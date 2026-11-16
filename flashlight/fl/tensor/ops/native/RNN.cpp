/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/native/RNN.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/ops/native/CpuTensorHelpers.h"

namespace fl::native {

namespace {

// Reorder GRU gates for a single direction: [gate0, gate1, gate2] -->
// [gate1, gate0, gate2]. OneDNN's LBR-GRU expects gates in a different
// order than the flat weight layout.
Tensor reorderGruGatesSingle(int d1, int d2, const Tensor& weights) {
  int weightsSize = d1 * d2;
  return fl::concatenate(
      0,
      weights.flat(fl::range(weightsSize, 2 * weightsSize)),
      weights.flat(fl::range(0, weightsSize)),
      weights.flat(fl::range(2 * weightsSize, fl::end)));
}

// Handles both single-direction and bidirectional weight chunks.
Tensor reorderLbrGruWeights(int d1, int d2, const Tensor& weights) {
  int perDirSize = d1 * d2 * 3;
  int numDirs = weights.elements() / perDirSize;
  if (weights.elements() != perDirSize * numDirs || numDirs < 1) {
    throw std::invalid_argument(
        "RNN reorderLbrGruWeights: weights size " +
        std::to_string(weights.elements()) + " not a multiple of " +
        std::to_string(perDirSize));
  }
  if (numDirs == 1) {
    return reorderGruGatesSingle(d1, d2, weights);
  }
  Tensor result = makeCpuTensor(Shape({0}), weights.type(), nullptr);
  for (int dir = 0; dir < numDirs; ++dir) {
    auto chunk =
        weights.flat(fl::range(dir * perDirSize, (dir + 1) * perDirSize));
    result = fl::concatenate(0, result, reorderGruGatesSingle(d1, d2, chunk));
  }
  return result;
}

Tensor unreorderLbrGruWeights(int d1, int d2, const Tensor& weights) {
  return reorderLbrGruWeights(d1, d2, weights);
}

} // namespace

ParsedWeightsAndBias parseWeights(
    const Tensor& weights,
    RnnMode mode,
    int numLayers,
    int directionMult,
    int inSize,
    int numGates,
    int hiddenSize) {
  ParsedWeightsAndBias out;

  // For bidirectional, layers 1+ receive hiddenSize*directionMult as input
  int inputSizeNonFirst = hiddenSize * directionMult;
  int weightsInputSize1L = directionMult * inSize * numGates * hiddenSize;
  int weightsInputSizeNF =
      directionMult * inputSizeNonFirst * numGates * hiddenSize;
  int weightsHiddenSize = directionMult * hiddenSize * numGates * hiddenSize;
  int lbrGruBias = mode == RnnMode::GRU ? 1 : 0;

  // firstLayerDifferent when L0 input size differs from L1+ input size
  bool firstLayerDifferent =
      (numLayers > 1) && (inSize != inputSizeNonFirst);
  int numWeightsLayers = firstLayerDifferent ? numLayers - 1 : numLayers;
  int weightsOffset =
      firstLayerDifferent ? weightsInputSize1L + weightsHiddenSize : 0;

  auto weightsFlat = weights.flatten().astype(weights.type());

  // Parse first layer if different
  if (firstLayerDifferent) {
    out.weightsInput1L = weightsFlat.flat(fl::range(weightsInputSize1L));
    out.weightsHidden1L = weightsFlat.flat(
        fl::range(weightsInputSize1L, weightsInputSize1L + weightsHiddenSize));
    if (mode == RnnMode::GRU) {
      out.weightsInput1L =
          reorderLbrGruWeights(inSize, hiddenSize, out.weightsInput1L);
      out.weightsHidden1L =
          reorderLbrGruWeights(hiddenSize, hiddenSize, out.weightsHidden1L);
    }
  }

  // Parse remaining layers (or all layers if not firstLayerDifferent)
  // Use makeCpuTensor to avoid creating JIT tensors when the default backend
  // is JIT -- fl::concatenate requires all tensors have the same backend.
  auto weightsInput = makeCpuTensor(Shape({0}), weights.type(), nullptr);
  auto weightsHidden = makeCpuTensor(Shape({0}), weights.type(), nullptr);
  Tensor weightsFlatOffset =
      weightsFlat.flat(fl::range(weightsOffset, fl::end));

  // For non-first layers, the input weight chunk uses inputSizeNonFirst
  int chunkSizeInput = firstLayerDifferent
      ? weightsInputSizeNF
      : weightsInputSize1L; // when !firstLayerDifferent, inSize ==
                            // inputSizeNonFirst
  int chunkSizeHidden = weightsHiddenSize;
  int layerChunkSize = chunkSizeInput + chunkSizeHidden;

  // The input size for GRU reorder on non-first layers
  int reorderInSize = firstLayerDifferent ? inputSizeNonFirst : inSize;

  for (int i = 0; i < numWeightsLayers; ++i) {
    auto inputWeightsChunk = weightsFlatOffset.flat(
        fl::range(layerChunkSize * i, layerChunkSize * i + chunkSizeInput));
    auto hiddenWeightsChunk = weightsFlatOffset.flat(fl::range(
        layerChunkSize * i + chunkSizeInput, layerChunkSize * (i + 1)));

    if (mode == RnnMode::GRU) {
      inputWeightsChunk =
          reorderLbrGruWeights(reorderInSize, hiddenSize, inputWeightsChunk);
      hiddenWeightsChunk =
          reorderLbrGruWeights(hiddenSize, hiddenSize, hiddenWeightsChunk);
    }

    weightsInput = fl::concatenate(0, weightsInput, inputWeightsChunk);
    weightsHidden = fl::concatenate(0, weightsHidden, hiddenWeightsChunk);
  }
  out.weightsInput = weightsInput;
  out.weightsHidden = weightsHidden;

  // Bias parsing
  int biasStartOffset = firstLayerDifferent
      ? (weightsInputSize1L + weightsHiddenSize +
         numWeightsLayers * (chunkSizeInput + chunkSizeHidden))
      : numLayers * layerChunkSize;
  int numBiases = 2;
  Tensor biasFlat = weightsFlat.flat(fl::range(biasStartOffset, fl::end));

  Tensor bias = makeCpuTensor(Shape({0}), weights.type(), nullptr);
  for (int i = 0; i < numLayers; ++i) {
    if (mode == RnnMode::GRU) {
      // GRU has 2 bias sets of 3 gates each per direction
      int biasPerDir = hiddenSize * 6; // 3 gates * 2 bias sets
      int lbrGruChunkSize = directionMult * biasPerDir;
      int base = i * lbrGruChunkSize;

      // Process per-direction for bidirectional
      Tensor layerBias = makeCpuTensor(Shape({0}), weights.type(), nullptr);
      for (int d = 0; d < directionMult; ++d) {
        int db = base + d * biasPerDir;
        auto b1_g0 = biasFlat.flat(fl::range(db + 0, db + hiddenSize));
        auto b1_g1 =
            biasFlat.flat(fl::range(db + hiddenSize, db + 2 * hiddenSize));
        auto b1_g2 = biasFlat.flat(
            fl::range(db + 2 * hiddenSize, db + 3 * hiddenSize));
        auto b2_g0 = biasFlat.flat(
            fl::range(db + 3 * hiddenSize, db + 4 * hiddenSize));
        auto b2_g1 = biasFlat.flat(
            fl::range(db + 4 * hiddenSize, db + 5 * hiddenSize));
        auto b2_g2 = biasFlat.flat(
            fl::range(db + 5 * hiddenSize, db + 6 * hiddenSize));
        // Reorder gates: [g0,g1,g2] --> [g1,g0,g2] for lbr_gru
        // Combined bias = b1 + b2 for regular gates, b2 for lbr extra gate
        auto combined = fl::concatenate(
            0,
            b1_g1 + b2_g1, // gate1 (reordered to position 0)
            b1_g0 + b2_g0, // gate0 (reordered to position 1)
            b1_g2,         // gate2 (only from b1; b2_g2 goes to lbr slot)
            b2_g2);        // lbr_gru extra bias slot
        layerBias = fl::concatenate(0, layerBias, combined);
      }
      bias = fl::concatenate(0, bias, layerBias);
    } else {
      int layerBiasSize = directionMult * numGates * hiddenSize;
      int layerStride = layerBiasSize * numBiases;
      auto biases1 = biasFlat.flat(
          fl::range(layerStride * i, layerStride * i + layerBiasSize));
      auto biases2 = biasFlat.flat(
          fl::range(layerStride * i + layerBiasSize, layerStride * (i + 1)));
      bias = fl::concatenate(0, bias, biases1 + biases2);
    }
  }

  if (firstLayerDifferent) {
    int layerBiasSize = directionMult * (numGates + lbrGruBias) * hiddenSize;
    out.bias1L = bias.flat(fl::range(layerBiasSize));
    if (numLayers > 1) {
      bias = bias.flat(fl::range(layerBiasSize, fl::end));
    }
  }
  out.bias = bias;

  if (firstLayerDifferent && numLayers == 1) {
    out.weightsInput = out.weightsInput1L;
    out.weightsHidden = out.weightsHidden1L;
    out.bias = out.bias1L;
  }

  return out;
}

Tensor reconstructFlatWeightGrad(
    const Tensor& diffWeightsLayerFlat,
    const Tensor& diffWeightsIterFlat,
    const Tensor& diffBias,
    RnnMode mode,
    int numLayers,
    int directionMult,
    int inSize,
    int numGates,
    int hiddenSize,
    fl::dtype wType) {
  int inputSizeNonFirst = hiddenSize * directionMult;
  bool firstLayerDifferent =
      (numLayers > 1) && (inSize != inputSizeNonFirst);

  int weightsInputSize1L = directionMult * inSize * numGates * hiddenSize;
  int weightsInputSizeNF =
      directionMult * inputSizeNonFirst * numGates * hiddenSize;
  int weightsHiddenSize = directionMult * hiddenSize * numGates * hiddenSize;
  int lbrGruBias = mode == RnnMode::GRU ? 1 : 0;

  Tensor flatGrad = makeCpuTensor(Shape({0}), wType, nullptr);

  if (firstLayerDifferent) {
    auto dwI1L = diffWeightsLayerFlat.flat(fl::range(weightsInputSize1L));
    auto dwH1L = diffWeightsIterFlat.flat(fl::range(weightsHiddenSize));
    if (mode == RnnMode::GRU) {
      dwI1L = unreorderLbrGruWeights(inSize, hiddenSize, dwI1L);
      dwH1L = unreorderLbrGruWeights(hiddenSize, hiddenSize, dwH1L);
    }
    flatGrad = fl::concatenate(0, flatGrad, dwI1L, dwH1L);
  }

  int numWeightsLayers = firstLayerDifferent ? numLayers - 1 : numLayers;
  int chunkSizeInput =
      firstLayerDifferent ? weightsInputSizeNF : weightsInputSize1L;
  int chunkSizeHidden = weightsHiddenSize;

  int reorderInSize = firstLayerDifferent ? inputSizeNonFirst : inSize;

  int wlOffset = firstLayerDifferent ? weightsInputSize1L : 0;
  int whOffset = firstLayerDifferent ? weightsHiddenSize : 0;

  for (int i = 0; i < numWeightsLayers; ++i) {
    auto dwI = diffWeightsLayerFlat.flat(fl::range(
        wlOffset + chunkSizeInput * i,
        wlOffset + chunkSizeInput * (i + 1)));
    auto dwH = diffWeightsIterFlat.flat(fl::range(
        whOffset + chunkSizeHidden * i,
        whOffset + chunkSizeHidden * (i + 1)));
    if (mode == RnnMode::GRU) {
      dwI = unreorderLbrGruWeights(reorderInSize, hiddenSize, dwI);
      dwH = unreorderLbrGruWeights(hiddenSize, hiddenSize, dwH);
    }
    flatGrad = fl::concatenate(0, flatGrad, dwI, dwH);
  }

  // Bias gradient
  int H = hiddenSize;
  if (mode == RnnMode::GRU) {
    for (int i = 0; i < numLayers; ++i) {
      int lbrGruBiasSize =
          directionMult * (numGates + lbrGruBias) * hiddenSize;
      auto diffBiasLayer = diffBias.flat(
          fl::range(lbrGruBiasSize * i, lbrGruBiasSize * (i + 1)));
      // Un-reorder GRU bias gates per direction
      int perDirBias = (numGates + lbrGruBias) * H;
      for (int d = 0; d < directionMult; ++d) {
        int db = d * perDirBias;
        // OneDNN gives combined bias in [g1, g0, g2, lbr] order
        auto dg1 = diffBiasLayer.flat(fl::range(db, db + H));
        auto dg0 = diffBiasLayer.flat(fl::range(db + H, db + 2 * H));
        auto dg2 = diffBiasLayer.flat(fl::range(db + 2 * H, db + 3 * H));
        auto dlbr = diffBiasLayer.flat(fl::range(db + 3 * H, db + 4 * H));
        // Reverse to original flat format:
        // bias1 = [g0, g1, g2], bias2 = [g0, g1, g2]
        flatGrad = fl::concatenate(
            0, flatGrad,
            dg0, dg1, dg2,   // bias1: [g0, g1, g2]
            dg0, dg1, dlbr); // bias2: [g0, g1, lbr]
      }
    }
  } else {
    int layerBiasSize = directionMult * numGates * hiddenSize;
    for (int i = 0; i < numLayers; ++i) {
      auto diffBiasLayer = diffBias.flat(
          fl::range(layerBiasSize * i, layerBiasSize * (i + 1)));
      // Duplicate: bias1_grad = bias2_grad = combined_grad
      flatGrad = fl::concatenate(0, flatGrad, diffBiasLayer, diffBiasLayer);
    }
  }

  return flatGrad;
}

template <typename Scalar>
NativeRnnForwardResult manualRnnForward(
    const Tensor& input,
    const Tensor& hiddenState,
    const Tensor& cellState,
    const Tensor& weightsInput,
    const Tensor& weightsHidden,
    const Tensor& bias,
    int hiddenSize,
    RnnMode mode,
    int numGates,
    int directionMult) {
  int seqLen = input.dim(0);
  int N = input.ndim() < 2 ? 1 : input.dim(1);
  int inSize = input.ndim() < 3 ? 1 : input.dim(2);
  int H = hiddenSize;
  int D = directionMult;
  int outSize = H * D;
  int extraBias = (mode == RnnMode::GRU) ? 1 : 0;
  int bPerDir = (numGates + extraBias) * H;
  int wxPerDir = inSize * numGates * H;
  int whPerDir = H * numGates * H;

  auto readHost = [](const Tensor& t, int expectedSize) {
    std::vector<Scalar> v(expectedSize, Scalar(0));
    if (!t.isEmpty()) {
      t.asContiguousTensor().hostSync(v.data());
    }
    return v;
  };

  auto xHost = readHost(input, seqLen * N * inSize);
  auto wxHost = readHost(weightsInput, D * wxPerDir);
  auto whHost = readHost(weightsHidden, D * whPerDir);
  auto bHost = readHost(bias, D * bPerDir);
  auto hxHost = readHost(hiddenState, D * N * H);
  auto cxHost =
      readHost(cellState, mode == RnnMode::LSTM ? D * N * H : 0);

  auto sigmoid = [](Scalar x) -> Scalar {
    return Scalar(1) / (Scalar(1) + std::exp(-x));
  };

  std::vector<Scalar> yHost(seqLen * N * outSize, Scalar(0));
  std::vector<Scalar> hyHost(D * N * H, Scalar(0));
  std::vector<Scalar> cyHost(
      mode == RnnMode::LSTM ? D * N * H : 0, Scalar(0));

  for (int d = 0; d < D; d++) {
    Scalar* Wx = wxHost.data() + d * wxPerDir;
    Scalar* Wh = whHost.data() + d * whPerDir;
    Scalar* b = bHost.data() + d * bPerDir;
    Scalar* h0 = hxHost.data() + d * N * H;
    bool isForwardDir = (d == 0);

    std::vector<Scalar> h_prev(N * H);
    std::copy(h0, h0 + N * H, h_prev.begin());

    std::vector<Scalar> c_prev;
    if (mode == RnnMode::LSTM) {
      c_prev.resize(N * H);
      Scalar* c0 = cxHost.data() + d * N * H;
      std::copy(c0, c0 + N * H, c_prev.begin());
    }

    for (int t_idx = 0; t_idx < seqLen; t_idx++) {
      int t = isForwardDir ? t_idx : (seqLen - 1 - t_idx);
      Scalar* xt = xHost.data() + t * N * inSize;
      std::vector<Scalar> h_cur(N * H);

      if (mode == RnnMode::TANH || mode == RnnMode::RELU) {
        for (int n = 0; n < N; n++) {
          for (int h = 0; h < H; h++) {
            Scalar val = b[h];
            for (int i = 0; i < inSize; i++)
              val += xt[n * inSize + i] * Wx[h * inSize + i];
            for (int j = 0; j < H; j++)
              val += h_prev[n * H + j] * Wh[h * H + j];
            h_cur[n * H + h] = (mode == RnnMode::TANH)
                ? std::tanh(val)
                : std::max(Scalar(0), val);
          }
        }
      } else if (mode == RnnMode::LSTM) {
        std::vector<Scalar> c_cur(N * H);
        for (int n = 0; n < N; n++) {
          std::vector<Scalar> pre(4 * H);
          for (int g = 0; g < 4; g++) {
            for (int h = 0; h < H; h++) {
              int gh = g * H + h;
              Scalar val = b[gh];
              for (int i = 0; i < inSize; i++)
                val += xt[n * inSize + i] * Wx[gh * inSize + i];
              for (int j = 0; j < H; j++)
                val += h_prev[n * H + j] * Wh[gh * H + j];
              pre[gh] = val;
            }
          }
          for (int h = 0; h < H; h++) {
            Scalar i_g = sigmoid(pre[0 * H + h]);
            Scalar f_g = sigmoid(pre[1 * H + h]);
            Scalar c_g = std::tanh(pre[2 * H + h]);
            Scalar o_g = sigmoid(pre[3 * H + h]);
            c_cur[n * H + h] = f_g * c_prev[n * H + h] + i_g * c_g;
            h_cur[n * H + h] = o_g * std::tanh(c_cur[n * H + h]);
          }
        }
        c_prev = c_cur;
      } else if (mode == RnnMode::GRU) {
        for (int n = 0; n < N; n++) {
          for (int h = 0; h < H; h++) {
            Scalar r_pre = b[0 * H + h];
            for (int i = 0; i < inSize; i++)
              r_pre += xt[n * inSize + i] * Wx[(0 * H + h) * inSize + i];
            for (int j = 0; j < H; j++)
              r_pre += h_prev[n * H + j] * Wh[(0 * H + h) * H + j];
            Scalar r_val = sigmoid(r_pre);

            Scalar z_pre = b[1 * H + h];
            for (int i = 0; i < inSize; i++)
              z_pre += xt[n * inSize + i] * Wx[(1 * H + h) * inSize + i];
            for (int j = 0; j < H; j++)
              z_pre += h_prev[n * H + j] * Wh[(1 * H + h) * H + j];
            Scalar z_val = sigmoid(z_pre);

            Scalar n_hid = b[3 * H + h];
            for (int j = 0; j < H; j++)
              n_hid += h_prev[n * H + j] * Wh[(2 * H + h) * H + j];
            Scalar n_pre = b[2 * H + h];
            for (int i = 0; i < inSize; i++)
              n_pre += xt[n * inSize + i] * Wx[(2 * H + h) * inSize + i];
            n_pre += r_val * n_hid;
            Scalar n_val = std::tanh(n_pre);

            h_cur[n * H + h] =
                (Scalar(1) - z_val) * n_val + z_val * h_prev[n * H + h];
          }
        }
      }

      // Write to y output
      for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
          yHost[t * N * outSize + n * outSize + d * H + h] = h_cur[n * H + h];
        }
      }
      h_prev = h_cur;
    }

    // hy = final hidden state for this direction
    std::copy(h_prev.begin(), h_prev.end(), hyHost.data() + d * N * H);
    if (mode == RnnMode::LSTM) {
      std::copy(c_prev.begin(), c_prev.end(), cyHost.data() + d * N * H);
    }
  }

  NativeRnnForwardResult result;
  result.y = makeCpuTensor({seqLen, N, outSize}, yHost.data()).copy();
  result.hy = makeCpuTensor({D, N, H}, hyHost.data()).copy();
  if (mode == RnnMode::LSTM) {
    result.cy = makeCpuTensor({D, N, H}, cyHost.data()).copy();
  }
  result.config = {mode, numGates, D, 1, H, inSize, N, seqLen};
  return result;
}

template <typename Scalar>
NativeRnnBackwardResult manualRnnBackward(
    const Tensor& input,
    const Tensor& hiddenState,
    const Tensor& cellState,
    const Tensor& weightsInput,
    const Tensor& weightsHidden,
    const Tensor& bias,
    const Tensor& dy,
    const Tensor& dhy,
    const Tensor& dcy,
    const RnnConfig& config) {
  NativeRnnBackwardResult result;
  int D = config.directionMult;
  int seqLen = config.seqLength;
  int N = config.batchSize;
  int inSize = input.dim(2);
  int H = config.hiddenSize;
  int outSize = H * D;
  int G = config.numGates;
  RnnMode mode = config.mode;
  int extraBias = (mode == RnnMode::GRU) ? 1 : 0;
  int bPerDir = (G + extraBias) * H;

  // Per-direction weight sizes (GOI format: [G*O, I])
  int wxPerDir = inSize * G * H;
  int whPerDir = H * G * H;

  // Read all data to host in the native precision
  auto readHost = [](const Tensor& t, int expectedSize) {
    std::vector<Scalar> v(expectedSize, Scalar(0));
    if (!t.isEmpty()) {
      t.asContiguousTensor().hostSync(v.data());
    }
    return v;
  };

  auto xHost = readHost(input, seqLen * N * inSize);
  auto wxHost = readHost(weightsInput, D * wxPerDir);
  auto whHost = readHost(weightsHidden, D * whPerDir);
  auto bHost = readHost(bias, D * bPerDir);
  auto hxHost = readHost(hiddenState, D * N * H);
  auto dyHost = readHost(dy, seqLen * N * outSize);
  auto dhyHost = readHost(dhy, D * N * H);

  // LSTM-specific
  auto cxHost = readHost(cellState, mode == RnnMode::LSTM ? D * N * H : 0);
  auto dcyHost = readHost(dcy, mode == RnnMode::LSTM ? D * N * H : 0);

  // Output gradients
  std::vector<Scalar> dxHost(seqLen * N * inSize, Scalar(0));
  std::vector<Scalar> dhxHost(D * N * H, Scalar(0));
  std::vector<Scalar> dcxHost(
      mode == RnnMode::LSTM ? D * N * H : 0, Scalar(0));
  std::vector<Scalar> dWxHost(D * wxPerDir, Scalar(0));
  std::vector<Scalar> dWhHost(D * whPerDir, Scalar(0));
  std::vector<Scalar> dBHost(D * bPerDir, Scalar(0));

  auto sigmoid = [](Scalar x) -> Scalar {
    return Scalar(1) / (Scalar(1) + std::exp(-x));
  };

  // Process each direction independently
  for (int d = 0; d < D; d++) {
    Scalar* Wx = wxHost.data() + d * wxPerDir;
    Scalar* Wh = whHost.data() + d * whPerDir;
    Scalar* b = bHost.data() + d * bPerDir;
    Scalar* h0 = hxHost.data() + d * N * H;
    Scalar* dWx = dWxHost.data() + d * wxPerDir;
    Scalar* dWh = dWhHost.data() + d * whPerDir;
    Scalar* dB = dBHost.data() + d * bPerDir;
    Scalar* dhx_out = dhxHost.data() + d * N * H;
    bool isForwardDir = (d == 0);

    // --- FORWARD RECOMPUTE to get per-timestep intermediates ---
    std::vector<std::vector<Scalar>> h_all(
        seqLen + 1, std::vector<Scalar>(N * H));
    std::copy(h0, h0 + N * H, h_all[0].begin());

    // LSTM: cell states and post-activation gate values
    std::vector<std::vector<Scalar>> cell_all;
    std::vector<std::vector<Scalar>> gates_act;
    Scalar* c0 = nullptr;
    if (mode == RnnMode::LSTM) {
      cell_all.resize(seqLen + 1, std::vector<Scalar>(N * H, Scalar(0)));
      c0 = cxHost.data() + d * N * H;
      std::copy(c0, c0 + N * H, cell_all[0].begin());
      gates_act.resize(seqLen, std::vector<Scalar>(N * 4 * H));
    }

    // GRU: save gate activations and linear-before-reset term
    std::vector<std::vector<Scalar>> gru_z, gru_r, gru_n, gru_n_hidden;
    if (mode == RnnMode::GRU) {
      gru_z.resize(seqLen, std::vector<Scalar>(N * H));
      gru_r.resize(seqLen, std::vector<Scalar>(N * H));
      gru_n.resize(seqLen, std::vector<Scalar>(N * H));
      gru_n_hidden.resize(seqLen, std::vector<Scalar>(N * H));
    }

    for (int t_idx = 0; t_idx < seqLen; t_idx++) {
      int t = isForwardDir ? t_idx : (seqLen - 1 - t_idx);
      Scalar* xt = xHost.data() + t * N * inSize;
      Scalar* h_prev = h_all[t_idx].data();
      Scalar* h_cur = h_all[t_idx + 1].data();

      if (mode == RnnMode::TANH || mode == RnnMode::RELU) {
        for (int n = 0; n < N; n++) {
          for (int h = 0; h < H; h++) {
            Scalar val = b[h];
            for (int i = 0; i < inSize; i++)
              val += xt[n * inSize + i] * Wx[h * inSize + i];
            for (int j = 0; j < H; j++)
              val += h_prev[n * H + j] * Wh[h * H + j];
            h_cur[n * H + h] = (mode == RnnMode::TANH)
                ? std::tanh(val)
                : std::max(Scalar(0), val);
          }
        }
      } else if (mode == RnnMode::LSTM) {
        Scalar* c_prev = cell_all[t_idx].data();
        Scalar* c_cur = cell_all[t_idx + 1].data();
        Scalar* ga = gates_act[t_idx].data();
        for (int n = 0; n < N; n++) {
          std::vector<Scalar> pre(4 * H);
          for (int g = 0; g < 4; g++) {
            for (int h = 0; h < H; h++) {
              int gh = g * H + h;
              Scalar val = b[gh];
              for (int i = 0; i < inSize; i++)
                val += xt[n * inSize + i] * Wx[gh * inSize + i];
              for (int j = 0; j < H; j++)
                val += h_prev[n * H + j] * Wh[gh * H + j];
              pre[gh] = val;
            }
          }
          for (int h = 0; h < H; h++) {
            Scalar i_g = sigmoid(pre[0 * H + h]);
            Scalar f_g = sigmoid(pre[1 * H + h]);
            Scalar c_g = std::tanh(pre[2 * H + h]);
            Scalar o_g = sigmoid(pre[3 * H + h]);
            ga[n * 4 * H + 0 * H + h] = i_g;
            ga[n * 4 * H + 1 * H + h] = f_g;
            ga[n * 4 * H + 2 * H + h] = c_g;
            ga[n * 4 * H + 3 * H + h] = o_g;
            c_cur[n * H + h] = f_g * c_prev[n * H + h] + i_g * c_g;
            h_cur[n * H + h] = o_g * std::tanh(c_cur[n * H + h]);
          }
        }
      } else if (mode == RnnMode::GRU) {
        Scalar* z_save = gru_z[t_idx].data();
        Scalar* r_save = gru_r[t_idx].data();
        Scalar* n_save = gru_n[t_idx].data();
        Scalar* nh_save = gru_n_hidden[t_idx].data();
        for (int n = 0; n < N; n++) {
          for (int h = 0; h < H; h++) {
            Scalar r_pre = b[0 * H + h];
            for (int i = 0; i < inSize; i++)
              r_pre += xt[n * inSize + i] * Wx[(0 * H + h) * inSize + i];
            for (int j = 0; j < H; j++)
              r_pre += h_prev[n * H + j] * Wh[(0 * H + h) * H + j];
            Scalar r_val = sigmoid(r_pre);

            Scalar z_pre = b[1 * H + h];
            for (int i = 0; i < inSize; i++)
              z_pre += xt[n * inSize + i] * Wx[(1 * H + h) * inSize + i];
            for (int j = 0; j < H; j++)
              z_pre += h_prev[n * H + j] * Wh[(1 * H + h) * H + j];
            Scalar z_val = sigmoid(z_pre);

            Scalar n_hid = b[3 * H + h];
            for (int j = 0; j < H; j++)
              n_hid += h_prev[n * H + j] * Wh[(2 * H + h) * H + j];
            Scalar n_pre = b[2 * H + h];
            for (int i = 0; i < inSize; i++)
              n_pre += xt[n * inSize + i] * Wx[(2 * H + h) * inSize + i];
            n_pre += r_val * n_hid;
            Scalar n_val = std::tanh(n_pre);

            r_save[n * H + h] = r_val;
            z_save[n * H + h] = z_val;
            n_save[n * H + h] = n_val;
            nh_save[n * H + h] = n_hid;
            h_cur[n * H + h] =
                (Scalar(1) - z_val) * n_val + z_val * h_prev[n * H + h];
          }
        }
      }
    }

    // --- BACKWARD PASS (BPTT) ---
    std::vector<Scalar> dh_next(N * H, Scalar(0));
    std::copy(
        dhyHost.data() + d * N * H,
        dhyHost.data() + d * N * H + N * H,
        dh_next.begin());

    std::vector<Scalar> dc_next(N * H, Scalar(0));
    if (mode == RnnMode::LSTM) {
      std::copy(
          dcyHost.data() + d * N * H,
          dcyHost.data() + d * N * H + N * H,
          dc_next.begin());
    }

    for (int t_idx = seqLen - 1; t_idx >= 0; t_idx--) {
      int t = isForwardDir ? t_idx : (seqLen - 1 - t_idx);
      Scalar* xt = xHost.data() + t * N * inSize;
      Scalar* h_prev = h_all[t_idx].data();
      Scalar* h_cur = h_all[t_idx + 1].data();
      std::vector<Scalar> new_dh_next(N * H, Scalar(0));

      if (mode == RnnMode::TANH || mode == RnnMode::RELU) {
        for (int n = 0; n < N; n++) {
          std::vector<Scalar> d_pre(H);
          for (int h = 0; h < H; h++) {
            int dy_idx = t * N * outSize + n * outSize + d * H + h;
            Scalar dh = dyHost[dy_idx] + dh_next[n * H + h];
            if (mode == RnnMode::TANH) {
              Scalar ht = h_cur[n * H + h];
              d_pre[h] = dh * (Scalar(1) - ht * ht);
            } else { // RELU
              d_pre[h] = (h_cur[n * H + h] > Scalar(0)) ? dh : Scalar(0);
            }
          }
          for (int i = 0; i < inSize; i++) {
            Scalar acc = Scalar(0);
            for (int h = 0; h < H; h++)
              acc += d_pre[h] * Wx[h * inSize + i];
            dxHost[t * N * inSize + n * inSize + i] += acc;
          }
          for (int j = 0; j < H; j++) {
            Scalar acc = Scalar(0);
            for (int h = 0; h < H; h++)
              acc += d_pre[h] * Wh[h * H + j];
            new_dh_next[n * H + j] += acc;
          }
          for (int h = 0; h < H; h++)
            for (int i = 0; i < inSize; i++)
              dWx[h * inSize + i] += d_pre[h] * xt[n * inSize + i];
          for (int h = 0; h < H; h++)
            for (int j = 0; j < H; j++)
              dWh[h * H + j] += d_pre[h] * h_prev[n * H + j];
          for (int h = 0; h < H; h++)
            dB[h] += d_pre[h];
        }
      } else if (mode == RnnMode::LSTM) {
        Scalar* c_prev = cell_all[t_idx].data();
        Scalar* c_cur = cell_all[t_idx + 1].data();
        Scalar* ga = gates_act[t_idx].data();
        std::vector<Scalar> new_dc_next(N * H, Scalar(0));

        for (int n = 0; n < N; n++) {
          std::vector<Scalar> d_gates(4 * H);
          for (int h = 0; h < H; h++) {
            int dy_idx = t * N * outSize + n * outSize + d * H + h;
            Scalar dh = dyHost[dy_idx] + dh_next[n * H + h];
            Scalar i_g = ga[n * 4 * H + 0 * H + h];
            Scalar f_g = ga[n * 4 * H + 1 * H + h];
            Scalar c_g = ga[n * 4 * H + 2 * H + h];
            Scalar o_g = ga[n * 4 * H + 3 * H + h];
            Scalar tc = std::tanh(c_cur[n * H + h]);

            Scalar do_pre = dh * tc;
            Scalar dc =
                dh * o_g * (Scalar(1) - tc * tc) + dc_next[n * H + h];

            Scalar di_pre = dc * c_g;
            Scalar df_pre = dc * c_prev[n * H + h];
            Scalar dc_gate_pre = dc * i_g;
            new_dc_next[n * H + h] = dc * f_g;

            d_gates[0 * H + h] = di_pre * i_g * (Scalar(1) - i_g);
            d_gates[1 * H + h] = df_pre * f_g * (Scalar(1) - f_g);
            d_gates[2 * H + h] = dc_gate_pre * (Scalar(1) - c_g * c_g);
            d_gates[3 * H + h] = do_pre * o_g * (Scalar(1) - o_g);
          }
          for (int i = 0; i < inSize; i++) {
            Scalar acc = Scalar(0);
            for (int gh = 0; gh < 4 * H; gh++)
              acc += d_gates[gh] * Wx[gh * inSize + i];
            dxHost[t * N * inSize + n * inSize + i] += acc;
          }
          for (int j = 0; j < H; j++) {
            Scalar acc = Scalar(0);
            for (int gh = 0; gh < 4 * H; gh++)
              acc += d_gates[gh] * Wh[gh * H + j];
            new_dh_next[n * H + j] += acc;
          }
          for (int gh = 0; gh < 4 * H; gh++)
            for (int i = 0; i < inSize; i++)
              dWx[gh * inSize + i] += d_gates[gh] * xt[n * inSize + i];
          for (int gh = 0; gh < 4 * H; gh++)
            for (int j = 0; j < H; j++)
              dWh[gh * H + j] += d_gates[gh] * h_prev[n * H + j];
          for (int gh = 0; gh < 4 * H; gh++)
            dB[gh] += d_gates[gh];
        }
        dc_next = new_dc_next;
      } else if (mode == RnnMode::GRU) {
        Scalar* z_save = gru_z[t_idx].data();
        Scalar* r_save = gru_r[t_idx].data();
        Scalar* n_save = gru_n[t_idx].data();
        Scalar* nh_save = gru_n_hidden[t_idx].data();

        for (int n = 0; n < N; n++) {
          for (int h = 0; h < H; h++) {
            int dy_idx = t * N * outSize + n * outSize + d * H + h;
            Scalar dh = dyHost[dy_idx] + dh_next[n * H + h];
            Scalar z = z_save[n * H + h];
            Scalar r = r_save[n * H + h];
            Scalar nv = n_save[n * H + h];
            Scalar nh = nh_save[n * H + h];

            Scalar dz_pre = dh * (h_prev[n * H + h] - nv);
            Scalar dn = dh * (Scalar(1) - z);
            Scalar dh_prev_from_z = dh * z;

            Scalar dn_pre = dn * (Scalar(1) - nv * nv);
            Scalar dr_from_n = dn_pre * nh;
            Scalar dnh = dn_pre * r;

            Scalar dz_gate = dz_pre * z * (Scalar(1) - z);
            Scalar dr_gate = dr_from_n * r * (Scalar(1) - r);

            for (int i = 0; i < inSize; i++) {
              dxHost[t * N * inSize + n * inSize + i] +=
                  dr_gate * Wx[(0 * H + h) * inSize + i] +
                  dz_gate * Wx[(1 * H + h) * inSize + i] +
                  dn_pre * Wx[(2 * H + h) * inSize + i];
            }

            new_dh_next[n * H + h] += dh_prev_from_z;

            for (int j = 0; j < H; j++) {
              new_dh_next[n * H + j] +=
                  dr_gate * Wh[(0 * H + h) * H + j] +
                  dz_gate * Wh[(1 * H + h) * H + j] +
                  dnh * Wh[(2 * H + h) * H + j];
            }

            for (int i = 0; i < inSize; i++) {
              dWx[(0 * H + h) * inSize + i] += dr_gate * xt[n * inSize + i];
              dWx[(1 * H + h) * inSize + i] += dz_gate * xt[n * inSize + i];
              dWx[(2 * H + h) * inSize + i] += dn_pre * xt[n * inSize + i];
            }

            for (int j = 0; j < H; j++) {
              dWh[(0 * H + h) * H + j] += dr_gate * h_prev[n * H + j];
              dWh[(1 * H + h) * H + j] += dz_gate * h_prev[n * H + j];
              dWh[(2 * H + h) * H + j] += dnh * h_prev[n * H + j];
            }

            dB[0 * H + h] += dr_gate;
            dB[1 * H + h] += dz_gate;
            dB[2 * H + h] += dn_pre;
            dB[3 * H + h] += dnh;
          }
        }
      }
      dh_next = new_dh_next;
    }

    // Store dhx for this direction
    std::copy(dh_next.begin(), dh_next.end(), dhx_out);
    if (mode == RnnMode::LSTM) {
      Scalar* dcx_out = dcxHost.data() + d * N * H;
      std::copy(dc_next.begin(), dc_next.end(), dcx_out);
    }
  }

  // Create output tensors from host arrays
  int hySize = D; // single layer
  result.diffInput =
      makeCpuTensor({seqLen, N, inSize}, dxHost.data()).copy();
  result.diffHx =
      makeCpuTensor({hySize, N, H}, dhxHost.data()).copy();
  if (mode == RnnMode::LSTM) {
    result.diffCx =
        makeCpuTensor({hySize, N, H}, dcxHost.data()).copy();
  }

  // Weight gradients in GOI format (same as parseWeights output)
  int wlSize = D * wxPerDir;
  int whSize = D * whPerDir;
  int biasOutSize = D * bPerDir;
  result.diffWeightsLayerFlat =
      makeCpuTensor({wlSize}, dWxHost.data()).copy();
  result.diffWeightsIterFlat =
      makeCpuTensor({whSize}, dWhHost.data()).copy();
  result.diffBiasFlat =
      makeCpuTensor({biasOutSize}, dBHost.data()).copy();

  return result;
}

// Explicit template instantiations
template NativeRnnForwardResult manualRnnForward<float>(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, int, RnnMode, int, int);
template NativeRnnForwardResult manualRnnForward<double>(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, int, RnnMode, int, int);
template NativeRnnBackwardResult manualRnnBackward<float>(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const RnnConfig&);
template NativeRnnBackwardResult manualRnnBackward<double>(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const RnnConfig&);

} // namespace fl::native
