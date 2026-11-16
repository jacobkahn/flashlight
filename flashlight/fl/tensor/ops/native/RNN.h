/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "flashlight/fl/tensor/Tensor.h"

namespace fl::native {

/**
 * Configuration for RNN operations, independent of any backend library.
 */
struct RnnConfig {
  RnnMode mode;
  int numGates;
  int directionMult;
  int numLayers;
  int hiddenSize;
  int inSize;
  int batchSize;
  int seqLength;
};

/**
 * Parsed weight tensors split from the flat weight buffer.
 * Used by both OneDNN primitive forward and manual forward/backward.
 */
struct ParsedWeightsAndBias {
  Tensor weightsInput1L;   // first layer input weights (when inSize != h*D)
  Tensor weightsHidden1L;  // first layer hidden weights
  Tensor bias1L;           // first layer bias
  Tensor weightsInput;     // remaining layers input weights
  Tensor weightsHidden;    // remaining layers hidden weights
  Tensor bias;             // remaining layers bias
};

/**
 * Result from manual single-layer RNN forward.
 */
struct NativeRnnForwardResult {
  Tensor y;   // output: [seqLen, batch, outSize]
  Tensor hy;  // final hidden: [D, batch, H]
  Tensor cy;  // final cell (LSTM only): [D, batch, H]
  RnnConfig config;
};

/**
 * Result from manual single-layer RNN backward.
 */
struct NativeRnnBackwardResult {
  Tensor diffInput;
  Tensor diffHx;
  Tensor diffCx;
  Tensor diffWeightsLayerFlat;
  Tensor diffWeightsIterFlat;
  Tensor diffBiasFlat;
};

/**
 * Parse the flat weight tensor into per-layer input/hidden/bias tensors.
 *
 * Flat layout (matching getNumRnnParams):
 *   [L0_Wi, L0_Wh, L1_Wi, L1_Wh, ..., biases]
 * where:
 *   L0_Wi size = directionMult * inSize * numGates * hiddenSize
 *   L0_Wh size = directionMult * hiddenSize * numGates * hiddenSize
 *   L1+_Wi size = directionMult * (hiddenSize * directionMult) * numGates * hiddenSize
 *   L1+_Wh size = directionMult * hiddenSize * numGates * hiddenSize
 *   biases: 2 bias sets per layer per direction, each numGates * hiddenSize
 */
ParsedWeightsAndBias parseWeights(
    const Tensor& weights,
    RnnMode mode,
    int numLayers,
    int directionMult,
    int inSize,
    int numGates,
    int hiddenSize);

/**
 * Reconstruct flat weight gradient from per-component gradients.
 * Must mirror parseWeights exactly.
 */
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
    fl::dtype wType);

/**
 * Manual single-layer RNN forward for types not supported by backend
 * primitives (e.g., f64). Multi-layer is handled by calling this per-layer.
 *
 * @tparam Scalar  float or double
 */
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
    int directionMult);

/**
 * Manual single-layer RNN backward (BPTT) for all CPU backends.
 * OneDNN 3.x CPU does not support RNN backward primitives, so all backward
 * goes through this path.
 *
 * @tparam Scalar  float or double
 */
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
    const RnnConfig& config);

// Explicit instantiations declared here, defined in RNN.cpp
extern template NativeRnnForwardResult manualRnnForward<float>(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, int, RnnMode, int, int);
extern template NativeRnnForwardResult manualRnnForward<double>(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, int, RnnMode, int, int);
extern template NativeRnnBackwardResult manualRnnBackward<float>(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const RnnConfig&);
extern template NativeRnnBackwardResult manualRnnBackward<double>(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const RnnConfig&);

} // namespace fl::native
