/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/onednn/OneDnnAutogradOps.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <dnnl.hpp>

#include "flashlight/fl/tensor/ops/onednn/DnnlUtils.h"
#include "flashlight/fl/tensor/ops/onednn/OneDnnOps.h"
#include "flashlight/fl/tensor/ops/onednn/Utils.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/ops/native/RNN.h"

namespace fl {
namespace detail {
namespace onednn {

namespace {

// State saved from forward for use in backward.
// Contains OneDNN-specific state (workspace, reordered weight memories)
// plus the backend-independent RNN config.
struct OneDnnRnnForwardState : ForwardState {
  dnnl::memory workspace;
  dnnl::memory weightsLayerMem;
  dnnl::memory weightsIterMem;

  fl::native::RnnConfig config;

  dnnl::algorithm activation;
  dnnl::rnn_direction direction;

  Tensor y;
  Tensor hy;
  Tensor cy;

  fl::native::ParsedWeightsAndBias parsedWeights;
  bool splitLayers = false;
  // For split-layer case: save sub-states
  std::shared_ptr<OneDnnRnnForwardState> stateL0;
  std::shared_ptr<OneDnnRnnForwardState> stateL1N;
  Tensor forwardInput; // original input to rnnForward (for split backward)
};

struct RnnResult {
  Tensor y;
  Tensor hy;
  Tensor cy;
  std::shared_ptr<OneDnnRnnForwardState> state;
};

// OneDNN primitive-based forward (f32/bf16 only).
RnnResult rnnImpl(
    const Tensor& input,
    const Tensor& hiddenState,
    const Tensor& cellState,
    const Tensor& weightsInput,
    const Tensor& weightsHidden,
    const Tensor& bias,
    int hiddenSize,
    int numLayers,
    RnnMode mode,
    dnnl::algorithm activation,
    int numGates,
    dnnl::rnn_direction direction,
    int directionMult,
    dnnl::prop_kind kind,
    float dropout,
    const dnnl::engine& dnnlEngine,
    AsyncCPUStream& asyncStream,
    dnnl::stream& nativeStream) {
  RnnResult result;
  auto state = std::make_shared<OneDnnRnnForwardState>();

  // TNC format: dim(0)=T=seqLength, dim(1)=N=batch, dim(2)=C=inSize
  int seqLength = input.dim(0);
  int batchSize = input.ndim() < 2 ? 1 : input.dim(1);
  int inSize = input.ndim() < 3 ? 1 : input.dim(2);
  dnnl::memory::dims inputDims = {seqLength, batchSize, inSize};
  int outSize = hiddenSize * directionMult;
  dnnl::memory::dims outputDims = {seqLength, batchSize, outSize};
  auto dType = detail::dnnlMapToType(input.type());
  dnnl::memory::dims hDims = {
      numLayers, directionMult, batchSize, hiddenSize};
  dnnl::memory::dims cDims = hDims;
  int extraBias = mode == RnnMode::GRU ? 1 : 0;
  dnnl::memory::dims biasDims = {
      numLayers, directionMult, numGates + extraBias, hiddenSize};
  dnnl::memory::dims weightsInputDims = {
      numLayers, directionMult, inSize, numGates, hiddenSize};
  dnnl::memory::dims weightsHiddenDims = {
      numLayers, directionMult, hiddenSize, numGates, hiddenSize};

  auto y = makeCPUTensor({seqLength, batchSize, outSize}, input.type());
  int hySize = numLayers * directionMult;
  auto hy = makeCPUTensor({hySize, batchSize, hiddenSize}, input.type());
  Tensor cy;
  if (mode == RnnMode::LSTM) {
    cy = makeCPUTensor(hy.shape(), input.type());
  }

  auto tnc = dnnl::memory::format_tag::tnc;
  auto ldnc = dnnl::memory::format_tag::ldnc;
  auto ldgoi = dnnl::memory::format_tag::ldgoi;
  auto ldgo = dnnl::memory::format_tag::ldgo;

  // Store contiguous tensors in local variables so the same buffer is used
  // for both DnnlMemoryWrapper (which creates a dnnl::memory with a raw ptr)
  // and the keepAlive SharedData refs.
  auto contInput = input.asContiguousTensor();
  auto contWeightsInput = weightsInput.asContiguousTensor();
  auto contWeightsHidden = weightsHidden.asContiguousTensor();
  auto contBias = bias.asContiguousTensor();
  Tensor contCellState;

  const detail::DnnlMemoryWrapper inputMemInit(
      contInput, {inputDims}, tnc, dnnlEngine);
  const detail::DnnlMemoryWrapper outputMemInit(
      y, {outputDims}, tnc, dnnlEngine);
  Tensor contHiddenState;
  detail::DnnlMemoryWrapper hiddenInMemInit;
  if (!hiddenState.isEmpty()) {
    contHiddenState = hiddenState.asContiguousTensor();
    hiddenInMemInit = detail::DnnlMemoryWrapper(
        contHiddenState, {hDims}, ldnc, dnnlEngine);
  }
  const detail::DnnlMemoryWrapper hiddenOutMemInit(
      hy, {hDims}, ldnc, dnnlEngine);
  const detail::DnnlMemoryWrapper weightsInputMemRawInit(
      contWeightsInput, {weightsInputDims}, ldgoi, dnnlEngine);
  const detail::DnnlMemoryWrapper weightsHiddenMemRawInit(
      contWeightsHidden, {weightsHiddenDims}, ldgoi, dnnlEngine);
  const detail::DnnlMemoryWrapper biasMemInit(
      contBias, {biasDims}, ldgo, dnnlEngine);

  auto weightsInputMemDesc = dnnl::memory::desc(
      weightsInputDims, dType, dnnl::memory::format_tag::ldigo);
  auto weightsInputMemInit =
      dnnl::memory(weightsInputMemDesc, dnnlEngine);
  auto weightsHiddenMemDesc = dnnl::memory::desc(
      weightsHiddenDims, dType, dnnl::memory::format_tag::ldigo);
  auto weightsHiddenMemInit =
      dnnl::memory(weightsHiddenMemDesc, dnnlEngine);

  std::unordered_map<int, dnnl::memory> rnnFwdArgs = {
      {DNNL_ARG_SRC_LAYER, inputMemInit.getMemory()},
      {DNNL_ARG_SRC_ITER, hiddenInMemInit.getMemory()},
      {DNNL_ARG_WEIGHTS_LAYER, weightsInputMemInit},
      {DNNL_ARG_WEIGHTS_ITER, weightsHiddenMemInit},
      {DNNL_ARG_BIAS, biasMemInit.getMemory()},
      {DNNL_ARG_DST_LAYER, outputMemInit.getMemory()},
      {DNNL_ARG_DST_ITER, hiddenOutMemInit.getMemory()}};

  dnnl::memory workspace;
  std::vector<dnnl::primitive> network;
  std::vector<std::unordered_map<int, dnnl::memory>> fwdArgs;

  network.push_back(
      dnnl::reorder(weightsInputMemRawInit.getMemory(), weightsInputMemInit));
  fwdArgs.push_back(
      {{DNNL_ARG_FROM, weightsInputMemRawInit.getMemory()},
       {DNNL_ARG_TO, weightsInputMemInit}});
  network.push_back(dnnl::reorder(
      weightsHiddenMemRawInit.getMemory(), weightsHiddenMemInit));
  fwdArgs.push_back(
      {{DNNL_ARG_FROM, weightsHiddenMemRawInit.getMemory()},
       {DNNL_ARG_TO, weightsHiddenMemInit}});

  if (mode == RnnMode::RELU || mode == RnnMode::TANH) {
    auto vanillaPd = dnnl::vanilla_rnn_forward::primitive_desc(
        dnnlEngine, kind, activation, direction,
        inputMemInit.getDescriptor(), hiddenInMemInit.getDescriptor(),
        weightsInputMemDesc, weightsHiddenMemDesc,
        biasMemInit.getDescriptor(), outputMemInit.getDescriptor(),
        hiddenOutMemInit.getDescriptor());
    network.push_back(dnnl::vanilla_rnn_forward(vanillaPd));
    workspace = dnnl::memory(vanillaPd.workspace_desc(), dnnlEngine);
  } else if (mode == RnnMode::LSTM) {
    detail::DnnlMemoryWrapper cellInMemInit;
    if (!cellState.isEmpty()) {
      contCellState = cellState.asContiguousTensor();
      cellInMemInit = detail::DnnlMemoryWrapper(
          contCellState, {cDims}, ldnc, dnnlEngine);
    }
    detail::DnnlMemoryWrapper cellOutMemInit(cy, cDims, ldnc, dnnlEngine);
    auto lstmPd = dnnl::lstm_forward::primitive_desc(
        dnnlEngine, kind, direction,
        inputMemInit.getDescriptor(), hiddenInMemInit.getDescriptor(),
        cellInMemInit.getDescriptor(), weightsInputMemDesc,
        weightsHiddenMemDesc, biasMemInit.getDescriptor(),
        outputMemInit.getDescriptor(), hiddenOutMemInit.getDescriptor(),
        cellOutMemInit.getDescriptor());
    network.push_back(dnnl::lstm_forward(lstmPd));
    workspace = dnnl::memory(lstmPd.workspace_desc(), dnnlEngine);
    rnnFwdArgs.insert({DNNL_ARG_SRC_ITER_C, cellInMemInit.getMemory()});
    rnnFwdArgs.insert({DNNL_ARG_DST_ITER_C, cellOutMemInit.getMemory()});
  } else if (mode == RnnMode::GRU) {
    auto gruPd = dnnl::lbr_gru_forward::primitive_desc(
        dnnlEngine, kind, direction,
        inputMemInit.getDescriptor(), hiddenInMemInit.getDescriptor(),
        weightsInputMemDesc, weightsHiddenMemDesc,
        biasMemInit.getDescriptor(), outputMemInit.getDescriptor(),
        hiddenOutMemInit.getDescriptor());
    network.push_back(dnnl::lbr_gru_forward(gruPd));
    workspace = dnnl::memory(gruPd.workspace_desc(), dnnlEngine);
  }
  rnnFwdArgs.insert({DNNL_ARG_WORKSPACE, workspace});
  fwdArgs.push_back(rnnFwdArgs);

  // Collect SharedData to keep tensor buffers alive during async execution.
  std::vector<std::shared_ptr<CPUTensor::SharedData>> keepAlive;
  keepAlive.push_back(toCPUTensor(contInput).getSharedData());
  keepAlive.push_back(toCPUTensor(y).getSharedData());
  keepAlive.push_back(toCPUTensor(hy).getSharedData());
  keepAlive.push_back(toCPUTensor(contWeightsInput).getSharedData());
  keepAlive.push_back(toCPUTensor(contWeightsHidden).getSharedData());
  keepAlive.push_back(toCPUTensor(contBias).getSharedData());
  if (!hiddenState.isEmpty()) {
    keepAlive.push_back(toCPUTensor(contHiddenState).getSharedData());
  }
  if (mode == RnnMode::LSTM && !cellState.isEmpty()) {
    keepAlive.push_back(toCPUTensor(contCellState).getSharedData());
  }
  if (mode == RnnMode::LSTM) {
    keepAlive.push_back(toCPUTensor(cy).getSharedData());
  }

  detail::executeNetwork(
      std::move(network), std::move(fwdArgs),
      asyncStream, nativeStream,
      std::move(keepAlive));

  state->workspace = workspace;
  state->weightsLayerMem = weightsInputMemInit;
  state->weightsIterMem = weightsHiddenMemInit;
  state->config = {
      mode, numGates, directionMult, numLayers, hiddenSize, inSize,
      batchSize, seqLength};
  state->activation = activation;
  state->direction = direction;
  state->y = y;
  state->hy = hy;
  state->cy = cy;

  result.y = y;
  result.hy = hy;
  result.cy = cy;
  result.state = state;
  return result;
}

} // namespace

std::tuple<Tensor, Tensor, Tensor, ForwardBackwardState> rnnForward(
    const Tensor& inputOrig,
    const Tensor& hiddenStateOrig,
    const Tensor& cellStateOrig,
    const Tensor& weightsOrig,
    int hiddenSize,
    int numLayers,
    RnnMode mode,
    bool bidirectional,
    float dropout,
    bool /*wantState*/,
    const dnnl::engine& engine,
    AsyncCPUStream& asyncStream,
    dnnl::stream& nativeStream) {
  if (dropout > 0.0) {
    throw std::invalid_argument("onednn RNN: dropout > 0.0 unsupported");
  }

  auto origType = inputOrig.type();

  auto direction = bidirectional
      ? dnnl::rnn_direction::bidirectional_concat
      : dnnl::rnn_direction::unidirectional_left2right;
  int directionMult = bidirectional ? 2 : 1;
  int numGates = 1;
  auto activation = dnnl::algorithm::undef;
  switch (mode) {
    case RnnMode::LSTM: numGates = 4; break;
    case RnnMode::GRU: numGates = 3; break;
    case RnnMode::RELU: activation = dnnl::algorithm::eltwise_relu; break;
    case RnnMode::TANH: activation = dnnl::algorithm::eltwise_tanh; break;
    default: break;
  }

  int inSize = inputOrig.dim(2);
  bool canSingleCall =
      (numLayers == 1) || (inSize == hiddenSize * directionMult);

  // Forward dispatch:
  // - f64: manual forward (OneDNN only supports f32/bf16)
  // - f32/bf16: OneDNN forward (hardware-accelerated)
  auto runFwd = [&](const Tensor& inp, const Tensor& hx, const Tensor& cx,
                     const Tensor& wI, const Tensor& wH, const Tensor& b,
                     int nLayers) -> RnnResult {
    if (origType == fl::dtype::f64) {
      // Use native CPU implementation
      auto nativeResult = fl::native::manualRnnForward<double>(
          inp, hx, cx, wI, wH, b, hiddenSize, mode, numGates, directionMult);
      // Wrap into OneDNN state
      RnnResult r;
      r.y = nativeResult.y;
      r.hy = nativeResult.hy;
      r.cy = nativeResult.cy;
      auto state = std::make_shared<OneDnnRnnForwardState>();
      state->config = nativeResult.config;
      state->activation = activation;
      state->direction = direction;
      state->y = r.y;
      state->hy = r.hy;
      state->cy = r.cy;
      r.state = state;
      return r;
    }
    // Cast to f32 for OneDNN if needed
    auto castIfNeeded = [](const Tensor& t) {
      return (!t.isEmpty() && t.type() != fl::dtype::f32)
          ? t.astype(fl::dtype::f32)
          : t;
    };
    return rnnImpl(
        castIfNeeded(inp), castIfNeeded(hx),
        cx.isEmpty() ? cx : castIfNeeded(cx),
        castIfNeeded(wI), castIfNeeded(wH), castIfNeeded(b),
        hiddenSize, nLayers, mode, activation,
        numGates, direction, directionMult,
        dnnl::prop_kind::forward_training, dropout,
        engine, asyncStream, nativeStream);
  };
  const auto& input = inputOrig;
  const auto& hiddenState = hiddenStateOrig;
  const auto& cellState = cellStateOrig;
  const auto& weights = weightsOrig;

  auto parsedWeights = fl::native::parseWeights(
      weights, mode, numLayers, directionMult, inSize, numGates, hiddenSize);

  RnnResult result;
  if (canSingleCall) {
    result = runFwd(
        input, hiddenState, cellState,
        parsedWeights.weightsInput, parsedWeights.weightsHidden,
        parsedWeights.bias, numLayers);
  } else {
    // Multi-layer split: first layer has different input size
    Tensor hx0, cx0;
    if (!hiddenState.isEmpty()) {
      hx0 = hiddenState(fl::range(0, directionMult));
    }
    if (!cellState.isEmpty()) {
      cx0 = cellState(fl::range(0, directionMult));
    }

    RnnResult resultL1 = runFwd(
        input, hx0, cx0,
        parsedWeights.weightsInput1L, parsedWeights.weightsHidden1L,
        parsedWeights.bias1L, 1);

    Tensor hxRest, cxRest;
    if (!hiddenState.isEmpty()) {
      hxRest = hiddenState(fl::range(directionMult, fl::end));
    }
    if (!cellState.isEmpty()) {
      cxRest = cellState(fl::range(directionMult, fl::end));
    }

    RnnResult resultL2N = runFwd(
        resultL1.y, hxRest, cxRest,
        parsedWeights.weightsInput, parsedWeights.weightsHidden,
        parsedWeights.bias, numLayers - 1);

    result.y = resultL2N.y;
    result.hy = fl::concatenate(0, resultL1.hy, resultL2N.hy);
    if (mode == RnnMode::LSTM) {
      result.cy = fl::concatenate(0, resultL1.cy, resultL2N.cy);
    }
    auto combinedState = std::make_shared<OneDnnRnnForwardState>();
    combinedState->splitLayers = true;
    combinedState->stateL0 = resultL1.state;
    combinedState->stateL1N = resultL2N.state;
    combinedState->forwardInput = input;
    combinedState->config = {
        mode, numGates, directionMult, numLayers, hiddenSize, inSize, 0, 0};
    combinedState->activation = activation;
    combinedState->direction = direction;
    result.state = combinedState;
  }

  if (result.state) {
    result.state->parsedWeights = parsedWeights;
  }

  ForwardBackwardState fbState = result.state
      ? result.state
      : nullptr;
  return std::make_tuple(
      std::move(result.y), std::move(result.hy), std::move(result.cy),
      std::move(fbState));
}

std::tuple<Tensor, Tensor, Tensor, Tensor> rnnBackward(
    const Tensor& inputOrig,
    const Tensor& hiddenStateOrig,
    const Tensor& cellStateOrig,
    const Tensor& weightsOrig,
    const Tensor& dyOrig,
    const Tensor& dhyOrig,
    const Tensor& dcyOrig,
    const Tensor& output,
    int numLayers,
    int hiddenSize,
    RnnMode mode,
    bool bidirectional,
    float dropProb,
    const ForwardBackwardState& fbState) {
  if (!fbState) {
    throw std::invalid_argument(
        "onednn rnnBackward: null ForwardBackwardState");
  }
  auto stateRaw = std::static_pointer_cast<OneDnnRnnForwardState>(fbState);

  auto origType = inputOrig.type();

  // All backward goes through manual BPTT (OneDNN 3.x CPU doesn't
  // support RNN backward primitives).
  const auto& input = inputOrig;
  const auto& hiddenState = hiddenStateOrig;
  const auto& cellState = cellStateOrig;
  const auto& weights = weightsOrig;
  const auto& dy = dyOrig;
  const auto& dhy = dhyOrig;
  const auto& dcy = dcyOrig;

  int directionMult = bidirectional ? 2 : 1;
  int inSize = input.dim(2);
  int numGates = 1;
  switch (mode) {
    case RnnMode::LSTM: numGates = 4; break;
    case RnnMode::GRU: numGates = 3; break;
    default: break;
  }

  auto runBwd = [&](const Tensor& inp, const Tensor& hx, const Tensor& cx,
                     const Tensor& wI, const Tensor& wH, const Tensor& b,
                     const Tensor& dyIn, const Tensor& dhyIn,
                     const Tensor& dcyIn,
                     const std::shared_ptr<OneDnnRnnForwardState>& st,
                     int nLayers)
      -> fl::native::NativeRnnBackwardResult {
    if (origType == fl::dtype::f64) {
      return fl::native::manualRnnBackward<double>(
          inp, hx, cx, wI, wH, b, dyIn, dhyIn, dcyIn, st->config);
    } else {
      return fl::native::manualRnnBackward<float>(
          inp.astype(fl::dtype::f32), hx.astype(fl::dtype::f32),
          cx.isEmpty() ? cx : cx.astype(fl::dtype::f32),
          wI.astype(fl::dtype::f32), wH.astype(fl::dtype::f32),
          b.astype(fl::dtype::f32),
          dyIn.astype(fl::dtype::f32), dhyIn.astype(fl::dtype::f32),
          dcyIn.isEmpty() ? dcyIn : dcyIn.astype(fl::dtype::f32),
          st->config);
    }
  };

  if (stateRaw->splitLayers) {
    auto stateL0 = stateRaw->stateL0;
    auto stateL1N = stateRaw->stateL1N;

    auto parsedWeights = fl::native::parseWeights(
        weights, mode, numLayers, directionMult, inSize, numGates, hiddenSize);

    Tensor dhyL0 = dhy.isEmpty() ? Tensor() : dhy(fl::range(0, directionMult));
    Tensor dhyL1N =
        dhy.isEmpty() ? Tensor() : dhy(fl::range(directionMult, fl::end));
    Tensor dcyL0, dcyL1N;
    if (mode == RnnMode::LSTM) {
      dcyL0 = dcy.isEmpty() ? Tensor() : dcy(fl::range(0, directionMult));
      dcyL1N =
          dcy.isEmpty() ? Tensor() : dcy(fl::range(directionMult, fl::end));
    }

    Tensor hxL0 = hiddenState.isEmpty()
        ? Tensor()
        : hiddenState(fl::range(0, directionMult));
    Tensor hxL1N = hiddenState.isEmpty()
        ? Tensor()
        : hiddenState(fl::range(directionMult, fl::end));
    Tensor cxL0, cxL1N;
    if (mode == RnnMode::LSTM) {
      cxL0 = cellState.isEmpty()
          ? Tensor()
          : cellState(fl::range(0, directionMult));
      cxL1N = cellState.isEmpty()
          ? Tensor()
          : cellState(fl::range(directionMult, fl::end));
    }

    auto resL1N = runBwd(
        stateL0->y, hxL1N, cxL1N,
        parsedWeights.weightsInput, parsedWeights.weightsHidden,
        parsedWeights.bias,
        dy, dhyL1N, dcyL1N, stateL1N, numLayers - 1);

    auto resL0 = runBwd(
        input, hxL0, cxL0,
        parsedWeights.weightsInput1L, parsedWeights.weightsHidden1L,
        parsedWeights.bias1L,
        resL1N.diffInput, dhyL0, dcyL0, stateL0, 1);

    auto combinedDiffWL = fl::concatenate(
        0, resL0.diffWeightsLayerFlat, resL1N.diffWeightsLayerFlat);
    auto combinedDiffWH = fl::concatenate(
        0, resL0.diffWeightsIterFlat, resL1N.diffWeightsIterFlat);
    auto combinedDiffBias =
        fl::concatenate(0, resL0.diffBiasFlat, resL1N.diffBiasFlat);

    auto dweights = fl::native::reconstructFlatWeightGrad(
        combinedDiffWL, combinedDiffWH, combinedDiffBias,
        mode, numLayers, directionMult, inSize, numGates, hiddenSize,
        input.type());

    auto diffHx = fl::concatenate(0, resL0.diffHx, resL1N.diffHx);
    Tensor diffCx;
    if (mode == RnnMode::LSTM) {
      diffCx = fl::concatenate(0, resL0.diffCx, resL1N.diffCx);
    }

    return {resL0.diffInput, diffHx, diffCx, dweights};
  }

  // Non-split case
  auto parsedWeights = fl::native::parseWeights(
      weights, mode, numLayers, directionMult, inSize, numGates, hiddenSize);

  auto res = runBwd(
      input, hiddenState, cellState,
      parsedWeights.weightsInput, parsedWeights.weightsHidden,
      parsedWeights.bias, dy, dhy, dcy,
      stateRaw, numLayers);

  auto dweights = fl::native::reconstructFlatWeightGrad(
      res.diffWeightsLayerFlat, res.diffWeightsIterFlat, res.diffBiasFlat,
      mode, numLayers, directionMult, inSize, numGates, hiddenSize,
      input.type());

  return {res.diffInput, res.diffHx, res.diffCx, dweights};
}

} // namespace onednn
} // namespace detail
} // namespace fl
