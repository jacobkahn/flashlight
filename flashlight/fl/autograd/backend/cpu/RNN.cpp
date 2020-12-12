/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <dnnl.hpp>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/autograd/backend/cpu/DnnlUtils.h"
#include "flashlight/fl/common/DevicePtr.h"

namespace fl {
namespace {

struct ParsedWeightsAndBias {
  // First layer - will be empty if inSize == hiddenSize
  af::array weightsInput1L;
  af::array weightsHidden1L;
  af::array bias1L;
  // All other layers
  af::array weightsInput;
  af::array weightsHidden;
  af::array bias;
};

ParsedWeightsAndBias parseWeights(
    const af::array& weights,
    RnnMode mode,
    int numLayers,
    int directionMult,
    int inSize,
    int numGates,
    int hiddenSize) {
  ParsedWeightsAndBias out;

  // Per-layer sizes for weightsInput and weightsHidden.
  // If inSize == hiddenSize, then weightsInputSize == weightsHiddenSize for all
  // layers, else all but the first layer
  size_t weightsInputSize1L = directionMult * inSize * numGates * hiddenSize;
  size_t weightsHiddenSize = directionMult * hiddenSize * numGates * hiddenSize;
  size_t weightsInputSize = weightsHiddenSize;
  size_t biasSize = numLayers * directionMult * numGates * hiddenSize;

  bool firstLayerDifferent = inSize != hiddenSize;
  // Adjusted if skipping first layer parsing
  int numWeightsLayers = firstLayerDifferent ? numLayers - 1 : numLayers;
  int weightsOffset =
      firstLayerDifferent ? weightsInputSize1L + weightsHiddenSize : 0;
  // If skipping the first layer, parse then skip over the first layer
  // weights and parse the remaining layers. Parsing all bias layers is still
  // fine since biases for each layer have the same size
  if (firstLayerDifferent) {
    out.weightsInput1L = weights(af::seq(weightsInputSize1L));
    out.weightsHidden1L = weights(af::seq(
        weightsInputSize1L, weightsInputSize1L + weightsHiddenSize - 1));
  }

  auto weightsFlat = af::flat(weights).as(weights.type());
  // cuDNN RNN weights, for each layer, are arranged with a chunk of
  // input-hidden weights for each layer followed by a chunk of hidden-hidden
  // weights for each layer:
  // {[layers x [hiddenSize, inputSize]], [layers x  [hiddenSize, hiddenSize]]}
  // Rearrange this to what oneDNN expects (or will reorder if not optimal),
  // which is numLayers chunks of two chunks containing input-hidden and
  // hidden-hidden:
  // {[layers x [[hiddenSize x inSize], [hiddenSize x hiddenSize]]]}
  // Note that the loop is over the total number of layers in case we're doing a
  // single-layer operation where input size and hidden size are different but
  // we'll call another primitive with the output of that first layer as the
  // input to the next layers
  auto weightsInput = af::array(0, weights.type());
  auto weightsHidden = af::array(0, weights.type());
  af::array weightsFlatOffset = weightsFlat(af::seq(weightsOffset, af::end));
  // Specifically ignore the first layer's weights, so inSize == hiddenSize
  for (size_t i = 0; i < numWeightsLayers; ++i) {
    // number of input/hidden weights
    // TODO: Will change for bidirectional, GRU and LSTM (different numGates)
    int chunkSize = hiddenSize * hiddenSize;
    // weights per layer
    int layerChunkSize = chunkSize + chunkSize;

    // Grab input-hidden weights and chunk them together
    auto inputWeightsChunk = weightsFlatOffset(
        af::seq(layerChunkSize * i, layerChunkSize * i + chunkSize - 1));
    weightsInput = af::join(2, weightsInput, inputWeightsChunk);

    // Grab hidden-hidden weights and chunk them together
    int hiddenWeightsSize = hiddenSize * hiddenSize;
    auto inputHiddenChunk = weightsFlatOffset(af::seq(
        layerChunkSize * i + chunkSize,
        layerChunkSize * i + chunkSize + chunkSize - 1));
    weightsHidden = af::join(2, weightsHidden, inputHiddenChunk);
  }
  out.weightsInput = weightsInput;
  out.weightsHidden = weightsHidden;

  // Reduce the weights to form biases. cuDNN uses two separate bias terms:
  // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNMode_t -
  // oneDNN expects only one bias term. Sum together the coefficients for both
  // bias terms to get a single bias term for oneDNN. The gradients for each
  // term can be computed as one since the gradients with respect to the bias
  // subarrays will simply be half of the computed gradient with oneDNN
  af::array bias(0, weights.type());
  size_t biasStartOffset = numLayers * weightsHiddenSize +
      (numLayers - 1) * weightsInputSize + weightsInputSize1L;
  // In vanilla RNN modes, the biases can be simply added:
  // two biases for each bias in fl cuDNN with CUDNN_RNN_DOUBLE_BIAS (default)
  int numBiases = 2;
  // First, grab a subarray which contains only both bias terms; then add them
  af::array biasFlat = weightsFlat(af::seq(biasStartOffset, af::end));
  af::print("biasFlat", biasFlat);
  // Layout is:
  // {numLayers x [numBiases x [bias shape]]}
  for (size_t i = 0; i < numLayers; ++i) {
    // this will definitely change with LSTM/GRU
    // int layerStride = (inSize + hiddenSize) * numBiases; //
    // The number of bias terms in the tensor per-layer
    int layerStride = biasSize / numLayers * numBiases;
    auto biases1 = biasFlat(af::seq(
        layerStride * i, layerStride * i + layerStride / numBiases - 1));
    auto biases2 = biasFlat(af::seq(
        layerStride * i + layerStride / numBiases, layerStride * (i + 1) - 1));
    auto layerBiasCombined = biases1 + biases2;
    bias = af::join(0, bias, layerBiasCombined);
  }

  if (firstLayerDifferent) {
    out.bias1L = bias(af::seq(biasSize / numLayers));
    // bias for the second --> last layer
    bias = bias(af::seq(biasSize / numLayers, af::end));
  }
  out.bias = bias;

  return out;
}

// TODO, implement RNN, also check to ensure there will appropriate checks to
// guard the use of half precision in case CPU implementation doesn't support
// it.
/*
 * Does forward for a single RNN primitive
 */
std::tuple<Variable, Variable, Variable> rnnImpl(
    const af::array& input,
    const af::array& hiddenState,
    const af::array& cellState,
    const af::array& weightsInput,
    const af::array& weightsHidden,
    const af::array& bias,
    int hiddenSize,
    int numLayers,
    RnnMode mode,
    dnnl::algorithm activation,
    int numGates,
    dnnl::rnn_direction direction,
    int directionMult,
    dnnl::prop_kind kind,
    float dropout) {
  auto dnnlEngine = detail::DnnlEngine::getInstance().getEngine();

  // Dimensions
  int inSize = input.dims(0);
  int batchSize = input.dims(1);
  int seqLength = input.dims(2);
  dnnl::memory::dims inputDims = {seqLength, batchSize, inSize};
  dnnl::memory::dims outputDims = {
      seqLength, batchSize, hiddenSize * directionMult};
  auto dType = detail::dnnlMapToType(input.type());
  int totalLayers = numLayers * directionMult;
  int outSize = hiddenSize * directionMult;
  dnnl::memory::dims hDims = {totalLayers, 1, batchSize, hiddenSize};
  dnnl::memory::dims cDims = {totalLayers, 1, batchSize, hiddenSize};
  dnnl::memory::dims biasDims = {
      numLayers, directionMult, numGates, hiddenSize};
  // ldigo
  dnnl::memory::dims weightsInputDims = {
      numLayers, directionMult, inSize, numGates, hiddenSize};
  dnnl::memory::dims weightsHiddenDims = {
      numLayers, directionMult, hiddenSize, numGates, hiddenSize};

  // Out tensors: output (y), hidden state output (hy), cell state output (cy)
  auto y = af::array(outSize, batchSize, seqLength, input.type());
  auto hy = af::array(hiddenSize, batchSize, totalLayers, input.type());
  af::array cy;
  if (mode == RnnMode::LSTM) {
    auto cy = af::array(hy.dims(), input.type());
  }

  // Memory for forward
  // input
  DevicePtr inputPtr(input);
  auto inputMemDesc =
      dnnl::memory::desc(inputDims, dType, dnnl::memory::format_tag::tnc);
  auto inputMemInit = dnnl::memory(inputMemDesc, dnnlEngine, inputPtr.get());
  // output
  DevicePtr outputPtr(y);
  auto outputMemDesc =
      dnnl::memory::desc(outputDims, dType, dnnl::memory::format_tag::tnc);
  auto outputMemInit = dnnl::memory(outputMemDesc, dnnlEngine, outputPtr.get());
  // input hidden state
  DevicePtr hiddenInPtr(hiddenState);
  dnnl::memory::desc hiddenInMemDesc;
  dnnl::memory hiddenInMemInit;
  if (!hiddenState.isempty()) {
    hiddenInMemDesc =
        dnnl::memory::desc(hDims, dType, dnnl::memory::format_tag::ldnc);
    hiddenInMemInit =
        dnnl::memory(hiddenInMemDesc, dnnlEngine, hiddenInPtr.get());
  } else {
    std::cout << "Hidden state empty!" << std::endl;
    hiddenInMemDesc = dnnl::memory::desc();
    hiddenInMemInit = dnnl::memory();
  }
  // output hidden state
  DevicePtr hiddenOutPtr(hy);
  auto hiddenOutMemDesc =
      dnnl::memory::desc(hDims, dType, dnnl::memory::format_tag::ldnc);
  auto hiddenOutMemInit =
      dnnl::memory(hiddenOutMemDesc, dnnlEngine, hiddenOutPtr.get());

  DevicePtr weightsInputPtr(weightsInput);
  // TODO(jacobkahn): don't force a format tag - use any and do a reorder based
  // on the format of the primitive - what it says - like you're supposed to
  auto weightsInputMemDesc = dnnl::memory::desc(
      weightsInputDims, dType, dnnl::memory::format_tag::ldigo);
  auto weightsInputMemInit = dnnl::memory(weightsInputMemDesc, dnnlEngine);
  // Primitive for reordering input weights: ldgoi --> ldigo
  auto weightsInputMemRawDesc = dnnl::memory::desc(
      weightsInputDims, dType, dnnl::memory::format_tag::ldgoi);
  auto weightsInputMemRawInit =
      dnnl::memory(weightsInputMemRawDesc, dnnlEngine, weightsInputPtr.get());

  DevicePtr weightsHiddenPtr(weightsHidden);
  auto weightsHiddenMemDesc = dnnl::memory::desc(
      weightsHiddenDims, dType, dnnl::memory::format_tag::ldigo);
  auto weightsHiddenMemInit = dnnl::memory(weightsHiddenMemDesc, dnnlEngine);
  // Primitive for reordering iter/hidden weights: ldgoi --> ldigo
  auto weightsHiddenMemRawDesc = dnnl::memory::desc(
      weightsHiddenDims, dType, dnnl::memory::format_tag::ldgoi);
  auto weightsHiddenMemRawInit =
      dnnl::memory(weightsHiddenMemRawDesc, dnnlEngine, weightsHiddenPtr.get());

  DevicePtr biasPtr(bias);
  auto biasMemDesc =
      dnnl::memory::desc(biasDims, dType, dnnl::memory::format_tag::ldgo);
  auto biasMemInit = dnnl::memory(biasMemDesc, dnnlEngine, biasPtr.get());

  // Add arguments
  std::unordered_map<int, dnnl::memory> rnnFwdArgs = {
      {DNNL_ARG_SRC_LAYER, inputMemInit},
      {DNNL_ARG_SRC_ITER, hiddenInMemInit},
      {DNNL_ARG_WEIGHTS_LAYER, weightsInputMemInit},
      {DNNL_ARG_WEIGHTS_ITER, weightsHiddenMemInit},
      {DNNL_ARG_BIAS, biasMemInit},
      {DNNL_ARG_DST_LAYER, outputMemInit},
      {DNNL_ARG_DST_ITER, hiddenOutMemInit}};

  // Workspace memory, if needed
  dnnl::memory workspace;
  std::vector<dnnl::primitive> network;
  std::vector<std::unordered_map<int, dnnl::memory>> fwdArgs;

  // reorder input weights
  network.push_back(dnnl::reorder(weightsInputMemRawInit, weightsInputMemInit));
  fwdArgs.push_back({{DNNL_ARG_FROM, weightsInputMemRawInit},
                     {DNNL_ARG_TO, weightsInputMemInit}});
  // reorder iter weights
  network.push_back(
      dnnl::reorder(weightsHiddenMemRawInit, weightsHiddenMemInit));
  fwdArgs.push_back({{DNNL_ARG_FROM, weightsHiddenMemRawInit},
                     {DNNL_ARG_TO, weightsHiddenMemInit}});

  // Initialize descriptors
  if (mode == RnnMode::RELU || mode == RnnMode::TANH) {
    auto vanilla = dnnl::vanilla_rnn_forward::desc(
        kind,
        activation,
        direction,
        inputMemDesc,
        hiddenInMemDesc,
        weightsInputMemDesc, // weights "layer"
        weightsHiddenMemDesc, // weights "iter"
        biasMemDesc,
        outputMemDesc,
        hiddenOutMemDesc);
    auto vanillaPd =
        dnnl::vanilla_rnn_forward::primitive_desc(vanilla, dnnlEngine);
    network.push_back(dnnl::vanilla_rnn_forward(vanillaPd));
    workspace = dnnl::memory(vanillaPd.workspace_desc(), dnnlEngine);

  } else if (mode == RnnMode::LSTM) {
    // LSTM-only
    // input cell state
    // TODO(jacobkahn): function that takes the array and
    // returns the desciptor and memory -- takes an argument for
    // which determines whether or not it's ok to return empty
    // descriptors if the array is empty
    DevicePtr cellInPtr(cellState);
    dnnl::memory::desc cellInMemDesc;
    dnnl::memory cellInMemInit;
    if (!cellState.isempty()) {
      cellInMemDesc =
          dnnl::memory::desc({cDims}, dType, dnnl::memory::format_tag::ldnc);
      cellInMemInit = dnnl::memory(cellInMemDesc, dnnlEngine, cellInPtr.get());
    } else {
      cellInMemDesc = dnnl::memory::desc();
      cellInMemInit = dnnl::memory();
    }

    // output cell state
    DevicePtr cellOutPtr(cy);
    auto cellOutMemDesc =
        dnnl::memory::desc({cDims}, dType, dnnl::memory::format_tag::ldnc);
    auto cellOutMemInit =
        dnnl::memory(cellOutMemDesc, dnnlEngine, cellOutPtr.get());

    auto lstm = dnnl::lstm_forward::desc(
        kind,
        direction,
        inputMemDesc,
        hiddenInMemDesc,
        cellInMemDesc,
        weightsInputMemDesc, // weights "layer"
        weightsHiddenMemDesc, // weights "iter"
        biasMemDesc,
        outputMemDesc,
        hiddenOutMemDesc,
        cellOutMemDesc);
    auto lstmPd = dnnl::lstm_forward::primitive_desc(lstm, dnnlEngine);
    network.push_back(dnnl::lstm_forward(lstmPd));
    workspace = dnnl::memory(lstmPd.workspace_desc(), dnnlEngine);
    // TODO: add cell memory arguments to fwd Args
    rnnFwdArgs.insert({DNNL_ARG_SRC_ITER_C, cellInMemInit});
    rnnFwdArgs.insert({DNNL_ARG_DST_ITER_C, cellOutMemInit});

  } else if (mode == RnnMode::GRU) {
    // Use linear-before-reset so we can have parity with cuDNN
    auto gru = dnnl::lbr_gru_forward::desc(
        kind,
        direction,
        inputMemDesc,
        hiddenInMemDesc,
        weightsInputMemDesc,
        weightsHiddenMemDesc,
        biasMemDesc,
        outputMemDesc,
        hiddenOutMemDesc);
    auto gruPd = dnnl::lbr_gru_forward::primitive_desc(gru, dnnlEngine);
    network.push_back(dnnl::lbr_gru_forward(gruPd));
    workspace = dnnl::memory(gruPd.workspace_desc(), dnnlEngine);
  }
  rnnFwdArgs.insert({DNNL_ARG_WORKSPACE, workspace});

  fwdArgs.push_back(rnnFwdArgs);

  detail::executeNetwork(network, fwdArgs);

  // Variable yv(y, {dummy}, dyGradFunc);
  // Variable hyv(hy, {dummy}, dhyGradFunc);
  // Variable cyv(cy, {dummy}, dcyGradFunc);
  Variable yv(y, false);
  Variable hyv(hy, false);
  Variable cyv(cy, false);
  return std::make_tuple(yv, hyv, cyv);
}

} // namespace

std::tuple<Variable, Variable, Variable> rnn(
    const Variable& inputV,
    const Variable& hiddenStateV,
    const Variable& cellStateV,
    const Variable& weightsV,
    int hiddenSize,
    int numLayers,
    RnnMode mode,
    bool bidirectional,
    float dropout) {
  // Constants
  auto direction = bidirectional
      ? dnnl::rnn_direction::bidirectional_concat
      : dnnl::rnn_direction::unidirectional_left2right;
  int directionMult = bidirectional ? 2 : 1;
  auto kind = (inputV.isCalcGrad() || weightsV.isCalcGrad())
      ? dnnl::prop_kind::forward_training
      : dnnl::prop_kind::forward_inference;
  int numGates = 1;
  auto activation = dnnl::algorithm::undef;
  switch (mode) {
    case RnnMode::LSTM:
      numGates = 4;
      break;
    case RnnMode::GRU:
      numGates = 3;
      break;
    case RnnMode::RELU:
      activation = dnnl::algorithm::eltwise_relu;
      break;
    case RnnMode::TANH:
      activation = dnnl::algorithm::eltwise_tanh;
    default:
      break;
  }

  auto& input = inputV.array();
  auto& hiddenState = hiddenStateV.array();
  auto& cellState = cellStateV.array();
  auto& weights = weightsV.array();
  int inSize = input.dims(0);

  // In flashlight, all RNN weights are stored as one contiguous tensor, so we
  // have to parse out the input weights, input biases, hidden weights, and
  // hidden biases from one tensor. Order doesn't matter since the arrangement
  // is a black box
  auto parsedWeights = parseWeights(
      weights, mode, numLayers, directionMult, inSize, numGates, hiddenSize);

  // The oneDNN RNN primitive has an API limitation where input size and
  // hidden size can only differ if the primitive has exactly one layer.
  // Therefore, for computations for more than one layer, first do the
  // operation for one layer, which gives an output vector of size [hidden
  // size, batch size, sequence length * number of directions], then use
  // that output as the input for layers [2, L]. Since the input size dim 0
  // is now the hidden size, the primitive can fuse computation for
  // arbitrarily-many layers.
  if (inputV.dims(0) == hiddenSize || numLayers == 1) {
    std::cout << "------ In single kernel case" << std::endl;
    // Input and hidden size are the same, or we only have one layer, which
    // means we can call the impl as is and parse weights "normally"
    // TODO: gradFunc will be a mightmare
    return rnnImpl(
        input,
        hiddenState,
        cellState,
        parsedWeights.weightsInput,
        parsedWeights.weightsHidden,
        parsedWeights.bias,
        hiddenSize,
        numLayers,
        mode,
        activation,
        numGates,
        direction,
        directionMult,
        kind,
        dropout);
  } else {
    std::cout << "------ In multi kernel case" << std::endl;
    // We require more than one layer with different input and hidden states -
    // see the above.
    // int firstLayerWeightsSize

    // TODO: change this to array later -- rnnImpl should return arrays
    Variable outL1; // input to layers [2..N]
    Variable hiddenOutL1;
    Variable cellOutL1;
    // Seek to the first layer's hidden/cell state, weights, and bias
    std::tie(outL1, hiddenOutL1, cellOutL1) = rnnImpl(
        input,
        hiddenState(af::span, af::span, 0),
        cellState(af::span, af::span, 0),
        parsedWeights.weightsInput1L,
        parsedWeights.weightsHidden1L,
        parsedWeights.bias1L,
        hiddenSize,
        1,
        mode,
        activation,
        numGates,
        direction,
        directionMult,
        kind,
        dropout);

    /* Layers [2..N] */
    Variable out;
    Variable hiddenOutL2N;
    Variable cellOutL2N;

    // Seek  past the first layer's hidden/cell state, weights, and bias
    std::tie(out, hiddenOutL2N, cellOutL2N) = rnnImpl(
        outL1.array(), // fixme
        hiddenState(af::span, af::span, af::seq(1, af::end)),
        cellState(af::span, af::span, af::seq(0, cellState.dims(2))),
        parsedWeights.weightsInput,
        parsedWeights.weightsHidden,
        parsedWeights.bias,
        hiddenSize,
        numLayers - 1, // layers [2..N]
        mode,
        activation,
        numGates,
        direction,
        directionMult,
        kind,
        dropout);

    // TODO: operate on arrays, not variables
    Variable hiddenOut =
        Variable(af::join(3, hiddenOutL1.array(), hiddenOutL2N.array()), false);
    Variable cellOut =
        Variable(af::join(3, cellOutL1.array(), cellOutL2N.array()), false);
    return std::make_tuple(out, hiddenOut, cellOut);
  }
}

} // namespace fl
