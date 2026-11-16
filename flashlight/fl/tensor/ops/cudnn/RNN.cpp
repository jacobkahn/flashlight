/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/cudnn/CudnnAutogradOps.h"

#include <cuda_runtime.h>
#include <cudnn.h>

#include "flashlight/fl/tensor/ops/cudnn/CudnnTensorHelpers.h"
#include "flashlight/fl/tensor/ops/cudnn/CudnnUtils.h"
#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/tensor/Compute.h"

namespace fl {
namespace detail {
namespace cudnn {

namespace {

// RAII wrapper for cudnnRNNDataDescriptor_t
struct RNNDataDescriptor {
  cudnnRNNDataDescriptor_t descriptor;

  RNNDataDescriptor(
      cudnnDataType_t dataType,
      int seqLength,
      int batchSize,
      int vectorSize,
      const int* seqLengthArray) {
    CUDNN_CHECK_ERR(cudnnCreateRNNDataDescriptor(&descriptor));
    CUDNN_CHECK_ERR(cudnnSetRNNDataDescriptor(
        descriptor,
        dataType,
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
        seqLength,
        batchSize,
        vectorSize,
        seqLengthArray,
        nullptr /* paddingFill */));
  }

  ~RNNDataDescriptor() {
    cudnnDestroyRNNDataDescriptor(descriptor);
  }

  RNNDataDescriptor(const RNNDataDescriptor&) = delete;
  RNNDataDescriptor& operator=(const RNNDataDescriptor&) = delete;
};

// Create a 3D tensor descriptor for hidden/cell states
// Shape: [numLayers * numDirs, batchSize, hiddenSize]
struct HiddenDescriptor {
  cudnnTensorDescriptor_t descriptor;

  HiddenDescriptor(
      cudnnDataType_t dataType,
      int totalLayers,
      int batchSize,
      int hiddenSize) {
    CUDNN_CHECK_ERR(cudnnCreateTensorDescriptor(&descriptor));
    std::array<int, 3> dims = {totalLayers, batchSize, hiddenSize};
    std::array<int, 3> strides = {
        batchSize * hiddenSize,
        hiddenSize,
        1};
    CUDNN_CHECK_ERR(cudnnSetTensorNdDescriptor(
        descriptor, dataType, 3, dims.data(), strides.data()));
  }

  ~HiddenDescriptor() {
    cudnnDestroyTensorDescriptor(descriptor);
  }

  HiddenDescriptor(const HiddenDescriptor&) = delete;
  HiddenDescriptor& operator=(const HiddenDescriptor&) = delete;
};

// Opaque state passed from rnnForward to rnnBackward via ForwardBackwardState.
struct CudnnRnnState : ForwardState {
  Tensor reserveSpace;
};

} // namespace

std::tuple<Tensor, Tensor, Tensor, ForwardBackwardState> rnnForward(
    const Tensor& input,
    const Tensor& hiddenStateIn,
    const Tensor& cellStateIn,
    const Tensor& weights,
    int hiddenSize,
    int numLayers,
    RnnMode mode,
    bool bidirectional,
    float dropProb,
    bool wantState) {
  FL_TENSOR_DTYPES_MATCH_CHECK(input, hiddenStateIn, cellStateIn, weights);

  bool train = wantState;
  auto state = std::make_shared<CudnnRnnState>();

  Tensor x = input.asContiguousTensor();
  Tensor hiddenState = hiddenStateIn.asContiguousTensor();
  Tensor cellState = cellStateIn.asContiguousTensor();

  auto dims = x.shape();
  // Row-major: input is {seqLength, batchSize, inputSize}
  int seqLength = dims.ndim() < 3 ? 1 : dims[0];
  int batchSize = dims.ndim() < 2 ? 1 : (dims.ndim() < 3 ? dims[0] : dims[1]);
  int inputSize = dims[dims.ndim() - 1];

  int totalLayers = numLayers * (bidirectional ? 2 : 1);
  int outSize = hiddenSize * (bidirectional ? 2 : 1);

  DropoutDescriptor dropout(dropProb, input);
  RNNDescriptor rnnDesc(
      input.type(), inputSize, hiddenSize, numLayers, mode, bidirectional,
      dropout);

  // Row-major: hidden state is {totalLayers, batchSize, hiddenSize}
  if (!hiddenState.isEmpty()) {
    auto hxDims = hiddenState.shape();
    int hxTotalLayers = hxDims[0];
    int hxBatchSize = hiddenState.ndim() < 2 ? 1 : hxDims[1];
    int hxHiddenSize = hiddenState.ndim() < 3 ? 1 : hxDims[2];

    if (!(hxHiddenSize == hiddenSize && hxBatchSize == batchSize &&
          hxTotalLayers == totalLayers)) {
      throw std::invalid_argument("invalid hidden state dims for RNN");
    }
  }

  // Row-major: cell state is {totalLayers, batchSize, hiddenSize}
  if (!cellState.isEmpty() &&
      !(mode == RnnMode::LSTM && cellState.dim(0) == totalLayers &&
        cellState.dim(1) == batchSize && cellState.dim(2) == hiddenSize)) {
    throw std::invalid_argument("invalid cell state dims for RNN");
  }

  auto handle = getCudnnHandle();
  const auto& cudnnStream = getCudnnStream();

  // Get weight space size
  size_t weightSpaceSize;
  CUDNN_CHECK_ERR(cudnnGetRNNWeightSpaceSize(
      handle, rnnDesc.descriptor, &weightSpaceSize));
  if (weightSpaceSize != weights.bytes()) {
    throw std::invalid_argument(
        "invalid # of parameters or wrong input shape for RNN");
  }

  // Sequence length array (host) - all sequences same length
  std::vector<int> seqLengthArray(batchSize, seqLength);

  // Device copy of sequence lengths
  int* devSeqLengths = nullptr;
  cudaMalloc(&devSeqLengths, batchSize * sizeof(int));
  cudaMemcpy(
      devSeqLengths, seqLengthArray.data(), batchSize * sizeof(int),
      cudaMemcpyHostToDevice);

  // Create RNN data descriptors
  auto cudnnType = cudnnMapToType(x.type());
  RNNDataDescriptor xDesc(
      cudnnType, seqLength, batchSize, inputSize, seqLengthArray.data());
  RNNDataDescriptor yDesc(
      cudnnType, seqLength, batchSize, outSize, seqLengthArray.data());

  // Hidden/cell state descriptors
  HiddenDescriptor hDesc(cudnnType, totalLayers, batchSize, hiddenSize);
  HiddenDescriptor cDesc(cudnnType, totalLayers, batchSize, hiddenSize);

  // Get temp space sizes
  cudnnForwardMode_t fwdMode =
      train ? CUDNN_FWD_MODE_TRAINING : CUDNN_FWD_MODE_INFERENCE;
  size_t workspaceSize, reserveSize;
  CUDNN_CHECK_ERR(cudnnGetRNNTempSpaceSizes(
      handle, rnnDesc.descriptor, fwdMode, xDesc.descriptor,
      &workspaceSize, &reserveSize));

  // Row-major: output {seqLength, batchSize, outSize}, hidden {totalLayers, batchSize, hiddenSize}
  Tensor y =
      makeTensorLike(input, {seqLength, batchSize, outSize}, input.type());
  Tensor hy =
      makeTensorLike(input, {totalLayers, batchSize, hiddenSize}, x.type());
  Tensor cy;
  if (mode == RnnMode::LSTM) {
    cy = makeTensorLike(input, hy.shape(), x.type());
  }

  Tensor workspace = makeTensorLike(
      input, {static_cast<long long>(workspaceSize)}, fl::dtype::b8);
  state->reserveSpace = makeTensorLike(
      input, {static_cast<long long>(reserveSize)}, fl::dtype::b8);

  {
    auto contiguousX = x.asContiguousTensor();
    auto contiguousWeights = weights.asContiguousTensor();
    DevicePtr xRaw(contiguousX);
    DevicePtr hxRaw(hiddenState);
    DevicePtr cxRaw(cellState);
    DevicePtr wRaw(contiguousWeights);
    DevicePtr yRaw(y);
    DevicePtr hyRaw(hy);
    DevicePtr cyRaw(cy);
    DevicePtr workspaceRaw(workspace);
    DevicePtr reserveSpaceRaw(state->reserveSpace);
    relativeSync(cudnnStream, {
      contiguousX, hiddenState, cellState, contiguousWeights, y, hy, cy,
      workspace, state->reserveSpace,
    });

    CUDNN_CHECK_ERR(cudnnRNNForward(
        handle,
        rnnDesc.descriptor,
        fwdMode,
        devSeqLengths,
        xDesc.descriptor,
        xRaw.get(),
        yDesc.descriptor,
        yRaw.get(),
        hDesc.descriptor,
        hxRaw.get(),
        hyRaw.get(),
        cDesc.descriptor,
        cxRaw.get(),
        cyRaw.get(),
        weightSpaceSize,
        wRaw.get(),
        workspaceSize,
        workspaceRaw.get(),
        reserveSize,
        reserveSpaceRaw.get()));
  }

  cudaFree(devSeqLengths);

  relativeSync({y, hy, cy}, cudnnStream);

  ForwardBackwardState fbState = wantState
      ? state
      : nullptr;
  return std::make_tuple(std::move(y), std::move(hy), std::move(cy), std::move(fbState));
}

std::tuple<Tensor, Tensor, Tensor, Tensor> rnnBackward(
    const Tensor& input,
    const Tensor& hiddenState,
    const Tensor& cellState,
    const Tensor& weights,
    const Tensor& dy,
    const Tensor& dhy,
    const Tensor& dcy,
    const Tensor& output,
    int numLayers,
    int hiddenSize,
    RnnMode mode,
    bool bidirectional,
    float dropProb,
    const ForwardBackwardState& fbState) {
  if (!fbState) {
    throw std::invalid_argument(
        "cudnn::rnnBackward given null ForwardBackwardState");
  }
  auto state = std::static_pointer_cast<CudnnRnnState>(fbState);

  auto handle = getCudnnHandle();
  const auto& cudnnStream = getCudnnStream();

  auto x = input.asContiguousTensor();
  auto& y = output;

  auto dims = x.shape();
  // Row-major: input is {seqLength, batchSize, inputSize}
  int seqLength = dims.ndim() < 3 ? 1 : dims[0];
  int batchSize = dims.ndim() < 2 ? 1 : (dims.ndim() < 3 ? dims[0] : dims[1]);
  int inputSize = dims[dims.ndim() - 1];
  int totalLayers = numLayers * (bidirectional ? 2 : 1);
  int outSize = hiddenSize * (bidirectional ? 2 : 1);

  DropoutDescriptor dropout(dropProb, input);
  RNNDescriptor rnnDesc(
      input.type(), inputSize, hiddenSize, numLayers, mode, bidirectional,
      dropout);

  // Get weight space size
  size_t weightSpaceSize;
  CUDNN_CHECK_ERR(cudnnGetRNNWeightSpaceSize(
      handle, rnnDesc.descriptor, &weightSpaceSize));

  // Sequence length array (host) and device copy
  std::vector<int> seqLengthArray(batchSize, seqLength);
  int* devSeqLengths = nullptr;
  cudaMalloc(&devSeqLengths, batchSize * sizeof(int));
  cudaMemcpy(
      devSeqLengths, seqLengthArray.data(), batchSize * sizeof(int),
      cudaMemcpyHostToDevice);

  auto cudnnType = cudnnMapToType(x.type());
  RNNDataDescriptor xDesc(
      cudnnType, seqLength, batchSize, inputSize, seqLengthArray.data());
  RNNDataDescriptor yDesc(
      cudnnType, seqLength, batchSize, outSize, seqLengthArray.data());
  HiddenDescriptor hDesc(cudnnType, totalLayers, batchSize, hiddenSize);
  HiddenDescriptor cDesc(cudnnType, totalLayers, batchSize, hiddenSize);

  // Get temp space sizes
  size_t workspaceSize, reserveSize;
  CUDNN_CHECK_ERR(cudnnGetRNNTempSpaceSizes(
      handle, rnnDesc.descriptor, CUDNN_FWD_MODE_TRAINING, xDesc.descriptor,
      &workspaceSize, &reserveSize));

  Tensor workspace = makeTensorLike(
      input, {static_cast<long long>(workspaceSize)}, fl::dtype::b8);

  // Fill in zero dy if empty
  Tensor dyFilled =
      dy.isEmpty() ? fullLike(input, y.shape(), 0.0, y.type()) : dy;

  Tensor dx = makeTensorLike(input, input.shape(), input.type());
  Tensor dhx = makeTensorLike(input, hiddenState.shape(), hiddenState.type());
  Tensor dcx = makeTensorLike(input, cellState.shape(), cellState.type());

  DevicePtr yRaw(output);
  DevicePtr workspaceRaw(workspace);
  DevicePtr reserveSpaceRaw(state->reserveSpace);
  relativeSync(cudnnStream, {output, workspace, state->reserveSpace});

  {
    DevicePtr dyRaw(dyFilled);
    DevicePtr dhyRaw(dhy);
    DevicePtr dcyRaw(dcy);

    DevicePtr wRaw(weights);

    DevicePtr hxRaw(hiddenState);
    DevicePtr cxRaw(cellState);

    DevicePtr dxRaw(dx);
    DevicePtr dhxRaw(dhx);
    DevicePtr dcxRaw(dcx);
    relativeSync(
        cudnnStream,
        {dyFilled, dhy, dcy, weights, hiddenState, cellState, dx, dhx, dcx});

    CUDNN_CHECK_ERR(cudnnRNNBackwardData_v8(
        handle,
        rnnDesc.descriptor,
        devSeqLengths,
        yDesc.descriptor,
        yRaw.get(),
        dyRaw.get(),
        xDesc.descriptor,
        dxRaw.get(),
        hDesc.descriptor,
        hxRaw.get(),
        dhyRaw.get(),
        dhxRaw.get(),
        cDesc.descriptor,
        cxRaw.get(),
        dcyRaw.get(),
        dcxRaw.get(),
        weightSpaceSize,
        wRaw.get(),
        workspaceSize,
        workspaceRaw.get(),
        reserveSize,
        reserveSpaceRaw.get()));
  }

  Tensor dw = fullLike(input, weights.shape(), 0, weights.type());

  {
    DevicePtr xRaw(x);
    DevicePtr dwRaw(dw);
    DevicePtr hxRaw(hiddenState);
    relativeSync(cudnnStream, {x, dw, hiddenState});

    CUDNN_CHECK_ERR(cudnnRNNBackwardWeights_v8(
        handle,
        rnnDesc.descriptor,
        CUDNN_WGRAD_MODE_ADD,
        devSeqLengths,
        xDesc.descriptor,
        xRaw.get(),
        hDesc.descriptor,
        hxRaw.get(),
        yDesc.descriptor,
        yRaw.get(),
        weightSpaceSize,
        dwRaw.get(),
        workspaceSize,
        workspaceRaw.get(),
        reserveSize,
        reserveSpaceRaw.get()));
  }

  cudaFree(devSeqLengths);

  relativeSync({dx, dhx, dcx, dw}, cudnnStream);
  return std::make_tuple(dx, dhx, dcx, dw);
}

} // namespace cudnn
} // namespace detail
} // namespace fl

