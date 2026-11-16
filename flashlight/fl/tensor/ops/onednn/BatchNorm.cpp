/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/onednn/OneDnnAutogradOps.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <dnnl.hpp>

#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/tensor/ops/onednn/DnnlUtils.h"
#include "flashlight/fl/tensor/ops/onednn/OneDnnOps.h"
#include "flashlight/fl/tensor/ops/onednn/Utils.h"
#include "flashlight/fl/tensor/Index.h"

namespace fl {
namespace detail {
namespace onednn {

namespace {

constexpr auto formatNCHW = dnnl::memory::format_tag::nchw;
constexpr auto formatX = dnnl::memory::format_tag::x;

int getNfeatures(const Shape& inputShape, const std::vector<int>& axes) {
  int nfeatures = 1;
  for (auto ax : axes) {
    nfeatures *= inputShape.dim(ax);
  }
  return nfeatures;
}

dnnl::memory::dims getInputOutputDims(
    int minAxis, int maxAxis,
    const Tensor& input, int nfeatures) {
  // Row-major layout: input shape is {N, C, H, W, ...}
  // OneDNN batch_normalization normalizes per channel (axis 1 in NCHW).
  // Map the input to 4D NCHW where:
  //   N = product of dims before minAxis (pre-feature batch)
  //   C = nfeatures (product of feature axes dims)
  //   H = product of dims after maxAxis (post-feature spatial)
  //   W = 1
  int preBatch = 1;
  for (int i = 0; i < minAxis; ++i) {
    preBatch *= input.dim(i);
  }
  int postSpatial = 1;
  for (int i = maxAxis + 1; i < input.ndim(); ++i) {
    postSpatial *= input.dim(i);
  }
  return {static_cast<dnnl::memory::dim>(preBatch),
          static_cast<dnnl::memory::dim>(nfeatures),
          static_cast<dnnl::memory::dim>(postSpatial),
          1};
}

// Opaque state for batchnorm forward -> backward
struct OneDnnBatchNormState : ForwardState {
  dnnl::batch_normalization_forward::primitive_desc fwdPrimDesc;
  Tensor weights;
  Tensor bias;
  Tensor mean;  // keep Tensor alive so buffer stays valid for backward
  Tensor var;   // keep Tensor alive so buffer stays valid for backward
  dnnl::memory::dims weightsDims;
  dnnl::memory::dims biasDims;
  dnnl::memory::desc outputMemoryDescriptor;
  dnnl::memory workspace; // required for backward
};

} // namespace

BatchnormForwardResult batchnormForward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& runningMeanIn,
    const Tensor& runningVarIn,
    const std::vector<int>& axes,
    bool train,
    double momentum,
    double epsilon,
    bool wantState,
    const dnnl::engine& dnnlEngine,
    AsyncCPUStream& asyncStream,
    dnnl::stream& nativeStream,
    const dnnl::primitive_attr& attr) {
  if (input.type() == fl::dtype::f16) {
    throw std::runtime_error("OneDNN batchnorm op - f16 inputs not supported.");
  }

  auto state = std::make_shared<OneDnnBatchNormState>();

  auto output = makeCPUTensor(input.shape(), input.type());
  int nfeatures = getNfeatures(input.shape(), axes);

  // Copy running stats so we don't mutate the const inputs
  Tensor runningMean = runningMeanIn.isEmpty()
      ? fullCPU({nfeatures}, 0., input.type())
      : runningMeanIn.copy();
  Tensor runningVar = runningVarIn.isEmpty()
      ? fullCPU({nfeatures}, 1., input.type())
      : runningVarIn.copy();

  auto maxAxis = *std::max_element(axes.begin(), axes.end());
  auto minAxis = *std::min_element(axes.begin(), axes.end());
  bool axesContinuous = (axes.size() == static_cast<size_t>(maxAxis - minAxis + 1));
  if (!axesContinuous) {
    throw std::invalid_argument("axis array should be continuous");
  }

  auto weightNonempty =
      weight.isEmpty() ? fullCPU({nfeatures}, 1., fl::dtype::f32) : weight;
  auto biasNonempty =
      bias.isEmpty() ? fullCPU({nfeatures}, 0., fl::dtype::f32) : bias;

  state->weights = weightNonempty;
  state->bias = biasNonempty;
  // Store copies of mean/var Tensors so their buffers survive until backward
  state->mean = runningMean;
  state->var = runningVar;
  state->weightsDims = detail::convertToDnnlDims({nfeatures});
  state->biasDims = detail::convertToDnnlDims({nfeatures});
  auto inputOutputDims = getInputOutputDims(minAxis, maxAxis, input, nfeatures);

  // DnnlMemoryWrapper format tags assume contiguous layout
  auto contiguousInput = input.asContiguousTensor();
  const detail::DnnlMemoryWrapper inputMemory(
      contiguousInput, inputOutputDims, formatNCHW, dnnlEngine);
  const detail::DnnlMemoryWrapper outputMemory(
      output, inputOutputDims, formatNCHW, dnnlEngine);
  const detail::DnnlMemoryWrapper meanMemory(
      state->mean, {nfeatures}, formatX, dnnlEngine);
  const detail::DnnlMemoryWrapper varMemory(
      state->var, {nfeatures}, formatX, dnnlEngine);
  const detail::DnnlMemoryWrapper weightsMemory(
      state->weights, state->weightsDims, formatX, dnnlEngine);
  const detail::DnnlMemoryWrapper biasMemory(
      state->bias, state->biasDims, formatX, dnnlEngine);

  auto kind = train ? dnnl::prop_kind::forward_training
                    : dnnl::prop_kind::forward_inference;
  dnnl::normalization_flags flag = train
      ? dnnl::normalization_flags::none
      : dnnl::normalization_flags::use_global_stats;
  flag = flag | dnnl::normalization_flags::use_scale |
      dnnl::normalization_flags::use_shift;
  state->fwdPrimDesc = dnnl::batch_normalization_forward::primitive_desc(
      dnnlEngine, kind,
      inputMemory.getDescriptor(),
      outputMemory.getDescriptor(),
      epsilon, flag, attr);
  state->outputMemoryDescriptor = outputMemory.getDescriptor();
  auto bn = dnnl::batch_normalization_forward(state->fwdPrimDesc);
  std::unordered_map<int, dnnl::memory> bnFwdArgs = {
      {DNNL_ARG_SRC, inputMemory.getMemory()},
      {DNNL_ARG_MEAN, meanMemory.getMemory()},
      {DNNL_ARG_VARIANCE, varMemory.getMemory()},
      {DNNL_ARG_DST, outputMemory.getMemory()},
      {DNNL_ARG_SCALE, weightsMemory.getMemory()},
      {DNNL_ARG_SHIFT, biasMemory.getMemory()}};
  // Allocate workspace for forward_training (needed by backward)
  if (train && wantState) {
    auto wsDesc = state->fwdPrimDesc.workspace_desc();
    if (wsDesc.get_size() > 0) {
      state->workspace = dnnl::memory(wsDesc, dnnlEngine);
      bnFwdArgs[DNNL_ARG_WORKSPACE] = state->workspace;
    }
  }

  std::vector<dnnl::primitive> network;
  std::vector<std::unordered_map<int, dnnl::memory>> fwdArgs = {bnFwdArgs};
  network.push_back(bn);

  std::vector<std::shared_ptr<CPUTensor::SharedData>> keepAlive;
  keepAlive.reserve(8);
  keepAlive.push_back(toCPUTensor(contiguousInput).getSharedData());
  keepAlive.push_back(toCPUTensor(output).getSharedData());
  keepAlive.push_back(toCPUTensor(runningMean).getSharedData());
  keepAlive.push_back(toCPUTensor(runningVar).getSharedData());
  keepAlive.push_back(toCPUTensor(state->weights).getSharedData());
  keepAlive.push_back(toCPUTensor(state->bias).getSharedData());
  keepAlive.push_back(toCPUTensor(state->mean).getSharedData());
  keepAlive.push_back(toCPUTensor(state->var).getSharedData());
  detail::executeNetwork(
      std::move(network), std::move(fwdArgs),
      asyncStream, nativeStream,
      std::move(keepAlive));

  ForwardBackwardState fbState = wantState
      ? state
      : nullptr;
  if (!train) {
    return {output, state->mean, state->var, runningMean, runningVar, fbState};
  }
  // In training, OneDNN wrote the batch statistics into state->mean/var (their
  // own buffers: CPU tensor copies are deep). Running statistics follow
  // cuDNN's convention: running = (1 - momentum) * running + momentum * batch,
  // using the unbiased batch variance.
  if (momentum == 0.) {
    return {output, state->mean, state->var, runningMean, runningVar, fbState};
  }
  const double perFeature = static_cast<double>(input.elements()) / nfeatures;
  const double unbias = perFeature > 1 ? perFeature / (perFeature - 1) : 1.;
  auto newRunningMean = (runningMean * (1. - momentum) + state->mean * momentum)
                            .astype(runningMean.type());
  auto newRunningVar =
      (runningVar * (1. - momentum) + state->var * (momentum * unbias))
          .astype(runningVar.type());
  return {
      output, state->mean, state->var, newRunningMean, newRunningVar, fbState};
}

std::tuple<Tensor, Tensor, Tensor> batchnormBackward(
    const Tensor& gradOutput,
    const Tensor& saveMean,
    const Tensor& saveVar,
    const Tensor& input,
    const Tensor& weight,
    const std::vector<int>& axes,
    bool train,
    float epsilon,
    const ForwardBackwardState& fbState,
    const dnnl::engine& dnnlEngine,
    AsyncCPUStream& asyncStream,
    dnnl::stream& nativeStream) {
  if (!fbState) {
    throw std::invalid_argument(
        "onednn::batchnormBackward given null ForwardBackwardState");
  }
  auto state = std::static_pointer_cast<OneDnnBatchNormState>(fbState);

  auto maxAxis = *std::max_element(axes.begin(), axes.end());
  auto minAxis = *std::min_element(axes.begin(), axes.end());
  const bool axesContinuous = (axes.size() == static_cast<size_t>(maxAxis - minAxis + 1));
  if (!axesContinuous) {
    throw std::invalid_argument("axis array should be continuous");
  }

  const int nfeatures = getNfeatures(input.shape(), axes);
  auto inputOutputDims = getInputOutputDims(minAxis, maxAxis, input, nfeatures);

  auto gradInput = makeCPUTensor(input.shape(), input.type());
  auto gradWeights = makeCPUTensor(state->weights.shape(), state->weights.type());
  auto gradBias = makeCPUTensor(state->bias.shape(), state->bias.type());

  // DnnlMemoryWrapper format tags assume contiguous layout
  auto contiguousInput = input.asContiguousTensor();
  auto contiguousGradOutput = gradOutput.asContiguousTensor();
  const detail::DnnlMemoryWrapper inputMemory(
      contiguousInput, inputOutputDims, formatNCHW, dnnlEngine);
  const detail::DnnlMemoryWrapper gradOutputMem(
      contiguousGradOutput, inputOutputDims, formatNCHW, dnnlEngine);
  const detail::DnnlMemoryWrapper gradInputMem(
      gradInput, inputOutputDims, formatNCHW, dnnlEngine);
  const detail::DnnlMemoryWrapper gradWeightsMem(
      gradWeights, state->weightsDims, formatX, dnnlEngine);
  const detail::DnnlMemoryWrapper gradBiasMem(
      gradBias, state->biasDims, formatX, dnnlEngine);
  // Recreate mean/var/weights/bias memory wrappers from stored Tensors
  const detail::DnnlMemoryWrapper meanMemory(
      state->mean, state->weightsDims, formatX, dnnlEngine);
  const detail::DnnlMemoryWrapper varMemory(
      state->var, state->weightsDims, formatX, dnnlEngine);
  const detail::DnnlMemoryWrapper weightsMemory(
      state->weights, state->weightsDims, formatX, dnnlEngine);
  const detail::DnnlMemoryWrapper biasMemory(
      state->bias, state->biasDims, formatX, dnnlEngine);

  auto bwdPrimitiveDesc = dnnl::batch_normalization_backward::primitive_desc(
      dnnlEngine, dnnl::prop_kind::backward,
      gradInputMem.getDescriptor(),
      inputMemory.getDescriptor(),
      gradOutputMem.getDescriptor(),
      epsilon,
      dnnl::normalization_flags::use_scale |
          dnnl::normalization_flags::use_shift,
      state->fwdPrimDesc);
  auto bwdPrim =
      std::make_shared<dnnl::batch_normalization_backward>(bwdPrimitiveDesc);

  std::vector<dnnl::primitive> networkBackwards;
  std::unordered_map<int, dnnl::memory> bwdArgMap = {
      {DNNL_ARG_SRC, inputMemory.getMemory()},
      {DNNL_ARG_MEAN, meanMemory.getMemory()},
      {DNNL_ARG_VARIANCE, varMemory.getMemory()},
      {DNNL_ARG_SCALE, weightsMemory.getMemory()},
      {DNNL_ARG_DIFF_SRC, gradInputMem.getMemory()},
      {DNNL_ARG_DIFF_DST, gradOutputMem.getMemory()},
      {DNNL_ARG_DIFF_SCALE, gradWeightsMem.getMemory()},
      {DNNL_ARG_DIFF_SHIFT, gradBiasMem.getMemory()}};
  // Pass workspace from forward if available
  if (state->workspace) {
    bwdArgMap[DNNL_ARG_WORKSPACE] = state->workspace;
  }

  std::vector<std::unordered_map<int, dnnl::memory>> bwdArgs = {bwdArgMap};
  networkBackwards.push_back(*bwdPrim);

  std::vector<std::shared_ptr<CPUTensor::SharedData>> keepAlive;
  keepAlive.reserve(9);
  keepAlive.push_back(toCPUTensor(contiguousInput).getSharedData());
  keepAlive.push_back(toCPUTensor(contiguousGradOutput).getSharedData());
  keepAlive.push_back(toCPUTensor(gradInput).getSharedData());
  keepAlive.push_back(toCPUTensor(gradWeights).getSharedData());
  keepAlive.push_back(toCPUTensor(gradBias).getSharedData());
  keepAlive.push_back(toCPUTensor(state->mean).getSharedData());
  keepAlive.push_back(toCPUTensor(state->var).getSharedData());
  keepAlive.push_back(toCPUTensor(state->weights).getSharedData());
  keepAlive.push_back(toCPUTensor(state->bias).getSharedData());
  detail::executeNetwork(
      std::move(networkBackwards), std::move(bwdArgs),
      asyncStream, nativeStream,
      std::move(keepAlive));

  return {gradInput, gradWeights, gradBias};
}

} // namespace onednn
} // namespace detail
} // namespace fl
