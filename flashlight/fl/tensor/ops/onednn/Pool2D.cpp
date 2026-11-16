/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/onednn/OneDnnAutogradOps.h"

#include <unordered_map>
#include <vector>

#include <dnnl.h>

#include "flashlight/fl/tensor/ops/onednn/DnnlUtils.h"
#include "flashlight/fl/tensor/ops/onednn/OneDnnOps.h"
#include "flashlight/fl/tensor/ops/onednn/Utils.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Tensor.h"

using namespace dnnl;

namespace fl {
namespace detail {
namespace onednn {

namespace {

// Row-major: {N, C, H, W}
constexpr size_t kBatchSizeIdx = 0;
constexpr size_t kChannelSizeIdx = 1;
constexpr size_t kHIdx = 2;
constexpr size_t kWIdx = 3;

constexpr auto formatAny = memory::format_tag::any;
constexpr auto formatNCHW = memory::format_tag::nchw;

struct DimsData {
  memory::dims inputDims;
  memory::dims outputDims;
  memory::dims windowDims;
  memory::dims strideDims;
  std::vector<int64_t> paddingDims;
};

DimsData getDimsData(
    const Shape& input,
    const Shape& output,
    int wx, int wy, int sx, int sy, int px, int py) {
  DimsData d;
  // Row-major: input/output are already {N, C, H, W}
  d.inputDims = detail::convertToDnnlDims(
      {input[0], input[1], input[2], input[3]});
  d.outputDims = detail::convertToDnnlDims(
      {input[0], input[1], output[2], output[3]});
  d.windowDims = {wy, wx};
  d.strideDims = {sy, sx};
  d.paddingDims = {py, px};
  return d;
}

// Opaque state for pool2d forward -> backward
struct OneDnnPool2DState : ForwardState {
  memory workspace;
  memory outputMemory;
  DimsData dimsData;
  pooling_forward::primitive_desc poolingFwdPrimDesc;
};

} // namespace

ForwardBackwardState pool2dForward(
    Tensor& out,
    const Tensor& input,
    int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode, bool wantState,
    const dnnl::engine& dnnlEngine,
    AsyncCPUStream& asyncStream,
    dnnl::stream& nativeStream,
    const dnnl::primitive_attr& attr) {
  auto state = std::make_shared<OneDnnPool2DState>();

  // Row-major: {N, C, H, W}
  auto b = input.dim(kBatchSizeIdx);
  auto c = input.ndim() > kChannelSizeIdx ? input.dim(kChannelSizeIdx) : 1;
  auto iy = input.ndim() > kHIdx ? input.dim(kHIdx) : 1;
  auto ix = input.ndim() > kWIdx ? input.dim(kWIdx) : 1;

  auto oH = 1 + (iy + 2 * py - wy) / sy;
  auto oW = 1 + (ix + 2 * px - wx) / sx;
  Shape outputShape{b, c, oH, oW};
  auto output = (!out.isEmpty() && out.shape() == outputShape && out.type() == input.type())
      ? out : makeCPUTensor(outputShape, input.type());

  state->dimsData =
      getDimsData({b, c, iy, ix}, output.shape(), wx, wy, sx, sy, px, py);
  auto& d = state->dimsData;
  auto dataType = detail::dnnlMapToType(input.type());

  auto inputMD = memory::desc({d.inputDims}, dataType, formatNCHW);
  auto outputMD = memory::desc({d.outputDims}, dataType, formatAny);

  // DnnlMemoryWrapper format tags assume contiguous layout
  auto contiguousInput = input.asContiguousTensor();
  const detail::DnnlMemoryWrapper inputMemInit(
      contiguousInput, {d.inputDims}, formatNCHW, dnnlEngine);
  const detail::DnnlMemoryWrapper outputMemInit(
      output, {d.outputDims}, formatNCHW, dnnlEngine);

  auto forwardMode = wantState ? prop_kind::forward : prop_kind::forward_inference;

  auto poolingMode = detail::dnnlMapToPoolingMode(mode);
  state->poolingFwdPrimDesc = pooling_forward::primitive_desc(
      dnnlEngine, forwardMode, poolingMode,
      inputMD, outputMD, d.strideDims, d.windowDims,
      memory::dims{0, 0}, d.paddingDims, d.paddingDims, attr);
  auto& primDesc = state->poolingFwdPrimDesc;

  std::vector<primitive> network;
  std::vector<std::unordered_map<int, dnnl::memory>> fwdArgs;
  auto inputDesc = primDesc.src_desc();
  auto outputDesc = primDesc.dst_desc();
  auto inputMemory = detail::dnnlAlignOrdering(
      network, fwdArgs, inputMemInit.getMemory(), inputDesc, dnnlEngine);
  state->outputMemory = outputMemInit.getMemory();
  if (outputMemInit.getMemory().get_desc() != outputDesc) {
    state->outputMemory = memory(outputDesc, dnnlEngine);
  }

  std::shared_ptr<pooling_forward> pooling;
  std::unordered_map<int, dnnl::memory> fwdPoolingArgs;
  fwdPoolingArgs[DNNL_ARG_SRC] = inputMemory;
  fwdPoolingArgs[DNNL_ARG_DST] = state->outputMemory;
  if (wantState) {
    state->workspace = memory(primDesc.workspace_desc(), dnnlEngine);
    pooling = std::make_shared<pooling_forward>(primDesc);
    fwdPoolingArgs[DNNL_ARG_WORKSPACE] = state->workspace;
  } else {
    pooling = std::make_shared<pooling_forward>(primDesc);
  }
  network.push_back(*pooling);
  fwdArgs.push_back(fwdPoolingArgs);

  if (state->outputMemory != outputMemInit.getMemory()) {
    network.push_back(
        dnnl::reorder(state->outputMemory, outputMemInit.getMemory()));
    fwdArgs.push_back(
        {{DNNL_ARG_FROM, state->outputMemory},
         {DNNL_ARG_TO, outputMemInit.getMemory()}});
  }

  std::vector<std::shared_ptr<CPUTensor::SharedData>> keepAlive;
  keepAlive.reserve(2);
  keepAlive.push_back(toCPUTensor(contiguousInput).getSharedData());
  keepAlive.push_back(toCPUTensor(output).getSharedData());
  detail::executeNetwork(
      std::move(network), std::move(fwdArgs),
      asyncStream, nativeStream,
      std::move(keepAlive));

  ForwardBackwardState fbState = wantState
      ? state
      : nullptr;
  out = std::move(output);
  return fbState;
}

Tensor pool2dBackward(
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& poolOutput,
    int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode, const ForwardBackwardState& fbState,
    const dnnl::engine& dnnlEngineBwd,
    AsyncCPUStream& asyncStream,
    dnnl::stream& nativeStream) {
  if (!fbState) {
    throw std::invalid_argument(
        "onednn::pool2dBackward given null ForwardBackwardState");
  }
  auto state = std::static_pointer_cast<OneDnnPool2DState>(fbState);

  auto gradInput = makeCPUTensor(input.shape(), input.type());

  DimsData& d = state->dimsData;
  auto poolingMode = detail::dnnlMapToPoolingMode(mode);

  // DnnlMemoryWrapper format tags assume contiguous layout
  auto contiguousGradOutput = gradOutput.asContiguousTensor();
  const detail::DnnlMemoryWrapper gradInputMemInit(
      gradInput, {d.inputDims}, formatNCHW, dnnlEngineBwd);
  const detail::DnnlMemoryWrapper gradOutputMemInit(
      contiguousGradOutput, {d.outputDims}, formatNCHW, dnnlEngineBwd);

  auto gradInputMD = gradInputMemInit.getMemory().get_desc();
  auto gradOutputMD = gradOutputMemInit.getMemory().get_desc();
  auto bwdPrimitiveDesc = pooling_backward::primitive_desc(
      dnnlEngineBwd, poolingMode, gradInputMD, gradOutputMD,
      d.strideDims, d.windowDims,
      memory::dims{0, 0}, d.paddingDims, d.paddingDims,
      state->poolingFwdPrimDesc);

  std::vector<primitive> networkBackward;
  std::vector<std::unordered_map<int, dnnl::memory>> bwdArgs;
  auto gradOutputMemory = detail::dnnlAlignOrdering(
      networkBackward, bwdArgs,
      gradOutputMemInit.getMemory(),
      state->outputMemory.get_desc(), dnnlEngineBwd);

  auto poolBwd = pooling_backward(bwdPrimitiveDesc);
  std::unordered_map<int, dnnl::memory> bwdPoolingArgs = {
      {DNNL_ARG_DIFF_SRC, gradInputMemInit.getMemory()},
      {DNNL_ARG_DIFF_DST, gradOutputMemory},
      {DNNL_ARG_WORKSPACE, state->workspace}};
  bwdArgs.push_back(bwdPoolingArgs);
  networkBackward.push_back(poolBwd);

  std::vector<std::shared_ptr<CPUTensor::SharedData>> keepAlive;
  keepAlive.reserve(2);
  keepAlive.push_back(toCPUTensor(contiguousGradOutput).getSharedData());
  keepAlive.push_back(toCPUTensor(gradInput).getSharedData());
  detail::executeNetwork(
      std::move(networkBackward), std::move(bwdArgs),
      asyncStream, nativeStream,
      std::move(keepAlive));

  return gradInput;
}

} // namespace onednn
} // namespace detail
} // namespace fl
