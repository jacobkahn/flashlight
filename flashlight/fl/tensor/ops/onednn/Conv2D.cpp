/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/onednn/OneDnnAutogradOps.h"

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include <dnnl.hpp>

#include "flashlight/fl/tensor/ops/onednn/DnnlUtils.h"
#include "flashlight/fl/tensor/ops/onednn/OneDnnOps.h"
#include "flashlight/fl/tensor/ops/onednn/Utils.h"

using namespace dnnl;

namespace fl {
namespace detail {
namespace onednn {

namespace {

// Row-major: input/output: {N, C, H, W}; weights: {O, I, H, W}
constexpr size_t kIOBatchSizeIdx = 0;
constexpr size_t kIOChannelSizeIdx = 1;
constexpr size_t kHIdx = 2;
constexpr size_t kWIdx = 3;
constexpr size_t kWeightOutputChannelSizeIdx = 0;

constexpr auto formatAny = memory::format_tag::any;
constexpr auto formatNCHW = memory::format_tag::nchw;
constexpr auto formatBias = memory::format_tag::x;

struct OneDnnConv2DData {
  memory::dims inputDims;
  memory::dims weightDims;
  memory::dims outputDims;
  memory::dims biasDims;
  memory::dims strideDims;
  memory::dims dilationDims;
  memory::dims paddingDims;
  memory::desc inputMemDesc;
  memory::desc outputMemDesc;
  memory::desc weightMemDesc;
  memory::desc biasMemDesc;
  convolution_forward::primitive_desc fwdPrimDesc;
};

OneDnnConv2DData createOneDnnConv2DData(
    const dnnl::engine& dnnlEngine,
    fl::dtype inputType,
    const Shape& inputShape,
    const Shape& weightsShape,
    const Shape& biasShape,
    const Shape& outputShape,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const dnnl::primitive_attr& attr = dnnl::primitive_attr()) {
  const dnnl::memory::data_type dataType = detail::dnnlMapToType(inputType);
  const auto formatWeight =
      (groups == 1) ? memory::format_tag::oihw : memory::format_tag::goihw;
  const bool hasBias = biasShape.elements() > 0;

  OneDnnConv2DData out;
  out.inputDims = detail::convertToDnnlDims(
      {inputShape.dim(kIOBatchSizeIdx),
       inputShape.dim(kIOChannelSizeIdx),
       inputShape.dim(kHIdx),
       inputShape.dim(kWIdx)});
  if (groups == 1) {
    out.weightDims = detail::convertToDnnlDims(
        {weightsShape.dim(kWeightOutputChannelSizeIdx),
         inputShape.dim(kIOChannelSizeIdx),
         weightsShape.dim(kHIdx),
         weightsShape.dim(kWIdx)});
  } else {
    out.weightDims = detail::convertToDnnlDims(
        {groups,
         weightsShape.dim(kWeightOutputChannelSizeIdx) / groups,
         inputShape.dim(kIOChannelSizeIdx) / groups,
         weightsShape.dim(kHIdx),
         weightsShape.dim(kWIdx)});
  }
  out.outputDims = detail::convertToDnnlDims(
      {inputShape.dim(kIOBatchSizeIdx),
       weightsShape.dim(kWeightOutputChannelSizeIdx),
       outputShape.dim(kHIdx),
       outputShape.dim(kWIdx)});
  out.biasDims = detail::convertToDnnlDims(
      {weightsShape.dim(kWeightOutputChannelSizeIdx)});
  out.strideDims = {sy, sx};
  out.paddingDims = {py, px};
  out.dilationDims = {dy - 1, dx - 1};

  out.inputMemDesc = memory::desc({out.inputDims}, dataType, formatAny);
  out.outputMemDesc = memory::desc({out.outputDims}, dataType, formatAny);
  out.weightMemDesc = memory::desc({out.weightDims}, dataType, formatWeight);
  out.biasMemDesc = memory::desc({out.biasDims}, dataType, formatAny);

  const auto forwardMode = prop_kind::forward_training;
  convolution_forward::primitive_desc fwdPrimitiveDescriptor;
  if (hasBias) {
    fwdPrimitiveDescriptor = convolution_forward::primitive_desc(
        dnnlEngine, forwardMode, algorithm::convolution_direct,
        out.inputMemDesc, out.weightMemDesc, out.biasMemDesc,
        out.outputMemDesc, out.strideDims, out.dilationDims,
        out.paddingDims, out.paddingDims, attr);
  } else {
    fwdPrimitiveDescriptor = convolution_forward::primitive_desc(
        dnnlEngine, forwardMode, algorithm::convolution_direct,
        out.inputMemDesc, out.weightMemDesc,
        out.outputMemDesc, out.strideDims, out.dilationDims,
        out.paddingDims, out.paddingDims, attr);
  }
  out.fwdPrimDesc = std::move(fwdPrimitiveDescriptor);

  return out;
}

} // namespace

ForwardBackwardState conv2dForward(
    Tensor& out,
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    bool /*wantState*/,
    const dnnl::engine& dnnlEngine,
    AsyncCPUStream& asyncStream,
    dnnl::stream& nativeStream,
    const dnnl::primitive_attr& attr,
    const Tensor& sumSrc) {
  if (input.type() == fl::dtype::f16) {
    throw std::runtime_error("Half precision is not supported in CPU.");
  }

  // Row-major output: {N, O, oH, oW}
  Shape outputShape{input.dim(kIOBatchSizeIdx),
       weights.dim(kWeightOutputChannelSizeIdx),
       1 +
           (input.dim(kHIdx) + (2 * py) - (1 + (weights.dim(kHIdx) - 1) * dy)) /
               sy,
       1 +
           (input.dim(kWIdx) + (2 * px) - (1 + (weights.dim(kWIdx) - 1) * dx)) /
               sx};
  auto output = (!out.isEmpty() && out.shape() == outputShape && out.type() == input.type())
      ? out : makeCPUTensor(outputShape, input.type());
  auto hasBias = bias.elements() > 0;

  auto formatWeight =
      (groups == 1) ? memory::format_tag::oihw : memory::format_tag::goihw;

  OneDnnConv2DData conv2DData = createOneDnnConv2DData(
      dnnlEngine, input.type(), input.shape(), weights.shape(), bias.shape(),
      output.shape(), sx, sy, px, py, dx, dy, groups, attr);

  // DnnlMemoryWrapper format tags assume contiguous layout
  auto contiguousInput = input.asContiguousTensor();
  auto contiguousWeights = weights.asContiguousTensor();
  const detail::DnnlMemoryWrapper inputMemInit(
      contiguousInput, {conv2DData.inputDims}, formatNCHW, dnnlEngine);
  const detail::DnnlMemoryWrapper outputMemInit(
      output, {conv2DData.outputDims}, formatNCHW, dnnlEngine);
  const detail::DnnlMemoryWrapper weightsMem(
      contiguousWeights, {conv2DData.weightDims}, formatWeight, dnnlEngine);

  std::vector<primitive> network;
  std::vector<std::unordered_map<int, dnnl::memory>> fwdArgs;

  auto inputDesc = conv2DData.fwdPrimDesc.src_desc();
  auto weightsDesc = conv2DData.fwdPrimDesc.weights_desc();
  auto outputDesc = conv2DData.fwdPrimDesc.dst_desc();
  auto inputMemory = detail::dnnlAlignOrdering(
      network, fwdArgs, inputMemInit.getMemory(), inputDesc, dnnlEngine);
  auto weightsMemory = detail::dnnlAlignOrdering(
      network, fwdArgs, weightsMem.getMemory(), weightsDesc, dnnlEngine);
  auto outputMemory = outputMemInit.getMemory();
  if (outputMemInit.getMemory().get_desc() != outputDesc) {
    outputMemory = memory(outputDesc, dnnlEngine);
  }

  // If sumSrc is provided (for append_sum post-op), pre-fill the destination
  // with its data. The conv primitive will compute dst = conv(...) + dst_old.
  Tensor contiguousSumSrc;
  if (!sumSrc.isEmpty()) {
    contiguousSumSrc = sumSrc.asContiguousTensor();
    const detail::DnnlMemoryWrapper sumSrcMem(
        contiguousSumSrc, conv2DData.outputDims, formatNCHW, dnnlEngine);
    network.push_back(dnnl::reorder(sumSrcMem.getMemory(), outputMemory));
    fwdArgs.push_back(
        {{DNNL_ARG_FROM, sumSrcMem.getMemory()},
         {DNNL_ARG_TO, outputMemory}});
  }

  const detail::DnnlMemoryWrapper biasMemory(
      bias, conv2DData.biasDims, formatBias, dnnlEngine);
  auto conv = std::make_shared<convolution_forward>(conv2DData.fwdPrimDesc);

  network.push_back(*conv);

  std::unordered_map<int, dnnl::memory> convFwdArgs = {
      {DNNL_ARG_SRC, inputMemory},
      {DNNL_ARG_WEIGHTS, weightsMemory},
      {DNNL_ARG_DST, outputMemory}};
  if (hasBias) {
    convFwdArgs[DNNL_ARG_BIAS] = biasMemory.getMemory();
  }
  fwdArgs.push_back(convFwdArgs);

  if (outputMemory != outputMemInit.getMemory()) {
    network.push_back(dnnl::reorder(outputMemory, outputMemInit.getMemory()));
    fwdArgs.push_back(
        {{DNNL_ARG_FROM, outputMemory},
         {DNNL_ARG_TO, outputMemInit.getMemory()}});
  }

  // Keep tensor data alive until async execution completes. The
  // DnnlMemoryWrapper creates dnnl::memory with raw pointers that do not
  // prevent the underlying buffer from being freed.
  std::vector<std::shared_ptr<CPUTensor::SharedData>> keepAlive;
  keepAlive.reserve(5);
  keepAlive.push_back(toCPUTensor(contiguousInput).getSharedData());
  keepAlive.push_back(toCPUTensor(contiguousWeights).getSharedData());
  keepAlive.push_back(toCPUTensor(output).getSharedData());
  if (hasBias) {
    keepAlive.push_back(toCPUTensor(bias).getSharedData());
  }
  if (!contiguousSumSrc.isEmpty()) {
    keepAlive.push_back(toCPUTensor(contiguousSumSrc).getSharedData());
  }
  detail::executeNetwork(
      std::move(network), std::move(fwdArgs),
      asyncStream, nativeStream,
      std::move(keepAlive));

  out = std::move(output);
  return nullptr;
}

Tensor conv2dBackwardData(
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& weights,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const ForwardBackwardState& /*state*/,
    const dnnl::engine& dnnlEngineBwd,
    AsyncCPUStream& asyncStream,
    dnnl::stream& nativeStream) {
  auto gradInput = makeCPUTensor(input.shape(), input.type());

  auto formatWeight =
      (groups == 1) ? memory::format_tag::oihw : memory::format_tag::goihw;

  Tensor bias; // dummy
  OneDnnConv2DData conv2DData = createOneDnnConv2DData(
      dnnlEngineBwd, input.type(), input.shape(), weights.shape(), bias.shape(),
      gradOutput.shape(), sx, sy, px, py, dx, dy, groups);

  convolution_backward_data::primitive_desc bwdDataPrimitiveDesc(
      dnnlEngineBwd, algorithm::convolution_direct,
      conv2DData.inputMemDesc, conv2DData.weightMemDesc,
      conv2DData.outputMemDesc, conv2DData.strideDims,
      conv2DData.dilationDims, conv2DData.paddingDims,
      conv2DData.paddingDims, conv2DData.fwdPrimDesc);

  // DnnlMemoryWrapper format tags assume contiguous layout
  auto contiguousGradOutput = gradOutput.asContiguousTensor();
  auto contiguousWeights = weights.asContiguousTensor();
  const detail::DnnlMemoryWrapper gradOutputMemInit(
      contiguousGradOutput, conv2DData.outputDims, formatNCHW, dnnlEngineBwd);
  const detail::DnnlMemoryWrapper gradInputMemInit(
      gradInput, conv2DData.inputDims, formatNCHW, dnnlEngineBwd);
  const detail::DnnlMemoryWrapper weightsMemInitBwd(
      contiguousWeights, conv2DData.weightDims, formatWeight, dnnlEngineBwd);

  std::vector<primitive> networkBackwards;
  std::vector<std::unordered_map<int, dnnl::memory>> bwdDataArgs;

  auto gradOutputDesc = bwdDataPrimitiveDesc.diff_dst_desc();
  auto weightsDesc = bwdDataPrimitiveDesc.weights_desc();
  auto gradInputDesc = bwdDataPrimitiveDesc.diff_src_desc();
  auto gradOutputMemory = detail::dnnlAlignOrdering(
      networkBackwards, bwdDataArgs,
      gradOutputMemInit.getMemory(), gradOutputDesc, dnnlEngineBwd);
  auto weightsMemoryBackwards = detail::dnnlAlignOrdering(
      networkBackwards, bwdDataArgs,
      weightsMemInitBwd.getMemory(), weightsDesc, dnnlEngineBwd);
  auto gradInputMemory = gradInputMemInit.getMemory();
  if (gradInputMemInit.getMemory().get_desc() != gradInputDesc) {
    gradInputMemory = memory(gradInputDesc, dnnlEngineBwd);
  }

  auto convBwdData =
      std::make_shared<convolution_backward_data>(bwdDataPrimitiveDesc);

  bwdDataArgs.push_back(
      {{DNNL_ARG_DIFF_SRC, gradInputMemory},
       {DNNL_ARG_WEIGHTS, weightsMemoryBackwards},
       {DNNL_ARG_DIFF_DST, gradOutputMemory}});
  networkBackwards.push_back(*convBwdData);

  if (gradInputMemory != gradInputMemInit.getMemory()) {
    networkBackwards.push_back(
        dnnl::reorder(gradInputMemory, gradInputMemInit.getMemory()));
    bwdDataArgs.push_back(
        {{DNNL_ARG_FROM, gradInputMemory},
         {DNNL_ARG_TO, gradInputMemInit.getMemory()}});
  }

  std::vector<std::shared_ptr<CPUTensor::SharedData>> keepAlive;
  keepAlive.reserve(3);
  keepAlive.push_back(toCPUTensor(contiguousGradOutput).getSharedData());
  keepAlive.push_back(toCPUTensor(contiguousWeights).getSharedData());
  keepAlive.push_back(toCPUTensor(gradInput).getSharedData());
  detail::executeNetwork(
      std::move(networkBackwards), std::move(bwdDataArgs),
      asyncStream, nativeStream,
      std::move(keepAlive));

  return gradInput;
}

std::pair<Tensor, Tensor> conv2dBackwardFilterBias(
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const ForwardBackwardState& /*state*/,
    const dnnl::engine& dnnlEngineBwd,
    AsyncCPUStream& asyncStream,
    dnnl::stream& nativeStream) {
  auto gradWeights = makeCPUTensor(weights.shape(), weights.type());

  auto formatWeight =
      (groups == 1) ? memory::format_tag::oihw : memory::format_tag::goihw;
  OneDnnConv2DData conv2DData = createOneDnnConv2DData(
      dnnlEngineBwd, input.type(), input.shape(), weights.shape(), bias.shape(),
      gradOutput.shape(), sx, sy, px, py, dx, dy, groups);

  Tensor gradBias;
  bool computeBiasGrad = !bias.isEmpty() && !conv2DData.biasMemDesc.is_zero();
  if (computeBiasGrad) {
    gradBias = makeCPUTensor(bias.shape(), bias.type());
  }

  convolution_backward_weights::primitive_desc bwdWeightPrimitiveDesc;
  if (computeBiasGrad) {
    bwdWeightPrimitiveDesc = convolution_backward_weights::primitive_desc(
        dnnlEngineBwd, algorithm::convolution_direct,
        conv2DData.inputMemDesc, conv2DData.weightMemDesc,
        conv2DData.biasMemDesc, conv2DData.outputMemDesc,
        conv2DData.strideDims, conv2DData.dilationDims,
        conv2DData.paddingDims, conv2DData.paddingDims,
        conv2DData.fwdPrimDesc);
  } else {
    bwdWeightPrimitiveDesc = convolution_backward_weights::primitive_desc(
        dnnlEngineBwd, algorithm::convolution_direct,
        conv2DData.inputMemDesc, conv2DData.weightMemDesc,
        conv2DData.outputMemDesc, conv2DData.strideDims,
        conv2DData.dilationDims, conv2DData.paddingDims,
        conv2DData.paddingDims, conv2DData.fwdPrimDesc);
  }

  // DnnlMemoryWrapper format tags assume contiguous layout
  auto contiguousInput = input.asContiguousTensor();
  auto contiguousGradOutput = gradOutput.asContiguousTensor();
  const detail::DnnlMemoryWrapper inputRawMemInitBwd(
      contiguousInput, conv2DData.inputDims, formatNCHW, dnnlEngineBwd);
  const detail::DnnlMemoryWrapper gradOutputMemInit(
      contiguousGradOutput, conv2DData.outputDims, formatNCHW, dnnlEngineBwd);
  const detail::DnnlMemoryWrapper gradWeightsMemInit(
      gradWeights, conv2DData.weightDims, formatWeight, dnnlEngineBwd);

  std::vector<primitive> networkBackwards;
  std::vector<std::unordered_map<int, dnnl::memory>> bwdWeightsArgs;

  auto inputDesc = bwdWeightPrimitiveDesc.src_desc();
  auto gradOutputDesc = bwdWeightPrimitiveDesc.diff_dst_desc();
  auto gradWeightsDesc = bwdWeightPrimitiveDesc.diff_weights_desc();
  auto inputMemoryBackwards = detail::dnnlAlignOrdering(
      networkBackwards, bwdWeightsArgs,
      inputRawMemInitBwd.getMemory(), inputDesc, dnnlEngineBwd);
  auto gradOutputMemory = detail::dnnlAlignOrdering(
      networkBackwards, bwdWeightsArgs,
      gradOutputMemInit.getMemory(), gradOutputDesc, dnnlEngineBwd);
  auto gradWeightsMemory = gradWeightsMemInit.getMemory();
  if (gradWeightsMemInit.getMemory().get_desc() != gradWeightsDesc) {
    gradWeightsMemory = memory(gradWeightsDesc, dnnlEngineBwd);
  }

  // Declare gradBiasMem outside the if-block so its memory stays valid
  // for the async executeNetwork call.
  detail::DnnlMemoryWrapper gradBiasMem;

  std::unordered_map<int, dnnl::memory> bwdConvWeightsArgs = {
      {DNNL_ARG_SRC, inputMemoryBackwards},
      {DNNL_ARG_DIFF_WEIGHTS, gradWeightsMemory},
      {DNNL_ARG_DIFF_DST, gradOutputMemory}};

  if (computeBiasGrad) {
    gradBiasMem = detail::DnnlMemoryWrapper(
        gradBias, conv2DData.biasDims, formatBias, dnnlEngineBwd);
    bwdConvWeightsArgs[DNNL_ARG_DIFF_BIAS] = gradBiasMem.getMemory();
  }
  networkBackwards.push_back(
      convolution_backward_weights(bwdWeightPrimitiveDesc));
  bwdWeightsArgs.push_back(bwdConvWeightsArgs);

  if (gradWeightsMemory != gradWeightsMemInit.getMemory()) {
    networkBackwards.push_back(
        dnnl::reorder(gradWeightsMemory, gradWeightsMemInit.getMemory()));
    bwdWeightsArgs.push_back(
        {{DNNL_ARG_FROM, gradWeightsMemory},
         {DNNL_ARG_TO, gradWeightsMemInit.getMemory()}});
  }

  std::vector<std::shared_ptr<CPUTensor::SharedData>> keepAlive;
  keepAlive.reserve(4);
  keepAlive.push_back(toCPUTensor(contiguousInput).getSharedData());
  keepAlive.push_back(toCPUTensor(contiguousGradOutput).getSharedData());
  keepAlive.push_back(toCPUTensor(gradWeights).getSharedData());
  if (!gradBias.isEmpty()) {
    keepAlive.push_back(toCPUTensor(gradBias).getSharedData());
  }
  detail::executeNetwork(
      std::move(networkBackwards), std::move(bwdWeightsArgs),
      asyncStream, nativeStream,
      std::move(keepAlive));

  return {gradWeights, gradBias};
}

} // namespace onednn
} // namespace detail
} // namespace fl
