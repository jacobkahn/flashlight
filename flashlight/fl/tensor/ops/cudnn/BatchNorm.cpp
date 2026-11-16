/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/cudnn/CudnnAutogradOps.h"

#include <algorithm>
#include <stdexcept>

#include <cudnn.h>
#include <cudnn_frontend.h>

#include "flashlight/fl/tensor/ops/cudnn/CudnnPlanCache.h"
#include "flashlight/fl/tensor/ops/cudnn/CudnnTensorHelpers.h"
#include "flashlight/fl/tensor/ops/cudnn/CudnnUtils.h"
#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/tensor/Compute.h"

namespace fl {
namespace detail {
namespace cudnn {

namespace fe = cudnn_frontend;

namespace {

// Compute batchnorm metadata for synthetic NCHW descriptors.
// This mirrors the legacy getBatchnormMetadata but returns simple structs.
struct BNMeta {
  bool isSpatial; // SPATIAL vs PER_ACTIVATION
  int batchSize;
  int nfeatures;
  int trailing;
  Shape inDescDims; // synthetic 4D dims for cuDNN
  Shape wtDescDims; // 1D weight/bias dims
};

BNMeta getBNMeta(
    const Tensor& input,
    const std::vector<int>& axes,
    bool train) {
  BNMeta meta;
  meta.nfeatures = 1;
  for (auto ax : axes) {
    if (ax > input.ndim() - 1) {
      throw std::invalid_argument(
          "batchnorm - passed axes (axis value " + std::to_string(ax) +
          ") exceeds the number of dimensions of the input (" +
          std::to_string(input.ndim()) + ")");
    }
    meta.nfeatures *= input.dim(ax);
  }

  auto maxAxis = *std::max_element(axes.begin(), axes.end());
  auto minAxis = *std::min_element(axes.begin(), axes.end());
  bool axes_continuous =
      (axes.size() == static_cast<size_t>(maxAxis - minAxis + 1));
  if (!axes_continuous) {
    throw std::invalid_argument("unsupported axis config for cuDNN batchnorm");
  }

  if (minAxis == 0) {
    meta.isSpatial = false; // PER_ACTIVATION
    meta.batchSize = static_cast<int>(input.elements() / meta.nfeatures);
    meta.trailing = 1;
    meta.inDescDims = Shape{meta.batchSize, meta.nfeatures, 1, 1};
  } else {
    meta.isSpatial = true; // SPATIAL
    meta.batchSize = 1;
    for (int i = 0; i < minAxis; ++i) {
      meta.batchSize *= input.dim(i);
    }
    meta.trailing =
        static_cast<int>(input.elements() / (meta.nfeatures * meta.batchSize));
    meta.inDescDims = Shape{
        meta.batchSize,
        meta.nfeatures,
        static_cast<long long>(meta.trailing),
        1};
  }
  meta.wtDescDims = Shape{1, meta.nfeatures, 1};
  return meta;
}

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
    bool /*wantState*/) {
  if ((input.type() == fl::dtype::f16 || input.type() == fl::dtype::bf16) &&
      weight.type() != fl::dtype::f32) {
    throw std::invalid_argument(
        "fl::batchnorm: non-input tensors must be of type f32");
  }
  FL_TENSOR_DTYPES_MATCH_CHECK(weight, bias, runningMeanIn, runningVarIn);

  Tensor runningMean = runningMeanIn.copy();
  Tensor runningVar = runningVarIn.copy();
  auto output = makeTensorLike(input, input.shape(), input.type());

  auto bnMeta = getBNMeta(input, axes, train);

  if (!weight.isEmpty() &&
      weight.elements() != bnMeta.wtDescDims.elements()) {
    throw std::invalid_argument("[BatchNorm] Invalid shape for weight.");
  }
  if (!bias.isEmpty() && bias.elements() != bnMeta.wtDescDims.elements()) {
    throw std::invalid_argument("[BatchNorm] Invalid shape for bias.");
  }

  Tensor weightArray = weight.isEmpty()
      ? fullLike(input, bnMeta.wtDescDims, 1.0, fl::dtype::f32)
      : weight.astype(fl::dtype::f32);
  Tensor biasArray = bias.isEmpty()
      ? fullLike(input, bnMeta.wtDescDims, 0.0, fl::dtype::f32)
      : bias.astype(fl::dtype::f32);

  fl::dtype scalarsType =
      (input.type() == fl::dtype::f16 || input.type() == fl::dtype::bf16)
      ? fl::dtype::f32 : input.type();

  // Use legacy cuDNN API for batchnorm — the graph API batchnorm
  // requires NHWC layout for full fusion support. Legacy API is still
  // supported and works correctly with NCHW synthetic descriptors.
  cudnnBatchNormMode_t mode;
  Shape inDescDims, wtDescDims;
  std::vector<int> inDescStrides;

  // Compute legacy metadata
  if (!bnMeta.isSpatial) {
    mode = CUDNN_BATCHNORM_PER_ACTIVATION;
    inDescDims = bnMeta.inDescDims;
    inDescStrides = {1, bnMeta.batchSize, bnMeta.batchSize, bnMeta.batchSize};
  } else {
    mode = train ? CUDNN_BATCHNORM_SPATIAL_PERSISTENT : CUDNN_BATCHNORM_SPATIAL;
    inDescDims = bnMeta.inDescDims;
    inDescStrides = {
        bnMeta.nfeatures * bnMeta.trailing,
        bnMeta.trailing, 1, 1};
  }
  wtDescDims = bnMeta.wtDescDims;

  auto inDesc = TensorDescriptor(input.type(), inDescDims, inDescStrides);
  auto wtDesc = TensorDescriptor(weightArray.type(), wtDescDims);

  Tensor saveMean, saveVar;
  {
    auto contiguousInput = input.asContiguousTensor();
    DevicePtr inRaw(contiguousInput);
    DevicePtr outRaw(output);
    DevicePtr wtRaw(weightArray);
    DevicePtr bsRaw(biasArray);
    DevicePtr runMeanRaw(runningMean);
    DevicePtr runVarRaw(runningVar);
    const auto& cudnnStream = getCudnnStream();
    relativeSync(
        cudnnStream,
        {contiguousInput, output, weightArray, biasArray, runningMean,
         runningVar});

    if (train) {
      saveMean = makeTensorLike(input, {wtDescDims[1]}, scalarsType);
      saveVar = makeTensorLike(input, {wtDescDims[1]}, scalarsType);

      DevicePtr saveMeanRaw(saveMean);
      DevicePtr saveVarRaw(saveVar);
      relativeSync(cudnnStream, {saveMean, saveVar});
      CUDNN_CHECK_ERR(cudnnBatchNormalizationForwardTraining(
          getCudnnHandle(),
          mode,
          kOne(scalarsType),
          kZero(scalarsType),
          inDesc.descriptor,
          inRaw.get(),
          inDesc.descriptor,
          outRaw.get(),
          wtDesc.descriptor,
          wtRaw.get(),
          bsRaw.get(),
          momentum,
          runMeanRaw.get(),
          runVarRaw.get(),
          epsilon,
          saveMeanRaw.get(),
          saveVarRaw.get()));
    } else {
      CUDNN_CHECK_ERR(cudnnBatchNormalizationForwardInference(
          getCudnnHandle(),
          mode,
          kOne(scalarsType),
          kZero(scalarsType),
          inDesc.descriptor,
          inRaw.get(),
          inDesc.descriptor,
          outRaw.get(),
          wtDesc.descriptor,
          wtRaw.get(),
          bsRaw.get(),
          runMeanRaw.get(),
          runVarRaw.get(),
          epsilon));
    }
    relativeSync({output}, cudnnStream);
  }
  return {output, saveMean, saveVar, runningMean, runningVar, nullptr};
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
    const ForwardBackwardState& /*state*/) {
  if (!train) {
    throw std::logic_error(
        "can't compute batchnorm grad when train was not specified");
  }

  auto bnMeta = getBNMeta(input, axes, train);

  cudnnBatchNormMode_t mode;
  Shape inDescDims, wtDescDims;
  std::vector<int> inDescStrides;

  if (!bnMeta.isSpatial) {
    mode = CUDNN_BATCHNORM_PER_ACTIVATION;
    inDescDims = bnMeta.inDescDims;
    inDescStrides = {1, bnMeta.batchSize, bnMeta.batchSize, bnMeta.batchSize};
  } else {
    mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    inDescDims = bnMeta.inDescDims;
    inDescStrides = {
        bnMeta.nfeatures * bnMeta.trailing, bnMeta.trailing, 1, 1};
  }
  wtDescDims = bnMeta.wtDescDims;

  auto wt = weight.isEmpty() ? fullLike(input, wtDescDims, 1.0, fl::dtype::f32)
                             : weight;

  auto scalarsType =
      (input.type() == fl::dtype::f16 || input.type() == fl::dtype::bf16)
      ? fl::dtype::f32 : input.type();
  const void* one1 = kOne(scalarsType);
  const void* zero0 = kZero(scalarsType);

  auto iDesc = TensorDescriptor(input.type(), inDescDims, inDescStrides);
  auto wDesc = TensorDescriptor(wt.type(), wtDescDims);
  auto gradIn = makeTensorLike(input, input.shape(), input.type());
  auto gradWt = makeTensorLike(input, wt.shape(), wt.type());
  auto gradBs = makeTensorLike(input, wt.shape(), wt.type());
  {
    auto contiguousInput = input.asContiguousTensor();
    auto contiguousGradOutput = gradOutput.asContiguousTensor();
    DevicePtr iRaw(contiguousInput);
    DevicePtr wRaw(wt);

    DevicePtr gradInRaw(gradIn);
    DevicePtr gradWtRaw(gradWt);
    DevicePtr gradBsRaw(gradBs);

    DevicePtr gradOpRaw(contiguousGradOutput);

    DevicePtr saveMeanRaw(saveMean);
    DevicePtr saveVarRaw(saveVar);
    const auto& cudnnStream = getCudnnStream();
    relativeSync(
        cudnnStream,
        {contiguousInput, contiguousGradOutput, gradIn, wt, gradWt, gradBs,
         saveMean, saveVar});

    CUDNN_CHECK_ERR(cudnnBatchNormalizationBackward(
        getCudnnHandle(),
        mode,
        one1,
        zero0,
        one1,
        zero0,
        iDesc.descriptor,
        iRaw.get(),
        iDesc.descriptor,
        gradOpRaw.get(),
        iDesc.descriptor,
        gradInRaw.get(),
        wDesc.descriptor,
        wRaw.get(),
        gradWtRaw.get(),
        gradBsRaw.get(),
        epsilon,
        saveMeanRaw.get(),
        saveVarRaw.get()));
    relativeSync({gradIn, gradWt, gradBs}, cudnnStream);
  }

  return std::make_tuple(gradIn, gradWt, gradBs);
}

} // namespace cudnn
} // namespace detail
} // namespace fl

