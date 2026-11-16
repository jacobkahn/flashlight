/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/cudnn/CudnnAutogradOps.h"

#include "flashlight/fl/tensor/ops/cudnn/CudnnTensorHelpers.h"
#include "flashlight/fl/tensor/ops/cudnn/CudnnUtils.h"
#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/tensor/Compute.h"

namespace fl {
namespace detail {
namespace cudnn {

ForwardBackwardState pool2dForward(
    Tensor& out,
    const Tensor& input,
    int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode, bool /*wantState*/) {
  auto inDesc = TensorDescriptor(input);
  auto poolDesc = PoolingDescriptor(wx, wy, sx, sy, px, py, mode);

  // Row-major: dim(0)=N, dim(1)=C, dim(2)=H, dim(3)=W
  auto ix = input.ndim() < 4 ? 1 : input.dim(3); // W
  auto iy = input.ndim() < 3 ? 1 : input.dim(2); // H
  auto ox = 1 + (ix + 2 * px - wx) / sx;
  auto oy = 1 + (iy + 2 * py - wy) / sy;

  Shape outputShape{input.dim(0), input.ndim() < 2 ? 1 : input.dim(1), oy, ox};
  auto output = (!out.isEmpty() && out.shape() == outputShape && out.type() == input.type())
      ? out : makeTensorLike(input, outputShape, input.type());
  auto outDesc = TensorDescriptor(output);
  {
    DevicePtr inputraw(input);
    DevicePtr resultraw(output);
    const auto& cudnnStream = getCudnnStream();
    relativeSync(cudnnStream, {input, output});

    auto handle = getCudnnHandle();
    const void* one = kOne(input.type());
    const void* zero = kZero(input.type());

    CUDNN_CHECK_ERR(cudnnPoolingForward(
        handle,
        poolDesc.descriptor,
        one,
        inDesc.descriptor,
        inputraw.get(),
        zero,
        outDesc.descriptor,
        resultraw.get()));

    relativeSync({output}, cudnnStream);
  }

  out = std::move(output);
  return nullptr;
}

Tensor pool2dBackward(
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& poolOutput,
    int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode, const ForwardBackwardState& /*state*/) {
  auto i_desc = TensorDescriptor(input);
  auto o_desc = TensorDescriptor(poolOutput);
  auto p_desc = PoolingDescriptor(wx, wy, sx, sy, px, py, mode);

  auto gradInput = makeTensorLike(input, input.shape(), input.type());

  auto hndl = getCudnnHandle();
  const auto& cudnnStream = getCudnnStream();
  const void* oneg = kOne(input.type());
  const void* zerog = kZero(input.type());

  {
    DevicePtr inraw(input);
    DevicePtr outraw(poolOutput);
    DevicePtr gradresultraw(gradOutput);
    DevicePtr gradinputraw(gradInput);
    relativeSync(cudnnStream, {input, poolOutput, gradOutput, gradInput});

    CUDNN_CHECK_ERR(cudnnPoolingBackward(
        hndl,
        p_desc.descriptor,
        oneg,
        o_desc.descriptor,
        outraw.get(),
        o_desc.descriptor,
        gradresultraw.get(),
        i_desc.descriptor,
        inraw.get(),
        zerog,
        i_desc.descriptor,
        gradinputraw.get()));
    relativeSync({gradInput}, cudnnStream);
  }

  return gradInput;
}

} // namespace cudnn
} // namespace detail
} // namespace fl
