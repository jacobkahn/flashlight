/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Copy.h"

#include <stdexcept>

#include "flashlight/fl/runtime/Device.h"
#include "flashlight/fl/runtime/Stream.h"
#include "flashlight/fl/tensor/BackendRegistry.h"
#include "flashlight/fl/tensor/JitBackend.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl {

namespace {

// Pick the backend that should execute the copy.
TensorBackend& pickCopyBackend(
    DeviceType srcDevType,
    TensorBackend& srcBackend,
    DeviceType dstDevType,
    TensorBackend& dstBackend) {
  if (dstDevType != DeviceType::x64)
    return dstBackend;
  if (srcDevType != DeviceType::x64)
    return srcBackend;
  return dstBackend;
}

// Pick the stream on which copy work is enqueued.
const Stream& pickCopyStream(
    DeviceType srcDevType,
    const Stream& srcStream,
    DeviceType dstDevType,
    const Stream& dstStream) {
  if (dstDevType == DeviceType::CUDA)
    return dstStream;
  if (srcDevType == DeviceType::CUDA)
    return srcStream;
  return dstStream;
}

} // namespace

void copy(Tensor& dst, const Tensor& src) {
  if (dst.shape() != src.shape())
    throw std::invalid_argument("[fl::copy] shape mismatch");
  if (dst.type() != src.type())
    throw std::invalid_argument("[fl::copy] dtype mismatch");
  if (dst.elements() == 0)
    return;

  auto srcDevType = src.stream().device().type();
  auto dstDevType = dst.stream().device().type();

  auto& copyBackend = pickCopyBackend(
      srcDevType, src.backend(),
      dstDevType, dst.backend());
  auto& copyStream = pickCopyStream(
      srcDevType, src.stream(), dstDevType, dst.stream());

  copyBackend.copy(dst, src, copyStream);
}

Tensor copy(const Tensor& src, std::shared_ptr<const Stream> dstStream) {
  auto dstDevType = dstStream->device().type();
  auto dstBackendType = backendTypeForDevice(dstDevType);

  // JIT path: create TransferNode (lazy, no materialization)
  if (src.backendType() == TensorBackendType::Jit) {
    auto& jitBackend = static_cast<JitBackend&>(src.backend());
    return jitBackend.transfer(src, dstDevType, dstBackendType, dstStream);
  }

  // Non-JIT path: allocate uninitialized dst, then imperative copy
  auto& targetBackend = fl::getBackend(dstBackendType);
  Tensor dst = targetBackend.empty(src.shape(), src.type());
  copy(dst, src); // calls imperative overload above
  return dst;
}

} // namespace fl
