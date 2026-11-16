/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/ops/onednn/OneDnnAsyncCPUStream.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace fl {
namespace detail {

#define FL_CPU_BACKEND_UNIMPLEMENTED \
  throw std::invalid_argument(       \
      "CPUBackend::" + std::string(__func__) + " - unimplemented.");

// Storage of `t` laid out contiguously. Native kernels index their input
// linearly from bufferPtr(), so a strided view is copied first (on the same
// stream, so FIFO order still holds); the returned SharedData keeps that copy
// alive for the enqueued task.
inline std::shared_ptr<CPUTensor::SharedData> contiguousSharedData(
    const Tensor& t) {
  if (t.isContiguous()) {
    return toCPUTensor(t).getSharedData();
  }
  auto contiguous = t.asContiguousTensor();
  return toCPUTensor(contiguous).getSharedData();
}

// Helper: enqueue a native (non-OneDNN) op on the async worker thread.
// Pre-allocates an output CPUTensor, then enqueues the op lambda which
// receives the output and input Tensors.  Inside the lambda, use
// toCPUTensor(t).bufferPtr() to access data — do NOT call host()/sync().
//
// IMPORTANT: Tensor copy = deep copy (clone + sync). To share buffers between
// the caller and the lambda, we capture shared_ptr<SharedData> and reconstruct
// lightweight Tensor wrappers inside the lambda.
//
// relativeSync: currently a no-op because all ops share a single FIFO
// stream, so ordering is implicit.  Kept as a cheap guard for future
// multi-stream support.
template <typename F>
Tensor enqueueNativeOp(
    const Tensor& input,
    const Shape& outShape,
    dtype outType,
    std::shared_ptr<OneDnnAsyncCPUStream>& asyncStream,
    F&& op) {
  auto output = toTensor<CPUTensor>(outShape, outType, nullptr, Location::Host);
  // Capture SharedData (shared_ptr) so the lambda shares the same buffers
  // as the returned output and the input — no deep copy, no sync.
  auto outData = toCPUTensor(output).getSharedData();
  auto inData = contiguousSharedData(input);
  auto inShape = input.shape();
  auto inType = input.type();
  asyncStream->relativeSync(input.stream());
  asyncStream->enqueue(
      [outData, inData, outShape, inShape, outType, inType,
       stream = asyncStream, op = std::forward<F>(op)]() {
        // Create lightweight Tensor wrappers sharing the same buffers
        Tensor out(std::make_unique<CPUTensor>(outData, outShape, outType, stream));
        Tensor in(std::make_unique<CPUTensor>(inData, inShape, inType, stream));
        op(out, in);
      });
  return output;
}

template <typename F>
Tensor enqueueNativeOp(
    const Tensor& input,
    std::shared_ptr<OneDnnAsyncCPUStream>& asyncStream,
    F&& op) {
  return enqueueNativeOp(
      input, input.shape(), input.type(), asyncStream, std::forward<F>(op));
}

} // namespace detail
} // namespace fl
