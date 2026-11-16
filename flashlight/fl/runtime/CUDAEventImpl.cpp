/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/runtime/CUDAEventImpl.h"

#include <cuda_runtime.h>

#include "flashlight/fl/runtime/CUDAStream.h"
#include "flashlight/fl/runtime/CUDAUtils.h"
#include "flashlight/fl/runtime/Stream.h"
#include "flashlight/fl/runtime/EventImpl.h"

namespace fl {
namespace detail {

struct CUDAEventImpl : EventImpl {
  cudaEvent_t event{nullptr};

  CUDAEventImpl() {
    FL_CUDA_CHECK(
        cudaEventCreate(&event, cudaEventDefault | cudaEventDisableTiming));
  }

  ~CUDAEventImpl() override {
    if (event) {
      cudaEventDestroy(event);
    }
  }

  // Non-copyable (shared via shared_ptr)
  CUDAEventImpl(const CUDAEventImpl&) = delete;
  CUDAEventImpl& operator=(const CUDAEventImpl&) = delete;

  void wait() const override {
    FL_CUDA_CHECK(cudaEventSynchronize(event));
  }

  bool isReady() const override {
    auto err = cudaEventQuery(event);
    if (err == cudaSuccess) {
      return true;
    }
    if (err == cudaErrorNotReady) {
      return false;
    }
    FL_CUDA_CHECK(err);
    return false; // unreachable
  }
};

std::shared_ptr<EventImpl> makeCUDAEventImpl(const Stream& stream) {
  auto impl = std::make_shared<CUDAEventImpl>();
  FL_CUDA_CHECK(
      cudaEventRecord(impl->event, stream.impl<CUDAStream>().handle()));
  return impl;
}

} // namespace detail
} // namespace fl
