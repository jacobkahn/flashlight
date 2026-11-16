/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/runtime/SynchronousStream.h"

#include <stdexcept>

#include "flashlight/fl/runtime/Event.h"
#include "flashlight/fl/runtime/EventImpl.h"

#if FL_BACKEND_CUDA
#include "flashlight/fl/runtime/CUDAStream.h"
#endif

namespace fl {

X64Device& SynchronousStream::device() {
  return device_;
}

const X64Device& SynchronousStream::device() const {
  return device_;
}

void SynchronousStream::relativeSync(const SynchronousStream& waitOn) const {
  waitOn.sync();
}

void SynchronousStream::relativeSync(const Stream& waitOn) const {
  switch (waitOn.type()) {
    case StreamType::Synchronous:
      relativeSync(waitOn.impl<SynchronousStream>());
      break;
    case StreamType::AsyncCPU:
      waitOn.sync();
      break;
#if FL_BACKEND_CUDA
    case StreamType::CUDA:
      waitOn.sync();
      break;
#endif
    default:
      throw std::runtime_error(
          "[SynchronousStream::relativeSync] "
          "Unsupported stream type for relative synchronization");
  }
}

Event SynchronousStream::createEvent() const {
  return Event(std::make_shared<detail::NoOpEventImpl>());
}

} // namespace fl
