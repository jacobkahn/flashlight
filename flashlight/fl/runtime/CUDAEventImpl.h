/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

namespace fl {

class Stream;

namespace detail {

struct EventImpl;

/**
 * Factory: creates a CUDAEventImpl and records it on the given CUDA stream.
 * Defined in CUDAEventImpl.cpp (compiled only when FL_USE_CUDA).
 */
std::shared_ptr<EventImpl> makeCUDAEventImpl(const Stream& stream);

} // namespace detail
} // namespace fl
