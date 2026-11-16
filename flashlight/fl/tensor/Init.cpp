/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Init.h"

#include <mutex>

#include "flashlight/fl/common/Logging.h"

namespace fl {
namespace {
std::once_flag flInitFlag;
}

/**
 * Initialize Flashlight. Sets up logging and signal handlers.
 *
 * Tensor backend initialization is lazy — the default backend singleton is
 * created on first use (e.g., when the first Tensor is constructed). This
 * avoids unnecessarily initializing backends (and their heavy dependencies
 * like CUDA) that may be overridden via setDefaultTensorType<T>().
 *
 * Body is only run once per process. Subsequent calls are no-ops.
 */
void init() {
  std::call_once(flInitFlag, []() {
    initLogging();
  });
}

} // namespace fl
