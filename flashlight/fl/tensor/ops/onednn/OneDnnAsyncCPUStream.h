/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <dnnl.hpp>

#include "flashlight/fl/runtime/AsyncCPUStream.h"

namespace fl {

/**
 * An AsyncCPUStream that additionally owns a dnnl::stream.
 *
 * Each instance has its own dnnl::stream (which is NOT thread-safe), so
 * multiple OneDnnAsyncCPUStreams can coexist — each worker thread uses its
 * own dnnl::stream for multi-stream parallelism.
 */
class OneDnnAsyncCPUStream : public AsyncCPUStream {
  dnnl::stream nativeStream_;

 public:
  /**
   * Create an OneDnnAsyncCPUStream and register it with the active x64 device.
   */
  static std::shared_ptr<OneDnnAsyncCPUStream> create(
      const dnnl::engine& engine);

  dnnl::stream& nativeStream() { return nativeStream_; }
  const dnnl::stream& nativeStream() const { return nativeStream_; }

 private:
  explicit OneDnnAsyncCPUStream(const dnnl::engine& engine);
};

} // namespace fl
