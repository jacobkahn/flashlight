/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/onednn/OneDnnAsyncCPUStream.h"

namespace fl {

OneDnnAsyncCPUStream::OneDnnAsyncCPUStream(const dnnl::engine& engine)
    : AsyncCPUStream(), // protected base ctor — starts worker thread
      nativeStream_(engine) {}

std::shared_ptr<OneDnnAsyncCPUStream> OneDnnAsyncCPUStream::create(
    const dnnl::engine& engine) {
  auto* raw = new OneDnnAsyncCPUStream(engine);
  auto s = std::shared_ptr<OneDnnAsyncCPUStream>(raw);
  raw->device().addStream(s);
  return s;
}

} // namespace fl
