/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Random.h"

#include "flashlight/fl/tensor/BackendRegistry.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/JitTensorBase.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl {

namespace {

TensorBackend& backendFor(StreamOrDevice& s) {
  if (s.hasDevice()) {
    return getOrCreateJitBackend(
        getBackend(backendTypeForDevice(s.device().type())));
  }
  return defaultTensorBackend();
}

} // namespace

void setSeed(const int seed) {
  defaultTensorBackend().setSeed(seed);
}

Tensor randn(const Shape& shape, dtype type, StreamOrDevice s) {
  return backendFor(s).randn(shape, type, s.takeStream());
}

Tensor rand(const Shape& shape, dtype type, StreamOrDevice s) {
  return backendFor(s).rand(shape, type, s.takeStream());
}

} // namespace fl
