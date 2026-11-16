/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/native/Binary.h"

namespace fl {
namespace native {

Tensor logicalAndCpu(const Tensor& lhs, const Tensor& rhs) {
  return cpuBinaryOpWithBroadcast<int32_t>(
      lhs.astype(dtype::s32), rhs.astype(dtype::s32),
      [](int32_t a, int32_t b) -> int32_t {
        return (a != 0 && b != 0) ? 1 : 0;
      }).astype(dtype::b8);
}

Tensor logicalOrCpu(const Tensor& lhs, const Tensor& rhs) {
  return cpuBinaryOpWithBroadcast<int32_t>(
      lhs.astype(dtype::s32), rhs.astype(dtype::s32),
      [](int32_t a, int32_t b) -> int32_t {
        return (a != 0 || b != 0) ? 1 : 0;
      }).astype(dtype::b8);
}

} // namespace native
} // namespace fl
