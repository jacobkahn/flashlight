/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <vector>

#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/ops/native/CpuTensorHelpers.h"

namespace fl {
namespace native {

template <typename T, typename V>
Tensor fullWithTypeCpu(const Shape& shape, V value, const dtype type) {
  std::vector<T> data(shape.elements());
  std::fill(data.begin(), data.end(), static_cast<T>(value));
  return makeCpuTensor(shape, type, data.data());
}

} // namespace native
} // namespace fl
