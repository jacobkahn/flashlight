/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <vector>

#include "flashlight/fl/common/Utils.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl::test {

// All dtypes in each category
inline const std::vector<fl::dtype> kAllFpDtypes = {
    fl::dtype::f16,
    fl::dtype::f32,
    fl::dtype::f64,
    fl::dtype::bf16};

inline const std::vector<fl::dtype> kAllIntegralDtypes = {
    // b8 excluded: casting clips values to 0/1 which breaks most fixtures
    fl::dtype::u8,
    fl::dtype::s16,
    fl::dtype::u16,
    fl::dtype::s32,
    fl::dtype::u32,
    fl::dtype::s64,
    fl::dtype::u64};

// Apply func to each dtype in the list that the default backend supports.
// Unsupported dtypes are silently skipped (not GTEST_SKIP) because multiple
// dtypes are tested per TEST() — skipping would skip the entire test.
inline void forEachSupportedDtype(
    const std::vector<fl::dtype>& dtypes,
    std::function<void(fl::dtype)> func) {
  for (auto dt : dtypes) {
    if (fl::dtypeSupported(dt)) {
      func(dt);
    }
  }
}

inline void forEachSupportedFpDtype(std::function<void(fl::dtype)> func) {
  forEachSupportedDtype(kAllFpDtypes, func);
}

inline void forEachSupportedIntegralDtype(
    std::function<void(fl::dtype)> func) {
  forEachSupportedDtype(kAllIntegralDtypes, func);
}

inline void forEachSupportedDtype(std::function<void(fl::dtype)> func) {
  forEachSupportedFpDtype(func);
  forEachSupportedIntegralDtype(func);
}

} // namespace fl::test
