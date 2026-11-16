/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <utility>
#include <vector>

#include "flashlight/fl/tensor/Tensor.h"

namespace fl {
namespace native {

/**
 * CPU nonzero: find indices of non-zero elements in a flattened tensor.
 */
Tensor nonzeroCpu(const Tensor& tensor);

/**
 * Pad implementation. Operates on pre-allocated output/input Tensors via
 * bufferPtr() — safe for async worker threads.
 */
void padImpl(
    Tensor& out,
    const Tensor& in,
    const Shape& inShape,
    const Shape& outShape,
    const std::vector<std::pair<int, int>>& padWidths,
    PadType padType);

/**
 * CPU element-wise conditional select: result[i] = cond[i] ? x[i] : y[i].
 * Condition must be b8 or u8. x and y must have the same shape.
 */
Tensor whereCpu(const Tensor& condition, const Tensor& x, const Tensor& y);

/**
 * CPU tile via coordinate decomposition. Used as f64 fallback when
 * OneDNN concat-based tile doesn't support the dtype.
 */
Tensor tileCpu(
    const Tensor& tensor,
    const Shape& tensorShape,
    const Shape& tileDims);

} // namespace native
} // namespace fl
