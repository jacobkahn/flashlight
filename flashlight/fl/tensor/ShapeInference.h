/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl {

Shape inferReductionOutputShape(
    const Shape& inputShape,
    const std::vector<int>& axes,
    bool keepDims);

/**
 * Pad each shape to the same rank by appending trailing 1s, where the
 * target rank is max(maxRank(shapes), minOutputRank).
 */
std::vector<Shape> broadcastShapesToMaxRank(
    const std::vector<Shape>& shapes,
    unsigned minOutputRank);

/// Convenience overload that extracts shapes from tensors.
std::vector<Shape> broadcastShapesToMaxRank(
    const std::vector<Tensor>& tensors,
    unsigned minOutputRank);

Shape inferTransposeOutputShape(const Shape& inputShape, const Shape& axes);

Shape inferTileOutputShape(const Shape& inputShape, const Shape& tileDims);

/**
 * Infer the output shape of element-wise conditional select.
 *
 * Current Flashlight backends require condition, x, and y to have identical
 * shapes. Frontend-style broadcasted where semantics should be added as a
 * separate API/backend change.
 */
Shape inferWhereOutputShape(
    const Shape& conditionShape,
    const Shape& xShape,
    const Shape& yShape);

/**
 * Infer the output shape of concatenation along axisToConcat.
 *
 * Each input is described by its shape and element count. Empty inputs
 * (elements == 0) contribute 0 along the concat axis.
 */
Shape inferConcatenateOutputShape(
    const std::vector<Shape>& shapes,
    const std::vector<Dim>& elements,
    unsigned axisToConcat);

/// Convenience overload that extracts shapes/elements from tensors.
Shape inferConcatenateOutputShape(
    const std::vector<Tensor>& tensors,
    unsigned axisToConcat);

Shape inferPadOutputShape(
    const Shape& inputShape,
    const std::vector<std::pair<int, int>>& padWidths);

Shape inferMatmulOutputShape(
    const Shape& lhsShape,
    const Shape& rhsShape,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp);

} // namespace fl
