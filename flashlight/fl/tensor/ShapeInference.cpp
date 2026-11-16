/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ShapeInference.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace fl {

Shape inferReductionOutputShape(
    const Shape& inputShape,
    const std::vector<int>& axes,
    bool keepDims) {
  // NOTE the tensor API requires this, see reduction tests
  if (inputShape.ndim() == 0) {
    return inputShape;
  }
  for (const auto axis : axes) {
    if (axis < 0 || axis >= inputShape.ndim()) {
      std::ostringstream oss;
      oss << "[inferReductionOutputShape] Invalid axis for reduction: " << axis
          << " for tensor of shape: " << inputShape;
      throw std::invalid_argument(oss.str());
    }
  }
  std::unordered_set<int> axesToReduce;
  if (axes.empty()) {
    for (int axis = 0; axis < inputShape.ndim(); axis++) {
      axesToReduce.insert(axis);
    }
  } else {
    axesToReduce.insert(axes.begin(), axes.end());
  }
  std::vector<Dim> outputDims;
  for (int axis = 0; axis < inputShape.ndim(); axis++) {
    if (axesToReduce.contains(axis)) {
      if (keepDims) {
        outputDims.push_back(1);
      }
    } else {
      outputDims.push_back(inputShape.dim(axis));
    }
  }
  return Shape(outputDims);
}

std::vector<Shape> broadcastShapesToMaxRank(
    const std::vector<Shape>& shapes,
    const unsigned minOutputRank) {
  assert(!shapes.empty());
  int maxRank = minOutputRank;
  for (const auto& shape : shapes) {
    maxRank = std::max(maxRank, shape.ndim());
  }
  std::vector<Shape> result;
  for (const auto& shape : shapes) {
    auto dims = shape.get();
    dims.insert(dims.end(), maxRank - dims.size(), 1);
    result.push_back(Shape(dims));
  }
  return result;
}

std::vector<Shape> broadcastShapesToMaxRank(
    const std::vector<Tensor>& tensors,
    const unsigned minOutputRank) {
  std::vector<Shape> shapes;
  shapes.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    shapes.push_back(tensor.shape());
  }
  return broadcastShapesToMaxRank(shapes, minOutputRank);
}

Shape inferTransposeOutputShape(const Shape& inputShape, const Shape& axes) {
  const auto inputRank = inputShape.ndim();
  std::vector<Dim> outputDims = inputShape.get();
  std::vector<Dim> oldToNewAxes = axes.get();
  if (axes.ndim() == 0) { // default, reverse all axes
    oldToNewAxes.resize(inputRank);
    std::reverse(outputDims.begin(), outputDims.end());
    std::iota(oldToNewAxes.begin(), oldToNewAxes.end(), 0);
    std::reverse(oldToNewAxes.begin(), oldToNewAxes.end());
  } else if (axes.ndim() == inputRank) {
    for (int axis = 0; axis < axes.ndim(); axis++) {
      outputDims[axis] = inputShape.dim(oldToNewAxes[axis]);
    }
  } else {
    std::ostringstream oss;
    oss << "[inferTransposeOutputShape] Invalid axes: " << axes
        << " for shape: " << inputShape;
    throw std::runtime_error(oss.str());
  }
  return Shape(outputDims);
}

Shape inferTileOutputShape(const Shape& inputShape, const Shape& tileDims) {
  std::vector<Dim> paddedTensorDims = inputShape.get();
  std::vector<Dim> paddedTileDims = tileDims.get();
  const auto inputRank = inputShape.ndim();
  const auto tileRank = tileDims.ndim();
  if (inputRank > tileRank) {
    const auto diff = inputRank - tileRank;
    // Left-align tile dims: tile factors apply starting from dim 0
    paddedTileDims.insert(paddedTileDims.end(), diff, 1);
  } else {
    const auto diff = tileRank - inputRank;
    paddedTensorDims.insert(paddedTensorDims.begin(), diff, 1);
  }
  std::vector<Dim> outputDims;
  for (unsigned i = 0; i < paddedTensorDims.size(); i++) {
    outputDims.push_back(paddedTensorDims[i] * paddedTileDims[i]);
  }
  return Shape(outputDims);
}

Shape inferWhereOutputShape(
    const Shape& conditionShape,
    const Shape& xShape,
    const Shape& yShape) {
  if (conditionShape != xShape || conditionShape != yShape) {
    std::ostringstream oss;
    oss << "[inferWhereOutputShape] where currently requires identical "
        << "condition, x, and y shapes, got condition: " << conditionShape
        << ", x: " << xShape << ", y: " << yShape;
    throw std::invalid_argument(oss.str());
  }
  return conditionShape;
}

Shape inferConcatenateOutputShape(
    const std::vector<Shape>& shapes,
    const std::vector<Dim>& elements,
    const unsigned axisToConcat) {
  if (shapes.empty()) {
    throw std::runtime_error(
        "[inferConcatenateOutputShape] Nothing to concatenate");
  }
  const auto padded = broadcastShapesToMaxRank(shapes, axisToConcat + 1);
  const unsigned outputRank = padded.front().ndim();
  if (axisToConcat >= outputRank) {
    std::ostringstream oss;
    oss << "[inferConcatenateOutputShape] Axis too big: " << axisToConcat;
    throw std::runtime_error(oss.str());
  }
  std::vector<Dim> outputDims;
  for (unsigned axis = 0; axis < outputRank; axis++) {
    if (axis == axisToConcat) {
      Dim concatenatedDim = 0;
      for (unsigned i = 0; i < shapes.size(); i++) {
        // Empty tensors (0 elements) contribute 0 along the concat axis
        if (elements[i] > 0) {
          concatenatedDim += padded[i].dim(axis);
        }
      }
      outputDims.push_back(concatenatedDim);
    } else {
      // For non-concat dims, use the first non-empty tensor's dimension,
      // or 1 if all tensors are empty
      Dim dim = 1;
      for (unsigned i = 0; i < shapes.size(); i++) {
        if (elements[i] > 0) {
          dim = padded[i].dim(axis);
          break;
        }
      }
      outputDims.push_back(dim);
    }
  }
  return Shape(outputDims);
}

Shape inferConcatenateOutputShape(
    const std::vector<Tensor>& tensors,
    const unsigned axisToConcat) {
  std::vector<Shape> shapes;
  std::vector<Dim> elements;
  shapes.reserve(tensors.size());
  elements.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    shapes.push_back(tensor.shape());
    elements.push_back(tensor.elements());
  }
  return inferConcatenateOutputShape(shapes, elements, axisToConcat);
}

Shape inferPadOutputShape(
    const Shape& inputShape,
    const std::vector<std::pair<int, int>>& padWidths) {
  std::vector<Dim> outputDims = inputShape.get();
  if (padWidths.size() > static_cast<size_t>(inputShape.ndim())) {
    throw std::runtime_error("[inferPadOutputShape] too many paddings");
  }
  for (unsigned axis = 0; axis < padWidths.size(); axis++) {
    const auto& [beforeDim, afterDim] = padWidths[axis];
    outputDims[axis] += beforeDim + afterDim;
  }
  return Shape(outputDims);
}

Shape inferMatmulOutputShape(
    const Shape& lhsShape,
    const Shape& rhsShape,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp) {
  std::vector<Dim> lhsDims = lhsShape.get();
  std::vector<Dim> rhsDims = rhsShape.get();
  const auto lhsRank = lhsDims.size();
  const auto rhsRank = rhsDims.size();
  const bool isLhsScalarOrVector = lhsRank <= 1;
  const bool isRhsScalarOrVector = rhsRank <= 1;

  // Row-major: matrix dims are the last two dims [..., M, K] x [..., K, N]
  // Scalars/vectors are padded by appending dims at the end
  if (isLhsScalarOrVector) { // pad to (..., 1, K) or (..., 1, 1)
    lhsDims.insert(lhsDims.end(), 2 - lhsRank, 1);
    // For a vector {K}, we get {K, 1}; swap to {1, K}
    std::swap(lhsDims[lhsDims.size() - 2], lhsDims[lhsDims.size() - 1]);
  } else if (lhsProp == MatrixProperty::Transpose) {
    std::swap(lhsDims[lhsDims.size() - 2], lhsDims[lhsDims.size() - 1]);
  }
  if (isRhsScalarOrVector) { // pad to (..., K, 1)
    rhsDims.insert(rhsDims.end(), 2 - rhsRank, 1);
  } else if (rhsProp == MatrixProperty::Transpose) {
    std::swap(rhsDims[rhsDims.size() - 2], rhsDims[rhsDims.size() - 1]);
  }

  // In row-major: lhs is [..., M, K], rhs is [..., K, N]
  const auto lM = lhsDims[lhsDims.size() - 2];
  const auto lK = lhsDims[lhsDims.size() - 1];
  const auto rK = rhsDims[rhsDims.size() - 2];
  const auto rN = rhsDims[rhsDims.size() - 1];

  // shape compatibility check on contraction dim
  bool onlyOneIsBatched = (lhsRank >= 3) != (rhsRank >= 3);
  if (lK != rK) {
    std::ostringstream oss;
    oss << "Cannot perform matmul for tensors of shapes: " << lhsShape
        << " and " << rhsShape;
    throw std::invalid_argument(oss.str());
  }

  // Broadcast batch dims (leading dims, right-aligned)
  std::vector<Dim> lhsBatch(lhsDims.begin(), lhsDims.end() - 2);
  std::vector<Dim> rhsBatch(rhsDims.begin(), rhsDims.end() - 2);
  std::vector<Dim> batchDims;
  if (!onlyOneIsBatched && !lhsBatch.empty() && !rhsBatch.empty()) {
    if (lhsBatch != rhsBatch) {
      std::ostringstream oss;
      oss << "Cannot perform matmul for tensors of shapes: " << lhsShape
          << " and " << rhsShape;
      throw std::invalid_argument(oss.str());
    }
    batchDims = lhsBatch;
  } else if (lhsBatch.size() >= rhsBatch.size()) {
    batchDims = lhsBatch;
  } else {
    batchDims = rhsBatch;
  }

  // Output: batch dims + {M, N}
  std::vector<Dim> outputDims = batchDims;
  outputDims.push_back(lM);
  outputDims.push_back(rN);

  Shape outputShape(outputDims);
  if (isLhsScalarOrVector || isRhsScalarOrVector) {
    outputShape = {outputShape.elements()};
  }
  return outputShape;
}

} // namespace fl
