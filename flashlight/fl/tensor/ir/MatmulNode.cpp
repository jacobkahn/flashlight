/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/MatmulNode.h"

#include "flashlight/fl/tensor/ShapeInference.h"

namespace fl {

MatmulNode::MatmulNode(
    NodePtr lhs,
    NodePtr rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp,
    const Shape& shape,
    dtype dtype,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait({lhs, rhs}, shape, dtype, std::move(stream)), lhsProp_(lhsProp), rhsProp_(rhsProp) {}

MatmulNodePtr MatmulNode::create(
    NodePtr lhs,
    NodePtr rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp,
    std::shared_ptr<const Stream> stream) {
  auto outputShape =
      inferMatmulOutputShape(lhs->shape(), rhs->shape(), lhsProp, rhsProp);
  auto outputDtype = promoteDtypes(lhs->dataType(), rhs->dataType());
  return std::make_shared<MatmulNode>(
      lhs, rhs, lhsProp, rhsProp, outputShape, outputDtype, std::move(stream), PrivateHelper{});
}

MatrixProperty MatmulNode::lhsProp() const {
  return lhsProp_;
}

MatrixProperty MatmulNode::rhsProp() const {
  return rhsProp_;
}

NodePtr MatmulNode::lhs() const {
  return getInput(kLhsIdx);
}

NodePtr MatmulNode::rhs() const {
  return getInput(kRhsIdx);
}

} // namespace fl
