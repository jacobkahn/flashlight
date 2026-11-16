/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include <sstream>
#include <stdexcept>

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/ir/OpProperties.h"

namespace fl {

BinaryNode::BinaryNode(NodePtr lhs, NodePtr rhs, BinaryOp op, const Shape& shape, dtype dtype,
                       std::shared_ptr<const Stream> stream, PrivateHelper)
    : NodeTrait({lhs, rhs}, shape, dtype, std::move(stream)), op_(op) {}

BinaryNodePtr BinaryNode::create(
    NodePtr lhs, NodePtr rhs, BinaryOp op,
    std::shared_ptr<const Stream> stream) {
  // broadcastShapes throws if shapes are incompatible
  auto outputShape = broadcastShapes(lhs->shape(), rhs->shape());
  auto outputDtype = inferOutputDtype(op, lhs->dataType(), rhs->dataType());
  return std::make_shared<BinaryNode>(
      lhs, rhs, op, outputShape, outputDtype, std::move(stream), PrivateHelper{});
}

BinaryOp BinaryNode::op() const {
  return op_;
}

NodePtr BinaryNode::lhs() const {
  return getInput(kLhsIdx);
}

NodePtr BinaryNode::rhs() const {
  return getInput(kRhsIdx);
}

} // namespace fl
