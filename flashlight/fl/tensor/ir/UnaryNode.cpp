/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/UnaryNode.h"

#include "flashlight/fl/tensor/ir/OpProperties.h"

namespace fl {

UnaryNode::UnaryNode(
    NodePtr input, UnaryOp op, const Shape& shape, dtype dtype,
    std::shared_ptr<const Stream> stream, PrivateHelper)
    : NodeTrait({input}, shape, dtype, std::move(stream)), op_(op) {}

UnaryNodePtr UnaryNode::create(
    NodePtr input, UnaryOp op,
    std::shared_ptr<const Stream> stream) {
  auto outputDtype = inferOutputDtype(op, input->dataType());
  return std::make_shared<UnaryNode>(
      input, op, Shape(input->shape()), outputDtype,
      std::move(stream), PrivateHelper{});
}

UnaryOp UnaryNode::op() const {
  return op_;
}

NodePtr UnaryNode::input() const {
  return getInput(kInputIdx);
}

} // namespace fl
