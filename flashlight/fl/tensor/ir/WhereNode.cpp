/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/WhereNode.h"

#include <stdexcept>

#include "flashlight/fl/tensor/ShapeInference.h"
#include "flashlight/fl/tensor/Types.h"

namespace fl {

WhereNode::WhereNode(
    NodePtr condition,
    NodePtr x,
    NodePtr y,
    const Shape& outputShape,
    dtype outputDtype,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait(
          {condition, x, y},
          outputShape,
          outputDtype,
          std::move(stream)) {}

WhereNodePtr WhereNode::create(
    NodePtr condition,
    NodePtr x,
    NodePtr y,
    std::shared_ptr<const Stream> stream) {
  if (condition->dataType() != dtype::b8) {
    throw std::invalid_argument(
        "fl::WhereNode: condition must be of type b8 (boolean)");
  }
  auto outputShape =
      inferWhereOutputShape(condition->shape(), x->shape(), y->shape());
  auto outputDtype = promoteDtypes(x->dataType(), y->dataType());
  return std::make_shared<WhereNode>(
      condition,
      x,
      y,
      outputShape,
      outputDtype,
      std::move(stream),
      PrivateHelper{});
}

NodePtr WhereNode::condition() const {
  return getInput(kConditionIdx);
}

NodePtr WhereNode::x() const {
  return getInput(kXIdx);
}

NodePtr WhereNode::y() const {
  return getInput(kYIdx);
}

} // namespace fl
