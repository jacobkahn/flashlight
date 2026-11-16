/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/ReshapeNode.h"

#include <sstream>
#include <stdexcept>

namespace fl {

ReshapeNode::ReshapeNode(
    NodePtr input,
    const Shape& newShape,
    dtype dtype,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait({input}, newShape, dtype, std::move(stream)) {}

ReshapeNodePtr ReshapeNode::create(
    NodePtr input,
    const Shape& newShape,
    std::shared_ptr<const Stream> stream) {
  if (input->shape().elements() != newShape.elements()) {
    std::ostringstream oss;
    oss << "[ReshapeNode::create] Cannot reshape from " << input->shape()
        << " to " << newShape << ": element count mismatch";
    throw std::invalid_argument(oss.str());
  }
  return std::make_shared<ReshapeNode>(
      input, newShape, input->dataType(), std::move(stream), PrivateHelper{});
}

NodePtr ReshapeNode::input() const {
  return getInput(kInputIdx);
}

} // namespace fl
