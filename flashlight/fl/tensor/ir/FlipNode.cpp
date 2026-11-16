/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/FlipNode.h"

#include <stdexcept>

namespace fl {

FlipNode::FlipNode(
    NodePtr input,
    unsigned axis,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait({input}, input->shape(), input->dataType(), std::move(stream)),
      axis_(axis) {}

FlipNodePtr FlipNode::create(
    NodePtr input,
    unsigned axis,
    std::shared_ptr<const Stream> stream) {
  if (axis >= static_cast<unsigned>(input->shape().ndim())) {
    throw std::invalid_argument("fl::FlipNode: axis out of range");
  }
  return std::make_shared<FlipNode>(
      input, axis, std::move(stream), PrivateHelper{});
}

NodePtr FlipNode::input() const {
  return getInput(kInputIdx);
}

unsigned FlipNode::axis() const {
  return axis_;
}

} // namespace fl
