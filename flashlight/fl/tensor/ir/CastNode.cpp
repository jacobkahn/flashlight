/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/CastNode.h"

namespace fl {

CastNode::CastNode(
    NodePtr input,
    dtype targetDtype,
    const Shape& shape,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait({input}, shape, targetDtype, std::move(stream)),
      targetDtype_(targetDtype) {}

NodePtr CastNode::create(
    NodePtr input,
    dtype targetDtype,
    std::shared_ptr<const Stream> stream) {
  if (input->dataType() == targetDtype) {
    return input; // no-op cast elimination
  }
  return std::make_shared<CastNode>(
      input, targetDtype, Shape(input->shape()), std::move(stream),
      PrivateHelper{});
}

NodePtr CastNode::input() const {
  return getInput(kInputIdx);
}

dtype CastNode::targetDtype() const {
  return targetDtype_;
}

} // namespace fl
