/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/ClipNode.h"

namespace fl {

ClipNode::ClipNode(
    NodePtr input,
    NodePtr low,
    NodePtr high,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait(
          {input, low, high},
          input->shape(),
          input->dataType(),
          std::move(stream)) {}

ClipNodePtr ClipNode::create(
    NodePtr input,
    NodePtr low,
    NodePtr high,
    std::shared_ptr<const Stream> stream) {
  return std::make_shared<ClipNode>(
      input, low, high, std::move(stream), PrivateHelper{});
}

NodePtr ClipNode::input() const {
  return getInput(kInputIdx);
}

NodePtr ClipNode::low() const {
  return getInput(kLowIdx);
}

NodePtr ClipNode::high() const {
  return getInput(kHighIdx);
}

} // namespace fl
