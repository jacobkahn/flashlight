/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/ProjectionNode.h"

#include <stdexcept>

namespace fl {

ProjectionNode::ProjectionNode(
    NodePtr parent,
    unsigned outputIndex,
    const Shape& shape,
    dtype type,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait({std::move(parent)}, shape, type, std::move(stream)),
      outputIndex_(outputIndex) {}

ProjectionNodePtr ProjectionNode::create(
    NodePtr parent,
    OutputIndex outputIndex,
    const Shape& shape,
    dtype type,
    std::shared_ptr<const Stream> stream) {
  if (parent->numOutputs() == 0) {
    throw std::invalid_argument(
        "[ProjectionNode::create] parent is not a multi-output node");
  }
  if (outputIndex >= parent->numOutputs()) {
    throw std::invalid_argument(
        "[ProjectionNode::create] outputIndex " +
        std::to_string(outputIndex) + " >= parent numOutputs " +
        std::to_string(parent->numOutputs()));
  }
  if (!stream) {
    stream = parent->stream();
  }
  return std::make_shared<ProjectionNode>(
      std::move(parent), outputIndex, shape, type, std::move(stream),
      PrivateHelper{});
}

unsigned ProjectionNode::outputIndex() const {
  return outputIndex_;
}

NodePtr ProjectionNode::parent() const {
  return getInput(0);
}

} // namespace fl
