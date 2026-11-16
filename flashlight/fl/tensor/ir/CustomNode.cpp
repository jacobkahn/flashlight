/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/CustomNode.h"

#include <stdexcept>
#include <utility>

#include "flashlight/fl/tensor/ir/OutputDesc.h"

namespace fl {

// Single-output constructor
CustomNode::CustomNode(
    std::string&& name,
    std::vector<NodePtr>&& inputs,
    const Shape& shape,
    dtype dtype,
    EvalFunc&& evalFunc,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait(std::move(inputs), shape, dtype, std::move(stream)),
      name_(name),
      evalFunc_(std::move(evalFunc)) {}

// Multi-output constructor
CustomNode::CustomNode(
    std::string&& name,
    std::vector<NodePtr>&& inputs,
    std::vector<OutputDesc> outputDescs,
    MultiOutputEvalFunc&& evalFunc,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait(
          std::move(inputs),
          outputDescs.at(0).shape,
          outputDescs.at(0).type,
          std::move(stream)),
      name_(name),
      multiOutputEvalFunc_(std::move(evalFunc)) {
  outputs_.resize(outputDescs.size());
}

CustomNodePtr CustomNode::create(
    std::string&& name,
    std::vector<NodePtr>&& inputs,
    const Shape& shape,
    dtype dtype,
    EvalFunc&& evalFunc,
    std::shared_ptr<const Stream> stream) {
  return std::make_shared<CustomNode>(
      std::move(name), std::move(inputs), shape, dtype, std::move(evalFunc),
      std::move(stream), PrivateHelper{});
}

CustomNodePtr CustomNode::createMultiOutput(
    std::string&& name,
    std::vector<NodePtr>&& inputs,
    std::vector<OutputDesc> outputDescs,
    MultiOutputEvalFunc&& evalFunc,
    std::shared_ptr<const Stream> stream) {
  return std::make_shared<CustomNode>(
      std::move(name), std::move(inputs), std::move(outputDescs),
      std::move(evalFunc), std::move(stream), PrivateHelper{});
}

const std::string& CustomNode::name() const {
  return name_;
}

const CustomNode::EvalFunc& CustomNode::evalFunc() const {
  return evalFunc_;
}

bool CustomNode::isMultiOutput() const {
  return multiOutputEvalFunc_.has_value();
}

const CustomNode::MultiOutputEvalFunc& CustomNode::multiOutputEvalFunc() const {
  if (!multiOutputEvalFunc_.has_value()) {
    throw std::runtime_error(
        "[CustomNode::multiOutputEvalFunc] not a multi-output node");
  }
  return multiOutputEvalFunc_.value();
}

} // namespace fl
