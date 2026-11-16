/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/BatchNormNode.h"

#include <stdexcept>

namespace fl {

BatchNormNode::BatchNormNode(
    std::vector<NodePtr> inputs,
    std::vector<int> axes,
    bool train,
    double momentum,
    double epsilon,
    std::vector<OutputDesc> outputDescs,
    const Shape& shape,
    dtype type,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait(std::move(inputs), shape, type, std::move(stream)),
      axes_(std::move(axes)),
      train_(train),
      momentum_(momentum),
      epsilon_(epsilon),
      outputDescs_(std::move(outputDescs)) {
  // Initialize multi-output storage
  outputs_.resize(kNumOutputs);
}

BatchNormNodePtr BatchNormNode::create(
    std::vector<NodePtr> inputs,
    std::vector<int> axes,
    bool train,
    double momentum,
    double epsilon,
    std::vector<OutputDesc> outputDescs,
    std::shared_ptr<const Stream> stream) {
  if (outputDescs.size() != kNumOutputs) {
    throw std::invalid_argument(
        "[BatchNormNode::create] outputDescs must have " +
        std::to_string(kNumOutputs) + " entries, got " +
        std::to_string(outputDescs.size()));
  }
  if (inputs.size() != 5) {
    throw std::invalid_argument(
        "[BatchNormNode::create] expected 5 inputs "
        "(input, weight, bias, runningMean, runningVar), got " +
        std::to_string(inputs.size()));
  }

  // The node's own shape/dtype is that of the primary output (Result)
  const auto& resultShape = outputDescs[Out::Result].shape;
  const auto resultType = outputDescs[Out::Result].type;

  return std::make_shared<BatchNormNode>(
      std::move(inputs), std::move(axes), train, momentum, epsilon,
      std::move(outputDescs), resultShape, resultType, std::move(stream),
      PrivateHelper{});
}

const std::vector<int>& BatchNormNode::axes() const {
  return axes_;
}

bool BatchNormNode::train() const {
  return train_;
}

double BatchNormNode::momentum() const {
  return momentum_;
}

double BatchNormNode::epsilon() const {
  return epsilon_;
}

NodePtr BatchNormNode::inputTensor() const {
  return getInput(0);
}

NodePtr BatchNormNode::weight() const {
  return getInput(1);
}

NodePtr BatchNormNode::bias() const {
  return getInput(2);
}

NodePtr BatchNormNode::runningMean() const {
  return getInput(3);
}

NodePtr BatchNormNode::runningVar() const {
  return getInput(4);
}

const std::vector<OutputDesc>& BatchNormNode::outputDescs() const {
  return outputDescs_;
}

bool BatchNormNode::needsState(uint32_t liveMask) const {
  return (liveMask & ((1u << SaveMean) | (1u << SaveVar))) != 0;
}

} // namespace fl
