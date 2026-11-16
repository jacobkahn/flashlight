/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/ConcatenateNode.h"

#include <stdexcept>

#include "flashlight/fl/tensor/ShapeInference.h"

namespace fl {

namespace {

std::vector<Shape> inputShapes(const std::vector<NodePtr>& inputs) {
  std::vector<Shape> shapes;
  shapes.reserve(inputs.size());
  for (const auto& input : inputs) {
    shapes.push_back(input->shape());
  }
  return shapes;
}

std::vector<Dim> inputElements(const std::vector<NodePtr>& inputs) {
  std::vector<Dim> elements;
  elements.reserve(inputs.size());
  for (const auto& input : inputs) {
    elements.push_back(input->shape().elements());
  }
  return elements;
}

dtype inferConcatenateDtype(const std::vector<NodePtr>& inputs) {
  if (inputs.empty()) {
    throw std::invalid_argument("ConcatenateNode requires at least one input");
  }
  auto result = inputs.front()->dataType();
  for (size_t i = 1; i < inputs.size(); ++i) {
    result = promoteDtypes(result, inputs[i]->dataType());
  }
  return result;
}

} // namespace

ConcatenateNode::ConcatenateNode(
    std::vector<NodePtr> inputs,
    unsigned axis,
    const Shape& outputShape,
    dtype dtype,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait(std::move(inputs), outputShape, dtype, std::move(stream)),
      axis_(axis) {}

ConcatenateNodePtr ConcatenateNode::create(
    const std::vector<NodePtr>& inputs,
    unsigned axis,
    std::shared_ptr<const Stream> stream) {
  if (inputs.empty()) {
    throw std::invalid_argument("ConcatenateNode requires at least one input");
  }
  auto outputShape = inferConcatenateOutputShape(
      inputShapes(inputs), inputElements(inputs), axis);
  return std::make_shared<ConcatenateNode>(
      inputs,
      axis,
      outputShape,
      inferConcatenateDtype(inputs),
      std::move(stream),
      PrivateHelper{});
}

NodePtr ConcatenateNode::input(unsigned inputIdx) const {
  return getInput(inputIdx);
}

unsigned ConcatenateNode::numInputs() const {
  return inputs().size();
}

unsigned ConcatenateNode::axis() const {
  return axis_;
}

} // namespace fl
