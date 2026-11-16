/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/Node.h"

#include "flashlight/fl/tensor/ir/ExternalUse.h"

#include <cassert>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace fl {

void Node::nodeImplTypeCheck(NodeType expect, NodeType actual) {
  if (expect != actual) {
    std::ostringstream oss;
    oss << "[fl::Node::nodeImplTypeCheck] "
        << "specified node type: [" << actual << "] "
        << "doesn't match actual node type: [" << expect << "]";
    throw std::invalid_argument(oss.str());
  }
}

void Node::linkInput(unsigned inputIdx, NodePtr input) {
  inputs_.at(inputIdx) = input;
  // update metadata
  auto inputUseIter = input->uses_.emplace(
      input->uses_.end(), std::make_unique<Use>(*this, inputIdx));
  inputUseIters_[inputIdx] = inputUseIter;
}

void Node::unlinkInput(unsigned inputIdx) {
  const auto oldInput = inputs_.at(inputIdx);
  const auto oldInputUseIter = inputUseIters_[inputIdx];
  assert(oldInput && "No input at given index");
  // reset
  inputUseIters_[inputIdx] = oldInput->uses_.end();
  inputs_[inputIdx] = nullptr;
  // clean up metadata
  oldInput->uses_.erase(oldInputUseIter);
}

Node::Node(
    std::vector<NodePtr>&& inputs,
    const Shape& shape,
    dtype dtype,
    std::shared_ptr<const Stream> stream)
    : inputs_(inputs),
      shape_(shape),
      dtype_(dtype),
      stream_(std::move(stream)) {
  inputs_.resize(inputs.size());
  inputUseIters_.resize(inputs.size());
  for (unsigned inputIdx = 0; inputIdx < inputs.size(); inputIdx++) {
    linkInput(inputIdx, inputs[inputIdx]);
  }
}

Node::~Node() {
  for (unsigned inputIdx = 0; inputIdx < inputs_.size(); inputIdx++) {
    unlinkInput(inputIdx);
  }
}

NodePtr Node::getInput(unsigned inputIdx) const {
  return inputs_.at(inputIdx);
}

const std::vector<NodePtr>& Node::inputs() const {
  return inputs_;
}

void Node::setInput(unsigned inputIdx, NodePtr newInput) {
  unlinkInput(inputIdx);
  linkInput(inputIdx, newInput);
}

const Shape& Node::shape() const {
  return shape_;
}

dtype Node::dataType() const {
  return dtype_;
}

const std::shared_ptr<const Stream>& Node::stream() const {
  return stream_;
}

const UseList& Node::uses() const {
  return uses_;
}

const ExternalUseList& Node::externalUses() const {
  return externalUses_;
}

void Node::replaceAllUsesWith(NodePtr newNode) {
  if (newNode.get() != this) {
    for (const auto& use : uses_) {
      auto& userNode = use->user();
      userNode.inputs_[use->inputIdx()] = newNode;
    }
    newNode->uses_.splice(newNode->uses_.begin(), this->uses_);
    for (const auto& externalUse : externalUses_) {
      externalUse->usee_ = newNode;
    }
    newNode->externalUses_.splice(
        newNode->externalUses_.begin(), this->externalUses_);
  }
}

const std::optional<Tensor>& Node::getResult() const {
  return result_;
}

void Node::setResult(Tensor&& tensor) {
  if (result_.has_value()) {
    throw std::invalid_argument("[Node::setResult] Result already set");
  } else {
    result_ = std::move(tensor);
  }
}

void Node::unsetResult() {
  if (result_.has_value()) {
    result_ = std::nullopt;
  } else if (!outputs_.empty()) {
    // Multi-output node: clear all outputs
    for (auto& o : outputs_) {
      o = std::nullopt;
    }
  } else {
    throw std::invalid_argument("[Node::unsetResult] Result not set");
  }
}

void Node::setOutputBuffer(Tensor buffer) {
  outputBuffer_ = std::move(buffer);
}

const std::optional<Tensor>& Node::getOutputBuffer() const {
  return outputBuffer_;
}

void Node::clearOutputBuffer() {
  outputBuffer_ = std::nullopt;
}

void Node::setOutput(OutputIndex index, Tensor result) {
  outputs_.at(index) = std::move(result);
}

const Tensor& Node::getOutput(OutputIndex index) const {
  return outputs_.at(index).value();
}

unsigned Node::numOutputs() const {
  return outputs_.size();
}

bool Node::hasOutput(OutputIndex index) const {
  return index < outputs_.size() && outputs_[index].has_value();
}

bool Node::isBinary() const {
  return type() == NodeType::Binary;
}

bool Node::isCast() const {
  return type() == NodeType::Cast;
}

bool Node::isClip() const {
  return type() == NodeType::Clip;
}

bool Node::isConcatenate() const {
  return type() == NodeType::Concatenate;
}

bool Node::isCustom() const {
  return type() == NodeType::Custom;
}

bool Node::isCumsum() const {
  return type() == NodeType::Cumsum;
}

bool Node::isIndex() const {
  return type() == NodeType::Index;
}

bool Node::isIndexedUpdate() const {
  return type() == NodeType::IndexedUpdate;
}

bool Node::isScalar() const {
  return type() == NodeType::Scalar;
}

bool Node::isValue() const {
  return type() == NodeType::Value;
}

bool Node::isFlip() const {
  return type() == NodeType::Flip;
}

bool Node::isRoll() const {
  return type() == NodeType::Roll;
}

bool Node::isUnary() const {
  return type() == NodeType::Unary;
}

bool Node::isReshape() const {
  return type() == NodeType::Reshape;
}

bool Node::isReduction() const {
  return type() == NodeType::Reduction;
}

bool Node::isTranspose() const {
  return type() == NodeType::Transpose;
}

bool Node::isTriangular() const {
  return type() == NodeType::Triangular;
}

bool Node::isFusedElementwise() const {
  return type() == NodeType::FusedElementwise;
}

bool Node::isFusedReduceElementwise() const {
  return type() == NodeType::FusedReduceElementwise;
}

bool Node::isMatmul() const {
  return type() == NodeType::Matmul;
}

bool Node::isConv2d() const {
  return type() == NodeType::Conv2d;
}

bool Node::isProjection() const {
  return type() == NodeType::Projection;
}

bool Node::isBatchNorm() const {
  return type() == NodeType::BatchNorm;
}

bool Node::isPool2d() const {
  return type() == NodeType::Pool2d;
}

bool Node::isMultiValue() const {
  return type() == NodeType::MultiValue;
}

bool Node::isTransfer() const {
  return type() == NodeType::Transfer;
}

bool Node::isPad() const {
  return type() == NodeType::Pad;
}

bool Node::isTile() const {
  return type() == NodeType::Tile;
}

bool Node::isWhere() const {
  return type() == NodeType::Where;
}

} // namespace fl
