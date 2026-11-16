/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/FusedElementwiseNode.h"

#include <stdexcept>

#include "flashlight/fl/tensor/ir/OpProperties.h"

namespace fl {

FusedElementwiseNode::FusedElementwiseNode(
    std::vector<NodePtr>&& inputs,
    std::vector<FusedOp> program,
    unsigned numExternalInputs,
    const Shape& shape,
    dtype dtype,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait(std::move(inputs), shape, dtype, std::move(stream)),
      program_(std::move(program)),
      numExternalInputs_(numExternalInputs) {}

FusedElementwiseNodePtr FusedElementwiseNode::create(
    std::vector<NodePtr> inputs,
    std::vector<FusedOp> program,
    unsigned numExternalInputs,
    const Shape& shape,
    std::shared_ptr<const Stream> stream) {
  if (program.empty()) {
    throw std::invalid_argument(
        "[FusedElementwiseNode::create] program must not be empty");
  }
  auto outputDtype = inferFusedProgramOutputDtype(
      inputs, program, numExternalInputs);
  return std::make_shared<FusedElementwiseNode>(
      std::move(inputs),
      std::move(program),
      numExternalInputs,
      shape,
      outputDtype,
      std::move(stream),
      PrivateHelper{});
}

const std::vector<FusedOp>& FusedElementwiseNode::program() const {
  return program_;
}

unsigned FusedElementwiseNode::numExternalInputs() const {
  return numExternalInputs_;
}

unsigned FusedElementwiseNode::numSteps() const {
  return program_.size();
}

} // namespace fl
