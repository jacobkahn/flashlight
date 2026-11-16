/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/FusedReduceElementwiseNode.h"

#include <stdexcept>

#include "flashlight/fl/tensor/ir/OpProperties.h"

namespace fl {

FusedReduceElementwiseNode::FusedReduceElementwiseNode(
    std::vector<NodePtr>&& inputs,
    std::vector<FusedOp> program,
    unsigned numExternalInputs,
    ReductionOp reductionOp,
    std::vector<int> axes,
    bool keepDims,
    const Shape& shape,
    dtype dtype,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait(std::move(inputs), shape, dtype, std::move(stream)),
      program_(std::move(program)),
      numExternalInputs_(numExternalInputs),
      reductionOp_(reductionOp),
      axes_(std::move(axes)),
      keepDims_(keepDims) {}

FusedReduceElementwiseNodePtr FusedReduceElementwiseNode::create(
    std::vector<NodePtr> inputs,
    std::vector<FusedOp> program,
    unsigned numExternalInputs,
    ReductionOp reductionOp,
    std::vector<int> axes,
    bool keepDims,
    const Shape& shape,
    std::shared_ptr<const Stream> stream) {
  if (program.empty()) {
    throw std::invalid_argument(
        "[FusedReduceElementwiseNode::create] program must not be empty");
  }
  auto elemDtype = inferFusedProgramOutputDtype(
      inputs, program, numExternalInputs);
  auto outputDtype = inferOutputDtype(reductionOp, elemDtype);
  return std::make_shared<FusedReduceElementwiseNode>(
      std::move(inputs),
      std::move(program),
      numExternalInputs,
      reductionOp,
      std::move(axes),
      keepDims,
      shape,
      outputDtype,
      std::move(stream),
      PrivateHelper{});
}

const std::vector<FusedOp>& FusedReduceElementwiseNode::program() const {
  return program_;
}

unsigned FusedReduceElementwiseNode::numExternalInputs() const {
  return numExternalInputs_;
}

unsigned FusedReduceElementwiseNode::numSteps() const {
  return program_.size();
}

ReductionOp FusedReduceElementwiseNode::reductionOp() const {
  return reductionOp_;
}

const std::vector<int>& FusedReduceElementwiseNode::axes() const {
  return axes_;
}

bool FusedReduceElementwiseNode::keepDims() const {
  return keepDims_;
}

} // namespace fl
