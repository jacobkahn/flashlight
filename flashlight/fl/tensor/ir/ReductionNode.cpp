/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/ReductionNode.h"

#include "flashlight/fl/tensor/ShapeInference.h"
#include "flashlight/fl/tensor/ir/OpProperties.h"

namespace fl {

ReductionNode::ReductionNode(
    NodePtr input,
    ReductionOp op,
    std::vector<int> axes,
    bool keepDims,
    const Shape& shape,
    dtype dtype,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait({input}, shape, dtype, std::move(stream)),
      op_(op),
      axes_(std::move(axes)),
      keepDims_(keepDims) {}

ReductionNodePtr ReductionNode::create(
    NodePtr input,
    ReductionOp op,
    std::vector<int> axes,
    bool keepDims,
    std::shared_ptr<const Stream> stream) {
  auto outputShape =
      inferReductionOutputShape(input->shape(), axes, keepDims);
  auto outputDtype = inferOutputDtype(op, input->dataType());
  return std::make_shared<ReductionNode>(
      input, op, std::move(axes), keepDims, outputShape, outputDtype, std::move(stream), PrivateHelper{});
}

ReductionOp ReductionNode::op() const {
  return op_;
}

const std::vector<int>& ReductionNode::axes() const {
  return axes_;
}

bool ReductionNode::keepDims() const {
  return keepDims_;
}

NodePtr ReductionNode::input() const {
  return getInput(kInputIdx);
}

} // namespace fl
