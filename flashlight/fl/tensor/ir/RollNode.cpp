/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/RollNode.h"

#include <limits>
#include <stdexcept>

namespace fl {

namespace {

int normalizeShift(int shift, Dim dim) {
  if (dim <= 0 || dim > std::numeric_limits<int>::max()) {
    return shift;
  }
  const auto dimInt = static_cast<int>(dim);
  auto normalized = shift % dimInt;
  if (normalized < 0) {
    normalized += dimInt;
  }
  return normalized;
}

} // namespace

RollNode::RollNode(
    NodePtr input,
    int shift,
    unsigned axis,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait({input}, input->shape(), input->dataType(), std::move(stream)),
      shift_(shift),
      axis_(axis) {}

RollNodePtr RollNode::create(
    NodePtr input,
    int shift,
    unsigned axis,
    std::shared_ptr<const Stream> stream) {
  if (axis >= static_cast<unsigned>(input->shape().ndim())) {
    throw std::invalid_argument("fl::RollNode: axis out of range");
  }
  return std::make_shared<RollNode>(
      input,
      normalizeShift(shift, input->shape().dim(axis)),
      axis,
      std::move(stream),
      PrivateHelper{});
}

NodePtr RollNode::input() const {
  return getInput(kInputIdx);
}

int RollNode::shift() const {
  return shift_;
}

unsigned RollNode::axis() const {
  return axis_;
}

} // namespace fl
