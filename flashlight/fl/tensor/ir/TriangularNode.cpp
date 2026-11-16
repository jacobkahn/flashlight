/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/TriangularNode.h"

#include <stdexcept>

namespace fl {

TriangularNode::TriangularNode(
    NodePtr input,
    TriangularMode mode,
    int diagonal,
    const Shape& outputShape,
    dtype dtype,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait({input}, outputShape, dtype, std::move(stream)),
      mode_(mode),
      diagonal_(diagonal) {}

TriangularNodePtr TriangularNode::create(
    NodePtr input,
    TriangularMode mode,
    int diagonal,
    std::shared_ptr<const Stream> stream) {
  if (input->shape().ndim() < 2) {
    throw std::invalid_argument(
        "TriangularNode requires input with rank at least 2");
  }
  return std::make_shared<TriangularNode>(
      input,
      mode,
      diagonal,
      input->shape(),
      input->dataType(),
      std::move(stream),
      PrivateHelper{});
}

NodePtr TriangularNode::input() const {
  return getInput(kInputIdx);
}

TriangularMode TriangularNode::mode() const {
  return mode_;
}

int TriangularNode::diagonal() const {
  return diagonal_;
}

} // namespace fl
