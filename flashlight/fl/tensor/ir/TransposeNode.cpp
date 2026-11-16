/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/TransposeNode.h"

#include "flashlight/fl/tensor/ShapeInference.h"

namespace fl {

TransposeNode::TransposeNode(
    NodePtr input,
    Shape axes,
    const Shape& outputShape,
    dtype dtype,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait({input}, outputShape, dtype, std::move(stream)),
      axes_(std::move(axes)) {}

TransposeNodePtr TransposeNode::create(
    NodePtr input,
    const Shape& axes,
    std::shared_ptr<const Stream> stream) {
  auto outputShape = inferTransposeOutputShape(input->shape(), axes);
  return std::make_shared<TransposeNode>(
      input, Shape(axes), outputShape, input->dataType(), std::move(stream),
      PrivateHelper{});
}

NodePtr TransposeNode::input() const {
  return getInput(kInputIdx);
}

const Shape& TransposeNode::axes() const {
  return axes_;
}

} // namespace fl
