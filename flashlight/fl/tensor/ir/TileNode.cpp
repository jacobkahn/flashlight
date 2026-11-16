/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/TileNode.h"

#include "flashlight/fl/tensor/ShapeInference.h"

namespace fl {

TileNode::TileNode(
    NodePtr input,
    const Shape& tileDims,
    const Shape& outputShape,
    dtype dtype,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait({input}, outputShape, dtype, std::move(stream)),
      tileDims_(tileDims) {}

TileNodePtr TileNode::create(
    NodePtr input,
    const Shape& tileDims,
    std::shared_ptr<const Stream> stream) {
  auto outputShape = inferTileOutputShape(input->shape(), tileDims);
  return std::make_shared<TileNode>(
      input, tileDims, outputShape, input->dataType(),
      std::move(stream), PrivateHelper{});
}

NodePtr TileNode::input() const {
  return getInput(kInputIdx);
}

const Shape& TileNode::tileDims() const {
  return tileDims_;
}

} // namespace fl
