/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/Pool2dNode.h"

namespace fl {

Pool2dNode::Pool2dNode(
    NodePtr input,
    int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode,
    const Shape& outputShape,
    dtype outputType,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait({std::move(input)}, outputShape, outputType, std::move(stream)),
      wx_(wx), wy_(wy), sx_(sx), sy_(sy), px_(px), py_(py),
      mode_(mode) {}

Pool2dNodePtr Pool2dNode::create(
    NodePtr input,
    int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode,
    const Shape& outputShape,
    dtype outputType,
    std::shared_ptr<const Stream> stream) {
  return std::make_shared<Pool2dNode>(
      std::move(input), wx, wy, sx, sy, px, py, mode,
      outputShape, outputType, std::move(stream), PrivateHelper{});
}

NodePtr Pool2dNode::input() const {
  return getInput(0);
}

int Pool2dNode::wx() const { return wx_; }
int Pool2dNode::wy() const { return wy_; }
int Pool2dNode::sx() const { return sx_; }
int Pool2dNode::sy() const { return sy_; }
int Pool2dNode::px() const { return px_; }
int Pool2dNode::py() const { return py_; }
PoolingMode Pool2dNode::mode() const { return mode_; }

void Pool2dNode::setCachedState(ForwardBackwardState s) const {
  cachedState_ = std::move(s);
}

const ForwardBackwardState& Pool2dNode::cachedState() const {
  return cachedState_;
}

} // namespace fl
