/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/ir/Node.h"

namespace fl {

class Pool2dNode;
using Pool2dNodePtr = std::shared_ptr<Pool2dNode>;

/**
 * A node representing a 2D pooling forward operation.
 *
 * Single-output node with mutable cached state (same pattern as Conv2dNode).
 * The cached state is used by pool2dBackward.
 *
 * Input: [0] input — (N, C, H, W)
 */
class Pool2dNode : public NodeTrait<Pool2dNode> {
  int wx_, wy_, sx_, sy_, px_, py_;
  PoolingMode mode_;

  // Mutable cache for ForwardBackwardState from evaluation
  mutable ForwardBackwardState cachedState_;

  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::Pool2d;

  Pool2dNode(
      NodePtr input,
      int wx, int wy, int sx, int sy, int px, int py,
      PoolingMode mode,
      const Shape& outputShape,
      dtype outputType,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static Pool2dNodePtr create(
      NodePtr input,
      int wx, int wy, int sx, int sy, int px, int py,
      PoolingMode mode,
      const Shape& outputShape,
      dtype outputType,
      std::shared_ptr<const Stream> stream = nullptr);

  // Accessors
  NodePtr input() const;
  int wx() const;
  int wy() const;
  int sx() const;
  int sy() const;
  int px() const;
  int py() const;
  PoolingMode mode() const;

  // Cached state from evaluation (for backward pass)
  void setCachedState(ForwardBackwardState s) const;
  const ForwardBackwardState& cachedState() const;
};

} // namespace fl
