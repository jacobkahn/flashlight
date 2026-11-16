/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/fl/tensor/ir/Node.h"

namespace fl {

class ReshapeNode;
using ReshapeNodePtr = std::shared_ptr<ReshapeNode>;

/**
 * A node that represents a reshape operation.
 * Output shape is the given new shape; dtype is preserved from input.
 */
class ReshapeNode : public NodeTrait<ReshapeNode> {
  static constexpr unsigned kInputIdx = 0;

  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::Reshape;
  ReshapeNode(
      NodePtr input,
      const Shape& newShape,
      dtype dtype,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static ReshapeNodePtr create(
      NodePtr input,
      const Shape& newShape,
      std::shared_ptr<const Stream> stream = nullptr);

  NodePtr input() const;
};

} // namespace fl
