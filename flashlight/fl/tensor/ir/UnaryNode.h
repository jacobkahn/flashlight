/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/fl/tensor/OpEnums.h"
#include "flashlight/fl/tensor/ir/Node.h"

namespace fl {

class UnaryNode;
using UnaryNodePtr = std::shared_ptr<UnaryNode>;

/**
 * A node that represents element-wise unary operations.
 * Output shape equals input shape.
 */
class UnaryNode : public NodeTrait<UnaryNode> {
  const UnaryOp op_;

  static constexpr unsigned kInputIdx = 0;

  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::Unary;
  UnaryNode(NodePtr input, UnaryOp op, const Shape& shape, dtype dtype,
            std::shared_ptr<const Stream> stream, PrivateHelper);

  static UnaryNodePtr create(
      NodePtr input, UnaryOp op,
      std::shared_ptr<const Stream> stream = nullptr);

  UnaryOp op() const;
  NodePtr input() const;
};

} // namespace fl
