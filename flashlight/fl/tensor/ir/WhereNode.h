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

class WhereNode;
using WhereNodePtr = std::shared_ptr<WhereNode>;

/**
 * A node representing element-wise conditional selection.
 *
 * Selects values from x where condition is true, otherwise from y. The current
 * IR node matches Flashlight backend behavior: condition, x, and y must have
 * the same shape, and condition must be dtype::b8.
 *
 * Inputs:
 *   [0] condition tensor
 *   [1] true branch tensor
 *   [2] false branch tensor
 */
class WhereNode : public NodeTrait<WhereNode> {
  static constexpr unsigned kConditionIdx = 0;
  static constexpr unsigned kXIdx = 1;
  static constexpr unsigned kYIdx = 2;

  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::Where;

  WhereNode(
      NodePtr condition,
      NodePtr x,
      NodePtr y,
      const Shape& outputShape,
      dtype outputDtype,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static WhereNodePtr create(
      NodePtr condition,
      NodePtr x,
      NodePtr y,
      std::shared_ptr<const Stream> stream = nullptr);

  NodePtr condition() const;
  NodePtr x() const;
  NodePtr y() const;
};

} // namespace fl
