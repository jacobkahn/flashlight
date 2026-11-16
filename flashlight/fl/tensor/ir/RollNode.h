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

class RollNode;
using RollNodePtr = std::shared_ptr<RollNode>;

/**
 * A node representing circular shift along one axis.
 *
 * Shape and dtype are preserved. Shift is normalized modulo the selected axis
 * size when that size fits the public int shift representation.
 *
 * Input: [0] input tensor
 */
class RollNode : public NodeTrait<RollNode> {
  int shift_;
  unsigned axis_;

  static constexpr unsigned kInputIdx = 0;

  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::Roll;

  RollNode(
      NodePtr input,
      int shift,
      unsigned axis,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static RollNodePtr create(
      NodePtr input,
      int shift,
      unsigned axis,
      std::shared_ptr<const Stream> stream = nullptr);

  NodePtr input() const;
  int shift() const;
  unsigned axis() const;
};

} // namespace fl
