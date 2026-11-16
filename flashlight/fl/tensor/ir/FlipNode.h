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

class FlipNode;
using FlipNodePtr = std::shared_ptr<FlipNode>;

/**
 * A node representing axis reversal.
 *
 * Reverses the input tensor along one axis. Shape and dtype are preserved.
 *
 * Input: [0] input tensor
 */
class FlipNode : public NodeTrait<FlipNode> {
  unsigned axis_;

  static constexpr unsigned kInputIdx = 0;

  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::Flip;

  FlipNode(
      NodePtr input,
      unsigned axis,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static FlipNodePtr create(
      NodePtr input,
      unsigned axis,
      std::shared_ptr<const Stream> stream = nullptr);

  NodePtr input() const;
  unsigned axis() const;
};

} // namespace fl
