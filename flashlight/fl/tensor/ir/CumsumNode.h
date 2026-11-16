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

class CumsumNode;
using CumsumNodePtr = std::shared_ptr<CumsumNode>;

/**
 * A node representing cumulative sum along one axis.
 *
 * Shape and dtype are preserved.
 *
 * Input: [0] input tensor
 */
class CumsumNode : public NodeTrait<CumsumNode> {
  unsigned axis_;

  static constexpr unsigned kInputIdx = 0;

  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::Cumsum;

  CumsumNode(
      NodePtr input,
      unsigned axis,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static CumsumNodePtr create(
      NodePtr input,
      unsigned axis,
      std::shared_ptr<const Stream> stream = nullptr);

  NodePtr input() const;
  unsigned axis() const;
};

} // namespace fl
