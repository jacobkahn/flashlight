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

class ClipNode;
using ClipNodePtr = std::shared_ptr<ClipNode>;

/**
 * A node representing value clipping.
 *
 * Computes min(max(input, low), high). Shape and dtype follow the input tensor,
 * matching the existing TensorBackend::clip contract.
 *
 * Inputs: [0] input tensor, [1] low bound, [2] high bound
 */
class ClipNode : public NodeTrait<ClipNode> {
  static constexpr unsigned kInputIdx = 0;
  static constexpr unsigned kLowIdx = 1;
  static constexpr unsigned kHighIdx = 2;

  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::Clip;

  ClipNode(
      NodePtr input,
      NodePtr low,
      NodePtr high,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static ClipNodePtr create(
      NodePtr input,
      NodePtr low,
      NodePtr high,
      std::shared_ptr<const Stream> stream = nullptr);

  NodePtr input() const;
  NodePtr low() const;
  NodePtr high() const;
};

} // namespace fl
