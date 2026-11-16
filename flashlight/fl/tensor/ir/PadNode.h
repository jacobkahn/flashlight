/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "flashlight/fl/tensor/ir/Node.h"

namespace fl {

class PadNode;
using PadNodePtr = std::shared_ptr<PadNode>;

/**
 * A node representing tensor padding.
 *
 * `padWidths` is canonicalized to one `(before, after)` pair per input axis,
 * with missing trailing axes represented as `{0, 0}`.
 *
 * Input: [0] input tensor
 */
class PadNode : public NodeTrait<PadNode> {
  std::vector<std::pair<int, int>> padWidths_;
  PadType type_;

  static constexpr unsigned kInputIdx = 0;

  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::Pad;

  PadNode(
      NodePtr input,
      std::vector<std::pair<int, int>> padWidths,
      PadType type,
      const Shape& outputShape,
      dtype dtype,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static PadNodePtr create(
      NodePtr input,
      const std::vector<std::pair<int, int>>& padWidths,
      PadType type,
      std::shared_ptr<const Stream> stream = nullptr);

  NodePtr input() const;
  const std::vector<std::pair<int, int>>& padWidths() const;
  PadType padType() const;
};

} // namespace fl
