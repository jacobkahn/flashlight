/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

#include "flashlight/fl/tensor/OpEnums.h"
#include "flashlight/fl/tensor/ir/Node.h"

namespace fl {

class ReductionNode;
using ReductionNodePtr = std::shared_ptr<ReductionNode>;

/**
 * A node that represents reduction operations along specified axes.
 */
class ReductionNode : public NodeTrait<ReductionNode> {
  const ReductionOp op_;
  const std::vector<int> axes_;
  const bool keepDims_;

  static constexpr unsigned kInputIdx = 0;

  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::Reduction;
  ReductionNode(
      NodePtr input,
      ReductionOp op,
      std::vector<int> axes,
      bool keepDims,
      const Shape& shape,
      dtype dtype,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static ReductionNodePtr create(
      NodePtr input,
      ReductionOp op,
      std::vector<int> axes,
      bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr);

  ReductionOp op() const;
  const std::vector<int>& axes() const;
  bool keepDims() const;
  NodePtr input() const;
};

} // namespace fl
