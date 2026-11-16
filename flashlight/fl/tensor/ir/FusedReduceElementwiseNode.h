/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

#include "flashlight/fl/tensor/ir/FusedElementwiseNode.h"
#include "flashlight/fl/tensor/ir/Node.h"
#include "flashlight/fl/tensor/ir/ReductionNode.h"

namespace fl {

class FusedReduceElementwiseNode;
using FusedReduceElementwiseNodePtr =
    std::shared_ptr<FusedReduceElementwiseNode>;

/**
 * A node representing fused elementwise operations followed by a reduction.
 *
 * Pattern: `reduce(elementwise_chain(inputs...))` becomes a single node
 * that avoids materializing the intermediate elementwise result.
 *
 * Stores:
 * - An elementwise mini-program (same format as FusedElementwiseNode)
 * - A reduction operation with axes and keepDims
 *
 * The fallback evaluator runs the elementwise program then reduces.
 * Backend-specific passes can lower this to a single fused kernel
 * (e.g., warp-level reduction with inline elementwise transforms).
 */
class FusedReduceElementwiseNode
    : public NodeTrait<FusedReduceElementwiseNode> {
  std::vector<FusedOp> program_;
  unsigned numExternalInputs_;
  ReductionOp reductionOp_;
  std::vector<int> axes_;
  bool keepDims_;

  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::FusedReduceElementwise;
  FusedReduceElementwiseNode(
      std::vector<NodePtr>&& inputs,
      std::vector<FusedOp> program,
      unsigned numExternalInputs,
      ReductionOp reductionOp,
      std::vector<int> axes,
      bool keepDims,
      const Shape& shape,
      dtype dtype,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static FusedReduceElementwiseNodePtr create(
      std::vector<NodePtr> inputs,
      std::vector<FusedOp> program,
      unsigned numExternalInputs,
      ReductionOp reductionOp,
      std::vector<int> axes,
      bool keepDims,
      const Shape& shape,
      std::shared_ptr<const Stream> stream = nullptr);

  const std::vector<FusedOp>& program() const;
  unsigned numExternalInputs() const;
  unsigned numSteps() const;
  ReductionOp reductionOp() const;
  const std::vector<int>& axes() const;
  bool keepDims() const;
};

} // namespace fl
