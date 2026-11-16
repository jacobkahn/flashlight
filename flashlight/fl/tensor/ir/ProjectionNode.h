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

class ProjectionNode;
using ProjectionNodePtr = std::shared_ptr<ProjectionNode>;

/**
 * Extracts a single output from a multi-output parent node.
 *
 * Inspired by XLA's GetTupleElement, TVM's TupleGetItem, and LLVM's
 * extractvalue. Preserves the single-result Node invariant: each
 * ProjectionNode produces one tensor (the specified output of its parent).
 *
 * Zero-cost at eval time: the parent node evaluates all outputs;
 * ProjectionNode just reads parent->getOutput(outputIndex_).
 */
class ProjectionNode : public NodeTrait<ProjectionNode> {
  unsigned outputIndex_;

  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::Projection;

  ProjectionNode(
      NodePtr parent,
      unsigned outputIndex,
      const Shape& shape,
      dtype type,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static ProjectionNodePtr create(
      NodePtr parent,
      OutputIndex outputIndex,
      const Shape& shape,
      dtype type,
      std::shared_ptr<const Stream> stream = nullptr);

  // Which output of the parent to extract
  unsigned outputIndex() const;

  // The parent multi-output node (always input 0)
  NodePtr parent() const;
};

} // namespace fl
