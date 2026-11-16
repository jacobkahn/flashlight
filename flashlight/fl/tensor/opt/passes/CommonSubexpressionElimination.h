/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <unordered_map>

#include "flashlight/fl/tensor/opt/Pass.h"

namespace fl {

/**
 * An optimization pass that eliminates common subexpressions in a JIT DAG.
 *
 * Two nodes are considered equivalent if they have the same type, parameters,
 * and identical inputs (by pointer identity). Bottom-up traversal ensures that
 * equivalent subtrees are resolved to the same node before their consumers
 * are checked.
 *
 * Supports: ScalarNode, BinaryNode, UnaryNode, ReductionNode,
 * FusedElementwiseNode. ValueNode, CustomNode, IndexNode, and
 * IndexedUpdateNode are not CSE'd (they have unique identity or non-comparable
 * internals).
 */
class CommonSubexpressionElimination : public BottomUpPass {
  std::unordered_map<std::string, NodePtr> cache_;

 protected:
  NodePtr rewrite(NodePtr node) override;

 public:
  NodePtr apply(NodePtr root) override;
};

} // namespace fl
