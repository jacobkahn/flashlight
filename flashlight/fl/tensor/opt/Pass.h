/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_set>

#include "flashlight/fl/tensor/ir/Node.h"

namespace fl {

/**
 * True if the node already holds its evaluated result(s). Passes treat such
 * nodes as leaves: rewriting or absorbing one would discard its result and
 * recompute it, which wastes work and changes values for nondeterministic ops
 * (e.g. rand).
 */
inline bool isEvaluated(const Node& node) {
  if (node.getResult().has_value()) {
    return true;
  }
  for (unsigned i = 0; i < node.numOutputs(); ++i) {
    if (node.hasOutput(i)) {
      return true;
    }
  }
  return false;
}

/**
 * A transformation pass over a JIT tree.
 */
class Pass {
 public:
  Pass() = default;
  virtual ~Pass() = default;

  /**
   * Apply in-place transformation to nodes within the tree.
   *
   * @param[in] node the root node of the JIT tree to be transformed
   * @return root to the updated tree
   */
  virtual NodePtr apply(NodePtr node) = 0;
};

/**
 * Base class for passes that traverse the graph bottom-up (inputs first,
 * then the node itself). Handles visited tracking and input updates.
 * Subclass overrides rewrite() to transform individual nodes.
 */
class BottomUpPass : public Pass {
  std::unordered_set<Node*> visited_;
  NodePtr traverse(NodePtr node);

 protected:
  /**
   * Transform a single node. Called after all inputs have been processed.
   * Return the original node if no change, or a new replacement node.
   */
  virtual NodePtr rewrite(NodePtr node) = 0;

  /**
   * Mark a node as visited so it won't be traversed again.
   */
  void markVisited(Node* node) { visited_.insert(node); }

 public:
  NodePtr apply(NodePtr root) override;
};

/**
 * Base class for passes that traverse the graph top-down (node first,
 * then inputs). Useful for greedy fusion where the outermost match
 * should be found first.
 */
class TopDownPass : public Pass {
  std::unordered_set<Node*> visited_;
  NodePtr traverse(NodePtr node);

 protected:
  /**
   * Transform a single node. Called before inputs are processed.
   * Return the original node if no change, or a new replacement node.
   */
  virtual NodePtr rewrite(NodePtr node) = 0;

  /**
   * Mark a node as visited so it won't be traversed again.
   */
  void markVisited(Node* node) { visited_.insert(node); }

 public:
  NodePtr apply(NodePtr root) override;
};

} // namespace fl
