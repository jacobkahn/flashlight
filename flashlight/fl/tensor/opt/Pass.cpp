/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/opt/Pass.h"

#include <algorithm>
#include <vector>

namespace fl {

NodePtr BottomUpPass::traverse(NodePtr root) {
  // Iterative post-order traversal to avoid stack overflow on deep graphs.
  // Pass 1: DFS to collect nodes. Each node appears once (visited check).
  std::vector<NodePtr> order;
  {
    std::vector<NodePtr> stack{root};
    while (!stack.empty()) {
      auto node = stack.back();
      stack.pop_back();
      if (!visited_.insert(node.get()).second) {
        continue;
      }
      // Evaluated nodes are leaves: neither rewritten nor descended into.
      if (isEvaluated(*node)) {
        continue;
      }
      order.push_back(node);
      // Push inputs in reverse so leftmost input is processed first
      auto& inputs = node->inputs();
      for (auto it = inputs.rbegin(); it != inputs.rend(); ++it) {
        stack.push_back(*it);
      }
    }
  }

  // Pass 2: process in reverse order (leaves first, root last).
  // Uses replaceAllUsesWith so parent inputs are automatically updated.
  std::reverse(order.begin(), order.end());
  NodePtr currentRoot = root;
  for (auto& node : order) {
    auto newNode = rewrite(node);
    if (newNode != node) {
      node->replaceAllUsesWith(newNode);
      if (node.get() == root.get()) {
        currentRoot = newNode;
      }
    }
  }

  return currentRoot;
}

NodePtr BottomUpPass::apply(NodePtr root) {
  visited_.clear();
  return traverse(root);
}

NodePtr TopDownPass::traverse(NodePtr root) {
  // Iterative pre-order traversal to avoid stack overflow on deep graphs.
  NodePtr currentRoot = root;
  std::vector<NodePtr> stack{root};
  while (!stack.empty()) {
    auto node = stack.back();
    stack.pop_back();
    if (!visited_.insert(node.get()).second) {
      continue;
    }
    // Evaluated nodes are leaves: neither rewritten nor descended into.
    if (isEvaluated(*node)) {
      continue;
    }
    auto result = rewrite(node);
    if (result != node) {
      node->replaceAllUsesWith(result);
      if (node.get() == root.get()) {
        currentRoot = result;
      }
      node = result;
      // Mark the new node as visited too
      visited_.insert(node.get());
    }
    // Push inputs in reverse so leftmost input is processed first
    auto& inputs = node->inputs();
    for (auto it = inputs.rbegin(); it != inputs.rend(); ++it) {
      stack.push_back(*it);
    }
  }
  return currentRoot;
}

NodePtr TopDownPass::apply(NodePtr root) {
  visited_.clear();
  return traverse(root);
}

} // namespace fl
