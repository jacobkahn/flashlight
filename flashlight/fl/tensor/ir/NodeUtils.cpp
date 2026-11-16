/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/NodeUtils.h"

#include <queue>
#include <unordered_set>
#include <vector>

#include "flashlight/fl/tensor/ir/Node.h"

namespace fl {

unsigned countNodes(const NodePtr& root) {
  if (!root) {
    return 0;
  }
  std::unordered_set<Node*> visited;
  std::queue<Node*> q;
  q.push(root.get());
  visited.insert(root.get());
  while (!q.empty()) {
    auto* cur = q.front();
    q.pop();
    for (const auto& inp : cur->inputs()) {
      if (inp && visited.insert(inp.get()).second) {
        q.push(inp.get());
      }
    }
  }
  return static_cast<unsigned>(visited.size());
}

unsigned graphDepth(const NodePtr& root) {
  if (!root) {
    return 0;
  }
  // Iterative DFS with explicit stack storing (node, depth).
  std::unordered_set<Node*> visited;
  std::vector<std::pair<Node*, unsigned>> stack;
  stack.emplace_back(root.get(), 1);
  unsigned maxDepth = 0;
  while (!stack.empty()) {
    auto [cur, depth] = stack.back();
    stack.pop_back();
    if (depth > maxDepth) {
      maxDepth = depth;
    }
    for (const auto& inp : cur->inputs()) {
      if (inp && visited.insert(inp.get()).second) {
        stack.emplace_back(inp.get(), depth + 1);
      }
    }
  }
  return maxDepth;
}



namespace detail {

std::ostream& operator<<(std::ostream& os, const UseVal& useVal) {
  os << "(" << useVal.user << ", " << useVal.inputIdx << ")";
  return os;
}

} // namespace detail

bool operator==(const UseList& actual, const UseValList& expect) {
  if (actual.size() != expect.size()) {
    return false;
  }
  unsigned i = 0;
  for (const auto& actualUse : actual) {
    const auto& expectUse = expect[i];
    if (&actualUse->user() != expectUse.user.get() ||
        actualUse->inputIdx() != expectUse.inputIdx) {
      return false;
    }
    i++;
  }
  return true;
}

std::ostream& operator<<(std::ostream& os, const Use* use) {
  os << "(" << &use->user() << ", " << use->inputIdx() << ")";
  return os;
}

} // namespace fl
