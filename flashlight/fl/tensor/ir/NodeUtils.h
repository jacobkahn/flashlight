/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "flashlight/fl/tensor/ir/Node.h"

// Some helpful definitions/functions
namespace fl {

/**
 * Count the total number of unique nodes reachable from \p root (inclusive).
 * Uses BFS; cost is O(V + E).
 */
unsigned countNodes(const NodePtr& root);

/**
 * Return the length of the longest root-to-leaf path.
 * A single node with no inputs has depth 1.  An empty (null) root returns 0.
 */
unsigned graphDepth(const NodePtr& root);

namespace detail {

// TODO a similar construct could be made for nodes for pattern matching in
// graph rewrite and testing.
struct UseVal {
  const NodePtr user;
  const unsigned inputIdx;
};

std::ostream& operator<<(std::ostream& os, const UseVal& useVal);

} // namespace detail

using NodeList = std::vector<NodePtr>;
using UseValList = std::vector<detail::UseVal>;

/**
 * Example usage:
 *
 *   node->uses() == UseValList({{n1, 0}, ...})
 */
bool operator==(const UseList& actual, const UseValList& expect);

std::ostream& operator<<(std::ostream& os, const Use* useVal);

} // namespace fl
