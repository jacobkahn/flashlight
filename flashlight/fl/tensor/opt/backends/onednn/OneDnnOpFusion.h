/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include <dnnl.hpp>

#include "flashlight/fl/tensor/ir/Node.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/opt/Pass.h"

namespace fl {

// OneDNN eltwise algorithm + parameters for a unary operation.
struct EltwiseInfo {
  dnnl::algorithm alg;
  float alpha;
  float beta;
};

// Map a UnaryOp to its OneDNN eltwise equivalent, if one exists.
std::optional<EltwiseInfo> tryUnaryOpToOneDnnEltwise(UnaryOp op);

/**
 * Leverage OneDNN's post-ops to fuse operations.
 *
 * Uses the PatternMatcher DSL for declarative node identification and walks
 * fusable chains iteratively, accumulating up to 32 OneDNN post-ops.
 *
 * NOTE
 * 1. OneDNN binary post-ops only accept the RHS argument. For commutative ops
 *    (add, mul, min, max, eq, neq), the pass swaps LHS/RHS when the fusable
 *    chain is on the RHS side, enabling longer fused chains.
 * 2. Currently we avoid recomputation -- fuse iff intermediate nodes are _only_
 *    used as input nodes in the chain. There might be places where benefit of
 *    aggressive fusion outweighs cost of recomputation, need to investigate
 *    more (think Halide).
 *
 * n1   n2
 *  \  /
 *   b2   n3
 *    \  /
 *     b1
 *
 * -->
 *
 *    n1 n2 n3
 *     \ |  /
 * ---------------- CustomNode with evaluation logic that uses OneDNN post-op
 * | n1 = n1 + n2 |
 * | n1 = n2 + n3 |
 * ----------------
 */
class OneDnnOpFusion : public Pass {
  struct SearchState;

  // Avoid re-visit, since fuser only need to apply once to each node.
  std::unordered_set<NodePtr> visited_{};
  // Nodes whose inputs have already been pushed onto the worklist.
  std::unordered_set<NodePtr> expanded_{};
  // Deferred worklist — inputs are pushed here instead of recursing.
  std::vector<NodePtr> worklist_;

  // 1. Fuse _along_ some path from `node`.
  // 2. recursively optimize other inputs along the fused path.
  NodePtr rewriteFrom(NodePtr node);

  // keep searching for nodes to fuse starting from `node`
  NodePtr searchAndFuse(NodePtr node, SearchState& state);

  // Actual fusion of an op-chain, `node` is a leaf input.
  NodePtr fuseNodes(NodePtr node, SearchState& state);

 public:
  OneDnnOpFusion() = default;
  ~OneDnnOpFusion() = default;

  NodePtr apply(NodePtr root) override;
};

} // namespace fl
