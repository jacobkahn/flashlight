/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/opt/passes/ElementwiseFusion.h"

#include <unordered_map>
#include <vector>

#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/FusedElementwiseNode.h"
#include "flashlight/fl/tensor/ir/OpProperties.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"

namespace fl {

namespace {

// A node can be absorbed into a fused group if it's a fusable elementwise op
// and has exactly one use with no external uses (its result doesn't need to be
// materialized separately).
bool canAbsorb(NodePtr node) {
  if (isEvaluated(*node) || node->uses().size() != 1 ||
      !node->externalUses().empty()) {
    return false;
  }
  if (node->isUnary()) {
    return isFusableElementwise(node->impl<UnaryNode>().op());
  }
  if (node->isBinary()) {
    return isFusableElementwise(node->impl<BinaryNode>().op());
  }
  return false;
}

// A node is a fusion candidate if it's a fusable elementwise op.
bool isFusionCandidate(NodePtr node) {
  if (node->isUnary()) {
    return isFusableElementwise(node->impl<UnaryNode>().op());
  }
  if (node->isBinary()) {
    return isFusableElementwise(node->impl<BinaryNode>().op());
  }
  return false;
}

// DFS post-order collection of absorbable nodes starting from root.
// Post-order gives topological order (inputs before outputs).
void collectGroupPostOrder(
    NodePtr node,
    std::unordered_set<NodePtr>& inGroup,
    std::vector<NodePtr>& topoOrder) {
  if (inGroup.contains(node)) {
    return;
  }
  inGroup.insert(node);
  for (const auto& input : node->inputs()) {
    if (canAbsorb(input)) {
      collectGroupPostOrder(input, inGroup, topoOrder);
    }
  }
  topoOrder.push_back(node);
}

// Build a FusedElementwiseNode from a topologically-ordered group of nodes.
NodePtr buildFusedNode(
    const std::vector<NodePtr>& topoOrder,
    const std::unordered_set<NodePtr>& inGroup) {
  if (topoOrder.size() < 2) {
    return topoOrder.back();
  }

  // Collect external inputs in order of first appearance
  std::vector<NodePtr> externalInputs;
  std::unordered_map<NodePtr, unsigned> nodeToSlot;
  for (const auto& node : topoOrder) {
    for (const auto& input : node->inputs()) {
      if (!inGroup.contains(input) &&
          !nodeToSlot.contains(input)) {
        nodeToSlot[input] = externalInputs.size();
        externalInputs.push_back(input);
      }
    }
  }

  // Guard: all external inputs must have the same shape as the output.
  // Fused kernels use flat linear indexing (in[idx]) with no broadcast
  // handling; mismatched shapes cause OOB reads. Reject fusion if any
  // external input shape differs from the root output shape.
  const Shape& rootShape = topoOrder.back()->shape();
  for (const auto& ext : externalInputs) {
    if (ext->shape() != rootShape) {
      return topoOrder.back(); // bail — don't fuse
    }
  }

  const unsigned numExternal = externalInputs.size();

  // Assign slot indices to group nodes
  for (unsigned i = 0; i < topoOrder.size(); i++) {
    nodeToSlot[topoOrder[i]] = numExternal + i;
  }

  // Build the program
  std::vector<FusedOp> program;
  program.reserve(topoOrder.size());
  for (const auto& node : topoOrder) {
    if (node->isUnary()) {
      const auto& un = node->impl<UnaryNode>();
      unsigned src0 = nodeToSlot.at(un.input());
      program.push_back(FusedOp{un.op(), src0, 0});
    } else {
      const auto& bn = node->impl<BinaryNode>();
      unsigned src0 = nodeToSlot.at(bn.lhs());
      unsigned src1 = nodeToSlot.at(bn.rhs());
      program.push_back(FusedOp{bn.op(), src0, src1});
    }
  }

  NodePtr root = topoOrder.back();
  auto fusedNode = FusedElementwiseNode::create(
      std::vector<NodePtr>(externalInputs.begin(), externalInputs.end()),
      std::move(program),
      numExternal,
      root->shape(),
      root->stream());

  root->replaceAllUsesWith(fusedNode);
  return fusedNode;
}

} // namespace

NodePtr ElementwiseFusion::rewrite(NodePtr node) {
  if (!isFusionCandidate(node)) {
    return node;
  }

  // Check if any input can be absorbed
  bool hasAbsorbableInput = false;
  for (const auto& input : node->inputs()) {
    if (canAbsorb(input)) {
      hasAbsorbableInput = true;
      break;
    }
  }

  if (!hasAbsorbableInput) {
    return node;
  }

  // Collect the fusion group via post-order DFS (gives topological order)
  std::vector<NodePtr> topoOrder;
  std::unordered_set<NodePtr> inGroup;
  collectGroupPostOrder(node, inGroup, topoOrder);

  // Mark all group members as visited so the base class won't re-process them
  for (const auto& gn : inGroup) {
    markVisited(gn.get());
  }

  return buildFusedNode(topoOrder, inGroup);
}

} // namespace fl
