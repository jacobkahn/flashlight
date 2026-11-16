/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/opt/passes/ReductionOfElementwiseFusion.h"

#include <vector>

#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/FusedElementwiseNode.h"
#include "flashlight/fl/tensor/ir/FusedReduceElementwiseNode.h"
#include "flashlight/fl/tensor/ir/OpProperties.h"
#include "flashlight/fl/tensor/ir/ReductionNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"

namespace fl {

namespace {

// Check if an elementwise input to a reduction can be absorbed.
// It must have exactly one internal use (the reduction) and no external uses.
bool canAbsorbIntoReduction(NodePtr node) {
  return !isEvaluated(*node) && node->uses().size() == 1 &&
      node->externalUses().empty();
}

// Try to fuse a ReductionNode with its elementwise input.
// Returns the fused node if successful, or the original node if not.
NodePtr tryFuseReduction(ReductionNodePtr reduction) {
  NodePtr input = reduction->input();

  if (!canAbsorbIntoReduction(input)) {
    return reduction;
  }

  if (input->isFusedElementwise()) {
    // Absorb a FusedElementwiseNode
    const auto& fused = input->impl<FusedElementwiseNode>();
    auto fusedNode = FusedReduceElementwiseNode::create(
        std::vector<NodePtr>(input->inputs().begin(), input->inputs().end()),
        fused.program(),
        fused.numExternalInputs(),
        reduction->op(),
        reduction->axes(),
        reduction->keepDims(),
        reduction->shape(),
        reduction->stream());
    reduction->replaceAllUsesWith(fusedNode);
    return fusedNode;
  }

  // Absorb a single fusable elementwise op (unary or binary)
  if (input->isUnary()) {
    const auto& un = input->impl<UnaryNode>();
    if (!isFusableElementwise(un.op())) {
      return reduction;
    }
    std::vector<NodePtr> inputs = {un.input()};
    std::vector<FusedOp> program = {FusedOp{un.op(), 0, 0}};
    auto fusedNode = FusedReduceElementwiseNode::create(
        std::move(inputs),
        std::move(program),
        1,
        reduction->op(),
        reduction->axes(),
        reduction->keepDims(),
        reduction->shape(),
        reduction->stream());
    reduction->replaceAllUsesWith(fusedNode);
    return fusedNode;
  }

  if (input->isBinary()) {
    const auto& bn = input->impl<BinaryNode>();
    if (!isFusableElementwise(bn.op())) {
      return reduction;
    }
    // Guard: inputs must have same shape (no broadcast) for flat-indexed kernel.
    if (bn.lhs()->shape() != bn.rhs()->shape()) {
      return reduction;
    }
    std::vector<NodePtr> inputs;
    unsigned lhsSlot, rhsSlot;
    if (bn.lhs() == bn.rhs()) {
      // Shared input
      inputs = {bn.lhs()};
      lhsSlot = 0;
      rhsSlot = 0;
    } else {
      inputs = {bn.lhs(), bn.rhs()};
      lhsSlot = 0;
      rhsSlot = 1;
    }
    unsigned numExternal = static_cast<unsigned>(inputs.size());
    std::vector<FusedOp> program = {FusedOp{bn.op(), lhsSlot, rhsSlot}};
    auto fusedNode = FusedReduceElementwiseNode::create(
        std::move(inputs),
        std::move(program),
        numExternal,
        reduction->op(),
        reduction->axes(),
        reduction->keepDims(),
        reduction->shape(),
        reduction->stream());
    reduction->replaceAllUsesWith(fusedNode);
    return fusedNode;
  }

  return reduction;
}

} // namespace

NodePtr ReductionOfElementwiseFusion::rewrite(NodePtr node) {
  if (node->isReduction()) {
    return tryFuseReduction(Node::cast<ReductionNodePtr>(node));
  }
  return node;
}

} // namespace fl
