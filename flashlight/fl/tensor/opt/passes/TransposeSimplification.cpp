/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/opt/passes/TransposeSimplification.h"

#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/MatmulNode.h"
#include "flashlight/fl/tensor/ir/ReductionNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/TransposeNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"

namespace fl {

namespace {

// Normalize axes: if empty (reverse-all convention), expand to {N-1, ..., 0}.
Shape normalizeAxes(const Shape& axes, int ndim) {
  if (axes.ndim() == 0) {
    std::vector<Dim> reversed(ndim);
    for (int i = 0; i < ndim; ++i) {
      reversed[i] = ndim - 1 - i;
    }
    return Shape(reversed);
  }
  return axes;
}

bool isIdentityPermutation(const Shape& axes) {
  for (int i = 0; i < axes.ndim(); ++i) {
    if (axes[i] != i) {
      return false;
    }
  }
  return true;
}

// Compose permutations: result[i] = inner[outer[i]]
Shape composePermutations(const Shape& outer, const Shape& inner) {
  std::vector<Dim> result(outer.ndim());
  for (int i = 0; i < outer.ndim(); ++i) {
    result[i] = inner[outer[i]];
  }
  return Shape(result);
}

// Check if a permutation only swaps the last two dimensions.
bool isLastTwoDimsSwap(const Shape& axes) {
  if (axes.ndim() < 2) {
    return false;
  }
  int n = axes.ndim();
  // All dims before the last two must be identity.
  for (int i = 0; i < n - 2; ++i) {
    if (axes[i] != i) {
      return false;
    }
  }
  return axes[n - 2] == n - 1 && axes[n - 1] == n - 2;
}

bool hasSingleUse(const NodePtr& node) {
  return !isEvaluated(*node) &&
      node->uses().size() + node->externalUses().size() == 1;
}

// Apply a permutation to a shape: result[i] = shape[axes[i]]
Shape permuteShape(const Shape& shape, const Shape& axes) {
  std::vector<Dim> dims(axes.ndim());
  for (int i = 0; i < axes.ndim(); ++i) {
    dims[i] = shape[axes[i]];
  }
  return Shape(dims);
}

// --- Rule 1 & 2: Identity elimination and transpose-transpose composition ---
NodePtr simplifyTranspose(TransposeNodePtr node) {
  int ndim = node->input()->shape().ndim();
  auto axes = normalizeAxes(node->axes(), ndim);

  // Rule 1: identity permutation → return input directly
  if (isIdentityPermutation(axes)) {
    return node->input();
  }

  // Rule 2: input is also a transpose → compose
  if (node->input()->isTranspose()) {
    auto& inner = node->input()->impl<TransposeNode>();
    auto innerAxes = normalizeAxes(inner.axes(), ndim);
    auto composed = composePermutations(axes, innerAxes);
    if (isIdentityPermutation(composed)) {
      // Double transpose cancels
      return inner.input();
    }
    // Compose into a single transpose
    auto result = TransposeNode::create(
        inner.input(), composed, node->stream());
    node->replaceAllUsesWith(result);
    return result;
  }

  // Rule 3: transpose of scalar → scalar with permuted shape
  if (node->input()->isScalar()) {
    auto& scalar = node->input()->impl<ScalarNode>();
    auto permutedShape = permuteShape(scalar.shape(), axes);
    auto result = ScalarNode::create(
        permutedShape, scalar.dataType(), scalar.scalar<double>(),
        node->stream());
    node->replaceAllUsesWith(result);
    return result;
  }

  return node;
}

// --- Rule 4: Fold transpose into matmul ---
NodePtr simplifyMatmul(MatmulNodePtr node) {
  auto lhs = node->lhs();
  auto rhs = node->rhs();
  bool changed = false;
  auto lhsProp = node->lhsProp();
  auto rhsProp = node->rhsProp();
  NodePtr newLhs = lhs;
  NodePtr newRhs = rhs;

  // Absorb transpose on lhs
  if (lhs->isTranspose() && hasSingleUse(lhs) &&
      lhsProp == MatrixProperty::None) {
    auto& tr = lhs->impl<TransposeNode>();
    auto axes = normalizeAxes(tr.axes(), lhs->shape().ndim());
    if (isLastTwoDimsSwap(axes)) {
      newLhs = tr.input();
      lhsProp = MatrixProperty::Transpose;
      changed = true;
    }
  }

  // Absorb transpose on rhs
  if (rhs->isTranspose() && hasSingleUse(rhs) &&
      rhsProp == MatrixProperty::None) {
    auto& tr = rhs->impl<TransposeNode>();
    auto axes = normalizeAxes(tr.axes(), rhs->shape().ndim());
    if (isLastTwoDimsSwap(axes)) {
      newRhs = tr.input();
      rhsProp = MatrixProperty::Transpose;
      changed = true;
    }
  }

  if (!changed) {
    return node;
  }

  auto result = MatmulNode::create(
      newLhs, newRhs, lhsProp, rhsProp, node->stream());
  node->replaceAllUsesWith(result);
  return result;
}

// --- Rule 5: Fold transpose into reduction ---
NodePtr simplifyReduction(ReductionNodePtr node) {
  auto input = node->input();
  if (!input->isTranspose() || !hasSingleUse(input)) {
    return node;
  }

  auto& tr = input->impl<TransposeNode>();
  int ndim = tr.input()->shape().ndim();
  auto axes = normalizeAxes(tr.axes(), ndim);

  // Remap reduction axes through the transpose permutation.
  // If reduce_axes = {a, b} and transpose axes = {2, 0, 1},
  // then we reduce axis a of the transposed tensor, which corresponds
  // to axis axes[a] of the original tensor.
  auto oldReduceAxes = node->axes();
  std::vector<int> newReduceAxes(oldReduceAxes.size());
  for (size_t i = 0; i < oldReduceAxes.size(); ++i) {
    newReduceAxes[i] = static_cast<int>(axes[oldReduceAxes[i]]);
  }

  // The fold is only valid if the result layout is unchanged: the axes that
  // survive the reduction must keep their order. Without keepDims they must
  // map to increasing original axes; with keepDims every surviving axis must
  // stay in place (e.g. sum(transpose(x, {1,0,2}), {2}) is {B,A}, while
  // sum(x, {2}) is {A,B}).
  std::vector<bool> reduced(ndim, false);
  for (int a : oldReduceAxes) {
    reduced[a] = true;
  }
  int prev = -1;
  for (int j = 0; j < ndim; ++j) {
    if (reduced[j]) {
      continue;
    }
    const int orig = static_cast<int>(axes[j]);
    if (node->keepDims() ? orig != j : orig < prev) {
      return node;
    }
    prev = orig;
  }

  auto result = ReductionNode::create(
      tr.input(), node->op(), newReduceAxes, node->keepDims(), node->stream());
  node->replaceAllUsesWith(result);
  return result;
}

// --- Rule 6: Commute unary through transpose ---
NodePtr simplifyUnary(UnaryNodePtr node) {
  auto input = node->input();
  if (!input->isTranspose() || !hasSingleUse(input)) {
    return node;
  }

  auto& tr = input->impl<TransposeNode>();
  // unary(transpose(x, axes)) → transpose(unary(x), axes)
  auto newUnary = UnaryNode::create(tr.input(), node->op(), node->stream());
  auto result = TransposeNode::create(newUnary, tr.axes(), node->stream());
  node->replaceAllUsesWith(result);
  return result;
}

// --- Rule 7: Commute binary through matching transposes ---
NodePtr simplifyBinary(BinaryNodePtr node) {
  auto lhs = node->lhs();
  auto rhs = node->rhs();
  if (!lhs->isTranspose() || !rhs->isTranspose()) {
    return node;
  }
  if (!hasSingleUse(lhs) || !hasSingleUse(rhs)) {
    return node;
  }

  auto& trLhs = lhs->impl<TransposeNode>();
  auto& trRhs = rhs->impl<TransposeNode>();

  int ndim = trLhs.input()->shape().ndim();
  auto axesLhs = normalizeAxes(trLhs.axes(), ndim);
  auto axesRhs = normalizeAxes(trRhs.axes(), trRhs.input()->shape().ndim());

  // Only commute if both transposes have the same axes.
  if (axesLhs.ndim() != axesRhs.ndim()) {
    return node;
  }
  for (int i = 0; i < axesLhs.ndim(); ++i) {
    if (axesLhs[i] != axesRhs[i]) {
      return node;
    }
  }

  // binary(transpose(x, axes), transpose(y, axes)) → transpose(binary(x, y), axes)
  auto newBinary = BinaryNode::create(
      trLhs.input(), trRhs.input(), node->op(), node->stream());
  auto result = TransposeNode::create(newBinary, axesLhs, node->stream());
  node->replaceAllUsesWith(result);
  return result;
}

} // namespace

NodePtr TransposeSimplification::rewrite(NodePtr node) {
  switch (node->type()) {
    case NodeType::Transpose:
      return simplifyTranspose(Node::cast<TransposeNodePtr>(node));
    case NodeType::Matmul:
      return simplifyMatmul(Node::cast<MatmulNodePtr>(node));
    case NodeType::Reduction:
      return simplifyReduction(Node::cast<ReductionNodePtr>(node));
    case NodeType::Unary:
      return simplifyUnary(Node::cast<UnaryNodePtr>(node));
    case NodeType::Binary:
      return simplifyBinary(Node::cast<BinaryNodePtr>(node));
    default:
      return node;
  }
}

} // namespace fl
