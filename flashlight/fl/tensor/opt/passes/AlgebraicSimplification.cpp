/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/opt/passes/AlgebraicSimplification.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/CastNode.h"
#include "flashlight/fl/tensor/ir/ConcatenateNode.h"
#include "flashlight/fl/tensor/ir/CumsumNode.h"
#include "flashlight/fl/tensor/ir/FlipNode.h"
#include "flashlight/fl/tensor/ir/PadNode.h"
#include "flashlight/fl/tensor/ir/ReshapeNode.h"
#include "flashlight/fl/tensor/ir/RollNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/TileNode.h"
#include "flashlight/fl/tensor/ir/TransposeNode.h"
#include "flashlight/fl/tensor/ir/TriangularNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/ir/WhereNode.h"

namespace fl {

namespace {

// ==================== Helpers ====================

bool isScalarWithValue(const NodePtr& node, double value) {
  return node->isScalar() && node->impl<ScalarNode>().scalar<double>() == value;
}

bool hasSingleUse(const NodePtr& node) {
  return !isEvaluated(*node) &&
      node->uses().size() + node->externalUses().size() == 1;
}

// A rewrite may only substitute a node with the same shape and dtype. Binary
// identities are subject to broadcasting and dtype promotion: `x + 0` is not
// `x` when the zero has a larger shape or a wider dtype.
bool sameMetadata(const NodePtr& a, const NodePtr& b) {
  return a->shape() == b->shape() && a->dataType() == b->dataType();
}

// ==================== Binary Rules ====================

NodePtr simplifyBinary(BinaryNodePtr node) {
  auto op = node->op();
  auto lhs = node->lhs();
  auto rhs = node->rhs();
  const NodePtr self = node;
  const bool lhsKeepsMetadata = sameMetadata(self, lhs);
  const bool rhsKeepsMetadata = sameMetadata(self, rhs);

  switch (op) {
    case BinaryOp::Add:
      if (lhsKeepsMetadata && isScalarWithValue(rhs, 0.0)) {
        return lhs;
      }
      if (rhsKeepsMetadata && isScalarWithValue(lhs, 0.0)) {
        return rhs;
      }
      break;

    case BinaryOp::Sub:
      if (lhsKeepsMetadata && isScalarWithValue(rhs, 0.0)) {
        return lhs;
      }
      if (rhsKeepsMetadata && isScalarWithValue(lhs, 0.0)) {
        // 0 - neg(y) → y (avoid creating neg(neg(y)) since new nodes
        // aren't revisited by the bottom-up traversal)
        if (rhs->isUnary() &&
            rhs->impl<UnaryNode>().op() == UnaryOp::Negative) {
          return rhs->impl<UnaryNode>().input();
        }
        auto result = UnaryNode::create(rhs, UnaryOp::Negative, node->stream());
        node->replaceAllUsesWith(result);
        return result;
      }
      break;

    case BinaryOp::Mul:
      if (lhsKeepsMetadata && isScalarWithValue(rhs, 1.0)) {
        return lhs;
      }
      if (rhsKeepsMetadata && isScalarWithValue(lhs, 1.0)) {
        return rhs;
      }
      break;

    case BinaryOp::Div:
      if (lhsKeepsMetadata && isScalarWithValue(rhs, 1.0)) {
        return lhs;
      }
      break;

    case BinaryOp::Pow:
      if (isScalarWithValue(rhs, 0.0)) {
        auto result = ScalarNode::create(
            node->shape(), node->dataType(), 1.0, node->stream());
        node->replaceAllUsesWith(result);
        return result;
      }
      // The remaining power rules all produce a value with lhs's metadata.
      if (!lhsKeepsMetadata) {
        break;
      }
      if (isScalarWithValue(rhs, 1.0)) {
        return lhs;
      }
      if (isScalarWithValue(rhs, 2.0)) {
        auto result =
            BinaryNode::create(lhs, lhs, BinaryOp::Mul, node->stream());
        node->replaceAllUsesWith(result);
        return result;
      }
      if (isScalarWithValue(rhs, 0.5)) {
        auto result = UnaryNode::create(lhs, UnaryOp::Sqrt, node->stream());
        node->replaceAllUsesWith(result);
        return result;
      }
      break;

    default:
      break;
  }

  // TODO: Annihilator rules (x*0→0, 0/x→0) — skipped because they are
  // not IEEE 754 safe (NaN*0=NaN, not 0; 0/0=NaN, not 0). Add behind
  // a fast-math flag if needed for performance.

  return node;
}

// ==================== Unary Rules ====================

NodePtr simplifyUnary(UnaryNodePtr node) {
  auto op = node->op();
  auto input = node->input();

  if (input->isUnary()) {
    auto& inner = input->impl<UnaryNode>();
    auto innerOp = inner.op();

    // neg(neg(x)) → x
    if (op == UnaryOp::Negative && innerOp == UnaryOp::Negative) {
      return inner.input();
    }
    // not(not(x)) → x (only when x is already boolean — LogicalNot
    // changes non-bool dtypes to b8, so the involution only holds for b8)
    if (op == UnaryOp::LogicalNot && innerOp == UnaryOp::LogicalNot &&
        inner.input()->dataType() == dtype::b8) {
      return inner.input();
    }
    // abs(abs(x)) → abs(x)
    if (op == UnaryOp::Absolute && innerOp == UnaryOp::Absolute) {
      return input;
    }

    // TODO: Inverse pairs — exp(log(x))→x and log(exp(x))→x. Skipped
    // because they are not IEEE 754 safe: exp(log(x)) returns NaN for
    // x≤0, but the simplified x returns the negative value. Add behind
    // a fast-math flag if needed. XLA and PyTorch apply these.
  }

  return node;
}

// ==================== Cast Rules ====================

NodePtr simplifyCast(NodePtr node) {
  auto& cast = node->impl<CastNode>();
  auto input = cast.input();

  // TODO: Round-trip cast elimination — cast(cast(x, t1), t2) where
  // t2==x.dtype() → x. Skipped because it changes precision behavior:
  // cast(f32→f16→f32) quantizes to f16, but the simplified x keeps
  // full f32 precision. XLA and PyTorch apply this. Add behind a
  // fast-math or allow-precision-loss flag if needed.

  // Same-dtype casts are already eliminated by CastNode::create.

  return node;
}

// ==================== Concatenate Rules ====================

NodePtr simplifyConcatenate(NodePtr node) {
  auto& concat = node->impl<ConcatenateNode>();

  // concat([x], axis) → x only when this does not erase Flashlight's
  // rank-promotion behavior for higher-axis single-input concatenation.
  if (concat.numInputs() == 1) {
    auto input = concat.input(0);
    if (sameMetadata(node, input)) {
      return input;
    }
    return node;
  }

  // concat(... empty ..., axis) → concat(non_empty, axis) where legal.
  // Empty tensors can affect rank promotion in Flashlight, so only keep the
  // rewrite if the replacement has exactly the same metadata.
  std::vector<NodePtr> nonemptyInputs;
  nonemptyInputs.reserve(concat.numInputs());
  bool foundEmpty = false;
  for (unsigned i = 0; i < concat.numInputs(); ++i) {
    auto input = concat.input(i);
    if (input->shape().elements() == 0) {
      foundEmpty = true;
    } else {
      nonemptyInputs.push_back(input);
    }
  }
  if (foundEmpty && !nonemptyInputs.empty()) {
    NodePtr replacement = nonemptyInputs.size() == 1
        ? nonemptyInputs.front()
        : ConcatenateNode::create(
              nonemptyInputs, concat.axis(), node->stream());
    if (sameMetadata(node, replacement)) {
      if (replacement != nonemptyInputs.front()) {
        node->replaceAllUsesWith(replacement);
      }
      return replacement;
    }
  }

  // concat(..., concat(..., axis), ..., axis) → concat(..., ..., axis).
  // This is the standard layout-insensitive concatenate canonical form used
  // by mature graph compilers.
  std::vector<NodePtr> flattenedInputs;
  flattenedInputs.reserve(concat.numInputs());
  bool flattened = false;
  for (unsigned i = 0; i < concat.numInputs(); ++i) {
    auto input = concat.input(i);
    if (input->isConcatenate() &&
        input->impl<ConcatenateNode>().axis() == concat.axis()) {
      auto& inner = input->impl<ConcatenateNode>();
      for (unsigned j = 0; j < inner.numInputs(); ++j) {
        flattenedInputs.push_back(inner.input(j));
      }
      flattened = true;
    } else {
      flattenedInputs.push_back(input);
    }
  }
  if (flattened) {
    auto replacement =
        ConcatenateNode::create(flattenedInputs, concat.axis(), node->stream());
    if (sameMetadata(node, replacement)) {
      node->replaceAllUsesWith(replacement);
      return replacement;
    }
  }

  return node;
}

// ==================== Reshape Rules ====================

NodePtr simplifyReshape(NodePtr node) {
  auto& reshape = node->impl<ReshapeNode>();
  auto input = reshape.input();

  // reshape(x, x.shape()) → x
  if (node->shape() == input->shape()) {
    return input;
  }

  // reshape(reshape(x, s1), s2) → reshape(x, s2)
  if (input->isReshape()) {
    auto& inner = input->impl<ReshapeNode>();
    auto result =
        ReshapeNode::create(inner.input(), node->shape(), node->stream());
    node->replaceAllUsesWith(result);
    return result;
  }

  return node;
}

// ==================== Tile Rules ====================

bool isAllOnes(const Shape& dims) {
  for (int i = 0; i < dims.ndim(); ++i) {
    if (dims[i] != 1) {
      return false;
    }
  }
  return true;
}

bool isAllOnes(const std::vector<Dim>& dims) {
  for (auto dim : dims) {
    if (dim != 1) {
      return false;
    }
  }
  return true;
}

std::vector<Dim> effectiveTileDims(
    const Shape& inputShape,
    const Shape& tileDims) {
  auto effective = tileDims.get();
  if (inputShape.ndim() > tileDims.ndim()) {
    effective.insert(effective.end(), inputShape.ndim() - tileDims.ndim(), 1);
  }
  return effective;
}

void leftPadOnesToRank(std::vector<Dim>& dims, int rank) {
  if (static_cast<int>(dims.size()) < rank) {
    dims.insert(dims.begin(), rank - dims.size(), 1);
  }
}

bool isIdentityTile(
    const Shape& inputShape,
    const Shape& outputShape,
    const Shape& tileDims) {
  return inputShape == outputShape &&
      isAllOnes(effectiveTileDims(inputShape, tileDims));
}

bool isExplicitPermutation(const Shape& axes, int rank) {
  if (axes.ndim() != rank) {
    return false;
  }
  std::vector<bool> seen(rank, false);
  for (int i = 0; i < axes.ndim(); ++i) {
    const auto axis = axes[i];
    if (axis < 0 || axis >= rank || seen[axis]) {
      return false;
    }
    seen[axis] = true;
  }
  return true;
}

bool swapsLastTwoDimsOnly(const Shape& axes, int rank) {
  if (!isExplicitPermutation(axes, rank) || rank < 2) {
    return false;
  }
  for (int i = 0; i < rank - 2; ++i) {
    if (axes[i] != i) {
      return false;
    }
  }
  return axes[rank - 2] == rank - 1 && axes[rank - 1] == rank - 2;
}

Shape normalizeTransposeAxes(const Shape& axes, int rank) {
  if (axes.ndim() == 0) {
    std::vector<Dim> reversed(rank);
    for (int i = 0; i < rank; ++i) {
      reversed[i] = rank - 1 - i;
    }
    return Shape(reversed);
  }
  return axes;
}

NodePtr simplifyTile(NodePtr node) {
  auto& tile = node->impl<TileNode>();
  auto input = tile.input();

  // tile(x, {1,1,...,1}) → x only when it preserves rank and shape.
  // Extra leading all-ones tile dims can still rank-expand the tensor.
  if (isIdentityTile(input->shape(), node->shape(), tile.tileDims())) {
    return input;
  }

  // tile(Scalar(shape, val), dims) → Scalar(tiled_shape, val)
  if (input->isScalar()) {
    auto& scalar = input->impl<ScalarNode>();
    auto result = ScalarNode::create(
        node->shape(),
        scalar.dataType(),
        scalar.scalar<double>(),
        node->stream());
    node->replaceAllUsesWith(result);
    return result;
  }

  // tile(tile(x, A), B) → tile(x, A*B)
  if (input->isTile()) {
    auto& inner = input->impl<TileNode>();
    const int rank = node->shape().ndim();

    auto innerDims =
        effectiveTileDims(inner.input()->shape(), inner.tileDims());
    leftPadOnesToRank(innerDims, rank);

    auto outerDims = effectiveTileDims(input->shape(), tile.tileDims());
    leftPadOnesToRank(outerDims, rank);

    std::vector<Dim> combined(rank);
    for (int i = 0; i < rank; ++i) {
      combined[i] = innerDims[i] * outerDims[i];
    }
    auto result =
        TileNode::create(inner.input(), Shape(combined), node->stream());
    node->replaceAllUsesWith(result);
    return result;
  }

  // tile(transpose(x, axes), T) → transpose(tile(x, T'), axes), where T'
  // applies each tile factor to the original axis before transpose.
  //
  // This conservative rule only handles explicit full-rank transposes and
  // rank-preserving tiles. Rank-expanding tiles interact with leading-dim
  // padding and are left unchanged for now.
  if (input->isTranspose() && hasSingleUse(input) &&
      node->shape().ndim() == input->shape().ndim()) {
    auto& transpose = input->impl<TransposeNode>();
    const int rank = input->shape().ndim();
    if (isExplicitPermutation(transpose.axes(), rank)) {
      auto postTransposeTileDims =
          effectiveTileDims(input->shape(), tile.tileDims());
      std::vector<Dim> preTransposeTileDims(rank, 1);
      for (int i = 0; i < rank; ++i) {
        preTransposeTileDims[transpose.axes()[i]] = postTransposeTileDims[i];
      }
      auto tiledInput = TileNode::create(
          transpose.input(), Shape(preTransposeTileDims), node->stream());
      auto result =
          TransposeNode::create(tiledInput, transpose.axes(), node->stream());
      node->replaceAllUsesWith(result);
      return result;
    }
  }

  return node;
}

// ==================== Pad Rules ====================

bool isAllZeroPad(const std::vector<std::pair<int, int>>& padWidths) {
  for (const auto& [before, after] : padWidths) {
    if (before != 0 || after != 0) {
      return false;
    }
  }
  return true;
}

std::vector<std::pair<int, int>> addPadWidths(
    const std::vector<std::pair<int, int>>& lhs,
    const std::vector<std::pair<int, int>>& rhs) {
  std::vector<std::pair<int, int>> result;
  result.reserve(lhs.size());
  for (size_t i = 0; i < lhs.size(); ++i) {
    result.emplace_back(
        lhs[i].first + rhs[i].first, lhs[i].second + rhs[i].second);
  }
  return result;
}

NodePtr simplifyPad(NodePtr node) {
  auto& pad = node->impl<PadNode>();
  auto input = pad.input();

  // pad(x, all_zero_widths, type) → x
  if (isAllZeroPad(pad.padWidths())) {
    return input;
  }

  // pad(pad(x, A, Constant), B, Constant) → pad(x, A+B, Constant).
  // Keep this constant-only for now: repeated symmetric padding is not
  // equivalent to one larger symmetric pad in general.
  if (pad.padType() == PadType::Constant && input->isPad()) {
    auto& inner = input->impl<PadNode>();
    if (inner.padType() == PadType::Constant &&
        inner.padWidths().size() == pad.padWidths().size()) {
      auto combinedWidths = addPadWidths(inner.padWidths(), pad.padWidths());
      auto result = PadNode::create(
          inner.input(), combinedWidths, pad.padType(), node->stream());
      node->replaceAllUsesWith(result);
      return result;
    }
  }

  // pad(transpose(x, axes), W, type)
  //   → transpose(pad(x, W', type), axes)
  //
  // W' applies the post-transpose width for each output axis to the
  // corresponding original input axis.
  if (input->isTranspose() && hasSingleUse(input)) {
    auto& transpose = input->impl<TransposeNode>();
    const int rank = transpose.input()->shape().ndim();
    auto axes = normalizeTransposeAxes(transpose.axes(), rank);
    if (isExplicitPermutation(axes, rank)) {
      std::vector<std::pair<int, int>> preTransposePadWidths(rank, {0, 0});
      for (int i = 0; i < rank; ++i) {
        preTransposePadWidths[axes[i]] = pad.padWidths()[i];
      }
      auto paddedInput = PadNode::create(
          transpose.input(),
          preTransposePadWidths,
          pad.padType(),
          node->stream());
      auto result =
          TransposeNode::create(paddedInput, transpose.axes(), node->stream());
      node->replaceAllUsesWith(result);
      return result;
    }
  }

  return node;
}

// ==================== Cumsum Rules ====================

NodePtr simplifyCumsum(NodePtr node) {
  auto& cumsum = node->impl<CumsumNode>();
  auto input = cumsum.input();

  // cumsum(x, axis) → x when the scan axis has at most one element.
  if (input->shape().dim(cumsum.axis()) <= 1) {
    return input;
  }

  return node;
}

// ==================== Flip Rules ====================

NodePtr simplifyFlip(NodePtr node) {
  auto& flip = node->impl<FlipNode>();
  auto input = flip.input();
  const auto axis = flip.axis();

  // flip(x, axis) → x when reversing that axis cannot change element order.
  if (input->shape().dim(axis) <= 1) {
    return input;
  }

  // flip(flip(x, axis), axis) → x
  if (input->isFlip()) {
    auto& inner = input->impl<FlipNode>();
    if (inner.axis() == axis) {
      return inner.input();
    }
  }

  // flip(transpose(x, axes), axis)
  //   → transpose(flip(x, axes[axis]), axes)
  //
  // Moving flip before transpose keeps the reversal on the original input
  // axis that becomes the flipped output axis. This is rank-preserving, so
  // both explicit axes and the default reverse-axes transpose are safe.
  if (input->isTranspose() && hasSingleUse(input)) {
    auto& transpose = input->impl<TransposeNode>();
    const int rank = transpose.input()->shape().ndim();
    auto axes = normalizeTransposeAxes(transpose.axes(), rank);
    if (isExplicitPermutation(axes, rank)) {
      auto flippedInput = FlipNode::create(
          transpose.input(), static_cast<unsigned>(axes[axis]), node->stream());
      auto result =
          TransposeNode::create(flippedInput, transpose.axes(), node->stream());
      node->replaceAllUsesWith(result);
      return result;
    }
  }

  return node;
}

// ==================== Triangular Rules ====================

TriangularMode oppositeTriangularMode(TriangularMode mode) {
  return mode == TriangularMode::Lower
      ? TriangularMode::Upper
      : TriangularMode::Lower;
}

NodePtr simplifyTriangular(NodePtr node) {
  auto& triangular = node->impl<TriangularNode>();
  auto input = triangular.input();
  const auto diagonal = triangular.diagonal();
  const auto shape = node->shape();
  const auto rank = shape.ndim();
  const auto rows = shape.dim(rank - 2);
  const auto cols = shape.dim(rank - 1);

  if (shape.elements() == 0) {
    return input;
  }

  if (triangular.mode() == TriangularMode::Lower &&
      static_cast<Dim>(diagonal) >= cols - 1) {
    return input;
  }
  if (triangular.mode() == TriangularMode::Upper &&
      static_cast<Dim>(diagonal) <= 1 - rows) {
    return input;
  }

  // tril(0) / triu(0) → 0. This is dtype- and shape-preserving for any
  // diagonal and mirrors constant-mask simplifications in mature compilers.
  if (isScalarWithValue(input, 0.0)) {
    return input;
  }

  // Applying same-mode triangular masks intersects the kept regions:
  // tril(tril(x, a), b) → tril(x, min(a, b))
  // triu(triu(x, a), b) → triu(x, max(a, b))
  if (input->isTriangular()) {
    auto& inner = input->impl<TriangularNode>();
    if (inner.mode() == triangular.mode()) {
      const auto combined = triangular.mode() == TriangularMode::Lower
          ? std::min(inner.diagonal(), diagonal)
          : std::max(inner.diagonal(), diagonal);
      if (combined == inner.diagonal()) {
        return input;
      }
      auto result = TriangularNode::create(
          inner.input(), triangular.mode(), combined, node->stream());
      node->replaceAllUsesWith(result);
      return result;
    }
  }

  // tril(transpose(x), d) → transpose(triu(x, -d))
  // triu(transpose(x), d) → transpose(tril(x, -d))
  //
  // This only holds when transpose preserves all leading axes and swaps the
  // last two matrix axes, matching the triangular op's last-two-dims contract.
  if (input->isTranspose() && hasSingleUse(input) &&
      diagonal != std::numeric_limits<int>::min()) {
    auto& transpose = input->impl<TransposeNode>();
    const auto inputRank = transpose.input()->shape().ndim();
    auto axes = normalizeTransposeAxes(transpose.axes(), inputRank);
    if (swapsLastTwoDimsOnly(axes, inputRank)) {
      auto maskedInput = TriangularNode::create(
          transpose.input(),
          oppositeTriangularMode(triangular.mode()),
          -diagonal,
          node->stream());
      auto result =
          TransposeNode::create(maskedInput, transpose.axes(), node->stream());
      node->replaceAllUsesWith(result);
      return result;
    }
  }

  return node;
}

// ==================== Roll Rules ====================

NodePtr simplifyRoll(NodePtr node) {
  auto& roll = node->impl<RollNode>();
  auto input = roll.input();
  const auto axis = roll.axis();

  // roll(x, 0, axis) → x. Singleton axes normalize to shift 0, but keep
  // the shape guard explicit for empty axes where construction preserves shift.
  if (roll.shift() == 0 || input->shape().dim(axis) <= 1) {
    return input;
  }

  // roll(roll(x, a, axis), b, axis) → roll(x, a + b, axis)
  if (input->isRoll()) {
    auto& inner = input->impl<RollNode>();
    if (inner.axis() == axis) {
      auto result = RollNode::create(
          inner.input(), inner.shift() + roll.shift(), axis, node->stream());
      if (result->shift() == 0) {
        return inner.input();
      }
      node->replaceAllUsesWith(result);
      return result;
    }
  }

  // roll(transpose(x, axes), shift, axis)
  //   → transpose(roll(x, shift, axes[axis]), axes)
  if (input->isTranspose() && hasSingleUse(input)) {
    auto& transpose = input->impl<TransposeNode>();
    const int rank = transpose.input()->shape().ndim();
    auto axes = normalizeTransposeAxes(transpose.axes(), rank);
    if (isExplicitPermutation(axes, rank)) {
      auto rolledInput = RollNode::create(
          transpose.input(),
          roll.shift(),
          static_cast<unsigned>(axes[axis]),
          node->stream());
      auto result =
          TransposeNode::create(rolledInput, transpose.axes(), node->stream());
      node->replaceAllUsesWith(result);
      return result;
    }
  }

  return node;
}

// ==================== Where Rules ====================

NodePtr simplifyWhere(NodePtr node) {
  auto& where = node->impl<WhereNode>();
  auto condition = where.condition();
  auto x = where.x();
  auto y = where.y();

  // where(c, x, x) → x
  if (x == y) {
    return x;
  }

  if (!condition->isScalar() || condition->dataType() != dtype::b8) {
    return node;
  }

  const bool condValue = condition->impl<ScalarNode>().scalar<bool>();

  // Avoid folding mixed-dtype cases until WhereNode metadata and runtime
  // dtype semantics are aligned across backends.
  if (condValue && node->dataType() == x->dataType()) {
    return x;
  }
  if (!condValue && node->dataType() == y->dataType()) {
    return y;
  }

  return node;
}

} // namespace

NodePtr AlgebraicSimplification::rewrite(NodePtr node) {
  switch (node->type()) {
    case NodeType::Binary:
      return simplifyBinary(Node::cast<BinaryNodePtr>(node));
    case NodeType::Unary:
      return simplifyUnary(Node::cast<UnaryNodePtr>(node));
    case NodeType::Cast:
      return simplifyCast(node);
    case NodeType::Concatenate:
      return simplifyConcatenate(node);
    case NodeType::Cumsum:
      return simplifyCumsum(node);
    case NodeType::Flip:
      return simplifyFlip(node);
    case NodeType::Pad:
      return simplifyPad(node);
    case NodeType::Reshape:
      return simplifyReshape(node);
    case NodeType::Roll:
      return simplifyRoll(node);
    case NodeType::Tile:
      return simplifyTile(node);
    case NodeType::Triangular:
      return simplifyTriangular(node);
    case NodeType::Where:
      return simplifyWhere(node);
    default:
      return node;
  }
}

} // namespace fl
