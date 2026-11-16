/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

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
#include "flashlight/fl/tensor/opt/passes/AlgebraicSimplification.h"

using namespace fl;

namespace {

NodePtr opaque(const Shape& shape, dtype type = dtype::f32) {
  auto a = ScalarNode::create(shape, type, 1.0);
  auto b = ScalarNode::create(shape, type, 0.5);
  return BinaryNode::create(a, b, BinaryOp::Add);
}

NodePtr scalar(const Shape& shape, double val, dtype type = dtype::f32) {
  return ScalarNode::create(shape, type, val);
}

} // namespace

// ==================== Additive Identity ====================

TEST(JitAlgebraicSimplificationTest, AddZeroRight) {
  auto x = opaque({3, 4});
  auto zero = scalar({3, 4}, 0.0);
  auto add = BinaryNode::create(x, zero, BinaryOp::Add);
  AlgebraicSimplification pass;
  auto result = pass.apply(add);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, AddZeroLeft) {
  auto x = opaque({3, 4});
  auto zero = scalar({3, 4}, 0.0);
  auto add = BinaryNode::create(zero, x, BinaryOp::Add);
  AlgebraicSimplification pass;
  auto result = pass.apply(add);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, SubZeroRight) {
  auto x = opaque({3, 4});
  auto zero = scalar({3, 4}, 0.0);
  auto sub = BinaryNode::create(x, zero, BinaryOp::Sub);
  AlgebraicSimplification pass;
  auto result = pass.apply(sub);
  ASSERT_EQ(result, x);
}

// ==================== Sub(0, x) Canonicalization ====================

TEST(JitAlgebraicSimplificationTest, SubZeroLeftToNeg) {
  auto x = opaque({3, 4});
  auto zero = scalar({3, 4}, 0.0);
  auto sub = BinaryNode::create(zero, x, BinaryOp::Sub);
  AlgebraicSimplification pass;
  auto result = pass.apply(sub);
  ASSERT_TRUE(result->isUnary());
  auto& unary = result->impl<UnaryNode>();
  ASSERT_EQ(unary.op(), UnaryOp::Negative);
  ASSERT_EQ(unary.input(), x);
}

// ==================== Multiplicative Identity ====================

TEST(JitAlgebraicSimplificationTest, MulOneRight) {
  auto x = opaque({3, 4});
  auto one = scalar({3, 4}, 1.0);
  auto mul = BinaryNode::create(x, one, BinaryOp::Mul);
  AlgebraicSimplification pass;
  auto result = pass.apply(mul);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, MulOneLeft) {
  auto x = opaque({3, 4});
  auto one = scalar({3, 4}, 1.0);
  auto mul = BinaryNode::create(one, x, BinaryOp::Mul);
  AlgebraicSimplification pass;
  auto result = pass.apply(mul);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, DivOne) {
  auto x = opaque({3, 4});
  auto one = scalar({3, 4}, 1.0);
  auto div = BinaryNode::create(x, one, BinaryOp::Div);
  AlgebraicSimplification pass;
  auto result = pass.apply(div);
  ASSERT_EQ(result, x);
}

// ==================== Power Special Cases ====================

TEST(JitAlgebraicSimplificationTest, PowZero) {
  auto x = opaque({3, 4});
  auto zero = scalar({3, 4}, 0.0);
  auto pow = BinaryNode::create(x, zero, BinaryOp::Pow);
  AlgebraicSimplification pass;
  auto result = pass.apply(pow);
  ASSERT_TRUE(result->isScalar());
  ASSERT_EQ(result->impl<ScalarNode>().scalar<double>(), 1.0);
  ASSERT_EQ(result->shape(), Shape({3, 4}));
}

TEST(JitAlgebraicSimplificationTest, PowOne) {
  auto x = opaque({3, 4});
  auto one = scalar({3, 4}, 1.0);
  auto pow = BinaryNode::create(x, one, BinaryOp::Pow);
  AlgebraicSimplification pass;
  auto result = pass.apply(pow);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, PowTwo) {
  auto x = opaque({3, 4});
  auto two = scalar({3, 4}, 2.0);
  auto pow = BinaryNode::create(x, two, BinaryOp::Pow);
  AlgebraicSimplification pass;
  auto result = pass.apply(pow);
  ASSERT_TRUE(result->isBinary());
  auto& mul = result->impl<BinaryNode>();
  ASSERT_EQ(mul.op(), BinaryOp::Mul);
  ASSERT_EQ(mul.lhs(), x);
  ASSERT_EQ(mul.rhs(), x);
}

TEST(JitAlgebraicSimplificationTest, PowHalf) {
  auto x = opaque({3, 4});
  auto half = scalar({3, 4}, 0.5);
  auto pow = BinaryNode::create(x, half, BinaryOp::Pow);
  AlgebraicSimplification pass;
  auto result = pass.apply(pow);
  ASSERT_TRUE(result->isUnary());
  auto& sqrt = result->impl<UnaryNode>();
  ASSERT_EQ(sqrt.op(), UnaryOp::Sqrt);
  ASSERT_EQ(sqrt.input(), x);
}

// ==================== Unary Involutions ====================

TEST(JitAlgebraicSimplificationTest, NegNeg) {
  auto x = opaque({3, 4});
  auto neg1 = UnaryNode::create(x, UnaryOp::Negative);
  auto neg2 = UnaryNode::create(neg1, UnaryOp::Negative);
  AlgebraicSimplification pass;
  auto result = pass.apply(neg2);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, NotNot) {
  // Use a comparison to produce a proper b8 node
  auto a = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto b = ScalarNode::create({3, 4}, dtype::f32, 0.5);
  auto x = BinaryNode::create(a, b, BinaryOp::Gt); // b8 output
  auto not1 = UnaryNode::create(x, UnaryOp::LogicalNot);
  auto not2 = UnaryNode::create(not1, UnaryOp::LogicalNot);
  AlgebraicSimplification pass;
  auto result = pass.apply(not2);
  ASSERT_EQ(result, x);
}

// ==================== Unary Idempotent ====================

TEST(JitAlgebraicSimplificationTest, AbsAbs) {
  auto x = opaque({3, 4});
  auto abs1 = UnaryNode::create(x, UnaryOp::Absolute);
  auto abs2 = UnaryNode::create(abs1, UnaryOp::Absolute);
  AlgebraicSimplification pass;
  auto result = pass.apply(abs2);
  // abs(abs(x)) → abs(x), not x
  ASSERT_TRUE(result->isUnary());
  ASSERT_EQ(result, abs1);
}

// ==================== Reshape Rules ====================

TEST(JitAlgebraicSimplificationTest, ReshapeIdentity) {
  auto x = opaque({3, 4});
  auto reshape = ReshapeNode::create(x, Shape({3, 4}));
  AlgebraicSimplification pass;
  auto result = pass.apply(reshape);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, ReshapeChain) {
  auto x = opaque({12});
  auto r1 = ReshapeNode::create(x, Shape({3, 4}));
  auto r2 = ReshapeNode::create(r1, Shape({4, 3}));
  AlgebraicSimplification pass;
  auto result = pass.apply(r2);
  ASSERT_TRUE(result->isReshape());
  auto& reshape = result->impl<ReshapeNode>();
  ASSERT_EQ(reshape.input(), x);
  ASSERT_EQ(result->shape(), Shape({4, 3}));
}

// ==================== Negative Tests ====================

TEST(JitAlgebraicSimplificationTest, AddNonZeroNoChange) {
  auto x = opaque({3, 4});
  auto two = scalar({3, 4}, 2.0);
  auto add = BinaryNode::create(x, two, BinaryOp::Add);
  AlgebraicSimplification pass;
  auto result = pass.apply(add);
  ASSERT_EQ(result, add);
}

TEST(JitAlgebraicSimplificationTest, MulNonOneNoChange) {
  auto x = opaque({3, 4});
  auto two = scalar({3, 4}, 2.0);
  auto mul = BinaryNode::create(x, two, BinaryOp::Mul);
  AlgebraicSimplification pass;
  auto result = pass.apply(mul);
  ASSERT_EQ(result, mul);
}

TEST(JitAlgebraicSimplificationTest, NegOfNonNegNoChange) {
  auto x = opaque({3, 4});
  auto neg = UnaryNode::create(x, UnaryOp::Negative);
  AlgebraicSimplification pass;
  auto result = pass.apply(neg);
  ASSERT_EQ(result, neg);
}

TEST(JitAlgebraicSimplificationTest, ReshapeDifferentShapeNoChange) {
  auto x = opaque({3, 4});
  auto reshape = ReshapeNode::create(x, Shape({4, 3}));
  AlgebraicSimplification pass;
  auto result = pass.apply(reshape);
  ASSERT_EQ(result, reshape);
}

// Identity rules must not drop broadcasting: x{4} + 0{2,3,4} is {2,3,4}.
TEST(JitAlgebraicSimplificationTest, AddBroadcastZeroNoChange) {
  auto x = opaque({4});
  auto zero = scalar({2, 3, 4}, 0.0);
  for (auto op : {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul}) {
    auto lhsZero = BinaryNode::create(zero, x, op);
    auto rhsIdentity = BinaryNode::create(
        x, op == BinaryOp::Mul ? scalar({2, 3, 4}, 1.0) : zero, op);
    AlgebraicSimplification pass;
    auto result = pass.apply(rhsIdentity);
    ASSERT_EQ(result, rhsIdentity);
    ASSERT_EQ(result->shape(), Shape({2, 3, 4}));
    if (op != BinaryOp::Mul) {
      auto lhsResult = AlgebraicSimplification().apply(lhsZero);
      ASSERT_EQ(lhsResult->shape(), Shape({2, 3, 4}));
    }
  }
}

TEST(JitAlgebraicSimplificationTest, PowBroadcastExponentNoChange) {
  auto x = opaque({4});
  for (double exponent : {1.0, 2.0, 0.5}) {
    auto pow =
        BinaryNode::create(x, scalar({2, 4}, exponent), BinaryOp::Pow);
    auto result = AlgebraicSimplification().apply(pow);
    ASSERT_EQ(result, pow);
    ASSERT_EQ(result->shape(), Shape({2, 4}));
  }
}

// Identity rules must not drop dtype promotion: f16 x + f32 zero is f32.
TEST(JitAlgebraicSimplificationTest, AddPromotingZeroNoChange) {
  auto x = opaque({3, 4}, dtype::f16);
  auto zero = scalar({3, 4}, 0.0, dtype::f32);
  auto add = BinaryNode::create(x, zero, BinaryOp::Add);
  ASSERT_EQ(add->dataType(), dtype::f32);
  auto result = AlgebraicSimplification().apply(add);
  ASSERT_EQ(result, add);
  ASSERT_EQ(result->dataType(), dtype::f32);
}

// ==================== Tile Rules ====================

TEST(JitAlgebraicSimplificationTest, TileIdentity) {
  auto x = opaque({3, 4});
  auto tile = TileNode::create(x, Shape({1, 1}));
  AlgebraicSimplification pass;
  auto result = pass.apply(tile);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, TileIdentityWithImplicitTrailingOnes) {
  auto x = opaque({3, 4});
  auto tile = TileNode::create(x, Shape({1}));
  AlgebraicSimplification pass;
  auto result = pass.apply(tile);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, TileAllOnesRankExpansionNoFold) {
  auto x = opaque({3});
  auto tile = TileNode::create(x, Shape({1, 1}));
  AlgebraicSimplification pass;
  auto result = pass.apply(tile);
  ASSERT_EQ(result, tile);
  ASSERT_EQ(result->shape(), Shape({1, 3}));
}

TEST(JitAlgebraicSimplificationTest, TileScalarFolding) {
  auto x = scalar({3, 4}, 2.0);
  auto tile = TileNode::create(x, Shape({2, 1}));
  AlgebraicSimplification pass;
  auto result = pass.apply(tile);
  ASSERT_TRUE(result->isScalar());
  ASSERT_EQ(result->shape(), Shape({6, 4}));
  ASSERT_EQ(result->dataType(), dtype::f32);
  ASSERT_EQ(result->impl<ScalarNode>().scalar<double>(), 2.0);
}

TEST(JitAlgebraicSimplificationTest, TileChainSameRank) {
  auto x = opaque({3, 4});
  auto inner = TileNode::create(x, Shape({2, 3}));
  auto outer = TileNode::create(inner, Shape({4, 5}));
  AlgebraicSimplification pass;
  auto result = pass.apply(outer);
  ASSERT_TRUE(result->isTile());
  auto& tile = result->impl<TileNode>();
  ASSERT_EQ(tile.input(), x);
  ASSERT_EQ(tile.tileDims(), Shape({8, 15}));
  ASSERT_EQ(tile.shape(), Shape({24, 60}));
}

TEST(JitAlgebraicSimplificationTest, TileChainRankMismatch) {
  auto x = opaque({3});
  auto inner = TileNode::create(x, Shape({2, 3})); // shape {2, 9}
  auto outer = TileNode::create(inner, Shape({4})); // shape {8, 9}
  AlgebraicSimplification pass;
  auto result = pass.apply(outer);
  ASSERT_TRUE(result->isTile());
  auto& tile = result->impl<TileNode>();
  ASSERT_EQ(tile.input(), x);
  ASSERT_EQ(tile.tileDims(), Shape({8, 3}));
  ASSERT_EQ(tile.shape(), Shape({8, 9}));
}

TEST(JitAlgebraicSimplificationTest, TileTransposeInteractionSameRank) {
  auto x = opaque({2, 3, 4});
  auto transpose = TransposeNode::create(x, Shape({2, 0, 1}));
  auto tile = TileNode::create(transpose, Shape({5, 6, 7}));
  AlgebraicSimplification pass;
  auto result = pass.apply(tile);
  ASSERT_TRUE(result->isTranspose());
  auto& outerTranspose = result->impl<TransposeNode>();
  ASSERT_EQ(outerTranspose.axes(), Shape({2, 0, 1}));
  ASSERT_TRUE(outerTranspose.input()->isTile());
  auto& innerTile = outerTranspose.input()->impl<TileNode>();
  ASSERT_EQ(innerTile.input(), x);
  ASSERT_EQ(innerTile.tileDims(), Shape({6, 7, 5}));
  ASSERT_EQ(result->shape(), Shape({20, 12, 21}));
}

TEST(
    JitAlgebraicSimplificationTest,
    TileTransposeInteractionImplicitTrailingOnes) {
  auto x = opaque({2, 3, 4});
  auto transpose = TransposeNode::create(x, Shape({1, 2, 0}));
  auto tile = TileNode::create(transpose, Shape({5}));
  AlgebraicSimplification pass;
  auto result = pass.apply(tile);
  ASSERT_TRUE(result->isTranspose());
  auto& outerTranspose = result->impl<TransposeNode>();
  ASSERT_EQ(outerTranspose.axes(), Shape({1, 2, 0}));
  ASSERT_TRUE(outerTranspose.input()->isTile());
  auto& innerTile = outerTranspose.input()->impl<TileNode>();
  ASSERT_EQ(innerTile.input(), x);
  ASSERT_EQ(innerTile.tileDims(), Shape({1, 5, 1}));
  ASSERT_EQ(result->shape(), Shape({15, 4, 2}));
}

TEST(JitAlgebraicSimplificationTest, TileTransposeRankExpansionNoFold) {
  auto x = opaque({3});
  auto transpose = TransposeNode::create(x, Shape({0}));
  auto tile = TileNode::create(transpose, Shape({1, 1}));
  AlgebraicSimplification pass;
  auto result = pass.apply(tile);
  ASSERT_EQ(result, tile);
  ASSERT_EQ(result->shape(), Shape({1, 3}));
}

TEST(JitAlgebraicSimplificationTest, TileTransposeDefaultAxesNoFold) {
  auto x = opaque({3, 4});
  auto transpose = TransposeNode::create(x, Shape({}));
  auto tile = TileNode::create(transpose, Shape({2, 3}));
  AlgebraicSimplification pass;
  auto result = pass.apply(tile);
  ASSERT_EQ(result, tile);
}

TEST(JitAlgebraicSimplificationTest, TileTransposeMultiUseNoFold) {
  auto x = opaque({3, 4});
  auto transpose = TransposeNode::create(x, Shape({1, 0}));
  auto otherUse = UnaryNode::create(transpose, UnaryOp::Negative);
  auto tile = TileNode::create(transpose, Shape({2, 3}));
  AlgebraicSimplification pass;
  auto result = pass.apply(tile);
  ASSERT_EQ(result, tile);
  ASSERT_EQ(otherUse->impl<UnaryNode>().input(), transpose);
}

// ==================== Concatenate Rules ====================

TEST(JitAlgebraicSimplificationTest, ConcatenateSingleInputIdentity) {
  auto x = opaque({3, 4});
  auto concat = ConcatenateNode::create({x}, 0);
  AlgebraicSimplification pass;
  auto result = pass.apply(concat);
  ASSERT_EQ(result, x);
}

TEST(
    JitAlgebraicSimplificationTest,
    ConcatenateSingleInputRankExpansionNoFold) {
  auto x = opaque({3});
  auto concat = ConcatenateNode::create({x}, 1);
  AlgebraicSimplification pass;
  auto result = pass.apply(concat);
  ASSERT_EQ(result, concat);
  ASSERT_EQ(result->shape(), Shape({3, 1}));
}

TEST(JitAlgebraicSimplificationTest, ConcatenateDropsEmptyInputs) {
  auto x = opaque({2, 3});
  auto empty = scalar({0}, 0.0);
  auto y = opaque({4, 3});
  auto concat = ConcatenateNode::create({x, empty, y}, 0);
  AlgebraicSimplification pass;
  auto result = pass.apply(concat);
  ASSERT_TRUE(result->isConcatenate());
  auto& simplified = result->impl<ConcatenateNode>();
  ASSERT_EQ(simplified.numInputs(), 2u);
  ASSERT_EQ(simplified.input(0), x);
  ASSERT_EQ(simplified.input(1), y);
  ASSERT_EQ(simplified.shape(), Shape({6, 3}));
}

TEST(JitAlgebraicSimplificationTest, ConcatenateEmptyToIdentity) {
  auto empty = scalar({0}, 0.0);
  auto x = opaque({3, 4});
  auto concat = ConcatenateNode::create({empty, x}, 0);
  AlgebraicSimplification pass;
  auto result = pass.apply(concat);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, ConcatenateKeepsRankPromotingEmpty) {
  auto x = opaque({3});
  auto empty = scalar({0}, 0.0);
  auto concat = ConcatenateNode::create({empty, x}, 1);
  AlgebraicSimplification pass;
  auto result = pass.apply(concat);
  ASSERT_EQ(result, concat);
  ASSERT_EQ(result->shape(), Shape({3, 1}));
}

TEST(JitAlgebraicSimplificationTest, ConcatenateFlattensNestedSameAxis) {
  auto x = opaque({1, 3});
  auto y = opaque({2, 3});
  auto z = opaque({4, 3});
  auto inner = ConcatenateNode::create({y, z}, 0);
  auto outer = ConcatenateNode::create({x, inner}, 0);
  AlgebraicSimplification pass;
  auto result = pass.apply(outer);
  ASSERT_TRUE(result->isConcatenate());
  auto& simplified = result->impl<ConcatenateNode>();
  ASSERT_EQ(simplified.numInputs(), 3u);
  ASSERT_EQ(simplified.input(0), x);
  ASSERT_EQ(simplified.input(1), y);
  ASSERT_EQ(simplified.input(2), z);
  ASSERT_EQ(simplified.shape(), Shape({7, 3}));
}

TEST(JitAlgebraicSimplificationTest, ConcatenateNestedDifferentAxisNoFold) {
  auto x = opaque({2, 1});
  auto y = opaque({2, 2});
  auto z = opaque({2, 3});
  auto inner = ConcatenateNode::create({x, y}, 1);
  auto outer = ConcatenateNode::create({inner, z}, 0);
  AlgebraicSimplification pass;
  auto result = pass.apply(outer);
  ASSERT_EQ(result, outer);
}

// ==================== Pad Rules ====================

TEST(JitAlgebraicSimplificationTest, PadZeroWidthsIdentity) {
  auto x = opaque({3, 4});
  auto pad = PadNode::create(x, {}, PadType::Constant);
  AlgebraicSimplification pass;
  auto result = pass.apply(pad);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, PadNestedConstantFuses) {
  auto x = opaque({2, 3});
  auto inner = PadNode::create(x, {{1, 0}, {0, 2}}, PadType::Constant);
  auto outer = PadNode::create(inner, {{0, 2}, {3, 0}}, PadType::Constant);
  AlgebraicSimplification pass;
  auto result = pass.apply(outer);
  ASSERT_TRUE(result->isPad());
  auto& pad = result->impl<PadNode>();
  ASSERT_EQ(pad.input(), x);
  ASSERT_EQ(
      pad.padWidths(), (std::vector<std::pair<int, int>>{{1, 2}, {3, 2}}));
  ASSERT_EQ(pad.padType(), PadType::Constant);
  ASSERT_EQ(pad.shape(), Shape({5, 8}));
}

TEST(JitAlgebraicSimplificationTest, PadNestedSymmetricNoFold) {
  auto x = opaque({2, 3});
  auto inner = PadNode::create(x, {{1, 0}, {0, 2}}, PadType::Symmetric);
  auto outer = PadNode::create(inner, {{0, 2}, {3, 0}}, PadType::Symmetric);
  AlgebraicSimplification pass;
  auto result = pass.apply(outer);
  ASSERT_EQ(result, outer);
}

TEST(JitAlgebraicSimplificationTest, PadTransposeInteractionExplicitAxes) {
  auto x = opaque({2, 3});
  auto transpose = TransposeNode::create(x, Shape({1, 0}));
  auto pad = PadNode::create(transpose, {{1, 2}, {3, 4}}, PadType::Edge);
  AlgebraicSimplification pass;
  auto result = pass.apply(pad);
  ASSERT_TRUE(result->isTranspose());
  auto& outerTranspose = result->impl<TransposeNode>();
  ASSERT_EQ(outerTranspose.axes(), Shape({1, 0}));
  ASSERT_TRUE(outerTranspose.input()->isPad());
  auto& innerPad = outerTranspose.input()->impl<PadNode>();
  ASSERT_EQ(innerPad.input(), x);
  ASSERT_EQ(
      innerPad.padWidths(), (std::vector<std::pair<int, int>>{{3, 4}, {1, 2}}));
  ASSERT_EQ(innerPad.padType(), PadType::Edge);
  ASSERT_EQ(result->shape(), Shape({6, 9}));
}

TEST(JitAlgebraicSimplificationTest, PadTransposeInteractionDefaultAxes) {
  auto x = opaque({2, 3, 4});
  auto transpose = TransposeNode::create(x, Shape({}));
  auto pad =
      PadNode::create(transpose, {{1, 0}, {0, 2}, {3, 4}}, PadType::Constant);
  AlgebraicSimplification pass;
  auto result = pass.apply(pad);
  ASSERT_TRUE(result->isTranspose());
  auto& outerTranspose = result->impl<TransposeNode>();
  ASSERT_EQ(outerTranspose.axes(), Shape({}));
  ASSERT_TRUE(outerTranspose.input()->isPad());
  auto& innerPad = outerTranspose.input()->impl<PadNode>();
  ASSERT_EQ(innerPad.input(), x);
  ASSERT_EQ(
      innerPad.padWidths(),
      (std::vector<std::pair<int, int>>{{3, 4}, {0, 2}, {1, 0}}));
  ASSERT_EQ(result->shape(), Shape({5, 5, 9}));
}

TEST(JitAlgebraicSimplificationTest, PadTransposeMultiUseNoFold) {
  auto x = opaque({3, 4});
  auto transpose = TransposeNode::create(x, Shape({1, 0}));
  auto otherUse = UnaryNode::create(transpose, UnaryOp::Negative);
  auto pad = PadNode::create(transpose, {{1, 2}, {3, 4}}, PadType::Constant);
  AlgebraicSimplification pass;
  auto result = pass.apply(pad);
  ASSERT_EQ(result, pad);
  ASSERT_EQ(otherUse->impl<UnaryNode>().input(), transpose);
}

// ==================== Flip Rules ====================

TEST(JitAlgebraicSimplificationTest, FlipSingletonAxisIdentity) {
  auto x = opaque({3, 1});
  auto flip = FlipNode::create(x, 1);
  AlgebraicSimplification pass;
  auto result = pass.apply(flip);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, FlipFlipSameAxisCancels) {
  auto x = opaque({3, 4});
  auto inner = FlipNode::create(x, 1);
  auto outer = FlipNode::create(inner, 1);
  AlgebraicSimplification pass;
  auto result = pass.apply(outer);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, FlipFlipDifferentAxesNoFold) {
  auto x = opaque({3, 4});
  auto inner = FlipNode::create(x, 0);
  auto outer = FlipNode::create(inner, 1);
  AlgebraicSimplification pass;
  auto result = pass.apply(outer);
  ASSERT_EQ(result, outer);
}

TEST(JitAlgebraicSimplificationTest, FlipTransposeInteractionExplicitAxes) {
  auto x = opaque({2, 3, 4});
  auto transpose = TransposeNode::create(x, Shape({2, 0, 1}));
  auto flip = FlipNode::create(transpose, 0);
  AlgebraicSimplification pass;
  auto result = pass.apply(flip);
  ASSERT_TRUE(result->isTranspose());
  auto& outerTranspose = result->impl<TransposeNode>();
  ASSERT_EQ(outerTranspose.axes(), Shape({2, 0, 1}));
  ASSERT_TRUE(outerTranspose.input()->isFlip());
  auto& innerFlip = outerTranspose.input()->impl<FlipNode>();
  ASSERT_EQ(innerFlip.input(), x);
  ASSERT_EQ(innerFlip.axis(), 2u);
  ASSERT_EQ(result->shape(), Shape({4, 2, 3}));
}

TEST(JitAlgebraicSimplificationTest, FlipTransposeInteractionDefaultAxes) {
  auto x = opaque({2, 3, 4});
  auto transpose = TransposeNode::create(x, Shape({}));
  auto flip = FlipNode::create(transpose, 0);
  AlgebraicSimplification pass;
  auto result = pass.apply(flip);
  ASSERT_TRUE(result->isTranspose());
  auto& outerTranspose = result->impl<TransposeNode>();
  ASSERT_EQ(outerTranspose.axes(), Shape({}));
  ASSERT_TRUE(outerTranspose.input()->isFlip());
  auto& innerFlip = outerTranspose.input()->impl<FlipNode>();
  ASSERT_EQ(innerFlip.input(), x);
  ASSERT_EQ(innerFlip.axis(), 2u);
  ASSERT_EQ(result->shape(), Shape({4, 3, 2}));
}

TEST(JitAlgebraicSimplificationTest, FlipTransposeMultiUseNoFold) {
  auto x = opaque({3, 4});
  auto transpose = TransposeNode::create(x, Shape({1, 0}));
  auto otherUse = UnaryNode::create(transpose, UnaryOp::Negative);
  auto flip = FlipNode::create(transpose, 0);
  AlgebraicSimplification pass;
  auto result = pass.apply(flip);
  ASSERT_EQ(result, flip);
  ASSERT_EQ(otherUse->impl<UnaryNode>().input(), transpose);
}

// ==================== Triangular Rules ====================

TEST(JitAlgebraicSimplificationTest, TriangularLowerIdentity) {
  auto x = opaque({3, 4});
  auto lower = TriangularNode::create(x, TriangularMode::Lower, 3);
  AlgebraicSimplification pass;
  auto result = pass.apply(lower);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, TriangularUpperIdentity) {
  auto x = opaque({3, 4});
  auto upper = TriangularNode::create(x, TriangularMode::Upper, -2);
  AlgebraicSimplification pass;
  auto result = pass.apply(upper);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, TriangularZeroScalarIdentity) {
  auto zero = scalar({3, 4}, 0.0);
  auto lower = TriangularNode::create(zero, TriangularMode::Lower, -1);
  AlgebraicSimplification pass;
  auto result = pass.apply(lower);
  ASSERT_EQ(result, zero);
}

TEST(JitAlgebraicSimplificationTest, TriangularLowerChainFusesToStricterDiagonal) {
  auto x = opaque({4, 5});
  auto inner = TriangularNode::create(x, TriangularMode::Lower, 2);
  auto outer = TriangularNode::create(inner, TriangularMode::Lower, -1);
  AlgebraicSimplification pass;
  auto result = pass.apply(outer);
  ASSERT_TRUE(result->isTriangular());
  auto& triangular = result->impl<TriangularNode>();
  ASSERT_EQ(triangular.input(), x);
  ASSERT_EQ(triangular.mode(), TriangularMode::Lower);
  ASSERT_EQ(triangular.diagonal(), -1);
}

TEST(JitAlgebraicSimplificationTest, TriangularUpperChainFusesToStricterDiagonal) {
  auto x = opaque({4, 5});
  auto inner = TriangularNode::create(x, TriangularMode::Upper, -2);
  auto outer = TriangularNode::create(inner, TriangularMode::Upper, 1);
  AlgebraicSimplification pass;
  auto result = pass.apply(outer);
  ASSERT_TRUE(result->isTriangular());
  auto& triangular = result->impl<TriangularNode>();
  ASSERT_EQ(triangular.input(), x);
  ASSERT_EQ(triangular.mode(), TriangularMode::Upper);
  ASSERT_EQ(triangular.diagonal(), 1);
}

TEST(JitAlgebraicSimplificationTest, TriangularMixedModeNoFold) {
  auto x = opaque({4, 5});
  auto inner = TriangularNode::create(x, TriangularMode::Lower, 0);
  auto outer = TriangularNode::create(inner, TriangularMode::Upper, 0);
  AlgebraicSimplification pass;
  auto result = pass.apply(outer);
  ASSERT_EQ(result, outer);
}

TEST(JitAlgebraicSimplificationTest, TriangularTransposeSwapsModeAndDiagonal) {
  auto x = opaque({2, 3, 4});
  auto transpose = TransposeNode::create(x, Shape({0, 2, 1}));
  auto lower = TriangularNode::create(transpose, TriangularMode::Lower, 1);
  AlgebraicSimplification pass;
  auto result = pass.apply(lower);
  ASSERT_TRUE(result->isTranspose());
  auto& outerTranspose = result->impl<TransposeNode>();
  ASSERT_EQ(outerTranspose.axes(), Shape({0, 2, 1}));
  ASSERT_TRUE(outerTranspose.input()->isTriangular());
  auto& innerTriangular = outerTranspose.input()->impl<TriangularNode>();
  ASSERT_EQ(innerTriangular.input(), x);
  ASSERT_EQ(innerTriangular.mode(), TriangularMode::Upper);
  ASSERT_EQ(innerTriangular.diagonal(), -1);
  ASSERT_EQ(result->shape(), Shape({2, 4, 3}));
}

TEST(JitAlgebraicSimplificationTest, TriangularTransposeMultiUseNoFold) {
  auto x = opaque({3, 4});
  auto transpose = TransposeNode::create(x, Shape({1, 0}));
  auto otherUse = UnaryNode::create(transpose, UnaryOp::Negative);
  auto lower = TriangularNode::create(transpose, TriangularMode::Lower, 0);
  AlgebraicSimplification pass;
  auto result = pass.apply(lower);
  ASSERT_EQ(result, lower);
  ASSERT_EQ(otherUse->impl<UnaryNode>().input(), transpose);
}

// ==================== Cumsum Rules ====================

TEST(JitAlgebraicSimplificationTest, CumsumSingletonAxisIdentity) {
  auto x = opaque({3, 1});
  auto cumsum = CumsumNode::create(x, 1);
  AlgebraicSimplification pass;
  auto result = pass.apply(cumsum);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, CumsumNonSingletonAxisNoFold) {
  auto x = opaque({3, 4});
  auto cumsum = CumsumNode::create(x, 1);
  AlgebraicSimplification pass;
  auto result = pass.apply(cumsum);
  ASSERT_EQ(result, cumsum);
}

// ==================== Roll Rules ====================

TEST(JitAlgebraicSimplificationTest, RollZeroShiftIdentity) {
  auto x = opaque({3, 4});
  auto roll = RollNode::create(x, 4, 1);
  AlgebraicSimplification pass;
  auto result = pass.apply(roll);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, RollSingletonAxisIdentity) {
  auto x = opaque({3, 1});
  auto roll = RollNode::create(x, 3, 1);
  AlgebraicSimplification pass;
  auto result = pass.apply(roll);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, RollChainSameAxisFuses) {
  auto x = opaque({3, 4});
  auto inner = RollNode::create(x, 1, 1);
  auto outer = RollNode::create(inner, 2, 1);
  AlgebraicSimplification pass;
  auto result = pass.apply(outer);
  ASSERT_TRUE(result->isRoll());
  auto& roll = result->impl<RollNode>();
  ASSERT_EQ(roll.input(), x);
  ASSERT_EQ(roll.shift(), 3);
  ASSERT_EQ(roll.axis(), 1u);
}

TEST(JitAlgebraicSimplificationTest, RollChainSameAxisCancels) {
  auto x = opaque({3, 4});
  auto inner = RollNode::create(x, 1, 1);
  auto outer = RollNode::create(inner, 3, 1);
  AlgebraicSimplification pass;
  auto result = pass.apply(outer);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, RollChainDifferentAxesNoFold) {
  auto x = opaque({3, 4});
  auto inner = RollNode::create(x, 1, 0);
  auto outer = RollNode::create(inner, 1, 1);
  AlgebraicSimplification pass;
  auto result = pass.apply(outer);
  ASSERT_EQ(result, outer);
}

TEST(JitAlgebraicSimplificationTest, RollTransposeInteractionExplicitAxes) {
  auto x = opaque({2, 3, 4});
  auto transpose = TransposeNode::create(x, Shape({2, 0, 1}));
  auto roll = RollNode::create(transpose, 1, 0);
  AlgebraicSimplification pass;
  auto result = pass.apply(roll);
  ASSERT_TRUE(result->isTranspose());
  auto& outerTranspose = result->impl<TransposeNode>();
  ASSERT_EQ(outerTranspose.axes(), Shape({2, 0, 1}));
  ASSERT_TRUE(outerTranspose.input()->isRoll());
  auto& innerRoll = outerTranspose.input()->impl<RollNode>();
  ASSERT_EQ(innerRoll.input(), x);
  ASSERT_EQ(innerRoll.shift(), 1);
  ASSERT_EQ(innerRoll.axis(), 2u);
  ASSERT_EQ(result->shape(), Shape({4, 2, 3}));
}

TEST(JitAlgebraicSimplificationTest, RollTransposeInteractionDefaultAxes) {
  auto x = opaque({2, 3, 4});
  auto transpose = TransposeNode::create(x, Shape({}));
  auto roll = RollNode::create(transpose, -1, 0);
  AlgebraicSimplification pass;
  auto result = pass.apply(roll);
  ASSERT_TRUE(result->isTranspose());
  auto& outerTranspose = result->impl<TransposeNode>();
  ASSERT_EQ(outerTranspose.axes(), Shape({}));
  ASSERT_TRUE(outerTranspose.input()->isRoll());
  auto& innerRoll = outerTranspose.input()->impl<RollNode>();
  ASSERT_EQ(innerRoll.input(), x);
  ASSERT_EQ(innerRoll.shift(), 3);
  ASSERT_EQ(innerRoll.axis(), 2u);
  ASSERT_EQ(result->shape(), Shape({4, 3, 2}));
}

TEST(JitAlgebraicSimplificationTest, RollTransposeMultiUseNoFold) {
  auto x = opaque({3, 4});
  auto transpose = TransposeNode::create(x, Shape({1, 0}));
  auto otherUse = UnaryNode::create(transpose, UnaryOp::Negative);
  auto roll = RollNode::create(transpose, 1, 0);
  AlgebraicSimplification pass;
  auto result = pass.apply(roll);
  ASSERT_EQ(result, roll);
  ASSERT_EQ(otherUse->impl<UnaryNode>().input(), transpose);
}

// ==================== Where Rules ====================

TEST(JitAlgebraicSimplificationTest, WhereSameBranch) {
  auto condition = scalar({3, 4}, 1.0, dtype::b8);
  auto x = opaque({3, 4});
  auto where = WhereNode::create(condition, x, x);
  AlgebraicSimplification pass;
  auto result = pass.apply(where);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, WhereTrueScalarCondition) {
  auto condition = scalar({3, 4}, 1.0, dtype::b8);
  auto x = opaque({3, 4});
  auto y = opaque({3, 4});
  auto where = WhereNode::create(condition, x, y);
  AlgebraicSimplification pass;
  auto result = pass.apply(where);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, WhereFalseScalarCondition) {
  auto condition = scalar({3, 4}, 0.0, dtype::b8);
  auto x = opaque({3, 4});
  auto y = opaque({3, 4});
  auto where = WhereNode::create(condition, x, y);
  AlgebraicSimplification pass;
  auto result = pass.apply(where);
  ASSERT_EQ(result, y);
}

TEST(JitAlgebraicSimplificationTest, WhereTrueMixedDtypeNoFold) {
  auto condition = scalar({3, 4}, 1.0, dtype::b8);
  auto x = opaque({3, 4}, dtype::s32);
  auto y = opaque({3, 4}, dtype::f32);
  auto where = WhereNode::create(condition, x, y);
  AlgebraicSimplification pass;
  auto result = pass.apply(where);
  ASSERT_EQ(result, where);
}

TEST(JitAlgebraicSimplificationTest, WhereFalseMixedDtypeNoFold) {
  auto condition = scalar({3, 4}, 0.0, dtype::b8);
  auto x = opaque({3, 4}, dtype::f32);
  auto y = opaque({3, 4}, dtype::s32);
  auto where = WhereNode::create(condition, x, y);
  AlgebraicSimplification pass;
  auto result = pass.apply(where);
  ASSERT_EQ(result, where);
}

// ==================== Multi-Rule Composition ====================

TEST(JitAlgebraicSimplificationTest, SubZeroThenNegNegCancels) {
  // Sub(0, Sub(0, x)) → first pass: each Sub(0,y)→Neg(y)
  // → Neg(Neg(x)) → x
  auto x = opaque({3, 4});
  auto z1 = scalar({3, 4}, 0.0);
  auto z2 = scalar({3, 4}, 0.0);
  auto inner = BinaryNode::create(z1, x, BinaryOp::Sub);
  auto outer = BinaryNode::create(z2, inner, BinaryOp::Sub);
  AlgebraicSimplification pass;
  auto result = pass.apply(outer);
  ASSERT_EQ(result, x);
}

TEST(JitAlgebraicSimplificationTest, MulOneChainCleanup) {
  // Mul(1, Mul(1, x)) → first inner Mul(1,x)→x, then outer Mul(1,x)→x
  auto x = opaque({3, 4});
  auto one1 = scalar({3, 4}, 1.0);
  auto one2 = scalar({3, 4}, 1.0);
  auto inner = BinaryNode::create(one1, x, BinaryOp::Mul);
  auto outer = BinaryNode::create(one2, inner, BinaryOp::Mul);
  AlgebraicSimplification pass;
  auto result = pass.apply(outer);
  ASSERT_EQ(result, x);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
