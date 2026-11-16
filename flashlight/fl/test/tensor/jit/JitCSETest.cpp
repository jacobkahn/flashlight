/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/CastNode.h"
#include "flashlight/fl/tensor/ir/ClipNode.h"
#include "flashlight/fl/tensor/ir/ConcatenateNode.h"
#include "flashlight/fl/tensor/ir/CumsumNode.h"
#include "flashlight/fl/tensor/ir/FlipNode.h"
#include "flashlight/fl/tensor/ir/FusedElementwiseNode.h"
#include "flashlight/fl/tensor/ir/PadNode.h"
#include "flashlight/fl/tensor/ir/ReductionNode.h"
#include "flashlight/fl/tensor/ir/ReshapeNode.h"
#include "flashlight/fl/tensor/ir/RollNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/TileNode.h"
#include "flashlight/fl/tensor/ir/TransposeNode.h"
#include "flashlight/fl/tensor/ir/TriangularNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/ir/WhereNode.h"
#include "flashlight/fl/tensor/opt/passes/CommonSubexpressionElimination.h"

using namespace fl;

TEST(JitCSETest, noCSESingleNode) {
  Shape shape({3, 4});
  auto s = ScalarNode::create(shape, dtype::f32, 1.0);
  CommonSubexpressionElimination cse;
  auto result = cse.apply(s);
  ASSERT_EQ(result, s);
}

TEST(JitCSETest, cseIdenticalScalars) {
  Shape shape({3, 4});
  auto s1 = ScalarNode::create(shape, dtype::f32, 42.0);
  auto s2 = ScalarNode::create(shape, dtype::f32, 42.0);
  // add(s1, s2) -- s1 and s2 are identical scalars
  auto add = BinaryNode::create(s1, s2, BinaryOp::Add);
  CommonSubexpressionElimination cse;
  auto result = cse.apply(add);
  // After CSE, both inputs should point to the same node
  ASSERT_EQ(result->inputs()[0], result->inputs()[1]);
}

TEST(JitCSETest, noCseDifferentScalars) {
  Shape shape({3, 4});
  auto s1 = ScalarNode::create(shape, dtype::f32, 1.0);
  auto s2 = ScalarNode::create(shape, dtype::f32, 2.0);
  auto add = BinaryNode::create(s1, s2, BinaryOp::Add);
  CommonSubexpressionElimination cse;
  auto result = cse.apply(add);
  // Different values -- no CSE
  ASSERT_NE(result->inputs()[0], result->inputs()[1]);
}

// Scalars that agree to 6 significant digits are still distinct constants.
TEST(JitCSETest, noCseNearbyScalars) {
  Shape shape({3, 4});
  auto s1 = ScalarNode::create(shape, dtype::f32, 1000000.0);
  auto s2 = ScalarNode::create(shape, dtype::f32, 1000001.0);
  auto add = BinaryNode::create(s1, s2, BinaryOp::Add);
  CommonSubexpressionElimination cse;
  auto result = cse.apply(add);
  ASSERT_NE(result->inputs()[0], result->inputs()[1]);
}

TEST(JitCSETest, cseIdenticalBinarySubtrees) {
  // Build: root = mul(a+b, a'+b')
  // where a==a' and b==b' (separate ScalarNodes with same values)
  Shape shape({3, 4});
  auto a = ScalarNode::create(shape, dtype::f32, 1.0);
  auto b = ScalarNode::create(shape, dtype::f32, 2.0);
  auto a2 = ScalarNode::create(shape, dtype::f32, 1.0);
  auto b2 = ScalarNode::create(shape, dtype::f32, 2.0);
  auto add1 = BinaryNode::create(a, b, BinaryOp::Add);
  auto add2 = BinaryNode::create(a2, b2, BinaryOp::Add);
  auto mul = BinaryNode::create(add1, add2, BinaryOp::Mul);

  CommonSubexpressionElimination cse;
  auto result = cse.apply(mul);
  // Both add nodes should have been merged
  ASSERT_EQ(result->inputs()[0], result->inputs()[1]);
}

TEST(JitCSETest, cseIdenticalUnarySubtrees) {
  // Build: root = add(exp(s1), exp(s2)) where s1 == s2
  Shape shape({3, 4});
  auto s1 = ScalarNode::create(shape, dtype::f32, 3.0);
  auto s2 = ScalarNode::create(shape, dtype::f32, 3.0);
  auto exp1 = UnaryNode::create(s1, UnaryOp::Exp);
  auto exp2 = UnaryNode::create(s2, UnaryOp::Exp);
  auto add = BinaryNode::create(exp1, exp2, BinaryOp::Add);

  CommonSubexpressionElimination cse;
  auto result = cse.apply(add);
  ASSERT_EQ(result->inputs()[0], result->inputs()[1]);
}

TEST(JitCSETest, noCseDifferentOps) {
  Shape shape({3, 4});
  auto s = ScalarNode::create(shape, dtype::f32, 1.0);
  auto exp_node = UnaryNode::create(s, UnaryOp::Exp);
  auto log_node = UnaryNode::create(s, UnaryOp::Log);
  auto add = BinaryNode::create(exp_node, log_node, BinaryOp::Add);

  CommonSubexpressionElimination cse;
  auto result = cse.apply(add);
  // Different ops -- no CSE
  ASSERT_NE(result->inputs()[0], result->inputs()[1]);
}

TEST(JitCSETest, cseWithSharedInputs) {
  // Build: root = add(exp(s), exp(s)) where s is the SAME node
  Shape shape({3, 4});
  auto s = ScalarNode::create(shape, dtype::f32, 5.0);
  auto exp1 = UnaryNode::create(s, UnaryOp::Exp);
  auto exp2 = UnaryNode::create(s, UnaryOp::Exp);
  auto add = BinaryNode::create(exp1, exp2, BinaryOp::Add);

  CommonSubexpressionElimination cse;
  auto result = cse.apply(add);
  // Both exp nodes share the same input and op -- should be merged
  ASSERT_EQ(result->inputs()[0], result->inputs()[1]);
}

TEST(JitCSETest, cseReductionNodes) {
  // Build: root = add(sum(s1, [0]), sum(s2, [0])) where s1 == s2
  Shape shape({3, 4});
  auto s1 = ScalarNode::create(shape, dtype::f32, 1.0);
  auto s2 = ScalarNode::create(shape, dtype::f32, 1.0);
  auto sum1 = ReductionNode::create(s1, ReductionOp::Sum, {0}, false);
  auto sum2 = ReductionNode::create(s2, ReductionOp::Sum, {0}, false);
  auto add = BinaryNode::create(sum1, sum2, BinaryOp::Add);

  CommonSubexpressionElimination cse;
  auto result = cse.apply(add);
  ASSERT_EQ(result->inputs()[0], result->inputs()[1]);
}

TEST(JitCSETest, noCseReductionDifferentAxes) {
  Shape shape({3, 4});
  auto s = ScalarNode::create(shape, dtype::f32, 1.0);
  auto sum0 = ReductionNode::create(s, ReductionOp::Sum, {0}, false);
  auto sum1 = ReductionNode::create(s, ReductionOp::Sum, {1}, false);
  // Different shapes from different reduction axes -- BinaryNode::create
  // would fail if shapes don't broadcast. Use separate roots instead.
  CommonSubexpressionElimination cse;
  auto r0 = cse.apply(sum0);
  auto r1 = cse.apply(sum1);
  ASSERT_NE(r0, r1);
}

TEST(JitCSETest, cseIdenticalReshapes) {
  Shape shape({3, 4});
  Shape newShape({12});
  auto s1 = ScalarNode::create(shape, dtype::f32, 1.0);
  auto s2 = ScalarNode::create(shape, dtype::f32, 1.0);
  auto r1 = ReshapeNode::create(s1, newShape);
  auto r2 = ReshapeNode::create(s2, newShape);
  auto add = BinaryNode::create(r1, r2, BinaryOp::Add);

  CommonSubexpressionElimination cse;
  auto result = cse.apply(add);
  ASSERT_EQ(result->inputs()[0], result->inputs()[1]);
}

TEST(JitCSETest, cseDifferentReshapeShapes) {
  // Different target shapes -- should NOT CSE
  auto s = ScalarNode::create({12}, dtype::f32, 1.0);
  auto r1 = ReshapeNode::create(s, Shape({3, 4}));
  auto r2 = ReshapeNode::create(s, Shape({4, 3}));
  // Can't add them directly (different shapes), just CSE individually
  CommonSubexpressionElimination cse;
  auto res1 = cse.apply(r1);
  auto res2 = cse.apply(r2);
  ASSERT_NE(res1, res2);
}

TEST(JitCSETest, cseIdenticalTransposes) {
  Shape shape({3, 4});
  Shape axes({1, 0});
  auto s1 = ScalarNode::create(shape, dtype::f32, 1.0);
  auto s2 = ScalarNode::create(shape, dtype::f32, 1.0);
  auto t1 = TransposeNode::create(s1, axes);
  auto t2 = TransposeNode::create(s2, axes);
  auto add = BinaryNode::create(t1, t2, BinaryOp::Add);

  CommonSubexpressionElimination cse;
  auto result = cse.apply(add);
  ASSERT_EQ(result->inputs()[0], result->inputs()[1]);
}

TEST(JitCSETest, cseIdenticalTiles) {
  auto s1 = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto s2 = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto t1 = TileNode::create(s1, Shape({2, 3}));
  auto t2 = TileNode::create(s2, Shape({2, 3}));
  auto add = BinaryNode::create(t1, t2, BinaryOp::Add);

  CommonSubexpressionElimination cse;
  auto result = cse.apply(add);
  ASSERT_EQ(result->inputs()[0], result->inputs()[1]);
}

TEST(JitCSETest, noCseDifferentTileDims) {
  auto s = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto t1 = TileNode::create(s, Shape({2, 3}));
  auto t2 = TileNode::create(s, Shape({3, 2}));

  CommonSubexpressionElimination cse;
  auto res1 = cse.apply(t1);
  auto res2 = cse.apply(t2);
  ASSERT_NE(res1, res2);
}

TEST(JitCSETest, cseIdenticalConcatenates) {
  auto x1 = ScalarNode::create({2, 3}, dtype::f32, 1.0);
  auto y1 = ScalarNode::create({4, 3}, dtype::f32, 2.0);
  auto x2 = ScalarNode::create({2, 3}, dtype::f32, 1.0);
  auto y2 = ScalarNode::create({4, 3}, dtype::f32, 2.0);
  auto c1 = ConcatenateNode::create({x1, y1}, 0);
  auto c2 = ConcatenateNode::create({x2, y2}, 0);
  auto add = BinaryNode::create(c1, c2, BinaryOp::Add);

  CommonSubexpressionElimination cse;
  auto result = cse.apply(add);
  ASSERT_EQ(result->inputs()[0], result->inputs()[1]);
}

TEST(JitCSETest, noCseDifferentConcatenateAxes) {
  auto x = ScalarNode::create({2, 2}, dtype::f32, 1.0);
  auto y = ScalarNode::create({2, 2}, dtype::f32, 2.0);
  auto c1 = ConcatenateNode::create({x, y}, 0);
  auto c2 = ConcatenateNode::create({x, y}, 1);

  CommonSubexpressionElimination cse;
  auto res1 = cse.apply(c1);
  auto res2 = cse.apply(c2);
  ASSERT_NE(res1, res2);
}

TEST(JitCSETest, cseIdenticalPads) {
  auto x1 = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto x2 = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto p1 = PadNode::create(x1, {{1, 2}, {3, 4}}, PadType::Constant);
  auto p2 = PadNode::create(x2, {{1, 2}, {3, 4}}, PadType::Constant);
  auto add = BinaryNode::create(p1, p2, BinaryOp::Add);

  CommonSubexpressionElimination cse;
  auto result = cse.apply(add);
  ASSERT_EQ(result->inputs()[0], result->inputs()[1]);
}

TEST(JitCSETest, noCseDifferentPadWidths) {
  auto x = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto p1 = PadNode::create(x, {{1, 2}, {3, 4}}, PadType::Constant);
  auto p2 = PadNode::create(x, {{1, 2}, {4, 3}}, PadType::Constant);

  CommonSubexpressionElimination cse;
  auto res1 = cse.apply(p1);
  auto res2 = cse.apply(p2);
  ASSERT_NE(res1, res2);
}

TEST(JitCSETest, noCseDifferentPadTypes) {
  auto x = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto p1 = PadNode::create(x, {{1, 2}, {3, 4}}, PadType::Constant);
  auto p2 = PadNode::create(x, {{1, 2}, {3, 4}}, PadType::Edge);

  CommonSubexpressionElimination cse;
  auto res1 = cse.apply(p1);
  auto res2 = cse.apply(p2);
  ASSERT_NE(res1, res2);
}

TEST(JitCSETest, cseIdenticalFlips) {
  auto s1 = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto s2 = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto f1 = FlipNode::create(s1, 1);
  auto f2 = FlipNode::create(s2, 1);
  auto add = BinaryNode::create(f1, f2, BinaryOp::Add);

  CommonSubexpressionElimination cse;
  auto result = cse.apply(add);
  ASSERT_EQ(result->inputs()[0], result->inputs()[1]);
}

TEST(JitCSETest, noCseDifferentFlipAxes) {
  auto s = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto f1 = FlipNode::create(s, 0);
  auto f2 = FlipNode::create(s, 1);

  CommonSubexpressionElimination cse;
  auto res1 = cse.apply(f1);
  auto res2 = cse.apply(f2);
  ASSERT_NE(res1, res2);
}

TEST(JitCSETest, cseIdenticalClips) {
  auto x1 = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto lo1 = ScalarNode::create({3, 4}, dtype::f32, 0.0);
  auto hi1 = ScalarNode::create({3, 4}, dtype::f32, 2.0);
  auto x2 = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto lo2 = ScalarNode::create({3, 4}, dtype::f32, 0.0);
  auto hi2 = ScalarNode::create({3, 4}, dtype::f32, 2.0);
  auto c1 = ClipNode::create(x1, lo1, hi1);
  auto c2 = ClipNode::create(x2, lo2, hi2);
  auto add = BinaryNode::create(c1, c2, BinaryOp::Add);

  CommonSubexpressionElimination cse;
  auto result = cse.apply(add);
  ASSERT_EQ(result->inputs()[0], result->inputs()[1]);
}

TEST(JitCSETest, noCseDifferentClipBounds) {
  auto x = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto lo = ScalarNode::create({3, 4}, dtype::f32, 0.0);
  auto hi1 = ScalarNode::create({3, 4}, dtype::f32, 2.0);
  auto hi2 = ScalarNode::create({3, 4}, dtype::f32, 3.0);
  auto c1 = ClipNode::create(x, lo, hi1);
  auto c2 = ClipNode::create(x, lo, hi2);

  CommonSubexpressionElimination cse;
  auto res1 = cse.apply(c1);
  auto res2 = cse.apply(c2);
  ASSERT_NE(res1, res2);
}

TEST(JitCSETest, cseIdenticalCumsums) {
  auto s1 = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto s2 = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto c1 = CumsumNode::create(s1, 1);
  auto c2 = CumsumNode::create(s2, 1);
  auto add = BinaryNode::create(c1, c2, BinaryOp::Add);

  CommonSubexpressionElimination cse;
  auto result = cse.apply(add);
  ASSERT_EQ(result->inputs()[0], result->inputs()[1]);
}

TEST(JitCSETest, noCseDifferentCumsumAxes) {
  auto s = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto c1 = CumsumNode::create(s, 0);
  auto c2 = CumsumNode::create(s, 1);

  CommonSubexpressionElimination cse;
  auto res1 = cse.apply(c1);
  auto res2 = cse.apply(c2);
  ASSERT_NE(res1, res2);
}

TEST(JitCSETest, cseIdenticalRolls) {
  auto s1 = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto s2 = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto r1 = RollNode::create(s1, -1, 1);
  auto r2 = RollNode::create(s2, 3, 1);
  auto add = BinaryNode::create(r1, r2, BinaryOp::Add);

  CommonSubexpressionElimination cse;
  auto result = cse.apply(add);
  ASSERT_EQ(result->inputs()[0], result->inputs()[1]);
}

TEST(JitCSETest, noCseDifferentRollParameters) {
  auto s = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto r1 = RollNode::create(s, 1, 0);
  auto r2 = RollNode::create(s, 1, 1);

  CommonSubexpressionElimination cse;
  auto res1 = cse.apply(r1);
  auto res2 = cse.apply(r2);
  ASSERT_NE(res1, res2);
}

TEST(JitCSETest, cseIdenticalTriangularNodes) {
  auto s1 = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto s2 = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto t1 = TriangularNode::create(s1, TriangularMode::Lower, 1);
  auto t2 = TriangularNode::create(s2, TriangularMode::Lower, 1);
  auto add = BinaryNode::create(t1, t2, BinaryOp::Add);

  CommonSubexpressionElimination cse;
  auto result = cse.apply(add);
  ASSERT_EQ(result->inputs()[0], result->inputs()[1]);
}

TEST(JitCSETest, noCseDifferentTriangularModeOrDiagonal) {
  auto s = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto lower = TriangularNode::create(s, TriangularMode::Lower, 1);
  auto upper = TriangularNode::create(s, TriangularMode::Upper, 1);
  auto lowerDifferentDiagonal =
      TriangularNode::create(s, TriangularMode::Lower, 0);

  CommonSubexpressionElimination cse;
  auto resLower = cse.apply(lower);
  auto resUpper = cse.apply(upper);
  auto resLowerDifferentDiagonal = cse.apply(lowerDifferentDiagonal);
  ASSERT_NE(resLower, resUpper);
  ASSERT_NE(resLower, resLowerDifferentDiagonal);
}

TEST(JitCSETest, cseIdenticalWheres) {
  auto c1 = ScalarNode::create({3, 4}, dtype::b8, true);
  auto x1 = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto y1 = ScalarNode::create({3, 4}, dtype::f32, 2.0);
  auto c2 = ScalarNode::create({3, 4}, dtype::b8, true);
  auto x2 = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto y2 = ScalarNode::create({3, 4}, dtype::f32, 2.0);
  auto w1 = WhereNode::create(c1, x1, y1);
  auto w2 = WhereNode::create(c2, x2, y2);
  auto add = BinaryNode::create(w1, w2, BinaryOp::Add);

  CommonSubexpressionElimination cse;
  auto result = cse.apply(add);
  ASSERT_EQ(result->inputs()[0], result->inputs()[1]);
}

TEST(JitCSETest, noCseDifferentWhereConditions) {
  auto x = ScalarNode::create({3, 4}, dtype::f32, 1.0);
  auto y = ScalarNode::create({3, 4}, dtype::f32, 2.0);
  auto trueCond = ScalarNode::create({3, 4}, dtype::b8, true);
  auto falseCond = ScalarNode::create({3, 4}, dtype::b8, false);
  auto w1 = WhereNode::create(trueCond, x, y);
  auto w2 = WhereNode::create(falseCond, x, y);

  CommonSubexpressionElimination cse;
  auto res1 = cse.apply(w1);
  auto res2 = cse.apply(w2);
  ASSERT_NE(res1, res2);
}

TEST(JitCSETest, cseIdenticalCasts) {
  Shape shape({3, 4});
  auto s1 = ScalarNode::create(shape, dtype::f32, 1.0);
  auto s2 = ScalarNode::create(shape, dtype::f32, 1.0);
  auto c1 = CastNode::create(s1, dtype::s32);
  auto c2 = CastNode::create(s2, dtype::s32);
  auto add = BinaryNode::create(c1, c2, BinaryOp::Add);

  CommonSubexpressionElimination cse;
  auto result = cse.apply(add);
  ASSERT_EQ(result->inputs()[0], result->inputs()[1]);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
