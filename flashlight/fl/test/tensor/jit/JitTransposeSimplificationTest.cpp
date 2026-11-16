/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/MatmulNode.h"
#include "flashlight/fl/tensor/ir/ReductionNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/TransposeNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/opt/passes/TransposeSimplification.h"

using namespace fl;

namespace {

// Create an opaque node that the pass won't simplify (binary of two scalars).
// This prevents Rule 3 (transpose-of-scalar) from firing before other rules.
NodePtr opaque(const Shape& shape, dtype type = dtype::f32) {
  auto a = ScalarNode::create(shape, type, 1.0);
  auto b = ScalarNode::create(shape, type, 0.0);
  return BinaryNode::create(a, b, BinaryOp::Add);
}

} // namespace

// ========================= Rule 1: Identity elimination =========================

TEST(JitTransposeSimplificationTest, identityPermutation) {
  auto x = opaque({3, 4, 5});
  auto tr = TransposeNode::create(x, Shape({0, 1, 2}));
  TransposeSimplification pass;
  auto result = pass.apply(tr);
  ASSERT_EQ(result, x);
}

TEST(JitTransposeSimplificationTest, identityReverse1D) {
  auto x = opaque({5});
  auto tr = TransposeNode::create(x, Shape({}));
  TransposeSimplification pass;
  auto result = pass.apply(tr);
  ASSERT_EQ(result, x);
}

// ========================= Rule 2: Composition/cancellation =========================

TEST(JitTransposeSimplificationTest, transposeTransposeCancels) {
  auto x = opaque({3, 4, 5});
  auto tr1 = TransposeNode::create(x, Shape({2, 0, 1}));
  auto tr2 = TransposeNode::create(tr1, Shape({1, 2, 0}));
  TransposeSimplification pass;
  auto result = pass.apply(tr2);
  ASSERT_EQ(result, x);
}

TEST(JitTransposeSimplificationTest, transposeTransposeComposes) {
  auto x = opaque({3, 4, 5});
  // inner={1,0,2}, outer={2,1,0} → compose: result[i]=inner[outer[i]]
  // result[0]=inner[2]=2, result[1]=inner[1]=0, result[2]=inner[0]=1 → {2,0,1}
  auto tr1 = TransposeNode::create(x, Shape({1, 0, 2}));
  auto tr2 = TransposeNode::create(tr1, Shape({2, 1, 0}));
  TransposeSimplification pass;
  auto result = pass.apply(tr2);
  ASSERT_TRUE(result->isTranspose());
  auto& tr = result->impl<TransposeNode>();
  ASSERT_EQ(tr.axes(), Shape({2, 0, 1}));
  ASSERT_EQ(tr.input(), x);
}

TEST(JitTransposeSimplificationTest, doubleReverseCancels) {
  auto x = opaque({3, 4});
  auto tr1 = TransposeNode::create(x, Shape({})); // reverse → {1, 0}
  auto tr2 = TransposeNode::create(tr1, Shape({})); // reverse again → identity
  TransposeSimplification pass;
  auto result = pass.apply(tr2);
  ASSERT_EQ(result, x);
}

// ========================= Rule 3: Scalar folding =========================

TEST(JitTransposeSimplificationTest, transposeScalar) {
  auto s = ScalarNode::create({3, 4, 5}, dtype::f32, 7.0);
  auto tr = TransposeNode::create(s, Shape({2, 0, 1}));
  TransposeSimplification pass;
  auto result = pass.apply(tr);
  ASSERT_TRUE(result->isScalar());
  ASSERT_EQ(result->shape(), Shape({5, 3, 4}));
}

// ========================= Rule 4: Matmul absorption =========================

TEST(JitTransposeSimplificationTest, matmulAbsorbLhsTranspose) {
  auto x = opaque({4, 3}); // will be transposed to {3, 4}
  auto y = opaque({4, 5});
  auto tr = TransposeNode::create(x, Shape({1, 0}));
  // matmul({3,4}, {4,5}) → {3,5}
  auto mm = MatmulNode::create(
      tr, y, MatrixProperty::None, MatrixProperty::None);
  TransposeSimplification pass;
  auto result = pass.apply(mm);
  ASSERT_TRUE(result->isMatmul());
  auto& matmul = result->impl<MatmulNode>();
  ASSERT_EQ(matmul.lhsProp(), MatrixProperty::Transpose);
  ASSERT_EQ(matmul.rhsProp(), MatrixProperty::None);
  ASSERT_EQ(matmul.lhs(), x);
}

TEST(JitTransposeSimplificationTest, matmulAbsorbRhsTranspose) {
  auto x = opaque({3, 4});
  auto y = opaque({5, 4}); // will be transposed to {4, 5}
  auto tr = TransposeNode::create(y, Shape({1, 0}));
  // matmul({3,4}, {4,5}) → {3,5}
  auto mm = MatmulNode::create(
      x, tr, MatrixProperty::None, MatrixProperty::None);
  TransposeSimplification pass;
  auto result = pass.apply(mm);
  ASSERT_TRUE(result->isMatmul());
  auto& matmul = result->impl<MatmulNode>();
  ASSERT_EQ(matmul.lhsProp(), MatrixProperty::None);
  ASSERT_EQ(matmul.rhsProp(), MatrixProperty::Transpose);
  ASSERT_EQ(matmul.rhs(), y);
}

TEST(JitTransposeSimplificationTest, matmulAbsorbBothTranspose) {
  auto x = opaque({4, 3}); // transposed to {3, 4}
  auto y = opaque({5, 4}); // transposed to {4, 5}
  auto trL = TransposeNode::create(x, Shape({1, 0}));
  auto trR = TransposeNode::create(y, Shape({1, 0}));
  auto mm = MatmulNode::create(
      trL, trR, MatrixProperty::None, MatrixProperty::None);
  TransposeSimplification pass;
  auto result = pass.apply(mm);
  ASSERT_TRUE(result->isMatmul());
  auto& matmul = result->impl<MatmulNode>();
  ASSERT_EQ(matmul.lhsProp(), MatrixProperty::Transpose);
  ASSERT_EQ(matmul.rhsProp(), MatrixProperty::Transpose);
}

TEST(JitTransposeSimplificationTest, matmulNoAbsorbNonSwap) {
  // 3D: transpose with axes {0, 2, 1} IS a last-2 swap → should absorb.
  // But {1, 2, 0} is NOT a last-2 swap → should NOT absorb.
  auto x = opaque({2, 3, 4});
  auto y = opaque({2, 5, 4});
  auto tr = TransposeNode::create(x, Shape({1, 2, 0})); // not last-2 swap
  // tr shape: {3, 4, 2}. matmul({3,4,2}, {2,5,4}): batch=? Won't work.
  // Use shapes that make valid matmul after transpose:
  auto x2 = opaque({3, 4, 5});
  auto y2 = opaque({3, 4, 3});
  // transpose({3,4,5}, {1,2,0}) → {4,5,3}. matmul({4,5,3},{3,4,3}) is invalid...
  // Let me just test that a non-swap transpose is NOT absorbed by checking
  // directly without creating a MatmulNode (since shapes may not be compatible).
  // Instead: create a valid matmul where the lhs IS a non-swap transpose.
  auto a = opaque({2, 4, 3});
  auto b = opaque({2, 3, 5});
  // transpose(a, {0, 2, 1}) → {2, 3, 4}, then matmul({2,3,4},{2,4,5}) — wait
  // that doesn't work either: inner dims 4 vs 4 is fine, but let me use
  // {0, 2, 1} which IS a last-2 swap — bad test.
  // Use {2, 0, 1}: shape {3, 2, 4}, matmul({3,2,4}, {2,3,5}) — batch mismatch.
  // Simplest: just test 2D where {1,0} is always a last-2 swap.
  // For 3D non-swap: {1, 0, 2} — axes that swap first two dims.
  auto c = opaque({3, 4, 5});
  auto d = opaque({4, 5, 7}); // batch=4, inner=5, outer=7
  auto tr2 = TransposeNode::create(c, Shape({1, 0, 2})); // {4,3,5}
  // matmul({4,3,5}, {4,5,7}) → {4,3,7}
  auto mm = MatmulNode::create(
      tr2, d, MatrixProperty::None, MatrixProperty::None);
  TransposeSimplification pass;
  auto result = pass.apply(mm);
  auto& matmul = result->impl<MatmulNode>();
  // {1,0,2} is NOT a last-two-dims swap → should NOT absorb
  ASSERT_EQ(matmul.lhsProp(), MatrixProperty::None);
  ASSERT_TRUE(matmul.lhs()->isTranspose());
}

TEST(JitTransposeSimplificationTest, matmulNoAbsorbAlreadyTransposed) {
  auto x = opaque({4, 3}); // transposed to {3, 4}
  auto y = opaque({3, 5}); // matmul({3,4} transposed → {4,3}, {3,5}) → {4,5}
  auto tr = TransposeNode::create(x, Shape({1, 0}));
  auto mm = MatmulNode::create(
      tr, y, MatrixProperty::Transpose, MatrixProperty::None);
  TransposeSimplification pass;
  auto result = pass.apply(mm);
  // lhsProp is already Transpose → should NOT absorb the TransposeNode
  auto& matmul = result->impl<MatmulNode>();
  ASSERT_EQ(matmul.lhsProp(), MatrixProperty::Transpose);
  ASSERT_TRUE(matmul.lhs()->isTranspose());
}

// ========================= Rule 5: Reduction folding =========================

TEST(JitTransposeSimplificationTest, reductionFoldTranspose) {
  auto x = opaque({3, 4, 5});
  auto tr = TransposeNode::create(x, Shape({2, 0, 1}));
  // Reduce axis 0 of transposed tensor = axis axes[0]=2 of original
  auto red = ReductionNode::create(tr, ReductionOp::Sum, {0}, false);
  TransposeSimplification pass;
  auto result = pass.apply(red);
  ASSERT_TRUE(result->isReduction());
  auto& reduction = result->impl<ReductionNode>();
  ASSERT_EQ(reduction.axes().size(), 1u);
  ASSERT_EQ(reduction.axes()[0], 2);
  ASSERT_EQ(reduction.input(), x);
}

// sum(transpose(x, {1, 0, 2}), {2}) is {B, A}; folding into sum(x, {2}) would
// produce {A, B}, so the transpose must stay.
TEST(JitTransposeSimplificationTest, reductionNoFoldWhenAxesReorder) {
  auto x = opaque({3, 4, 5});
  auto tr = TransposeNode::create(x, Shape({1, 0, 2}));
  auto red = ReductionNode::create(tr, ReductionOp::Sum, {2}, false);
  TransposeSimplification pass;
  auto result = pass.apply(red);
  ASSERT_EQ(result, red);
  ASSERT_EQ(result->shape(), Shape({4, 3}));
}

TEST(JitTransposeSimplificationTest, reductionFoldMultiAxes) {
  auto x = opaque({3, 4, 5});
  auto tr = TransposeNode::create(x, Shape({1, 2, 0}));
  // Reduce axes {0, 2}: axes[0]=1, axes[2]=0 of original
  auto red = ReductionNode::create(tr, ReductionOp::Sum, {0, 2}, false);
  TransposeSimplification pass;
  auto result = pass.apply(red);
  ASSERT_TRUE(result->isReduction());
  auto& reduction = result->impl<ReductionNode>();
  ASSERT_EQ(reduction.axes().size(), 2u);
  ASSERT_EQ(reduction.axes()[0], 1);
  ASSERT_EQ(reduction.axes()[1], 0);
  ASSERT_EQ(reduction.input(), x);
}

// ========================= Rule 6: Unary commutation =========================

TEST(JitTransposeSimplificationTest, unaryCommuteThroughTranspose) {
  auto x = opaque({3, 4});
  auto tr = TransposeNode::create(x, Shape({1, 0}));
  auto unary = UnaryNode::create(tr, UnaryOp::Exp);
  TransposeSimplification pass;
  auto result = pass.apply(unary);
  // Should become transpose(exp(x), {1,0})
  ASSERT_TRUE(result->isTranspose());
  auto& outerTr = result->impl<TransposeNode>();
  ASSERT_TRUE(outerTr.input()->isUnary());
  auto& inner = outerTr.input()->impl<UnaryNode>();
  ASSERT_EQ(inner.op(), UnaryOp::Exp);
  ASSERT_EQ(inner.input(), x);
}

TEST(JitTransposeSimplificationTest, unaryNoCommuteMultiUse) {
  auto x = opaque({3, 4});
  auto tr = TransposeNode::create(x, Shape({1, 0}));
  // Create two uses of the transpose so it's not single-use
  auto u1 = UnaryNode::create(tr, UnaryOp::Exp);
  auto u2 = UnaryNode::create(tr, UnaryOp::Log);
  auto sum = BinaryNode::create(u1, u2, BinaryOp::Add);
  TransposeSimplification pass;
  auto result = pass.apply(sum);
  // transpose has 2 uses — should NOT commute through either unary.
  // The binary's inputs should still be unary nodes with transpose inputs.
  auto lhs = result->inputs()[0];
  auto rhs = result->inputs()[1];
  ASSERT_TRUE(lhs->isUnary());
  ASSERT_TRUE(rhs->isUnary());
  ASSERT_TRUE(lhs->impl<UnaryNode>().input()->isTranspose());
  ASSERT_TRUE(rhs->impl<UnaryNode>().input()->isTranspose());
}

// ========================= Rule 7: Binary commutation =========================

TEST(JitTransposeSimplificationTest, binaryCommuteSameAxes) {
  auto x = opaque({3, 4});
  auto y = opaque({3, 4});
  auto trX = TransposeNode::create(x, Shape({1, 0}));
  auto trY = TransposeNode::create(y, Shape({1, 0}));
  auto add = BinaryNode::create(trX, trY, BinaryOp::Add);
  TransposeSimplification pass;
  auto result = pass.apply(add);
  // Should become transpose(add(x, y), {1, 0})
  ASSERT_TRUE(result->isTranspose());
  auto& outerTr = result->impl<TransposeNode>();
  ASSERT_TRUE(outerTr.input()->isBinary());
  auto& inner = outerTr.input()->impl<BinaryNode>();
  ASSERT_EQ(inner.lhs(), x);
  ASSERT_EQ(inner.rhs(), y);
}

TEST(JitTransposeSimplificationTest, binaryNoCommuteDifferentAxes) {
  auto x = opaque({3, 4, 5});
  auto y = opaque({3, 4, 5});
  auto trX = TransposeNode::create(x, Shape({1, 0, 2})); // → {4,3,5}
  auto trY = TransposeNode::create(y, Shape({0, 2, 1})); // → {3,5,4}
  // These have different shapes after transpose, but Binary may still accept
  // them if they broadcast. The key point is: different axes → no commutation.
  // Use same output shapes by picking axes that produce same shape but are different permutations.
  auto a = opaque({4, 4, 4}); // symmetric shape so any permutation gives {4,4,4}
  auto b = opaque({4, 4, 4});
  auto trA = TransposeNode::create(a, Shape({1, 0, 2}));
  auto trB = TransposeNode::create(b, Shape({2, 1, 0}));
  auto add = BinaryNode::create(trA, trB, BinaryOp::Add);
  TransposeSimplification pass;
  auto result = pass.apply(add);
  // Different axes — should NOT commute. Result should still be Binary.
  ASSERT_TRUE(result->isBinary());
}

// ========================= Multi-rule composition =========================

TEST(JitTransposeSimplificationTest, unaryThenMatmulAbsorb) {
  // exp(transpose(x, {1,0})) fed into matmul.
  // Bottom-up: exp commutes → transpose(exp(x), {1,0}).
  // Then matmul sees transpose lhs → absorbs.
  auto x = opaque({4, 3});
  auto y = opaque({4, 5});
  auto tr = TransposeNode::create(x, Shape({1, 0})); // {3, 4}
  auto expN = UnaryNode::create(tr, UnaryOp::Exp); // {3, 4}
  auto mm = MatmulNode::create(
      expN, y, MatrixProperty::None, MatrixProperty::None);
  TransposeSimplification pass;
  auto result = pass.apply(mm);
  ASSERT_TRUE(result->isMatmul());
  auto& matmul = result->impl<MatmulNode>();
  ASSERT_EQ(matmul.lhsProp(), MatrixProperty::Transpose);
  ASSERT_TRUE(matmul.lhs()->isUnary());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
