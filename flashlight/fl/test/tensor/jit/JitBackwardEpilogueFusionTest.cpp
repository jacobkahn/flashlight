/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/CastNode.h"
#include "flashlight/fl/tensor/ir/CustomNode.h"
#include "flashlight/fl/tensor/ir/MatmulNode.h"
#include "flashlight/fl/tensor/ir/ProjectionNode.h"
#include "flashlight/fl/tensor/ir/ReductionNode.h"
#include "flashlight/fl/tensor/ir/ReshapeNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"
#include "flashlight/fl/tensor/backend/cpu/CPUBackend.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/opt/backends/cuda/CublasLtBackwardEpilogueFusion.h"

using namespace fl;

namespace {

// Helper: create a ValueNode wrapping a bare CPUTensor (structural only).
NodePtr makeValue(const Shape& shape, dtype type = dtype::f32) {
  Tensor bare = toTensor<CPUTensor>(shape, type, nullptr, Location::Host);
  return ValueNode::create(std::move(bare));
}

// Build the relu mask pattern: Cast(f32, Gt(z, Scalar(0)))
NodePtr makeReluMask(const NodePtr& z) {
  auto scalar0 = ScalarNode::create(z->shape(), dtype::f32, 0.0);
  auto gt = BinaryNode::create(z, scalar0, BinaryOp::Gt);
  return CastNode::create(gt, dtype::f32);
}

// Build the dgelu derivative chain (erf-based) from pre-gelu activation z.
// Mirrors the structure emitted by the dedicated gelu gradFunc:
//   cdf = 0.5 * (1 + erf(z * M_SQRT1_2))
//   pdf = z * exp(-0.5 * z * z) * (M_2_SQRTPI * M_SQRT1_2 * 0.5)
//   dgelu = cdf + pdf
NodePtr makeDgeluChain(const NodePtr& z) {
  // cdf = 0.5 * (1.0 + erf(z * M_SQRT1_2))
  auto sqrt1_2 = ScalarNode::create(z->shape(), dtype::f32, M_SQRT1_2);
  auto zScaled = BinaryNode::create(z, sqrt1_2, BinaryOp::Mul);
  auto erfNode = UnaryNode::create(zScaled, UnaryOp::Erf);
  auto one = ScalarNode::create(z->shape(), dtype::f32, 1.0);
  auto onePlusErf = BinaryNode::create(one, erfNode, BinaryOp::Add);
  auto half = ScalarNode::create(z->shape(), dtype::f32, 0.5);
  auto cdf = BinaryNode::create(half, onePlusErf, BinaryOp::Mul);

  // pdf = z * exp(-0.5 * z * z) * coeff
  auto zz = BinaryNode::create(z, z, BinaryOp::Mul);
  auto negHalf = ScalarNode::create(z->shape(), dtype::f32, -0.5);
  auto negHalfZZ = BinaryNode::create(negHalf, zz, BinaryOp::Mul);
  auto expNode = UnaryNode::create(negHalfZZ, UnaryOp::Exp);
  auto zExp = BinaryNode::create(z, expNode, BinaryOp::Mul);
  double coeff = M_2_SQRTPI * M_SQRT1_2 * 0.5;
  auto coeffScalar = ScalarNode::create(z->shape(), dtype::f32, coeff);
  auto pdf = BinaryNode::create(zExp, coeffScalar, BinaryOp::Mul);

  // dgelu = cdf + pdf
  return BinaryNode::create(cdf, pdf, BinaryOp::Add);
}

} // namespace

// ==================== Existing DRELU tests ====================

// Test: Basic relu mask + matmul fusion
TEST(JitBackwardEpilogueFusion, FusesReluMaskWithMatmul) {
  // matmul([4, 8], [8, 6]) -> [4, 6]
  auto mmLhs = makeValue({4, 8}); // gradOutput
  auto mmRhs = makeValue({8, 6}); // W
  auto mm = MatmulNode::create(
      mmLhs, mmRhs, MatrixProperty::None, MatrixProperty::None);

  // relu mask from z [4, 6]
  auto z = makeValue({4, 6});
  auto reluMask = makeReluMask(z);

  // Mul(reluMask, matmulOutput) — mask * matmul result
  auto mul = BinaryNode::create(reluMask, mm, BinaryOp::Mul);

  CublasLtBackwardEpilogueFusion pass;
  auto result = pass.apply(mul);

  // Should be replaced with a CustomNode
  ASSERT_TRUE(result->isCustom());
  EXPECT_EQ(result->impl<CustomNode>().name(), "CublasLtDrelu");
  EXPECT_EQ(result->shape(), Shape({4, 6}));
  // CustomNode should have 3 inputs: mmLhs, mmRhs, z
  EXPECT_EQ(result->inputs().size(), 3u);
}

// Test: Commuted order — matmul on LHS, mask on RHS
TEST(JitBackwardEpilogueFusion, FusesCommutedOrder) {
  auto mmLhs = makeValue({4, 8});
  auto mmRhs = makeValue({8, 6});
  auto mm = MatmulNode::create(
      mmLhs, mmRhs, MatrixProperty::None, MatrixProperty::None);

  auto z = makeValue({4, 6});
  auto reluMask = makeReluMask(z);

  // Mul(matmulOutput, reluMask) — reversed order
  auto mul = BinaryNode::create(mm, reluMask, BinaryOp::Mul);

  CublasLtBackwardEpilogueFusion pass;
  auto result = pass.apply(mul);

  ASSERT_TRUE(result->isCustom());
  EXPECT_EQ(result->impl<CustomNode>().name(), "CublasLtDrelu");
}

// Test: Fusion through reshape(s) between matmul and mul
TEST(JitBackwardEpilogueFusion, FusesThroughReshape) {
  // matmul produces [4, 6], then reshape to [2, 2, 6], then mul with mask
  auto mmLhs = makeValue({4, 8});
  auto mmRhs = makeValue({8, 6});
  auto mm = MatmulNode::create(
      mmLhs, mmRhs, MatrixProperty::None, MatrixProperty::None);

  auto reshaped = ReshapeNode::create(mm, {2, 2, 6});

  auto z = makeValue({2, 2, 6});
  auto reluMask = makeReluMask(z);

  auto mul = BinaryNode::create(reluMask, reshaped, BinaryOp::Mul);

  CublasLtBackwardEpilogueFusion pass;
  auto result = pass.apply(mul);

  ASSERT_TRUE(result->isCustom());
  EXPECT_EQ(result->impl<CustomNode>().name(), "CublasLtDrelu");
  EXPECT_EQ(result->shape(), Shape({2, 2, 6}));
}

// Test: Multi-use matmul should NOT be fused (for DRELU)
TEST(JitBackwardEpilogueFusion, DoesNotFuseMultiUseMatmul) {
  auto mmLhs = makeValue({4, 8});
  auto mmRhs = makeValue({8, 6});
  auto mm = MatmulNode::create(
      mmLhs, mmRhs, MatrixProperty::None, MatrixProperty::None);

  // Create a second use of the matmul
  auto dummy = BinaryNode::create(mm, makeValue({4, 6}), BinaryOp::Add);

  auto z = makeValue({4, 6});
  auto reluMask = makeReluMask(z);
  auto mul = BinaryNode::create(reluMask, mm, BinaryOp::Mul);

  CublasLtBackwardEpilogueFusion pass;
  auto result = pass.apply(mul);

  // Should NOT be fused — matmul has 2 uses
  EXPECT_FALSE(result->isCustom());
  EXPECT_TRUE(result->isBinary());
}

// Test: Multi-use mask (CastNode) should NOT be fused
TEST(JitBackwardEpilogueFusion, DoesNotFuseMultiUseMask) {
  auto mmLhs = makeValue({4, 8});
  auto mmRhs = makeValue({8, 6});
  auto mm = MatmulNode::create(
      mmLhs, mmRhs, MatrixProperty::None, MatrixProperty::None);

  auto z = makeValue({4, 6});
  auto reluMask = makeReluMask(z);

  // Create a second use of the mask
  auto dummy = BinaryNode::create(reluMask, makeValue({4, 6}), BinaryOp::Add);

  auto mul = BinaryNode::create(reluMask, mm, BinaryOp::Mul);

  CublasLtBackwardEpilogueFusion pass;
  auto result = pass.apply(mul);

  // Should NOT be fused — mask has 2 uses
  EXPECT_FALSE(result->isCustom());
  EXPECT_TRUE(result->isBinary());
}

// Test: Non-f32 matmul should not be fused (for now)
TEST(JitBackwardEpilogueFusion, DoesNotFuseNonF32) {
  auto mmLhs = makeValue({4, 8}, dtype::f64);
  auto mmRhs = makeValue({8, 6}, dtype::f64);
  auto mm = MatmulNode::create(
      mmLhs, mmRhs, MatrixProperty::None, MatrixProperty::None);

  auto z = makeValue({4, 6}, dtype::f64);
  auto scalar0 = ScalarNode::create(z->shape(), dtype::f64, 0.0);
  auto gt = BinaryNode::create(z, scalar0, BinaryOp::Gt);
  // Cast to f32 (this is what relu backward does)
  auto mask = CastNode::create(gt, dtype::f32);

  // But matmul output is f64, so the mul dtypes don't match the pass
  auto mul = BinaryNode::create(mask, mm, BinaryOp::Mul);

  CublasLtBackwardEpilogueFusion pass;
  auto result = pass.apply(mul);

  // Should NOT be fused — matmul is f64
  EXPECT_FALSE(result->isCustom());
}

// ==================== DGELU tests ====================

// Test: DGELU — dgelu chain + matmul fusion
TEST(JitBackwardEpilogueFusion, FusesDgeluWithMatmul) {
  auto mmLhs = makeValue({4, 8});
  auto mmRhs = makeValue({8, 8});
  auto mm = MatmulNode::create(
      mmLhs, mmRhs, MatrixProperty::None, MatrixProperty::None);

  auto z = makeValue({4, 8});
  auto dgeluChain = makeDgeluChain(z);

  // Mul(dgelu_chain, matmul)
  auto mul = BinaryNode::create(dgeluChain, mm, BinaryOp::Mul);

  CublasLtBackwardEpilogueFusion pass;
  auto result = pass.apply(mul);

  ASSERT_TRUE(result->isCustom());
  EXPECT_EQ(result->impl<CustomNode>().name(), "CublasLtDgelu");
  EXPECT_EQ(result->shape(), Shape({4, 8}));
  EXPECT_EQ(result->inputs().size(), 3u);
}

// Test: DGELU — commuted operand order
TEST(JitBackwardEpilogueFusion, FusesDgeluCommuted) {
  auto mmLhs = makeValue({4, 8});
  auto mmRhs = makeValue({8, 8});
  auto mm = MatmulNode::create(
      mmLhs, mmRhs, MatrixProperty::None, MatrixProperty::None);

  auto z = makeValue({4, 8});
  auto dgeluChain = makeDgeluChain(z);

  // Mul(matmul, dgelu_chain) — reversed order
  auto mul = BinaryNode::create(mm, dgeluChain, BinaryOp::Mul);

  CublasLtBackwardEpilogueFusion pass;
  auto result = pass.apply(mul);

  ASSERT_TRUE(result->isCustom());
  EXPECT_EQ(result->impl<CustomNode>().name(), "CublasLtDgelu");
}

// Test: DGELU — non-f32 should not be fused
TEST(JitBackwardEpilogueFusion, DgeluNonF32Rejected) {
  auto mmLhs = makeValue({4, 8}, dtype::f64);
  auto mmRhs = makeValue({8, 8}, dtype::f64);
  auto mm = MatmulNode::create(
      mmLhs, mmRhs, MatrixProperty::None, MatrixProperty::None);

  // Build dgelu chain with f64
  auto z = makeValue({4, 8}, dtype::f64);
  auto dgeluChain = makeDgeluChain(z);

  auto mul = BinaryNode::create(dgeluChain, mm, BinaryOp::Mul);

  CublasLtBackwardEpilogueFusion pass;
  auto result = pass.apply(mul);

  EXPECT_FALSE(result->isCustom());
}

// Test: DGELU — cuBLASLt needs the GELU aux leading dimension (row length)
// to be a multiple of 8, so other widths are left unfused.
TEST(JitBackwardEpilogueFusion, DgeluUnalignedWidthNotFused) {
  auto mmLhs = makeValue({4, 8});
  auto mmRhs = makeValue({8, 6});
  auto mm = MatmulNode::create(
      mmLhs, mmRhs, MatrixProperty::None, MatrixProperty::None);
  auto z = makeValue({4, 6});
  auto mul = BinaryNode::create(makeDgeluChain(z), mm, BinaryOp::Mul);

  CublasLtBackwardEpilogueFusion pass;
  auto result = pass.apply(mul);

  EXPECT_FALSE(result->isCustom());
}

// Test: forward gelu times a matmul (e.g. GEGLU) is not a gelu derivative and
// must not be rewritten to the DGELU epilogue.
TEST(JitBackwardEpilogueFusion, ForwardGeluNotMatchedAsDgelu) {
  auto mmLhs = makeValue({4, 8});
  auto mmRhs = makeValue({8, 8});
  auto mm = MatmulNode::create(
      mmLhs, mmRhs, MatrixProperty::None, MatrixProperty::None);
  auto z = makeValue({4, 8});
  // gelu(z) = z * 0.5 * (1 + erf(z * M_SQRT1_2))
  auto sqrt1_2 = ScalarNode::create(z->shape(), dtype::f32, M_SQRT1_2);
  auto erfNode = UnaryNode::create(
      BinaryNode::create(z, sqrt1_2, BinaryOp::Mul), UnaryOp::Erf);
  auto one = ScalarNode::create(z->shape(), dtype::f32, 1.0);
  auto half = ScalarNode::create(z->shape(), dtype::f32, 0.5);
  auto cdf = BinaryNode::create(
      half, BinaryNode::create(one, erfNode, BinaryOp::Add), BinaryOp::Mul);
  auto gelu = BinaryNode::create(z, cdf, BinaryOp::Mul);
  auto mul = BinaryNode::create(gelu, mm, BinaryOp::Mul);

  CublasLtBackwardEpilogueFusion pass;
  auto result = pass.apply(mul);

  EXPECT_FALSE(result->isCustom());
}

// ==================== BGRADB tests ====================

// Test: BGRADB — Sum(Matmul(A, B), {0}) where matmul has other uses.
// The BGRADB epilogue reduces the GEMM's B operand (a vector as long as D's
// column count), not Sum(D, {0}), so this pattern is deliberately not fused.
TEST(JitBackwardEpilogueFusion, BgradbNotFused) {
  auto mmLhs = makeValue({4, 8});
  auto mmRhs = makeValue({8, 6});
  auto mm = MatmulNode::create(
      mmLhs, mmRhs, MatrixProperty::None, MatrixProperty::None);

  // Create another use of the matmul (e.g., it feeds into the next layer)
  auto otherUse = BinaryNode::create(mm, makeValue({4, 6}), BinaryOp::Add);

  // Sum(matmul, {0}, keepDims=true)
  auto sum = ReductionNode::create(mm, ReductionOp::Sum, {0}, true);

  CublasLtBackwardEpilogueFusion pass;
  auto result = pass.apply(sum);

  EXPECT_EQ(result, sum);
  EXPECT_EQ(otherUse->inputs()[0], mm);
}

// Test: BGRADB — single-use matmul should NOT be fused
TEST(JitBackwardEpilogueFusion, BgradbSingleUseMatmulNotFused) {
  auto mmLhs = makeValue({4, 8});
  auto mmRhs = makeValue({8, 6});
  auto mm = MatmulNode::create(
      mmLhs, mmRhs, MatrixProperty::None, MatrixProperty::None);

  // No other use of the matmul — just the sum
  auto sum = ReductionNode::create(mm, ReductionOp::Sum, {0}, true);

  CublasLtBackwardEpilogueFusion pass;
  auto result = pass.apply(sum);

  // Should NOT be fused — single-use matmul doesn't benefit from BGRADB
  EXPECT_FALSE(result->isProjection());
  EXPECT_TRUE(result->isReduction());
}

// ==================== DRELU_BGRAD tests ====================

// Test: DRELU_BGRAD — Sum(Mul(relu_mask, Matmul(A, B)), {0})
TEST(JitBackwardEpilogueFusion, FusesDreluBgrad) {
  auto mmLhs = makeValue({4, 8});
  auto mmRhs = makeValue({8, 6});
  auto mm = MatmulNode::create(
      mmLhs, mmRhs, MatrixProperty::None, MatrixProperty::None);

  auto z = makeValue({4, 6});
  auto reluMask = makeReluMask(z);
  auto mul = BinaryNode::create(reluMask, mm, BinaryOp::Mul);

  auto sum = ReductionNode::create(mul, ReductionOp::Sum, {0}, true);

  CublasLtBackwardEpilogueFusion pass;
  auto result = pass.apply(sum);

  ASSERT_TRUE(result->isProjection());
  auto& proj = result->impl<ProjectionNode>();
  EXPECT_EQ(proj.outputIndex(), 1u); // bias grad
  auto parent = proj.parent();
  ASSERT_TRUE(parent->isCustom());
  EXPECT_EQ(parent->impl<CustomNode>().name(), "CublasLtDreluBgrad");
  EXPECT_EQ(parent->inputs().size(), 3u); // mmLhs, mmRhs, z
}

// ==================== DGELU_BGRAD tests ====================

// Test: DGELU_BGRAD — Sum(Mul(dgelu_chain(z), Matmul(A, B)), {0})
TEST(JitBackwardEpilogueFusion, FusesDgeluBgrad) {
  auto mmLhs = makeValue({4, 8});
  auto mmRhs = makeValue({8, 8});
  auto mm = MatmulNode::create(
      mmLhs, mmRhs, MatrixProperty::None, MatrixProperty::None);

  auto z = makeValue({4, 8});
  auto dgeluChain = makeDgeluChain(z);
  auto mul = BinaryNode::create(dgeluChain, mm, BinaryOp::Mul);

  auto sum = ReductionNode::create(mul, ReductionOp::Sum, {0}, true);

  CublasLtBackwardEpilogueFusion pass;
  auto result = pass.apply(sum);

  ASSERT_TRUE(result->isProjection());
  auto& proj = result->impl<ProjectionNode>();
  EXPECT_EQ(proj.outputIndex(), 1u); // bias grad
  auto parent = proj.parent();
  ASSERT_TRUE(parent->isCustom());
  EXPECT_EQ(parent->impl<CustomNode>().name(), "CublasLtDgeluBgrad");
  EXPECT_EQ(parent->inputs().size(), 3u); // mmLhs, mmRhs, z
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
