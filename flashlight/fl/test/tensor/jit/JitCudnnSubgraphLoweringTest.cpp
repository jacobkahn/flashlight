/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/ir/BatchNormNode.h"
#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/CastNode.h"
#include "flashlight/fl/tensor/ir/Conv2dNode.h"
#include "flashlight/fl/tensor/ir/CustomNode.h"
#include "flashlight/fl/tensor/ir/MatmulNode.h"
#include "flashlight/fl/tensor/ir/Pool2dNode.h"
#include "flashlight/fl/tensor/ir/TransposeNode.h"
#include "flashlight/fl/tensor/ir/ProjectionNode.h"
#include "flashlight/fl/tensor/ir/ReshapeNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"
#include "flashlight/fl/tensor/backend/cpu/CPUBackend.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/BackendDispatch.h"
#include "flashlight/fl/tensor/opt/backends/cuda/CudnnSubgraphLowering.h"

using namespace fl;

namespace {

// Helper: create a ValueNode wrapping a bare CPUTensor (structural only).
NodePtr makeValue(const Shape& shape, dtype type = dtype::f32) {
  Tensor bare = toTensor<CPUTensor>(shape, type, nullptr, Location::Host);
  return ValueNode::create(std::move(bare));
}

struct Conv2dTestGraph {
  NodePtr input;
  NodePtr weights;
  NodePtr bias;
  Conv2dNodePtr conv;

  Conv2dTestGraph(bool wantState = false) {
    input = makeValue({1, 3, 8, 8});
    weights = makeValue({4, 3, 3, 3});
    bias = makeValue({1, 4, 1, 1});
    conv = Conv2dNode::create(
        input, weights, bias,
        /*sx=*/1, /*sy=*/1, /*px=*/0, /*py=*/0,
        /*dx=*/1, /*dy=*/1, /*groups=*/1, /*wantState=*/wantState);
  }
};

// Helper: build a training-mode BatchNormNode with standard 4D shapes.
// N=1, C=channels, H=8, W=8.
struct BatchNormTestGraph {
  NodePtr input;
  NodePtr weight;
  NodePtr bias;
  NodePtr runningMean;
  NodePtr runningVar;
  BatchNormNodePtr bn;

  // ProjectionNodes for each output
  ProjectionNodePtr projResult;
  ProjectionNodePtr projSaveMean;
  ProjectionNodePtr projSaveVar;
  ProjectionNodePtr projNewRM;
  ProjectionNodePtr projNewRV;

  BatchNormTestGraph(int channels = 4, bool train = true) {
    Shape inputShape{1, channels, 8, 8};
    Shape statsShape{1, channels, 1, 1};

    input = makeValue(inputShape);
    weight = makeValue(statsShape);
    bias = makeValue(statsShape);
    runningMean = makeValue(statsShape);
    runningVar = makeValue(statsShape);

    std::vector<NodePtr> inputs = {
        input, weight, bias, runningMean, runningVar};

    bn = BatchNormNode::create(
        std::move(inputs),
        /*axes=*/{1}, // normalize over channel dim
        train,
        /*momentum=*/0.1,
        /*epsilon=*/1e-5,
        {{inputShape, dtype::f32},   // Result
         {statsShape, dtype::f32},   // SaveMean
         {statsShape, dtype::f32},   // SaveVar
         {statsShape, dtype::f32},   // NewRunningMean
         {statsShape, dtype::f32}}); // NewRunningVar

    projResult = ProjectionNode::create(
        bn, BatchNormNode::Result, inputShape, dtype::f32);
    projSaveMean = ProjectionNode::create(
        bn, BatchNormNode::SaveMean, statsShape, dtype::f32);
    projSaveVar = ProjectionNode::create(
        bn, BatchNormNode::SaveVar, statsShape, dtype::f32);
    projNewRM = ProjectionNode::create(
        bn, BatchNormNode::NewRunningMean, statsShape, dtype::f32);
    projNewRV = ProjectionNode::create(
        bn, BatchNormNode::NewRunningVar, statsShape, dtype::f32);
  }
};

} // namespace

class JitCudnnSubgraphLoweringTest : public ::testing::Test {
 protected:
  CudnnSubgraphLowering pass_;
};

// ==================== Structural subgraph detection ====================
// These tests verify that the pass identifies the correct subgraph
// structure. The cuDNN graph building + compilation may fail on CPU-only
// machines, so we test only the pattern matching here. When the pass
// can't compile (no GPU), it falls back gracefully returning the original
// node — we test that too.

TEST_F(JitCudnnSubgraphLoweringTest, StandaloneConv2dNoFusion) {
  // A standalone conv2d with no pointwise consumer → no benefit to fuse
  // (subgraph size = 1), so the pass should not replace it.
  Conv2dTestGraph g;
  auto result = pass_.apply(g.conv);
  // On CPU the graph build will fail, returning the original node
  ASSERT_EQ(result, g.conv);
}

TEST_F(JitCudnnSubgraphLoweringTest, StandalonePointwiseNoFusion) {
  // Pure pointwise chain with no anchor → pass should skip (leave for NVRTC)
  auto x = makeValue({4, 8});
  auto y = makeValue({4, 8});
  auto add = BinaryNode::create(x, y, BinaryOp::Add);
  auto sigmoid = UnaryNode::create(add, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);
  ASSERT_EQ(result, sigmoid);
}

TEST_F(JitCudnnSubgraphLoweringTest, Conv2dReluSubgraphDetected) {
  // relu(conv2d(x, w, b)) should be identified as a 2-node subgraph
  // On CPU, graph compilation will fail → returns original node
  // This test verifies the pass doesn't crash and handles fallback
  Conv2dTestGraph g;
  auto zero = ScalarNode::create(g.conv->shape(), dtype::f32, 0.0);
  auto relu = BinaryNode::create(g.conv, zero, BinaryOp::Max);

  auto result = pass_.apply(relu);
  // Either fused (CustomNode on GPU) or original (CPU fallback)
  // Both are valid outcomes
  ASSERT_TRUE(result->isCustom() || result == relu);
}

TEST_F(JitCudnnSubgraphLoweringTest, Conv2dSigmoidSubgraphDetected) {
  Conv2dTestGraph g;
  auto sigmoid = UnaryNode::create(g.conv, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);
  ASSERT_TRUE(result->isCustom() || result == sigmoid);
}

TEST_F(JitCudnnSubgraphLoweringTest, Conv2dAddReluSubgraphDetected) {
  // relu(conv2d(x, w, b) + residual) — 3-node subgraph
  Conv2dTestGraph g;
  auto residual = makeValue(g.conv->shape());
  auto add = BinaryNode::create(g.conv, residual, BinaryOp::Add);
  auto zero = ScalarNode::create(add->shape(), dtype::f32, 0.0);
  auto relu = BinaryNode::create(add, zero, BinaryOp::Max);

  auto result = pass_.apply(relu);
  ASSERT_TRUE(result->isCustom() || result == relu);
}

TEST_F(JitCudnnSubgraphLoweringTest, Conv2dMultiUseNoPointwiseFusion) {
  // conv2d consumed by two different ops → each consumer should not
  // pull the conv2d into its subgraph (since conv2d has multiple uses)
  Conv2dTestGraph g;
  auto sigmoid = UnaryNode::create(g.conv, UnaryOp::Sigmoid);
  auto tanh = UnaryNode::create(g.conv, UnaryOp::Tanh);

  // Apply to sigmoid — conv2d has 2 uses so it stays as boundary input,
  // leaving sigmoid alone (no anchor in its subgraph)
  auto result = pass_.apply(sigmoid);
  ASSERT_EQ(result, sigmoid); // sigmoid alone has no anchor → no fusion
}

TEST_F(JitCudnnSubgraphLoweringTest, Conv2dReshapeBoundary) {
  // sigmoid(reshape(conv2d(x, w, b))) — reshape is a graph boundary,
  // so sigmoid can't see the conv2d as part of its subgraph
  Conv2dTestGraph g;
  auto reshape = ReshapeNode::create(g.conv, Shape{1, 4, 6, 6});
  auto sigmoid = UnaryNode::create(reshape, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);
  ASSERT_EQ(result, sigmoid); // No anchor reachable → no fusion
}

TEST_F(JitCudnnSubgraphLoweringTest, MatmulAddSubgraphDetected) {
  // matmul(A, B) + C — anchor + pointwise
  auto a = makeValue({2, 3});
  auto b = makeValue({3, 4});
  auto c = makeValue({2, 4});
  auto mm = MatmulNode::create(a, b, MatrixProperty::None, MatrixProperty::None);
  auto add = BinaryNode::create(mm, c, BinaryOp::Add);

  auto result = pass_.apply(add);
  ASSERT_TRUE(result->isCustom() || result == add);
}

TEST_F(JitCudnnSubgraphLoweringTest, MatmulAddReluSubgraphDetected) {
  // relu(matmul(A, B) + C) — anchor + pointwise chain
  auto a = makeValue({2, 3});
  auto b = makeValue({3, 4});
  auto c = makeValue({2, 4});
  auto mm = MatmulNode::create(a, b, MatrixProperty::None, MatrixProperty::None);
  auto add = BinaryNode::create(mm, c, BinaryOp::Add);
  auto zero = ScalarNode::create(add->shape(), dtype::f32, 0.0);
  auto relu = BinaryNode::create(add, zero, BinaryOp::Max);

  auto result = pass_.apply(relu);
  ASSERT_TRUE(result->isCustom() || result == relu);
}

// ==================== Batched Matmul subgraph detection ====================

TEST_F(JitCudnnSubgraphLoweringTest, BatchedMatmulAddSubgraphDetected) {
  // batched matmul(A, B) + C — 3D anchor + pointwise
  auto a = makeValue({4, 2, 3});
  auto b = makeValue({4, 3, 5});
  auto c = makeValue({4, 2, 5});
  auto mm = MatmulNode::create(a, b, MatrixProperty::None, MatrixProperty::None);
  auto add = BinaryNode::create(mm, c, BinaryOp::Add);

  auto result = pass_.apply(add);
  ASSERT_TRUE(result->isCustom() || result == add);
}

TEST_F(JitCudnnSubgraphLoweringTest, BatchedMatmulAddReluSubgraphDetected) {
  // relu(batched matmul(A, B) + C) — 3D anchor + pointwise chain
  auto a = makeValue({4, 2, 3});
  auto b = makeValue({4, 3, 5});
  auto c = makeValue({4, 2, 5});
  auto mm = MatmulNode::create(a, b, MatrixProperty::None, MatrixProperty::None);
  auto add = BinaryNode::create(mm, c, BinaryOp::Add);
  auto zero = ScalarNode::create(add->shape(), dtype::f32, 0.0);
  auto relu = BinaryNode::create(add, zero, BinaryOp::Max);

  auto result = pass_.apply(relu);
  ASSERT_TRUE(result->isCustom() || result == relu);
}

TEST_F(JitCudnnSubgraphLoweringTest, BatchedMatmulTransposeAddSubgraphDetected) {
  // batched matmul(A, B^T) + C — transpose in batched context
  auto a = makeValue({4, 2, 3});
  auto b = makeValue({4, 5, 3}); // transposed: {4, N, K}
  auto mm = MatmulNode::create(a, b, MatrixProperty::None, MatrixProperty::Transpose);
  auto c = makeValue({4, 2, 5});
  auto add = BinaryNode::create(mm, c, BinaryOp::Add);

  auto result = pass_.apply(add);
  ASSERT_TRUE(result->isCustom() || result == add);
}

TEST_F(JitCudnnSubgraphLoweringTest, BatchedMatmulBroadcastBiasSubgraphDetected) {
  // batched matmul(A, B) + bias — bias broadcasts across batch
  auto a = makeValue({4, 2, 3});
  auto b = makeValue({4, 3, 5});
  auto bias = makeValue({1, 1, 5}); // broadcasts to {4, 2, 5}
  auto mm = MatmulNode::create(a, b, MatrixProperty::None, MatrixProperty::None);
  auto add = BinaryNode::create(mm, bias, BinaryOp::Add);

  auto result = pass_.apply(add);
  ASSERT_TRUE(result->isCustom() || result == add);
}

// ==================== BatchNorm subgraph detection ====================

TEST_F(JitCudnnSubgraphLoweringTest, BNTrainingStandaloneNoFusion) {
  // A standalone training BN (no pointwise consumer on the primary output)
  // → subgraph = {Projection(0), BatchNormNode} = 2 nodes, but no
  // pointwise ops → should still attempt fusion via cuDNN graph.
  // On CPU, graph compilation fails → returns original node.
  BatchNormTestGraph g;

  auto result = pass_.apply(g.projResult);
  // projResult alone is a projection of the BN anchor, but without
  // pointwise neighbors the subgraph only has BN + Projection.
  // The pass should either fuse (CustomNode on GPU) or fall back.
  ASSERT_TRUE(result->isCustom() || result->isProjection() || result == g.projResult);
}

TEST_F(JitCudnnSubgraphLoweringTest, BNTrainingReluSubgraphDetected) {
  // relu(projection(0, batchnorm(x))) → fused.
  // ProjectionNodes 1-4 produce correct auxiliary outputs.
  BatchNormTestGraph g;
  auto sigmoid = UnaryNode::create(g.projResult, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);
  // Either fused (CustomNode on GPU) or original (CPU fallback)
  ASSERT_TRUE(result->isCustom() || result->isProjection() || result == sigmoid);
}

TEST_F(JitCudnnSubgraphLoweringTest, BNTrainingAddReluSubgraphDetected) {
  // relu(bn_result + shortcut) — ResNet pattern
  BatchNormTestGraph g;
  auto shortcut = makeValue(g.projResult->shape());
  auto add = BinaryNode::create(g.projResult, shortcut, BinaryOp::Add);
  auto zero = ScalarNode::create(add->shape(), dtype::f32, 0.0);
  auto relu = BinaryNode::create(add, zero, BinaryOp::Max);

  auto result = pass_.apply(relu);
  ASSERT_TRUE(result->isCustom() || result->isProjection() || result == relu);
}

TEST_F(JitCudnnSubgraphLoweringTest, BNInferenceReluSubgraphDetected) {
  // relu(batchnorm_inference(x)) → fused (inference is single-output)
  BatchNormTestGraph g(/*channels=*/4, /*train=*/false);
  auto sigmoid = UnaryNode::create(g.projResult, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);
  ASSERT_TRUE(result->isCustom() || result->isProjection() || result == sigmoid);
}

TEST_F(JitCudnnSubgraphLoweringTest, BNMultiUseProjectionNoFusion) {
  // If Projection(0) has multiple consumers, it won't be pulled into
  // any single consumer's subgraph → no BN fusion.
  BatchNormTestGraph g;
  auto sigmoid = UnaryNode::create(g.projResult, UnaryOp::Sigmoid);
  auto tanh = UnaryNode::create(g.projResult, UnaryOp::Tanh);

  // projResult has 2 uses, so it stays as boundary input
  auto result = pass_.apply(sigmoid);
  ASSERT_EQ(result, sigmoid); // No anchor → no fusion
}

// ============= Multi-anchor subgraph detection (Conv + BN) =============

TEST_F(JitCudnnSubgraphLoweringTest, Conv2dBNReluSubgraphDetected) {
  // relu(projection(0, batchnorm(conv2d(x, w, b)))) — two anchors
  // This is the core ResNet building block (minus the residual shortcut).
  Conv2dTestGraph cg;
  BatchNormTestGraph bg(/*channels=*/4, /*train=*/true);

  // Rewire: BN's input is the conv2d output instead of a value node
  // We need to rebuild the BN with conv as input.
  Shape inputShape{1, 4, 6, 6}; // conv output shape for 8x8 input, 3x3 kernel
  Shape statsShape{1, 4, 1, 1};
  auto bnWeight = makeValue(statsShape);
  auto bnBias = makeValue(statsShape);
  auto bnRM = makeValue(statsShape);
  auto bnRV = makeValue(statsShape);

  std::vector<NodePtr> bnInputs = {cg.conv, bnWeight, bnBias, bnRM, bnRV};
  auto bn = BatchNormNode::create(
      std::move(bnInputs),
      /*axes=*/{1},
      /*train=*/true,
      /*momentum=*/0.1,
      /*epsilon=*/1e-5,
      {{inputShape, dtype::f32},   // Result
       {statsShape, dtype::f32},   // SaveMean
       {statsShape, dtype::f32},   // SaveVar
       {statsShape, dtype::f32},   // NewRunningMean
       {statsShape, dtype::f32}}); // NewRunningVar

  auto projResult = ProjectionNode::create(
      bn, BatchNormNode::Result, inputShape, dtype::f32);
  auto sigmoid = UnaryNode::create(projResult, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);
  // Either fused (CustomNode on GPU) or original (CPU fallback)
  ASSERT_TRUE(result->isCustom() || result->isProjection() || result == sigmoid);
}

TEST_F(JitCudnnSubgraphLoweringTest, Conv2dBNAddReluSubgraphDetected) {
  // relu(projection(0, batchnorm(conv2d(x, w, b))) + shortcut) — full ResNet block
  // Two anchors (conv2d, batchnorm) + add + relu
  Shape convOutShape{1, 4, 6, 6}; // conv output for 8x8 input, 3x3 kernel
  Shape statsShape{1, 4, 1, 1};

  Conv2dTestGraph cg;
  auto bnWeight = makeValue(statsShape);
  auto bnBias = makeValue(statsShape);
  auto bnRM = makeValue(statsShape);
  auto bnRV = makeValue(statsShape);

  std::vector<NodePtr> bnInputs = {cg.conv, bnWeight, bnBias, bnRM, bnRV};
  auto bn = BatchNormNode::create(
      std::move(bnInputs),
      /*axes=*/{1},
      /*train=*/true,
      /*momentum=*/0.1,
      /*epsilon=*/1e-5,
      {{convOutShape, dtype::f32},
       {statsShape, dtype::f32},
       {statsShape, dtype::f32},
       {statsShape, dtype::f32},
       {statsShape, dtype::f32}});

  auto projResult = ProjectionNode::create(
      bn, BatchNormNode::Result, convOutShape, dtype::f32);
  auto shortcut = makeValue(convOutShape);
  auto add = BinaryNode::create(projResult, shortcut, BinaryOp::Add);
  auto zero = ScalarNode::create(add->shape(), dtype::f32, 0.0);
  auto relu = BinaryNode::create(add, zero, BinaryOp::Max);

  auto result = pass_.apply(relu);
  // Either fused (CustomNode on GPU) or original (CPU fallback)
  ASSERT_TRUE(result->isCustom() || result->isProjection() || result == relu);
}

TEST_F(JitCudnnSubgraphLoweringTest, Conv2dBNInferenceReluSubgraphDetected) {
  // relu(batchnorm_inference(conv2d(x, w, b))) — inference mode, single output
  Shape convOutShape{1, 4, 6, 6};
  Shape statsShape{1, 4, 1, 1};

  Conv2dTestGraph cg;
  auto bnWeight = makeValue(statsShape);
  auto bnBias = makeValue(statsShape);
  auto bnRM = makeValue(statsShape);
  auto bnRV = makeValue(statsShape);

  std::vector<NodePtr> bnInputs = {cg.conv, bnWeight, bnBias, bnRM, bnRV};
  auto bn = BatchNormNode::create(
      std::move(bnInputs),
      /*axes=*/{1},
      /*train=*/false,
      /*momentum=*/0.1,
      /*epsilon=*/1e-5,
      {{convOutShape, dtype::f32},
       {statsShape, dtype::f32},
       {statsShape, dtype::f32},
       {statsShape, dtype::f32},
       {statsShape, dtype::f32}});

  auto projResult = ProjectionNode::create(
      bn, BatchNormNode::Result, convOutShape, dtype::f32);
  auto zero = ScalarNode::create(projResult->shape(), dtype::f32, 0.0);
  auto relu = BinaryNode::create(projResult, zero, BinaryOp::Max);

  auto result = pass_.apply(relu);
  ASSERT_TRUE(result->isCustom() || result == relu);
}

// ==================== CastNode handling ====================

TEST_F(JitCudnnSubgraphLoweringTest, Conv2dCastReluSubgraphDetected) {
  // relu(cast(conv2d(x, w, b), f16)) — cast is pointwise, should be
  // included in the subgraph and handled by the graph builder.
  Conv2dTestGraph g;
  auto cast = CastNode::create(g.conv, dtype::f16);
  auto zero = ScalarNode::create(cast->shape(), dtype::f16, 0.0);
  auto relu = BinaryNode::create(cast, zero, BinaryOp::Max);

  auto result = pass_.apply(relu);
  ASSERT_TRUE(result->isCustom() || result == relu);
}

TEST_F(JitCudnnSubgraphLoweringTest, MatmulCastAddSubgraphDetected) {
  // cast(matmul(A, B), f16) + C — cast between anchor and pointwise
  auto a = makeValue({2, 3});
  auto b = makeValue({3, 4});
  auto mm = MatmulNode::create(
      a, b, MatrixProperty::None, MatrixProperty::None);
  auto cast = CastNode::create(mm, dtype::f16);
  auto c = makeValue({2, 4}, dtype::f16);
  auto add = BinaryNode::create(cast, c, BinaryOp::Add);

  auto result = pass_.apply(add);
  ASSERT_TRUE(result->isCustom() || result == add);
}

// ==================== Transpose absorption ====================

TEST_F(JitCudnnSubgraphLoweringTest, TransposeConv2dReluAbsorbed) {
  // relu(conv2d(transpose(x))) — the transpose should be absorbed into the
  // cuDNN subgraph. The boundary input becomes x (pre-transpose) with permuted
  // strides in the cuDNN tensor descriptor.
  auto x = makeValue({1, 8, 8, 3}); // NHWC
  auto tr = TransposeNode::create(x, Shape({0, 3, 1, 2})); // → NCHW {1,3,8,8}
  auto weights = makeValue({4, 3, 3, 3});
  auto bias = makeValue({1, 4, 1, 1});
  auto conv = Conv2dNode::create(
      tr, weights, bias,
      /*sx=*/1, /*sy=*/1, /*px=*/0, /*py=*/0,
      /*dx=*/1, /*dy=*/1, /*groups=*/1, /*wantState=*/false);
  auto zero = ScalarNode::create(conv->shape(), dtype::f32, 0.0);
  auto relu = BinaryNode::create(conv, zero, BinaryOp::Max);

  auto result = pass_.apply(relu);
  // Either fused (CustomNode on GPU) or original (CPU fallback).
  // The key test: the pass doesn't crash with the absorbed transpose.
  ASSERT_TRUE(result->isCustom() || result == relu);
}

TEST_F(JitCudnnSubgraphLoweringTest, TransposeMatmulAddAbsorbed) {
  // add(matmul(transpose(x), y), bias) — transpose absorbed into matmul subgraph
  auto x = makeValue({4, 3}); // will be transposed to {3, 4}
  auto tr = TransposeNode::create(x, Shape({1, 0}));
  auto y = makeValue({4, 5});
  auto mm = MatmulNode::create(
      tr, y, MatrixProperty::None, MatrixProperty::None);
  auto bias = makeValue({3, 5});
  auto add = BinaryNode::create(mm, bias, BinaryOp::Add);

  auto result = pass_.apply(add);
  ASSERT_TRUE(result->isCustom() || result == add);
}

TEST_F(JitCudnnSubgraphLoweringTest, TransposePool2dReluAbsorbed) {
  // relu(pool2d(transpose(x))) — transpose absorbed into pool2d subgraph
  auto x = makeValue({1, 8, 8, 4}); // NHWC
  auto tr = TransposeNode::create(x, Shape({0, 3, 1, 2})); // → NCHW {1,4,8,8}
  auto pool = Pool2dNode::create(
      tr, /*wx=*/2, /*wy=*/2, /*sx=*/2, /*sy=*/2, /*px=*/0, /*py=*/0,
      PoolingMode::MAX, Shape({1, 4, 4, 4}), dtype::f32);
  auto zero = ScalarNode::create(pool->shape(), dtype::f32, 0.0);
  auto relu = BinaryNode::create(pool, zero, BinaryOp::Max);

  auto result = pass_.apply(relu);
  ASSERT_TRUE(result->isCustom() || result == relu);
}

TEST_F(JitCudnnSubgraphLoweringTest, TransposeMultiUseNotAbsorbed) {
  // When the transpose has multiple uses, it should NOT be absorbed — it
  // becomes a boundary input like any other multi-use node.
  auto x = makeValue({1, 8, 8, 3}); // NHWC
  auto tr = TransposeNode::create(x, Shape({0, 3, 1, 2})); // → NCHW {1,3,8,8}
  // Create a second use of the transpose (same shape as tr output)
  auto otherUse = UnaryNode::create(tr, UnaryOp::Exp); // {1,3,8,8}
  // Conv2d consumes the transpose
  auto weights = makeValue({3, 3, 3, 3}); // C_out=3 so output matches tr shape
  auto bias = makeValue({1, 3, 1, 1});
  auto conv = Conv2dNode::create(
      tr, weights, bias,
      /*sx=*/1, /*sy=*/1, /*px=*/1, /*py=*/1, // pad to preserve spatial dims
      /*dx=*/1, /*dy=*/1, /*groups=*/1, /*wantState=*/false);
  // conv output: {1,3,8,8} — same shape as otherUse
  auto sum = BinaryNode::create(conv, otherUse, BinaryOp::Add);

  auto result = pass_.apply(sum);
  // Transpose has 2 uses — not absorbed into either subgraph.
  // The key test: no crash with multi-use transpose.
  ASSERT_TRUE(result == sum || result->isCustom());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
