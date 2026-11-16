/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/ir/NodeUtils.h"
#include "flashlight/fl/tensor/ir/BatchNormNode.h"
#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/Conv2dNode.h"
#include "flashlight/fl/tensor/ir/CustomNode.h"
#include "flashlight/fl/tensor/ir/MatmulNode.h"
#include "flashlight/fl/tensor/ir/Pool2dNode.h"
#include "flashlight/fl/tensor/ir/ProjectionNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"
#include "flashlight/fl/tensor/backend/cpu/CPUBackend.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/eval/Evaluator.h"
#include "flashlight/fl/tensor/BackendDispatch.h"
#include "flashlight/fl/tensor/opt/backends/onednn/OneDnnPrimitivePostOpFusion.h"

using namespace fl;

namespace {

// Helper: create a ValueNode wrapping a bare CPUTensor.
NodePtr makeValue(const Shape& shape, dtype type = dtype::f32) {
  Tensor bare = toTensor<CPUTensor>(shape, type, nullptr, Location::Host);
  return ValueNode::create(std::move(bare));
}

// Helper: create a Conv2dNode with dummy inputs.
// input: (N=1, C=3, H=8, W=8), weights: (outC=4, inC=3, kH=3, kW=3)
// bias: (1, outC=4, 1, 1), output: (1, 4, 6, 6)
struct Conv2dTestGraph {
  NodePtr input;
  NodePtr weights;
  NodePtr bias;
  Conv2dNodePtr conv;

  Conv2dTestGraph() {
    input = makeValue({1, 3, 8, 8});
    weights = makeValue({4, 3, 3, 3});
    bias = makeValue({1, 4, 1, 1});
    conv = Conv2dNode::create(
        input, weights, bias,
        /*sx=*/1, /*sy=*/1, /*px=*/0, /*py=*/0,
        /*dx=*/1, /*dy=*/1, /*groups=*/1, /*wantState=*/true);
  }
};

// Helper: create a Pool2dNode with dummy input.
// input: (N=1, C=3, H=8, W=8), pool 2x2 stride 2 → output (1, 3, 4, 4)
struct Pool2dTestGraph {
  NodePtr input;
  Pool2dNodePtr pool;

  Pool2dTestGraph() {
    input = makeValue({1, 3, 8, 8});
    pool = Pool2dNode::create(
        input,
        /*wx=*/2, /*wy=*/2, /*sx=*/2, /*sy=*/2, /*px=*/0, /*py=*/0,
        PoolingMode::MAX,
        Shape({1, 3, 4, 4}), dtype::f32);
  }
};

// Helper: create a BatchNormNode with dummy inputs and a ProjectionNode(Result).
// input: (N=2, C=4, H=4, W=4), weight/bias: (1, 4, 1, 1),
// runningMean/runningVar: (1, 4, 1, 1), output: same as input
struct BatchNormTestGraph {
  NodePtr input;
  NodePtr weight;
  NodePtr bias;
  NodePtr runningMean;
  NodePtr runningVar;
  BatchNormNodePtr bn;
  ProjectionNodePtr resultProj; // Projection(BN, Result=0)

  explicit BatchNormTestGraph(bool train = false) {
    input = makeValue({2, 4, 4, 4});
    weight = makeValue({1, 4, 1, 1});
    bias = makeValue({1, 4, 1, 1});
    runningMean = makeValue({1, 4, 1, 1});
    runningVar = makeValue({1, 4, 1, 1});

    Shape outShape({2, 4, 4, 4});
    Shape perChannelShape({1, 4, 1, 1});
    std::vector<OutputDesc> descs = {
        {outShape, dtype::f32},         // Result
        {perChannelShape, dtype::f32},  // SaveMean
        {perChannelShape, dtype::f32},  // SaveVar
        {perChannelShape, dtype::f32},  // NewRunningMean
        {perChannelShape, dtype::f32},  // NewRunningVar
    };

    bn = BatchNormNode::create(
        {input, weight, bias, runningMean, runningVar},
        /*axes=*/{1}, train,
        /*momentum=*/0.1, /*epsilon=*/1e-5,
        std::move(descs));

    resultProj = ProjectionNode::create(
        bn, BatchNormNode::Result, outShape, dtype::f32);
  }
};

} // namespace

class JitOneDnnPrimitivePostOpFusionTest : public ::testing::Test {
 protected:
  OneDnnPrimitivePostOpFusion pass_;
};

// ==================== Conv2d structural tests ====================

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dSingleEltwise) {
  // sigmoid(conv2d(...)) → fused CustomNode with 1 eltwise post-op
  Conv2dTestGraph g;
  auto sigmoid = UnaryNode::create(g.conv, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);

  ASSERT_NE(result, sigmoid);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs(), NodeList({g.input, g.weights, g.bias}));
  ASSERT_EQ(result->shape(), sigmoid->shape());
  ASSERT_EQ(result->dataType(), sigmoid->dataType());
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dReluViaMax) {
  // max(conv2d(...), Scalar(0)) → fused CustomNode with relu post-op
  Conv2dTestGraph g;
  auto zero = ScalarNode::create(g.conv->shape(), dtype::f32, 0.0);
  auto maxNode = BinaryNode::create(g.conv, zero, BinaryOp::Max);

  auto result = pass_.apply(maxNode);

  ASSERT_NE(result, maxNode);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs(), NodeList({g.input, g.weights, g.bias}));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dReluViaMaxSwapped) {
  // max(Scalar(0), conv2d(...)) → same fusion (scalar on LHS)
  Conv2dTestGraph g;
  auto zero = ScalarNode::create(g.conv->shape(), dtype::f32, 0.0);
  auto maxNode = BinaryNode::create(zero, g.conv, BinaryOp::Max);

  auto result = pass_.apply(maxNode);

  ASSERT_NE(result, maxNode);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs(), NodeList({g.input, g.weights, g.bias}));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dEltwiseChain) {
  // sigmoid(tanh(conv)) → 1 CustomNode with 2 eltwise post-ops
  Conv2dTestGraph g;
  auto tanhNode = UnaryNode::create(g.conv, UnaryOp::Tanh);
  auto sigmoid = UnaryNode::create(tanhNode, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);

  ASSERT_NE(result, sigmoid);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs(), NodeList({g.input, g.weights, g.bias}));
  ASSERT_EQ(result->shape(), sigmoid->shape());
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dSumOnly) {
  // add(conv, x) → 1 CustomNode with sum-only post-op, 4 inputs
  Conv2dTestGraph g;
  auto other = makeValue({1, 4, 6, 6});
  auto add = BinaryNode::create(g.conv, other, BinaryOp::Add);

  auto result = pass_.apply(add);

  ASSERT_NE(result, add);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs().size(), 4u);
  ASSERT_EQ(result->inputs()[0], g.input);
  ASSERT_EQ(result->inputs()[1], g.weights);
  ASSERT_EQ(result->inputs()[2], g.bias);
  ASSERT_EQ(result->inputs()[3], other);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dSumEltwise) {
  // sigmoid(add(conv2d, other)) → fused CustomNode with sum + eltwise
  Conv2dTestGraph g;
  auto other = makeValue({1, 4, 6, 6});
  auto add = BinaryNode::create(g.conv, other, BinaryOp::Add);
  auto zero = ScalarNode::create(add->shape(), dtype::f32, 0.0);
  auto relu = BinaryNode::create(add, zero, BinaryOp::Max);

  auto result = pass_.apply(relu);

  ASSERT_NE(result, relu);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs().size(), 4u);
  ASSERT_EQ(result->inputs()[0], g.input);
  ASSERT_EQ(result->inputs()[1], g.weights);
  ASSERT_EQ(result->inputs()[2], g.bias);
  ASSERT_EQ(result->inputs()[3], other);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dSumEltwiseSwapped) {
  // sigmoid(add(other, conv2d)) → same fusion (add is commutative)
  Conv2dTestGraph g;
  auto other = makeValue({1, 4, 6, 6});
  auto add = BinaryNode::create(other, g.conv, BinaryOp::Add);
  auto sigmoid = UnaryNode::create(add, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);

  ASSERT_NE(result, sigmoid);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs()[0], g.input);
  ASSERT_EQ(result->inputs()[3], other);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dSumEltwiseChain) {
  // tanh(sigmoid(add(conv, x))) → 1 CustomNode with sum + 2 eltwise
  Conv2dTestGraph g;
  auto other = makeValue({1, 4, 6, 6});
  auto add = BinaryNode::create(g.conv, other, BinaryOp::Add);
  auto sigmoid = UnaryNode::create(add, UnaryOp::Sigmoid);
  auto tanhNode = UnaryNode::create(sigmoid, UnaryOp::Tanh);

  auto result = pass_.apply(tanhNode);

  ASSERT_NE(result, tanhNode);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs().size(), 4u);
  ASSERT_EQ(result->inputs()[0], g.input);
  ASSERT_EQ(result->inputs()[1], g.weights);
  ASSERT_EQ(result->inputs()[2], g.bias);
  ASSERT_EQ(result->inputs()[3], other);
}

// ==================== Pool2d structural tests ====================

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Pool2dSingleEltwise) {
  // tanh(pool2d(input)) → fused CustomNode
  Pool2dTestGraph g;
  auto tanhNode = UnaryNode::create(g.pool, UnaryOp::Tanh);

  auto result = pass_.apply(tanhNode);

  ASSERT_NE(result, tanhNode);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs(), NodeList({g.input}));
  ASSERT_EQ(result->shape(), tanhNode->shape());
  ASSERT_EQ(result->dataType(), tanhNode->dataType());
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Pool2dReluViaMax) {
  // max(pool2d(...), Scalar(0)) → fused CustomNode with relu post-op
  Pool2dTestGraph g;
  auto zero = ScalarNode::create(g.pool->shape(), dtype::f32, 0.0);
  auto maxNode = BinaryNode::create(g.pool, zero, BinaryOp::Max);

  auto result = pass_.apply(maxNode);

  ASSERT_NE(result, maxNode);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs(), NodeList({g.input}));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Pool2dEltwiseChain) {
  // sigmoid(tanh(pool)) → 1 CustomNode with 2 eltwise post-ops
  Pool2dTestGraph g;
  auto tanhNode = UnaryNode::create(g.pool, UnaryOp::Tanh);
  auto sigmoid = UnaryNode::create(tanhNode, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);

  ASSERT_NE(result, sigmoid);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs(), NodeList({g.input}));
}

// ==================== BatchNorm structural tests ====================

TEST_F(JitOneDnnPrimitivePostOpFusionTest, BNInferenceSingleEltwise) {
  // sigmoid(Projection(BN_inference, Result)) → fused CustomNode
  BatchNormTestGraph g(/*train=*/false);
  auto sigmoid = UnaryNode::create(g.resultProj, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);

  ASSERT_NE(result, sigmoid);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs().size(), 5u);
  ASSERT_EQ(result->inputs()[0], g.input);
  ASSERT_EQ(result->inputs()[1], g.weight);
  ASSERT_EQ(result->inputs()[2], g.bias);
  ASSERT_EQ(result->inputs()[3], g.runningMean);
  ASSERT_EQ(result->inputs()[4], g.runningVar);
  ASSERT_EQ(result->shape(), sigmoid->shape());
  ASSERT_EQ(result->dataType(), sigmoid->dataType());
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, BNInferenceReluViaMax) {
  // max(Projection(BN_inference, Result), Scalar(0)) → fused (relu)
  BatchNormTestGraph g(/*train=*/false);
  auto zero = ScalarNode::create(g.resultProj->shape(), dtype::f32, 0.0);
  auto maxNode = BinaryNode::create(g.resultProj, zero, BinaryOp::Max);

  auto result = pass_.apply(maxNode);

  ASSERT_NE(result, maxNode);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs().size(), 5u);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, BNInferenceEltwiseChain) {
  // tanh(sigmoid(proj(bn_inference))) → 1 CustomNode with 2 eltwise post-ops
  BatchNormTestGraph g(/*train=*/false);
  auto sigmoid = UnaryNode::create(g.resultProj, UnaryOp::Sigmoid);
  auto tanhNode = UnaryNode::create(sigmoid, UnaryOp::Tanh);

  auto result = pass_.apply(tanhNode);

  ASSERT_NE(result, tanhNode);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs().size(), 5u);
  ASSERT_EQ(result->inputs()[0], g.input);
}

// ==================== Matmul structural tests ====================

TEST_F(JitOneDnnPrimitivePostOpFusionTest, MatmulSingleEltwise) {
  // sigmoid(matmul(A, B)) → fused CustomNode with 1 eltwise post-op
  auto lhs = makeValue({4, 3});
  auto rhs = makeValue({3, 5});
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  auto sigmoid = UnaryNode::create(mm, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);

  ASSERT_NE(result, sigmoid);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs(), NodeList({lhs, rhs}));
  ASSERT_EQ(result->shape(), sigmoid->shape());
  ASSERT_EQ(result->dataType(), sigmoid->dataType());
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, MatmulReluViaMax) {
  // max(matmul(A, B), Scalar(0)) → fused CustomNode with relu post-op
  auto lhs = makeValue({4, 3});
  auto rhs = makeValue({3, 5});
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  auto zero = ScalarNode::create(mm->shape(), dtype::f32, 0.0);
  auto maxNode = BinaryNode::create(mm, zero, BinaryOp::Max);

  auto result = pass_.apply(maxNode);

  ASSERT_NE(result, maxNode);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs(), NodeList({lhs, rhs}));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, MatmulEltwiseChain) {
  // sigmoid(tanh(matmul(A, B))) → 1 CustomNode with 2 eltwise post-ops
  auto lhs = makeValue({4, 3});
  auto rhs = makeValue({3, 5});
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  auto tanhNode = UnaryNode::create(mm, UnaryOp::Tanh);
  auto sigmoid = UnaryNode::create(tanhNode, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);

  ASSERT_NE(result, sigmoid);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs(), NodeList({lhs, rhs}));
  ASSERT_EQ(result->shape(), sigmoid->shape());
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, MatmulBiasOnly) {
  // add(matmul(A, B), bias) → fused CustomNode with native bias, 3 inputs
  auto lhs = makeValue({4, 3});
  auto rhs = makeValue({3, 5});
  auto bias = makeValue({1, 5});
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  auto add = BinaryNode::create(mm, bias, BinaryOp::Add);

  auto result = pass_.apply(add);

  ASSERT_NE(result, add);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs().size(), 3u);
  ASSERT_EQ(result->inputs()[0], lhs);
  ASSERT_EQ(result->inputs()[1], rhs);
  ASSERT_EQ(result->inputs()[2], bias);
  ASSERT_EQ(result->shape(), add->shape());
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, MatmulBiasSwapped) {
  // add(bias, matmul(A, B)) → same fusion (add is commutative)
  auto lhs = makeValue({4, 3});
  auto rhs = makeValue({3, 5});
  auto bias = makeValue({5});
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  auto add = BinaryNode::create(bias, mm, BinaryOp::Add);

  auto result = pass_.apply(add);

  ASSERT_NE(result, add);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs().size(), 3u);
  ASSERT_EQ(result->inputs()[0], lhs);
  ASSERT_EQ(result->inputs()[1], rhs);
  ASSERT_EQ(result->inputs()[2], bias);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, MatmulBiasEltwise) {
  // relu(add(matmul(A, B), bias)) → fused with native bias + eltwise
  auto lhs = makeValue({4, 3});
  auto rhs = makeValue({3, 5});
  auto bias = makeValue({1, 5});
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  auto add = BinaryNode::create(mm, bias, BinaryOp::Add);
  auto zero = ScalarNode::create(add->shape(), dtype::f32, 0.0);
  auto relu = BinaryNode::create(add, zero, BinaryOp::Max);

  auto result = pass_.apply(relu);

  ASSERT_NE(result, relu);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs().size(), 3u);
  ASSERT_EQ(result->inputs()[0], lhs);
  ASSERT_EQ(result->inputs()[1], rhs);
  ASSERT_EQ(result->inputs()[2], bias);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, MatmulBiasEltwiseChain) {
  // sigmoid(relu(add(matmul(A, B), bias))) → native bias + 2 eltwise post-ops
  auto lhs = makeValue({4, 3});
  auto rhs = makeValue({3, 5});
  auto bias = makeValue({5});
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  auto add = BinaryNode::create(mm, bias, BinaryOp::Add);
  auto zero = ScalarNode::create(add->shape(), dtype::f32, 0.0);
  auto relu = BinaryNode::create(add, zero, BinaryOp::Max);
  auto sigmoid = UnaryNode::create(relu, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);

  ASSERT_NE(result, sigmoid);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->inputs().size(), 3u);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, MatmulBiasNoFusionBroadcast) {
  // add(matmul({4,3}x{3,5}->{4,5}), other({4,5})) — other has same shape as
  // output (not a valid native bias shape) → no matmul+bias fusion.
  auto lhs = makeValue({4, 3});
  auto rhs = makeValue({3, 5});
  auto other = makeValue({4, 5}); // full shape, not bias-compatible
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  auto add = BinaryNode::create(mm, other, BinaryOp::Add);
  auto sigmoid = UnaryNode::create(add, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);
  ASSERT_EQ(result, sigmoid); // no fusion
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, MatmulBiasNoFusionMultiUseMatmul) {
  // matmul has multiple uses → no bias fusion
  auto lhs = makeValue({4, 3});
  auto rhs = makeValue({3, 5});
  auto bias = makeValue({1, 5});
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  auto extra = UnaryNode::create(mm, UnaryOp::Negative); // 2nd use
  auto add = BinaryNode::create(mm, bias, BinaryOp::Add);

  auto result = pass_.apply(add);
  ASSERT_EQ(result, add);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, MatmulNoFusionMultiUse) {
  // matmul has two consumers → no fusion
  auto lhs = makeValue({4, 3});
  auto rhs = makeValue({3, 5});
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  auto sigmoid = UnaryNode::create(mm, UnaryOp::Sigmoid);
  auto tanh = UnaryNode::create(mm, UnaryOp::Tanh);

  auto result = pass_.apply(sigmoid);
  ASSERT_EQ(result, sigmoid);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, MatmulNoFusionF64) {
  // f64 matmul + sigmoid → no fusion (OneDNN doesn't support f64 post-ops)
  auto lhs = makeValue({4, 3}, dtype::f64);
  auto rhs = makeValue({3, 5}, dtype::f64);
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  auto sigmoid = UnaryNode::create(mm, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);
  ASSERT_EQ(result, sigmoid);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, PlainMatmulNoChange) {
  // Plain matmul without activation → pass doesn't touch it
  auto lhs = makeValue({4, 3});
  auto rhs = makeValue({3, 5});
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);

  auto result = pass_.apply(mm);
  ASSERT_EQ(result, mm);
}

// ==================== No-fusion structural tests ====================

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dNoFusionMultiUse) {
  // conv2d has two consumers → no fusion
  Conv2dTestGraph g;
  auto sigmoid = UnaryNode::create(g.conv, UnaryOp::Sigmoid);
  auto tanh = UnaryNode::create(g.conv, UnaryOp::Tanh);

  auto result = pass_.apply(sigmoid);
  ASSERT_EQ(result, sigmoid);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dNoFusionNonEltwise) {
  // sin(conv2d) — sin has no OneDNN eltwise mapping → no fusion
  Conv2dTestGraph g;
  auto sinNode = UnaryNode::create(g.conv, UnaryOp::Sin);

  auto result = pass_.apply(sinNode);
  ASSERT_EQ(result, sinNode);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dNoFusionNonZeroMax) {
  // max(conv2d, Scalar(1)) — not a relu pattern (non-zero scalar) → no fusion
  Conv2dTestGraph g;
  auto one = ScalarNode::create(g.conv->shape(), dtype::f32, 1.0);
  auto maxNode = BinaryNode::create(g.conv, one, BinaryOp::Max);

  auto result = pass_.apply(maxNode);
  ASSERT_EQ(result, maxNode);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dNoFusionF64) {
  // f64 conv2d + sigmoid → no fusion (OneDNN doesn't support f64 post-ops)
  auto input = makeValue({1, 3, 8, 8}, dtype::f64);
  auto weights = makeValue({4, 3, 3, 3}, dtype::f64);
  auto bias = makeValue({1, 4, 1, 1}, dtype::f64);
  auto conv = Conv2dNode::create(
      input, weights, bias,
      1, 1, 0, 0, 1, 1, 1, true);
  auto sigmoid = UnaryNode::create(conv, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);
  ASSERT_EQ(result, sigmoid);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Pool2dNoFusionMultiUse) {
  // pool2d has two consumers → no fusion
  Pool2dTestGraph g;
  auto sigmoid = UnaryNode::create(g.pool, UnaryOp::Sigmoid);
  auto tanh = UnaryNode::create(g.pool, UnaryOp::Tanh);

  auto result = pass_.apply(sigmoid);
  ASSERT_EQ(result, sigmoid);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Pool2dNoFusionF64) {
  // f64 pool2d + tanh → no fusion
  auto input = makeValue({1, 3, 8, 8}, dtype::f64);
  auto pool = Pool2dNode::create(
      input,
      2, 2, 2, 2, 0, 0,
      PoolingMode::MAX,
      Shape({1, 3, 4, 4}), dtype::f64);
  auto tanhNode = UnaryNode::create(pool, UnaryOp::Tanh);

  auto result = pass_.apply(tanhNode);
  ASSERT_EQ(result, tanhNode);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, ActivationOfBinaryNoFusion) {
  // sigmoid(a + b) — inner node is a binary add, not conv2d/pool2d → no fusion
  auto a = makeValue({4});
  auto b = makeValue({4});
  auto add = BinaryNode::create(a, b, BinaryOp::Add);
  auto sigmoid = UnaryNode::create(add, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);
  ASSERT_EQ(result, sigmoid);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, PlainConv2dNoChange) {
  // Conv2d without an activation wrapper → pass doesn't touch it
  Conv2dTestGraph g;

  auto result = pass_.apply(g.conv);
  ASSERT_EQ(result, g.conv);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, PlainPool2dNoChange) {
  Pool2dTestGraph g;

  auto result = pass_.apply(g.pool);
  ASSERT_EQ(result, g.pool);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, MinNotMatchedAsRelu) {
  // min(conv2d, Scalar(0)) — min is NOT relu → no fusion
  Conv2dTestGraph g;
  auto zero = ScalarNode::create(g.conv->shape(), dtype::f32, 0.0);
  auto minNode = BinaryNode::create(g.conv, zero, BinaryOp::Min);

  auto result = pass_.apply(minNode);
  ASSERT_EQ(result, minNode);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, BNTrainingNoFusion) {
  // sigmoid(Projection(BN_training, Result)) → NO fusion
  BatchNormTestGraph g(/*train=*/true);
  auto sigmoid = UnaryNode::create(g.resultProj, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);
  ASSERT_EQ(result, sigmoid);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, BNInferenceMultiUseNoFusion) {
  // Projection(BN, Result) has two uses → no fusion
  BatchNormTestGraph g(/*train=*/false);
  auto sigmoid = UnaryNode::create(g.resultProj, UnaryOp::Sigmoid);
  auto tanh = UnaryNode::create(g.resultProj, UnaryOp::Tanh);

  auto result = pass_.apply(sigmoid);
  ASSERT_EQ(result, sigmoid);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, BNInferenceNonResultProjNoFusion) {
  // sigmoid(Projection(BN, SaveMean=1)) → no fusion (not Result output)
  BatchNormTestGraph g(/*train=*/false);
  auto saveMeanProj = ProjectionNode::create(
      g.bn, BatchNormNode::SaveMean, Shape({1, 4, 1, 1}), dtype::f32);
  auto sigmoid = UnaryNode::create(saveMeanProj, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);
  ASSERT_EQ(result, sigmoid);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, BNInferenceF64NoFusion) {
  // f64 BN + sigmoid → no fusion
  auto input = makeValue({2, 4, 4, 4}, dtype::f64);
  auto weight = makeValue({1, 4, 1, 1}, dtype::f64);
  auto bias = makeValue({1, 4, 1, 1}, dtype::f64);
  auto runningMean = makeValue({1, 4, 1, 1}, dtype::f64);
  auto runningVar = makeValue({1, 4, 1, 1}, dtype::f64);

  Shape outShape({2, 4, 4, 4});
  Shape pcShape({1, 4, 1, 1});
  auto bn = BatchNormNode::create(
      {input, weight, bias, runningMean, runningVar},
      {1}, false, 0.1, 1e-5,
      {{outShape, dtype::f64}, {pcShape, dtype::f64}, {pcShape, dtype::f64},
       {pcShape, dtype::f64}, {pcShape, dtype::f64}});
  auto proj = ProjectionNode::create(bn, BatchNormNode::Result, outShape, dtype::f64);
  auto sigmoid = UnaryNode::create(proj, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);
  ASSERT_EQ(result, sigmoid);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dSumNoFusionBroadcast) {
  // add(conv2d({1,4,6,6}), other({1,1,6,6})) — sum source has different shape
  // (broadcast). OneDNN sum post-op requires exact shape match → no fusion.
  Conv2dTestGraph g;
  auto other = makeValue({1, 1, 6, 6}); // broadcasts with conv output {1,4,6,6}
  auto add = BinaryNode::create(g.conv, other, BinaryOp::Add);
  auto sigmoid = UnaryNode::create(add, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);
  ASSERT_EQ(result, sigmoid); // no fusion
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dSumOnlyNoFusionBroadcast) {
  // add(conv2d({1,4,6,6}), other({4,1,1})) — sum-only path, broadcast → no fusion.
  Conv2dTestGraph g;
  auto other = makeValue({4, 1, 1}); // broadcasts with conv output
  auto add = BinaryNode::create(g.conv, other, BinaryOp::Add);

  auto result = pass_.apply(add);
  ASSERT_EQ(result, add); // no fusion
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dSumNoFusionMultiUseConv) {
  // conv2d has multiple uses → no sum fusion
  Conv2dTestGraph g;
  auto extra = UnaryNode::create(g.conv, UnaryOp::Negative); // 2nd use
  auto other = makeValue({1, 4, 6, 6});
  auto add = BinaryNode::create(g.conv, other, BinaryOp::Add);
  auto sigmoid = UnaryNode::create(add, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);
  ASSERT_EQ(result, sigmoid);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dSumNoFusionMultiUseAdd) {
  // add has multiple uses → no fusion (inner single-use gate)
  Conv2dTestGraph g;
  auto other = makeValue({1, 4, 6, 6});
  auto add = BinaryNode::create(g.conv, other, BinaryOp::Add);
  auto extra = UnaryNode::create(add, UnaryOp::Negative); // 2nd use of add
  auto sigmoid = UnaryNode::create(add, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);
  ASSERT_EQ(result, sigmoid);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dSumNoFusionNonAdd) {
  // sub(conv2d, other) → not an add, no fusion
  Conv2dTestGraph g;
  auto other = makeValue({1, 4, 6, 6});
  auto sub = BinaryNode::create(g.conv, other, BinaryOp::Sub);
  auto sigmoid = UnaryNode::create(sub, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);
  ASSERT_EQ(result, sigmoid);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, ChainBreaksAtNonFusableUnary) {
  // sin(sigmoid(conv)) — sin has no OneDNN eltwise mapping.
  // Chain walk starts at sin but doesn't match → returns sin unchanged.
  // Inner sigmoid(conv) fuses when TopDownPass visits it later.
  Conv2dTestGraph g;
  auto sigmoid = UnaryNode::create(g.conv, UnaryOp::Sigmoid);
  auto sinNode = UnaryNode::create(sigmoid, UnaryOp::Sin);

  auto result = pass_.apply(sinNode);
  ASSERT_EQ(result, sinNode);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, ChainBreaksAtMultiUseIntermediate) {
  // tanh(sigmoid(conv)) where sigmoid has 2 consumers.
  // Chain walk: tanh matches eltwise, tries to descend into sigmoid but
  // sigmoid has multi-use → chain stops. tanh alone can't reach conv2d.
  Conv2dTestGraph g;
  auto sigmoid = UnaryNode::create(g.conv, UnaryOp::Sigmoid);
  auto extra = UnaryNode::create(sigmoid, UnaryOp::Negative); // 2nd use
  auto tanhNode = UnaryNode::create(sigmoid, UnaryOp::Tanh);

  auto result = pass_.apply(tanhNode);
  ASSERT_EQ(result, tanhNode);
}

// ==================== Numerical correctness ====================

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dReluNumerical) {
  // relu(conv2d) fused vs unfused
  auto inputData = std::vector<float>(1 * 3 * 8 * 8, 1.0f);
  auto weightsData = std::vector<float>(4 * 3 * 3 * 3, 0.1f);
  auto biasData = std::vector<float>(4, -5.0f);

  Shape inputShape({1, 3, 8, 8});
  Shape weightsShape({4, 3, 3, 3});
  Shape biasShape({4});

  auto inputNode = ValueNode::create(
      toTensor<CPUTensor>(inputShape, dtype::f32, inputData.data(), Location::Host));
  auto weightsNode = ValueNode::create(
      toTensor<CPUTensor>(weightsShape, dtype::f32, weightsData.data(), Location::Host));
  auto biasNode = ValueNode::create(
      toTensor<CPUTensor>(biasShape, dtype::f32, biasData.data(), Location::Host));

  auto conv = Conv2dNode::create(
      inputNode, weightsNode, biasNode,
      1, 1, 0, 0, 1, 1, 1, false);
  auto zero = ScalarNode::create(conv->shape(), dtype::f32, 0.0);
  auto relu = BinaryNode::create(conv, zero, BinaryOp::Max);

  auto fused = pass_.apply(relu);
  ASSERT_TRUE(fused->isCustom());

  Evaluator evaluator(makeBackendDispatch(CPUBackend::getInstance()));
  evaluator.eval(fused);
  const auto& fusedResult = fused->getResult().value();

  auto conv2 = Conv2dNode::create(
      inputNode, weightsNode, biasNode,
      1, 1, 0, 0, 1, 1, 1, false);
  evaluator.eval(conv2);
  auto reluRef = fl::maximum(conv2->getResult().value(), 0.0f);

  ASSERT_EQ(fusedResult.shape(), reluRef.shape());
  ASSERT_TRUE(fl::allClose(fusedResult, reluRef, 1e-5));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dEltwiseChainNumerical) {
  // sigmoid(relu(conv2d)) fused vs unfused — verifies chain ordering
  auto inputData = std::vector<float>(1 * 3 * 8 * 8, 1.0f);
  auto weightsData = std::vector<float>(4 * 3 * 3 * 3, 0.1f);
  auto biasData = std::vector<float>(4, -5.0f);

  Shape inputShape({1, 3, 8, 8});
  Shape weightsShape({4, 3, 3, 3});
  Shape biasShape({4});

  auto inputNode = ValueNode::create(
      toTensor<CPUTensor>(inputShape, dtype::f32, inputData.data(), Location::Host));
  auto weightsNode = ValueNode::create(
      toTensor<CPUTensor>(weightsShape, dtype::f32, weightsData.data(), Location::Host));
  auto biasNode = ValueNode::create(
      toTensor<CPUTensor>(biasShape, dtype::f32, biasData.data(), Location::Host));

  auto conv = Conv2dNode::create(
      inputNode, weightsNode, biasNode,
      1, 1, 0, 0, 1, 1, 1, false);
  auto zero = ScalarNode::create(conv->shape(), dtype::f32, 0.0);
  auto relu = BinaryNode::create(conv, zero, BinaryOp::Max);
  auto sigmoid = UnaryNode::create(relu, UnaryOp::Sigmoid);

  auto fused = pass_.apply(sigmoid);
  ASSERT_TRUE(fused->isCustom());

  Evaluator evaluator(makeBackendDispatch(CPUBackend::getInstance()));
  evaluator.eval(fused);
  const auto& fusedResult = fused->getResult().value();

  auto conv2 = Conv2dNode::create(
      inputNode, weightsNode, biasNode,
      1, 1, 0, 0, 1, 1, 1, false);
  evaluator.eval(conv2);
  auto reluRef = fl::maximum(conv2->getResult().value(), 0.0f);
  auto sigmoidRef = fl::sigmoid(reluRef);

  ASSERT_EQ(fusedResult.shape(), sigmoidRef.shape());
  ASSERT_TRUE(fl::allClose(fusedResult, sigmoidRef, 1e-5));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Pool2dTanhNumerical) {
  // tanh(pool2d) fused vs unfused
  auto inputData = std::vector<float>(1 * 3 * 8 * 8);
  for (size_t i = 0; i < inputData.size(); i++) {
    inputData[i] = static_cast<float>(i) * 0.01f;
  }

  Shape inputShape({1, 3, 8, 8});
  auto inputNode = ValueNode::create(
      toTensor<CPUTensor>(inputShape, dtype::f32, inputData.data(), Location::Host));

  auto pool = Pool2dNode::create(
      inputNode,
      2, 2, 2, 2, 0, 0,
      PoolingMode::MAX,
      Shape({1, 3, 4, 4}), dtype::f32);
  auto tanhNode = UnaryNode::create(pool, UnaryOp::Tanh);

  auto fused = pass_.apply(tanhNode);
  ASSERT_TRUE(fused->isCustom());

  Evaluator evaluator(makeBackendDispatch(CPUBackend::getInstance()));
  evaluator.eval(fused);
  const auto& fusedResult = fused->getResult().value();

  auto pool2 = Pool2dNode::create(
      inputNode,
      2, 2, 2, 2, 0, 0,
      PoolingMode::MAX,
      Shape({1, 3, 4, 4}), dtype::f32);
  evaluator.eval(pool2);
  auto tanhRef = fl::tanh(pool2->getResult().value());

  ASSERT_EQ(fusedResult.shape(), tanhRef.shape());
  ASSERT_TRUE(fl::allClose(fusedResult, tanhRef, 1e-5));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, BNInferenceReluNumerical) {
  // relu(projection(BN_inference, Result)) fused vs unfused
  const int N = 2, C = 4, H = 4, W = 4;
  const int total = N * C * H * W;
  const int cSize = 1 * C * 1 * 1;

  auto inputData = std::vector<float>(total);
  for (int i = 0; i < total; i++) {
    inputData[i] = static_cast<float>(i % 7) - 3.0f;
  }
  auto weightData = std::vector<float>(cSize, 1.0f);
  auto biasData = std::vector<float>(cSize, 0.0f);
  auto meanData = std::vector<float>(cSize, 0.0f);
  auto varData = std::vector<float>(cSize, 1.0f);

  Shape inShape({N, C, H, W});
  Shape pcShape({1, C, 1, 1});

  auto inputNode = ValueNode::create(
      toTensor<CPUTensor>(inShape, dtype::f32, inputData.data(), Location::Host));
  auto weightNode = ValueNode::create(
      toTensor<CPUTensor>(pcShape, dtype::f32, weightData.data(), Location::Host));
  auto biasNode = ValueNode::create(
      toTensor<CPUTensor>(pcShape, dtype::f32, biasData.data(), Location::Host));
  auto meanNode = ValueNode::create(
      toTensor<CPUTensor>(pcShape, dtype::f32, meanData.data(), Location::Host));
  auto varNode = ValueNode::create(
      toTensor<CPUTensor>(pcShape, dtype::f32, varData.data(), Location::Host));

  std::vector<OutputDesc> descs = {
      {inShape, dtype::f32}, {pcShape, dtype::f32}, {pcShape, dtype::f32},
      {pcShape, dtype::f32}, {pcShape, dtype::f32}};

  auto bn = BatchNormNode::create(
      {inputNode, weightNode, biasNode, meanNode, varNode},
      {1}, /*train=*/false, /*momentum=*/0.0, /*epsilon=*/1e-5, descs);
  auto proj = ProjectionNode::create(bn, BatchNormNode::Result, inShape, dtype::f32);
  auto zero = ScalarNode::create(inShape, dtype::f32, 0.0);
  auto relu = BinaryNode::create(proj, zero, BinaryOp::Max);

  auto fused = pass_.apply(relu);
  ASSERT_TRUE(fused->isCustom());

  Evaluator evaluator(makeBackendDispatch(CPUBackend::getInstance()));
  evaluator.eval(fused);
  const auto& fusedResult = fused->getResult().value();

  auto bn2 = BatchNormNode::create(
      {inputNode, weightNode, biasNode, meanNode, varNode},
      {1}, false, 0.0, 1e-5, descs);
  auto proj2 = ProjectionNode::create(bn2, BatchNormNode::Result, inShape, dtype::f32);
  evaluator.eval(proj2);
  auto reluRef = fl::maximum(proj2->getResult().value(), 0.0f);

  ASSERT_EQ(fusedResult.shape(), reluRef.shape());
  ASSERT_TRUE(fl::allClose(fusedResult, reluRef, 1e-5));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dSumReluNumerical) {
  // relu(conv2d + other) fused vs unfused
  auto inputData = std::vector<float>(1 * 3 * 8 * 8, 1.0f);
  auto weightsData = std::vector<float>(4 * 3 * 3 * 3, 0.1f);
  auto biasData = std::vector<float>(4, -5.0f);
  auto otherData = std::vector<float>(1 * 4 * 6 * 6, 10.0f);

  Shape inputShape({1, 3, 8, 8});
  Shape weightsShape({4, 3, 3, 3});
  Shape biasShape({4});
  Shape outShape({1, 4, 6, 6});

  auto inputNode = ValueNode::create(
      toTensor<CPUTensor>(inputShape, dtype::f32, inputData.data(), Location::Host));
  auto weightsNode = ValueNode::create(
      toTensor<CPUTensor>(weightsShape, dtype::f32, weightsData.data(), Location::Host));
  auto biasNode = ValueNode::create(
      toTensor<CPUTensor>(biasShape, dtype::f32, biasData.data(), Location::Host));
  auto otherNode = ValueNode::create(
      toTensor<CPUTensor>(outShape, dtype::f32, otherData.data(), Location::Host));

  auto conv = Conv2dNode::create(
      inputNode, weightsNode, biasNode,
      1, 1, 0, 0, 1, 1, 1, false);
  auto add = BinaryNode::create(conv, otherNode, BinaryOp::Add);
  auto zero = ScalarNode::create(outShape, dtype::f32, 0.0);
  auto relu = BinaryNode::create(add, zero, BinaryOp::Max);

  auto fused = pass_.apply(relu);
  ASSERT_TRUE(fused->isCustom());

  Evaluator evaluator(makeBackendDispatch(CPUBackend::getInstance()));
  evaluator.eval(fused);
  const auto& fusedResult = fused->getResult().value();

  auto conv2 = Conv2dNode::create(
      inputNode, weightsNode, biasNode,
      1, 1, 0, 0, 1, 1, 1, false);
  evaluator.eval(conv2);
  auto convResult = conv2->getResult().value();
  evaluator.eval(otherNode);
  auto addRef = convResult + otherNode->getResult().value();
  auto reluRef = fl::maximum(addRef, 0.0f);

  ASSERT_EQ(fusedResult.shape(), reluRef.shape());
  ASSERT_TRUE(fl::allClose(fusedResult, reluRef, 1e-5));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, Conv2dSumOnlyNumerical) {
  // add(conv2d, other) fused vs unfused — sum-only, no eltwise
  auto inputData = std::vector<float>(1 * 3 * 8 * 8, 1.0f);
  auto weightsData = std::vector<float>(4 * 3 * 3 * 3, 0.1f);
  auto biasData = std::vector<float>(4, -5.0f);
  auto otherData = std::vector<float>(1 * 4 * 6 * 6, 10.0f);

  Shape inputShape({1, 3, 8, 8});
  Shape weightsShape({4, 3, 3, 3});
  Shape biasShape({4});
  Shape outShape({1, 4, 6, 6});

  auto inputNode = ValueNode::create(
      toTensor<CPUTensor>(inputShape, dtype::f32, inputData.data(), Location::Host));
  auto weightsNode = ValueNode::create(
      toTensor<CPUTensor>(weightsShape, dtype::f32, weightsData.data(), Location::Host));
  auto biasNode = ValueNode::create(
      toTensor<CPUTensor>(biasShape, dtype::f32, biasData.data(), Location::Host));
  auto otherNode = ValueNode::create(
      toTensor<CPUTensor>(outShape, dtype::f32, otherData.data(), Location::Host));

  auto conv = Conv2dNode::create(
      inputNode, weightsNode, biasNode,
      1, 1, 0, 0, 1, 1, 1, false);
  auto add = BinaryNode::create(conv, otherNode, BinaryOp::Add);

  auto fused = pass_.apply(add);
  ASSERT_TRUE(fused->isCustom());

  Evaluator evaluator(makeBackendDispatch(CPUBackend::getInstance()));
  evaluator.eval(fused);
  const auto& fusedResult = fused->getResult().value();

  auto conv2 = Conv2dNode::create(
      inputNode, weightsNode, biasNode,
      1, 1, 0, 0, 1, 1, 1, false);
  evaluator.eval(conv2);
  evaluator.eval(otherNode);
  auto addRef = conv2->getResult().value() + otherNode->getResult().value();

  ASSERT_EQ(fusedResult.shape(), addRef.shape());
  ASSERT_TRUE(fl::allClose(fusedResult, addRef, 1e-5));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, MatmulReluNumerical) {
  // relu(matmul(A, B)) fused vs unfused
  auto lhsData = std::vector<float>(4 * 3);
  auto rhsData = std::vector<float>(3 * 5);
  for (size_t i = 0; i < lhsData.size(); i++) {
    lhsData[i] = static_cast<float>(i) * 0.1f - 0.5f;
  }
  for (size_t i = 0; i < rhsData.size(); i++) {
    rhsData[i] = static_cast<float>(i) * 0.2f - 1.0f;
  }

  Shape lhsShape({4, 3});
  Shape rhsShape({3, 5});

  auto lhsNode = ValueNode::create(
      toTensor<CPUTensor>(lhsShape, dtype::f32, lhsData.data(), Location::Host));
  auto rhsNode = ValueNode::create(
      toTensor<CPUTensor>(rhsShape, dtype::f32, rhsData.data(), Location::Host));

  auto mm = MatmulNode::create(
      lhsNode, rhsNode, MatrixProperty::None, MatrixProperty::None);
  auto zero = ScalarNode::create(mm->shape(), dtype::f32, 0.0);
  auto relu = BinaryNode::create(mm, zero, BinaryOp::Max);

  auto fused = pass_.apply(relu);
  ASSERT_TRUE(fused->isCustom());

  Evaluator evaluator(makeBackendDispatch(CPUBackend::getInstance()));
  evaluator.eval(fused);
  const auto& fusedResult = fused->getResult().value();

  auto mm2 = MatmulNode::create(
      lhsNode, rhsNode, MatrixProperty::None, MatrixProperty::None);
  evaluator.eval(mm2);
  auto reluRef = fl::maximum(mm2->getResult().value(), 0.0f);

  ASSERT_EQ(fusedResult.shape(), reluRef.shape());
  ASSERT_TRUE(fl::allClose(fusedResult, reluRef, 1e-5));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, MatmulBiasReluNumerical) {
  // relu(matmul(A, B) + bias) fused vs unfused — the linear layer pattern
  auto lhsData = std::vector<float>(4 * 3);
  auto rhsData = std::vector<float>(3 * 5);
  auto biasData = std::vector<float>{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
  for (size_t i = 0; i < lhsData.size(); i++) {
    lhsData[i] = static_cast<float>(i) * 0.1f - 0.5f;
  }
  for (size_t i = 0; i < rhsData.size(); i++) {
    rhsData[i] = static_cast<float>(i) * 0.2f - 1.0f;
  }

  Shape lhsShape({4, 3});
  Shape rhsShape({3, 5});
  Shape biasShape({1, 5});

  auto lhsNode = ValueNode::create(
      toTensor<CPUTensor>(lhsShape, dtype::f32, lhsData.data(), Location::Host));
  auto rhsNode = ValueNode::create(
      toTensor<CPUTensor>(rhsShape, dtype::f32, rhsData.data(), Location::Host));
  auto biasNode = ValueNode::create(
      toTensor<CPUTensor>(biasShape, dtype::f32, biasData.data(), Location::Host));

  auto mm = MatmulNode::create(
      lhsNode, rhsNode, MatrixProperty::None, MatrixProperty::None);
  auto add = BinaryNode::create(mm, biasNode, BinaryOp::Add);
  auto zero = ScalarNode::create(add->shape(), dtype::f32, 0.0);
  auto relu = BinaryNode::create(add, zero, BinaryOp::Max);

  auto fused = pass_.apply(relu);
  ASSERT_TRUE(fused->isCustom());
  ASSERT_EQ(fused->inputs().size(), 3u); // lhs, rhs, bias

  Evaluator evaluator(makeBackendDispatch(CPUBackend::getInstance()));
  evaluator.eval(fused);
  const auto& fusedResult = fused->getResult().value();

  // Compute reference: matmul + bias + relu
  auto mm2 = MatmulNode::create(
      lhsNode, rhsNode, MatrixProperty::None, MatrixProperty::None);
  evaluator.eval(mm2);
  evaluator.eval(biasNode);
  auto addRef = mm2->getResult().value() + biasNode->getResult().value();
  auto reluRef = fl::maximum(addRef, 0.0f);

  ASSERT_EQ(fusedResult.shape(), reluRef.shape());
  ASSERT_TRUE(fl::allClose(fusedResult, reluRef, 1e-5));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, MatmulBiasOnlyNumerical) {
  // add(matmul(A, B), bias) fused vs unfused — bias-only, no eltwise
  auto lhsData = std::vector<float>(4 * 3);
  auto rhsData = std::vector<float>(3 * 5);
  auto biasData = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  for (size_t i = 0; i < lhsData.size(); i++) {
    lhsData[i] = static_cast<float>(i) * 0.1f;
  }
  for (size_t i = 0; i < rhsData.size(); i++) {
    rhsData[i] = static_cast<float>(i) * 0.2f;
  }

  Shape lhsShape({4, 3});
  Shape rhsShape({3, 5});
  Shape biasShape({5});

  auto lhsNode = ValueNode::create(
      toTensor<CPUTensor>(lhsShape, dtype::f32, lhsData.data(), Location::Host));
  auto rhsNode = ValueNode::create(
      toTensor<CPUTensor>(rhsShape, dtype::f32, rhsData.data(), Location::Host));
  auto biasNode = ValueNode::create(
      toTensor<CPUTensor>(biasShape, dtype::f32, biasData.data(), Location::Host));

  auto mm = MatmulNode::create(
      lhsNode, rhsNode, MatrixProperty::None, MatrixProperty::None);
  auto add = BinaryNode::create(mm, biasNode, BinaryOp::Add);

  auto fused = pass_.apply(add);
  ASSERT_TRUE(fused->isCustom());

  Evaluator evaluator(makeBackendDispatch(CPUBackend::getInstance()));
  evaluator.eval(fused);
  const auto& fusedResult = fused->getResult().value();

  auto mm2 = MatmulNode::create(
      lhsNode, rhsNode, MatrixProperty::None, MatrixProperty::None);
  evaluator.eval(mm2);
  evaluator.eval(biasNode);
  auto addRef = mm2->getResult().value() + biasNode->getResult().value();

  ASSERT_EQ(fusedResult.shape(), addRef.shape());
  ASSERT_TRUE(fl::allClose(fusedResult, addRef, 1e-5));
}

// ==================== Gelu structural tests ====================

namespace {

// Helper: build the composed gelu(x) subgraph.
// gelu(x) = x * (0.5 * (1.0 + erf(x * M_SQRT1_2)))
NodePtr makeComposedGelu(const NodePtr& x) {
  auto sqrt1_2 = ScalarNode::create(x->shape(), x->dataType(), M_SQRT1_2);
  auto xScaled = BinaryNode::create(x, sqrt1_2, BinaryOp::Mul);
  auto erfNode = UnaryNode::create(xScaled, UnaryOp::Erf);
  auto one = ScalarNode::create(x->shape(), x->dataType(), 1.0);
  auto addOne = BinaryNode::create(one, erfNode, BinaryOp::Add);
  auto half = ScalarNode::create(x->shape(), x->dataType(), 0.5);
  auto cdfScaled = BinaryNode::create(half, addOne, BinaryOp::Mul);
  return BinaryNode::create(x, cdfScaled, BinaryOp::Mul);
}

} // namespace

TEST_F(JitOneDnnPrimitivePostOpFusionTest, GeluAnchorMatmul) {
  // gelu(matmul(A, B)) → fused CustomNode with gelu eltwise post-op
  auto lhs = makeValue({4, 3});
  auto rhs = makeValue({3, 5});
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  auto gelu = makeComposedGelu(mm);

  auto result = pass_.apply(gelu);

  ASSERT_NE(result, gelu);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->impl<CustomNode>().name(), "OneDnnFusedMatmulChain");
  ASSERT_EQ(result->inputs(), NodeList({lhs, rhs}));
  ASSERT_EQ(result->shape(), gelu->shape());
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, GeluBiasAnchorMatmul) {
  // gelu(matmul(A,B) + bias) → fused matmul with native bias + gelu post-op
  auto lhs = makeValue({4, 3});
  auto rhs = makeValue({3, 5});
  auto bias = makeValue({1, 5});
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  auto add = BinaryNode::create(mm, bias, BinaryOp::Add);
  auto gelu = makeComposedGelu(add);

  auto result = pass_.apply(gelu);

  ASSERT_NE(result, gelu);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->impl<CustomNode>().name(), "OneDnnFusedMatmulChain");
  ASSERT_EQ(result->inputs().size(), 3u);
  ASSERT_EQ(result->inputs()[0], lhs);
  ASSERT_EQ(result->inputs()[1], rhs);
  ASSERT_EQ(result->inputs()[2], bias);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, GeluChainAnchorMatmul) {
  // sigmoid(gelu(matmul(A,B))) → fused matmul with 2 post-ops (gelu + sigmoid)
  auto lhs = makeValue({4, 3});
  auto rhs = makeValue({3, 5});
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  auto gelu = makeComposedGelu(mm);
  auto sigmoid = UnaryNode::create(gelu, UnaryOp::Sigmoid);

  auto result = pass_.apply(sigmoid);

  ASSERT_NE(result, sigmoid);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->impl<CustomNode>().name(), "OneDnnFusedMatmulChain");
  ASSERT_EQ(result->inputs(), NodeList({lhs, rhs}));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, GeluAnchorConv2d) {
  // gelu(conv2d(...)) → fused conv2d with gelu post-op
  Conv2dTestGraph g;
  auto gelu = makeComposedGelu(g.conv);

  auto result = pass_.apply(gelu);

  ASSERT_NE(result, gelu);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->impl<CustomNode>().name(), "OneDnnFusedConv2dChain");
  ASSERT_EQ(result->inputs(), NodeList({g.input, g.weights, g.bias}));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, StandaloneGelu) {
  // gelu(x) with no anchor below → standalone eltwise
  auto x = makeValue({4, 8});
  auto gelu = makeComposedGelu(x);

  auto result = pass_.apply(gelu);

  ASSERT_NE(result, gelu);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->impl<CustomNode>().name(), "OneDnnStandaloneEltwise");
  ASSERT_EQ(result->inputs(), NodeList({x}));
  ASSERT_EQ(result->shape(), gelu->shape());
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, GeluNoFusionMultiUseInterior) {
  // Interior node of gelu has multi-use → no gelu pattern match
  auto x = makeValue({4, 8});
  auto sqrt1_2 = ScalarNode::create(x->shape(), dtype::f32, M_SQRT1_2);
  auto xScaled = BinaryNode::create(x, sqrt1_2, BinaryOp::Mul);
  auto erfNode = UnaryNode::create(xScaled, UnaryOp::Erf);
  auto extra = UnaryNode::create(erfNode, UnaryOp::Negative); // 2nd use of erf
  auto one = ScalarNode::create(x->shape(), dtype::f32, 1.0);
  auto addOne = BinaryNode::create(one, erfNode, BinaryOp::Add);
  auto half = ScalarNode::create(x->shape(), dtype::f32, 0.5);
  auto cdfScaled = BinaryNode::create(half, addOne, BinaryOp::Mul);
  auto gelu = BinaryNode::create(x, cdfScaled, BinaryOp::Mul);

  auto result = pass_.apply(gelu);
  ASSERT_EQ(result, gelu); // no match
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, GeluNoFusionF64) {
  // f64 gelu(matmul) → no post-op fusion (but would get standalone)
  auto lhs = makeValue({4, 3}, dtype::f64);
  auto rhs = makeValue({3, 5}, dtype::f64);
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  auto gelu = makeComposedGelu(mm);

  auto result = pass_.apply(gelu);

  // f64 blocks post-op fusion, so it falls through to standalone.
  ASSERT_NE(result, gelu);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->impl<CustomNode>().name(), "OneDnnStandaloneEltwise");
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, GeluSharedMatmulStandalone) {
  // gelu(mm) where mm has a consumer outside the gelu pattern: mm can't be
  // absorbed, but gelu still lowers to a standalone eltwise reading mm (and
  // must never consume the node it replaces).
  auto lhs = makeValue({4, 3});
  auto rhs = makeValue({3, 5});
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  auto gelu = makeComposedGelu(mm);
  auto otherUse = UnaryNode::create(mm, UnaryOp::Negative);

  auto result = pass_.apply(gelu);

  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->impl<CustomNode>().name(), "OneDnnStandaloneEltwise");
  ASSERT_EQ(result->inputs(), NodeList({mm}));
  ASSERT_EQ(otherUse->inputs(), NodeList({mm}));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, GeluSharedConvNotAbsorbed) {
  // A conv read outside the activation chain must not be fused with it: the
  // other reader would otherwise observe the post-activation value.
  Conv2dTestGraph g;
  auto gelu = makeComposedGelu(g.conv);
  auto otherUse = UnaryNode::create(g.conv, UnaryOp::Negative);

  auto result = pass_.apply(gelu);

  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->impl<CustomNode>().name(), "OneDnnStandaloneEltwise");
  ASSERT_EQ(result->inputs(), NodeList({g.conv}));
  ASSERT_EQ(otherUse->inputs(), NodeList({g.conv}));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, GeluOfNegSharedInput) {
  // gelu(neg(x)) with x read elsewhere: the chain is exactly [neg, gelu] over
  // x (previously the walk stopped at neg but kept it in the chain, computing
  // gelu(-(-x))).
  auto x = makeValue({4, 8});
  auto neg = UnaryNode::create(x, UnaryOp::Negative);
  auto gelu = makeComposedGelu(neg);
  auto otherUse = UnaryNode::create(x, UnaryOp::Exp);

  auto result = pass_.apply(gelu);

  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->impl<CustomNode>().name(), "OneDnnStandaloneEltwise");
  ASSERT_EQ(result->inputs(), NodeList({x}));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, GeluBiasNoFusionF64) {
  // f64 gelu(matmul + bias) can't use post-ops, and the standalone fallback
  // would drop the bias, so nothing is fused.
  auto lhs = makeValue({4, 3}, dtype::f64);
  auto rhs = makeValue({3, 5}, dtype::f64);
  auto bias = makeValue({1, 5}, dtype::f64);
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  auto add = BinaryNode::create(mm, bias, BinaryOp::Add);
  auto gelu = makeComposedGelu(add);

  auto result = pass_.apply(gelu);

  ASSERT_EQ(result, gelu);
}

// ==================== Gelu numerical tests ====================

TEST_F(JitOneDnnPrimitivePostOpFusionTest, GeluMatmulNumerical) {
  // gelu(matmul(A, B)) fused vs composed reference
  auto lhsData = std::vector<float>(4 * 3);
  auto rhsData = std::vector<float>(3 * 5);
  for (size_t i = 0; i < lhsData.size(); i++) {
    lhsData[i] = static_cast<float>(i) * 0.1f - 0.5f;
  }
  for (size_t i = 0; i < rhsData.size(); i++) {
    rhsData[i] = static_cast<float>(i) * 0.2f - 1.0f;
  }

  Shape lhsShape({4, 3});
  Shape rhsShape({3, 5});

  auto lhsNode = ValueNode::create(
      toTensor<CPUTensor>(lhsShape, dtype::f32, lhsData.data(), Location::Host));
  auto rhsNode = ValueNode::create(
      toTensor<CPUTensor>(rhsShape, dtype::f32, rhsData.data(), Location::Host));

  auto mm = MatmulNode::create(
      lhsNode, rhsNode, MatrixProperty::None, MatrixProperty::None);
  auto gelu = makeComposedGelu(mm);

  auto fused = pass_.apply(gelu);
  ASSERT_TRUE(fused->isCustom());
  ASSERT_EQ(fused->impl<CustomNode>().name(), "OneDnnFusedMatmulChain");

  Evaluator evaluator(makeBackendDispatch(CPUBackend::getInstance()));
  evaluator.eval(fused);
  const auto& fusedResult = fused->getResult().value();

  // Reference: compute matmul then apply gelu element-wise.
  auto mm2 = MatmulNode::create(
      lhsNode, rhsNode, MatrixProperty::None, MatrixProperty::None);
  evaluator.eval(mm2);
  auto mmResult = mm2->getResult().value();
  // gelu(x) = x * 0.5 * (1 + erf(x * sqrt(0.5)))
  auto geluRef = mmResult * 0.5f * (1.0f + fl::erf(mmResult * static_cast<float>(M_SQRT1_2)));

  ASSERT_EQ(fusedResult.shape(), geluRef.shape());
  ASSERT_TRUE(fl::allClose(fusedResult, geluRef, 1e-5));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, StandaloneGeluNumerical) {
  // Standalone gelu(x) lowered vs composed reference
  auto data = std::vector<float>(32);
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = static_cast<float>(i) * 0.2f - 3.0f;
  }

  Shape shape({4, 8});
  auto xNode = ValueNode::create(
      toTensor<CPUTensor>(shape, dtype::f32, data.data(), Location::Host));

  auto gelu = makeComposedGelu(xNode);

  auto fused = pass_.apply(gelu);
  ASSERT_TRUE(fused->isCustom());
  ASSERT_EQ(fused->impl<CustomNode>().name(), "OneDnnStandaloneEltwise");

  Evaluator evaluator(makeBackendDispatch(CPUBackend::getInstance()));
  evaluator.eval(fused);
  const auto& fusedResult = fused->getResult().value();

  // Reference
  evaluator.eval(xNode);
  auto x = xNode->getResult().value();
  auto geluRef = x * 0.5f * (1.0f + fl::erf(x * static_cast<float>(M_SQRT1_2)));

  ASSERT_EQ(fusedResult.shape(), geluRef.shape());
  ASSERT_TRUE(fl::allClose(fusedResult, geluRef, 1e-5));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, GeluBiasMatmulNumerical) {
  // gelu(matmul(A, B) + bias) fused vs composed reference
  auto lhsData = std::vector<float>(4 * 3);
  auto rhsData = std::vector<float>(3 * 5);
  auto biasData = std::vector<float>{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
  for (size_t i = 0; i < lhsData.size(); i++) {
    lhsData[i] = static_cast<float>(i) * 0.1f - 0.5f;
  }
  for (size_t i = 0; i < rhsData.size(); i++) {
    rhsData[i] = static_cast<float>(i) * 0.2f - 1.0f;
  }

  Shape lhsShape({4, 3});
  Shape rhsShape({3, 5});
  Shape biasShape({1, 5});

  auto lhsNode = ValueNode::create(
      toTensor<CPUTensor>(lhsShape, dtype::f32, lhsData.data(), Location::Host));
  auto rhsNode = ValueNode::create(
      toTensor<CPUTensor>(rhsShape, dtype::f32, rhsData.data(), Location::Host));
  auto biasNode = ValueNode::create(
      toTensor<CPUTensor>(biasShape, dtype::f32, biasData.data(), Location::Host));

  auto mm = MatmulNode::create(
      lhsNode, rhsNode, MatrixProperty::None, MatrixProperty::None);
  auto add = BinaryNode::create(mm, biasNode, BinaryOp::Add);
  auto gelu = makeComposedGelu(add);

  auto fused = pass_.apply(gelu);
  ASSERT_TRUE(fused->isCustom());
  ASSERT_EQ(fused->impl<CustomNode>().name(), "OneDnnFusedMatmulChain");
  ASSERT_EQ(fused->inputs().size(), 3u);

  Evaluator evaluator(makeBackendDispatch(CPUBackend::getInstance()));
  evaluator.eval(fused);
  const auto& fusedResult = fused->getResult().value();

  // Reference: matmul + bias + gelu
  auto mm2 = MatmulNode::create(
      lhsNode, rhsNode, MatrixProperty::None, MatrixProperty::None);
  evaluator.eval(mm2);
  evaluator.eval(biasNode);
  auto addRef = mm2->getResult().value() + biasNode->getResult().value();
  auto geluRef = addRef * 0.5f * (1.0f + fl::erf(addRef * static_cast<float>(M_SQRT1_2)));

  ASSERT_EQ(fusedResult.shape(), geluRef.shape());
  ASSERT_TRUE(fl::allClose(fusedResult, geluRef, 1e-5));
}

// ==================== Silu structural tests ====================

namespace {

// Helper: build the composed silu(x) = x * sigmoid(x) subgraph.
NodePtr makeComposedSilu(const NodePtr& x) {
  auto sig = UnaryNode::create(x, UnaryOp::Sigmoid);
  return BinaryNode::create(x, sig, BinaryOp::Mul);
}

// Helper: build silu as produced by swish(x, beta=1.0):
//   x * sigmoid(Scalar(1.0) * x)
NodePtr makeSwishBeta1(const NodePtr& x) {
  auto one = ScalarNode::create(x->shape(), x->dataType(), 1.0);
  auto scaled = BinaryNode::create(one, x, BinaryOp::Mul);
  auto sig = UnaryNode::create(scaled, UnaryOp::Sigmoid);
  return BinaryNode::create(x, sig, BinaryOp::Mul);
}

} // namespace

TEST_F(JitOneDnnPrimitivePostOpFusionTest, SiluAnchorConv2d) {
  // silu(conv2d(...)) → fused conv2d with swish post-op
  Conv2dTestGraph g;
  auto silu = makeComposedSilu(g.conv);

  auto result = pass_.apply(silu);

  ASSERT_NE(result, silu);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->impl<CustomNode>().name(), "OneDnnFusedConv2dChain");
  ASSERT_EQ(result->inputs(), NodeList({g.input, g.weights, g.bias}));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, SiluAnchorMatmul) {
  // silu(matmul(A, B)) → fused matmul with swish post-op
  auto lhs = makeValue({4, 3});
  auto rhs = makeValue({3, 5});
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  auto silu = makeComposedSilu(mm);

  auto result = pass_.apply(silu);

  ASSERT_NE(result, silu);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->impl<CustomNode>().name(), "OneDnnFusedMatmulChain");
  ASSERT_EQ(result->inputs(), NodeList({lhs, rhs}));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, SiluSwapped) {
  // sigmoid(x) * x (operands swapped) → still matches
  Conv2dTestGraph g;
  auto sig = UnaryNode::create(g.conv, UnaryOp::Sigmoid);
  auto silu = BinaryNode::create(sig, g.conv, BinaryOp::Mul); // swapped

  auto result = pass_.apply(silu);

  ASSERT_NE(result, silu);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->impl<CustomNode>().name(), "OneDnnFusedConv2dChain");
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, SwishBeta1AnchorMatmul) {
  // x * sigmoid(1.0 * x) (the actual autograd swish output) → matches
  auto lhs = makeValue({4, 3});
  auto rhs = makeValue({3, 5});
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  auto silu = makeSwishBeta1(mm);

  auto result = pass_.apply(silu);

  ASSERT_NE(result, silu);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->impl<CustomNode>().name(), "OneDnnFusedMatmulChain");
  ASSERT_EQ(result->inputs(), NodeList({lhs, rhs}));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, SiluBiasAnchorMatmul) {
  // silu(matmul(A, B) + bias) → fused matmul with bias + swish
  auto lhs = makeValue({4, 3});
  auto rhs = makeValue({3, 5});
  auto bias = makeValue({1, 5});
  auto mm = MatmulNode::create(lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  auto add = BinaryNode::create(mm, bias, BinaryOp::Add);
  auto silu = makeComposedSilu(add);

  auto result = pass_.apply(silu);

  ASSERT_NE(result, silu);
  ASSERT_TRUE(result->isCustom());
  ASSERT_EQ(result->impl<CustomNode>().name(), "OneDnnFusedMatmulChain");
  ASSERT_EQ(result->inputs().size(), 3u);
  ASSERT_EQ(result->inputs()[0], lhs);
  ASSERT_EQ(result->inputs()[1], rhs);
  ASSERT_EQ(result->inputs()[2], bias);
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, SiluNoMatchDifferentSigmoidInput) {
  // x * sigmoid(y) where y != x → no silu match
  Conv2dTestGraph g;
  auto other = makeValue(g.conv->shape());
  auto sig = UnaryNode::create(other, UnaryOp::Sigmoid);
  auto mul = BinaryNode::create(g.conv, sig, BinaryOp::Mul);

  auto result = pass_.apply(mul);

  // No silu matched; no other pattern either → unchanged
  ASSERT_EQ(result, mul);
}

// ==================== Silu numerical tests ====================

TEST_F(JitOneDnnPrimitivePostOpFusionTest, SiluMatmulNumerical) {
  // silu(matmul(A, B)) fused vs composed reference
  auto lhsData = std::vector<float>(4 * 3);
  auto rhsData = std::vector<float>(3 * 5);
  for (size_t i = 0; i < lhsData.size(); i++) {
    lhsData[i] = static_cast<float>(i) * 0.1f - 0.5f;
  }
  for (size_t i = 0; i < rhsData.size(); i++) {
    rhsData[i] = static_cast<float>(i) * 0.2f - 1.0f;
  }

  Shape lhsShape({4, 3});
  Shape rhsShape({3, 5});

  auto lhsNode = ValueNode::create(
      toTensor<CPUTensor>(lhsShape, dtype::f32, lhsData.data(), Location::Host));
  auto rhsNode = ValueNode::create(
      toTensor<CPUTensor>(rhsShape, dtype::f32, rhsData.data(), Location::Host));

  auto mm = MatmulNode::create(
      lhsNode, rhsNode, MatrixProperty::None, MatrixProperty::None);
  auto silu = makeComposedSilu(mm);

  auto fused = pass_.apply(silu);
  ASSERT_TRUE(fused->isCustom());
  ASSERT_EQ(fused->impl<CustomNode>().name(), "OneDnnFusedMatmulChain");

  Evaluator evaluator(makeBackendDispatch(CPUBackend::getInstance()));
  evaluator.eval(fused);
  const auto& fusedResult = fused->getResult().value();

  // Reference: matmul then silu(x) = x * sigmoid(x)
  auto mm2 = MatmulNode::create(
      lhsNode, rhsNode, MatrixProperty::None, MatrixProperty::None);
  evaluator.eval(mm2);
  auto mmResult = mm2->getResult().value();
  auto siluRef = mmResult * fl::sigmoid(mmResult);

  ASSERT_EQ(fusedResult.shape(), siluRef.shape());
  ASSERT_TRUE(fl::allClose(fusedResult, siluRef, 1e-5));
}

TEST_F(JitOneDnnPrimitivePostOpFusionTest, SwishBeta1MatmulNumerical) {
  // x * sigmoid(1.0 * x) fused vs composed reference (the actual autograd path)
  auto lhsData = std::vector<float>(4 * 3);
  auto rhsData = std::vector<float>(3 * 5);
  for (size_t i = 0; i < lhsData.size(); i++) {
    lhsData[i] = static_cast<float>(i) * 0.1f - 0.5f;
  }
  for (size_t i = 0; i < rhsData.size(); i++) {
    rhsData[i] = static_cast<float>(i) * 0.2f - 1.0f;
  }

  Shape lhsShape({4, 3});
  Shape rhsShape({3, 5});

  auto lhsNode = ValueNode::create(
      toTensor<CPUTensor>(lhsShape, dtype::f32, lhsData.data(), Location::Host));
  auto rhsNode = ValueNode::create(
      toTensor<CPUTensor>(rhsShape, dtype::f32, rhsData.data(), Location::Host));

  auto mm = MatmulNode::create(
      lhsNode, rhsNode, MatrixProperty::None, MatrixProperty::None);
  auto silu = makeSwishBeta1(mm);

  auto fused = pass_.apply(silu);
  ASSERT_TRUE(fused->isCustom());
  ASSERT_EQ(fused->impl<CustomNode>().name(), "OneDnnFusedMatmulChain");

  Evaluator evaluator(makeBackendDispatch(CPUBackend::getInstance()));
  evaluator.eval(fused);
  const auto& fusedResult = fused->getResult().value();

  // Reference
  auto mm2 = MatmulNode::create(
      lhsNode, rhsNode, MatrixProperty::None, MatrixProperty::None);
  evaluator.eval(mm2);
  auto mmResult = mm2->getResult().value();
  auto siluRef = mmResult * fl::sigmoid(mmResult);

  ASSERT_EQ(fusedResult.shape(), siluRef.shape());
  ASSERT_TRUE(fl::allClose(fusedResult, siluRef, 1e-5));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
