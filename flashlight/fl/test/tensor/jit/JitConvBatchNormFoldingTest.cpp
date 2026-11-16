/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/ir/BatchNormNode.h"
#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/Conv2dNode.h"
#include "flashlight/fl/tensor/ir/ProjectionNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"
#include "flashlight/fl/tensor/backend/cpu/CPUBackend.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/opt/passes/ConvBatchNormFolding.h"

using namespace fl;

namespace {

// Helper: create a ValueNode wrapping a bare CPUTensor.
NodePtr makeValue(const Shape& shape, dtype type = dtype::f32) {
  Tensor bare = toTensor<CPUTensor>(shape, type, nullptr, Location::Host);
  return ValueNode::create(std::move(bare));
}

// Helper: create a ValueNode from a concrete Tensor.
NodePtr makeValueFrom(Tensor t) {
  return ValueNode::create(std::move(t));
}

// Helper: build standard BN output descriptors.
std::vector<OutputDesc> makeBNDescs(
    const Shape& outShape, const Shape& perCh, dtype type = dtype::f32) {
  return {
      {outShape, type},  // Result
      {perCh, type},     // SaveMean
      {perCh, type},     // SaveVar
      {perCh, type},     // NewRunningMean
      {perCh, type},     // NewRunningVar
  };
}

// Build a Conv2d -> BN(inference) -> Projection(Result) subgraph.
struct ConvBNInferenceGraph {
  NodePtr input;
  NodePtr convWeights;
  NodePtr convBias;
  Conv2dNodePtr conv;
  NodePtr bnWeight;   // gamma
  NodePtr bnBias;     // beta
  NodePtr runningMean;
  NodePtr runningVar;
  BatchNormNodePtr bn;
  ProjectionNodePtr resultProj;

  ConvBNInferenceGraph(
      bool train = false,
      std::vector<int> axes = {1},
      bool withConvBias = true) {
    input = makeValue({1, 3, 8, 8});
    convWeights = makeValue({4, 3, 3, 3});
    convBias = withConvBias ? makeValue({1, 4, 1, 1}) : makeValue({});
    conv = Conv2dNode::create(
        input, convWeights, convBias,
        /*sx=*/1, /*sy=*/1, /*px=*/0, /*py=*/0,
        /*dx=*/1, /*dy=*/1, /*groups=*/1, /*wantState=*/true);

    bnWeight = makeValue({1, 4, 1, 1});
    bnBias = makeValue({1, 4, 1, 1});
    runningMean = makeValue({1, 4, 1, 1});
    runningVar = makeValue({1, 4, 1, 1});

    Shape outShape({1, 4, 6, 6});
    Shape perCh({1, 4, 1, 1});

    bn = BatchNormNode::create(
        {conv, bnWeight, bnBias, runningMean, runningVar},
        axes, train,
        /*momentum=*/0.1, /*epsilon=*/1e-5,
        makeBNDescs(outShape, perCh));

    resultProj = ProjectionNode::create(
        bn, BatchNormNode::Result, outShape, dtype::f32);
  }
};

} // namespace

class JitConvBatchNormFoldingTest : public ::testing::Test {
 protected:
  ConvBatchNormFolding pass_;
};

// ==================== Structural tests ====================

TEST_F(JitConvBatchNormFoldingTest, foldsInferenceBN) {
  ConvBNInferenceGraph g(/*train=*/false);
  auto result = pass_.apply(g.resultProj);

  // Result should be a Conv2dNode (not the original projection).
  ASSERT_NE(result, g.resultProj);
  ASSERT_TRUE(result->isConv2d());

  // The new conv should use the same input as the original.
  auto& newConv = result->impl<Conv2dNode>();
  ASSERT_EQ(newConv.input(), g.input);

  // Weights and bias should be new ValueNodes (folded values).
  ASSERT_TRUE(newConv.weights()->isValue());
  ASSERT_TRUE(newConv.bias()->isValue());
  ASSERT_NE(newConv.weights(), g.convWeights);
  ASSERT_NE(newConv.bias(), g.convBias);

  // Conv params preserved.
  ASSERT_EQ(newConv.sx(), 1);
  ASSERT_EQ(newConv.sy(), 1);
  ASSERT_EQ(newConv.px(), 0);
  ASSERT_EQ(newConv.py(), 0);
  ASSERT_EQ(newConv.groups(), 1);
}

TEST_F(JitConvBatchNormFoldingTest, doesNotFoldTrainingBN) {
  ConvBNInferenceGraph g(/*train=*/true);
  auto result = pass_.apply(g.resultProj);
  ASSERT_EQ(result, g.resultProj);
}

TEST_F(JitConvBatchNormFoldingTest, doesNotFoldWrongAxes) {
  ConvBNInferenceGraph g(/*train=*/false, /*axes=*/{0});
  auto result = pass_.apply(g.resultProj);
  ASSERT_EQ(result, g.resultProj);
}

TEST_F(JitConvBatchNormFoldingTest, doesNotFoldMultiUseConv) {
  ConvBNInferenceGraph g(/*train=*/false);

  // Add a second user of the conv node.
  auto extraUser = UnaryNode::create(g.conv, UnaryOp::Negative);

  auto result = pass_.apply(g.resultProj);
  ASSERT_EQ(result, g.resultProj);
}

TEST_F(JitConvBatchNormFoldingTest, doesNotFoldNonConvInput) {
  auto v1 = makeValue({1, 4, 6, 6});
  auto v2 = makeValue({1, 4, 6, 6});
  auto add = BinaryNode::create(v1, v2, BinaryOp::Add);

  Shape outShape({1, 4, 6, 6});
  Shape perCh({1, 4, 1, 1});
  auto bn = BatchNormNode::create(
      {add, makeValue(perCh), makeValue(perCh),
       makeValue(perCh), makeValue(perCh)},
      {1}, /*train=*/false, 0.1, 1e-5,
      makeBNDescs(outShape, perCh));

  auto proj = ProjectionNode::create(bn, BatchNormNode::Result, outShape, dtype::f32);
  auto result = pass_.apply(proj);
  ASSERT_EQ(result, proj);
}

TEST_F(JitConvBatchNormFoldingTest, doesNotFoldNonValueWeights) {
  auto input = makeValue({1, 3, 8, 8});
  auto w1 = makeValue({4, 3, 3, 3});
  auto w2 = makeValue({4, 3, 3, 3});
  auto dynamicWeights = BinaryNode::create(w1, w2, BinaryOp::Add);

  auto conv = Conv2dNode::create(
      input, dynamicWeights, makeValue({1, 4, 1, 1}),
      1, 1, 0, 0, 1, 1, 1, true);

  Shape outShape({1, 4, 6, 6});
  Shape perCh({1, 4, 1, 1});
  auto bn = BatchNormNode::create(
      {conv, makeValue(perCh), makeValue(perCh),
       makeValue(perCh), makeValue(perCh)},
      {1}, false, 0.1, 1e-5,
      makeBNDescs(outShape, perCh));

  auto proj = ProjectionNode::create(bn, BatchNormNode::Result, outShape, dtype::f32);
  auto result = pass_.apply(proj);
  ASSERT_EQ(result, proj);
}

TEST_F(JitConvBatchNormFoldingTest, doesNotFoldNonResultProjection) {
  ConvBNInferenceGraph g(/*train=*/false);

  auto saveMeanProj = ProjectionNode::create(
      g.bn, BatchNormNode::SaveMean, {1, 4, 1, 1}, dtype::f32);
  auto result = pass_.apply(saveMeanProj);
  ASSERT_EQ(result, saveMeanProj);
}

TEST_F(JitConvBatchNormFoldingTest, foldsConvWithoutBias) {
  ConvBNInferenceGraph g(/*train=*/false, /*axes=*/{1}, /*withConvBias=*/false);
  auto result = pass_.apply(g.resultProj);

  ASSERT_NE(result, g.resultProj);
  ASSERT_TRUE(result->isConv2d());

  auto& newConv = result->impl<Conv2dNode>();
  ASSERT_TRUE(newConv.bias()->isValue());
}

// ==================== Numerical tests ====================

TEST_F(JitConvBatchNormFoldingTest, foldedWeightsAndBiasCorrect) {
  // Single output channel, known scalar values.
  // weights=3.0, bias=0.5, gamma=2.0, beta=0.1, mean=1.0, var=4.0
  //   scale = 2.0 / sqrt(4.0 + 1e-5) ~ 1.0
  //   w_new = 3.0 * 1.0 = 3.0
  //   b_new = (0.5 - 1.0) * 1.0 + 0.1 = -0.4
  const float eps = 1e-5f;

  auto inputNode = makeValue({1, 1, 4, 4});
  auto conv = Conv2dNode::create(
      inputNode,
      makeValueFrom(fl::full({1, 1, 1, 1}, 3.0f)),
      makeValueFrom(fl::full({1, 1, 1, 1}, 0.5f)),
      1, 1, 0, 0, 1, 1, 1, true);

  Shape outShape = conv->shape();
  Shape perCh({1, 1, 1, 1});
  auto bn = BatchNormNode::create(
      {conv,
       makeValueFrom(fl::full(perCh, 2.0f)),   // gamma
       makeValueFrom(fl::full(perCh, 0.1f)),   // beta
       makeValueFrom(fl::full(perCh, 1.0f)),   // mean
       makeValueFrom(fl::full(perCh, 4.0f))},  // var
      {1}, /*train=*/false, 0.1, eps,
      makeBNDescs(outShape, perCh));
  auto proj = ProjectionNode::create(
      bn, BatchNormNode::Result, outShape, dtype::f32);

  auto folded = pass_.apply(proj);
  ASSERT_TRUE(folded->isConv2d());

  auto& fc = folded->impl<Conv2dNode>();
  Tensor fw = fc.weights()->impl<ValueNode>().value();
  Tensor fb = fc.bias()->impl<ValueNode>().value();
  fl::eval(fw);
  fl::eval(fb);

  ASSERT_NEAR(fw.asScalar<float>(), 3.0f, 1e-3);
  ASSERT_EQ(fb.shape(), Shape({1, 1, 1, 1}));
  ASSERT_NEAR(fb.asScalar<float>(), -0.4f, 1e-3);
}

TEST_F(JitConvBatchNormFoldingTest, foldedWeightsWithoutConvBias) {
  // No conv bias. b_new = (-mean) * scale + beta
  // weights=2.0, gamma=1.0, beta=0.5, mean=3.0, var=1.0
  //   scale ~ 1.0, w_new = 2.0, b_new = -3.0 + 0.5 = -2.5
  const float eps = 1e-5f;

  auto inputNode = makeValue({1, 1, 4, 4});
  auto conv = Conv2dNode::create(
      inputNode,
      makeValueFrom(fl::full({1, 1, 1, 1}, 2.0f)),
      makeValueFrom(Tensor()),  // empty bias
      1, 1, 0, 0, 1, 1, 1, true);

  Shape outShape = conv->shape();
  Shape perCh({1, 1, 1, 1});
  auto bn = BatchNormNode::create(
      {conv,
       makeValueFrom(fl::full(perCh, 1.0f)),
       makeValueFrom(fl::full(perCh, 0.5f)),
       makeValueFrom(fl::full(perCh, 3.0f)),
       makeValueFrom(fl::full(perCh, 1.0f))},
      {1}, /*train=*/false, 0.1, eps,
      makeBNDescs(outShape, perCh));
  auto proj = ProjectionNode::create(
      bn, BatchNormNode::Result, outShape, dtype::f32);

  auto folded = pass_.apply(proj);
  ASSERT_TRUE(folded->isConv2d());

  auto& fc = folded->impl<Conv2dNode>();
  Tensor fw = fc.weights()->impl<ValueNode>().value();
  Tensor fb = fc.bias()->impl<ValueNode>().value();
  fl::eval(fw);
  fl::eval(fb);

  ASSERT_NEAR(fw.asScalar<float>(), 2.0f, 1e-3);
  ASSERT_NEAR(fb.asScalar<float>(), -2.5f, 1e-3);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  init();
  return RUN_ALL_TESTS();
}
