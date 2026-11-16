/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/BackendDispatch.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/ir/NodeUtils.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/eval/Evaluator.h"
#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/CustomNode.h"
#include "flashlight/fl/tensor/ir/FusedElementwiseNode.h"
#include "flashlight/fl/tensor/ir/FusedReduceElementwiseNode.h"
#include "flashlight/fl/tensor/ir/ReductionNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"
#include "flashlight/fl/tensor/opt/passes/ElementwiseFusion.h"
#include "flashlight/fl/tensor/opt/passes/ReductionOfElementwiseFusion.h"

using namespace fl;

class JitReductionOfElementwiseFusionTest : public ::testing::Test {
 protected:
  TensorBackend& defaultBackend_ = DefaultTensorBackend_t::getInstance();
  ElementwiseFusion elemFusionPass_;
  ReductionOfElementwiseFusion reductionFusionPass_;
};

TEST_F(JitReductionOfElementwiseFusionTest, noFusionReductionOfValue) {
  // sum(v1) -- v1 is a ValueNode, not elementwise -- no fusion
  Shape shape({3, 4});
  const auto v1 =
      ValueNode::create(defaultBackend_.full(shape, 1.0f, dtype::f32));
  const auto sumNode =
      ReductionNode::create(v1, ReductionOp::Sum, {0}, false);
  const auto result = reductionFusionPass_.apply(sumNode);
  ASSERT_EQ(result, sumNode); // unchanged
}

TEST_F(JitReductionOfElementwiseFusionTest, fuseReductionOfUnary) {
  // sum(exp(v1)) -> FusedReduceElementwise(Sum, [Exp], v1)
  Shape shape({3, 4});
  const auto v1 =
      ValueNode::create(defaultBackend_.full(shape, 1.0f, dtype::f32));
  const auto expNode = UnaryNode::create(v1, UnaryOp::Exp);
  const auto sumNode =
      ReductionNode::create(expNode, ReductionOp::Sum, {0}, false);
  const auto result = reductionFusionPass_.apply(sumNode);
  ASSERT_TRUE(result->isFusedReduceElementwise());
  const auto& fused = result->impl<FusedReduceElementwiseNode>();
  ASSERT_EQ(fused.numExternalInputs(), 1u);
  ASSERT_EQ(fused.numSteps(), 1u);
  ASSERT_TRUE(fused.program()[0].isUnary());
  ASSERT_EQ(fused.program()[0].unaryOp(), UnaryOp::Exp);
  ASSERT_EQ(fused.reductionOp(), ReductionOp::Sum);
  ASSERT_EQ(fused.axes(), std::vector<int>({0}));
  ASSERT_FALSE(fused.keepDims());
  ASSERT_EQ(fused.inputs(), NodeList({v1}));
  // Shape: reduce dim 0 of {3,4} -> {4}
  ASSERT_EQ(fused.shape(), Shape({4}));
}

TEST_F(JitReductionOfElementwiseFusionTest, fuseReductionOfBinary) {
  // sum(add(v1, v2)) -> FusedReduceElementwise(Sum, [Add], v1, v2)
  Shape shape({3, 4});
  const auto v1 =
      ValueNode::create(defaultBackend_.full(shape, 1.0f, dtype::f32));
  const auto v2 =
      ValueNode::create(defaultBackend_.full(shape, 2.0f, dtype::f32));
  const auto addNode = BinaryNode::create(v1, v2, BinaryOp::Add);
  const auto sumNode =
      ReductionNode::create(addNode, ReductionOp::Sum, {1}, true);
  const auto result = reductionFusionPass_.apply(sumNode);
  ASSERT_TRUE(result->isFusedReduceElementwise());
  const auto& fused = result->impl<FusedReduceElementwiseNode>();
  ASSERT_EQ(fused.numExternalInputs(), 2u);
  ASSERT_EQ(fused.numSteps(), 1u);
  ASSERT_TRUE(fused.program()[0].isBinary());
  ASSERT_EQ(fused.program()[0].binaryOp(), BinaryOp::Add);
  ASSERT_EQ(fused.reductionOp(), ReductionOp::Sum);
  ASSERT_EQ(fused.axes(), std::vector<int>({1}));
  ASSERT_TRUE(fused.keepDims());
  // Shape: reduce dim 1 of {3,4} keepDims -> {3,1}
  ASSERT_EQ(fused.shape(), Shape({3, 1}));
}

TEST_F(JitReductionOfElementwiseFusionTest, fuseReductionOfFusedElementwise) {
  // sum(exp(add(v1, v2))) -- after ElementwiseFusion:
  //   FusedElementwise([Add, Exp], v1, v2)
  // Then ReductionOfElementwiseFusion:
  //   FusedReduceElementwise(Sum, [Add, Exp], v1, v2)
  Shape shape({3, 4});
  const auto v1 =
      ValueNode::create(defaultBackend_.full(shape, 1.0f, dtype::f32));
  const auto v2 =
      ValueNode::create(defaultBackend_.full(shape, 2.0f, dtype::f32));
  const auto addNode = BinaryNode::create(v1, v2, BinaryOp::Add);
  const auto expNode = UnaryNode::create(addNode, UnaryOp::Exp);
  const auto sumNode =
      ReductionNode::create(expNode, ReductionOp::Sum, {}, false);

  // First fuse elementwise
  auto intermediate = elemFusionPass_.apply(sumNode);
  // sum's input should now be a FusedElementwiseNode
  ASSERT_TRUE(intermediate->isReduction());

  // Then fuse reduction of elementwise
  const auto result = reductionFusionPass_.apply(intermediate);
  ASSERT_TRUE(result->isFusedReduceElementwise());
  const auto& fused = result->impl<FusedReduceElementwiseNode>();
  ASSERT_EQ(fused.numExternalInputs(), 2u);
  ASSERT_EQ(fused.numSteps(), 2u);
  // Step 0: Add(s0, s1)
  ASSERT_TRUE(fused.program()[0].isBinary());
  ASSERT_EQ(fused.program()[0].binaryOp(), BinaryOp::Add);
  // Step 1: Exp(s2)
  ASSERT_TRUE(fused.program()[1].isUnary());
  ASSERT_EQ(fused.program()[1].unaryOp(), UnaryOp::Exp);
  ASSERT_EQ(fused.reductionOp(), ReductionOp::Sum);
  ASSERT_EQ(fused.axes(), std::vector<int>({})); // all axes
  // Shape: reduce all dims of {3,4} -> {} (scalar)
  ASSERT_EQ(fused.shape(), Shape({}));
}

TEST_F(JitReductionOfElementwiseFusionTest, noFusionMultipleUsers) {
  // exp(v1) is used by both sum and neg -- can't absorb
  Shape shape({3, 4});
  const auto v1 =
      ValueNode::create(defaultBackend_.full(shape, 1.0f, dtype::f32));
  const auto expNode = UnaryNode::create(v1, UnaryOp::Exp);
  const auto sumNode =
      ReductionNode::create(expNode, ReductionOp::Sum, {0}, false);
  const auto negNode = UnaryNode::create(expNode, UnaryOp::Negative);
  // expNode has 2 users -- not absorbable
  const auto result = reductionFusionPass_.apply(sumNode);
  ASSERT_EQ(result, sumNode);
}

TEST_F(JitReductionOfElementwiseFusionTest, fuseReductionSharedBinaryInput) {
  // sum(mul(v1, v1)) -- shared input
  Shape shape({3, 4});
  const auto v1 =
      ValueNode::create(defaultBackend_.full(shape, 2.0f, dtype::f32));
  const auto mulNode = BinaryNode::create(v1, v1, BinaryOp::Mul);
  const auto sumNode =
      ReductionNode::create(mulNode, ReductionOp::Sum, {0}, false);
  const auto result = reductionFusionPass_.apply(sumNode);
  ASSERT_TRUE(result->isFusedReduceElementwise());
  const auto& fused = result->impl<FusedReduceElementwiseNode>();
  ASSERT_EQ(fused.numExternalInputs(), 1u); // just v1
  ASSERT_EQ(fused.numSteps(), 1u);
  ASSERT_TRUE(fused.program()[0].isBinary());
  ASSERT_EQ(fused.program()[0].binaryOp(), BinaryOp::Mul);
  ASSERT_EQ(fused.program()[0].src0, 0u);
  ASSERT_EQ(fused.program()[0].src1, 0u); // same input
}

TEST_F(JitReductionOfElementwiseFusionTest, noFusionNonFusableOp) {
  // sum(custom(v1)) -- custom is not fusable elementwise
  Shape shape({3, 4});
  const auto v1 =
      ValueNode::create(defaultBackend_.full(shape, 1.0f, dtype::f32));
  const auto customNode = CustomNode::create(
      "identity",
      {v1},
      shape,
      dtype::f32,
      [](const std::vector<const Tensor*>& inputs) {
        return inputs[0]->copy();
      });
  const auto sumNode =
      ReductionNode::create(customNode, ReductionOp::Sum, {0}, false);
  const auto result = reductionFusionPass_.apply(sumNode);
  ASSERT_EQ(result, sumNode); // custom is not fusable
}

TEST_F(JitReductionOfElementwiseFusionTest, fuseDifferentReductionOps) {
  // mean(exp(v1))
  Shape shape({3, 4});
  const auto v1 =
      ValueNode::create(defaultBackend_.full(shape, 1.0f, dtype::f32));
  const auto expNode = UnaryNode::create(v1, UnaryOp::Exp);
  const auto meanNode =
      ReductionNode::create(expNode, ReductionOp::Mean, {0, 1}, false);
  const auto result = reductionFusionPass_.apply(meanNode);
  ASSERT_TRUE(result->isFusedReduceElementwise());
  const auto& fused = result->impl<FusedReduceElementwiseNode>();
  ASSERT_EQ(fused.reductionOp(), ReductionOp::Mean);
  ASSERT_EQ(fused.axes(), (std::vector<int>{0, 1}));
  ASSERT_FALSE(fused.keepDims());
}

TEST_F(JitReductionOfElementwiseFusionTest, nodeMetadata) {
  // Verify FusedReduceElementwiseNode metadata
  Shape shape({3, 4});
  const auto v1 =
      ValueNode::create(defaultBackend_.full(shape, 1.0f, dtype::f32));
  const auto expNode = UnaryNode::create(v1, UnaryOp::Exp);
  const auto sumNode =
      ReductionNode::create(expNode, ReductionOp::Sum, {0}, true);
  const auto result = reductionFusionPass_.apply(sumNode);
  ASSERT_TRUE(result->isFusedReduceElementwise());
  ASSERT_EQ(result->type(), NodeType::FusedReduceElementwise);
  // Shape: reduce dim 0 of {3,4} keepDims -> {1,4}
  ASSERT_EQ(result->shape(), Shape({1, 4}));
  // Has 1 input (v1)
  ASSERT_EQ(result->inputs().size(), 1u);
  ASSERT_EQ(result->inputs()[0], v1);
}

TEST_F(JitReductionOfElementwiseFusionTest, noFusionBroadcastBinary) {
  // sum(add(v1{3,4}, v2{1,4})) -- add has broadcast inputs.
  // The fused reduction kernel uses flat indexing with no broadcast handling,
  // so the pass must reject fusion of this binary into the reduction.
  Shape bigShape({3, 4});
  Shape smallShape({1, 4});
  const auto v1 =
      ValueNode::create(defaultBackend_.full(bigShape, 1.0f, dtype::f32));
  const auto v2 =
      ValueNode::create(defaultBackend_.full(smallShape, 2.0f, dtype::f32));
  const auto addNode = BinaryNode::create(v1, v2, BinaryOp::Add);
  ASSERT_EQ(addNode->shape(), bigShape); // broadcast output
  const auto sumNode =
      ReductionNode::create(addNode, ReductionOp::Sum, {0}, false);
  const auto result = reductionFusionPass_.apply(sumNode);
  // Fusion must be rejected because v1 and v2 have different shapes.
  ASSERT_FALSE(result->isFusedReduceElementwise());
  ASSERT_EQ(result, sumNode);
}

// ==================== Numerical Correctness Tests ====================

namespace {

NodePtr makeValueF32(const Shape& shape, std::vector<float> data) {
  Tensor bare = toTensor<CPUTensor>(shape, dtype::f32, data.data(), Location::Host);
  return ValueNode::create(std::move(bare));
}

void expectClose(
    const Tensor& result,
    const std::vector<float>& expected,
    float tol = 1e-5f) {
  ASSERT_EQ(result.elements(), static_cast<long long>(expected.size()));
  std::vector<float> actual(expected.size());
  result.hostSync(actual.data());
  for (size_t i = 0; i < expected.size(); i++) {
    ASSERT_NEAR(actual[i], expected[i], tol) << "at index " << i;
  }
}

} // namespace

class JitReductionOfElemFusionNumericalTest : public ::testing::Test {
 protected:
  JitReductionOfElemFusionNumericalTest()
      : evaluator_(makeBackendDispatch(CPUBackend::getInstance())) {}
  ReductionOfElementwiseFusion reductionFusionPass_;
  Evaluator evaluator_;
};

TEST_F(JitReductionOfElemFusionNumericalTest, NumericalReduceOfUnary) {
  // sum(exp(x)) over all axes
  // x = [1, 2, 3, 4], exp(x) = [e^1, e^2, e^3, e^4], sum = e+e^2+e^3+e^4
  Shape shape({4});
  auto x = makeValueF32(shape, {1, 2, 3, 4});
  auto expNode = UnaryNode::create(x, UnaryOp::Exp);
  auto sumNode = ReductionNode::create(expNode, ReductionOp::Sum, {}, false);
  auto result = reductionFusionPass_.apply(sumNode);
  evaluator_.eval(result);
  float expectedVal =
      std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f) + std::exp(4.0f);
  expectClose(result->getResult().value(), {expectedVal}, 1e-3f);
}

TEST_F(JitReductionOfElemFusionNumericalTest, NumericalReduceOfBinary) {
  // sum(x + y, axis=0) where x and y are {3, 2}
  // x = [[1,2],[3,4],[5,6]], y = [[10,20],[30,40],[50,60]]
  // x+y = [[11,22],[33,44],[55,66]]
  // sum(axis=0) = [11+33+55, 22+44+66] = [99, 132]
  Shape shape({3, 2});
  auto x = makeValueF32(shape, {1, 2, 3, 4, 5, 6});
  auto y = makeValueF32(shape, {10, 20, 30, 40, 50, 60});
  auto addNode = BinaryNode::create(x, y, BinaryOp::Add);
  auto sumNode = ReductionNode::create(addNode, ReductionOp::Sum, {0}, false);
  auto result = reductionFusionPass_.apply(sumNode);
  evaluator_.eval(result);
  expectClose(result->getResult().value(), {99, 132}, 1e-4f);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  init();
  return RUN_ALL_TESTS();
}
