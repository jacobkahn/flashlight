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
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"
#include "flashlight/fl/tensor/opt/passes/ElementwiseFusion.h"

using namespace fl;

class JitElementwiseFusionTest : public ::testing::Test {
 protected:
  TensorBackend& defaultBackend_ = DefaultTensorBackend_t::getInstance();
  ElementwiseFusion fusionPass_;
};

TEST_F(JitElementwiseFusionTest, noFusionSingleNode) {
  // v1 -> exp
  // exp has external uses so can't be absorbed,
  // and since it's a single node, there's nothing to fuse.
  Shape shape({3, 4});
  const auto v1 = ValueNode::create(defaultBackend_.full(shape, 1.0f, dtype::f32));
  const auto expNode = UnaryNode::create(v1, UnaryOp::Exp);
  const auto result = fusionPass_.apply(expNode);
  // No fusion -- single elementwise node with no fusable input
  ASSERT_EQ(result, expNode);
}

TEST_F(JitElementwiseFusionTest, fuseUnaryChain) {
  // v1 -> exp -> neg -> result
  // exp is single-use, no external uses => absorbable
  Shape shape({3, 4});
  const auto v1 = ValueNode::create(defaultBackend_.full(shape, 1.0f, dtype::f32));
  const auto expNode = UnaryNode::create(v1, UnaryOp::Exp);
  const auto negNode = UnaryNode::create(expNode, UnaryOp::Negative);
  const auto result = fusionPass_.apply(negNode);
  // Should fuse exp + neg into a single FusedElementwiseNode
  ASSERT_TRUE(result->isFusedElementwise());
  const auto& fused = result->impl<FusedElementwiseNode>();
  ASSERT_EQ(fused.numExternalInputs(), 1u); // v1
  ASSERT_EQ(fused.numSteps(), 2u);
  // Step 0: Exp(s0) -> s1
  ASSERT_TRUE(fused.program()[0].isUnary());
  ASSERT_EQ(fused.program()[0].unaryOp(), UnaryOp::Exp);
  ASSERT_EQ(fused.program()[0].src0, 0u);
  // Step 1: Negative(s1) -> s2
  ASSERT_TRUE(fused.program()[1].isUnary());
  ASSERT_EQ(fused.program()[1].unaryOp(), UnaryOp::Negative);
  ASSERT_EQ(fused.program()[1].src0, 1u); // output of step 0
  ASSERT_EQ(fused.shape(), shape);
  ASSERT_EQ(fused.inputs(), NodeList({v1}));
}

TEST_F(JitElementwiseFusionTest, fuseBinaryAndUnary) {
  // v1   v2
  //   \ /
  //   add -> exp -> result
  // add is single-use, absorbable into exp's group
  Shape shape({3, 4});
  const auto v1 = ValueNode::create(defaultBackend_.full(shape, 1.0f, dtype::f32));
  const auto v2 = ValueNode::create(defaultBackend_.full(shape, 2.0f, dtype::f32));
  const auto addNode = BinaryNode::create(v1, v2, BinaryOp::Add);
  const auto expNode = UnaryNode::create(addNode, UnaryOp::Exp);
  const auto result = fusionPass_.apply(expNode);
  ASSERT_TRUE(result->isFusedElementwise());
  const auto& fused = result->impl<FusedElementwiseNode>();
  ASSERT_EQ(fused.numExternalInputs(), 2u); // v1, v2
  ASSERT_EQ(fused.numSteps(), 2u);
  // Step 0: Add(s0, s1) -> s2
  ASSERT_TRUE(fused.program()[0].isBinary());
  ASSERT_EQ(fused.program()[0].binaryOp(), BinaryOp::Add);
  // Step 1: Exp(s2) -> s3
  ASSERT_TRUE(fused.program()[1].isUnary());
  ASSERT_EQ(fused.program()[1].unaryOp(), UnaryOp::Exp);
  ASSERT_EQ(fused.shape(), shape);
}

TEST_F(JitElementwiseFusionTest, noFusionMultipleUsers) {
  // v1 -> exp -> neg
  //           \-> add(exp, v2)
  // exp has 2 uses, so it's NOT absorbable.
  // neg is a single node with no absorbable input.
  // add's input exp is not absorbable.
  Shape shape({3, 4});
  const auto v1 = ValueNode::create(defaultBackend_.full(shape, 1.0f, dtype::f32));
  const auto v2 = ValueNode::create(defaultBackend_.full(shape, 2.0f, dtype::f32));
  const auto expNode = UnaryNode::create(v1, UnaryOp::Exp);
  const auto negNode = UnaryNode::create(expNode, UnaryOp::Negative);
  const auto addNode = BinaryNode::create(expNode, v2, BinaryOp::Add);

  // Apply to addNode (has 2 users of exp)
  const auto result = fusionPass_.apply(addNode);
  // exp has 2 uses so is not absorbable -- no fusion
  ASSERT_EQ(result, addNode);
}

TEST_F(JitElementwiseFusionTest, noFusionNonElementwise) {
  // v1 -> custom -> exp
  // custom is not elementwise, so exp can't absorb it
  Shape shape({3, 4});
  const auto v1 = ValueNode::create(defaultBackend_.full(shape, 1.0f, dtype::f32));
  const auto customNode = CustomNode::create(
      "identity",
      {v1},
      shape,
      dtype::f32,
      [](const std::vector<const Tensor*>& inputs) { return inputs[0]->copy(); });
  const auto expNode = UnaryNode::create(customNode, UnaryOp::Exp);
  const auto result = fusionPass_.apply(expNode);
  // custom is not fusable, exp is alone -- no fusion
  ASSERT_EQ(result, expNode);
}

TEST_F(JitElementwiseFusionTest, fuseThreeNodeChain) {
  // v1 -> exp -> neg -> sqrt -> result
  Shape shape({3, 4});
  const auto v1 = ValueNode::create(defaultBackend_.full(shape, 1.0f, dtype::f32));
  const auto expNode = UnaryNode::create(v1, UnaryOp::Exp);
  const auto negNode = UnaryNode::create(expNode, UnaryOp::Negative);
  const auto sqrtNode = UnaryNode::create(negNode, UnaryOp::Sqrt);
  const auto result = fusionPass_.apply(sqrtNode);
  ASSERT_TRUE(result->isFusedElementwise());
  const auto& fused = result->impl<FusedElementwiseNode>();
  ASSERT_EQ(fused.numExternalInputs(), 1u);
  ASSERT_EQ(fused.numSteps(), 3u);
  ASSERT_EQ(fused.program()[0].unaryOp(), UnaryOp::Exp);
  ASSERT_EQ(fused.program()[1].unaryOp(), UnaryOp::Negative);
  ASSERT_EQ(fused.program()[2].unaryOp(), UnaryOp::Sqrt);
}

TEST_F(JitElementwiseFusionTest, fuseDiamond) {
  //    v1
  //   /  \
  //  exp  neg
  //   \  /
  //   add -> result
  // exp and neg are both single-use, absorbable
  Shape shape({3, 4});
  const auto v1 = ValueNode::create(defaultBackend_.full(shape, 1.0f, dtype::f32));
  const auto expNode = UnaryNode::create(v1, UnaryOp::Exp);
  const auto negNode = UnaryNode::create(v1, UnaryOp::Negative);
  const auto addNode = BinaryNode::create(expNode, negNode, BinaryOp::Add);
  const auto result = fusionPass_.apply(addNode);
  ASSERT_TRUE(result->isFusedElementwise());
  const auto& fused = result->impl<FusedElementwiseNode>();
  ASSERT_EQ(fused.numExternalInputs(), 1u); // just v1
  ASSERT_EQ(fused.numSteps(), 3u); // exp, neg, add
  ASSERT_EQ(fused.shape(), shape);
}

TEST_F(JitElementwiseFusionTest, noFusionBroadcastInputs) {
  // v1({4,8}) + v2({1,8}) -> exp
  // add has broadcast: v2 shape != output shape {4,8}.
  // Fused kernels use flat indexing with no broadcast handling,
  // so the pass must reject this fusion.
  Shape bigShape({4, 8});
  Shape smallShape({1, 8});
  const auto v1 = ValueNode::create(defaultBackend_.full(bigShape, 1.0f, dtype::f32));
  const auto v2 = ValueNode::create(defaultBackend_.full(smallShape, 2.0f, dtype::f32));
  const auto addNode = BinaryNode::create(v1, v2, BinaryOp::Add);
  ASSERT_EQ(addNode->shape(), bigShape); // broadcast output
  const auto expNode = UnaryNode::create(addNode, UnaryOp::Exp);
  const auto result = fusionPass_.apply(expNode);
  // Fusion must be rejected because v2 has a different shape from the output.
  ASSERT_FALSE(result->isFusedElementwise());
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

class JitElementwiseFusionNumericalTest : public ::testing::Test {
 protected:
  JitElementwiseFusionNumericalTest()
      : evaluator_(makeBackendDispatch(CPUBackend::getInstance())) {}
  ElementwiseFusion fusionPass_;
  Evaluator evaluator_;
};

TEST_F(JitElementwiseFusionNumericalTest, NumericalUnaryChain) {
  // neg(exp(x))
  Shape shape({4});
  auto x = makeValueF32(shape, {0, 1, 2, 3});
  auto expNode = UnaryNode::create(x, UnaryOp::Exp);
  auto negNode = UnaryNode::create(expNode, UnaryOp::Negative);
  auto result = fusionPass_.apply(negNode);
  evaluator_.eval(result);
  expectClose(
      result->getResult().value(),
      {-std::exp(0.0f), -std::exp(1.0f), -std::exp(2.0f), -std::exp(3.0f)},
      1e-4f);
}

TEST_F(JitElementwiseFusionNumericalTest, NumericalBinaryUnaryMix) {
  // exp(x) + y
  Shape shape({4});
  auto x = makeValueF32(shape, {0, 1, 2, 3});
  auto y = makeValueF32(shape, {10, 20, 30, 40});
  auto expNode = UnaryNode::create(x, UnaryOp::Exp);
  auto addNode = BinaryNode::create(expNode, y, BinaryOp::Add);
  auto result = fusionPass_.apply(addNode);
  evaluator_.eval(result);
  expectClose(
      result->getResult().value(),
      {std::exp(0.0f) + 10, std::exp(1.0f) + 20,
       std::exp(2.0f) + 30, std::exp(3.0f) + 40},
      1e-4f);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  init();
  return RUN_ALL_TESTS();
}
