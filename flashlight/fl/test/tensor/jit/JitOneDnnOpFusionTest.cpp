/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <functional>
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
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"
#include "flashlight/fl/tensor/opt/backends/onednn/OneDnnOpFusion.h"

using namespace fl;

class JitOneDnnOpFusionTest : public ::testing::Test {
 protected:
  OneDnnOpFusion oneDnnFuser_;
};

TEST_F(JitOneDnnOpFusionTest, singleScalarNode) {
  // c1
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto res = oneDnnFuser_.apply(c1);
  // nothing changes
  ASSERT_EQ(res, c1);
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({}));
}

TEST_F(JitOneDnnOpFusionTest, singleBinaryNode) {
  // c1  c2
  //  \  /
  //   add
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto c2 = ScalarNode::create(shape, dtype, 2);
  const auto add = BinaryNode::create(c1, c2, BinaryOp::Add);
  const auto res = oneDnnFuser_.apply(add);
  // nothing changes
  ASSERT_EQ(res, add);
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({{add, 0}}));
  ASSERT_EQ(c2->inputs(), NodeList({}));
  ASSERT_EQ(c2->uses(), UseValList({{add, 1}}));
  ASSERT_EQ(add->inputs(), NodeList({c1, c2}));
  ASSERT_EQ(add->uses(), UseValList({}));
}

TEST_F(JitOneDnnOpFusionTest, sharedBinaryNode) {
  //   c1
  //  /  \
  //  \  /
  //   sub
  //  /  \
  //  \  /
  //   add
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto sub = BinaryNode::create(c1, c1, BinaryOp::Sub);
  const auto add = BinaryNode::create(sub, sub, BinaryOp::Add);
  const auto res = oneDnnFuser_.apply(add);
  // nothing changes
  ASSERT_EQ(res, add);
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({{sub, 0}, {sub, 1}}));
  ASSERT_EQ(sub->inputs(), NodeList({c1, c1}));
  ASSERT_EQ(sub->uses(), UseValList({{add, 0}, {add, 1}}));
  ASSERT_EQ(add->inputs(), NodeList({sub, sub}));
  ASSERT_EQ(add->uses(), UseValList({}));
}

TEST_F(JitOneDnnOpFusionTest, nonFusableRoot) {
  // c1  c2
  //  \  /
  //   mul  c3
  //    \  /
  //     add
  //      |
  //    custom
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto c2 = ScalarNode::create(shape, dtype, 2);
  const auto c3 = ScalarNode::create(shape, dtype, 3);
  const auto mul = BinaryNode::create(c1, c2, BinaryOp::Mul);
  const auto add = BinaryNode::create(mul, c3, BinaryOp::Add);
  const auto custom = CustomNode::create(
      "identity", {add}, shape, dtype, [](const std::vector<const Tensor*>& inputs) {
        return *inputs[0];
      });
  // c1  c2
  //  \  /
  //   mul  c3            c1 c2  c3       ..... (same as before)
  //    \  /               \  |  /         \ /
  //     add      ---->  fusedCustomNode   add
  //      |                   |
  //    custom              custom
  ASSERT_EQ(custom, oneDnnFuser_.apply(custom));
  const auto fusedNode = custom->inputs().at(0);
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({{mul, 0}, {fusedNode, 0}}));
  ASSERT_EQ(c2->inputs(), NodeList({}));
  ASSERT_EQ(c2->uses(), UseValList({{mul, 1}, {fusedNode, 1}}));
  ASSERT_EQ(c3->inputs(), NodeList({}));
  ASSERT_EQ(c3->uses(), UseValList({{add, 1}, {fusedNode, 2}}));
  ASSERT_EQ(mul->inputs(), NodeList({c1, c2}));
  ASSERT_EQ(mul->uses(), UseValList({{add, 0}}));
  ASSERT_EQ(add->inputs(), NodeList({mul, c3}));
  ASSERT_EQ(add->uses(), UseValList({})); // no more use, replaced by fused node
  ASSERT_EQ(fusedNode->inputs(), NodeList({c1, c2, c3}));
  ASSERT_EQ(fusedNode->uses(), UseValList({{custom, 0}}));
  ASSERT_TRUE(fusedNode->isCustom());
  ASSERT_EQ(custom->inputs(), NodeList({fusedNode}));
  ASSERT_EQ(custom->uses(), UseValList({}));
}

TEST_F(JitOneDnnOpFusionTest, nestedFusableChains) {
  // c2   c3
  //  \  /
  //   sub  c4
  //     \  /
  //  c1  div
  //   \  /
  //    mul  c5
  //     \  /
  //      add
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto c2 = ScalarNode::create(shape, dtype, 2);
  const auto c3 = ScalarNode::create(shape, dtype, 3);
  const auto c4 = ScalarNode::create(shape, dtype, 4);
  const auto c5 = ScalarNode::create(shape, dtype, 5);
  const auto sub = BinaryNode::create(c2, c3, BinaryOp::Sub);
  const auto div = BinaryNode::create(sub, c4, BinaryOp::Div);
  const auto mul = BinaryNode::create(c1, div, BinaryOp::Mul);
  const auto add = BinaryNode::create(mul, c5, BinaryOp::Add);
  // With commutative swap: mul is commutative, c1 (LHS) is not fusable,
  // div (RHS) is fusable. So c1 becomes a binary post-op RHS and the pass
  // descends into div, extending the fusion chain. The entire expression
  // becomes a single fused node:
  //   primary: sub(c2, c3)
  //   post-op: div(c4)
  //   post-op: mul(c1)  -- swapped
  //   post-op: add(c5)
  //
  // c2   c3
  //  \  /
  //   sub  c4                    c2  c3  c4  c1  c5
  //     \  /                       \  |  |  |  /
  //  c1  div     ---->          fusedCustomRoot
  //   \  /
  //    mul  c5
  //     \  /
  //      add
  const auto fusedCustomRoot = oneDnnFuser_.apply(add);
  // All 5 scalars are inputs to the single fused node
  ASSERT_EQ(fusedCustomRoot->inputs(), NodeList({c2, c3, c4, c1, c5}));
  ASSERT_TRUE(fusedCustomRoot->isCustom());
  ASSERT_EQ(fusedCustomRoot->shape(), shape);
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({{mul, 0}, {fusedCustomRoot, 3}}));
  ASSERT_EQ(c2->inputs(), NodeList({}));
  ASSERT_EQ(c2->uses(), UseValList({{sub, 0}, {fusedCustomRoot, 0}}));
  ASSERT_EQ(c3->inputs(), NodeList({}));
  ASSERT_EQ(c3->uses(), UseValList({{sub, 1}, {fusedCustomRoot, 1}}));
  ASSERT_EQ(c4->inputs(), NodeList({}));
  ASSERT_EQ(c4->uses(), UseValList({{div, 1}, {fusedCustomRoot, 2}}));
  ASSERT_EQ(c5->inputs(), NodeList({}));
  ASSERT_EQ(c5->uses(), UseValList({{add, 1}, {fusedCustomRoot, 4}}));
  // Original intermediate nodes still exist but have no fused-node uses
  ASSERT_EQ(sub->inputs(), NodeList({c2, c3}));
  ASSERT_EQ(sub->uses(), UseValList({{div, 0}}));
  ASSERT_EQ(div->inputs(), NodeList({sub, c4}));
  ASSERT_EQ(div->uses(), UseValList({{mul, 1}}));
  ASSERT_EQ(mul->inputs(), NodeList({c1, div}));
  ASSERT_EQ(mul->uses(), UseValList({{add, 0}}));
  ASSERT_EQ(add->inputs(), NodeList({mul, c5}));
  ASSERT_EQ(add->uses(), UseValList({}));
  ASSERT_EQ(fusedCustomRoot->uses(), UseValList({}));
  ASSERT_EQ(fusedCustomRoot->shape(), shape);
  ASSERT_TRUE(fusedCustomRoot->isCustom());
}

TEST_F(JitOneDnnOpFusionTest, commutativeSwap) {
  // c1   c2
  //  \  /
  //   sub     c3    sub is single-use, fusable chain on RHS of add
  //    \     /
  //     add         add is commutative; LHS (c3) is leaf, RHS (sub) is fusable
  //                 --> swap: walk down sub, c3 becomes post-op arg
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto c2 = ScalarNode::create(shape, dtype, 2);
  const auto c3 = ScalarNode::create(shape, dtype, 3);
  const auto sub = BinaryNode::create(c1, c2, BinaryOp::Sub);
  const auto add = BinaryNode::create(c3, sub, BinaryOp::Add);
  // Without commutativity: LHS (c3) is not fusable --> chain = [add], length 1
  //   --> no fusion (min 2 required).
  // With commutativity: swap add(c3, sub) --> walk down sub, c3 is post-op RHS
  //   --> chain = [sub, add(c3)], length 2 --> fuse!
  const auto fusedRoot = oneDnnFuser_.apply(add);
  ASSERT_TRUE(fusedRoot->isCustom());
  // fused node inputs: c1 (sub LHS), c2 (sub RHS = primary binop RHS), c3 (add post-op RHS)
  ASSERT_EQ(fusedRoot->inputs(), NodeList({c1, c2, c3}));
  ASSERT_EQ(fusedRoot->shape(), shape);
}

TEST_F(JitOneDnnOpFusionTest, nonCommutativeNoSwap) {
  // c1   c2
  //  \  /
  //   mul     c3    mul is single-use, fusable chain on RHS of div
  //    \     /
  //     div         div is NOT commutative; can only walk LHS (c3, leaf)
  //                 --> chain = [div], length 1 --> no fusion
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto c2 = ScalarNode::create(shape, dtype, 2);
  const auto c3 = ScalarNode::create(shape, dtype, 3);
  const auto mul = BinaryNode::create(c1, c2, BinaryOp::Mul);
  const auto div = BinaryNode::create(c3, mul, BinaryOp::Div);
  const auto res = oneDnnFuser_.apply(div);
  // No fusion -- tree unchanged
  ASSERT_EQ(res, div);
  ASSERT_EQ(div->inputs(), NodeList({c3, mul}));
  ASSERT_EQ(mul->inputs(), NodeList({c1, c2}));
}

TEST_F(JitOneDnnOpFusionTest, unaryInChain) {
  // c1   c2
  //  \  /
  //   mul         binary primary (bottommost)
  //    |
  //   exp    c3   eltwise post-op
  //     \   /
  //      add      binary post-op
  //
  // Chain walk (top-down): add --> exp --> mul --> c1 (stop)
  // Accumulated: [{c3, Add}, eltwise_exp, {c2, Mul}]
  // Reversed: [{c2, Mul}, eltwise_exp, {c3, Add}]
  // Primary: binary_mul(c1, c2), post-ops: [eltwise_exp, binary_add(c3)]
  // --> fused CustomNode(c1, c2, c3)
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto c2 = ScalarNode::create(shape, dtype, 2);
  const auto c3 = ScalarNode::create(shape, dtype, 3);
  const auto mul = BinaryNode::create(c1, c2, BinaryOp::Mul);
  const auto exp = UnaryNode::create(mul, UnaryOp::Exp);
  const auto add = BinaryNode::create(exp, c3, BinaryOp::Add);
  const auto fusedRoot = oneDnnFuser_.apply(add);
  ASSERT_TRUE(fusedRoot->isCustom());
  ASSERT_EQ(fusedRoot->inputs(), NodeList({c1, c2, c3}));
  ASSERT_EQ(fusedRoot->shape(), shape);
}

TEST_F(JitOneDnnOpFusionTest, maxPostOpsLimit) {
  // Build a chain of 35 binary adds: c0 + c1 + c2 + ... + c34
  // The OneDNN limit is 32 post-ops. The pass should stop accumulating
  // at the limit and not exceed it. With 35 ops in the chain, the pass
  // should split into multiple fused nodes or stop early.
  Shape shape(Shape({2, 2}));
  auto dt = dtype::f32;
  constexpr unsigned kChainLength = 35;

  std::vector<NodePtr> scalars;
  for (unsigned i = 0; i <= kChainLength; i++) {
    scalars.push_back(ScalarNode::create(shape, dt, static_cast<int>(i)));
  }

  // Build chain: add(add(add(c0, c1), c2), c3) ...
  NodePtr chain = BinaryNode::create(scalars[0], scalars[1], BinaryOp::Add);
  for (unsigned i = 2; i <= kChainLength; i++) {
    chain = BinaryNode::create(chain, scalars[i], BinaryOp::Add);
  }

  // Should not crash -- the pass must respect the post-op limit.
  const auto result = oneDnnFuser_.apply(chain);
  // The result should be a valid node (fused or partially fused).
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->shape(), shape);
}

// ==================== Numerical Correctness Tests ====================
// These tests verify that the fused CustomNode produced by OneDnnOpFusion
// computes the correct numerical result, not just the correct graph shape.

namespace {

// Create a bare CPUTensor from float data wrapped in a ValueNode.
NodePtr makeValueF32(const Shape& shape, std::vector<float> data) {
  Tensor bare = toTensor<CPUTensor>(shape, dtype::f32, data.data(), Location::Host);
  return ValueNode::create(std::move(bare));
}

// Extract float data from a bare CPUTensor result and compare against expected.
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

class JitOneDnnOpFusionNumericalTest : public ::testing::Test {
 protected:
  JitOneDnnOpFusionNumericalTest()
      : evaluator_(makeBackendDispatch(CPUBackend::getInstance())) {}
  OneDnnOpFusion oneDnnFuser_;
  Evaluator evaluator_;
};

TEST_F(JitOneDnnOpFusionNumericalTest, NumericalBinaryChain) {
  // (a + b) * c  with real f32 data
  // Chain walk: mul -> add -> a (leaf), so chain = [add, mul], fused.
  Shape shape({4});
  auto a = makeValueF32(shape, {1, 2, 3, 4});
  auto b = makeValueF32(shape, {10, 20, 30, 40});
  auto c = makeValueF32(shape, {2, 2, 2, 2});
  auto add = BinaryNode::create(a, b, BinaryOp::Add);
  auto mul = BinaryNode::create(add, c, BinaryOp::Mul);
  auto fused = oneDnnFuser_.apply(mul);
  ASSERT_TRUE(fused->isCustom());
  evaluator_.eval(fused);
  expectClose(fused->getResult().value(), {22, 44, 66, 88});
}

TEST_F(JitOneDnnOpFusionNumericalTest, NumericalUnaryInChain) {
  // exp(a * b) + c
  // Chain walk: add -> exp -> mul -> a (leaf)
  // chain = [mul, exp, add(c)], fused.
  Shape shape({4});
  auto a = makeValueF32(shape, {0, 0, 0, 0});
  auto b = makeValueF32(shape, {1, 1, 1, 1});
  auto c = makeValueF32(shape, {10, 20, 30, 40});
  auto mul = BinaryNode::create(a, b, BinaryOp::Mul);
  auto exp = UnaryNode::create(mul, UnaryOp::Exp);
  auto add = BinaryNode::create(exp, c, BinaryOp::Add);
  auto fused = oneDnnFuser_.apply(add);
  ASSERT_TRUE(fused->isCustom());
  evaluator_.eval(fused);
  // exp(0*1) + c = 1 + c
  expectClose(fused->getResult().value(), {11, 21, 31, 41});
}

TEST_F(JitOneDnnOpFusionNumericalTest, NumericalCommutativeSwap) {
  // c + (a - b)
  // sub is fusable chain on RHS of add; add is commutative -> swap
  // chain = [sub, add(c)], fused.
  Shape shape({4});
  auto a = makeValueF32(shape, {10, 20, 30, 40});
  auto b = makeValueF32(shape, {1, 2, 3, 4});
  auto c = makeValueF32(shape, {100, 200, 300, 400});
  auto sub = BinaryNode::create(a, b, BinaryOp::Sub);
  auto add = BinaryNode::create(c, sub, BinaryOp::Add);
  auto fused = oneDnnFuser_.apply(add);
  ASSERT_TRUE(fused->isCustom());
  evaluator_.eval(fused);
  // c + (a - b) = 100+(10-1), 200+(20-2), ...
  expectClose(fused->getResult().value(), {109, 218, 327, 436});
}

TEST_F(JitOneDnnOpFusionNumericalTest, NumericalLongerChain) {
  // tanh(a * b + c)
  // Chain: add -> mul -> a (leaf), post-ops: [mul(b), add(c), tanh]
  Shape shape({4});
  auto a = makeValueF32(shape, {1, 2, 3, 4});
  auto b = makeValueF32(shape, {0.1f, 0.2f, 0.3f, 0.4f});
  auto c = makeValueF32(shape, {0, 0, 0, 0});
  auto mul = BinaryNode::create(a, b, BinaryOp::Mul);
  auto add = BinaryNode::create(mul, c, BinaryOp::Add);
  auto tanhNode = UnaryNode::create(add, UnaryOp::Tanh);
  auto fused = oneDnnFuser_.apply(tanhNode);
  // The chain may or may not fuse depending on chain length (tanh->add->mul = 3 ops).
  // Either way, evaluation should produce correct results.
  evaluator_.eval(fused);
  expectClose(
      fused->getResult().value(),
      {std::tanh(0.1f), std::tanh(0.4f), std::tanh(0.9f), std::tanh(1.6f)},
      1e-4f);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  init();
  return RUN_ALL_TESTS();
}
