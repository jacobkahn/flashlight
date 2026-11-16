/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <functional>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/JitTensorBase.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/ClipNode.h"
#include "flashlight/fl/tensor/ir/ConcatenateNode.h"
#include "flashlight/fl/tensor/ir/CumsumNode.h"
#include "flashlight/fl/tensor/ir/FlipNode.h"
#include "flashlight/fl/tensor/ir/IndexedUpdateNode.h"
#include "flashlight/fl/tensor/ir/NodeUtils.h"
#include "flashlight/fl/tensor/ir/PadNode.h"
#include "flashlight/fl/tensor/ir/RollNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/TileNode.h"
#include "flashlight/fl/tensor/ir/TriangularNode.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"
#include "flashlight/fl/tensor/ir/WhereNode.h"

using namespace fl;

namespace {

class JitTensorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fl::setDefaultJitTensorType<DefaultBaseTensorType_t>();
  }
  TensorBackend& defaultBackend_ = DefaultTensorBackend_t::getInstance();
};

template <typename Op>
void testBinaryOp(Op func, BinaryOp op) {
  // c0  c1
  //  \  /
  //  node
  Shape shape({2, 2});
  auto dtype = dtype::s32;
  const auto t0 = full(shape, 0, dtype);
  const auto t1 = full(shape, 1, dtype);
  const auto c0 = toJitTensorBase(t0).node();
  const auto c1 = toJitTensorBase(t1).node();
  const auto tensor = func(t0, t1);
  const auto& jitTensor = toJitTensorBase(tensor);
  const auto node = jitTensor.node();
  const auto& binaryNode = node->template impl<BinaryNode>();
  ASSERT_EQ(node->inputs(), NodeList({c0, c1}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(binaryNode.lhs(), c0);
  ASSERT_EQ(binaryNode.rhs(), c1);
  ASSERT_EQ(binaryNode.op(), op);
  ASSERT_EQ(node->shape(), shape);
  ASSERT_EQ(c0->uses(), UseValList({{node, 0}}));
  ASSERT_EQ(c1->uses(), UseValList({{node, 1}}));
}

struct ShiftLeftFunctor {
  auto operator()(const Tensor& lhs, const Tensor& rhs) {
    return lhs << rhs;
  }
};

struct ShiftRightFunctor {
  auto operator()(const Tensor& lhs, const Tensor& rhs) {
    return lhs >> rhs;
  }
};

struct PowerFunctor {
  auto operator()(const Tensor& lhs, const Tensor& rhs) {
    return fl::power(lhs, rhs);
  }
};

struct MaxFunctor {
  auto operator()(const Tensor& lhs, const Tensor& rhs) {
    return fl::maximum(lhs, rhs);
  }
};

struct MinFunctor {
  auto operator()(const Tensor& lhs, const Tensor& rhs) {
    return fl::minimum(lhs, rhs);
  }
};

} // namespace

TEST_F(JitTensorTest, constructor) {
  const auto dataTensor = defaultBackend_.rand(Shape({2, 2}), dtype::f32);
  const auto data = dataTensor.toHostVectorSync<float>();
  const Tensor tensor =
      Tensor::fromBuffer(dataTensor.shape(), data.data(), Location::Host);
  const auto& jitTensor = toJitTensorBase(tensor);
  const auto node = jitTensor.node();
  ASSERT_EQ(node->inputs(), NodeList({}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(node->shape(), dataTensor.shape());
  ASSERT_TRUE(node->isValue());
  ASSERT_TRUE(allClose(dataTensor, node->getResult().value()));
}

TEST_F(JitTensorTest, full) {
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  int val = 42;
  const auto tensor = full(shape, val, dtype);
  const auto& jitTensor = toJitTensorBase(tensor);
  const auto& node = jitTensor.node()->impl<ScalarNode>();
  ASSERT_EQ(node.inputs(), NodeList({}));
  ASSERT_EQ(node.uses(), UseValList({}));
  ASSERT_EQ(node.shape(), shape);
  ASSERT_EQ(node.dataType(), dtype);
  ASSERT_EQ(node.scalar<int>(), val);
}

TEST_F(JitTensorTest, add) {
  testBinaryOp(std::plus<>(), BinaryOp::Add);
}

TEST_F(JitTensorTest, sub) {
  testBinaryOp(std::minus<>(), BinaryOp::Sub);
}

TEST_F(JitTensorTest, mul) {
  testBinaryOp(std::multiplies<>(), BinaryOp::Mul);
}

TEST_F(JitTensorTest, div) {
  testBinaryOp(std::divides<>(), BinaryOp::Div);
}

TEST_F(JitTensorTest, eq) {
  testBinaryOp(std::equal_to<>(), BinaryOp::Eq);
}

TEST_F(JitTensorTest, neq) {
  testBinaryOp(std::not_equal_to<>(), BinaryOp::Neq);
}

TEST_F(JitTensorTest, gt) {
  testBinaryOp(std::greater<>(), BinaryOp::Gt);
}

TEST_F(JitTensorTest, gte) {
  testBinaryOp(std::greater_equal<>(), BinaryOp::Gte);
}

TEST_F(JitTensorTest, lt) {
  testBinaryOp(std::less<>(), BinaryOp::Lt);
}

TEST_F(JitTensorTest, mod) {
  testBinaryOp(std::modulus<>(), BinaryOp::Mod);
}

TEST_F(JitTensorTest, and) {
  testBinaryOp(std::logical_and<>(), BinaryOp::And);
}

TEST_F(JitTensorTest, or) {
  testBinaryOp(std::logical_and<>(), BinaryOp::And);
}

TEST_F(JitTensorTest, bitwiseAnd) {
  testBinaryOp(std::bit_and<>(), BinaryOp::BitAnd);
}

TEST_F(JitTensorTest, bitwiseOr) {
  testBinaryOp(std::bit_or<>(), BinaryOp::BitOr);
}

TEST_F(JitTensorTest, bitwiseXor) {
  testBinaryOp(std::bit_xor<>(), BinaryOp::BitXor);
}

TEST_F(JitTensorTest, shiftLeft) {
  testBinaryOp(ShiftLeftFunctor(), BinaryOp::Shl);
}

TEST_F(JitTensorTest, shiftRight) {
  testBinaryOp(ShiftRightFunctor(), BinaryOp::Shr);
}

TEST_F(JitTensorTest, max) {
  testBinaryOp(MaxFunctor(), BinaryOp::Max);
}

TEST_F(JitTensorTest, min) {
  testBinaryOp(MinFunctor(), BinaryOp::Min);
}

TEST_F(JitTensorTest, pow) {
  testBinaryOp(PowerFunctor(), BinaryOp::Pow);
}

TEST_F(JitTensorTest, tileBuildsTileNode) {
  auto input = full({2, 3}, 1, dtype::s32);
  auto inputNode = toJitTensorBase(input).node();
  auto tiled = tile(input, {2});
  auto node = toJitTensorBase(tiled).node();
  ASSERT_TRUE(node->isTile());
  auto& tileNode = node->impl<TileNode>();
  ASSERT_EQ(tileNode.input(), inputNode);
  ASSERT_EQ(tileNode.tileDims(), Shape({2}));
  ASSERT_EQ(tileNode.shape(), Shape({4, 3}));
  ASSERT_EQ(inputNode->uses(), UseValList({{node, 0}}));
}

TEST_F(JitTensorTest, concatenateBuildsConcatenateNode) {
  auto a = full({2, 3}, 1, dtype::s32);
  auto b = full({2, 4}, 2, dtype::s32);
  auto aNode = toJitTensorBase(a).node();
  auto bNode = toJitTensorBase(b).node();
  auto result = concatenate({a, b}, 1);
  auto node = toJitTensorBase(result).node();
  ASSERT_TRUE(node->isConcatenate());
  auto& concatNode = node->impl<ConcatenateNode>();
  ASSERT_EQ(concatNode.input(0), aNode);
  ASSERT_EQ(concatNode.input(1), bNode);
  ASSERT_EQ(concatNode.numInputs(), 2u);
  ASSERT_EQ(concatNode.axis(), 1u);
  ASSERT_EQ(concatNode.shape(), Shape({2, 7}));
  ASSERT_EQ(concatNode.dataType(), dtype::s32);
  ASSERT_EQ(aNode->uses(), UseValList({{node, 0}}));
  ASSERT_EQ(bNode->uses(), UseValList({{node, 1}}));
}

TEST_F(JitTensorTest, padBuildsPadNode) {
  auto input = full({2, 3}, 1, dtype::s32);
  auto inputNode = toJitTensorBase(input).node();
  auto result = pad(input, {{1, 2}, {3, 4}}, PadType::Constant);
  auto node = toJitTensorBase(result).node();
  ASSERT_TRUE(node->isPad());
  auto& padNode = node->impl<PadNode>();
  ASSERT_EQ(padNode.input(), inputNode);
  ASSERT_EQ(
      padNode.padWidths(), (std::vector<std::pair<int, int>>{{1, 2}, {3, 4}}));
  ASSERT_EQ(padNode.padType(), PadType::Constant);
  ASSERT_EQ(padNode.shape(), Shape({5, 10}));
  ASSERT_EQ(padNode.dataType(), dtype::s32);
  ASSERT_EQ(inputNode->uses(), UseValList({{node, 0}}));
}

TEST_F(JitTensorTest, flipBuildsFlipNode) {
  auto input = full({2, 3}, 1, dtype::s32);
  auto inputNode = toJitTensorBase(input).node();
  auto flipped = fl::flip(input, 1);
  auto node = toJitTensorBase(flipped).node();
  ASSERT_TRUE(node->isFlip());
  auto& flipNode = node->impl<FlipNode>();
  ASSERT_EQ(flipNode.input(), inputNode);
  ASSERT_EQ(flipNode.axis(), 1u);
  ASSERT_EQ(flipNode.shape(), Shape({2, 3}));
  ASSERT_EQ(flipNode.dataType(), dtype::s32);
  ASSERT_EQ(inputNode->uses(), UseValList({{node, 0}}));
}

TEST_F(JitTensorTest, trilBuildsTriangularNode) {
  auto input = full({2, 3}, 1, dtype::s32);
  auto inputNode = toJitTensorBase(input).node();
  auto result = fl::tril(input, 1);
  auto node = toJitTensorBase(result).node();
  ASSERT_TRUE(node->isTriangular());
  auto& triangularNode = node->impl<TriangularNode>();
  ASSERT_EQ(triangularNode.input(), inputNode);
  ASSERT_EQ(triangularNode.mode(), TriangularMode::Lower);
  ASSERT_EQ(triangularNode.diagonal(), 1);
  ASSERT_EQ(triangularNode.shape(), Shape({2, 3}));
  ASSERT_EQ(triangularNode.dataType(), dtype::s32);
  ASSERT_EQ(inputNode->uses(), UseValList({{node, 0}}));
}

TEST_F(JitTensorTest, triuBuildsTriangularNode) {
  auto input = full({2, 3}, 1, dtype::s32);
  auto inputNode = toJitTensorBase(input).node();
  auto result = fl::triu(input, -1);
  auto node = toJitTensorBase(result).node();
  ASSERT_TRUE(node->isTriangular());
  auto& triangularNode = node->impl<TriangularNode>();
  ASSERT_EQ(triangularNode.input(), inputNode);
  ASSERT_EQ(triangularNode.mode(), TriangularMode::Upper);
  ASSERT_EQ(triangularNode.diagonal(), -1);
  ASSERT_EQ(triangularNode.shape(), Shape({2, 3}));
  ASSERT_EQ(triangularNode.dataType(), dtype::s32);
  ASSERT_EQ(inputNode->uses(), UseValList({{node, 0}}));
}

TEST_F(JitTensorTest, clipBuildsClipNode) {
  auto input = full({2, 3}, 2, dtype::s32);
  auto low = full({2, 3}, 0, dtype::s32);
  auto high = full({2, 3}, 1, dtype::s32);
  auto inputNode = toJitTensorBase(input).node();
  auto lowNode = toJitTensorBase(low).node();
  auto highNode = toJitTensorBase(high).node();
  auto clipped = fl::clip(input, low, high);
  auto node = toJitTensorBase(clipped).node();
  ASSERT_TRUE(node->isClip());
  auto& clipNode = node->impl<ClipNode>();
  ASSERT_EQ(clipNode.input(), inputNode);
  ASSERT_EQ(clipNode.low(), lowNode);
  ASSERT_EQ(clipNode.high(), highNode);
  ASSERT_EQ(clipNode.shape(), Shape({2, 3}));
  ASSERT_EQ(clipNode.dataType(), dtype::s32);
  ASSERT_EQ(inputNode->uses(), UseValList({{node, 0}}));
  ASSERT_EQ(lowNode->uses(), UseValList({{node, 1}}));
  ASSERT_EQ(highNode->uses(), UseValList({{node, 2}}));
}

TEST_F(JitTensorTest, cumsumBuildsCumsumNode) {
  auto input = full({2, 3}, 1, dtype::s32);
  auto inputNode = toJitTensorBase(input).node();
  auto result = fl::cumsum(input, 1);
  auto node = toJitTensorBase(result).node();
  ASSERT_TRUE(node->isCumsum());
  auto& cumsumNode = node->impl<CumsumNode>();
  ASSERT_EQ(cumsumNode.input(), inputNode);
  ASSERT_EQ(cumsumNode.axis(), 1u);
  ASSERT_EQ(cumsumNode.shape(), Shape({2, 3}));
  ASSERT_EQ(cumsumNode.dataType(), dtype::s32);
  ASSERT_EQ(inputNode->uses(), UseValList({{node, 0}}));
}

TEST_F(JitTensorTest, rollBuildsRollNode) {
  auto input = full({2, 3}, 1, dtype::s32);
  auto inputNode = toJitTensorBase(input).node();
  auto result = fl::roll(input, -1, 1);
  auto node = toJitTensorBase(result).node();
  ASSERT_TRUE(node->isRoll());
  auto& rollNode = node->impl<RollNode>();
  ASSERT_EQ(rollNode.input(), inputNode);
  ASSERT_EQ(rollNode.shift(), 2);
  ASSERT_EQ(rollNode.axis(), 1u);
  ASSERT_EQ(rollNode.shape(), Shape({2, 3}));
  ASSERT_EQ(rollNode.dataType(), dtype::s32);
  ASSERT_EQ(inputNode->uses(), UseValList({{node, 0}}));
}

TEST_F(JitTensorTest, whereBuildsWhereNode) {
  auto condition = full({2, 2}, 1, dtype::b8);
  auto x = full({2, 2}, 1, dtype::s32);
  auto y = full({2, 2}, 2, dtype::s32);
  auto conditionNode = toJitTensorBase(condition).node();
  auto xNode = toJitTensorBase(x).node();
  auto yNode = toJitTensorBase(y).node();
  auto result = where(condition, x, y);
  auto node = toJitTensorBase(result).node();
  ASSERT_TRUE(node->isWhere());
  auto& whereNode = node->impl<WhereNode>();
  ASSERT_EQ(whereNode.condition(), conditionNode);
  ASSERT_EQ(whereNode.x(), xNode);
  ASSERT_EQ(whereNode.y(), yNode);
  ASSERT_EQ(whereNode.shape(), Shape({2, 2}));
  ASSERT_EQ(whereNode.dataType(), dtype::s32);
  ASSERT_EQ(conditionNode->uses(), UseValList({{node, 0}}));
  ASSERT_EQ(xNode->uses(), UseValList({{node, 1}}));
  ASSERT_EQ(yNode->uses(), UseValList({{node, 2}}));
}

TEST_F(JitTensorTest, whereRejectsNonBooleanCondition) {
  auto condition = full({2, 2}, 1, dtype::s32);
  auto x = full({2, 2}, 1, dtype::s32);
  auto y = full({2, 2}, 2, dtype::s32);
  EXPECT_THROW(where(condition, x, y), std::invalid_argument);
}

TEST_F(JitTensorTest, explicitEval) {
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto t0 = full(shape, 11, dtype);
  const auto t1 = full(shape, 22, dtype);
  auto sum = t0 + t1;
  ASSERT_FALSE(toJitTensorBase(sum).node()->getResult().has_value());
  fl::eval(sum); // explicit eval call
  ASSERT_TRUE(allClose(
      toJitTensorBase(sum).node()->getResult().value(),
      defaultBackend_.full(shape, 33, dtype)));
}

TEST_F(JitTensorTest, forcedEval) {
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto t0 = full(shape, 11, dtype);
  const auto t1 = full(shape, 22, dtype);
  auto sum = t0 + t1;
  ASSERT_FALSE(toJitTensorBase(sum).node()->getResult().has_value());
  sum.toString(); // forces eval
  ASSERT_TRUE(allClose(
      toJitTensorBase(sum).node()->getResult().value(),
      defaultBackend_.full(shape, 33, dtype)));
}

TEST_F(JitTensorTest, assignment) {
  // we don't test the computation result (that's Evaluator's job) -- we only
  // test the graph we are building.

  //     Expression         The node lhs represents
  //
  // t0 = full(..., 0)      c0
  // t1 = full(..., 1)      c1
  // t2 = full(..., 2)      c2
  //
  //                        c0  c1
  //                         \  /
  // sum1 = t0 + t1          add1
  //
  //                        c0  c2
  //                         \  /
  // t0 += t2                add2
  //
  //                        c0  c2
  //                         \  /
  //                         add2  c1
  //                           \   /
  // sum3 = t0 + t1            add3
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  auto t0 = full(shape, 0, dtype);
  auto t1 = full(shape, 1, dtype);
  auto t2 = full(shape, 2, dtype);
  auto sum1 = t0 + t1;
  auto c0 = toJitTensorBase(t0).node();
  auto c1 = toJitTensorBase(t1).node();
  auto c2 = toJitTensorBase(t2).node();
  auto add1 = toJitTensorBase(sum1).node();
  t0 += t2;
  auto sum3 = t0 + t1;
  auto add2 = toJitTensorBase(t0).node();
  auto add3 = toJitTensorBase(sum3).node();

  // assignment won't affect other tensors
  ASSERT_EQ(toJitTensorBase(t1).node(), c1);
  ASSERT_EQ(toJitTensorBase(sum1).node(), add1);

  // assignment won't affect other tensor's graphs
  ASSERT_EQ(c0->inputs(), NodeList({}));
  ASSERT_EQ(c0->uses(), UseValList({{add1, 0}, {add2, 0}}));
  ASSERT_TRUE(c0->isScalar());
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({{add1, 1}, {add3, 1}}));
  ASSERT_TRUE(c1->isScalar());
  ASSERT_EQ(add1->inputs(), NodeList({c0, c1}));
  ASSERT_EQ(add1->uses(), UseValList({}));
  ASSERT_TRUE(add1->isBinary());

  // in-place add assignment creates new graph (like SSA)
  ASSERT_EQ(c2->inputs(), NodeList({}));
  ASSERT_EQ(c2->uses(), UseValList({{add2, 1}}));
  ASSERT_TRUE(c2->isScalar());
  ASSERT_EQ(add2->inputs(), NodeList({c0, c2}));
  ASSERT_EQ(add2->uses(), UseValList({{add3, 0}}));
  ASSERT_TRUE(add2->isBinary());

  // future use of assigned tensor uses its new graph
  ASSERT_EQ(add3->inputs(), NodeList({add2, c1}));
  ASSERT_EQ(add3->uses(), UseValList({}));
  ASSERT_TRUE(add3->isBinary());
}

TEST_F(JitTensorTest, assignmentWithShallowCopyAndCopy) {
  // we don't test the computation result (that's Evaluator's job) -- we only
  // test the graph we are building.

  //     Expression         The node lhs represents
  //
  // t0 = full(..., 0)        c0
  // t1 = full(..., 1)        c1
  //
  // t0c = t0.copy()          c0
  // t0sc = t0.shallowCopy()  c0
  //
  //                        c0  c1
  //                         \  /
  // t0 += t1                add
  //
  // t0c                      c0
  //
  //                        c0  c1
  //                         \  /
  // t0sc                    add
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  auto t0 = full(shape, 0, dtype);
  auto t1 = full(shape, 1, dtype);
  auto c0 = toJitTensorBase(t0).node();
  auto c1 = toJitTensorBase(t1).node();
  auto t0c = t0.copy();
  auto t0sc = toJitTensorBase(t0).shallowCopy();
  t0 += t1;
  auto add = toJitTensorBase(t0).node();

  // assignment won't affect copy
  ASSERT_EQ(toJitTensorBase(t0c).node(), c0);
  ASSERT_EQ(c0->inputs(), NodeList({}));
  ASSERT_EQ(c0->uses(), UseValList({{add, 0}}));
  ASSERT_TRUE(c0->isScalar());

  // assignment affects shallow copy
  ASSERT_EQ(toJitTensorBase(t0sc).node(), add);
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({{add, 1}}));
  ASSERT_TRUE(c1->isScalar());
  ASSERT_EQ(add->inputs(), NodeList({c0, c1}));
  ASSERT_EQ(add->uses(), UseValList({}));
  ASSERT_TRUE(add->isBinary());
}

TEST_F(JitTensorTest, indexedRead) {
  const Shape shape({5, 6, 7});
  const auto dtype = dtype::s32;
  const auto t0 = full(shape, 0, dtype);
  const std::vector<Index> indices{2, range(1, 4, 2)};
  auto indexResult = t0(indices);
  const auto indexResultCopy = indexResult.copy();
  const auto indexResultShallowCopy =
      toJitTensorBase(indexResult).shallowCopy();
  const auto c0 = toJitTensorBase(t0).node();
  const auto i0 = toJitTensorBase(indexResult).node();

  // graph is properly constructed, i.e., i0 --> c0
  ASSERT_EQ(c0->inputs(), NodeList({}));
  ASSERT_EQ(c0->uses(), UseValList({{i0, 0}}));
  ASSERT_TRUE(c0->isScalar());
  ASSERT_EQ(i0->inputs(), NodeList({c0}));
  ASSERT_EQ(i0->uses(), UseValList({}));
  ASSERT_EQ(i0->shape(), Shape({2, 7}));
  ASSERT_TRUE(i0->isIndex());
  ASSERT_EQ(i0->impl<IndexNode>().indices().size(), 2);
  ASSERT_EQ(i0->impl<IndexNode>().indexedNode(), c0);

  // copy/shallow-copy all yield the same node
  ASSERT_EQ(toJitTensorBase(indexResultCopy).node(), i0);
  ASSERT_EQ(toJitTensorBase(indexResultShallowCopy).node(), i0);
}

TEST_F(JitTensorTest, indexingWithOriginalDataUpdated) {
  const Shape shape(Shape({5, 6, 7}));
  const auto dtype = dtype::s32;
  auto t0 = full(shape, 0, dtype);
  const auto t1 = full(shape, 1, dtype);
  const std::vector<Index> indices{2, range(1, 4, 2)};
  const auto t0Copy = t0.copy();
  const auto t0ShallowCopy = toJitTensorBase(t0).shallowCopy();
  const auto t0Indexed = t0(indices);
  const auto t0CopyIndexed = t0Copy(indices);
  const auto t0ShallowCopyIndexed = t0ShallowCopy(indices);
  const auto c0 = toJitTensorBase(t0).node();
  t0 += t1; // affects t0, t0ShallowCopy and their indexed nodes
  const auto add = toJitTensorBase(t0).node();
  const auto c1 = toJitTensorBase(t1).node();
  const auto i0 = toJitTensorBase(t0Indexed).node();
  const auto i1 = toJitTensorBase(t0CopyIndexed).node();
  const auto i2 = toJitTensorBase(t0ShallowCopyIndexed).node();

  // shallow copy and copy have the right nodes
  ASSERT_EQ(toJitTensorBase(t0Copy).node(), c0);
  ASSERT_EQ(toJitTensorBase(t0ShallowCopy).node(), add);

  // graph is properly constructed
  //
  // c0   c1  c0
  //   \ /     |
  //   add    i1
  //    |
  //  i0/i2
  ASSERT_EQ(c0->inputs(), NodeList({}));
  ASSERT_EQ(c0->uses(), UseValList({{i1, 0}, {add, 0}}));
  ASSERT_TRUE(c0->isScalar());
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({{add, 1}}));
  ASSERT_TRUE(c1->isScalar());
  ASSERT_EQ(add->inputs(), NodeList({c0, c1}));
  ASSERT_EQ(add->uses(), UseValList({{i0, 0}, {i2, 0}}));
  ASSERT_TRUE(add->isBinary());
  ASSERT_EQ(i0->inputs(), NodeList({add}));
  ASSERT_EQ(i0->uses(), UseValList({}));
  ASSERT_EQ(i0->shape(), Shape({2, 7}));
  ASSERT_TRUE(i0->isIndex());
  ASSERT_EQ(i0->impl<IndexNode>().indices().size(), 2);
  ASSERT_EQ(i0->impl<IndexNode>().indexedNode(), add);
  ASSERT_EQ(i1->inputs(), NodeList({c0}));
  ASSERT_EQ(i1->uses(), UseValList({}));
  ASSERT_EQ(i1->shape(), Shape({2, 7}));
  ASSERT_TRUE(i1->isIndex());
  ASSERT_EQ(i1->impl<IndexNode>().indices().size(), 2);
  ASSERT_EQ(i1->impl<IndexNode>().indexedNode(), c0);
  ASSERT_EQ(i2->inputs(), NodeList({add}));
  ASSERT_EQ(i2->uses(), UseValList({}));
  ASSERT_EQ(i2->shape(), Shape({2, 7}));
  ASSERT_TRUE(i2->isIndex());
  ASSERT_EQ(i2->impl<IndexNode>().indices().size(), 2);
  ASSERT_EQ(i2->impl<IndexNode>().indexedNode(), add);
}

TEST_F(JitTensorTest, indexingWithIndexedDataUpdated) {
  const auto dtype = dtype::s32;
  auto t0 = full({5, 6, 7}, 0, dtype);
  const auto t1 = full({6, 7}, 1, dtype);
  const std::vector<Index> indices1{2};
  const std::vector<Index> indices2{2, range(1, 4, 2)};
  const auto t0Copy = t0.copy();
  const auto t0ShallowCopy = toJitTensorBase(t0).shallowCopy();
  const auto t0Indexed = t0(indices2);
  const auto t0CopyIndexed = t0Copy(indices2);
  const auto t0ShallowCopyIndexed = t0ShallowCopy(indices2);
  const auto c0 = toJitTensorBase(t0).node();
  t0(indices1) = t1; // affects t0, t0ShallowCopy and their indexed nodes
  const auto update = toJitTensorBase(t0).node();
  const auto c1 = toJitTensorBase(t1).node();
  const auto i0 = toJitTensorBase(t0Indexed).node();
  const auto i1 = toJitTensorBase(t0CopyIndexed).node();
  const auto i2 = toJitTensorBase(t0ShallowCopyIndexed).node();

  // shallow copy and copy have the right nodes
  ASSERT_EQ(toJitTensorBase(t0Copy).node(), c0);
  ASSERT_EQ(toJitTensorBase(t0ShallowCopy).node(), update);

  // graph is properly constructed
  //
  // indices1
  //    |
  // c0 | c1  c0
  //   \|/     |
  //  update  i1
  //    |
  //  i0/i2
  ASSERT_EQ(c0->inputs(), NodeList({}));
  ASSERT_EQ(c0->uses(), UseValList({{i1, 0}, {update, 0}}));
  ASSERT_TRUE(c0->isScalar());
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({{update, 1}}));
  ASSERT_TRUE(c1->isScalar());
  ASSERT_EQ(update->inputs(), NodeList({c0, c1}));
  ASSERT_EQ(update->uses(), UseValList({{i0, 0}, {i2, 0}}));
  ASSERT_TRUE(update->isIndexedUpdate());
  ASSERT_EQ(update->impl<IndexedUpdateNode>().indexedNode(), c0);
  ASSERT_EQ(update->impl<IndexedUpdateNode>().updateDataNode(), c1);
  ASSERT_EQ(update->impl<IndexedUpdateNode>().indexings().size(), 1);
  ASSERT_EQ(i0->inputs(), NodeList({update}));
  ASSERT_EQ(i0->uses(), UseValList({}));
  ASSERT_EQ(i0->shape(), Shape({2, 7}));
  ASSERT_TRUE(i0->isIndex());
  ASSERT_EQ(i0->impl<IndexNode>().indices().size(), 2);
  ASSERT_EQ(i0->impl<IndexNode>().indexedNode(), update);
  ASSERT_EQ(i1->inputs(), NodeList({c0}));
  ASSERT_EQ(i1->uses(), UseValList({}));
  ASSERT_EQ(i1->shape(), Shape({2, 7}));
  ASSERT_TRUE(i1->isIndex());
  ASSERT_EQ(i1->impl<IndexNode>().indices().size(), 2);
  ASSERT_EQ(i1->impl<IndexNode>().indexedNode(), c0);
  ASSERT_EQ(i2->inputs(), NodeList({update}));
  ASSERT_EQ(i2->uses(), UseValList({}));
  ASSERT_EQ(i2->shape(), Shape({2, 7}));
  ASSERT_TRUE(i2->isIndex());
  ASSERT_EQ(i2->impl<IndexNode>().indices().size(), 2);
  ASSERT_EQ(i2->impl<IndexNode>().indexedNode(), update);
}

// Verify that eval() compacts the JIT graph by replacing deep node chains
// with a single ValueNode. Without compaction, repeated indexed updates
// create O(N)-deep graphs that cause optimizer timeouts.
TEST_F(JitTensorTest, EvalCompactsGraph) {
  auto t = fl::rand({100});

  // Build a chain of indexed updates
  for (int i = 0; i < 50; i++) {
    t(i) = fl::full({1}, static_cast<float>(i));
  }

  // Before eval: node should be a deep IndexedUpdateNode chain
  auto& jitTensor = toJitTensorBase(t);
  auto preEvalNode = jitTensor.node();
  ASSERT_TRUE(preEvalNode->isIndexedUpdate());

  // Count depth before eval
  unsigned depthBefore = 0;
  {
    auto n = preEvalNode;
    while (n->isIndexedUpdate()) {
      depthBefore++;
      n = n->impl<IndexedUpdateNode>().indexedNode();
    }
  }
  ASSERT_GE(depthBefore, 50u);

  // fl::eval computes the result AND compacts the graph
  fl::eval(t);

  // After eval: node should be a ValueNode (graph compacted)
  auto postEvalNode = jitTensor.node();
  ASSERT_TRUE(postEvalNode->isValue())
      << "After eval, node should be a ValueNode but got "
      << nodeTypeToString(postEvalNode->type());
  ASSERT_EQ(countNodes(postEvalNode), 1u);

  // Value should be correct
  ASSERT_EQ(t(0).scalarSync<float>(), 0.0f);
  ASSERT_EQ(t(49).scalarSync<float>(), 49.0f);
}

// Verify that repeated eval+update cycles maintain O(1) graph depth.
// This is the pattern used by Jacobian tests: perturb, eval, perturb, eval, ...
TEST_F(JitTensorTest, RepeatedEvalKeepsGraphShallow) {
  auto t = fl::rand({100});
  fl::eval(t);

  for (int iter = 0; iter < 500; iter++) {
    // Perturb 3 elements (mimics Jacobian test pattern)
    t(iter % 100) = fl::full({1}, static_cast<float>(iter));
    t((iter + 1) % 100) = fl::full({1}, static_cast<float>(iter + 1));
    t((iter + 2) % 100) = fl::full({1}, static_cast<float>(iter + 2));
    fl::eval(t);

    // After each eval, graph should be compacted to a ValueNode
    auto& jitTensor = toJitTensorBase(t);
    ASSERT_TRUE(jitTensor.node()->isValue())
        << "After eval at iter " << iter
        << ", node should be ValueNode but got "
        << nodeTypeToString(jitTensor.node()->type());
    ASSERT_EQ(countNodes(jitTensor.node()), 1u);
  }
}

// Stress test: large number of indexed updates without intermediate evals.
// The optimizer and evaluator should handle deep graphs without stack overflow.
TEST_F(JitTensorTest, DeepGraphNoStackOverflow) {
  auto t = fl::rand({10});

  // Build a very deep graph (5000 indexed updates)
  for (int i = 0; i < 5000; i++) {
    t(i % 10) = fl::full({1}, static_cast<float>(i));
  }

  // Single eval should handle the deep graph without crashing
  fl::eval(t);

  auto& jitTensor = toJitTensorBase(t);
  ASSERT_TRUE(jitTensor.node()->isValue());
  ASSERT_EQ(countNodes(jitTensor.node()), 1u);
  // Last update to each position should be the final value
  ASSERT_EQ(t(0).scalarSync<float>(), 4990.0f);
  ASSERT_EQ(t(9).scalarSync<float>(), 4999.0f);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  init();
  return RUN_ALL_TESTS();
}
