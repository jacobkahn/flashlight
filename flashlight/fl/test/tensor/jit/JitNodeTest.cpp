/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <optional>
#include <stdexcept>
#include <vector>

#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/JitTensorBase.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/CastNode.h"
#include "flashlight/fl/tensor/ir/ClipNode.h"
#include "flashlight/fl/tensor/ir/ConcatenateNode.h"
#include "flashlight/fl/tensor/ir/CumsumNode.h"
#include "flashlight/fl/tensor/ir/CustomNode.h"
#include "flashlight/fl/tensor/ir/FlipNode.h"
#include "flashlight/fl/tensor/ir/FusedElementwiseNode.h"
#include "flashlight/fl/tensor/ir/IndexNode.h"
#include "flashlight/fl/tensor/ir/IndexedUpdateNode.h"
#include "flashlight/fl/tensor/ir/MatmulNode.h"
#include "flashlight/fl/tensor/ir/Node.h"
#include "flashlight/fl/tensor/ir/NodeUtils.h"
#include "flashlight/fl/tensor/ir/PadNode.h"
#include "flashlight/fl/tensor/ir/ReductionNode.h"
#include "flashlight/fl/tensor/ir/ReshapeNode.h"
#include "flashlight/fl/tensor/ir/RollNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/TileNode.h"
#include "flashlight/fl/tensor/ir/TransposeNode.h"
#include "flashlight/fl/tensor/ir/TriangularNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"
#include "flashlight/fl/tensor/ir/WhereNode.h"

using namespace fl;

TEST(JitNodeTest, ScalarNodeMetaData) {
  const Shape shape({2, 2});
  const auto type = dtype::s32;
  const int value = 20;
  const auto node = ScalarNode::create(shape, type, value);
  ASSERT_EQ(node->inputs(), NodeList({}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(node->isScalar(), true);
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->shape(), shape);
  ASSERT_EQ(node->dataType(), type);
  ASSERT_EQ(node->scalar<int>(), value);
}

TEST(JitNodeTest, ValueNodeMetaData) {
  const auto tensor = full(Shape({3, 3}), 42);
  const auto node = ValueNode::create(tensor.copy());
  ASSERT_EQ(node->inputs(), NodeList({}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(node->isValue(), true);
  ASSERT_EQ(node->shape(), tensor.shape());
  ASSERT_TRUE(node->getResult().has_value());
  ASSERT_TRUE(allClose(node->getResult().value(), tensor));
}

TEST(JitNodeTest, BinaryNodeMetaData) {
  // Row-major broadcasting: align from the right
  Shape lhsShape({1, 4});
  Shape rhsShape({3, 5, 1});
  Shape outShape({3, 5, 4}); // test broadcast
  const auto c1 = ScalarNode::create(lhsShape, dtype::f32, 42);
  const auto c2 = ScalarNode::create(rhsShape, dtype::f32, 42);
  const auto op = BinaryOp::Add;
  const auto node = BinaryNode::create(c1, c2, op);
  ASSERT_EQ(node->inputs(), NodeList({c1, c2}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(node->isBinary(), true);
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->lhs(), c1);
  ASSERT_EQ(node->rhs(), c2);
  ASSERT_EQ(node->op(), op);
  ASSERT_EQ(node->shape(), outShape);
}

TEST(JitNodeTest, CustomNodeMetaData) {
  Shape shape({2, 2});
  auto type = dtype::f32;
  const auto c1 = ScalarNode::create(shape, type, 42);
  const auto c2 = ScalarNode::create(shape, type, 23);
  const auto t1 = full(shape, 42, type);
  const auto t2 = full(shape, 23, type);
  const auto name = "foobar";
  const auto node = CustomNode::create(
      name,
      {c1, c2},
      shape,
      type,
      [](const std::vector<const Tensor*>& inputs) -> Tensor {
        return inputs.at(0)->copy();
      });
  ASSERT_EQ(node->inputs(), NodeList({c1, c2}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(node->isCustom(), true);
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->name(), name);
  ASSERT_EQ(node->shape(), shape);
  ASSERT_TRUE(allClose(node->evalFunc()({&t1, &t2}), t1));
}

TEST(JitNodeTest, IndexNodeMetaData) {
  const auto c0 = ScalarNode::create({5, 6, 7}, dtype::f32, 0);
  std::vector<Index> indices{1, range(0, 3, 2)};
  const auto node = IndexNode::create(c0, indices);
  ASSERT_EQ(node->inputs(), NodeList({c0}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(node->isIndex(), true);
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->indexedNode(), c0);
  ASSERT_EQ(node->indices().size(), 2); // can't check equality easily...
  // - 1 reduces first dimension -- {6, 7}
  // - [0:3:2] takes 2 elements  -- {2, 7}
  ASSERT_EQ(node->shape(), Shape({2, 7}));
}

TEST(JitNodeTest, IndexNodeSameShapeTensorIndices) {
  const Shape shape({5, 6, 7});
  const auto c0 = ScalarNode::create(shape, dtype::f32, 0);
  const auto t1 = getOrCreateJitBackend(DefaultTensorBackend_t::getInstance())
                      .full(shape, 1, dtype::f32);
  const auto c1 = toJitTensorBase(t1).node();
  std::vector<Index> indices{t1};
  const auto node = IndexNode::create(c0, indices);
  ASSERT_EQ(node->inputs(), NodeList({c0, c1})); // includes tensor index
  ASSERT_EQ(node->indexedNode(), c0);
  ASSERT_EQ(node->indices().size(), 1); // can't check equality easily...
  // if tensor index has same shape as indexed tensor, output shape is simply
  // the flattened tensor shape
  ASSERT_EQ(node->shape(), Shape({210}));
}

TEST(JitNodeTest, IndexNodeDifferentShapeTensorIndices) {
  const auto c0 = ScalarNode::create({5, 6, 7}, dtype::f32, 0);
  const auto t1 = getOrCreateJitBackend(DefaultTensorBackend_t::getInstance())
                      .full({2, 3}, 1, dtype::f32);
  const auto c1 = toJitTensorBase(t1).node();
  std::vector<Index> indices{t1};
  const auto node = IndexNode::create(c0, indices);
  ASSERT_EQ(node->inputs(), NodeList({c0, c1})); // includes tensor index
  ASSERT_EQ(node->indexedNode(), c0);
  ASSERT_EQ(node->indices().size(), 1); // can't check equality easily...
  // tensor index shape flattened as base, then reduce 1 dimension
  ASSERT_EQ(node->shape(), Shape({6, 6, 7}));
}

TEST(JitNodeTest, UnaryNodeMetaData) {
  Shape shape({3, 4});
  const auto c1 = ScalarNode::create(shape, dtype::f32, 42);
  const auto op = UnaryOp::Exp;
  const auto node = UnaryNode::create(c1, op);
  ASSERT_EQ(node->inputs(), NodeList({c1}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(node->isUnary(), true);
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->input(), c1);
  ASSERT_EQ(node->op(), op);
  // Unary ops preserve shape
  ASSERT_EQ(node->shape(), shape);
}

TEST(JitNodeTest, ReductionNodeMetaData) {
  Shape inputShape({3, 4, 5});
  const auto c1 = ScalarNode::create(inputShape, dtype::f32, 42);
  const auto op = ReductionOp::Sum;
  std::vector<int> axes = {1};
  const bool keepDims = false;
  const auto node = ReductionNode::create(c1, op, axes, keepDims);
  ASSERT_EQ(node->inputs(), NodeList({c1}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(node->isReduction(), true);
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->input(), c1);
  ASSERT_EQ(node->op(), op);
  ASSERT_EQ(node->axes(), axes);
  ASSERT_EQ(node->keepDims(), keepDims);
  // Reducing axis 1 from {3, 4, 5} -> {3, 5}
  ASSERT_EQ(node->shape(), Shape({3, 5}));
}

TEST(JitNodeTest, ReductionNodeKeepDims) {
  Shape inputShape({3, 4, 5});
  const auto c1 = ScalarNode::create(inputShape, dtype::f32, 42);
  std::vector<int> axes = {0, 2};
  const bool keepDims = true;
  const auto node =
      ReductionNode::create(c1, ReductionOp::Mean, axes, keepDims);
  // keepDims=true: {3, 4, 5} reduce axes {0,2} -> {1, 4, 1}
  ASSERT_EQ(node->shape(), Shape({1, 4, 1}));
  ASSERT_EQ(node->keepDims(), true);
}

TEST(JitNodeTest, TileNodeMetaData) {
  const auto input = ScalarNode::create({3}, dtype::f32, 42);
  const auto node = TileNode::create(input, {2, 3});
  ASSERT_EQ(node->inputs(), NodeList({input}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_TRUE(node->isTile());
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->input(), input);
  ASSERT_EQ(node->tileDims(), Shape({2, 3}));
  ASSERT_EQ(node->shape(), Shape({2, 9}));
  ASSERT_EQ(node->dataType(), dtype::f32);
}

TEST(JitNodeTest, ConcatenateNodeMetaData) {
  const auto a = ScalarNode::create({2, 3}, dtype::s32, 1);
  const auto b = ScalarNode::create({2, 4}, dtype::f32, 2);
  const auto node = ConcatenateNode::create({a, b}, 1);
  ASSERT_EQ(node->inputs(), NodeList({a, b}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_TRUE(node->isConcatenate());
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->input(0), a);
  ASSERT_EQ(node->input(1), b);
  ASSERT_EQ(node->numInputs(), 2u);
  ASSERT_EQ(node->axis(), 1u);
  ASSERT_EQ(node->shape(), Shape({2, 7}));
  ASSERT_EQ(node->dataType(), dtype::f32);
}

TEST(JitNodeTest, ConcatenateNodeRejectsEmptyInputList) {
  EXPECT_THROW(ConcatenateNode::create({}, 0), std::invalid_argument);
}

TEST(JitNodeTest, PadNodeMetaData) {
  const auto input = ScalarNode::create({3, 4}, dtype::f32, 42);
  const std::vector<std::pair<int, int>> padWidths{{1, 2}, {0, 3}};
  const auto node = PadNode::create(input, padWidths, PadType::Edge);
  ASSERT_EQ(node->inputs(), NodeList({input}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_TRUE(node->isPad());
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->input(), input);
  ASSERT_EQ(node->padWidths(), padWidths);
  ASSERT_EQ(node->padType(), PadType::Edge);
  ASSERT_EQ(node->shape(), Shape({6, 7}));
  ASSERT_EQ(node->dataType(), dtype::f32);
}

TEST(JitNodeTest, PadNodeNormalizesMissingAxes) {
  const auto input = ScalarNode::create({3, 4}, dtype::f32, 42);
  const auto node = PadNode::create(input, {{1, 1}}, PadType::Constant);
  ASSERT_EQ(
      node->padWidths(), (std::vector<std::pair<int, int>>{{1, 1}, {0, 0}}));
  ASSERT_EQ(node->shape(), Shape({5, 4}));
}

TEST(JitNodeTest, PadNodeRejectsTooManyAxes) {
  const auto input = ScalarNode::create({3, 4}, dtype::f32, 42);
  EXPECT_THROW(
      PadNode::create(input, {{1, 1}, {1, 1}, {1, 1}}, PadType::Constant),
      std::runtime_error);
}

TEST(JitNodeTest, PadNodeRejectsNegativeWidths) {
  const auto input = ScalarNode::create({3, 4}, dtype::f32, 42);
  EXPECT_THROW(
      PadNode::create(input, {{-1, 0}, {0, 0}}, PadType::Constant),
      std::invalid_argument);
}

TEST(JitNodeTest, TriangularNodeMetaData) {
  const auto input = ScalarNode::create({2, 3, 4}, dtype::f32, 42);
  const auto node =
      TriangularNode::create(input, TriangularMode::Lower, 1);
  ASSERT_EQ(node->inputs(), NodeList({input}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_TRUE(node->isTriangular());
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->input(), input);
  ASSERT_EQ(node->mode(), TriangularMode::Lower);
  ASSERT_EQ(node->diagonal(), 1);
  ASSERT_EQ(node->shape(), Shape({2, 3, 4}));
  ASSERT_EQ(node->dataType(), dtype::f32);
}

TEST(JitNodeTest, TriangularNodeRejectsRankLessThanTwo) {
  const auto input = ScalarNode::create({4}, dtype::f32, 42);
  EXPECT_THROW(
      TriangularNode::create(input, TriangularMode::Upper, 0),
      std::invalid_argument);
}

TEST(JitNodeTest, FlipNodeMetaData) {
  const auto input = ScalarNode::create({3, 4}, dtype::f32, 42);
  const auto node = FlipNode::create(input, 1);
  ASSERT_EQ(node->inputs(), NodeList({input}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_TRUE(node->isFlip());
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->input(), input);
  ASSERT_EQ(node->axis(), 1u);
  ASSERT_EQ(node->shape(), Shape({3, 4}));
  ASSERT_EQ(node->dataType(), dtype::f32);
}

TEST(JitNodeTest, FlipNodeRejectsInvalidAxis) {
  const auto input = ScalarNode::create({3, 4}, dtype::f32, 42);
  EXPECT_THROW(FlipNode::create(input, 2), std::invalid_argument);
}

TEST(JitNodeTest, ClipNodeMetaData) {
  const auto input = ScalarNode::create({3, 4}, dtype::f32, 2);
  const auto low = ScalarNode::create({3, 4}, dtype::f32, 0);
  const auto high = ScalarNode::create({3, 4}, dtype::f32, 1);
  const auto node = ClipNode::create(input, low, high);
  ASSERT_EQ(node->inputs(), NodeList({input, low, high}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_TRUE(node->isClip());
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->input(), input);
  ASSERT_EQ(node->low(), low);
  ASSERT_EQ(node->high(), high);
  ASSERT_EQ(node->shape(), Shape({3, 4}));
  ASSERT_EQ(node->dataType(), dtype::f32);
}

TEST(JitNodeTest, CumsumNodeMetaData) {
  const auto input = ScalarNode::create({3, 4}, dtype::f32, 42);
  const auto node = CumsumNode::create(input, 1);
  ASSERT_EQ(node->inputs(), NodeList({input}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_TRUE(node->isCumsum());
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->input(), input);
  ASSERT_EQ(node->axis(), 1u);
  ASSERT_EQ(node->shape(), Shape({3, 4}));
  ASSERT_EQ(node->dataType(), dtype::f32);
}

TEST(JitNodeTest, CumsumNodeRejectsInvalidAxis) {
  const auto input = ScalarNode::create({3, 4}, dtype::f32, 42);
  EXPECT_THROW(CumsumNode::create(input, 2), std::invalid_argument);
}

TEST(JitNodeTest, RollNodeMetaData) {
  const auto input = ScalarNode::create({3, 4}, dtype::f32, 42);
  const auto node = RollNode::create(input, -1, 1);
  ASSERT_EQ(node->inputs(), NodeList({input}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_TRUE(node->isRoll());
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->input(), input);
  ASSERT_EQ(node->shift(), 3);
  ASSERT_EQ(node->axis(), 1u);
  ASSERT_EQ(node->shape(), Shape({3, 4}));
  ASSERT_EQ(node->dataType(), dtype::f32);
}

TEST(JitNodeTest, RollNodeRejectsInvalidAxis) {
  const auto input = ScalarNode::create({3, 4}, dtype::f32, 42);
  EXPECT_THROW(RollNode::create(input, 1, 2), std::invalid_argument);
}

TEST(JitNodeTest, WhereNodeMetaData) {
  const auto condition = ScalarNode::create({2, 2}, dtype::b8, true);
  const auto x = ScalarNode::create({2, 2}, dtype::s32, 1);
  const auto y = ScalarNode::create({2, 2}, dtype::f32, 2);
  const auto node = WhereNode::create(condition, x, y);
  ASSERT_EQ(node->inputs(), NodeList({condition, x, y}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_TRUE(node->isWhere());
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->condition(), condition);
  ASSERT_EQ(node->x(), x);
  ASSERT_EQ(node->y(), y);
  ASSERT_EQ(node->shape(), Shape({2, 2}));
  ASSERT_EQ(node->dataType(), dtype::f32);
}

TEST(JitNodeTest, WhereNodeRejectsNonBooleanCondition) {
  const auto condition = ScalarNode::create({2, 2}, dtype::f32, 1);
  const auto x = ScalarNode::create({2, 2}, dtype::f32, 1);
  const auto y = ScalarNode::create({2, 2}, dtype::f32, 2);
  EXPECT_THROW(WhereNode::create(condition, x, y), std::invalid_argument);
}

TEST(JitNodeTest, WhereNodeRejectsMismatchedShapes) {
  const auto condition = ScalarNode::create({2, 2}, dtype::b8, true);
  const auto x = ScalarNode::create({2, 2}, dtype::f32, 1);
  const auto y = ScalarNode::create({4}, dtype::f32, 2);
  EXPECT_THROW(WhereNode::create(condition, x, y), std::invalid_argument);
}

TEST(JitNodeTest, FusedElementwiseNodeMetaData) {
  // Build: out = (a + b) * neg(a)
  // External inputs: a (slot 0), b (slot 1)
  // Step 0: Add(s0, s1)       -> s2
  // Step 1: Negative(s0)      -> s3
  // Step 2: Mul(s2, s3)       -> s4 (output)
  Shape shape({3, 4});
  const auto a = ScalarNode::create(shape, dtype::f32, 1);
  const auto b = ScalarNode::create(shape, dtype::f32, 2);
  std::vector<FusedOp> program = {
      FusedOp{BinaryOp::Add, 0, 1},
      FusedOp{UnaryOp::Negative, 0, 0},
      FusedOp{BinaryOp::Mul, 2, 3},
  };
  const auto node = FusedElementwiseNode::create({a, b}, program, 2, shape);
  ASSERT_EQ(node->inputs(), NodeList({a, b}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(node->isFusedElementwise(), true);
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->shape(), shape);
  ASSERT_EQ(node->numExternalInputs(), 2u);
  ASSERT_EQ(node->numSteps(), 3u);
  ASSERT_TRUE(node->program()[0].isBinary());
  ASSERT_EQ(node->program()[0].binaryOp(), BinaryOp::Add);
  ASSERT_TRUE(node->program()[1].isUnary());
  ASSERT_EQ(node->program()[1].unaryOp(), UnaryOp::Negative);
  ASSERT_TRUE(node->program()[2].isBinary());
  ASSERT_EQ(node->program()[2].binaryOp(), BinaryOp::Mul);
}

// ======================== Dtype Inference Tests ========================

TEST(JitNodeTest, DtypeInferenceBinary) {
  Shape shape({2, 2});
  const auto f32_node = ScalarNode::create(shape, dtype::f32, 1);
  const auto s32_node = ScalarNode::create(shape, dtype::s32, 2);
  // Arithmetic: f32 + s32 -> f32 (float wins)
  const auto add = BinaryNode::create(f32_node, s32_node, BinaryOp::Add);
  ASSERT_EQ(add->dataType(), dtype::f32);
  // Comparison: f32 > s32 -> b8
  const auto gt = BinaryNode::create(f32_node, s32_node, BinaryOp::Gt);
  ASSERT_EQ(gt->dataType(), dtype::b8);
}

TEST(JitNodeTest, DtypeInferenceUnary) {
  Shape shape({2, 2});
  const auto f32_node = ScalarNode::create(shape, dtype::f32, 1);
  // Exp preserves dtype
  const auto exp_node = UnaryNode::create(f32_node, UnaryOp::Exp);
  ASSERT_EQ(exp_node->dataType(), dtype::f32);
  // LogicalNot -> b8
  const auto not_node = UnaryNode::create(f32_node, UnaryOp::LogicalNot);
  ASSERT_EQ(not_node->dataType(), dtype::b8);
}

TEST(JitNodeTest, DtypeInferenceReduction) {
  Shape shape({3, 4});
  const auto s32_node = ScalarNode::create(shape, dtype::s32, 1);
  const auto f32_node = ScalarNode::create(shape, dtype::f32, 1);
  // Mean of s32 -> f32
  const auto mean =
      ReductionNode::create(s32_node, ReductionOp::Mean, {0}, false);
  ASSERT_EQ(mean->dataType(), dtype::f32);
  // Sum of f32 -> f32
  const auto sum =
      ReductionNode::create(f32_node, ReductionOp::Sum, {0}, false);
  ASSERT_EQ(sum->dataType(), dtype::f32);
  // CountNonzero -> u32 (matches CPU backend behavior)
  const auto cnt =
      ReductionNode::create(f32_node, ReductionOp::CountNonzero, {0}, false);
  ASSERT_EQ(cnt->dataType(), dtype::u32);
}

TEST(JitNodeTest, DtypeInferenceMatmul) {
  const auto f32_node = ScalarNode::create({2, 3}, dtype::f32, 1);
  const auto f64_node = ScalarNode::create({3, 4}, dtype::f64, 1);
  const auto matmul = MatmulNode::create(
      f32_node, f64_node, MatrixProperty::None, MatrixProperty::None);
  ASSERT_EQ(matmul->dataType(), dtype::f64);
}

TEST(JitNodeTest, DtypeInferenceCustom) {
  Shape shape({2, 2});
  const auto node = CustomNode::create(
      "test", {}, shape, dtype::f16, [](const std::vector<const Tensor*>&) {
        return Tensor();
      });
  ASSERT_EQ(node->dataType(), dtype::f16);
}

TEST(JitNodeTest, DtypeInferenceValue) {
  const auto tensor = full(Shape({3, 3}), 42.0f, dtype::f32);
  const auto node = ValueNode::create(tensor.copy());
  ASSERT_EQ(node->dataType(), dtype::f32);
}

TEST(JitNodeTest, DtypeInferenceScalar) {
  const auto node = ScalarNode::create({2, 2}, dtype::f64, 1.0);
  ASSERT_EQ(node->dataType(), dtype::f64);
}

TEST(JitNodeTest, DtypeInferenceFusedElementwise) {
  // Program: Add(f32, s32) -> f32, then Exp(f32) -> f32
  Shape shape({3, 4});
  const auto a = ScalarNode::create(shape, dtype::f32, 1);
  const auto b = ScalarNode::create(shape, dtype::s32, 2);
  std::vector<FusedOp> program = {
      FusedOp{BinaryOp::Add, 0, 1},
      FusedOp{UnaryOp::Exp, 2, 0},
  };
  const auto node = FusedElementwiseNode::create({a, b}, program, 2, shape);
  ASSERT_EQ(node->dataType(), dtype::f32);
}

TEST(JitNodeTest, IndexNodePreservesDtype) {
  const auto c0 = ScalarNode::create({5, 6, 7}, dtype::f64, 0);
  std::vector<Index> indices{1, range(0, 3, 2)};
  const auto node = IndexNode::create(c0, indices);
  ASSERT_EQ(node->dataType(), dtype::f64);
}

TEST(JitNodeTest, IndexedUpdateNodePreservesDtype) {
  const auto c0 = ScalarNode::create(Shape({5, 6, 7}), dtype::f32, 0);
  const auto c1 = ScalarNode::create(Shape({2, 3, 6, 7}), dtype::f32, 0);
  const auto t2 = getOrCreateJitBackend(DefaultTensorBackend_t::getInstance())
                      .full(Shape({2, 3}), 1, dtype::f32);
  const auto c2 = toJitTensorBase(t2).node();
  std::vector<Index> indices{t2};
  const auto node = IndexedUpdateNode::create(c0, {indices}, c1);
  ASSERT_EQ(node->inputs(), NodeList({c0, c1, c2})); // includes tensor index
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(node->isIndexedUpdate(), true);
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->indexedNode(), c0);
  ASSERT_EQ(node->indexings().size(), 1); // can't check equality easily...
  ASSERT_EQ(node->shape(), c0->shape());
}

TEST(JitNodeTest, getSetResult) {
  const auto node = ScalarNode::create(Shape({2, 2}), dtype::f32, 42);
  const auto tensor = full(Shape({2, 2}), 23, dtype::s32);
  // NOTE it's client's responsibility to ensure the result is correct
  node->setResult(tensor.copy());
  ASSERT_TRUE(node->getResult().has_value());
  ASSERT_TRUE(allClose(node->getResult().value(), tensor));
  ASSERT_THROW(node->setResult(tensor.copy()), std::invalid_argument);
}

TEST(JitNodeTest, inputsAndUses) {
  // c1   c2
  //   \  /
  //   add
  //   / \
  //   \ /
  //   mul
  Shape shape({2, 2, 2});
  auto type = dtype::s32;
  auto c1 = ScalarNode::create(shape, type, 1);
  auto c2 = ScalarNode::create(shape, type, 2);
  auto add = BinaryNode::create(c1, c2, BinaryOp::Add);
  auto mul = BinaryNode::create(add, add, BinaryOp::Mul);

  // inputs
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c2->inputs(), NodeList({}));
  ASSERT_EQ(add->inputs(), NodeList({c1, c2}));
  ASSERT_EQ(mul->inputs(), NodeList({add, add}));

  // uses
  ASSERT_EQ(c1->uses(), UseValList({{add, 0}}));
  ASSERT_EQ(c2->uses(), UseValList({{add, 1}}));
  ASSERT_EQ(add->uses(), UseValList({{mul, 0}, {mul, 1}}));
  ASSERT_EQ(mul->uses(), UseValList({}));
}

TEST(JitNodeTest, replaceAllUsesWithRootNodes) {
  // c1  c2
  Shape shape({2, 2, 2});
  auto type = dtype::s32;
  auto c1 = ScalarNode::create(shape, type, 1);
  auto c2 = ScalarNode::create(shape, type, 2);

  // c1  c2
  // --->
  // c1  c2
  c1->replaceAllUsesWith(c2);
  // nothing changed
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c2->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({}));
  ASSERT_EQ(c2->uses(), UseValList({}));
}

TEST(JitNodeTest, replaceAllUsesWithSharedNodes) {
  // c1   c2   c3  c4
  //   \  /     \  /
  //   add1     add2
  //   / \
  //   \ /
  //   mul
  Shape shape({2, 2, 2});
  auto type = dtype::s32;
  auto c1 = ScalarNode::create(shape, type, 1);
  auto c2 = ScalarNode::create(shape, type, 2);
  auto c3 = ScalarNode::create(shape, type, 3);
  auto c4 = ScalarNode::create(shape, type, 4);
  auto add1 = BinaryNode::create(c1, c2, BinaryOp::Add);
  auto add2 = BinaryNode::create(c3, c4, BinaryOp::Add);
  auto mul = BinaryNode::create(add1, add1, BinaryOp::Mul);

  // c1   c2   c3  c4
  //   \  /     \  /
  //   add1     add2
  //            / \
  //            \ /
  //            mul
  add1->replaceAllUsesWith(add2);
  // make sure nothing got messed up
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c2->inputs(), NodeList({}));
  ASSERT_EQ(c3->inputs(), NodeList({}));
  ASSERT_EQ(c4->inputs(), NodeList({}));
  ASSERT_EQ(add1->inputs(), NodeList({c1, c2}));
  ASSERT_EQ(add2->inputs(), NodeList({c3, c4}));
  ASSERT_EQ(mul->inputs(), NodeList({add2, add2}));
  ASSERT_EQ(c1->uses(), UseValList({{add1, 0}}));
  ASSERT_EQ(c2->uses(), UseValList({{add1, 1}}));
  ASSERT_EQ(c3->uses(), UseValList({{add2, 0}}));
  ASSERT_EQ(c4->uses(), UseValList({{add2, 1}}));
  ASSERT_EQ(add1->uses(), UseValList({}));
  ASSERT_EQ(add2->uses(), UseValList({{mul, 0}, {mul, 1}}));
  ASSERT_EQ(mul->uses(), UseValList({}));

  // c1   c3  c4
  //       \  /
  //       add2
  //       / \
  //       \ /
  //       mul  c2
  //        \   /
  //        add1
  c1->replaceAllUsesWith(mul);
  // make sure nothing got messed up
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c2->inputs(), NodeList({}));
  ASSERT_EQ(c3->inputs(), NodeList({}));
  ASSERT_EQ(c4->inputs(), NodeList({}));
  ASSERT_EQ(add1->inputs(), NodeList({mul, c2}));
  ASSERT_EQ(add2->inputs(), NodeList({c3, c4}));
  ASSERT_EQ(mul->inputs(), NodeList({add2, add2}));
  ASSERT_EQ(c1->uses(), UseValList({}));
  ASSERT_EQ(c2->uses(), UseValList({{add1, 1}}));
  ASSERT_EQ(c3->uses(), UseValList({{add2, 0}}));
  ASSERT_EQ(c4->uses(), UseValList({{add2, 1}}));
  ASSERT_EQ(add1->uses(), UseValList({}));
  ASSERT_EQ(add2->uses(), UseValList({{mul, 0}, {mul, 1}}));
  ASSERT_EQ(mul->uses(), UseValList({{add1, 0}}));
}

// ======================== Reshape/Transpose/Cast Node Tests
// ========================

TEST(JitNodeTest, ReshapeNodeMetaData) {
  Shape inputShape({3, 4});
  Shape newShape({12});
  const auto c1 = ScalarNode::create(inputShape, dtype::f32, 42);
  const auto node = ReshapeNode::create(c1, newShape);
  ASSERT_EQ(node->inputs(), NodeList({c1}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(node->isReshape(), true);
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->input(), c1);
  ASSERT_EQ(node->shape(), newShape);
  ASSERT_EQ(node->dataType(), dtype::f32);
}

TEST(JitNodeTest, ReshapeNodeInvalidShape) {
  Shape inputShape({3, 4});
  Shape badShape({5});
  const auto c1 = ScalarNode::create(inputShape, dtype::f32, 42);
  ASSERT_THROW(ReshapeNode::create(c1, badShape), std::invalid_argument);
}

TEST(JitNodeTest, TransposeNodeMetaData) {
  Shape inputShape({3, 4, 5});
  Shape axes({2, 0, 1});
  const auto c1 = ScalarNode::create(inputShape, dtype::f32, 42);
  const auto node = TransposeNode::create(c1, axes);
  ASSERT_EQ(node->inputs(), NodeList({c1}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(node->isTranspose(), true);
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->input(), c1);
  ASSERT_EQ(node->axes(), axes);
  // {3, 4, 5} with axes {2, 0, 1} -> {5, 3, 4}
  ASSERT_EQ(node->shape(), Shape({5, 3, 4}));
  ASSERT_EQ(node->dataType(), dtype::f32);
}

TEST(JitNodeTest, CastNodeMetaData) {
  Shape shape({3, 4});
  const auto c1 = ScalarNode::create(shape, dtype::f32, 42);
  const auto node = CastNode::create(c1, dtype::s32);
  ASSERT_TRUE(node->isCast());
  ASSERT_EQ(node->shape(), shape);
  ASSERT_EQ(node->dataType(), dtype::s32);
  ASSERT_EQ(node->inputs(), NodeList({c1}));
  ASSERT_EQ(node->impl<CastNode>().targetDtype(), dtype::s32);
  ASSERT_EQ(node->impl<CastNode>().input(), c1);
}

TEST(JitNodeTest, CastNodeNoOp) {
  Shape shape({3, 4});
  const auto c1 = ScalarNode::create(shape, dtype::f32, 42);
  const auto node = CastNode::create(c1, dtype::f32);
  // No-op cast should return the input directly
  ASSERT_EQ(node, c1);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  init();
  return RUN_ALL_TESTS();
}
