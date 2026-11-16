/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/CustomNode.h"
#include "flashlight/fl/tensor/ir/MatmulNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"
#include "flashlight/fl/tensor/opt/PatternMatcher.h"

using namespace fl;

// =========================================================================
// Pattern matcher tests
// =========================================================================

TEST(PatternMatcherTest, matchAnyAlwaysMatches) {
  auto node = ScalarNode::create(Shape{2, 3}, fl::dtype::f32, 1.0);
  MatchResult result;
  ASSERT_TRUE(MatchAnyNoCapture::match(node, result));
}

TEST(PatternMatcherTest, captureStoresNode) {
  auto node = ScalarNode::create(Shape{2, 3}, fl::dtype::f32, 1.0);
  MatchResult result;
  auto pat = capture("myNode");
  ASSERT_TRUE(pat.match(node, result));
  ASSERT_EQ(result.get("myNode"), node);
}

TEST(PatternMatcherTest, matchUnaryOp) {
  auto input = ScalarNode::create(Shape{2, 3}, fl::dtype::f32, 1.0);
  auto expNode = UnaryNode::create(input, UnaryOp::Exp);

  MatchResult result;
  auto pat = matchUnary(UnaryOp::Exp, capture("input"));
  ASSERT_TRUE(pat.match(expNode, result));
  ASSERT_EQ(result.get("input"), input);
}

TEST(PatternMatcherTest, matchUnaryOpWrongOp) {
  auto input = ScalarNode::create(Shape{2, 3}, fl::dtype::f32, 1.0);
  auto expNode = UnaryNode::create(input, UnaryOp::Exp);

  MatchResult result;
  auto pat = matchUnary(UnaryOp::Log, capture("input"));
  ASSERT_FALSE(pat.match(expNode, result));
}

TEST(PatternMatcherTest, matchBinaryOp) {
  auto lhs = ScalarNode::create(Shape{2, 3}, fl::dtype::f32, 1.0);
  auto rhs = ScalarNode::create(Shape{2, 3}, fl::dtype::f32, 2.0);
  auto addNode = BinaryNode::create(lhs, rhs, BinaryOp::Add);

  MatchResult result;
  auto pat = matchBinary(BinaryOp::Add, capture("lhs"), capture("rhs"));
  ASSERT_TRUE(pat.match(addNode, result));
  ASSERT_EQ(result.get("lhs"), lhs);
  ASSERT_EQ(result.get("rhs"), rhs);
}

TEST(PatternMatcherTest, matchMatmul) {
  auto lhs = ScalarNode::create(Shape{4, 3}, fl::dtype::f32, 1.0);
  auto rhs = ScalarNode::create(Shape{3, 5}, fl::dtype::f32, 2.0);
  auto mm = MatmulNode::create(
      lhs, rhs, MatrixProperty::None, MatrixProperty::None);

  MatchResult result;
  auto pat = matchMatmul(capture("A"), capture("B"));
  ASSERT_TRUE(pat.match(mm, result));
  ASSERT_EQ(result.get("A"), lhs);
  ASSERT_EQ(result.get("B"), rhs);
}

TEST(PatternMatcherTest, matchMatmulNotMatmul) {
  auto lhs = ScalarNode::create(Shape{4, 3}, fl::dtype::f32, 1.0);
  auto rhs = ScalarNode::create(Shape{4, 3}, fl::dtype::f32, 2.0);
  auto addNode = BinaryNode::create(lhs, rhs, BinaryOp::Add);

  MatchResult result;
  auto pat = matchMatmul(capture("A"), capture("B"));
  ASSERT_FALSE(pat.match(addNode, result));
}

TEST(PatternMatcherTest, nestedPattern) {
  // Match: exp(add(matmul(A, B), C))
  auto a = ScalarNode::create(Shape{4, 3}, fl::dtype::f32, 1.0);
  auto b = ScalarNode::create(Shape{3, 5}, fl::dtype::f32, 2.0);
  auto c = ScalarNode::create(Shape{4, 5}, fl::dtype::f32, 3.0);
  auto mm = MatmulNode::create(a, b, MatrixProperty::None, MatrixProperty::None);
  auto add = BinaryNode::create(mm, c, BinaryOp::Add);
  auto exp = UnaryNode::create(add, UnaryOp::Exp);

  MatchResult result;
  auto pat = matchUnary(
      UnaryOp::Exp,
      matchBinary(
          BinaryOp::Add,
          matchMatmul(capture("A"), capture("B")),
          capture("C")));
  ASSERT_TRUE(pat.match(exp, result));
  ASSERT_EQ(result.get("A"), a);
  ASSERT_EQ(result.get("B"), b);
  ASSERT_EQ(result.get("C"), c);
}

TEST(PatternMatcherTest, withCaptureWrapsPattern) {
  auto a = ScalarNode::create(Shape{4, 3}, fl::dtype::f32, 1.0);
  auto b = ScalarNode::create(Shape{3, 5}, fl::dtype::f32, 2.0);
  auto mm = MatmulNode::create(a, b, MatrixProperty::None, MatrixProperty::None);
  auto exp = UnaryNode::create(mm, UnaryOp::Exp);

  MatchResult result;
  auto pat = matchUnary(
      UnaryOp::Exp,
      withCapture("matmulNode", matchMatmul(capture("A"), capture("B"))));
  ASSERT_TRUE(pat.match(exp, result));
  ASSERT_EQ(result.get("matmulNode"), mm);
  ASSERT_EQ(result.get("A"), a);
  ASSERT_EQ(result.get("B"), b);
}

TEST(PatternMatcherTest, matchNodeType) {
  auto a = ScalarNode::create(Shape{4, 3}, fl::dtype::f32, 1.0);

  MatchResult result;
  auto pat = MatchNodeType(NodeType::Scalar, "scalarNode");
  ASSERT_TRUE(pat.match(a, result));
  ASSERT_EQ(result.get("scalarNode"), a);

  result.clear();
  auto patBinary = MatchNodeType(NodeType::Binary);
  ASSERT_FALSE(patBinary.match(a, result));
}

// =========================================================================
// MatmulNode tests
// =========================================================================

TEST(MatmulNodeTest, basicMetadata) {
  auto lhs = ScalarNode::create(Shape{4, 3}, fl::dtype::f32, 1.0);
  auto rhs = ScalarNode::create(Shape{5, 3}, fl::dtype::f32, 2.0);
  auto mm = MatmulNode::create(
      lhs, rhs, MatrixProperty::None, MatrixProperty::Transpose);

  ASSERT_EQ(mm->type(), NodeType::Matmul);
  ASSERT_TRUE(mm->isMatmul());
  ASSERT_FALSE(mm->isBinary());
  ASSERT_EQ(mm->inputs().size(), 2);
  ASSERT_EQ(mm->impl<MatmulNode>().lhs(), lhs);
  ASSERT_EQ(mm->impl<MatmulNode>().rhs(), rhs);
  ASSERT_EQ(mm->impl<MatmulNode>().lhsProp(), MatrixProperty::None);
  ASSERT_EQ(mm->impl<MatmulNode>().rhsProp(), MatrixProperty::Transpose);
}

TEST(MatmulNodeTest, outputShape) {
  // [4, 3] x [3, 5] = [4, 5]
  auto lhs = ScalarNode::create(Shape{4, 3}, fl::dtype::f32, 1.0);
  auto rhs = ScalarNode::create(Shape{3, 5}, fl::dtype::f32, 2.0);
  auto mm = MatmulNode::create(
      lhs, rhs, MatrixProperty::None, MatrixProperty::None);
  ASSERT_EQ(mm->shape(), Shape({4, 5}));
}

TEST(MatmulNodeTest, outputShapeTransposed) {
  // [3, 4]^T x [3, 5] = [4, 5]
  auto lhs = ScalarNode::create(Shape{3, 4}, fl::dtype::f32, 1.0);
  auto rhs = ScalarNode::create(Shape{3, 5}, fl::dtype::f32, 2.0);
  auto mm = MatmulNode::create(
      lhs, rhs, MatrixProperty::Transpose, MatrixProperty::None);
  ASSERT_EQ(mm->shape(), Shape({4, 5}));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
