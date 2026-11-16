/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <functional>

#include <gtest/gtest.h>

#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/tensor/BackendDispatch.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/JitTensorBase.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/eval/Evaluator.h"
#include "flashlight/fl/tensor/ir/CastNode.h"
#include "flashlight/fl/tensor/ir/ClipNode.h"
#include "flashlight/fl/tensor/ir/ConcatenateNode.h"
#include "flashlight/fl/tensor/ir/CumsumNode.h"
#include "flashlight/fl/tensor/ir/ExternalUse.h"
#include "flashlight/fl/tensor/ir/FlipNode.h"
#include "flashlight/fl/tensor/ir/PadNode.h"
#include "flashlight/fl/tensor/ir/ReshapeNode.h"
#include "flashlight/fl/tensor/ir/RollNode.h"
#include "flashlight/fl/tensor/ir/TileNode.h"
#include "flashlight/fl/tensor/ir/TransposeNode.h"
#include "flashlight/fl/tensor/ir/TriangularNode.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"
#include "flashlight/fl/tensor/ir/WhereNode.h"

using namespace fl;

namespace {

TensorBackendType defaultEvaluatorBackendType() {
  return DefaultTensorBackend_t::getInstance().backendType();
}

bool defaultEvaluatorBackendSupportsGeneralTriangularOps() {
  const auto backendType = defaultEvaluatorBackendType();
  return backendType == TensorBackendType::CPU ||
      backendType == TensorBackendType::Cuda;
}

} // namespace

class JitEvaluatorTest : public ::testing::Test {
 protected:
  JitEvaluatorTest()
      : evaluator_(makeBackendDispatch(DefaultTensorBackend_t::getInstance())) {
  }
  Evaluator evaluator_;
};

TEST_F(JitEvaluatorTest, evalValueNode) {
  const auto tensor = fl::rand(Shape({2, 2}), dtype::f32);
  const auto node = ValueNode::create(tensor.copy());
  evaluator_.eval(node);
  ASSERT_TRUE(allClose(node->getResult().value(), tensor));
}

TEST_F(JitEvaluatorTest, evalScalarNode) {
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  int val = 42;
  const auto tensor = full(shape, val, dtype);
  const auto node = ScalarNode::create(shape, dtype, val);
  evaluator_.eval(node);
  ASSERT_TRUE(allClose(node->getResult().value(), tensor));
}

TEST_F(JitEvaluatorTest, evalBinaryNode) {
  // c1  c2
  //  \  /
  //   add
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto c2 = ScalarNode::create(shape, dtype, 2);
  const auto add = BinaryNode::create(c1, c2, BinaryOp::Add);
  evaluator_.eval(add);
  ASSERT_TRUE(allClose(add->getResult().value(), full(shape, 3, dtype)));
}

TEST_F(JitEvaluatorTest, evalCustomNode) {
  // c1  c2  c3
  //  \  |  /
  //   custom
  //
  // evaluation logic: (c1 + c2) * c3
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto c2 = ScalarNode::create(shape, dtype, 2);
  const auto c3 = ScalarNode::create(shape, dtype, 3);
  const auto custom = CustomNode::create(
      "addThenMul",
      {c1, c2, c3},
      shape,
      dtype,
      [](const std::vector<const Tensor*> inputs) {
        return (*inputs[0] + *inputs[1]) * *inputs[2];
      });
  evaluator_.eval(custom);
  ASSERT_TRUE(allClose(custom->getResult().value(), full(shape, 9, dtype)));
}

TEST_F(JitEvaluatorTest, evalIndexNodeWithoutTensorIdx) {
  const auto value = iota({4, 5, 6}, {1}, dtype::s32);
  const auto valueNode = ValueNode::create(value.copy());
  const std::vector<Index> indices{1, range(0, 3, 2), range(3)};
  const auto indexNode = IndexNode::create(valueNode, indices);
  evaluator_.eval(indexNode);
  ASSERT_TRUE(allClose(indexNode->getResult().value(), value(indices)));
  ASSERT_TRUE(allClose(valueNode->getResult().value(), value));
}

TEST_F(JitEvaluatorTest, evalIndexNodeWithTensorIdx) {
  const auto dtype = dtype::s32;
  const auto value = iota({10, 10}, {1}, dtype);
  const auto tensorIdx = iota({2, 3}, {1}, dtype);
  auto& jitBackend =
      getOrCreateJitBackend(DefaultTensorBackend_t::getInstance());
  const auto jitTensorIdx = Tensor(
      std::make_unique<JitTensorBase>(
          CustomNode::create(
              "createIota",
              {},
              tensorIdx.shape(),
              tensorIdx.type(),
              [tensorIdx](const std::vector<const Tensor*> /* inputs */) {
                return tensorIdx;
              }),
          jitBackend));
  const auto valueNode = ValueNode::create(value.copy());
  const auto indexNode = IndexNode::create(valueNode, {jitTensorIdx});
  evaluator_.eval(indexNode);
  const Tensor& resultTensor = indexNode->getResult().value();
  ASSERT_EQ(resultTensor.shape(), indexNode->shape());
  ASSERT_TRUE(allClose(resultTensor, value(tensorIdx)));
  ASSERT_TRUE(allClose(valueNode->getResult().value(), value));
  ASSERT_TRUE(allClose(
      toJitTensorBase(jitTensorIdx).node()->getResult().value(), tensorIdx));
}

TEST_F(JitEvaluatorTest, evalIndexedUpdateNodeWithoutTensorIdx) {
  // TODO
  // test multiple indexing update, like x(a)(b) = y (AF doesn't support this
  // atm, and OneDNN can't use `allClose` because it requires f64)
  const auto dtype = dtype::s32;
  const std::vector<Index> indices{1, range(0, 3, 2)};
  const auto indexedValue = iota({4, 5, 6}, {1}, dtype);
  const auto updateValue = iota({2, 6}, {1}, dtype);
  const auto indexedValueNode = ValueNode::create(indexedValue.copy());
  const auto updateValueNode = ValueNode::create(updateValue.copy());
  const auto indexedUpdateNode =
      IndexedUpdateNode::create(indexedValueNode, {indices}, updateValueNode);
  evaluator_.eval(indexedUpdateNode);
  // udpate expected result
  const auto resultValue = indexedValue.copy();
  resultValue(indices) = updateValue;
  ASSERT_TRUE(allClose(indexedUpdateNode->getResult().value(), resultValue));
  ASSERT_TRUE(allClose(indexedValueNode->getResult().value(), indexedValue));
  ASSERT_TRUE(allClose(updateValueNode->getResult().value(), updateValue));
}

TEST_F(JitEvaluatorTest, evalIndexedUpdateNodeWithTensorIdx) {
  const auto dtype = dtype::s32;
  // Tensor index has values 0..5 -- must fit in dim 0 of indexedValue
  const auto tensor = iota(Shape({2, 3}), {1}, dtype);
  auto& jitBackend =
      getOrCreateJitBackend(DefaultTensorBackend_t::getInstance());
  const auto jitTensor = Tensor(
      std::make_unique<JitTensorBase>(
          CustomNode::create(
              "createIota",
              {},
              tensor.shape(),
              tensor.type(),
              [tensor](const std::vector<const Tensor*> /* inputs */) {
                return tensor;
              }),
          jitBackend));
  const auto indexedValue = iota({8, 5, 6}, {1}, dtype);
  const auto updateValue = iota({6, 5, 6}, {1}, dtype);
  const auto indexedValueNode = ValueNode::create(indexedValue.copy());
  const auto updateValueNode = ValueNode::create(updateValue.copy());
  const auto indexedUpdateNode = IndexedUpdateNode::create(
      indexedValueNode, {{jitTensor}}, updateValueNode);
  evaluator_.eval(indexedUpdateNode);
  // udpate expected result
  const auto resultValue = indexedValue.copy();
  resultValue({tensor}) = updateValue;
  ASSERT_TRUE(allClose(indexedUpdateNode->getResult().value(), resultValue));
  ASSERT_TRUE(allClose(indexedValueNode->getResult().value(), indexedValue));
  ASSERT_TRUE(allClose(updateValueNode->getResult().value(), updateValue));
}

TEST_F(JitEvaluatorTest, evalSharedInput) {
  //   c1
  //  /  \
  //  \  /
  //   add
  // Sanity check
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto add = BinaryNode::create(c1, c1, BinaryOp::Add);
  evaluator_.eval(add);
  ASSERT_TRUE(allClose(add->getResult().value(), full(shape, 2, dtype)));
}

TEST_F(JitEvaluatorTest, evalRetainResults) {
  // c1  c2
  //  \  /
  //   add
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto c2 = ScalarNode::create(shape, dtype, 2);
  const auto add = BinaryNode::create(c1, c2, BinaryOp::Add);
  ExternalUse u1(c1); // this forces evaluator to keep c1's result
  evaluator_.eval(add);
  ASSERT_TRUE(allClose(c1->getResult().value(), full(shape, 1, dtype)));
  ASSERT_FALSE(c2->getResult().has_value()); // result not retained
  ASSERT_TRUE(allClose(add->getResult().value(), full(shape, 3, dtype)));
}

TEST_F(JitEvaluatorTest, evalReshapeNode) {
  const auto tensor = fl::rand(Shape({3, 4}), dtype::f32);
  const auto valueNode = ValueNode::create(tensor.copy());
  Shape newShape({12});
  const auto reshapeNode = ReshapeNode::create(valueNode, newShape);
  evaluator_.eval(reshapeNode);
  ASSERT_EQ(reshapeNode->getResult().value().shape(), newShape);
  ASSERT_TRUE(allClose(reshapeNode->getResult().value(), tensor.flatten()));
}

TEST_F(JitEvaluatorTest, evalTransposeNode) {
  const auto tensor = fl::rand(Shape({3, 4}), dtype::f32);
  const auto valueNode = ValueNode::create(tensor.copy());
  Shape axes({1, 0});
  const auto transposeNode = TransposeNode::create(valueNode, axes);
  evaluator_.eval(transposeNode);
  ASSERT_EQ(transposeNode->getResult().value().shape(), Shape({4, 3}));
  ASSERT_TRUE(
      allClose(transposeNode->getResult().value(), transpose(tensor, axes)));
}

TEST_F(JitEvaluatorTest, evalTileNode) {
  const auto tensor = Tensor::fromVector<float>({2, 2}, {1, 2, 3, 4});
  const auto valueNode = ValueNode::create(tensor.copy());
  const auto tileNode = TileNode::create(valueNode, Shape({2, 3}));
  evaluator_.eval(tileNode);
  ASSERT_EQ(tileNode->getResult().value().shape(), Shape({4, 6}));
  ASSERT_TRUE(allClose(tileNode->getResult().value(), tile(tensor, {2, 3})));
}

TEST_F(JitEvaluatorTest, evalConcatenateNode) {
  const auto a = Tensor::fromVector<float>({2, 2}, {1, 2, 3, 4});
  const auto b = Tensor::fromVector<float>({2, 1}, {5, 6});
  const auto aNode = ValueNode::create(a.copy());
  const auto bNode = ValueNode::create(b.copy());
  const auto concatNode = ConcatenateNode::create({aNode, bNode}, 1);
  evaluator_.eval(concatNode);
  ASSERT_EQ(concatNode->getResult().value().shape(), Shape({2, 3}));
  ASSERT_TRUE(
      allClose(concatNode->getResult().value(), fl::concatenate({a, b}, 1)));
}

TEST_F(JitEvaluatorTest, evalPadNode) {
  const auto tensor = Tensor::fromVector<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const auto valueNode = ValueNode::create(tensor.copy());
  const std::vector<std::pair<int, int>> padWidths{{1, 1}, {2, 0}};
  const auto padNode = PadNode::create(valueNode, padWidths, PadType::Constant);
  evaluator_.eval(padNode);
  ASSERT_EQ(padNode->getResult().value().shape(), Shape({4, 5}));
  ASSERT_TRUE(allClose(
      padNode->getResult().value(),
      fl::pad(tensor, padWidths, PadType::Constant)));
}

TEST_F(JitEvaluatorTest, evalFlipNode) {
  const auto tensor = Tensor::fromVector<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const auto valueNode = ValueNode::create(tensor.copy());
  const auto flipNode = FlipNode::create(valueNode, 1);
  evaluator_.eval(flipNode);
  ASSERT_EQ(flipNode->getResult().value().shape(), Shape({2, 3}));
  ASSERT_TRUE(allClose(flipNode->getResult().value(), fl::flip(tensor, 1)));
}

TEST_F(JitEvaluatorTest, evalTriangularNode) {
  if (!defaultEvaluatorBackendSupportsGeneralTriangularOps()) {
    GTEST_SKIP() << defaultEvaluatorBackendType()
                 << " backend does not yet support generalized triangular ops";
  }

  const auto tensor = fl::rand({2, 3, 3}, dtype::f32);
  const auto valueNode = ValueNode::create(tensor.copy());
  const auto lowerNode =
      TriangularNode::create(valueNode, TriangularMode::Lower, 1);
  evaluator_.eval(lowerNode);
  ASSERT_EQ(lowerNode->getResult().value().shape(), Shape({2, 3, 3}));
  ASSERT_TRUE(allClose(lowerNode->getResult().value(), fl::tril(tensor, 1)));

  const auto upperNode =
      TriangularNode::create(valueNode, TriangularMode::Upper, -1);
  evaluator_.eval(upperNode);
  ASSERT_EQ(upperNode->getResult().value().shape(), Shape({2, 3, 3}));
  ASSERT_TRUE(allClose(upperNode->getResult().value(), fl::triu(tensor, -1)));
}

TEST_F(JitEvaluatorTest, evalClipNode) {
  const auto tensor = Tensor::fromVector<float>({4}, {-1, 1, 5, 7});
  const auto low = Tensor::fromVector<float>({4}, {0, 0, 0, 0});
  const auto high = Tensor::fromVector<float>({4}, {4, 4, 4, 4});
  const auto inputNode = ValueNode::create(tensor.copy());
  const auto lowNode = ValueNode::create(low.copy());
  const auto highNode = ValueNode::create(high.copy());
  const auto clipNode = ClipNode::create(inputNode, lowNode, highNode);
  evaluator_.eval(clipNode);
  ASSERT_EQ(clipNode->getResult().value().shape(), Shape({4}));
  ASSERT_TRUE(
      allClose(clipNode->getResult().value(), fl::clip(tensor, low, high)));
}

TEST_F(JitEvaluatorTest, evalCumsumNode) {
  const auto tensor = Tensor::fromVector<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const auto valueNode = ValueNode::create(tensor.copy());
  const auto cumsumNode = CumsumNode::create(valueNode, 1);
  evaluator_.eval(cumsumNode);
  ASSERT_EQ(cumsumNode->getResult().value().shape(), Shape({2, 3}));
  ASSERT_TRUE(allClose(cumsumNode->getResult().value(), fl::cumsum(tensor, 1)));
}

TEST_F(JitEvaluatorTest, evalRollNode) {
  const auto tensor = Tensor::fromVector<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const auto valueNode = ValueNode::create(tensor.copy());
  const auto rollNode = RollNode::create(valueNode, -1, 1);
  evaluator_.eval(rollNode);
  ASSERT_EQ(rollNode->getResult().value().shape(), Shape({2, 3}));
  ASSERT_TRUE(allClose(rollNode->getResult().value(), fl::roll(tensor, -1, 1)));
}

TEST_F(JitEvaluatorTest, evalWhereNode) {
  const auto condition = Tensor::fromVector<char>({2, 2}, {1, 0, 0, 1});
  const auto x = Tensor::fromVector<int>({2, 2}, {0, 1, 2, 3});
  const auto y = Tensor::fromVector<int>({2, 2}, {4, 5, 6, 7});
  const auto conditionNode = ValueNode::create(condition.copy());
  const auto xNode = ValueNode::create(x.copy());
  const auto yNode = ValueNode::create(y.copy());
  const auto whereNode = WhereNode::create(conditionNode, xNode, yNode);
  evaluator_.eval(whereNode);
  ASSERT_EQ(whereNode->getResult().value().shape(), Shape({2, 2}));
  ASSERT_TRUE(allClose(
      whereNode->getResult().value(),
      Tensor::fromVector<int>({2, 2}, {0, 5, 6, 3})));
}

TEST_F(JitEvaluatorTest, evalCastNode) {
  const auto tensor = fl::rand(Shape({3, 4}), dtype::f32);
  const auto valueNode = ValueNode::create(tensor.copy());
  const auto castNode = CastNode::create(valueNode, dtype::f64);
  evaluator_.eval(castNode);
  ASSERT_EQ(castNode->getResult().value().type(), dtype::f64);
  ASSERT_EQ(castNode->getResult().value().shape(), Shape({3, 4}));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // Use the bare backend (not JIT-wrapped) since this test creates IR nodes
  // manually and compares evaluator results (bare tensors) with tensors created
  // via free functions like full()/iota()/rand().
  setDefaultTensorType<DefaultBaseTensorType_t>();
  fl::initLogging();
  return RUN_ALL_TESTS();
}
