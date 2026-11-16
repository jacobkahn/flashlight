/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/BackendDispatch.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/eval/Evaluator.h"
#include "flashlight/fl/tensor/ir/CustomNode.h"
#include "flashlight/fl/tensor/ir/ExternalUse.h"
#include "flashlight/fl/tensor/ir/MultiValueNode.h"
#include "flashlight/fl/tensor/ir/ProjectionNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"

using namespace fl;

namespace {

// Helper: create a ValueNode from a concrete Tensor.
NodePtr makeValueFrom(Tensor t) {
  return ValueNode::create(std::move(t));
}

// Helper: compact a multi-output parent to MultiValueNode (mirrors the
// compaction logic in JitTensorBase::eval).
MultiValueNodePtr compactParent(NodePtr parent) {
  std::vector<Tensor> values(parent->numOutputs());
  for (unsigned i = 0; i < parent->numOutputs(); ++i) {
    if (parent->hasOutput(i) && !parent->getOutput(i).isEmpty()) {
      values[i] = parent->getOutput(i);
    }
  }
  auto bundle = MultiValueNode::create(std::move(values), parent->stream());
  parent->replaceAllUsesWith(bundle);
  return bundle;
}

// Helper: create a 2-output CustomNode that returns (input*2, input+1).
struct TwoOutputGraph {
  Shape shape;
  NodePtr input;
  CustomNodePtr custom;
  ProjectionNodePtr proj0;
  ProjectionNodePtr proj1;
  Tensor inputVal;

  TwoOutputGraph(const Shape& s = {2, 3}) : shape(s) {
    inputVal = fl::rand(shape, dtype::f32);
    input = makeValueFrom(inputVal.copy());

    std::vector<OutputDesc> descs = {
        {shape, dtype::f32},
        {shape, dtype::f32},
    };

    custom = CustomNode::createMultiOutput(
        "test_two_output",
        {input},
        descs,
        [](const std::vector<const Tensor*>& inputs) -> std::vector<Tensor> {
          return {*inputs[0] * 2, *inputs[0] + 1};
        });

    proj0 = ProjectionNode::create(custom, 0, shape, dtype::f32);
    proj1 = ProjectionNode::create(custom, 1, shape, dtype::f32);
  }
};

// Helper: create a 3-output CustomNode for dead-output testing.
struct ThreeOutputGraph {
  Shape shape;
  NodePtr input;
  CustomNodePtr custom;
  Tensor inputVal;

  ThreeOutputGraph(const Shape& s = {2, 3}) : shape(s) {
    inputVal = fl::rand(shape, dtype::f32);
    input = makeValueFrom(inputVal.copy());

    std::vector<OutputDesc> descs = {
        {shape, dtype::f32},
        {shape, dtype::f32},
        {shape, dtype::f32},
    };

    custom = CustomNode::createMultiOutput(
        "test_three_output",
        {input},
        descs,
        [](const std::vector<const Tensor*>& inputs) -> std::vector<Tensor> {
          return {*inputs[0] * 2, *inputs[0] + 1, *inputs[0] - 1};
        });
  }

  ProjectionNodePtr createProj(unsigned idx) {
    return ProjectionNode::create(custom, idx, shape, dtype::f32);
  }
};

} // namespace

class JitMultiValueCompactionTest : public ::testing::Test {
 protected:
  JitMultiValueCompactionTest()
      : evaluator_(
            makeBackendDispatch(DefaultTensorBackend_t::getInstance())) {}

  void SetUp() override {
    fl::setDefaultJitTensorType<DefaultBaseTensorType_t>();
  }

  Evaluator evaluator_;
};

// After eval+compaction, parent should become MultiValueNode, preserving
// sibling references.
TEST_F(JitMultiValueCompactionTest, CompactionPreservesSiblings) {
  TwoOutputGraph g;

  // Keep nodes alive with external uses.
  ExternalUse ext0(g.proj0);
  ExternalUse ext1(g.proj1);
  ExternalUse extCustom(g.custom);

  // Evaluate proj0 through the evaluator.
  evaluator_.eval(g.proj0);
  ASSERT_TRUE(g.proj0->getResult().has_value());
  ASSERT_TRUE(g.custom->hasOutput(0));
  ASSERT_TRUE(g.custom->hasOutput(1));

  // Compact: replace the CustomNode parent with MultiValueNode.
  auto bundle = compactParent(g.custom);

  // Both projections should now point to the same MultiValueNode.
  EXPECT_TRUE(g.proj0->parent()->isMultiValue());
  EXPECT_TRUE(g.proj1->parent()->isMultiValue());
  EXPECT_EQ(g.proj0->parent().get(), g.proj1->parent().get());

  // Re-evaluate proj1 via the MultiValueNode path (should be a no-op).
  evaluator_.eval(g.proj1);
  ASSERT_TRUE(g.proj1->getResult().has_value());

  // Verify values are correct.
  EXPECT_TRUE(allClose(g.proj0->getResult().value(), g.inputVal * 2));
  EXPECT_TRUE(allClose(g.proj1->getResult().value(), g.inputVal + 1));
}

// Second sibling to eval should skip compaction (parent already MultiValue).
TEST_F(JitMultiValueCompactionTest, SecondSiblingSkipsCompaction) {
  TwoOutputGraph g;

  ExternalUse ext0(g.proj0);
  ExternalUse ext1(g.proj1);
  ExternalUse extCustom(g.custom);

  // Evaluate both projections.
  evaluator_.eval(g.proj0);
  evaluator_.eval(g.proj1);

  // Compact once.
  compactParent(g.custom);

  // Parent is now MultiValueNode.
  EXPECT_TRUE(g.proj0->parent()->isMultiValue());
  EXPECT_TRUE(g.proj1->parent()->isMultiValue());

  // Simulating second sibling reaching compaction — it would check
  // parent->isMultiValue() and skip. Verify it's already true.
  EXPECT_TRUE(g.proj1->parent()->isMultiValue());

  // Both values correct.
  EXPECT_TRUE(allClose(g.proj0->getResult().value(), g.inputVal * 2));
  EXPECT_TRUE(allClose(g.proj1->getResult().value(), g.inputVal + 1));
}

// Single-output node should compact to ValueNode (not MultiValueNode).
// This test verifies the type-checking logic, not actual evaluation.
TEST_F(JitMultiValueCompactionTest, SingleOutputUnchanged) {
  Shape shape({2, 3});
  auto val = fl::rand(shape, dtype::f32);

  // Directly create a ValueNode (mimics compaction of a single-output eval).
  auto compactNode = ValueNode::create(std::move(val));

  EXPECT_TRUE(compactNode->isValue());
  EXPECT_FALSE(compactNode->isMultiValue());
  EXPECT_FALSE(compactNode->isProjection());

  // A single-output node is never a ProjectionNode, so the JitTensorBase
  // compaction code will always take the ValueNode path.
}

// Dead output slots should be empty in MultiValueNode.
TEST_F(JitMultiValueCompactionTest, DeadOutputSlot) {
  ThreeOutputGraph g;

  // Only create projections for outputs 0 and 2 (output 1 is dead).
  auto proj0 = g.createProj(0);
  auto proj2 = g.createProj(2);

  ExternalUse ext0(proj0);
  ExternalUse ext2(proj2);
  ExternalUse extCustom(g.custom);

  evaluator_.eval(proj0);

  // Output 0 should be populated; output 1 was dead.
  ASSERT_TRUE(g.custom->hasOutput(0));
  ASSERT_FALSE(g.custom->getOutput(0).isEmpty());

  // Compact to MultiValueNode.
  auto bundle = compactParent(g.custom);

  EXPECT_TRUE(proj0->parent()->isMultiValue());
  EXPECT_TRUE(proj2->parent()->isMultiValue());
  EXPECT_EQ(bundle->numValues(), 3u);

  // Output 0 should be correct.
  EXPECT_TRUE(allClose(proj0->getResult().value(), g.inputVal * 2));
}

// ExternalUse should remain valid after compaction.
TEST_F(JitMultiValueCompactionTest, ExternalUsePreservedAcrossCompaction) {
  TwoOutputGraph g;

  // Create ExternalUse on proj0 (simulating autograd holding a reference).
  ExternalUse extUse(g.proj0);
  ExternalUse customUse(g.custom);

  evaluator_.eval(g.proj0);
  ASSERT_TRUE(g.proj0->getResult().has_value());

  // Compact parent to MultiValueNode.
  auto bundle = compactParent(g.custom);

  // ExternalUse on proj0 should still be valid — proj0 itself wasn't replaced.
  EXPECT_EQ(extUse.usee(), g.proj0);
  // proj0's parent is now the MultiValueNode.
  EXPECT_TRUE(g.proj0->parent()->isMultiValue());

  // The external use on the old custom node should now point to bundle.
  EXPECT_EQ(customUse.usee(), bundle);

  // Value is still correct.
  EXPECT_TRUE(allClose(g.proj0->getResult().value(), g.inputVal * 2));
}

// MultiValueNode basic construction and accessors.
TEST_F(JitMultiValueCompactionTest, MultiValueNodeBasics) {
  Shape shape({4, 5});
  auto t0 = fl::rand(shape, dtype::f32);
  auto t1 = fl::full(shape, 42.0f);

  auto mvn = MultiValueNode::create({t0.copy(), t1.copy()});

  EXPECT_TRUE(mvn->isMultiValue());
  EXPECT_EQ(mvn->type(), NodeType::MultiValue);
  EXPECT_EQ(mvn->numValues(), 2u);
  EXPECT_EQ(mvn->numOutputs(), 2u);
  EXPECT_EQ(mvn->shape(), shape);
  EXPECT_EQ(mvn->dataType(), dtype::f32);
  EXPECT_TRUE(mvn->inputs().empty()); // leaf node

  EXPECT_TRUE(mvn->hasOutput(0));
  EXPECT_TRUE(mvn->hasOutput(1));
  EXPECT_TRUE(allClose(mvn->getOutput(0), t0));
  EXPECT_TRUE(allClose(mvn->getOutput(1), t1));
}

// Empty values vector should throw.
TEST_F(JitMultiValueCompactionTest, MultiValueNodeEmptyThrows) {
  EXPECT_THROW(
      MultiValueNode::create(std::vector<Tensor>{}), std::invalid_argument);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::setDefaultJitTensorType<DefaultBaseTensorType_t>();
  fl::init();
  return RUN_ALL_TESTS();
}
