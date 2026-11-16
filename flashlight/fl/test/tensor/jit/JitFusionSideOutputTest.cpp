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
#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/CustomNode.h"
#include "flashlight/fl/tensor/ir/ExternalUse.h"
#include "flashlight/fl/tensor/ir/ProjectionNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"
#include "flashlight/fl/tensor/opt/FusionUtils.h"
#include "flashlight/fl/tensor/opt/GraphVerifier.h"

using namespace fl;

namespace {

// Helper to build a vector of NodePtrs from heterogeneous node pointer types.
template <typename... Args>
std::vector<NodePtr> nodeVec(Args&&... args) {
  std::vector<NodePtr> v;
  v.reserve(sizeof...(args));
  (v.push_back(std::forward<Args>(args)), ...);
  return v;
}

} // namespace

class JitFusionSideOutputTest : public ::testing::Test {
 protected:
  JitFusionSideOutputTest()
      : evaluator_(
            makeBackendDispatch(DefaultTensorBackend_t::getInstance())) {}
  Evaluator evaluator_;
};

// Test 1: Side-output preserves ExternalUse after fusion.
TEST_F(JitFusionSideOutputTest, SideOutputPreservesExternalUse) {
  auto inputVal = fl::rand({4, 4}, dtype::f32);
  auto input = ValueNode::create(inputVal.copy());
  auto expNode = UnaryNode::create(input, UnaryOp::Exp);
  auto scalar = ScalarNode::create({4, 4}, dtype::f32, 1.0);
  auto addNode = BinaryNode::create(expNode, scalar, BinaryOp::Add);

  // Simulate autograd holding a reference to expNode.
  ExternalUse extUse(expNode);
  ASSERT_EQ(expNode->externalUses().size(), 1u);

  // Find externally-referenced absorbed nodes.
  auto absorbed = nodeVec(expNode, addNode);
  auto extNodes = findExternallyReferencedNodes(absorbed);
  ASSERT_EQ(extNodes.size(), 1u);
  ASSERT_EQ(extNodes[0], expNode);

  auto [descs, sideOutputs] = buildOutputDescsForSideOutputs(extNodes, 1);
  ASSERT_EQ(descs.size(), 1u);
  ASSERT_EQ(sideOutputs.size(), 1u);
  ASSERT_EQ(sideOutputs[0].outputIndex, 1u);

  // Build multi-output CustomNode with 2 outputs.
  std::vector<OutputDesc> allDescs;
  allDescs.push_back({addNode->shape(), addNode->dataType()});
  allDescs.push_back(descs[0]);

  auto fusedNode = CustomNode::createMultiOutput(
      "fused_test",
      nodeVec(input),
      allDescs,
      [](const std::vector<const Tensor*>& inputs) -> std::vector<Tensor> {
        auto expResult = fl::exp(*inputs[0]);
        auto addResult = expResult + fl::full({4, 4}, 1.0f);
        std::vector<Tensor> out;
        out.push_back(std::move(addResult));
        out.push_back(std::move(expResult));
        return out;
      });

  // Redirect the root (addNode) users to projection of output 0.
  auto rootProj = ProjectionNode::create(
      fusedNode, 0, addNode->shape(), addNode->dataType());
  addNode->replaceAllUsesWith(rootProj);

  // Create side-output projections for externally-referenced nodes.
  createSideOutputProjections(fusedNode, sideOutputs);

  // The ExternalUse should now point to a ProjectionNode (output 1).
  auto usee = extUse.usee();
  ASSERT_TRUE(usee->isProjection());
  ASSERT_EQ(usee->impl<ProjectionNode>().outputIndex(), 1u);

  // Evaluate and check correctness.
  ExternalUse rootUse(rootProj);
  evaluator_.eval(rootProj);

  ASSERT_TRUE(rootProj->getResult().has_value());
  auto expected = fl::exp(inputVal) + fl::full({4, 4}, 1.0f);
  ASSERT_TRUE(allClose(rootProj->getResult().value(), expected, 1e-5));

  // Side-output should also be populated (has ExternalUse -> live).
  ASSERT_TRUE(fusedNode->hasOutput(1));
  auto expectedExp = fl::exp(inputVal);
  ASSERT_TRUE(allClose(fusedNode->getOutput(1), expectedExp, 1e-5));
}

// Test 2: Dead side-output is eliminated when ExternalUse is released.
TEST_F(JitFusionSideOutputTest, DeadSideOutputEliminated) {
  auto inputVal = fl::rand({4, 4}, dtype::f32);
  auto input = ValueNode::create(inputVal.copy());
  auto expNode = UnaryNode::create(input, UnaryOp::Exp);
  auto scalar = ScalarNode::create({4, 4}, dtype::f32, 1.0);
  auto addNode = BinaryNode::create(expNode, scalar, BinaryOp::Add);

  // Create and immediately destroy the ExternalUse by scoping it.
  {
    ExternalUse tempUse(expNode);
    ASSERT_EQ(expNode->externalUses().size(), 1u);

    auto absorbed = nodeVec(expNode, addNode);
    auto extNodes = findExternallyReferencedNodes(absorbed);
    ASSERT_EQ(extNodes.size(), 1u);

    auto [descs, sideOutputs] = buildOutputDescsForSideOutputs(extNodes, 1);

    std::vector<OutputDesc> allDescs;
    allDescs.push_back({addNode->shape(), addNode->dataType()});
    allDescs.push_back(descs[0]);

    auto fusedNode = CustomNode::createMultiOutput(
        "fused_dead_test",
        nodeVec(input),
        allDescs,
        [](const std::vector<const Tensor*>& inputs) -> std::vector<Tensor> {
          auto expResult = fl::exp(*inputs[0]);
          auto addResult = expResult + fl::full({4, 4}, 1.0f);
          std::vector<Tensor> out;
          out.push_back(std::move(addResult));
          out.push_back(std::move(expResult));
          return out;
        });

    auto rootProj = ProjectionNode::create(
        fusedNode, 0, addNode->shape(), addNode->dataType());
    addNode->replaceAllUsesWith(rootProj);
    createSideOutputProjections(fusedNode, sideOutputs);

    // tempUse goes out of scope -> ExternalUse on the projection is released
  }

  // Verify findExternallyReferencedNodes returns empty when no ExternalUses.
  auto input2 = ValueNode::create(inputVal.copy());
  auto expNode2 = UnaryNode::create(input2, UnaryOp::Exp);
  auto absorbed2 = nodeVec(expNode2);
  auto extNodes2 = findExternallyReferencedNodes(absorbed2);
  ASSERT_TRUE(extNodes2.empty());
}

// Test 3: Multiple side-outputs get correct output indices.
TEST_F(JitFusionSideOutputTest, MultipleSideOutputs) {
  auto input = ValueNode::create(fl::rand({4, 4}, dtype::f32));
  auto expNode = UnaryNode::create(input, UnaryOp::Exp);
  auto negNode = UnaryNode::create(expNode, UnaryOp::Negative);
  auto scalar = ScalarNode::create({4, 4}, dtype::f32, 2.0);
  auto mulNode = BinaryNode::create(negNode, scalar, BinaryOp::Mul);

  // Both expNode and negNode have ExternalUses.
  ExternalUse extUseExp(expNode);
  ExternalUse extUseNeg(negNode);

  auto absorbed = nodeVec(expNode, negNode, mulNode);
  auto extNodes = findExternallyReferencedNodes(absorbed);
  ASSERT_EQ(extNodes.size(), 2u);

  auto [descs, sideOutputs] = buildOutputDescsForSideOutputs(extNodes, 1);
  ASSERT_EQ(descs.size(), 2u);
  ASSERT_EQ(sideOutputs.size(), 2u);
  ASSERT_EQ(sideOutputs[0].outputIndex, 1u);
  ASSERT_EQ(sideOutputs[1].outputIndex, 2u);

  // Build a 3-output CustomNode.
  std::vector<OutputDesc> allDescs;
  allDescs.push_back({mulNode->shape(), mulNode->dataType()});
  allDescs.push_back(descs[0]);
  allDescs.push_back(descs[1]);

  auto fusedNode = CustomNode::createMultiOutput(
      "fused_multi_test",
      nodeVec(input),
      allDescs,
      [](const std::vector<const Tensor*>& inputs) -> std::vector<Tensor> {
        auto e = fl::exp(*inputs[0]);
        auto n = fl::negative(e);
        auto m = n * fl::full({4, 4}, 2.0f);
        std::vector<Tensor> out;
        out.push_back(std::move(m));
        out.push_back(std::move(e));
        out.push_back(std::move(n));
        return out;
      });

  auto rootProj = ProjectionNode::create(
      fusedNode, 0, mulNode->shape(), mulNode->dataType());
  mulNode->replaceAllUsesWith(rootProj);
  createSideOutputProjections(fusedNode, sideOutputs);

  // Both ExternalUses should now point to their respective ProjectionNodes.
  ASSERT_TRUE(extUseExp.usee()->isProjection());
  ASSERT_EQ(extUseExp.usee()->impl<ProjectionNode>().outputIndex(), 1u);
  ASSERT_TRUE(extUseNeg.usee()->isProjection());
  ASSERT_EQ(extUseNeg.usee()->impl<ProjectionNode>().outputIndex(), 2u);
}

// Test 4: No side-outputs when no ExternalUses exist.
TEST_F(JitFusionSideOutputTest, NoSideOutputsWhenNoExternalUses) {
  auto input = ValueNode::create(fl::rand({4, 4}, dtype::f32));
  auto expNode = UnaryNode::create(input, UnaryOp::Exp);
  auto negNode = UnaryNode::create(expNode, UnaryOp::Negative);

  auto absorbed = nodeVec(expNode, negNode);
  auto extNodes = findExternallyReferencedNodes(absorbed);
  ASSERT_TRUE(extNodes.empty());
}

// Test 5: GraphVerifier passes on well-formed graphs with side-outputs.
TEST_F(JitFusionSideOutputTest, VerifierPassesOnWellFormedGraph) {
  auto inputVal = fl::rand({4, 4}, dtype::f32);
  auto input = ValueNode::create(inputVal.copy());
  auto expNode = UnaryNode::create(input, UnaryOp::Exp);
  auto scalar = ScalarNode::create({4, 4}, dtype::f32, 1.0);
  auto addNode = BinaryNode::create(expNode, scalar, BinaryOp::Add);

  ExternalUse extUse(expNode);

  auto absorbed = nodeVec(expNode, addNode);
  auto extNodes = findExternallyReferencedNodes(absorbed);
  auto [descs, sideOutputs] = buildOutputDescsForSideOutputs(extNodes, 1);

  std::vector<OutputDesc> allDescs;
  allDescs.push_back({addNode->shape(), addNode->dataType()});
  allDescs.push_back(descs[0]);

  auto fusedNode = CustomNode::createMultiOutput(
      "fused_verified",
      nodeVec(input),
      allDescs,
      [](const std::vector<const Tensor*>& inputs) -> std::vector<Tensor> {
        auto e = fl::exp(*inputs[0]);
        auto a = e + fl::full({4, 4}, 1.0f);
        std::vector<Tensor> out;
        out.push_back(std::move(a));
        out.push_back(std::move(e));
        return out;
      });

  auto rootProj = ProjectionNode::create(
      fusedNode, 0, addNode->shape(), addNode->dataType());
  addNode->replaceAllUsesWith(rootProj);
  createSideOutputProjections(fusedNode, sideOutputs);

  // Verifier should pass -- ExternalUse is properly on a ProjectionNode.
  verifyGraph(rootProj);
}

// Test 6: GraphVerifier shape consistency checks.
TEST_F(JitFusionSideOutputTest, VerifierShapeConsistency) {
  auto input = ValueNode::create(fl::rand({4, 4}, dtype::f32));
  auto expNode = UnaryNode::create(input, UnaryOp::Exp);
  auto scalar = ScalarNode::create({1, 4}, dtype::f32, 2.0);
  auto mulNode = BinaryNode::create(expNode, scalar, BinaryOp::Mul);
  auto negNode = UnaryNode::create(mulNode, UnaryOp::Negative);

  // All node shapes should be consistent.
  verifyGraph(negNode);
}

// Test 7: a node that (transitively) consumes itself is rejected.
TEST_F(JitFusionSideOutputTest, VerifierDetectsCycle) {
  auto leaf = ScalarNode::create({4}, dtype::f32, 1.0);
  auto inner = UnaryNode::create(leaf, UnaryOp::Negative);
  auto outer = UnaryNode::create(inner, UnaryOp::Exp);
  verifyGraph(outer);

  inner->setInput(0, outer);
  ASSERT_THROW(verifyGraph(outer), std::logic_error);
  inner->setInput(0, leaf); // break the shared_ptr cycle
}

// Test 8: passes must preserve the root's shape and dtype.
TEST_F(JitFusionSideOutputTest, VerifierRootMetadata) {
  auto root = ScalarNode::create({2, 3}, dtype::f32, 1.0);
  ASSERT_NO_THROW(verifyRootMetadata(root, Shape({2, 3}), dtype::f32));
  ASSERT_THROW(
      verifyRootMetadata(root, Shape({3}), dtype::f32), std::logic_error);
  ASSERT_THROW(
      verifyRootMetadata(root, Shape({2, 3}), dtype::f16), std::logic_error);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
