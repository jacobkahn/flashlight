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
#include "flashlight/fl/tensor/ir/BatchNormNode.h"
#include "flashlight/fl/tensor/ir/ExternalUse.h"
#include "flashlight/fl/tensor/ir/ProjectionNode.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"

using namespace fl;

namespace {

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

// Build a BatchNorm graph with real data for evaluation.
struct BNGraph {
  NodePtr input;
  NodePtr weight;
  NodePtr bias;
  NodePtr runningMean;
  NodePtr runningVar;
  BatchNormNodePtr bn;

  // ProjectionNodes for all 5 outputs (nullptr if not created)
  ProjectionNodePtr proj[BatchNormNode::kNumOutputs];

  // Keep copies for reference computation
  Tensor inputVal, weightVal, biasVal, rmVal, rvVal;

  BNGraph(
      const Shape& inShape = {1, 4, 8, 8},
      bool train = false,
      std::vector<int> axes = {1}) {
    Shape perCh({1, inShape.dim(1), 1, 1});

    inputVal = fl::rand(inShape, dtype::f32);
    weightVal = fl::full(perCh, 1.0f);
    biasVal = fl::full(perCh, 0.0f);
    rmVal = fl::full(perCh, 0.0f);
    rvVal = fl::full(perCh, 1.0f);

    input = makeValueFrom(inputVal.copy());
    weight = makeValueFrom(weightVal.copy());
    bias = makeValueFrom(biasVal.copy());
    runningMean = makeValueFrom(rmVal.copy());
    runningVar = makeValueFrom(rvVal.copy());

    bn = BatchNormNode::create(
        {input, weight, bias, runningMean, runningVar},
        axes, train,
        /*momentum=*/0.1, /*epsilon=*/1e-5,
        makeBNDescs(inShape, perCh));
  }

  // Create a projection for a given output index.
  ProjectionNodePtr createProj(BatchNormNode::Out idx) {
    const auto& desc = bn->outputDescs()[idx];
    proj[idx] = ProjectionNode::create(bn, idx, desc.shape, desc.type);
    return proj[idx];
  }
};

} // namespace

class JitDeadOutputEliminationTest : public ::testing::Test {
 protected:
  JitDeadOutputEliminationTest()
      : evaluator_(makeBackendDispatch(DefaultTensorBackend_t::getInstance())) {
  }
  Evaluator evaluator_;
};

// All ProjectionNodes have users → all outputs populated.
// Use train=true so SaveMean/SaveVar are actually computed by the backend.
TEST_F(JitDeadOutputEliminationTest, allOutputsLive) {
  BNGraph g({1, 4, 8, 8}, /*train=*/true);
  // Create projections for all 5 outputs, each with an external use.
  std::vector<ExternalUse> extUses;
  for (unsigned i = 0; i < BatchNormNode::kNumOutputs; i++) {
    auto p = g.createProj(static_cast<BatchNormNode::Out>(i));
    extUses.emplace_back(p);
  }

  // Keep BN alive so we can inspect outputs after eval.
  ExternalUse bnUse(g.bn);

  evaluator_.eval(g.proj[BatchNormNode::Result]);

  // All outputs should be populated (non-empty tensors).
  for (unsigned i = 0; i < BatchNormNode::kNumOutputs; i++) {
    ASSERT_TRUE(g.bn->hasOutput(i))
        << "Output " << i << " should be set";
    ASSERT_FALSE(g.bn->getOutput(i).isEmpty())
        << "Output " << i << " should be non-empty";
  }
}

// Only Result ProjectionNode has users → SaveMean/SaveVar are empty.
TEST_F(JitDeadOutputEliminationTest, deadSaveStats) {
  BNGraph g({1, 4, 8, 8}, /*train=*/true);

  // Create projections for Result, SaveMean, SaveVar.
  auto resultProj = g.createProj(BatchNormNode::Result);
  g.createProj(BatchNormNode::SaveMean);
  g.createProj(BatchNormNode::SaveVar);

  // Only Result has an external use.
  ExternalUse resultUse(resultProj);
  // Keep BN alive for inspection.
  ExternalUse bnUse(g.bn);

  evaluator_.eval(resultProj);

  // Result should be populated.
  ASSERT_TRUE(g.bn->hasOutput(BatchNormNode::Result));
  ASSERT_FALSE(g.bn->getOutput(BatchNormNode::Result).isEmpty());

  // SaveMean and SaveVar projections exist but have no users → dead outputs.
  ASSERT_TRUE(g.bn->hasOutput(BatchNormNode::SaveMean));
  ASSERT_TRUE(g.bn->getOutput(BatchNormNode::SaveMean).isEmpty());

  ASSERT_TRUE(g.bn->hasOutput(BatchNormNode::SaveVar));
  ASSERT_TRUE(g.bn->getOutput(BatchNormNode::SaveVar).isEmpty());
}

// Only Result is live → outputs 1-4 all empty.
TEST_F(JitDeadOutputEliminationTest, inferenceAllButResult) {
  BNGraph g;

  auto resultProj = g.createProj(BatchNormNode::Result);
  ExternalUse resultUse(resultProj);
  // Keep BN alive for inspection.
  ExternalUse bnUse(g.bn);

  evaluator_.eval(resultProj);

  // Result should be populated.
  ASSERT_TRUE(g.bn->hasOutput(BatchNormNode::Result));
  ASSERT_FALSE(g.bn->getOutput(BatchNormNode::Result).isEmpty());

  // All other outputs should be empty (dead — no projections created).
  for (unsigned i = 1; i < BatchNormNode::kNumOutputs; i++) {
    ASSERT_TRUE(g.bn->hasOutput(i))
        << "Output " << i << " should be set (to empty)";
    ASSERT_TRUE(g.bn->getOutput(i).isEmpty())
        << "Output " << i << " should be empty (dead)";
  }

  // cachedState_ should not have been populated (wantState=false).
  ASSERT_EQ(g.bn->cachedState_, nullptr);
}

// ProjectionNode with external use but no internal uses → output is populated.
TEST_F(JitDeadOutputEliminationTest, externalUseKeepsAlive) {
  BNGraph g({1, 4, 8, 8}, /*train=*/true);

  auto resultProj = g.createProj(BatchNormNode::Result);
  auto saveMeanProj = g.createProj(BatchNormNode::SaveMean);

  // Both have external uses.
  ExternalUse resultUse(resultProj);
  ExternalUse saveMeanUse(saveMeanProj);
  // Keep BN alive for inspection.
  ExternalUse bnUse(g.bn);

  evaluator_.eval(resultProj);

  // Both should be populated (train=true, so SaveMean is non-empty).
  ASSERT_FALSE(g.bn->getOutput(BatchNormNode::Result).isEmpty());
  ASSERT_FALSE(g.bn->getOutput(BatchNormNode::SaveMean).isEmpty());
}

// Verify Result correctness when other outputs are dead.
TEST_F(JitDeadOutputEliminationTest, numericalInference) {
  const Shape inShape({1, 2, 3, 3});
  BNGraph g(inShape, /*train=*/false, {1});

  // Only Result is live.
  auto resultProj = g.createProj(BatchNormNode::Result);
  ExternalUse resultUse(resultProj);

  evaluator_.eval(resultProj);

  // Result should be accessible via projection.
  ASSERT_TRUE(resultProj->getResult().has_value());

  // Compute reference via eager batchnorm.
  auto& backend = DefaultTensorBackend_t::getInstance();
  auto ref = backend.batchnormForward(
      g.inputVal, g.weightVal, g.biasVal, g.rmVal, g.rvVal,
      {1}, /*train=*/false, 0.1, 1e-5, /*wantState=*/false);

  ASSERT_TRUE(allClose(resultProj->getResult().value(), ref.output, 1e-5));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  // Use the bare backend (not JIT-wrapped): this test builds IR nodes from
  // tensors created via free functions and evaluates them with a bare
  // Evaluator, so those tensors must be backend tensors.
  setDefaultTensorType<DefaultBaseTensorType_t>();
  return RUN_ALL_TESTS();
}
