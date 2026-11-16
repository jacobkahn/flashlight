/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/opt/Optimizer.h"

#include "flashlight/fl/tensor/opt/GraphVerifier.h"
#include "flashlight/fl/tensor/opt/JitOptimizerExtension.h"
#include "flashlight/fl/tensor/opt/passes/CommonSubexpressionElimination.h"
#include "flashlight/fl/tensor/opt/passes/ElementwiseFusion.h"
#include "flashlight/fl/tensor/opt/passes/ReductionOfElementwiseFusion.h"
#include "flashlight/fl/tensor/opt/passes/ConvBatchNormFolding.h"
#include "flashlight/fl/tensor/opt/passes/AlgebraicSimplification.h"
#include "flashlight/fl/tensor/opt/passes/ScalarFolding.h"
#include "flashlight/fl/tensor/opt/passes/TransposeSimplification.h"

namespace fl {

Optimizer::Optimizer(JitOptimizerExtension* ext) {
  // Phase 1: Generic simplification
  passes_.emplace_back(std::make_unique<ConvBatchNormFolding>());
  passes_.emplace_back(std::make_unique<TransposeSimplification>());
  passes_.emplace_back(std::make_unique<ScalarFolding>());
  passes_.emplace_back(std::make_unique<AlgebraicSimplification>());
  passes_.emplace_back(std::make_unique<CommonSubexpressionElimination>());

  // Phase 2: Backend compound-op passes (e.g. cuDNN subgraph lowering).
  // These run BEFORE elementwise fusion so they can claim anchor ops
  // (conv2d, matmul, etc.) and their adjacent pointwise neighborhoods.
  if (ext) {
    for (auto& p : ext->compoundOpPasses()) {
      passes_.push_back(std::move(p));
    }
  }

  // Phase 3: Generic elementwise fusion (fuses remaining pointwise chains)
  passes_.emplace_back(std::make_unique<ElementwiseFusion>());
  passes_.emplace_back(std::make_unique<ReductionOfElementwiseFusion>());

  // Phase 4: Backend lowering passes (e.g. FusedElementwiseNode → NVRTC)
  if (ext) {
    for (auto& p : ext->loweringPasses()) {
      passes_.push_back(std::move(p));
    }
  }
}

NodePtr Optimizer::optimize(NodePtr node) {
  const Shape rootShape = node->shape();
  const dtype rootType = node->dataType();
  for (const auto& pass : passes_) {
    node = pass->apply(node);
    verifyRootMetadata(node, rootShape, rootType);
    verifyGraph(node);
  }
  return node;
}

} // namespace fl
