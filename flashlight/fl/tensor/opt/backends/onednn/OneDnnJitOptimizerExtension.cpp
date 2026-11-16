/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/opt/backends/onednn/OneDnnJitOptimizerExtension.h"

#include "flashlight/fl/tensor/opt/backends/onednn/OneDnnOpFusion.h"
#include "flashlight/fl/tensor/opt/backends/onednn/OneDnnPrimitivePostOpFusion.h"
#include "flashlight/fl/tensor/backend/cpu/CPUBackend.h"

namespace fl {

std::vector<std::unique_ptr<Pass>> OneDnnJitOptimizerExtension::compoundOpPasses() {
  std::vector<std::unique_ptr<Pass>> passes;
  // Primitive post-op fusion runs in Phase 2 (before ElementwiseFusion) so it
  // can claim conv2d/pool2d + adjacent pointwise neighborhoods (add, relu, etc.)
  // before ElementwiseFusion absorbs those pointwise ops into FusedElementwiseNodes.
  passes.emplace_back(std::make_unique<OneDnnPrimitivePostOpFusion>());
  return passes;
}

std::vector<std::unique_ptr<Pass>> OneDnnJitOptimizerExtension::loweringPasses() {
  std::vector<std::unique_ptr<Pass>> passes;
  passes.emplace_back(std::make_unique<OneDnnOpFusion>());
  return passes;
}

bool OneDnnJitOptimizerExtension::isDataTypeSupported(
    const fl::dtype& dtype) const {
  return CPUBackend::getInstance().isDataTypeSupported(dtype);
}

} // namespace fl
