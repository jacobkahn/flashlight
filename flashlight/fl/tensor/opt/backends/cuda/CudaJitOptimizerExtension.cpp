/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/opt/backends/cuda/CudaJitOptimizerExtension.h"

#include "flashlight/fl/tensor/backend/cuda/CudaBackend.h"
#include "flashlight/fl/tensor/opt/backends/cuda/CublasLtBackwardEpilogueFusion.h"
#include "flashlight/fl/tensor/opt/backends/cuda/CudaFusedKernelLowering.h"
#include "flashlight/fl/tensor/opt/backends/cuda/CudnnSubgraphLowering.h"

namespace fl {

std::vector<std::unique_ptr<Pass>>
CudaJitOptimizerExtension::compoundOpPasses() {
  std::vector<std::unique_ptr<Pass>> passes;
  passes.emplace_back(std::make_unique<CudnnSubgraphLowering>());
  passes.emplace_back(std::make_unique<CublasLtBackwardEpilogueFusion>());
  return passes;
}

std::vector<std::unique_ptr<Pass>>
CudaJitOptimizerExtension::loweringPasses() {
  std::vector<std::unique_ptr<Pass>> passes;
  passes.emplace_back(std::make_unique<CudaFusedKernelLowering>());
  return passes;
}

bool CudaJitOptimizerExtension::isDataTypeSupported(
    const fl::dtype& dtype) const {
  return CudaBackend::getInstance().isDataTypeSupported(dtype);
}

} // namespace fl
