/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/opt/JitOptimizerExtension.h"

namespace fl {

/**
 * CUDA-specific JIT optimizer extension.
 *
 * Compound-op passes: CudnnSubgraphLowering (claims anchor ops + pointwise
 *   neighborhoods for cuDNN graph execution)
 * Lowering passes: CudaFusedKernelLowering (lowers remaining fused
 *   elementwise to NVRTC kernels)
 */
class CudaJitOptimizerExtension : public JitOptimizerExtension {
 public:
  std::vector<std::unique_ptr<Pass>> compoundOpPasses() override;
  std::vector<std::unique_ptr<Pass>> loweringPasses() override;
  bool isDataTypeSupported(const fl::dtype& dtype) const override;
};

} // namespace fl
