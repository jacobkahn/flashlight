/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorExtension.h"
#include "flashlight/fl/tensor/opt/Pass.h"

namespace fl {

/**
 * Tensor Extension to enable backend-specific JIT graph optimization.
 *
 * The Optimizer builds its pass pipeline in four phases:
 *   1. Generic simplification (ScalarFolding, CSE)
 *   2. compoundOpPasses() — claim compound operations before fusion
 *   3. Generic elementwise fusion (ElementwiseFusion, ReductionOfElementwiseFusion)
 *   4. loweringPasses() — lower fused nodes to backend-specific implementations
 */
class JitOptimizerExtension : public TensorExtension<JitOptimizerExtension> {
 public:
  static constexpr auto extensionType = TensorExtensionType::JitOptimizer;

  /**
   * Passes that claim compound operations (e.g. conv2d+relu, matmul+add)
   * before generic elementwise fusion absorbs the adjacent pointwise ops.
   *
   * Default: empty (no compound op passes).
   */
  virtual std::vector<std::unique_ptr<Pass>> compoundOpPasses() {
    return {};
  }

  /**
   * Passes that lower fused nodes to backend-specific implementations
   * (e.g. FusedElementwiseNode → NVRTC kernels).
   *
   * Default: empty (no lowering passes).
   */
  virtual std::vector<std::unique_ptr<Pass>> loweringPasses() {
    return {};
  }

  virtual bool isDataTypeSupported(const fl::dtype& dtype) const = 0;
};

} // namespace fl
