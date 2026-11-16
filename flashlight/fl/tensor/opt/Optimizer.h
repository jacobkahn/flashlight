/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

#include "flashlight/fl/tensor/ir/Node.h"
#include "flashlight/fl/tensor/opt/Pass.h"

namespace fl {

class JitOptimizerExtension;

/**
 * A JIT tree optimizer.
 *
 * Builds a fixed four-phase pass pipeline:
 *   1. ScalarFolding, CSE
 *   2. Backend compound-op passes (from JitOptimizerExtension::compoundOpPasses)
 *   3. ElementwiseFusion, ReductionOfElementwiseFusion
 *   4. Backend lowering passes (from JitOptimizerExtension::loweringPasses)
 */
class Optimizer {
  std::vector<std::unique_ptr<Pass>> passes_;

 public:
  /**
   * @param ext Backend-specific optimizer extension, or nullptr for
   *   generic-only optimization.
   */
  explicit Optimizer(JitOptimizerExtension* ext = nullptr);

  /**
   * Apply in-place optimization to nodes within the tree.
   *
   * @param[in] node the root node of the JIT tree to be optimized
   * @return root to the updated tree
   */
  NodePtr optimize(NodePtr node);
};

} // namespace fl
