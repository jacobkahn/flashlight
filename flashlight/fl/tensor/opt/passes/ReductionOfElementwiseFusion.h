/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/opt/Pass.h"

namespace fl {

/**
 * An optimization pass that fuses elementwise operations feeding into a
 * reduction into a single FusedReduceElementwiseNode.
 *
 * Pattern: `reduce(elementwise_chain(inputs...))` becomes a single node
 * that avoids materializing the intermediate elementwise result.
 *
 * This pass should run after ElementwiseFusion, which first fuses elementwise
 * chains into FusedElementwiseNodes. This pass then looks for:
 * - ReductionNode whose input is a FusedElementwiseNode (single use)
 * - ReductionNode whose input is a single fusable elementwise op (single use)
 *
 * The fused node is evaluated by first running the elementwise program, then
 * applying the reduction. Backend-specific passes can lower this to a single
 * fused kernel.
 */
class ReductionOfElementwiseFusion : public BottomUpPass {
 protected:
  NodePtr rewrite(NodePtr node) override;
};

} // namespace fl
