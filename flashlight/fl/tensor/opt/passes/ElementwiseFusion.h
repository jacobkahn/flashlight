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
 * An optimization pass that fuses chains of elementwise unary and binary
 * operations into a single FusedElementwiseNode.
 *
 * A node is fusable if:
 * - It is a UnaryNode or BinaryNode with a fusable elementwise op
 * - All its uses are internal (no external uses, single-use intermediate)
 *
 * The pass walks bottom-up, and for each fusable root collects the maximal
 * chain of fusable producers, then replaces it with a FusedElementwiseNode.
 */
class ElementwiseFusion : public TopDownPass {
 protected:
  NodePtr rewrite(NodePtr node) override;
};

} // namespace fl
