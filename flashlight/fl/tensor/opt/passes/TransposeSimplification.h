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
 * An optimization pass that simplifies transpose operations in the JIT graph.
 *
 * Rewrite rules (applied bottom-up):
 * - Identity elimination: transpose(x, {0,1,...}) → x
 * - Composition/cancellation: transpose(transpose(x, p1), p2) → transpose(x, compose(p2,p1))
 * - Scalar folding: transpose(ScalarNode) → ScalarNode with permuted shape
 * - Matmul absorption: matmul(transpose(x, swap_last2), y) → matmul(x, y, Transpose)
 * - Reduction folding: reduce(transpose(x, axes), rAxes) → reduce(x, permuted_rAxes)
 * - Unary commutation: unary(transpose(x, axes)) → transpose(unary(x), axes)
 * - Binary commutation: binary(transpose(x, axes), transpose(y, axes)) → transpose(binary(x, y), axes)
 */
class TransposeSimplification : public BottomUpPass {
 protected:
  NodePtr rewrite(NodePtr node) override;
};

} // namespace fl
