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
 * An optimization pass that applies algebraic identities to simplify the
 * JIT graph. All rules are IEEE 754 safe (no NaN/inf semantics changes).
 *
 * Binary rules:
 *   x+0, 0+x, x-0 → x           (additive identity)
 *   x*1, 1*x, x/1 → x           (multiplicative identity)
 *   x^1 → x, x^0 → 1            (power identity)
 *   x^2 → x*x, x^0.5 → sqrt(x)  (strength reduction)
 *   0-x → neg(x)                  (canonicalization)
 *
 * Unary rules:
 *   neg(neg(x)) → x, not(not(x)) → x  (involutions)
 *   abs(abs(x)) → abs(x)               (idempotent)
 *
 * Reshape rules:
 *   reshape(x, x.shape()) → x             (identity)
 *   reshape(reshape(x, s1), s2) → reshape(x, s2)  (chain)
 */
class AlgebraicSimplification : public BottomUpPass {
 protected:
  NodePtr rewrite(NodePtr node) override;
};

} // namespace fl
