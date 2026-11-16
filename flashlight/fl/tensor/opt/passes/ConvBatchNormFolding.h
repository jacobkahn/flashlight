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
 * Folds inference-mode BatchNorm into a preceding Conv2d by algebraically
 * combining the BN affine transform into the conv weights and bias.
 *
 * Pattern: Projection(BatchNorm(Conv2d(x, w, b), gamma, beta, mean, var), Result)
 *   where train=false, axes={1}, conv2d has single use
 *
 * Folding:
 *   scale = gamma / sqrt(var + eps)
 *   w_new = w * scale.reshape(outC, 1, 1, 1)
 *   b_new = (b - mean) * scale + beta
 *
 * Replaces Projection(BN, Result) with Conv2d(x, w_new, b_new), eliminating
 * the BatchNorm entirely.
 */
class ConvBatchNormFolding : public BottomUpPass {
 protected:
  NodePtr rewrite(NodePtr node) override;
};

} // namespace fl
