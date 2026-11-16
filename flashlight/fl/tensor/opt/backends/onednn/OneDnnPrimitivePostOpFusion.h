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
 * Fuses chains of activations (and optional residual adds) applied after
 * conv2d, pool2d, or inference batchnorm into OneDNN post-op chains,
 * eliminating separate kernels. OneDNN supports up to 32 chained post-ops.
 *
 * The pass walks top-down from each node through single-use eltwise
 * activations, optionally through a residual add, to an anchor primitive.
 * The entire chain is replaced with a single CustomNode.
 *
 * Matched patterns (N ≥ 1 eltwise ops):
 *   eltwise_N(...eltwise_1(conv2d))           → conv2d + [eltwise post-ops]
 *   eltwise_N(...eltwise_1(add(conv2d, x)))   → conv2d + [sum, eltwise post-ops]
 *   add(conv2d, x)                            → conv2d + [sum] (sum-only)
 *   eltwise_N(...eltwise_1(pool2d))           → pool2d + [eltwise post-ops]
 *   eltwise_N(...eltwise_1(projection(bn)))   → bn + [eltwise post-ops] (inference only)
 *
 * Activations: UnaryOp with OneDNN eltwise mapping, or max(x, 0) for relu.
 * Sum post-ops are only supported for conv2d anchors.
 * Every intermediate node and the anchor must have exactly one consumer.
 */
class OneDnnPrimitivePostOpFusion : public TopDownPass {
  NodePtr rewrite(NodePtr node) override;
};

} // namespace fl
