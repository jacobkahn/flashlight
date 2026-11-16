/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/ir/Node.h"
#include "flashlight/fl/tensor/opt/Pass.h"

namespace fl {

/**
 * A CUDA-specific JIT pass that identifies subgraphs containing cuDNN-
 * expressible anchor operations (conv2d, batchnorm, pool2d, matmul)
 * along with their adjacent pointwise neighborhoods, and replaces them
 * with a single CustomNode backed by a compiled cuDNN execution plan.
 *
 * This pass runs BEFORE ElementwiseFusion so it can claim pointwise ops
 * adjacent to cuDNN anchors. Remaining standalone pointwise chains fall
 * through to NVRTC fusion.
 *
 * Subsumes CudnnConvEpilogueFusion and CudaMatmulEpilogueFusion.
 *
 * Uses TopDownPass so the outermost (consumer) node is examined first,
 * allowing maximal subgraph growth from consumer to producer.
 */
class CudnnSubgraphLowering : public TopDownPass {
 protected:
  NodePtr rewrite(NodePtr node) override;
};

} // namespace fl
