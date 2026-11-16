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
 * A backend-specific pass that lowers FusedElementwiseNodes to CUDA kernels
 * compiled via NVRTC.
 *
 * Replaces each FusedElementwiseNode with a CustomNode whose eval function
 * launches the compiled kernel. The kernel reads all external inputs and
 * executes the fused program in a single pass over the data.
 */
class CudaFusedKernelLowering : public BottomUpPass {
  NodePtr lowerFusedNode(NodePtr node);
  NodePtr lowerFusedReduceNode(NodePtr node);

 protected:
  NodePtr rewrite(NodePtr node) override;
};

} // namespace fl
