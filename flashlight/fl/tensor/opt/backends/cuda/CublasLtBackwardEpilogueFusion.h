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
 * A CUDA-specific JIT pass that fuses backward matmul with activation
 * derivatives and/or bias gradient reduction using cuBLASLt epilogues.
 *
 * Supported patterns:
 *
 *   DRELU:       Mul(relu_mask, Matmul(A, B))
 *   DGELU:       Mul(dgelu_chain(z), Matmul(A, B))
 *   BGRADB:      Sum(Matmul(A, B), {0})  [matmul has other uses]
 *   DRELU_BGRAD: Sum(Mul(relu_mask, Matmul(A, B)), {0})
 *   DGELU_BGRAD: Sum(Mul(dgelu_chain(z), Matmul(A, B)), {0})
 *
 * Each pattern is replaced with a CustomNode that calls cublasLtMatmul
 * with the appropriate epilogue. Multi-output patterns (BGRADB, *_BGRAD)
 * use CustomNode::createMultiOutput + ProjectionNode.
 *
 * Uses BottomUpPass: inputs are processed first, then each node is
 * examined for fusible patterns.
 */
class CublasLtBackwardEpilogueFusion : public BottomUpPass {
 protected:
  NodePtr rewrite(NodePtr node) override;
};

} // namespace fl
