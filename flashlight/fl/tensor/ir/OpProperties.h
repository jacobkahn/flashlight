/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/FusedElementwiseNode.h"
#include "flashlight/fl/tensor/ir/ReductionNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"

namespace fl {

/**
 * Infer the output dtype of a unary operation.
 *
 * Rules:
 * - Most ops preserve the input dtype (exp, log, sqrt, etc.)
 * - Predicate ops return b8 (logicalNot, isnan, isinf)
 */
dtype inferOutputDtype(UnaryOp op, dtype inputType);

/**
 * Infer the output dtype of a binary operation.
 *
 * Rules:
 * - Arithmetic ops return the promoted type of the two inputs
 * - Comparison and logical ops return b8
 * - Bitwise/shift ops return the promoted type
 *
 * Type promotion follows NumPy-style rules: wider type wins, float > int,
 * signed > unsigned at same width.
 */
dtype inferOutputDtype(BinaryOp op, dtype lhsType, dtype rhsType);

/**
 * Infer the output dtype of a reduction operation.
 *
 * Rules:
 * - Most reductions preserve dtype (amin, amax, sum, median)
 * - Mean/std promote integer inputs to f32
 * - CountNonzero returns s32
 * - Any/all return b8
 */
dtype inferOutputDtype(ReductionOp op, dtype inputType);

// promoteDtypes() is declared in fl/tensor/Types.h

/**
 * Returns true if the unary op is elementwise (output shape == input shape).
 */
bool isElementwise(UnaryOp op);

/**
 * Returns true if the binary op is elementwise (output shape == broadcast of
 * input shapes).
 */
bool isElementwise(BinaryOp op);

/**
 * Returns true if the given unary op can participate in kernel fusion.
 * Ops like tril/triu are shape-preserving but not pointwise-fusable.
 */
bool isFusableElementwise(UnaryOp op);

/**
 * Returns true if the given binary op can participate in kernel fusion.
 */
bool isFusableElementwise(BinaryOp op);

/**
 * Returns true if the binary op is commutative (a op b == b op a).
 */
bool isCommutative(BinaryOp op);

/**
 * Infer the output dtype of a fused elementwise program by simulating
 * dtype propagation through each step.
 *
 * @param inputs The input nodes to the fused program.
 * @param program The sequence of fused operations.
 * @param numExternalInputs How many of the inputs are external (vs intermediate).
 * @return The dtype of the last program step's output.
 */
dtype inferFusedProgramOutputDtype(
    const std::vector<NodePtr>& inputs,
    const std::vector<FusedOp>& program,
    unsigned numExternalInputs);

} // namespace fl
