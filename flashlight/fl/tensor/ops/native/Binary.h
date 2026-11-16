/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/Tensor.h"

namespace fl {
namespace native {

/**
 * Generic CPU binary op with broadcasting support. T is the output element
 * type. The op lambda takes (T, T) and returns T.
 * Input tensors are read as type T via toHostVectorSync<T>().
 */
template <typename T, typename Op>
Tensor cpuBinaryOpWithBroadcast(
    const Tensor& lhs,
    const Tensor& rhs,
    Op op);

/**
 * Same-shape binary op on CPU with full type dispatch.
 * Both tensors must have the same shape. Types are dispatched independently.
 */
template <typename T, typename OP>
Tensor sameShapeBinopCpu(const Tensor& lhs, const Tensor& rhs, OP op);

/**
 * Dispatch a function by tensor dtype. The function receives a
 * zero-initialized value of the matching C++ type (use decltype to recover it).
 */
template <typename Func>
auto dispatchByType(fl::dtype type, Func func) -> decltype(func(float{}));

/**
 * Element-wise logical AND with broadcasting. Returns a b8 tensor.
 */
Tensor logicalAndCpu(const Tensor& lhs, const Tensor& rhs);

/**
 * Element-wise logical OR with broadcasting. Returns a b8 tensor.
 */
Tensor logicalOrCpu(const Tensor& lhs, const Tensor& rhs);

} // namespace native
} // namespace fl

// Template implementations
#include "flashlight/fl/tensor/ops/native/Binary-inl.h"
