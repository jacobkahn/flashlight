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
 * Cumulative sum along an axis. Operates on pre-allocated output/input
 * Tensors via bufferPtr() — safe for async worker threads.
 */
void cumsumImpl(Tensor& out, const Tensor& in, unsigned axis);

/**
 * Max with index along an axis (finds the maximum value and its index).
 * Operates on pre-allocated Tensors via bufferPtr().
 */
void maxWithIndexImpl(
    Tensor& valOut,
    Tensor& idxOut,
    const Tensor& in,
    const Shape& inShape,
    unsigned axis);

/**
 * Min with index along an axis (finds the minimum value and its index).
 * Operates on pre-allocated Tensors via bufferPtr().
 */
void minWithIndexImpl(
    Tensor& valOut,
    Tensor& idxOut,
    const Tensor& in,
    const Shape& inShape,
    unsigned axis);

} // namespace native
} // namespace fl
