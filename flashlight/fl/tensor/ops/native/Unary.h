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
 * Flip tensor along an axis. Operates on pre-allocated output/input
 * Tensors via bufferPtr() — safe for async worker threads.
 */
void flipImpl(Tensor& out, const Tensor& in, unsigned dim);

/**
 * Roll tensor along an axis. Operates on pre-allocated output/input
 * Tensors via bufferPtr() — safe for async worker threads.
 */
void rollImpl(Tensor& out, const Tensor& in, int shift, unsigned axis);

/**
 * Lower triangular masking along the last two dimensions. Zeros elements where
 * col - row > diagonal.
 * Operates on pre-allocated Tensors via bufferPtr().
 */
void trilImpl(Tensor& out, const Tensor& in, int diagonal);

/**
 * Upper triangular masking along the last two dimensions. Zeros elements where
 * col - row < diagonal.
 * Operates on pre-allocated Tensors via bufferPtr().
 */
void triuImpl(Tensor& out, const Tensor& in, int diagonal);

/**
 * Absolute value for integer types. Operates on pre-allocated Tensors
 * via bufferPtr().
 */
void absoluteIntImpl(Tensor& out, const Tensor& in);

} // namespace native
} // namespace fl
