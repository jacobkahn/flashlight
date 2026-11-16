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
 * Sort values in-place along an axis. Operates on pre-allocated output/input
 * Tensors via bufferPtr() — safe for async worker threads.
 */
void sortImpl(Tensor& out, const Tensor& in, Dim axis, SortMode sortMode);

/**
 * Sort values and indices in-place along an axis. Operates on pre-allocated
 * output/input Tensors via bufferPtr() — safe for async worker threads.
 */
void sortWithIndicesImpl(
    Tensor& valOut,
    Tensor& idxOut,
    const Tensor& in,
    Dim axis,
    SortMode sortMode);

} // namespace native
} // namespace fl
