/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "flashlight/fl/tensor/Tensor.h"

namespace fl::native {

/**
 * CPU coordinate-based transpose for types not supported by backend primitives.
 *
 * Performs a permutation of axes on a contiguous source buffer, writing into
 * a contiguous destination buffer in the permuted layout. Operates on raw bytes
 * using memcpy, so it works for any element type.
 *
 * @param[out] dst        Destination buffer (must be pre-allocated, same total
 *                        bytes as src).
 * @param[in]  src        Source buffer (contiguous, row-major).
 * @param[in]  srcDims    Shape of the source tensor.
 * @param[in]  axes       Permutation in NumPy convention: axes[newAxis] = oldAxis.
 *                        Output axis `i` corresponds to input axis `axes[i]`.
 * @param[in]  typeSize   Size of each element in bytes.
 */
void transposeCpu(
    void* dst,
    const void* src,
    const std::vector<Dim>& srcDims,
    const std::vector<Dim>& axes,
    size_t typeSize);

} // namespace fl::native
