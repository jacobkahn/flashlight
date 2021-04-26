/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <af/array.h>

namespace fl {
namespace cuda {

af::array getSparseRowIndices(const af::array& rowOffsets);

void denseToSparseCSRt(
    af::array& values,
    af::array& rowIndices,
    af::array& rowOffsets,
    af::array& columnIndices,
    const af::array& input,
    const dim_t axis);

/**
 * Computes a sparse softmax along the second dimension of some input.
 */
af::array sputnikSparseSoftmax(const af::array& input, const dim_t axis);

} // namespace cuda
} // namespace fl
