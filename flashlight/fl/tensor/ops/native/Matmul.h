/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/Tensor.h"

namespace fl::native {

/**
 * CBLAS-based double-precision matrix multiply.
 *
 * Handles batched matmul with broadcasting, vector inputs (1D tensors),
 * and transpose flags. Uses cblas_dgemm for the inner GEMM call.
 *
 * Both inputs are cast to f64. The output is f64.
 */
Tensor matmulF64(
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp);

} // namespace fl::native
