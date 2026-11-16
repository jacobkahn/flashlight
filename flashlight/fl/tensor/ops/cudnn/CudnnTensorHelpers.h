/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/BackendRegistry.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl {
namespace detail {

/**
 * Create a bare tensor on the same backend as `like` (CudaBackend or
 * ArrayFireBackend), bypassing the default tensor type. The cuDNN ops are
 * shared by both backends, so their results must stay on the caller's backend.
 * Routes through the unified detail::makeTensor() primitive.
 *
 * Throws if `like` is a JIT tensor: cuDNN ops only see materialized inputs.
 */
Tensor makeTensorLike(const Tensor& like, const Shape& shape, fl::dtype type);

/**
 * Create a filled bare tensor on the same backend as `like`, bypassing the
 * default tensor type. Routes through the unified detail::full() primitive.
 */
Tensor fullLike(
    const Tensor& like,
    const Shape& shape,
    double value,
    fl::dtype type);

} // namespace detail
} // namespace fl
