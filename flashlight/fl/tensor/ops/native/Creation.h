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
 * CPU full: fill a tensor with a scalar value.
 */
template <typename T, typename V>
Tensor fullWithTypeCpu(const Shape& shape, V value, dtype type);

/**
 * CPU iota along a single axis.
 */
Tensor iotaSingleAxisCpu(
    unsigned ndims,
    unsigned axis,
    unsigned axisDim,
    dtype type);

/**
 * CPU identity matrix.
 */
Tensor identityCpu(Dim dim, dtype type);

} // namespace native
} // namespace fl

// Template implementations
#include "flashlight/fl/tensor/ops/native/Creation-inl.h"
