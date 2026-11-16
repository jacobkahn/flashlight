/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>

#include "flashlight/fl/tensor/Shape.h"

namespace fl {
namespace native {

/**
 * Fill a host buffer with a scalar value, handling both contiguous and
 * strided (non-contiguous) layouts.
 *
 * @param data pointer to the raw host buffer
 * @param numElements logical element count
 * @param val the value to fill
 * @param ndim number of dimensions (0 for contiguous fast path)
 * @param dims shape dimensions (row-major), may be nullptr if ndim == 0
 * @param strides element strides (row-major), may be nullptr if ndim == 0
 */
template <typename T, typename DimT>
void fillBufferCpu(
    T* data,
    Dim numElements,
    T val,
    int ndim,
    const DimT* dims,
    const DimT* strides) {
  // Check if layout is contiguous (row-major).
  bool contiguous = true;
  if (ndim > 0 && dims && strides) {
    Dim expected = 1;
    for (int d = ndim - 1; d >= 0; --d) {
      if (dims[d] > 1 && strides[d] != expected) {
        contiguous = false;
        break;
      }
      expected *= dims[d];
    }
  }

  if (contiguous) {
    std::fill(data, data + numElements, val);
  } else {
    for (Dim i = 0; i < numElements; ++i) {
      Dim offset = 0;
      Dim idx = i;
      for (int d = ndim - 1; d >= 0; --d) {
        offset += (idx % dims[d]) * strides[d];
        idx /= dims[d];
      }
      data[offset] = val;
    }
  }
}

} // namespace native
} // namespace fl
