/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/native/Transpose.h"

#include <cstring>
#include <numeric>

namespace fl::native {

void transposeCpu(
    void* dst,
    const void* src,
    const std::vector<Dim>& srcDims,
    const std::vector<Dim>& axes,
    size_t typeSize) {
  const auto ndim = srcDims.size();
  const size_t totalElements = std::accumulate(
      srcDims.begin(), srcDims.end(), size_t{1}, std::multiplies<>());

  // Compute dest shape: dstDims[i] = srcDims[axes[i]]
  std::vector<Dim> dstDims(ndim);
  for (size_t i = 0; i < ndim; i++) {
    dstDims[i] = srcDims[axes[i]];
  }

  // Compute source row-major strides
  std::vector<size_t> srcStrides(ndim);
  srcStrides.back() = 1;
  for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) {
    srcStrides[d] = srcStrides[d + 1] * srcDims[d + 1];
  }

  // Compute dest row-major strides
  std::vector<size_t> dstStrides(ndim);
  dstStrides.back() = 1;
  for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) {
    dstStrides[d] = dstStrides[d + 1] * dstDims[d + 1];
  }

  const auto* srcBytes = static_cast<const char*>(src);
  auto* dstBytes = static_cast<char*>(dst);

  // For each source element: decompose into coordinates, permute, write to dest
  std::vector<Dim> srcCoords(ndim);
  for (size_t i = 0; i < totalElements; ++i) {
    // Decompose flat index into source coordinates
    size_t rem = i;
    for (size_t d = 0; d < ndim; ++d) {
      srcCoords[d] = rem / srcStrides[d];
      rem %= srcStrides[d];
    }
    // Apply permutation: dst axis d takes from src axis axes[d]
    size_t dstLinear = 0;
    for (size_t d = 0; d < ndim; ++d) {
      dstLinear += srcCoords[axes[d]] * dstStrides[d];
    }
    std::memcpy(
        dstBytes + dstLinear * typeSize,
        srcBytes + i * typeSize,
        typeSize);
  }
}

} // namespace fl::native
