/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/native/Reduction.h"

#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/ops/native/CpuTensorHelpers.h"

namespace fl {
namespace native {

namespace {

template <typename T>
void cumsumTyped(
    T* dst, const T* src, size_t n,
    const std::vector<Dim>& dims, unsigned axis) {
  Dim axisDim = dims[axis];
  size_t innerSize = 1;
  for (int d = static_cast<int>(dims.size()) - 1;
       d > static_cast<int>(axis); d--) {
    innerSize *= dims[d];
  }
  if (n == 0) {
    return; // empty input: nothing to do (and no division by zero below)
  }
  size_t outerSize = n / (axisDim * innerSize);
  std::memcpy(dst, src, n * sizeof(T));
  for (size_t outer = 0; outer < outerSize; outer++) {
    for (size_t inner = 0; inner < innerSize; inner++) {
      T acc = 0;
      for (Dim a = 0; a < axisDim; a++) {
        size_t idx = outer * axisDim * innerSize + a * innerSize + inner;
        acc += dst[idx];
        dst[idx] = acc;
      }
    }
  }
}

template <typename T, typename LessThan>
void maxWithIndexTyped(
    T* valDst, unsigned* idxDst, const T* src, size_t n,
    const std::vector<Dim>& dims, unsigned axis, LessThan lt) {
  Dim axisSize = dims[axis];
  size_t axisStride = 1;
  for (int d = static_cast<int>(dims.size()) - 1;
       d > static_cast<int>(axis); d--) {
    axisStride *= dims[d];
  }
  size_t outIdx = 0;
  auto visited = std::make_unique<bool[]>(n);
  std::memset(visited.get(), 0, n);
  for (size_t idx = 0; idx < n; idx++) {
    if (!visited[idx]) {
      visited[idx] = true;
      T maxVal = src[idx];
      unsigned maxAxisIdx = 0;
      for (Dim a = 1; a < axisSize; a++) {
        size_t elemIdx = idx + a * axisStride;
        visited[elemIdx] = true;
        T elemVal = src[elemIdx];
        if (lt(maxVal, elemVal)) {
          maxVal = elemVal;
          maxAxisIdx = static_cast<unsigned>(a);
        }
      }
      valDst[outIdx] = maxVal;
      idxDst[outIdx] = maxAxisIdx;
      outIdx++;
    }
  }
}

} // namespace

void cumsumImpl(Tensor& out, const Tensor& in, unsigned axis) {
  auto dims = in.shape().get();
  size_t n = in.shape().elements();
  dispatchByType(in.type(), [&](auto typeTag) {
    using T = decltype(typeTag);
    cumsumTyped(
        static_cast<T*>(toCPUTensor(out).bufferPtr()),
        static_cast<const T*>(toCPUTensor(in).bufferPtr()),
        n, dims, axis);
  });
}

void maxWithIndexImpl(
    Tensor& valOut, Tensor& idxOut, const Tensor& in,
    const Shape& inShape, unsigned axis) {
  auto dims = inShape.get();
  size_t n = inShape.elements();
  dispatchByType(in.type(), [&](auto typeTag) {
    using T = decltype(typeTag);
    maxWithIndexTyped(
        static_cast<T*>(toCPUTensor(valOut).bufferPtr()),
        static_cast<unsigned*>(toCPUTensor(idxOut).bufferPtr()),
        static_cast<const T*>(toCPUTensor(in).bufferPtr()),
        n, dims, axis, std::less<T>());
  });
}

void minWithIndexImpl(
    Tensor& valOut, Tensor& idxOut, const Tensor& in,
    const Shape& inShape, unsigned axis) {
  auto dims = inShape.get();
  size_t n = inShape.elements();
  dispatchByType(in.type(), [&](auto typeTag) {
    using T = decltype(typeTag);
    maxWithIndexTyped(
        static_cast<T*>(toCPUTensor(valOut).bufferPtr()),
        static_cast<unsigned*>(toCPUTensor(idxOut).bufferPtr()),
        static_cast<const T*>(toCPUTensor(in).bufferPtr()),
        n, dims, axis, std::greater<T>());
  });
}

} // namespace native
} // namespace fl
