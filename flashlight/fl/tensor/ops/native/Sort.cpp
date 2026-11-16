/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/native/Sort.h"

#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/ops/native/CpuTensorHelpers.h"

namespace fl {
namespace native {

namespace {

template <typename T>
void sortValuesTyped(
    T* dst, const T* src, size_t n,
    const std::vector<Dim>& dims, Dim axis, SortMode sortMode) {
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
      std::vector<T> slice(axisDim);
      for (Dim a = 0; a < axisDim; a++) {
        slice[a] = dst[outer * axisDim * innerSize + a * innerSize + inner];
      }
      if (sortMode == SortMode::Descending) {
        std::stable_sort(slice.begin(), slice.end(), std::greater<T>());
      } else {
        std::stable_sort(slice.begin(), slice.end());
      }
      for (Dim a = 0; a < axisDim; a++) {
        dst[outer * axisDim * innerSize + a * innerSize + inner] = slice[a];
      }
    }
  }
}

template <typename T>
void sortValuesIndicesTyped(
    T* valDst, int* idxDst, const T* src, size_t n,
    const std::vector<Dim>& dims, Dim axis, SortMode sortMode) {
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
  for (size_t outer = 0; outer < outerSize; outer++) {
    for (size_t inner = 0; inner < innerSize; inner++) {
      std::vector<std::pair<T, int>> slice(axisDim);
      for (Dim a = 0; a < axisDim; a++) {
        size_t idx = outer * axisDim * innerSize + a * innerSize + inner;
        slice[a] = {src[idx], static_cast<int>(a)};
      }
      if (sortMode == SortMode::Descending) {
        std::stable_sort(slice.begin(), slice.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
      } else {
        std::stable_sort(slice.begin(), slice.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
      }
      for (Dim a = 0; a < axisDim; a++) {
        size_t idx = outer * axisDim * innerSize + a * innerSize + inner;
        valDst[idx] = slice[a].first;
        idxDst[idx] = slice[a].second;
      }
    }
  }
}

} // namespace

void sortImpl(Tensor& out, const Tensor& in, Dim axis, SortMode sortMode) {
  auto dims = in.shape().get();
  size_t n = in.shape().elements();
  dispatchByType(in.type(), [&](auto typeTag) {
    using T = decltype(typeTag);
    sortValuesTyped(
        static_cast<T*>(toCPUTensor(out).bufferPtr()),
        static_cast<const T*>(toCPUTensor(in).bufferPtr()),
        n, dims, axis, sortMode);
  });
}

void sortWithIndicesImpl(
    Tensor& valOut, Tensor& idxOut, const Tensor& in,
    Dim axis, SortMode sortMode) {
  auto dims = in.shape().get();
  size_t n = in.shape().elements();
  dispatchByType(in.type(), [&](auto typeTag) {
    using T = decltype(typeTag);
    sortValuesIndicesTyped(
        static_cast<T*>(toCPUTensor(valOut).bufferPtr()),
        static_cast<int*>(toCPUTensor(idxOut).bufferPtr()),
        static_cast<const T*>(toCPUTensor(in).bufferPtr()),
        n, dims, axis, sortMode);
  });
}

} // namespace native
} // namespace fl
