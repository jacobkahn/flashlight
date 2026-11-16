/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "flashlight/fl/tensor/Tensor.h"

namespace fl {
namespace native {

/**
 * Create a CPU-backed tensor from host data, regardless of the default
 * tensor backend. Use this in native CPU ops to avoid inadvertent
 * host-to-device copies when the default backend is a GPU type.
 */
Tensor makeCpuTensor(const Shape& shape, fl::dtype type, const void* hostPtr);

/**
 * Convenience overload that deduces the dtype from the C++ type.
 */
template <typename T>
Tensor makeCpuTensor(const Shape& shape, const T* hostPtr) {
  return makeCpuTensor(shape, fl::dtype_traits<T>::fl_type, hostPtr);
}

/**
 * Dispatch a function by tensor dtype, passing a zero-initialized value of
 * the matching C++ type. Used to call typed template implementations.
 */
template <typename Func>
auto dispatchByType(fl::dtype type, Func func) -> decltype(func(float{})) {
  switch (type) {
    case fl::dtype::f32: return func(float{});
    case fl::dtype::f64: return func(double{});
    case fl::dtype::s32: return func(int32_t{});
    case fl::dtype::u32: return func(uint32_t{});
    case fl::dtype::s16: return func(int16_t{});
    case fl::dtype::u8: return func(uint8_t{});
    case fl::dtype::b8: return func(char{});
    default:
      throw std::runtime_error(
          "[native] Unsupported dtype for CPU fallback");
  }
}

/**
 * Compute contiguous row-major strides for a shape represented as dimensions.
 */
inline std::vector<size_t> rowMajorStrides(const std::vector<Dim>& dims) {
  std::vector<size_t> strides(dims.size());
  size_t stride = 1;
  for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
    strides[d] = stride;
    stride *= static_cast<size_t>(dims[d]);
  }
  return strides;
}

/**
 * Compute row-major linear index from multi-dimensional coordinates.
 */
inline size_t coordsToLinearIndex(
    const std::vector<Dim>& coords,
    const std::vector<size_t>& strides) {
  size_t idx = 0;
  for (size_t d = 0; d < coords.size(); ++d) {
    idx += static_cast<size_t>(coords[d]) * strides[d];
  }
  return idx;
}

/**
 * Compute row-major linear index for a broadcast input shape. Coordinates in
 * dimensions of size 1 map to input coordinate 0.
 */
inline size_t broadcastCoordsToLinearIndex(
    const std::vector<Dim>& coords,
    const std::vector<Dim>& dims,
    const std::vector<size_t>& strides) {
  size_t idx = 0;
  for (size_t d = 0; d < coords.size(); ++d) {
    if (dims[d] != 1) {
      idx += static_cast<size_t>(coords[d]) * strides[d];
    }
  }
  return idx;
}

/**
 * Decompose a row-major linear index into caller-owned coordinates.
 */
inline void linearIndexToCoords(
    size_t idx,
    const std::vector<Dim>& dims,
    const std::vector<size_t>& strides,
    std::vector<Dim>& coords) {
  coords.resize(dims.size());
  for (size_t d = 0; d < dims.size(); ++d) {
    coords[d] = static_cast<Dim>(idx / strides[d]);
    idx %= strides[d];
  }
}

} // namespace native
} // namespace fl
