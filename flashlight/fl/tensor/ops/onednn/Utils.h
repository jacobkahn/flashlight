/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_set>

#include "flashlight/fl/tensor/BackendRegistry.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/Types.h"

#include <dnnl.hpp>

namespace fl {
namespace detail {

/**
 * Create a bare CPUTensor, bypassing the default tensor type.
 * Routes through the unified detail::makeTensor() primitive.
 */
inline Tensor makeCPUTensor(const Shape& shape, fl::dtype type) {
  return detail::makeTensor(shape, type, TensorBackendType::CPU);
}

/**
 * Create a filled bare CPUTensor, bypassing the default tensor type.
 * Routes through the unified detail::full() primitive.
 */
inline Tensor fullCPU(const Shape& shape, double value, fl::dtype type) {
  return detail::full(shape, value, type, TensorBackendType::CPU);
}

/**
 * Convert an fl::dtype into a OneDNN dnnl::memory::data_type
 *
 * @param[in] type a Flashlight data type.
 * @return the corresponding OneDNN memory data type.
 * @throw invalid_argument if corresponding type doesn't exist.
 */
dnnl::memory::data_type flToOneDnnType(const fl::dtype type);

/**
 * Convert a OneDNN dnnl::memory::data_type into an fl::dtype
 *
 * @param[in] type a OneDNN memory data type.
 * @return the corresponding Flashlight data type.
 * @throw invalid_argument if corresponding type doesn't exist.
 */
fl::dtype oneDnnToFlType(const dnnl::memory::data_type type);

/**
 * Return whether the given Flashlight type has a corresponding OneDNN memory
 * type.
 *
 * @param[in] shape a Flashlight shape.
 * @return true if given type has a corresponding OneDNN memory type.
 */
bool isTypeSupportedByOneDnn(const fl::dtype type);

/**
 * Get a string representation of given OneDNN memory data type.
 *
 * @return a string reprenseting given OneDNN memory data type.
 */
std::string oneDnnDataTypeToStr(const dnnl::memory::data_type type);

/**
 * Convert Flashlight dimentions to OneDNN dimensions.
 *
 * @return the corresponding OneDNN dims for given shape.
 */
dnnl::memory::dims flDimsToOneDnnDims(const std::vector<Dim>& flDims);

/**
 * Convert a Flashlight Shape to OneDNN dimensions.
 *
 * @return the corresponding OneDNN dims for given shape.
 */
dnnl::memory::dims shapeToOneDnnDims(const Shape& shape);

/**
 * Convert OneDNN dimensions to a Flashlight Shape.
 *
 * @return the corresponding shape for given OneDNN dims.
 */
Shape oneDnnDimsToShape(const dnnl::memory::dims& dims);

/**
 * Return the input type that can represent a larger range of data.
 *
 * @param[in] t1 the first input type.
 * @param[in] t2 the second input type.
 * @return the input type that can represent a larger range of data.
 */
dnnl::memory::data_type getTypeWithLargerRange(
    dnnl::memory::data_type t1,
    dnnl::memory::data_type t2);

/**
 * Create a contiguous row-major OneDNN memory descriptor based on given
 * Flashlight Shape and OneDNN type.
 *
 * @param[in] shape the Flashlight Shape.
 * @param[in] type the OneDNN data type.
 * @return the memory descriptor created.
 */
dnnl::memory::desc oneDnnContiguousMemDescFromShape(
    const Shape& shape,
    const dnnl::memory::data_type type);

/**
 * Return a copy of the given vector with items at given indices removed.
 *
 * @param[in] items vector to copy and filter.
 * @param[in] indicesToFilter indices of items to be filtered.
 * @return the filtered copy of given vector.
 */
template <typename T>
std::vector<T> removeIndices(
    const std::vector<T>& items,
    const std::vector<int>& indicesToFilter) {
  std::vector<T> itemsKept;
  std::unordered_set<int> axesToFilterSet(
      indicesToFilter.begin(), indicesToFilter.end());
  for (int idx = 0; idx < items.size(); idx++) {
    if (!axesToFilterSet.contains(idx)) {
      itemsKept.push_back(items[idx]);
    }
  }
  return itemsKept;
}

/**
 * Construct a non-owning dnnl::memory from a raw buffer + metadata.
 * O(ndim) metadata construction, no allocation. The caller is responsible
 * for keeping the buffer alive for the lifetime of the returned memory.
 *
 * @param[in] buffer raw data pointer
 * @param[in] shape Flashlight shape
 * @param[in] strides row-major strides
 * @param[in] type Flashlight dtype
 * @param[in] engine OneDNN engine to associate with
 * @return a non-owning dnnl::memory wrapping the buffer
 */
dnnl::memory makeDnnlMemory(
    void* buffer,
    const Shape& shape,
    const std::vector<Dim>& strides,
    fl::dtype type,
    const dnnl::engine& engine);

} // namespace detail
} // namespace fl
