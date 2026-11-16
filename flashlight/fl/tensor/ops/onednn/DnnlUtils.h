/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <memory>

#include <dnnl.hpp>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/ops/onednn/Utils.h"

namespace fl {

class AsyncCPUStream;
class Tensor;

namespace detail {

// Aliases for backward compatibility -- prefer flDimsToOneDnnDims/shapeToOneDnnDims from Utils.h
inline dnnl::memory::dims convertToDnnlDims(const std::vector<Dim>& dims) {
  return flDimsToOneDnnDims(dims);
}
inline dnnl::memory::dims convertShapeToDnnlDims(const Shape& shape) {
  return shapeToOneDnnDims(shape);
}

// Alias for backward compatibility -- prefer flToOneDnnType from Utils.h
inline dnnl::memory::data_type dnnlMapToType(const fl::dtype t) {
  return flToOneDnnType(t);
}

/**
 * A light wrapper around dnnl::memory that manages underlying memory lifetime
 * in accordance with fl::DevicePtr.
 */
class DnnlMemoryWrapper {
 public:
  DnnlMemoryWrapper(
      const Tensor& tensor,
      dnnl::memory::dims dims,
      dnnl::memory::format_tag format,
      const dnnl::engine& engine);
  DnnlMemoryWrapper() = default;

  DnnlMemoryWrapper& operator=(DnnlMemoryWrapper&& other);

  dnnl::memory getMemory() const;

  dnnl::memory::desc getDescriptor() const;

 private:
  dnnl::memory::desc descriptor_;
  dnnl::memory memory_;
  fl::DevicePtr devicePtr_;
};

/**
 * Given some an dnnl network (a ``std::vector<dnnl::primitive>``), a
 * ``dnnl::memory`` with some ordering, and a
 * ``dnnl::memory::primitive_desc``, determines whether or not the memory
 * needs to be ordered based on the primitive descriptor's required ordering.
 *
 * If so, adds a ``dnnl::reorder`` layer to the network, and returns a new
 * memory descriptor that will be properly reordered.
 */
dnnl::memory dnnlAlignOrdering(
    std::vector<dnnl::primitive>& net,
    std::vector<std::unordered_map<int, dnnl::memory>>& netArgs,
    const dnnl::memory& memory,
    const dnnl::memory::desc& desc,
    const dnnl::engine& engine);

/**
 * Enqueues a sequence of DNNL primitives for execution on the given
 * AsyncCPUStream, using the given dnnl::stream for primitive execution.
 *
 * For each primitive, passes the corresponding arguments map for that index
 * to the execution stream. The number of primitives and the number of
 * arguments must be equal, else throws.
 *
 * Primitives and args are moved into the enqueued task (taken by value).
 */
void executeNetwork(
    std::vector<dnnl::primitive> net,
    std::vector<std::unordered_map<int, dnnl::memory>> args,
    AsyncCPUStream& stream,
    dnnl::stream& nativeStream,
    std::vector<std::shared_ptr<CPUTensor::SharedData>> keepAlive = {});

/**
 * Given a flashlight pooling mode, returns the corresponding dnnl pooling
 * mode.
 */
dnnl::algorithm dnnlMapToPoolingMode(const PoolingMode mode);

} // namespace detail
} // namespace fl
