/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/common/Defines.h"

#include <memory>

namespace fl {

class Tensor;

/**
 * DevicePtr provides an RAII wrapper for accessing the device pointer of a
 * Flashlight Tensor. It holds a shallow copy of the tensor, keeping the
 * underlying buffer alive for the lifetime of the DevicePtr. A DevicePtr is
 * movable, but not copyable.
 *
 * The returned pointer points to the tensor's raw device memory, which may not
 * be contiguous. Callers that pass the pointer to external libraries (cuDNN,
 * oneDNN, cuBLAS, etc.) should query the tensor's shape and strides and
 * provide them to the library via its descriptor/configuration API. Callers
 * that require contiguous memory should call `Tensor::asContiguous()` or
 * `Tensor::asContiguousTensor()` before constructing a DevicePtr.
 *
 * Example Usage :
 * \code{.cpp}
 * auto A = Tensor({10, 10});
 * {
 *     DevicePtr devPtr(A); // calls `.deviceSync<>()` on the tensor.
 *     void* ptr = devPtr.get();
 * }
 * // devPtr is destructed; buffer stays alive as long as A exists
 * \endcode
 *
 */
class FL_API DevicePtr {
 public:
  /**
   * Creates a null DevicePtr.
   */
  DevicePtr() : ptr_(nullptr) {}

  /**
   * @param in input tensor to get device pointer from
   */
  explicit DevicePtr(const Tensor& in);

  ~DevicePtr();

  DevicePtr(const DevicePtr& other) = delete;

  DevicePtr& operator=(const DevicePtr& other) = delete;

  DevicePtr(DevicePtr&& d) noexcept;

  DevicePtr& operator=(DevicePtr&& other) noexcept;

  bool operator==(const DevicePtr& other) const {
    return get() == other.get();
  }

  void* get() const;

  template <typename T>
  T* getAs() const {
    return reinterpret_cast<T*>(ptr_);
  }

 protected:
  std::unique_ptr<Tensor> tensor_;

 private:
  void* ptr_;
};

struct DevicePtrHasher {
  std::size_t operator()(const DevicePtr& k) const {
    return std::hash<void*>()(k.get());
  }
};

} // namespace fl
