/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <utility>

#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl {

DevicePtr::DevicePtr(const Tensor& in)
    : tensor_(std::make_unique<Tensor>(in.shallowCopy())) {
  if (tensor_->isEmpty()) {
    ptr_ = nullptr;
  } else {
    ptr_ = tensor_->deviceSync<void>();
  }
}

DevicePtr::~DevicePtr() {
  // No explicit unlock needed. Buffer lifetime is managed by the Tensor
  // object (held via shallow copy in tensor_). Backend-specific cleanup
  // (if any) is handled by the tensor adapter's destructor.
}

DevicePtr::DevicePtr(DevicePtr&& d) noexcept
    : tensor_(std::move(d.tensor_)), ptr_(d.ptr_) {
  d.ptr_ = nullptr;
}

DevicePtr& DevicePtr::operator=(DevicePtr&& other) noexcept {
  // Old tensor_ (if any) is destroyed here; its adapter destructor
  // handles any backend-specific cleanup.
  tensor_ = std::move(other.tensor_);
  ptr_ = other.ptr_;
  other.ptr_ = nullptr;
  return *this;
}

void* DevicePtr::get() const {
  return ptr_;
}

} // namespace fl
