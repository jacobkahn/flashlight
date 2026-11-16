/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/BackendRegistry.h"

#include <format>
#include <stdexcept>

#include "flashlight/fl/tensor/TensorBackend.h"

namespace fl {

BackendRegistry& BackendRegistry::getInstance() {
  static BackendRegistry instance;
  return instance;
}

void BackendRegistry::registerBackend(
    TensorBackendType type,
    std::function<TensorBackend&()> factory) {
  std::lock_guard<std::mutex> lock(mutex_);
  factories_[type] = std::move(factory);
}

TensorBackend& BackendRegistry::getBackend(TensorBackendType type) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = factories_.find(type);
  if (it == factories_.end()) {
    throw std::runtime_error(std::format(
        "[BackendRegistry::getBackend] Backend not available: {}",
        tensorBackendTypeToString(type)));
  }
  return it->second();
}

bool BackendRegistry::hasBackend(TensorBackendType type) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return factories_.contains(type);
}

std::vector<TensorBackendType> BackendRegistry::getAvailableBackends() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<TensorBackendType> result;
  result.reserve(factories_.size());
  for (const auto& [type, _] : factories_) {
    result.push_back(type);
  }
  return result;
}

// Free functions
TensorBackend& getBackend(TensorBackendType type) {
  return BackendRegistry::getInstance().getBackend(type);
}

bool hasBackend(TensorBackendType type) {
  return BackendRegistry::getInstance().hasBackend(type);
}

std::vector<TensorBackendType> getAvailableBackends() {
  return BackendRegistry::getInstance().getAvailableBackends();
}

TensorBackendType backendTypeForDevice(DeviceType deviceType) {
  switch (deviceType) {
    case DeviceType::x64:
      return TensorBackendType::CPU;
    case DeviceType::CUDA:
      return TensorBackendType::Cuda;
  }
  throw std::runtime_error("[backendTypeForDevice] Unknown DeviceType");
}

namespace detail {

Tensor makeTensor(
    const Shape& shape,
    fl::dtype type,
    TensorBackendType backend) {
  return getBackend(backend).empty(shape, type, nullptr);
}

Tensor full(
    const Shape& shape,
    double value,
    fl::dtype type,
    TensorBackendType backend) {
  return getBackend(backend).full(shape, value, type, nullptr);
}

} // namespace detail

} // namespace fl
