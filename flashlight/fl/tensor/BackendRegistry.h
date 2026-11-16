/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "flashlight/fl/runtime/DeviceType.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl {

class TensorBackend;

/**
 * A registry that maps TensorBackendType to TensorBackend singletons.
 *
 * Backends self-register during static initialization via
 * FL_REGISTER_TENSOR_BACKEND. The registry is lazily initialized and
 * thread-safe.
 */
class BackendRegistry {
 public:
  static BackendRegistry& getInstance();

  /**
   * Register a backend factory. Called during static initialization.
   */
  void registerBackend(
      TensorBackendType type,
      std::function<TensorBackend&()> factory);

  /**
   * Get a backend by type. The backend singleton is created on first access.
   * Throws if the backend type is not registered.
   */
  TensorBackend& getBackend(TensorBackendType type);

  /**
   * Check if a backend type is registered (compiled in).
   */
  bool hasBackend(TensorBackendType type) const;

  /**
   * Get all registered backend types.
   */
  std::vector<TensorBackendType> getAvailableBackends() const;

 private:
  BackendRegistry() = default;

  mutable std::mutex mutex_;
  std::unordered_map<TensorBackendType, std::function<TensorBackend&()>>
      factories_;
};

/**
 * Get a tensor backend by type.
 *
 * @param[in] type the backend type to look up
 * @return a reference to the backend singleton
 * @throws std::runtime_error if the backend is not available
 */
TensorBackend& getBackend(TensorBackendType type);

/**
 * Check if a backend type is available (compiled in and registered).
 */
bool hasBackend(TensorBackendType type);

/**
 * Get all available backend types.
 */
std::vector<TensorBackendType> getAvailableBackends();

/**
 * Map a DeviceType to the corresponding TensorBackendType.
 */
TensorBackendType backendTypeForDevice(DeviceType deviceType);

namespace detail {

/**
 * Create an uninitialized bare tensor on the specified backend.
 * This is the single internal primitive for device-targeted allocation.
 * Per-backend helpers (makeCPUTensor, makeCudaTensor, etc.) route through this.
 */
Tensor makeTensor(
    const Shape& shape,
    fl::dtype type,
    TensorBackendType backend);

/**
 * Create a filled bare tensor on the specified backend.
 */
Tensor full(
    const Shape& shape,
    double value,
    fl::dtype type,
    TensorBackendType backend);

/**
 * RAII helper for backend self-registration during static initialization.
 */
struct BackendRegistrar {
  BackendRegistrar(
      TensorBackendType type,
      std::function<TensorBackend&()> factory) {
    BackendRegistry::getInstance().registerBackend(
        type, std::move(factory));
  }
};

} // namespace detail

/**
 * Register a tensor backend factory. Place in the backend's .cpp file.
 *
 * Usage:
 *   FL_REGISTER_TENSOR_BACKEND(ArrayFire, ArrayFireBackend::getInstance)
 *
 * The first argument is the TensorBackendType enum value name (without
 * TensorBackendType:: prefix). The second is a callable returning
 * TensorBackend&.
 */
#define FL_REGISTER_TENSOR_BACKEND(BACKEND_TYPE, FACTORY) \
  static ::fl::detail::BackendRegistrar                   \
      _fl_backend_registrar_##BACKEND_TYPE(               \
          ::fl::TensorBackendType::BACKEND_TYPE,          \
          []() -> ::fl::TensorBackend& { return FACTORY(); })

} // namespace fl
