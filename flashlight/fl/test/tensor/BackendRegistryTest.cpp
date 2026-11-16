/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/BackendRegistry.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/TensorBackend.h"

using namespace fl;

// The default tensor backend is a JitBackend wrapping a concrete backend
// (CPU, CUDA, ArrayFire). The registry tracks the concrete backends, not the
// JIT wrapper. These tests verify that the underlying concrete backend is
// properly registered.

TEST(BackendRegistryTest, defaultBackendIsRegistered) {
  // The default backend is JIT; the wrapped/underlying backend must be registered
  auto backends = fl::getAvailableBackends();
  ASSERT_FALSE(backends.empty());
}

TEST(BackendRegistryTest, getBackendReturnsCorrectType) {
  // Use the CPU backend type which is always available and doesn't require
  // hardware initialization
  ASSERT_TRUE(fl::hasBackend(TensorBackendType::CPU));
  auto& backend = fl::getBackend(TensorBackendType::CPU);
  ASSERT_EQ(backend.backendType(), TensorBackendType::CPU);
}

TEST(BackendRegistryTest, getBackendReturnsSameInstance) {
  ASSERT_TRUE(fl::hasBackend(TensorBackendType::CPU));
  auto& backend1 = fl::getBackend(TensorBackendType::CPU);
  auto& backend2 = fl::getBackend(TensorBackendType::CPU);
  ASSERT_EQ(&backend1, &backend2);
}

TEST(BackendRegistryTest, availableBackendsNotEmpty) {
  auto backends = fl::getAvailableBackends();
  ASSERT_FALSE(backends.empty());
}

TEST(BackendRegistryTest, hasBackendFalseForTracer) {
  // Tracer backend type exists in the enum but is never registered
  ASSERT_FALSE(fl::hasBackend(TensorBackendType::Tracer));
}

TEST(BackendRegistryTest, getBackendThrowsForUnregistered) {
  ASSERT_THROW(fl::getBackend(TensorBackendType::Tracer), std::runtime_error);
}

TEST(BackendRegistryTest, backendCanCreateTensor) {
  // Use the default tensor backend (JIT) to create a tensor
  auto& backend = defaultTensorBackend();
  auto t = backend.full({2, 3}, 1.0f, fl::dtype::f32);
  ASSERT_EQ(t.shape(), Shape({2, 3}));
  ASSERT_EQ(t.scalarSync<float>(), 1.0f);
}
