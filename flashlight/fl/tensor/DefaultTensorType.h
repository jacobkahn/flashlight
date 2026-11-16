/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if FL_USE_ARRAYFIRE
  #include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"
  #include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#endif
#if FL_USE_CPU
  #include "flashlight/fl/tensor/backend/cpu/CPUBackend.h"
  #include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#endif
#if FL_USE_NATIVE_CUDA
  #include "flashlight/fl/tensor/backend/cuda/CudaBackend.h"
  #include "flashlight/fl/tensor/backend/cuda/CudaTensor.h"
#endif
#if FL_USE_TENSOR_STUB
  #include "flashlight/fl/tensor/backend/stub/StubBackend.h"
  #include "flashlight/fl/tensor/backend/stub/StubTensor.h"
#endif
#include "flashlight/fl/tensor/JitTensorBase.h"

namespace fl {

/**
 * Select the default tensor type in Flashlight.
 *
 * DefaultBaseTensorType_t is always the raw storage type (CudaTensor,
 * CPUTensor, ArrayFireTensor, StubTensor).
 *
 * DefaultTensorType_t is JitTensorBase (JIT wrapping is always-on).
 *
 * DefaultTensorBackend_t always points to the underlying backend (CudaBackend,
 * CPUBackend, etc.).
 *
 * Priority among built backends: native CUDA, CPU, ArrayFire, stub.
 *
 * FL_DEFAULT_BACKEND_COMPILE_FLAG is the compile time value which will
 * be true if the default backend is available.
 */
#if FL_USE_NATIVE_CUDA
using DefaultBaseTensorType_t = fl::CudaTensor;
using DefaultTensorBackend_t = fl::CudaBackend;
#define FL_DEFAULT_BACKEND_COMPILE_FLAG FL_USE_NATIVE_CUDA

#elif FL_USE_CPU
using DefaultBaseTensorType_t = fl::CPUTensor;
using DefaultTensorBackend_t = fl::CPUBackend;
#define FL_DEFAULT_BACKEND_COMPILE_FLAG FL_USE_CPU

#elif FL_USE_ARRAYFIRE
using DefaultBaseTensorType_t = fl::ArrayFireTensor;
using DefaultTensorBackend_t = fl::ArrayFireBackend;
#define FL_DEFAULT_BACKEND_COMPILE_FLAG FL_USE_ARRAYFIRE

#elif FL_USE_TENSOR_STUB
using DefaultBaseTensorType_t = fl::StubTensor;
using DefaultTensorBackend_t = fl::StubBackend;
#define FL_DEFAULT_BACKEND_COMPILE_FLAG FL_USE_TENSOR_STUB

#else
#error Unreachable - no tensor backend selected.
#endif

using DefaultTensorType_t = fl::JitTensorBase;

/**
 * Returns a TensorBackend instance for the default tensor type, even if changed
 * at runtime.
 */
TensorBackend& defaultTensorBackend();

// ---- Template implementations (need DefaultBaseTensorType_t etc.) ----

template <typename WrappedT>
void setDefaultJitTensorType() {
  auto& wrappedBackend = WrappedT().backend();
  auto& jb = getOrCreateJitBackend(wrappedBackend);
  detail::DefaultTensorType::getInstance().swap({
      /* denseFactory */ [](const Shape& s, dtype t, const void* p,
                            MemoryLocation m) {
        return toTensor<WrappedT>(s, t, p, m);
      },
      /* sparseFactory */ [](Dim nr, Dim nc, const Tensor& v, const Tensor& ri,
                              const Tensor& ci, StorageType st) {
        return toTensor<WrappedT>(nr, nc, v, ri, ci, st);
      },
      /* jitBackend */ &jb
  });
}

template <typename WrappedT, typename Func>
void withJitTensorType(Func func) {
  auto& wrappedBackend = WrappedT().backend();
  auto& jb = getOrCreateJitBackend(wrappedBackend);
  auto old = detail::DefaultTensorType::getInstance().swap({
      [](const Shape& s, dtype t, const void* p, MemoryLocation m) {
        return toTensor<WrappedT>(s, t, p, m);
      },
      [](Dim nr, Dim nc, const Tensor& v, const Tensor& ri,
         const Tensor& ci, StorageType st) {
        return toTensor<WrappedT>(nr, nc, v, ri, ci, st);
      },
      &jb
  });
  func();
  detail::DefaultTensorType::getInstance().swap(std::move(old));
}

template <typename T>
void setDefaultTensorType() {
  // Set non-JIT default. Used by tests that compare against raw
  // backend tensors (e.g. evaluator tests).
  detail::DefaultTensorType::getInstance().swap({
      /* denseFactory */ [](const Shape& s, dtype t, const void* p,
                            MemoryLocation m) {
        return toTensor<T>(s, t, p, m);
      },
      /* sparseFactory */ [](Dim nr, Dim nc, const Tensor& v, const Tensor& ri,
                              const Tensor& ci, StorageType st) {
        return toTensor<T>(nr, nc, v, ri, ci, st);
      },
      /* jitBackend */ nullptr
  });
}

template <typename T, typename B>
void withTensorType(B func) {
  auto old = detail::DefaultTensorType::getInstance().swap({
      [](const Shape& s, dtype t, const void* p, MemoryLocation m) {
        return toTensor<T>(s, t, p, m);
      },
      [](Dim nr, Dim nc, const Tensor& v, const Tensor& ri,
         const Tensor& ci, StorageType st) {
        return toTensor<T>(nr, nc, v, ri, ci, st);
      },
      nullptr
  });
  func();
  detail::DefaultTensorType::getInstance().swap(std::move(old));
}

} // namespace fl
