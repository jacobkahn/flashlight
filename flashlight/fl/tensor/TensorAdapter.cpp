/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/TensorAdapter.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/JitTensorBase.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"


namespace fl::detail {

DefaultTensorType& DefaultTensorType::getInstance() {
  // Thread-local: each thread gets its own default, lazily initialized to
  // the compile-time default backend. This makes setDefaultJitTensorType and
  // withJitTensorType thread-safe without locks.
  thread_local DefaultTensorType instance;
  return instance;
}

DefaultTensorType::DefaultTensorType() {
#if FL_DEFAULT_BACKEND_COMPILE_FLAG
  denseFactory_ = [](const Shape& s, dtype t, const void* p, MemoryLocation m) {
    return toTensor<DefaultBaseTensorType_t>(s, t, p, m);
  };
  sparseFactory_ = [](Dim nr, Dim nc, const Tensor& v, const Tensor& ri,
                       const Tensor& ci, StorageType st) {
    return toTensor<DefaultBaseTensorType_t>(nr, nc, v, ri, ci, st);
  };
  // Resolved lazily: initializing the default backend (e.g. ArrayFire + CUDA)
  // is expensive and unnecessary when another default is selected first.
  defaultJitPending_ = true;
#else
  throw std::runtime_error(
      "Cannot construct DefaultTensorType: Flashlight built "
      "without an available tensor backend.");
#endif
}

JitBackend* DefaultTensorType::resolvedJitBackend() {
  if (defaultJitPending_) {
#if FL_DEFAULT_BACKEND_COMPILE_FLAG
    jitBackend_ = &getOrCreateJitBackend(DefaultTensorBackend_t::getInstance());
#endif
    defaultJitPending_ = false;
  }
  return jitBackend_;
}

JitBackend& DefaultTensorType::jitBackend() {
  if (!resolvedJitBackend()) {
    throw std::runtime_error(
        "DefaultTensorType::jitBackend() called in non-JIT mode");
  }
  return *jitBackend_;
}

std::unique_ptr<TensorAdapterBase> DefaultTensorType::createAdapter(
    const Shape& shape, dtype type, const void* ptr, MemoryLocation loc) {
  Tensor wrapped = denseFactory_(shape, type, ptr, loc);
  if (!resolvedJitBackend()) {
    return detail::releaseAdapter(std::move(wrapped));
  }
  return std::make_unique<JitTensorBase>(
      ValueNode::create(std::move(wrapped)), *jitBackend_);
}

std::unique_ptr<TensorAdapterBase> DefaultTensorType::createAdapter(
    Dim nRows, Dim nCols, const Tensor& values,
    const Tensor& rowIdx, const Tensor& colIdx, StorageType storageType) {
  Tensor wrapped = sparseFactory_(nRows, nCols, values, rowIdx, colIdx, storageType);
  if (!resolvedJitBackend()) {
    return detail::releaseAdapter(std::move(wrapped));
  }
  return std::make_unique<JitTensorBase>(
      ValueNode::create(std::move(wrapped)), *jitBackend_);
}

TensorBackend& DefaultTensorType::getBackend() {
  if (!resolvedJitBackend()) {
    // Non-JIT mode: return the raw backend directly.
    auto t = denseFactory_({0}, dtype::f32, nullptr, MemoryLocation::Host);
    return t.backend();
  }
  return *jitBackend_;
}

DefaultTensorType::State DefaultTensorType::swap(State newState) noexcept {
  State old{
      std::move(denseFactory_),
      std::move(sparseFactory_),
      jitBackend_,
      defaultJitPending_};
  denseFactory_ = std::move(newState.denseFactory);
  sparseFactory_ = std::move(newState.sparseFactory);
  jitBackend_ = newState.jitBackend;
  defaultJitPending_ = newState.defaultJitPending;
  return old;
}

} // namespace fl::detail
