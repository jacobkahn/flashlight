/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <utility>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl {

class JitBackend;

/**
 * Convert a Tensor from an adapter or via arguments for an adapter.
 *
 * @param[in] t the thing to convert
 * @return a Tensor containing the ArrayFire array
 */
template <typename Impl, typename... T>
Tensor toTensor(T&&... t) {
  return Tensor(std::make_unique<Impl>(std::forward<T>(t)...));
}

/**
 * The implementation interface for Flashlight Tensor backends.
 *
 * Defines the implementation requirements and behaviors for a particular tensor
 * backend. These implementations can be tested against default
 * backend/reference implementations.
 *
 * Derived types from TensorAdapterBase should not be constructed or operated on
 * literally. Calls to their member implementations are dispatched via visiting
 * Tensor or other interfaces.
 */
class FL_API TensorAdapterBase {
 public:
  TensorAdapterBase() = default;
  virtual ~TensorAdapterBase() = default;

  /**
   * Construct a tensor from some existing data.
   *
   * @param[in] shape the shape of the new tensor
   * @param[in] ptr the buffer containing underlying tensor data
   * @param[in] type the type of the new tensor
   * @param[in] memoryLocation the location of the buffer
   */
  TensorAdapterBase(
      const Shape& shape,
      fl::dtype type,
      void* ptr,
      MemoryLocation memoryLocation);

  TensorAdapterBase(
      const Dim nRows,
      const Dim nCols,
      const Tensor& values,
      const Tensor& rowIdx,
      const Tensor& colIdx,
      StorageType storageType);

  /**
   * Copies the tensor adapter. The copy is not required to be eager -- the
   * implementation can use copy-on-write.
   */
  virtual std::unique_ptr<TensorAdapterBase> clone() const = 0;

  /**
   * Gets the tensor's associated backend.
   *
   * @return TensorBackendType enum associated with the backend
   */
  virtual TensorBackendType backendType() const = 0;

  /**
   * Gets the backend for a tensor with this adapter implementation.
   *
   * @return the TensorBackend instance backing this particular tensor.
   */
  virtual TensorBackend& backend() const = 0;

  /**
   * Deep copy the tensor, including underlying data.
   */
  virtual Tensor copy() = 0;

  /**
   * Shallow copy the tensor - return a tensor that points to the same
   * underlying data.
   */
  virtual Tensor shallowCopy() = 0;

  /**
   * Get the shape of a tensor.
   *
   * @return the shape of the tensor
   */
  virtual const Shape& shape() const = 0;

  /**
   * Get the data type of tensor.
   *
   * @return the dtype of the tensor
   */
  virtual dtype type() const = 0;

  /**
   * Returns if the tensor is sparse.
   *
   * @return true if the tensor is sparse, else false
   */
  virtual bool isSparse() const = 0;

  /**
   * Get a tensor's location, host or some device.
   *
   * @return the tensor's location
   */
  virtual Location location() const = 0;

  /**
   * Populate a pointer with a scalar for the first element of the tensor.
   * May block until data is ready (e.g. stream sync on CPU backends).
   * For JIT backends, assumes the tensor has been evaluated.
   */
  virtual void scalarSync(void* out) = 0;

  /**
   * Returns a pointer to the tensor in device memory. For JIT backends this
   * may return a pre-allocated buffer without triggering evaluation.
   */
  virtual void device(void** out) = 0;

  /**
   * Returns a pointer to the tensor in device memory (blocking). Triggers
   * evaluation and synchronization so the data is ready for use. Default
   * implementation forwards to device().
   */
  virtual void deviceSync(void** out) { device(out); }

  /**
   * Copies tensor data into a host buffer.
   * May block until data is ready (e.g. stream sync on CPU backends).
   * For JIT backends, assumes the tensor has been evaluated.
   */
  virtual void hostSync(void* out) = 0;

  // TODO: remove unlock()/isLocked() from the TensorAdapter interface.
  // Buffer lifetime should be managed by the Tensor object itself, not by
  // explicit lock/unlock calls. Backends that need internal locking should
  // handle it in their own device()/destructor lifecycle.

  /**
   * Signals that the caller is done using a device pointer previously
   * obtained via device(). No-op for most backends.
   */
  virtual void unlock() = 0;

  /**
   * Returns true if a device pointer is currently held via device().
   * Returns false for most backends.
   */
  virtual bool isLocked() = 0;

  /**
   * Returns a bool based on Tensor contiguousness in memory.
   */
  virtual bool isContiguous() const = 0;

  /**
   * Get the dimension-wise strides for this tensor - the number of bytes to
   * step in each direction when traversing.
   */
  virtual Shape strides() const = 0;

  /**
   * Get the stream which contains(ed) the computation required to realize an
   * up-to-date value for this tensor. For instance, `device()` may not yield a
   * pointer to the up-to-date value -- to use this pointer, `Stream::sync` or
   * `Stream::relativeSync` is required.
   *
   * @return an immutable reference to the stream that contains(ed) the
   * computations which create this tensor.
   */
  virtual const Stream& stream() const = 0;

  /**
   * Returns a tensor with elements cast as a particular type
   *
   * @param[in] the type to which to cast the tensor
   * @return a tensor with element-wise cast to the new type
   */
  virtual Tensor astype(const dtype type) = 0;

  /**
   * Index into a tensor with a variable number of indices.
   *
   * @param[in] indices a vector of Index references
   * @return an indexed tensor
   */
  virtual Tensor index(const std::vector<Index>& indices) = 0;

  /**
   * Returns a representation of the tensor in 1 dimension.
   *
   * @return a 1D version of this tensor
   */
  virtual Tensor flatten() const = 0;

  /**
   * Returns a tensor indexed from this tensor but indexed as a 1D/flattened
   * tensor.
   *
   * @return a 1D version of this tensor 1D-indexed with the given index.
   */
  virtual Tensor flat(const Index& idx) const = 0;

  /**
   * Returns a copy of the tensor that is contiguous in memory.
   */
  virtual Tensor asContiguousTensor() = 0;

  /**
   * Sets arbitrary data on a tensor. May be a no-op for some backends.
   */
  virtual void setContext(void* context) = 0;

  /**
   * Sets arbitrary data on a tensor. May be a no-op for some backends.
   *
   * @return An arbitrary payload
   */
  virtual void* getContext() = 0;

  /**
   * Return a string representation of a Tensor. Not intended to be portable
   * across backends.
   */
  virtual std::string toString() = 0;

  /**
   * Write a string representation of a tensor to an output stream.
   */
  virtual std::ostream& operator<<(std::ostream& ostr) = 0;

  /******************** Assignment Operators ********************/
#define ASSIGN_OP_TYPE(OP, TYPE) virtual void OP(const TYPE& val) = 0;
#define ASSIGN_OP(OP)                 \
  ASSIGN_OP_TYPE(OP, Tensor);         \
  ASSIGN_OP_TYPE(OP, double);         \
  ASSIGN_OP_TYPE(OP, float);          \
  ASSIGN_OP_TYPE(OP, int);            \
  ASSIGN_OP_TYPE(OP, unsigned);       \
  ASSIGN_OP_TYPE(OP, bool);           \
  ASSIGN_OP_TYPE(OP, char);           \
  ASSIGN_OP_TYPE(OP, unsigned char);  \
  ASSIGN_OP_TYPE(OP, short);          \
  ASSIGN_OP_TYPE(OP, unsigned short); \
  ASSIGN_OP_TYPE(OP, long);           \
  ASSIGN_OP_TYPE(OP, unsigned long);  \
  ASSIGN_OP_TYPE(OP, long long);      \
  ASSIGN_OP_TYPE(OP, unsigned long long);

  ASSIGN_OP(assign); // =
  ASSIGN_OP(inPlaceAdd); // +=
  ASSIGN_OP(inPlaceSubtract); // -=
  ASSIGN_OP(inPlaceMultiply); // *=
  ASSIGN_OP(inPlaceDivide); // /=
#undef ASSIGN_OP_TYPE
#undef ASSIGN_OP
};

namespace detail {

/*
 * Thread-local state for the default tensor type and backend. Each thread gets
 * its own instance, defaulting to a JIT over the compile-time default backend.
 * That backend is only initialized on first use, so selecting another default
 * (setDefaultJitTensorType<T>() / withJitTensorType<T>()) never initializes
 * it.
 *
 * The dense/sparse factories always create bare backend tensors (CPUTensor,
 * CudaTensor, ArrayFireTensor, etc.). If jitBackend is non-null,
 * createAdapter() wraps them in a JitTensorBase; otherwise it returns the
 * bare adapter directly (used by test helpers like setDefaultTensorType<T>()).
 */
class FL_API DefaultTensorType {
 public:
  using DenseFactory = std::function<Tensor(
      const Shape&, dtype, const void*, MemoryLocation)>;
  using SparseFactory = std::function<Tensor(
      Dim, Dim, const Tensor&, const Tensor&, const Tensor&, StorageType)>;

  struct State {
    DenseFactory denseFactory;
    SparseFactory sparseFactory;
    // Pointer to a process-lifetime JitBackend (owned by a static map in
    // getOrCreateJitBackend). nullptr = non-JIT mode (bare adapters only),
    // unless defaultJitPending is set.
    JitBackend* jitBackend;
    // True until the compile-time default JIT backend is first needed.
    bool defaultJitPending{false};
  };

 private:
  DenseFactory denseFactory_;
  SparseFactory sparseFactory_;
  JitBackend* jitBackend_{nullptr};
  bool defaultJitPending_{false};

  // The JitBackend to wrap new tensors with (nullptr in non-JIT mode),
  // initializing the compile-time default on first use.
  JitBackend* resolvedJitBackend();

 public:
  static DefaultTensorType& getInstance(); // thread-local

  DefaultTensorType();

  /**
   * Swap the default tensor factories and JitBackend. Returns the previous
   * state.
   */
  State swap(State newState) noexcept;

  JitBackend& jitBackend();

  std::unique_ptr<TensorAdapterBase> createAdapter(
      const Shape& shape = {0},
      dtype type = dtype::f32,
      const void* ptr = nullptr,
      MemoryLocation loc = MemoryLocation::Host);

  std::unique_ptr<TensorAdapterBase> createAdapter(
      Dim nRows,
      Dim nCols,
      const Tensor& values,
      const Tensor& rowIdx,
      const Tensor& colIdx,
      StorageType storageType);

  TensorBackend& getBackend();

  DefaultTensorType(DefaultTensorType const&) = delete;
  void operator=(DefaultTensorType const&) = delete;
};

/**
 * Get an instance of the default tensor adapter.
 */
template <typename... T>
std::unique_ptr<TensorAdapterBase> getDefaultAdapter(T&&... t) {
  return DefaultTensorType::getInstance().createAdapter(std::forward<T>(t)...);
}

} // namespace detail

/**
 * Set the default tensor type for this thread. The template parameter
 * WrappedT is the inner/wrapped tensor type (e.g. CPUTensor, CudaTensor,
 * ArrayFireTensor). JIT wrapping is applied automatically.
 */
template <typename WrappedT>
void setDefaultJitTensorType();

/**
 * Temporarily set the default tensor type for this thread, restoring the
 * previous default when the callable returns.
 */
template <typename WrappedT, typename Func>
void withJitTensorType(Func func);

/**
 * Set the default tensor type directly (non-JIT). Only for special cases
 * like tests that need to bypass JIT wrapping.
 */
template <typename T>
void setDefaultTensorType();

/**
 * Temporarily set the default tensor type directly (non-JIT).
 */
template <typename T, typename B>
void withTensorType(B func);

} // namespace fl

// Template implementations are in the header that has DefaultTensorType_t
// and the concrete backend types available — they are defined after
// DefaultTensorType.h is included.
