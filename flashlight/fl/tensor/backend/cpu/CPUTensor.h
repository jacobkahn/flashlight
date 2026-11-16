/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/backend/cpu/AlignedBuffer.h"
#include "flashlight/fl/tensor/backend/cpu/CPUBackend.h"

#include <memory>
#include <vector>

#include "flashlight/fl/runtime/AsyncCPUStream.h"

namespace fl {

/**
 * Tensor adapter for CPU computation. Uses OneDNN for accelerated
 * operations, but the tensor storage itself is OneDNN-free.
 */
class CPUTensor : public TensorAdapterBase {
  friend class CPUBackend;

 public:
  struct SharedData {
    // Owns the raw data allocation (64-byte aligned). Null for views that
    // borrow from a parent buffer, or when bufferOwner holds the allocation.
    std::shared_ptr<AlignedBuffer> buffer;
    // Opaque owner that keeps the buffer alive when not using AlignedBuffer.
    // E.g., wraps a dnnl::memory whose internal allocation we borrow.
    std::shared_ptr<void> bufferOwner;
    // Raw data pointer. For owning tensors via AlignedBuffer, equals
    // buffer->data. For tensors backed by bufferOwner, points into that
    // allocation. For views, points into the parent's buffer (with offset).
    void* bufferPtr{nullptr};
    // Keeps the parent's data alive for sub-views (created via index()).
    // Sub-views point into the parent's buffer, so the parent must outlive them.
    std::shared_ptr<SharedData> parent;
    // Whether the data in the buffer is ready (its computation finished).
    bool isDataReady{false};
    ~SharedData() = default;
  };

  constexpr static TensorBackendType tensorBackendType =
      TensorBackendType::CPU;

 private:
  // State for tensor-as-index write-back. When a tensor is created via
  // index() with a tensor/boolean index, it stores the parent reference and
  // the linear indices mapping gathered elements back to the parent buffer.
  // In-place ops and assign() scatter-write modified data back to the parent.
  struct ParentIndexState {
    std::shared_ptr<SharedData> parent;
    std::vector<int> indices; // linear indices into parent's flat buffer
  };

  // shared among tensors that are shallow copied
  std::shared_ptr<SharedData> sharedData_;
  Shape shape_;
  // The true FL dtype.
  fl::dtype dtype_{fl::dtype::f32};
  // Row-major strides for this tensor's view of the buffer.
  std::vector<Dim> strides_;
  std::shared_ptr<AsyncCPUStream> stream_;
  std::optional<ParentIndexState> parentIndexState_;

  // Scatter modified data back to parent tensor at indexed positions.
  // No-op if this tensor was not created via tensor-as-index indexing.
  void scatterToParent();

  // Return the underlying data handle.
  // If `isDataReady` is false, sync and set it to true.
  void* getOrEvalDataHandle();

  // Trigger computation to convert to contiguous tensor if needed, block until
  // data is fully ready.
  const void* getContiguousData();

  // return the # of bytes for the data represented by this tensor.
  size_t getSizeInBytes() const;

 public:

  /**
   * Helper constructor for shallow-copying / view creation.
   *
   * @param[in] sharedData shared data among shallow copies or view from
   * indexing
   * @param[in] shape Flashlight shape of this Tensor (which may be a view)
   */
  CPUTensor(
      std::shared_ptr<SharedData> sharedData,
      const Shape& shape);

  /**
   * Construct a CPUTensor sharing existing data with a given shape and dtype.
   * Used internally to create lightweight Tensor wrappers that share the same
   * underlying buffer (e.g., for async enqueue without deep-copying).
   *
   * @param[in] sharedData shared data among shallow copies
   * @param[in] shape Flashlight shape
   * @param[in] dtype the dtype of the tensor
   * @param[in] stream optional stream to associate with this tensor
   */
  CPUTensor(
      std::shared_ptr<SharedData> sharedData,
      const Shape& shape,
      fl::dtype dtype,
      std::shared_ptr<AsyncCPUStream> stream = nullptr);

  /**
   * Construct a CPUTensor with given shape and externally-owned buffer.
   * The dataOwner keeps the buffer alive; bufferPtr points to the actual data.
   *
   * @param[in] shape the shape of the new tensor
   * @param[in] type the dtype of the new tensor
   * @param[in] bufferPtr pointer to the buffer data
   * @param[in] dataOwner opaque shared_ptr that keeps bufferPtr alive
   * @param[in] stream optional stream to associate with this tensor
   */
  CPUTensor(
      const Shape& shape,
      fl::dtype type,
      void* bufferPtr,
      std::shared_ptr<void> dataOwner,
      std::shared_ptr<AsyncCPUStream> stream = nullptr);

  /**
   * Construct an empty CPUTensor.
   */
  CPUTensor();

  /**
   * Construct a CPUTensor using some data.
   *
   * @param[in] shape the shape of the new tensor
   * @param[in] type the type of the new tensor
   * @param[in] ptr the buffer containing underlying tensor data
   * @param[in] memoryLocation the location of the buffer
   */
  CPUTensor(
      const Shape& shape,
      fl::dtype type,
      const void* ptr,
      Location memoryLocation);

  // Constructor for a sparse CPUTensor; currently not supported.
  CPUTensor(
      const Dim nRows,
      const Dim nCols,
      const Tensor& values,
      const Tensor& rowIdx,
      const Tensor& colIdx,
      StorageType storageType);

  ~CPUTensor() override = default;

  /**
   * Returns the shared data pointer that manages the underlying buffer.
   * Used to keep buffers alive during async execution.
   */
  const std::shared_ptr<SharedData>& getSharedData() const {
    return sharedData_;
  }

  std::unique_ptr<TensorAdapterBase> clone() const override;
  TensorBackendType backendType() const override;
  CPUBackend& backend() const override;
  Tensor copy() override;
  Tensor shallowCopy() override;
  const Shape& shape() const override;
  dtype type() const override;
  bool isSparse() const override;
  Location location() const override;
  void scalarSync(void* out) override;
  void device(void** out) override;
  void hostSync(void* out) override;
  void unlock() override;
  bool isLocked() override;
  bool isContiguous() const override;
  Shape strides() const override;
  const Stream& stream() const override;
  Tensor astype(const dtype type) override;
  Tensor index(const std::vector<Index>& indices) override;
  Tensor flatten() const override;
  Tensor flat(const Index& idx) const override;
  Tensor asContiguousTensor() override;
  void setContext(void* context) override;
  void* getContext() override;
  std::string toString() override;
  std::ostream& operator<<(std::ostream& ostr) override;

  /******************** Assignment Operators ********************/
#define ASSIGN_OP_TYPE(OP, TYPE) void OP(const TYPE& val) override;

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

  /**
   * Get the raw data pointer for this tensor.
   * For owning tensors, this is buffer->data. For views, this points into the
   * parent's buffer (with offset).
   */
  void* bufferPtr();
  const void* bufferPtr() const;

  /**
   * Get the strides vector for this tensor.
   */
  const std::vector<Dim>& stridesVec() const;

  /**
   * Deep comparison over shape, type, and data (with some tolerance for float).
   *
   * @return true if this CPUTensor equals the given one.
   */
  bool equals(CPUTensor&& other);

  /**
   * Override the FL dtype reported by type().
   */
  void setFlType(fl::dtype type) {
    dtype_ = type;
  }

};

// Safe to drop `const`, as these are just checked version of `Tensor::impl`
CPUTensor& toCPUTensor(const Tensor& tensor);
CPUTensor& toCPUTensor(Tensor& tensor);

} // namespace fl
