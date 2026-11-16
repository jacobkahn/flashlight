/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <optional>

#include "flashlight/fl/runtime/CUDAStream.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/TensorAdapter.h"

namespace fl {

class CudaBackend;

/**
 * GPU memory buffer with stream-ordered allocation.
 */
struct CudaBuffer {
  void* data{nullptr};
  size_t sizeBytes{0};
  int deviceId{0};
  cudaStream_t stream{nullptr};

  CudaBuffer() = default;
  CudaBuffer(size_t bytes, int devId, cudaStream_t s);
  ~CudaBuffer();

  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;
};

/**
 * Tensor adapter backed by raw CUDA device memory.
 * Row-major C-order. Strides are in elements.
 */
class CudaTensor : public TensorAdapterBase {
  friend class CudaBackend;

  // State for tensor-as-index write-back. When a tensor is created via
  // index() with a tensor/boolean index, it stores the parent buffer and
  // the flat index map. assign()/in-place ops scatter-write back to parent.
  struct ParentIndexState {
    std::shared_ptr<CudaBuffer> parentBuffer;
    fl::dtype parentType;
    std::shared_ptr<CudaBuffer> indexMapBuffer; // device buffer of int indices
    size_t numIndices;
    size_t parentOffsetElements;
  };

  std::shared_ptr<CudaBuffer> buffer_;
  Shape shape_;
  Shape strides_; // element strides, C-order
  fl::dtype type_{fl::dtype::f32};
  size_t offsetElements_{0};
  std::shared_ptr<CUDAStream> stream_;
  std::optional<ParentIndexState> parentIndexState_;

  // Scatter modified data back to parent tensor at indexed positions.
  void scatterToParent();

 public:
  // Compute C-order strides from shape
  static Shape computeStrides(const Shape& shape);
  CudaTensor();

  /**
   * Construct from shape/type, optionally copying from host data.
   */
  CudaTensor(
      const Shape& shape,
      fl::dtype type,
      const void* ptr = nullptr,
      MemoryLocation memoryLocation = MemoryLocation::Host);

  /**
   * Sparse tensor construction (not supported -- throws).
   */
  CudaTensor(
      const Dim nRows,
      const Dim nCols,
      const Tensor& values,
      const Tensor& rowIdx,
      const Tensor& colIdx,
      StorageType storageType);

  /**
   * Construct directly from a buffer + metadata (for internal use).
   */
  CudaTensor(
      std::shared_ptr<CudaBuffer> buffer,
      Shape shape,
      Shape strides,
      fl::dtype type,
      size_t offsetElements,
      std::shared_ptr<CUDAStream> stream);

  ~CudaTensor() override = default;

  // Create a view with a different shape (metadata-only, shares buffer).
  // The tensor must be contiguous.
  Tensor reshapeView(const Shape& newShape) const;

  // Get a raw device pointer to this tensor's data (offset-adjusted).
  void* devicePtr() const;

  // Number of bytes per element for the tensor's dtype.
  size_t bytesPerElement() const;

  // Total number of data bytes for this tensor.
  size_t totalBytes() const;

  // True if contiguous in row-major C-order.
  bool isContiguousRowMajor() const;

  // TensorAdapterBase interface
  std::unique_ptr<TensorAdapterBase> clone() const override;
  TensorBackendType backendType() const override;
  TensorBackend& backend() const override;
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

  // Assignment operators
#define FL_CUDA_TENSOR_ASSIGN_OP_TYPE(OP, TYPE) \
  void OP(const TYPE& val) override;
#define FL_CUDA_TENSOR_ASSIGN_OP(OP)                 \
  FL_CUDA_TENSOR_ASSIGN_OP_TYPE(OP, Tensor);         \
  FL_CUDA_TENSOR_ASSIGN_OP_TYPE(OP, double);         \
  FL_CUDA_TENSOR_ASSIGN_OP_TYPE(OP, float);          \
  FL_CUDA_TENSOR_ASSIGN_OP_TYPE(OP, int);            \
  FL_CUDA_TENSOR_ASSIGN_OP_TYPE(OP, unsigned);       \
  FL_CUDA_TENSOR_ASSIGN_OP_TYPE(OP, bool);           \
  FL_CUDA_TENSOR_ASSIGN_OP_TYPE(OP, char);           \
  FL_CUDA_TENSOR_ASSIGN_OP_TYPE(OP, unsigned char);  \
  FL_CUDA_TENSOR_ASSIGN_OP_TYPE(OP, short);          \
  FL_CUDA_TENSOR_ASSIGN_OP_TYPE(OP, unsigned short); \
  FL_CUDA_TENSOR_ASSIGN_OP_TYPE(OP, long);           \
  FL_CUDA_TENSOR_ASSIGN_OP_TYPE(OP, unsigned long);  \
  FL_CUDA_TENSOR_ASSIGN_OP_TYPE(OP, long long);      \
  FL_CUDA_TENSOR_ASSIGN_OP_TYPE(OP, unsigned long long);

  FL_CUDA_TENSOR_ASSIGN_OP(assign);
  FL_CUDA_TENSOR_ASSIGN_OP(inPlaceAdd);
  FL_CUDA_TENSOR_ASSIGN_OP(inPlaceSubtract);
  FL_CUDA_TENSOR_ASSIGN_OP(inPlaceMultiply);
  FL_CUDA_TENSOR_ASSIGN_OP(inPlaceDivide);
#undef FL_CUDA_TENSOR_ASSIGN_OP_TYPE
#undef FL_CUDA_TENSOR_ASSIGN_OP
};

} // namespace fl
