/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/cuda/CudaTensor.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>

#include "flashlight/fl/runtime/CUDAUtils.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/backend/cuda/CudaBackend.h"
#include "flashlight/fl/tensor/ops/cuda/CudaOps.h"

namespace fl {

namespace {

size_t dtypeSize(fl::dtype t) {
  switch (t) {
    case fl::dtype::f16: return 2;
    case fl::dtype::f32: return 4;
    case fl::dtype::f64: return 8;
    case fl::dtype::b8: return 1;
    case fl::dtype::s16: return 2;
    case fl::dtype::s32: return 4;
    case fl::dtype::s64: return 8;
    case fl::dtype::u8: return 1;
    case fl::dtype::u16: return 2;
    case fl::dtype::u32: return 4;
    case fl::dtype::u64: return 8;
    case fl::dtype::bf16: return 2;
  }
  throw std::invalid_argument("Unknown dtype");
}

// Get the default CUDAStream — reuses CudaBackend's singleton stream.
// All CudaTensors share this stream so that stream-ordered allocation
// (cudaMallocAsync/cudaFreeAsync) works correctly: frees are ordered
// with respect to the same stream that cuBLAS/cuDNN/etc. operate on.
std::shared_ptr<CUDAStream> getDefaultCudaStream() {
  return CudaBackend::getInstance().cudaStreamPtr();
}

// Maximum number of dimensions for strided copy kernel
constexpr int kMaxStridedDims = 8;

struct StridedCopyInfo {
  int ndim;
  long long shape[kMaxStridedDims];
  long long dstStrides[kMaxStridedDims]; // actual strides of destination
  long long srcStrides[kMaxStridedDims]; // row-major strides of contiguous source
};

// CUDA kernel: copy elements from contiguous source to strided destination.
// Each thread handles one element indexed by its linear position in the shape.
__global__ void stridedCopyKernel(
    char* __restrict__ dst,
    const char* __restrict__ src,
    StridedCopyInfo info,
    size_t elemSize,
    size_t nElems) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= nElems) return;

  // Decompose linear index into coordinates using source (row-major) strides,
  // then compute destination offset using destination strides.
  size_t dstOffset = 0;
  size_t rem = idx;
  for (int d = 0; d < info.ndim; ++d) {
    size_t coord = rem / info.srcStrides[d];
    rem %= info.srcStrides[d];
    dstOffset += coord * info.dstStrides[d];
  }

  // Byte-level copy of one element
  const char* s = src + idx * elemSize;
  char* d = dst + dstOffset * elemSize;
  for (size_t b = 0; b < elemSize; ++b) {
    d[b] = s[b];
  }
}

// CUDA kernel: copy elements from strided source to contiguous destination.
// Reverse of stridedCopyKernel: reads from strided positions, writes linearly.
__global__ void stridedGatherKernel(
    char* __restrict__ dst,
    const char* __restrict__ src,
    StridedCopyInfo info,
    size_t elemSize,
    size_t nElems) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= nElems) return;

  // Decompose linear index into coordinates using destination (row-major)
  // strides, then compute source offset using source (strided) strides.
  size_t srcOffset = 0;
  size_t rem = idx;
  for (int d = 0; d < info.ndim; ++d) {
    size_t coord = rem / info.dstStrides[d];
    rem %= info.dstStrides[d];
    srcOffset += coord * info.srcStrides[d];
  }

  // Byte-level copy of one element
  const char* s = src + srcOffset * elemSize;
  char* d = dst + idx * elemSize;
  for (size_t b = 0; b < elemSize; ++b) {
    d[b] = s[b];
  }
}

// Gather kernel: out[i] = src[indices[i]]  (byte-level, element-granularity)
__global__ void gatherKernel(
    char* __restrict__ dst,
    const char* __restrict__ src,
    const int* __restrict__ indices,
    size_t elemSize,
    size_t nElems) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= nElems) return;
  int srcIdx = indices[idx];
  const char* s = src + srcIdx * elemSize;
  char* d = dst + idx * elemSize;
  for (size_t b = 0; b < elemSize; ++b) {
    d[b] = s[b];
  }
}

// Scatter kernel: dst[indices[i]] = src[i]  (byte-level, element-granularity)
__global__ void scatterKernel(
    char* __restrict__ dst,
    const char* __restrict__ src,
    const int* __restrict__ indices,
    size_t elemSize,
    size_t nElems) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= nElems) return;
  int dstIdx = indices[idx];
  const char* s = src + idx * elemSize;
  char* d = dst + dstIdx * elemSize;
  for (size_t b = 0; b < elemSize; ++b) {
    d[b] = s[b];
  }
}

} // namespace

// --- CudaBuffer ---

CudaBuffer::CudaBuffer(size_t bytes, int devId, cudaStream_t s)
    : sizeBytes(bytes), deviceId(devId), stream(s) {
  if (bytes > 0) {
    FL_CUDA_CHECK(cudaMallocAsync(&data, bytes, stream));
  }
}

CudaBuffer::~CudaBuffer() {
  if (data) {
    // Best-effort free -- don't throw from destructor
    cudaFreeAsync(data, stream);
  }
}

// --- CudaTensor ---

Shape CudaTensor::computeStrides(const Shape& shape) {
  if (shape.ndim() == 0) {
    return Shape{};
  }
  std::vector<Dim> strides(shape.ndim());
  Dim stride = 1;
  for (int i = shape.ndim() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return Shape(strides);
}

CudaTensor::CudaTensor()
    : shape_({0}),
      strides_(computeStrides({0})),
      type_(fl::dtype::f32) {
  stream_ = getDefaultCudaStream();
}

CudaTensor::CudaTensor(
    const Shape& shape,
    fl::dtype type,
    const void* ptr,
    MemoryLocation memoryLocation)
    : shape_(shape),
      strides_(computeStrides(shape)),
      type_(type) {
  stream_ = getDefaultCudaStream();
  size_t numElements = shape.elements();
  size_t bytes = numElements * dtypeSize(type);
  int devId = fl::cuda::getActiveDeviceId();

  buffer_ = std::make_shared<CudaBuffer>(bytes, devId, stream_->handle());

  if (ptr && bytes > 0) {
    if (memoryLocation == MemoryLocation::Host) {
      FL_CUDA_CHECK(cudaMemcpyAsync(
          buffer_->data, ptr, bytes, cudaMemcpyHostToDevice, stream_->handle()));
    } else {
      FL_CUDA_CHECK(cudaMemcpyAsync(
          buffer_->data, ptr, bytes, cudaMemcpyDeviceToDevice, stream_->handle()));
    }
  }
}

CudaTensor::CudaTensor(
    const Dim /*nRows*/,
    const Dim /*nCols*/,
    const Tensor& /*values*/,
    const Tensor& /*rowIdx*/,
    const Tensor& /*colIdx*/,
    StorageType /*storageType*/) {
  throw std::runtime_error("CudaTensor: sparse tensors not supported");
}

CudaTensor::CudaTensor(
    std::shared_ptr<CudaBuffer> buffer,
    Shape shape,
    Shape strides,
    fl::dtype type,
    size_t offsetElements,
    std::shared_ptr<CUDAStream> stream)
    : buffer_(std::move(buffer)),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      type_(type),
      offsetElements_(offsetElements),
      stream_(std::move(stream)) {}

void CudaTensor::scatterToParent() {
  if (!parentIndexState_) {
    return;
  }
  auto& state = *parentIndexState_;
  size_t n = shape_.elements();
  if (n == 0) return;

  size_t elemSz = dtypeSize(type_);
  auto stream = stream_->handle();

  // Scatter this tensor's data back to parent buffer at indexed positions
  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;
  scatterKernel<<<gridSize, blockSize, 0, stream>>>(
      static_cast<char*>(state.parentBuffer->data) +
          state.parentOffsetElements * elemSz,
      static_cast<const char*>(devicePtr()),
      static_cast<const int*>(state.indexMapBuffer->data),
      elemSz, n);
  FL_CUDA_CHECK(cudaGetLastError());
}

Tensor CudaTensor::reshapeView(const Shape& newShape) const {
  auto strides = computeStrides(newShape);
  return toTensor<CudaTensor>(buffer_, newShape, strides, type_, offsetElements_, stream_);
}

void* CudaTensor::devicePtr() const {
  if (!buffer_ || !buffer_->data) {
    return nullptr;
  }
  return static_cast<char*>(buffer_->data) + offsetElements_ * dtypeSize(type_);
}

size_t CudaTensor::bytesPerElement() const {
  return dtypeSize(type_);
}

size_t CudaTensor::totalBytes() const {
  return shape_.elements() * dtypeSize(type_);
}

bool CudaTensor::isContiguousRowMajor() const {
  auto expected = computeStrides(shape_);
  if (expected.ndim() != strides_.ndim()) {
    return false;
  }
  for (int i = 0; i < expected.ndim(); ++i) {
    if (expected[i] != strides_[i]) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<TensorAdapterBase> CudaTensor::clone() const {
  return std::make_unique<CudaTensor>(
      buffer_, shape_, strides_, type_, offsetElements_, stream_);
}

TensorBackendType CudaTensor::backendType() const {
  return TensorBackendType::Cuda;
}

TensorBackend& CudaTensor::backend() const {
  return CudaBackend::getInstance();
}

Tensor CudaTensor::copy() {
  size_t nElems = shape_.elements();
  size_t bytes = nElems * dtypeSize(type_);
  auto newBuffer = std::make_shared<CudaBuffer>(
      bytes, buffer_ ? buffer_->deviceId : fl::cuda::getActiveDeviceId(),
      stream_->handle());
  if (bytes > 0 && buffer_) {
    if (isContiguousRowMajor()) {
      FL_CUDA_CHECK(cudaMemcpyAsync(
          newBuffer->data, devicePtr(), bytes,
          cudaMemcpyDeviceToDevice, stream_->handle()));
    } else {
      // Non-contiguous: use strided gather kernel to copy directly
      // (avoids mutual recursion with asContiguousTensor)
      int ndim = shape_.ndim();
      if (ndim == 0) {
        // Scalar
        FL_CUDA_CHECK(cudaMemcpyAsync(
            newBuffer->data, devicePtr(), dtypeSize(type_),
            cudaMemcpyDeviceToDevice, stream_->handle()));
      } else if (ndim <= kMaxStridedDims) {
        StridedCopyInfo info{};
        info.ndim = ndim;
        auto rowMajorStrides = computeStrides(shape_);
        for (int d = 0; d < ndim; ++d) {
          info.shape[d] = shape_[d];
          info.srcStrides[d] = strides_[d];       // source is strided
          info.dstStrides[d] = rowMajorStrides[d]; // dest is contiguous
        }
        size_t elemSz = dtypeSize(type_);
        int blockSize = 256;
        int gridSize = (nElems + blockSize - 1) / blockSize;
        stridedGatherKernel<<<gridSize, blockSize, 0, stream_->handle()>>>(
            static_cast<char*>(newBuffer->data),
            static_cast<const char*>(devicePtr()),
            info, elemSz, nElems);
        FL_CUDA_CHECK(cudaGetLastError());
      } else {
        throw std::runtime_error(
            "CudaTensor::copy: ndim exceeds maximum for strided copy");
      }
    }
  }
  auto newStrides = computeStrides(shape_);
  return toTensor<CudaTensor>(
      std::move(newBuffer), shape_, newStrides, type_, 0, stream_);
}

Tensor CudaTensor::shallowCopy() {
  return toTensor<CudaTensor>(
      buffer_, shape_, strides_, type_, offsetElements_, stream_);
}

const Shape& CudaTensor::shape() const {
  return shape_;
}

dtype CudaTensor::type() const {
  return type_;
}

bool CudaTensor::isSparse() const {
  return false;
}

Location CudaTensor::location() const {
  return Location::Device;
}

void CudaTensor::scalarSync(void* out) {
  if (shape_.elements() == 0) {
    throw std::invalid_argument("CudaTensor::scalar: empty tensor");
  }
  size_t elemSize = dtypeSize(type_);
  FL_CUDA_CHECK(cudaMemcpyAsync(
      out, devicePtr(), elemSize, cudaMemcpyDeviceToHost, stream_->handle()));
  FL_CUDA_CHECK(cudaStreamSynchronize(stream_->handle()));
}

void CudaTensor::device(void** out) {
  *out = devicePtr();
}

void CudaTensor::hostSync(void* out) {
  size_t bytes = totalBytes();
  if (bytes > 0) {
    if (!isContiguousRowMajor()) {
      auto contig = asContiguousTensor();
      auto& ca = contig.getAdapter<CudaTensor>();
      FL_CUDA_CHECK(cudaMemcpyAsync(
          out, ca.devicePtr(), bytes, cudaMemcpyDeviceToHost, stream_->handle()));
    } else {
      FL_CUDA_CHECK(cudaMemcpyAsync(
          out, devicePtr(), bytes, cudaMemcpyDeviceToHost, stream_->handle()));
    }
    FL_CUDA_CHECK(cudaStreamSynchronize(stream_->handle()));
  }
}

void CudaTensor::unlock() {
  // No-op: buffer lifetime managed by CudaBuffer shared_ptr.
}

bool CudaTensor::isLocked() {
  return false;
}

bool CudaTensor::isContiguous() const {
  return isContiguousRowMajor();
}

Shape CudaTensor::strides() const {
  return strides_;
}

const Stream& CudaTensor::stream() const {
  return *stream_;
}

// GPU-native type casting kernel.
template <typename Dst, typename Src>
__global__ void castKernel(
    const Src* __restrict__ input,
    Dst* __restrict__ output,
    size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (static_cast<size_t>(idx) >= n) {
    return;
  }
  output[idx] = static_cast<Dst>(input[idx]);
}

// Specialization for bool output -- nonzero check.
template <typename Src>
__global__ void castToBoolKernel(
    const Src* __restrict__ input,
    char* __restrict__ output,
    size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (static_cast<size_t>(idx) >= n) {
    return;
  }
  output[idx] = input[idx] != static_cast<Src>(0) ? 1 : 0;
}

// Dispatch cast kernel for a given source type --> destination dtype.
template <typename Src>
void launchCastFromSrc(
    const void* srcPtr, void* dstPtr, size_t n,
    fl::dtype dstType, cudaStream_t stream) {
  constexpr int kBlock = 256;
  int grid = (static_cast<int>(n) + kBlock - 1) / kBlock;
  auto* src = static_cast<const Src*>(srcPtr);

  switch (dstType) {
    case fl::dtype::f32:
      castKernel<float, Src><<<grid, kBlock, 0, stream>>>(
          src, static_cast<float*>(dstPtr), n); break;
    case fl::dtype::f64:
      castKernel<double, Src><<<grid, kBlock, 0, stream>>>(
          src, static_cast<double*>(dstPtr), n); break;
    case fl::dtype::s32:
      castKernel<int, Src><<<grid, kBlock, 0, stream>>>(
          src, static_cast<int*>(dstPtr), n); break;
    case fl::dtype::s64:
      castKernel<long long, Src><<<grid, kBlock, 0, stream>>>(
          src, static_cast<long long*>(dstPtr), n); break;
    case fl::dtype::u32:
      castKernel<unsigned, Src><<<grid, kBlock, 0, stream>>>(
          src, static_cast<unsigned*>(dstPtr), n); break;
    case fl::dtype::u64:
      castKernel<unsigned long long, Src><<<grid, kBlock, 0, stream>>>(
          src, static_cast<unsigned long long*>(dstPtr), n); break;
    case fl::dtype::b8:
      castToBoolKernel<Src><<<grid, kBlock, 0, stream>>>(
          src, static_cast<char*>(dstPtr), n); break;
    case fl::dtype::u8:
      castKernel<unsigned char, Src><<<grid, kBlock, 0, stream>>>(
          src, static_cast<unsigned char*>(dstPtr), n); break;
    case fl::dtype::s16:
      castKernel<short, Src><<<grid, kBlock, 0, stream>>>(
          src, static_cast<short*>(dstPtr), n); break;
    case fl::dtype::u16:
      castKernel<unsigned short, Src><<<grid, kBlock, 0, stream>>>(
          src, static_cast<unsigned short*>(dstPtr), n); break;
    case fl::dtype::f16:
      castKernel<__half, Src><<<grid, kBlock, 0, stream>>>(
          src, static_cast<__half*>(dstPtr), n); break;
    case fl::dtype::bf16:
      castKernel<__nv_bfloat16, Src><<<grid, kBlock, 0, stream>>>(
          src, static_cast<__nv_bfloat16*>(dstPtr), n); break;
  }
}

// Dispatch cast for (srcType, dstType) pair.
void launchCastKernel(
    const void* srcPtr, void* dstPtr, size_t n,
    fl::dtype srcType, fl::dtype dstType, cudaStream_t stream) {
  switch (srcType) {
    case fl::dtype::f32:
      launchCastFromSrc<float>(srcPtr, dstPtr, n, dstType, stream); break;
    case fl::dtype::f64:
      launchCastFromSrc<double>(srcPtr, dstPtr, n, dstType, stream); break;
    case fl::dtype::s32:
      launchCastFromSrc<int>(srcPtr, dstPtr, n, dstType, stream); break;
    case fl::dtype::s64:
      launchCastFromSrc<long long>(srcPtr, dstPtr, n, dstType, stream); break;
    case fl::dtype::u32:
      launchCastFromSrc<unsigned>(srcPtr, dstPtr, n, dstType, stream); break;
    case fl::dtype::u64:
      launchCastFromSrc<unsigned long long>(srcPtr, dstPtr, n, dstType, stream); break;
    case fl::dtype::b8:
      launchCastFromSrc<char>(srcPtr, dstPtr, n, dstType, stream); break;
    case fl::dtype::u8:
      launchCastFromSrc<unsigned char>(srcPtr, dstPtr, n, dstType, stream); break;
    case fl::dtype::s16:
      launchCastFromSrc<short>(srcPtr, dstPtr, n, dstType, stream); break;
    case fl::dtype::u16:
      launchCastFromSrc<unsigned short>(srcPtr, dstPtr, n, dstType, stream); break;
    case fl::dtype::f16:
      launchCastFromSrc<__half>(srcPtr, dstPtr, n, dstType, stream); break;
    case fl::dtype::bf16:
      launchCastFromSrc<__nv_bfloat16>(srcPtr, dstPtr, n, dstType, stream); break;
  }
}

Tensor CudaTensor::astype(const dtype newType) {
  if (newType == type_) {
    return shallowCopy();
  }

  // Strided views must be compacted before element-wise cast
  if (!isContiguousRowMajor()) {
    return asContiguousTensor().astype(newType);
  }

  size_t n = shape_.elements();
  if (n == 0) {
    return toTensor<CudaTensor>(shape_, newType, nullptr, MemoryLocation::Host);
  }

  auto result = toTensor<CudaTensor>(
      shape_, newType, nullptr, MemoryLocation::Host);

  const void* srcPtr = devicePtr();
  void* dstPtr = result.getAdapter<CudaTensor>().devicePtr();
  cudaStream_t stream = stream_->handle();

  launchCastKernel(srcPtr, dstPtr, n, type_, newType, stream);
  FL_CUDA_CHECK(cudaGetLastError());

  return result;
}

Tensor CudaTensor::index(const std::vector<Index>& indices) {
  // Start with current view parameters
  auto newShape = std::vector<Dim>();
  auto newStrides = std::vector<Dim>();
  size_t newOffset = offsetElements_;

  unsigned ndim = shape_.ndim();
  for (unsigned i = 0; i < ndim; ++i) {
    Dim dimSize = shape_[i];
    Dim dimStride = strides_[i];

    if (i >= indices.size()) {
      // Unspecified dims are implicitly span
      newShape.push_back(dimSize);
      newStrides.push_back(dimStride);
      continue;
    }

    const auto& idx = indices[i];
    switch (idx.type()) {
      case detail::IndexType::Span:
        newShape.push_back(dimSize);
        newStrides.push_back(dimStride);
        break;

      case detail::IndexType::Literal: {
        Dim lit = idx.get<Dim>();
        if (lit < 0) lit += dimSize;
        if (lit < 0 || lit >= dimSize) {
          throw std::out_of_range("CudaTensor::index: literal index out of range");
        }
        // Collapse this dimension: adjust offset, don't add to output shape
        newOffset += lit * dimStride;
        break;
      }

      case detail::IndexType::Range: {
        const auto& r = idx.get<range>();
        Dim start = r.start();
        if (start < 0) start += dimSize;
        Dim end;
        if (r.end().has_value()) {
          end = r.endVal();
          if (end < 0) end += dimSize;
        } else {
          end = dimSize;
        }
        Dim stride = r.stride();
        if (stride <= 0) {
          throw std::invalid_argument("CudaTensor::index: non-positive stride");
        }
        Dim rangeSize = (end - start + stride - 1) / stride;
        if (rangeSize < 0) rangeSize = 0;
        newOffset += start * dimStride;
        newShape.push_back(rangeSize);
        newStrides.push_back(dimStride * stride);
        break;
      }

      case detail::IndexType::Tensor: {
        const auto& idxTensor = idx.get<Tensor>();
        // Use the backend's stream for consistency with nonzero/astype ops
        auto cudaStream = CudaBackend::getInstance().cudaStream();

        // Convert b8 mask to integer indices, or cast to s32
        Tensor intIndices;
        if (idxTensor.type() == dtype::b8) {
          intIndices = fl::nonzero(idxTensor);
        } else {
          intIndices = idxTensor.astype(dtype::s32);
        }

        Dim numIdx = intIndices.elements();
        if (numIdx == 0) {
          return toTensor<CudaTensor>(
              Shape({0}), type_, nullptr, MemoryLocation::Host);
        }

        // For same-shape tensor index (flat indexing), gather from flat view
        // For dim-0 indexing, gather along first axis
        bool isFlatIndex = (shape_ == idxTensor.shape());
        Dim innerSize = 1;
        Shape outShapeTensor;
        if (isFlatIndex) {
          // Flat indexing: result is 1D
          outShapeTensor = Shape({numIdx});
        } else {
          // Dim-0 indexing: result shape = [numIdx, dim1, dim2, ...]
          std::vector<Dim> outDims;
          outDims.push_back(numIdx);
          for (int d = 1; d < static_cast<int>(shape_.ndim()); d++) {
            outDims.push_back(shape_[d]);
            innerSize *= shape_[d];
          }
          outShapeTensor = Shape(outDims);
        }

        // Ensure source is contiguous for gather
        Tensor srcContig;
        const CudaTensor* srcAdapter;
        if (isContiguousRowMajor()) {
          srcAdapter = this;
        } else {
          srcContig = asContiguousTensor();
          srcAdapter = &srcContig.getAdapter<CudaTensor>();
        }

        // Allocate output
        auto result = toTensor<CudaTensor>(
            outShapeTensor, type_, nullptr, MemoryLocation::Host);
        auto& resultAdapter = result.getAdapter<CudaTensor>();

        size_t elemSz = dtypeSize(type_);

        if (innerSize == 1) {
          // Simple flat gather: out[i] = src[indices[i]]
          int blockSize = 256;
          int gridSize = (numIdx + blockSize - 1) / blockSize;
          auto& idxAdapter = intIndices.getAdapter<CudaTensor>();
          gatherKernel<<<gridSize, blockSize, 0, cudaStream>>>(
              static_cast<char*>(resultAdapter.devicePtr()),
              static_cast<const char*>(srcAdapter->devicePtr()),
              static_cast<const int*>(idxAdapter.devicePtr()),
              elemSz, numIdx);
          FL_CUDA_CHECK(cudaGetLastError());
        } else {
          // Dim-0 gather: for each index, copy innerSize elements
          // Build expanded index map: for idx i → [idx[i]*innerSize + 0, ..., idx[i]*innerSize + innerSize-1]
          Dim totalOut = numIdx * innerSize;
          auto expandedIdxBuf = std::make_shared<CudaBuffer>(
              totalOut * sizeof(int),
              buffer_ ? buffer_->deviceId : fl::cuda::getActiveDeviceId(),
              cudaStream);
          auto* expandedPtr = static_cast<int*>(expandedIdxBuf->data);
          auto& idxAdapter = intIndices.getAdapter<CudaTensor>();
          const int* rawIdx = static_cast<const int*>(idxAdapter.devicePtr());
          Dim innerSz = innerSize;

          // Use thrust::tabulate to build the expanded map
          auto policy = thrust::cuda::par.on(cudaStream);
          thrust::tabulate(
              policy,
              thrust::device_pointer_cast(expandedPtr),
              thrust::device_pointer_cast(expandedPtr + totalOut),
              [rawIdx, innerSz] __device__ (int i) {
                int outerIdx = i / innerSz;
                int innerOff = i % innerSz;
                return rawIdx[outerIdx] * static_cast<int>(innerSz) + innerOff;
              });

          int blockSize = 256;
          int gridSize = (totalOut + blockSize - 1) / blockSize;
          gatherKernel<<<gridSize, blockSize, 0, cudaStream>>>(
              static_cast<char*>(resultAdapter.devicePtr()),
              static_cast<const char*>(srcAdapter->devicePtr()),
              expandedPtr,
              elemSz, totalOut);
          FL_CUDA_CHECK(cudaGetLastError());

          // Store expanded index map for scatter-back
          resultAdapter.parentIndexState_ = ParentIndexState{
              buffer_, type_,
              std::move(expandedIdxBuf),
              static_cast<size_t>(totalOut),
              offsetElements_};
          return result;
        }

        // Store index map for scatter-back (flat case)
        // Copy index buffer reference for write-back
        auto& idxAdapter = intIndices.getAdapter<CudaTensor>();
        auto idxMapBuf = std::make_shared<CudaBuffer>(
            numIdx * sizeof(int),
            buffer_ ? buffer_->deviceId : fl::cuda::getActiveDeviceId(),
            cudaStream);
        FL_CUDA_CHECK(cudaMemcpyAsync(
            idxMapBuf->data, idxAdapter.devicePtr(),
            numIdx * sizeof(int),
            cudaMemcpyDeviceToDevice, cudaStream));

        resultAdapter.parentIndexState_ = ParentIndexState{
            buffer_, type_,
            std::move(idxMapBuf),
            static_cast<size_t>(numIdx),
            offsetElements_};
        return result;
      }
    }
  }

  Shape outShape(newShape);
  Shape outStrides(newStrides);

  return toTensor<CudaTensor>(
      buffer_, outShape, outStrides, type_, newOffset, stream_);
}

Tensor CudaTensor::flatten() const {
  // Returns a COPY as a 1D tensor. This ensures that callers who capture
  // the result (e.g. `Tensor orig = t.flatten()(i)`) get an independent
  // snapshot, not a view that aliases `t`. Without COW, view-based flatten
  // causes aliasing bugs in patterns like the Jacobian test.
  auto contig = const_cast<CudaTensor*>(this)->asContiguousTensor();
  auto& ca = contig.getAdapter<CudaTensor>();
  Dim nElems = shape_.elements();
  Shape flatShape({nElems});

  auto result = toTensor<CudaTensor>(
      flatShape, type_, nullptr, MemoryLocation::Host);
  if (nElems > 0) {
    auto& ra = result.getAdapter<CudaTensor>();
    size_t bytes = nElems * dtypeSize(type_);
    FL_CUDA_CHECK(cudaMemcpyAsync(
        ra.devicePtr(), ca.devicePtr(), bytes,
        cudaMemcpyDeviceToDevice,
        CudaBackend::getInstance().cudaStream()));
  }
  return result;
}

Tensor CudaTensor::flat(const Index& idx) const {
  // Create a VIEW directly into this tensor's buffer for write-through.
  // This does NOT go through flatten() to avoid the copy.
  // Requires the tensor to be contiguous; non-contiguous tensors are
  // materialized first.
  Dim totalElems = shape_.elements();

  // Get a contiguous representation sharing the original buffer if possible
  std::shared_ptr<CudaBuffer> buf = buffer_;
  size_t baseOffset = offsetElements_;
  auto stream = stream_;
  if (!isContiguousRowMajor()) {
    auto contig = const_cast<CudaTensor*>(this)->asContiguousTensor();
    auto& ca = contig.getAdapter<CudaTensor>();
    buf = ca.buffer_;
    baseOffset = ca.offsetElements_;
    stream = ca.stream_;
  }

  switch (idx.type()) {
    case detail::IndexType::Literal: {
      Dim lit = idx.get<Dim>();
      if (lit < 0) lit += totalElems;
      if (lit < 0 || lit >= totalElems) {
        throw std::out_of_range("CudaTensor::flat: index out of range");
      }
      // Scalar view: shape={}, strides={}, offset adjusted
      return toTensor<CudaTensor>(
          buf, Shape{}, Shape{}, type_,
          baseOffset + lit, stream);
    }

    case detail::IndexType::Span: {
      Shape flatShape({totalElems});
      return toTensor<CudaTensor>(
          buf, flatShape, computeStrides(flatShape), type_,
          baseOffset, stream);
    }

    case detail::IndexType::Range: {
      const auto& r = idx.get<range>();
      Dim start = r.start();
      if (start < 0) start += totalElems;
      Dim end;
      if (r.end().has_value()) {
        end = r.endVal();
        if (end < 0) end += totalElems;
      } else {
        end = totalElems;
      }
      Dim stride = r.stride();
      if (stride <= 0) {
        throw std::invalid_argument("CudaTensor::flat: non-positive stride");
      }
      Dim rangeSize = (end - start + stride - 1) / stride;
      if (rangeSize < 0) rangeSize = 0;
      return toTensor<CudaTensor>(
          buf, Shape{rangeSize}, Shape{stride}, type_,
          baseOffset + start, stream);
    }

    case detail::IndexType::Tensor: {
      // Create a 1D view sharing the buffer, then use index() for gather
      // with scatter-back support (parentIndexState_).
      Shape flatShape({totalElems});
      auto flatView = toTensor<CudaTensor>(
          buf, flatShape, computeStrides(flatShape), type_,
          baseOffset, stream);
      return flatView({idx});
    }
  }
  throw std::runtime_error("CudaTensor::flat: unknown index type");
}

Tensor CudaTensor::asContiguousTensor() {
  if (isContiguousRowMajor()) {
    return shallowCopy();
  }
  // Materialize strided data into a new contiguous buffer using GPU kernel
  return copy();
}

void CudaTensor::setContext(void* /*context*/) {}
void* CudaTensor::getContext() { return nullptr; }

std::string CudaTensor::toString() {
  std::ostringstream ss;
  ss << "CudaTensor [shape=" << shape_ << ", dtype=" << static_cast<int>(type_) << "]";
  return ss.str();
}

std::ostream& CudaTensor::operator<<(std::ostream& ostr) {
  ostr << toString();
  return ostr;
}

// --- Assignment ops ---

// Copy data from src into this tensor's buffer (write-through for views).
// Handles contiguous-to-contiguous (fast memcpy) and contiguous-to-strided
// (kernel-based scatter) cases.
void CudaTensor::assign(const Tensor& val) {
  // If source dtype differs from destination, cast first
  if (val.type() != type_) {
    assign(val.astype(type_));
    return;
  }

  auto& src = val.getAdapter<CudaTensor>();
  size_t nElems = shape_.elements();
  size_t srcElems = src.shape_.elements();
  if (nElems == 0 && srcElems == 0) return;

  if (nElems != srcElems) {
    throw std::invalid_argument(
        "CudaTensor::assign: element count mismatch (" +
        std::to_string(nElems) + " vs " + std::to_string(srcElems) + ")");
  }

  auto stream = CudaBackend::getInstance().cudaStream();

  // Fast path: both contiguous --> single memcpy
  if (isContiguousRowMajor() && src.isContiguousRowMajor()) {
    size_t bytes = nElems * dtypeSize(type_);
    FL_CUDA_CHECK(cudaMemcpyAsync(
        devicePtr(), src.devicePtr(), bytes,
        cudaMemcpyDeviceToDevice, stream));
    scatterToParent();
    return;
  }

  // Ensure source is contiguous
  Tensor srcContig = src.isContiguousRowMajor()
      ? val : val.asContiguousTensor();
  auto& sc = srcContig.getAdapter<CudaTensor>();

  if (isContiguousRowMajor()) {
    // Contiguous destination, just memcpy from contiguous source
    size_t bytes = nElems * dtypeSize(type_);
    FL_CUDA_CHECK(cudaMemcpyAsync(
        devicePtr(), sc.devicePtr(), bytes,
        cudaMemcpyDeviceToDevice, stream));
    scatterToParent();
    return;
  }

  // Non-contiguous destination: use strided copy kernel
  int ndim = shape_.ndim();
  if (ndim == 0) {
    // Scalar -- always contiguous, should have been caught above
    size_t bytes = dtypeSize(type_);
    FL_CUDA_CHECK(cudaMemcpyAsync(
        devicePtr(), sc.devicePtr(), bytes,
        cudaMemcpyDeviceToDevice, stream));
    scatterToParent();
    return;
  }
  if (ndim > kMaxStridedDims) {
    throw std::runtime_error(
        "CudaTensor::assign: ndim exceeds maximum for strided copy");
  }

  StridedCopyInfo info{};
  info.ndim = ndim;
  auto rowMajorStrides = computeStrides(shape_);
  for (int d = 0; d < ndim; ++d) {
    info.shape[d] = shape_[d];
    info.dstStrides[d] = strides_[d];
    info.srcStrides[d] = rowMajorStrides[d];
  }

  size_t elemSz = dtypeSize(type_);
  int blockSize = 256;
  int gridSize = (nElems + blockSize - 1) / blockSize;
  stridedCopyKernel<<<gridSize, blockSize, 0, stream>>>(
      static_cast<char*>(devicePtr()),
      static_cast<const char*>(sc.devicePtr()),
      info, elemSz, nElems);
  FL_CUDA_CHECK(cudaGetLastError());
  scatterToParent();
}

// Scalar assignment: create a full tensor and delegate to assign(Tensor).
// This handles both contiguous and non-contiguous destinations.
#define FL_CUDA_ASSIGN_SCALAR_DEF(TYPE)                                  \
  void CudaTensor::assign(const TYPE& val) {                             \
    auto filled = CudaBackend::getInstance().full(shape_, val, type_);   \
    assign(filled);                                                      \
  }

FL_CUDA_ASSIGN_SCALAR_DEF(double)
FL_CUDA_ASSIGN_SCALAR_DEF(float)
FL_CUDA_ASSIGN_SCALAR_DEF(int)
FL_CUDA_ASSIGN_SCALAR_DEF(unsigned)
FL_CUDA_ASSIGN_SCALAR_DEF(bool)
FL_CUDA_ASSIGN_SCALAR_DEF(char)
FL_CUDA_ASSIGN_SCALAR_DEF(unsigned char)
FL_CUDA_ASSIGN_SCALAR_DEF(short)
FL_CUDA_ASSIGN_SCALAR_DEF(unsigned short)
FL_CUDA_ASSIGN_SCALAR_DEF(long)
FL_CUDA_ASSIGN_SCALAR_DEF(unsigned long)
FL_CUDA_ASSIGN_SCALAR_DEF(long long)
FL_CUDA_ASSIGN_SCALAR_DEF(unsigned long long)
#undef FL_CUDA_ASSIGN_SCALAR_DEF

// In-place ops: compute result, then copy back into this tensor's buffer
#define FL_CUDA_INPLACE_OP_DEF(OP, BACKEND_OP)                         \
  void CudaTensor::OP(const Tensor& val) {                             \
    auto self = toTensor<CudaTensor>(                                   \
        buffer_, shape_, strides_, type_, offsetElements_, stream_);    \
    Tensor result;                                                      \
    cuda::BACKEND_OP(                                                   \
        CudaBackend::getInstance().cudaStream(), result, self, val);    \
    assign(result);                                                     \
  }                                                                     \
  void CudaTensor::OP(const double& val) {                             \
    OP(CudaBackend::getInstance().full(shape_, val, type_));            \
  }                                                                     \
  void CudaTensor::OP(const float& val) {                              \
    OP(CudaBackend::getInstance().full(shape_, val, type_));            \
  }                                                                     \
  void CudaTensor::OP(const int& val) {                                \
    OP(CudaBackend::getInstance().full(shape_, val, type_));            \
  }                                                                     \
  void CudaTensor::OP(const unsigned& val) {                           \
    OP(CudaBackend::getInstance().full(shape_, val, type_));            \
  }                                                                     \
  void CudaTensor::OP(const bool& val) {                               \
    OP(CudaBackend::getInstance().full(shape_, val, type_));            \
  }                                                                     \
  void CudaTensor::OP(const char& val) {                               \
    OP(CudaBackend::getInstance().full(shape_, val, type_));            \
  }                                                                     \
  void CudaTensor::OP(const unsigned char& val) {                      \
    OP(CudaBackend::getInstance().full(shape_, val, type_));            \
  }                                                                     \
  void CudaTensor::OP(const short& val) {                              \
    OP(CudaBackend::getInstance().full(shape_, val, type_));            \
  }                                                                     \
  void CudaTensor::OP(const unsigned short& val) {                     \
    OP(CudaBackend::getInstance().full(shape_, val, type_));            \
  }                                                                     \
  void CudaTensor::OP(const long& val) {                               \
    OP(CudaBackend::getInstance().full(shape_, val, type_));            \
  }                                                                     \
  void CudaTensor::OP(const unsigned long& val) {                      \
    OP(CudaBackend::getInstance().full(shape_, val, type_));            \
  }                                                                     \
  void CudaTensor::OP(const long long& val) {                          \
    OP(CudaBackend::getInstance().full(shape_, val, type_));            \
  }                                                                     \
  void CudaTensor::OP(const unsigned long long& val) {                 \
    OP(CudaBackend::getInstance().full(shape_, val, type_));            \
  }

FL_CUDA_INPLACE_OP_DEF(inPlaceAdd, add)
FL_CUDA_INPLACE_OP_DEF(inPlaceSubtract, sub)
FL_CUDA_INPLACE_OP_DEF(inPlaceMultiply, mul)
FL_CUDA_INPLACE_OP_DEF(inPlaceDivide, div)
#undef FL_CUDA_INPLACE_OP_DEF

} // namespace fl
