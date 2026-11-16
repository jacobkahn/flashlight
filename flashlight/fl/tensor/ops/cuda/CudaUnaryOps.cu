/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include "flashlight/fl/tensor/backend/cuda/CudaTensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <cub/device/device_segmented_sort.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include "flashlight/fl/runtime/CUDAUtils.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/ops/cuda/CudaOps.h"

namespace fl::cuda {

namespace {

// Helper: create output tensor and run a unary Thrust transform in f32.
// When out is non-empty, writes into its buffer instead of allocating.
template <typename UnaryOp>
void unaryTransformF32(
    cudaStream_t stream,
    Tensor& out,
    const Tensor& input,
    fl::dtype outType,
    UnaryOp op) {
  auto asF32 = (input.type() == fl::dtype::f32)
      ? input : input.astype(fl::dtype::f32);
  auto& in = asF32.getAdapter<CudaTensor>();
  size_t n = input.elements();
  auto contigIn = in.isContiguousRowMajor()
      ? asF32
      : asF32.asContiguousTensor();
  auto& ci = contigIn.getAdapter<CudaTensor>();

  // Only reuse out when output type matches compute type (f32),
  // otherwise the buffer has the wrong element type for f32 writes.
  Tensor result = (!out.isEmpty() && outType == fl::dtype::f32)
      ? out
      : toTensor<CudaTensor>(input.shape(), fl::dtype::f32, nullptr, MemoryLocation::Host);
  auto& outAdapter = result.getAdapter<CudaTensor>();

  auto inPtr = thrust::device_pointer_cast(
      static_cast<const float*>(ci.devicePtr()));
  auto outPtr = thrust::device_pointer_cast(
      static_cast<float*>(outAdapter.devicePtr()));

  thrust::transform(
      thrust::cuda::par.on(stream), inPtr, inPtr + n, outPtr, op);
  out = (outType != fl::dtype::f32) ? result.astype(outType) : result;
}

// Helper: unary Thrust transform in f64 (native double precision).
// When out is non-empty, writes into its buffer instead of allocating.
template <typename UnaryOp>
void unaryTransformF64(
    cudaStream_t stream,
    Tensor& out,
    const Tensor& input,
    fl::dtype outType,
    UnaryOp op) {
  auto asF64 = (input.type() == fl::dtype::f64)
      ? input : input.astype(fl::dtype::f64);
  auto& in = asF64.getAdapter<CudaTensor>();
  size_t n = input.elements();
  auto contigIn = in.isContiguousRowMajor()
      ? asF64
      : asF64.asContiguousTensor();
  auto& ci = contigIn.getAdapter<CudaTensor>();

  Tensor result = (!out.isEmpty() && outType == fl::dtype::f64)
      ? out
      : toTensor<CudaTensor>(input.shape(), fl::dtype::f64, nullptr, MemoryLocation::Host);
  auto& outAdapter = result.getAdapter<CudaTensor>();

  auto inPtr = thrust::device_pointer_cast(
      static_cast<const double*>(ci.devicePtr()));
  auto outPtr = thrust::device_pointer_cast(
      static_cast<double*>(outAdapter.devicePtr()));

  thrust::transform(
      thrust::cuda::par.on(stream), inPtr, inPtr + n, outPtr, op);
  out = (outType != fl::dtype::f64) ? result.astype(outType) : result;
}

// Gather kernel: copy strided slices into contiguous segments.
// Input layout: data[outer * axisSize * innerSize + j * innerSize + inner]
// Output layout: gathered[seg * axisSize + j] where seg = outer * innerSize + inner
__global__ void gatherForSortKernel(
    const float* __restrict__ input,
    float* __restrict__ gathered,
    int numSegments, int axisSize, int innerSize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = numSegments * axisSize;
  if (idx >= total) {
    return;
  }

  int seg = idx / axisSize;
  int j = idx % axisSize;
  int outer = seg / innerSize;
  int inner = seg % innerSize;
  gathered[idx] = input[outer * (axisSize * innerSize) + j * innerSize + inner];
}

// Scatter kernel: copy sorted contiguous segments back to strided layout.
__global__ void scatterFromSortKernel(
    const float* __restrict__ gathered,
    const int* __restrict__ gatheredIdx,
    float* __restrict__ outValues,
    int* __restrict__ outIndices,
    int numSegments, int axisSize, int innerSize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = numSegments * axisSize;
  if (idx >= total) {
    return;
  }

  int seg = idx / axisSize;
  int j = idx % axisSize;
  int outer = seg / innerSize;
  int inner = seg % innerSize;
  int dstIdx = outer * (axisSize * innerSize) + j * innerSize + inner;
  outValues[dstIdx] = gathered[idx];
  outIndices[dstIdx] = gatheredIdx[idx];
}

constexpr int kSortBlockSize = 256;

// Scaled counting iterator for CUB segment offsets: iter[i] = (start+i)*scale
struct SortScaleOp {
  int scale;
  __host__ __device__ __forceinline__ int operator()(int i) const {
    return i * scale;
  }
};

using SortScaledIter = cub::TransformInputIterator<
    int, SortScaleOp, cub::CountingInputIterator<int>>;

SortScaledIter makeSortScaledIter(int start, int scale) {
  return SortScaledIter(
      cub::CountingInputIterator<int>(start), SortScaleOp{scale});
}

// Run a CUB API call with the standard two-pass pattern.
template <typename CubCallFn>
void cubSortTwoPass(CubCallFn&& fn, cudaStream_t stream) {
  size_t tempBytes = 0;
  fn(nullptr, tempBytes);
  void* d_temp = nullptr;
  if (tempBytes > 0) {
    FL_CUDA_CHECK(cudaMallocAsync(&d_temp, tempBytes, stream));
  }
  fn(d_temp, tempBytes);
  if (d_temp) {
    FL_CUDA_CHECK(cudaFreeAsync(d_temp, stream));
  }
}

// GPU-native sort along an axis using CUB DeviceSegmentedSort.
// All segments are sorted in a single kernel launch instead of per-segment loops.
void gpuSortAlongAxis(
    const float* d_input,
    float* d_outValues,
    int* d_outIndices,
    const Shape& shape,
    Dim axis,
    SortMode sortMode,
    cudaStream_t stream) {
  int axisSize = static_cast<int>(shape[axis]);
  int outerSize = 1, innerSize = 1;
  for (int d = 0; d < static_cast<int>(axis); ++d) {
    outerSize *= static_cast<int>(shape[d]);
  }
  for (int d = static_cast<int>(axis) + 1; d < shape.ndim(); ++d) {
    innerSize *= static_cast<int>(shape[d]);
  }
  int numSegments = outerSize * innerSize;
  int n = numSegments * axisSize;

  auto policy = thrust::cuda::par.on(stream);
  bool ascending = (sortMode == SortMode::Ascending);

  // CUB segment offset iterators
  auto beginOffsets = makeSortScaledIter(0, axisSize);
  auto endOffsets = makeSortScaledIter(1, axisSize);

  if (innerSize == 1) {
    // Contiguous segments -- sort directly with CUB, no gather/scatter needed.
    // Initialize per-segment indices: [0, 1, ..., axisSize-1, 0, 1, ...]
    int* d_indicesIn = nullptr;
    FL_CUDA_CHECK(cudaMallocAsync(&d_indicesIn, n * sizeof(int), stream));
    auto idxPtr = thrust::device_pointer_cast(d_indicesIn);
    thrust::tabulate(policy, idxPtr, idxPtr + n,
        [axisSize] __device__ (int i) { return i % axisSize; });

    if (ascending) {
      cubSortTwoPass([&](void* temp, size_t& bytes) {
        cub::DeviceSegmentedSort::SortPairs(
            temp, bytes,
            d_input, d_outValues,
            d_indicesIn, d_outIndices,
            n, numSegments,
            beginOffsets, endOffsets, stream);
      }, stream);
    } else {
      cubSortTwoPass([&](void* temp, size_t& bytes) {
        cub::DeviceSegmentedSort::SortPairsDescending(
            temp, bytes,
            d_input, d_outValues,
            d_indicesIn, d_outIndices,
            n, numSegments,
            beginOffsets, endOffsets, stream);
      }, stream);
    }

    FL_CUDA_CHECK(cudaFreeAsync(d_indicesIn, stream));
  } else {
    // Strided data -- gather into contiguous segments, CUB sort, scatter back.
    float* d_gathered = nullptr;
    int* d_indicesIn = nullptr;
    float* d_sortedKeys = nullptr;
    int* d_sortedIndices = nullptr;
    FL_CUDA_CHECK(cudaMallocAsync(&d_gathered, n * sizeof(float), stream));
    FL_CUDA_CHECK(cudaMallocAsync(&d_indicesIn, n * sizeof(int), stream));
    FL_CUDA_CHECK(cudaMallocAsync(&d_sortedKeys, n * sizeof(float), stream));
    FL_CUDA_CHECK(cudaMallocAsync(&d_sortedIndices, n * sizeof(int), stream));

    // Step 1: Gather strided data into contiguous segments
    int grid = (n + kSortBlockSize - 1) / kSortBlockSize;
    gatherForSortKernel<<<grid, kSortBlockSize, 0, stream>>>(
        d_input, d_gathered, numSegments, axisSize, innerSize);
    FL_CUDA_CHECK(cudaGetLastError());

    // Step 2: Initialize per-segment indices
    auto idxPtr = thrust::device_pointer_cast(d_indicesIn);
    thrust::tabulate(policy, idxPtr, idxPtr + n,
        [axisSize] __device__ (int i) { return i % axisSize; });

    // Step 3: CUB segmented sort -- single launch for all segments
    if (ascending) {
      cubSortTwoPass([&](void* temp, size_t& bytes) {
        cub::DeviceSegmentedSort::SortPairs(
            temp, bytes,
            d_gathered, d_sortedKeys,
            d_indicesIn, d_sortedIndices,
            n, numSegments,
            beginOffsets, endOffsets, stream);
      }, stream);
    } else {
      cubSortTwoPass([&](void* temp, size_t& bytes) {
        cub::DeviceSegmentedSort::SortPairsDescending(
            temp, bytes,
            d_gathered, d_sortedKeys,
            d_indicesIn, d_sortedIndices,
            n, numSegments,
            beginOffsets, endOffsets, stream);
      }, stream);
    }

    // Step 4: Scatter back to strided layout
    scatterFromSortKernel<<<grid, kSortBlockSize, 0, stream>>>(
        d_sortedKeys, d_sortedIndices, d_outValues, d_outIndices,
        numSegments, axisSize, innerSize);
    FL_CUDA_CHECK(cudaGetLastError());

    FL_CUDA_CHECK(cudaFreeAsync(d_gathered, stream));
    FL_CUDA_CHECK(cudaFreeAsync(d_indicesIn, stream));
    FL_CUDA_CHECK(cudaFreeAsync(d_sortedKeys, stream));
    FL_CUDA_CHECK(cudaFreeAsync(d_sortedIndices, stream));
  }
}

// Topk extraction kernel: copy first k elements per segment from sorted output.
__global__ void topkGatherKernel(
    const float* __restrict__ sortedValues,
    const int* __restrict__ sortedIndices,
    float* __restrict__ topkValues,
    int* __restrict__ topkIndices,
    int outerSize, int k, int fullAxisSize, int innerSize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outerSize * k * innerSize;
  if (idx >= total) {
    return;
  }

  // Decompose output index
  int inner = idx % innerSize;
  int rem = idx / innerSize;
  int j = rem % k;
  int outer = rem / k;

  int srcIdx = outer * (fullAxisSize * innerSize) + j * innerSize + inner;
  topkValues[idx] = sortedValues[srcIdx];
  topkIndices[idx] = sortedIndices[srcIdx];
}

} // anonymous namespace

// Macro for unary ops with f32 and f64 paths.
// F32_EXPR uses float x, F64_EXPR uses double x.
#define FL_CUDA_UNARY_OP(NAME, F32_EXPR, F64_EXPR)                          \
  void NAME(cudaStream_t stream, Tensor& out, const Tensor& tensor) {       \
    if (tensor.type() == fl::dtype::f64) {                                   \
      unaryTransformF64(stream, out, tensor, tensor.type(),                  \
          [] __device__ (double x) -> double { return F64_EXPR; });          \
    } else {                                                                 \
      unaryTransformF32(stream, out, tensor, tensor.type(),                  \
          [] __device__ (float x) -> float { return F32_EXPR; });            \
    }                                                                        \
  }

FL_CUDA_UNARY_OP(exp, ::expf(x), ::exp(x))
FL_CUDA_UNARY_OP(log, ::logf(x), ::log(x))
FL_CUDA_UNARY_OP(negative, -x, -x)
FL_CUDA_UNARY_OP(log1p, ::log1pf(x), ::log1p(x))
FL_CUDA_UNARY_OP(sin, ::sinf(x), ::sin(x))
FL_CUDA_UNARY_OP(cos, ::cosf(x), ::cos(x))
FL_CUDA_UNARY_OP(sqrt, ::sqrtf(x), ::sqrt(x))
FL_CUDA_UNARY_OP(tanh, ::tanhf(x), ::tanh(x))
FL_CUDA_UNARY_OP(floor, ::floorf(x), ::floor(x))
FL_CUDA_UNARY_OP(ceil, ::ceilf(x), ::ceil(x))
FL_CUDA_UNARY_OP(rint, ::rintf(x), ::rint(x))
FL_CUDA_UNARY_OP(absolute, ::fabsf(x), ::fabs(x))
FL_CUDA_UNARY_OP(sigmoid,
    1.0f / (1.0f + ::expf(-x)),
    1.0 / (1.0 + ::exp(-x)))
FL_CUDA_UNARY_OP(erf, ::erff(x), ::erf(x))
FL_CUDA_UNARY_OP(sign,
    static_cast<float>((x > 0.0f) - (x < 0.0f)),
    static_cast<double>((x > 0.0) - (x < 0.0)))

#undef FL_CUDA_UNARY_OP

// Ops with b8 output -- use f32 compute regardless (result is boolean).
void logicalNot(cudaStream_t stream, Tensor& out, const Tensor& tensor) {
  unaryTransformF32(stream, out, tensor, fl::dtype::b8,
      [] __device__ (float x) -> float { return x == 0 ? 1.0f : 0.0f; });
}

void isnan(cudaStream_t stream, Tensor& out, const Tensor& tensor) {
  unaryTransformF32(stream, out, tensor, fl::dtype::b8,
      [] __device__ (float x) -> float { return ::isnan(x) ? 1.0f : 0.0f; });
}

void isinf(cudaStream_t stream, Tensor& out, const Tensor& tensor) {
  unaryTransformF32(stream, out, tensor, fl::dtype::b8,
      [] __device__ (float x) -> float { return ::isinf(x) ? 1.0f : 0.0f; });
}

void clip(cudaStream_t stream, Tensor& out,
    const Tensor& tensor, const Tensor& low, const Tensor& high) {
  // Element-wise bounds that broadcast like any binary op (not just their
  // first element); the result keeps the input's dtype.
  Tensor lowerBounded;
  maximum(stream, lowerBounded, tensor, low);
  Tensor result;
  minimum(stream, result, lowerBounded, high);
  out = (result.type() == tensor.type()) ? result : result.astype(tensor.type());
}

template <typename T>
void flipImpl(cudaStream_t stream, Tensor& out, const Tensor& tensor, const unsigned dim) {
  size_t n = tensor.elements();
  auto contig = tensor.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? tensor : tensor.asContiguousTensor();
  auto& ci = contig.getAdapter<CudaTensor>();

  auto result = toTensor<CudaTensor>(tensor.shape(), tensor.type(), nullptr, MemoryLocation::Host);
  auto& resultAdapter = result.getAdapter<CudaTensor>();

  Dim dimSize = tensor.dim(dim);
  Dim innerStride = 1;
  for (unsigned d = tensor.ndim() - 1; d > dim; --d) {
    innerStride *= tensor.dim(d);
  }

  auto inPtr = thrust::device_pointer_cast(static_cast<const T*>(ci.devicePtr()));
  auto outPtr = thrust::device_pointer_cast(static_cast<T*>(resultAdapter.devicePtr()));

  thrust::tabulate(
      thrust::cuda::par.on(stream), outPtr, outPtr + n,
      [inPtr, dimSize, innerStride] __device__ (int idx) -> T {
        int dimIdx = (idx / innerStride) % dimSize;
        int flippedDimIdx = dimSize - 1 - dimIdx;
        int newIdx = idx + (flippedDimIdx - dimIdx) * innerStride;
        return inPtr[newIdx];
      });
  out = result;
}

// Flip and roll only move elements, so they dispatch on element size: a
// same-sized integer type copies every dtype bit-exactly.
template <typename F>
void dispatchByElementSize(fl::dtype type, F&& f) {
  switch (fl::getTypeSize(type)) {
    case 1: return f(uint8_t{});
    case 2: return f(uint16_t{});
    case 4: return f(uint32_t{});
    case 8: return f(uint64_t{});
  }
  throw std::invalid_argument("unsupported element size");
}

void flip(cudaStream_t stream, Tensor& out, const Tensor& tensor, const unsigned dim) {
  dispatchByElementSize(tensor.type(), [&](auto element) {
    flipImpl<decltype(element)>(stream, out, tensor, dim);
  });
}

template <typename T>
void rollImpl(cudaStream_t stream, Tensor& out, const Tensor& tensor, const int shift, const unsigned axis) {
  size_t n = tensor.elements();
  auto contig = tensor.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? tensor : tensor.asContiguousTensor();
  auto& ci = contig.getAdapter<CudaTensor>();

  auto result = toTensor<CudaTensor>(tensor.shape(), tensor.type(), nullptr, MemoryLocation::Host);
  auto& resultAdapter = result.getAdapter<CudaTensor>();

  Dim dimSize = tensor.dim(axis);
  Dim innerStride = 1;
  for (unsigned d = tensor.ndim() - 1; d > axis; --d) {
    innerStride *= tensor.dim(d);
  }

  auto inPtr = thrust::device_pointer_cast(static_cast<const T*>(ci.devicePtr()));
  auto outPtr = thrust::device_pointer_cast(static_cast<T*>(resultAdapter.devicePtr()));

  int sh = shift;
  thrust::tabulate(
      thrust::cuda::par.on(stream), outPtr, outPtr + n,
      [inPtr, dimSize, innerStride, sh] __device__ (int idx) -> T {
        int dimIdx = (idx / innerStride) % dimSize;
        int srcDimIdx = ((dimIdx - sh) % dimSize + dimSize) % dimSize;
        int srcIdx = idx + (srcDimIdx - dimIdx) * innerStride;
        return inPtr[srcIdx];
      });
  out = result;
}

void roll(cudaStream_t stream, Tensor& out, const Tensor& tensor, const int shift, const unsigned axis) {
  dispatchByElementSize(tensor.type(), [&](auto element) {
    rollImpl<decltype(element)>(stream, out, tensor, shift, axis);
  });
}

void trilImpl(cudaStream_t stream, Tensor& out, const Tensor& tensor, int diagonal) {
  if (tensor.ndim() < 2) {
    throw std::invalid_argument("tril requires input with rank at least 2");
  }
  const size_t elements = tensor.elements();
  const size_t typeSize = fl::getTypeSize(tensor.type());
  const size_t bytes = elements * typeSize;
  auto contig = tensor.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? tensor : tensor.asContiguousTensor();
  auto& ci = contig.getAdapter<CudaTensor>();

  auto result = toTensor<CudaTensor>(tensor.shape(), tensor.type(), nullptr, MemoryLocation::Host);
  auto& resultAdapter = result.getAdapter<CudaTensor>();

  const size_t rows = static_cast<size_t>(tensor.dim(tensor.ndim() - 2));
  const size_t cols = static_cast<size_t>(tensor.dim(tensor.ndim() - 1));

  auto inPtr = thrust::device_pointer_cast(static_cast<const unsigned char*>(ci.devicePtr()));
  auto outPtr = thrust::device_pointer_cast(static_cast<unsigned char*>(resultAdapter.devicePtr()));

  thrust::tabulate(
      thrust::cuda::par.on(stream), outPtr, outPtr + bytes,
      [inPtr, typeSize, rows, cols, diagonal] __device__ (size_t idx) -> unsigned char {
        const auto elementIdx = idx / typeSize;
        const auto row = (elementIdx / cols) % rows;
        const auto col = elementIdx % cols;
        const auto offset =
            static_cast<long long>(col) - static_cast<long long>(row);
        return (offset <= diagonal) ? inPtr[idx] : 0;
      });
  out = result;
}

void tril(cudaStream_t stream, Tensor& out, const Tensor& tensor, int diagonal) {
  trilImpl(stream, out, tensor, diagonal);
}

void triuImpl(cudaStream_t stream, Tensor& out, const Tensor& tensor, int diagonal) {
  if (tensor.ndim() < 2) {
    throw std::invalid_argument("triu requires input with rank at least 2");
  }
  const size_t elements = tensor.elements();
  const size_t typeSize = fl::getTypeSize(tensor.type());
  const size_t bytes = elements * typeSize;
  auto contig = tensor.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? tensor : tensor.asContiguousTensor();
  auto& ci = contig.getAdapter<CudaTensor>();

  auto result = toTensor<CudaTensor>(tensor.shape(), tensor.type(), nullptr, MemoryLocation::Host);
  auto& resultAdapter = result.getAdapter<CudaTensor>();

  const size_t rows = static_cast<size_t>(tensor.dim(tensor.ndim() - 2));
  const size_t cols = static_cast<size_t>(tensor.dim(tensor.ndim() - 1));

  auto inPtr = thrust::device_pointer_cast(static_cast<const unsigned char*>(ci.devicePtr()));
  auto outPtr = thrust::device_pointer_cast(static_cast<unsigned char*>(resultAdapter.devicePtr()));

  thrust::tabulate(
      thrust::cuda::par.on(stream), outPtr, outPtr + bytes,
      [inPtr, typeSize, rows, cols, diagonal] __device__ (size_t idx) -> unsigned char {
        const auto elementIdx = idx / typeSize;
        const auto row = (elementIdx / cols) % rows;
        const auto col = elementIdx % cols;
        const auto offset =
            static_cast<long long>(col) - static_cast<long long>(row);
        return (offset >= diagonal) ? inPtr[idx] : 0;
      });
  out = result;
}

void triu(cudaStream_t stream, Tensor& out, const Tensor& tensor, int diagonal) {
  triuImpl(stream, out, tensor, diagonal);
}

template <typename T>
void whereImpl(cudaStream_t stream, Tensor& out,
    const Tensor& condition, const Tensor& x, const Tensor& y) {
  size_t n = x.elements();
  // computeType must match T (chosen from both operands in where()), and the
  // result has the IR's dtype: promoteDtypes(x, y).
  const fl::dtype computeType =
      std::is_same_v<T, double> ? fl::dtype::f64 : fl::dtype::f32;
  const fl::dtype outType = fl::promoteDtypes(x.type(), y.type());

  // Condition is always checked as f32 (it's a boolean mask)
  auto condF32 = (condition.type() == fl::dtype::f32)
      ? condition : condition.astype(fl::dtype::f32);
  auto xCast = (x.type() == computeType) ? x : x.astype(computeType);
  auto yCast = (y.type() == computeType) ? y : y.astype(computeType);

  // Ensure contiguity
  if (!condF32.getAdapter<CudaTensor>().isContiguousRowMajor()) {
    condF32 = condF32.asContiguousTensor();
  }
  if (!xCast.getAdapter<CudaTensor>().isContiguousRowMajor()) {
    xCast = xCast.asContiguousTensor();
  }
  if (!yCast.getAdapter<CudaTensor>().isContiguousRowMajor()) {
    yCast = yCast.asContiguousTensor();
  }

  auto result = toTensor<CudaTensor>(
      x.shape(), computeType, nullptr, MemoryLocation::Host);
  auto& resultAdapter = result.getAdapter<CudaTensor>();

  auto cPtr = thrust::device_pointer_cast(
      static_cast<const float*>(condF32.getAdapter<CudaTensor>().devicePtr()));
  auto xPtr = thrust::device_pointer_cast(
      static_cast<const T*>(xCast.getAdapter<CudaTensor>().devicePtr()));
  auto yPtr = thrust::device_pointer_cast(
      static_cast<const T*>(yCast.getAdapter<CudaTensor>().devicePtr()));
  auto outPtr = thrust::device_pointer_cast(
      static_cast<T*>(resultAdapter.devicePtr()));

  thrust::tabulate(
      thrust::cuda::par.on(stream), outPtr, outPtr + n,
      [cPtr, xPtr, yPtr] __device__ (int idx) -> T {
        return cPtr[idx] != 0 ? xPtr[idx] : yPtr[idx];
      });

  if (outType != computeType) {
    out = result.astype(outType);
  } else {
    out = result;
  }
}

void where(cudaStream_t stream, Tensor& out,
    const Tensor& condition, const Tensor& x, const Tensor& y) {
  if (x.type() == fl::dtype::f64 || y.type() == fl::dtype::f64) {
    whereImpl<double>(stream, out, condition, x, y);
  } else {
    whereImpl<float>(stream, out, condition, x, y);
  }
}

// =========================================================================
// GPU-native sort operations via CUB DeviceSegmentedSort
// =========================================================================

void sort(cudaStream_t stream, Tensor& out,
    const Tensor& input, const Dim axis, const SortMode sortMode) {
  auto asF32 = input.astype(fl::dtype::f32);
  auto asF32C = asF32.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? asF32 : asF32.asContiguousTensor();
  const auto* inPtr = static_cast<const float*>(
      asF32C.getAdapter<CudaTensor>().devicePtr());

  int n = static_cast<int>(input.elements());
  auto outValues = toTensor<CudaTensor>(
      input.shape(), fl::dtype::f32, nullptr, MemoryLocation::Host);
  auto outIndices = toTensor<CudaTensor>(
      input.shape(), fl::dtype::s32, nullptr, MemoryLocation::Host);
  auto* outValPtr = static_cast<float*>(
      outValues.getAdapter<CudaTensor>().devicePtr());
  auto* outIdxPtr = static_cast<int*>(
      outIndices.getAdapter<CudaTensor>().devicePtr());

  gpuSortAlongAxis(
      inPtr, outValPtr, outIdxPtr, input.shape(), axis, sortMode, stream);

  if (input.type() != fl::dtype::f32) {
    out = outValues.astype(input.type());
  } else {
    out = outValues;
  }
}

void sort(cudaStream_t stream,
    Tensor& values, Tensor& indices, const Tensor& input,
    const Dim axis, const SortMode sortMode) {
  auto asF32 = input.astype(fl::dtype::f32);
  auto asF32C = asF32.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? asF32 : asF32.asContiguousTensor();
  const auto* inPtr = static_cast<const float*>(
      asF32C.getAdapter<CudaTensor>().devicePtr());

  auto outValues = toTensor<CudaTensor>(
      input.shape(), fl::dtype::f32, nullptr, MemoryLocation::Host);
  auto outIndices = toTensor<CudaTensor>(
      input.shape(), fl::dtype::s32, nullptr, MemoryLocation::Host);
  auto* outValPtr = static_cast<float*>(
      outValues.getAdapter<CudaTensor>().devicePtr());
  auto* outIdxPtr = static_cast<int*>(
      outIndices.getAdapter<CudaTensor>().devicePtr());

  gpuSortAlongAxis(
      inPtr, outValPtr, outIdxPtr, input.shape(), axis, sortMode, stream);

  values = (input.type() != fl::dtype::f32)
      ? outValues.astype(input.type()) : outValues;
  indices = outIndices;
}

void argsort(cudaStream_t stream, Tensor& out,
    const Tensor& input, const Dim axis, const SortMode sortMode) {
  auto asF32 = input.astype(fl::dtype::f32);
  auto asF32C = asF32.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? asF32 : asF32.asContiguousTensor();
  const auto* inPtr = static_cast<const float*>(
      asF32C.getAdapter<CudaTensor>().devicePtr());

  auto outValues = toTensor<CudaTensor>(
      input.shape(), fl::dtype::f32, nullptr, MemoryLocation::Host);
  auto outIndices = toTensor<CudaTensor>(
      input.shape(), fl::dtype::s32, nullptr, MemoryLocation::Host);
  auto* outValPtr = static_cast<float*>(
      outValues.getAdapter<CudaTensor>().devicePtr());
  auto* outIdxPtr = static_cast<int*>(
      outIndices.getAdapter<CudaTensor>().devicePtr());

  gpuSortAlongAxis(
      inPtr, outValPtr, outIdxPtr, input.shape(), axis, sortMode, stream);

  out = outIndices;
}

void topk(cudaStream_t stream,
    Tensor& values, Tensor& indices, const Tensor& input,
    const unsigned k, const Dim axis, const SortMode sortMode) {
  auto asF32 = input.astype(fl::dtype::f32);
  auto asF32C = asF32.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? asF32 : asF32.asContiguousTensor();
  const auto* inPtr = static_cast<const float*>(
      asF32C.getAdapter<CudaTensor>().devicePtr());

  // Full sort
  auto sortedValues = toTensor<CudaTensor>(
      input.shape(), fl::dtype::f32, nullptr, MemoryLocation::Host);
  auto sortedIndices = toTensor<CudaTensor>(
      input.shape(), fl::dtype::s32, nullptr, MemoryLocation::Host);
  auto* sortedValPtr = static_cast<float*>(
      sortedValues.getAdapter<CudaTensor>().devicePtr());
  auto* sortedIdxPtr = static_cast<int*>(
      sortedIndices.getAdapter<CudaTensor>().devicePtr());

  gpuSortAlongAxis(
      inPtr, sortedValPtr, sortedIdxPtr, input.shape(), axis, sortMode, stream);

  // Build output shape: same as input but axis dimension = k
  std::vector<Dim> outDims;
  for (int d = 0; d < input.ndim(); ++d) {
    outDims.push_back(
        d == static_cast<int>(axis) ? static_cast<Dim>(k) : input.shape()[d]);
  }
  Shape outShape(outDims);

  // Extract top-k on GPU
  auto topkVals = toTensor<CudaTensor>(
      outShape, fl::dtype::f32, nullptr, MemoryLocation::Host);
  auto topkIdxs = toTensor<CudaTensor>(
      outShape, fl::dtype::s32, nullptr, MemoryLocation::Host);
  auto* topkValPtr = static_cast<float*>(
      topkVals.getAdapter<CudaTensor>().devicePtr());
  auto* topkIdxPtr = static_cast<int*>(
      topkIdxs.getAdapter<CudaTensor>().devicePtr());

  int outerSize = 1;
  for (int d = 0; d < static_cast<int>(axis); ++d) {
    outerSize *= static_cast<int>(input.shape()[d]);
  }
  int innerSize = 1;
  for (int d = static_cast<int>(axis) + 1; d < input.ndim(); ++d) {
    innerSize *= static_cast<int>(input.shape()[d]);
  }

  int outN = static_cast<int>(outShape.elements());
  int grid = (outN + kSortBlockSize - 1) / kSortBlockSize;
  topkGatherKernel<<<grid, kSortBlockSize, 0, stream>>>(
      sortedValPtr, sortedIdxPtr, topkValPtr, topkIdxPtr,
      outerSize, static_cast<int>(k),
      static_cast<int>(input.shape()[axis]), innerSize);
  FL_CUDA_CHECK(cudaGetLastError());

  values = (input.type() != fl::dtype::f32)
      ? topkVals.astype(input.type()) : topkVals;
  indices = topkIdxs;
}

} // namespace fl::cuda
