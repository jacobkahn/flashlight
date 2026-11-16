/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include "flashlight/fl/runtime/CUDAUtils.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/backend/cuda/CudaBackend.h"
#include "flashlight/fl/tensor/backend/cuda/CudaTensor.h"
#include "flashlight/fl/tensor/ops/cuda/CudaOps.h"

namespace fl::cuda {

namespace {

constexpr int kBlockSize = 256;

// Compute the shape after reducing given axes, optionally keeping dims.
Shape computeReducedShape(
    const Shape& in, const std::vector<int>& axes, bool keepDims) {
  std::vector<Dim> out;
  for (int d = 0; d < in.ndim(); ++d) {
    bool reduced = false;
    for (int a : axes) {
      if (a == d) { reduced = true; break; }
    }
    if (reduced) {
      if (keepDims) {
        out.push_back(1);
      }
    } else {
      out.push_back(in[d]);
    }
  }
  if (out.empty()) {
    return Shape{};
  }
  return Shape(out);
}

// Normalize axes: if empty, reduce all dims.
std::vector<int> normalizeAxes(int ndim, const std::vector<int>& axes) {
  if (axes.empty()) {
    std::vector<int> all(ndim);
    std::iota(all.begin(), all.end(), 0);
    return all;
  }
  return axes;
}

// Check if axes form a contiguous range.
bool areAxesContiguous(const std::vector<int>& axes) {
  if (axes.size() <= 1) {
    return true;
  }
  auto sorted = axes;
  std::sort(sorted.begin(), sorted.end());
  for (size_t i = 1; i < sorted.size(); ++i) {
    if (sorted[i] != sorted[i-1] + 1) {
      return false;
    }
  }
  return true;
}

// Compute outerSize, reduceSize, innerSize for contiguous axis reduction.
void computeReductionDims(
    const Shape& shape,
    const std::vector<int>& axes,
    Dim& outerSize,
    Dim& reduceSize,
    Dim& innerSize) {
  auto sorted = axes;
  std::sort(sorted.begin(), sorted.end());
  int firstAxis = sorted.front();
  int lastAxis = sorted.back();

  outerSize = 1;
  for (int d = 0; d < firstAxis; ++d) outerSize *= shape[d];

  reduceSize = 1;
  for (int a : sorted) reduceSize *= shape[a];

  innerSize = 1;
  for (int d = lastAxis + 1; d < shape.ndim(); ++d) innerSize *= shape[d];
}

// =========================================================================
// GPU reduction kernels -- templated for f32/f64
// =========================================================================

enum class ReduceType { Sum, Amin, Amax, CountNonzero, Any, All };

// =========================================================================
// CUB helpers for segmented reductions (contiguous segments, innerSize == 1)
// =========================================================================

// Scaled counting iterator: iter[i] = (start + i) * scale
// Used to generate segment offsets without device memory allocation.
struct ScaleOp {
  int scale;
  __host__ __device__ __forceinline__ int operator()(int i) const {
    return i * scale;
  }
};

using ScaledCountingIterator = cub::TransformInputIterator<
    int, ScaleOp, cub::CountingInputIterator<int>>;

ScaledCountingIterator makeScaledIter(int start, int scale) {
  return ScaledCountingIterator(
      cub::CountingInputIterator<int>(start), ScaleOp{scale});
}

// Custom CUB reduction operators.
template <typename T>
struct CubAnyOp {
  __host__ __device__ __forceinline__ T operator()(T a, T b) const {
    return (a != T(0) || b != T(0)) ? T(1) : T(0);
  }
};

template <typename T>
struct CubAllOp {
  __host__ __device__ __forceinline__ T operator()(T a, T b) const {
    return (a != T(0) && b != T(0)) ? T(1) : T(0);
  }
};

// Transform op for CountNonzero: maps each value to 1 if nonzero, 0 otherwise.
template <typename T>
struct NonzeroTransformOp {
  __host__ __device__ __forceinline__ T operator()(T x) const {
    return x != T(0) ? T(1) : T(0);
  }
};

// Run a CUB API call with the standard two-pass pattern (size query + execute).
template <typename CubCallFn>
void cubTwoPass(CubCallFn&& fn, cudaStream_t stream) {
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

// CUB segmented reduction for contiguous segments (innerSize == 1).
template <typename T>
void cubSegmentedReduce(
    const T* d_in, T* d_out,
    int numSegments, int segmentSize,
    ReduceType type, cudaStream_t stream) {
  auto beginOffsets = makeScaledIter(0, segmentSize);
  auto endOffsets = makeScaledIter(1, segmentSize);

  switch (type) {
    case ReduceType::Sum:
      cubTwoPass([&](void* temp, size_t& bytes) {
        cub::DeviceSegmentedReduce::Sum(
            temp, bytes, d_in, d_out,
            numSegments, beginOffsets, endOffsets, stream);
      }, stream);
      break;
    case ReduceType::Amin:
      cubTwoPass([&](void* temp, size_t& bytes) {
        cub::DeviceSegmentedReduce::Min(
            temp, bytes, d_in, d_out,
            numSegments, beginOffsets, endOffsets, stream);
      }, stream);
      break;
    case ReduceType::Amax:
      cubTwoPass([&](void* temp, size_t& bytes) {
        cub::DeviceSegmentedReduce::Max(
            temp, bytes, d_in, d_out,
            numSegments, beginOffsets, endOffsets, stream);
      }, stream);
      break;
    case ReduceType::CountNonzero: {
      cub::TransformInputIterator<T, NonzeroTransformOp<T>, const T*>
          transformedIn(d_in, NonzeroTransformOp<T>{});
      cubTwoPass([&](void* temp, size_t& bytes) {
        cub::DeviceSegmentedReduce::Sum(
            temp, bytes, transformedIn, d_out,
            numSegments, beginOffsets, endOffsets, stream);
      }, stream);
      break;
    }
    case ReduceType::Any:
      cubTwoPass([&](void* temp, size_t& bytes) {
        cub::DeviceSegmentedReduce::Reduce(
            temp, bytes, d_in, d_out,
            numSegments, beginOffsets, endOffsets,
            CubAnyOp<T>{}, T(0), stream);
      }, stream);
      break;
    case ReduceType::All:
      cubTwoPass([&](void* temp, size_t& bytes) {
        cub::DeviceSegmentedReduce::Reduce(
            temp, bytes, d_in, d_out,
            numSegments, beginOffsets, endOffsets,
            CubAllOp<T>{}, T(1), stream);
      }, stream);
      break;
  }
}

// CUB segmented argmin/argmax for contiguous segments (innerSize == 1).
// Outputs values and indices into separate arrays.
template <typename T>
void cubSegmentedArgReduce(
    const T* d_in, T* d_outValues, unsigned* d_outIndices,
    int numSegments, int segmentSize,
    bool isMin, cudaStream_t stream) {
  auto beginOffsets = makeScaledIter(0, segmentSize);
  auto endOffsets = makeScaledIter(1, segmentSize);

  using KVPair = cub::KeyValuePair<int, T>;
  KVPair* d_kvPairs = nullptr;
  FL_CUDA_CHECK(cudaMallocAsync(
      &d_kvPairs, numSegments * sizeof(KVPair), stream));

  if (isMin) {
    cubTwoPass([&](void* temp, size_t& bytes) {
      cub::DeviceSegmentedReduce::ArgMin(
          temp, bytes, d_in, d_kvPairs,
          numSegments, beginOffsets, endOffsets, stream);
    }, stream);
  } else {
    cubTwoPass([&](void* temp, size_t& bytes) {
      cub::DeviceSegmentedReduce::ArgMax(
          temp, bytes, d_in, d_kvPairs,
          numSegments, beginOffsets, endOffsets, stream);
    }, stream);
  }

  // Extract values and indices from KeyValuePairs
  auto kvPtr = thrust::device_pointer_cast(d_kvPairs);
  auto valPtr = thrust::device_pointer_cast(d_outValues);
  auto idxPtr = thrust::device_pointer_cast(d_outIndices);
  auto policy = thrust::cuda::par.on(stream);

  thrust::transform(policy, kvPtr, kvPtr + numSegments, valPtr,
      [] __device__ (const KVPair& kv) -> T { return kv.value; });
  thrust::transform(policy, kvPtr, kvPtr + numSegments, idxPtr,
      [] __device__ (const KVPair& kv) -> unsigned {
        return static_cast<unsigned>(kv.key);
      });

  FL_CUDA_CHECK(cudaFreeAsync(d_kvPairs, stream));
}

// =========================================================================
// Fallback kernels for strided data (innerSize > 1)
// =========================================================================

// Thrust functors templated for value type.
template <typename T>
struct IsNonzero {
  __host__ __device__ T operator()(T x) const {
    return x != T(0) ? T(1) : T(0);
  }
};

template <typename T>
struct AnyOp {
  __host__ __device__ T operator()(T a, T b) const {
    return (a != T(0) || b != T(0)) ? T(1) : T(0);
  }
};

template <typename T>
struct AllOp {
  __host__ __device__ T operator()(T a, T b) const {
    return (a != T(0) && b != T(0)) ? T(1) : T(0);
  }
};

template <typename T>
__device__ __forceinline__ T reduceInit(ReduceType type) {
  switch (type) {
    case ReduceType::Sum:
    case ReduceType::CountNonzero:
    case ReduceType::Any:
      return T(0);
    case ReduceType::All:
      return T(1);
    case ReduceType::Amin:
      return T(CUDART_INF_F);
    case ReduceType::Amax:
      return T(-CUDART_INF_F);
  }
  return T(0);
}

template <typename T>
__device__ __forceinline__ T reduceOp(ReduceType type, T acc, T val) {
  switch (type) {
    case ReduceType::Sum:
      return acc + val;
    case ReduceType::Amin:
      return acc < val ? acc : val;
    case ReduceType::Amax:
      return acc > val ? acc : val;
    case ReduceType::CountNonzero:
      return acc + (val != T(0) ? T(1) : T(0));
    case ReduceType::Any:
      return (acc != T(0) || val != T(0)) ? T(1) : T(0);
    case ReduceType::All:
      return (acc != T(0) && val != T(0)) ? T(1) : T(0);
  }
  return acc;
}

// Axis-wise reduction kernel for strided data (innerSize > 1).
// Each thread computes one output element by sequentially looping over the
// reduction dimension. Used as fallback when CUB can't operate directly.
template <typename T>
__global__ void axisReduceKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int outerSize,
    int reduceSize,
    int innerSize,
    ReduceType type) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int numOutputs = outerSize * innerSize;
  if (tid >= numOutputs) {
    return;
  }

  int outerIdx = tid / innerSize;
  int innerIdx = tid % innerSize;
  int baseIdx = outerIdx * reduceSize * innerSize + innerIdx;

  T acc = reduceInit<T>(type);
  for (int r = 0; r < reduceSize; ++r) {
    acc = reduceOp<T>(type, acc, input[baseIdx + r * innerSize]);
  }
  output[tid] = acc;
}

// GPU-native axis-wise reduction for contiguous axes -- templated.
template <typename T>
void gpuAxisReduceTyped(
    Tensor& out,
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims,
    fl::dtype computeType,
    fl::dtype outType,
    ReduceType type,
    cudaStream_t stream) {
  auto typed = (input.type() == computeType)
      ? input : input.astype(computeType);
  auto typedC = typed.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? typed : typed.asContiguousTensor();
  const auto* inPtr = static_cast<const T*>(
      typedC.getAdapter<CudaTensor>().devicePtr());

  auto outShape = computeReducedShape(input.shape(), axes, keepDims);
  Dim outerSize, reduceSize, innerSize;
  computeReductionDims(input.shape(), axes, outerSize, reduceSize, innerSize);

  Dim numOutputs = outerSize * innerSize;
  // Reuse pre-assigned buffer when output type matches compute type
  bool reuse = !out.isEmpty() && outType == computeType;
  Tensor result = reuse
      ? out
      : toTensor<CudaTensor>(outShape, computeType, nullptr, MemoryLocation::Host);
  auto* outPtr = static_cast<T*>(
      result.getAdapter<CudaTensor>().devicePtr());

  if (innerSize == 1) {
    // Segments are contiguous -- use CUB for optimized parallel reduction
    cubSegmentedReduce<T>(
        inPtr, outPtr,
        static_cast<int>(numOutputs),
        static_cast<int>(reduceSize),
        type, stream);
  } else {
    // Strided data -- fall back to per-thread kernel
    int gridSize =
        (static_cast<int>(numOutputs) + kBlockSize - 1) / kBlockSize;
    axisReduceKernel<T><<<gridSize, kBlockSize, 0, stream>>>(
        inPtr, outPtr,
        static_cast<int>(outerSize),
        static_cast<int>(reduceSize),
        static_cast<int>(innerSize),
        type);
  }
  FL_CUDA_CHECK(cudaGetLastError());

  if (outType != computeType) {
    out = result.astype(outType);
  } else if (!reuse) {
    out = result;
  }
}

// Full reduction (all axes) using Thrust -- templated.
template <typename T>
void gpuFullReduceTyped(
    Tensor& out,
    const Tensor& input,
    bool keepDims,
    fl::dtype computeType,
    fl::dtype outType,
    ReduceType type,
    cudaStream_t stream) {
  auto typed = (input.type() == computeType)
      ? input : input.astype(computeType);
  auto typedC = typed.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? typed : typed.asContiguousTensor();
  const auto* inPtr = static_cast<const T*>(
      typedC.getAdapter<CudaTensor>().devicePtr());
  size_t n = input.elements();

  auto dptr = thrust::device_pointer_cast(inPtr);
  auto policy = thrust::cuda::par.on(stream);
  T result;

  switch (type) {
    case ReduceType::Sum:
      result = thrust::reduce(policy, dptr, dptr + n, T(0), thrust::plus<T>());
      break;
    case ReduceType::Amin:
      result = *thrust::min_element(policy, dptr, dptr + n);
      break;
    case ReduceType::Amax:
      result = *thrust::max_element(policy, dptr, dptr + n);
      break;
    case ReduceType::CountNonzero:
      result = thrust::transform_reduce(
          policy, dptr, dptr + n, IsNonzero<T>(), T(0), thrust::plus<T>());
      break;
    case ReduceType::Any:
      result = thrust::reduce(policy, dptr, dptr + n, T(0), AnyOp<T>());
      break;
    case ReduceType::All:
      result = thrust::reduce(policy, dptr, dptr + n, T(1), AllOp<T>());
      break;
  }

  Shape outShape = keepDims
      ? Shape(std::vector<Dim>(input.ndim(), 1))
      : Shape{};

  // Reuse pre-assigned buffer when output type matches compute type.
  // Thrust synchronizes the stream, so the host result is ready; copy it
  // directly into the existing device buffer.
  if (!out.isEmpty() && outType == computeType) {
    auto* outPtr = static_cast<T*>(out.getAdapter<CudaTensor>().devicePtr());
    cudaMemcpyAsync(outPtr, &result, sizeof(T), cudaMemcpyHostToDevice, stream);
    return;
  }

  // Create scalar output directly as CudaTensor (not via default tensor type)
  if constexpr (std::is_same_v<T, double>) {
    std::vector<double> v = {result};
    Tensor t = toTensor<CudaTensor>(outShape, fl::dtype::f64, v.data(), Location::Host);
    out = (outType != fl::dtype::f64) ? t.astype(outType) : t;
  } else {
    std::vector<float> v = {result};
    Tensor t = toTensor<CudaTensor>(outShape, fl::dtype::f32, v.data(), Location::Host);
    out = (outType != fl::dtype::f32) ? t.astype(outType) : t;
  }
}

// General axis reduction kernel (non-contiguous axes) -- templated.
constexpr int kMaxReduceDims = 8;

struct GeneralReduceParams {
  int keptSizes[kMaxReduceDims];
  int keptOutStrides[kMaxReduceDims];
  int keptInStrides[kMaxReduceDims];
  int numKept;
  int reducedSizes[kMaxReduceDims];
  int reducedIterStrides[kMaxReduceDims];
  int reducedInStrides[kMaxReduceDims];
  int numReduced;
  int numOutputs;
  int numReduceElems;
};

template <typename T>
__global__ void generalAxisReduceKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    GeneralReduceParams params,
    ReduceType type) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= params.numOutputs) {
    return;
  }

  int baseInputIdx = 0;
  int rem = tid;
  for (int i = 0; i < params.numKept; ++i) {
    int coord = rem / params.keptOutStrides[i];
    rem %= params.keptOutStrides[i];
    baseInputIdx += coord * params.keptInStrides[i];
  }

  T acc = reduceInit<T>(type);
  for (int ri = 0; ri < params.numReduceElems; ++ri) {
    int inputIdx = baseInputIdx;
    int rrem = ri;
    for (int j = 0; j < params.numReduced; ++j) {
      int coord = rrem / params.reducedIterStrides[j];
      rrem %= params.reducedIterStrides[j];
      inputIdx += coord * params.reducedInStrides[j];
    }
    acc = reduceOp<T>(type, acc, input[inputIdx]);
  }

  output[tid] = acc;
}

// GPU-native general axis reduction -- templated.
template <typename T>
void gpuGeneralAxisReduceTyped(
    Tensor& out,
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims,
    fl::dtype computeType,
    fl::dtype outType,
    ReduceType type,
    cudaStream_t stream) {
  auto typed = (input.type() == computeType)
      ? input : input.astype(computeType);
  auto typedC = typed.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? typed : typed.asContiguousTensor();
  const auto* inPtr = static_cast<const T*>(
      typedC.getAdapter<CudaTensor>().devicePtr());

  const auto& inShape = input.shape();
  int ndim = inShape.ndim();

  std::vector<int> inStrides(ndim);
  {
    int s = 1;
    for (int d = ndim - 1; d >= 0; --d) {
      inStrides[d] = s;
      s *= static_cast<int>(inShape[d]);
    }
  }

  std::set<int> reducedSet(axes.begin(), axes.end());
  std::vector<int> keptDims, reducedDims;
  for (int d = 0; d < ndim; ++d) {
    if (reducedSet.contains(d)) {
      reducedDims.push_back(d);
    } else {
      keptDims.push_back(d);
    }
  }

  GeneralReduceParams params{};
  params.numKept = static_cast<int>(keptDims.size());
  params.numReduced = static_cast<int>(reducedDims.size());

  params.numOutputs = 1;
  for (int i = 0; i < params.numKept; ++i) {
    params.keptSizes[i] = static_cast<int>(inShape[keptDims[i]]);
    params.keptInStrides[i] = inStrides[keptDims[i]];
    params.numOutputs *= params.keptSizes[i];
  }
  if (params.numKept > 0) {
    int s = 1;
    for (int i = params.numKept - 1; i >= 0; --i) {
      params.keptOutStrides[i] = s;
      s *= params.keptSizes[i];
    }
  }

  params.numReduceElems = 1;
  for (int i = 0; i < params.numReduced; ++i) {
    params.reducedSizes[i] = static_cast<int>(inShape[reducedDims[i]]);
    params.reducedInStrides[i] = inStrides[reducedDims[i]];
    params.numReduceElems *= params.reducedSizes[i];
  }
  if (params.numReduced > 0) {
    int s = 1;
    for (int i = params.numReduced - 1; i >= 0; --i) {
      params.reducedIterStrides[i] = s;
      s *= params.reducedSizes[i];
    }
  }

  auto outShape = computeReducedShape(inShape, axes, keepDims);
  // Reuse pre-assigned buffer when output type matches compute type
  bool reuse = !out.isEmpty() && outType == computeType;
  Tensor result = reuse
      ? out
      : toTensor<CudaTensor>(outShape, computeType, nullptr, MemoryLocation::Host);
  auto* outPtr = static_cast<T*>(
      result.getAdapter<CudaTensor>().devicePtr());

  int gridSize = (params.numOutputs + kBlockSize - 1) / kBlockSize;
  generalAxisReduceKernel<T><<<gridSize, kBlockSize, 0, stream>>>(
      inPtr, outPtr, params, type);
  FL_CUDA_CHECK(cudaGetLastError());

  if (outType != computeType) {
    out = result.astype(outType);
  } else if (!reuse) {
    out = result;
  }
}

// Dispatch to the appropriate GPU reduction strategy.
void gpuReduce(
    Tensor& out,
    cudaStream_t stream,
    const Tensor& input,
    const std::vector<int>& rawAxes,
    bool keepDims,
    fl::dtype outType,
    ReduceType type) {
  auto axes = normalizeAxes(input.ndim(), rawAxes);
  size_t n = input.elements();
  if (n == 0) {
    auto outShape = computeReducedShape(input.shape(), axes, keepDims);
    out = toTensor<CudaTensor>(
        outShape, outType, nullptr, MemoryLocation::Host);
    return;
  }

  // For 0-dim (scalar) inputs, always treat as full reduction regardless of
  // axes, to avoid OOB access in computeReductionDims.
  bool useF64 = (input.type() == fl::dtype::f64);
  bool fullReduce = (input.ndim() == 0) ||
      (static_cast<int>(axes.size()) == input.ndim());

  if (fullReduce) {
    if (useF64) {
      gpuFullReduceTyped<double>(
          out, input, keepDims, fl::dtype::f64, outType, type, stream);
    } else {
      gpuFullReduceTyped<float>(
          out, input, keepDims, fl::dtype::f32, outType, type, stream);
    }
    return;
  }

  if (areAxesContiguous(axes)) {
    if (useF64) {
      gpuAxisReduceTyped<double>(
          out, input, axes, keepDims, fl::dtype::f64, outType, type, stream);
    } else {
      gpuAxisReduceTyped<float>(
          out, input, axes, keepDims, fl::dtype::f32, outType, type, stream);
    }
    return;
  }

  if (useF64) {
    gpuGeneralAxisReduceTyped<double>(
        out, input, axes, keepDims, fl::dtype::f64, outType, type, stream);
  } else {
    gpuGeneralAxisReduceTyped<float>(
        out, input, axes, keepDims, fl::dtype::f32, outType, type, stream);
  }
}

} // namespace

void amin(
    cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes, const bool keepDims) {
  gpuReduce(out, stream, input, axes, keepDims, input.type(), ReduceType::Amin);
}

void amax(
    cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes, const bool keepDims) {
  gpuReduce(out, stream, input, axes, keepDims, input.type(), ReduceType::Amax);
}

void sum(
    cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes, const bool keepDims) {
  // Sum of boolean should produce u32 (count of true values)
  auto outType = (input.type() == fl::dtype::b8) ? fl::dtype::u32 : input.type();
  gpuReduce(out, stream, input, axes, keepDims, outType, ReduceType::Sum);
}

void mean(
    cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes, const bool keepDims) {
  // Mean of integers should produce f32 (matching OpProperties inference)
  auto promoted = !isFloatingPoint(input.type())
      ? input.astype(fl::dtype::f32) : input;
  // Capture shape/ndim before sum() to avoid lifetime issues with input ref
  auto inShape = input.shape();
  auto inNdim = input.ndim();
  Tensor s;
  sum(stream, s, promoted, axes, keepDims);
  // Compute element count for the reduced axes
  Dim count;
  if (inNdim == 0) {
    // Scalar input: reducing over axes={0} means the single element
    count = 1;
  } else {
    auto reducedAxes = normalizeAxes(inNdim, axes);
    count = 1;
    for (int a : reducedAxes) count *= inShape.dim(a);
  }
  div(stream, out, s,
      CudaBackend::getInstance().full(s.shape(), static_cast<float>(count), s.type()));
}

// Median pick kernel: for each segment of sorted data, extract the median.
__global__ void medianPickKernel(
    const float* __restrict__ sorted,
    float* __restrict__ output,
    int numSegments, int axisSize, int innerSize) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numSegments) {
    return;
  }

  int outer = tid / innerSize;
  int inner = tid % innerSize;
  int base = outer * axisSize * innerSize + inner;

  int mid = axisSize / 2;
  if (axisSize % 2 == 1) {
    output[tid] = sorted[base + mid * innerSize];
  } else {
    float a = sorted[base + (mid - 1) * innerSize];
    float b = sorted[base + mid * innerSize];
    output[tid] = (a + b) * 0.5f;
  }
}

void median(
    cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes, const bool keepDims) {
  auto reducedAxes = normalizeAxes(input.ndim(), axes);
  auto outShape = computeReducedShape(input.shape(), reducedAxes, keepDims);

  // Full reduction: flatten, sort on GPU, pick middle element(s) on GPU
  bool fullReduce = (static_cast<int>(reducedAxes.size()) == input.ndim());
  if (fullReduce) {
    Dim n = input.elements();
    Tensor flat;
    reshape(stream, flat, input, Shape{n});
    Tensor sorted;
    sort(stream, sorted, flat, 0, SortMode::Ascending);
    auto asF32 = sorted.astype(fl::dtype::f32);
    auto asF32C = asF32.getAdapter<CudaTensor>().isContiguousRowMajor()
        ? asF32 : asF32.asContiguousTensor();
    const auto* sortedPtr = static_cast<const float*>(
        asF32C.getAdapter<CudaTensor>().devicePtr());

    auto medShape = outShape.ndim() == 0 ? Shape{} : outShape;
    // Reuse pre-assigned buffer when output type is f32 (median compute type)
    bool reuse = !out.isEmpty() && out.type() == fl::dtype::f32;
    Tensor result = reuse
        ? out
        : toTensor<CudaTensor>(medShape, fl::dtype::f32, nullptr, MemoryLocation::Host);
    auto* resultPtr = static_cast<float*>(
        result.getAdapter<CudaTensor>().devicePtr());

    // 1 output element, axisSize = n, innerSize = 1
    medianPickKernel<<<1, 1, 0, stream>>>(
        sortedPtr, resultPtr, 1, static_cast<int>(n), 1);
    FL_CUDA_CHECK(cudaGetLastError());

    // Median always returns f32 for non-float inputs (matching OpProperties)
    if (isFloatingPoint(input.type()) && input.type() != fl::dtype::f32) {
      out = result.astype(input.type());
    } else if (!reuse) {
      out = result;
    }
    return;
  }

  // For multi-axis reduction, transpose reduced axes to the end and merge them.
  // This converts multi-axis median into a single-axis median along the last dim.
  Tensor toSort = input;
  int sortAxis;
  if (reducedAxes.size() > 1) {
    // Build permutation: kept axes first (in order), reduced axes last (in order)
    std::unordered_set<int> reducedSet(reducedAxes.begin(), reducedAxes.end());
    std::vector<Dim> perm;
    for (int d = 0; d < input.ndim(); ++d) {
      if (!reducedSet.contains(d)) {
        perm.push_back(d);
      }
    }
    for (int a : reducedAxes) {
      perm.push_back(a);
    }
    Tensor transposed;
    transpose(stream, transposed, input, Shape(perm));

    // Merge reduced axes (now trailing) into a single dimension
    Dim reduceDimSize = 1;
    for (int a : reducedAxes) {
      reduceDimSize *= input.dim(a);
    }
    int keptDims = input.ndim() - static_cast<int>(reducedAxes.size());
    std::vector<Dim> mergedDims;
    for (int d = 0; d < keptDims; ++d) {
      mergedDims.push_back(transposed.dim(d));
    }
    mergedDims.push_back(reduceDimSize);
    reshape(stream, toSort, transposed, Shape(mergedDims));
    sortAxis = keptDims; // last axis
  } else {
    sortAxis = reducedAxes[0];
  }

  // Sort along the reduction axis, then pick middle element per segment.
  Tensor sorted;
  sort(stream, sorted, toSort, sortAxis, SortMode::Ascending);
  auto asF32 = sorted.astype(fl::dtype::f32);
  auto asF32C = asF32.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? asF32 : asF32.asContiguousTensor();
  const auto* sortedPtr = static_cast<const float*>(
      asF32C.getAdapter<CudaTensor>().devicePtr());

  const auto& sortShape = toSort.shape();
  int axisSize = static_cast<int>(sortShape[sortAxis]);
  int outerSize = 1;
  for (int d = 0; d < sortAxis; ++d) {
    outerSize *= static_cast<int>(sortShape[d]);
  }
  int innerSize = 1;
  for (int d = sortAxis + 1; d < sortShape.ndim(); ++d) {
    innerSize *= static_cast<int>(sortShape[d]);
  }

  int numOutputs = outerSize * innerSize;
  // Reuse pre-assigned buffer when output type is f32 (median compute type)
  bool reuse = !out.isEmpty() && out.type() == fl::dtype::f32;
  Tensor result = reuse
      ? out
      : toTensor<CudaTensor>(outShape, fl::dtype::f32, nullptr, MemoryLocation::Host);
  auto* resultPtr = static_cast<float*>(
      result.getAdapter<CudaTensor>().devicePtr());

  int grid = (numOutputs + kBlockSize - 1) / kBlockSize;
  medianPickKernel<<<grid, kBlockSize, 0, stream>>>(
      sortedPtr, resultPtr, numOutputs, axisSize, innerSize);
  FL_CUDA_CHECK(cudaGetLastError());

  // Median always returns f32 for non-float inputs (matching OpProperties)
  if (isFloatingPoint(input.type()) && input.type() != fl::dtype::f32) {
    out = result.astype(input.type());
  } else if (!reuse) {
    out = result;
  }
}

void var(
    cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes,
    bool bias, const bool keepDims) {
  // Capture shape before operations that might invalidate the input reference
  auto inNdim = input.ndim();
  auto inShape = input.shape();
  Tensor m;
  mean(stream, m, input, axes, true);
  Tensor diff;
  sub(stream, diff, input, m);
  Tensor sq;
  mul(stream, sq, diff, diff);
  Tensor s;
  sum(stream, s, sq, axes, keepDims);
  Dim count;
  if (inNdim == 0) {
    count = 1;
  } else {
    auto reducedAxes = normalizeAxes(inNdim, axes);
    count = 1;
    for (int a : reducedAxes) count *= inShape.dim(a);
  }
  // bias=true → sample variance (divide by n-1, Bessel's correction)
  // bias=false → population variance (divide by n)
  if (bias && count <= 1) {
    // Sample variance with count <= 1 is 0 (avoid division by zero)
    out = CudaBackend::getInstance().full(s.shape(), 0.0f, s.type());
    return;
  }
  float denom = bias ? static_cast<float>(count - 1)
                     : static_cast<float>(count);
  div(stream, out, s,
      CudaBackend::getInstance().full(s.shape(), denom, s.type()));
}

void std(
    cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes, const bool keepDims) {
  Tensor v;
  var(stream, v, input, axes, false, keepDims);
  sqrt(stream, out, v);
}

void norm(
    cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes,
    double p, const bool keepDims) {
  // Promote integer inputs to f32 for norm computation
  auto promoted = !isFloatingPoint(input.type())
      ? input.astype(fl::dtype::f32) : input;
  auto& be = CudaBackend::getInstance();
  if (p == 2.0) {
    Tensor sq;
    mul(stream, sq, promoted, promoted);
    Tensor s;
    sum(stream, s, sq, axes, keepDims);
    sqrt(stream, out, s);
    return;
  } else if (p == 1.0) {
    Tensor absx;
    absolute(stream, absx, promoted);
    sum(stream, out, absx, axes, keepDims);
    return;
  }
  Tensor absx;
  absolute(stream, absx, promoted);
  auto pTensor = be.full(promoted.shape(), static_cast<float>(p), promoted.type());
  Tensor raised;
  power(stream, raised, absx, pTensor);
  Tensor s;
  sum(stream, s, raised, axes, keepDims);
  auto invP = be.full(s.shape(), 1.0f / static_cast<float>(p), s.type());
  power(stream, out, s, invP);
}

void countNonzero(
    cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes, const bool keepDims) {
  gpuReduce(
      out, stream, input, axes, keepDims, fl::dtype::u32, ReduceType::CountNonzero);
}

void any(
    cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes, const bool keepDims) {
  gpuReduce(
      out, stream, input, axes, keepDims, fl::dtype::b8, ReduceType::Any);
}

void all(
    cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes, const bool keepDims) {
  gpuReduce(
      out, stream, input, axes, keepDims, fl::dtype::b8, ReduceType::All);
}

// Cumulative sum: CUB DeviceScan for contiguous segments, fallback kernel
// for strided data.
template <typename T>
__global__ void cumsumKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int outerSize, int axisSize, int innerSize) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int numSegments = outerSize * innerSize;
  if (tid >= numSegments) {
    return;
  }

  int outer = tid / innerSize;
  int inner = tid % innerSize;
  int base = outer * axisSize * innerSize + inner;

  T acc = T(0);
  for (int j = 0; j < axisSize; ++j) {
    int idx = base + j * innerSize;
    acc += input[idx];
    output[idx] = acc;
  }
}

template <typename T>
void cumsumTyped(
    const T* inPtr, T* outPtr,
    int outerSize, int axisSize, int innerSize,
    cudaStream_t stream) {
  if (innerSize == 1) {
    // Contiguous segments -- use CUB DeviceScan per segment.
    // Pre-compute temp size once (same for all segments of equal length),
    // allocate once, and reuse across all segment scans.
    size_t tempBytes = 0;
    cub::DeviceScan::InclusiveSum(
        nullptr, tempBytes, inPtr, outPtr, axisSize, stream);
    void* d_temp = nullptr;
    if (tempBytes > 0) {
      FL_CUDA_CHECK(cudaMallocAsync(&d_temp, tempBytes, stream));
    }
    for (int seg = 0; seg < outerSize; ++seg) {
      cub::DeviceScan::InclusiveSum(
          d_temp, tempBytes,
          inPtr + seg * axisSize,
          outPtr + seg * axisSize,
          axisSize, stream);
    }
    if (d_temp) {
      FL_CUDA_CHECK(cudaFreeAsync(d_temp, stream));
    }
  } else {
    // Strided data -- fall back to per-thread kernel
    int numSegments = outerSize * innerSize;
    int grid = (numSegments + kBlockSize - 1) / kBlockSize;
    cumsumKernel<T><<<grid, kBlockSize, 0, stream>>>(
        inPtr, outPtr, outerSize, axisSize, innerSize);
  }
}

void cumsum(
    cudaStream_t stream, Tensor& out,
    const Tensor& input, const unsigned axis) {
  bool useF64 = (input.type() == fl::dtype::f64);
  fl::dtype computeType = useF64 ? fl::dtype::f64 : fl::dtype::f32;

  auto typed = (input.type() == computeType)
      ? input : input.astype(computeType);
  auto typedC = typed.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? typed : typed.asContiguousTensor();

  const auto& shape = input.shape();
  int axisSize = static_cast<int>(shape[axis]);

  int outerSize = 1;
  for (int d = 0; d < static_cast<int>(axis); ++d) {
    outerSize *= static_cast<int>(shape[d]);
  }
  int innerSize = 1;
  for (int d = static_cast<int>(axis) + 1; d < shape.ndim(); ++d) {
    innerSize *= static_cast<int>(shape[d]);
  }

  auto result = toTensor<CudaTensor>(
      shape, computeType, nullptr, MemoryLocation::Host);

  if (useF64) {
    const auto* inPtr = static_cast<const double*>(
        typedC.getAdapter<CudaTensor>().devicePtr());
    auto* outPtr = static_cast<double*>(
        result.getAdapter<CudaTensor>().devicePtr());
    cumsumTyped<double>(inPtr, outPtr, outerSize, axisSize, innerSize, stream);
  } else {
    const auto* inPtr = static_cast<const float*>(
        typedC.getAdapter<CudaTensor>().devicePtr());
    auto* outPtr = static_cast<float*>(
        result.getAdapter<CudaTensor>().devicePtr());
    cumsumTyped<float>(inPtr, outPtr, outerSize, axisSize, innerSize, stream);
  }
  FL_CUDA_CHECK(cudaGetLastError());

  if (input.type() != computeType) {
    out = result.astype(input.type());
  } else {
    out = result;
  }
}

// Arg-reduce: CUB for contiguous segments (innerSize == 1),
// fallback kernel for strided data.
template <typename T>
__global__ void argReduceKernel(
    const T* __restrict__ input,
    T* __restrict__ outValues,
    unsigned* __restrict__ outIndices,
    int outerSize, int axisSize, int innerSize,
    bool isMin) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int numOutputs = outerSize * innerSize;
  if (tid >= numOutputs) {
    return;
  }

  int outer = tid / innerSize;
  int inner = tid % innerSize;
  int base = outer * axisSize * innerSize + inner;

  T bestVal = input[base];
  int bestIdx = 0;
  for (int k = 1; k < axisSize; ++k) {
    T v = input[base + k * innerSize];
    bool better = isMin ? (v < bestVal) : (v > bestVal);
    if (better) {
      bestVal = v;
      bestIdx = k;
    }
  }
  outValues[tid] = bestVal;
  outIndices[tid] = bestIdx;
}

// Shared implementation for min/max with CUB fast path.
template <typename T>
void argReduceImpl(
    cudaStream_t stream,
    const T* inPtr,
    T* outValPtr,
    unsigned* outIdxPtr,
    int outerSize, int axisSize, int innerSize,
    bool isMin) {
  int numOutputs = outerSize * innerSize;

  if (innerSize == 1) {
    // Contiguous segments -- use CUB ArgMin/ArgMax
    cubSegmentedArgReduce<T>(
        inPtr, outValPtr, outIdxPtr,
        numOutputs, axisSize, isMin, stream);
  } else {
    // Strided data -- fall back to per-thread kernel
    int grid = (numOutputs + kBlockSize - 1) / kBlockSize;
    argReduceKernel<T><<<grid, kBlockSize, 0, stream>>>(
        inPtr, outValPtr, outIdxPtr,
        outerSize, axisSize, innerSize, isMin);
  }
}

void min(
    cudaStream_t stream,
    Tensor& values, Tensor& indices, const Tensor& input,
    const unsigned axis, const bool keepDims) {
  bool useF64 = (input.type() == fl::dtype::f64);
  fl::dtype computeType = useF64 ? fl::dtype::f64 : fl::dtype::f32;

  auto typed = (input.type() == computeType)
      ? input : input.astype(computeType);
  auto typedC = typed.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? typed : typed.asContiguousTensor();

  const auto& shape = input.shape();
  int axisSize = static_cast<int>(shape[axis]);

  std::vector<Dim> outDims;
  for (int d = 0; d < shape.ndim(); ++d) {
    if (d == static_cast<int>(axis)) {
      if (keepDims) {
        outDims.push_back(1);
      }
    } else {
      outDims.push_back(shape[d]);
    }
  }
  Shape outShape(outDims.empty() ? std::vector<Dim>{} : outDims);

  int outerSize = 1;
  for (int d = 0; d < static_cast<int>(axis); ++d) {
    outerSize *= static_cast<int>(shape[d]);
  }
  int innerSize = 1;
  for (int d = static_cast<int>(axis) + 1; d < shape.ndim(); ++d) {
    innerSize *= static_cast<int>(shape[d]);
  }

  int numOutputs = outerSize * innerSize;
  Shape flatShape({static_cast<Dim>(numOutputs)});
  auto valResult = toTensor<CudaTensor>(
      flatShape, computeType, nullptr, MemoryLocation::Host);
  auto idxResult = toTensor<CudaTensor>(
      flatShape, fl::dtype::u32, nullptr, MemoryLocation::Host);
  auto* outIdxPtr = static_cast<unsigned*>(
      idxResult.getAdapter<CudaTensor>().devicePtr());

  if (useF64) {
    const auto* inPtr = static_cast<const double*>(
        typedC.getAdapter<CudaTensor>().devicePtr());
    auto* outValPtr = static_cast<double*>(
        valResult.getAdapter<CudaTensor>().devicePtr());
    argReduceImpl<double>(
        stream, inPtr, outValPtr, outIdxPtr,
        outerSize, axisSize, innerSize, /*isMin=*/true);
  } else {
    const auto* inPtr = static_cast<const float*>(
        typedC.getAdapter<CudaTensor>().devicePtr());
    auto* outValPtr = static_cast<float*>(
        valResult.getAdapter<CudaTensor>().devicePtr());
    argReduceImpl<float>(
        stream, inPtr, outValPtr, outIdxPtr,
        outerSize, axisSize, innerSize, /*isMin=*/true);
  }
  FL_CUDA_CHECK(cudaGetLastError());

  values = fl::reshape(valResult, outShape);
  if (input.type() != computeType) {
    values = values.astype(input.type());
  }
  indices = fl::reshape(idxResult, outShape);
}

void max(
    cudaStream_t stream,
    Tensor& values, Tensor& indices, const Tensor& input,
    const unsigned axis, const bool keepDims) {
  bool useF64 = (input.type() == fl::dtype::f64);
  fl::dtype computeType = useF64 ? fl::dtype::f64 : fl::dtype::f32;

  auto typed = (input.type() == computeType)
      ? input : input.astype(computeType);
  auto typedC = typed.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? typed : typed.asContiguousTensor();

  const auto& shape = input.shape();
  int axisSize = static_cast<int>(shape[axis]);

  std::vector<Dim> outDims;
  for (int d = 0; d < shape.ndim(); ++d) {
    if (d == static_cast<int>(axis)) {
      if (keepDims) {
        outDims.push_back(1);
      }
    } else {
      outDims.push_back(shape[d]);
    }
  }
  Shape outShape(outDims.empty() ? std::vector<Dim>{} : outDims);

  int outerSize = 1;
  for (int d = 0; d < static_cast<int>(axis); ++d) {
    outerSize *= static_cast<int>(shape[d]);
  }
  int innerSize = 1;
  for (int d = static_cast<int>(axis) + 1; d < shape.ndim(); ++d) {
    innerSize *= static_cast<int>(shape[d]);
  }

  int numOutputs = outerSize * innerSize;
  Shape flatShape({static_cast<Dim>(numOutputs)});
  auto valResult = toTensor<CudaTensor>(
      flatShape, computeType, nullptr, MemoryLocation::Host);
  auto idxResult = toTensor<CudaTensor>(
      flatShape, fl::dtype::u32, nullptr, MemoryLocation::Host);
  auto* outIdxPtr = static_cast<unsigned*>(
      idxResult.getAdapter<CudaTensor>().devicePtr());

  if (useF64) {
    const auto* inPtr = static_cast<const double*>(
        typedC.getAdapter<CudaTensor>().devicePtr());
    auto* outValPtr = static_cast<double*>(
        valResult.getAdapter<CudaTensor>().devicePtr());
    argReduceImpl<double>(
        stream, inPtr, outValPtr, outIdxPtr,
        outerSize, axisSize, innerSize, /*isMin=*/false);
  } else {
    const auto* inPtr = static_cast<const float*>(
        typedC.getAdapter<CudaTensor>().devicePtr());
    auto* outValPtr = static_cast<float*>(
        valResult.getAdapter<CudaTensor>().devicePtr());
    argReduceImpl<float>(
        stream, inPtr, outValPtr, outIdxPtr,
        outerSize, axisSize, innerSize, /*isMin=*/false);
  }
  FL_CUDA_CHECK(cudaGetLastError());

  values = fl::reshape(valResult, outShape);
  if (input.type() != computeType) {
    values = values.astype(input.type());
  }
  indices = fl::reshape(idxResult, outShape);
}

void argmax(
    cudaStream_t stream, Tensor& out,
    const Tensor& input, const unsigned axis, const bool keepDims) {
  Tensor values, indices;
  max(stream, values, indices, input, axis, keepDims);
  out = indices;
}

void argmin(
    cudaStream_t stream, Tensor& out,
    const Tensor& input, const unsigned axis, const bool keepDims) {
  Tensor values, indices;
  min(stream, values, indices, input, axis, keepDims);
  out = indices;
}

} // namespace fl::cuda
