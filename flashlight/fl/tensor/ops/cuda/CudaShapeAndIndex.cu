/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include "flashlight/fl/tensor/backend/cuda/CudaTensor.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>

#include "flashlight/fl/runtime/CUDAUtils.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl::cuda {

namespace {

constexpr int kMaxDims = 8;
constexpr int kBlockSize = 256;

// POD struct for passing shape/stride info to device lambdas and kernels.
struct ShapeInfo {
  int dims[kMaxDims];
  int strides[kMaxDims];
  int ndim;
};

// POD wrapper for passing fixed-size int arrays by value to kernels.
// (C-style arrays decay to pointers, which can't be passed to device code.)
struct IntArray {
  int data[kMaxDims];
};

ShapeInfo makeShapeInfo(const Shape& shape) {
  ShapeInfo info{};
  info.ndim = shape.ndim();
  int s = 1;
  for (int d = info.ndim - 1; d >= 0; --d) {
    info.dims[d] = static_cast<int>(shape[d]);
    info.strides[d] = s;
    s *= info.dims[d];
  }
  return info;
}

ShapeInfo makeShapeInfo(const std::vector<Dim>& dims) {
  ShapeInfo info{};
  info.ndim = static_cast<int>(dims.size());
  int s = 1;
  for (int d = info.ndim - 1; d >= 0; --d) {
    info.dims[d] = static_cast<int>(dims[d]);
    info.strides[d] = s;
    s *= info.dims[d];
  }
  return info;
}

bool isF64(const Tensor& t) {
  return t.type() == fl::dtype::f64;
}

} // namespace

void reshape(cudaStream_t /* stream */, Tensor& out, const Tensor& tensor, const Shape& shape) {
  if (tensor.elements() != shape.elements()) {
    throw std::invalid_argument(
        "cuda::reshape: cannot reshape tensor of "
        + std::to_string(tensor.elements()) + " elements to shape "
        + shape.toString());
  }
  // Reshape is a metadata-only operation on contiguous tensors.
  auto contig = tensor.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? tensor : tensor.asContiguousTensor();
  out = contig.getAdapter<CudaTensor>().reshapeView(shape);
}

// GPU-native transpose kernel.
// Each output element reads from the permuted input position.
template <typename T>
__global__ void transposeKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int n,
    ShapeInfo inInfo,
    ShapeInfo outInfo,
    IntArray perm) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }

  // Decompose output index into output coordinates
  int rem = tid;
  int inIdx = 0;
  for (int d = 0; d < outInfo.ndim; ++d) {
    int coord = rem / outInfo.strides[d];
    rem %= outInfo.strides[d];
    // Output dim d corresponds to input dim perm[d]
    inIdx += coord * inInfo.strides[perm.data[d]];
  }
  output[tid] = input[inIdx];
}

template <typename T>
void transposeImpl(cudaStream_t stream, Tensor& out, const Tensor& tensor,
    const Shape& outShape, const Shape& shape,
    const std::vector<Dim>& outDims, IntArray perm) {
  fl::dtype computeType = isF64(tensor) ? fl::dtype::f64 : fl::dtype::f32;
  auto cast = (tensor.type() == computeType)
      ? tensor : tensor.astype(computeType);
  auto castC = cast.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? cast : cast.asContiguousTensor();
  const auto* inPtr = static_cast<const T*>(
      castC.getAdapter<CudaTensor>().devicePtr());

  auto result = toTensor<CudaTensor>(
      outShape, computeType, nullptr, MemoryLocation::Host);
  auto* outPtr = static_cast<T*>(
      result.getAdapter<CudaTensor>().devicePtr());

  int n = static_cast<int>(tensor.elements());
  auto inInfo = makeShapeInfo(shape);
  auto outInfo = makeShapeInfo(outDims);

  int grid = (n + kBlockSize - 1) / kBlockSize;
  transposeKernel<T><<<grid, kBlockSize, 0, stream>>>(
      inPtr, outPtr, n, inInfo, outInfo, perm);
  FL_CUDA_CHECK(cudaGetLastError());

  if (tensor.type() != computeType) {
    out = result.astype(tensor.type());
  } else {
    out = result;
  }
}

void transpose(cudaStream_t stream, Tensor& out, const Tensor& tensor, const Shape& axes) {
  if (tensor.ndim() < 2) {
    out = tensor.copy();
    return;
  }

  const auto& shape = tensor.shape();
  int ndim = shape.ndim();

  // Default axes: reverse all dims
  IntArray perm{};
  if (axes.ndim() == 0) {
    for (int i = 0; i < ndim; ++i) {
      perm.data[i] = ndim - 1 - i;
    }
  } else {
    if (axes.ndim() != ndim) {
      throw std::invalid_argument(
          "cuda::transpose: axes must have same ndim as tensor");
    }
    for (int i = 0; i < ndim; ++i) {
      perm.data[i] = static_cast<int>(axes[i]);
    }
  }

  // Compute output shape
  std::vector<Dim> outDims(ndim);
  for (int i = 0; i < ndim; ++i) {
    outDims[i] = shape[perm.data[i]];
  }
  Shape outShape(outDims);

  if (isF64(tensor)) {
    transposeImpl<double>(stream, out, tensor, outShape, shape, outDims, perm);
  } else {
    transposeImpl<float>(stream, out, tensor, outShape, shape, outDims, perm);
  }
}

// GPU-native tile kernel.
// Each output element reads from input at (coord % inputDim) per dimension.
template <typename T>
__global__ void tileKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int n,
    ShapeInfo inInfo,
    ShapeInfo outInfo) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }

  // Decompose output index into output coordinates
  int rem = tid;
  int inIdx = 0;
  for (int d = 0; d < outInfo.ndim; ++d) {
    int coord = rem / outInfo.strides[d];
    rem %= outInfo.strides[d];
    int inCoord = coord % inInfo.dims[d];
    inIdx += inCoord * inInfo.strides[d];
  }
  output[tid] = input[inIdx];
}

template <typename T>
void tileImpl(cudaStream_t stream, Tensor& out, const Tensor& tensor,
    const Shape& outShape, const std::vector<Dim>& effectiveInDims,
    const std::vector<Dim>& outDims) {
  fl::dtype computeType = isF64(tensor) ? fl::dtype::f64 : fl::dtype::f32;
  auto cast = (tensor.type() == computeType)
      ? tensor : tensor.astype(computeType);
  auto castC = cast.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? cast : cast.asContiguousTensor();
  const auto* inPtr = static_cast<const T*>(
      castC.getAdapter<CudaTensor>().devicePtr());

  auto result = toTensor<CudaTensor>(
      outShape, computeType, nullptr, MemoryLocation::Host);
  auto* outPtr = static_cast<T*>(
      result.getAdapter<CudaTensor>().devicePtr());

  int n = static_cast<int>(outShape.elements());
  auto inInfo = makeShapeInfo(effectiveInDims);
  auto outInfo = makeShapeInfo(outDims);

  int grid = (n + kBlockSize - 1) / kBlockSize;
  tileKernel<T><<<grid, kBlockSize, 0, stream>>>(
      inPtr, outPtr, n, inInfo, outInfo);
  FL_CUDA_CHECK(cudaGetLastError());

  if (tensor.type() != computeType) {
    out = result.astype(tensor.type());
  } else {
    out = result;
  }
}

void tile(cudaStream_t stream, Tensor& out, const Tensor& tensor, const Shape& tileDims) {
  const auto& shape = tensor.shape();
  int ndim = std::max(shape.ndim(), tileDims.ndim());

  // Input dims: right-aligned (prepend 1s). Tile dims: left-aligned (append 1s).
  // This matches ArrayFire convention where tile factors apply starting from dim 0.
  std::vector<Dim> outDims(ndim);
  std::vector<Dim> effectiveInDims(ndim, 1);
  int inOffset = ndim - shape.ndim();   // right-align input
  for (int d = 0; d < ndim; ++d) {
    Dim inDim = (d >= inOffset) ? shape[d - inOffset] : 1;
    Dim tileFactor = d < tileDims.ndim() ? tileDims[d] : 1;  // left-align tiles
    effectiveInDims[d] = inDim;
    outDims[d] = inDim * tileFactor;
  }
  Shape outShape(outDims);

  if (isF64(tensor)) {
    tileImpl<double>(stream, out, tensor, outShape, effectiveInDims, outDims);
  } else {
    tileImpl<float>(stream, out, tensor, outShape, effectiveInDims, outDims);
  }
}

// GPU-native concatenation via device-to-device copies.
// Each input tensor's slabs are copied into the correct output positions.
void concatenate(cudaStream_t stream, Tensor& out,
    const std::vector<Tensor>& tensors, const unsigned axis) {
  if (tensors.empty()) {
    throw std::invalid_argument("cuda::concatenate: empty tensor list");
  }
  if (tensors.size() == 1) {
    out = tensors[0].copy();
    return;
  }

  // Validate: all tensors same ndim, same shape except along axis
  int ndim = tensors[0].ndim();
  Dim totalAlongAxis = 0;
  for (const auto& t : tensors) {
    if (t.ndim() != ndim) {
      throw std::invalid_argument(
          "cuda::concatenate: tensors must have same ndim");
    }
    for (int d = 0; d < ndim; ++d) {
      if (d != static_cast<int>(axis) && t.dim(d) != tensors[0].dim(d)) {
        throw std::invalid_argument(
            "cuda::concatenate: shape mismatch at dim "
            + std::to_string(d));
      }
    }
    totalAlongAxis += t.dim(axis);
  }

  // Output shape
  std::vector<Dim> outDims;
  for (int d = 0; d < ndim; ++d) {
    outDims.push_back(
        d == static_cast<int>(axis) ? totalAlongAxis : tensors[0].dim(d));
  }
  Shape outShape(outDims);

  // Determine compute type: use f64 if any tensor is f64
  bool useF64 = false;
  for (const auto& t : tensors) {
    if (isF64(t)) { useF64 = true; break; }
  }
  fl::dtype computeType = useF64 ? fl::dtype::f64 : fl::dtype::f32;
  size_t elemSize = useF64 ? sizeof(double) : sizeof(float);

  auto result = toTensor<CudaTensor>(
      outShape, computeType, nullptr, MemoryLocation::Host);
  auto* outBasePtr = static_cast<char*>(
      result.getAdapter<CudaTensor>().devicePtr());

  // innerStride = product of dims after axis
  Dim innerStride = 1;
  for (int d = ndim - 1; d > static_cast<int>(axis); --d) {
    innerStride *= tensors[0].dim(d);
  }
  // outerCount = product of dims before axis
  Dim outerCount = 1;
  for (int d = 0; d < static_cast<int>(axis); ++d) {
    outerCount *= tensors[0].dim(d);
  }

  Dim offset = 0;
  for (const auto& t : tensors) {
    auto cast = (t.type() == computeType)
        ? t : t.astype(computeType);
    auto castC = cast.getAdapter<CudaTensor>().isContiguousRowMajor()
        ? cast : cast.asContiguousTensor();
    const auto* inPtr = static_cast<const char*>(
        castC.getAdapter<CudaTensor>().devicePtr());

    Dim tDimSize = t.dim(axis);
    // Copy each outer slab via cudaMemcpy
    for (Dim outer = 0; outer < outerCount; ++outer) {
      const char* src = inPtr + outer * (tDimSize * innerStride) * elemSize;
      char* dst = outBasePtr
          + (outer * (totalAlongAxis * innerStride)
             + offset * innerStride) * elemSize;
      size_t bytes = tDimSize * innerStride * elemSize;
      FL_CUDA_CHECK(cudaMemcpyAsync(
          dst, src, bytes, cudaMemcpyDeviceToDevice, stream));
    }
    offset += tDimSize;
  }

  if (tensors[0].type() != computeType) {
    out = result.astype(tensors[0].type());
  } else {
    out = result;
  }
}

// GPU-native nonzero via Thrust copy_if.
void nonzero(cudaStream_t stream, Tensor& out, const Tensor& tensor) {
  auto asF32 = (tensor.type() == fl::dtype::f32)
      ? tensor : tensor.astype(fl::dtype::f32);
  auto asF32C = asF32.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? asF32 : asF32.asContiguousTensor();
  const auto* inPtr = static_cast<const float*>(
      asF32C.getAdapter<CudaTensor>().devicePtr());
  size_t n = tensor.elements();

  auto policy = thrust::cuda::par.on(stream);
  auto dptr = thrust::device_pointer_cast(inPtr);

  // Count nonzero elements first
  size_t count = thrust::count_if(policy, dptr, dptr + n,
      [] __device__ (float x) { return x != 0.0f; });

  if (count == 0) {
    out = Tensor(Shape{0}, fl::dtype::s32);
    return;
  }

  // Allocate output indices on device
  auto outTensor = toTensor<CudaTensor>(
      Shape{static_cast<Dim>(count)}, fl::dtype::s32,
      nullptr, MemoryLocation::Host);
  auto* outPtr = static_cast<int*>(
      outTensor.getAdapter<CudaTensor>().devicePtr());
  auto outDptr = thrust::device_pointer_cast(outPtr);

  // Copy indices where input is nonzero
  thrust::counting_iterator<int> countIter(0);
  thrust::copy_if(
      policy, countIter, countIter + static_cast<int>(n), dptr, outDptr,
      [] __device__ (float x) { return x != 0.0f; });

  out = outTensor;
}

// GPU-native pad kernel.
// Each output element checks if its coordinates are within the unpadded region
// and reads from input, or outputs zero.
// padType: 0=Constant, 1=Edge, 2=Symmetric
template <typename T>
__global__ void padKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int n,
    ShapeInfo inInfo,
    ShapeInfo outInfo,
    IntArray padBefore,
    int padType) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }

  // Decompose output index into output coordinates
  int rem = tid;
  int inIdx = 0;
  bool inBounds = true;
  for (int d = 0; d < outInfo.ndim; ++d) {
    int outCoord = rem / outInfo.strides[d];
    rem %= outInfo.strides[d];
    int inCoord = outCoord - padBefore.data[d];
    int dimSize = inInfo.dims[d];

    if (inCoord < 0 || inCoord >= dimSize) {
      if (padType == 0) {
        // Constant: zero-fill
        inBounds = false;
        break;
      } else if (padType == 1) {
        // Edge: clamp to boundary
        if (inCoord < 0) inCoord = 0;
        if (inCoord >= dimSize) inCoord = dimSize - 1;
      } else {
        // Symmetric: reflect
        if (inCoord < 0) {
          inCoord = -inCoord - 1;
        }
        if (inCoord >= dimSize) {
          inCoord = 2 * dimSize - inCoord - 1;
        }
        if (inCoord < 0) inCoord = 0;
        if (inCoord >= dimSize) inCoord = dimSize - 1;
      }
    }
    inIdx += inCoord * inInfo.strides[d];
  }
  output[tid] = inBounds ? input[inIdx] : T(0);
}

template <typename T>
void padImpl(cudaStream_t stream, Tensor& out,
    const Tensor& input, const Shape& outShape,
    const ShapeInfo& inInfo, const ShapeInfo& outInfo,
    const IntArray& padBefore, int padTypeInt) {
  fl::dtype computeType = isF64(input) ? fl::dtype::f64 : fl::dtype::f32;
  auto cast = (input.type() == computeType)
      ? input : input.astype(computeType);
  auto castC = cast.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? cast : cast.asContiguousTensor();
  const auto* inPtr = static_cast<const T*>(
      castC.getAdapter<CudaTensor>().devicePtr());

  auto result = toTensor<CudaTensor>(
      outShape, computeType, nullptr, MemoryLocation::Host);
  auto* outPtr = static_cast<T*>(
      result.getAdapter<CudaTensor>().devicePtr());

  int n = static_cast<int>(outShape.elements());
  int grid = (n + kBlockSize - 1) / kBlockSize;
  padKernel<T><<<grid, kBlockSize, 0, stream>>>(
      inPtr, outPtr, n, inInfo, outInfo, padBefore, padTypeInt);
  FL_CUDA_CHECK(cudaGetLastError());

  if (input.type() != computeType) {
    out = result.astype(input.type());
  } else {
    out = result;
  }
}

void pad(cudaStream_t stream, Tensor& out,
    const Tensor& input,
    const std::vector<std::pair<int, int>>& padWidths,
    const PadType type) {
  const auto& shape = input.shape();
  int ndim = shape.ndim();
  if (static_cast<int>(padWidths.size()) != ndim) {
    throw std::invalid_argument(
        "cuda::pad: padWidths size must match tensor ndim");
  }

  // Compute output shape
  std::vector<Dim> outDims(ndim);
  IntArray padBefore{};
  for (int d = 0; d < ndim; ++d) {
    outDims[d] = shape[d] + padWidths[d].first + padWidths[d].second;
    padBefore.data[d] = padWidths[d].first;
  }
  Shape outShape(outDims);

  auto inInfo = makeShapeInfo(shape);
  auto outInfo = makeShapeInfo(outDims);

  int padTypeInt = (type == PadType::Constant) ? 0
                 : (type == PadType::Edge) ? 1 : 2;

  if (isF64(input)) {
    padImpl<double>(stream, out, input, outShape, inInfo, outInfo, padBefore, padTypeInt);
  } else {
    padImpl<float>(stream, out, input, outShape, inInfo, outInfo, padBefore, padTypeInt);
  }
}

} // namespace fl::cuda
