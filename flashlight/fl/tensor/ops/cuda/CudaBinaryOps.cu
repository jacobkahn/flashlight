/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include "flashlight/fl/tensor/backend/cuda/CudaTensor.h"

#include <thrust/device_ptr.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl::cuda {

namespace {

// Fixed-size dim info passable to device lambdas
constexpr int kMaxBcastDims = 8;

struct BroadcastInfo {
  int ndim;
  long long outShape[kMaxBcastDims];
  long long outStrides[kMaxBcastDims];
  long long lhsStrides[kMaxBcastDims]; // 0 if broadcast
  long long rhsStrides[kMaxBcastDims]; // 0 if broadcast
};

// Compute row-major (C-order) strides
inline void computeStrides(
    const Shape& shape, long long* strides, int ndim) {
  long long stride = 1;
  for (int d = ndim - 1; d >= 0; --d) {
    strides[d] = stride;
    stride *= shape[d];
  }
}

BroadcastInfo makeBroadcastInfo(
    const Shape& lhs, const Shape& rhs, const Shape& out) {
  BroadcastInfo info{};
  info.ndim = out.ndim();
  if (info.ndim > kMaxBcastDims) {
    throw std::runtime_error(
        "CudaBinaryOps: tensor ndim exceeds maximum supported ("
        + std::to_string(kMaxBcastDims) + ")");
  }

  // Output strides
  computeStrides(out, info.outStrides, info.ndim);
  for (int i = 0; i < info.ndim; ++i) {
    info.outShape[i] = out[i];
  }

  // Compute input strides, zeroing broadcast dims.
  // Right-align: input dim offset = ndim - input.ndim()
  int lOff = info.ndim - lhs.ndim();
  int rOff = info.ndim - rhs.ndim();

  // First compute natural strides for each input
  long long lNatural[kMaxBcastDims] = {};
  long long rNatural[kMaxBcastDims] = {};
  {
    long long s = 1;
    for (int d = lhs.ndim() - 1; d >= 0; --d) {
      lNatural[d] = s;
      s *= lhs[d];
    }
  }
  {
    long long s = 1;
    for (int d = rhs.ndim() - 1; d >= 0; --d) {
      rNatural[d] = s;
      s *= rhs[d];
    }
  }

  for (int i = 0; i < info.ndim; ++i) {
    int li = i - lOff;
    int ri = i - rOff;
    Dim ld = (li >= 0 && li < lhs.ndim()) ? lhs[li] : 1;
    Dim rd = (ri >= 0 && ri < rhs.ndim()) ? rhs[ri] : 1;
    info.lhsStrides[i] = (ld == 1 && info.outShape[i] > 1)
        ? 0 : (li >= 0 ? lNatural[li] : 0);
    info.rhsStrides[i] = (rd == 1 && info.outShape[i] > 1)
        ? 0 : (ri >= 0 ? rNatural[ri] : 0);
  }
  return info;
}

// Binary transform with broadcasting -- f32 compute path.
// When out is non-empty, writes into its buffer instead of allocating.
template <typename BinaryOp>
void binaryTransformF32(
    cudaStream_t stream,
    Tensor& out,
    const Tensor& lhs,
    const Tensor& rhs,
    fl::dtype outType,
    BinaryOp op) {
  auto outShape = broadcastShapes(lhs.shape(), rhs.shape());
  size_t n = outShape.elements();
  if (n == 0) {
    out = toTensor<CudaTensor>(outShape, outType, nullptr, MemoryLocation::Host);
    return;
  }

  auto info = makeBroadcastInfo(lhs.shape(), rhs.shape(), outShape);

  auto lhsF32 = (lhs.type() == fl::dtype::f32)
      ? lhs : lhs.astype(fl::dtype::f32);
  auto rhsF32 = (rhs.type() == fl::dtype::f32)
      ? rhs : rhs.astype(fl::dtype::f32);
  auto lhsC = lhsF32.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? lhsF32 : lhsF32.asContiguousTensor();
  auto rhsC = rhsF32.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? rhsF32 : rhsF32.asContiguousTensor();

  // Only reuse out when output type matches compute type (f32),
  // otherwise the buffer has the wrong element type for f32 writes.
  Tensor result = (!out.isEmpty() && outType == fl::dtype::f32)
      ? out
      : toTensor<CudaTensor>(outShape, fl::dtype::f32, nullptr, MemoryLocation::Host);
  auto& outAdapter = result.getAdapter<CudaTensor>();

  auto lPtr = thrust::device_pointer_cast(
      static_cast<const float*>(lhsC.getAdapter<CudaTensor>().devicePtr()));
  auto rPtr = thrust::device_pointer_cast(
      static_cast<const float*>(rhsC.getAdapter<CudaTensor>().devicePtr()));
  auto oPtr = thrust::device_pointer_cast(
      static_cast<float*>(outAdapter.devicePtr()));

  bool sameshape = (lhs.shape() == rhs.shape());
  if (sameshape) {
    thrust::transform(
        thrust::cuda::par.on(stream), lPtr, lPtr + n, rPtr, oPtr, op);
  } else {
    thrust::tabulate(
        thrust::cuda::par.on(stream), oPtr, oPtr + n,
        [lPtr, rPtr, info, op] __device__ (int idx) -> float {
          int lIdx = 0, rIdx = 0;
          int rem = idx;
          for (int d = 0; d < info.ndim; ++d) {
            int dimIdx = rem / info.outStrides[d];
            rem %= info.outStrides[d];
            lIdx += dimIdx * info.lhsStrides[d];
            rIdx += dimIdx * info.rhsStrides[d];
          }
          return op(lPtr[lIdx], rPtr[rIdx]);
        });
  }
  out = (outType != fl::dtype::f32) ? result.astype(outType) : result;
}

// Binary transform with broadcasting -- f64 compute path (native double).
// When out is non-empty, writes into its buffer instead of allocating.
template <typename BinaryOp>
void binaryTransformF64(
    cudaStream_t stream,
    Tensor& out,
    const Tensor& lhs,
    const Tensor& rhs,
    fl::dtype outType,
    BinaryOp op) {
  auto outShape = broadcastShapes(lhs.shape(), rhs.shape());
  size_t n = outShape.elements();
  if (n == 0) {
    out = toTensor<CudaTensor>(outShape, outType, nullptr, MemoryLocation::Host);
    return;
  }

  auto info = makeBroadcastInfo(lhs.shape(), rhs.shape(), outShape);

  auto lhsF64 = (lhs.type() == fl::dtype::f64)
      ? lhs : lhs.astype(fl::dtype::f64);
  auto rhsF64 = (rhs.type() == fl::dtype::f64)
      ? rhs : rhs.astype(fl::dtype::f64);
  auto lhsC = lhsF64.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? lhsF64 : lhsF64.asContiguousTensor();
  auto rhsC = rhsF64.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? rhsF64 : rhsF64.asContiguousTensor();

  Tensor result = (!out.isEmpty() && outType == fl::dtype::f64)
      ? out
      : toTensor<CudaTensor>(outShape, fl::dtype::f64, nullptr, MemoryLocation::Host);
  auto& outAdapter = result.getAdapter<CudaTensor>();

  auto lPtr = thrust::device_pointer_cast(
      static_cast<const double*>(lhsC.getAdapter<CudaTensor>().devicePtr()));
  auto rPtr = thrust::device_pointer_cast(
      static_cast<const double*>(rhsC.getAdapter<CudaTensor>().devicePtr()));
  auto oPtr = thrust::device_pointer_cast(
      static_cast<double*>(outAdapter.devicePtr()));

  bool sameshape = (lhs.shape() == rhs.shape());
  if (sameshape) {
    thrust::transform(
        thrust::cuda::par.on(stream), lPtr, lPtr + n, rPtr, oPtr, op);
  } else {
    thrust::tabulate(
        thrust::cuda::par.on(stream), oPtr, oPtr + n,
        [lPtr, rPtr, info, op] __device__ (int idx) -> double {
          int lIdx = 0, rIdx = 0;
          int rem = idx;
          for (int d = 0; d < info.ndim; ++d) {
            int dimIdx = rem / info.outStrides[d];
            rem %= info.outStrides[d];
            lIdx += dimIdx * info.lhsStrides[d];
            rIdx += dimIdx * info.rhsStrides[d];
          }
          return op(lPtr[lIdx], rPtr[rIdx]);
        });
  }
  out = (outType != fl::dtype::f64) ? result.astype(outType) : result;
}

// Check if either operand is f64
bool hasF64(const Tensor& lhs, const Tensor& rhs) {
  return lhs.type() == fl::dtype::f64 || rhs.type() == fl::dtype::f64;
}

} // namespace

// Arithmetic -- dispatch to f64 when either operand is f64. The output dtype
// follows the same promotion rule as the IR (promoteDtypes), so the result
// matches its node's metadata (e.g. bf16 + f16 -> f32).
#define FL_CUDA_BINARY_OP(NAME, F32_EXPR, F64_EXPR)                           \
  void NAME(cudaStream_t stream, Tensor& out,                                 \
            const Tensor& lhs, const Tensor& rhs) {                           \
    if (hasF64(lhs, rhs)) {                                                    \
      binaryTransformF64(stream, out, lhs, rhs, fl::dtype::f64,               \
          [] __device__ (double a, double b) -> double { return F64_EXPR; });   \
    } else {                                                                   \
      binaryTransformF32(stream, out, lhs, rhs,                               \
          fl::promoteDtypes(lhs.type(), rhs.type()),                           \
          [] __device__ (float a, float b) -> float { return F32_EXPR; });     \
    }                                                                          \
  }

FL_CUDA_BINARY_OP(add, a + b, a + b)
FL_CUDA_BINARY_OP(sub, a - b, a - b)
FL_CUDA_BINARY_OP(mul, a * b, a * b)
FL_CUDA_BINARY_OP(div, a / b, a / b)
FL_CUDA_BINARY_OP(minimum, ::fminf(a, b), ::fmin(a, b))
FL_CUDA_BINARY_OP(maximum, ::fmaxf(a, b), ::fmax(a, b))
FL_CUDA_BINARY_OP(power, ::powf(a, b), ::pow(a, b))
FL_CUDA_BINARY_OP(mod, ::fmodf(a, b), ::fmod(a, b))

#undef FL_CUDA_BINARY_OP

// Comparison -- output is always b8, f32 compute is sufficient
#define FL_CUDA_CMP_OP(NAME, EXPR)                                             \
  void NAME(cudaStream_t stream, Tensor& out,                                  \
            const Tensor& lhs, const Tensor& rhs) {                            \
    binaryTransformF32(stream, out, lhs, rhs, fl::dtype::b8,                   \
        [] __device__ (float a, float b) -> float { return EXPR; });            \
  }

FL_CUDA_CMP_OP(eq, a == b ? 1.0f : 0.0f)
FL_CUDA_CMP_OP(neq, a != b ? 1.0f : 0.0f)
FL_CUDA_CMP_OP(lessThan, a < b ? 1.0f : 0.0f)
FL_CUDA_CMP_OP(lessThanEqual, a <= b ? 1.0f : 0.0f)
FL_CUDA_CMP_OP(greaterThan, a > b ? 1.0f : 0.0f)
FL_CUDA_CMP_OP(greaterThanEqual, a >= b ? 1.0f : 0.0f)
FL_CUDA_CMP_OP(logicalOr, (a != 0 || b != 0) ? 1.0f : 0.0f)
FL_CUDA_CMP_OP(logicalAnd, (a != 0 && b != 0) ? 1.0f : 0.0f)

#undef FL_CUDA_CMP_OP

// Bitwise -- always integer operations, f32 path is fine. Output dtype is
// promoted like the IR does.
void bitwiseAnd(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs) {
  binaryTransformF32(stream, out, lhs, rhs,
      fl::promoteDtypes(lhs.type(), rhs.type()),
      [] __device__ (float a, float b) -> float {
        return static_cast<float>(
            static_cast<int>(a) & static_cast<int>(b));
      });
}

void bitwiseOr(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs) {
  binaryTransformF32(stream, out, lhs, rhs,
      fl::promoteDtypes(lhs.type(), rhs.type()),
      [] __device__ (float a, float b) -> float {
        return static_cast<float>(
            static_cast<int>(a) | static_cast<int>(b));
      });
}

void bitwiseXor(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs) {
  binaryTransformF32(stream, out, lhs, rhs,
      fl::promoteDtypes(lhs.type(), rhs.type()),
      [] __device__ (float a, float b) -> float {
        return static_cast<float>(
            static_cast<int>(a) ^ static_cast<int>(b));
      });
}

void lShift(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs) {
  binaryTransformF32(stream, out, lhs, rhs,
      fl::promoteDtypes(lhs.type(), rhs.type()),
      [] __device__ (float a, float b) -> float {
        return static_cast<float>(
            static_cast<int>(a) << static_cast<int>(b));
      });
}

void rShift(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs) {
  binaryTransformF32(stream, out, lhs, rhs,
      fl::promoteDtypes(lhs.type(), rhs.type()),
      [] __device__ (float a, float b) -> float {
        return static_cast<float>(
            static_cast<int>(a) >> static_cast<int>(b));
      });
}

} // namespace fl::cuda
