/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/cuda/CudaBackend.h"

#include "flashlight/fl/tensor/BackendDispatch-inl.h"
#include "flashlight/fl/tensor/BackendRegistry.h"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>

#include "flashlight/fl/runtime/CUDAUtils.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/backend/cuda/CudaTensor.h"
#include "flashlight/fl/tensor/ops/cuda/CudaNumerics.h"
#include "flashlight/fl/tensor/ops/cuda/CudaOps.h"

namespace fl {

FL_REGISTER_TENSOR_BACKEND(Cuda, CudaBackend::getInstance);

namespace {

// Helper: dispatch a fill operation by dtype
template <typename T>
void fillKernel(void* data, size_t n, T value, cudaStream_t stream) {
  auto ptr = thrust::device_pointer_cast(static_cast<T*>(data));
  thrust::fill(thrust::cuda::par.on(stream), ptr, ptr + n, value);
}

void fillByDtype(void* data, size_t n, double value, fl::dtype type, cudaStream_t stream) {
  switch (type) {
    case fl::dtype::f32: fillKernel(data, n, static_cast<float>(value), stream); break;
    case fl::dtype::f64: fillKernel(data, n, value, stream); break;
    case fl::dtype::s32: fillKernel(data, n, static_cast<int>(value), stream); break;
    case fl::dtype::s64: fillKernel(data, n, static_cast<long long>(value), stream); break;
    case fl::dtype::u32: fillKernel(data, n, static_cast<unsigned>(value), stream); break;
    case fl::dtype::u64: fillKernel(data, n, static_cast<unsigned long long>(value), stream); break;
    case fl::dtype::b8: fillKernel(data, n, static_cast<char>(value), stream); break;
    case fl::dtype::u8: fillKernel(data, n, static_cast<unsigned char>(value), stream); break;
    case fl::dtype::s16: fillKernel(data, n, static_cast<short>(value), stream); break;
    case fl::dtype::u16: fillKernel(data, n, static_cast<unsigned short>(value), stream); break;
    case fl::dtype::f16:
      fillKernel(data, n, __float2half(static_cast<float>(value)), stream); break;
    case fl::dtype::bf16:
      fillKernel(data, n, __float2bfloat16(static_cast<float>(value)), stream); break;
    default: throw std::invalid_argument("fillByDtype: unsupported dtype");
  }
}

Tensor makeCudaTensor(const Shape& shape, fl::dtype type, cudaStream_t stream,
                      const std::shared_ptr<CUDAStream>& streamPtr) {
  int devId = fl::cuda::getActiveDeviceId();
  size_t elemSize = 0;
  switch (type) {
    case fl::dtype::f32: elemSize = 4; break;
    case fl::dtype::f64: elemSize = 8; break;
    case fl::dtype::s32: elemSize = 4; break;
    case fl::dtype::s64: elemSize = 8; break;
    case fl::dtype::u32: elemSize = 4; break;
    case fl::dtype::u64: elemSize = 8; break;
    case fl::dtype::b8: elemSize = 1; break;
    case fl::dtype::u8: elemSize = 1; break;
    case fl::dtype::s16: elemSize = 2; break;
    case fl::dtype::u16: elemSize = 2; break;
    case fl::dtype::f16: elemSize = 2; break;
    case fl::dtype::bf16: elemSize = 2; break;
  }
  size_t bytes = shape.elements() * elemSize;
  auto buffer = std::make_shared<CudaBuffer>(bytes, devId, stream);
  auto strides = CudaTensor::computeStrides(shape);
  return toTensor<CudaTensor>(
      std::move(buffer), shape, strides, type, 0, streamPtr);
}

} // namespace

// --- CudaBackend singleton ---

// TODO: support per-op stream configuration rather than a single global stream.
// Tensors already store their computation stream (CudaTensor::stream_), but ops
// currently ignore it and use this backend-wide stream. We should infer the
// execution stream from input tensors and call cublasSetStream/curandSetStream
// per-op as needed. See MLX for design inspiration.
CudaBackend::CudaBackend() {
  stream_ = CUDAStream::createManaged();

  // Initialize cuBLAS
  cublasStatus_t cbStatus = cublasCreate(&cublasHandle_);
  if (cbStatus != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("CudaBackend: failed to create cuBLAS handle");
  }
  cublasSetStream(cublasHandle_, stream_->handle());
  // The default math mode keeps f32 GEMMs strict; TF32 only on opt-in.
  if (fl::cuda::detail::allowTF32()) {
    cublasSetMathMode(cublasHandle_, CUBLAS_TF32_TENSOR_OP_MATH);
  }

  // Initialize cuRAND
  curandStatus_t crStatus = curandCreateGenerator(
      &curandGen_, CURAND_RNG_PSEUDO_DEFAULT);
  if (crStatus != CURAND_STATUS_SUCCESS) {
    throw std::runtime_error("CudaBackend: failed to create cuRAND generator");
  }
  curandSetStream(curandGen_, stream_->handle());
  curandSetPseudoRandomGeneratorSeed(curandGen_, 0);

  // Initialize cuBLASLt
  cublasStatus_t ltStatus = cublasLtCreate(&cublasLtHandle_);
  if (ltStatus != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("CudaBackend: failed to create cuBLASLt handle");
  }
}

CudaBackend::~CudaBackend() {
  if (cublasHandle_) {
    cublasDestroy(cublasHandle_);
  }
  if (cublasLtHandle_) {
    cublasLtDestroy(cublasLtHandle_);
  }
  if (curandGen_) {
    curandDestroyGenerator(curandGen_);
  }
}

CudaBackend& CudaBackend::getInstance() {
  static CudaBackend instance;
  return instance;
}

cudaStream_t CudaBackend::cudaStream() const {
  return stream_->handle();
}

const std::shared_ptr<CUDAStream>& CudaBackend::cudaStreamPtr() const {
  return stream_;
}

cublasHandle_t CudaBackend::cublasHandle() const {
  return cublasHandle_;
}

cublasLtHandle_t CudaBackend::cublasLtHandle() const {
  return cublasLtHandle_;
}

TensorBackendType CudaBackend::backendType() const {
  return TensorBackendType::Cuda;
}

std::shared_ptr<Stream> CudaBackend::jitStream() const {
  return stream_;
}

BackendDispatch CudaBackend::createDispatch() {
  return makeDirectDispatch(*this);
}

void CudaBackend::eval(const Tensor& /*tensor*/) {
  FL_CUDA_CHECK(cudaStreamSynchronize(stream_->handle()));
}

bool CudaBackend::supportsDataType(const fl::dtype& dtype) const {
  switch (dtype) {
    case fl::dtype::f16: return true;
    case fl::dtype::f32: return true;
    case fl::dtype::f64: return true;
    case fl::dtype::b8: return true;
    case fl::dtype::s16: return true;
    case fl::dtype::s32: return true;
    case fl::dtype::s64: return true;
    case fl::dtype::u8: return true;
    case fl::dtype::u16: return true;
    case fl::dtype::u32: return true;
    case fl::dtype::u64: return true;
    case fl::dtype::bf16: return true;
  }
  return false;
}

// --- Memory management (stubs) ---
void CudaBackend::getMemMgrInfo(const char*, int, std::ostream*) {}
void CudaBackend::setMemMgrLogStream(std::ostream*) {}
void CudaBackend::setMemMgrLoggingEnabled(bool) {}
void CudaBackend::setMemMgrFlushInterval(size_t) {}

// --- Random ---
void CudaBackend::setSeed(const int seed) {
  curandSetPseudoRandomGeneratorSeed(curandGen_, seed);
}

Tensor CudaBackend::randn(const Shape& shape, dtype type,
    std::shared_ptr<const Stream> /* stream */) {
  auto t = makeCudaTensor(shape, fl::dtype::f32, cudaStream(), stream_);
  auto& ct = t.getAdapter<CudaTensor>();
  curandGenerateNormal(
      curandGen_, static_cast<float*>(ct.devicePtr()),
      // cuRAND requires even count
      (shape.elements() + 1) / 2 * 2,
      0.0f, 1.0f);
  if (type != fl::dtype::f32) {
    return t.astype(type);
  }
  return t;
}

Tensor CudaBackend::rand(const Shape& shape, dtype type,
    std::shared_ptr<const Stream> /* stream */) {
  auto t = makeCudaTensor(shape, fl::dtype::f32, cudaStream(), stream_);
  auto& ct = t.getAdapter<CudaTensor>();
  curandGenerateUniform(
      curandGen_, static_cast<float*>(ct.devicePtr()),
      shape.elements());
  if (type != fl::dtype::f32) {
    return t.astype(type);
  }
  return t;
}

// --- Creation: fromScalar and full ---

Tensor CudaBackend::fromScalar(double value, dtype type,
    std::shared_ptr<const Stream> /* stream */) {
  return full(Shape{}, value, type);
}

Tensor CudaBackend::full(const Shape& dims, double value, dtype type,
    std::shared_ptr<const Stream> /* stream */) {
  auto t = makeCudaTensor(dims, type, cudaStream(), stream_);
  if (dims.elements() > 0) {
    auto& ct = t.getAdapter<CudaTensor>();
    fillByDtype(ct.devicePtr(), dims.elements(), value, type, cudaStream());
  }
  return t;
}

Tensor CudaBackend::empty(const Shape& shape, dtype type,
    std::shared_ptr<const Stream> /* stream */) {
  // All CUDA ops currently run on the backend stream (BackendDispatch takes no
  // stream), so buffers are allocated and freed in that stream's order too.
  // Honoring a node's stream here would allocate on one stream while kernels
  // use another. Revisit when dispatch accepts per-node streams.
  return makeCudaTensor(shape, type, cudaStream(), stream_);
}

Tensor CudaBackend::identity(const Dim dim, const dtype type,
    std::shared_ptr<const Stream> /* stream */) {
  // Set diagonal to 1 on host (optimize later with a kernel)
  std::vector<float> host(dim * dim, 0);
  for (Dim i = 0; i < dim; ++i) host[i * dim + i] = 1;
  auto result = toTensor<CudaTensor>(
      Shape{dim, dim}, fl::dtype::f32, host.data(), Location::Host);
  if (type != fl::dtype::f32) {
    result = result.astype(type);
  }
  return result;
}

Tensor CudaBackend::arange(const Shape& shape, const Dim seqDim, const dtype type,
    std::shared_ptr<const Stream> /* stream */) {
  auto t = makeCudaTensor(shape, type, cudaStream(), stream_);
  auto& ct = t.getAdapter<CudaTensor>();
  size_t n = shape.elements();
  if (type == fl::dtype::f32) {
    auto ptr = thrust::device_pointer_cast(static_cast<float*>(ct.devicePtr()));
    // For arange, seqDim determines which dimension sequences along
    Dim seqLen = shape[seqDim];
    Dim innerStride = 1;
    for (int d = shape.ndim() - 1; d > seqDim; --d) {
      innerStride *= shape[d];
    }
    thrust::tabulate(
        thrust::cuda::par.on(cudaStream()), ptr, ptr + n,
        [seqLen, innerStride] __device__ (int idx) -> float {
          return static_cast<float>((idx / innerStride) % seqLen);
        });
  } else {
    // Generate as f32, then cast
    auto f32 = arange(shape, seqDim, fl::dtype::f32);
    return f32.astype(type);
  }
  return t;
}

Tensor CudaBackend::iota(
    const Shape& dims, const Shape& tileDims, const dtype type,
    std::shared_ptr<const Stream> /* stream */) {
  // Generate flat consecutive indices [0, 1, ..., N-1] then reshape
  auto flat = arange({dims.elements()}, 0, type);
  auto base = fl::reshape(flat, dims);
  if (tileDims.ndim() > 0) {
    Tensor result;
    tile(result, base, tileDims);
    return result;
  }
  return base;
}

// --- Enum-Dispatched Ops ---
// Dispatchers delegate to namespace fl::cuda free functions.

void CudaBackend::binary(Tensor& out, BinaryOp op, const Tensor& lhs, const Tensor& rhs,
    std::shared_ptr<const Stream> /* stream */) {
  auto s = cudaStream();
  switch (op) {
    case BinaryOp::Add: cuda::add(s, out, lhs, rhs); return;
    case BinaryOp::Sub: cuda::sub(s, out, lhs, rhs); return;
    case BinaryOp::Mul: cuda::mul(s, out, lhs, rhs); return;
    case BinaryOp::Div: cuda::div(s, out, lhs, rhs); return;
    case BinaryOp::Eq: cuda::eq(s, out, lhs, rhs); return;
    case BinaryOp::Neq: cuda::neq(s, out, lhs, rhs); return;
    case BinaryOp::Gt: cuda::greaterThan(s, out, lhs, rhs); return;
    case BinaryOp::Gte: cuda::greaterThanEqual(s, out, lhs, rhs); return;
    case BinaryOp::Lt: cuda::lessThan(s, out, lhs, rhs); return;
    case BinaryOp::Lte: cuda::lessThanEqual(s, out, lhs, rhs); return;
    case BinaryOp::Min: cuda::minimum(s, out, lhs, rhs); return;
    case BinaryOp::Max: cuda::maximum(s, out, lhs, rhs); return;
    case BinaryOp::Pow: cuda::power(s, out, lhs, rhs); return;
    case BinaryOp::Mod: cuda::mod(s, out, lhs, rhs); return;
    case BinaryOp::And: cuda::logicalAnd(s, out, lhs, rhs); return;
    case BinaryOp::Or: cuda::logicalOr(s, out, lhs, rhs); return;
    case BinaryOp::Shl: cuda::lShift(s, out, lhs, rhs); return;
    case BinaryOp::Shr: cuda::rShift(s, out, lhs, rhs); return;
    case BinaryOp::BitAnd: cuda::bitwiseAnd(s, out, lhs, rhs); return;
    case BinaryOp::BitOr: cuda::bitwiseOr(s, out, lhs, rhs); return;
    case BinaryOp::BitXor: cuda::bitwiseXor(s, out, lhs, rhs); return;
  }
  throw std::runtime_error("[CudaBackend::binary] Unknown binary operation type");
}

void CudaBackend::unary(Tensor& out, UnaryOp op, const Tensor& input,
    std::shared_ptr<const Stream> /* stream */) {
  auto s = cudaStream();
  switch (op) {
    case UnaryOp::Exp: cuda::exp(s, out, input); return;
    case UnaryOp::Log: cuda::log(s, out, input); return;
    case UnaryOp::Log1p: cuda::log1p(s, out, input); return;
    case UnaryOp::Sin: cuda::sin(s, out, input); return;
    case UnaryOp::Cos: cuda::cos(s, out, input); return;
    case UnaryOp::Sqrt: cuda::sqrt(s, out, input); return;
    case UnaryOp::Tanh: cuda::tanh(s, out, input); return;
    case UnaryOp::Floor: cuda::floor(s, out, input); return;
    case UnaryOp::Ceil: cuda::ceil(s, out, input); return;
    case UnaryOp::Rint: cuda::rint(s, out, input); return;
    case UnaryOp::Negative: cuda::negative(s, out, input); return;
    case UnaryOp::Absolute: cuda::absolute(s, out, input); return;
    case UnaryOp::Sigmoid: cuda::sigmoid(s, out, input); return;
    case UnaryOp::Erf: cuda::erf(s, out, input); return;
    case UnaryOp::LogicalNot: cuda::logicalNot(s, out, input); return;
    case UnaryOp::Sign: cuda::sign(s, out, input); return;
    case UnaryOp::Isnan: cuda::isnan(s, out, input); return;
    case UnaryOp::Isinf: cuda::isinf(s, out, input); return;
  }
  throw std::runtime_error("[CudaBackend::unary] Unknown unary operation type");
}

void CudaBackend::reduction(Tensor& out, ReductionOp op, const Tensor& input,
    const std::vector<int>& axes, bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  auto s = cudaStream();
  switch (op) {
    case ReductionOp::Amin: cuda::amin(s, out, input, axes, keepDims); return;
    case ReductionOp::Amax: cuda::amax(s, out, input, axes, keepDims); return;
    case ReductionOp::Sum: cuda::sum(s, out, input, axes, keepDims); return;
    case ReductionOp::Mean: cuda::mean(s, out, input, axes, keepDims); return;
    case ReductionOp::Median: cuda::median(s, out, input, axes, keepDims); return;
    case ReductionOp::Std: cuda::std(s, out, input, axes, keepDims); return;
    case ReductionOp::CountNonzero: cuda::countNonzero(s, out, input, axes, keepDims); return;
    case ReductionOp::Any: cuda::any(s, out, input, axes, keepDims); return;
    case ReductionOp::All: cuda::all(s, out, input, axes, keepDims); return;
  }
  throw std::runtime_error("[CudaBackend::reduction] Unknown reduction operation type");
}

// --- Virtual method overrides that delegate to cuda:: free functions ---

void CudaBackend::flip(Tensor& out, const Tensor& tensor, const unsigned dim,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::flip(cudaStream(), out, tensor, dim);
}

void CudaBackend::clip(Tensor& out,
    const Tensor& tensor, const Tensor& low, const Tensor& high,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::clip(cudaStream(), out, tensor, low, high);
}

void CudaBackend::roll(Tensor& out,
    const Tensor& tensor, const int shift, const unsigned axis,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::roll(cudaStream(), out, tensor, shift, axis);
}

void CudaBackend::tril(Tensor& out, const Tensor& tensor, int diagonal,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::tril(cudaStream(), out, tensor, diagonal);
}

void CudaBackend::triu(Tensor& out, const Tensor& tensor, int diagonal,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::triu(cudaStream(), out, tensor, diagonal);
}

void CudaBackend::where(Tensor& out,
    const Tensor& condition, const Tensor& x, const Tensor& y,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::where(cudaStream(), out, condition, x, y);
}

void CudaBackend::topk(
    Tensor& values, Tensor& indices, const Tensor& input,
    const unsigned k, const Dim axis, const SortMode sortMode,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::topk(cudaStream(), values, indices, input, k, axis, sortMode);
}

void CudaBackend::sort(Tensor& out,
    const Tensor& input, const Dim axis, const SortMode sortMode,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::sort(cudaStream(), out, input, axis, sortMode);
}

void CudaBackend::sort(
    Tensor& values, Tensor& indices, const Tensor& input,
    const Dim axis, const SortMode sortMode,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::sort(cudaStream(), values, indices, input, axis, sortMode);
}

void CudaBackend::argsort(Tensor& out,
    const Tensor& input, const Dim axis, const SortMode sortMode,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::argsort(cudaStream(), out, input, axis, sortMode);
}

void CudaBackend::matmul(Tensor& out,
    const Tensor& lhs, const Tensor& rhs,
    MatrixProperty lhsProp, MatrixProperty rhsProp,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::matmul(out, lhs, rhs, lhsProp, rhsProp, cudaStream(), cublasHandle());
}

void CudaBackend::reshape(Tensor& out, const Tensor& tensor, const Shape& shape,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::reshape(cudaStream(), out, tensor, shape);
}

void CudaBackend::transpose(Tensor& out, const Tensor& tensor, const Shape& axes,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::transpose(cudaStream(), out, tensor, axes);
}

void CudaBackend::tile(Tensor& out, const Tensor& tensor, const Shape& shape,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::tile(cudaStream(), out, tensor, shape);
}

void CudaBackend::concatenate(Tensor& out,
    const std::vector<Tensor>& tensors, const unsigned axis,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::concatenate(cudaStream(), out, tensors, axis);
}

void CudaBackend::nonzero(Tensor& out, const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::nonzero(cudaStream(), out, tensor);
}

void CudaBackend::pad(Tensor& out,
    const Tensor& input,
    const std::vector<std::pair<int, int>>& padWidths,
    const PadType type,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::pad(cudaStream(), out, input, padWidths, type);
}

void CudaBackend::min(
    Tensor& values, Tensor& indices, const Tensor& input,
    const unsigned axis, const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::min(cudaStream(), values, indices, input, axis, keepDims);
}

void CudaBackend::max(
    Tensor& values, Tensor& indices, const Tensor& input,
    const unsigned axis, const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::max(cudaStream(), values, indices, input, axis, keepDims);
}

void CudaBackend::cumsum(Tensor& out, const Tensor& input, const unsigned axis,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::cumsum(cudaStream(), out, input, axis);
}

void CudaBackend::argmax(Tensor& out,
    const Tensor& input, const unsigned axis, const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::argmax(cudaStream(), out, input, axis, keepDims);
}

void CudaBackend::argmin(Tensor& out,
    const Tensor& input, const unsigned axis, const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::argmin(cudaStream(), out, input, axis, keepDims);
}

void CudaBackend::var(Tensor& out,
    const Tensor& input, const std::vector<int>& axes,
    bool bias, const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::var(cudaStream(), out, input, axes, bias, keepDims);
}

void CudaBackend::norm(Tensor& out,
    const Tensor& input, const std::vector<int>& axes,
    double p, const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  cuda::norm(cudaStream(), out, input, axes, p, keepDims);
}

// --- Data Transfer ---

// CudaBackend::copy is in CudaBackendTransfer.cpp (NVCC can't compile CPUTensor.h)

// --- Autograd ops (delegate to cuda:: free functions) ---

ForwardBackwardState
CudaBackend::conv2dForward(
    Tensor& out,
    const Tensor& input, const Tensor& weights, const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    bool wantState) {
  return cuda::conv2dForward(
      out, input, weights, bias, sx, sy, px, py, dx, dy, groups, wantState);
}

Tensor CudaBackend::conv2dBackwardData(
    const Tensor& gradOutput, const Tensor& input, const Tensor& weight,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const ForwardBackwardState& state) {
  return cuda::conv2dBackwardData(
      gradOutput, input, weight, sx, sy, px, py, dx, dy, groups, state);
}

std::pair<Tensor, Tensor> CudaBackend::conv2dBackwardFilterBias(
    const Tensor& gradOutput, const Tensor& input,
    const Tensor& weights, const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const ForwardBackwardState& state) {
  return cuda::conv2dBackwardFilterBias(
      gradOutput, input, weights, bias, sx, sy, px, py, dx, dy, groups, state);
}

ForwardBackwardState
CudaBackend::pool2dForward(
    Tensor& out,
    const Tensor& input, int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode, bool wantState) {
  return cuda::pool2dForward(
      out, input, wx, wy, sx, sy, px, py, mode, wantState);
}

Tensor CudaBackend::pool2dBackward(
    const Tensor& gradOutput, const Tensor& input, const Tensor& poolOutput,
    int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode, const ForwardBackwardState& state) {
  return cuda::pool2dBackward(
      gradOutput, input, poolOutput, wx, wy, sx, sy, px, py, mode, state);
}

BatchnormForwardResult
CudaBackend::batchnormForward(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& runningMean, const Tensor& runningVar,
    const std::vector<int>& axes, bool train, double momentum, double epsilon,
    bool wantState) {
  return cuda::batchnormForward(
      input, weight, bias,
      runningMean, runningVar, axes, train, momentum, epsilon, wantState);
}

std::tuple<Tensor, Tensor, Tensor> CudaBackend::batchnormBackward(
    const Tensor& gradOutput, const Tensor& saveMean, const Tensor& saveVar,
    const Tensor& input, const Tensor& weight,
    const std::vector<int>& axes, bool train, float epsilon,
    const ForwardBackwardState& state) {
  return cuda::batchnormBackward(
      gradOutput, saveMean, saveVar, input, weight,
      axes, train, epsilon, state);
}

std::tuple<Tensor, Tensor, Tensor, ForwardBackwardState>
CudaBackend::rnnForward(
    const Tensor& input, const Tensor& hiddenState, const Tensor& cellState,
    const Tensor& weights, int hiddenSize, int numLayers,
    RnnMode mode, bool bidirectional, float dropout,
    bool wantState) {
  return cuda::rnnForward(
      input, hiddenState, cellState, weights,
      hiddenSize, numLayers, mode, bidirectional, dropout, wantState);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> CudaBackend::rnnBackward(
    const Tensor& input, const Tensor& hiddenState, const Tensor& cellState,
    const Tensor& weights,
    const Tensor& dy, const Tensor& dhy, const Tensor& dcy,
    const Tensor& output, int numLayers, int hiddenSize,
    RnnMode mode, bool bidirectional, float dropProb,
    const ForwardBackwardState& state) {
  return cuda::rnnBackward(
      input, hiddenState, cellState, weights,
      dy, dhy, dcy, output, numLayers, hiddenSize,
      mode, bidirectional, dropProb, state);
}

// --- Utilities ---
void CudaBackend::print(const Tensor& tensor) {
  size_t n = tensor.elements();
  std::vector<float> host(n);
  auto asF32 = tensor.astype(fl::dtype::f32);
  asF32.hostSync(host.data());
  std::cout << "CudaTensor [" << tensor.shape() << "]:" << std::endl;
  for (size_t i = 0; i < std::min(n, size_t(100)); ++i) {
    std::cout << host[i] << " ";
  }
  if (n > 100) {
    std::cout << "...";
  }
  std::cout << std::endl;
}

} // namespace fl
