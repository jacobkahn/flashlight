/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/cpu/CPUBackend.h"

#include "flashlight/fl/tensor/BackendDispatch-inl.h"
#include "flashlight/fl/tensor/BackendRegistry.h"
#include "flashlight/fl/tensor/ops/onednn/OneDnnOps.h"
#include "flashlight/fl/tensor/ops/onednn/OneDnnAsyncCPUStream.h"

#include <cstring>
#include <dnnl.hpp>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/ops/onednn/Utils.h"

namespace fl {

FL_REGISTER_TENSOR_BACKEND(CPU, CPUBackend::getInstance);

CPUBackend::CPUBackend() {
#if FL_USE_MKL_RNG
  vslNewStream(&randStream_, VSL_BRNG_MCG31, std::rand());
#else
  randEngine_ = RandEngineType{static_cast<uint_fast32_t>(std::rand())};
#endif // FL_USE_MKL_RNG
  onednnOps_ = std::make_unique<OneDnnOps>();
  asyncStream_ = OneDnnAsyncCPUStream::create(onednnOps_->engine());

  // On ARM, verify at runtime that oneDNN was built with ACL support.
  // Without ACL, compute-heavy ops (matmul, conv, pooling) use reference
  // kernels that are significantly slower.
#if FL_ONEDNN_ARM
  #ifndef DNNL_AARCH64_USE_ACL
  throw std::runtime_error(
      "[CPUBackend] ARM backend requires oneDNN built with ACL support "
      "(DNNL_AARCH64_USE_ACL=ON). The linked oneDNN library was not built "
      "with ACL. Please rebuild oneDNN with -DDNNL_AARCH64_USE_ACL=ON and "
      "ensure the ARM Compute Library is installed.");
  #endif
#endif
}

CPUBackend::~CPUBackend() {
  // Drain all pending async tasks before destroying OneDnnOps (which owns
  // the dnnl::engine and dnnl::stream). The worker thread captures
  // &nativeStream_ by reference in enqueued tasks (e.g., reorder, binary ops).
  // If we let default member destruction proceed first, the stream is destroyed
  // while tasks may still be queued, causing use-after-free in the worker thread.
  if (asyncStream_) {
    asyncStream_->sync();
  }
}

CPUBackend& CPUBackend::getInstance() {
  static CPUBackend instance;
  return instance;
}

TensorBackendType CPUBackend::backendType() const {
  return TensorBackendType::CPU;
}

BackendDispatch CPUBackend::createDispatch() {
  return makeDirectDispatch(*this);
}

const Stream& CPUBackend::stream() const {
  return *asyncStream_;
}

std::shared_ptr<AsyncCPUStream> CPUBackend::asyncStream() const {
  return asyncStream_;
}

std::shared_ptr<Stream> CPUBackend::jitStream() const {
  return asyncStream_;
}

OneDnnOps& CPUBackend::onednnOps() {
  return *onednnOps_;
}

const OneDnnOps& CPUBackend::onednnOps() const {
  return *onednnOps_;
}

dnnl::stream& CPUBackend::nativeStream() {
  return asyncStream_->nativeStream();
}

void CPUBackend::inPlaceAdd(CPUTensor& self, double val) {
  onednnOps_->inPlaceAdd(self, val, nativeStream());
}
void CPUBackend::inPlaceSub(CPUTensor& self, double val) {
  onednnOps_->inPlaceSub(self, val, nativeStream());
}
void CPUBackend::inPlaceMul(CPUTensor& self, double val) {
  onednnOps_->inPlaceMul(self, val, nativeStream());
}
void CPUBackend::inPlaceDiv(CPUTensor& self, double val) {
  onednnOps_->inPlaceDiv(self, val, nativeStream());
}
void CPUBackend::inPlaceAdd(CPUTensor& self, const Tensor& other) {
  onednnOps_->inPlaceAdd(self, other, nativeStream());
}
void CPUBackend::inPlaceSub(CPUTensor& self, const Tensor& other) {
  onednnOps_->inPlaceSub(self, other, nativeStream());
}
void CPUBackend::inPlaceMul(CPUTensor& self, const Tensor& other) {
  onednnOps_->inPlaceMul(self, other, nativeStream());
}
void CPUBackend::inPlaceDiv(CPUTensor& self, const Tensor& other) {
  onednnOps_->inPlaceDiv(self, other, nativeStream());
}
void CPUBackend::fillInPlace(CPUTensor& self, double val) {
  onednnOps_->fillInPlace(self, val, nativeStream());
}
Tensor CPUBackend::reorderAstype(CPUTensor& self, fl::dtype type) {
  return onednnOps_->reorderAstype(self, type, nativeStream());
}
void CPUBackend::assignTensor(CPUTensor& self, const Tensor& src) {
  onednnOps_->assignTensor(self, src, nativeStream());
}

/* -------------------------- Compute Functions -------------------------- */

void CPUBackend::eval(const Tensor& tensor) {
  // Sync the tensor's stream to ensure all enqueued async operations
  // (reorders, binary ops, reductions, etc.) have completed and the
  // tensor's data is materialized.
  tensor.stream().sync();
}

bool CPUBackend::supportsDataType(const fl::dtype& type) const {
  // f16/f64 have OneDNN type mappings but CPU engine can't compute on them
  // natively for most ops. f64 fallbacks exist but are too slow for practical
  // use (see TODO.md). bf16 is supported: OneDNN auto-upcasts to f32
  // internally on CPUs without AVX512_BF16, and computes natively on CPUs
  // with AVX512_BF16 (Cooper Lake, Sapphire Rapids+).
  if ((type == fl::dtype::f16 || type == fl::dtype::f64) &&
      onednnOps_->engine().get_kind() == dnnl::engine::kind::cpu) {
    return false;
  }
  // s16/u16 are mapped to f16 bytes and s64/u64 to f64 bytes for raw storage,
  // but arithmetic on these mappings produces wrong results. Mark unsupported
  // so tests skip them.
  if (type == fl::dtype::s16 || type == fl::dtype::u16 ||
      type == fl::dtype::s64 || type == fl::dtype::u64) {
    return false;
  }
  // u32 is stored as s32 (same byte size), fully supported
  if (type == fl::dtype::u32) {
    return true;
  }
  return detail::isTypeSupportedByOneDnn(type);
}

void CPUBackend::getMemMgrInfo(
    const char* /* msg */,
    const int /* deviceId */,
    std::ostream* /* ostream */) {
  throw std::runtime_error(
      "[CPUBackend] Currently no memory manager support");
}

void CPUBackend::setMemMgrLogStream(std::ostream* /* stream */) {
  throw std::runtime_error(
      "[CPUBackend] Currently no memory manager support");
}

void CPUBackend::setMemMgrLoggingEnabled(const bool /* enabled */) {
  throw std::runtime_error(
      "[CPUBackend] Currently no memory manager support");
}

void CPUBackend::setMemMgrFlushInterval(const size_t /* interval */) {
  throw std::runtime_error(
      "[CPUBackend] Currently no memory manager support");
}

/* -------------------------- Rand Functions -------------------------- */

void CPUBackend::setSeed(const int seed) {
#if FL_USE_MKL_RNG
  vslDeleteStream(&randStream_);
  vslNewStream(&randStream_, VSL_BRNG_MCG31, seed);
#else
  randEngine_.seed(seed);
#endif // FL_USE_MKL_RNG
}

Tensor CPUBackend::randnCpu(const Shape& shape, const dtype type) {
  std::vector<float> data(shape.elements());
#if FL_USE_MKL_RNG
  const auto alg = VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2;
  vsRngGaussian(alg, randStream_, data.size(), data.data(), 0, 1);
#else
  std::normal_distribution<float> normal_dist(0, 1);
  auto* data_ptr = data.data();
  for (decltype(data.size()) i = 0; i < data.size(); ++i) {
    data_ptr[i] = normal_dist(randEngine_);
  }
#endif // FL_USE_MKL_RNG
  return toTensor<CPUTensor>(shape, dtype::f32, data.data(), Location::Host)
      .astype(type);
}

Tensor CPUBackend::randCpu(const Shape& shape, const dtype type) {
  // For integer types, generate random integers in the full type range
  // (matching ArrayFire's af::randu behavior for integer types).
  // Casting [0,1) floats to integers would truncate everything to 0.
  if (!isFloatingPoint(type)) {
    auto n = shape.elements();
    switch (type) {
      case dtype::s32: {
        std::vector<int> data(n);
        std::uniform_int_distribution<int> dist(
            std::numeric_limits<int>::min(),
            std::numeric_limits<int>::max());
        for (size_t i = 0; i < n; ++i) data[i] = dist(randEngine_);
        return toTensor<CPUTensor>(shape, type, data.data(), Location::Host);
      }
      case dtype::u32: {
        std::vector<unsigned int> data(n);
        std::uniform_int_distribution<unsigned int> dist(
            0, std::numeric_limits<unsigned int>::max());
        for (size_t i = 0; i < n; ++i) data[i] = dist(randEngine_);
        return toTensor<CPUTensor>(shape, type, data.data(), Location::Host);
      }
      case dtype::s16: {
        std::vector<short> data(n);
        std::uniform_int_distribution<short> dist(
            std::numeric_limits<short>::min(),
            std::numeric_limits<short>::max());
        for (size_t i = 0; i < n; ++i) data[i] = dist(randEngine_);
        return toTensor<CPUTensor>(shape, type, data.data(), Location::Host);
      }
      case dtype::s64: {
        std::vector<long long> data(n);
        std::uniform_int_distribution<long long> dist(
            std::numeric_limits<long long>::min(),
            std::numeric_limits<long long>::max());
        for (size_t i = 0; i < n; ++i) data[i] = dist(randEngine_);
        return toTensor<CPUTensor>(shape, type, data.data(), Location::Host);
      }
      case dtype::u8: {
        std::vector<unsigned char> data(n);
        std::uniform_int_distribution<unsigned short> dist(0, 255);
        for (size_t i = 0; i < n; ++i)
          data[i] = static_cast<unsigned char>(dist(randEngine_));
        return toTensor<CPUTensor>(shape, type, data.data(), Location::Host);
      }
      case dtype::b8: {
        std::vector<char> data(n);
        std::uniform_int_distribution<int> dist(0, 1);
        for (size_t i = 0; i < n; ++i)
          data[i] = static_cast<char>(dist(randEngine_));
        return toTensor<CPUTensor>(shape, type, data.data(), Location::Host);
      }
      default:
        break; // fall through to float path
    }
  }
  std::vector<float> data(shape.elements());
#if FL_USE_MKL_RNG
  const auto alg = VSL_RNG_METHOD_UNIFORM_STD;
  vsRngUniform(alg, randStream_, data.size(), data.data(), 0, 1);
#else
  std::uniform_real_distribution<float> uniform_dist{};
  auto* data_ptr = data.data();
  for (decltype(data.size()) i = 0; i < data.size(); ++i) {
    data_ptr[i] = uniform_dist(randEngine_);
  }
#endif // FL_USE_MKL_RNG
  return toTensor<CPUTensor>(shape, dtype::f32, data.data(), Location::Host)
      .astype(type);
}

Tensor CPUBackend::randn(const Shape& shape, dtype type, std::shared_ptr<const Stream> /* stream */) {
  if (onednnOps_->engine().get_kind() == dnnl::engine::kind::cpu) {
    return randnCpu(shape, type);
  } else {
    throw std::runtime_error(
        "[CPUBackend::randn] unimplemented for non-CPU engine");
  }
}

Tensor CPUBackend::rand(const Shape& shape, dtype type, std::shared_ptr<const Stream> /* stream */) {
  if (onednnOps_->engine().get_kind() == dnnl::engine::kind::cpu) {
    return randCpu(shape, type);
  } else {
    throw std::runtime_error(
        "[CPUBackend::rand] unimplemented for non-CPU engine");
  }
}

/* -------- Tensor ops: see CPU{Creation,UnaryOps,BinaryOps,ShapeAndIndex,Reductions}.cpp -------- */


void CPUBackend::copy(
    Tensor& dst, const Tensor& src, const Stream& stream) {
  if (dst.isEmpty()) {
    dst = empty(src.shape(), src.type());
  }

  // Ensure src data is ready (async stream may have pending work)
  src.stream().sync();

  Tensor contigSrc = src.isContiguous() ? src : src.asContiguousTensor();

  void* dstPtr;
  dst.device(&dstPtr);
  void* srcPtr;
  contigSrc.device(&srcPtr);
  std::memcpy(dstPtr, srcPtr, dst.elements() * fl::getTypeSize(dst.type()));
}

void CPUBackend::print(const Tensor& tensor) {
  std::cout << "CPUTensor" << std::endl
            << tensor.getAdapter<CPUTensor>().toString() << std::endl;
}

/************************ Enum-Dispatched Ops ************************/

void CPUBackend::binary(Tensor& out, BinaryOp op, const Tensor& lhs, const Tensor& rhs,
    std::shared_ptr<const Stream> stream) {
  using A = dnnl::algorithm;
  auto s8 = dnnl::memory::data_type::s8;
  switch (op) {
    // Arithmetic — direct OneDNN dispatch, reuses out buffer
    case BinaryOp::Add:
      onednnOps_->applyBinop(out, lhs, rhs, A::binary_add, nativeStream(), asyncStream_); return;
    case BinaryOp::Sub:
      onednnOps_->applyBinop(out, lhs, rhs, A::binary_sub, nativeStream(), asyncStream_); return;
    case BinaryOp::Mul:
      onednnOps_->applyBinop(out, lhs, rhs, A::binary_mul, nativeStream(), asyncStream_); return;
    case BinaryOp::Div:
      onednnOps_->applyBinop(out, lhs, rhs, A::binary_div, nativeStream(), asyncStream_); return;
    case BinaryOp::Min:
      onednnOps_->applyBinop(out, lhs, rhs, A::binary_min, nativeStream(), asyncStream_); return;
    case BinaryOp::Max:
      onednnOps_->applyBinop(out, lhs, rhs, A::binary_max, nativeStream(), asyncStream_); return;
    // Comparison — output forced to s8/b8
    case BinaryOp::Eq:
      onednnOps_->applyBinop(out, lhs, rhs, A::binary_eq, nativeStream(), asyncStream_, s8); return;
    case BinaryOp::Neq:
      onednnOps_->applyBinop(out, lhs, rhs, A::binary_ne, nativeStream(), asyncStream_, s8); return;
    case BinaryOp::Gt:
      onednnOps_->applyBinop(out, lhs, rhs, A::binary_gt, nativeStream(), asyncStream_, s8); return;
    case BinaryOp::Gte:
      onednnOps_->applyBinop(out, lhs, rhs, A::binary_ge, nativeStream(), asyncStream_, s8); return;
    case BinaryOp::Lt:
      onednnOps_->applyBinop(out, lhs, rhs, A::binary_lt, nativeStream(), asyncStream_, s8); return;
    case BinaryOp::Lte:
      onednnOps_->applyBinop(out, lhs, rhs, A::binary_le, nativeStream(), asyncStream_, s8); return;
    // Non-OneDNN fallbacks — reconciliation handles out
    case BinaryOp::Pow: out = power(lhs, rhs, std::move(stream)); return;
    case BinaryOp::Mod: out = mod(lhs, rhs, std::move(stream)); return;
    case BinaryOp::And: out = logicalAnd(lhs, rhs, std::move(stream)); return;
    case BinaryOp::Or: out = logicalOr(lhs, rhs, std::move(stream)); return;
    case BinaryOp::Shl: out = lShift(lhs, rhs, std::move(stream)); return;
    case BinaryOp::Shr: out = rShift(lhs, rhs, std::move(stream)); return;
    case BinaryOp::BitAnd: out = bitwiseAnd(lhs, rhs, std::move(stream)); return;
    case BinaryOp::BitOr: out = bitwiseOr(lhs, rhs, std::move(stream)); return;
    case BinaryOp::BitXor: out = bitwiseXor(lhs, rhs, std::move(stream)); return;
  }
  throw std::runtime_error("[CPUBackend::binary] Unknown binary operation type");
}

void CPUBackend::unary(Tensor& out, UnaryOp op, const Tensor& input,
    std::shared_ptr<const Stream> stream) {
  using A = dnnl::algorithm;
  switch (op) {
    // Direct OneDNN eltwise dispatch, reuses out buffer
    case UnaryOp::Exp:
      onednnOps_->applyEltwiseOp(out, input, A::eltwise_exp, nativeStream(), asyncStream_); return;
    case UnaryOp::Log:
      onednnOps_->applyEltwiseOp(out, input, A::eltwise_log, nativeStream(), asyncStream_); return;
    case UnaryOp::Sqrt:
      onednnOps_->applyEltwiseOp(out, input, A::eltwise_sqrt, nativeStream(), asyncStream_); return;
    case UnaryOp::Tanh:
      onednnOps_->applyEltwiseOp(out, input, A::eltwise_tanh, nativeStream(), asyncStream_); return;
    case UnaryOp::Rint:
      onednnOps_->applyEltwiseOp(out, input, A::eltwise_round, nativeStream(), asyncStream_); return;
    case UnaryOp::Negative:
      onednnOps_->applyEltwiseOp(out, input, A::eltwise_linear, nativeStream(), asyncStream_, -1, 0); return;
    case UnaryOp::Sigmoid:
      onednnOps_->applyEltwiseOp(out, input, A::eltwise_logistic, nativeStream(), asyncStream_); return;
    // Absolute has split path (int vs float) — keep named op
    case UnaryOp::Absolute: out = absolute(input, std::move(stream)); return;
    // Native fallbacks — reconciliation handles out
    case UnaryOp::Log1p: out = log1p(input, std::move(stream)); return;
    case UnaryOp::Sin: out = sin(input, std::move(stream)); return;
    case UnaryOp::Cos: out = cos(input, std::move(stream)); return;
    case UnaryOp::Floor: out = floor(input, std::move(stream)); return;
    case UnaryOp::Ceil: out = ceil(input, std::move(stream)); return;
    case UnaryOp::Erf: out = erf(input, std::move(stream)); return;
    case UnaryOp::LogicalNot: out = logicalNot(input, std::move(stream)); return;
    case UnaryOp::Sign: out = sign(input, std::move(stream)); return;
    case UnaryOp::Isnan: out = isnan(input, std::move(stream)); return;
    case UnaryOp::Isinf: out = isinf(input, std::move(stream)); return;
  }
  throw std::runtime_error("[CPUBackend::unary] Unknown unary operation type");
}

void CPUBackend::reduction(Tensor& out, ReductionOp op, const Tensor& input,
    const std::vector<int>& axes, bool keepDims,
    std::shared_ptr<const Stream> stream) {
  using A = dnnl::algorithm;
  switch (op) {
    // Direct OneDNN dispatch, reuses out buffer
    case ReductionOp::Amin:
      onednnOps_->applyReductionOp(out, input, A::reduction_min, axes, keepDims,
          nativeStream(), asyncStream_); return;
    case ReductionOp::Amax:
      onednnOps_->applyReductionOp(out, input, A::reduction_max, axes, keepDims,
          nativeStream(), asyncStream_); return;
    // Sum/mean have type promotion logic — keep named ops for now
    case ReductionOp::Sum: out = sum(input, axes, keepDims, std::move(stream)); return;
    case ReductionOp::Mean: out = mean(input, axes, keepDims, std::move(stream)); return;
    // Native/composed fallbacks
    case ReductionOp::Median: out = median(input, axes, keepDims, std::move(stream)); return;
    case ReductionOp::Std: out = std(input, axes, keepDims, std::move(stream)); return;
    case ReductionOp::CountNonzero: out = countNonzero(input, axes, keepDims, std::move(stream)); return;
    case ReductionOp::Any: out = any(input, axes, keepDims, std::move(stream)); return;
    case ReductionOp::All: out = all(input, axes, keepDims, std::move(stream)); return;
  }
  throw std::runtime_error("[CPUBackend::reduction] Unknown reduction operation type");
}

} // namespace fl
