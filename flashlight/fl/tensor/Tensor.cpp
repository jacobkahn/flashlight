/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Tensor.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/runtime/StreamOrDevice.h"
#include "flashlight/fl/tensor/BackendRegistry.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/JitTensorBase.h"
#include "flashlight/fl/tensor/OpEnums.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/TensorBackend.h"

#define FL_TENSOR_BACKENDS_MATCH_CHECK(...)             \
  if (!detail::areBackendsEqual(__VA_ARGS__)) {         \
    throw std::invalid_argument(                        \
        std::string(__func__) +                         \
        " called with tensors of different backends."); \
  }

namespace fl {

Tensor::Tensor(std::unique_ptr<TensorAdapterBase> adapter)
    : impl_(std::move(adapter)) {}

std::unique_ptr<TensorAdapterBase> Tensor::releaseAdapter() {
  return std::move(impl_);
}

Tensor::~Tensor() = default;

Tensor::Tensor(const Tensor& tensor)
    : impl_(tensor.impl_->clone()),
      autogradMeta_(tensor.autogradMeta_) {}

Tensor::Tensor(Tensor&& other) noexcept
    : impl_(std::move(other.impl_)),
      autogradMeta_(std::move(other.autogradMeta_)) {}

Tensor::Tensor() : impl_(detail::getDefaultAdapter()) {}

Tensor::Tensor(
    const Shape& shape,
    fl::dtype type,
    const void* ptr,
    MemoryLocation memoryLocation)
    : impl_(detail::getDefaultAdapter(shape, type, ptr, memoryLocation)) {}

Tensor::Tensor(
    const Dim nRows,
    const Dim nCols,
    const Tensor& values,
    const Tensor& rowIdx,
    const Tensor& colIdx,
    StorageType storageType)
    : impl_(detail::getDefaultAdapter(
          nRows,
          nCols,
          values,
          rowIdx,
          colIdx,
          storageType)) {}

Tensor::Tensor(const Shape& shape, fl::dtype type /* = fl::dtype::f32 */)
    : impl_(detail::getDefaultAdapter(shape, type)) {}

Tensor::Tensor(const Shape& shape, fl::dtype type, const Device& device) {
  auto backendType = backendTypeForDevice(device.type());
  auto& jitBackend = getOrCreateJitBackend(getBackend(backendType));
  *this = jitBackend.empty(shape, type, nullptr);
}

Tensor::Tensor(fl::dtype type)
    : impl_(detail::getDefaultAdapter(Shape({0}), type)) {}

Tensor Tensor::copy() const {
  auto result = impl_->copy();
  if (requiresGrad()) {
    return Tensor(result, true);
  }
  return result;
}

Tensor Tensor::shallowCopy() const {
  return impl_->shallowCopy();
}

const Shape& Tensor::shape() const {
  return impl_->shape();
}

Location Tensor::location() const {
  return impl_->location();
}

size_t Tensor::elements() const {
  return impl_->shape().elements();
}

Dim Tensor::dim(const size_t dim) const {
  return shape().dim(dim);
}

int Tensor::ndim() const {
  return shape().ndim();
}

bool Tensor::isEmpty() const {
  return elements() == 0;
}

bool Tensor::hasAdapter() const {
  return impl_.get() != nullptr;
}

size_t Tensor::bytes() const {
  return elements() * getTypeSize(type());
}

dtype Tensor::type() const {
  return impl_->type();
}

bool Tensor::isSparse() const {
  return impl_->isSparse();
}

Tensor Tensor::astype(const dtype newType) const {
  auto result = impl_->astype(newType);
  if (requiresGrad()) {
    return autograd::detail::autogradAstype(*this, newType, result);
  }
  return result;
}

Tensor Tensor::operator()(const std::vector<Index>& indices) const {
  auto result = impl_->index(indices);
  if (requiresGrad()) {
    return autograd::detail::autogradIndex(*this, indices, result);
  }
  return result;
}

Tensor Tensor::flatten() const {
  return impl_->flatten();
}

Tensor Tensor::flat(const Index& idx) const {
  auto result = impl_->flat(idx);
  if (requiresGrad()) {
    return autograd::detail::autogradFlat(*this, idx, result);
  }
  return result;
}

Tensor Tensor::asContiguousTensor() const {
  return impl_->asContiguousTensor();
}

TensorBackendType Tensor::backendType() const {
  return impl_->backendType();
}

TensorBackend& Tensor::backend() const {
  return impl_->backend();
}

// Non-template helpers for inline template methods in Tensor.h.
// These bridge the TensorAdapterBase/JitTensorBase dependency.
std::optional<std::pair<Event, Tensor>>
Tensor::to(const Device& device) const {
  if (backendType() != TensorBackendType::Jit) {
    // TODO: call backend::copy(src, device) for non-JIT backends
    return std::nullopt;
  }
  return toJitTensorBase(*this).to(device);
}

void Tensor::readScalar(void* out) const {
  impl_->scalarSync(out);
}

void Tensor::readDevice(void** out) const {
  impl_->device(out);
}

void Tensor::readDeviceSync(void** out) const {
  impl_->deviceSync(out);
}

void Tensor::readHost(void* out) const {
  impl_->hostSync(out);
}

void Tensor::unlock() const {
  impl_->unlock();
}

bool Tensor::isLocked() const {
  return impl_->isLocked();
}

bool Tensor::isContiguous() const {
  return impl_->isContiguous();
}

Shape Tensor::strides() const {
  return impl_->strides();
}

const Stream& Tensor::stream() const {
  return impl_->stream();
}

void Tensor::setContext(void* context) {
  impl_->setContext(context);
}

void* Tensor::getContext() const {
  return impl_->getContext();
}

std::string Tensor::toString() const {
  return impl_->toString();
}

std::ostream& Tensor::operator<<(std::ostream& ostr) const {
  return impl_->operator<<(ostr);
}

/******************** Assignment Operators ********************/
#define FL_ASSIGN_OP_TYPE(OP, FUN, TYPE) \
  Tensor& Tensor::OP(TYPE val) {         \
    impl_->FUN(val);                     \
    return *this;                        \
  }
#define FL_ASSIGN_TENSOR_OP(OP, FUN) FL_ASSIGN_OP_TYPE(OP, FUN, const Tensor&);
#define FL_ASSIGN_SCALAR_OP(OP, FUN)                 \
  FL_ASSIGN_OP_TYPE(OP, FUN, const double&);         \
  FL_ASSIGN_OP_TYPE(OP, FUN, const float&);          \
  FL_ASSIGN_OP_TYPE(OP, FUN, const int&);            \
  FL_ASSIGN_OP_TYPE(OP, FUN, const unsigned&);       \
  FL_ASSIGN_OP_TYPE(OP, FUN, const bool&);           \
  FL_ASSIGN_OP_TYPE(OP, FUN, const char&);           \
  FL_ASSIGN_OP_TYPE(OP, FUN, const unsigned char&);  \
  FL_ASSIGN_OP_TYPE(OP, FUN, const short&);          \
  FL_ASSIGN_OP_TYPE(OP, FUN, const unsigned short&); \
  FL_ASSIGN_OP_TYPE(OP, FUN, const long&);           \
  FL_ASSIGN_OP_TYPE(OP, FUN, const unsigned long&);  \
  FL_ASSIGN_OP_TYPE(OP, FUN, const long long&);      \
  FL_ASSIGN_OP_TYPE(OP, FUN, const unsigned long long&);

#define FL_ASSIGN_OP(OP, FUN)   \
  FL_ASSIGN_TENSOR_OP(OP, FUN); \
  FL_ASSIGN_SCALAR_OP(OP, FUN);

// (operator, function name on impl)
FL_ASSIGN_SCALAR_OP(operator=, assign);
FL_ASSIGN_OP(operator+=, inPlaceAdd);
FL_ASSIGN_OP(operator-=, inPlaceSubtract);
FL_ASSIGN_OP(operator*=, inPlaceMultiply);
FL_ASSIGN_OP(operator/=, inPlaceDivide);
#undef FL_ASSIGN_OP_TYPE
#undef FL_ASSIGN_TENSOR_OP
#undef FL_ASSIGN_SCALAR_OP
#undef FL_ASSIGN_OP

// Move assignment operator when `this` is a lvalue, e.g., `x = std::move(y)`.
// In such cases, we let `this` take over the tensor data of `other`.
Tensor& Tensor::operator=(Tensor&& other) & {
  this->impl_ = std::move(other.impl_);
  this->autogradMeta_ = std::move(other.autogradMeta_);
  return *this;
}

// Move assignment operator when `this` is a rvalue, e.g., `x(0) =
// std::move(y)`. In such cases, we copy the data from `other` to `this`.
Tensor& Tensor::operator=(Tensor&& other) && {
  this->impl_->assign(other);
  // Note: rvalue assignment (indexing) does NOT transfer autograd metadata.
  return *this;
}

// Copy assignment operator when `this` is a lvalue, e.g., `x = y`.
// In such cases, we let `this` take over the _cloned_ data from `other`.
Tensor& Tensor::operator=(const Tensor& other) & {
  this->impl_ = other.impl_->clone();
  this->autogradMeta_ = other.autogradMeta_;
  return *this;
}

// Copy assignment operator when `this` is a lvalue, e.g., `x(0) = y`.
// In such cases, we copy the data from `other` to `this`.
Tensor& Tensor::operator=(const Tensor& other) && {
  this->impl_->assign(other);
  // Note: rvalue assignment (indexing) does NOT transfer autograd metadata.
  return *this;
}

/* --------------------------- Tensor Operators --------------------------- */

/******************** Tensor Creation Functions ********************/

// Resolve the JIT backend for a StreamOrDevice: if a Device is specified,
// look up the corresponding backend; otherwise use the thread-default.
static TensorBackend& backendFor(StreamOrDevice& s) {
  if (s.hasDevice()) {
    return getOrCreateJitBackend(
        getBackend(backendTypeForDevice(s.device().type())));
  }
  return defaultTensorBackend();
}

#define FL_CREATE_FUN_LITERAL_TYPE(TYPE)                                       \
  template <>                                                                  \
  FL_API Tensor fromScalar(TYPE value, const dtype type, StreamOrDevice s) {   \
    return backendFor(s).fromScalar(                                           \
        static_cast<double>(value), type, s.takeStream());                    \
  }                                                                            \
  template <>                                                                  \
  FL_API Tensor full(                                                          \
      const Shape& dims, TYPE value, const dtype type, StreamOrDevice s) {     \
    return backendFor(s).full(                                                 \
        dims, static_cast<double>(value), type, s.takeStream());              \
  }
FL_CREATE_FUN_LITERAL_TYPE(const double&);
FL_CREATE_FUN_LITERAL_TYPE(const float&);
FL_CREATE_FUN_LITERAL_TYPE(const int&);
FL_CREATE_FUN_LITERAL_TYPE(const unsigned&);
FL_CREATE_FUN_LITERAL_TYPE(const char&);
FL_CREATE_FUN_LITERAL_TYPE(const unsigned char&);
FL_CREATE_FUN_LITERAL_TYPE(const long&);
FL_CREATE_FUN_LITERAL_TYPE(const unsigned long&);
FL_CREATE_FUN_LITERAL_TYPE(const long long&);
FL_CREATE_FUN_LITERAL_TYPE(const unsigned long long&);
FL_CREATE_FUN_LITERAL_TYPE(const bool&);
FL_CREATE_FUN_LITERAL_TYPE(const short&);
FL_CREATE_FUN_LITERAL_TYPE(const unsigned short&);
#undef FL_CREATE_FUN_LITERAL_TYPE

Tensor empty(const Shape& shape, dtype type, StreamOrDevice s) {
  return backendFor(s).empty(shape, type, s.takeStream());
}

Tensor identity(const Dim dim, const dtype type, StreamOrDevice s) {
  return backendFor(s).identity(dim, type, s.takeStream());
}

#define FL_ARANGE_FUN_DEF(TYPE)                                             \
  template <>                                                               \
  FL_API Tensor arange(                                                     \
      TYPE start, TYPE end, TYPE step, const dtype type, StreamOrDevice s) {\
    return fl::arange(                                                      \
        {static_cast<long>((end - start) / step)}, 0, type, std::move(s))   \
        * step + start;                                                     \
  }
FL_ARANGE_FUN_DEF(const double&);
FL_ARANGE_FUN_DEF(const float&);
FL_ARANGE_FUN_DEF(const int&);
FL_ARANGE_FUN_DEF(const unsigned&);
FL_ARANGE_FUN_DEF(const long&);
FL_ARANGE_FUN_DEF(const unsigned long&);
FL_ARANGE_FUN_DEF(const long long&);
FL_ARANGE_FUN_DEF(const unsigned long long&);

Tensor arange(const Shape& shape, const Dim seqDim, const dtype type,
    StreamOrDevice s) {
  return backendFor(s).arange(shape, seqDim, type, s.takeStream());
}

Tensor iota(const Shape& dims, const Shape& tileDims, const dtype type,
    StreamOrDevice s) {
  return backendFor(s).iota(dims, tileDims, type, s.takeStream());
}

/************************ Shaping and Indexing *************************/

Tensor reshape(const Tensor& tensor, const Shape& shape, StreamOrDevice s) {
  Tensor out;
  tensor.backend().reshape(out, tensor, shape, s.takeStream());
  return out;
}

Tensor squeeze(const Tensor& tensor) {
  std::vector<Dim> dims;
  for (unsigned i = 0; i < tensor.ndim(); ++i) {
    if (tensor.dim(i) != 1) {
      dims.push_back(tensor.dim(i));
    }
  }
  // dims empty means all dims were 1 (or tensor was already scalar) => scalar
  return reshape(tensor, Shape(dims));
}

Tensor squeeze(const Tensor& tensor, const unsigned axis) {
  if (axis >= tensor.ndim()) {
    throw std::invalid_argument(
        "squeeze: axis " + std::to_string(axis) +
        " out of range for tensor with " + std::to_string(tensor.ndim()) +
        " dimensions");
  }
  if (tensor.dim(axis) != 1) {
    throw std::invalid_argument(
        "squeeze: can only squeeze dimensions of size 1, but dim(" +
        std::to_string(axis) + ") = " + std::to_string(tensor.dim(axis)));
  }
  std::vector<Dim> dims;
  for (unsigned i = 0; i < tensor.ndim(); ++i) {
    if (i != axis) {
      dims.push_back(tensor.dim(i));
    }
  }
  return reshape(tensor, Shape(dims));
}

Tensor unsqueeze(const Tensor& tensor, const unsigned axis) {
  if (axis > tensor.ndim()) {
    throw std::invalid_argument(
        "unsqueeze: axis " + std::to_string(axis) +
        " out of range for tensor with " + std::to_string(tensor.ndim()) +
        " dimensions");
  }
  std::vector<Dim> dims(tensor.shape().get());
  dims.insert(dims.begin() + axis, 1);
  return reshape(tensor, Shape(dims));
}

Tensor flatten(const Tensor& tensor) {
  return tensor.flatten();
}

Tensor transpose(const Tensor& tensor, const Shape& axes /* = {} */,
    StreamOrDevice s) {
  Tensor out;
  tensor.backend().transpose(out, tensor, axes, s.takeStream());
  return out;
}

Tensor tile(const Tensor& tensor, const Shape& shape, StreamOrDevice s) {
  Tensor out;
  tensor.backend().tile(out, tensor, shape, s.takeStream());
  return out;
}

Tensor concatenate(const std::vector<Tensor>& tensors, const unsigned axis,
    StreamOrDevice s) {
  if (tensors.empty()) {
    throw std::invalid_argument("concatenate: called on empty set of tensors");
  }

  // Check all backends match
  const TensorBackendType b = tensors.front().backendType();
  const bool matches =
      std::all_of(tensors.begin(), tensors.end(), [b](const Tensor& t) {
        return t.backendType() == b;
      });
  if (!matches) {
    throw std::invalid_argument(
        "concatenate: tried to concatenate tensors of different backends");
  }

  Tensor out;
  tensors.front().backend().concatenate(out, tensors, axis, s.takeStream());
  return out;
}

Tensor nonzero(const Tensor& tensor, StreamOrDevice s) {
  Tensor out;
  tensor.backend().nonzero(out, tensor, s.takeStream());
  return out;
}

Tensor pad(
    const Tensor& input,
    const std::vector<std::pair<int, int>>& padWidths,
    const PadType type,
    StreamOrDevice s) {
  Tensor out;
  input.backend().pad(out, input, padWidths, type, s.takeStream());
  return out;
}

/************************** Unary Operators ***************************/
Tensor exp(const Tensor& tensor, StreamOrDevice s) {
  Tensor out;
  tensor.backend().unary(out, UnaryOp::Exp, tensor, s.takeStream());
  return out;
}

Tensor log(const Tensor& tensor, StreamOrDevice s) {
  Tensor out;
  tensor.backend().unary(out, UnaryOp::Log, tensor, s.takeStream());
  return out;
}

Tensor negative(const Tensor& tensor, StreamOrDevice s) {
  Tensor out;
  tensor.backend().unary(out, UnaryOp::Negative, tensor, s.takeStream());
  return out;
}

Tensor logicalNot(const Tensor& tensor, StreamOrDevice s) {
  Tensor out;
  tensor.backend().unary(out, UnaryOp::LogicalNot, tensor, s.takeStream());
  return out;
}

Tensor log1p(const Tensor& tensor, StreamOrDevice s) {
  Tensor out;
  tensor.backend().unary(out, UnaryOp::Log1p, tensor, s.takeStream());
  return out;
}

Tensor sin(const Tensor& tensor, StreamOrDevice s) {
  Tensor out;
  tensor.backend().unary(out, UnaryOp::Sin, tensor, s.takeStream());
  return out;
}

Tensor cos(const Tensor& tensor, StreamOrDevice s) {
  Tensor out;
  tensor.backend().unary(out, UnaryOp::Cos, tensor, s.takeStream());
  return out;
}

Tensor sqrt(const Tensor& tensor, StreamOrDevice s) {
  Tensor out;
  tensor.backend().unary(out, UnaryOp::Sqrt, tensor, s.takeStream());
  return out;
}

Tensor tanh(const Tensor& tensor, StreamOrDevice s) {
  Tensor out;
  tensor.backend().unary(out, UnaryOp::Tanh, tensor, s.takeStream());
  return out;
}

Tensor floor(const Tensor& tensor, StreamOrDevice s) {
  Tensor out;
  tensor.backend().unary(out, UnaryOp::Floor, tensor, s.takeStream());
  return out;
}

Tensor ceil(const Tensor& tensor, StreamOrDevice s) {
  Tensor out;
  tensor.backend().unary(out, UnaryOp::Ceil, tensor, s.takeStream());
  return out;
}

Tensor rint(const Tensor& tensor, StreamOrDevice s) {
  Tensor out;
  tensor.backend().unary(out, UnaryOp::Rint, tensor, s.takeStream());
  return out;
}

Tensor absolute(const Tensor& tensor, StreamOrDevice s) {
  Tensor out;
  tensor.backend().unary(out, UnaryOp::Absolute, tensor, s.takeStream());
  return out;
}

Tensor sigmoid(const Tensor& tensor, StreamOrDevice s) {
  Tensor out;
  tensor.backend().unary(out, UnaryOp::Sigmoid, tensor, s.takeStream());
  return out;
}

Tensor erf(const Tensor& tensor, StreamOrDevice s) {
  Tensor out;
  tensor.backend().unary(out, UnaryOp::Erf, tensor, s.takeStream());
  return out;
}

Tensor flip(const Tensor& tensor, const unsigned dim, StreamOrDevice s) {
  Tensor out;
  tensor.backend().flip(out, tensor, dim, s.takeStream());
  return out;
}

Tensor clip(const Tensor& tensor, const Tensor& low, const Tensor& high,
    StreamOrDevice s) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(tensor, low, high);
  Tensor out;
  tensor.backend().clip(out, tensor, low, high, s.takeStream());
  return out;
}

Tensor clip(const Tensor& tensor, const Tensor& low, const double& high,
    StreamOrDevice s) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(tensor, low);
  Tensor out;
  tensor.backend().clip(out, tensor, low, high, s.takeStream());
  return out;
}

Tensor clip(const Tensor& tensor, const double& low, const Tensor& high,
    StreamOrDevice s) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(tensor, high);
  Tensor out;
  tensor.backend().clip(out, tensor, low, high, s.takeStream());
  return out;
}

Tensor clip(const Tensor& tensor, const double& low, const double& high,
    StreamOrDevice s) {
  Tensor out;
  tensor.backend().clip(out, tensor, low, high, s.takeStream());
  return out;
}

Tensor roll(const Tensor& tensor, const int shift, const unsigned axis,
    StreamOrDevice s) {
  Tensor out;
  tensor.backend().roll(out, tensor, shift, axis, s.takeStream());
  return out;
}

Tensor isnan(const Tensor& tensor, StreamOrDevice s) {
  Tensor out;
  tensor.backend().unary(out, UnaryOp::Isnan, tensor, s.takeStream());
  return out;
}

Tensor isinf(const Tensor& tensor, StreamOrDevice s) {
  Tensor out;
  tensor.backend().unary(out, UnaryOp::Isinf, tensor, s.takeStream());
  return out;
}

Tensor sign(const Tensor& tensor, StreamOrDevice s) {
  Tensor out;
  tensor.backend().unary(out, UnaryOp::Sign, tensor, s.takeStream());
  return out;
}

Tensor tril(const Tensor& tensor, int diagonal, StreamOrDevice s) {
  Tensor out;
  tensor.backend().tril(out, tensor, diagonal, s.takeStream());
  return out;
}

Tensor tril(const Tensor& tensor, StreamOrDevice s) {
  return tril(tensor, 0, std::move(s));
}

Tensor triu(const Tensor& tensor, int diagonal, StreamOrDevice s) {
  Tensor out;
  tensor.backend().triu(out, tensor, diagonal, s.takeStream());
  return out;
}

Tensor triu(const Tensor& tensor, StreamOrDevice s) {
  return triu(tensor, 0, std::move(s));
}

Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y,
    StreamOrDevice s) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(condition, x, y);
  Tensor out;
  condition.backend().where(out, condition, x, y, s.takeStream());
  return out;
}

Tensor where(const Tensor& condition, const Tensor& x, const double& y,
    StreamOrDevice s) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(condition, x);
  Tensor out;
  condition.backend().where(out, condition, x, y, s.takeStream());
  return out;
}

Tensor where(const Tensor& condition, const double& x, const Tensor& y,
    StreamOrDevice s) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(condition, y);
  Tensor out;
  condition.backend().where(out, condition, x, y, s.takeStream());
  return out;
}

void topk(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned k,
    const Dim axis,
    const SortMode sortMode /* = SortMode::Descending */,
    StreamOrDevice s) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(values, indices, input);
  input.backend().topk(values, indices, input, k, axis, sortMode,
      s.takeStream());
}

Tensor sort(const Tensor& input, const Dim axis, const SortMode sortMode,
    StreamOrDevice s) {
  Tensor out;
  input.backend().sort(out, input, axis, sortMode, s.takeStream());
  return out;
}

void sort(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode /* = SortMode::Descending */,
    StreamOrDevice s) {
  return values.backend().sort(values, indices, input, axis, sortMode,
      s.takeStream());
}

Tensor argsort(const Tensor& input, const Dim axis, const SortMode sortMode,
    StreamOrDevice s) {
  Tensor out;
  input.backend().argsort(out, input, axis, sortMode, s.takeStream());
  return out;
}

void indexAdd(
    Tensor& self,
    unsigned axis,
    const Tensor& indices,
    const Tensor& src,
    StreamOrDevice s) {
  self.backend().indexAdd(self, axis, indices, src, s.takeStream());
}

/************************** Binary Operators ***************************/
#define FL_BINARY_OP_DEF(OP, FUNC, BINOP)                              \
  Tensor FUNC(const Tensor& lhs, const Tensor& rhs, StreamOrDevice s) {\
    FL_TENSOR_BACKENDS_MATCH_CHECK(lhs, rhs);                          \
    Tensor out;                                                        \
    lhs.backend().binary(out, BINOP, lhs, rhs, s.takeStream());       \
    return out;                                                        \
  }                                                                    \
  Tensor operator OP(const Tensor& lhs, const Tensor& rhs) {           \
    return FUNC(lhs, rhs);                                             \
  }                                                                    \
  Tensor FUNC(const Tensor& lhs, double rhs, StreamOrDevice s) {       \
    auto stream = s.takeStream();                                      \
    Tensor out;                                                        \
    lhs.backend().binary(out, BINOP,                                   \
        lhs,                                                           \
        lhs.backend().full(lhs.shape(), rhs,                           \
            promoteScalarDtype(lhs.type(), rhs)),                      \
        std::move(stream));                                            \
    return out;                                                        \
  }                                                                    \
  Tensor FUNC(double lhs, const Tensor& rhs, StreamOrDevice s) {       \
    auto stream = s.takeStream();                                      \
    Tensor out;                                                        \
    rhs.backend().binary(out, BINOP,                                   \
        rhs.backend().full(rhs.shape(), lhs,                           \
            promoteScalarDtype(rhs.type(), lhs)),                      \
        rhs,                                                           \
        std::move(stream));                                            \
    return out;                                                        \
  }                                                                    \
  Tensor operator OP(const Tensor& lhs, double rhs) {                  \
    return FUNC(lhs, rhs);                                             \
  }                                                                    \
  Tensor operator OP(double lhs, const Tensor& rhs) {                  \
    return FUNC(lhs, rhs);                                             \
  }

/********** Arithmetic operators **********/
// Named functions are pure tensor ops (no autograd graph).
// Operators dispatch to autograd::detail::autograd*() when either operand
// requires grad. Gradient logic lives in fl/autograd/Functions.cpp.

// --- addition ---
Tensor add(const Tensor& lhs, const Tensor& rhs, StreamOrDevice s) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(lhs, rhs);
  Tensor out;
  lhs.backend().binary(out, BinaryOp::Add, lhs, rhs, s.takeStream());
  return out;
}
Tensor add(const Tensor& lhs, double rhs, StreamOrDevice s) {
  auto stream = s.takeStream();
  Tensor out;
  lhs.backend().binary(out, BinaryOp::Add,
      lhs, lhs.backend().full(lhs.shape(), rhs, promoteScalarDtype(lhs.type(), rhs)),
      std::move(stream));
  return out;
}
Tensor add(double lhs, const Tensor& rhs, StreamOrDevice s) {
  auto stream = s.takeStream();
  Tensor out;
  rhs.backend().binary(out, BinaryOp::Add,
      rhs.backend().full(rhs.shape(), lhs, promoteScalarDtype(rhs.type(), lhs)), rhs,
      std::move(stream));
  return out;
}
Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
  if (lhs.requiresGrad() || rhs.requiresGrad()) {
    return autograd::detail::autogradAdd(lhs, rhs);
  }
  return add(lhs, rhs);
}
Tensor operator+(const Tensor& lhs, double rhs) {
  if (lhs.requiresGrad()) {
    return autograd::detail::autogradAdd(lhs, rhs);
  }
  return add(lhs, rhs);
}
Tensor operator+(double lhs, const Tensor& rhs) {
  return rhs + lhs;
}

// --- subtraction ---
Tensor sub(const Tensor& lhs, const Tensor& rhs, StreamOrDevice s) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(lhs, rhs);
  Tensor out;
  lhs.backend().binary(out, BinaryOp::Sub, lhs, rhs, s.takeStream());
  return out;
}
Tensor sub(const Tensor& lhs, double rhs, StreamOrDevice s) {
  auto stream = s.takeStream();
  Tensor out;
  lhs.backend().binary(out, BinaryOp::Sub,
      lhs, lhs.backend().full(lhs.shape(), rhs, promoteScalarDtype(lhs.type(), rhs)),
      std::move(stream));
  return out;
}
Tensor sub(double lhs, const Tensor& rhs, StreamOrDevice s) {
  auto stream = s.takeStream();
  Tensor out;
  rhs.backend().binary(out, BinaryOp::Sub,
      rhs.backend().full(rhs.shape(), lhs, promoteScalarDtype(rhs.type(), lhs)), rhs,
      std::move(stream));
  return out;
}
Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
  if (lhs.requiresGrad() || rhs.requiresGrad()) {
    return autograd::detail::autogradSub(lhs, rhs);
  }
  return sub(lhs, rhs);
}
Tensor operator-(const Tensor& lhs, double rhs) {
  if (lhs.requiresGrad()) {
    return autograd::detail::autogradSub(lhs, rhs);
  }
  return sub(lhs, rhs);
}
Tensor operator-(double lhs, const Tensor& rhs) {
  if (rhs.requiresGrad()) {
    return autograd::detail::autogradSub(lhs, rhs);
  }
  return sub(lhs, rhs);
}

// --- multiplication ---
Tensor mul(const Tensor& lhs, const Tensor& rhs, StreamOrDevice s) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(lhs, rhs);
  Tensor out;
  lhs.backend().binary(out, BinaryOp::Mul, lhs, rhs, s.takeStream());
  return out;
}
Tensor mul(const Tensor& lhs, double rhs, StreamOrDevice s) {
  auto stream = s.takeStream();
  Tensor out;
  lhs.backend().binary(out, BinaryOp::Mul,
      lhs, lhs.backend().full(lhs.shape(), rhs, promoteScalarDtype(lhs.type(), rhs)),
      std::move(stream));
  return out;
}
Tensor mul(double lhs, const Tensor& rhs, StreamOrDevice s) {
  auto stream = s.takeStream();
  Tensor out;
  rhs.backend().binary(out, BinaryOp::Mul,
      rhs.backend().full(rhs.shape(), lhs, promoteScalarDtype(rhs.type(), lhs)), rhs,
      std::move(stream));
  return out;
}
Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
  if (lhs.requiresGrad() || rhs.requiresGrad()) {
    return autograd::detail::autogradMul(lhs, rhs);
  }
  return mul(lhs, rhs);
}
Tensor operator*(const Tensor& lhs, double rhs) {
  if (lhs.requiresGrad()) {
    return autograd::detail::autogradMul(lhs, rhs);
  }
  return mul(lhs, rhs);
}
Tensor operator*(double lhs, const Tensor& rhs) {
  return rhs * lhs;
}

// --- division ---
Tensor div(const Tensor& lhs, const Tensor& rhs, StreamOrDevice s) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(lhs, rhs);
  Tensor out;
  lhs.backend().binary(out, BinaryOp::Div, lhs, rhs, s.takeStream());
  return out;
}
Tensor div(const Tensor& lhs, double rhs, StreamOrDevice s) {
  auto stream = s.takeStream();
  Tensor out;
  lhs.backend().binary(out, BinaryOp::Div,
      lhs, lhs.backend().full(lhs.shape(), rhs, promoteScalarDtype(lhs.type(), rhs)),
      std::move(stream));
  return out;
}
Tensor div(double lhs, const Tensor& rhs, StreamOrDevice s) {
  auto stream = s.takeStream();
  Tensor out;
  rhs.backend().binary(out, BinaryOp::Div,
      rhs.backend().full(rhs.shape(), lhs, promoteScalarDtype(rhs.type(), lhs)), rhs,
      std::move(stream));
  return out;
}
Tensor operator/(const Tensor& lhs, const Tensor& rhs) {
  if (lhs.requiresGrad() || rhs.requiresGrad()) {
    return autograd::detail::autogradDiv(lhs, rhs);
  }
  return div(lhs, rhs);
}
Tensor operator/(const Tensor& lhs, double rhs) {
  if (lhs.requiresGrad()) {
    return autograd::detail::autogradDiv(lhs, rhs);
  }
  return div(lhs, rhs);
}
Tensor operator/(double lhs, const Tensor& rhs) {
  if (rhs.requiresGrad()) {
    return autograd::detail::autogradDiv(lhs, rhs);
  }
  return div(lhs, rhs);
}

/********** Non-differentiable operators **********/
FL_BINARY_OP_DEF(==, eq, BinaryOp::Eq);
FL_BINARY_OP_DEF(!=, neq, BinaryOp::Neq);
FL_BINARY_OP_DEF(<, lessThan, BinaryOp::Lt);
FL_BINARY_OP_DEF(<=, lessThanEqual, BinaryOp::Lte);
FL_BINARY_OP_DEF(>, greaterThan, BinaryOp::Gt);
FL_BINARY_OP_DEF(>=, greaterThanEqual, BinaryOp::Gte);
FL_BINARY_OP_DEF(||, logicalOr, BinaryOp::Or);
FL_BINARY_OP_DEF(&&, logicalAnd, BinaryOp::And);
FL_BINARY_OP_DEF(%, mod, BinaryOp::Mod);
FL_BINARY_OP_DEF(&, bitwiseAnd, BinaryOp::BitAnd);
FL_BINARY_OP_DEF(|, bitwiseOr, BinaryOp::BitOr);
FL_BINARY_OP_DEF(^, bitwiseXor, BinaryOp::BitXor);
FL_BINARY_OP_DEF(<<, lShift, BinaryOp::Shl);
FL_BINARY_OP_DEF(>>, rShift, BinaryOp::Shr);
#undef FL_BINARY_OP_DEF

Tensor minimum(const Tensor& lhs, const Tensor& rhs, StreamOrDevice s) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(lhs, rhs);
  Tensor out;
  lhs.backend().binary(out, BinaryOp::Min, lhs, rhs, s.takeStream());
  return out;
}

Tensor maximum(const Tensor& lhs, const Tensor& rhs, StreamOrDevice s) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(lhs, rhs);
  Tensor out;
  lhs.backend().binary(out, BinaryOp::Max, lhs, rhs, s.takeStream());
  return out;
}

Tensor minimum(const Tensor& lhs, const double& rhs, StreamOrDevice s) {
  auto stream = s.takeStream();
  Tensor out;
  lhs.backend().binary(out, BinaryOp::Min,
      lhs, lhs.backend().full(lhs.shape(), rhs, promoteDtypes(lhs.type(), scalarDtype(rhs))),
      std::move(stream));
  return out;
}

Tensor minimum(const double& lhs, const Tensor& rhs, StreamOrDevice s) {
  auto stream = s.takeStream();
  Tensor out;
  rhs.backend().binary(out, BinaryOp::Min,
      rhs.backend().full(rhs.shape(), lhs, promoteDtypes(rhs.type(), scalarDtype(lhs))), rhs,
      std::move(stream));
  return out;
}

Tensor maximum(const Tensor& lhs, const double& rhs, StreamOrDevice s) {
  auto stream = s.takeStream();
  Tensor out;
  lhs.backend().binary(out, BinaryOp::Max,
      lhs, lhs.backend().full(lhs.shape(), rhs, promoteDtypes(lhs.type(), scalarDtype(rhs))),
      std::move(stream));
  return out;
}

Tensor maximum(const double& lhs, const Tensor& rhs, StreamOrDevice s) {
  auto stream = s.takeStream();
  Tensor out;
  rhs.backend().binary(out, BinaryOp::Max,
      rhs.backend().full(rhs.shape(), lhs, promoteDtypes(rhs.type(), scalarDtype(lhs))), rhs,
      std::move(stream));
  return out;
}

Tensor power(const Tensor& lhs, const Tensor& rhs, StreamOrDevice s) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(lhs, rhs);
  Tensor out;
  lhs.backend().binary(out, BinaryOp::Pow, lhs, rhs, s.takeStream());
  return out;
}

Tensor power(const Tensor& lhs, const double& rhs, StreamOrDevice s) {
  auto stream = s.takeStream();
  Tensor out;
  lhs.backend().binary(out, BinaryOp::Pow,
      lhs, lhs.backend().full(lhs.shape(), rhs, promoteDtypes(lhs.type(), scalarDtype(rhs))),
      std::move(stream));
  return out;
}

Tensor power(const double& lhs, const Tensor& rhs, StreamOrDevice s) {
  auto stream = s.takeStream();
  Tensor out;
  rhs.backend().binary(out, BinaryOp::Pow,
      rhs.backend().full(rhs.shape(), lhs, promoteDtypes(rhs.type(), scalarDtype(lhs))), rhs,
      std::move(stream));
  return out;
}

/******************************* BLAS ********************************/
Tensor matmul(
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp,
    StreamOrDevice s) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(lhs, rhs);
  Tensor out;
  lhs.backend().matmul(out, lhs, rhs, lhsProp, rhsProp, s.takeStream());
  return out;
}

/************************** Reductions ***************************/

Tensor amin(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */,
    StreamOrDevice s) {
  Tensor out;
  input.backend().reduction(
      out, ReductionOp::Amin, input, axes, keepDims, s.takeStream());
  return out;
}

Tensor amax(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */,
    StreamOrDevice s) {
  Tensor out;
  input.backend().reduction(
      out, ReductionOp::Amax, input, axes, keepDims, s.takeStream());
  return out;
}

void min(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims,
    StreamOrDevice s) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(values, indices, input);
  return input.backend().min(values, indices, input, axis, keepDims, s.takeStream());
}

void max(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims /* = false */,
    StreamOrDevice s) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(values, indices, input);
  return input.backend().max(values, indices, input, axis, keepDims, s.takeStream());
}

Tensor sum(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */,
    StreamOrDevice s) {
  Tensor out;
  input.backend().reduction(
      out, ReductionOp::Sum, input, axes, keepDims, s.takeStream());
  return out;
}

Tensor cumsum(const Tensor& input, const unsigned axis, StreamOrDevice s) {
  Tensor out;
  input.backend().cumsum(out, input, axis, s.takeStream());
  return out;
}

Tensor argmax(
    const Tensor& input,
    const unsigned axis,
    const bool keepDims /* = false */,
    StreamOrDevice s) {
  Tensor out;
  input.backend().argmax(out, input, axis, keepDims, s.takeStream());
  return out;
}

Tensor argmin(
    const Tensor& input,
    const unsigned axis,
    const bool keepDims /* = false */,
    StreamOrDevice s) {
  Tensor out;
  input.backend().argmin(out, input, axis, keepDims, s.takeStream());
  return out;
}

Tensor mean(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */,
    StreamOrDevice s) {
  Tensor out;
  input.backend().reduction(
      out, ReductionOp::Mean, input, axes, keepDims, s.takeStream());
  return out;
}

Tensor median(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */,
    StreamOrDevice s) {
  Tensor out;
  input.backend().reduction(
      out, ReductionOp::Median, input, axes, keepDims, s.takeStream());
  return out;
}

Tensor var(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    const bool bias,
    const bool keepDims /* = false */,
    StreamOrDevice s) {
  Tensor out;
  input.backend().var(out, input, axes, bias, keepDims, s.takeStream());
  return out;
}

Tensor std(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */,
    StreamOrDevice s) {
  Tensor out;
  input.backend().reduction(
      out, ReductionOp::Std, input, axes, keepDims, s.takeStream());
  return out;
}

Tensor norm(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    double p /* = 2 */,
    const bool keepDims /* = false */,
    StreamOrDevice s) {
  Tensor out;
  input.backend().norm(out, input, axes, p, keepDims, s.takeStream());
  return out;
}

Tensor countNonzero(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */,
    StreamOrDevice s) {
  Tensor out;
  input.backend().reduction(
      out, ReductionOp::CountNonzero, input, axes, keepDims, s.takeStream());
  return out;
}

Tensor any(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */,
    StreamOrDevice s) {
  Tensor out;
  input.backend().reduction(
      out, ReductionOp::Any, input, axes, keepDims, s.takeStream());
  return out;
}

Tensor all(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */,
    StreamOrDevice s) {
  Tensor out;
  input.backend().reduction(
      out, ReductionOp::All, input, axes, keepDims, s.takeStream());
  return out;
}

/************************** Utilities ***************************/

std::ostream& operator<<(std::ostream& ostr, const Tensor& t) {
  t.operator<<(ostr);
  return ostr;
}

void print(const Tensor& tensor) {
  tensor.backend().print(tensor);
}

bool allClose(
    const fl::Tensor& a,
    const fl::Tensor& b,
    const double absTolerance) {
  if (a.type() != b.type()) {
    return false;
  }
  if (a.shape() != b.shape()) {
    return false;
  }
  if (a.elements() == 0 && b.elements() == 0) {
    return true;
  }
  return fl::amax(fl::abs(a - b)).astype(dtype::f64).scalarSync<double>() <
      absTolerance;
}

bool isInvalidArray(const Tensor& tensor) {
  return fl::any(fl::isnan(tensor)).asScalar<bool>() ||
      fl::any(fl::isinf(tensor)).asScalar<bool>();
}

std::string tensorBackendTypeToString(const TensorBackendType type) {
  switch (type) {
    case TensorBackendType::Stub:
      return "Stub";
    case TensorBackendType::Tracer:
      return "Tracer";
    case TensorBackendType::ArrayFire:
      return "ArrayFire";
    case TensorBackendType::CPU:
      return "CPU";
    case TensorBackendType::Jit:
      return "Jit";
    case TensorBackendType::Cuda:
      return "Cuda";
  }
  throw std::runtime_error("Unreachable -- unrecognized tensor backend type");
}

std::ostream& operator<<(std::ostream& os, const TensorBackendType type) {
  os << tensorBackendTypeToString(type);
  return os;
}

namespace detail {

std::unique_ptr<TensorAdapterBase> releaseAdapter(Tensor&& t) {
  return t.releaseAdapter();
}

std::unique_ptr<TensorAdapterBase> releaseAdapterUnsafe(Tensor& t) {
  return t.releaseAdapter();
}

bool areTensorTypesEqual(const Tensor& a, const Tensor& b) {
  return a.type() == b.type();
}

Tensor tileAs(const Tensor& input, const Shape& rdims) {
  // Scalar tensor
  if (input.ndim() == 0) {
    return fl::tile(input, rdims);
  }

  // Right-align: input dims align with trailing target dims
  Shape dims(std::vector<Dim>(rdims.ndim(), 1));
  Shape idims = input.shape();
  int offset = rdims.ndim() - idims.ndim();
  for (int i = 0; i < rdims.ndim(); i++) {
    int ii = i - offset;
    int idimsSize = ii >= 0 ? idims[ii] : 1;
    if (rdims[i] % idimsSize != 0) {
      std::stringstream ss;
      ss << "Invalid dims for tileAs for input dims " << idims
         << " to output dims " << rdims;
      throw std::invalid_argument(ss.str());
    }
    dims[i] = rdims[i] / idimsSize;
  }
  return fl::tile(input, dims);
}

Tensor sumAs(const Tensor& input, const Shape& rdims) {
  Shape idims = input.shape();
  auto result = input;
  // Right-align: input trailing dims correspond to rdims trailing dims
  int offset = input.ndim() - rdims.ndim();
  for (int i = 0; i < input.ndim(); i++) {
    int ri = i - offset;
    if (ri < 0 || idims[i] != rdims[ri]) {
      result = fl::sum(result, {i}, /* keepDims = */ true);
    }
  }

  return fl::reshape(result.astype(input.type()), rdims);
}

Shape expandedShapeFromReducedDims(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims /* = false */) {
  // Fast path - tensor already retained its shape
  if (keepDims) {
    return input.shape();
  }
  // If we output a scalar,
  if (input.ndim() == 0) {
    return {};
  }

  unsigned preNDims = input.ndim() + axes.size();
  Shape newShape(std::vector<Dim>(preNDims, 1));
  unsigned axesIdx = 0;
  unsigned inputIdx = 0;
  for (unsigned i = 0; i < preNDims; ++i) {
    if (i == axes[axesIdx]) {
      // This dim was reduced over, leave as 1 in the new shape
      axesIdx++;
    } else {
      // Dim wasn't reduced over - add the shape from the new tensor
      newShape[i] = input.dim(inputIdx);
      inputIdx++;
    }
  }
  return newShape;
}

Tensor expandFromReduction(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims) {
  return fl::reshape(
      input, expandedShapeFromReducedDims(input, axes, keepDims));
}

} // namespace detail

} // namespace fl
