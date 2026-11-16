/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/stub/StubBackend.h"

#include "flashlight/fl/tensor/BackendRegistry.h"

#include <stdexcept>

#include "flashlight/fl/tensor/Tensor.h"

#define FL_STUB_BACKEND_UNIMPLEMENTED \
  throw std::invalid_argument(        \
      "StubBackend::" + std::string(__func__) + " - unimplemented.");

namespace fl {

FL_REGISTER_TENSOR_BACKEND(Stub, StubBackend::getInstance);

StubBackend::StubBackend() {
  // Set up state
}

StubBackend& StubBackend::getInstance() {
  static StubBackend instance;
  return instance;
}

TensorBackendType StubBackend::backendType() const {
  // Implementers of a backend should create their own option in the
  // TensorBackendType enum and return it here.
  return TensorBackendType::Stub;
}

/* -------------------------- Compute Functions -------------------------- */

void StubBackend::eval(const Tensor& /* tensor */) {
  // Launch computation for a given tensor. Can be a noop for non-async
  // runtimes.
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

bool StubBackend::supportsDataType(const fl::dtype& /* dtype */) const {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::getMemMgrInfo(
    const char* /* msg */,
    const int /* deviceId */,
    std::ostream* /* ostream */) {
  // Can be a noop if no memory manager is implemented.
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::setMemMgrLogStream(std::ostream* /* stream */) {
  // Can be a noop if no memory manager is implemented.
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::setMemMgrLoggingEnabled(const bool /* enabled */) {
  // Can be a noop if no memory manager is implemented.
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::setMemMgrFlushInterval(const size_t /* interval */) {
  // Can be a noop if no memory manager is implemented.
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

/* -------------------------- Rand Functions -------------------------- */

void StubBackend::setSeed(const int /* seed */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::randn(
    const Shape& /* shape */,
    dtype /* type */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::rand(
    const Shape& /* shape */,
    dtype /* type */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

/* --------------------------- Tensor Operators --------------------------- */

/******************** Tensor Creation Functions ********************/
Tensor StubBackend::fromScalar(
    double /* value */,
    dtype /* type */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::full(
    const Shape& /* shape */,
    double /* value */,
    dtype /* type */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::empty(
    const Shape& /* shape */,
    dtype /* type */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::identity(
    const Dim /* dim */,
    const dtype /* type */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::arange(
    const Shape& /* shape */,
    const Dim /* seqDim */,
    const dtype /* type */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::iota(
    const Shape& /* dims */,
    const Shape& /* tileDims */,
    const dtype /* type */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

/************************ Shaping and Indexing *************************/
void StubBackend::reshape(
    Tensor& /* out */,
    const Tensor& /* tensor */,
    const Shape& /* shape */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::transpose(
    Tensor& /* out */,
    const Tensor& /* tensor */,
    const Shape& /* axes */ /* = {} */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::tile(
    Tensor& /* out */,
    const Tensor& /* tensor */,
    const Shape& /* shape */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::concatenate(
    Tensor& /* out */,
    const std::vector<Tensor>& /* tensors */,
    const unsigned /* axis */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::nonzero(
    Tensor& /* out */,
    const Tensor& /* tensor */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::pad(
    Tensor& /* out */,
    const Tensor& /* input */,
    const std::vector<std::pair<int, int>>& /* padWidths */,
    const PadType /* type */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

/************************** Unary Operators ***************************/

Tensor StubBackend::exp(
    const Tensor& /* tensor */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::log(
    const Tensor& /* tensor */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::negative(
    const Tensor& /* tensor */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::logicalNot(
    const Tensor& /* tensor */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::log1p(
    const Tensor& /* tensor */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::sin(
    const Tensor& /* tensor */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::cos(
    const Tensor& /* tensor */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::sqrt(
    const Tensor& /* tensor */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::tanh(
    const Tensor& /* tensor */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::floor(
    const Tensor& /* tensor */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::ceil(
    const Tensor& /* tensor */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::rint(
    const Tensor& /* tensor */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::absolute(
    const Tensor& /* tensor */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::sigmoid(
    const Tensor& /* tensor */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::erf(
    const Tensor& /* tensor */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::flip(
    Tensor& /* out */,
    const Tensor& /* tensor */,
    const unsigned /* dim */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::clip(
    Tensor& /* out */,
    const Tensor& /* tensor */,
    const Tensor& /* low */,
    const Tensor& /* high */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::roll(
    Tensor& /* out */,
    const Tensor& /* tensor */,
    const int /* shift */,
    const unsigned /* axis */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::isnan(
    const Tensor& /* tensor */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::isinf(
    const Tensor& /* tensor */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::sign(
    const Tensor& /* tensor */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::tril(
    Tensor& /* out */,
    const Tensor& /* tensor */,
    int /* diagonal */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::triu(
    Tensor& /* out */,
    const Tensor& /* tensor */,
    int /* diagonal */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::where(
    Tensor& /* out */,
    const Tensor& /* condition */,
    const Tensor& /* x */,
    const Tensor& /* y */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::topk(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const unsigned /* k */,
    const Dim /* axis */,
    const SortMode /* sortMode */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::sort(
    Tensor& /* out */,
    const Tensor& /* input */,
    const Dim /* axis */,
    const SortMode /* sortMode */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::sort(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const Dim /* axis */,
    const SortMode /* sortMode */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::argsort(
    Tensor& /* out */,
    const Tensor& /* input */,
    const Dim /* axis */,
    const SortMode /* sortMode */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

/************************** Binary Operators ***************************/
#define FL_STUB_BINARY_OP_DEF(FUNC)                                            \
  Tensor StubBackend::FUNC(                                                    \
      const Tensor& /* lhs */,                                                 \
      const Tensor& /* rhs */,                                                 \
      std::shared_ptr<const Stream> /* stream */) {                            \
    throw std::runtime_error(                                                  \
        "StubBackend::" + std::string(#FUNC) +                                 \
        " unimplemented for two-Tensor inputs.");                              \
  }

FL_STUB_BINARY_OP_DEF(add);
FL_STUB_BINARY_OP_DEF(sub);
FL_STUB_BINARY_OP_DEF(mul);
FL_STUB_BINARY_OP_DEF(div);
FL_STUB_BINARY_OP_DEF(eq);
FL_STUB_BINARY_OP_DEF(neq);
FL_STUB_BINARY_OP_DEF(lessThan);
FL_STUB_BINARY_OP_DEF(lessThanEqual);
FL_STUB_BINARY_OP_DEF(greaterThan);
FL_STUB_BINARY_OP_DEF(greaterThanEqual);
FL_STUB_BINARY_OP_DEF(logicalOr);
FL_STUB_BINARY_OP_DEF(logicalAnd);
FL_STUB_BINARY_OP_DEF(mod);
FL_STUB_BINARY_OP_DEF(bitwiseAnd);
FL_STUB_BINARY_OP_DEF(bitwiseOr);
FL_STUB_BINARY_OP_DEF(bitwiseXor);
FL_STUB_BINARY_OP_DEF(lShift);
FL_STUB_BINARY_OP_DEF(rShift);
#undef FL_STUB_BINARY_OP_DEF

Tensor StubBackend::minimum(
    const Tensor& /* lhs */,
    const Tensor& /* rhs */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::maximum(
    const Tensor& /* lhs */,
    const Tensor& /* rhs */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::power(
    const Tensor& /* lhs */,
    const Tensor& /* rhs */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

/************************** BLAS ***************************/

void StubBackend::matmul(
    Tensor& /* out */,
    const Tensor& /* lhs */,
    const Tensor& /* rhs */,
    MatrixProperty /* lhsProp */,
    MatrixProperty /* rhsProp */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

/************************** Reductions ***************************/

Tensor StubBackend::amin(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::amax(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::min(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::max(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::sum(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::cumsum(
    Tensor& /* out */,
    const Tensor& /* input */,
    const unsigned /* axis */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::argmax(
    Tensor& /* out */,
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::argmin(
    Tensor& /* out */,
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::mean(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::median(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::var(
    Tensor& /* out */,
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* bias */,
    const bool /* keepDims */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::std(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::norm(
    Tensor& /* out */,
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    double /* p */ /* = 2 */,
    const bool /* keepDims */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::countNonzero(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::any(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::all(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */,
    std::shared_ptr<const Stream> /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::copy(
    Tensor& /* dst */, const Tensor& /* src */, const Stream& /* stream */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::print(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::binary(Tensor& /* out */, BinaryOp, const Tensor&,
    const Tensor&, std::shared_ptr<const Stream>) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::unary(Tensor& /* out */, UnaryOp, const Tensor&,
    std::shared_ptr<const Stream>) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::reduction(Tensor& /* out */, ReductionOp, const Tensor&,
    const std::vector<int>&, bool,
    std::shared_ptr<const Stream>) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

} // namespace fl
