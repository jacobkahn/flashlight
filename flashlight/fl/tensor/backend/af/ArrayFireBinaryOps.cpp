/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"

#include <stdexcept>

#include <af/arith.h>
#include <af/data.h>
#include <af/gfor.h>

#include "flashlight/fl/tensor/OpEnums.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"

namespace fl {
namespace {

// A binary operation on two ArrayFire arrays
using binaryOpFunc_t =
    af::array (*)(const af::array& lhs, const af::array& rhs);

// Promote an AF array to a target ndim by reshaping so data sits at the
// correct high AF dims (AF is col-major, FL is row-major). This ensures
// that corresponding FL dimensions map to the same AF positions for
// broadcasting.
af::array promoteAfArray(const Tensor& t, int targetNdim) {
  auto arr = toArray(t);
  if (t.ndim() < targetNdim) {
    // Right-align: in FL, shorter shape gets 1s prepended. In AF (with full
    // reversal), this means data stays at LOW AF dims and new 1-dims are at
    // high AF dims. Since AF dim4 already has trailing 1s, the data is
    // already in the right place -- just reshape to set the right dimensions.
    af::dim4 paddedDims(1, 1, 1, 1);
    auto origDims = arr.dims();
    for (int i = 0; i < t.ndim(); ++i) {
      paddedDims[i] = origDims[i];
    }
    arr = af::moddims(arr, paddedDims);
  }
  return arr;
}

Tensor doBinaryOpOrBroadcast(
    const Tensor& lhs,
    const Tensor& rhs,
    binaryOpFunc_t func) {
  // Dims are the same or scalar <> 1-el tensor - no broadcasting
  if (lhs.shape() == rhs.shape() ||
      (lhs.elements() <= 1 && rhs.elements() <= 1)) {
    return toTensor<ArrayFireTensor>(
        func(toArray(lhs), toArray(rhs)), lhs.ndim());
  }

  // broadcastShapes throws if shapes are incompatible
  auto outShape = broadcastShapes(lhs.shape(), rhs.shape());
  int outNdim = outShape.ndim();

  // When inputs have different ndims, promote the shorter one so that
  // corresponding FL dimensions align in AF space.
  return toTensor<ArrayFireTensor>(
      af::batchFunc(
          promoteAfArray(lhs, outNdim),
          promoteAfArray(rhs, outNdim),
          func),
      outNdim);
}
} // namespace

// Operations on fl::Tensor call the respective operator overloads that are
// already defined on af::arrays. Scalar overloads are handled by default
// implementations in TensorBackend.
#define FL_AF_BINARY_OP_DEF(OP, FUNC)                                   \
  Tensor ArrayFireBackend::FUNC(const Tensor& lhs, const Tensor& rhs,  \
      std::shared_ptr<const Stream> /* stream */) {                     \
    return doBinaryOpOrBroadcast(lhs, rhs, af::operator OP);            \
  }

// Definitions
FL_AF_BINARY_OP_DEF(+, add);
FL_AF_BINARY_OP_DEF(-, sub);
FL_AF_BINARY_OP_DEF(*, mul);
FL_AF_BINARY_OP_DEF(/, div);
FL_AF_BINARY_OP_DEF(==, eq);
FL_AF_BINARY_OP_DEF(!=, neq);
FL_AF_BINARY_OP_DEF(<, lessThan);
FL_AF_BINARY_OP_DEF(<=, lessThanEqual);
FL_AF_BINARY_OP_DEF(>, greaterThan);
FL_AF_BINARY_OP_DEF(>=, greaterThanEqual);
FL_AF_BINARY_OP_DEF(||, logicalOr);
FL_AF_BINARY_OP_DEF(&&, logicalAnd);
FL_AF_BINARY_OP_DEF(%, mod);
FL_AF_BINARY_OP_DEF(&, bitwiseAnd);
FL_AF_BINARY_OP_DEF(|, bitwiseOr);
FL_AF_BINARY_OP_DEF(^, bitwiseXor);
FL_AF_BINARY_OP_DEF(<<, lShift);
FL_AF_BINARY_OP_DEF(>>, rShift);
#undef FL_AF_BINARY_OP_DEF

Tensor ArrayFireBackend::minimum(const Tensor& lhs, const Tensor& rhs,
    std::shared_ptr<const Stream> /* stream */) {
  return doBinaryOpOrBroadcast(lhs, rhs, af::min);
}

Tensor ArrayFireBackend::maximum(const Tensor& lhs, const Tensor& rhs,
    std::shared_ptr<const Stream> /* stream */) {
  return doBinaryOpOrBroadcast(lhs, rhs, af::max);
}

Tensor ArrayFireBackend::power(const Tensor& lhs, const Tensor& rhs,
    std::shared_ptr<const Stream> /* stream */) {
  return doBinaryOpOrBroadcast(lhs, rhs, af::pow);
}

void ArrayFireBackend::binary(
    Tensor& out,
    BinaryOp op,
    const Tensor& lhs,
    const Tensor& rhs,
    std::shared_ptr<const Stream> stream) {
  switch (op) {
    case BinaryOp::Add: out = add(lhs, rhs, std::move(stream)); return;
    case BinaryOp::Sub: out = sub(lhs, rhs, std::move(stream)); return;
    case BinaryOp::Mul: out = mul(lhs, rhs, std::move(stream)); return;
    case BinaryOp::Div: out = div(lhs, rhs, std::move(stream)); return;
    case BinaryOp::Eq: out = eq(lhs, rhs, std::move(stream)); return;
    case BinaryOp::Neq: out = neq(lhs, rhs, std::move(stream)); return;
    case BinaryOp::Gt: out = greaterThan(lhs, rhs, std::move(stream)); return;
    case BinaryOp::Gte: out = greaterThanEqual(lhs, rhs, std::move(stream)); return;
    case BinaryOp::Lt: out = lessThan(lhs, rhs, std::move(stream)); return;
    case BinaryOp::Lte: out = lessThanEqual(lhs, rhs, std::move(stream)); return;
    case BinaryOp::Max: out = maximum(lhs, rhs, std::move(stream)); return;
    case BinaryOp::Min: out = minimum(lhs, rhs, std::move(stream)); return;
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
  throw std::runtime_error(
      "[ArrayFireBackend::binary] Unknown binary operation type");
}
} // namespace fl
