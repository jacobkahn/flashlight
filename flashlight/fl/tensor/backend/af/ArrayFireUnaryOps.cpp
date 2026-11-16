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

#include "flashlight/fl/tensor/OpEnums.h"
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#include "flashlight/fl/tensor/backend/af/Utils.h"

namespace fl {

Tensor ArrayFireBackend::exp(const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  return toTensor<ArrayFireTensor>(af::exp(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::log(const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  return toTensor<ArrayFireTensor>(af::log(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::negative(const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  return toTensor<ArrayFireTensor>(-toArray(tensor), tensor.ndim());
}

Tensor ArrayFireBackend::logicalNot(const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  return toTensor<ArrayFireTensor>(!toArray(tensor), tensor.ndim());
}

Tensor ArrayFireBackend::log1p(const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  return toTensor<ArrayFireTensor>(af::log1p(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::sin(const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  return toTensor<ArrayFireTensor>(af::sin(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::cos(const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  return toTensor<ArrayFireTensor>(af::cos(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::sqrt(const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  return toTensor<ArrayFireTensor>(af::sqrt(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::tanh(const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  return toTensor<ArrayFireTensor>(af::tanh(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::floor(const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  return toTensor<ArrayFireTensor>(af::floor(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::ceil(const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  return toTensor<ArrayFireTensor>(af::ceil(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::rint(const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  return toTensor<ArrayFireTensor>(af::round(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::absolute(const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  return toTensor<ArrayFireTensor>(af::abs(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::sigmoid(const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  return toTensor<ArrayFireTensor>(af::sigmoid(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::erf(const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  return toTensor<ArrayFireTensor>(af::erf(toArray(tensor)), tensor.ndim());
}

void ArrayFireBackend::flip(Tensor& out, const Tensor& tensor, const unsigned dim,
    std::shared_ptr<const Stream> /* stream */) {
  unsigned afDim = detail::flToAfAxis(dim, tensor.ndim());
  out = toTensor<ArrayFireTensor>(
      af::flip(toArray(tensor), afDim), tensor.ndim());
}

void ArrayFireBackend::clip(
    Tensor& out,
    const Tensor& tensor,
    const Tensor& low,
    const Tensor& high,
    std::shared_ptr<const Stream> /* stream */) {
  out = toTensor<ArrayFireTensor>(
      af::clamp(toArray(tensor), toArray(low), toArray(high)), tensor.ndim());
}

void ArrayFireBackend::roll(
    Tensor& out,
    const Tensor& tensor,
    const int shift,
    const unsigned axis,
    std::shared_ptr<const Stream> /* stream */) {
  if (axis > AF_MAX_DIMS) {
    throw std::invalid_argument(
        "ArrayFireBackend::roll - given axis > 3 - unsupported");
  }
  unsigned afAxis = detail::flToAfAxis(axis, tensor.ndim());
  std::vector<Dim> shifts(AF_MAX_DIMS, 0);
  shifts[afAxis] = shift;
  out = toTensor<ArrayFireTensor>(
      af::shift(toArray(tensor), shifts[0], shifts[1], shifts[2], shifts[3]),
      tensor.ndim());
}

Tensor ArrayFireBackend::isnan(const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  return toTensor<ArrayFireTensor>(af::isNaN(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::isinf(const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  return toTensor<ArrayFireTensor>(af::isInf(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::sign(const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  const auto& arr = toArray(tensor);
  auto result = af::select(arr == 0, 0.0, 1 - 2 * af::sign(arr));
  return toTensor<ArrayFireTensor>(std::move(result), tensor.ndim());
}

void ArrayFireBackend::tril(Tensor& out, const Tensor& tensor, int diagonal,
    std::shared_ptr<const Stream> /* stream */) {
  if (diagonal != 0) {
    throw std::invalid_argument(
        "ArrayFireBackend::tril only supports diagonal = 0");
  }
  auto arr = toArray(tensor);
  unsigned nd = tensor.ndim();
  if (nd < 2) {
    throw std::invalid_argument("tril requires input with rank at least 2");
  }
  if (nd != 2) {
    throw std::invalid_argument(
        "ArrayFireBackend::tril only supports rank-2 inputs");
  }
  // 2D: FL(i,j) = AF(j,i). af::upper keeps r<=c --> j<=i --> FL lower triangle.
  out = toTensor<ArrayFireTensor>(
      af::upper(arr, /* is_unit_diag = */ false), nd);
}

void ArrayFireBackend::triu(Tensor& out, const Tensor& tensor, int diagonal,
    std::shared_ptr<const Stream> /* stream */) {
  if (diagonal != 0) {
    throw std::invalid_argument(
        "ArrayFireBackend::triu only supports diagonal = 0");
  }
  auto arr = toArray(tensor);
  unsigned nd = tensor.ndim();
  if (nd < 2) {
    throw std::invalid_argument("triu requires input with rank at least 2");
  }
  if (nd != 2) {
    throw std::invalid_argument(
        "ArrayFireBackend::triu only supports rank-2 inputs");
  }
  out = toTensor<ArrayFireTensor>(
      af::lower(arr, /* is_unit_diag = */ false), nd);
}

void ArrayFireBackend::unary(
    Tensor& out,
    UnaryOp op,
    const Tensor& input,
    std::shared_ptr<const Stream> stream) {
  switch (op) {
    case UnaryOp::Exp: out = exp(input, std::move(stream)); return;
    case UnaryOp::Log: out = log(input, std::move(stream)); return;
    case UnaryOp::Log1p: out = log1p(input, std::move(stream)); return;
    case UnaryOp::Sin: out = sin(input, std::move(stream)); return;
    case UnaryOp::Cos: out = cos(input, std::move(stream)); return;
    case UnaryOp::Sqrt: out = sqrt(input, std::move(stream)); return;
    case UnaryOp::Tanh: out = tanh(input, std::move(stream)); return;
    case UnaryOp::Floor: out = floor(input, std::move(stream)); return;
    case UnaryOp::Ceil: out = ceil(input, std::move(stream)); return;
    case UnaryOp::Rint: out = rint(input, std::move(stream)); return;
    case UnaryOp::Negative: out = negative(input, std::move(stream)); return;
    case UnaryOp::Absolute: out = absolute(input, std::move(stream)); return;
    case UnaryOp::Sigmoid: out = sigmoid(input, std::move(stream)); return;
    case UnaryOp::Erf: out = erf(input, std::move(stream)); return;
    case UnaryOp::LogicalNot: out = logicalNot(input, std::move(stream)); return;
    case UnaryOp::Sign: out = sign(input, std::move(stream)); return;
    case UnaryOp::Isnan: out = isnan(input, std::move(stream)); return;
    case UnaryOp::Isinf: out = isinf(input, std::move(stream)); return;
  }
  throw std::runtime_error(
      "[ArrayFireBackend::unary] Unknown unary operation type");
}
} // namespace fl
