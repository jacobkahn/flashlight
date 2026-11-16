/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/cudnn/CudnnTensorHelpers.h"

#include <stdexcept>

namespace fl {
namespace detail {

namespace {

TensorBackendType bareBackendOf(const Tensor& like) {
  const auto type = like.backendType();
  if (type == TensorBackendType::Jit) {
    throw std::invalid_argument(
        "cuDNN op received a JIT tensor; inputs must be materialized "
        "backend tensors");
  }
  return type;
}

} // namespace

Tensor makeTensorLike(const Tensor& like, const Shape& shape, fl::dtype type) {
  return detail::makeTensor(shape, type, bareBackendOf(like));
}

Tensor fullLike(
    const Tensor& like,
    const Shape& shape,
    double value,
    fl::dtype type) {
  return detail::full(shape, value, type, bareBackendOf(like));
}

} // namespace detail
} // namespace fl
