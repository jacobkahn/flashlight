/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/cpu/CPUBackend.h"

#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/ops/native/Creation.h"

namespace fl {

Tensor CPUBackend::fromScalar(double value, dtype type,
    std::shared_ptr<const Stream> /* stream */) {
  return full(Shape({}), value, type);
}

Tensor CPUBackend::full(
    const Shape& shape, double value, dtype type,
    std::shared_ptr<const Stream> /* stream */) {
  switch (type) {
    case dtype::f16:
      return fullWithType<float>(shape, value, dtype::f32)
          .astype(dtype::f16);
    case dtype::bf16:
      return fullWithType<float>(shape, value, dtype::f32)
          .astype(dtype::bf16);
    case dtype::f32:
      return fullWithType<float>(shape, value, type);
    case dtype::f64:
      return fullWithType<double>(shape, value, type);
    case dtype::b8:
      return fullWithType<char>(shape, value, type);
    case dtype::s16:
      return fullWithType<short>(shape, value, type);
    case dtype::s32:
      return fullWithType<int>(shape, value, type);
    case dtype::s64:
      return fullWithType<long long>(shape, value, type);
    case dtype::u8:
      return fullWithType<unsigned char>(shape, value, type);
    case dtype::u16:
      return fullWithType<unsigned short>(shape, value, type);
    case dtype::u32:
      return fullWithType<unsigned int>(shape, value, type);
    case dtype::u64:
      return fullWithType<unsigned long long>(shape, value, type);
  }
  throw std::invalid_argument("CPUBackend::full: unsupported dtype");
}

template <typename T, typename V>
Tensor
CPUBackend::fullWithType(const Shape& shape, V value, const dtype type) {
  return fl::native::fullWithTypeCpu<T, V>(shape, value, type);
}

Tensor CPUBackend::empty(
    const Shape& shape, dtype type, std::shared_ptr<const Stream> /* stream */) {
  return toTensor<CPUTensor>(shape, type, nullptr, Location::Host);
}

Tensor CPUBackend::identity(const Dim dim, const dtype type, std::shared_ptr<const Stream> /* stream */) {
  return fl::native::identityCpu(dim, type);
}

Tensor
CPUBackend::arange(const Shape& shape, const Dim seqDim, const dtype type, std::shared_ptr<const Stream> /* stream */) {
  if (seqDim < 0 || seqDim >= shape.ndim()) {
    std::ostringstream oss;
    oss << "[CPUBackend::arange] Invalid seqDim: " << seqDim
        << ", for shape: " << shape;
    throw std::invalid_argument(oss.str());
  }
  std::vector<Dim> tileDims = shape.get();
  tileDims[seqDim] = 1;
  Tensor result;
  tile(result,
      fl::native::iotaSingleAxisCpu(shape.ndim(), seqDim, shape[seqDim], type),
      Shape(tileDims));
  return result;
}

Tensor CPUBackend::iota(
    const Shape& dims,
    const Shape& tileDims,
    const dtype type,
    std::shared_ptr<const Stream> /* stream */) {
  // Create a tensor with sequential values [0, N) then tile
  size_t n = dims.elements();
  std::vector<float> data(n);
  std::iota(data.begin(), data.end(), 0.0f);
  auto base = toTensor<CPUTensor>(dims, dtype::f32, data.data(), Location::Host);
  Tensor result;
  tile(result, base, tileDims);
  if (type != dtype::f32) {
    return result.astype(type);
  }
  return result;
}

} // namespace fl
