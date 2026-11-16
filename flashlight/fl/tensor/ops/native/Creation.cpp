/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/native/Creation.h"

#include <numeric>
#include <stdexcept>
#include <vector>

#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/ops/native/CpuTensorHelpers.h"

namespace fl {
namespace native {

namespace {

template <typename T>
Tensor iotaWithTypeCpu(const Shape& shape, const dtype type) {
  std::vector<T> data(shape.elements());
  std::iota(data.begin(), data.end(), 0);
  return makeCpuTensor(shape, data.data());
}

} // namespace

Tensor iotaSingleAxisCpu(
    const unsigned ndims,
    const unsigned axis,
    const unsigned axisDim,
    const dtype type) {
  std::vector<Dim> dims(ndims, 1);
  dims[axis] = axisDim;
  const Shape shape(dims);
  switch (type) {
    case dtype::f16:
      return iotaWithTypeCpu<float>(shape, dtype::f32).astype(dtype::f16);
    case dtype::f32:
      return iotaWithTypeCpu<float>(shape, type);
    case dtype::f64:
      return iotaWithTypeCpu<double>(shape, type);
    case dtype::b8:
      return iotaWithTypeCpu<char>(shape, type);
    case dtype::s16:
      return iotaWithTypeCpu<short>(shape, type);
    case dtype::s32:
      return iotaWithTypeCpu<int>(shape, type);
    case dtype::s64:
      return iotaWithTypeCpu<long long>(shape, type);
    case dtype::u8:
      return iotaWithTypeCpu<unsigned char>(shape, type);
    case dtype::u16:
      return iotaWithTypeCpu<unsigned short>(shape, type);
    case dtype::u32:
      return iotaWithTypeCpu<unsigned int>(shape, type);
    case dtype::u64:
      return iotaWithTypeCpu<unsigned long long>(shape, type);
    default:
      break;
  }
  throw std::invalid_argument("iotaSingleAxisCpu: unsupported dtype");
}

Tensor identityCpu(const Dim dim, const dtype type) {
  std::vector<float> data(dim * dim, 0.0f);
  for (Dim i = 0; i < dim; ++i) {
    data[i * dim + i] = 1.0f;
  }
  return makeCpuTensor(Shape({dim, dim}), data.data())
      .astype(type);
}

} // namespace native
} // namespace fl
