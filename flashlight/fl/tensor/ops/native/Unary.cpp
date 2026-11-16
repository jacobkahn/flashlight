/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/native/Unary.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/ops/native/CpuTensorHelpers.h"

namespace fl {
namespace native {

void flipImpl(Tensor& out, const Tensor& in, unsigned dim) {
  auto dims = in.shape().get();
  auto strides = rowMajorStrides(dims);
  std::vector<Dim> coords(dims.size());
  size_t n = in.shape().elements();
  size_t typeSize = fl::getTypeSize(in.type());
  auto* dst = static_cast<char*>(toCPUTensor(out).bufferPtr());
  const auto* src = static_cast<const char*>(toCPUTensor(in).bufferPtr());
  for (size_t i = 0; i < n; i++) {
    linearIndexToCoords(i, dims, strides, coords);
    coords[dim] = dims[dim] - 1 - coords[dim];
    std::memcpy(
        dst + coordsToLinearIndex(coords, strides) * typeSize,
        src + i * typeSize,
        typeSize);
  }
}

void rollImpl(Tensor& out, const Tensor& in, int shift, unsigned axis) {
  auto dims = in.shape().get();
  auto strides = rowMajorStrides(dims);
  std::vector<Dim> coords(dims.size());
  size_t n = in.shape().elements();
  Dim axisDim = dims[axis];
  size_t typeSize = fl::getTypeSize(in.type());
  auto* dst = static_cast<char*>(toCPUTensor(out).bufferPtr());
  const auto* src = static_cast<const char*>(toCPUTensor(in).bufferPtr());
  for (size_t i = 0; i < n; i++) {
    linearIndexToCoords(i, dims, strides, coords);
    coords[axis] = ((coords[axis] + shift) % axisDim + axisDim) % axisDim;
    std::memcpy(
        dst + coordsToLinearIndex(coords, strides) * typeSize,
        src + i * typeSize,
        typeSize);
  }
}

void trilImpl(Tensor& out, const Tensor& in, int diagonal) {
  auto dims = in.shape().get();
  if (dims.size() < 2) {
    throw std::invalid_argument("tril requires input with rank at least 2");
  }
  auto strides = rowMajorStrides(dims);
  std::vector<Dim> coords(dims.size());
  size_t n = in.shape().elements();
  size_t typeSize = fl::getTypeSize(in.type());
  auto* dst = static_cast<char*>(toCPUTensor(out).bufferPtr());
  const auto* src = static_cast<const char*>(toCPUTensor(in).bufferPtr());
  std::memcpy(dst, src, n * typeSize);
  const auto rowAxis = dims.size() - 2;
  const auto colAxis = dims.size() - 1;
  for (size_t i = 0; i < n; i++) {
    linearIndexToCoords(i, dims, strides, coords);
    if (coords[colAxis] - coords[rowAxis] > diagonal) {
      std::memset(dst + i * typeSize, 0, typeSize);
    }
  }
}

void triuImpl(Tensor& out, const Tensor& in, int diagonal) {
  auto dims = in.shape().get();
  if (dims.size() < 2) {
    throw std::invalid_argument("triu requires input with rank at least 2");
  }
  auto strides = rowMajorStrides(dims);
  std::vector<Dim> coords(dims.size());
  size_t n = in.shape().elements();
  size_t typeSize = fl::getTypeSize(in.type());
  auto* dst = static_cast<char*>(toCPUTensor(out).bufferPtr());
  const auto* src = static_cast<const char*>(toCPUTensor(in).bufferPtr());
  std::memcpy(dst, src, n * typeSize);
  const auto rowAxis = dims.size() - 2;
  const auto colAxis = dims.size() - 1;
  for (size_t i = 0; i < n; i++) {
    linearIndexToCoords(i, dims, strides, coords);
    if (coords[colAxis] - coords[rowAxis] < diagonal) {
      std::memset(dst + i * typeSize, 0, typeSize);
    }
  }
}

void absoluteIntImpl(Tensor& out, const Tensor& in) {
  size_t n = in.shape().elements();
  auto* dst = toCPUTensor(out).bufferPtr();
  const auto* src = toCPUTensor(in).bufferPtr();
  switch (in.type()) {
    case dtype::s32: {
      auto* d = static_cast<int*>(dst);
      const auto* s = static_cast<const int*>(src);
      for (size_t i = 0; i < n; i++) d[i] = std::abs(s[i]);
      break;
    }
    case dtype::b8: {
      auto* d = static_cast<char*>(dst);
      const auto* s = static_cast<const char*>(src);
      for (size_t i = 0; i < n; i++) d[i] = std::abs(s[i]);
      break;
    }
    case dtype::u8:
    case dtype::u32:
      std::memcpy(dst, src, n * fl::getTypeSize(in.type()));
      break;
    default:
      throw std::invalid_argument(
          "[absoluteIntImpl] unsupported type for integer abs");
  }
}

} // namespace native
} // namespace fl
