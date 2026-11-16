/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/native/ShapeAndIndex.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/ops/native/CpuTensorHelpers.h"

namespace fl {
namespace native {

Tensor nonzeroCpu(const Tensor& tensor) {
  auto flat = tensor.flatten();
  size_t n = flat.elements();
  size_t typeSize = fl::getTypeSize(flat.type());
  std::vector<char> data(n * typeSize);
  flat.hostSync(data.data());
  std::vector<char> zero(typeSize, 0);
  std::vector<int> indices;
  for (size_t i = 0; i < n; i++) {
    if (std::memcmp(data.data() + i * typeSize, zero.data(), typeSize) != 0) {
      indices.push_back(static_cast<int>(i));
    }
  }
  if (indices.empty()) {
    // A bare CPU tensor: this runs inside evaluation, where the default
    // tensor type would create a JIT tensor.
    return makeCpuTensor(Shape({0}), dtype::s32, nullptr);
  }
  return makeCpuTensor(
      Shape({static_cast<Dim>(indices.size())}),
      indices.data());
}

void padImpl(
    Tensor& out, const Tensor& in,
    const Shape& inShape, const Shape& outShape,
    const std::vector<std::pair<int, int>>& padWidths, PadType padType) {
  auto inDims = inShape.get();
  auto outDims = outShape.get();
  auto inStrides = rowMajorStrides(inDims);
  auto outStrides = rowMajorStrides(outDims);
  int ndim = static_cast<int>(inDims.size());
  size_t typeSize = fl::getTypeSize(in.type());
  size_t inN = inShape.elements();
  size_t outN = outShape.elements();
  auto* dst = static_cast<char*>(toCPUTensor(out).bufferPtr());
  const auto* src = static_cast<const char*>(toCPUTensor(in).bufferPtr());
  std::memset(dst, 0, outN * typeSize);
  if (padType == PadType::Edge || padType == PadType::Symmetric) {
    std::vector<Dim> coords(outDims.size());
    std::vector<Dim> srcCoords(ndim);
    for (size_t i = 0; i < outN; i++) {
      linearIndexToCoords(i, outDims, outStrides, coords);
      for (int d = 0; d < ndim; d++) {
        Dim c = coords[d] - padWidths[d].first;
        if (padType == PadType::Edge) {
          srcCoords[d] = std::clamp(c, Dim(0), inDims[d] - 1);
        } else {
          Dim nn = inDims[d];
          if (c < 0) c = -c - 1;
          if (nn > 1) {
            Dim period = 2 * nn;
            c = c % period;
            if (c >= nn) c = period - c - 1;
          } else {
            c = 0;
          }
          srcCoords[d] = c;
        }
      }
      std::memcpy(
          dst + i * typeSize,
          src + coordsToLinearIndex(srcCoords, inStrides) * typeSize,
          typeSize);
    }
  } else {
    std::vector<Dim> coords(inDims.size());
    for (size_t i = 0; i < inN; i++) {
      linearIndexToCoords(i, inDims, inStrides, coords);
      for (int d = 0; d < ndim; d++) coords[d] += padWidths[d].first;
      std::memcpy(
          dst + coordsToLinearIndex(coords, outStrides) * typeSize,
          src + i * typeSize,
          typeSize);
    }
  }
}

Tensor whereCpu(const Tensor& condition, const Tensor& x, const Tensor& y) {
  Tensor condU8 = (condition.type() == dtype::u8)
      ? condition
      : condition.astype(dtype::u8);
  return dispatchByType(x.type(), [&](auto typeTag) {
    using T = decltype(typeTag);
    auto condData = condU8.toHostVectorSync<uint8_t>();
    auto xData = x.toHostVectorSync<T>();
    auto yData = y.toHostVectorSync<T>();
    const Dim n = x.elements();
    std::vector<T> result(n);
    for (Dim i = 0; i < n; i++) {
      result[i] = condData[i] ? xData[i] : yData[i];
    }
    return makeCpuTensor(x.shape(), result.data());
  });
}

Tensor tileCpu(
    const Tensor& tensor,
    const Shape& paddedTensorShape,
    const Shape& paddedTileDims) {
  // Compute output shape
  std::vector<Dim> outDims(paddedTensorShape.ndim());
  for (int d = 0; d < paddedTensorShape.ndim(); d++) {
    outDims[d] = paddedTensorShape[d] * paddedTileDims[d];
  }
  Shape outShape(outDims);
  auto tensorDims = paddedTensorShape.get();
  auto tensorStrides = rowMajorStrides(tensorDims);
  auto outStrides = rowMajorStrides(outDims);

  size_t typeSize = fl::getTypeSize(tensor.type());
  size_t inN = tensor.elements();
  size_t outN = outShape.elements();
  std::vector<char> src(inN * typeSize);
  tensor.hostSync(src.data());
  std::vector<char> dst(outN * typeSize);

  std::vector<Dim> outCoords(outDims.size());
  std::vector<Dim> srcCoords(tensorDims.size());
  for (size_t i = 0; i < outN; i++) {
    linearIndexToCoords(i, outDims, outStrides, outCoords);
    for (size_t d = 0; d < outCoords.size(); d++) {
      srcCoords[d] = outCoords[d] % tensorDims[d];
    }
    std::memcpy(
        dst.data() + i * typeSize,
        src.data() + coordsToLinearIndex(srcCoords, tensorStrides) * typeSize,
        typeSize);
  }
  return makeCpuTensor(outShape, tensor.type(), dst.data());
}

} // namespace native
} // namespace fl
