/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/ops/native/CpuTensorHelpers.h"

namespace fl {
namespace native {

namespace detail {

template <typename L, typename R, typename T, typename OP>
void applyBinopCpu(
    const void* lhs,
    const void* rhs,
    T* dst,
    unsigned count,
    OP op) {
  const L* lhsData = static_cast<const L*>(lhs);
  const R* rhsData = static_cast<const R*>(rhs);
  for (unsigned i = 0; i < count; i++) {
    dst[i] = op(lhsData[i], rhsData[i]);
  }
}

template <typename L, typename T, typename OP>
void applyBinopCpu(
    const void* lhs,
    const void* rhs,
    const dtype rhsType,
    T* dst,
    unsigned count,
    OP op) {
  switch (rhsType) {
    case fl::dtype::f16:
    case fl::dtype::bf16:
      throw std::runtime_error(
          "Fallback implementation currently doesn't support f16/bf16");
    case fl::dtype::f32:
      applyBinopCpu<L, float>(lhs, rhs, dst, count, op); break;
    case fl::dtype::f64:
      applyBinopCpu<L, double>(lhs, rhs, dst, count, op); break;
    case fl::dtype::b8:
      applyBinopCpu<L, char>(lhs, rhs, dst, count, op); break;
    case fl::dtype::s16:
      applyBinopCpu<L, short>(lhs, rhs, dst, count, op); break;
    case fl::dtype::s32:
      applyBinopCpu<L, int>(lhs, rhs, dst, count, op); break;
    case fl::dtype::s64:
      applyBinopCpu<L, long long>(lhs, rhs, dst, count, op); break;
    case fl::dtype::u8:
      applyBinopCpu<L, unsigned char>(lhs, rhs, dst, count, op); break;
    case fl::dtype::u16:
      applyBinopCpu<L, unsigned short>(lhs, rhs, dst, count, op); break;
    case fl::dtype::u32:
      applyBinopCpu<L, unsigned int>(lhs, rhs, dst, count, op); break;
    case fl::dtype::u64:
      applyBinopCpu<L, unsigned long long>(lhs, rhs, dst, count, op); break;
  }
}

template <typename T, typename OP>
void applyBinopCpu(
    const void* lhs,
    const dtype lhsType,
    const void* rhs,
    const dtype rhsType,
    T* dst,
    unsigned count,
    OP op) {
  switch (lhsType) {
    case fl::dtype::f16:
    case fl::dtype::bf16:
      throw std::runtime_error(
          "Fallback implementation currently doesn't support f16/bf16");
    case fl::dtype::f32:
      applyBinopCpu<float>(lhs, rhs, rhsType, dst, count, op); break;
    case fl::dtype::f64:
      applyBinopCpu<double>(lhs, rhs, rhsType, dst, count, op); break;
    case fl::dtype::b8:
      applyBinopCpu<char>(lhs, rhs, rhsType, dst, count, op); break;
    case fl::dtype::s16:
      applyBinopCpu<short>(lhs, rhs, rhsType, dst, count, op); break;
    case fl::dtype::s32:
      applyBinopCpu<int>(lhs, rhs, rhsType, dst, count, op); break;
    case fl::dtype::s64:
      applyBinopCpu<long long>(lhs, rhs, rhsType, dst, count, op); break;
    case fl::dtype::u8:
      applyBinopCpu<unsigned char>(lhs, rhs, rhsType, dst, count, op); break;
    case fl::dtype::u16:
      applyBinopCpu<unsigned short>(lhs, rhs, rhsType, dst, count, op); break;
    case fl::dtype::u32:
      applyBinopCpu<unsigned int>(lhs, rhs, rhsType, dst, count, op); break;
    case fl::dtype::u64:
      applyBinopCpu<unsigned long long>(lhs, rhs, rhsType, dst, count, op); break;
  }
}

} // namespace detail

// dispatchByType is now in CpuTensorHelpers.h

template <typename T, typename Op>
Tensor cpuBinaryOpWithBroadcast(
    const Tensor& lhs,
    const Tensor& rhs,
    Op op) {
  auto lhsData = lhs.toHostVectorSync<T>();
  auto rhsData = rhs.toHostVectorSync<T>();

  // Compute broadcast output shape
  auto lhsDims = lhs.shape().get();
  auto rhsDims = rhs.shape().get();
  const auto maxNdim = std::max(lhsDims.size(), rhsDims.size());
  while (lhsDims.size() < maxNdim) {
    lhsDims.insert(lhsDims.begin(), 1);
  }
  while (rhsDims.size() < maxNdim) {
    rhsDims.insert(rhsDims.begin(), 1);
  }
  std::vector<Dim> outDims(maxNdim);
  for (size_t i = 0; i < maxNdim; ++i) {
    if (lhsDims[i] != rhsDims[i] && lhsDims[i] != 1 && rhsDims[i] != 1) {
      std::stringstream ss;
      ss << "[native::cpuBinaryOpWithBroadcast] Cannot broadcast shapes: "
         << lhs.shape() << " and " << rhs.shape();
      throw std::invalid_argument(ss.str());
    }
    outDims[i] = std::max(lhsDims[i], rhsDims[i]);
  }
  Shape outShape(outDims);

  std::vector<T> result(outShape.elements());
  if (lhs.shape() == rhs.shape()) {
    for (size_t i = 0; i < result.size(); ++i) {
      result[i] = op(lhsData[i], rhsData[i]);
    }
  } else {
    auto outStrides = rowMajorStrides(outDims);
    auto lhsStrides = rowMajorStrides(lhsDims);
    auto rhsStrides = rowMajorStrides(rhsDims);
    std::vector<Dim> coords(maxNdim);
    for (size_t i = 0; i < result.size(); ++i) {
      linearIndexToCoords(i, outDims, outStrides, coords);
      result[i] = op(
          lhsData[broadcastCoordsToLinearIndex(coords, lhsDims, lhsStrides)],
          rhsData[broadcastCoordsToLinearIndex(coords, rhsDims, rhsStrides)]);
    }
  }
  return makeCpuTensor(outShape, result.data());
}

template <typename T, typename OP>
Tensor sameShapeBinopCpu(const Tensor& lhs, const Tensor& rhs, OP op) {
  if (!lhs.isContiguous()) {
    return sameShapeBinopCpu<T>(lhs.asContiguousTensor(), rhs, op);
  }
  if (!rhs.isContiguous()) {
    return sameShapeBinopCpu<T>(lhs, rhs.asContiguousTensor(), op);
  }
  if (lhs.shape() != rhs.shape()) {
    throw std::invalid_argument(
        "[native::sameShapeBinopCpu] requires input tensors of the same shape");
  }
  lhs.stream().sync();
  rhs.stream().sync();
  void* lhsData;
  void* rhsData;
  lhs.device(&lhsData);
  rhs.device(&rhsData);
  std::vector<T> dst(lhs.elements());
  T* dstData = dst.data();
  detail::applyBinopCpu(
      lhsData, lhs.type(), rhsData, rhs.type(), dstData, lhs.elements(), op);
  return makeCpuTensor(lhs.shape(), dstData);
}

} // namespace native
} // namespace fl
