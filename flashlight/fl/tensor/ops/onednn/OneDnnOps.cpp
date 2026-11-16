/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/onednn/OneDnnOps.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "flashlight/fl/runtime/AsyncCPUStream.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/ops/native/Binary.h"
#include "flashlight/fl/tensor/ops/native/CpuTensorHelpers.h"
#include "flashlight/fl/tensor/ops/native/Fill.h"
#include "flashlight/fl/tensor/ops/native/Matmul.h"
#include "flashlight/fl/tensor/ops/native/Reduction.h"
#include "flashlight/fl/tensor/ops/native/ShapeAndIndex.h"
#include "flashlight/fl/tensor/ops/native/Transpose.h"
#include "flashlight/fl/tensor/ops/onednn/Utils.h"

namespace fl {

namespace {

// Helper: create a Tensor from a dnnl::memory output.
// The dnnl::memory is moved into a shared_ptr<void> to keep its allocation alive.
Tensor dnnlMemToTensor(
    const Shape& shape,
    dnnl::memory&& mem,
    std::shared_ptr<AsyncCPUStream> stream = nullptr) {
  auto type = detail::oneDnnToFlType(mem.get_desc().get_data_type());
  void* buf = mem.get_data_handle();
  auto owner = std::make_shared<dnnl::memory>(std::move(mem));
  return toTensor<CPUTensor>(shape, type, buf, std::move(owner), std::move(stream));
}

/**
 * ASSUME row-major layout. Compute the strides if we keep the dimensions but
 * permute the axes.
 */
// Compute destination strides (in source axis order) for a transpose reorder.
// `axes` follows NumPy convention: axes[newAxis] = oldAxis, meaning the output
// axis `newAxis` corresponds to the input axis `axes[newAxis]`.
// The result is a stride array indexed by *old* axis, such that creating a
// dnnl::memory::desc with (oldDims, type, strides) and reordering from the
// contiguous source into it produces a contiguous destination buffer in the
// permuted layout.
dnnl::memory::dims getStridesAfterPermuteAxes(
    const dnnl::memory::dims& oldDims,
    const std::vector<Dim>& axes) {
  assert(oldDims.size() == axes.size());
  const auto ndim = oldDims.size();
  // Compute the new (permuted) shape
  std::vector<dnnl::memory::dim> newDims(ndim);
  for (size_t i = 0; i < ndim; i++) {
    newDims[i] = oldDims[axes[i]];
  }
  // Compute contiguous row-major strides for the permuted shape
  std::vector<dnnl::memory::dim> newStrides(ndim, 1);
  for (int i = static_cast<int>(ndim) - 2; i >= 0; i--) {
    newStrides[i] = newStrides[i + 1] * newDims[i + 1];
  }
  // Map strides back to old (source) axis order
  std::vector<dnnl::memory::dim> strides(ndim);
  for (size_t i = 0; i < ndim; i++) {
    strides[axes[i]] = newStrides[i];
  }
  return strides;
}

struct BinaryOpOutputDesc {
  BinaryOpOutputDesc(dnnl::memory::desc desc, Shape shape)
      : dstMemDesc(std::move(desc)), dstShape(std::move(shape)) {}
  dnnl::memory::desc dstMemDesc;
  const Shape dstShape;
};

BinaryOpOutputDesc getBinaryOpOutputDesc(
    const Shape& lhsShape,
    const dnnl::memory::desc& lhsMemDesc,
    const Shape& rhsShape,
    const dnnl::memory::desc& rhsMemDesc,
    std::optional<dnnl::memory::data_type> optDstType) {
  // allow implicit casting
  const auto dstType = optDstType.value_or(detail::getTypeWithLargerRange(
      lhsMemDesc.get_data_type(), rhsMemDesc.get_data_type()));
  // Fast path: same shapes
  if (lhsShape == rhsShape) {
    return BinaryOpOutputDesc(
        detail::oneDnnContiguousMemDescFromShape(lhsShape, dstType),
        lhsShape);
  }
  // Pad shorter shape with leading 1s to match ndim for OneDNN broadcasting
  auto lhsDims = lhsShape.get();
  auto rhsDims = rhsShape.get();
  const auto maxNdim = std::max(lhsDims.size(), rhsDims.size());
  while (lhsDims.size() < maxNdim) {
    lhsDims.insert(lhsDims.begin(), 1);
  }
  while (rhsDims.size() < maxNdim) {
    rhsDims.insert(rhsDims.begin(), 1);
  }
  // Check broadcast compatibility and compute output dimensions
  std::vector<Dim> dstDims;
  for (size_t i = 0; i < maxNdim; ++i) {
    if (lhsDims[i] != rhsDims[i] && lhsDims[i] != 1 && rhsDims[i] != 1) {
      std::stringstream ss;
      ss << "[OneDnnOps] Cannot broadcast shapes: "
         << lhsShape << " and " << rhsShape;
      throw std::invalid_argument(ss.str());
    }
    dstDims.push_back(std::max(lhsDims[i], rhsDims[i]));
  }
  Shape dstShape(dstDims);
  return BinaryOpOutputDesc(
      detail::oneDnnContiguousMemDescFromShape(dstShape, dstType),
      dstShape);
}

dnnl::memory::desc transposeInnerMatrix(const dnnl::memory::desc& memDesc) {
  const auto ndims = memDesc.get_ndims();
  if (ndims < 2) {
    std::ostringstream oss;
    oss << "[transposeInnerMatrix] expected ndims to be >= 2, got: " << ndims;
    throw std::runtime_error(oss.str());
  }
  // Swap the last two axes (inner matrix dimensions) for transpose.
  std::vector<int> transposeAxesPermutation;
  for (int i = 0; i < ndims; i++) {
    transposeAxesPermutation.push_back(i);
  }
  std::swap(
      transposeAxesPermutation[ndims - 2], transposeAxesPermutation[ndims - 1]);
  return memDesc.permute_axes(transposeAxesPermutation);
}

// Pad dims for tile. FL tile convention matches NumPy/PyTorch: tileDim[i]
// tiles the i-th dimension. When tileDims has more dims than tensor, extra
// leading dims are prepended to the tensor shape (size 1). When tensor has
// more dims than tileDims, extra trailing tile counts default to 1.
std::tuple<Shape, Shape> padDimsForTile(
    const Shape& tensorShape,
    const Shape& tileDims) {
  std::vector<Dim> paddedTensorDims = tensorShape.get();
  std::vector<Dim> paddedTileDims = tileDims.get();
  const auto tensorShapeNDims = tensorShape.ndim();
  const auto tileDimsNDims = tileDims.ndim();
  if (tensorShapeNDims > tileDimsNDims) {
    // More tensor dims: pad tileDims with trailing 1s (no replication)
    const auto diff = tensorShapeNDims - tileDimsNDims;
    paddedTileDims.insert(paddedTileDims.end(), diff, 1);
  } else if (tileDimsNDims > tensorShapeNDims) {
    // More tile dims: pad tensor shape with leading 1s (new dimensions)
    const auto diff = tileDimsNDims - tensorShapeNDims;
    paddedTensorDims.insert(paddedTensorDims.begin(), diff, 1);
  }
  return {Shape(paddedTensorDims), Shape(paddedTileDims)};
}

// Write a scalar value into a dnnl::memory in the tensor's native dtype.
void writeScalarToMemory(dnnl::memory& mem, dnnl::memory::data_type type, double val) {
  switch (type) {
    case dnnl::memory::data_type::f32:
      *static_cast<float*>(mem.get_data_handle()) = static_cast<float>(val);
      break;
    case dnnl::memory::data_type::f64:
      *static_cast<double*>(mem.get_data_handle()) = val;
      break;
    case dnnl::memory::data_type::s32:
      *static_cast<int32_t*>(mem.get_data_handle()) = static_cast<int32_t>(val);
      break;
    case dnnl::memory::data_type::s8:
      *static_cast<int8_t*>(mem.get_data_handle()) = static_cast<int8_t>(val);
      break;
    case dnnl::memory::data_type::u8:
      *static_cast<uint8_t*>(mem.get_data_handle()) = static_cast<uint8_t>(val);
      break;
    default:
      throw std::invalid_argument(
          "writeScalarToMemory: unsupported dtype");
  }
}

// Compute strided offset for logical element i (shared helper for CPU fallbacks).
Dim stridedOffset(
    Dim i, int ndim,
    const dnnl::memory::dims& dims,
    const dnnl::memory::dims& strides) {
  Dim offset = 0;
  for (int d = ndim - 1; d >= 0; --d) {
    offset += (i % dims[d]) * strides[d];
    i /= dims[d];
  }
  return offset;
}

// Check if a memory layout is contiguous (row-major).
bool isContiguousLayout(
    const dnnl::memory::dims& dims,
    const dnnl::memory::dims& strides) {
  dnnl::memory::dim expected = 1;
  for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
    if (dims[d] > 1 && strides[d] != expected) return false;
    expected *= dims[d];
  }
  return true;
}

// CPU fallback for in-place scalar binary op (stride-aware).
template <typename T>
void inPlaceBinaryScalarCpu(
    dnnl::memory& mem,
    const dnnl::memory::desc& desc,
    Dim numElements,
    double val,
    dnnl::algorithm alg) {
  T* data = static_cast<T*>(mem.get_data_handle());
  T v = static_cast<T>(val);
  const auto dims = desc.get_dims();
  const auto strides = desc.get_strides();
  const bool contig = isContiguousLayout(dims, strides);
  const int ndim = static_cast<int>(dims.size());

  for (Dim i = 0; i < numElements; ++i) {
    Dim off = contig ? i : stridedOffset(i, ndim, dims, strides);
    switch (alg) {
      case dnnl::algorithm::binary_add: data[off] += v; break;
      case dnnl::algorithm::binary_sub: data[off] -= v; break;
      case dnnl::algorithm::binary_mul: data[off] *= v; break;
      case dnnl::algorithm::binary_div: data[off] /= v; break;
      default:
        throw std::runtime_error(
            "inPlaceBinaryScalarCpu: unsupported algorithm");
    }
  }
}

// CPU fallback for in-place tensor binary op (stride-aware).
template <typename T>
void inPlaceBinaryTensorCpu(
    dnnl::memory& selfMem,
    const dnnl::memory::desc& selfDesc,
    dnnl::memory& otherMem,
    const dnnl::memory::desc& otherDesc,
    Dim numElements,
    dnnl::algorithm alg) {
  T* dst = static_cast<T*>(selfMem.get_data_handle());
  const T* src = static_cast<const T*>(otherMem.get_data_handle());

  const auto dstDims = selfDesc.get_dims();
  const auto dstStrides = selfDesc.get_strides();
  const auto srcDims = otherDesc.get_dims();
  const auto srcStrides = otherDesc.get_strides();
  const int ndim = static_cast<int>(dstDims.size());

  auto isContig = [](const dnnl::memory::dims& d,
                     const dnnl::memory::dims& s) -> bool {
    dnnl::memory::dim expected = 1;
    for (int i = static_cast<int>(d.size()) - 1; i >= 0; --i) {
      if (d[i] > 1 && s[i] != expected) return false;
      expected *= d[i];
    }
    return true;
  };

  bool dstContig = isContig(dstDims, dstStrides);
  bool srcContig = isContig(srcDims, srcStrides);

  auto stridedOff = [ndim](Dim i,
      const dnnl::memory::dims& d,
      const dnnl::memory::dims& s) -> Dim {
    Dim offset = 0;
    for (int dd = ndim - 1; dd >= 0; --dd) {
      offset += (i % d[dd]) * s[dd];
      i /= d[dd];
    }
    return offset;
  };

  for (Dim i = 0; i < numElements; ++i) {
    Dim dOff = dstContig ? i : stridedOff(i, dstDims, dstStrides);
    Dim sOff = srcContig ? i : stridedOff(i, srcDims, srcStrides);
    switch (alg) {
      case dnnl::algorithm::binary_add: dst[dOff] += src[sOff]; break;
      case dnnl::algorithm::binary_sub: dst[dOff] -= src[sOff]; break;
      case dnnl::algorithm::binary_mul: dst[dOff] *= src[sOff]; break;
      case dnnl::algorithm::binary_div: dst[dOff] /= src[sOff]; break;
      default:
        throw std::runtime_error(
            "inPlaceBinaryTensorCpu: unsupported algorithm");
    }
  }
}

// An enqueued task must own every buffer it touches. A dnnl::memory created
// for a freshly allocated output owns its buffer, but one wrapping a
// pre-assigned `out` does not: keep `out`'s storage alive until the task runs,
// otherwise the evaluator may release it first and the task writes into freed
// memory.
std::shared_ptr<CPUTensor::SharedData> keepAliveIfPreassigned(
    bool allocated,
    const Tensor& out) {
  return allocated ? nullptr : toCPUTensor(out).getSharedData();
}

} // namespace

// -------------- OneDnnOps constructor/destructor ----------------

OneDnnOps::OneDnnOps()
    : engine_(dnnl::engine::kind::cpu, 0) {}

OneDnnOps::~OneDnnOps() = default;

// -------------- OneDnnOps wrapper methods ----------------

void OneDnnOps::applyBinop(
    Tensor& out,
    const Tensor& lhs,
    const Tensor& rhs,
    dnnl::algorithm alg,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream,
    std::optional<dnnl::memory::data_type> dstType) {
  detail::onednn::applyBinop(
      out, lhs, rhs, alg, engine_, nativeStream, std::move(asyncStream), dstType);
}

void OneDnnOps::applyEltwiseOp(
    Tensor& out,
    const Tensor& tensor,
    dnnl::algorithm alg,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream,
    float alpha,
    float beta) {
  detail::onednn::applyEltwiseOp(
      out, tensor, alg, engine_, nativeStream, std::move(asyncStream), alpha, beta);
}

void OneDnnOps::applyReductionOp(
    Tensor& out,
    const Tensor& tensor,
    dnnl::algorithm alg,
    const std::vector<int>& axes,
    bool keepDims,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream,
    float p,
    float eps) {
  detail::onednn::applyReductionOp(
      out, tensor, alg, axes, keepDims, engine_, nativeStream,
      std::move(asyncStream), p, eps);
}

Tensor OneDnnOps::reshape(
    const Tensor& tensor,
    const Shape& shape,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream) {
  return detail::onednn::reshape(
      tensor, shape, engine_, nativeStream, std::move(asyncStream));
}

Tensor OneDnnOps::transpose(
    const Tensor& tensor,
    const Shape& axes,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream) {
  return detail::onednn::transpose(
      tensor, axes, engine_, nativeStream, std::move(asyncStream));
}

Tensor OneDnnOps::tile(
    const Tensor& tensor,
    const Shape& tileDims,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream) {
  return detail::onednn::tile(
      tensor, tileDims, engine_, nativeStream, std::move(asyncStream));
}

Tensor OneDnnOps::concatenate(
    const std::vector<Tensor>& tensors,
    unsigned axis,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream) {
  return detail::onednn::concatenate(
      tensors, axis, engine_, nativeStream, std::move(asyncStream));
}

void OneDnnOps::matmul(
    Tensor& out,
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream,
    const dnnl::primitive_attr& attr,
    const Tensor& bias) {
  detail::onednn::matmul(
      out, lhs, rhs, lhsProp, rhsProp, engine_, nativeStream,
      std::move(asyncStream), attr, bias);
}

Tensor OneDnnOps::where(
    const Tensor& condition,
    const Tensor& x,
    const Tensor& y,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream) {
  return detail::onednn::where(
      condition, x, y, engine_, nativeStream, std::move(asyncStream));
}

void OneDnnOps::inPlaceAdd(CPUTensor& self, double val, dnnl::stream& nativeStream) {
  detail::onednn::inPlaceBinaryScalar(
      self, val, dnnl::algorithm::binary_add, engine_, nativeStream);
}
void OneDnnOps::inPlaceSub(CPUTensor& self, double val, dnnl::stream& nativeStream) {
  detail::onednn::inPlaceBinaryScalar(
      self, val, dnnl::algorithm::binary_sub, engine_, nativeStream);
}
void OneDnnOps::inPlaceMul(CPUTensor& self, double val, dnnl::stream& nativeStream) {
  detail::onednn::inPlaceBinaryScalar(
      self, val, dnnl::algorithm::binary_mul, engine_, nativeStream);
}
void OneDnnOps::inPlaceDiv(CPUTensor& self, double val, dnnl::stream& nativeStream) {
  detail::onednn::inPlaceBinaryScalar(
      self, val, dnnl::algorithm::binary_div, engine_, nativeStream);
}

void OneDnnOps::inPlaceAdd(CPUTensor& self, const Tensor& other, dnnl::stream& nativeStream) {
  detail::onednn::inPlaceBinaryTensor(
      self, other, dnnl::algorithm::binary_add, engine_, nativeStream);
}
void OneDnnOps::inPlaceSub(CPUTensor& self, const Tensor& other, dnnl::stream& nativeStream) {
  detail::onednn::inPlaceBinaryTensor(
      self, other, dnnl::algorithm::binary_sub, engine_, nativeStream);
}
void OneDnnOps::inPlaceMul(CPUTensor& self, const Tensor& other, dnnl::stream& nativeStream) {
  detail::onednn::inPlaceBinaryTensor(
      self, other, dnnl::algorithm::binary_mul, engine_, nativeStream);
}
void OneDnnOps::inPlaceDiv(CPUTensor& self, const Tensor& other, dnnl::stream& nativeStream) {
  detail::onednn::inPlaceBinaryTensor(
      self, other, dnnl::algorithm::binary_div, engine_, nativeStream);
}

void OneDnnOps::fillInPlace(CPUTensor& self, double val, dnnl::stream& nativeStream) {
  detail::onednn::fillInPlace(self, val, engine_, nativeStream);
}

Tensor OneDnnOps::reorderAstype(CPUTensor& self, fl::dtype type, dnnl::stream& nativeStream) {
  return detail::onednn::reorderAstype(self, type, engine_, nativeStream);
}

void OneDnnOps::assignTensor(CPUTensor& self, const Tensor& src, dnnl::stream& nativeStream) {
  detail::onednn::assignTensor(self, src, engine_, nativeStream);
}

// =====================================================================
// detail::onednn free function implementations
// =====================================================================

namespace detail::onednn {

void applyBinop(
    Tensor& out,
    const Tensor& lhs,
    const Tensor& rhs,
    dnnl::algorithm alg,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream,
    std::optional<dnnl::memory::data_type> dstType) {
  // f64 is not supported by OneDNN binary primitives on CPU — use CPU fallback.
  if (lhs.type() == dtype::f64 || rhs.type() == dtype::f64) {
    auto lhsF64 = (lhs.type() == dtype::f64) ? lhs : lhs.astype(dtype::f64);
    auto rhsF64 = (rhs.type() == dtype::f64) ? rhs : rhs.astype(dtype::f64);
    using A = dnnl::algorithm;
    switch (alg) {
      case A::binary_add:
        out = fl::native::cpuBinaryOpWithBroadcast<double>(lhsF64, rhsF64,
            [](double a, double b) { return a + b; }); return;
      case A::binary_sub:
        out = fl::native::cpuBinaryOpWithBroadcast<double>(lhsF64, rhsF64,
            [](double a, double b) { return a - b; }); return;
      case A::binary_mul:
        out = fl::native::cpuBinaryOpWithBroadcast<double>(lhsF64, rhsF64,
            [](double a, double b) { return a * b; }); return;
      case A::binary_div:
        out = fl::native::cpuBinaryOpWithBroadcast<double>(lhsF64, rhsF64,
            [](double a, double b) { return a / b; }); return;
      case A::binary_min:
        out = fl::native::cpuBinaryOpWithBroadcast<double>(lhsF64, rhsF64,
            [](double a, double b) { return std::min(a, b); }); return;
      case A::binary_max:
        out = fl::native::cpuBinaryOpWithBroadcast<double>(lhsF64, rhsF64,
            [](double a, double b) { return std::max(a, b); }); return;
      // Comparison ops: read as double, produce b8 (int8_t / s8)
      case A::binary_eq:
      case A::binary_ne:
      case A::binary_lt:
      case A::binary_le:
      case A::binary_gt:
      case A::binary_ge: {
        auto lhsData = lhsF64.toHostVectorSync<double>();
        auto rhsData = rhsF64.toHostVectorSync<double>();
        auto lhsDims = lhsF64.shape().get();
        auto rhsDims = rhsF64.shape().get();
        const auto maxNdim = std::max(lhsDims.size(), rhsDims.size());
        while (lhsDims.size() < maxNdim) lhsDims.insert(lhsDims.begin(), 1);
        while (rhsDims.size() < maxNdim) rhsDims.insert(rhsDims.begin(), 1);
        std::vector<Dim> outDims(maxNdim);
        for (size_t i = 0; i < maxNdim; ++i) {
          outDims[i] = std::max(lhsDims[i], rhsDims[i]);
        }
        Shape outShape(outDims);
        std::vector<char> result(outShape.elements());
        auto cmpFn = [alg](double a, double b) -> char {
          switch (alg) {
            case A::binary_eq: return a == b ? 1 : 0;
            case A::binary_ne: return a != b ? 1 : 0;
            case A::binary_lt: return a < b ? 1 : 0;
            case A::binary_le: return a <= b ? 1 : 0;
            case A::binary_gt: return a > b ? 1 : 0;
            case A::binary_ge: return a >= b ? 1 : 0;
            default: return 0;
          }
        };
        if (lhsF64.shape() == rhsF64.shape()) {
          for (size_t i = 0; i < result.size(); ++i) {
            result[i] = cmpFn(lhsData[i], rhsData[i]);
          }
        } else {
          auto outStrides = native::rowMajorStrides(outDims);
          auto lhsStrides = native::rowMajorStrides(lhsDims);
          auto rhsStrides = native::rowMajorStrides(rhsDims);
          std::vector<Dim> coords(maxNdim);
          for (size_t i = 0; i < result.size(); ++i) {
            native::linearIndexToCoords(i, outDims, outStrides, coords);
            result[i] = cmpFn(
                lhsData[native::broadcastCoordsToLinearIndex(
                    coords, lhsDims, lhsStrides)],
                rhsData[native::broadcastCoordsToLinearIndex(
                    coords, rhsDims, rhsStrides)]);
          }
        }
        out = native::makeCpuTensor(outShape, result.data()); return;
      }
      default:
        throw std::runtime_error(
            "[OneDnnOps::applyBinop] unsupported f64 algorithm");
    }
  }

  // OneDNN binary primitives require both sources to have the same data type.
  // Promote to a common type if they differ.
  if (lhs.type() != rhs.type()) {
    auto commonType = promoteDtypes(lhs.type(), rhs.type());
    auto lhsP = (lhs.type() == commonType) ? lhs : lhs.astype(commonType);
    auto rhsP = (rhs.type() == commonType) ? rhs : rhs.astype(commonType);
    applyBinop(out, lhsP, rhsP, alg, engine, nativeStream, asyncStream, dstType);
    return;
  }

  // prepare memories
  auto& lhsTensor = toCPUTensor(lhs);
  auto& rhsTensor = toCPUTensor(rhs);
  auto lhsMem = detail::makeDnnlMemory(
      lhsTensor.bufferPtr(), lhs.shape(), lhsTensor.stridesVec(),
      lhs.type(), engine);
  auto rhsMem = detail::makeDnnlMemory(
      rhsTensor.bufferPtr(), rhs.shape(), rhsTensor.stridesVec(),
      rhs.type(), engine);
  auto lhsMemDesc = lhsMem.get_desc();
  auto rhsMemDesc = rhsMem.get_desc();

  // OneDNN binary requires same ndim — pad shorter desc with leading 1s.
  // Must reshape both the descriptors AND the memory objects, since OneDNN
  // checks ndim consistency between args and the primitive descriptor.
  const int lhsNdim = lhsMemDesc.get_ndims();
  const int rhsNdim = rhsMemDesc.get_ndims();
  if (lhsNdim != rhsNdim) {
    const int maxNdim = std::max(lhsNdim, rhsNdim);
    auto padDims = [maxNdim](const dnnl::memory::dims& dims) {
      dnnl::memory::dims padded(maxNdim, 1);
      int offset = maxNdim - static_cast<int>(dims.size());
      for (size_t i = 0; i < dims.size(); ++i) {
        padded[offset + i] = dims[i];
      }
      return padded;
    };
    if (lhsNdim < maxNdim) {
      lhsMemDesc = lhsMemDesc.reshape(padDims(lhsMemDesc.get_dims()));
      lhsMem = dnnl::memory(lhsMemDesc, engine, lhsMem.get_data_handle());
    }
    if (rhsNdim < maxNdim) {
      rhsMemDesc = rhsMemDesc.reshape(padDims(rhsMemDesc.get_dims()));
      rhsMem = dnnl::memory(rhsMemDesc, engine, rhsMem.get_data_handle());
    }
  }

  const auto outputDesc = getBinaryOpOutputDesc(
      lhs.shape(), lhsMemDesc, rhs.shape(), rhsMemDesc, dstType);
  const bool allocated = out.isEmpty();
  dnnl::memory dstMem = allocated
      ? dnnl::memory(outputDesc.dstMemDesc, engine)
      : dnnl::memory(outputDesc.dstMemDesc, engine,
                     toCPUTensor(out).bufferPtr());

  // prepare primitive
  const auto binaryPrimitiveDesc = dnnl::binary::primitive_desc(
      engine, alg, lhsMemDesc, rhsMemDesc, outputDesc.dstMemDesc);
  const auto binaryPrimitive = dnnl::binary(binaryPrimitiveDesc);

  // prepare arguments
  const std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC_0, lhsMem},
      {DNNL_ARG_SRC_1, rhsMem},
      {DNNL_ARG_DST, dstMem},
  };

  // execute primitive asynchronously
  auto& opStream = static_cast<const AsyncCPUStream&>(lhs.stream());
  relativeSync(opStream, {rhs});
  opStream.enqueue(
      [prim = binaryPrimitive, execArgs = std::move(args),
       lhsKeepAlive = lhsTensor.getSharedData(),
       rhsKeepAlive = rhsTensor.getSharedData(),
       outKeepAlive = keepAliveIfPreassigned(allocated, out),
       &ns = nativeStream]() {
        prim.execute(ns, execArgs);
      });
  if (allocated) {
    out = dnnlMemToTensor(outputDesc.dstShape, std::move(dstMem), asyncStream);
  }
  // Preserve FL type for types that don't have a 1:1 OneDNN mapping (e.g. u32 --> s32).
  if (!dstType.has_value() && lhs.type() == rhs.type() &&
      !detail::isTypeSupportedByOneDnn(lhs.type())) {
    toCPUTensor(out).setFlType(lhs.type());
  }
}

void applyEltwiseOp(
    Tensor& out,
    const Tensor& tensor,
    dnnl::algorithm alg,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream,
    float alpha,
    float beta) {
  // f64 is not supported by OneDNN eltwise primitives on CPU — use CPU fallback.
  if (tensor.type() == dtype::f64) {
    using A = dnnl::algorithm;
    auto data = tensor.toHostVectorSync<double>();
    for (auto& v : data) {
      switch (alg) {
        case A::eltwise_abs: v = std::abs(v); break;
        case A::eltwise_exp: v = std::exp(v); break;
        case A::eltwise_log: v = std::log(v); break;
        case A::eltwise_sqrt: v = std::sqrt(v); break;
        case A::eltwise_logistic: v = 1.0 / (1.0 + std::exp(-v)); break;
        case A::eltwise_tanh: v = std::tanh(v); break;
        case A::eltwise_relu: v = std::max(0.0, v); break;
        case A::eltwise_elu: v = v >= 0 ? v : alpha * (std::exp(v) - 1.0); break;
        case A::eltwise_linear: v = alpha * v + beta; break;
        case A::eltwise_round: v = std::round(v); break;
        case A::eltwise_gelu_erf:
          v = v * 0.5 * (1.0 + std::erf(v * M_SQRT1_2)); break;
        case A::eltwise_clip:
        case A::eltwise_clip_v2: v = std::max(static_cast<double>(alpha), std::min(static_cast<double>(beta), v)); break;
        default:
          throw std::runtime_error(
              "[OneDnnOps::applyEltwiseOp] unsupported f64 eltwise algorithm");
      }
    }
    out = toTensor<CPUTensor>(
        tensor.shape(), dtype::f64, data.data(), Location::Host);
    return;
  }

  // prepare memories. OneDNN eltwise needs matching source and destination
  // layouts and the destination is contiguous, so a strided view (e.g. a
  // column slice) is made contiguous first.
  std::optional<Tensor> contiguousSrc;
  if (!tensor.isContiguous()) {
    contiguousSrc = tensor.asContiguousTensor();
  }
  const Tensor& src = contiguousSrc ? *contiguousSrc : tensor;
  auto& srcTensor = toCPUTensor(src);
  auto srcMem = detail::makeDnnlMemory(
      srcTensor.bufferPtr(), src.shape(), srcTensor.stridesVec(),
      src.type(), engine);
  const auto srcMemDesc = srcMem.get_desc();
  const auto dstMemDesc = detail::oneDnnContiguousMemDescFromShape(
      tensor.shape(), detail::flToOneDnnType(tensor.type()));
  const bool allocated = out.isEmpty();
  dnnl::memory dstMem = allocated
      ? dnnl::memory(dstMemDesc, engine)
      : dnnl::memory(dstMemDesc, engine, toCPUTensor(out).bufferPtr());

  // prepare unary primitive
  const auto unaryPrimitiveDesc = dnnl::eltwise_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward_inference,
      alg,
      srcMemDesc,
      dstMemDesc,
      alpha,
      beta);
  const auto unaryPrimitive = dnnl::eltwise_forward(unaryPrimitiveDesc);

  // prepare arguments.
  const std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC, srcMem},
      {DNNL_ARG_DST, dstMem},
  };

  // execute primitive asynchronously
  auto& opStream = static_cast<const AsyncCPUStream&>(tensor.stream());
  opStream.enqueue(
      [prim = unaryPrimitive, execArgs = std::move(args),
       srcKeepAlive = srcTensor.getSharedData(),
       outKeepAlive = keepAliveIfPreassigned(allocated, out),
       &ns = nativeStream]() {
        prim.execute(ns, execArgs);
      });
  if (allocated) {
    out = dnnlMemToTensor(tensor.shape(), std::move(dstMem), asyncStream);
  }
  // Preserve flType_ for types stored differently in OneDNN (e.g. u32 as s32)
  if (!detail::isTypeSupportedByOneDnn(tensor.type())) {
    toCPUTensor(out).setFlType(tensor.type());
  }
}

void applyReductionOp(
    Tensor& out,
    const Tensor& input,
    dnnl::algorithm alg,
    const std::vector<int>& axes,
    bool keepDims,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream,
    float p,
    float eps) {
  // Scalar input: reduction is a no-op, just return a copy
  if (input.ndim() == 0) {
    out = input.copy();
    return;
  }

  // compute final shape
  std::vector<int> axesToReduce;
  if (axes.empty()) {
    axesToReduce.resize(input.ndim());
    std::iota(axesToReduce.begin(), axesToReduce.end(), 0);
  } else {
    axesToReduce = axes;
  }
  std::vector<Dim> dstDims = input.shape().get();
  for (int axis : axesToReduce) {
    if (axis < 0 || axis >= input.ndim()) {
      std::ostringstream oss;
      oss << "[OneDnnOps] Invalid axis for reduction: " << axis
          << " for tensor of shape: " << input.shape();
      throw std::invalid_argument(oss.str());
    }
    dstDims[axis] = 1;
  }

  // f64 not supported by OneDNN reduction primitives — use CPU fallback.
  if (input.type() == dtype::f64) {
    Shape dstShape(dstDims);
    auto srcData = input.toHostVectorSync<double>();
    auto srcDims = input.shape().get();
    const Dim dstElements = dstShape.elements();
    std::vector<double> dstData(dstElements);

    // Initialize based on algorithm
    using A = dnnl::algorithm;
    double initVal;
    switch (alg) {
      case A::reduction_sum: case A::reduction_mean: initVal = 0.0; break;
      case A::reduction_max: initVal = -std::numeric_limits<double>::infinity(); break;
      case A::reduction_min: initVal = std::numeric_limits<double>::infinity(); break;
      case A::reduction_mul: initVal = 1.0; break;
      default:
        throw std::runtime_error(
            "[OneDnnOps::applyReductionOp] unsupported f64 reduction algorithm");
    }
    std::fill(dstData.begin(), dstData.end(), initVal);

    // Compute reduction
    const int ndim = static_cast<int>(srcDims.size());
    const Dim srcElements = input.elements();
    for (Dim si = 0; si < srcElements; ++si) {
      Dim rem = si;
      Dim di = 0;
      Dim dstStride = 1;
      for (int d = ndim - 1; d >= 0; --d) {
        Dim coord = rem % srcDims[d];
        rem /= srcDims[d];
        Dim dstCoord = (dstDims[d] == 1) ? 0 : coord;
        di += dstCoord * dstStride;
        dstStride *= dstDims[d];
      }
      switch (alg) {
        case A::reduction_sum: case A::reduction_mean:
          dstData[di] += srcData[si]; break;
        case A::reduction_max:
          dstData[di] = std::max(dstData[di], srcData[si]); break;
        case A::reduction_min:
          dstData[di] = std::min(dstData[di], srcData[si]); break;
        case A::reduction_mul:
          dstData[di] *= srcData[si]; break;
        default: break;
      }
    }
    if (alg == A::reduction_mean) {
      Dim reducedCount = srcElements / dstElements;
      for (auto& v : dstData) v /= static_cast<double>(reducedCount);
    }

    // Remove reduced axes if not keepDims
    if (!keepDims) {
      dstDims = detail::removeIndices(input.shape().get(), axesToReduce);
      dstShape = Shape(dstDims);
    }
    out = toTensor<CPUTensor>(
        dstShape, dtype::f64, dstData.data(), Location::Host);
    return;
  }

  Shape dstShape(dstDims);

  // prepare part of memories
  auto& srcTensor = toCPUTensor(input);
  auto srcMem = detail::makeDnnlMemory(
      srcTensor.bufferPtr(), input.shape(), srcTensor.stridesVec(),
      input.type(), engine);
  const auto srcMemDesc = srcMem.get_desc();
  auto dstArgMemDesc = detail::oneDnnContiguousMemDescFromShape(
      dstShape, detail::flToOneDnnType(input.type()));

  // If all reduced axes already have size 1, reduction is a no-op.
  if (srcMemDesc.get_dims() == dstArgMemDesc.get_dims()) {
    out = input.copy();
    if (!keepDims) {
      auto filteredDims = detail::removeIndices(input.shape().get(), axesToReduce);
      out = fl::reshape(out, Shape(filteredDims));
    }
    return;
  }

  // prepare reduction primitive
  const auto reductionPrimitiveDesc = dnnl::reduction::primitive_desc(
      engine, alg, srcMemDesc, dstArgMemDesc, p, eps);
  const auto reductionPrimitive = dnnl::reduction(reductionPrimitiveDesc);

  const bool allocated = out.isEmpty();
  dnnl::memory dstMem = allocated
      ? dnnl::memory(dstArgMemDesc, engine)
      : dnnl::memory(dstArgMemDesc, engine, toCPUTensor(out).bufferPtr());

  // prepare arguments.
  const std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC, srcMem},
      {DNNL_ARG_DST, dstMem},
  };

  // execute primitive asynchronously
  auto& opStream = static_cast<const AsyncCPUStream&>(input.stream());
  opStream.enqueue(
      [prim = reductionPrimitive, execArgs = std::move(args),
       srcKeepAlive = srcTensor.getSharedData(),
       outKeepAlive = keepAliveIfPreassigned(allocated, out),
       &ns = nativeStream]() {
        prim.execute(ns, execArgs);
      });

  if (allocated) {
    // Create result with keepDims shape (matches the memory descriptor)
    out = dnnlMemToTensor(dstShape, std::move(dstMem), asyncStream);
    // If keepDims is false, reshape to remove reduced dimensions
    if (!keepDims) {
      auto finalShape = Shape(detail::removeIndices(dstShape.get(), axesToReduce));
      out = fl::reshape(out, finalShape);
    }
  }
}

Tensor reshape(
    const Tensor& tensor,
    const Shape& shape,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream) {
  if (tensor.shape().elements() != shape.elements()) {
    std::ostringstream oss;
    oss << "[OneDnnOps::reshape] Cannot reshape tensor from "
        << tensor.shape() << " to " << shape;
    throw std::invalid_argument(oss.str());
  }

  auto& srcTensor = toCPUTensor(tensor);
  const auto oneDnnType = detail::flToOneDnnType(tensor.type());
  auto srcMem = detail::makeDnnlMemory(
      srcTensor.bufferPtr(), tensor.shape(), srcTensor.stridesVec(),
      tensor.type(), engine);
  const auto srcMemDesc = srcMem.get_desc();
  const auto reshapedMemDesc =
      detail::oneDnnContiguousMemDescFromShape(shape, oneDnnType);
  auto reshapedMem = dnnl::memory(reshapedMemDesc, engine);

  const auto contiguousDstDesc =
      detail::oneDnnContiguousMemDescFromShape(tensor.shape(), oneDnnType);

  try {
    const auto reorderPrimitiveDesc =
        dnnl::reorder::primitive_desc(engine, srcMemDesc, engine, contiguousDstDesc);
    const auto reorderPrimitive = dnnl::reorder(reorderPrimitiveDesc);

    auto& opStream = static_cast<const AsyncCPUStream&>(tensor.stream());
    opStream.enqueue(
        [prim = reorderPrimitive, src = srcMem, dst = reshapedMem,
         srcKeepAlive = srcTensor.getSharedData(),
         &ns = nativeStream]() mutable {
          prim.execute(ns, src, dst);
        });
  } catch (const dnnl::error&) {
    tensor.stream().sync();
    if (tensor.isContiguous()) {
      void* srcData = srcTensor.bufferPtr();
      void* dstData = reshapedMem.get_data_handle();
      std::memcpy(dstData, srcData,
          tensor.elements() * fl::getTypeSize(tensor.type()));
    } else {
      Tensor(srcTensor.clone()).hostSync(reshapedMem.get_data_handle());
    }
  }
  auto result = dnnlMemToTensor(shape, std::move(reshapedMem), asyncStream);
  if (!detail::isTypeSupportedByOneDnn(tensor.type())) {
    toCPUTensor(result).setFlType(tensor.type());
  }
  return result;
}

Tensor transpose(
    const Tensor& tensor,
    const Shape& axes,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream) {
  if (tensor.ndim() <= 1) {
    return tensor.copy();
  }
  Shape newShape = tensor.shape();
  std::vector<Dim> oldToNewAxes = axes.get();
  if (axes.ndim() == 0) { // default, reverse all axes
    oldToNewAxes.resize(tensor.ndim());
    std::reverse(newShape.get().begin(), newShape.get().end());
    std::iota(oldToNewAxes.begin(), oldToNewAxes.end(), 0);
    std::reverse(oldToNewAxes.begin(), oldToNewAxes.end());
  } else if (axes.ndim() == tensor.ndim()) {
    for (int axis = 0; axis < axes.ndim(); axis++) {
      newShape[axis] = tensor.dim(oldToNewAxes[axis]);
    }
  } else {
    throw std::invalid_argument(
        "[OneDnnOps::transpose] Invalid axes: " + axes.toString() +
        " for shape: " + tensor.shape().toString());
  }

  auto& srcTensor = toCPUTensor(tensor);
  const auto oneDnnType = detail::flToOneDnnType(tensor.type());
  auto srcMem = detail::makeDnnlMemory(
      srcTensor.bufferPtr(), tensor.shape(), srcTensor.stridesVec(),
      tensor.type(), engine);
  const auto srcMemDesc = srcMem.get_desc();
  const auto srcMemDims = srcMemDesc.get_dims();
  const auto dstMemDesc =
      detail::oneDnnContiguousMemDescFromShape(newShape, oneDnnType);
  auto dstMem = dnnl::memory(dstMemDesc, engine);

  const auto reorderDstStrides =
      getStridesAfterPermuteAxes(srcMemDims, oldToNewAxes);
  const auto reorderDstMemDesc =
      dnnl::memory::desc(srcMemDims, oneDnnType, reorderDstStrides);
  try {
    const auto reorderPrimitiveDesc = dnnl::reorder::primitive_desc(
        engine, srcMemDesc, engine, reorderDstMemDesc);
    const auto reorderPrimitive = dnnl::reorder(reorderPrimitiveDesc);

    auto& opStream = static_cast<const AsyncCPUStream&>(tensor.stream());
    opStream.enqueue(
        [prim = reorderPrimitive, src = srcMem, dst = dstMem,
         srcKeepAlive = srcTensor.getSharedData(),
         &ns = nativeStream]() mutable {
          prim.execute(ns, src, dst);
        });
  } catch (const dnnl::error&) {
    tensor.stream().sync();
    const auto typeSize = fl::getTypeSize(tensor.type());
    fl::native::transposeCpu(
        dstMem.get_data_handle(),
        srcTensor.bufferPtr(),
        tensor.shape().get(),
        oldToNewAxes,
        typeSize);
  }
  auto result = dnnlMemToTensor(newShape, std::move(dstMem), asyncStream);
  if (!detail::isTypeSupportedByOneDnn(tensor.type())) {
    toCPUTensor(result).setFlType(tensor.type());
  }
  return result;
}

Tensor tile(
    const Tensor& tensor,
    const Shape& tileDims,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream) {
  const auto [paddedTensorShape, paddedTileDims] =
      padDimsForTile(tensor.shape(), tileDims);

  // If all tile dims are 1, return a copy — no concat needed.
  {
    bool allOnes = true;
    for (int i = 0; i < paddedTileDims.ndim(); ++i) {
      if (paddedTileDims[i] != 1) { allOnes = false; break; }
    }
    if (allOnes) {
      return tensor.copy();
    }
  }

  // dnnl::concat doesn't support f64 — use native CPU fallback
  if (tensor.type() == dtype::f64) {
    return fl::native::tileCpu(tensor, paddedTensorShape, paddedTileDims);
  }
  auto& srcTensor = toCPUTensor(tensor);
  auto currTiledMem = detail::makeDnnlMemory(
      srcTensor.bufferPtr(), tensor.shape(), srcTensor.stridesVec(),
      tensor.type(), engine);
  auto currTiledMemDesc = currTiledMem.get_desc().reshape(
      detail::shapeToOneDnnDims(paddedTensorShape));
  std::vector<Dim> finalDims;
  for (int axis = 0; axis < paddedTileDims.ndim(); axis++) {
    const auto numTiles = paddedTileDims[axis];
    finalDims.push_back(numTiles * paddedTensorShape[axis]);
    if (numTiles > 1) {
      std::vector<dnnl::memory::desc> tileMemDescs(numTiles, currTiledMemDesc);

      // Compute the output shape: same as current shape but with this axis expanded
      auto concatDims = currTiledMemDesc.get_dims();
      concatDims[axis] *= numTiles;
      // Use an explicit contiguous destination desc so OneDNN doesn't choose
      // a non-standard (e.g. blocked) layout that dnnlMemToTensor can't handle.
      auto dstType = detail::flToOneDnnType(tensor.type());
      auto concatShape = Shape(std::vector<Dim>(concatDims.begin(), concatDims.end()));
      auto newTileMemDesc = detail::oneDnnContiguousMemDescFromShape(
          concatShape, dstType);
      const dnnl::concat::primitive_desc concatPrimitiveDesc(
          engine, newTileMemDesc, axis, tileMemDescs);
      const dnnl::concat concatPrimitive(concatPrimitiveDesc);
      auto newTiledMem = dnnl::memory(newTileMemDesc, engine);

      std::unordered_map<int, dnnl::memory> args{{DNNL_ARG_DST, newTiledMem}};
      for (int tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
        args.insert({DNNL_ARG_MULTIPLE_SRC + tileIdx, currTiledMem});
      }

      auto& opStream = static_cast<const AsyncCPUStream&>(tensor.stream());
      opStream.enqueue(
          [prim = concatPrimitive, execArgs = std::move(args),
           srcKeepAlive = srcTensor.getSharedData(),
           &ns = nativeStream]() {
            prim.execute(ns, execArgs);
          });
      currTiledMemDesc = newTileMemDesc;
      currTiledMem = newTiledMem;
    }
  }
  auto result = dnnlMemToTensor(Shape(finalDims), std::move(currTiledMem), asyncStream);
  if (!detail::isTypeSupportedByOneDnn(tensor.type())) {
    toCPUTensor(result).setFlType(tensor.type());
  }
  return result;
}

Tensor concatenate(
    const std::vector<Tensor>& tensors,
    unsigned axis,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream) {
  if (tensors.empty()) {
    throw std::invalid_argument("[OneDnnOps::concatenate] empty input");
  }
  if (tensors.size() == 1) {
    return tensors[0].copy();
  }

  // Expand all tensors to have at least axis+1 dimensions (pad with leading 1s)
  const unsigned minNdim = axis + 1;
  auto padToNdim = [minNdim, axis](const Shape& shape) -> std::vector<Dim> {
    auto dims = shape.get();
    if (dims.size() == 1 && dims[0] == 0 && minNdim > 1) {
      dims.assign(minNdim, 1);
      dims[axis] = 0;
      return dims;
    }
    while (dims.size() < minNdim) {
      dims.push_back(1);
    }
    return dims;
  };

  auto outDims = padToNdim(tensors[0].shape());
  const auto ndim = outDims.size();
  for (size_t t = 1; t < tensors.size(); t++) {
    auto tDims = padToNdim(tensors[t].shape());
    if (tDims.size() != ndim) {
      throw std::invalid_argument(
          "[OneDnnOps::concatenate] dimension mismatch");
    }
    for (size_t d = 0; d < ndim; d++) {
      if (d == axis) {
        outDims[d] += tDims[d];
      } else if (tDims[d] != outDims[d]) {
        throw std::invalid_argument(
            "[OneDnnOps::concatenate] shape mismatch on non-concat axis");
      }
    }
  }
  Shape outShape(outDims);

  if (outShape.elements() == 0) {
    return toTensor<CPUTensor>(
        outShape, tensors[0].type(), nullptr, Location::Host);
  }

  const auto outType = tensors[0].type();

  // dnnl::concat doesn't support f64 — CPU memcpy fallback
  if (outType == dtype::f64) {
    std::vector<double> out(outShape.elements());
    size_t outIdx = 0;
    Dim innerSize = 1;
    for (size_t d = axis + 1; d < ndim; d++) {
      innerSize *= outDims[d];
    }
    Dim outerSize = 1;
    for (size_t d = 0; d < axis; d++) {
      outerSize *= outDims[d];
    }
    out.clear();
    out.resize(outShape.elements());
    outIdx = 0;
    if (axis == 0) {
      for (const auto& t : tensors) {
        auto data = t.toHostVectorSync<double>();
        std::memcpy(out.data() + outIdx, data.data(), data.size() * sizeof(double));
        outIdx += data.size();
      }
    } else {
      for (Dim outer = 0; outer < outerSize; outer++) {
        for (const auto& t : tensors) {
          auto tDims = padToNdim(t.shape());
          Dim sliceSize = tDims[axis] * innerSize;
          auto data = t.toHostVectorSync<double>();
          std::memcpy(
              out.data() + outIdx,
              data.data() + outer * sliceSize,
              sliceSize * sizeof(double));
          outIdx += sliceSize;
        }
      }
    }
    return toTensor<CPUTensor>(outShape, dtype::f64, out.data(), Location::Host);
  }

  // Use dnnl::concat for all OneDNN-supported types.
  std::vector<Tensor> contiguousCopies;
  std::vector<dnnl::memory::desc> srcDescs;
  std::vector<dnnl::memory> srcMems;
  for (const auto& t : tensors) {
    auto& cpuT = toCPUTensor(t);
    Tensor contiguousCopy;
    CPUTensor* srcT = &cpuT;
    if (!t.isContiguous()) {
      contiguousCopies.push_back(t.asContiguousTensor());
      srcT = &toCPUTensor(contiguousCopies.back());
    }
    auto mem = detail::makeDnnlMemory(
        srcT->bufferPtr(), t.shape(), srcT->stridesVec(),
        t.type(), engine);
    auto desc = mem.get_desc();
    if (static_cast<unsigned>(desc.get_ndims()) < minNdim) {
      desc = desc.reshape(
          detail::shapeToOneDnnDims(Shape(padToNdim(t.shape()))));
    }
    srcDescs.push_back(desc);
    srcMems.push_back(mem);
  }

  // Use an explicit contiguous destination desc so OneDNN doesn't choose
  // a non-standard (e.g. blocked) layout that dnnlMemToTensor can't handle.
  auto dstMemDesc = detail::oneDnnContiguousMemDescFromShape(
      outShape, detail::flToOneDnnType(outType));
  const dnnl::concat::primitive_desc concatPd(engine, dstMemDesc, axis, srcDescs);
  const dnnl::concat concatPrim(concatPd);
  auto dstMem = dnnl::memory(dstMemDesc, engine);

  std::unordered_map<int, dnnl::memory> args{{DNNL_ARG_DST, dstMem}};
  for (size_t i = 0; i < srcMems.size(); ++i) {
    args.insert({DNNL_ARG_MULTIPLE_SRC + static_cast<int>(i), srcMems[i]});
  }

  auto& opStream = static_cast<const AsyncCPUStream&>(tensors[0].stream());
  for (size_t i = 1; i < tensors.size(); i++) {
    relativeSync(opStream, {tensors[i]});
  }
  // The task reads the contiguous copies made above, which are locals here.
  std::vector<std::shared_ptr<CPUTensor::SharedData>> srcKeepAlive;
  for (const auto& t : tensors) {
    srcKeepAlive.push_back(toCPUTensor(t).getSharedData());
  }
  for (const auto& t : contiguousCopies) {
    srcKeepAlive.push_back(toCPUTensor(t).getSharedData());
  }
  opStream.enqueue([prim = concatPrim, execArgs = std::move(args),
                    srcKeepAlive = std::move(srcKeepAlive),
                    &ns = nativeStream]() {
    prim.execute(ns, execArgs);
  });
  auto result = dnnlMemToTensor(outShape, std::move(dstMem), asyncStream);
  if (!detail::isTypeSupportedByOneDnn(outType)) {
    toCPUTensor(result).setFlType(outType);
  }
  return result;
}

void matmul(
    Tensor& out,
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream,
    const dnnl::primitive_attr& attr,
    const Tensor& bias) {
  // OneDNN matmul does not support f64. Use CBLAS-based native implementation.
  if (lhs.type() == fl::dtype::f64 || rhs.type() == fl::dtype::f64) {
    out = fl::native::matmulF64(lhs, rhs, lhsProp, rhsProp);
    return;
  }

  std::vector<Dim> lhsDims = lhs.shape().get();
  std::vector<Dim> rhsDims = rhs.shape().get();
  const bool isLhsScalarOrVector = lhsDims.size() <= 1;
  const bool isRhsScalarOrVector = rhsDims.size() <= 1;
  auto& lhsTensor = toCPUTensor(lhs);
  auto& rhsTensor = toCPUTensor(rhs);
  auto lhsMem = detail::makeDnnlMemory(
      lhsTensor.bufferPtr(), lhs.shape(), lhsTensor.stridesVec(),
      lhs.type(), engine);
  auto rhsMem = detail::makeDnnlMemory(
      rhsTensor.bufferPtr(), rhs.shape(), rhsTensor.stridesVec(),
      rhs.type(), engine);
  auto lhsMemDesc = lhsMem.get_desc();
  auto rhsMemDesc = rhsMem.get_desc();

  const bool isBatchMatmul =
      lhsDims.size() >= 3 || rhsDims.size() >= 3;
  if (isBatchMatmul) {
    const int maxNdim = std::max(lhsDims.size(), rhsDims.size());
    while (static_cast<int>(lhsDims.size()) < maxNdim) {
      lhsDims.insert(lhsDims.begin(), 1);
    }
    while (static_cast<int>(rhsDims.size()) < maxNdim) {
      rhsDims.insert(rhsDims.begin(), 1);
    }

    if (lhsMemDesc.get_ndims() < maxNdim) {
      lhsMemDesc = lhsMemDesc.reshape(detail::flDimsToOneDnnDims(lhsDims));
    }
    if (rhsMemDesc.get_ndims() < maxNdim) {
      rhsMemDesc = rhsMemDesc.reshape(detail::flDimsToOneDnnDims(rhsDims));
    }

    if (lhsProp == MatrixProperty::Transpose) {
      std::swap(lhsDims[maxNdim - 2], lhsDims[maxNdim - 1]);
      lhsMemDesc = transposeInnerMatrix(lhsMemDesc);
    }
    if (rhsProp == MatrixProperty::Transpose) {
      std::swap(rhsDims[maxNdim - 2], rhsDims[maxNdim - 1]);
      rhsMemDesc = transposeInnerMatrix(rhsMemDesc);
    }

    if (lhsDims[maxNdim - 1] != rhsDims[maxNdim - 2]) {
      std::ostringstream oss;
      oss << "Cannot perform matmul for tensors of shapes: " << lhs.shape()
          << " and " << rhs.shape();
      throw std::invalid_argument(oss.str());
    }

    std::vector<Dim> dstDims(maxNdim);
    for (int i = 0; i < maxNdim - 2; ++i) {
      dstDims[i] = std::max(lhsDims[i], rhsDims[i]);
    }
    dstDims[maxNdim - 2] = lhsDims[maxNdim - 2];
    dstDims[maxNdim - 1] = rhsDims[maxNdim - 1];
    Shape dstShape(dstDims);

    const auto dstType = detail::getTypeWithLargerRange(
        lhsMemDesc.get_data_type(), rhsMemDesc.get_data_type());
    auto dstMemDesc =
        detail::oneDnnContiguousMemDescFromShape(dstShape, dstType);
    const bool allocated = out.isEmpty();
    dnnl::memory dstMem = allocated
        ? dnnl::memory(dstMemDesc, engine)
        : dnnl::memory(dstMemDesc, engine, toCPUTensor(out).bufferPtr());

    auto lhsMatMem = dnnl::memory(lhsMemDesc, engine, lhsMem.get_data_handle());
    auto rhsMatMem = dnnl::memory(rhsMemDesc, engine, rhsMem.get_data_handle());

    const bool hasBias = !bias.isEmpty();
    dnnl::memory biasMem;
    if (hasBias) {
      auto& biasTensor = toCPUTensor(bias);
      biasMem = detail::makeDnnlMemory(
          biasTensor.bufferPtr(), bias.shape(), biasTensor.stridesVec(),
          bias.type(), engine);
      // Pad bias to match output ndims with leading 1s.
      auto biasDesc = biasMem.get_desc();
      if (biasDesc.get_ndims() < maxNdim) {
        dnnl::memory::dims padded(maxNdim, 1);
        auto dims = biasDesc.get_dims();
        int offset = maxNdim - static_cast<int>(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
          padded[offset + i] = dims[i];
        }
        biasDesc = biasDesc.reshape(padded);
      }
      biasMem = dnnl::memory(biasDesc, engine, biasMem.get_data_handle());
    }

    const auto pd = hasBias
        ? dnnl::matmul::primitive_desc(
              engine, lhsMemDesc, rhsMemDesc, biasMem.get_desc(), dstMemDesc, attr)
        : dnnl::matmul::primitive_desc(
              engine, lhsMemDesc, rhsMemDesc, dstMemDesc, attr);
    auto matmulPrim = dnnl::matmul(pd);
    std::unordered_map<int, dnnl::memory> matmulArgs = {
        {DNNL_ARG_SRC, lhsMatMem},
        {DNNL_ARG_WEIGHTS, rhsMatMem},
        {DNNL_ARG_DST, dstMem}};
    if (hasBias) {
      matmulArgs[DNNL_ARG_BIAS] = biasMem;
    }

    auto& opStream = static_cast<const AsyncCPUStream&>(lhs.stream());
    relativeSync(opStream, {rhs});
    if (hasBias) {
      relativeSync(opStream, {bias});
    }
    opStream.enqueue(
        [prim = std::move(matmulPrim), execArgs = std::move(matmulArgs),
         lhsKeepAlive = lhsTensor.getSharedData(),
         rhsKeepAlive = rhsTensor.getSharedData(),
         biasKeepAlive = hasBias ? toCPUTensor(bias).getSharedData() : nullptr,
         outKeepAlive = keepAliveIfPreassigned(allocated, out),
         &ns = nativeStream]() {
          prim.execute(ns, execArgs);
        });

    if (allocated) {
      out = dnnlMemToTensor(dstShape, std::move(dstMem), asyncStream);
    }
    return;
  }

  // 2D / scalar / vector path
  if (isLhsScalarOrVector) {
    lhsDims.insert(lhsDims.begin(), 2 - lhsDims.size(), 1);
    lhsMemDesc = lhsMemDesc.reshape(detail::flDimsToOneDnnDims(lhsDims));
  } else if (lhsProp == MatrixProperty::Transpose) {
    std::swap(lhsDims[0], lhsDims[1]);
    lhsMemDesc = transposeInnerMatrix(lhsMemDesc);
  }
  if (isRhsScalarOrVector) {
    rhsDims.insert(rhsDims.end(), 2 - rhsDims.size(), 1);
    rhsMemDesc = rhsMemDesc.reshape(detail::flDimsToOneDnnDims(rhsDims));
  } else if (rhsProp == MatrixProperty::Transpose) {
    std::swap(rhsDims[0], rhsDims[1]);
    rhsMemDesc = transposeInnerMatrix(rhsMemDesc);
  }

  if (!(lhsDims.at(1) == rhsDims.at(0) &&
        std::equal(
            lhsDims.begin() + 2,
            lhsDims.end(),
            rhsDims.begin() + 2,
            rhsDims.end()))) {
    std::ostringstream oss;
    oss << "Cannot perform matmul for tensors of shapes: " << lhs.shape()
        << " and " << rhs.shape();
    throw std::invalid_argument(oss.str());
  }
  std::vector<Dim> dstDims = lhsDims;
  dstDims[1] = rhsDims[1];
  Shape dstShape(dstDims);

  const auto dstType = detail::getTypeWithLargerRange(
      lhsMemDesc.get_data_type(), rhsMemDesc.get_data_type());
  const auto dstMemArgDesc =
      detail::oneDnnContiguousMemDescFromShape(dstShape, dstType);
  auto dstMemDesc = dstMemArgDesc;
  if (isLhsScalarOrVector || isRhsScalarOrVector) {
    const auto elems = dstShape.elements();
    dstMemDesc = dstMemArgDesc.reshape({elems});
    dstShape = {elems};
  }
  const bool allocated = out.isEmpty();
  dnnl::memory dstMem = allocated
      ? dnnl::memory(dstMemDesc, engine)
      : dnnl::memory(dstMemDesc, engine, toCPUTensor(out).bufferPtr());

  const bool hasBias = !bias.isEmpty();
  dnnl::memory biasMem;
  if (hasBias) {
    auto& biasTensor = toCPUTensor(bias);
    biasMem = detail::makeDnnlMemory(
        biasTensor.bufferPtr(), bias.shape(), biasTensor.stridesVec(),
        bias.type(), engine);
    // Pad bias to 2D with leading 1 if needed: (N,) → (1, N)
    auto biasDesc = biasMem.get_desc();
    if (biasDesc.get_ndims() < 2) {
      auto dims = biasDesc.get_dims();
      dims.insert(dims.begin(), 1);
      biasDesc = biasDesc.reshape(dims);
    }
    biasMem = dnnl::memory(biasDesc, engine, biasMem.get_data_handle());
  }

  const auto matmulPrimitiveDesc = hasBias
      ? dnnl::matmul::primitive_desc(
            engine, lhsMemDesc, rhsMemDesc, biasMem.get_desc(), dstMemArgDesc, attr)
      : dnnl::matmul::primitive_desc(
            engine, lhsMemDesc, rhsMemDesc, dstMemArgDesc, attr);
  const auto matmulPrimitive = dnnl::matmul(matmulPrimitiveDesc);

  std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC, lhsMem},
      {DNNL_ARG_WEIGHTS, rhsMem},
      {DNNL_ARG_DST, dstMem},
  };
  if (hasBias) {
    args[DNNL_ARG_BIAS] = biasMem;
  }

  auto& opStream = static_cast<const AsyncCPUStream&>(lhs.stream());
  relativeSync(opStream, {rhs});
  if (hasBias) {
    relativeSync(opStream, {bias});
  }
  opStream.enqueue(
      [prim = matmulPrimitive, execArgs = std::move(args),
       lhsKeepAlive = lhsTensor.getSharedData(),
       rhsKeepAlive = rhsTensor.getSharedData(),
       biasKeepAlive = hasBias ? toCPUTensor(bias).getSharedData() : nullptr,
       outKeepAlive = keepAliveIfPreassigned(allocated, out),
       &ns = nativeStream]() {
        prim.execute(ns, execArgs);
      });
  if (allocated) {
    out = dnnlMemToTensor(dstShape, std::move(dstMem), asyncStream);
  }
}

Tensor where(
    const Tensor& condition,
    const Tensor& xIn,
    const Tensor& yIn,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream) {
  if (condition.type() != dtype::b8 && condition.type() != dtype::u8) {
    throw std::invalid_argument(
        "[OneDnnOps::where] condition must be a boolean tensor");
  }
  // The result has the IR's dtype, promoteDtypes(x, y), so both branches are
  // promoted to it first.
  const auto outType = promoteDtypes(xIn.type(), yIn.type());
  std::optional<Tensor> xPromoted;
  std::optional<Tensor> yPromoted;
  if (xIn.type() != outType) {
    xPromoted = xIn.astype(outType);
  }
  if (yIn.type() != outType) {
    yPromoted = yIn.astype(outType);
  }
  const Tensor& x = xPromoted ? *xPromoted : xIn;
  const Tensor& y = yPromoted ? *yPromoted : yIn;
  Tensor condU8 = (condition.type() == dtype::u8)
      ? condition
      : condition.astype(dtype::u8);

  auto isSelectSupported = [](fl::dtype t) {
    return t == dtype::f32 || t == dtype::f16;
  };
  if (isSelectSupported(x.type()) && isSelectSupported(y.type())) {
    auto& condTensor = toCPUTensor(condU8);
    auto& xTensor = toCPUTensor(x);
    auto& yTensor = toCPUTensor(y);

    auto condMem = detail::makeDnnlMemory(
        condTensor.bufferPtr(), condU8.shape(), condTensor.stridesVec(),
        condU8.type(), engine);
    auto xMem = detail::makeDnnlMemory(
        xTensor.bufferPtr(), x.shape(), xTensor.stridesVec(),
        x.type(), engine);
    auto yMem = detail::makeDnnlMemory(
        yTensor.bufferPtr(), y.shape(), yTensor.stridesVec(),
        y.type(), engine);
    const auto xMemDesc = xMem.get_desc();
    // The result is wrapped as a contiguous tensor below, so the destination
    // must be contiguous even when x is a strided view.
    const auto dstMemDesc = detail::oneDnnContiguousMemDescFromShape(
        x.shape(), detail::flToOneDnnType(x.type()));
    auto dstMem = dnnl::memory(dstMemDesc, engine);

    const dnnl::binary::primitive_desc pd(
        engine, dnnl::algorithm::binary_select,
        condMem.get_desc(), xMemDesc, yMem.get_desc(), dstMemDesc);
    const dnnl::binary prim(pd);

    auto& opStream = static_cast<const AsyncCPUStream&>(x.stream());
    relativeSync(opStream, {condition, y});
    opStream.enqueue(
        [prim, condMem, xMem, yMem, dstMem,
         condKeepAlive = condTensor.getSharedData(),
         xKeepAlive = xTensor.getSharedData(),
         yKeepAlive = yTensor.getSharedData(), &ns = nativeStream]() {
          prim.execute(ns, {
              {DNNL_ARG_SRC_0, condMem},
              {DNNL_ARG_SRC_1, xMem},
              {DNNL_ARG_SRC_2, yMem},
              {DNNL_ARG_DST, dstMem},
          });
        });
    return dnnlMemToTensor(x.shape(), std::move(dstMem), asyncStream);
  }

  // f64/integer types not supported by OneDNN binary_select — CPU fallback.
  return fl::native::whereCpu(condU8, x, y);
}

void inPlaceBinaryScalar(
    CPUTensor& self,
    double val,
    dnnl::algorithm alg,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream) {
  auto selfMem = detail::makeDnnlMemory(
      self.bufferPtr(), self.shape(), self.stridesVec(),
      self.type(), engine);
  const auto selfMemDesc = selfMem.get_desc();
  const dnnl::memory::data_type type = selfMemDesc.get_data_type();
  auto keepAlive = self.getSharedData();

  // f64 not supported by OneDNN binary — use stride-aware CPU fallback
  if (type == dnnl::memory::data_type::f64) {
    const auto& opStream =
        static_cast<const AsyncCPUStream&>(self.stream());
    const Dim n = self.shape().elements();
    opStream.enqueue([selfMem, selfMemDesc, keepAlive, n, val, alg]() mutable {
      inPlaceBinaryScalarCpu<double>(selfMem, selfMemDesc, n, val, alg);
    });
    return;
  }

  // Create a scalar memory with shape (1, 1, ..., 1) for broadcasting.
  const int ndim = std::max(1, selfMemDesc.get_ndims());
  dnnl::memory::dims scalarDims(ndim, 1);
  dnnl::memory::dims scalarStrides(ndim, 1);
  dnnl::memory::desc scalarDesc(scalarDims, type, scalarStrides, true);
  dnnl::memory scalarMem(scalarDesc, engine);
  writeScalarToMemory(scalarMem, type, val);

  const dnnl::binary::primitive_desc primDesc(
      engine, alg, selfMemDesc, scalarDesc, selfMemDesc);
  const dnnl::binary prim(primDesc);

  const auto& opStream =
      static_cast<const AsyncCPUStream&>(self.stream());
  opStream.enqueue(
      [prim, selfMem, scalarMem, keepAlive, &ns = nativeStream]() {
        prim.execute(ns, {
            {DNNL_ARG_SRC_0, selfMem},
            {DNNL_ARG_SRC_1, scalarMem},
            {DNNL_ARG_DST, selfMem},
        });
      });
}

void inPlaceBinaryTensor(
    CPUTensor& self,
    const Tensor& other,
    dnnl::algorithm alg,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream) {
  auto& otherTensor = toCPUTensor(other);
  const auto& selfStream = self.stream();
  relativeSync(selfStream, {other});

  auto selfMem = detail::makeDnnlMemory(
      self.bufferPtr(), self.shape(), self.stridesVec(),
      self.type(), engine);
  auto otherMem = detail::makeDnnlMemory(
      otherTensor.bufferPtr(), other.shape(), otherTensor.stridesVec(),
      other.type(), engine);
  const auto selfDesc = selfMem.get_desc();
  const auto otherDesc = otherMem.get_desc();
  auto selfKeepAlive = self.getSharedData();
  auto otherKeepAlive = otherTensor.getSharedData();

  // f64 not supported by OneDNN binary — use stride-aware CPU fallback
  if (selfDesc.get_data_type() == dnnl::memory::data_type::f64) {
    const auto& opStream = static_cast<const AsyncCPUStream&>(selfStream);
    const Dim n = self.shape().elements();
    opStream.enqueue([selfMem, selfDesc, otherMem, otherDesc,
                      selfKeepAlive, otherKeepAlive, n,
                      alg]() mutable {
      inPlaceBinaryTensorCpu<double>(
          selfMem, selfDesc, otherMem, otherDesc, n, alg);
    });
    return;
  }

  if (selfDesc.get_ndims() != otherDesc.get_ndims()) {
    throw std::invalid_argument(
        "[OneDnnOps] in-place op shape mismatch: different ndims");
  }
  auto selfDims = selfDesc.get_dims();
  auto otherDims = otherDesc.get_dims();
  for (int i = 0; i < selfDesc.get_ndims(); i++) {
    if (selfDims[i] != otherDims[i] && otherDims[i] != 1) {
      throw std::invalid_argument(
          "[OneDnnOps] in-place op shape mismatch");
    }
  }

  const dnnl::binary::primitive_desc primDesc(
      engine, alg, selfDesc, otherDesc, selfDesc);
  const dnnl::binary prim(primDesc);

  const auto& opStream = static_cast<const AsyncCPUStream&>(selfStream);
  opStream.enqueue(
      [prim, selfMem, otherMem, selfKeepAlive, otherKeepAlive,
       &ns = nativeStream]() {
        prim.execute(ns, {
            {DNNL_ARG_SRC_0, selfMem},
            {DNNL_ARG_SRC_1, otherMem},
            {DNNL_ARG_DST, selfMem},
        });
      });
}

void fillInPlace(
    CPUTensor& self,
    double val,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream) {
  auto selfMem = detail::makeDnnlMemory(
      self.bufferPtr(), self.shape(), self.stridesVec(),
      self.type(), engine);
  const auto selfMemDesc = selfMem.get_desc();
  const auto type = selfMemDesc.get_data_type();
  auto keepAlive = self.getSharedData();

  if (type == dnnl::memory::data_type::f64) {
    const auto& opStream =
        static_cast<const AsyncCPUStream&>(self.stream());
    const Dim n = self.shape().elements();
    auto flType = self.type();
    if (flType == dtype::s64) {
      opStream.enqueue([selfMem, selfMemDesc, keepAlive, n, val]() mutable {
        auto dims = selfMemDesc.get_dims();
        auto strides = selfMemDesc.get_strides();
        fl::native::fillBufferCpu<int64_t>(
            static_cast<int64_t*>(selfMem.get_data_handle()),
            n, static_cast<int64_t>(val),
            static_cast<int>(dims.size()), dims.data(), strides.data());
      });
    } else if (flType == dtype::u64) {
      opStream.enqueue([selfMem, selfMemDesc, keepAlive, n, val]() mutable {
        auto dims = selfMemDesc.get_dims();
        auto strides = selfMemDesc.get_strides();
        fl::native::fillBufferCpu<uint64_t>(
            static_cast<uint64_t*>(selfMem.get_data_handle()),
            n, static_cast<uint64_t>(val),
            static_cast<int>(dims.size()), dims.data(), strides.data());
      });
    } else {
      opStream.enqueue([selfMem, selfMemDesc, keepAlive, n, val]() mutable {
        auto dims = selfMemDesc.get_dims();
        auto strides = selfMemDesc.get_strides();
        fl::native::fillBufferCpu<double>(
            static_cast<double*>(selfMem.get_data_handle()),
            n, val,
            static_cast<int>(dims.size()), dims.data(), strides.data());
      });
    }
    return;
  }

  const dnnl::eltwise_forward::primitive_desc primDesc(
      engine,
      dnnl::prop_kind::forward_inference,
      dnnl::algorithm::eltwise_linear,
      selfMemDesc,
      selfMemDesc,
      /*alpha=*/0.0f,
      /*beta=*/static_cast<float>(val));
  const dnnl::eltwise_forward prim(primDesc);

  const auto& opStream =
      static_cast<const AsyncCPUStream&>(self.stream());
  opStream.enqueue(
      [prim, selfMem, keepAlive, &ns = nativeStream]() {
        prim.execute(ns, {
            {DNNL_ARG_SRC, selfMem},
            {DNNL_ARG_DST, selfMem},
        });
      });
}

Tensor reorderAstype(
    CPUTensor& self,
    fl::dtype type,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream) {
  if (type == self.type()) {
    // Return a view sharing the same data
    auto viewSharedData = std::make_shared<CPUTensor::SharedData>();
    viewSharedData->bufferPtr = self.getSharedData()->bufferPtr;
    viewSharedData->buffer = self.getSharedData()->buffer;
    viewSharedData->bufferOwner = self.getSharedData()->bufferOwner;
    viewSharedData->parent = self.getSharedData();
    viewSharedData->isDataReady = self.getSharedData()->isDataReady;
    auto view = std::make_unique<CPUTensor>(viewSharedData, self.shape());
    // Preserve stream and strides via clone-like path
    return Tensor(std::move(view));
  }

  // OneDNN reorder path: try using OneDNN for type conversion.
  // Skip float->int because OneDNN uses round-to-nearest-even while
  // C++ convention (and test expectations) use truncation.
  const auto srcType = self.type();
  bool srcIsFloat = (srcType == dtype::f32 || srcType == dtype::f64 ||
                     srcType == dtype::f16);
  bool dstIsInt = (type == dtype::s32 || type == dtype::s16 ||
                   type == dtype::s64 || type == dtype::u8 ||
                   type == dtype::u16 || type == dtype::u32 ||
                   type == dtype::u64 || type == dtype::b8);

  if (!(srcIsFloat && dstIsInt)) {
    try {
      auto srcMem = detail::makeDnnlMemory(
          self.bufferPtr(), self.shape(), self.stridesVec(), srcType, engine);
      const auto srcMemDesc = srcMem.get_desc();
      const auto dstMemDesc = detail::oneDnnContiguousMemDescFromShape(
          self.shape(), detail::flToOneDnnType(type));
      auto dstMem = dnnl::memory(dstMemDesc, engine);
      const auto reorderPrimitiveDesc =
          dnnl::reorder::primitive_desc(engine, srcMemDesc, engine, dstMemDesc);
      const auto reorderPrimitive = dnnl::reorder(reorderPrimitiveDesc);

      auto selfStreamPtr = self.backend().asyncStream();
      selfStreamPtr->enqueue(
          [prim = reorderPrimitive, src = srcMem, dst = dstMem,
           keepAlive = self.getSharedData(),
           &ns = nativeStream]() mutable {
            prim.execute(ns, src, dst);
          });
      void* dstBuf = dstMem.get_data_handle();
      auto owner = std::make_shared<dnnl::memory>(std::move(dstMem));
      auto result = toTensor<CPUTensor>(
          self.shape(), type, dstBuf, std::move(owner), selfStreamPtr);
      if (!detail::isTypeSupportedByOneDnn(type)) {
        toCPUTensor(result).setFlType(type);
      }
      return result;
    } catch (const dnnl::error&) {
      // Fall through to CPU fallback — handled by CPUTensor
    }
  }

  // Return empty tensor to signal caller should use CPU fallback
  return Tensor();
}

void assignTensor(
    CPUTensor& self,
    const Tensor& src,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream) {
  auto& other = toCPUTensor(src);

  auto thisMem = detail::makeDnnlMemory(
      self.bufferPtr(), self.shape(), self.stridesVec(),
      self.type(), engine);
  auto otherMem = detail::makeDnnlMemory(
      other.bufferPtr(), other.shape(), other.stridesVec(),
      other.type(), engine);
  try {
    const auto reorderPrimitiveDesc = dnnl::reorder::primitive_desc(
        engine, otherMem.get_desc(), engine, thisMem.get_desc());
    const auto reorderPrimitive = dnnl::reorder(reorderPrimitiveDesc);

    auto& selfStream = static_cast<const AsyncCPUStream&>(self.stream());
    selfStream.enqueue(
        [prim = reorderPrimitive, srcMem = otherMem, dstMem = thisMem,
         keepSrc = other.getSharedData(), keepDst = self.getSharedData(),
         &ns = nativeStream]() mutable {
          prim.execute(ns, srcMem, dstMem);
        });
  } catch (const dnnl::error&) {
    // Reorder not supported (e.g. f64). Fall back to stride-aware CPU copy.
    self.stream().sync();
    const auto srcType = detail::flToOneDnnType(other.type());
    const auto dstType = detail::flToOneDnnType(self.type());
    const auto srcTypeSize = fl::getTypeSize(other.type());
    const auto* srcBase = static_cast<const char*>(other.bufferPtr());
    auto* dstBase = static_cast<char*>(self.bufferPtr());
    const Dim n = self.shape().elements();

    const auto& srcStridesVec = other.stridesVec();
    const auto srcDimsVec = other.shape().get();
    const auto& dstStridesVec = self.stridesVec();
    const auto dstDimsVec = self.shape().get();
    dnnl::memory::dims srcDims(srcDimsVec.begin(), srcDimsVec.end());
    dnnl::memory::dims srcStrides(srcStridesVec.begin(), srcStridesVec.end());
    dnnl::memory::dims dstDims(dstDimsVec.begin(), dstDimsVec.end());
    dnnl::memory::dims dstStrides(dstStridesVec.begin(), dstStridesVec.end());
    const int ndim = static_cast<int>(srcDims.size());

    bool srcContig = isContiguousLayout(srcDims, srcStrides);
    bool dstContig = isContiguousLayout(dstDims, dstStrides);

    if (srcContig && dstContig && srcType == dstType) {
      std::memcpy(dstBase, srcBase, n * srcTypeSize);
    } else {
      auto readDouble = [&](Dim srcOff) -> double {
        switch (srcType) {
          case dnnl::memory::data_type::f64:
            return reinterpret_cast<const double*>(srcBase)[srcOff];
          case dnnl::memory::data_type::f32:
            return reinterpret_cast<const float*>(srcBase)[srcOff];
          case dnnl::memory::data_type::s32:
            return reinterpret_cast<const int32_t*>(srcBase)[srcOff];
          case dnnl::memory::data_type::s8:
            return reinterpret_cast<const int8_t*>(srcBase)[srcOff];
          case dnnl::memory::data_type::u8:
            return reinterpret_cast<const uint8_t*>(srcBase)[srcOff];
          default:
            throw std::runtime_error(
                "[OneDnnOps::assignTensor] unsupported source dtype");
        }
      };

      auto writeDouble = [&](Dim dstOff, double val) {
        switch (dstType) {
          case dnnl::memory::data_type::f64:
            reinterpret_cast<double*>(dstBase)[dstOff] = val; break;
          case dnnl::memory::data_type::f32:
            reinterpret_cast<float*>(dstBase)[dstOff] =
                static_cast<float>(val); break;
          case dnnl::memory::data_type::s32:
            reinterpret_cast<int32_t*>(dstBase)[dstOff] =
                static_cast<int32_t>(val); break;
          case dnnl::memory::data_type::s8:
            reinterpret_cast<int8_t*>(dstBase)[dstOff] =
                static_cast<int8_t>(val); break;
          case dnnl::memory::data_type::u8:
            reinterpret_cast<uint8_t*>(dstBase)[dstOff] =
                static_cast<uint8_t>(val); break;
          default:
            throw std::runtime_error(
                "[OneDnnOps::assignTensor] unsupported dest dtype");
        }
      };

      for (Dim i = 0; i < n; ++i) {
        Dim srcOff = srcContig ? i : stridedOffset(i, ndim, srcDims, srcStrides);
        Dim dstOff = dstContig ? i : stridedOffset(i, ndim, dstDims, dstStrides);
        writeDouble(dstOff, readDouble(srcOff));
      }
    }
  }
}

} // namespace detail::onednn

} // namespace fl
