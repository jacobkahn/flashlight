/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/TensorBackend.h"

#include <cstring>
#include <stdexcept>
#include <vector>

#include "flashlight/fl/tensor/BackendDispatch.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl {

BackendDispatch TensorBackend::createDispatch() {
  return makeBackendDispatch(*this);
}

namespace detail {

bool areBackendsEqual(const Tensor& a, const Tensor& b) {
  return a.backendType() == b.backendType();
}

} // namespace detail

bool TensorBackend::isDataTypeSupported(const fl::dtype& dtype) const {
  bool supported = this->supportsDataType(dtype);
  for (auto& p : extensions_) {
    supported &= p.second->isDataTypeSupported(dtype);
  }
  return supported;
}

void TensorBackend::clip(
    Tensor& out,
    const Tensor& tensor,
    const Tensor& low,
    const double& high,
    std::shared_ptr<const Stream> stream) {
  clip(out,
      tensor, low, full(tensor.shape(), high, tensor.type()),
      std::move(stream));
}

void TensorBackend::clip(
    Tensor& out,
    const Tensor& tensor,
    const double& low,
    const Tensor& high,
    std::shared_ptr<const Stream> stream) {
  clip(out,
      tensor, full(tensor.shape(), low, tensor.type()), high,
      std::move(stream));
}

void TensorBackend::clip(
    Tensor& out,
    const Tensor& tensor,
    const double& low,
    const double& high,
    std::shared_ptr<const Stream> stream) {
  clip(out,
      tensor,
      full(tensor.shape(), low, tensor.type()),
      full(tensor.shape(), high, tensor.type()),
      std::move(stream));
}

void TensorBackend::where(
    Tensor& out,
    const Tensor& condition,
    const Tensor& x,
    const double& y,
    std::shared_ptr<const Stream> stream) {
  where(out,
      condition, x,
      full(condition.shape(), y, promoteScalarDtype(x.type(), y)),
      std::move(stream));
}

void TensorBackend::where(
    Tensor& out,
    const Tensor& condition,
    const double& x,
    const Tensor& y,
    std::shared_ptr<const Stream> stream) {
  where(out,
      condition,
      full(condition.shape(), x, promoteScalarDtype(y.type(), x)), y,
      std::move(stream));
}

void TensorBackend::indexAdd(
    Tensor& self,
    unsigned axis,
    const Tensor& indices,
    const Tensor& src,
    std::shared_ptr<const Stream> /* stream */) {
  // Default CPU implementation: read all data to host, accumulate, write back.
  // Backends may override for better performance.
  if (axis != 0) {
    throw std::runtime_error(
        "TensorBackend::indexAdd default impl only supports axis=0");
  }
  const Dim nIdx = indices.elements();
  const Dim innerSize = (self.ndim() > 1)
      ? self.elements() / self.dim(0)
      : 1;
  const size_t typeSize = fl::getTypeSize(self.type());

  // Read indices
  auto idxHost = indices.astype(fl::dtype::s32).asContiguousTensor();
  std::vector<int> idxData(nIdx);
  idxHost.hostSync(idxData.data());

  // Read src and self to host
  auto srcContig = src.asContiguousTensor();
  std::vector<char> srcBytes(srcContig.elements() * typeSize);
  srcContig.hostSync(srcBytes.data());

  std::vector<char> selfBytes(self.elements() * typeSize);
  self.hostSync(selfBytes.data());

  // Accumulate: for each index i, add src[i, ...] to self[idx[i], ...]
  for (Dim i = 0; i < nIdx; ++i) {
    int idx = idxData[i];
    char* dst = selfBytes.data() + idx * innerSize * typeSize;
    const char* s = srcBytes.data() + i * innerSize * typeSize;
    // Type-aware addition
    switch (self.type()) {
      case fl::dtype::f32:
        for (Dim j = 0; j < innerSize; ++j) {
          reinterpret_cast<float*>(dst)[j] +=
              reinterpret_cast<const float*>(s)[j];
        }
        break;
      case fl::dtype::f64:
        for (Dim j = 0; j < innerSize; ++j) {
          reinterpret_cast<double*>(dst)[j] +=
              reinterpret_cast<const double*>(s)[j];
        }
        break;
      case fl::dtype::s32:
        for (Dim j = 0; j < innerSize; ++j) {
          reinterpret_cast<int*>(dst)[j] +=
              reinterpret_cast<const int*>(s)[j];
        }
        break;
      default:
        throw std::runtime_error(
            "indexAdd: unsupported dtype " +
            std::to_string(static_cast<int>(self.type())));
    }
  }

  // Write back
  auto result = Tensor::fromBuffer(
      self.shape(), self.type(),
      reinterpret_cast<const uint8_t*>(selfBytes.data()),
      Location::Host);
  self = result.copy();
}

// Default throwing implementations for autograd ops.
// Backends that support these ops override them.
#define FL_AUTOGRAD_OP_NOT_SUPPORTED(NAME)                     \
  throw std::runtime_error(                                    \
      std::string(#NAME) + " not supported by this backend");

ForwardBackwardState
TensorBackend::conv2dForward(
    Tensor&, const Tensor&, const Tensor&, const Tensor&,
    int, int, int, int, int, int, int, bool) {
  FL_AUTOGRAD_OP_NOT_SUPPORTED(conv2dForward);
}

Tensor TensorBackend::conv2dBackwardData(
    const Tensor&, const Tensor&, const Tensor&,
    int, int, int, int, int, int, int,
    const ForwardBackwardState&) {
  FL_AUTOGRAD_OP_NOT_SUPPORTED(conv2dBackwardData);
}

std::pair<Tensor, Tensor> TensorBackend::conv2dBackwardFilterBias(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    int, int, int, int, int, int, int,
    const ForwardBackwardState&) {
  FL_AUTOGRAD_OP_NOT_SUPPORTED(conv2dBackwardFilterBias);
}

ForwardBackwardState
TensorBackend::pool2dForward(
    Tensor&, const Tensor&, int, int, int, int, int, int,
    PoolingMode, bool) {
  FL_AUTOGRAD_OP_NOT_SUPPORTED(pool2dForward);
}

Tensor TensorBackend::pool2dBackward(
    const Tensor&, const Tensor&, const Tensor&,
    int, int, int, int, int, int,
    PoolingMode, const ForwardBackwardState&) {
  FL_AUTOGRAD_OP_NOT_SUPPORTED(pool2dBackward);
}

BatchnormForwardResult
TensorBackend::batchnormForward(
    const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&,
    const std::vector<int>&, bool, double, double, bool) {
  FL_AUTOGRAD_OP_NOT_SUPPORTED(batchnormForward);
}

std::tuple<Tensor, Tensor, Tensor> TensorBackend::batchnormBackward(
    const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&,
    const std::vector<int>&, bool, float,
    const ForwardBackwardState&) {
  FL_AUTOGRAD_OP_NOT_SUPPORTED(batchnormBackward);
}

std::tuple<Tensor, Tensor, Tensor, ForwardBackwardState>
TensorBackend::rnnForward(
    const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, int, int,
    RnnMode, bool, float, bool) {
  FL_AUTOGRAD_OP_NOT_SUPPORTED(rnnForward);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> TensorBackend::rnnBackward(
    const Tensor&, const Tensor&, const Tensor&,
    const Tensor&,
    const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, int, int,
    RnnMode, bool, float,
    const ForwardBackwardState&) {
  FL_AUTOGRAD_OP_NOT_SUPPORTED(rnnBackward);
}

#undef FL_AUTOGRAD_OP_NOT_SUPPORTED

} // namespace fl
