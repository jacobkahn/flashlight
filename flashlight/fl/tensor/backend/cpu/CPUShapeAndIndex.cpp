/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/cpu/CPUBackend.h"

#include <stdexcept>
#include <vector>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/backend/cpu/CPUBackendUtils.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/ops/native/ShapeAndIndex.h"
#include "flashlight/fl/tensor/ops/native/Sort.h"
#include "flashlight/fl/tensor/ops/native/Unary.h"
#include "flashlight/fl/tensor/ops/onednn/OneDnnOps.h"

namespace fl {

using detail::enqueueNativeOp;

/************************ Shaping and Indexing *************************/

void CPUBackend::reshape(Tensor& out, const Tensor& tensor, const Shape& shape, std::shared_ptr<const Stream> /* stream */) {
  out = onednnOps_->reshape(tensor, shape, nativeStream(), asyncStream_);
}

void CPUBackend::transpose(
    Tensor& out,
    const Tensor& tensor,
    const Shape& axes /* = {} */,
    std::shared_ptr<const Stream> /* stream */) {
  out = onednnOps_->transpose(tensor, axes, nativeStream(), asyncStream_);
}

void CPUBackend::tile(Tensor& out, const Tensor& tensor, const Shape& tileDims, std::shared_ptr<const Stream> /* stream */) {
  out = onednnOps_->tile(tensor, tileDims, nativeStream(), asyncStream_);
}

void CPUBackend::concatenate(
    Tensor& out,
    const std::vector<Tensor>& tensors,
    const unsigned axis,
    std::shared_ptr<const Stream> /* stream */) {
  out = onednnOps_->concatenate(tensors, axis, nativeStream(), asyncStream_);
}

void CPUBackend::nonzero(Tensor& out, const Tensor& tensor, std::shared_ptr<const Stream> /* stream */) {
  out = fl::native::nonzeroCpu(tensor);
}

void CPUBackend::pad(
    Tensor& out,
    const Tensor& input,
    const std::vector<std::pair<int, int>>& padWidths,
    const PadType type,
    std::shared_ptr<const Stream> /* stream */) {
  // Compute output shape
  auto inDims = input.shape().get();
  std::vector<Dim> outDims(input.ndim());
  for (int d = 0; d < input.ndim(); d++) {
    outDims[d] = inDims[d] + padWidths[d].first + padWidths[d].second;
  }
  Shape outShape(outDims);
  out = enqueueNativeOp(input, outShape, input.type(), asyncStream_,
      [padWidths, type, inShape = input.shape(), outShape]
      (Tensor& out, const Tensor& in) {
        native::padImpl(out, in, inShape, outShape, padWidths, type);
      });
}

void CPUBackend::flip(Tensor& out, const Tensor& tensor, const unsigned dim, std::shared_ptr<const Stream> /* stream */) {
  out = enqueueNativeOp(tensor, asyncStream_,
      [dim](Tensor& out, const Tensor& in) {
        native::flipImpl(out, in, dim);
      });
}

void CPUBackend::clip(
    Tensor& out,
    const Tensor& tensor,
    const Tensor& low,
    const Tensor& high,
    std::shared_ptr<const Stream> /* stream */) {
  out = maximum(minimum(tensor, high), low);
}

void CPUBackend::clip(
    Tensor& out,
    const Tensor& tensor,
    const double& low,
    const double& high,
    std::shared_ptr<const Stream> /* stream */) {
  onednnOps_->applyEltwiseOp(out, tensor, dnnl::algorithm::eltwise_clip, nativeStream(), asyncStream_, low, high);
}

void CPUBackend::roll(
    Tensor& out,
    const Tensor& tensor,
    const int shift,
    const unsigned axis,
    std::shared_ptr<const Stream> /* stream */) {
  out = enqueueNativeOp(tensor, asyncStream_,
      [shift, axis](Tensor& out, const Tensor& in) {
        native::rollImpl(out, in, shift, axis);
      });
}

void CPUBackend::tril(
    Tensor& out,
    const Tensor& tensor,
    int diagonal,
    std::shared_ptr<const Stream> /* stream */) {
  if (tensor.ndim() < 2) {
    throw std::invalid_argument("tril requires input with rank at least 2");
  }
  out = enqueueNativeOp(tensor, asyncStream_,
      [diagonal](Tensor& out, const Tensor& in) {
        native::trilImpl(out, in, diagonal);
      });
}

void CPUBackend::triu(
    Tensor& out,
    const Tensor& tensor,
    int diagonal,
    std::shared_ptr<const Stream> /* stream */) {
  if (tensor.ndim() < 2) {
    throw std::invalid_argument("triu requires input with rank at least 2");
  }
  out = enqueueNativeOp(tensor, asyncStream_,
      [diagonal](Tensor& out, const Tensor& in) {
        native::triuImpl(out, in, diagonal);
      });
}

void CPUBackend::where(
    Tensor& out,
    const Tensor& condition,
    const Tensor& x,
    const Tensor& y,
    std::shared_ptr<const Stream> /* stream */) {
  out = onednnOps_->where(condition, x, y, nativeStream(), asyncStream_);
}

void CPUBackend::topk(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned k,
    const Dim axis,
    const SortMode sortMode,
    std::shared_ptr<const Stream> /* stream */) {
  Tensor sortedValues, sortedIndices;
  sort(sortedValues, sortedIndices, input, axis, sortMode);
  std::vector<Index> sliceIdx(input.ndim(), fl::span);
  sliceIdx[axis] = fl::range(0, static_cast<int>(k));
  values = sortedValues(sliceIdx);
  indices = sortedIndices(sliceIdx);
}

void CPUBackend::sort(
    Tensor& out,
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode,
    std::shared_ptr<const Stream> /* stream */) {
  out = enqueueNativeOp(input, asyncStream_,
      [axis, sortMode](Tensor& out, const Tensor& in) {
        native::sortImpl(out, in, axis, sortMode);
      });
}

void CPUBackend::sort(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode,
    std::shared_ptr<const Stream> /* stream */) {
  auto shape = input.shape();
  auto type = input.type();
  values = toTensor<CPUTensor>(shape, type, nullptr, Location::Host);
  indices = toTensor<CPUTensor>(shape, dtype::s32, nullptr, Location::Host);
  auto inData = detail::contiguousSharedData(input);
  auto valData = toCPUTensor(values).getSharedData();
  auto idxData = toCPUTensor(indices).getSharedData();
  asyncStream_->relativeSync(input.stream());
  asyncStream_->enqueue(
      [inData, valData, idxData, shape, type,
       stream = asyncStream_, axis, sortMode]() {
        Tensor in(std::make_unique<CPUTensor>(inData, shape, type, stream));
        Tensor v(std::make_unique<CPUTensor>(valData, shape, type, stream));
        Tensor idx(std::make_unique<CPUTensor>(idxData, shape, dtype::s32, stream));
        native::sortWithIndicesImpl(v, idx, in, axis, sortMode);
      });
}

void CPUBackend::argsort(
    Tensor& out,
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode,
    std::shared_ptr<const Stream> /* stream */) {
  auto shape = input.shape();
  auto type = input.type();
  auto values = toTensor<CPUTensor>(shape, type, nullptr, Location::Host);
  auto indices = toTensor<CPUTensor>(shape, dtype::s32, nullptr, Location::Host);
  auto inData = detail::contiguousSharedData(input);
  auto valData = toCPUTensor(values).getSharedData();
  auto idxData = toCPUTensor(indices).getSharedData();
  asyncStream_->relativeSync(input.stream());
  asyncStream_->enqueue(
      [inData, valData, idxData, shape, type,
       stream = asyncStream_, axis, sortMode]() {
        Tensor in(std::make_unique<CPUTensor>(inData, shape, type, stream));
        Tensor v(std::make_unique<CPUTensor>(valData, shape, type, stream));
        Tensor idx(std::make_unique<CPUTensor>(idxData, shape, dtype::s32, stream));
        native::sortWithIndicesImpl(v, idx, in, axis, sortMode);
      });
  out = indices;
}

} // namespace fl
