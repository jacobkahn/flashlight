/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/cpu/CPUBackend.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <vector>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/backend/cpu/CPUBackendUtils.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/ops/native/CpuTensorHelpers.h"
#include "flashlight/fl/tensor/ops/native/Reduction.h"
#include "flashlight/fl/tensor/ops/onednn/OneDnnOps.h"

namespace fl {

using detail::enqueueNativeOp;

/************************** Reductions ***************************/

Tensor CPUBackend::amin(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  Tensor out;
  onednnOps_->applyReductionOp(
      out, input, dnnl::algorithm::reduction_min, axes, keepDims, nativeStream(), asyncStream_);
  return out;
}

Tensor CPUBackend::amax(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  Tensor out;
  onednnOps_->applyReductionOp(
      out, input, dnnl::algorithm::reduction_max, axes, keepDims, nativeStream(), asyncStream_);
  return out;
}

void CPUBackend::min(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  auto inShape = input.shape();
  auto type = input.type();
  std::vector<Dim> outDims = inShape.get();
  if (keepDims) {
    outDims[axis] = 1;
  } else {
    outDims.erase(outDims.begin() + axis);
  }
  Shape outShape(outDims);
  values = toTensor<CPUTensor>(outShape, type, nullptr, Location::Host);
  indices = toTensor<CPUTensor>(outShape, dtype::u32, nullptr, Location::Host);
  auto inData = detail::contiguousSharedData(input);
  auto valData = toCPUTensor(values).getSharedData();
  auto idxData = toCPUTensor(indices).getSharedData();
  asyncStream_->relativeSync(input.stream());
  asyncStream_->enqueue(
      [inData, valData, idxData, inShape, outShape, type,
       stream = asyncStream_, axis]() {
        Tensor in(std::make_unique<CPUTensor>(inData, inShape, type, stream));
        Tensor v(std::make_unique<CPUTensor>(valData, outShape, type, stream));
        Tensor idx(std::make_unique<CPUTensor>(idxData, outShape, dtype::u32, stream));
        native::minWithIndexImpl(v, idx, in, inShape, axis);
      });
}

void CPUBackend::max(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  auto inShape = input.shape();
  auto type = input.type();
  std::vector<Dim> outDims = inShape.get();
  if (keepDims) {
    outDims[axis] = 1;
  } else {
    outDims.erase(outDims.begin() + axis);
  }
  Shape outShape(outDims);
  values = toTensor<CPUTensor>(outShape, type, nullptr, Location::Host);
  indices = toTensor<CPUTensor>(outShape, dtype::u32, nullptr, Location::Host);
  auto inData = detail::contiguousSharedData(input);
  auto valData = toCPUTensor(values).getSharedData();
  auto idxData = toCPUTensor(indices).getSharedData();
  asyncStream_->relativeSync(input.stream());
  asyncStream_->enqueue(
      [inData, valData, idxData, inShape, outShape, type,
       stream = asyncStream_, axis]() {
        Tensor in(std::make_unique<CPUTensor>(inData, inShape, type, stream));
        Tensor v(std::make_unique<CPUTensor>(valData, outShape, type, stream));
        Tensor idx(std::make_unique<CPUTensor>(idxData, outShape, dtype::u32, stream));
        native::maxWithIndexImpl(v, idx, in, inShape, axis);
      });
}

Tensor CPUBackend::sum(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  // b8 (stored as s8) can overflow with sum; promote to s32, return as u32
  // (matching ArrayFire convention where sum of bools returns unsigned)
  if (input.type() == dtype::b8) {
    Tensor out;
    onednnOps_->applyReductionOp(
        out, input.astype(dtype::s32), dnnl::algorithm::reduction_sum, axes, keepDims, nativeStream(), asyncStream_);
    return out.astype(dtype::u32);
  }
  Tensor out;
  onednnOps_->applyReductionOp(
      out, input, dnnl::algorithm::reduction_sum, axes, keepDims, nativeStream(), asyncStream_);
  return out;
}

void CPUBackend::cumsum(Tensor& out, const Tensor& input, const unsigned axis, std::shared_ptr<const Stream> /* stream */) {
  out = enqueueNativeOp(input, asyncStream_,
      [axis](Tensor& out, const Tensor& in) {
        native::cumsumImpl(out, in, axis);
      });
}

void CPUBackend::argmax(
    Tensor& out,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  Tensor values, indices;
  max(values, indices, input, axis, keepDims);
  out = indices;
}

void CPUBackend::argmin(
    Tensor& out,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  Tensor values, indices;
  min(values, indices, input, axis, keepDims);
  out = indices;
}

Tensor CPUBackend::mean(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  // Mean should always produce floating-point output
  Tensor out;
  if (input.type() != dtype::f32 && input.type() != dtype::f64 &&
      input.type() != dtype::f16) {
    onednnOps_->applyReductionOp(
        out, input.astype(dtype::f32), dnnl::algorithm::reduction_mean, axes, keepDims, nativeStream(), asyncStream_);
  } else {
    onednnOps_->applyReductionOp(
        out, input, dnnl::algorithm::reduction_mean, axes, keepDims, nativeStream(), asyncStream_);
  }
  return out;
}

Tensor CPUBackend::median(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  // Scalar or single-element: median is the value itself (as f32)
  if (input.elements() <= 1) {
    if (input.type() == dtype::f32 || input.type() == dtype::f64) {
      return input.copy();
    }
    return input.astype(dtype::f32);
  }
  // Compose: sort along axes, pick middle element
  // For simplicity, flatten the reduction axes, sort, pick middle
  if (axes.empty()) {
    // Full reduction — always return f32 (median may not be integer)
    return fl::native::dispatchByType(input.type(), [&](auto typeTag) {
      using T = decltype(typeTag);
      auto data = input.toHostVectorSync<T>();
      std::sort(data.begin(), data.end());
      size_t n = data.size();
      float med;
      if (n % 2 == 0) {
        med = static_cast<float>(
            (static_cast<double>(data[n / 2 - 1]) +
             static_cast<double>(data[n / 2])) / 2.0);
      } else {
        med = static_cast<float>(data[n / 2]);
      }
      return native::makeCpuTensor(Shape({}), &med);
    });
  }
  // Single axis median
  if (axes.size() == 1) {
    int axis = axes[0];
    Tensor sorted;
    this->sort(sorted, input, axis, SortMode::Ascending);
    Dim axisDim = input.shape()[axis];
    Dim mid = axisDim / 2;
    std::vector<Index> idx(input.ndim(), fl::span);
    if (axisDim % 2 == 1) {
      idx[axis] = fl::range(static_cast<int>(mid), static_cast<int>(mid + 1));
      auto result = sorted(idx);
      if (!keepDims) {
        return result.flatten();
      }
      return result;
    } else {
      idx[axis] = fl::range(static_cast<int>(mid - 1), static_cast<int>(mid + 1));
      auto slice = sorted(idx);
      return mean(slice, {axis}, keepDims);
    }
  }
  // Multi-axis median: flatten reduction axes, then do single-axis median
  // Reorder dims: non-reduction axes first, then reduction axes
  std::set<int> reduceSet(axes.begin(), axes.end());
  std::vector<int> perm;
  for (int d = 0; d < input.ndim(); ++d) {
    if (reduceSet.find(d) == reduceSet.end()) perm.push_back(d);
  }
  int firstReducePos = perm.size();
  for (int ax : axes) perm.push_back(ax);

  auto reordered = fl::transpose(input, Shape(std::vector<Dim>(perm.begin(), perm.end())));
  // Flatten the reduction axes into one
  auto reordShape = reordered.shape().get();
  Dim reducedDim = 1;
  for (size_t i = firstReducePos; i < reordShape.size(); ++i) {
    reducedDim *= reordShape[i];
  }
  std::vector<Dim> flatShape(reordShape.begin(), reordShape.begin() + firstReducePos);
  flatShape.push_back(reducedDim);
  auto flattened = fl::reshape(reordered, Shape(flatShape));
  auto result = median(flattened, {firstReducePos}, keepDims);
  if (keepDims) {
    // Restore the reduced axes as size-1 dimensions
    auto outDims = input.shape().get();
    for (int ax : axes) outDims[ax] = 1;
    result = fl::reshape(result, Shape(outDims));
  }
  return result;
}

void CPUBackend::var(
    Tensor& out,
    const Tensor& input,
    const std::vector<int>& axes,
    const bool bias,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  // Scalar or single-element: variance is 0. Results are bare CPU tensors
  // (`this->full`): this runs inside evaluation, where fl::full would create
  // a JIT tensor.
  if (input.elements() <= 1) {
    auto outType = (input.type() == dtype::f64) ? dtype::f64 : dtype::f32;
    std::vector<int> reduced = axes;
    if (reduced.empty()) {
      reduced.resize(input.ndim());
      std::iota(reduced.begin(), reduced.end(), 0);
    }
    std::vector<Dim> outDims;
    for (int d = 0; d < static_cast<int>(input.ndim()); ++d) {
      const bool isReduced =
          std::find(reduced.begin(), reduced.end(), d) != reduced.end();
      if (!isReduced) {
        outDims.push_back(input.dim(d));
      } else if (keepDims) {
        outDims.push_back(1);
      }
    }
    out = full(Shape(outDims), 0., outType);
    return;
  }
  // bias=true --> sample/unbiased variance: sum((x-mean)^2) / (N-1)
  // bias=false --> population/biased variance: mean((x-mean)^2)
  auto m = mean(input, axes, /*keepDims=*/true);
  auto diff = input - m;
  auto sq = diff * diff;
  if (!bias) {
    // Population variance: divide by N
    out = mean(sq, axes, keepDims);
  } else {
    // Sample variance: divide by N-1
    auto s = sum(sq, axes, keepDims);
    // Count reduction elements
    Dim count = 1;
    std::vector<int> axesToReduce = axes;
    if (axesToReduce.empty()) {
      axesToReduce.resize(input.ndim());
      std::iota(axesToReduce.begin(), axesToReduce.end(), 0);
    }
    for (int ax : axesToReduce) {
      count *= input.shape()[ax];
    }
    if (count <= 1) {
      // Sample variance of 0 or 1 element is 0
      out = full(s.shape(), 0., s.type());
      return;
    }
    out = s / static_cast<double>(count - 1);
  }
}

Tensor CPUBackend::std(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  Tensor v;
  var(v, input, axes, /*bias=*/false, keepDims);
  return fl::sqrt(v);
}

void CPUBackend::norm(
    Tensor& out,
    const Tensor& input,
    const std::vector<int>& axes,
    double p,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  // norm requires floating-point input
  if (input.type() != dtype::f32 && input.type() != dtype::f64 &&
      input.type() != dtype::f16) {
    norm(out, input.astype(dtype::f32), axes, p, keepDims);
    return;
  }
  if (p == 2.0) {
    // L2 norm: sum(|x|^2)^(1/2)
    onednnOps_->applyReductionOp(
        out, input, dnnl::algorithm::reduction_norm_lp_sum, axes, keepDims, nativeStream(), asyncStream_,
        /*p=*/2.0f, /*eps=*/0.f);
  } else if (p == 1.0) {
    // L1 norm
    out = sum(fl::abs(input), axes, keepDims);
  } else {
    // Lp norm = sum(|x|^p)^(1/p)
    auto absTensor = fl::abs(input);
    auto powered = fl::power(absTensor, p);
    auto s = sum(powered, axes, keepDims);
    out = fl::power(s, 1.0 / p);
  }
}

Tensor CPUBackend::countNonzero(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  std::vector<Dim> dims(input.ndim(), 1);
  auto zero = this->full(Shape(dims), 0, input.type());
  // can't use s32 for sum in OneDNN
  Tensor res;
  onednnOps_->applyBinop(
      res, input, zero, dnnl::algorithm::binary_ne, nativeStream(), asyncStream_,
      dnnl::memory::data_type::f32);
  return sum(res, axes, keepDims).astype(fl::dtype::u32);
}

Tensor CPUBackend::any(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  return countNonzero(input, axes, keepDims) != 0;
}

Tensor CPUBackend::all(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  // Scalar: all is just nonzero check
  if (input.elements() <= 1) {
    return input != 0;
  }
  // all = (countNonzero == totalElements along axes)
  auto nz = countNonzero(input, axes, keepDims);
  // Compute expected count along reduction axes
  std::vector<int> axesToReduce = axes;
  if (axesToReduce.empty()) {
    axesToReduce.resize(input.ndim());
    std::iota(axesToReduce.begin(), axesToReduce.end(), 0);
  }
  Dim count = 1;
  for (int ax : axesToReduce) {
    count *= input.shape()[ax];
  }
  return nz == static_cast<int>(count);
}

} // namespace fl
