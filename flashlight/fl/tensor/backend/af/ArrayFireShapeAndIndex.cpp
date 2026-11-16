/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"

#include <af/blas.h>

#include <numeric>

#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#include "flashlight/fl/tensor/backend/af/Utils.h"

namespace fl {

void ArrayFireBackend::reshape(Tensor& out, const Tensor& tensor, const Shape& shape,
    std::shared_ptr<const Stream> /* stream */) {
  out = toTensor<ArrayFireTensor>(
      af::moddims(toArray(tensor), detail::flToAfDims(shape)), shape.ndim());
}

void ArrayFireBackend::transpose(
    Tensor& out,
    const Tensor& tensor,
    const Shape& axes /* = {} */,
    std::shared_ptr<const Stream> /* stream */) {
  if (tensor.ndim() == 1) {
    out = tensor;
  } else if (
      tensor.ndim() == 2 && (axes.ndim() == 0 || axes == Shape({1, 0}))) {
    // fastpath for matrices
    out = toTensor<ArrayFireTensor>(
        af::transpose(toArray(tensor)), tensor.ndim());
  } else if (axes.ndim() == 0) {
    // Default: reverse all dimensions. This is the same in both FL and AF
    // space since reversing is symmetric.
    std::vector<Dim> dims(AF_MAX_DIMS);
    std::iota(std::begin(dims), std::end(dims), 0);
    // Compute the reversed dimensions for as many ndims as are in the input
    for (unsigned i = 0; i < tensor.ndim(); ++i) {
      dims[i] = tensor.ndim() - 1 - i;
    }

    // flip all dimensions
    out = toTensor<ArrayFireTensor>(
        af::reorder(toArray(tensor), dims[0], dims[1], dims[2], dims[3]),
        tensor.ndim());
  } else {
    if (axes.ndim() > AF_MAX_DIMS) {
      throw std::invalid_argument(
          "ArrayFire tensor transpose was given "
          "permutation dims with > 4 axes");
    }
    if (axes.ndim() != tensor.ndim()) {
      throw std::invalid_argument(
          "ArrayFire tensor transpose axes don't match tensor's for "
          "permutation - axes must have the same number of "
          "dimensions as the tensor");
    }
    // Convert FL permutation to AF permutation.
    // FL perm axes[i] means: result FL dim i comes from input FL dim axes[i].
    // AF perm d[k] means: result AF dim k comes from input AF dim d[k].
    // FL dim j = AF dim (ndim-1-j), so:
    //   d[ndim-1-i] = ndim-1-axes[i]
    // i.e. d[k] = ndim-1-axes[ndim-1-k]
    unsigned ndim = tensor.ndim();
    std::vector<dim_t> d(AF_MAX_DIMS);
    std::iota(std::begin(d), std::end(d), 0);
    for (size_t i = 0; i < axes.ndim(); ++i) {
      if (axes[i] > tensor.ndim() - 1) {
        throw std::invalid_argument(
            "ArrayFireBackend::transpose - given dimension is larger "
            "than the number of dimensions in the tensor");
      }
      d[ndim - 1 - i] = ndim - 1 - axes[i];
    }
    out = toTensor<ArrayFireTensor>(
        af::reorder(toArray(tensor), d[0], d[1], d[2], d[3]), tensor.ndim());
  }
}

void ArrayFireBackend::tile(Tensor& out, const Tensor& tensor, const Shape& shape,
    std::shared_ptr<const Stream> /* stream */) {
  int outNdim = std::max(tensor.ndim(), shape.ndim());

  // Build AF tile counts aligned to outNdim resolution.
  // FL dim i -> AF dim (outNdim-1-i).
  af::dim4 afTile(1, 1, 1, 1);
  for (int i = 0; i < shape.ndim(); ++i) {
    afTile[outNdim - 1 - i] = shape.dim(i);
  }

  // Right-align: shorter input gets 1s prepended in FL space. In AF (with full
  // reversal), data stays at LOW AF dims. Since AF dim4 already has trailing
  // 1s, no reshape is needed -- just ensure the tile counts align correctly.
  auto arr = toArray(tensor);

  out = toTensor<ArrayFireTensor>(af::tile(arr, afTile), outNdim);
}

void ArrayFireBackend::concatenate(
    Tensor& out,
    const std::vector<Tensor>& tensors,
    const unsigned axis,
    std::shared_ptr<const Stream> /* stream */) {
  if (tensors.empty()) {
    out = toTensor<ArrayFireTensor>(ArrayFireTensor()); // empty tensor
    return;
  }
  if (tensors.size() == 1) {
    out = tensors.front();
    return;
  }

  // Compute effective ndim before axis conversion (axis may extend rank).
  unsigned numDims = tensors[0].ndim();
  if (axis >= numDims) {
    numDims = axis + 1;
  }
  unsigned afAxis = detail::flToAfAxis(axis, numDims);

  // Helper: get an AF array promoted to the effective ndim. When the effective
  // ndim is larger than the input ndim, reshape so data sits at the correct
  // high AF dims (AF is col-major, FL is row-major).
  auto getArr = [&](const Tensor& t) -> af::array {
    auto arr = toArray(t);
    // When rank extends (numDims > input ndim), the new axis is at
    // AF position afAxis. Move data to high AF dims so the new axis
    // at low AF position has size 1.
    if (static_cast<int>(numDims) > t.ndim()) {
      af::dim4 paddedDims(1, 1, 1, 1);
      auto origDims = arr.dims();
      int offset = static_cast<int>(numDims) - t.ndim();
      for (int i = 0; i < t.ndim(); ++i) {
        paddedDims[i + offset] = origDims[i];
      }
      arr = af::moddims(arr, paddedDims);
    }
    return arr;
  };

  // af::join supports at most 4 arrays at a time; handle larger inputs
  // by iteratively joining in groups of 4.
  af::array result = getArr(tensors[0]);
  size_t i = 1;
  while (i < tensors.size()) {
    size_t remaining = tensors.size() - i;
    if (remaining >= 3) {
      result = af::join(
          afAxis, result, getArr(tensors[i]), getArr(tensors[i + 1]),
          getArr(tensors[i + 2]));
      i += 3;
    } else if (remaining == 2) {
      result = af::join(
          afAxis, result, getArr(tensors[i]), getArr(tensors[i + 1]));
      i += 2;
    } else {
      result = af::join(afAxis, result, getArr(tensors[i]));
      i += 1;
    }
  }

  // All tensors have the same numdims else AF would throw
  out = toTensor<ArrayFireTensor>(std::move(result), numDims);
}

void ArrayFireBackend::nonzero(Tensor& out, const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  out = toTensor<ArrayFireTensor>(
      af::where(toArray(tensor)), /* numDims = */ 1);
}

void ArrayFireBackend::pad(
    Tensor& out,
    const Tensor& input,
    const std::vector<std::pair<int, int>>& padWidths,
    const PadType type,
    std::shared_ptr<const Stream> /* stream */) {
  if (padWidths.size() > AF_MAX_DIMS) {
    throw std::invalid_argument(
        "ArrayFireBackend::pad - given padWidths for more than 4 dimensions");
  }

  // convert ((begin_1, end_1), ..., (begin_k, end_k)) to ((begin_1, ...,
  // begin_k), (end_1, ..., end_k)) for ArrayFire.
  // FL dim i maps to AF dim (ndim-1-i), so reverse the padWidths order.
  unsigned ndim = padWidths.size();
  af::dim4 beginPadding, endPadding;
  for (size_t i = 0; i < padWidths.size(); ++i) {
    auto& [first, second] = padWidths[i];
    beginPadding[ndim - 1 - i] = first;
    endPadding[ndim - 1 - i] = second;
  }

  out = toTensor<ArrayFireTensor>(
      af::pad(
          toArray(input),
          beginPadding,
          endPadding,
          detail::flToAfPadType(type)),
      /* numDims = */ // TODO: check
      std::max(input.ndim(), static_cast<int>(padWidths.size())));
}
} // namespace fl
