/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"

#include <algorithm>
#include <stdexcept>

#include <af/arith.h>
#include <af/gfor.h>

#include "flashlight/fl/tensor/OpEnums.h"
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#include "flashlight/fl/tensor/backend/af/Utils.h"

namespace fl {
namespace {

using reduceFunc_t = af::array (*)(const af::array&, const int);

template <typename T = reduceFunc_t>
af::array afReduceAxes(
    const af::array& input,
    const std::vector<int>& axes,
    T func,
    const bool keepDims = false) {
  auto arr = input;
  for (int dim : axes) {
    arr = func(arr, dim);
  }
  return fl::detail::condenseIndices(arr, keepDims);
}

unsigned
getReducedNumDims(unsigned inSize, unsigned axisSize, const bool keepDims) {
  if (keepDims) {
    return inSize;
  } else {
    if (inSize < axisSize) {
      return 0;
    } else {
      return inSize - axisSize;
    }
  }
}

bool isAllAxisReduction(const Tensor& input, const std::vector<int>& axes) {
  if (input.ndim() == 0 || axes.empty()) {
    return true;
  }
  if (input.ndim() != axes.size()) {
    return false;
  }
  // Check that all dims are present
  auto _axes = axes;
  std::sort(_axes.begin(), _axes.end());
  for (size_t i = 0; i < _axes.size(); ++i) {
    if (_axes[i] != i) {
      return false;
    }
  }
  return true;
}
} // namespace

Tensor ArrayFireBackend::amin(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  if (isAllAxisReduction(input, axes)) {
    // Reduce along all axes returning a singleton tensor
    // TODO: modify this to af::min<af::array> to take advantage of the
    // ArrayFire reduce_all kernels once available
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(
            af::min(af::min(af::min(af::min(toArray(input)))))),
        /* numDims = */ 0);
  } else {
    auto afAxes = detail::flToAfAxes(axes, input.ndim());
    return toTensor<ArrayFireTensor>(
        afReduceAxes(toArray(input), afAxes, af::min, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

Tensor ArrayFireBackend::amax(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  if (isAllAxisReduction(input, axes)) {
    // Reduce along all axes returning a singleton tensor
    // TODO: modify this to af::max<af::array> to take advantage of the
    // ArrayFire reduce_all kernels once available
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(
            af::max(af::max(af::max(af::max(toArray(input)))))),
        /* numDims = */ 0);
  } else {
    auto afAxes = detail::flToAfAxes(axes, input.ndim());
    return toTensor<ArrayFireTensor>(
        afReduceAxes(toArray(input), afAxes, af::max, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

void ArrayFireBackend::min(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  unsigned afAxis = detail::flToAfAxis(axis, input.ndim());
  af::min(toArray(values), toArray(indices), toArray(input), afAxis);
  values = toTensor<ArrayFireTensor>(
      detail::condenseIndices(toArray(values), keepDims),
      getReducedNumDims(input.ndim(), 1, keepDims));
  indices = toTensor<ArrayFireTensor>(
      detail::condenseIndices(toArray(indices), keepDims),
      getReducedNumDims(input.ndim(), 1, keepDims));
}

void ArrayFireBackend::max(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  unsigned afAxis = detail::flToAfAxis(axis, input.ndim());
  af::max(toArray(values), toArray(indices), toArray(input), afAxis);
  values = toTensor<ArrayFireTensor>(
      detail::condenseIndices(toArray(values), keepDims),
      getReducedNumDims(input.ndim(), 1, keepDims));
  indices = toTensor<ArrayFireTensor>(
      detail::condenseIndices(toArray(indices), keepDims),
      getReducedNumDims(input.ndim(), 1, keepDims));
}

Tensor ArrayFireBackend::sum(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  if (isAllAxisReduction(input, axes)) {
    // Reduce along all axes returning a singleton tensor
    // TODO: modify this to af::sum<af::array> to take advantage of the
    // ArrayFire reduce_all kernels once available
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(
            af::sum(af::sum(af::sum(af::sum(toArray(input)))))),
        /* numDims = */ 0);
  } else {
    auto afAxes = detail::flToAfAxes(axes, input.ndim());
    return toTensor<ArrayFireTensor>(
        afReduceAxes(toArray(input), afAxes, af::sum, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

void ArrayFireBackend::cumsum(Tensor& out, const Tensor& input, const unsigned axis,
    std::shared_ptr<const Stream> /* stream */) {
  unsigned afAxis = detail::flToAfAxis(axis, input.ndim());
  out = toTensor<ArrayFireTensor>(
      af::accum(toArray(input), afAxis), /* numDims = */ input.ndim());
}

void ArrayFireBackend::argmax(
    Tensor& out,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  unsigned afAxis = detail::flToAfAxis(axis, input.ndim());
  af::array tmpVal, indices;
  af::max(tmpVal, indices, toArray(input), afAxis);
  out = toTensor<ArrayFireTensor>(
      detail::condenseIndices(indices, keepDims),
      getReducedNumDims(input.ndim(), 1, keepDims));
}

void ArrayFireBackend::argmin(
    Tensor& out,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  unsigned afAxis = detail::flToAfAxis(axis, input.ndim());
  af::array tmpVal, indices;
  af::min(tmpVal, indices, toArray(input), afAxis);
  out = toTensor<ArrayFireTensor>(
      detail::condenseIndices(indices, keepDims),
      getReducedNumDims(input.ndim(), 1, keepDims));
}

Tensor ArrayFireBackend::mean(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  if (isAllAxisReduction(input, axes)) {
    // Reduce along all axes returning a singleton tensor
    // TODO: modify this to af::mean<af::array> to take advantage of the
    // ArrayFire reduce_all kernels once available
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(
            af::mean(af::mean(af::mean(af::mean(toArray(input)))))),
        /* numDims = */ 0);
  } else {
    auto afAxes = detail::flToAfAxes(axes, input.ndim());
    return toTensor<ArrayFireTensor>(
        afReduceAxes<af::array(const af::array&, const dim_t)>(
            toArray(input), afAxes, af::mean, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

Tensor ArrayFireBackend::median(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  if (isAllAxisReduction(input, axes)) {
    // Stay on device: sort the flattened array and pick the middle element(s)
    auto flat = af::flat(toArray(input));
    // Cast integer types to f32 since median may be fractional (e.g., median of {0,1} = 0.5)
    if (flat.type() != f32 && flat.type() != f64) {
      flat = flat.as(f32);
    }
    auto sorted = af::sort(flat);
    dim_t n = flat.elements();
    af::array result;
    if (n % 2 == 1) {
      result = sorted(af::seq(n / 2, n / 2));
    } else {
      result = (sorted(af::seq(n / 2 - 1, n / 2 - 1)) +
                sorted(af::seq(n / 2, n / 2))) /
          2.0;
    }
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(result),
        /* numDims = */ 0);
  } else {
    auto afAxes = detail::flToAfAxes(axes, input.ndim());
    return toTensor<ArrayFireTensor>(
        afReduceAxes<af::array(const af::array&, const dim_t)>(
            toArray(input), afAxes, af::median, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

void ArrayFireBackend::var(
    Tensor& out,
    const Tensor& input,
    const std::vector<int>& axes,
    const bool bias,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  af_var_bias biasMode = bias ? AF_VARIANCE_SAMPLE : AF_VARIANCE_POPULATION;
  // Use ArrayFire default for one dimension which may be optimized
  auto& arr = toArray(input);
  auto afAxes = detail::flToAfAxes(axes, input.ndim());
  if (isAllAxisReduction(input, axes)) {
    // Stay on device: compute variance via mean subtraction
    auto flat = af::flat(arr);
    auto m = af::mean(flat);
    auto diff = flat - af::tile(m, static_cast<unsigned>(flat.elements()));
    auto sq = af::pow(diff, 2);
    auto s = af::sum(sq);
    dim_t n = flat.elements();
    if (bias) {
      n -= 1;
    }
    auto result = s / static_cast<double>(n);
    out = toTensor<ArrayFireTensor>(
        detail::condenseIndices(result), /* numDims = */ 0);
  } else if (axes.size() == 1) {
    out = toTensor<ArrayFireTensor>(
        detail::condenseIndices(af::var(arr, biasMode, afAxes[0]), keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  } else {
    auto meanArr = mean(input, axes, /* keepDims = */ true);
    auto x = af::batchFunc(arr, toArray(meanArr), af::operator-);

    x = af::pow(x, 2);
    x = afReduceAxes(x, afAxes, af::sum, /* keepDims = */ true);

    int denominator = 1;
    auto dims = arr.dims();
    for (auto dim : afAxes) {
      denominator *= dims[dim];
    }
    if (bias) {
      denominator--;
    }

    x = x / denominator;
    out = toTensor<ArrayFireTensor>(
        detail::condenseIndices(x, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

Tensor ArrayFireBackend::std(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  const bool bias = false; // TODO: make this configurable
  af_var_bias biasMode = bias ? AF_VARIANCE_SAMPLE : AF_VARIANCE_POPULATION;
  if (isAllAxisReduction(input, axes)) {
    // Stay on device: compute as sqrt(var(...))
    Tensor varResult;
    this->var(varResult, input, axes, /* bias = */ bias, keepDims);
    return this->sqrt(varResult);
  } else if (axes.size() == 1) {
    unsigned afAxis = detail::flToAfAxis(axes[0], input.ndim());
    // Use arrayfire default for one dimension which may be optimized
    // TODO: update this? stddev is deprecated.
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(
            af::stdev(toArray(input), biasMode, afAxis), keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
  Tensor varResult;
  this->var(varResult, input, axes, /* bias = */ bias, keepDims);
  return this->sqrt(varResult);
}

void ArrayFireBackend::norm(
    Tensor& out,
    const Tensor& input,
    const std::vector<int>& axes,
    double p /* = 2 */,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  if (isAllAxisReduction(input, axes)) {
    // TODO: update to af::norm<af::array> if device-side specialization is
    // available. Either that or use the all-axis specializations with the below
    // implementation
    auto result = af::pow(af::abs(af::flat(toArray(input))), p);
    // Replace with af::sum<af::array>
    result = af::sum(af::sum(af::sum(result)));
    result = af::pow(result, 1 / p);
    out = toTensor<ArrayFireTensor>(
        detail::condenseIndices(result), /* numDims = */ 0);
  } else {
    auto afAxes = detail::flToAfAxes(axes, input.ndim());
    auto result = af::pow(af::abs(toArray(input)), p);
    result = afReduceAxes(result, afAxes, af::sum, keepDims);
    result = af::pow(result, 1 / p);
    out = toTensor<ArrayFireTensor>(
        std::move(result),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

Tensor ArrayFireBackend::countNonzero(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  auto& arr = toArray(input);
  unsigned numDims;
  af::array out;
  if (isAllAxisReduction(input, axes)) {
    out = detail::condenseIndices(
        af::sum(af::sum(af::sum(af::count(arr)))), keepDims);
    numDims = 0;
  } else {
    auto afAxes = detail::flToAfAxes(axes, input.ndim());
    if (axes.size() == 1) {
      out = af::count(arr, afAxes.front());
      numDims = getReducedNumDims(input.ndim(), axes.size(), keepDims);
    } else {
      out = afReduceAxes(
          af::count(arr, afAxes.front()),
          std::vector<int>(afAxes.begin() + 1, afAxes.end()),
          af::sum,
          keepDims);
      numDims = getReducedNumDims(input.ndim(), axes.size(), keepDims);
    }
  }
  return toTensor<ArrayFireTensor>(
      detail::condenseIndices(out, keepDims), numDims);
}

Tensor ArrayFireBackend::any(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  if (isAllAxisReduction(input, axes)) {
    // Reduce along all axes returning a singleton tensor
    // TODO: modify this to af::anyTrue<af::array> to take advantage of the
    // ArrayFire reduce_all kernels once available
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(
            af::anyTrue(af::anyTrue(af::anyTrue(af::anyTrue(toArray(input)))))),
        /* numDims = */ 0);
  } else {
    auto afAxes = detail::flToAfAxes(axes, input.ndim());
    return toTensor<ArrayFireTensor>(
        afReduceAxes(toArray(input), afAxes, af::anyTrue, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

Tensor ArrayFireBackend::all(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims,
    std::shared_ptr<const Stream> /* stream */) {
  if (isAllAxisReduction(input, axes)) {
    // Reduce along all axes returning a singleton tensor
    // TODO: modify this to af::allTrue<af::array> to take advantage of the
    // ArrayFire reduce_all kernels once available
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(
            af::allTrue(af::allTrue(af::allTrue(af::allTrue(toArray(input)))))),
        /* numDims = */ 0);
  } else {
    auto afAxes = detail::flToAfAxes(axes, input.ndim());
    return toTensor<ArrayFireTensor>(
        afReduceAxes(toArray(input), afAxes, af::allTrue, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

void ArrayFireBackend::reduction(
    Tensor& out,
    ReductionOp op,
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims,
    std::shared_ptr<const Stream> stream) {
  switch (op) {
    case ReductionOp::Amin: out = amin(input, axes, keepDims, std::move(stream)); return;
    case ReductionOp::Amax: out = amax(input, axes, keepDims, std::move(stream)); return;
    case ReductionOp::Sum: out = sum(input, axes, keepDims, std::move(stream)); return;
    case ReductionOp::Mean: out = mean(input, axes, keepDims, std::move(stream)); return;
    case ReductionOp::Median: out = median(input, axes, keepDims, std::move(stream)); return;
    case ReductionOp::Std: out = std(input, axes, keepDims, std::move(stream)); return;
    case ReductionOp::CountNonzero: out = countNonzero(input, axes, keepDims, std::move(stream)); return;
    case ReductionOp::Any: out = any(input, axes, keepDims, std::move(stream)); return;
    case ReductionOp::All: out = all(input, axes, keepDims, std::move(stream)); return;
  }
  throw std::runtime_error(
      "[ArrayFireBackend::reduction] Unknown reduction operation type");
}
} // namespace fl
