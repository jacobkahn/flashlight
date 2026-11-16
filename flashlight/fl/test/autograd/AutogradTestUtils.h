/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl {
namespace detail {

class AutogradTestF16 : public ::testing::Test {};

using JacobianFunc = std::function<Tensor(Tensor&)>;

// Flat host copy of a tensor's values, widened to double.
inline std::vector<double> toHostDoubles(const Tensor& tensor) {
  if (tensor.type() == fl::dtype::f64) {
    return tensor.toHostVectorSync<double>();
  }
  auto values = tensor.astype(fl::dtype::f32).toHostVectorSync<float>();
  return std::vector<double>(values.begin(), values.end());
}

// A tensor of `shape` and `type` holding `values` (row-major). Built from the
// host in one step, so each perturbation costs a single transfer rather than
// several tiny device ops.
inline Tensor fromHostDoubles(
    const Shape& shape,
    const std::vector<double>& values,
    fl::dtype type) {
  if (type == fl::dtype::f64) {
    return Tensor::fromVector<double>(shape, values);
  }
  std::vector<float> narrowed(values.begin(), values.end());
  auto tensor = Tensor::fromVector<float>(shape, narrowed);
  return type == fl::dtype::f32 ? tensor : tensor.astype(type);
}

// Compares the Jacobian of `func` at `input` from central finite differences
// against the one from backward(). Both Jacobians are assembled on the host
// from fresh lazy graphs per column/row: the input is never mutated (in-place
// writes would grow IndexedUpdate chains), and no device matrix is updated
// one column at a time.
inline bool jacobianTestImpl(
    const JacobianFunc& func,
    Tensor& input,
    float precision = 1E-5,
    float perturbation = 1E-4) {
  const auto firstResult = func(input);
  const auto outShape = firstResult.shape();
  const auto outType = firstResult.type();
  const auto nOut = static_cast<Dim>(firstResult.elements());
  const auto nIn = static_cast<Dim>(input.elements());

  // fwdJacobian[o * nIn + i] = d out[o] / d in[i], by central differences on
  // non-grad perturbed copies of the input.
  std::vector<double> fwdJacobian(nOut * nIn);
  auto values = toHostDoubles(input);
  for (Dim i = 0; i < nIn; ++i) {
    const double orig = values[i];
    values[i] = orig - perturbation;
    Tensor minus(fromHostDoubles(input.shape(), values, input.type()), false);
    values[i] = orig + perturbation;
    Tensor plus(fromHostDoubles(input.shape(), values, input.type()), false);
    values[i] = orig;
    auto diff = toHostDoubles((func(plus) - func(minus)) * 0.5 / perturbation);
    for (Dim o = 0; o < nOut; ++o) {
      fwdJacobian[o * nIn + i] = diff[o];
    }
  }

  // bwdJacobian row o = gradient of out[o] with respect to the input.
  std::vector<double> bwdJacobian(nOut * nIn);
  std::vector<double> oneHot(nOut, 0.);
  for (Dim o = 0; o < nOut; ++o) {
    oneHot[o] = 1.;
    auto cotangent = fromHostDoubles(outShape, oneHot, outType);
    oneHot[o] = 0.;
    input.zeroGrad();
    auto out = func(input);
    out.backward(cotangent);
    auto grad = toHostDoubles(input.grad());
    std::copy(grad.begin(), grad.end(), bwdJacobian.begin() + o * nIn);
  }

  for (size_t k = 0; k < fwdJacobian.size(); ++k) {
    if (!(std::abs(fwdJacobian[k] - bwdJacobian[k]) < precision)) {
      return false;
    }
  }
  return true;
}

} // namespace detail
} // namespace fl
