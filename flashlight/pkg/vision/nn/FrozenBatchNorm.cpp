/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/nn/FrozenBatchNorm.h"

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"

namespace fl {

FrozenBatchNorm::FrozenBatchNorm(
    int featAxis,
    int featSize,
    double momentum /* = 0.1 */,
    double eps /*  = 1e-5*/,
    bool affine /*  = true*/,
    bool trackStats /*  = true*/)
    : FrozenBatchNorm(
          std::vector<int>(1, featAxis),
          featSize,
          momentum,
          eps,
          affine,
          trackStats) {}

FrozenBatchNorm::FrozenBatchNorm(
    const std::vector<int>& featAxis,
    int featSize,
    double momentum /* = 0.1*/,
    double eps /* = 1e-5 */,
    bool affine /* = true*/,
    bool trackStats /* = true*/)
    : BatchNorm(featAxis, featSize, momentum, eps, affine, trackStats) {
  BatchNorm::initialize();
}

std::unique_ptr<Module> FrozenBatchNorm::clone() const {
  return std::make_unique<FrozenBatchNorm>(*this);
}

Tensor FrozenBatchNorm::forward(const Tensor& input) {
  auto scale = params_[0] / autograd::sqrt(runningVar_ + epsilon_);
  auto bias = params_[1] - runningMean_ * scale;
  bias = autograd::reshape(bias, {1, 1, bias.dim(0), 1}).astype(input.type());
  scale = autograd::reshape(scale, {1, 1, scale.dim(0), 1}).astype(input.type());
  return (input * autograd::tileAs(scale, input)) + autograd::tileAs(bias, input);
}

void FrozenBatchNorm::setRunningMean(const fl::Tensor& x) {
  runningMean_ = x;
}

void FrozenBatchNorm::setRunningVar(const fl::Tensor& x) {
  runningVar_ = x;
}

void FrozenBatchNorm::train() {
  for (auto& param : params_) {
    param.setRequiresGrad(false);
  }
  runningVar_.setRequiresGrad(false);
  runningMean_.setRequiresGrad(false);
  train_ = false;
}

std::string FrozenBatchNorm::prettyString() const {
  std::ostringstream ss;
  ss << "FrozenBatchNorm";
  ss << " ( axis : { ";
  for (auto x : featAxis_) {
    ss << x << " ";
  }
  ss << "}, size : " << featSize_ << " )";
  return ss.str();
}

} // namespace fl
