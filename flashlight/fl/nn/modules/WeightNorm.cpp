/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <format>
#include <stdexcept>

#include "flashlight/fl/nn/modules/Conv2D.h"
#include "flashlight/fl/nn/modules/Linear.h"
#include "flashlight/fl/nn/modules/WeightNorm.h"

namespace fl {

WeightNorm::WeightNorm(const WeightNorm& other)
    : module_(other.module_->clone()),
      dim_(other.dim_),
      normDim_(other.normDim_) {
  initParams();
}

WeightNorm& WeightNorm::operator=(const WeightNorm& other) {
  module_ = other.clone();
  dim_ = other.dim_;
  normDim_ = other.normDim_;
  initParams();
  return *this;
}

void WeightNorm::transformDims() {
  normDim_.clear();
  int vNumdims = module_->param(0).ndim();
  if (dim_ < 0 || dim_ > vNumdims) {
    throw std::invalid_argument("invalid dimension for WeightNorm");
  }
  for (int i = 0; i < vNumdims; i++) {
    if (i != dim_) {
      normDim_.push_back(i);
    }
  }
}

void WeightNorm::computeWeight() {
  auto v = params_[0];
  auto g = params_[1];
  Tensor nm;
  // speed of norm operation is the best while doing it across {1} dim
  // tested for convlm model training
  if (dim_ == 0) {
    nm = autograd::reshape(v, {0, -1});
    nm = autograd::norm(nm, {1}, /* p = */ 2, /* keepDims = */ true);
  } else if (dim_ == 3) {
    // TODO{fl::Tensor}{enforce 4D parameters from child module?}
    nm = autograd::reshape(v, {-1, 1, 1, 0});
    nm = autograd::transpose(nm, {3, 0, 1, 2});
    nm = autograd::norm(nm, {1}, /* p = */ 2, /* keepDims = */ true);
    nm = autograd::transpose(nm, {1, 2, 3, 0});
  } else {
    throw std::invalid_argument(
        std::format("Wrong dimension for Weight Norm: {}", dim_));
  }
  auto wt = v * autograd::tileAs(g / nm, v);
  module_->setParams(wt, 0);
}

void WeightNorm::initParams() {
  auto moduleParams = module_->params();
  auto& v = moduleParams.at(0);
  Tensor g(
      autograd::norm(v, normDim_, /* p = */ 2, /* keepDims = */ true), true);
  if (moduleParams.size() == 2) {
    auto& b = moduleParams[1];
    params_ = {v, g, b};
  } else if (moduleParams.size() == 1) {
    params_ = {v, g};
  } else {
    throw std::invalid_argument("WeightNorm only supports Linear and Conv2D");
  }
}

void WeightNorm::setParams(const Tensor& var, int position) {
  Module::setParams(var, position);
  // it is necessary to copy all params to the parent module
  // due to copies stored in the parent module (not pointers)
  if (position == 2) {
    module_->setParams(var, 1);
  } else if (position <= 1) {
    computeWeight();
  }
}

std::vector<Tensor> WeightNorm::forward(const std::vector<Tensor>& inputs) {
  if (train_) {
    computeWeight();
  }
  return module_->forward(inputs);
}

ModulePtr WeightNorm::module() const {
  return module_;
}

void WeightNorm::train() {
  Module::train();
  module_->train();
}

void WeightNorm::eval() {
  Module::eval();
  module_->eval();
  computeWeight();
}

std::unique_ptr<Module> WeightNorm::clone() const {
  return std::make_unique<WeightNorm>(*this);
}

std::string WeightNorm::prettyString() const {
  return std::format("WeightNorm ({}, {})", module_->prettyString(), dim_);
}

} // namespace fl
