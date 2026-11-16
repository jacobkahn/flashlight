/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/View.h"

#include <format>
#include <utility>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace fl {

View::View(Shape dims) : dims_(std::move(dims)) {}

Tensor View::forward(const Tensor& input) {
  Shape dims = dims_;
  return autograd::reshape(input, dims);
}

std::unique_ptr<Module> View::clone() const {
  return std::make_unique<View>(*this);
}

std::string View::prettyString() const {
  return std::format("View ({})", dims_.toString());
}

} // namespace fl
