/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Dropout.h"

#include <format>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"

namespace fl {

Dropout::Dropout(double drop_ratio) : ratio_(drop_ratio) {}

Tensor Dropout::forward(const Tensor& input) {
  if (train_) {
    return autograd::dropout(input, ratio_);
  } else {
    return input;
  }
}

std::unique_ptr<Module> Dropout::clone() const {
  return std::make_unique<Dropout>(*this);
}

std::string Dropout::prettyString() const {
  return std::format("Dropout ({})", ratio_);
}

} // namespace fl
