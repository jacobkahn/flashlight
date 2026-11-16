/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Reorder.h"

#include <format>
#include <stdexcept>
#include <utility>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"

namespace fl {

Reorder::Reorder(Shape shape) : shape_(std::move(shape)) {}

Tensor Reorder::forward(const Tensor& input) {
  if (input.ndim() != shape_.ndim()) {
    throw std::invalid_argument(
        "Reorder::forward - input tensor has different "
        "number of dimensions than reorder shape.");
  }
  return autograd::transpose(input, shape_);
}

std::unique_ptr<Module> Reorder::clone() const {
  return std::make_unique<Reorder>(*this);
}

std::string Reorder::prettyString() const {
  return std::format("Reorder ({})", shape_.toString());
}

} // namespace fl
