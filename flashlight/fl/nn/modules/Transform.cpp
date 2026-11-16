/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Transform.h"

#include <format>

#include "flashlight/fl/tensor/Tensor.h"

namespace fl {

Transform::Transform(
    const std::function<Tensor(const Tensor&)>& func,
    const std::string& name /* = "" */)
    : func_(func), name_(name) {}

Tensor Transform::forward(const Tensor& input) {
  return func_(input);
}

std::unique_ptr<Module> Transform::clone() const {
  return std::make_unique<Transform>(*this);
}

std::string Transform::prettyString() const {
  return std::format("Transform ('{}')", name_);
}

} // namespace fl
