/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Padding.h"

#include <format>

#include "flashlight/fl/autograd/Functions.h"

namespace fl {

Padding::Padding(std::vector<std::pair<int, int>> padding, double val)
    : m_pad(std::move(padding)), m_val(val) {}

Tensor Padding::forward(const Tensor& input) {
  return autograd::pad(input, m_pad, m_val);
}

std::unique_ptr<Module> Padding::clone() const {
  return std::make_unique<Padding>(*this);
}

std::string Padding::prettyString() const {
  std::string pads;
  for (const auto& p : m_pad) {
    pads += std::format("({}, {}), ", p.first, p.second);
  }
  return std::format("Padding ({}, {{ {}}})", m_val, pads);
}

} // namespace fl
