/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/PrecisionCast.h"

#include <format>

#include "flashlight/fl/common/Utils.h"
#include "flashlight/fl/tensor/Types.h"

namespace fl {

PrecisionCast::PrecisionCast(fl::dtype targetType) : targetType_(targetType) {}

std::vector<Tensor> PrecisionCast::forward(
    const std::vector<Tensor>& inputs) {
  std::vector<Tensor> outputs;
  for (const auto& input : inputs) {
    auto output = input.astype(targetType_);
    outputs.push_back(output);
  }
  return outputs;
}

Tensor PrecisionCast::forward(const Tensor& input) {
  return forward(std::vector<Tensor>{input}).front();
}

Tensor PrecisionCast::operator()(const Tensor& input) {
  return this->forward(input);
}

std::unique_ptr<Module> PrecisionCast::clone() const {
  return std::make_unique<PrecisionCast>(*this);
}

std::string PrecisionCast::prettyString() const {
  return std::format("PrecisionCast * -> {}", dtypeToString(targetType_));
}

} // namespace fl
