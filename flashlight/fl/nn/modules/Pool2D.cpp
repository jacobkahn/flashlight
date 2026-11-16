/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Pool2D.h"

#include <format>
#include <stdexcept>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace fl {

using detail::IntOrPadMode;

Pool2D::Pool2D(
    int wx,
    int wy,
    int sx,
    int sy,
    IntOrPadMode px,
    IntOrPadMode py,
    PoolingMode mode)
    : xFilter_(wx),
      yFilter_(wy),
      xStride_(sx),
      yStride_(sy),
      xPad_(px.padVal),
      yPad_(py.padVal),
      mode_(mode) {}

Tensor Pool2D::forward(const Tensor& input) {
  auto px = derivePadding(
      input.dim(3),
      xFilter_,
      xStride_,
      xPad_,
      /* dilation= */ 1);
  auto py = derivePadding(
      input.dim(2),
      yFilter_,
      yStride_,
      yPad_,
      /* dilation= */ 1);

  if (!(px >= 0 && py >= 0)) {
    throw std::invalid_argument("invalid padding for Pool2D");
  }

  return autograd::pool2d(input, xFilter_, yFilter_, xStride_, yStride_, px, py, mode_);
}

std::unique_ptr<Module> Pool2D::clone() const {
  return std::make_unique<Pool2D>(*this);
}

std::string Pool2D::prettyString() const {
  const char* modeStr = [&] {
    switch (mode_) {
      case PoolingMode::MAX: return "-max";
      case PoolingMode::AVG_EXCLUDE_PADDING: return "-avg(without pad)";
      case PoolingMode::AVG_INCLUDE_PADDING: return "-avg(with pad)";
      default: return "";
    }
  }();
  auto padStr = [](int pad) -> std::string {
    return pad == static_cast<int>(PaddingMode::SAME)
        ? "SAME" : std::to_string(pad);
  };
  return std::format(
      "Pool2D{} ({}x{}, {},{}, {},{})",
      modeStr, xFilter_, yFilter_, xStride_, yStride_,
      padStr(xPad_), padStr(yPad_));
}

} // namespace fl
