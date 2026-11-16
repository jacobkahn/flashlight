/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <format>
#include <stdexcept>

#include "flashlight/fl/contrib/modules/AsymmetricConv1D.h"

#include "flashlight/fl/tensor/Index.h"

namespace fl {

void AsymmetricConv1D::checkParams() {
  if (xPad_ != static_cast<int>(PaddingMode::SAME) && xPad_ != 0) {
    throw std::invalid_argument(
        "AsymmetricConv1D: invalid xPad_, now supports only '0' or 'SAME' ");
  }
  if (futurePart_ < 0 || futurePart_ > 1) {
    throw std::invalid_argument(
        "AsymmetricConv1D: invalid futurePart_, should be in [0, 1]");
  }
}

AsymmetricConv1D::AsymmetricConv1D(
    int nIn,
    int nOut,
    int wx,
    int sx /*= 1 */,
    fl::detail::IntOrPadMode px /*= 0 */,
    float futurePart /* 0.5 */,
    int dx /* 1 */,
    bool bias /* true */,
    int groups /* 1 */)
    : Conv2D(nIn, nOut, wx, 1, sx, 1, px, 0, dx, 1, bias, groups),
      futurePart_(futurePart) {
  checkParams();
}

AsymmetricConv1D::AsymmetricConv1D(
    const Tensor& w,
    int sx /*= 1 */,
    fl::detail::IntOrPadMode px /*= 0 */,
    float futurePart /*= 0.5 */,
    int dx /*= 1 */,
    int groups /*= 1 */)
    : Conv2D(w, sx, 1, px, 0, dx, 1, groups), futurePart_(futurePart) {
  checkParams();
}

AsymmetricConv1D::AsymmetricConv1D(
    const Tensor& w,
    const Tensor& b,
    int sx /*= 1 */,
    fl::detail::IntOrPadMode px /*= 0 */,
    float futurePart /*= 0.5 */,
    int dx /*= 1 */,
    int groups /*= 1 */)
    : Conv2D(w, b, sx, 1, px, 0, dx, 1, groups), futurePart_(futurePart) {
  checkParams();
}

Tensor AsymmetricConv1D::forward(const Tensor& input) {
  // Row-major: input is {N, C, H, W}. xFilter maps to kW (dim 3).
  // Time is on dim 3 (W).
  auto px =
      fl::derivePadding(input.dim(3), xFilter_, xStride_, xPad_, xDilation_);
  if (!(px >= 0)) {
    throw std::invalid_argument("invalid padding for AsymmetricConv1D");
  }
  Tensor output;
  int cutPx = std::abs(2 * (0.5 - futurePart_)) * px;
  int asymmetryPx = px + cutPx;
  if (bias_) {
    output = autograd::conv2d(
        input,
        params_[0],
        params_[1],
        xStride_,
        yStride_,
        asymmetryPx,
        0,
        xDilation_,
        yDilation_,
        groups_);
  } else {
    output = autograd::conv2d(
        input,
        params_[0],
        xStride_,
        yStride_,
        asymmetryPx,
        0,
        xDilation_,
        yDilation_,
        groups_);
  }
  // Slice on dim 3 (W = time)
  if (futurePart_ < 0.5) {
    output = output(fl::span, fl::span, fl::span,
        fl::range(0, output.dim(3) - 2 * cutPx));
  } else if (futurePart_ > 0.5) {
    output = output(fl::span, fl::span, fl::span,
        fl::range(2 * cutPx, output.dim(3)));
  }
  return output;
}

std::unique_ptr<Module> AsymmetricConv1D::clone() const {
  return std::make_unique<AsymmetricConv1D>(*this);
}

std::string AsymmetricConv1D::prettyString() const {
  return std::format("AsymmetricConv1D ({})", Conv2D::prettyString());
}

} // namespace fl
