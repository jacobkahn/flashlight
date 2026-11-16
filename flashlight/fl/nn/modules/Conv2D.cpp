/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Conv2D.h"

#include <cmath>
#include <format>
#include <stdexcept>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl {

using detail::IntOrPadMode;

Conv2D::Conv2D(
    int nin,
    int nout,
    int wx,
    int wy,
    int sx,
    int sy,
    IntOrPadMode px,
    IntOrPadMode py,
    int dx,
    int dy,
    bool bias,
    int groups)
    : nIn_(nin),
      nOut_(nout),
      xFilter_(wx),
      yFilter_(wy),
      xStride_(sx),
      yStride_(sy),
      xPad_(px.padVal),
      yPad_(py.padVal),
      xDilation_(dx),
      yDilation_(dy),
      bias_(bias),
      groups_(groups) {
  initialize();
}

Conv2D::Conv2D(
    const Tensor& w,
    int sx,
    int sy,
    IntOrPadMode px,
    IntOrPadMode py,
    int dx,
    int dy,
    int groups)
    : UnaryModule({w}),
      nIn_(w.dim(1)),
      nOut_(w.dim(0)),
      xFilter_(w.dim(3)),
      yFilter_(w.dim(2)),
      xStride_(sx),
      yStride_(sy),
      xPad_(px.padVal),
      yPad_(py.padVal),
      xDilation_(dx),
      yDilation_(dy),
      bias_(false),
      groups_(groups) {}

Conv2D::Conv2D(
    const Tensor& w,
    const Tensor& b,
    int sx,
    int sy,
    IntOrPadMode px,
    IntOrPadMode py,
    int dx,
    int dy,
    int groups)
    : UnaryModule({w, b}),
      nIn_(w.dim(1)),
      nOut_(w.dim(0)),
      xFilter_(w.dim(3)),
      yFilter_(w.dim(2)),
      xStride_(sx),
      yStride_(sy),
      xPad_(px.padVal),
      yPad_(py.padVal),
      xDilation_(dx),
      yDilation_(dy),
      bias_(true),
      groups_(groups) {
  if (b.dim(1) != w.dim(0)) {
    throw std::invalid_argument(
        "output channel dimension mismatch between Conv2D weight and bias");
  }
  if (b.elements() != b.dim(1)) {
    throw std::invalid_argument(
        "only 2nd dimension of Conv2D bias may be non-singleton");
  }
}

Conv2D::Conv2D(const Conv2D& other)
    : UnaryModule(other.copyParams()),
      nIn_(other.nIn_),
      nOut_(other.nOut_),
      xFilter_(other.xFilter_),
      yFilter_(other.yFilter_),
      xStride_(other.xStride_),
      yStride_(other.yStride_),
      xPad_(other.xPad_),
      yPad_(other.yPad_),
      xDilation_(other.xDilation_),
      yDilation_(other.yDilation_),
      bias_(other.bias_),
      groups_(other.groups_) {
  train_ = other.train_;
}

Conv2D& Conv2D::operator=(const Conv2D& other) {
  params_ = other.copyParams();
  train_ = other.train_;
  nIn_ = other.nIn_;
  nOut_ = other.nOut_;
  xFilter_ = other.xFilter_;
  yFilter_ = other.yFilter_;
  xStride_ = other.xStride_;
  yStride_ = other.yStride_;
  xPad_ = other.xPad_;
  yPad_ = other.yPad_;
  xDilation_ = other.xDilation_;
  yDilation_ = other.yDilation_;
  bias_ = other.bias_;
  groups_ = other.groups_;
  return *this;
}

Tensor Conv2D::forward(const Tensor& input) {
  auto px = derivePadding(input.dim(3), xFilter_, xStride_, xPad_, xDilation_);
  auto py = derivePadding(input.dim(2), yFilter_, yStride_, yPad_, yDilation_);
  if (!(px >= 0 && py >= 0)) {
    throw std::invalid_argument("invalid padding for Conv2D");
  }

  if (bias_) {
    return autograd::conv2d(
        input,
        params_[0].astype(input.type()),
        params_[1].astype(input.type()),
        xStride_,
        yStride_,
        px,
        py,
        xDilation_,
        yDilation_,
        groups_);
  } else {
    return autograd::conv2d(
        input,
        params_[0].astype(input.type()),
        xStride_,
        yStride_,
        px,
        py,
        xDilation_,
        yDilation_,
        groups_);
  }
}

void Conv2D::initialize() {
  int fanIn = xFilter_ * yFilter_ * nIn_ / groups_;
  auto wt = kaimingUniform(
      Shape({nOut_, nIn_ / groups_, yFilter_, xFilter_}),
      fanIn,
      fl::dtype::f32,
      true);
  if (bias_) {
    double bound = std::sqrt(1.0 / fanIn);
    auto bs =
        uniform(Shape({1, nOut_, 1, 1}), -bound, bound, fl::dtype::f32, true);
    params_ = {wt, bs};
  } else {
    params_ = {wt};
  }

}

std::unique_ptr<Module> Conv2D::clone() const {
  return std::make_unique<Conv2D>(*this);
}

std::string Conv2D::prettyString() const {
  auto padStr = [](int pad) -> std::string {
    return pad == static_cast<int>(PaddingMode::SAME)
        ? "SAME" : std::to_string(pad);
  };
  return std::format(
      "Conv2D ({}->{}, {}x{}, {},{}, {},{}, {}, {}) ({})",
      nIn_, nOut_, xFilter_, yFilter_, xStride_, yStride_,
      padStr(xPad_), padStr(yPad_), xDilation_, yDilation_,
      bias_ ? "with bias" : "without bias");
}

} // namespace fl
