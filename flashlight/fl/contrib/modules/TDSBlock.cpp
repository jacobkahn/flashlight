/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/TDSBlock.h"

#include <format>

namespace fl {

TDSBlock::TDSBlock(
    int channels,
    int kernelSize,
    int width,
    double dropout /* = 0 */,
    int innerLinearDim /* = 0 */,
    int rightPadding /* = -1 */,
    bool lNormIncludeTime /* = true */) {
  // Row-major: input is {B, C, W, T} = {N, Cin, H, W_spatial}
  // Conv2D(c, c, kernelSize, 1): wx=kernelSize-->kW on dim 3 (T), wy=1-->kH on dim 2 (W)
  Sequential conv;
  auto convPadding = static_cast<int>(fl::PaddingMode::SAME);
  if (rightPadding != -1) {
    int totalPadding = kernelSize - 1;
    if (rightPadding > totalPadding) {
      throw std::invalid_argument(
          "right padding exceeds the 'SAME' padding required for TDSBlock");
    }
    // Pad on dim 3 (T/W): {{dim0}, {dim1}, {dim2}, {dim3}}
    conv.add(Padding(
        {{0, 0}, {0, 0}, {0, 0},
         {totalPadding - rightPadding, rightPadding}}, 0.0));
    convPadding = 0;
  }
  conv.add(Conv2D(channels, channels, kernelSize, 1, 1, 1, convPadding, 0));
  conv.add(ReLU());
  conv.add(Dropout(dropout));

  int linearDim = channels * width;
  if (innerLinearDim == 0) {
    innerLinearDim = linearDim;
  }
  // fc: flatten C*W_spatial into last dim for Linear, then unflatten
  // {B, C, W, T} --> reorder --> {B, T, C, W} --> view --> {B, T, C*W}
  // --> Linear --> ... --> view --> {B, T, C, W} --> reorder --> {B, C, W, T}
  Sequential fc;
  fc.add(Reorder({0, 3, 1, 2}));
  fc.add(View(Shape({0, -1, linearDim})));

  fc.add(Linear(linearDim, innerLinearDim));
  fc.add(ReLU());
  if (dropout > 0) {
    fc.add(Dropout(dropout));
  }
  fc.add(Linear(innerLinearDim, linearDim));
  fc.add(View(Shape({0, -1, channels, width})));
  fc.add(Reorder({0, 2, 3, 1}));
  if (dropout > 0) {
    fc.add(Dropout(dropout));
  }

  add(std::move(conv));
  // Row-major: {B, C, T, W}. Normalize over C, T, W = dims {1, 2, 3}
  if (lNormIncludeTime) {
    add(LayerNorm(std::vector<int>{1, 2, 3}));
  } else {
    // Normalize over C, W only = dims {1, 3}
    add(LayerNorm(std::vector<int>{1, 3}));
  }
  add(std::move(fc));
  if (lNormIncludeTime) {
    add(LayerNorm(std::vector<int>{1, 2, 3}));
  } else {
    add(LayerNorm(std::vector<int>{1, 3}));
  }
}

std::vector<Tensor> TDSBlock::forward(const std::vector<Tensor>& inputs) {
  auto out = inputs[0];
  out = module(0)->forward({out})[0].astype(out.type()) + out;
  out = module(1)->forward({out})[0];
  out = module(2)->forward({out})[0].astype(out.type()) + out;
  return module(3)->forward({out});
}

std::string TDSBlock::prettyString() const {
  auto convW = param(0);
  auto linW = param(4);
  // Row-major: Conv2D weight is {Cout, Cin/groups, kH, kW}
  int kw = convW.dim(3);  // kW = kernelSize (time dim)
  int c = convW.dim(0);   // Cout = channels
  // Linear weight is {outDim, inDim}
  int w = linW.dim(1) / c;  // linearDim / channels = width
  int l = linW.dim(1);      // inDim = linearDim
  int l2 = linW.dim(0);     // outDim = innerLinearDim
  return std::format(
      "Time-Depth Separable Block ({}, {}, {}) [{} -> {} -> {}]",
      kw, w, c, l, l2, l);
}

} // namespace fl
