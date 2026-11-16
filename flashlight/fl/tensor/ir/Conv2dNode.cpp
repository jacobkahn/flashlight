/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/Conv2dNode.h"

#include "flashlight/fl/tensor/ShapeInference.h"

#include <stdexcept>

namespace fl {

Conv2dNode::Conv2dNode(
    NodePtr input,
    NodePtr weights,
    NodePtr bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    bool wantState,
    const Shape& outputShape,
    dtype outputDtype,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait({input, weights, bias}, outputShape, outputDtype, std::move(stream)),
      sx_(sx), sy_(sy), px_(px), py_(py), dx_(dx), dy_(dy),
      groups_(groups), wantState_(wantState) {}

Conv2dNodePtr Conv2dNode::create(
    NodePtr input,
    NodePtr weights,
    NodePtr bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    bool wantState,
    std::shared_ptr<const Stream> stream) {
  // Conv2d output shape: NCHW layout
  // input:   (N, C_in, H_in, W_in)
  // weights: (C_out, C_in/groups, kH, kW)
  // output:  (N, C_out, H_out, W_out)
  const auto& inShape = input->shape();
  const auto& wShape = weights->shape();
  if (inShape.ndim() != 4) {
    throw std::invalid_argument(
        "Conv2dNode::create: input must be 4D (NCHW), got " +
        std::to_string(inShape.ndim()) + "D");
  }
  if (wShape.ndim() != 4) {
    throw std::invalid_argument(
        "Conv2dNode::create: weights must be 4D, got " +
        std::to_string(wShape.ndim()) + "D");
  }

  auto N = inShape.dim(0);
  auto H_in = inShape.dim(2);
  auto W_in = inShape.dim(3);
  auto C_out = wShape.dim(0);
  auto kH = wShape.dim(2);
  auto kW = wShape.dim(3);

  auto H_out = (H_in + 2 * py - dy * (kH - 1) - 1) / sy + 1;
  auto W_out = (W_in + 2 * px - dx * (kW - 1) - 1) / sx + 1;

  Shape outputShape({N, C_out, H_out, W_out});
  auto outputDtype = input->dataType();

  return std::make_shared<Conv2dNode>(
      input, weights, bias,
      sx, sy, px, py, dx, dy, groups, wantState,
      outputShape, outputDtype, std::move(stream), PrivateHelper{});
}

NodePtr Conv2dNode::input() const {
  return getInput(kInputIdx);
}

NodePtr Conv2dNode::weights() const {
  return getInput(kWeightsIdx);
}

NodePtr Conv2dNode::bias() const {
  return getInput(kBiasIdx);
}

int Conv2dNode::sx() const { return sx_; }
int Conv2dNode::sy() const { return sy_; }
int Conv2dNode::px() const { return px_; }
int Conv2dNode::py() const { return py_; }
int Conv2dNode::dx() const { return dx_; }
int Conv2dNode::dy() const { return dy_; }
int Conv2dNode::groups() const { return groups_; }
bool Conv2dNode::wantState() const { return wantState_; }

void Conv2dNode::setCachedState(ForwardBackwardState s) const {
  cachedState_ = std::move(s);
}

const ForwardBackwardState& Conv2dNode::cachedState() const {
  return cachedState_;
}

} // namespace fl
