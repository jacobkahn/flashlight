/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/AutogradOps.h"
#include "flashlight/fl/tensor/TensorBackend.h"

namespace fl {

Tensor conv2d(
    const Tensor& input,
    const Tensor& weights,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const int dx,
    const int dy,
    const int groups) {
  auto dummyBias = Tensor(input.type());
  return conv2d(input, weights, dummyBias, sx, sy, px, py, dx, dy, groups);
}

Tensor conv2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const int dx,
    const int dy,
    const int groups) {
  auto [output, _] = detail::conv2d(
      input, weights, bias, sx, sy, px, py, dx, dy, groups,
      /*wantState=*/false);
  return output;
}

Tensor pool2d(
    const Tensor& input,
    const int wx,
    const int wy,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const PoolingMode mode) {
  auto [output, _] = detail::pool2d(
      input, wx, wy, sx, sy, px, py, mode, /*wantState=*/false);
  return output;
}

std::tuple<Tensor, Tensor, Tensor> batchnorm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& runningMean,
    const Tensor& runningVar,
    const std::vector<int>& axes,
    const bool train,
    const double momentum,
    const double epsilon) {
  auto result = detail::batchnorm(
      input, weight, bias,
      runningMean, runningVar, axes, train, momentum, epsilon,
      /*wantState=*/false);
  return {std::move(result.output),
          std::move(result.newRunningMean),
          std::move(result.newRunningVar)};
}

std::tuple<Tensor, Tensor, Tensor> rnn(
    const Tensor& input,
    const Tensor& hiddenState,
    const Tensor& cellState,
    const Tensor& weights,
    const int hiddenSize,
    const int numLayers,
    const RnnMode mode,
    const bool bidirectional,
    const float dropout) {
  auto [y, hy, cy, _] = detail::rnn(
      input, hiddenState, cellState, weights,
      hiddenSize, numLayers, mode, bidirectional, dropout,
      /*wantState=*/false);
  return {y, hy, cy};
}

namespace detail {

std::pair<Tensor, ForwardBackwardState> conv2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    bool wantState) {
  Tensor out;
  auto state = input.backend().conv2dForward(
      out, input, weights, bias, sx, sy, px, py, dx, dy, groups, wantState);
  return {std::move(out), std::move(state)};
}

std::pair<Tensor, ForwardBackwardState> pool2d(
    const Tensor& input,
    int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode,
    bool wantState) {
  Tensor out;
  auto state = input.backend().pool2dForward(
      out, input, wx, wy, sx, sy, px, py, mode, wantState);
  return {std::move(out), std::move(state)};
}

BatchnormForwardResult batchnorm(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& runningMean, const Tensor& runningVar,
    const std::vector<int>& axes, bool train, double momentum, double epsilon,
    bool wantState) {
  return input.backend().batchnormForward(
      input, weight, bias,
      runningMean, runningVar, axes, train, momentum, epsilon, wantState);
}

std::tuple<Tensor, Tensor, Tensor, ForwardBackwardState> rnn(
    const Tensor& input, const Tensor& hiddenState, const Tensor& cellState,
    const Tensor& weights, int hiddenSize, int numLayers,
    RnnMode mode, bool bidirectional, float dropout,
    bool wantState) {
  return input.backend().rnnForward(
      input, hiddenState, cellState, weights,
      hiddenSize, numLayers, mode, bidirectional, dropout, wantState);
}

Tensor conv2dBackwardData(
    const Tensor& gradOutput, const Tensor& input, const Tensor& weight,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const ForwardBackwardState& state) {
  return input.backend().conv2dBackwardData(
      gradOutput, input, weight, sx, sy, px, py, dx, dy, groups, state);
}

std::pair<Tensor, Tensor> conv2dBackwardFilterBias(
    const Tensor& gradOutput, const Tensor& input,
    const Tensor& weights, const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const ForwardBackwardState& state) {
  return input.backend().conv2dBackwardFilterBias(
      gradOutput, input, weights, bias,
      sx, sy, px, py, dx, dy, groups, state);
}

Tensor pool2dBackward(
    const Tensor& gradOutput, const Tensor& input, const Tensor& poolOutput,
    int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode, const ForwardBackwardState& state) {
  return input.backend().pool2dBackward(
      gradOutput, input, poolOutput, wx, wy, sx, sy, px, py, mode, state);
}

std::tuple<Tensor, Tensor, Tensor> batchnormBackward(
    const Tensor& gradOutput, const Tensor& saveMean, const Tensor& saveVar,
    const Tensor& input, const Tensor& weight,
    const std::vector<int>& axes, bool train, float epsilon,
    const ForwardBackwardState& state) {
  return input.backend().batchnormBackward(
      gradOutput, saveMean, saveVar, input, weight,
      axes, train, epsilon, state);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> rnnBackward(
    const Tensor& input, const Tensor& hiddenState, const Tensor& cellState,
    const Tensor& weights,
    const Tensor& dy, const Tensor& dhy, const Tensor& dcy,
    const Tensor& output, int numLayers, int hiddenSize,
    RnnMode mode, bool bidirectional, float dropProb,
    const ForwardBackwardState& state) {
  return input.backend().rnnBackward(
      input, hiddenState, cellState, weights,
      dy, dhy, dcy, output, numLayers, hiddenSize,
      mode, bidirectional, dropProb, state);
}

} // namespace detail

} // namespace fl
