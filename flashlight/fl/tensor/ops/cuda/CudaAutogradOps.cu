/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/cudnn/CudnnAutogradOps.h"

#include "flashlight/fl/tensor/TensorBackend.h"

namespace fl::cuda {

ForwardBackwardState
conv2dForward(
    Tensor& out,
    const Tensor& input, const Tensor& weights, const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    bool wantState) {
  return detail::cudnn::conv2dForward(
      out, input, weights, bias, sx, sy, px, py, dx, dy, groups, wantState);
}

Tensor conv2dBackwardData(
    const Tensor& gradOutput, const Tensor& input, const Tensor& weight,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const ForwardBackwardState& state) {
  return detail::cudnn::conv2dBackwardData(
      gradOutput, input, weight, sx, sy, px, py, dx, dy, groups, state);
}

std::pair<Tensor, Tensor> conv2dBackwardFilterBias(
    const Tensor& gradOutput, const Tensor& input,
    const Tensor& weights, const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const ForwardBackwardState& state) {
  return detail::cudnn::conv2dBackwardFilterBias(
      gradOutput, input, weights, bias, sx, sy, px, py, dx, dy, groups, state);
}

ForwardBackwardState
pool2dForward(
    Tensor& out,
    const Tensor& input, int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode, bool wantState) {
  return detail::cudnn::pool2dForward(
      out, input, wx, wy, sx, sy, px, py, mode, wantState);
}

Tensor pool2dBackward(
    const Tensor& gradOutput, const Tensor& input, const Tensor& poolOutput,
    int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode, const ForwardBackwardState& state) {
  return detail::cudnn::pool2dBackward(
      gradOutput, input, poolOutput, wx, wy, sx, sy, px, py, mode, state);
}

BatchnormForwardResult
batchnormForward(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& runningMean, const Tensor& runningVar,
    const std::vector<int>& axes, bool train, double momentum, double epsilon,
    bool wantState) {
  return detail::cudnn::batchnormForward(
      input, weight, bias,
      runningMean, runningVar, axes, train, momentum, epsilon, wantState);
}

std::tuple<Tensor, Tensor, Tensor> batchnormBackward(
    const Tensor& gradOutput, const Tensor& saveMean, const Tensor& saveVar,
    const Tensor& input, const Tensor& weight,
    const std::vector<int>& axes, bool train, float epsilon,
    const ForwardBackwardState& state) {
  return detail::cudnn::batchnormBackward(
      gradOutput, saveMean, saveVar, input, weight,
      axes, train, epsilon, state);
}

std::tuple<Tensor, Tensor, Tensor, ForwardBackwardState>
rnnForward(
    const Tensor& input, const Tensor& hiddenState, const Tensor& cellState,
    const Tensor& weights, int hiddenSize, int numLayers,
    RnnMode mode, bool bidirectional, float dropout,
    bool wantState) {
  return detail::cudnn::rnnForward(
      input, hiddenState, cellState, weights,
      hiddenSize, numLayers, mode, bidirectional, dropout, wantState);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> rnnBackward(
    const Tensor& input, const Tensor& hiddenState, const Tensor& cellState,
    const Tensor& weights,
    const Tensor& dy, const Tensor& dhy, const Tensor& dcy,
    const Tensor& output, int numLayers, int hiddenSize,
    RnnMode mode, bool bidirectional, float dropProb,
    const ForwardBackwardState& state) {
  return detail::cudnn::rnnBackward(
      input, hiddenState, cellState, weights,
      dy, dhy, dcy, output, numLayers, hiddenSize,
      mode, bidirectional, dropProb, state);
}

} // namespace fl::cuda
