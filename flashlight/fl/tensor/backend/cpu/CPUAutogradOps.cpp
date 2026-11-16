/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/cpu/CPUBackend.h"

#include "flashlight/fl/tensor/ops/onednn/OneDnnAutogradOps.h"
#include "flashlight/fl/tensor/ops/onednn/OneDnnOps.h"

namespace fl {

ForwardBackwardState
CPUBackend::conv2dForward(
    Tensor& out,
    const Tensor& input, const Tensor& weights, const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    bool wantState) {
  return detail::onednn::conv2dForward(
      out, input, weights, bias, sx, sy, px, py, dx, dy, groups, wantState,
      onednnOps().engine(), *asyncStream_, nativeStream());
}

Tensor CPUBackend::conv2dBackwardData(
    const Tensor& gradOutput, const Tensor& input, const Tensor& weight,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const ForwardBackwardState& state) {
  return detail::onednn::conv2dBackwardData(
      gradOutput, input, weight, sx, sy, px, py, dx, dy, groups, state,
      onednnOps().engine(), *asyncStream_, nativeStream());
}

std::pair<Tensor, Tensor> CPUBackend::conv2dBackwardFilterBias(
    const Tensor& gradOutput, const Tensor& input,
    const Tensor& weights, const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const ForwardBackwardState& state) {
  return detail::onednn::conv2dBackwardFilterBias(
      gradOutput, input, weights, bias, sx, sy, px, py, dx, dy, groups, state,
      onednnOps().engine(), *asyncStream_, nativeStream());
}

ForwardBackwardState
CPUBackend::pool2dForward(
    Tensor& out,
    const Tensor& input, int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode, bool wantState) {
  return detail::onednn::pool2dForward(
      out, input, wx, wy, sx, sy, px, py, mode, wantState,
      onednnOps().engine(), *asyncStream_, nativeStream());
}

Tensor CPUBackend::pool2dBackward(
    const Tensor& gradOutput, const Tensor& input, const Tensor& poolOutput,
    int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode, const ForwardBackwardState& state) {
  return detail::onednn::pool2dBackward(
      gradOutput, input, poolOutput, wx, wy, sx, sy, px, py, mode, state,
      onednnOps().engine(), *asyncStream_, nativeStream());
}

BatchnormForwardResult
CPUBackend::batchnormForward(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& runningMean, const Tensor& runningVar,
    const std::vector<int>& axes, bool train, double momentum, double epsilon,
    bool wantState) {
  return detail::onednn::batchnormForward(
      input, weight, bias,
      runningMean, runningVar, axes, train, momentum, epsilon, wantState,
      onednnOps().engine(), *asyncStream_, nativeStream());
}

std::tuple<Tensor, Tensor, Tensor> CPUBackend::batchnormBackward(
    const Tensor& gradOutput, const Tensor& saveMean, const Tensor& saveVar,
    const Tensor& input, const Tensor& weight,
    const std::vector<int>& axes, bool train, float epsilon,
    const ForwardBackwardState& state) {
  return detail::onednn::batchnormBackward(
      gradOutput, saveMean, saveVar, input, weight,
      axes, train, epsilon, state,
      onednnOps().engine(), *asyncStream_, nativeStream());
}

std::tuple<Tensor, Tensor, Tensor, ForwardBackwardState>
CPUBackend::rnnForward(
    const Tensor& input, const Tensor& hiddenState, const Tensor& cellState,
    const Tensor& weights, int hiddenSize, int numLayers,
    RnnMode mode, bool bidirectional, float dropout,
    bool wantState) {
  return detail::onednn::rnnForward(
      input, hiddenState, cellState, weights,
      hiddenSize, numLayers, mode, bidirectional, dropout, wantState,
      onednnOps().engine(), *asyncStream_, nativeStream());
}

std::tuple<Tensor, Tensor, Tensor, Tensor> CPUBackend::rnnBackward(
    const Tensor& input, const Tensor& hiddenState, const Tensor& cellState,
    const Tensor& weights,
    const Tensor& dy, const Tensor& dhy, const Tensor& dcy,
    const Tensor& output, int numLayers, int hiddenSize,
    RnnMode mode, bool bidirectional, float dropProb,
    const ForwardBackwardState& state) {
  return detail::onednn::rnnBackward(
      input, hiddenState, cellState, weights,
      dy, dhy, dcy, output, numLayers, hiddenSize,
      mode, bidirectional, dropProb, state);
}

} // namespace fl
