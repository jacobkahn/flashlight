/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <tuple>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl {

/**
 * Applies a 2D convolution over an input signal given filter weights. In
 the
 * simplest case, the output with shape [\f$X_{out}\f$, \f$Y_{out}\f$,
 * \f$C_{out}\f$, \f$N\f$] of the convolution with input [\f$X_{in}\f$,
 * \f$Y_{in}\f$, \f$C_{in}\f$, \f$N\f$] and weight [\f$K_x\f$, \f$K_y\f$,
 * \f$C_{in}\f$, \f$C_{out}\f$] can be precisely described as:
 * \f[
      \text{out}(C_{out_j}, N_i) =
          \sum_{k = 0}^{C_{in} - 1} \text{weight}(k, C_{out_j}) \star
          \text{input}(k, N_i)
 * \f]
 * @param input a Tensor with shape [\f$X_{in}\f$, \f$Y_{in}\f$,
 \f$C_{in}\f$,
 * \f$N\f$]
 * @param weights a Tensor with shape [\f$K_x\f$, \f$K_y\f$, \f$C_{in}\f$,
 * \f$C_{out}\f$]
 * @param sx stride in the first dimension
 * @param sy stride in the second dimension
 * @param px number of positions of zero-padding on both sides in the first
 * dimension
 * @param py number of positions of zero-padding on both sides in the second
 * dimension
 * @param dx dilation along the first kernel dimension. A dilation of 1
 * is equivalent to a standard convolution along this axis.
 * @param dy dilation along the second kernel dimension. A dilation of 1
 * is equivalent to a standard convolution along this axis.
 * @param groups number of filter groups
 * @return a Tensor with shape [\f$X_{out}\f$, \f$Y_{out}\f$, \f$C_{out}\f$,
 * \f$N\f$]]
 */
FL_API Tensor conv2d(
    const Tensor& input,
    const Tensor& weights,
    const int sx = 1,
    const int sy = 1,
    const int px = 0,
    const int py = 0,
    const int dx = 1,
    const int dy = 1,
    const int groups = 1);

/**
 * Applies a 2D convolution over an input signal given filter weights and
 * biases.
 */
FL_API Tensor conv2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    const int sx = 1,
    const int sy = 1,
    const int px = 0,
    const int py = 0,
    const int dx = 1,
    const int dy = 1,
    const int groups = 1);

/**
 * Applies a 2D pooling over an input signal composed of several input planes.
 */
FL_API Tensor pool2d(
    const Tensor& input,
    const int wx,
    const int wy,
    const int sx = 1,
    const int sy = 1,
    const int px = 0,
    const int py = 0,
    const PoolingMode mode = PoolingMode::MAX);

/**
 * Applies Batch Normalization over a 4D input.
 * Returns {output, newRunningMean, newRunningVar}.
 */
FL_API std::tuple<Tensor, Tensor, Tensor> batchnorm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& runningMean,
    const Tensor& runningVar,
    const std::vector<int>& axes,
    const bool train,
    const double momentum,
    const double epsilon);

/**
 * Applies an RNN unit to an input sequence.
 */
FL_API std::tuple<Tensor, Tensor, Tensor> rnn(
    const Tensor& input,
    const Tensor& hiddenState,
    const Tensor& cellState,
    const Tensor& weights,
    const int hiddenSize,
    const int numLayers,
    const RnnMode mode,
    const bool bidirectional,
    const float dropout);

namespace detail {

// Forward ops: return output + opaque state (state is null when wantState
// is false, i.e. not training / no autograd).
FL_API std::pair<Tensor, ForwardBackwardState> conv2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    bool wantState);

FL_API std::pair<Tensor, ForwardBackwardState> pool2d(
    const Tensor& input,
    int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode,
    bool wantState);

FL_API BatchnormForwardResult batchnorm(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& runningMean, const Tensor& runningVar,
    const std::vector<int>& axes, bool train, double momentum, double epsilon,
    bool wantState);

FL_API std::tuple<Tensor, Tensor, Tensor, ForwardBackwardState> rnn(
    const Tensor& input, const Tensor& hiddenState, const Tensor& cellState,
    const Tensor& weights, int hiddenSize, int numLayers,
    RnnMode mode, bool bidirectional, float dropout,
    bool wantState);

// Backward ops: take opaque state from forward.
FL_API Tensor conv2dBackwardData(
    const Tensor& gradOutput, const Tensor& input, const Tensor& weight,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const ForwardBackwardState& state);

FL_API std::pair<Tensor, Tensor> conv2dBackwardFilterBias(
    const Tensor& gradOutput, const Tensor& input,
    const Tensor& weights, const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const ForwardBackwardState& state);

FL_API Tensor pool2dBackward(
    const Tensor& gradOutput, const Tensor& input, const Tensor& poolOutput,
    int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode, const ForwardBackwardState& state);

FL_API std::tuple<Tensor, Tensor, Tensor> batchnormBackward(
    const Tensor& gradOutput, const Tensor& saveMean, const Tensor& saveVar,
    const Tensor& input, const Tensor& weight,
    const std::vector<int>& axes, bool train, float epsilon,
    const ForwardBackwardState& state);

FL_API std::tuple<Tensor, Tensor, Tensor, Tensor> rnnBackward(
    const Tensor& input, const Tensor& hiddenState, const Tensor& cellState,
    const Tensor& weights,
    const Tensor& dy, const Tensor& dhy, const Tensor& dcy,
    const Tensor& output, int numLayers, int hiddenSize,
    RnnMode mode, bool bidirectional, float dropProb,
    const ForwardBackwardState& state);

} // namespace detail

} // namespace fl
