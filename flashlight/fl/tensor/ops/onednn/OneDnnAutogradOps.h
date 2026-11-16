/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <dnnl.hpp>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl {

class AsyncCPUStream;

namespace detail {
namespace onednn {

// -- conv2d --
ForwardBackwardState conv2dForward(
    Tensor& out,
    const Tensor& input, const Tensor& weights, const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    bool wantState,
    const dnnl::engine& engine,
    AsyncCPUStream& asyncStream,
    dnnl::stream& nativeStream,
    const dnnl::primitive_attr& attr = dnnl::primitive_attr(),
    const Tensor& sumSrc = Tensor());

Tensor conv2dBackwardData(
    const Tensor& gradOutput, const Tensor& input, const Tensor& weight,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const ForwardBackwardState& state,
    const dnnl::engine& engine,
    AsyncCPUStream& asyncStream,
    dnnl::stream& nativeStream);

std::pair<Tensor, Tensor> conv2dBackwardFilterBias(
    const Tensor& gradOutput, const Tensor& input,
    const Tensor& weights, const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const ForwardBackwardState& state,
    const dnnl::engine& engine,
    AsyncCPUStream& asyncStream,
    dnnl::stream& nativeStream);

// -- pool2d --
ForwardBackwardState pool2dForward(
    Tensor& out,
    const Tensor& input, int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode, bool wantState,
    const dnnl::engine& engine,
    AsyncCPUStream& asyncStream,
    dnnl::stream& nativeStream,
    const dnnl::primitive_attr& attr = dnnl::primitive_attr());

Tensor pool2dBackward(
    const Tensor& gradOutput, const Tensor& input, const Tensor& poolOutput,
    int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode, const ForwardBackwardState& state,
    const dnnl::engine& engine,
    AsyncCPUStream& asyncStream,
    dnnl::stream& nativeStream);

// -- batchnorm --
BatchnormForwardResult batchnormForward(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& runningMean, const Tensor& runningVar,
    const std::vector<int>& axes, bool train, double momentum, double epsilon,
    bool wantState,
    const dnnl::engine& engine,
    AsyncCPUStream& asyncStream,
    dnnl::stream& nativeStream,
    const dnnl::primitive_attr& attr = dnnl::primitive_attr());

std::tuple<Tensor, Tensor, Tensor> batchnormBackward(
    const Tensor& gradOutput, const Tensor& saveMean, const Tensor& saveVar,
    const Tensor& input, const Tensor& weight,
    const std::vector<int>& axes, bool train, float epsilon,
    const ForwardBackwardState& state,
    const dnnl::engine& engine,
    AsyncCPUStream& asyncStream,
    dnnl::stream& nativeStream);

// -- rnn --
std::tuple<Tensor, Tensor, Tensor, ForwardBackwardState> rnnForward(
    const Tensor& input, const Tensor& hiddenState, const Tensor& cellState,
    const Tensor& weights, int hiddenSize, int numLayers,
    RnnMode mode, bool bidirectional, float dropout,
    bool wantState,
    const dnnl::engine& engine,
    AsyncCPUStream& asyncStream,
    dnnl::stream& nativeStream);

std::tuple<Tensor, Tensor, Tensor, Tensor> rnnBackward(
    const Tensor& input, const Tensor& hiddenState, const Tensor& cellState,
    const Tensor& weights,
    const Tensor& dy, const Tensor& dhy, const Tensor& dcy,
    const Tensor& output, int numLayers, int hiddenSize,
    RnnMode mode, bool bidirectional, float dropProb,
    const ForwardBackwardState& state);

} // namespace onednn
} // namespace detail
} // namespace fl
