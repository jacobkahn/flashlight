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

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl::cuda {

// ======================= Unary Ops =======================

void exp(cudaStream_t stream, Tensor& out, const Tensor& tensor);
void log(cudaStream_t stream, Tensor& out, const Tensor& tensor);
void negative(cudaStream_t stream, Tensor& out, const Tensor& tensor);
void logicalNot(cudaStream_t stream, Tensor& out, const Tensor& tensor);
void log1p(cudaStream_t stream, Tensor& out, const Tensor& tensor);
void sin(cudaStream_t stream, Tensor& out, const Tensor& tensor);
void cos(cudaStream_t stream, Tensor& out, const Tensor& tensor);
void sqrt(cudaStream_t stream, Tensor& out, const Tensor& tensor);
void tanh(cudaStream_t stream, Tensor& out, const Tensor& tensor);
void floor(cudaStream_t stream, Tensor& out, const Tensor& tensor);
void ceil(cudaStream_t stream, Tensor& out, const Tensor& tensor);
void rint(cudaStream_t stream, Tensor& out, const Tensor& tensor);
void absolute(cudaStream_t stream, Tensor& out, const Tensor& tensor);
void sigmoid(cudaStream_t stream, Tensor& out, const Tensor& tensor);
void erf(cudaStream_t stream, Tensor& out, const Tensor& tensor);
void sign(cudaStream_t stream, Tensor& out, const Tensor& tensor);
void isnan(cudaStream_t stream, Tensor& out, const Tensor& tensor);
void isinf(cudaStream_t stream, Tensor& out, const Tensor& tensor);

void clip(cudaStream_t stream, Tensor& out,
    const Tensor& tensor, const Tensor& low, const Tensor& high);
void flip(cudaStream_t stream, Tensor& out,
    const Tensor& tensor, unsigned dim);
void roll(cudaStream_t stream, Tensor& out,
    const Tensor& tensor, int shift, unsigned axis);
void tril(cudaStream_t stream, Tensor& out, const Tensor& tensor, int diagonal);
void triu(cudaStream_t stream, Tensor& out, const Tensor& tensor, int diagonal);
void where(cudaStream_t stream, Tensor& out,
    const Tensor& condition, const Tensor& x, const Tensor& y);

// Sort
void sort(cudaStream_t stream, Tensor& out,
    const Tensor& input, Dim axis, SortMode sortMode);
void sort(cudaStream_t stream,
    Tensor& values, Tensor& indices, const Tensor& input,
    Dim axis, SortMode sortMode);
void argsort(cudaStream_t stream, Tensor& out,
    const Tensor& input, Dim axis, SortMode sortMode);
void topk(cudaStream_t stream,
    Tensor& values, Tensor& indices, const Tensor& input,
    unsigned k, Dim axis, SortMode sortMode);

// ======================= Binary Ops =======================

void add(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);
void sub(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);
void mul(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);
void div(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);
void eq(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);
void neq(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);
void lessThan(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);
void lessThanEqual(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);
void greaterThan(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);
void greaterThanEqual(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);
void logicalOr(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);
void logicalAnd(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);
void mod(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);
void minimum(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);
void maximum(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);
void power(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);
void bitwiseAnd(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);
void bitwiseOr(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);
void bitwiseXor(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);
void lShift(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);
void rShift(cudaStream_t stream, Tensor& out, const Tensor& lhs, const Tensor& rhs);

// ======================= Reductions =======================

void amin(cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes, bool keepDims);
void amax(cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes, bool keepDims);
void sum(cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes, bool keepDims);
void mean(cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes, bool keepDims);
void median(cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes, bool keepDims);
void std(cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes, bool keepDims);
void countNonzero(cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes, bool keepDims);
void any(cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes, bool keepDims);
void all(cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes, bool keepDims);

void var(cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes,
    bool bias, bool keepDims);
void norm(cudaStream_t stream, Tensor& out,
    const Tensor& input, const std::vector<int>& axes,
    double p, bool keepDims);
void cumsum(cudaStream_t stream, Tensor& out,
    const Tensor& input, unsigned axis);
void min(cudaStream_t stream,
    Tensor& values, Tensor& indices, const Tensor& input,
    unsigned axis, bool keepDims);
void max(cudaStream_t stream,
    Tensor& values, Tensor& indices, const Tensor& input,
    unsigned axis, bool keepDims);
void argmax(cudaStream_t stream, Tensor& out,
    const Tensor& input, unsigned axis, bool keepDims);
void argmin(cudaStream_t stream, Tensor& out,
    const Tensor& input, unsigned axis, bool keepDims);

// ======================= BLAS =======================

void matmul(Tensor& out,
    const Tensor& lhs, const Tensor& rhs,
    MatrixProperty lhsProp, MatrixProperty rhsProp,
    cudaStream_t stream, cublasHandle_t handle);

// ======================= Shaping & Indexing =======================

void reshape(cudaStream_t stream, Tensor& out,
    const Tensor& tensor, const Shape& shape);
void transpose(cudaStream_t stream, Tensor& out,
    const Tensor& tensor, const Shape& axes);
void tile(cudaStream_t stream, Tensor& out,
    const Tensor& tensor, const Shape& tileDims);
void concatenate(cudaStream_t stream, Tensor& out,
    const std::vector<Tensor>& tensors, unsigned axis);
void nonzero(cudaStream_t stream, Tensor& out, const Tensor& tensor);
void pad(cudaStream_t stream, Tensor& out,
    const Tensor& input,
    const std::vector<std::pair<int, int>>& padWidths,
    PadType type);

// ======================= Autograd Ops =======================

ForwardBackwardState conv2dForward(
    Tensor& out,
    const Tensor& input, const Tensor& weights, const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    bool wantState);
Tensor conv2dBackwardData(
    const Tensor& gradOutput, const Tensor& input, const Tensor& weight,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const ForwardBackwardState& state);
std::pair<Tensor, Tensor> conv2dBackwardFilterBias(
    const Tensor& gradOutput, const Tensor& input,
    const Tensor& weights, const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const ForwardBackwardState& state);

ForwardBackwardState pool2dForward(
    Tensor& out,
    const Tensor& input, int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode, bool wantState);
Tensor pool2dBackward(
    const Tensor& gradOutput, const Tensor& input, const Tensor& poolOutput,
    int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode, const ForwardBackwardState& state);

BatchnormForwardResult batchnormForward(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& runningMean, const Tensor& runningVar,
    const std::vector<int>& axes, bool train, double momentum, double epsilon,
    bool wantState);
std::tuple<Tensor, Tensor, Tensor> batchnormBackward(
    const Tensor& gradOutput, const Tensor& saveMean, const Tensor& saveVar,
    const Tensor& input, const Tensor& weight,
    const std::vector<int>& axes, bool train, float epsilon,
    const ForwardBackwardState& state);

std::tuple<Tensor, Tensor, Tensor, ForwardBackwardState> rnnForward(
    const Tensor& input, const Tensor& hiddenState, const Tensor& cellState,
    const Tensor& weights, int hiddenSize, int numLayers,
    RnnMode mode, bool bidirectional, float dropout,
    bool wantState);
std::tuple<Tensor, Tensor, Tensor, Tensor> rnnBackward(
    const Tensor& input, const Tensor& hiddenState, const Tensor& cellState,
    const Tensor& weights,
    const Tensor& dy, const Tensor& dhy, const Tensor& dcy,
    const Tensor& output, int numLayers, int hiddenSize,
    RnnMode mode, bool bidirectional, float dropProb,
    const ForwardBackwardState& state);

} // namespace fl::cuda
