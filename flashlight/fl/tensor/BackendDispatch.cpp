/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/BackendDispatch.h"

#include "flashlight/fl/tensor/TensorBackend.h"

namespace fl {

BackendDispatch makeBackendDispatch(TensorBackend& backend) {
  BackendDispatch d;
  d.ctx = &backend;
  d.backendType = backend.backendType();

  // --- Evaluator hot path (7) ---
  d.binary = [](TensorBackend* c, Tensor& out, BinaryOp op,
                const Tensor& lhs, const Tensor& rhs) {
    c->binary(out, op, lhs, rhs);
  };
  d.unary = [](TensorBackend* c, Tensor& out, UnaryOp op, const Tensor& input) {
    c->unary(out, op, input);
  };
  d.reduction = [](TensorBackend* c, Tensor& out, ReductionOp op,
                   const Tensor& input,
                   const std::vector<int>& axes, bool keepDims) {
    c->reduction(out, op, input, axes, keepDims);
  };
  d.reshape = [](TensorBackend* c, Tensor& out,
                 const Tensor& input, const Shape& shape) {
    c->reshape(out, input, shape);
  };
  d.transpose = [](TensorBackend* c, Tensor& out,
                   const Tensor& input, const Shape& axes) {
    c->transpose(out, input, axes);
  };
  d.matmul = [](TensorBackend* c, Tensor& out,
                const Tensor& lhs, const Tensor& rhs,
                MatrixProperty lhsProp, MatrixProperty rhsProp) {
    c->matmul(out, lhs, rhs, lhsProp, rhsProp);
  };
  d.full = [](TensorBackend* c, const Shape& shape, double value, dtype type) {
    return c->full(shape, value, type);
  };
  d.empty = [](TensorBackend* c, const Shape& shape, dtype type) {
    return c->empty(shape, type);
  };

  // --- Creation/random (5) ---
  d.randn = [](TensorBackend* c, const Shape& shape, dtype type) {
    return c->randn(shape, type);
  };
  d.rand = [](TensorBackend* c, const Shape& shape, dtype type) {
    return c->rand(shape, type);
  };
  d.identity = [](TensorBackend* c, Dim dim, dtype type) {
    return c->identity(dim, type);
  };
  d.arange = [](TensorBackend* c, const Shape& shape, Dim seqDim, dtype type) {
    return c->arange(shape, seqDim, type);
  };
  d.iota = [](TensorBackend* c, const Shape& dims,
              const Shape& tileDims, dtype type) {
    return c->iota(dims, tileDims, type);
  };

  // --- Admin/lifecycle (8) ---
  d.eval = [](TensorBackend* c, const Tensor& tensor) {
    c->eval(tensor);
  };
  d.supportsDataType = [](TensorBackend* c, const fl::dtype& dtype) -> bool {
    return c->supportsDataType(dtype);
  };
  d.setSeed = [](TensorBackend* c, int seed) {
    c->setSeed(seed);
  };
  d.getMemMgrInfo = [](TensorBackend* c, const char* msg, int deviceId,
                       std::ostream* ostream) {
    c->getMemMgrInfo(msg, deviceId, ostream);
  };
  d.setMemMgrLogStream = [](TensorBackend* c, std::ostream* stream) {
    c->setMemMgrLogStream(stream);
  };
  d.setMemMgrLoggingEnabled = [](TensorBackend* c, bool enabled) {
    c->setMemMgrLoggingEnabled(enabled);
  };
  d.setMemMgrFlushInterval = [](TensorBackend* c, size_t interval) {
    c->setMemMgrFlushInterval(interval);
  };
  d.print = [](TensorBackend* c, const Tensor& tensor) {
    c->print(tensor);
  };

  // --- Shaping/indexing (4) ---
  d.tile = [](TensorBackend* c, Tensor& out, const Tensor& tensor,
              const Shape& shape) {
    c->tile(out, tensor, shape);
  };
  d.concatenate = [](TensorBackend* c, Tensor& out,
                     const std::vector<Tensor>& tensors, unsigned axis) {
    c->concatenate(out, tensors, axis);
  };
  d.nonzero = [](TensorBackend* c, Tensor& out, const Tensor& tensor) {
    c->nonzero(out, tensor);
  };
  d.pad = [](TensorBackend* c, Tensor& out, const Tensor& input,
             const std::vector<std::pair<int, int>>& padWidths, PadType type) {
    c->pad(out, input, padWidths, type);
  };

  // --- Named unary/misc (8) ---
  d.flip = [](TensorBackend* c, Tensor& out, const Tensor& tensor, unsigned dim) {
    c->flip(out, tensor, dim);
  };
  d.clip = [](TensorBackend* c, Tensor& out, const Tensor& tensor,
              const Tensor& low, const Tensor& high) {
    c->clip(out, tensor, low, high);
  };
  d.roll = [](TensorBackend* c, Tensor& out, const Tensor& tensor,
              int shift, unsigned axis) {
    c->roll(out, tensor, shift, axis);
  };
  d.tril = [](TensorBackend* c, Tensor& out, const Tensor& tensor, int diagonal) {
    c->tril(out, tensor, diagonal);
  };
  d.triu = [](TensorBackend* c, Tensor& out, const Tensor& tensor, int diagonal) {
    c->triu(out, tensor, diagonal);
  };
  d.where = [](TensorBackend* c, Tensor& out, const Tensor& condition,
               const Tensor& x, const Tensor& y) {
    c->where(out, condition, x, y);
  };
  d.sort = [](TensorBackend* c, Tensor& out, const Tensor& input,
              Dim axis, SortMode sortMode) {
    c->sort(out, input, axis, sortMode);
  };
  d.argsort = [](TensorBackend* c, Tensor& out, const Tensor& input,
                 Dim axis, SortMode sortMode) {
    c->argsort(out, input, axis, sortMode);
  };

  // --- 2-output ops (3) ---
  d.topk = [](TensorBackend* c, Tensor& values, Tensor& indices,
              const Tensor& input, unsigned k, Dim axis, SortMode sortMode) {
    c->topk(values, indices, input, k, axis, sortMode);
  };
  d.sortWithIndices = [](TensorBackend* c, Tensor& values, Tensor& indices,
                        const Tensor& input, Dim axis, SortMode sortMode) {
    c->sort(values, indices, input, axis, sortMode);
  };

  // --- Named reductions (7) ---
  d.min = [](TensorBackend* c, Tensor& values, Tensor& indices,
             const Tensor& input, unsigned axis, bool keepDims) {
    c->min(values, indices, input, axis, keepDims);
  };
  d.max = [](TensorBackend* c, Tensor& values, Tensor& indices,
             const Tensor& input, unsigned axis, bool keepDims) {
    c->max(values, indices, input, axis, keepDims);
  };
  d.cumsum = [](TensorBackend* c, Tensor& out, const Tensor& input, unsigned axis) {
    c->cumsum(out, input, axis);
  };
  d.argmax = [](TensorBackend* c, Tensor& out, const Tensor& input,
                unsigned axis, bool keepDims) {
    c->argmax(out, input, axis, keepDims);
  };
  d.argmin = [](TensorBackend* c, Tensor& out, const Tensor& input,
                unsigned axis, bool keepDims) {
    c->argmin(out, input, axis, keepDims);
  };
  d.var = [](TensorBackend* c, Tensor& out, const Tensor& input,
             const std::vector<int>& axes, bool bias, bool keepDims) {
    c->var(out, input, axes, bias, keepDims);
  };
  d.norm = [](TensorBackend* c, Tensor& out, const Tensor& input,
              const std::vector<int>& axes, double p, bool keepDims) {
    c->norm(out, input, axes, p, keepDims);
  };

  // --- Autograd (9) ---
  d.conv2dForward = [](TensorBackend* c, Tensor& out,
                       const Tensor& input, const Tensor& weights,
                       const Tensor& bias,
                       int sx, int sy, int px, int py, int dx, int dy,
                       int groups, bool wantState) -> ForwardBackwardState {
    return c->conv2dForward(
        out, input, weights, bias, sx, sy, px, py, dx, dy, groups, wantState);
  };
  d.conv2dBackwardData = [](TensorBackend* c, const Tensor& gradOutput,
                            const Tensor& input, const Tensor& weight,
                            int sx, int sy, int px, int py, int dx, int dy,
                            int groups,
                            const ForwardBackwardState& state) -> Tensor {
    return c->conv2dBackwardData(
        gradOutput, input, weight, sx, sy, px, py, dx, dy, groups, state);
  };
  d.conv2dBackwardFilterBias = [](TensorBackend* c, const Tensor& gradOutput,
                                  const Tensor& input, const Tensor& weights,
                                  const Tensor& bias,
                                  int sx, int sy, int px, int py, int dx,
                                  int dy, int groups,
                                  const ForwardBackwardState& state)
      -> std::pair<Tensor, Tensor> {
    return c->conv2dBackwardFilterBias(
        gradOutput, input, weights, bias, sx, sy, px, py, dx, dy, groups,
        state);
  };
  d.pool2dForward = [](TensorBackend* c, Tensor& out,
                       const Tensor& input,
                       int wx, int wy, int sx, int sy, int px, int py,
                       PoolingMode mode, bool wantState) -> ForwardBackwardState {
    return c->pool2dForward(
        out, input, wx, wy, sx, sy, px, py, mode, wantState);
  };
  d.pool2dBackward = [](TensorBackend* c, const Tensor& gradOutput, const Tensor& input,
                        const Tensor& poolOutput,
                        int wx, int wy, int sx, int sy, int px, int py,
                        PoolingMode mode,
                        const ForwardBackwardState& state) -> Tensor {
    return c->pool2dBackward(
        gradOutput, input, poolOutput, wx, wy, sx, sy, px, py, mode, state);
  };
  d.batchnormForward = [](TensorBackend* c,
                          const Tensor& input, const Tensor& weight,
                          const Tensor& bias,
                          const Tensor& runningMean, const Tensor& runningVar,
                          const std::vector<int>& axes, bool train,
                          double momentum, double epsilon, bool wantState)
      -> BatchnormForwardResult {
    return c->batchnormForward(
        input, weight, bias, runningMean, runningVar,
        axes, train, momentum, epsilon, wantState);
  };
  d.batchnormBackward = [](TensorBackend* c, const Tensor& gradOutput,
                           const Tensor& saveMean, const Tensor& saveVar,
                           const Tensor& input, const Tensor& weight,
                           const std::vector<int>& axes, bool train,
                           float epsilon,
                           const ForwardBackwardState& state)
      -> std::tuple<Tensor, Tensor, Tensor> {
    return c->batchnormBackward(
        gradOutput, saveMean, saveVar, input, weight,
        axes, train, epsilon, state);
  };
  d.rnnForward = [](TensorBackend* c, const Tensor& input, const Tensor& hiddenState,
                    const Tensor& cellState, const Tensor& weights,
                    int hiddenSize, int numLayers,
                    RnnMode mode, bool bidirectional, float dropout,
                    bool wantState)
      -> std::tuple<Tensor, Tensor, Tensor, ForwardBackwardState> {
    return c->rnnForward(
        input, hiddenState, cellState, weights,
        hiddenSize, numLayers, mode, bidirectional, dropout, wantState);
  };
  d.rnnBackward = [](TensorBackend* c, const Tensor& input, const Tensor& hiddenState,
                     const Tensor& cellState, const Tensor& weights,
                     const Tensor& dy, const Tensor& dhy, const Tensor& dcy,
                     const Tensor& output, int numLayers, int hiddenSize,
                     RnnMode mode, bool bidirectional, float dropProb,
                     const ForwardBackwardState& state)
      -> std::tuple<Tensor, Tensor, Tensor, Tensor> {
    return c->rnnBackward(
        input, hiddenState, cellState, weights,
        dy, dhy, dcy, output, numLayers, hiddenSize,
        mode, bidirectional, dropProb, state);
  };

  return d;
}

} // namespace fl
