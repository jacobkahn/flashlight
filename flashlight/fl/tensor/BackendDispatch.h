/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <ostream>
#include <tuple>
#include <utility>
#include <vector>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/tensor/OpEnums.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl {

class TensorBackend;

/**
 * A lightweight struct of function pointers for dispatching tensor operations.
 *
 * Used by the JIT Evaluator instead of virtual dispatch through TensorBackend.
 * The default trampoline (makeBackendDispatch) forwards through TensorBackend
 * virtuals; backend-specific overrides can point directly to free functions.
 */
struct BackendDispatch {
  TensorBackend* ctx;

  // The backend type (e.g. ArrayFire, CPU, Cuda).
  TensorBackendType backendType;

  // --- Evaluator hot path (7) ---
  void (*binary)(TensorBackend* ctx, Tensor& out, BinaryOp op,
                 const Tensor& lhs, const Tensor& rhs);
  void (*unary)(TensorBackend* ctx, Tensor& out, UnaryOp op,
                const Tensor& input);
  void (*reduction)(TensorBackend* ctx, Tensor& out, ReductionOp op,
                    const Tensor& input,
                    const std::vector<int>& axes, bool keepDims);
  void (*reshape)(TensorBackend* ctx, Tensor& out,
                  const Tensor& input, const Shape& shape);
  void (*transpose)(TensorBackend* ctx, Tensor& out,
                    const Tensor& input, const Shape& axes);
  void (*matmul)(TensorBackend* ctx, Tensor& out,
                 const Tensor& lhs, const Tensor& rhs,
                 MatrixProperty lhsProp, MatrixProperty rhsProp);
  Tensor (*full)(TensorBackend* ctx, const Shape& shape,
                 double value, dtype type);
  Tensor (*empty)(TensorBackend* ctx, const Shape& shape, dtype type);

  // --- Creation/random (5) ---
  Tensor (*randn)(TensorBackend* ctx, const Shape& shape, dtype type);
  Tensor (*rand)(TensorBackend* ctx, const Shape& shape, dtype type);
  Tensor (*identity)(TensorBackend* ctx, Dim dim, dtype type);
  Tensor (*arange)(TensorBackend* ctx, const Shape& shape, Dim seqDim, dtype type);
  Tensor (*iota)(TensorBackend* ctx, const Shape& dims,
                 const Shape& tileDims, dtype type);

  // --- Admin/lifecycle (8) ---
  void (*eval)(TensorBackend* ctx, const Tensor& tensor);
  bool (*supportsDataType)(TensorBackend* ctx, const fl::dtype& dtype);
  void (*setSeed)(TensorBackend* ctx, int seed);
  void (*getMemMgrInfo)(TensorBackend* ctx, const char* msg, int deviceId,
                        std::ostream* ostream);
  void (*setMemMgrLogStream)(TensorBackend* ctx, std::ostream* stream);
  void (*setMemMgrLoggingEnabled)(TensorBackend* ctx, bool enabled);
  void (*setMemMgrFlushInterval)(TensorBackend* ctx, size_t interval);
  void (*print)(TensorBackend* ctx, const Tensor& tensor);

  // --- Shaping/indexing (4) ---
  void (*tile)(TensorBackend* ctx, Tensor& out, const Tensor& tensor,
               const Shape& shape);
  void (*concatenate)(TensorBackend* ctx, Tensor& out,
                      const std::vector<Tensor>& tensors, unsigned axis);
  void (*nonzero)(TensorBackend* ctx, Tensor& out, const Tensor& tensor);
  void (*pad)(TensorBackend* ctx, Tensor& out, const Tensor& input,
              const std::vector<std::pair<int, int>>& padWidths, PadType type);

  // --- Named unary/misc (8) ---
  void (*flip)(TensorBackend* ctx, Tensor& out, const Tensor& tensor, unsigned dim);
  void (*clip)(TensorBackend* ctx, Tensor& out, const Tensor& tensor,
               const Tensor& low, const Tensor& high);
  void (*roll)(TensorBackend* ctx, Tensor& out, const Tensor& tensor,
               int shift, unsigned axis);
  void (*tril)(
      TensorBackend* ctx, Tensor& out, const Tensor& tensor, int diagonal);
  void (*triu)(
      TensorBackend* ctx, Tensor& out, const Tensor& tensor, int diagonal);
  void (*where)(TensorBackend* ctx, Tensor& out, const Tensor& condition,
                const Tensor& x, const Tensor& y);
  void (*sort)(TensorBackend* ctx, Tensor& out, const Tensor& input,
               Dim axis, SortMode sortMode);
  void (*argsort)(TensorBackend* ctx, Tensor& out, const Tensor& input,
                  Dim axis, SortMode sortMode);

  // --- 2-output ops (3) ---
  void (*topk)(TensorBackend* ctx, Tensor& values, Tensor& indices,
               const Tensor& input, unsigned k, Dim axis, SortMode sortMode);
  void (*sortWithIndices)(TensorBackend* ctx, Tensor& values, Tensor& indices,
                         const Tensor& input, Dim axis, SortMode sortMode);

  // --- Named reductions (7) ---
  void (*min)(TensorBackend* ctx, Tensor& values, Tensor& indices,
              const Tensor& input, unsigned axis, bool keepDims);
  void (*max)(TensorBackend* ctx, Tensor& values, Tensor& indices,
              const Tensor& input, unsigned axis, bool keepDims);
  void (*cumsum)(TensorBackend* ctx, Tensor& out, const Tensor& input, unsigned axis);
  void (*argmax)(TensorBackend* ctx, Tensor& out, const Tensor& input,
                 unsigned axis, bool keepDims);
  void (*argmin)(TensorBackend* ctx, Tensor& out, const Tensor& input,
                 unsigned axis, bool keepDims);
  void (*var)(TensorBackend* ctx, Tensor& out, const Tensor& input,
              const std::vector<int>& axes, bool bias, bool keepDims);
  void (*norm)(TensorBackend* ctx, Tensor& out, const Tensor& input,
               const std::vector<int>& axes, double p, bool keepDims);

  // --- Autograd (9) ---
  ForwardBackwardState (*conv2dForward)(
      TensorBackend* ctx, Tensor& out,
      const Tensor& input, const Tensor& weights, const Tensor& bias,
      int sx, int sy, int px, int py, int dx, int dy, int groups,
      bool wantState);
  Tensor (*conv2dBackwardData)(
      TensorBackend* ctx, const Tensor& gradOutput, const Tensor& input,
      const Tensor& weight,
      int sx, int sy, int px, int py, int dx, int dy, int groups,
      const ForwardBackwardState& state);
  std::pair<Tensor, Tensor> (*conv2dBackwardFilterBias)(
      TensorBackend* ctx, const Tensor& gradOutput, const Tensor& input,
      const Tensor& weights, const Tensor& bias,
      int sx, int sy, int px, int py, int dx, int dy, int groups,
      const ForwardBackwardState& state);
  ForwardBackwardState (*pool2dForward)(
      TensorBackend* ctx, Tensor& out,
      const Tensor& input,
      int wx, int wy, int sx, int sy, int px, int py,
      PoolingMode mode, bool wantState);
  Tensor (*pool2dBackward)(
      TensorBackend* ctx, const Tensor& gradOutput, const Tensor& input,
      const Tensor& poolOutput,
      int wx, int wy, int sx, int sy, int px, int py,
      PoolingMode mode, const ForwardBackwardState& state);
  BatchnormForwardResult (*batchnormForward)(
      TensorBackend* ctx,
      const Tensor& input, const Tensor& weight, const Tensor& bias,
      const Tensor& runningMean, const Tensor& runningVar,
      const std::vector<int>& axes, bool train, double momentum, double epsilon,
      bool wantState);
  std::tuple<Tensor, Tensor, Tensor> (*batchnormBackward)(
      TensorBackend* ctx, const Tensor& gradOutput, const Tensor& saveMean,
      const Tensor& saveVar,
      const Tensor& input, const Tensor& weight,
      const std::vector<int>& axes, bool train, float epsilon,
      const ForwardBackwardState& state);
  std::tuple<Tensor, Tensor, Tensor, ForwardBackwardState> (*rnnForward)(
      TensorBackend* ctx, const Tensor& input, const Tensor& hiddenState,
      const Tensor& cellState, const Tensor& weights,
      int hiddenSize, int numLayers,
      RnnMode mode, bool bidirectional, float dropout, bool wantState);
  std::tuple<Tensor, Tensor, Tensor, Tensor> (*rnnBackward)(
      TensorBackend* ctx, const Tensor& input, const Tensor& hiddenState,
      const Tensor& cellState, const Tensor& weights,
      const Tensor& dy, const Tensor& dhy, const Tensor& dcy,
      const Tensor& output, int numLayers, int hiddenSize,
      RnnMode mode, bool bidirectional, float dropProb,
      const ForwardBackwardState& state);
};

/**
 * Creates a default dispatch that trampolines through TensorBackend virtuals.
 */
BackendDispatch makeBackendDispatch(TensorBackend& backend);

} // namespace fl
