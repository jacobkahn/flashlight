/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/runtime/DeviceType.h"
#include "flashlight/fl/runtime/Stream.h"
#include "flashlight/fl/tensor/BackendDispatch.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/eval/Evaluator.h"
#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/Node.h"
#include "flashlight/fl/tensor/ir/ReductionNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/opt/Optimizer.h"

namespace fl {

/**
 * A JIT Tensor backend, mainly responsible for building up the JIT graph for
 * some specific backend.
 */
class JitBackend : public TensorBackend {
  BackendDispatch dispatch_;
  Evaluator evaluator_;
  Optimizer optimizer_;
  std::shared_ptr<Stream> defaultStream_;

  Tensor createJitTensor(NodePtr node);

  Tensor createBinopJitTensor(
      const Tensor& lhs, const Tensor& rhs, BinaryOp op,
      std::shared_ptr<const Stream> stream);
  Tensor createUnaryJitTensor(
      const Tensor& tensor, UnaryOp op,
      std::shared_ptr<const Stream> stream);
  Tensor createReductionJitTensor(
      const Tensor& input, ReductionOp op,
      const std::vector<int>& axes, bool keepDims,
      std::shared_ptr<const Stream> stream);

 public:
  JitBackend(
      TensorBackend& wrappedBackend,
      std::shared_ptr<Stream> defaultStream = nullptr);
  ~JitBackend() override = default;
  TensorBackendType backendType() const override;

  // No copy or move construction or assignment
  JitBackend(JitBackend&&) = delete;
  JitBackend(const JitBackend&) = delete;
  JitBackend& operator=(JitBackend&&) = delete;
  JitBackend& operator=(const JitBackend&) = delete;

  Evaluator& evaluator();
  Optimizer& optimizer();
  TensorBackend& wrappedBackend();
  const BackendDispatch& dispatch() const;
  const Stream& defaultStream() const;
  std::shared_ptr<const Stream> defaultStreamShared() const;

  /* -------------------------- Compute Functions -------------------------- */
  void eval(const Tensor& tensor) override;
  bool supportsDataType(const fl::dtype& dtype) const override;
  // Memory management
  void getMemMgrInfo(const char* msg, const int deviceId, std::ostream* ostream)
      override;
  void setMemMgrLogStream(std::ostream* stream) override;
  void setMemMgrLoggingEnabled(const bool enabled) override;
  void setMemMgrFlushInterval(const size_t interval) override;

  /* -------------------------- Rand Functions -------------------------- */
  void setSeed(const int seed) override;
  Tensor randn(const Shape& shape, dtype type,
      std::shared_ptr<const Stream> stream = nullptr) override;
  Tensor rand(const Shape& shape, dtype type,
      std::shared_ptr<const Stream> stream = nullptr) override;

  /* --------------------------- Tensor Operators --------------------------- */
  /******************** Tensor Creation Functions ********************/
  Tensor fromScalar(double value, dtype type,
      std::shared_ptr<const Stream> stream = nullptr) override;
  Tensor full(const Shape& dims, double value, dtype type,
      std::shared_ptr<const Stream> stream = nullptr) override;
  Tensor empty(const Shape& shape, dtype type,
      std::shared_ptr<const Stream> stream = nullptr) override;

  Tensor identity(const Dim dim, const dtype type,
      std::shared_ptr<const Stream> stream = nullptr) override;
  Tensor arange(const Shape& shape, const Dim seqDim, const dtype type,
      std::shared_ptr<const Stream> stream = nullptr) override;
  Tensor iota(const Shape& dims, const Shape& tileDims, const dtype type,
      std::shared_ptr<const Stream> stream = nullptr) override;

  /************************ Shaping and Indexing *************************/
  void reshape(Tensor& out, const Tensor& tensor, const Shape& shape,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void transpose(Tensor& out, const Tensor& tensor, const Shape& axes /* = {} */,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void tile(Tensor& out, const Tensor& tensor, const Shape& shape,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void concatenate(Tensor& out, const std::vector<Tensor>& tensors, const unsigned axis,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void nonzero(Tensor& out, const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void pad(Tensor& out, const Tensor& input,
      const std::vector<std::pair<int, int>>& padWidths, const PadType type,
      std::shared_ptr<const Stream> stream = nullptr) override;

  /************************** Unary Operators ***************************/
  void unary(Tensor& out, UnaryOp op, const Tensor& input,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void flip(Tensor& out, const Tensor& tensor, const unsigned dim,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void clip(Tensor& out, const Tensor& tensor, const Tensor& low, const Tensor& high,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void roll(Tensor& out, const Tensor& tensor, const int shift, const unsigned axis,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void tril(Tensor& out, const Tensor& tensor, int diagonal,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void triu(Tensor& out, const Tensor& tensor, int diagonal,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void where(Tensor& out, const Tensor& condition, const Tensor& x, const Tensor& y,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void topk(
      Tensor& values, Tensor& indices, const Tensor& input,
      const unsigned k, const Dim axis, const SortMode sortMode,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void sort(Tensor& out, const Tensor& input, const Dim axis, const SortMode sortMode,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void sort(
      Tensor& values, Tensor& indices, const Tensor& input,
      const Dim axis, const SortMode sortMode,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void argsort(Tensor& out, const Tensor& input, const Dim axis, const SortMode sortMode,
      std::shared_ptr<const Stream> stream = nullptr) override;

  /************************** Binary Operators ***************************/
  void binary(Tensor& out, BinaryOp op, const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr) override;

  /******************************* BLAS ********************************/
  void matmul(
      Tensor& out, const Tensor& lhs, const Tensor& rhs,
      MatrixProperty lhsProp, MatrixProperty rhsProp,
      std::shared_ptr<const Stream> stream = nullptr) override;

  /************************** Reductions ***************************/
  void reduction(Tensor& out, ReductionOp op, const Tensor& input,
      const std::vector<int>& axes, bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void min(
      Tensor& values, Tensor& indices, const Tensor& input,
      const unsigned axis, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void max(
      Tensor& values, Tensor& indices, const Tensor& input,
      const unsigned axis, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void cumsum(Tensor& out, const Tensor& input, const unsigned axis,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void argmax(Tensor& out, const Tensor& input, const unsigned axis, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void argmin(Tensor& out, const Tensor& input, const unsigned axis, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void var(Tensor& out, const Tensor& input, const std::vector<int>& axes,
      const bool bias, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void norm(Tensor& out, const Tensor& input, const std::vector<int>& axes,
      double p, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) override;

  /*********************** Autograd Ops ************************/
  ForwardBackwardState conv2dForward(
      Tensor& out,
      const Tensor& input, const Tensor& weights, const Tensor& bias,
      int sx, int sy, int px, int py, int dx, int dy, int groups,
      bool wantState) override;
  Tensor conv2dBackwardData(
      const Tensor& gradOutput, const Tensor& input, const Tensor& weight,
      int sx, int sy, int px, int py, int dx, int dy, int groups,
      const ForwardBackwardState& state) override;
  std::pair<Tensor, Tensor> conv2dBackwardFilterBias(
      const Tensor& gradOutput, const Tensor& input,
      const Tensor& weights, const Tensor& bias,
      int sx, int sy, int px, int py, int dx, int dy, int groups,
      const ForwardBackwardState& state) override;

  ForwardBackwardState pool2dForward(
      Tensor& out,
      const Tensor& input, int wx, int wy, int sx, int sy, int px, int py,
      PoolingMode mode, bool wantState) override;
  Tensor pool2dBackward(
      const Tensor& gradOutput, const Tensor& input, const Tensor& poolOutput,
      int wx, int wy, int sx, int sy, int px, int py,
      PoolingMode mode, const ForwardBackwardState& state) override;

  BatchnormForwardResult batchnormForward(
      const Tensor& input, const Tensor& weight, const Tensor& bias,
      const Tensor& runningMean, const Tensor& runningVar,
      const std::vector<int>& axes, bool train, double momentum, double epsilon,
      bool wantState) override;
  std::tuple<Tensor, Tensor, Tensor> batchnormBackward(
      const Tensor& gradOutput, const Tensor& saveMean, const Tensor& saveVar,
      const Tensor& input, const Tensor& weight,
      const std::vector<int>& axes, bool train, float epsilon,
      const ForwardBackwardState& state) override;

  std::tuple<Tensor, Tensor, Tensor, ForwardBackwardState> rnnForward(
      const Tensor& input, const Tensor& hiddenState, const Tensor& cellState,
      const Tensor& weights, int hiddenSize, int numLayers,
      RnnMode mode, bool bidirectional, float dropout,
      bool wantState) override;
  std::tuple<Tensor, Tensor, Tensor, Tensor> rnnBackward(
      const Tensor& input, const Tensor& hiddenState, const Tensor& cellState,
      const Tensor& weights,
      const Tensor& dy, const Tensor& dhy, const Tensor& dcy,
      const Tensor& output, int numLayers, int hiddenSize,
      RnnMode mode, bool bidirectional, float dropProb,
      const ForwardBackwardState& state) override;

  /************************** Data Transfer ***************************/
  void copy(Tensor& dst, const Tensor& src, const Stream& stream) override;

  /// Lazy transfer: creates a TransferNode (no materialization).
  Tensor transfer(
      const Tensor& src,
      DeviceType dstDeviceType,
      TensorBackendType dstBackendType,
      std::shared_ptr<const Stream> dstStream);

  /************************** Utils ***************************/
  void print(const Tensor& tensor) override;
};

} // namespace fl
