/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "flashlight/fl/tensor/OpEnums.h"
#include "flashlight/fl/tensor/TensorBackend.h"

#include <af/array.h>

namespace fl {

/**
 * A tensor backend implementation of the ArrayFire tensor library.
 *
 * Given that ArrayFire has an internal DeviceManager singleton to manage its
 * global state, nothing is stored here as those internals are opaquely handled.
 * This class simply dispatches operations on global tensor functions to their
 * ArrayFire counterparts.
 */
class ArrayFireBackend : public TensorBackend {
  // TODO: consolidate the ArrayFire memory manager here so its global state can
  // be stored/we can reduce the number of singletons.
  std::once_flag memoryInitFlag;

  // These help ensure we are using native device id in public methods.
  std::unordered_map<int, int> nativeIdToId_;
  std::unordered_map<int, int> idToNativeId_;

  // keep track of the individual active stream on each ArrayFire device
  // NOTE using a `shared_ptr` to allow its capture in setActive callback;
  // see constructor for details.
  std::shared_ptr<std::unordered_map<int, std::shared_ptr<const Stream>>>
      afIdToStream_{std::make_shared<
          std::unordered_map<int, std::shared_ptr<const Stream>>>()};

  // Intentionally private. Only one instance should exist/it should be accessed
  // via getInstance().
  ArrayFireBackend();

 public:
  static ArrayFireBackend& getInstance();
  ~ArrayFireBackend() override = default;
  TensorBackendType backendType() const override;

  /**
   * The wrapped AF stream of the active device. Used as the default stream
   * of the JitBackend that wraps this backend.
   */
  std::shared_ptr<Stream> jitStream() const override;

  // No copy or move construction or assignment
  ArrayFireBackend(ArrayFireBackend&&) = delete;
  ArrayFireBackend(const ArrayFireBackend&) = delete;
  ArrayFireBackend& operator=(ArrayFireBackend&&) = delete;
  ArrayFireBackend& operator=(const ArrayFireBackend&) = delete;

  /* -------------------------- Compute Functions -------------------------- */
  void eval(const Tensor& tensor) override;

  /**
   * Return the stream from which the given array was created.
   *
   * @return an immutable reference to the stream from which `arr` was created.
   */
  const Stream& getStreamOfArray(const af::array& arr);
  bool supportsDataType(const fl::dtype& dtype) const override;
  // Memory management
  void getMemMgrInfo(const char* msg, const int nativeDeviceId, std::ostream* ostream)
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
  void pad(
      Tensor& out,
      const Tensor& input,
      const std::vector<std::pair<int, int>>& padWidths,
      const PadType type,
      std::shared_ptr<const Stream> stream = nullptr) override;

  /************************** Unary Operators ***************************/
  void unary(Tensor& out, UnaryOp op, const Tensor& input,
      std::shared_ptr<const Stream> stream = nullptr) override;

  Tensor exp(const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor log(const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor negative(const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor logicalNot(const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor log1p(const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor sin(const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor cos(const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor sqrt(const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor tanh(const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor floor(const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor ceil(const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor rint(const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor absolute(const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor sigmoid(const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor erf(const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr);
  void flip(Tensor& out, const Tensor& tensor, const unsigned dim,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void clip(Tensor& out, const Tensor& tensor, const Tensor& low, const Tensor& high,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void roll(Tensor& out, const Tensor& tensor, const int shift, const unsigned axis,
      std::shared_ptr<const Stream> stream = nullptr) override;
  Tensor isnan(const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor isinf(const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor sign(const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr);
  void tril(Tensor& out, const Tensor& tensor, int diagonal,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void triu(Tensor& out, const Tensor& tensor, int diagonal,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void where(Tensor& out, const Tensor& condition, const Tensor& x, const Tensor& y,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void topk(
      Tensor& values,
      Tensor& indices,
      const Tensor& input,
      const unsigned k,
      const Dim axis,
      const SortMode sortMode,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void sort(Tensor& out, const Tensor& input, const Dim axis, const SortMode sortMode,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void sort(
      Tensor& values,
      Tensor& indices,
      const Tensor& input,
      const Dim axis,
      const SortMode sortMode,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void argsort(Tensor& out, const Tensor& input, const Dim axis, const SortMode sortMode,
      std::shared_ptr<const Stream> stream = nullptr) override;

  /************************** Binary Operators ***************************/
  void binary(Tensor& out, BinaryOp op, const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr) override;

  Tensor add(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor sub(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor mul(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor div(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor eq(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor neq(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor lessThan(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor lessThanEqual(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor greaterThan(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor greaterThanEqual(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor logicalOr(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor logicalAnd(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor mod(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor bitwiseAnd(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor bitwiseOr(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor bitwiseXor(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor lShift(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor rShift(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor minimum(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor maximum(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor power(const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr);

  /******************************* BLAS ********************************/
  void matmul(
      Tensor& out,
      const Tensor& lhs,
      const Tensor& rhs,
      MatrixProperty lhsProp,
      MatrixProperty rhsProp,
      std::shared_ptr<const Stream> stream = nullptr) override;

  /************************** Reductions ***************************/
  void reduction(Tensor& out, ReductionOp op, const Tensor& input,
      const std::vector<int>& axes, bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) override;

  Tensor amin(const Tensor& input, const std::vector<int>& axes, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor amax(const Tensor& input, const std::vector<int>& axes, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr);
  void min(
      Tensor& values,
      Tensor& indices,
      const Tensor& input,
      const unsigned axis,
      const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void max(
      Tensor& values,
      Tensor& indices,
      const Tensor& input,
      const unsigned axis,
      const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) override;
  Tensor sum(const Tensor& input, const std::vector<int>& axes, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr);
  void cumsum(Tensor& out, const Tensor& input, const unsigned axis,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void argmax(Tensor& out, const Tensor& input, const unsigned axis, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void argmin(Tensor& out, const Tensor& input, const unsigned axis, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) override;
  Tensor mean(const Tensor& input, const std::vector<int>& axes, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor median(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr);
  void var(
      Tensor& out,
      const Tensor& input,
      const std::vector<int>& axes,
      const bool bias,
      const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) override;
  Tensor std(const Tensor& input, const std::vector<int>& axes, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr);
  void norm(
      Tensor& out,
      const Tensor& input,
      const std::vector<int>& axes,
      double p,
      const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) override;
  Tensor countNonzero(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor any(const Tensor& input, const std::vector<int>& axes, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor all(const Tensor& input, const std::vector<int>& axes, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr);

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

  /************************** Utils ***************************/
  void print(const Tensor& tensor) override;
};

} // namespace fl
