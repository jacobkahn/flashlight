/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorBackend.h"

#include <memory>
#include <optional>

#include "flashlight/fl/tensor/ops/onednn/OneDnnAsyncCPUStream.h"

#if FL_USE_MKL_RNG
  #include <mkl_vsl.h>
#else
  #include <random>
#endif // FL_USE_MKL_RNG

namespace fl {

class CPUTensor;
class OneDnnOps;

/**
 * A tensor backend implementation using the OneDNN library.
 */
class CPUBackend : public TensorBackend {
  std::unique_ptr<OneDnnOps> onednnOps_;
  std::shared_ptr<OneDnnAsyncCPUStream> asyncStream_;
#if FL_USE_MKL_RNG
  VSLStreamStatePtr randStream_;
#else
  using RandEngineType = std::mt19937;
  RandEngineType randEngine_;
#endif // FL_USE_MKL_RNG

  Tensor randnCpu(const Shape& shape, dtype type);
  Tensor randCpu(const Shape& shape, dtype type);

  template <typename T, typename V>
  Tensor fullWithType(const Shape& shape, V value, const dtype type);

 public:
  CPUBackend();

  static CPUBackend& getInstance();
  ~CPUBackend() override;
  TensorBackendType backendType() const override;
  BackendDispatch createDispatch() override;

  // No copy or move construction or assignment
  CPUBackend(CPUBackend&&) = delete;
  CPUBackend(const CPUBackend&) = delete;
  CPUBackend& operator=(CPUBackend&&) = delete;
  CPUBackend& operator=(const CPUBackend&) = delete;

  /**
   * Gets the backend's default AsyncCPUStream.
   *
   * @return the default AsyncCPUStream.
   */
  const Stream& stream() const;

  /**
   * Gets the backend's default AsyncCPUStream as a shared_ptr.
   *
   * @return the default AsyncCPUStream.
   */
  std::shared_ptr<AsyncCPUStream> asyncStream() const;
  std::shared_ptr<Stream> jitStream() const override;

  /**
   * Gets the OneDnnOps instance that owns the OneDNN engine.
   */
  OneDnnOps& onednnOps();
  const OneDnnOps& onednnOps() const;

  /**
   * Gets the dnnl::stream associated with the default async stream.
   */
  dnnl::stream& nativeStream();

  // ---- In-place forwarding (routes through onednnOps + nativeStream) ----
  void inPlaceAdd(CPUTensor& self, double val);
  void inPlaceSub(CPUTensor& self, double val);
  void inPlaceMul(CPUTensor& self, double val);
  void inPlaceDiv(CPUTensor& self, double val);
  void inPlaceAdd(CPUTensor& self, const Tensor& other);
  void inPlaceSub(CPUTensor& self, const Tensor& other);
  void inPlaceMul(CPUTensor& self, const Tensor& other);
  void inPlaceDiv(CPUTensor& self, const Tensor& other);
  void fillInPlace(CPUTensor& self, double val);
  Tensor reorderAstype(CPUTensor& self, fl::dtype type);
  void assignTensor(CPUTensor& self, const Tensor& src);

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
  void pad(
      Tensor& out,
      const Tensor& input,
      const std::vector<std::pair<int, int>>& padWidths,
      const PadType type,
      std::shared_ptr<const Stream> stream = nullptr) override;

  /************************ Enum-Dispatched Ops ************************/
  void binary(Tensor& out, BinaryOp op, const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void unary(Tensor& out, UnaryOp op, const Tensor& input,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void reduction(Tensor& out, ReductionOp op, const Tensor& input,
      const std::vector<int>& axes, bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) override;

  /************************** Unary Operators ***************************/
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
  void clip(Tensor& out, const Tensor& tensor, const double& low, const double& high,
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
  Tensor amin(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor amax(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims,
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
  Tensor sum(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr);
  void cumsum(Tensor& out, const Tensor& input, const unsigned axis,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void argmax(Tensor& out, const Tensor& input, const unsigned axis, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void argmin(Tensor& out, const Tensor& input, const unsigned axis, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) override;
  Tensor mean(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims,
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
  Tensor std(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims,
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
  Tensor any(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr);
  Tensor all(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr);

  /************************** Data Transfer ***************************/
  void copy(Tensor& dst, const Tensor& src, const Stream& stream) override;

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

  /************************** Utils ***************************/
  void print(const Tensor& tensor) override;
};

} // namespace fl
