/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/fl/tensor/TensorBackend.h"

namespace fl {

/**
 * A stub Tensor backend implementation to make it easy to get started with the
 * Flashlight Tensor API.
 *
 * This stub can be copied, renamed, and implemented as needed.
 */
class StubBackend : public TensorBackend {
 public:
  StubBackend();

  static StubBackend& getInstance();
  ~StubBackend() override = default;
  TensorBackendType backendType() const override;

  // No copy or move construction or assignment
  StubBackend(StubBackend&&) = delete;
  StubBackend(const StubBackend&) = delete;
  StubBackend& operator=(StubBackend&&) = delete;
  StubBackend& operator=(const StubBackend&) = delete;

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
  void concatenate(Tensor& out, const std::vector<Tensor>& tensors,
      const unsigned axis,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void nonzero(Tensor& out, const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void pad(Tensor& out,
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
  void clip(Tensor& out, const Tensor& tensor, const Tensor& low,
      const Tensor& high,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void roll(Tensor& out, const Tensor& tensor, const int shift,
      const unsigned axis,
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
  void where(Tensor& out, const Tensor& condition, const Tensor& x,
      const Tensor& y,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void topk(
      Tensor& values,
      Tensor& indices,
      const Tensor& input,
      const unsigned k,
      const Dim axis,
      const SortMode sortMode,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void sort(Tensor& out, const Tensor& input, const Dim axis,
      const SortMode sortMode,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void sort(
      Tensor& values,
      Tensor& indices,
      const Tensor& input,
      const Dim axis,
      const SortMode sortMode,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void argsort(Tensor& out, const Tensor& input, const Dim axis,
      const SortMode sortMode,
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
  void matmul(Tensor& out,
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
  void argmax(Tensor& out, const Tensor& input, const unsigned axis,
      const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) override;
  void argmin(Tensor& out, const Tensor& input, const unsigned axis,
      const bool keepDims,
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
  void var(Tensor& out,
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
  void norm(Tensor& out,
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

  /************************** Utils ***************************/
  void print(const Tensor& tensor) override;
};

} // namespace fl
