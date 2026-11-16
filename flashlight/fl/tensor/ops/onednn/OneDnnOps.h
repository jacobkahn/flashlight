/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <dnnl.hpp>

#include "flashlight/fl/tensor/Tensor.h"

namespace fl {

class AsyncCPUStream;
class CPUTensor;

// Stateless free functions — business logic parameterized by engine/stream.
// These live in detail::onednn so they can be independently tested and reused.
namespace detail::onednn {

// Core ops -- write into pre-assigned out buffer when non-empty,
// otherwise allocate a new tensor.
void applyBinop(
    Tensor& out,
    const Tensor& lhs,
    const Tensor& rhs,
    dnnl::algorithm alg,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream,
    std::optional<dnnl::memory::data_type> dstType = std::nullopt);

void applyEltwiseOp(
    Tensor& out,
    const Tensor& tensor,
    dnnl::algorithm alg,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream,
    float alpha = 0,
    float beta = 0);

void applyReductionOp(
    Tensor& out,
    const Tensor& tensor,
    dnnl::algorithm alg,
    const std::vector<int>& axes,
    bool keepDims,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream,
    float p = 0.f,
    float eps = 0.f);

// Shaping ops
Tensor reshape(
    const Tensor& tensor,
    const Shape& shape,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream);

Tensor transpose(
    const Tensor& tensor,
    const Shape& axes,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream);

Tensor tile(
    const Tensor& tensor,
    const Shape& tileDims,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream);

Tensor concatenate(
    const std::vector<Tensor>& tensors,
    unsigned axis,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream);

// BLAS -- writes into pre-assigned out buffer when non-empty, else allocates.
void matmul(
    Tensor& out,
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream,
    const dnnl::primitive_attr& attr = dnnl::primitive_attr(),
    const Tensor& bias = Tensor());

// Conditional
Tensor where(
    const Tensor& condition,
    const Tensor& x,
    const Tensor& y,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream,
    std::shared_ptr<AsyncCPUStream> asyncStream);

// In-place ops (used by CPUTensor)
void inPlaceBinaryScalar(
    CPUTensor& self,
    double val,
    dnnl::algorithm alg,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream);

void inPlaceBinaryTensor(
    CPUTensor& self,
    const Tensor& other,
    dnnl::algorithm alg,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream);

void fillInPlace(
    CPUTensor& self,
    double val,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream);

// Type conversion. Returns empty Tensor if OneDNN reorder fails (caller
// should fall back to CPU conversion).
Tensor reorderAstype(
    CPUTensor& self,
    fl::dtype type,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream);

// Tensor assignment
void assignTensor(
    CPUTensor& self,
    const Tensor& src,
    const dnnl::engine& engine,
    dnnl::stream& nativeStream);

} // namespace detail::onednn

/**
 * Stateful wrapper — owns the dnnl::engine, delegates to detail::onednn
 * free functions. The dnnl::stream is owned by OneDnnAsyncCPUStream,
 * allowing each worker thread to have its own stream for multi-stream
 * parallelism. All methods take a dnnl::stream& parameter.
 */
class OneDnnOps {
  dnnl::engine engine_;

 public:
  OneDnnOps();
  ~OneDnnOps();

  const dnnl::engine& engine() const { return engine_; }

  // ---- Core ops (write into out buffer when non-empty, else allocate) ----
  void applyBinop(
      Tensor& out,
      const Tensor& lhs,
      const Tensor& rhs,
      dnnl::algorithm alg,
      dnnl::stream& nativeStream,
      std::shared_ptr<AsyncCPUStream> asyncStream,
      std::optional<dnnl::memory::data_type> dstType = std::nullopt);

  void applyEltwiseOp(
      Tensor& out,
      const Tensor& tensor,
      dnnl::algorithm alg,
      dnnl::stream& nativeStream,
      std::shared_ptr<AsyncCPUStream> asyncStream,
      float alpha = 0,
      float beta = 0);

  void applyReductionOp(
      Tensor& out,
      const Tensor& tensor,
      dnnl::algorithm alg,
      const std::vector<int>& axes,
      bool keepDims,
      dnnl::stream& nativeStream,
      std::shared_ptr<AsyncCPUStream> asyncStream,
      float p = 0.f,
      float eps = 0.f);

  // ---- Shaping ops ----
  Tensor reshape(
      const Tensor& tensor,
      const Shape& shape,
      dnnl::stream& nativeStream,
      std::shared_ptr<AsyncCPUStream> asyncStream);

  Tensor transpose(
      const Tensor& tensor,
      const Shape& axes,
      dnnl::stream& nativeStream,
      std::shared_ptr<AsyncCPUStream> asyncStream);

  Tensor tile(
      const Tensor& tensor,
      const Shape& tileDims,
      dnnl::stream& nativeStream,
      std::shared_ptr<AsyncCPUStream> asyncStream);

  Tensor concatenate(
      const std::vector<Tensor>& tensors,
      unsigned axis,
      dnnl::stream& nativeStream,
      std::shared_ptr<AsyncCPUStream> asyncStream);

  // ---- BLAS ----
  void matmul(
      Tensor& out,
      const Tensor& lhs,
      const Tensor& rhs,
      MatrixProperty lhsProp,
      MatrixProperty rhsProp,
      dnnl::stream& nativeStream,
      std::shared_ptr<AsyncCPUStream> asyncStream,
      const dnnl::primitive_attr& attr = dnnl::primitive_attr(),
      const Tensor& bias = Tensor());

  // ---- Conditional ----
  Tensor where(
      const Tensor& condition,
      const Tensor& x,
      const Tensor& y,
      dnnl::stream& nativeStream,
      std::shared_ptr<AsyncCPUStream> asyncStream);

  // ---- In-place ops (named wrappers so callers don't need dnnl::algorithm) ----
  void inPlaceAdd(CPUTensor& self, double val, dnnl::stream& nativeStream);
  void inPlaceSub(CPUTensor& self, double val, dnnl::stream& nativeStream);
  void inPlaceMul(CPUTensor& self, double val, dnnl::stream& nativeStream);
  void inPlaceDiv(CPUTensor& self, double val, dnnl::stream& nativeStream);

  void inPlaceAdd(CPUTensor& self, const Tensor& other, dnnl::stream& nativeStream);
  void inPlaceSub(CPUTensor& self, const Tensor& other, dnnl::stream& nativeStream);
  void inPlaceMul(CPUTensor& self, const Tensor& other, dnnl::stream& nativeStream);
  void inPlaceDiv(CPUTensor& self, const Tensor& other, dnnl::stream& nativeStream);

  void fillInPlace(CPUTensor& self, double val, dnnl::stream& nativeStream);

  // ---- Type conversion ----
  Tensor reorderAstype(CPUTensor& self, fl::dtype type, dnnl::stream& nativeStream);

  // ---- Assignment ----
  void assignTensor(CPUTensor& self, const Tensor& src, dnnl::stream& nativeStream);
};

} // namespace fl
