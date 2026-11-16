/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <ostream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/tensor/OpEnums.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/TensorExtension.h"

namespace fl {

class Stream;
struct BackendDispatch;

/**
 * Base class for opaque forward-to-backward state.
 * Backends define concrete subclasses (e.g. OneDnnPool2DState, CudnnRnnState).
 */
struct ForwardState {
  virtual ~ForwardState() = default;
};
using ForwardBackwardState = std::shared_ptr<ForwardState>;

/**
 * Result of a batchnorm forward pass.
 */
struct BatchnormForwardResult {
  Tensor output;
  Tensor saveMean;
  Tensor saveVar;
  Tensor newRunningMean;  // empty if not tracking stats
  Tensor newRunningVar;   // empty if not tracking stats
  ForwardBackwardState state;
};

/**
 * A Tensor backend that can be used to store global state associated with a
 * particular tensor implementation.
 *
 * This abstraction facilitates adherence to the implementation requirements for
 * global operators that operate on tensors (e.g. those functions that are not
 * members of `fl::Tensor`).
 *
 * Flashlight Tensors dispatch to their corresponding backends using
 * fl::Tensor::backend() --> typeToBackend (see below) to grab the correct
 * instance.
 */
class TensorBackend {
 public:
  TensorBackend() = default;
  virtual ~TensorBackend() = default;
  virtual TensorBackendType backendType() const = 0;

  /**
   * Creates a BackendDispatch for this backend.  The default implementation
   * returns a trampoline that dispatches through TensorBackend virtuals.
   * Concrete backends override to return a direct-dispatch table that uses
   * qualified (non-virtual) calls for better performance.
   */
  virtual BackendDispatch createDispatch();

  /**
   * Returns the backend's async stream for JIT evaluation, if any.
   * Used by JIT infrastructure to create JitBackend instances.
   * Returns nullptr by default.
   */
  virtual std::shared_ptr<Stream> jitStream() const { return nullptr; }

  /* -------------------------- Compute Functions -------------------------- */
  virtual void eval(const Tensor& tensor) = 0;
  virtual bool supportsDataType(const fl::dtype& dtype) const = 0;
  // Memory Management
  virtual void
  getMemMgrInfo(const char* msg, const int deviceId, std::ostream* ostream) = 0;
  virtual void setMemMgrLogStream(std::ostream* stream) = 0;
  virtual void setMemMgrLoggingEnabled(const bool enabled) = 0;
  virtual void setMemMgrFlushInterval(const size_t interval) = 0;

  /* -------------------------- Rand Functions -------------------------- */
  virtual void setSeed(const int seed) = 0;
  virtual Tensor randn(const Shape& shape, dtype type,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual Tensor rand(const Shape& shape, dtype type,
      std::shared_ptr<const Stream> stream = nullptr) = 0;

  /* --------------------------- Tensor Operators ---------------------------
   * For operator documentation and expected behavior, see Tensor.h.
   */
  /******************** Tensor Creation Functions ********************/
  virtual Tensor fromScalar(double value, dtype type,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual Tensor full(const Shape& dims, double value, dtype type,
      std::shared_ptr<const Stream> stream = nullptr) = 0;

  virtual Tensor empty(const Shape& shape, dtype type,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual Tensor identity(const Dim dim, const dtype type,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual Tensor arange(const Shape& shape, const Dim seqDim, const dtype type,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual Tensor iota(const Shape& dims, const Shape& tileDims, const dtype type,
      std::shared_ptr<const Stream> stream = nullptr) = 0;

  /************************ Shaping and Indexing *************************/
  virtual void reshape(Tensor& out, const Tensor& tensor, const Shape& shape,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void transpose(Tensor& out, const Tensor& tensor, const Shape& axes /* = {} */,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void tile(Tensor& out, const Tensor& tensor, const Shape& shape,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void concatenate(Tensor& out, const std::vector<Tensor>& tensors,
      const unsigned axis,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void nonzero(Tensor& out, const Tensor& tensor,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void pad(Tensor& out, const Tensor& input,
      const std::vector<std::pair<int, int>>& padWidths, const PadType type,
      std::shared_ptr<const Stream> stream = nullptr) = 0;

  /************************** Unary Operators ***************************/
  virtual void flip(Tensor& out, const Tensor& tensor, const unsigned dim,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void clip(Tensor& out, const Tensor& tensor, const Tensor& low,
      const Tensor& high,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void clip(Tensor& out, const Tensor& tensor, const Tensor& low,
      const double& high,
      std::shared_ptr<const Stream> stream = nullptr);
  virtual void clip(Tensor& out, const Tensor& tensor, const double& low,
      const Tensor& high,
      std::shared_ptr<const Stream> stream = nullptr);
  virtual void clip(Tensor& out, const Tensor& tensor, const double& low,
      const double& high,
      std::shared_ptr<const Stream> stream = nullptr);
  virtual void roll(Tensor& out, const Tensor& tensor, const int shift,
      const unsigned axis,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void tril(Tensor& out, const Tensor& tensor, int diagonal,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void triu(Tensor& out, const Tensor& tensor, int diagonal,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void where(Tensor& out, const Tensor& condition, const Tensor& x,
      const Tensor& y,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void where(Tensor& out, const Tensor& condition, const Tensor& x,
      const double& y,
      std::shared_ptr<const Stream> stream = nullptr);
  virtual void where(Tensor& out, const Tensor& condition, const double& x,
      const Tensor& y,
      std::shared_ptr<const Stream> stream = nullptr);
  virtual void topk(
      Tensor& values, Tensor& indices, const Tensor& input,
      const unsigned k, const Dim axis, const SortMode sortMode,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void sort(Tensor& out, const Tensor& input, const Dim axis,
      const SortMode sortMode,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void sort(
      Tensor& values, Tensor& indices, const Tensor& input,
      const Dim axis, const SortMode sortMode,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void argsort(Tensor& out, const Tensor& input, const Dim axis,
      const SortMode sortMode,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void indexAdd(
      Tensor& self, unsigned axis, const Tensor& indices, const Tensor& src,
      std::shared_ptr<const Stream> stream = nullptr);

  /************************ Enum-Dispatched Ops ************************/
  virtual void binary(Tensor& out, BinaryOp op, const Tensor& lhs, const Tensor& rhs,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void unary(Tensor& out, UnaryOp op, const Tensor& input,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void reduction(Tensor& out, ReductionOp op, const Tensor& input,
      const std::vector<int>& axes, bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) = 0;

  /******************************* BLAS ********************************/
  virtual void matmul(Tensor& out,
      const Tensor& lhs, const Tensor& rhs,
      MatrixProperty lhsProp, MatrixProperty rhsProp,
      std::shared_ptr<const Stream> stream = nullptr) = 0;

  /************************** Reductions ***************************/
  virtual void min(
      Tensor& values, Tensor& indices, const Tensor& input,
      const unsigned axis, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void max(
      Tensor& values, Tensor& indices, const Tensor& input,
      const unsigned axis, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void cumsum(Tensor& out, const Tensor& input, const unsigned axis,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void argmax(Tensor& out, const Tensor& input, const unsigned axis,
      const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void argmin(Tensor& out, const Tensor& input, const unsigned axis,
      const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void var(Tensor& out, const Tensor& input, const std::vector<int>& axes,
      bool bias, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) = 0;
  virtual void norm(Tensor& out, const Tensor& input, const std::vector<int>& axes,
      double p, const bool keepDims,
      std::shared_ptr<const Stream> stream = nullptr) = 0;

  /*********************** Autograd Ops ************************/
  // -- conv2d --
  virtual ForwardBackwardState conv2dForward(
      Tensor& out,
      const Tensor& input, const Tensor& weights, const Tensor& bias,
      int sx, int sy, int px, int py, int dx, int dy, int groups,
      bool wantState = false);
  virtual Tensor conv2dBackwardData(
      const Tensor& gradOutput, const Tensor& input, const Tensor& weight,
      int sx, int sy, int px, int py, int dx, int dy, int groups,
      const ForwardBackwardState& state);
  virtual std::pair<Tensor, Tensor> conv2dBackwardFilterBias(
      const Tensor& gradOutput, const Tensor& input,
      const Tensor& weights, const Tensor& bias,
      int sx, int sy, int px, int py, int dx, int dy, int groups,
      const ForwardBackwardState& state);

  // -- pool2d --
  virtual ForwardBackwardState pool2dForward(
      Tensor& out,
      const Tensor& input, int wx, int wy, int sx, int sy, int px, int py,
      PoolingMode mode, bool wantState = false);
  virtual Tensor pool2dBackward(
      const Tensor& gradOutput, const Tensor& input, const Tensor& poolOutput,
      int wx, int wy, int sx, int sy, int px, int py,
      PoolingMode mode, const ForwardBackwardState& state);

  // -- batchnorm --
  virtual BatchnormForwardResult batchnormForward(
      const Tensor& input, const Tensor& weight, const Tensor& bias,
      const Tensor& runningMean, const Tensor& runningVar,
      const std::vector<int>& axes, bool train, double momentum, double epsilon,
      bool wantState = false);
  virtual std::tuple<Tensor, Tensor, Tensor> batchnormBackward(
      const Tensor& gradOutput, const Tensor& saveMean, const Tensor& saveVar,
      const Tensor& input, const Tensor& weight,
      const std::vector<int>& axes, bool train, float epsilon,
      const ForwardBackwardState& state);

  // -- rnn --
  virtual std::tuple<Tensor, Tensor, Tensor, ForwardBackwardState> rnnForward(
      const Tensor& input, const Tensor& hiddenState, const Tensor& cellState,
      const Tensor& weights, int hiddenSize, int numLayers,
      RnnMode mode, bool bidirectional, float dropout,
      bool wantState = false);
  virtual std::tuple<Tensor, Tensor, Tensor, Tensor> rnnBackward(
      const Tensor& input, const Tensor& hiddenState, const Tensor& cellState,
      const Tensor& weights,
      const Tensor& dy, const Tensor& dhy, const Tensor& dcy,
      const Tensor& output, int numLayers, int hiddenSize,
      RnnMode mode, bool bidirectional, float dropProb,
      const ForwardBackwardState& state);

  /************************** Data Transfer ***************************/
  /// Copy data from src into pre-allocated dst.
  /// Both must have the same shape and dtype. src and dst may be strided;
  /// the backend handles gather/scatter/staging internally.
  /// Copy data from src into dst. If dst is empty, the backend allocates
  /// the destination buffer with the optimal memory type for the transfer
  /// direction (inferred from src location).
  /// Both must have the same shape and dtype. src and dst may be strided;
  /// the backend handles gather/scatter/staging internally.
  virtual void copy(Tensor& dst, const Tensor& src, const Stream& stream) = 0;

  /************************** Utils ***************************/
  virtual void print(const Tensor& tensor) = 0;

  /**
   * Checks if a datatype is supported by a TensorBackend and its registered
   * extensions.
   *
   * @param[in] dtype the datatype to check
   *
   * @return true if the data type is supported, false otherwise
   */
  virtual bool isDataTypeSupported(const fl::dtype& dtype) const final;

  /********************* Tensor Extensions **********************/
  template <typename T>
  T& getExtension() {
    static_assert(
        std::is_base_of<TensorExtensionBase, T>::value,
        "TensorBackend::getExtension<T>() called with type T "
        "that is not derived from TensorExtensionBase.");

    TensorExtensionType e = T::getExtensionType();

    // If an extension isn't present, instantiate it via its registered
    // creation function - only do this once per extension.
    if (!extensions_.contains(e)) {
      auto& registrar = detail::TensorExtensionRegistrar::getInstance();
      auto& creationFunc =
          registrar.getTensorExtensionCreationFunc(this->backendType(), e);
      extensions_.emplace(e, creationFunc());
    }
    return *(static_cast<T*>(extensions_.at(e).get()));
  }

 protected:
  std::unordered_map<TensorExtensionType, std::unique_ptr<TensorExtensionBase>>
      extensions_;
};

/**
 * Convert a Tensor from one backend to another.
 *
 * The resulting tensor will have the same shape, type, and contents.
 *
 * @param[in] in a tensor rvalue reference
 * @return a tensor with backend type specified by the template
 */
template <typename T>
Tensor toTensorType(Tensor&& in) {
  static_assert(
      std::is_base_of<TensorAdapterBase, T>::value,
      "toTensorType: T must be a derived type of TensorAdapterBase");
  // Fast path - backend is the same
  // TODO: make fl::TensorBackendType a static constexpr on the class as well so
  // as to not need to instantiate a backend to check the type
  if (in.backendType() == T().backendType()) {
    return std::move(in);
  }

  // As per impl requirements, Tensor::deviceSync() should return a pointer to
  // host memory if the tensor resides on the host.
  return Tensor(std::make_unique<T>(
      in.shape(),
      in.type(),
      // TODO: use the void specialization instead of a reinterpret cast
      reinterpret_cast<void*>(in.deviceSync<char>()), // expects contiguous memory
      in.location()));
}

namespace detail {

/**
 * Compare the backends of two tensors.
 *
 * @return true if the backends of both tensors are the same, else false.
 */
bool areBackendsEqual(const Tensor& a, const Tensor& b);

/**
 * Compare the backends of multiple tensors.
 *
 * @return true if all tensors' backends are the same, false otherwise.
 */
template <typename... Args>
bool areBackendsEqual(const Tensor& a, const Tensor& b, const Args&... args) {
  return areBackendsEqual(a, b) && areBackendsEqual(a, args...) &&
      areBackendsEqual(b, args...);
}

} // namespace detail
} // namespace fl
