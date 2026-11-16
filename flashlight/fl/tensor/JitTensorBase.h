/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/runtime/Device.h"
#include "flashlight/fl/runtime/Event.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/JitBackend.h"
#include "flashlight/fl/tensor/eval/Evaluator.h"
#include "flashlight/fl/tensor/ir/IndexNode.h"
#include "flashlight/fl/tensor/ir/Node.h"
#include "flashlight/fl/tensor/opt/Optimizer.h"

namespace fl {

/**
 * A JIT Tensor that wraps another backend -- it lazily builds up the
 * computation graph and delegates to the wrapped backend for execution.
 *
 * JitTensorBase is a concrete class (not abstract). It stores a pointer to its
 * owning JitBackend, which in turn knows about the wrapped backend.
 */
class JitTensorBase : public TensorAdapterBase {
 public:
  // declaration made private to enable `std::make_shared`
  class SharedData;

  constexpr static TensorBackendType tensorBackendType = TensorBackendType::Jit;

 private:
  // shared among shallow copies
  std::shared_ptr<SharedData> sharedData_;
  // pointer to owning JitBackend
  JitBackend* backend_;

  // Blocking convenience: eval + sync + compact.
  const Tensor& materializeSync() const;

  // convenience method to construct new JitTensor from a data node
  Tensor fromDataNode(NodePtr node) const;

  // Chain a TransferNode onto srcNode targeting host (DeviceType::x64),
  // wrap in a temp JIT tensor, and materialize via standard mechanisms.
  // Only called for off-host tensors; host-to-host is passthrough (CoW).
  // TODO(jacobkahn): also check for ARM device type (not just x64).
  Tensor transferToHost(NodePtr srcNode) const;

  // Create a lazy TransferNode that moves data from host to a target device.
  // The copy stream is the target device's stream (e.g. CUDA stream for H2D).
  Tensor transferToDevice(NodePtr srcNode, const Device& device) const;

 protected:
  TensorBackend& wrappedBackend() const;
  Optimizer& optimizer() const;
  Evaluator& evaluator() const;

 public:
  // Primary constructors — require a JitBackend reference
  JitTensorBase(NodePtr node, JitBackend& backend);
  JitTensorBase(std::shared_ptr<SharedData> sharedData, JitBackend& backend);

  // Placeholder data constructors for to<JitTensorBase>() fast path (never called)
  JitTensorBase(
      const Shape& shape,
      fl::dtype type,
      const void* ptr,
      Location memoryLocation);
  JitTensorBase(
      Dim nRows,
      Dim nCols,
      const Tensor& values,
      const Tensor& rowIdx,
      const Tensor& colIdx,
      StorageType storageType);

  virtual ~JitTensorBase() override;
  TensorBackendType backendType() const override;
  JitBackend& backend() const override;
  Tensor copy() override;
  Tensor shallowCopy() override;
  std::unique_ptr<TensorAdapterBase> clone() const override;
  const Shape& shape() const override;
  dtype type() const override;
  bool isSparse() const override;
  Location location() const override;
  void scalarSync(void* out) override;
  void device(void** out) override;
  void deviceSync(void** out) override;
  void hostSync(void* out) override;
  void unlock() override;
  bool isLocked() override;
  bool isContiguous() const override;
  Shape strides() const override;
  const Stream& stream() const override;
  Tensor astype(const dtype type) override;
  Tensor index(const std::vector<Index>& indices) override;
  Tensor flatten() const override;
  Tensor flat(const Index& idx) const override;
  Tensor asContiguousTensor() override;
  void setContext(void* context) override;
  void* getContext() override;
  std::string toString() override;
  std::ostream& operator<<(std::ostream& ostr) override;

  /**
   * Return the node this JIT tensor represents.
   * NOTE `const` w.r.t. the underlying Tensor this represents.
   */
  NodePtr node() const;

  /**
   * Force evaluation of this tensor's JIT node.
   * NOTE `const` w.r.t. the underlying Tensor this represents.
   */
  void eval() const;

  /**
   * Replace the node chain with a leaf ValueNode wrapping the evaluated result.
   * Separate from eval() so callers can choose when to break the graph.
   */
  void compact() const;

  /**
   * Get or create a deferred Event for this tensor. eval() records it on the
   * computation stream, fulfilling it and unblocking any Future waiters.
   * Returns the same Event on repeated calls (multiple Futures share it).
   * If the tensor is already evaluated, returns a default-constructed
   * (always-ready) Event.
   */
  Event getOrCreateDataReadyEvent() const;

  /**
   * Lazy transfer to target device. Builds the graph and creates a
   * deferred event but does NOT trigger evaluation.
   *
   * Same-device: returns (deferred_event, clone of this tensor).
   * Cross-device to host: returns (deferred_event, new tensor with
   *   TransferNode).
   * Other cross-device: throws (not yet supported).
   *
   * Caller must call fl::eval() on the returned tensor to trigger
   * computation. The deferred event fires when the stream work completes.
   */
  std::pair<Event, Tensor> to(const Device& device) const;

  /******************** Assignment Operators ********************/
#define ASSIGN_OP_TYPE_STUB(OP, TYPE) void OP(const TYPE& val) override;

#define ASSIGN_OP_STUB(OP)                 \
  ASSIGN_OP_TYPE_STUB(OP, Tensor);         \
  ASSIGN_OP_TYPE_STUB(OP, double);         \
  ASSIGN_OP_TYPE_STUB(OP, float);          \
  ASSIGN_OP_TYPE_STUB(OP, int);            \
  ASSIGN_OP_TYPE_STUB(OP, unsigned);       \
  ASSIGN_OP_TYPE_STUB(OP, bool);           \
  ASSIGN_OP_TYPE_STUB(OP, char);           \
  ASSIGN_OP_TYPE_STUB(OP, unsigned char);  \
  ASSIGN_OP_TYPE_STUB(OP, short);          \
  ASSIGN_OP_TYPE_STUB(OP, unsigned short); \
  ASSIGN_OP_TYPE_STUB(OP, long);           \
  ASSIGN_OP_TYPE_STUB(OP, unsigned long);  \
  ASSIGN_OP_TYPE_STUB(OP, long long);      \
  ASSIGN_OP_TYPE_STUB(OP, unsigned long long);

  ASSIGN_OP_STUB(assign); // =
  ASSIGN_OP_STUB(inPlaceAdd); // +=
  ASSIGN_OP_STUB(inPlaceSubtract); // -=
  ASSIGN_OP_STUB(inPlaceMultiply); // *=
  ASSIGN_OP_STUB(inPlaceDivide); // /=
#undef ASSIGN_OP_TYPE
#undef ASSIGN_OP
};

const JitTensorBase& toJitTensorBase(const Tensor& tensor);
JitTensorBase& toJitTensorBase(Tensor& tensor);

/**
 * Get or create a JitBackend that wraps the given backend.
 * Thread-safe singleton registry keyed by TensorBackendType.
 */
JitBackend& getOrCreateJitBackend(TensorBackend& wrappedBackend);

} // namespace fl
