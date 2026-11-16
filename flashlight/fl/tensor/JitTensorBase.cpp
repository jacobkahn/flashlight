/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/JitTensorBase.h"

#include <memory>
#include <mutex>
#include <queue>
#include <unordered_set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "flashlight/fl/runtime/Device.h"
#include "flashlight/fl/runtime/EventImpl.h"
#include "flashlight/fl/tensor/BackendRegistry.h"
#include "flashlight/fl/tensor/Copy.h"

#include "flashlight/fl/tensor/ir/CastNode.h"
#include "flashlight/fl/tensor/ir/CustomNode.h"
#include "flashlight/fl/tensor/ir/ExternalUse.h"
#include "flashlight/fl/tensor/ir/IndexedUpdateNode.h"
#include "flashlight/fl/tensor/ir/MultiValueNode.h"
#include "flashlight/fl/tensor/ir/ProjectionNode.h"
#include "flashlight/fl/tensor/ir/ReshapeNode.h"
#include "flashlight/fl/tensor/ir/TransferNode.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"

#define FL_JIT_TENSOR_UNIMPLEMENTED \
  throw std::invalid_argument(      \
      "JitTensorBase::" + std::string(__func__) + " - unimplemented.");

namespace fl {

// represents the data referred to by indexinges
class DataStorage {
  ExternalUse externalUse_;

 public:
  DataStorage(NodePtr node) : externalUse_(node) {}

  void replaceNode(NodePtr newNode) {
    externalUse_.setUsee(newNode);
  }

  NodePtr node() {
    return externalUse_.usee();
  }
};

// data shared among shallow copies and views of a tensor.
class JitTensorBase::SharedData {
  // 1. Why `shared_ptr<DataStorage> dataStorage_`?
  //   Indexing essentially creates a shallow-copy that views the indexed region
  //   of the original tensor. So any update to the original tensor should
  //   reflect as an update to dataNode for all shallow copies.
  // 2. Why `indexings_`?
  //   Underlying data node update requires creating a __new__ viewNode_ based
  //   on the __old__ indexings and __new__ data node (recall SSA semantics).
  // 3. Why `oldDataNode_`?
  //   When underlying data node changes, not all SharedData objects are
  //   informed directly, so we lazily perform (2), and use `oldDataNode_` to
  //   guide us.
  std::shared_ptr<DataStorage> dataStorage_;
  // TODO consider
  // 1. using an immutable linked-list here to speed things up
  // 2. making `std::vector<Index>` into an immutable class and use it as
  //    shared_ptr (to avoid copying when passing to ViewNodoe), since it's all
  //    readonly -- JitTensor gives us free immutability for Tensor index. To
  //    push it further, we can use custom shared data structure that has a more
  //    efficient copy constructor/uses std::atomic with
  //    std::memory_order::memory_order_relaxed on a refcount to reduce
  //    contention overhead.
  // 3. index merging, e.g., (1, 3)(1, 2) --> (2, 3). Maybe as an optimization
  //    pass, need to think more.
  std::vector<std::vector<Index>> indexings_{};
  // Optional reshape applied before the indexing chain. Used by flat() to
  // express flat indexing as reshape-to-1D + regular index while keeping the
  // same DataStorage (so writes propagate back). This is the SSA encoding of
  // "copy + reshape + scatter" — analogous to XLA's Reshape + DynamicUpdateSlice.
  std::optional<Shape> preReshapeShape_{std::nullopt};
  std::optional<NodePtr> oldDataNode_{std::nullopt}; // None iff no indexings
  std::optional<NodePtr> viewNode_{std::nullopt}; // None iff no indexings
  mutable bool evaluationEnqueued_{false};
  // Deferred event, fulfilled by eval() to unblock Future waiters.
  std::optional<Event> dataReadyEvent_;
  // Pre-allocated device buffer for lazy device(). Shared between SharedData
  // (for pointer stability) and Node (for evaluator to discover).
  std::optional<Tensor> outputBuffer_;

  void updateViewNodeIfNeeded() {
    if (!hasView() || dataStorage_->node() == oldDataNode_) {
      return;
    }
    auto toBeIndexedNode = dataStorage_->node();
    // Reshape before indexing (e.g. flat() reshapes to 1D first)
    if (preReshapeShape_) {
      toBeIndexedNode =
          ReshapeNode::create(toBeIndexedNode, *preReshapeShape_);
    }
    for (const auto& indices : indexings_) {
      toBeIndexedNode = IndexNode::create(toBeIndexedNode, indices);
    }
    viewNode_ = toBeIndexedNode;
    oldDataNode_ = dataStorage_->node();
  }

 public:
  bool hasView() const {
    return !indexings_.empty() || preReshapeShape_.has_value();
  }

  SharedData(NodePtr dataNode)
      : SharedData(std::make_shared<DataStorage>(dataNode), {}) {}

  SharedData(
      std::shared_ptr<DataStorage> dataStorage,
      std::vector<std::vector<Index>> indexings,
      std::optional<Shape> preReshapeShape = std::nullopt)
      : dataStorage_(dataStorage),
        indexings_(std::move(indexings)),
        preReshapeShape_(std::move(preReshapeShape)) {
    updateViewNodeIfNeeded();
  }

  void updateDataNode(NodePtr newNode) {
    if (hasView()) {
      // SSA indexed update: creates a new node representing
      // "copy original, scatter newNode at indexings (with optional reshape)"
      newNode = IndexedUpdateNode::create(
          dataStorage_->node(), indexings_, newNode, preReshapeShape_);
    }
    dataStorage_->replaceNode(newNode);
    // Mutation invalidates the old completion signal and assigned buffer.
    dataReadyEvent_.reset();
    outputBuffer_.reset();
  }

  // NOTE intended for optimizer
  void replaceNode(NodePtr newNode) {
    if (viewNode_.has_value()) {
      viewNode_ = newNode;
    } else {
      // graph optimization applies to all shallow copies
      dataStorage_->replaceNode(newNode);
    }
  }

  NodePtr getNode() {
    updateViewNodeIfNeeded();
    return viewNode_.value_or(dataStorage_->node());
  }

  const std::shared_ptr<DataStorage>& dataStorage() const {
    return dataStorage_;
  }

  bool isEvaluationEnqueued() const { return evaluationEnqueued_; }
  void setEvaluationEnqueued() const { evaluationEnqueued_ = true; }

  Event getOrCreateDataReadyEvent() {
    // If already evaluated, return a ready event — no deferred wait needed.
    if (getNode()->getResult().has_value()) {
      return Event(); // default-constructed = always ready
    }
    if (!dataReadyEvent_) {
      dataReadyEvent_ = Event(std::make_shared<detail::DeferredEventImpl>());
    }
    return *dataReadyEvent_;
  }

  void recordDataReadyEvent(const Stream& stream) {
    if (dataReadyEvent_) {
      dataReadyEvent_->record(stream);
    }
  }

  std::shared_ptr<SharedData> applyIndices(std::vector<Index> indices) {
    std::vector<std::vector<Index>> newIndexings = this->indexings_;
    newIndexings.push_back(std::move(indices));
    return std::make_shared<SharedData>(
        dataStorage_, newIndexings, preReshapeShape_);
  }

  /**
   * Get or allocate a device buffer for lazy device(). The buffer is stored
   * on both SharedData (pointer stability for the caller) and the Node
   * (so the evaluator finds it during assignBuffers and writes into it).
   */
  Tensor& getOrAllocateDeviceBuffer(
      TensorBackend& backend, const Shape& shape, dtype type,
      std::shared_ptr<const Stream> stream) {
    if (!outputBuffer_) {
      outputBuffer_ = backend.empty(shape, type, std::move(stream));
      // Also store on the node so the evaluator writes into it. A shallow
      // copy shares the buffer, so the result lands in the caller's pointer.
      dataStorage_->node()->setOutputBuffer(outputBuffer_->shallowCopy());
    }
    return *outputBuffer_;
  }

  /**
   * After eval, ensure the result lives in the output buffer that device()
   * returned a pointer to. If the dispatch function ignored the assigned
   * buffer and allocated its own, copy the result into the assigned buffer.
   */
  void ensureResultInOutputBuffer(const NodePtr& node) {
    if (!outputBuffer_) {
      return; // no assigned buffer — nothing to reconcile
    }
    if (!node->getResult().has_value()) {
      return; // shouldn't happen after eval, but guard
    }
    // Compare device pointers: if they differ, the dispatch function
    // wrote into its own buffer. Copy the data into ours.
    const auto& result = node->getResult().value();
    void* resultPtr = nullptr;
    void* assignedPtr = nullptr;
    result.device(&resultPtr);
    outputBuffer_->device(&assignedPtr);

    if (resultPtr != assignedPtr) {
      // Copy result into the assigned buffer. fl::copy handles both
      // CPU (memcpy) and CUDA (cudaMemcpyAsync) transparently.
      fl::copy(*outputBuffer_, result);
      // Swap the result to point to our assigned buffer
      node->unsetResult();
      node->setResult(outputBuffer_->copy());
    }
  }

};

TensorBackend& JitTensorBase::wrappedBackend() const {
  return backend().wrappedBackend();
}

Optimizer& JitTensorBase::optimizer() const {
  return backend().optimizer();
}

Evaluator& JitTensorBase::evaluator() const {
  return backend().evaluator();
}

JitTensorBase::JitTensorBase(NodePtr node, JitBackend& backend)
    : JitTensorBase(std::make_shared<SharedData>(node), backend) {}

JitTensorBase::JitTensorBase(
    std::shared_ptr<SharedData> sharedData, JitBackend& backend)
    : sharedData_(sharedData), backend_(&backend) {}

// Placeholder data constructors — to<> fast path means these are never called
JitTensorBase::JitTensorBase(
    const Shape&, fl::dtype, const void*, Location) {
  throw std::logic_error(
      "JitTensorBase data constructor: use setDefaultJitTensorType<T>()");
}

JitTensorBase::JitTensorBase(
    Dim, Dim, const Tensor&, const Tensor&, const Tensor&, StorageType) {
  throw std::logic_error(
      "JitTensorBase sparse constructor: use setDefaultJitTensorType<T>()");
}

JitTensorBase::~JitTensorBase() {}

const Tensor& JitTensorBase::materializeSync() const {
  if (!node()->getResult().has_value()) {
    eval();
    stream().sync();
  }
  compact();
  return node()->getResult().value();
}

Tensor JitTensorBase::fromDataNode(NodePtr node) const {
  return Tensor(std::make_unique<JitTensorBase>(
      std::make_shared<SharedData>(node), *backend_));
}

Tensor JitTensorBase::transferToHost(NodePtr srcNode) const {
  // Resolve copy stream: prefer the source node's explicit stream, then the
  // JitBackend's default stream (which is the wrapped backend's jitStream).
  // Stream ordering on the device stream guarantees compute finishes before
  // the copy starts.
  // TODO: CudaBackend should override jitStream() to return its CUDA stream;
  // until then, CUDA JIT nodes rely on the evaluator's internal dispatch.
  auto copyStream = srcNode->stream();
  if (!copyStream) {
    copyStream = backend().defaultStreamShared();
  }

  auto xferNode = TransferNode::create(
      srcNode,
      DeviceType::x64,
      backendTypeForDevice(DeviceType::x64),
      copyStream);

  // Return a lazy JIT tensor — no eval, no sync. The caller or the
  // event/promise machinery decides when to materialize.
  return fromDataNode(xferNode);
}

Tensor JitTensorBase::transferToDevice(
    NodePtr srcNode, const Device& device) const {
  // The copy runs on the target device's stream (e.g. CUDA stream for H2D)
  // because cudaMemcpyAsync must be enqueued on a CUDA stream.
  auto targetBackendType = backendTypeForDevice(device.type());
  auto& targetBackend = fl::getBackend(targetBackendType);
  auto copyStream = targetBackend.jitStream();

  auto xferNode = TransferNode::create(
      srcNode, device.type(), targetBackendType, copyStream);

  return fromDataNode(xferNode);
}

Tensor JitTensorBase::copy() {
  // Since a node's computation result is immutable, copy is free.
  return Tensor(clone());
}

Tensor JitTensorBase::shallowCopy() {
  // NOTE IR-captured computation semantics is immutable
  return Tensor(std::make_unique<JitTensorBase>(sharedData_, *backend_));
}

std::unique_ptr<TensorAdapterBase> JitTensorBase::clone() const {
  // NOTE IR-captured computation semantics is immutable
  // copy forgets about indexing -- `node()` takes care of materialization
  return std::make_unique<JitTensorBase>(node(), *backend_);
}

JitBackend& JitTensorBase::backend() const {
  return *backend_;
}

TensorBackendType JitTensorBase::backendType() const {
  return TensorBackendType::Jit;
}

const Shape& JitTensorBase::shape() const {
  return node()->shape();
}

fl::dtype JitTensorBase::type() const {
  return node()->dataType();
}

bool JitTensorBase::isSparse() const {
  return false; // JIT only wraps dense tensors
}

Location JitTensorBase::location() const {
  // Use resolved stream (handles null node streams via BFS/backend fallback).
  // TODO: also check ARM device type, not just x64.
  auto devType = stream().device().type();
  return (devType == DeviceType::x64) ? Location::Host : Location::Device;
}

void JitTensorBase::scalarSync(void* out) {
  // Pure data read — caller (Tensor templates) must have eval'd + sync'd.
  // compact() severs the graph, then we read from the concrete result.
  compact();
  const auto& result = node()->getResult().value();
  switch (type()) {
    case dtype::f16:
    case dtype::bf16:
      throw std::runtime_error(
          "[JitTensorBase::scalar] f16/bf16 unsupported");
    case dtype::f32:
      *((float*)out) = result.scalarSync<float>();
      return;
    case dtype::f64:
      *((double*)out) = result.scalarSync<double>();
      return;
    case dtype::b8:
      *((char*)out) = result.scalarSync<char>();
      return;
    case dtype::s16:
      *((short*)out) = result.scalarSync<short>();
      return;
    case dtype::s32:
      *((int*)out) = result.scalarSync<int>();
      return;
    case dtype::s64:
      *((long long*)out) = result.scalarSync<long long>();
      return;
    case dtype::u8:
      *((unsigned char*)out) = result.scalarSync<unsigned char>();
      return;
    case dtype::u16:
      *((unsigned short*)out) = result.scalarSync<unsigned short>();
      return;
    case dtype::u32:
      *((unsigned int*)out) = result.scalarSync<unsigned int>();
      return;
    case dtype::u64:
      *((unsigned long long*)out) = result.scalarSync<unsigned long long>();
      return;
  }
  throw std::runtime_error("[JitTensorBase::scalar] Unknown data type");
}

void JitTensorBase::device(void** out) {
  if (node()->getResult().has_value()) {
    // Already evaluated — return result's device pointer directly.
    compact();
    node()->getResult().value().device(out);
    return;
  }
  // A view's buffer would be sized for the view but attached to the storage
  // node, which writes the whole base into it; materialize views instead.
  if (sharedData_->hasView()) {
    materializeSync().device(out);
    return;
  }
  // Lazy: allocate a buffer on the node's stream without triggering eval.
  // The evaluator will write into this buffer during execution.
  auto& buffer = sharedData_->getOrAllocateDeviceBuffer(
      wrappedBackend(), node()->shape(), node()->dataType(), node()->stream());
  buffer.device(out);
}

void JitTensorBase::deviceSync(void** out) {
  materializeSync().device(out);
}

void JitTensorBase::hostSync(void* out) {
  // Pure data read — caller (Tensor templates) must have eval'd + sync'd.
  compact();
  node()->getResult().value().hostSync(out);
}

void JitTensorBase::unlock() {
  // No-op: buffer lifetime managed by Tensor object.
}

bool JitTensorBase::isLocked() {
  return false;
}

bool JitTensorBase::isContiguous() const {
  // Already evaluated — delegate to result.
  if (node()->getResult().has_value()) {
    compact();
    return node()->getResult().value().isContiguous();
  }
  // ValueNode wraps a concrete tensor — query it directly, no eval needed.
  if (node()->isValue()) {
    return node()->impl<ValueNode>().value().isContiguous();
  }
  // These node types may produce non-contiguous output — materialize.
  if (node()->isIndex() || node()->isCustom() || node()->isProjection()) {
    return materializeSync().isContiguous();
  }
  // All other node types (binary, unary, reduction, matmul, reshape,
  // transpose, scalar, cast, conv2d, pool2d, batchnorm, fused ops,
  // transfer, indexed update, multi-value) always produce contiguous output.
  return true;
}

Shape JitTensorBase::strides() const {
  // Already evaluated — delegate to result.
  if (node()->getResult().has_value()) {
    compact();
    return node()->getResult().value().strides();
  }
  // ValueNode wraps a concrete tensor — query it directly, no eval needed.
  if (node()->isValue()) {
    return node()->impl<ValueNode>().value().strides();
  }
  // These node types may produce non-contiguous output — materialize.
  if (node()->isIndex() || node()->isCustom() || node()->isProjection()) {
    return materializeSync().strides();
  }
  // All other node types produce contiguous output — compute from shape.
  return contiguousStrides(node()->shape());
}

const Stream& JitTensorBase::stream() const {
  // 1. Explicit stream on this node
  if (node()->stream()) {
    return *node()->stream();
  }
  // 2. Already evaluated — use result tensor's stream
  if (node()->getResult().has_value()) {
    return node()->getResult().value().stream();
  }
  // 3. Infer from leaf ValueNodes via BFS (they always have results). Shared
  // nodes are visited once; without that the walk is exponential on DAGs.
  std::queue<NodePtr> worklist;
  std::unordered_set<Node*> visited;
  worklist.push(node());
  while (!worklist.empty()) {
    NodePtr n = worklist.front();
    worklist.pop();
    if (!visited.insert(n.get()).second) {
      continue;
    }
    if (n->stream()) {
      return *n->stream();
    }
    if (n->getResult().has_value()) {
      return n->getResult().value().stream();
    }
    for (const auto& input : n->inputs()) {
      worklist.push(input);
    }
  }
  // Pure scalar/constant graph — use backend default
  return backend().defaultStream();
}

Tensor JitTensorBase::astype(const dtype type) {
  if (type == this->type()) {
    return fromDataNode(this->node()); // no-op cast
  }
  return fromDataNode(CastNode::create(this->node(), type));
}

Tensor JitTensorBase::index(const std::vector<Index>& indices) {
  // TODO: support multi-tensor advanced indexing in JIT shape inference
  // (NumPy-style broadcasting rules). For now, materialize as a barrier.
  int tensorIndexCount = 0;
  for (const auto& idx : indices) {
    if (idx.type() == detail::IndexType::Tensor) {
      tensorIndexCount++;
    }
  }
  if (tensorIndexCount > 1) {
    const auto& selfResult = materializeSync();
    for (const auto& idx : indices) {
      if (idx.type() == detail::IndexType::Tensor) {
        toJitTensorBase(idx.get<Tensor>()).materializeSync();
      }
    }
    auto result = selfResult(indices);
    return fromDataNode(ValueNode::create(std::move(result)));
  }
  return Tensor(std::make_unique<JitTensorBase>(
      sharedData_->applyIndices(indices), *backend_));
}

Tensor JitTensorBase::flatten() const {
  Shape flatShape({node()->shape().elements()});
  return fromDataNode(ReshapeNode::create(this->node(), flatShape));
}

Tensor JitTensorBase::flat(const Index& idx) const {
  Shape flatShape({node()->shape().elements()});
  std::vector<std::vector<Index>> indexings;
  indexings.push_back({idx});
  return Tensor(std::make_unique<JitTensorBase>(
      std::make_shared<SharedData>(
          sharedData_->dataStorage(), std::move(indexings), flatShape),
      *backend_));
}

Tensor JitTensorBase::asContiguousTensor() {
  return fromDataNode(CustomNode::create(
      "asContiguousTensor", {this->node()}, Shape(shape()), node()->dataType(), [=](auto inputs) {
        return inputs.at(0)->asContiguousTensor();
      }));
}

void JitTensorBase::setContext(void* /* context */) {
  // no-op
}

void* JitTensorBase::getContext() {
  return nullptr;
}

std::string JitTensorBase::toString() {
  return materializeSync().toString();
}

std::ostream& JitTensorBase::operator<<(std::ostream& ostr) {
  ostr << toString();
  return ostr;
}

/******************** Assignment Operators ********************/
// NOTE Think SSA:
// x += 42
// ......
// --->
// x' = x + 42
// ...... (uses of x becomes x')
void JitTensorBase::assign(const Tensor& other) {
  sharedData_->updateDataNode(toJitTensorBase(other).node());
}

#define FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, TYPE)                   \
  void JitTensorBase::OP(const TYPE& scalar) {                     \
    this->assign(backend().full(this->shape(), scalar, this->type())); \
  }

#define FL_JIT_TENSOR_ASSIGN_BINOP_TENSOR(OP, BINOP) \
  void JitTensorBase::OP(const Tensor& other) {      \
    this->assign(this->shallowCopy() BINOP other);   \
  }

#define FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, TYPE) \
  void JitTensorBase::OP(const TYPE& scalar) {             \
    this->assign(this->shallowCopy() BINOP scalar);        \
  }

#define FL_JIT_TENSOR_ASSIGN_OP(OP)                   \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, double);         \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, float);          \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, int);            \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, unsigned);       \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, bool);           \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, char);           \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, unsigned char);  \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, short);          \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, unsigned short); \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, long);           \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, unsigned long);  \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, long long);      \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, unsigned long long);

#define FL_JIT_TENSOR_ASSIGN_BINOP(OP, BINOP)                   \
  FL_JIT_TENSOR_ASSIGN_BINOP_TENSOR(OP, BINOP);                 \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, double);         \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, float);          \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, int);            \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, unsigned);       \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, bool);           \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, char);           \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, unsigned char);  \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, short);          \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, unsigned short); \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, long);           \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, unsigned long);  \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, long long);      \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, unsigned long long);

FL_JIT_TENSOR_ASSIGN_OP(assign); // =
FL_JIT_TENSOR_ASSIGN_BINOP(inPlaceAdd, +); // +=
FL_JIT_TENSOR_ASSIGN_BINOP(inPlaceSubtract, -); // -=
FL_JIT_TENSOR_ASSIGN_BINOP(inPlaceMultiply, *); // *=
FL_JIT_TENSOR_ASSIGN_BINOP(inPlaceDivide, /); // /=
#undef FL_JIT_TENSOR_ASSIGN_OP_SCALAR
#undef FL_JIT_TENSOR_ASSIGN_BINOP_TENSOR
#undef FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR
#undef FL_JIT_TENSOR_ASSIGN_OP
#undef FL_JIT_TENSOR_ASSIGN_BINOP

NodePtr JitTensorBase::node() const {
  return sharedData_->getNode();
}

void JitTensorBase::eval() const {
  if (node()->getResult().has_value()) {
    return;
  }
  sharedData_->replaceNode(optimizer().optimize(node()));
  evaluator().eval(node());
  sharedData_->ensureResultInOutputBuffer(node());
  sharedData_->recordDataReadyEvent(stream());
}

void JitTensorBase::compact() const {
  if (!node()->getResult().has_value()) {
    return;
  }
  // Already a leaf — nothing to sever.
  if (node()->isValue() || node()->isMultiValue()) {
    return;
  }
  // Replace the (potentially deep) node chain with a leaf node wrapping the
  // evaluated result. This severs the old graph allowing it to be freed and
  // keeps graph depth O(1) across repeated eval() calls.
  if (node()->isProjection()) {
    // Multi-output: wrap all sibling outputs into a single MultiValueNode
    // so ProjectionNode relationships are preserved for downstream passes.
    auto& proj = node()->impl<ProjectionNode>();
    auto parent = proj.parent();
    if (!parent->isMultiValue()) {
      std::vector<Tensor> values(parent->numOutputs());
      for (unsigned i = 0; i < parent->numOutputs(); ++i) {
        if (parent->hasOutput(i)) {
          values[i] = parent->getOutput(i);
        }
      }
      auto bundle =
          MultiValueNode::create(std::move(values), parent->stream());
      parent->replaceAllUsesWith(bundle);
    }
  } else {
    // Single-output: replace with a ValueNode holding the concrete result.
    // Use shallowCopy() to share the underlying buffer instead of deep-copying.
    // This is critical for cross-device transfers (e.g. D2H via TransferNode):
    // the result CPUTensor's buffer may still be the target of an in-flight
    // cudaMemcpyAsync, and a deep copy (CPUTensor::clone) would race with it
    // by reading zeros before the async transfer completes.
    auto result = node()->getResult().value().shallowCopy();
    auto stream = node()->stream();
    auto compactNode = ValueNode::create(std::move(result), stream);
    sharedData_->replaceNode(compactNode);
  }
}

Event JitTensorBase::getOrCreateDataReadyEvent() const {
  return sharedData_->getOrCreateDataReadyEvent();
}

std::pair<Event, Tensor>
JitTensorBase::to(const Device& device) const {
  auto& srcDev = stream().device();
  if (srcDev.type() == device.type() &&
      srcDev.nativeId() == device.nativeId()) {
    // Same device: no transfer needed. Return a shallow copy sharing this
    // SharedData so that eval() on either copy records the deferred event
    // on the same SharedData.
    Event event = getOrCreateDataReadyEvent();
    Tensor self(std::make_unique<JitTensorBase>(sharedData_, *backend_));
    return {std::move(event), std::move(self)};
  }

  // Cross-device to host: build TransferNode (lazy, no eval)
  if (device.type() == DeviceType::x64) {
    Tensor transfer = transferToHost(node());
    Event event = toJitTensorBase(transfer).getOrCreateDataReadyEvent();
    return {std::move(event), std::move(transfer)};
  }

  // Cross-device to CUDA: build TransferNode targeting GPU (lazy, no eval)
  if (device.type() == DeviceType::CUDA) {
    Tensor transfer = transferToDevice(node(), device);
    Event event = toJitTensorBase(transfer).getOrCreateDataReadyEvent();
    return {std::move(event), std::move(transfer)};
  }

  throw std::runtime_error(
      "[JitTensorBase::to] unsupported target device type");
}

const JitTensorBase& toJitTensorBase(const Tensor& tensor) {
  auto type = tensor.backendType();
  if (type != TensorBackendType::Jit) {
    std::ostringstream oss;
    oss << "[toJitTensorBase] expected JIT-backed tensor, got " << type;
    throw std::invalid_argument(oss.str());
  }
  return tensor.getAdapter<JitTensorBase>();
}

JitTensorBase& toJitTensorBase(Tensor& tensor) {
  auto type = tensor.backendType();
  if (type != TensorBackendType::Jit) {
    std::ostringstream oss;
    oss << "[toJitTensorBase] expected JIT-backed tensor, got " << type;
    throw std::invalid_argument(oss.str());
  }
  return tensor.getAdapter<JitTensorBase>();
}

JitBackend& getOrCreateJitBackend(TensorBackend& wrappedBackend) {
  static std::mutex mu;
  static std::unordered_map<TensorBackendType, std::unique_ptr<JitBackend>>
      backends;

  auto type = wrappedBackend.backendType();
  std::lock_guard<std::mutex> lock(mu);
  auto it = backends.find(type);
  if (it != backends.end()) {
    return *it->second;
  }
  auto ptr = std::make_unique<JitBackend>(
      wrappedBackend, wrappedBackend.jitStream());
  auto& ref = *ptr;
  backends.emplace(type, std::move(ptr));
  return ref;
}

} // namespace fl
