/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/JitBackend.h"

#include <cassert>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "flashlight/fl/runtime/Device.h"
#include "flashlight/fl/tensor/JitBackendUtils.h"
#include "flashlight/fl/tensor/JitTensorBase.h"
#include "flashlight/fl/tensor/ShapeInference.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/ir/CastNode.h"
#include "flashlight/fl/tensor/ir/ClipNode.h"
#include "flashlight/fl/tensor/ir/ConcatenateNode.h"
#include "flashlight/fl/tensor/ir/CumsumNode.h"
#include "flashlight/fl/tensor/ir/CustomNode.h"
#include "flashlight/fl/tensor/ir/FlipNode.h"
#include "flashlight/fl/tensor/ir/MatmulNode.h"
#include "flashlight/fl/tensor/ir/NodeUtils.h"
#include "flashlight/fl/tensor/ir/PadNode.h"
#include "flashlight/fl/tensor/ir/ReductionNode.h"
#include "flashlight/fl/tensor/ir/ReshapeNode.h"
#include "flashlight/fl/tensor/ir/RollNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/TileNode.h"
#include "flashlight/fl/tensor/ir/TransferNode.h"
#include "flashlight/fl/tensor/ir/TransposeNode.h"
#include "flashlight/fl/tensor/ir/TriangularNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"
#include "flashlight/fl/tensor/ir/WhereNode.h"
#include "flashlight/fl/tensor/opt/JitOptimizerExtension.h"
#include "flashlight/fl/tensor/opt/JitOptimizerExtensionBackends.h"

namespace fl {

using detail::materializeMutable;
using detail::materializeSafe;
using detail::tensorsToNodes;

namespace {

const Tensor& materialize(const Tensor& tensor) {
  auto& jitTensor = toJitTensorBase(tensor);
  jitTensor.eval();
  jitTensor.stream().sync();
  return jitTensor.node()->getResult().value();
}

JitOptimizerExtension* getOptimizerExtension(TensorBackend& backend) {
  auto& registrar = detail::TensorExtensionRegistrar::getInstance();
  if (registrar.isTensorExtensionRegistered(
          backend.backendType(), TensorExtensionType::JitOptimizer)) {
    return &backend.getExtension<JitOptimizerExtension>();
  }
  return nullptr;
}

} // namespace

JitBackend::JitBackend(
    TensorBackend& wrappedBackend,
    std::shared_ptr<Stream> defaultStream)
    : dispatch_(wrappedBackend.createDispatch()),
      evaluator_(dispatch_),
      optimizer_(getOptimizerExtension(wrappedBackend)),
      defaultStream_(std::move(defaultStream)) {}

TensorBackendType JitBackend::backendType() const {
  return TensorBackendType::Jit;
}

Evaluator& JitBackend::evaluator() {
  return evaluator_;
}

Optimizer& JitBackend::optimizer() {
  return optimizer_;
}

TensorBackend& JitBackend::wrappedBackend() {
  return *dispatch_.ctx;
}

const BackendDispatch& JitBackend::dispatch() const {
  return dispatch_;
}

const Stream& JitBackend::defaultStream() const {
  return *defaultStream_;
}

std::shared_ptr<const Stream> JitBackend::defaultStreamShared() const {
  return defaultStream_;
}

Tensor JitBackend::createJitTensor(NodePtr node) {
  return Tensor(std::make_unique<JitTensorBase>(std::move(node), *this));
}

/* -------------------------- Compute Functions -------------------------- */

void JitBackend::eval(const Tensor& tensor) {
  auto& jitTensor = toJitTensorBase(tensor);
  assert(&jitTensor.backend() == this);
  jitTensor.eval(); // optimization + evaluation on target stream
  jitTensor.compact(); // sever old graph to keep depth O(1)
  // No sync — compute runs asynchronously on the worker thread.
  // Blocking only happens when data is actually needed (host/scalar/etc).
}

bool JitBackend::supportsDataType(const fl::dtype& dtype) const {
  return dispatch_.supportsDataType(dispatch_.ctx, dtype);
}

void JitBackend::getMemMgrInfo(
    const char* msg,
    const int deviceId,
    std::ostream* ostream) {
  dispatch_.getMemMgrInfo(dispatch_.ctx, msg, deviceId, ostream);
}

void JitBackend::setMemMgrLogStream(std::ostream* stream) {
  dispatch_.setMemMgrLogStream(dispatch_.ctx, stream);
}

void JitBackend::setMemMgrLoggingEnabled(const bool enabled) {
  dispatch_.setMemMgrLoggingEnabled(dispatch_.ctx, enabled);
}

void JitBackend::setMemMgrFlushInterval(const size_t interval) {
  dispatch_.setMemMgrFlushInterval(dispatch_.ctx, interval);
}

/* -------------------------- Rand Functions -------------------------- */

void JitBackend::setSeed(const int seed) {
  dispatch_.setSeed(dispatch_.ctx, seed);
}

Tensor JitBackend::randn(
    const Shape& shape,
    dtype type,
    std::shared_ptr<const Stream> stream) {
  return createJitTensor(
      CustomNode::create(
          "randn",
          tensorsToNodes(),
          Shape(shape),
          type,
          [dispatch = dispatch_, shape, type](
              const std::vector<const Tensor*>& /* inputs */) {
            return dispatch.randn(dispatch.ctx, shape, type);
          },
          std::move(stream)));
}

Tensor JitBackend::rand(
    const Shape& shape,
    dtype type,
    std::shared_ptr<const Stream> stream) {
  return createJitTensor(
      CustomNode::create(
          "rand",
          tensorsToNodes(),
          Shape(shape),
          type,
          [dispatch = dispatch_, shape, type](
              const std::vector<const Tensor*>& /* inputs */) {
            return dispatch.rand(dispatch.ctx, shape, type);
          },
          std::move(stream)));
}

/* --------------------------- Tensor Operators --------------------------- */

/******************** Tensor Creation Functions ********************/
Tensor JitBackend::fromScalar(
    double value,
    const dtype type,
    std::shared_ptr<const Stream> stream) {
  return full(Shape{}, value, type, std::move(stream));
}

Tensor JitBackend::full(
    const Shape& shape,
    double value,
    const dtype type,
    std::shared_ptr<const Stream> stream) {
  return createJitTensor(
      ScalarNode::create(shape, type, value, std::move(stream)));
}

Tensor JitBackend::empty(
    const Shape& shape,
    dtype type,
    std::shared_ptr<const Stream> /* stream */) {
  // Allocation, not computation — delegate to wrapped backend directly.
  return createJitTensor(
      ValueNode::create(dispatch_.empty(dispatch_.ctx, shape, type)));
}

Tensor JitBackend::identity(
    const Dim dim,
    const dtype type,
    std::shared_ptr<const Stream> /* stream */) {
  return createJitTensor(
      ValueNode::create(dispatch_.identity(dispatch_.ctx, dim, type)));
}

Tensor JitBackend::arange(
    const Shape& shape,
    const Dim seqDim,
    const dtype type,
    std::shared_ptr<const Stream> stream) {
  return createJitTensor(
      CustomNode::create(
          "arange",
          tensorsToNodes(),
          Shape(shape),
          type,
          [dispatch = dispatch_, shape, seqDim, type](
              const std::vector<const Tensor*>& /* inputs */) {
            return dispatch.arange(dispatch.ctx, shape, seqDim, type);
          },
          std::move(stream)));
}

Tensor JitBackend::iota(
    const Shape& dims,
    const Shape& tileDims,
    const dtype type,
    std::shared_ptr<const Stream> /* stream */) {
  return createJitTensor(
      ValueNode::create(dispatch_.iota(dispatch_.ctx, dims, tileDims, type)));
}

/************************ Shaping and Indexing *************************/
void JitBackend::reshape(
    Tensor& out,
    const Tensor& tensor,
    const Shape& shape,
    std::shared_ptr<const Stream> stream) {
  out = createJitTensor(
      ReshapeNode::create(
          toJitTensorBase(tensor).node(), shape, std::move(stream)));
}

void JitBackend::transpose(
    Tensor& out,
    const Tensor& tensor,
    const Shape& axes,
    std::shared_ptr<const Stream> stream) {
  out = createJitTensor(
      TransposeNode::create(
          toJitTensorBase(tensor).node(), axes, std::move(stream)));
}

void JitBackend::tile(
    Tensor& out,
    const Tensor& tensor,
    const Shape& tileDims,
    std::shared_ptr<const Stream> stream) {
  out = createJitTensor(
      TileNode::create(
          toJitTensorBase(tensor).node(), tileDims, std::move(stream)));
}

void JitBackend::concatenate(
    Tensor& out,
    const std::vector<Tensor>& tensors,
    const unsigned axisToConcat,
    std::shared_ptr<const Stream> stream) {
  out = createJitTensor(
      ConcatenateNode::create(
          tensorsToNodes(tensors), axisToConcat, std::move(stream)));
}

void JitBackend::nonzero(
    Tensor& out,
    const Tensor& tensor,
    std::shared_ptr<const Stream> /* stream */) {
  // NOTE must materialize since we can't infer output shape
  const auto& tensorResult = materialize(tensor);
  Tensor result;
  dispatch_.nonzero(dispatch_.ctx, result, tensorResult);
  out = createJitTensor(ValueNode::create(std::move(result)));
}

void JitBackend::pad(
    Tensor& out,
    const Tensor& input,
    const std::vector<std::pair<int, int>>& padWidths,
    const PadType type,
    std::shared_ptr<const Stream> stream) {
  out = createJitTensor(
      PadNode::create(
          toJitTensorBase(input).node(), padWidths, type, std::move(stream)));
}

/************************** Unary Operators ***************************/

Tensor JitBackend::createUnaryJitTensor(
    const Tensor& tensor,
    UnaryOp op,
    std::shared_ptr<const Stream> stream) {
  const auto inputNode = toJitTensorBase(tensor).node();
  return createJitTensor(UnaryNode::create(inputNode, op, std::move(stream)));
}

void JitBackend::unary(
    Tensor& out,
    UnaryOp op,
    const Tensor& input,
    std::shared_ptr<const Stream> stream) {
  out = createUnaryJitTensor(input, op, std::move(stream));
}

void JitBackend::flip(
    Tensor& out,
    const Tensor& tensor,
    const unsigned dim,
    std::shared_ptr<const Stream> stream) {
  out = createJitTensor(
      FlipNode::create(toJitTensorBase(tensor).node(), dim, std::move(stream)));
}

void JitBackend::clip(
    Tensor& out,
    const Tensor& tensor,
    const Tensor& low,
    const Tensor& high,
    std::shared_ptr<const Stream> stream) {
  out = createJitTensor(
      ClipNode::create(
          toJitTensorBase(tensor).node(),
          toJitTensorBase(low).node(),
          toJitTensorBase(high).node(),
          std::move(stream)));
}

void JitBackend::roll(
    Tensor& out,
    const Tensor& tensor,
    const int shift,
    const unsigned axis,
    std::shared_ptr<const Stream> stream) {
  out = createJitTensor(
      RollNode::create(
          toJitTensorBase(tensor).node(), shift, axis, std::move(stream)));
}

void JitBackend::tril(
    Tensor& out,
    const Tensor& tensor,
    int diagonal,
    std::shared_ptr<const Stream> stream) {
  out = createJitTensor(
      TriangularNode::create(
          toJitTensorBase(tensor).node(),
          TriangularMode::Lower,
          diagonal,
          std::move(stream)));
}

void JitBackend::triu(
    Tensor& out,
    const Tensor& tensor,
    int diagonal,
    std::shared_ptr<const Stream> stream) {
  out = createJitTensor(
      TriangularNode::create(
          toJitTensorBase(tensor).node(),
          TriangularMode::Upper,
          diagonal,
          std::move(stream)));
}

void JitBackend::where(
    Tensor& out,
    const Tensor& condition,
    const Tensor& x,
    const Tensor& y,
    std::shared_ptr<const Stream> stream) {
  out = createJitTensor(
      WhereNode::create(
          toJitTensorBase(condition).node(),
          toJitTensorBase(x).node(),
          toJitTensorBase(y).node(),
          std::move(stream)));
}

void JitBackend::topk(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned k,
    const Dim axis,
    const SortMode sortMode,
    std::shared_ptr<const Stream> stream) {
  auto cache = std::make_shared<std::optional<std::pair<Tensor, Tensor>>>();
  auto computeIfNeeded = [dispatch = dispatch_, k, axis, sortMode, cache](
                             const std::vector<const Tensor*>& inputs) {
    if (!cache->has_value()) {
      Tensor v, i;
      dispatch.topk(dispatch.ctx, v, i, *inputs[0], k, axis, sortMode);
      cache->emplace(std::move(v), std::move(i));
    }
  };
  auto outShape = Shape(input.shape());
  outShape[axis] = k;
  auto inputNodes = tensorsToNodes(input);
  values = createJitTensor(
      CustomNode::create(
          "topk_values",
          NodeList(inputNodes),
          Shape(outShape),
          input.type(),
          [computeIfNeeded, cache](const std::vector<const Tensor*>& inputs) {
            computeIfNeeded(inputs);
            return cache->value().first;
          },
          stream));
  indices = createJitTensor(
      CustomNode::create(
          "topk_indices",
          NodeList(inputNodes),
          Shape(outShape),
          dtype::s32,
          [computeIfNeeded, cache](const std::vector<const Tensor*>& inputs) {
            computeIfNeeded(inputs);
            return cache->value().second;
          },
          stream));
}

void JitBackend::sort(
    Tensor& out,
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode,
    std::shared_ptr<const Stream> stream) {
  out = createJitTensor(
      CustomNode::create(
          "sort",
          tensorsToNodes(input),
          Shape(input.shape()),
          input.type(),
          [dispatch = dispatch_, axis, sortMode](
              const std::vector<const Tensor*>& inputs) {
            Tensor result;
            dispatch.sort(dispatch.ctx, result, *inputs.at(0), axis, sortMode);
            return result;
          },
          std::move(stream)));
}

void JitBackend::sort(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode,
    std::shared_ptr<const Stream> stream) {
  auto cache = std::make_shared<std::optional<std::pair<Tensor, Tensor>>>();
  auto computeIfNeeded = [dispatch = dispatch_, axis, sortMode, cache](
                             const std::vector<const Tensor*>& inputs) {
    if (!cache->has_value()) {
      Tensor v, i;
      dispatch.sortWithIndices(dispatch.ctx, v, i, *inputs[0], axis, sortMode);
      cache->emplace(std::move(v), std::move(i));
    }
  };
  auto inputNodes = tensorsToNodes(input);
  values = createJitTensor(
      CustomNode::create(
          "sort_values",
          NodeList(inputNodes),
          Shape(input.shape()),
          input.type(),
          [computeIfNeeded, cache](const std::vector<const Tensor*>& inputs) {
            computeIfNeeded(inputs);
            return cache->value().first;
          },
          stream));
  indices = createJitTensor(
      CustomNode::create(
          "sort_indices",
          NodeList(inputNodes),
          Shape(input.shape()),
          dtype::s32,
          [computeIfNeeded, cache](const std::vector<const Tensor*>& inputs) {
            computeIfNeeded(inputs);
            return cache->value().second;
          },
          stream));
}

void JitBackend::argsort(
    Tensor& out,
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode,
    std::shared_ptr<const Stream> stream) {
  out = createJitTensor(
      CustomNode::create(
          "argsort",
          tensorsToNodes(input),
          Shape(input.shape()),
          dtype::s32,
          [dispatch = dispatch_, axis, sortMode](
              const std::vector<const Tensor*>& inputs) {
            Tensor result;
            dispatch.argsort(
                dispatch.ctx, result, *inputs.at(0), axis, sortMode);
            return result;
          },
          std::move(stream)));
}

/************************** Binary Operators ***************************/
Tensor JitBackend::createBinopJitTensor(
    const Tensor& lhs,
    const Tensor& rhs,
    BinaryOp op,
    std::shared_ptr<const Stream> stream) {
  const auto lhsNode = toJitTensorBase(lhs).node();
  const auto rhsNode = toJitTensorBase(rhs).node();
  return createJitTensor(
      BinaryNode::create(lhsNode, rhsNode, op, std::move(stream)));
}

void JitBackend::binary(
    Tensor& out,
    BinaryOp op,
    const Tensor& lhs,
    const Tensor& rhs,
    std::shared_ptr<const Stream> stream) {
  out = createBinopJitTensor(lhs, rhs, op, std::move(stream));
}

/************************** BLAS ***************************/

void JitBackend::matmul(
    Tensor& out,
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp,
    std::shared_ptr<const Stream> stream) {
  const auto lhsNode = toJitTensorBase(lhs).node();
  const auto rhsNode = toJitTensorBase(rhs).node();
  out = createJitTensor(
      MatmulNode::create(
          lhsNode, rhsNode, lhsProp, rhsProp, std::move(stream)));
}

/************************** Reductions ***************************/

Tensor JitBackend::createReductionJitTensor(
    const Tensor& input,
    ReductionOp op,
    const std::vector<int>& axes,
    bool keepDims,
    std::shared_ptr<const Stream> stream) {
  const auto inputNode = toJitTensorBase(input).node();
  return createJitTensor(
      ReductionNode::create(inputNode, op, axes, keepDims, std::move(stream)));
}

void JitBackend::reduction(
    Tensor& out,
    ReductionOp op,
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims,
    std::shared_ptr<const Stream> stream) {
  out = createReductionJitTensor(input, op, axes, keepDims, std::move(stream));
}

void JitBackend::min(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims,
    std::shared_ptr<const Stream> stream) {
  auto cache = std::make_shared<std::optional<std::pair<Tensor, Tensor>>>();
  auto computeIfNeeded = [dispatch = dispatch_, axis, keepDims, cache](
                             const std::vector<const Tensor*>& inputs) {
    if (!cache->has_value()) {
      Tensor v, i;
      dispatch.min(dispatch.ctx, v, i, *inputs[0], axis, keepDims);
      cache->emplace(std::move(v), std::move(i));
    }
  };
  auto outShape = inferReductionOutputShape(
      input.shape(), {static_cast<int>(axis)}, keepDims);
  auto inputNodes = tensorsToNodes(input);
  values = createJitTensor(
      CustomNode::create(
          "min_values",
          NodeList(inputNodes),
          Shape(outShape),
          input.type(),
          [computeIfNeeded, cache](const std::vector<const Tensor*>& inputs) {
            computeIfNeeded(inputs);
            return cache->value().first;
          },
          stream));
  indices = createJitTensor(
      CustomNode::create(
          "min_indices",
          NodeList(inputNodes),
          Shape(outShape),
          dtype::u32,
          [computeIfNeeded, cache](const std::vector<const Tensor*>& inputs) {
            computeIfNeeded(inputs);
            return cache->value().second;
          },
          stream));
}

void JitBackend::max(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims,
    std::shared_ptr<const Stream> stream) {
  auto cache = std::make_shared<std::optional<std::pair<Tensor, Tensor>>>();
  auto computeIfNeeded = [dispatch = dispatch_, axis, keepDims, cache](
                             const std::vector<const Tensor*>& inputs) {
    if (!cache->has_value()) {
      Tensor v, i;
      dispatch.max(dispatch.ctx, v, i, *inputs[0], axis, keepDims);
      cache->emplace(std::move(v), std::move(i));
    }
  };
  auto outShape = inferReductionOutputShape(
      input.shape(), {static_cast<int>(axis)}, keepDims);
  auto inputNodes = tensorsToNodes(input);
  values = createJitTensor(
      CustomNode::create(
          "max_values",
          NodeList(inputNodes),
          Shape(outShape),
          input.type(),
          [computeIfNeeded, cache](const std::vector<const Tensor*>& inputs) {
            computeIfNeeded(inputs);
            return cache->value().first;
          },
          stream));
  indices = createJitTensor(
      CustomNode::create(
          "max_indices",
          NodeList(inputNodes),
          Shape(outShape),
          dtype::u32,
          [computeIfNeeded, cache](const std::vector<const Tensor*>& inputs) {
            computeIfNeeded(inputs);
            return cache->value().second;
          },
          stream));
}

void JitBackend::cumsum(
    Tensor& out,
    const Tensor& input,
    const unsigned axis,
    std::shared_ptr<const Stream> stream) {
  out = createJitTensor(
      CumsumNode::create(
          toJitTensorBase(input).node(), axis, std::move(stream)));
}

void JitBackend::argmax(
    Tensor& out,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims,
    std::shared_ptr<const Stream> stream) {
  out = createJitTensor(
      CustomNode::create(
          "argmax",
          tensorsToNodes(input),
          inferReductionOutputShape(
              input.shape(), {static_cast<int>(axis)}, keepDims),
          dtype::u32,
          [dispatch = dispatch_, axis, keepDims](
              const std::vector<const Tensor*>& inputs) {
            Tensor result;
            dispatch.argmax(
                dispatch.ctx, result, *inputs.at(0), axis, keepDims);
            return result;
          },
          std::move(stream)));
}

void JitBackend::argmin(
    Tensor& out,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims,
    std::shared_ptr<const Stream> stream) {
  out = createJitTensor(
      CustomNode::create(
          "argmin",
          tensorsToNodes(input),
          inferReductionOutputShape(
              input.shape(), {static_cast<int>(axis)}, keepDims),
          dtype::u32,
          [dispatch = dispatch_, axis, keepDims](
              const std::vector<const Tensor*>& inputs) {
            Tensor result;
            dispatch.argmin(
                dispatch.ctx, result, *inputs.at(0), axis, keepDims);
            return result;
          },
          std::move(stream)));
}

void JitBackend::var(
    Tensor& out,
    const Tensor& input,
    const std::vector<int>& axes,
    const bool bias,
    const bool keepDims,
    std::shared_ptr<const Stream> stream) {
  out = createJitTensor(
      CustomNode::create(
          "var",
          tensorsToNodes(input),
          inferReductionOutputShape(input.shape(), axes, keepDims),
          isFloatingPoint(input.type()) ? input.type() : dtype::f32,
          [dispatch = dispatch_, axes, bias, keepDims](
              const std::vector<const Tensor*>& inputs) {
            Tensor result;
            dispatch.var(
                dispatch.ctx, result, *inputs.at(0), axes, bias, keepDims);
            return result;
          },
          std::move(stream)));
}

void JitBackend::norm(
    Tensor& out,
    const Tensor& input,
    const std::vector<int>& axes,
    double p,
    const bool keepDims,
    std::shared_ptr<const Stream> stream) {
  out = createJitTensor(
      CustomNode::create(
          "norm",
          tensorsToNodes(input),
          inferReductionOutputShape(input.shape(), axes, keepDims),
          isFloatingPoint(input.type()) ? input.type() : dtype::f32,
          [dispatch = dispatch_, axes, p, keepDims](
              const std::vector<const Tensor*>& inputs) {
            Tensor result;
            dispatch.norm(
                dispatch.ctx, result, *inputs.at(0), axes, p, keepDims);
            return result;
          },
          std::move(stream)));
}

void JitBackend::copy(Tensor& dst, const Tensor& src, const Stream& stream) {
  // Imperative copy: materialize and forward to wrapped backend.
  const auto& srcR = materializeSafe(src);
  materializeMutable(dst);
  wrappedBackend().copy(dst, srcR, stream);
  // Re-wrap in a JitTensor so callers see a consistent backend type.
  dst = createJitTensor(ValueNode::create(std::move(dst)));
}

Tensor JitBackend::transfer(
    const Tensor& src,
    DeviceType dstDeviceType,
    TensorBackendType dstBackendType,
    std::shared_ptr<const Stream> dstStream) {
  auto srcNode = toJitTensorBase(src).node();

  // Nodes without an explicit stream run on this backend's default stream.
  std::shared_ptr<const Stream> srcStream = srcNode->stream();
  if (!srcStream) {
    srcStream = defaultStream_;
  }
  auto srcDevType = srcStream ? srcStream->device().type() : DeviceType::x64;

  // Same device: tensors are immutable values, so the copy can share the node.
  if (srcDevType == dstDeviceType) {
    return createJitTensor(srcNode);
  }

  // Resolve which stream the copy work runs on
  std::shared_ptr<const Stream> copyStream;
  if (dstDeviceType == DeviceType::CUDA)
    copyStream = dstStream;
  else if (srcDevType == DeviceType::CUDA)
    copyStream = srcStream;
  else
    copyStream = dstStream;

  auto transferNode =
      TransferNode::create(srcNode, dstDeviceType, dstBackendType, copyStream);
  return createJitTensor(transferNode);
}

void JitBackend::print(const Tensor& tensor) {
  std::cout << "JitTensor" << std::endl
            << toJitTensorBase(const_cast<Tensor&>(tensor)).toString()
            << std::endl;
}

} // namespace fl
