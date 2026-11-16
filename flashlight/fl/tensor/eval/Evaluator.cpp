/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/eval/Evaluator.h"

#include <chrono>
#include <functional>
#include <stack>
#include <unordered_set>
#include <vector>

#include "flashlight/fl/runtime/Device.h"
#include "flashlight/fl/tensor/BackendRegistry.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/JitTensorBase.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/ir/ClipNode.h"
#include "flashlight/fl/tensor/ir/ConcatenateNode.h"
#include "flashlight/fl/tensor/ir/CumsumNode.h"
#include "flashlight/fl/tensor/ir/FlipNode.h"
#include "flashlight/fl/tensor/ir/PadNode.h"
#include "flashlight/fl/tensor/ir/TriangularNode.h"
#include "flashlight/fl/tensor/ir/ProjectionNode.h"
#include "flashlight/fl/tensor/ir/RollNode.h"
#include "flashlight/fl/tensor/ir/TileNode.h"
#include "flashlight/fl/tensor/ir/TransferNode.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"
#include "flashlight/fl/tensor/ir/WhereNode.h"

namespace fl {

namespace {

// Compute a bitmask of which outputs of a multi-output node are live.
// A ProjectionNode user is live if it has any internal uses (other nodes
// consume it) or any external uses (a JitTensorBase references it).
// Multi-output nodes have at most ~5 ProjectionNode users, so this is O(1).
uint32_t computeLiveOutputMask(const NodePtr& node) {
  uint32_t mask = 0;
  for (const auto& use : node->uses()) {
    if (use->user().isProjection()) {
      auto& proj = use->user().impl<ProjectionNode>();
      if (!proj.uses().empty() || !proj.externalUses().empty()) {
        mask |= (1u << proj.outputIndex());
      }
    }
  }
  return mask;
}

} // namespace

Evaluator::Evaluator(BackendDispatch dispatch) : dispatch_(dispatch) {}

namespace {

// Consume the pre-assigned output buffer from a node, if any.
// Returns an empty Tensor if none was assigned.
Tensor takeOutputBuffer(const NodePtr& node) {
  if (node->getOutputBuffer().has_value()) {
    Tensor buf = std::move(
        const_cast<std::optional<Tensor>&>(node->getOutputBuffer()).value());
    node->clearOutputBuffer();
    return buf;
  }
  return Tensor();
}

} // namespace

void Evaluator::profile(std::function<void()> func, NodePtr nodePtr) {
  if (!profilerEnabled_) {
    func();
    return;
  }
  const auto start = std::chrono::high_resolution_clock::now();
  func();
  const auto end = std::chrono::high_resolution_clock::now();
  const auto durNs =
      std::chrono::duration_cast<std::chrono::duration<float>>(end - start)
          .count();
  nodeToTotTimeMs_.insert({nodePtr, durNs * 1000});
}

void Evaluator::evalBinaryNode(BinaryNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& lhs = node->lhs()->getResult().value();
    const auto& rhs = node->rhs()->getResult().value();
    Tensor out = takeOutputBuffer(node);
    evalBinaryOp(out, node->op(), lhs, rhs);
    node->setResult(std::move(out));
  };
  profile(func, node);
}

void Evaluator::evalCustomNode(CustomNodePtr node) {
  std::vector<const Tensor*> inputTensors;
  for (auto& inputNode : node->inputs()) {
    inputTensors.push_back(&inputNode->getResult().value());
  }
  std::function<void()> func = [node, inputTensors = std::move(inputTensors)] {
    if (node->isMultiOutput()) {
      auto results = node->multiOutputEvalFunc()(inputTensors);
      // Dead output elimination: only store outputs that have live
      // ProjectionNode consumers. Dead outputs get an empty Tensor —
      // any ProjectionNode that later reads a dead slot simply gets
      // an empty Tensor as its result.
      const uint32_t liveMask = computeLiveOutputMask(node);
      for (unsigned i = 0; i < results.size(); i++) {
        node->setOutput(
            i, (liveMask & (1u << i)) ? std::move(results[i]) : Tensor());
      }
    } else {
      node->setResult(node->evalFunc()(inputTensors));
    }
  };
  profile(func, node);
}

void Evaluator::evalClipNode(ClipNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& input = node->input()->getResult().value();
    const auto& low = node->low()->getResult().value();
    const auto& high = node->high()->getResult().value();
    Tensor out = takeOutputBuffer(node);
    dispatch_.clip(dispatch_.ctx, out, input, low, high);
    node->setResult(std::move(out));
  };
  profile(func, node);
}

void Evaluator::evalConcatenateNode(ConcatenateNodePtr node) {
  std::function<void()> func = [this, node] {
    std::vector<Tensor> inputTensors;
    inputTensors.reserve(node->numInputs());
    for (const auto& inputNode : node->inputs()) {
      inputTensors.push_back(inputNode->getResult().value());
    }
    Tensor out = takeOutputBuffer(node);
    dispatch_.concatenate(dispatch_.ctx, out, inputTensors, node->axis());
    node->setResult(std::move(out));
  };
  profile(func, node);
}

void Evaluator::evalCumsumNode(CumsumNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& input = node->input()->getResult().value();
    Tensor out = takeOutputBuffer(node);
    dispatch_.cumsum(dispatch_.ctx, out, input, node->axis());
    node->setResult(std::move(out));
  };
  profile(func, node);
}

void Evaluator::evalFlipNode(FlipNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& input = node->input()->getResult().value();
    Tensor out = takeOutputBuffer(node);
    dispatch_.flip(dispatch_.ctx, out, input, node->axis());
    node->setResult(std::move(out));
  };
  profile(func, node);
}

void Evaluator::evalTriangularNode(TriangularNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& input = node->input()->getResult().value();
    Tensor out = takeOutputBuffer(node);
    if (node->mode() == TriangularMode::Lower) {
      dispatch_.tril(dispatch_.ctx, out, input, node->diagonal());
    } else {
      dispatch_.triu(dispatch_.ctx, out, input, node->diagonal());
    }
    node->setResult(std::move(out));
  };
  profile(func, node);
}

void Evaluator::evalIndexNode(IndexNodePtr node) {
  std::function<void()> func = [this, node]() {
    const auto& indexedTensor = node->indexedNode()->getResult().value();
    node->setResult(indexedTensor(unwrapTensorInIndices(node->indices())));
  };
  profile(func, node);
}

void Evaluator::evalIndexedUpdateNode(IndexedUpdateNodePtr node) {
  // TODO no need to copy if indexedNode has only 1 user here
  std::function<void()> func = [this, node]() {
    auto indexedTensor = node->indexedNode()->getResult().value().copy();
    const auto& updateDataTensor = node->updateDataNode()->getResult().value();

    if (node->hasPreReshape()) {
      // Flat update: use backend's flat() for write-through.
      // Both CPUTensor and CudaTensor flat() create views that share the
      // parent buffer, so assigns scatter back to the original tensor.
      const auto& flatIdx = node->indexings().front().front();
      auto unwrappedIdx = (flatIdx.type() == detail::IndexType::Tensor)
          ? Index(toJitTensorBase(flatIdx.get<Tensor>())
                      .node()
                      ->getResult()
                      .value())
          : flatIdx;
      indexedTensor.flat(unwrappedIdx) = updateDataTensor;
    } else {
      const auto firstUnwrappedIndices =
          unwrapTensorInIndices(node->indexings().front());
      // if we do X = Y, it's a copy instead of update, thus the special case
      // here
      if (node->indexings().size() == 1) {
        indexedTensor(firstUnwrappedIndices) = updateDataTensor;
      } else {
        auto currIndexResult = indexedTensor(firstUnwrappedIndices);
        for (unsigned i = 1; i < node->indexings().size() - 1; i++) {
          const auto unwrappedIndices =
              unwrapTensorInIndices(node->indexings().at(i));
          currIndexResult = currIndexResult(unwrappedIndices);
        }
        const auto lastUnwrappedIndices =
            unwrapTensorInIndices(node->indexings().back());
        currIndexResult(lastUnwrappedIndices) = updateDataTensor;
      }
    }
    node->setResult(std::move(indexedTensor));
  };
  profile(func, node);
}

std::vector<Index> Evaluator::unwrapTensorInIndices(
    const std::vector<Index>& indices) {
  std::vector<Index> unwrappedIndices;
  for (const auto& index : indices) {
    if (index.type() == detail::IndexType::Tensor) {
      const auto tensorIndexNode = toJitTensorBase(index.get<Tensor>()).node();
      unwrappedIndices.push_back(tensorIndexNode->getResult().value());
    } else {
      unwrappedIndices.push_back(index);
    }
  }
  return unwrappedIndices;
}

void Evaluator::evalScalarNode(ScalarNodePtr node) {
  std::function<void()> func = [this, node]() {
    Tensor out = takeOutputBuffer(node);
    evalScalar(out, node);
    node->setResult(std::move(out));
  };
  profile(func, node);
}

void Evaluator::evalCastNode(CastNodePtr node) {
  std::function<void()> func = [node] {
    const auto& input = node->input()->getResult().value();
    node->setResult(input.astype(node->targetDtype()));
  };
  profile(func, node);
}

void Evaluator::evalUnaryNode(UnaryNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& input = node->input()->getResult().value();
    Tensor out = takeOutputBuffer(node);
    evalUnaryOp(out, node->op(), input);
    node->setResult(std::move(out));
  };
  profile(func, node);
}

void Evaluator::evalReshapeNode(ReshapeNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& input = node->input()->getResult().value();
    Tensor out = takeOutputBuffer(node);
    dispatch_.reshape(dispatch_.ctx, out, input, node->shape());
    node->setResult(std::move(out));
  };
  profile(func, node);
}

void Evaluator::evalReductionNode(ReductionNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& input = node->input()->getResult().value();
    Tensor out = takeOutputBuffer(node);
    evalReductionOp(out, node->op(), input, node->axes(), node->keepDims());
    node->setResult(std::move(out));
  };
  profile(func, node);
}

void Evaluator::evalTransposeNode(TransposeNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& input = node->input()->getResult().value();
    Tensor out = takeOutputBuffer(node);
    dispatch_.transpose(dispatch_.ctx, out, input, node->axes());
    node->setResult(std::move(out));
  };
  profile(func, node);
}

void Evaluator::evalFusedElementwiseNode(FusedElementwiseNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& program = node->program();
    const unsigned numInputs = node->numExternalInputs();
    // Slots: [0..numInputs) = external inputs, [numInputs..numInputs+i) = step
    // outputs
    std::vector<Tensor> slots;
    slots.reserve(numInputs + program.size());
    for (unsigned i = 0; i < numInputs; i++) {
      slots.push_back(node->inputs()[i]->getResult().value());
    }
    for (unsigned si = 0; si < program.size(); si++) {
      const auto& step = program[si];
      // Use assigned buffer for the last step only
      bool isLast = (si == program.size() - 1);
      if (step.isUnary()) {
        Tensor stepOut = isLast ? takeOutputBuffer(node) : Tensor();
        evalUnaryOp(stepOut, step.unaryOp(), slots[step.src0]);
        slots.push_back(std::move(stepOut));
      } else {
        Tensor stepOut = isLast ? takeOutputBuffer(node) : Tensor();
        evalBinaryOp(
            stepOut, step.binaryOp(), slots[step.src0], slots[step.src1]);
        slots.push_back(std::move(stepOut));
      }
    }
    node->setResult(std::move(slots.back()));
  };
  profile(func, node);
}

void Evaluator::evalMatmulNode(MatmulNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& lhs = node->lhs()->getResult().value();
    const auto& rhs = node->rhs()->getResult().value();
    Tensor out = takeOutputBuffer(node);
    dispatch_.matmul(
        dispatch_.ctx, out, lhs, rhs, node->lhsProp(), node->rhsProp());
    node->setResult(std::move(out));
  };
  profile(func, node);
}

void Evaluator::evalConv2dNode(Conv2dNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& input = node->input()->getResult().value();
    const auto& weights = node->weights()->getResult().value();
    const auto& bias = node->bias()->getResult().value();
    Tensor out = takeOutputBuffer(node);
    auto state = dispatch_.conv2dForward(
        dispatch_.ctx,
        out,
        input,
        weights,
        bias,
        node->sx(),
        node->sy(),
        node->px(),
        node->py(),
        node->dx(),
        node->dy(),
        node->groups(),
        node->wantState());
    if (node->wantState()) {
      node->setCachedState(std::move(state));
    }
    node->setResult(std::move(out));
  };
  profile(func, node);
}

void Evaluator::evalFusedReduceElementwiseNode(
    FusedReduceElementwiseNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& program = node->program();
    const unsigned numInputs = node->numExternalInputs();
    // Evaluate the elementwise program
    std::vector<Tensor> slots;
    slots.reserve(numInputs + program.size());
    for (unsigned i = 0; i < numInputs; i++) {
      slots.push_back(node->inputs()[i]->getResult().value());
    }
    for (const auto& step : program) {
      if (step.isUnary()) {
        Tensor stepOut;
        evalUnaryOp(stepOut, step.unaryOp(), slots[step.src0]);
        slots.push_back(std::move(stepOut));
      } else {
        Tensor stepOut;
        evalBinaryOp(
            stepOut, step.binaryOp(), slots[step.src0], slots[step.src1]);
        slots.push_back(std::move(stepOut));
      }
    }
    // Apply the reduction to the elementwise result — use assigned buffer
    Tensor redOut = takeOutputBuffer(node);
    evalReductionOp(
        redOut,
        node->reductionOp(),
        slots.back(),
        node->axes(),
        node->keepDims());
    node->setResult(std::move(redOut));
  };
  profile(func, node);
}

void Evaluator::evalProjectionNode(ProjectionNodePtr node) {
  // Parent was already evaluated (topological order guarantee).
  // Zero-cost: just copy the tensor reference from parent's output slot.
  node->setResult(Tensor(node->parent()->getOutput(node->outputIndex())));
}

void Evaluator::evalBatchNormNode(BatchNormNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& input = node->inputTensor()->getResult().value();
    const auto& weight = node->weight()->getResult().value();
    const auto& bias = node->bias()->getResult().value();
    const auto& rm = node->runningMean()->getResult().value();
    const auto& rv = node->runningVar()->getResult().value();

    // Dead output elimination for BatchNorm's 5 outputs:
    //   0: Result, 1: SaveMean, 2: SaveVar,
    //   3: NewRunningMean, 4: NewRunningVar
    //
    // Scan the BN node's ProjectionNode users to determine which
    // outputs are actually consumed. A projection is "live" if it has
    // any internal graph uses or any external uses (i.e. a JitTensorBase
    // holds a reference to it). Inference-mode BN typically only
    // consumes Result, making outputs 1-4 dead.
    const uint32_t liveMask = computeLiveOutputMask(node);

    // Ask the node whether any backward-pass state outputs are live.
    const bool wantState = node->needsState(liveMask);

    auto result = dispatch_.batchnormForward(
        dispatch_.ctx,
        input,
        weight,
        bias,
        rm,
        rv,
        node->axes(),
        node->train(),
        node->momentum(),
        node->epsilon(),
        wantState);

    // Result is always stored — it's the primary output.
    node->setOutput(BatchNormNode::Result, std::move(result.output));

    // For each auxiliary output: store the real tensor if the
    // corresponding bit is set in liveMask, otherwise store an empty
    // Tensor(). Any ProjectionNode that reads a dead slot will
    // propagate the empty tensor — this is safe because nothing
    // downstream actually uses it.
    auto storeOrEmpty = [&](BatchNormNode::Out idx, Tensor&& val) {
      node->setOutput(
          idx, (liveMask & (1u << idx)) ? std::move(val) : Tensor());
    };
    storeOrEmpty(BatchNormNode::SaveMean, std::move(result.saveMean));
    storeOrEmpty(BatchNormNode::SaveVar, std::move(result.saveVar));
    storeOrEmpty(
        BatchNormNode::NewRunningMean, std::move(result.newRunningMean));
    storeOrEmpty(BatchNormNode::NewRunningVar, std::move(result.newRunningVar));

    // Only cache the opaque forward state when we requested it.
    if (wantState) {
      node->cachedState_ = std::move(result.state);
    }
  };
  profile(func, node);
}

void Evaluator::evalRollNode(RollNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& input = node->input()->getResult().value();
    Tensor out = takeOutputBuffer(node);
    dispatch_.roll(dispatch_.ctx, out, input, node->shift(), node->axis());
    node->setResult(std::move(out));
  };
  profile(func, node);
}

void Evaluator::evalPadNode(PadNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& input = node->input()->getResult().value();
    Tensor out = takeOutputBuffer(node);
    dispatch_.pad(
        dispatch_.ctx, out, input, node->padWidths(), node->padType());
    node->setResult(std::move(out));
  };
  profile(func, node);
}

void Evaluator::evalPool2dNode(Pool2dNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& input = node->input()->getResult().value();
    Tensor out = takeOutputBuffer(node);
    auto state = dispatch_.pool2dForward(
        dispatch_.ctx,
        out,
        input,
        node->wx(),
        node->wy(),
        node->sx(),
        node->sy(),
        node->px(),
        node->py(),
        node->mode(),
        /*wantState=*/true);

    node->setCachedState(std::move(state));
    node->setResult(std::move(out));
  };
  profile(func, node);
}

void Evaluator::evalTileNode(TileNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& input = node->input()->getResult().value();
    Tensor out = takeOutputBuffer(node);
    dispatch_.tile(dispatch_.ctx, out, input, node->tileDims());
    node->setResult(std::move(out));
  };
  profile(func, node);
}

void Evaluator::evalWhereNode(WhereNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& condition = node->condition()->getResult().value();
    const auto& x = node->x()->getResult().value();
    const auto& y = node->y()->getResult().value();
    Tensor out = takeOutputBuffer(node);
    dispatch_.where(dispatch_.ctx, out, condition, x, y);
    node->setResult(std::move(out));
  };
  profile(func, node);
}

void Evaluator::evalBinaryOp(
    Tensor& out,
    BinaryOp op,
    const Tensor& lhs,
    const Tensor& rhs) {
  dispatch_.binary(dispatch_.ctx, out, op, lhs, rhs);
}

void Evaluator::evalUnaryOp(Tensor& out, UnaryOp op, const Tensor& input) {
  dispatch_.unary(dispatch_.ctx, out, op, input);
}

void Evaluator::evalReductionOp(
    Tensor& out,
    ReductionOp op,
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims) {
  dispatch_.reduction(dispatch_.ctx, out, op, input, axes, keepDims);
}

void Evaluator::evalScalar(Tensor& out, ScalarNodePtr node) {
  const Shape& shape = node->shape();
  const auto dtype = node->dataType();
  switch (dtype) {
    case dtype::f16:
    case dtype::bf16:
    case dtype::f32:
    case dtype::f64:
      out = dispatch_.full(dispatch_.ctx, shape, node->scalar<double>(), dtype);
      return;
    case dtype::b8:
    case dtype::s16:
    case dtype::s32:
    case dtype::s64:
    case dtype::u8:
    case dtype::u16:
    case dtype::u32:
      out = dispatch_.full(
          dispatch_.ctx, shape, node->scalar<long long>(), dtype);
      return;
    case dtype::u64:
      out = dispatch_.full(
          dispatch_.ctx, shape, node->scalar<unsigned long long>(), dtype);
      return;
  }
  throw std::runtime_error("Unknown dtype");
}

void Evaluator::evalTransferNode(TransferNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& src = node->input()->getResult().value();
    auto& targetBackend = fl::getBackend(node->dstBackendType());

    // Dispatch copy to the right backend:
    // The backend with device-specific primitives handles the copy.
    auto srcDevType = src.stream().device().type();
    auto dstDevType = node->dstDeviceType();
    TensorBackend* copyBackend;
    if (dstDevType != DeviceType::x64)
      copyBackend = &targetBackend;
    else if (srcDevType != DeviceType::x64)
      copyBackend = &src.backend();
    else
      copyBackend = &targetBackend;

    // If a buffer was pre-assigned (e.g. by device()), use it. Device
    // destinations are allocated on the target backend here, since copy()
    // only sees the source and the copy stream. Host destinations are left
    // empty so copy() can pick the optimal host memory (e.g. pinned for D2H).
    Tensor dst = takeOutputBuffer(node);
    if (dst.isEmpty() && dstDevType != DeviceType::x64) {
      dst = targetBackend.empty(node->shape(), node->dataType());
    }
    copyBackend->copy(dst, src, *node->stream());
    node->setResult(std::move(dst));
  };
  profile(func, node);
}

void Evaluator::evalNodeDispatch(NodePtr node) {
  switch (node->type()) {
    case NodeType::Binary:
      return evalBinaryNode(Node::cast<BinaryNodePtr>(node));
    case NodeType::Cast:
      return evalCastNode(Node::cast<CastNodePtr>(node));
    case NodeType::Clip:
      return evalClipNode(Node::cast<ClipNodePtr>(node));
    case NodeType::Concatenate:
      return evalConcatenateNode(Node::cast<ConcatenateNodePtr>(node));
    case NodeType::Custom:
      return evalCustomNode(Node::cast<CustomNodePtr>(node));
    case NodeType::Cumsum:
      return evalCumsumNode(Node::cast<CumsumNodePtr>(node));
    case NodeType::Flip:
      return evalFlipNode(Node::cast<FlipNodePtr>(node));
    case NodeType::Index:
      return evalIndexNode(Node::cast<IndexNodePtr>(node));
    case NodeType::IndexedUpdate:
      return evalIndexedUpdateNode(Node::cast<IndexedUpdateNodePtr>(node));
    case NodeType::Scalar:
      return evalScalarNode(Node::cast<ScalarNodePtr>(node));
    case NodeType::Unary:
      return evalUnaryNode(Node::cast<UnaryNodePtr>(node));
    case NodeType::Reshape:
      return evalReshapeNode(Node::cast<ReshapeNodePtr>(node));
    case NodeType::Reduction:
      return evalReductionNode(Node::cast<ReductionNodePtr>(node));
    case NodeType::Roll:
      return evalRollNode(Node::cast<RollNodePtr>(node));
    case NodeType::Transpose:
      return evalTransposeNode(Node::cast<TransposeNodePtr>(node));
    case NodeType::Triangular:
      return evalTriangularNode(Node::cast<TriangularNodePtr>(node));
    case NodeType::FusedElementwise:
      return evalFusedElementwiseNode(
          Node::cast<FusedElementwiseNodePtr>(node));
    case NodeType::FusedReduceElementwise:
      return evalFusedReduceElementwiseNode(
          Node::cast<FusedReduceElementwiseNodePtr>(node));
    case NodeType::Matmul:
      return evalMatmulNode(Node::cast<MatmulNodePtr>(node));
    case NodeType::Conv2d:
      return evalConv2dNode(Node::cast<Conv2dNodePtr>(node));
    case NodeType::Projection:
      return evalProjectionNode(Node::cast<ProjectionNodePtr>(node));
    case NodeType::BatchNorm:
      return evalBatchNormNode(Node::cast<BatchNormNodePtr>(node));
    case NodeType::Pad:
      return evalPadNode(Node::cast<PadNodePtr>(node));
    case NodeType::Pool2d:
      return evalPool2dNode(Node::cast<Pool2dNodePtr>(node));
    case NodeType::Transfer:
      return evalTransferNode(Node::cast<TransferNodePtr>(node));
    case NodeType::Tile:
      return evalTileNode(Node::cast<TileNodePtr>(node));
    case NodeType::Where:
      return evalWhereNode(Node::cast<WhereNodePtr>(node));
    case NodeType::Value:
      return; // already has a result
    case NodeType::MultiValue:
      return; // already has results
  }
  throw std::runtime_error("[Evaluator::evalNodeDispatch] Unknown node type");
}

struct Evaluator::EvalPlan {
  std::vector<NodePtr> topoOrder;
  std::unordered_map<NodePtr, unsigned> useCount;
  std::vector<std::unordered_set<const Stream*>> syncEdges;
};

Evaluator::EvalPlan Evaluator::buildPlan(NodePtr root) {
  EvalPlan plan;

  // Check if a node has been evaluated — works for both single-output
  // and multi-output nodes.
  auto isEvaluated = [](const NodePtr& node) -> bool {
    if (node->getResult().has_value()) {
      return true;
    }
    // Multi-output nodes store results in outputs_ instead of result_.
    if (node->numOutputs() > 0 && node->hasOutput(0)) {
      return true;
    }
    return false;
  };

  // Iterative post-order DFS
  std::unordered_set<Node*> visited;
  // Stack entries: (node, expanded). When expanded=false, we push inputs.
  // When expanded=true, all inputs have been processed — emit the node.
  std::stack<std::pair<NodePtr, bool>> stack;
  stack.push({root, false});

  auto getUseCount = [](const NodePtr& node) -> unsigned {
    return node->uses().size() + node->externalUses().size();
  };

  while (!stack.empty()) {
    auto& [node, expanded] = stack.top();

    // Skip nodes that already have results
    if (isEvaluated(node)) {
      stack.pop();
      continue;
    }

    if (!expanded) {
      if (!visited.insert(node.get()).second) {
        stack.pop();
        continue;
      }
      expanded = true;
      // Push inputs in reverse order so they're processed left-to-right
      const auto& inputs = node->inputs();
      for (int i = static_cast<int>(inputs.size()) - 1; i >= 0; i--) {
        if (!isEvaluated(inputs[i])) {
          stack.push({inputs[i], false});
        }
      }
    } else {
      // Copy before popping: `node` refers into the popped stack entry.
      NodePtr current = node;
      stack.pop();

      // Record use count
      plan.useCount.emplace(current, getUseCount(current));

      // Compute sync edges for cross-stream dependencies
      std::unordered_set<const Stream*> syncs;
      const auto& targetStream = current->stream();
      if (targetStream) {
        for (const auto& input : current->inputs()) {
          const auto& inputStream = input->stream();
          if (inputStream && inputStream.get() != targetStream.get()) {
            syncs.insert(inputStream.get());
          }
        }
      }

      plan.topoOrder.push_back(current);
      plan.syncEdges.push_back(std::move(syncs));
    }
  }

  return plan;
}

void Evaluator::executePlan(EvalPlan& plan) {
  for (unsigned i = 0; i < plan.topoOrder.size(); i++) {
    const auto& node = plan.topoOrder[i];

    // Synchronize cross-stream dependencies
    if (!plan.syncEdges[i].empty()) {
      node->stream()->relativeSync(plan.syncEdges[i]);
    }

    // Allocate the output just before dispatch, so only buffers of nodes that
    // are live at this point in the plan exist at once.
    assignBuffer(node);

    // Dispatch the operation
    evalNodeDispatch(node);

    // Decrement use counts and free dead intermediates
    for (const auto& input : node->inputs()) {
      auto it = plan.useCount.find(input);
      if (it != plan.useCount.end()) {
        it->second--;
        if (it->second == 0 && !input->isValue()) {
          input->unsetResult();
        }
      }
    }
  }
}

void Evaluator::assignBuffer(const NodePtr& node) {
  if (node->getResult().has_value()) {
    return; // already evaluated
  }
  if (node->getOutputBuffer().has_value()) {
    return; // already assigned (e.g. by device())
  }

  // View-like nodes don't allocate output buffers — they alias input memory.
  // Also skip multi-output nodes (BatchNorm, Custom multi-output),
  // projections (they read from parent's output slots), and transfers
  // (output backend differs from dispatch backend; evalTransferNode
  // allocates via the copy backend's emptyCopyDst). Custom, IndexedUpdate and
  // Scalar nodes produce their own result tensor, so a pre-assigned buffer
  // would only be dead weight.
  if (node->isReshape() || node->isTranspose() || node->isIndex() ||
      node->isProjection() || node->isCast() || node->isValue() ||
      node->isMultiValue() || node->isTransfer() || node->numOutputs() > 0 ||
      node->isCustom() || node->isIndexedUpdate() || node->isScalar()) {
    return;
  }

  // Allocate via TensorBackend::empty with the node's stream.
  // Uses virtual dispatch (allocation is not in the execution hot loop).
  Tensor buffer =
      dispatch_.ctx->empty(node->shape(), node->dataType(), node->stream());
  node->setOutputBuffer(std::move(buffer));
}

void Evaluator::eval(NodePtr node) {
  auto plan = buildPlan(node);
  executePlan(plan);
  for (const auto& callback : postEvalCallbacks_) {
    callback(node, nodeToTotTimeMs_);
  }
  nodeToTotTimeMs_.clear();
}

void Evaluator::setProfilerState(bool active) {
  this->profilerEnabled_ = active;
}

bool Evaluator::getProfilerState() {
  return this->profilerEnabled_;
}

const std::unordered_map<NodePtr, float>& Evaluator::getProfilerStats() {
  return nodeToTotTimeMs_;
}

void Evaluator::clearProfilerStats() {
  nodeToTotTimeMs_.clear();
}

Evaluator::PostEvalCallbackHandle Evaluator::addPostEvalCallback(
    PostEvalCallback callback) {
  return postEvalCallbacks_.insert(postEvalCallbacks_.end(), callback);
}

void Evaluator::removePostEvalCallback(PostEvalCallbackHandle handle) {
  postEvalCallbacks_.erase(handle);
}

} // namespace fl
