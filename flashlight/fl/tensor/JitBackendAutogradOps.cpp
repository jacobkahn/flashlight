/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/JitBackend.h"

#include "flashlight/fl/tensor/JitBackendUtils.h"
#include "flashlight/fl/tensor/JitTensorBase.h"
#include "flashlight/fl/tensor/ir/NodeUtils.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/ir/BatchNormNode.h"
#include "flashlight/fl/tensor/ir/Conv2dNode.h"
#include "flashlight/fl/tensor/ir/CustomNode.h"
#include "flashlight/fl/tensor/ir/Pool2dNode.h"
#include "flashlight/fl/tensor/ir/ProjectionNode.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"

namespace fl {

using detail::materializeSafe;
using detail::tensorsToNodes;

namespace {

// Deferred conv2d state: holds a Conv2dNodePtr so the backward methods
// can retrieve the actual ForwardBackwardState cached during evaluation.
struct DeferredConv2dState : ForwardState {
  Conv2dNodePtr node;
  explicit DeferredConv2dState(Conv2dNodePtr n) : node(std::move(n)) {}
};

// Deferred pool2d state: holds a Pool2dNodePtr so pool2dBackward
// can retrieve the actual ForwardBackwardState cached during evaluation.
struct DeferredPool2dState : ForwardState {
  Pool2dNodePtr node;
  explicit DeferredPool2dState(Pool2dNodePtr n) : node(std::move(n)) {}
};

// Deferred batchnorm state: holds a BatchNormNodePtr so batchnormBackward
// can retrieve the actual ForwardBackwardState cached during evaluation.
struct DeferredBatchnormState : ForwardState {
  BatchNormNodePtr node;
  explicit DeferredBatchnormState(BatchNormNodePtr n) : node(std::move(n)) {}
};

} // namespace

/*********************** Autograd Ops ************************/

ForwardBackwardState
JitBackend::conv2dForward(
    Tensor& out,
    const Tensor& input, const Tensor& weights, const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    bool wantState) {
  auto inputNode = toJitTensorBase(input).node();
  auto weightsNode = toJitTensorBase(weights).node();
  // bias may be empty (default-constructed, non-JIT).
  // Wrap it in a ValueNode so Conv2dNode always has 3 inputs.
  NodePtr biasNode;
  if (bias.isEmpty()) {
    // Wrap a real empty tensor — the evaluator passes this to cuDNN which
    // checks elements() > 0 to decide whether to add bias.
    biasNode = ValueNode::create(
        dispatch_.full(dispatch_.ctx, {0}, 0, input.type()));
  } else if (bias.backendType() != TensorBackendType::Jit) {
    biasNode = ValueNode::create(Tensor(bias));
  } else {
    biasNode = toJitTensorBase(bias).node();
  }

  auto conv2dNode = Conv2dNode::create(
      inputNode, weightsNode, biasNode,
      sx, sy, px, py, dx, dy, groups, wantState);

  out = createJitTensor(conv2dNode);

  if (wantState) {
    return std::make_shared<DeferredConv2dState>(conv2dNode);
  }
  return nullptr;
}

Tensor JitBackend::conv2dBackwardData(
    const Tensor& gradOutput, const Tensor& input, const Tensor& weight,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const ForwardBackwardState& state) {
  auto inputNodes = tensorsToNodes(gradOutput, input, weight);
  return createJitTensor(CustomNode::create(
      "conv2dBackwardData",
      std::move(inputNodes),
      Shape(input.shape()),
      input.type(),
      [dispatch = dispatch_, sx, sy, px, py, dx, dy, groups, state,
       eval = &evaluator_](const std::vector<const Tensor*>& inputs) {
        ForwardBackwardState actualState = state;
        if (state) {
          auto* deferred = static_cast<DeferredConv2dState*>(state.get());
          if (!deferred->node->getResult().has_value()) {
            eval->eval(deferred->node);
          }
          actualState = deferred->node->cachedState();
        }

        return dispatch.conv2dBackwardData(
            dispatch.ctx, *inputs[0], *inputs[1], *inputs[2],
            sx, sy, px, py, dx, dy, groups, actualState);
      }));
}

std::pair<Tensor, Tensor> JitBackend::conv2dBackwardFilterBias(
    const Tensor& gradOutput, const Tensor& input,
    const Tensor& weights, const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const ForwardBackwardState& state) {
  auto cache = std::make_shared<std::optional<std::pair<Tensor, Tensor>>>();
  // Collect non-empty inputs as JIT nodes.
  std::vector<NodePtr> inputNodes;
  inputNodes.push_back(toJitTensorBase(gradOutput).node());
  inputNodes.push_back(toJitTensorBase(input).node());
  inputNodes.push_back(toJitTensorBase(weights).node());
  bool hasBias = !bias.isEmpty();
  int biasIdx = -1;
  if (hasBias) {
    biasIdx = static_cast<int>(inputNodes.size());
    inputNodes.push_back(toJitTensorBase(bias).node());
  }
  auto computeIfNeeded = [dispatch = dispatch_, sx, sy, px, py, dx, dy, groups,
                           state, hasBias, biasIdx, cache,
                           eval = &evaluator_](
      const std::vector<const Tensor*>& inputs) {
    if (!cache->has_value()) {
      ForwardBackwardState actualState = state;
      if (state) {
        auto* deferred = static_cast<DeferredConv2dState*>(state.get());
        if (!deferred->node->getResult().has_value()) {
          eval->eval(deferred->node);
        }
        actualState = deferred->node->cachedState();
      }

      Tensor bs = hasBias ? *inputs[biasIdx] : Tensor();
      auto [gw, gb] = dispatch.conv2dBackwardFilterBias(
          dispatch.ctx, *inputs[0], *inputs[1], *inputs[2], bs,
          sx, sy, px, py, dx, dy, groups, actualState);
      cache->emplace(std::move(gw), std::move(gb));
    }
  };
  auto gradWeightsOut = createJitTensor(CustomNode::create(
      "conv2dBackwardFilter", NodeList(inputNodes), Shape(weights.shape()),
      weights.type(),
      [computeIfNeeded, cache](const std::vector<const Tensor*>& inputs) {
        computeIfNeeded(inputs);
        return cache->value().first;
      }));
  auto gradBiasOut = createJitTensor(CustomNode::create(
      "conv2dBackwardBias", NodeList(inputNodes),
      Shape(hasBias ? bias.shape() : weights.shape()), weights.type(),
      [computeIfNeeded, cache](const std::vector<const Tensor*>& inputs) {
        computeIfNeeded(inputs);
        return cache->value().second;
      }));
  return {std::move(gradWeightsOut), std::move(gradBiasOut)};
}

ForwardBackwardState
JitBackend::pool2dForward(
    Tensor& out,
    const Tensor& input, int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode, bool wantState) {
  // Compute output shape statically: {N, C, oH, oW}
  auto N = input.dim(0);
  auto C = input.ndim() > 1 ? input.dim(1) : 1;
  auto H = input.ndim() > 2 ? input.dim(2) : 1;
  auto W = input.ndim() > 3 ? input.dim(3) : 1;
  auto oH = 1 + (H + 2 * py - wy) / sy;
  auto oW = 1 + (W + 2 * px - wx) / sx;
  Shape outShape({N, C, oH, oW});

  auto poolNode = Pool2dNode::create(
      toJitTensorBase(input).node(),
      wx, wy, sx, sy, px, py, mode,
      outShape, input.type());

  out = createJitTensor(poolNode);
  if (wantState) {
    return std::make_shared<DeferredPool2dState>(poolNode);
  }
  return nullptr;
}

Tensor JitBackend::pool2dBackward(
    const Tensor& gradOutput, const Tensor& input, const Tensor& poolOutput,
    int wx, int wy, int sx, int sy, int px, int py,
    PoolingMode mode, const ForwardBackwardState& state) {
  auto inputNodes = tensorsToNodes(gradOutput, input, poolOutput);
  return createJitTensor(CustomNode::create(
      "pool2dBackward",
      std::move(inputNodes),
      Shape(input.shape()),
      input.type(),
      [dispatch = dispatch_, wx, wy, sx, sy, px, py, mode, state,
       eval = &evaluator_](
          const std::vector<const Tensor*>& inputs) {
        ForwardBackwardState actualState = nullptr;
        if (state) {
          auto* deferred = static_cast<DeferredPool2dState*>(state.get());
          if (!deferred->node->getResult().has_value()) {
            eval->eval(deferred->node);
          }
          actualState = deferred->node->cachedState();
        }

        return dispatch.pool2dBackward(
            dispatch.ctx, *inputs[0], *inputs[1], *inputs[2],
            wx, wy, sx, sy, px, py, mode, actualState);
      }));
}

BatchnormForwardResult
JitBackend::batchnormForward(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& runningMean, const Tensor& runningVar,
    const std::vector<int>& axes, bool train, double momentum, double epsilon,
    bool wantState) {
  // Compute output shapes statically from input metadata.
  auto outputShape = Shape(input.shape());
  auto outputType = input.type();
  int nfeatures = 1;
  for (auto ax : axes) {
    nfeatures *= input.dim(ax);
  }
  Shape statsShape({nfeatures});
  auto paramsType =
      (outputType == fl::dtype::f16) ? fl::dtype::f32 : outputType;

  // Helper: get node from JIT tensor, or create empty ValueNode for absent optionals.
  auto toNodeOrEmpty = [](const Tensor& t) -> NodePtr {
    if (t.isEmpty()) {
      return ValueNode::create(Tensor());
    }
    return toJitTensorBase(t).node();
  };

  // Always 5 inputs: input, weight, bias, runningMean, runningVar
  std::vector<NodePtr> inputNodes;
  inputNodes.push_back(toJitTensorBase(input).node());
  inputNodes.push_back(toNodeOrEmpty(weight));
  inputNodes.push_back(toNodeOrEmpty(bias));
  inputNodes.push_back(toNodeOrEmpty(runningMean));
  inputNodes.push_back(toNodeOrEmpty(runningVar));

  auto bnNode = BatchNormNode::create(
      std::move(inputNodes), axes, train, momentum, epsilon,
      {{outputShape, outputType},
       {statsShape, paramsType},
       {statsShape, paramsType},
       {statsShape, paramsType},
       {statsShape, paramsType}});

  auto deferredState = wantState
      ? std::make_shared<DeferredBatchnormState>(bnNode)
      : nullptr;

  // Extract outputs via ProjectionNodes with named indices
  bool hasRunningStats = !runningMean.isEmpty();
  Tensor newRunningMeanOut, newRunningVarOut;
  if (hasRunningStats) {
    newRunningMeanOut = createJitTensor(ProjectionNode::create(
        bnNode, BatchNormNode::NewRunningMean, statsShape, paramsType));
    newRunningVarOut = createJitTensor(ProjectionNode::create(
        bnNode, BatchNormNode::NewRunningVar, statsShape, paramsType));
  }

  return {
      createJitTensor(ProjectionNode::create(
          bnNode, BatchNormNode::Result, outputShape, outputType)),
      createJitTensor(ProjectionNode::create(
          bnNode, BatchNormNode::SaveMean, statsShape, paramsType)),
      createJitTensor(ProjectionNode::create(
          bnNode, BatchNormNode::SaveVar, statsShape, paramsType)),
      std::move(newRunningMeanOut),
      std::move(newRunningVarOut),
      std::move(deferredState)};
}

std::tuple<Tensor, Tensor, Tensor> JitBackend::batchnormBackward(
    const Tensor& gradOutput, const Tensor& saveMean, const Tensor& saveVar,
    const Tensor& input, const Tensor& weight,
    const std::vector<int>& axes, bool train, float epsilon,
    const ForwardBackwardState& state) {
  auto cache = std::make_shared<
      std::optional<std::tuple<Tensor, Tensor, Tensor>>>();

  // Collect non-empty inputs as JIT nodes.
  std::vector<NodePtr> inputNodes;
  inputNodes.push_back(toJitTensorBase(gradOutput).node());
  inputNodes.push_back(toJitTensorBase(saveMean).node());
  inputNodes.push_back(toJitTensorBase(saveVar).node());
  inputNodes.push_back(toJitTensorBase(input).node());
  bool hasWeight = !weight.isEmpty();
  int weightIdx = -1;
  if (hasWeight) {
    weightIdx = static_cast<int>(inputNodes.size());
    inputNodes.push_back(toJitTensorBase(weight).node());
  }

  auto computeIfNeeded = [dispatch = dispatch_, axes, train, epsilon, state,
                           hasWeight, weightIdx, cache,
                           eval = &evaluator_](
      const std::vector<const Tensor*>& inputs) {
    if (!cache->has_value()) {
      // Unwrap the deferred state from lazy batchnormForward.
      ForwardBackwardState actualState = nullptr;
      if (state) {
        auto* deferred = static_cast<DeferredBatchnormState*>(state.get());
        // Ensure the BatchNormNode has been evaluated so cachedState_ is populated.
        // cachedState_ persists across unsetResult() calls (mutable).
        if (!deferred->node->cachedState_) {
          // Force-evaluate the BatchNormNode and its inputs.
          // This can happen if backward runs before forward outputs were consumed.
          eval->eval(deferred->node);
        }
        actualState = deferred->node->cachedState_;
      }
      Tensor wt = hasWeight ? *inputs[weightIdx] : Tensor();

      auto result = dispatch.batchnormBackward(
          dispatch.ctx, *inputs[0], *inputs[1], *inputs[2], *inputs[3], wt,
          axes, train, epsilon, actualState);
      cache->emplace(std::move(result));
    }
  };

  auto gradInputOut = createJitTensor(CustomNode::create(
      "batchnormBackwardInput", NodeList(inputNodes), Shape(input.shape()),
      input.type(),
      [computeIfNeeded, cache](const std::vector<const Tensor*>& inputs) {
        computeIfNeeded(inputs);
        return std::get<0>(cache->value());
      }));
  auto gradWeightShape = hasWeight ? Shape(weight.shape()) : Shape({1});
  auto gradWeightType = hasWeight ? weight.type() : input.type();
  auto gradWeightOut = createJitTensor(CustomNode::create(
      "batchnormBackwardWeight", NodeList(inputNodes), gradWeightShape,
      gradWeightType,
      [computeIfNeeded, cache](const std::vector<const Tensor*>& inputs) {
        computeIfNeeded(inputs);
        return std::get<1>(cache->value());
      }));
  auto gradBiasOut = createJitTensor(CustomNode::create(
      "batchnormBackwardBias", NodeList(inputNodes), gradWeightShape,
      gradWeightType,
      [computeIfNeeded, cache](const std::vector<const Tensor*>& inputs) {
        computeIfNeeded(inputs);
        return std::get<2>(cache->value());
      }));
  return {std::move(gradInputOut), std::move(gradWeightOut),
          std::move(gradBiasOut)};
}

std::tuple<Tensor, Tensor, Tensor, ForwardBackwardState>
JitBackend::rnnForward(
    const Tensor& input, const Tensor& hiddenState, const Tensor& cellState,
    const Tensor& weights, int hiddenSize, int numLayers,
    RnnMode mode, bool bidirectional, float dropout,
    bool wantState) {
  const auto& inputR = materializeSafe(input);
  const auto& hiddenStateR = materializeSafe(hiddenState);
  const auto& cellStateR = materializeSafe(cellState);
  const auto& weightsR = materializeSafe(weights);

  auto [y, hy, cy, state] = dispatch_.rnnForward(
      dispatch_.ctx, inputR, hiddenStateR, cellStateR, weightsR,
      hiddenSize, numLayers, mode, bidirectional, dropout, wantState);
  return {
      createJitTensor(ValueNode::create(std::move(y))),
      createJitTensor(ValueNode::create(std::move(hy))),
      createJitTensor(ValueNode::create(std::move(cy))),
      state};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> JitBackend::rnnBackward(
    const Tensor& input, const Tensor& hiddenState, const Tensor& cellState,
    const Tensor& weights,
    const Tensor& dy, const Tensor& dhy, const Tensor& dcy,
    const Tensor& output, int numLayers, int hiddenSize,
    RnnMode mode, bool bidirectional, float dropProb,
    const ForwardBackwardState& state) {
  const auto& inputR = materializeSafe(input);
  const auto& hiddenStateR = materializeSafe(hiddenState);
  const auto& cellStateR = materializeSafe(cellState);
  const auto& weightsR = materializeSafe(weights);
  const auto& dyR = materializeSafe(dy);
  const auto& dhyR = materializeSafe(dhy);
  const auto& dcyR = materializeSafe(dcy);
  const auto& outputR = materializeSafe(output);

  auto [dx, dhx, dcx, dw] = dispatch_.rnnBackward(
      dispatch_.ctx, inputR, hiddenStateR, cellStateR, weightsR,
      dyR, dhyR, dcyR, outputR, numLayers, hiddenSize,
      mode, bidirectional, dropProb, state);
  return {
      createJitTensor(ValueNode::create(std::move(dx))),
      createJitTensor(ValueNode::create(std::move(dhx))),
      createJitTensor(ValueNode::create(std::move(dcx))),
      createJitTensor(ValueNode::create(std::move(dw)))};
}

} // namespace fl
