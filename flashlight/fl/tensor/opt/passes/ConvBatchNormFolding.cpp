/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/opt/passes/ConvBatchNormFolding.h"

#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/ir/BatchNormNode.h"
#include "flashlight/fl/tensor/ir/Conv2dNode.h"
#include "flashlight/fl/tensor/ir/ProjectionNode.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"

namespace fl {

namespace {

// Get the concrete tensor from a ValueNode, or std::nullopt.
std::optional<Tensor> getValueTensor(const NodePtr& node) {
  if (node->isValue()) {
    return node->impl<ValueNode>().value();
  }
  return std::nullopt;
}

} // namespace

NodePtr ConvBatchNormFolding::rewrite(NodePtr node) {
  // Match: Projection(BN, Result)
  if (!node->isProjection()) {
    return node;
  }
  auto& proj = node->impl<ProjectionNode>();
  if (proj.outputIndex() != BatchNormNode::Result) {
    return node;
  }
  auto parent = proj.parent();
  if (!parent->isBatchNorm()) {
    return node;
  }
  auto& bn = parent->impl<BatchNormNode>();

  // Only fold inference-mode BN.
  if (bn.train()) {
    return node;
  }

  // BN must normalize over the channel axis only (axis 1 in NCHW).
  if (bn.axes() != std::vector<int>{1}) {
    return node;
  }

  // BN's input must be a Conv2dNode.
  auto convNode = bn.inputTensor();
  if (!convNode->isConv2d()) {
    return node;
  }

  // Conv2d must have single use (only feeds this BN).
  auto convUses = convNode->uses().size() + convNode->externalUses().size();
  if (convUses != 1) {
    return node;
  }

  // Extract conv parameters — weights must be concrete ValueNodes.
  auto convWeightsOpt = getValueTensor(convNode->impl<Conv2dNode>().weights());
  if (!convWeightsOpt) {
    return node;
  }
  const auto& convWeights = *convWeightsOpt; // {outC, inC/groups, kH, kW}

  // Conv bias: may be empty (no bias).
  Tensor convBias;
  auto convBiasOpt = getValueTensor(convNode->impl<Conv2dNode>().bias());
  if (convBiasOpt && !convBiasOpt->isEmpty()) {
    convBias = *convBiasOpt;
  }

  // Extract BN parameters — runningMean and runningVar must be concrete.
  auto meanOpt = getValueTensor(bn.runningMean());
  auto varOpt = getValueTensor(bn.runningVar());
  if (!meanOpt || meanOpt->isEmpty() || !varOpt || varOpt->isEmpty()) {
    return node;
  }
  // Flatten to 1D {C} for arithmetic.
  Tensor mean = meanOpt->flatten();
  Tensor var = varOpt->flatten();

  // BN weight (gamma): defaults to 1.0 if empty. Defaults are created on the
  // weights' own backend: the other operands are bare backend tensors, and
  // fl::full would create a JIT tensor.
  auto& backend = convWeights.backend();
  const Shape channels({convWeights.dim(0)});
  Tensor gamma;
  auto gammaOpt = getValueTensor(bn.weight());
  if (gammaOpt && !gammaOpt->isEmpty()) {
    gamma = gammaOpt->flatten();
  } else {
    gamma = backend.full(channels, 1.0, convWeights.type());
  }

  // BN bias (beta): defaults to 0.0 if empty.
  Tensor beta;
  auto betaOpt = getValueTensor(bn.bias());
  if (betaOpt && !betaOpt->isEmpty()) {
    beta = betaOpt->flatten();
  } else {
    beta = backend.full(channels, 0.0, convWeights.type());
  }

  // Compute folded parameters:
  //   scale = gamma / sqrt(var + eps)
  //   w_new = w * scale.reshape(outC, 1, 1, 1)
  //   b_new = (b - mean) * scale + beta
  Tensor scale = gamma / fl::sqrt(var + bn.epsilon());

  // Reshape scale for broadcasting with weights: {outC, 1, 1, 1}
  Tensor scaleReshaped = fl::reshape(scale, {convWeights.dim(0), 1, 1, 1});
  Tensor newWeights = convWeights * scaleReshaped;

  // Compute new bias.
  Tensor newBias;
  if (!convBias.isEmpty()) {
    newBias = (convBias.flatten() - mean) * scale + beta;
  } else {
    newBias = (-mean) * scale + beta;
  }

  // Create new ValueNodes for the folded parameters.
  auto newWeightsNode = ValueNode::create(std::move(newWeights));
  auto newBiasNode = ValueNode::create(
      fl::reshape(newBias, {1, newBias.dim(0), 1, 1}));

  // Create new Conv2dNode with folded weights/bias.
  auto& conv = convNode->impl<Conv2dNode>();
  return Conv2dNode::create(
      conv.input(), newWeightsNode, newBiasNode,
      conv.sx(), conv.sy(), conv.px(), conv.py(),
      conv.dx(), conv.dy(), conv.groups(), conv.wantState(),
      node->stream());
}

} // namespace fl
