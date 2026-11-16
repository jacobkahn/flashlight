/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/opt/backends/onednn/OneDnnPrimitivePostOpFusion.h"

#include <cmath>
#include <optional>

#include <dnnl.hpp>

#include "flashlight/fl/tensor/backend/cpu/CPUBackend.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/ops/onednn/OneDnnOps.h"
#include "flashlight/fl/tensor/ir/BatchNormNode.h"
#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/Conv2dNode.h"
#include "flashlight/fl/tensor/ir/CustomNode.h"
#include "flashlight/fl/tensor/ir/MatmulNode.h"
#include "flashlight/fl/tensor/ir/Pool2dNode.h"
#include "flashlight/fl/tensor/ir/ProjectionNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/ops/onednn/OneDnnAutogradOps.h"
#include "flashlight/fl/tensor/ops/onednn/Utils.h"
#include "flashlight/fl/tensor/opt/ActivationPatterns.h"
#include "flashlight/fl/tensor/opt/backends/onednn/OneDnnOpFusion.h"

namespace fl {

namespace {

// OneDNN dnnl_types.h: DNNL_MAX_POST_OPS = 32
static constexpr unsigned kMaxPostOps = 32;

// Try to match an activation applied to some inner node.
// Returns the eltwise info if the node is an activation we can fuse.
std::optional<EltwiseInfo> tryMatchActivation(const NodePtr& node) {
  // Case 1: UnaryNode with a OneDNN eltwise mapping
  if (node->isUnary()) {
    return tryUnaryOpToOneDnnEltwise(node->impl<UnaryNode>().op());
  }
  // Case 2: max(x, Scalar(0)) or max(Scalar(0), x) -> relu
  if (tryMatchRelu(node)) {
    return EltwiseInfo{dnnl::algorithm::eltwise_relu, 0, 0};
  }
  // Case 3: composed gelu subgraph -> eltwise_gelu_erf
  if (tryMatchGeluErf(node)) {
    return EltwiseInfo{dnnl::algorithm::eltwise_gelu_erf, 0, 0};
  }
  // Case 4: x * sigmoid(x) -> swish (silu)
  if (tryMatchSilu(node)) {
    return EltwiseInfo{dnnl::algorithm::eltwise_swish, 1, 0};
  }
  return std::nullopt;
}

// Get the "inner" operand of an activation node (the conv2d/pool2d candidate).
NodePtr getActivationInput(const NodePtr& node) {
  if (node->isUnary()) {
    return node->impl<UnaryNode>().input();
  }
  if (auto x = tryMatchRelu(node)) {
    return x;
  }
  if (auto x = tryMatchGeluErf(node)) {
    return x;
  }
  if (auto x = tryMatchSilu(node)) {
    return x;
  }
  return nullptr;
}

// Check single-use gate: the conv2d/pool2d should have exactly one consumer.
bool hasSingleUse(const NodePtr& node) {
  return !isEvaluated(*node) &&
      node->uses().size() + node->externalUses().size() == 1;
}

// Try to extract conv2d + other from a BinaryNode(Add).
// Returns {conv, other} if one operand is a single-use Conv2dNode.
std::optional<std::pair<Conv2dNodePtr, NodePtr>> tryMatchConv2dAdd(
    const NodePtr& addNode) {
  if (!addNode->isBinary()) {
    return std::nullopt;
  }
  const auto& bin = addNode->impl<BinaryNode>();
  if (bin.op() != BinaryOp::Add) {
    return std::nullopt;
  }

  auto tryExtract = [&addNode](const NodePtr& candidate, const NodePtr& other)
      -> std::optional<std::pair<Conv2dNodePtr, NodePtr>> {
    if (candidate->isConv2d() && hasSingleUse(candidate) &&
        other->shape() == addNode->shape()) {
      return std::make_pair(Node::cast<Conv2dNodePtr>(candidate), other);
    }
    return std::nullopt;
  };

  // Check both operand orders (add is commutative).
  if (auto r = tryExtract(bin.lhs(), bin.rhs())) {
    return r;
  }
  return tryExtract(bin.rhs(), bin.lhs());
}

// Check if a bias shape is valid for oneDNN matmul's native bias parameter.
// Native bias requires all dims except the last to be 1.
// E.g. for output (M, N): bias (N,) or (1, N). For (B, M, N): (N,) or (1, 1, N).
bool isValidMatmulBias(const Shape& outputShape, const Shape& biasShape) {
  if (biasShape.ndim() == 0 || outputShape.ndim() == 0) {
    return false;
  }
  // Last dim must match output's last dim.
  if (biasShape.dim(biasShape.ndim() - 1) != outputShape.dim(outputShape.ndim() - 1)) {
    return false;
  }
  // All other dims must be 1.
  for (int i = 0; i < biasShape.ndim() - 1; i++) {
    if (biasShape.dim(i) != 1) {
      return false;
    }
  }
  return true;
}

// Try to extract matmul + bias from a BinaryNode(Add).
// Returns {matmul, bias} if one operand is a single-use MatmulNode and the
// other is a valid native matmul bias (all dims except last are 1).
std::optional<std::pair<MatmulNodePtr, NodePtr>> tryMatchMatmulAdd(
    const NodePtr& addNode) {
  if (!addNode->isBinary()) {
    return std::nullopt;
  }
  const auto& bin = addNode->impl<BinaryNode>();
  if (bin.op() != BinaryOp::Add) {
    return std::nullopt;
  }

  auto tryExtract = [](const NodePtr& candidate, const NodePtr& other)
      -> std::optional<std::pair<MatmulNodePtr, NodePtr>> {
    if (candidate->isMatmul() && hasSingleUse(candidate) &&
        isValidMatmulBias(candidate->shape(), other->shape())) {
      return std::make_pair(Node::cast<MatmulNodePtr>(candidate), other);
    }
    return std::nullopt;
  };

  // Check both operand orders (add is commutative).
  if (auto r = tryExtract(bin.lhs(), bin.rhs())) {
    return r;
  }
  return tryExtract(bin.rhs(), bin.lhs());
}

// Collected chain info from top-down walk.
struct FusedChain {
  enum AnchorKind { Conv2d, Pool2d, BatchNorm, Matmul, Standalone };

  AnchorKind anchorKind;
  NodePtr anchorNode; // Conv2dNode, Pool2dNode, or ProjectionNode(BN)
  std::vector<EltwiseInfo> eltwiseChain; // collected top-down (outermost first)
  NodePtr sumSrc; // non-null only for conv2d + sum fusion
  NodePtr biasSrc; // non-null only for matmul + native bias fusion
};

// Walk a chain from `node` downward through single-use eltwise activations,
// optionally through an add (sum post-op), to an anchor primitive.
// Returns the collected chain if fusable, nullopt otherwise.
std::optional<FusedChain> findAnchorChain(const NodePtr& node) {
  FusedChain chain;

  // Step 1: Walk down through single-use eltwise activations.
  // Reserve one slot for a potential sum post-op (added in Step 2).
  NodePtr cur = node;
  // Set when the walk stops at an input that other nodes also read. Such an
  // input can feed the chain but must not be absorbed into the fused node.
  bool curIsShared = false;
  while (chain.eltwiseChain.size() < kMaxPostOps - 1) {
    auto elt = tryMatchActivation(cur);
    if (!elt) {
      break;
    }
    auto inner = getActivationInput(cur);
    if (!inner) {
      break;
    }
    // The chain must describe exactly the activations between `node` and
    // `cur`, so record the activation and step past it together.
    chain.eltwiseChain.push_back(*elt);
    cur = inner;
    // Only continue walking through `inner` if nothing outside the chain
    // reads it. For composed activations (gelu, silu), x appears twice in the
    // pattern (outer mul + sigmoid/erf input), so x has exactly 2 uses.
    if (!hasSingleUse(inner)) {
      bool isComposed = (elt->alg == dnnl::algorithm::eltwise_gelu_erf ||
                         elt->alg == dnnl::algorithm::eltwise_swish);
      unsigned totalUses =
          inner->uses().size() + inner->externalUses().size();
      if (!isComposed || totalUses != 2) {
        curIsShared = true;
        break;
      }
    }
  }

  // `cur` is now the first non-activation node in the chain.
  // If we collected any eltwise ops, `cur` is the node below the bottommost
  // activation. If we collected none, `cur == node` (unchanged).

  // Step 2: If the bottom of the eltwise chain is an add with a single-use
  // conv2d operand, record a sum post-op and advance to the conv2d.
  if (!chain.eltwiseChain.empty() && !curIsShared) {
    if (auto convAdd = tryMatchConv2dAdd(cur)) {
      auto& [convNode, sumSrcNode] = *convAdd;
      chain.anchorKind = FusedChain::Conv2d;
      chain.anchorNode = convNode;
      chain.sumSrc = sumSrcNode;
      // Eltwise chain is outermost-first; reverse to get execution order.
      std::reverse(chain.eltwiseChain.begin(), chain.eltwiseChain.end());
      return chain;
    }
    // Check for add(matmul, bias) — fuse bias into matmul native bias param.
    if (auto mmAdd = tryMatchMatmulAdd(cur)) {
      auto& [mmNode, biasSrcNode] = *mmAdd;
      chain.anchorKind = FusedChain::Matmul;
      chain.anchorNode = mmNode;
      chain.biasSrc = biasSrcNode;
      std::reverse(chain.eltwiseChain.begin(), chain.eltwiseChain.end());
      return chain;
    }
  }

  // Step 3: If no eltwise was found, check if the node itself is
  // add(conv2d, x) for sum-only fusion.
  if (chain.eltwiseChain.empty()) {
    if (auto convAdd = tryMatchConv2dAdd(node)) {
      auto& [convNode, sumSrcNode] = *convAdd;
      chain.anchorKind = FusedChain::Conv2d;
      chain.anchorNode = convNode;
      chain.sumSrc = sumSrcNode;
      return chain;
    }
    // Check for add(matmul, bias) — bias-only fusion (no eltwise).
    if (auto mmAdd = tryMatchMatmulAdd(node)) {
      auto& [mmNode, biasSrcNode] = *mmAdd;
      chain.anchorKind = FusedChain::Matmul;
      chain.anchorNode = mmNode;
      chain.biasSrc = biasSrcNode;
      return chain;
    }
  }

  // Step 4: Check if `cur` is an anchor primitive.
  if (chain.eltwiseChain.empty()) {
    // No eltwise collected, nothing to fuse.
    return std::nullopt;
  }

  // Reverse eltwise chain to execution order (innermost first).
  std::reverse(chain.eltwiseChain.begin(), chain.eltwiseChain.end());

  // Anchors are absorbed into the fused node, so a shared `cur` can only be
  // the input of a standalone lowering (Step 5).
  if (cur->isConv2d() && !curIsShared) {
    chain.anchorKind = FusedChain::Conv2d;
    chain.anchorNode = cur;
    return chain;
  }

  if (cur->type() == NodeType::Pool2d && !curIsShared) {
    chain.anchorKind = FusedChain::Pool2d;
    chain.anchorNode = cur;
    return chain;
  }

  // Projection(BN, Result) where bn.train() == false
  if (cur->isProjection() && !curIsShared) {
    auto& proj = cur->impl<ProjectionNode>();
    if (proj.outputIndex() == BatchNormNode::Result &&
        proj.parent()->isBatchNorm()) {
      auto bnNode = Node::cast<BatchNormNodePtr>(proj.parent());
      if (!bnNode->train()) {
        chain.anchorKind = FusedChain::BatchNorm;
        chain.anchorNode = cur; // keep the projection node as anchor
        return chain;
      }
    }
  }

  // For matmul anchor, allow 2 uses if a composed activation consumed both.
  auto matmulUsesOk = [&]() {
    if (hasSingleUse(cur)) return true;
    // Composed activations (gelu, silu) use x twice in the pattern. If the
    // eltwise chain contains one, allow the anchor to have exactly 2 uses.
    for (const auto& elt : chain.eltwiseChain) {
      if (elt.alg == dnnl::algorithm::eltwise_gelu_erf ||
          elt.alg == dnnl::algorithm::eltwise_swish) {
        unsigned totalUses =
            cur->uses().size() + cur->externalUses().size();
        return totalUses == 2;
      }
    }
    return false;
  };
  if (cur->isMatmul() && !curIsShared && matmulUsesOk()) {
    chain.anchorKind = FusedChain::Matmul;
    chain.anchorNode = cur;
    return chain;
  }

  // Step 5: No anchor primitive found. Standalone gelu is worth lowering
  // (7 composed nodes → 1 OneDNN eltwise_gelu_erf kernel).
  for (const auto& elt : chain.eltwiseChain) {
    if (elt.alg == dnnl::algorithm::eltwise_gelu_erf) {
      chain.anchorKind = FusedChain::Standalone;
      chain.anchorNode = cur; // the input to the eltwise chain
      return chain;
    }
  }

  // No anchor found.
  return std::nullopt;
}

// Build a fused CustomNode for conv2d anchor (with optional sum + eltwise chain).
NodePtr buildFusedConv2d(const NodePtr& topNode, const FusedChain& chain) {
  auto conv = Node::cast<Conv2dNodePtr>(chain.anchorNode);
  std::vector<NodePtr> inputs = {conv->input(), conv->weights(), conv->bias()};
  if (chain.sumSrc) {
    inputs.push_back(chain.sumSrc);
  }

  auto evalFunc = [convNode = conv,
                   eltwiseChain = chain.eltwiseChain,
                   hasSumSrc = bool(chain.sumSrc)](
                      const std::vector<const Tensor*>& inputTensors) -> Tensor {
    const auto& input = *inputTensors[0];
    const auto& weights = *inputTensors[1];
    const auto& bias = *inputTensors[2];

    auto& backend = CPUBackend::getInstance();
    auto& engine = backend.onednnOps().engine();
    auto& nativeStream = backend.nativeStream();
    auto asyncStream = backend.asyncStream();

    dnnl::post_ops postOps;
    if (hasSumSrc) {
      postOps.append_sum(1.0f);
    }
    for (const auto& elt : eltwiseChain) {
      postOps.append_eltwise(elt.alg, elt.alpha, elt.beta);
    }
    dnnl::primitive_attr attr;
    attr.set_post_ops(postOps);

    Tensor sumSrc;
    if (hasSumSrc) {
      sumSrc = *inputTensors[3];
    }

    Tensor output;
    auto state = detail::onednn::conv2dForward(
        output, input, weights, bias,
        convNode->sx(), convNode->sy(),
        convNode->px(), convNode->py(),
        convNode->dx(), convNode->dy(),
        convNode->groups(), convNode->wantState(),
        engine, *asyncStream, nativeStream, attr, sumSrc);

    if (convNode->wantState()) {
      convNode->setCachedState(std::move(state));
      convNode->setResult(Tensor(output));
    }

    return output;
  };

  return CustomNode::create(
      "OneDnnFusedConv2dChain",
      std::move(inputs),
      topNode->shape(),
      topNode->dataType(),
      std::move(evalFunc),
      topNode->stream());
}

// Build a fused CustomNode for pool2d anchor (eltwise chain only, no sum).
NodePtr buildFusedPool2d(const NodePtr& topNode, const FusedChain& chain) {
  auto pool = Node::cast<Pool2dNodePtr>(chain.anchorNode);
  std::vector<NodePtr> inputs = {pool->input()};

  auto evalFunc = [poolNode = pool, eltwiseChain = chain.eltwiseChain](
                      const std::vector<const Tensor*>& inputTensors) -> Tensor {
    const auto& input = *inputTensors[0];

    auto& backend = CPUBackend::getInstance();
    auto& engine = backend.onednnOps().engine();
    auto& nativeStream = backend.nativeStream();
    auto asyncStream = backend.asyncStream();

    dnnl::post_ops postOps;
    for (const auto& elt : eltwiseChain) {
      postOps.append_eltwise(elt.alg, elt.alpha, elt.beta);
    }
    dnnl::primitive_attr attr;
    attr.set_post_ops(postOps);

    Tensor output;
    auto state = detail::onednn::pool2dForward(
        output, input,
        poolNode->wx(), poolNode->wy(),
        poolNode->sx(), poolNode->sy(),
        poolNode->px(), poolNode->py(),
        poolNode->mode(), /*wantState=*/true,
        engine, *asyncStream, nativeStream, attr);

    poolNode->setCachedState(std::move(state));
    poolNode->setResult(Tensor(output));

    return output;
  };

  return CustomNode::create(
      "OneDnnFusedPool2dChain",
      std::move(inputs),
      topNode->shape(),
      topNode->dataType(),
      std::move(evalFunc),
      topNode->stream());
}

// Build a fused CustomNode for batchnorm anchor (eltwise chain only, no sum).
// anchorNode is the ProjectionNode(BN, Result).
NodePtr buildFusedBatchNorm(const NodePtr& topNode, const FusedChain& chain) {
  auto& proj = chain.anchorNode->impl<ProjectionNode>();
  auto bn = Node::cast<BatchNormNodePtr>(proj.parent());

  std::vector<NodePtr> inputs = {
      bn->inputTensor(), bn->weight(), bn->bias(),
      bn->runningMean(), bn->runningVar()};

  auto evalFunc = [bnNode = bn, eltwiseChain = chain.eltwiseChain](
                      const std::vector<const Tensor*>& inputTensors) -> Tensor {
    const auto& input = *inputTensors[0];
    const auto& weight = *inputTensors[1];
    const auto& bias = *inputTensors[2];
    const auto& runningMean = *inputTensors[3];
    const auto& runningVar = *inputTensors[4];

    auto& backend = CPUBackend::getInstance();
    auto& engine = backend.onednnOps().engine();
    auto& nativeStream = backend.nativeStream();
    auto asyncStream = backend.asyncStream();

    dnnl::post_ops postOps;
    for (const auto& elt : eltwiseChain) {
      postOps.append_eltwise(elt.alg, elt.alpha, elt.beta);
    }
    dnnl::primitive_attr attr;
    attr.set_post_ops(postOps);

    auto result = detail::onednn::batchnormForward(
        input, weight, bias, runningMean, runningVar,
        bnNode->axes(), bnNode->train(),
        bnNode->momentum(), bnNode->epsilon(),
        /*wantState=*/false,
        engine, *asyncStream, nativeStream, attr);

    return result.output;
  };

  return CustomNode::create(
      "OneDnnFusedBatchNormChain",
      std::move(inputs),
      topNode->shape(),
      topNode->dataType(),
      std::move(evalFunc),
      topNode->stream());
}

// Build a fused CustomNode for matmul anchor (optional bias + eltwise chain).
NodePtr buildFusedMatmul(const NodePtr& topNode, const FusedChain& chain) {
  auto mm = Node::cast<MatmulNodePtr>(chain.anchorNode);
  std::vector<NodePtr> inputs = {mm->lhs(), mm->rhs()};
  if (chain.biasSrc) {
    inputs.push_back(chain.biasSrc);
  }

  auto evalFunc = [mmNode = mm,
                   eltwiseChain = chain.eltwiseChain,
                   hasBias = bool(chain.biasSrc)](
                      const std::vector<const Tensor*>& inputTensors) -> Tensor {
    const auto& lhs = *inputTensors[0];
    const auto& rhs = *inputTensors[1];

    auto& backend = CPUBackend::getInstance();
    auto& engine = backend.onednnOps().engine();
    auto& nativeStream = backend.nativeStream();
    auto asyncStream = backend.asyncStream();

    dnnl::post_ops postOps;
    for (const auto& elt : eltwiseChain) {
      postOps.append_eltwise(elt.alg, elt.alpha, elt.beta);
    }
    dnnl::primitive_attr attr;
    attr.set_post_ops(postOps);

    Tensor bias;
    if (hasBias) {
      bias = *inputTensors[2];
    }

    Tensor result;
    detail::onednn::matmul(
        result, lhs, rhs, mmNode->lhsProp(), mmNode->rhsProp(),
        engine, nativeStream, asyncStream, attr, bias);
    return result;
  };

  return CustomNode::create(
      "OneDnnFusedMatmulChain",
      std::move(inputs),
      topNode->shape(),
      topNode->dataType(),
      std::move(evalFunc),
      topNode->stream());
}

// Build a standalone CustomNode for an eltwise chain with no anchor primitive.
// Used when the chain contains a composed gelu (7 nodes → 1 kernel).
NodePtr buildStandaloneEltwise(const NodePtr& topNode, const FusedChain& chain) {
  std::vector<NodePtr> inputs = {chain.anchorNode};

  auto evalFunc = [eltwiseChain = chain.eltwiseChain](
                      const std::vector<const Tensor*>& inputTensors) -> Tensor {
    auto& backend = CPUBackend::getInstance();
    auto& engine = backend.onednnOps().engine();
    auto& nativeStream = backend.nativeStream();
    auto asyncStream = backend.asyncStream();

    Tensor result = *inputTensors[0];
    for (const auto& elt : eltwiseChain) {
      Tensor tmp;
      detail::onednn::applyEltwiseOp(
          tmp, result, elt.alg, engine, nativeStream, asyncStream, elt.alpha, elt.beta);
      result = std::move(tmp);
    }
    return result;
  };

  return CustomNode::create(
      "OneDnnStandaloneEltwise",
      std::move(inputs),
      topNode->shape(),
      topNode->dataType(),
      std::move(evalFunc),
      topNode->stream());
}

} // namespace

NodePtr OneDnnPrimitivePostOpFusion::rewrite(NodePtr node) {
  auto chain = findAnchorChain(node);
  // The fused node replaces `node`, so it must not consume it.
  if (!chain || chain->anchorNode == node) {
    return node;
  }

  // f64 not supported by OneDNN post-ops (anchored fusion).
  // Standalone eltwise is OK — applyEltwiseOp has an f64 CPU fallback.
  // Standalone lowering applies only the eltwise chain, so it cannot stand in
  // for chains that also fold a bias or residual add.
  if (node->dataType() == dtype::f64 &&
      chain->anchorKind != FusedChain::Standalone) {
    if (chain->sumSrc || chain->biasSrc) {
      return node;
    }
    // If chain has gelu, demote to standalone (gelu still worth lowering).
    bool hasGelu = false;
    for (const auto& elt : chain->eltwiseChain) {
      if (elt.alg == dnnl::algorithm::eltwise_gelu_erf) {
        hasGelu = true;
        break;
      }
    }
    if (hasGelu) {
      chain->anchorKind = FusedChain::Standalone;
      // anchorNode should be the input to the eltwise chain, which is the
      // anchor node that was found (e.g., the matmul).
    } else {
      return node;
    }
  }

  switch (chain->anchorKind) {
    case FusedChain::Conv2d:
      return buildFusedConv2d(node, *chain);
    case FusedChain::Pool2d:
      return buildFusedPool2d(node, *chain);
    case FusedChain::BatchNorm:
      return buildFusedBatchNorm(node, *chain);
    case FusedChain::Matmul:
      return buildFusedMatmul(node, *chain);
    case FusedChain::Standalone:
      return buildStandaloneEltwise(node, *chain);
  }
  return node;
}

} // namespace fl
