/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/opt/backends/onednn/OneDnnOpFusion.h"
#include <optional>
#include <variant>

#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/CustomNode.h"
#include "flashlight/fl/tensor/ir/OpProperties.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/opt/PatternMatcher.h"
#include "flashlight/fl/tensor/backend/cpu/CPUBackend.h"
#include "flashlight/fl/tensor/ops/onednn/OneDnnOps.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/ops/onednn/Utils.h"

#include "dnnl.hpp"

namespace fl {

struct BinopInfo {
  NodePtr rhsNode;
  BinaryOp op;
};

using PostOpInfo = std::variant<BinopInfo, EltwiseInfo>;

struct OneDnnOpFusion::SearchState {
  SearchState(NodePtr root, std::vector<PostOpInfo> postOps)
      : searchRoot(root), accumulatedPostOps(std::move(postOps)) {}
  NodePtr searchRoot;
  // Accumulated in top-to-bottom order (outermost op first).
  // Reversed in fuseNodes to get bottom-to-top order for execution.
  std::vector<PostOpInfo> accumulatedPostOps;
};

std::optional<EltwiseInfo> tryUnaryOpToOneDnnEltwise(const UnaryOp op) {
  switch (op) {
    case UnaryOp::Exp:
      return EltwiseInfo{dnnl::algorithm::eltwise_exp, 0, 0};
    case UnaryOp::Log:
      return EltwiseInfo{dnnl::algorithm::eltwise_log, 0, 0};
    case UnaryOp::Sqrt:
      return EltwiseInfo{dnnl::algorithm::eltwise_sqrt, 0, 0};
    case UnaryOp::Tanh:
      return EltwiseInfo{dnnl::algorithm::eltwise_tanh, 0, 0};
    case UnaryOp::Absolute:
      return EltwiseInfo{dnnl::algorithm::eltwise_abs, 0, 0};
    case UnaryOp::Rint:
      return EltwiseInfo{dnnl::algorithm::eltwise_round, 0, 0};
    case UnaryOp::Negative:
      return EltwiseInfo{dnnl::algorithm::eltwise_linear, -1, 0};
    case UnaryOp::Sigmoid:
      return EltwiseInfo{dnnl::algorithm::eltwise_logistic, 0, 0};
    // These ops have no OneDNN eltwise equivalent
    case UnaryOp::Log1p:
    case UnaryOp::Sin:
    case UnaryOp::Cos:
    case UnaryOp::Floor:
    case UnaryOp::Ceil:
    case UnaryOp::Erf:
    case UnaryOp::LogicalNot:
    case UnaryOp::Sign:
    case UnaryOp::Isnan:
    case UnaryOp::Isinf:
      return std::nullopt;
  }
  throw std::runtime_error(
      "[tryUnaryOpToOneDnnEltwise] Unexpected unary operation type");
}

namespace {

// OneDNN dnnl_types.h: DNNL_MAX_POST_OPS = 32
static constexpr unsigned kOneDnnMaxNumPostOps = 32;

std::optional<dnnl::algorithm> tryBinopToOneDnnAlg(const BinaryOp op) {
  switch (op) {
    case BinaryOp::Add:
      return dnnl::algorithm::binary_add;
    case BinaryOp::Sub:
      return dnnl::algorithm::binary_sub;
    case BinaryOp::Mul:
      return dnnl::algorithm::binary_mul;
    case BinaryOp::Div:
      return dnnl::algorithm::binary_div;
    case BinaryOp::Eq:
      return dnnl::algorithm::binary_eq;
    case BinaryOp::Neq:
      return dnnl::algorithm::binary_ne;
    case BinaryOp::Gt:
      return dnnl::algorithm::binary_gt;
    case BinaryOp::Gte:
      return dnnl::algorithm::binary_ge;
    case BinaryOp::Lt:
      return dnnl::algorithm::binary_lt;
    case BinaryOp::Lte:
      return dnnl::algorithm::binary_le;
    case BinaryOp::Min:
      return dnnl::algorithm::binary_min;
    case BinaryOp::Max:
      return dnnl::algorithm::binary_max;
    case BinaryOp::Pow:
    case BinaryOp::Mod:
    case BinaryOp::And:
    case BinaryOp::Or:
    case BinaryOp::Shl:
    case BinaryOp::Shr:
    case BinaryOp::BitAnd:
    case BinaryOp::BitOr:
    case BinaryOp::BitXor:
      return std::nullopt;
  }
  throw std::runtime_error(
      "[tryBinopToOneDnnAlg] Unexpected binary operation type");
}

dnnl::algorithm binopToOneDnnAlg(const BinaryOp op) {
  const auto alg = tryBinopToOneDnnAlg(op);
  if (!alg.has_value()) {
    throw std::runtime_error("[binopToOneDnnAlg] unsupported binop for OneDNN");
  }
  return alg.value();
}

bool isNodeFusable(const NodePtr node) {
  // OneDNN binary/eltwise primitives don't support f64 — skip fusion.
  if (node->dataType() == dtype::f64) {
    return false;
  }
  if (node->isBinary()) {
    return tryBinopToOneDnnAlg(node->impl<BinaryNode>().op()).has_value();
  }
  if (node->isUnary()) {
    return tryUnaryOpToOneDnnEltwise(node->impl<UnaryNode>().op()).has_value();
  }
  return false;
}

bool isFusionProfitable(const NodePtr node) {
  // TODO
  // Even if we have multiple uses, it might be possible & profitable to fuse,
  // i.e., recomputation might be okay, think Halide.
  return node->uses().size() + node->externalUses().size() <= 1;
}

bool shouldNodeBeFused(const NodePtr node) {
  return !isEvaluated(*node) && isNodeFusable(node) &&
      isFusionProfitable(node);
}

// Capture names for PatternMatcher
constexpr const char* kLhsCapture = "lhs";
constexpr const char* kRhsCapture = "rhs";
constexpr const char* kInputCapture = "input";

} // namespace

NodePtr OneDnnOpFusion::rewriteFrom(NodePtr node) {
  SearchState state(node, /* accumulatedPostOps = */ {});
  auto fusedNode = searchAndFuse(node, state);
  node->replaceAllUsesWith(fusedNode);
  return fusedNode;
}

NodePtr OneDnnOpFusion::searchAndFuse(NodePtr node, SearchState& state) {
  auto binaryPat = matchAnyBinary(capture(kLhsCapture), capture(kRhsCapture));
  auto unaryPat = matchAnyUnary(capture(kInputCapture));

  while (true) {
    if (visited_.contains(node) || !shouldNodeBeFused(node) ||
        state.accumulatedPostOps.size() >= kOneDnnMaxNumPostOps) {
      return fuseNodes(node, state);
    }
    visited_.insert(node);

    MatchResult result;

    // Match fusable binary: binop(lhs, rhs)
    if (binaryPat.match(node, result)) {
      const auto op = node->impl<BinaryNode>().op();
      auto lhs = result.get(kLhsCapture);
      auto rhs = result.get(kRhsCapture);

      // Commutativity: for commutative ops, if the LHS is not fusable but
      // the RHS is, swap sides to extend the chain. OneDNN binary post-ops
      // only accept the RHS argument, so the non-fusable side becomes the
      // post-op RHS after swapping.
      if (isCommutative(op) &&
          !shouldNodeBeFused(lhs) && shouldNodeBeFused(rhs)) {
        state.accumulatedPostOps.push_back(BinopInfo{rewriteFrom(lhs), op});
        node = rhs;
        continue;
      }

      state.accumulatedPostOps.push_back(BinopInfo{rewriteFrom(rhs), op});
      node = lhs;
      continue;
    }

    // Match fusable unary: unaryOp(input)
    result.clear();
    if (unaryPat.match(node, result)) {
      auto eltwiseInfo =
          tryUnaryOpToOneDnnEltwise(node->impl<UnaryNode>().op()).value();
      state.accumulatedPostOps.push_back(eltwiseInfo);
      node = result.get(kInputCapture);
      continue;
    }

    throw std::runtime_error(
        "[OneDnnOpFusion::searchAndFuse] If node should be fused, "
        "it must be a binary or unary node");
  }
}

NodePtr OneDnnOpFusion::fuseNodes(NodePtr node, SearchState& state) {
  // Defer input processing to the worklist in apply() instead of recursing.
  // This prevents stack overflow on deep graphs (e.g., thousands of chained
  // IndexedUpdateNodes from Jacobian tests).
  // Each node's inputs are queued at most once; a shared node may terminate
  // many chains, and re-queueing its inputs every time grows exponentially.
  if (expanded_.insert(node).second) {
    for (const auto& input : node->inputs()) {
      worklist_.push_back(input);
    }
  }

  if (state.accumulatedPostOps.size() < 2) {
    return state.searchRoot;
  }

  // Reverse to get bottom-to-top execution order
  auto ops = state.accumulatedPostOps;
  std::reverse(ops.begin(), ops.end());

  // The first (bottommost) op must be binary to serve as the primary primitive.
  // If it's eltwise, we can't fuse (would need an eltwise primary, which is a
  // different code path -- defer for now).
  if (!std::holds_alternative<BinopInfo>(ops.front())) {
    return state.searchRoot;
  }

  // Build the input node list and post-op descriptors.
  // The primary binary op consumes inputs[0] (LHS) and inputs[1] (RHS).
  // Each subsequent binary post-op adds an input; eltwise post-ops add none.
  std::vector<NodePtr> inputNodes{node};
  inputNodes.push_back(std::get<BinopInfo>(ops.front()).rhsNode);
  dnnl::algorithm primaryAlg =
      binopToOneDnnAlg(std::get<BinopInfo>(ops.front()).op);

  // Descriptor for each post-op (ops[1..N-1])
  struct PostOpDesc {
    bool isBinary;
    dnnl::algorithm alg;
    float alpha = 0;
    float beta = 0;
  };
  std::vector<PostOpDesc> postOpDescs;
  for (size_t i = 1; i < ops.size(); i++) {
    if (auto* binop = std::get_if<BinopInfo>(&ops[i])) {
      inputNodes.push_back(binop->rhsNode);
      postOpDescs.push_back(
          {/* isBinary = */ true, binopToOneDnnAlg(binop->op), 0, 0});
    } else {
      const auto& eltwise = std::get<EltwiseInfo>(ops[i]);
      postOpDescs.push_back(
          {/* isBinary = */ false, eltwise.alg, eltwise.alpha, eltwise.beta});
    }
  }

  // TODO refactor with common logic in CPUBackend
  // The fused node replaces searchRoot, so its result must have searchRoot's
  // dtype (e.g. b8 for a chain ending in a comparison).
  auto evalFunc = [primaryAlg,
                   postOpDescs = std::move(postOpDescs),
                   dstShape = state.searchRoot->shape(),
                   dstType = detail::flToOneDnnType(
                       state.searchRoot->dataType())](
                      const std::vector<const Tensor*>& inputs) {
    const Tensor* lhs = inputs[0];
    const Tensor* rhs = inputs[1];

    auto& backend = CPUBackend::getInstance();
    auto& engine = backend.onednnOps().engine();

    // prepare memories
    auto& lhsTensor = toCPUTensor(*lhs);
    auto& rhsTensor = toCPUTensor(*rhs);
    auto lhsMem = detail::makeDnnlMemory(
        lhsTensor.bufferPtr(), lhs->shape(), lhsTensor.stridesVec(),
        lhs->type(), engine);
    auto rhsMem = detail::makeDnnlMemory(
        rhsTensor.bufferPtr(), rhs->shape(), rhsTensor.stridesVec(),
        rhs->type(), engine);
    auto lhsMemDesc = lhsMem.get_desc();
    auto rhsMemDesc = rhsMem.get_desc();
    const auto dstMemDesc =
        detail::oneDnnContiguousMemDescFromShape(dstShape, dstType);
    auto dstMem = dnnl::memory(dstMemDesc, engine);

    // OneDNN binary requires all tensors to have the same number of
    // dimensions. Pad shorter descriptors/memories with leading 1s.
    const int dstNdim = dstMemDesc.get_ndims();
    auto padToNdim = [&engine](dnnl::memory::desc& desc, dnnl::memory& mem,
                               int targetNdim) {
      if (desc.get_ndims() < targetNdim) {
        dnnl::memory::dims padded(targetNdim, 1);
        auto dims = desc.get_dims();
        int offset = targetNdim - static_cast<int>(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
          padded[offset + i] = dims[i];
        }
        desc = desc.reshape(padded);
        mem = dnnl::memory(desc, engine, mem.get_data_handle());
      }
    };
    padToNdim(lhsMemDesc, lhsMem, dstNdim);
    padToNdim(rhsMemDesc, rhsMem, dstNdim);

    // prepare part of arguments
    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC_0, lhsMem},
        {DNNL_ARG_SRC_1, rhsMem},
        {DNNL_ARG_DST, dstMem},
    };

    // prepare post ops
    dnnl::post_ops postOps;
    unsigned binaryInputIdx = 2; // next input index for binary post-ops
    for (unsigned i = 0; i < postOpDescs.size(); i++) {
      const auto& desc = postOpDescs[i];
      if (desc.isBinary) {
        auto& otherTensor = toCPUTensor(*inputs[binaryInputIdx++]);
        auto otherMem = detail::makeDnnlMemory(
            otherTensor.bufferPtr(),
            inputs[binaryInputIdx - 1]->shape(),
            otherTensor.stridesVec(),
            inputs[binaryInputIdx - 1]->type(), engine);
        auto otherDesc = otherMem.get_desc();
        padToNdim(otherDesc, otherMem, dstNdim);
        postOps.append_binary(desc.alg, otherDesc);
        args.insert(
            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1, otherMem});
      } else {
        postOps.append_eltwise(desc.alg, desc.alpha, desc.beta);
      }
    }

    // finish building primitive
    dnnl::primitive_attr binaryAttr;
    binaryAttr.set_post_ops(postOps);

    // prepare part of primitive
    const dnnl::binary::primitive_desc binaryPrimitiveDesc(
        engine, primaryAlg, lhsMemDesc, rhsMemDesc, dstMemDesc, binaryAttr);
    const auto binaryPrimitive = dnnl::binary(binaryPrimitiveDesc);

    // Sync the async stream before executing: input buffers may have been
    // written by tasks enqueued on the async worker thread (e.g., binary ops,
    // reductions). Without sync, the fused primitive reads stale data.
    backend.asyncStream()->sync();
    binaryPrimitive.execute(backend.nativeStream(), args);
    auto type = detail::oneDnnToFlType(dstMem.get_desc().get_data_type());
    void* buf = dstMem.get_data_handle();

    auto owner = std::make_shared<dnnl::memory>(std::move(dstMem));
    return toTensor<CPUTensor>(dstShape, type, buf, std::move(owner));
  };

  return CustomNode::create(
      "OneDnnFusedOp",
      std::move(inputNodes),
      state.searchRoot->shape(),
      state.searchRoot->dataType(),
      std::move(evalFunc),
      state.searchRoot->stream());
}

NodePtr OneDnnOpFusion::apply(NodePtr root) {
  // Iterative worklist instead of deep recursion through rewriteFrom. State is
  // reset up front so an exception in a previous apply() can't leak into this
  // one.
  visited_.clear();
  expanded_.clear();
  worklist_.clear();
  worklist_.push_back(root);
  NodePtr optimizedRoot = root;
  while (!worklist_.empty()) {
    auto node = worklist_.back();
    worklist_.pop_back();
    if (visited_.contains(node)) {
      continue;
    }
    auto result = rewriteFrom(node);
    if (node == root) {
      optimizedRoot = result;
    }
  }
  visited_.clear();
  expanded_.clear();
  worklist_.clear();
  return optimizedRoot;
}

} // namespace fl
