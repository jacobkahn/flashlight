/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>
#include <optional>
#include <unordered_map>
#include <vector>

#include "flashlight/fl/tensor/ir/BatchNormNode.h"
#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/Conv2dNode.h"
#include "flashlight/fl/tensor/ir/MatmulNode.h"
#include "flashlight/fl/tensor/ir/Node.h"
#include "flashlight/fl/tensor/ir/Pool2dNode.h"
#include "flashlight/fl/tensor/ir/ProjectionNode.h"
#include "flashlight/fl/tensor/ir/ReductionNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"

namespace fl {

/**
 * Lightweight subgraph pattern matcher for the JIT IR.
 *
 * Patterns are composed declaratively using template types:
 *   MatchAny          -- matches any node, captures it by name
 *   MatchUnary<P>     -- matches a UnaryNode with specific op, child matches P
 *   MatchAnyUnary<P>  -- matches any UnaryNode, child matches P
 *   MatchBinary<L, R> -- matches a BinaryNode with specific op
 *   MatchAnyBinary<L, R> -- matches any BinaryNode
 *   MatchMatmul<L, R> -- matches a MatmulNode
 *   MatchReduction<P> -- matches a ReductionNode with specific op
 *
 * Captures are stored in a flat map from string name to NodePtr.
 *
 * Example:
 *   // Match: relu(add(matmul(A, B), bias))
 *   using Pat = MatchUnary<
 *       UnaryOp::Sigmoid,
 *       MatchBinary<
 *           BinaryOp::Add,
 *           MatchMatmul<MatchAny<"lhs">, MatchAny<"rhs">>,
 *           MatchAny<"bias">>>;
 *
 *   MatchResult result;
 *   if (Pat::match(node, result)) {
 *     auto lhs = result.get("lhs");
 *     auto rhs = result.get("rhs");
 *     auto bias = result.get("bias");
 *     // ... replace with fused node
 *   }
 */

// Match result: maps capture names to matched nodes.
class MatchResult {
  std::unordered_map<std::string, NodePtr> captures_;

 public:
  void capture(const std::string& name, NodePtr node) {
    captures_[name] = node;
  }

  NodePtr get(const std::string& name) const {
    auto it = captures_.find(name);
    if (it == captures_.end()) {
      return nullptr;
    }
    return it->second;
  }

  void clear() {
    captures_.clear();
  }
};

// MatchAny: matches any node. Optionally captures it.
struct MatchAnyNoCapture {
  static bool match(const NodePtr& /*node*/, MatchResult& /*result*/) {
    return true;
  }
};

// Named capture version -- call capture() with a name.
// Since C++20 doesn't allow string literal template params portably,
// use a runtime string approach.
class MatchAnyCapture {
  std::string name_;

 public:
  explicit MatchAnyCapture(std::string name) : name_(std::move(name)) {}

  bool match(const NodePtr& node, MatchResult& result) const {
    result.capture(name_, node);
    return true;
  }
};

// MatchUnaryOp: matches a UnaryNode with a specific op.
template <typename ChildPattern>
class MatchUnaryOp {
  UnaryOp op_;
  ChildPattern child_;

 public:
  explicit MatchUnaryOp(UnaryOp op, ChildPattern child = {})
      : op_(op), child_(std::move(child)) {}

  bool match(const NodePtr& node, MatchResult& result) const {
    if (!node->isUnary()) {
      return false;
    }
    const auto& u = node->impl<UnaryNode>();
    if (u.op() != op_) {
      return false;
    }
    return child_.match(u.input(), result);
  }
};

// MatchAnyUnary: matches any UnaryNode (any op).
template <typename ChildPattern>
class MatchAnyUnary {
  ChildPattern child_;
  std::string captureName_;

 public:
  explicit MatchAnyUnary(ChildPattern child = {}, std::string captureName = "")
      : child_(std::move(child)), captureName_(std::move(captureName)) {}

  bool match(const NodePtr& node, MatchResult& result) const {
    if (!node->isUnary()) {
      return false;
    }
    if (!captureName_.empty()) {
      result.capture(captureName_, node);
    }
    const auto& u = node->impl<UnaryNode>();
    return child_.match(u.input(), result);
  }
};

// MatchBinaryOp: matches a BinaryNode with a specific op.
template <typename LhsPattern, typename RhsPattern>
class MatchBinaryOp {
  BinaryOp op_;
  LhsPattern lhs_;
  RhsPattern rhs_;

 public:
  explicit MatchBinaryOp(
      BinaryOp op,
      LhsPattern lhs = {},
      RhsPattern rhs = {})
      : op_(op), lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

  bool match(const NodePtr& node, MatchResult& result) const {
    if (!node->isBinary()) {
      return false;
    }
    const auto& b = node->impl<BinaryNode>();
    if (b.op() != op_) {
      return false;
    }
    return lhs_.match(b.lhs(), result) && rhs_.match(b.rhs(), result);
  }
};

// MatchAnyBinary: matches any BinaryNode.
template <typename LhsPattern, typename RhsPattern>
class MatchAnyBinary {
  LhsPattern lhs_;
  RhsPattern rhs_;
  std::string captureName_;

 public:
  explicit MatchAnyBinary(
      LhsPattern lhs = {},
      RhsPattern rhs = {},
      std::string captureName = "")
      : lhs_(std::move(lhs)),
        rhs_(std::move(rhs)),
        captureName_(std::move(captureName)) {}

  bool match(const NodePtr& node, MatchResult& result) const {
    if (!node->isBinary()) {
      return false;
    }
    if (!captureName_.empty()) {
      result.capture(captureName_, node);
    }
    const auto& b = node->impl<BinaryNode>();
    return lhs_.match(b.lhs(), result) && rhs_.match(b.rhs(), result);
  }
};

// MatchMatmul: matches a MatmulNode.
template <typename LhsPattern, typename RhsPattern>
class MatchMatmul {
  LhsPattern lhs_;
  RhsPattern rhs_;
  std::string captureName_;

 public:
  explicit MatchMatmul(
      LhsPattern lhs = {},
      RhsPattern rhs = {},
      std::string captureName = "")
      : lhs_(std::move(lhs)),
        rhs_(std::move(rhs)),
        captureName_(std::move(captureName)) {}

  bool match(const NodePtr& node, MatchResult& result) const {
    if (!node->isMatmul()) {
      return false;
    }
    if (!captureName_.empty()) {
      result.capture(captureName_, node);
    }
    const auto& m = node->impl<MatmulNode>();
    return lhs_.match(m.lhs(), result) && rhs_.match(m.rhs(), result);
  }
};

// MatchConv2d: matches a Conv2dNode.
template <typename InputPattern, typename WeightsPattern, typename BiasPattern>
class MatchConv2d {
  InputPattern input_;
  WeightsPattern weights_;
  BiasPattern bias_;
  std::string captureName_;

 public:
  explicit MatchConv2d(
      InputPattern input = {},
      WeightsPattern weights = {},
      BiasPattern bias = {},
      std::string captureName = "")
      : input_(std::move(input)),
        weights_(std::move(weights)),
        bias_(std::move(bias)),
        captureName_(std::move(captureName)) {}

  bool match(const NodePtr& node, MatchResult& result) const {
    if (!node->isConv2d()) {
      return false;
    }
    if (!captureName_.empty()) {
      result.capture(captureName_, node);
    }
    const auto& c = node->impl<Conv2dNode>();
    return input_.match(c.input(), result) &&
        weights_.match(c.weights(), result) &&
        bias_.match(c.bias(), result);
  }
};

// MatchReductionOp: matches a ReductionNode with a specific op.
template <typename ChildPattern>
class MatchReductionOp {
  ReductionOp op_;
  ChildPattern child_;

 public:
  explicit MatchReductionOp(ReductionOp op, ChildPattern child = {})
      : op_(op), child_(std::move(child)) {}

  bool match(const NodePtr& node, MatchResult& result) const {
    if (!node->isReduction()) {
      return false;
    }
    const auto& r = node->impl<ReductionNode>();
    if (r.op() != op_) {
      return false;
    }
    return child_.match(r.input(), result);
  }
};

// MatchNodeType: matches any node of a given NodeType.
class MatchNodeType {
  NodeType type_;
  std::string captureName_;

 public:
  explicit MatchNodeType(NodeType type, std::string captureName = "")
      : type_(type), captureName_(std::move(captureName)) {}

  bool match(const NodePtr& node, MatchResult& result) const {
    if (node->type() != type_) {
      return false;
    }
    if (!captureName_.empty()) {
      result.capture(captureName_, node);
    }
    return true;
  }
};

// MatchWithCapture: wraps any pattern and captures the matched node.
template <typename Pattern>
class MatchWithCapture {
  std::string name_;
  Pattern inner_;

 public:
  explicit MatchWithCapture(std::string name, Pattern inner = {})
      : name_(std::move(name)), inner_(std::move(inner)) {}

  bool match(const NodePtr& node, MatchResult& result) const {
    if (!inner_.match(node, result)) {
      return false;
    }
    result.capture(name_, node);
    return true;
  }
};

// MatchScalar: matches a ScalarNode with a specific value (within tolerance).
class MatchScalar {
  double value_;
  double tol_;
  std::string captureName_;

 public:
  explicit MatchScalar(
      double value,
      double tol = 1e-6,
      std::string captureName = "")
      : value_(value), tol_(tol), captureName_(std::move(captureName)) {}

  bool match(const NodePtr& node, MatchResult& result) const {
    if (!node->isScalar()) {
      return false;
    }
    const auto& s = node->impl<ScalarNode>();
    double nodeVal = s.scalar<double>();
    if (std::abs(nodeVal - value_) > tol_) {
      return false;
    }
    if (!captureName_.empty()) {
      result.capture(captureName_, node);
    }
    return true;
  }
};

// MatchScalarAny: matches any ScalarNode, optionally capturing it.
class MatchScalarAny {
  std::string captureName_;

 public:
  explicit MatchScalarAny(std::string captureName = "")
      : captureName_(std::move(captureName)) {}

  bool match(const NodePtr& node, MatchResult& result) const {
    if (!node->isScalar()) {
      return false;
    }
    if (!captureName_.empty()) {
      result.capture(captureName_, node);
    }
    return true;
  }
};

// =========================================================================
// Convenience factory functions
// =========================================================================

inline MatchAnyNoCapture any() {
  return {};
}

inline MatchAnyCapture capture(std::string name) {
  return MatchAnyCapture(std::move(name));
}

template <typename ChildPattern>
MatchUnaryOp<ChildPattern> matchUnary(UnaryOp op, ChildPattern child) {
  return MatchUnaryOp<ChildPattern>(op, std::move(child));
}

template <typename ChildPattern>
MatchAnyUnary<ChildPattern> matchAnyUnary(
    ChildPattern child,
    std::string captureName = "") {
  return MatchAnyUnary<ChildPattern>(
      std::move(child), std::move(captureName));
}

template <typename LhsPattern, typename RhsPattern>
MatchBinaryOp<LhsPattern, RhsPattern> matchBinary(
    BinaryOp op,
    LhsPattern lhs,
    RhsPattern rhs) {
  return MatchBinaryOp<LhsPattern, RhsPattern>(
      op, std::move(lhs), std::move(rhs));
}

template <typename LhsPattern, typename RhsPattern>
MatchAnyBinary<LhsPattern, RhsPattern> matchAnyBinary(
    LhsPattern lhs,
    RhsPattern rhs,
    std::string captureName = "") {
  return MatchAnyBinary<LhsPattern, RhsPattern>(
      std::move(lhs), std::move(rhs), std::move(captureName));
}

template <typename LhsPattern, typename RhsPattern>
MatchMatmul<LhsPattern, RhsPattern> matchMatmul(
    LhsPattern lhs,
    RhsPattern rhs,
    std::string captureName = "") {
  return MatchMatmul<LhsPattern, RhsPattern>(
      std::move(lhs), std::move(rhs), std::move(captureName));
}

template <typename InputPattern, typename WeightsPattern, typename BiasPattern>
MatchConv2d<InputPattern, WeightsPattern, BiasPattern> matchConv2d(
    InputPattern input,
    WeightsPattern weights,
    BiasPattern bias,
    std::string captureName = "") {
  return MatchConv2d<InputPattern, WeightsPattern, BiasPattern>(
      std::move(input), std::move(weights), std::move(bias),
      std::move(captureName));
}

template <typename ChildPattern>
MatchReductionOp<ChildPattern> matchReduction(
    ReductionOp op,
    ChildPattern child) {
  return MatchReductionOp<ChildPattern>(op, std::move(child));
}

template <typename Pattern>
MatchWithCapture<Pattern> withCapture(std::string name, Pattern inner) {
  return MatchWithCapture<Pattern>(std::move(name), std::move(inner));
}

inline MatchScalar matchScalar(
    double value,
    double tol = 1e-6,
    std::string captureName = "") {
  return MatchScalar(value, tol, std::move(captureName));
}

inline MatchScalarAny matchScalarAny(std::string captureName = "") {
  return MatchScalarAny(std::move(captureName));
}

// =========================================================================
// Multi-output node pattern matching
// =========================================================================
//
// Multi-output operations (e.g. BatchNorm) use ProjectionNode to expose
// individual outputs. To match a specific output of a multi-output op,
// compose MatchProjection with the parent node matcher:
//
//   auto pat = matchUnary(
//       UnaryOp::Relu,
//       matchProjection(BatchNormNode::Result, matchBatchNorm("bn")));
//   if (pat.match(node, result)) {
//     auto bnNode = result.get("bn");
//   }

// MatchProjection: matches a ProjectionNode at a specific output index
// whose parent matches ParentPattern.
template <typename ParentPattern>
class MatchProjection {
  unsigned outputIndex_;
  ParentPattern parent_;
  std::string captureName_;

 public:
  explicit MatchProjection(
      unsigned outputIndex,
      ParentPattern parent = {},
      std::string captureName = "")
      : outputIndex_(outputIndex),
        parent_(std::move(parent)),
        captureName_(std::move(captureName)) {}

  bool match(const NodePtr& node, MatchResult& result) const {
    if (node->type() != NodeType::Projection) {
      return false;
    }
    const auto& proj = node->impl<ProjectionNode>();
    if (proj.outputIndex() != outputIndex_) {
      return false;
    }
    if (!captureName_.empty()) {
      result.capture(captureName_, node);
    }
    return parent_.match(proj.parent(), result);
  }
};

// MatchBatchNorm: matches a BatchNormNode, optionally capturing it.
class MatchBatchNorm {
  std::string captureName_;

 public:
  explicit MatchBatchNorm(std::string captureName = "")
      : captureName_(std::move(captureName)) {}

  bool match(const NodePtr& node, MatchResult& result) const {
    if (node->type() != NodeType::BatchNorm) {
      return false;
    }
    if (!captureName_.empty()) {
      result.capture(captureName_, node);
    }
    return true;
  }
};

// MatchPool2d: matches a Pool2dNode, optionally capturing it.
class MatchPool2d {
  std::string captureName_;

 public:
  explicit MatchPool2d(std::string captureName = "")
      : captureName_(std::move(captureName)) {}

  bool match(const NodePtr& node, MatchResult& result) const {
    if (node->type() != NodeType::Pool2d) {
      return false;
    }
    if (!captureName_.empty()) {
      result.capture(captureName_, node);
    }
    return true;
  }
};

// ---- Factory functions for multi-output matching ----

template <typename ParentPattern>
MatchProjection<ParentPattern> matchProjection(
    unsigned outputIndex,
    ParentPattern parent,
    std::string captureName = "") {
  return MatchProjection<ParentPattern>(
      outputIndex, std::move(parent), std::move(captureName));
}

inline MatchBatchNorm matchBatchNorm(std::string captureName = "") {
  return MatchBatchNorm(std::move(captureName));
}

inline MatchPool2d matchPool2d(std::string captureName = "") {
  return MatchPool2d(std::move(captureName));
}

// ---- Pre-built patterns for BatchNorm outputs ----
// These match Projection(index, BatchNormNode) using the named Out enum.
// The capture name refers to the *parent BatchNormNode*, not the projection.

inline auto matchBatchNormResult(std::string bnCapture = "") {
  return matchProjection(BatchNormNode::Result, matchBatchNorm(std::move(bnCapture)));
}

inline auto matchBatchNormSaveMean(std::string bnCapture = "") {
  return matchProjection(BatchNormNode::SaveMean, matchBatchNorm(std::move(bnCapture)));
}

inline auto matchBatchNormSaveVar(std::string bnCapture = "") {
  return matchProjection(BatchNormNode::SaveVar, matchBatchNorm(std::move(bnCapture)));
}

inline auto matchBatchNormNewRunningMean(std::string bnCapture = "") {
  return matchProjection(BatchNormNode::NewRunningMean, matchBatchNorm(std::move(bnCapture)));
}

inline auto matchBatchNormNewRunningVar(std::string bnCapture = "") {
  return matchProjection(BatchNormNode::NewRunningVar, matchBatchNorm(std::move(bnCapture)));
}

} // namespace fl
