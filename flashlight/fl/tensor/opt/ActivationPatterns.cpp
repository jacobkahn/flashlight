/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/opt/ActivationPatterns.h"

#include "flashlight/fl/tensor/opt/Pass.h"

#include <cmath>

#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"

namespace fl {

namespace {

bool hasSingleUse(const NodePtr& node) {
  return !isEvaluated(*node) &&
      node->uses().size() + node->externalUses().size() == 1;
}

} // namespace

NodePtr tryMatchRelu(const NodePtr& node) {
  if (!node->isBinary()) {
    return nullptr;
  }
  const auto& bin = node->impl<BinaryNode>();
  if (bin.op() != BinaryOp::Max) {
    return nullptr;
  }
  auto isZeroScalar = [](const NodePtr& n) -> bool {
    return n->isScalar() && n->impl<ScalarNode>().scalar<double>() == 0.0;
  };
  if (isZeroScalar(bin.rhs())) {
    return bin.lhs();
  }
  if (isZeroScalar(bin.lhs())) {
    return bin.rhs();
  }
  return nullptr;
}

NodePtr tryMatchGeluErf(const NodePtr& node) {
  if (!node->isBinary()) {
    return nullptr;
  }
  const auto& outerMul = node->impl<BinaryNode>();
  if (outerMul.op() != BinaryOp::Mul) {
    return nullptr;
  }

  // Helper: check if a node is a ScalarNode with value ~ target.
  auto isScalarApprox =
      [](const NodePtr& n, double target, double tol) -> bool {
    if (!n->isScalar()) {
      return false;
    }
    return std::abs(n->impl<ScalarNode>().scalar<double>() - target) < tol;
  };

  // Try both sides of outerMul: one side is x, the other is cdf_scaled.
  auto tryMatch = [&](const NodePtr& xCandidate,
                      const NodePtr& cdfScaled) -> NodePtr {
    // cdf_scaled: Mul(Scalar(0.5), cdf_inner) -- check both sides
    if (!cdfScaled->isBinary() || !hasSingleUse(cdfScaled)) {
      return nullptr;
    }
    const auto& cdfMul = cdfScaled->impl<BinaryNode>();
    if (cdfMul.op() != BinaryOp::Mul) {
      return nullptr;
    }

    // Find which side of cdfMul is the 0.5 scalar.
    NodePtr cdfInner;
    if (isScalarApprox(cdfMul.lhs(), 0.5, 1e-6)) {
      cdfInner = cdfMul.rhs();
    } else if (isScalarApprox(cdfMul.rhs(), 0.5, 1e-6)) {
      cdfInner = cdfMul.lhs();
    } else {
      return nullptr;
    }

    // cdf_inner: Add(Scalar(1.0), erfNode) -- check both sides
    if (!cdfInner->isBinary() || !hasSingleUse(cdfInner)) {
      return nullptr;
    }
    const auto& addNode = cdfInner->impl<BinaryNode>();
    if (addNode.op() != BinaryOp::Add) {
      return nullptr;
    }

    NodePtr erfNode;
    if (isScalarApprox(addNode.lhs(), 1.0, 1e-6)) {
      erfNode = addNode.rhs();
    } else if (isScalarApprox(addNode.rhs(), 1.0, 1e-6)) {
      erfNode = addNode.lhs();
    } else {
      return nullptr;
    }

    // erfNode: Erf(...)
    if (!erfNode->isUnary() || !hasSingleUse(erfNode)) {
      return nullptr;
    }
    const auto& erf = erfNode->impl<UnaryNode>();
    if (erf.op() != UnaryOp::Erf) {
      return nullptr;
    }

    // erf input: Mul(x, Scalar(~0.7071))
    auto erfInput = erf.input();
    if (!erfInput->isBinary() || !hasSingleUse(erfInput)) {
      return nullptr;
    }
    const auto& erfMul = erfInput->impl<BinaryNode>();
    if (erfMul.op() != BinaryOp::Mul) {
      return nullptr;
    }

    // Find which side is x (must be same as xCandidate) and which is M_SQRT1_2.
    NodePtr innerX;
    if (isScalarApprox(erfMul.rhs(), M_SQRT1_2, 1e-4)) {
      innerX = erfMul.lhs();
    } else if (isScalarApprox(erfMul.lhs(), M_SQRT1_2, 1e-4)) {
      innerX = erfMul.rhs();
    } else {
      return nullptr;
    }

    // Verify inner x == outer x (same node pointer).
    if (innerX != xCandidate) {
      return nullptr;
    }

    return xCandidate;
  };

  // Try both orderings of outerMul (commutative).
  if (auto x = tryMatch(outerMul.lhs(), outerMul.rhs())) {
    return x;
  }
  return tryMatch(outerMul.rhs(), outerMul.lhs());
}

NodePtr tryMatchSilu(const NodePtr& node) {
  if (!node->isBinary()) {
    return nullptr;
  }
  const auto& bin = node->impl<BinaryNode>();
  if (bin.op() != BinaryOp::Mul) {
    return nullptr;
  }

  // See through Scalar(1.0) * x wrappers (produced by swish(x, beta=1.0)
  // which computes x * sigmoid(1.0 * x)).
  auto seeThrough1x = [](const NodePtr& n) -> NodePtr {
    if (!n->isBinary()) {
      return n;
    }
    const auto& inner = n->impl<BinaryNode>();
    if (inner.op() != BinaryOp::Mul) {
      return n;
    }
    auto isOneScalar = [](const NodePtr& s) -> bool {
      return s->isScalar() &&
             s->impl<ScalarNode>().scalar<double>() == 1.0;
    };
    if (isOneScalar(inner.lhs())) {
      return inner.rhs();
    }
    if (isOneScalar(inner.rhs())) {
      return inner.lhs();
    }
    return n;
  };

  auto trySide = [&](const NodePtr& x, const NodePtr& sig) -> NodePtr {
    if (sig->isUnary() && sig->impl<UnaryNode>().op() == UnaryOp::Sigmoid &&
        seeThrough1x(sig->impl<UnaryNode>().input()).get() == x.get()) {
      return x;
    }
    return nullptr;
  };
  if (auto r = trySide(bin.lhs(), bin.rhs())) {
    return r;
  }
  if (auto r = trySide(bin.rhs(), bin.lhs())) {
    return r;
  }
  return nullptr;
}

} // namespace fl
