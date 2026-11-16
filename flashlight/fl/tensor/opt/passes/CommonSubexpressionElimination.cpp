/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/opt/passes/CommonSubexpressionElimination.h"

#include <sstream>

#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/CastNode.h"
#include "flashlight/fl/tensor/ir/ClipNode.h"
#include "flashlight/fl/tensor/ir/ConcatenateNode.h"
#include "flashlight/fl/tensor/ir/CumsumNode.h"
#include "flashlight/fl/tensor/ir/FlipNode.h"
#include "flashlight/fl/tensor/ir/FusedElementwiseNode.h"
#include "flashlight/fl/tensor/ir/FusedReduceElementwiseNode.h"
#include "flashlight/fl/tensor/ir/MatmulNode.h"
#include "flashlight/fl/tensor/ir/PadNode.h"
#include "flashlight/fl/tensor/ir/ReductionNode.h"
#include "flashlight/fl/tensor/ir/ReshapeNode.h"
#include "flashlight/fl/tensor/ir/RollNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/TileNode.h"
#include "flashlight/fl/tensor/ir/TransposeNode.h"
#include "flashlight/fl/tensor/ir/TriangularNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/ir/WhereNode.h"

namespace fl {

namespace {

// Build a string key that uniquely identifies a node's computation.
// Returns empty string for non-CSE-able nodes.
std::string makeKey(const NodePtr& node) {
  std::ostringstream oss;
  switch (node->type()) {
    case NodeType::Scalar: {
      // The value must be keyed exactly: default stream precision (6 digits)
      // would merge distinct constants like 1000000 and 1000001.
      const auto& s = node->impl<ScalarNode>();
      oss << "S:" << s.shape().toString() << ":"
          << static_cast<int>(s.dataType()) << ":";
      if (isFloatingPoint(s.dataType())) {
        oss << std::hexfloat << s.scalar<double>();
      } else if (s.dataType() == dtype::u64) {
        oss << s.scalar<unsigned long long>();
      } else {
        oss << s.scalar<long long>();
      }
      return oss.str();
    }
    case NodeType::Binary: {
      const auto& b = node->impl<BinaryNode>();
      oss << "B:" << static_cast<int>(b.op()) << ":"
          << reinterpret_cast<uintptr_t>(b.lhs().get()) << ":"
          << reinterpret_cast<uintptr_t>(b.rhs().get());
      return oss.str();
    }
    case NodeType::Unary: {
      const auto& u = node->impl<UnaryNode>();
      oss << "U:" << static_cast<int>(u.op()) << ":"
          << reinterpret_cast<uintptr_t>(u.input().get());
      return oss.str();
    }
    case NodeType::Clip: {
      const auto& c = node->impl<ClipNode>();
      oss << "Clip:" << reinterpret_cast<uintptr_t>(c.input().get()) << ":"
          << reinterpret_cast<uintptr_t>(c.low().get()) << ":"
          << reinterpret_cast<uintptr_t>(c.high().get());
      return oss.str();
    }
    case NodeType::Concatenate: {
      const auto& c = node->impl<ConcatenateNode>();
      oss << "Concat:" << c.axis();
      for (const auto& input : node->inputs()) {
        oss << ":" << reinterpret_cast<uintptr_t>(input.get());
      }
      return oss.str();
    }
    case NodeType::Cumsum: {
      const auto& c = node->impl<CumsumNode>();
      oss << "Cumsum:" << c.axis() << ":"
          << reinterpret_cast<uintptr_t>(c.input().get());
      return oss.str();
    }
    case NodeType::Flip: {
      const auto& f = node->impl<FlipNode>();
      oss << "Flip:" << f.axis() << ":"
          << reinterpret_cast<uintptr_t>(f.input().get());
      return oss.str();
    }
    case NodeType::Reduction: {
      const auto& r = node->impl<ReductionNode>();
      oss << "R:" << static_cast<int>(r.op()) << ":" << r.keepDims() << ":"
          << reinterpret_cast<uintptr_t>(r.input().get());
      for (int ax : r.axes()) {
        oss << ":" << ax;
      }
      return oss.str();
    }
    case NodeType::FusedElementwise: {
      const auto& f = node->impl<FusedElementwiseNode>();
      oss << "FE:" << f.numExternalInputs();
      for (const auto& input : node->inputs()) {
        oss << ":" << reinterpret_cast<uintptr_t>(input.get());
      }
      for (const auto& step : f.program()) {
        oss << ":";
        if (step.isUnary()) {
          oss << "u" << static_cast<int>(step.unaryOp());
        } else {
          oss << "b" << static_cast<int>(step.binaryOp());
        }
        oss << "," << step.src0 << "," << step.src1;
      }
      return oss.str();
    }
    // ValueNode: each wraps a unique tensor -- can't CSE
    case NodeType::FusedReduceElementwise: {
      const auto& f = node->impl<FusedReduceElementwiseNode>();
      oss << "FRE:" << f.numExternalInputs() << ":"
          << static_cast<int>(f.reductionOp()) << ":" << f.keepDims();
      for (int ax : f.axes()) {
        oss << ":" << ax;
      }
      for (const auto& input : node->inputs()) {
        oss << ":" << reinterpret_cast<uintptr_t>(input.get());
      }
      for (const auto& step : f.program()) {
        oss << ":";
        if (step.isUnary()) {
          oss << "u" << static_cast<int>(step.unaryOp());
        } else {
          oss << "b" << static_cast<int>(step.binaryOp());
        }
        oss << "," << step.src0 << "," << step.src1;
      }
      return oss.str();
    }
    case NodeType::Matmul: {
      const auto& m = node->impl<MatmulNode>();
      oss << "M:" << static_cast<int>(m.lhsProp()) << ":"
          << static_cast<int>(m.rhsProp()) << ":"
          << reinterpret_cast<uintptr_t>(m.lhs().get()) << ":"
          << reinterpret_cast<uintptr_t>(m.rhs().get());
      return oss.str();
    }
    case NodeType::Pad: {
      const auto& p = node->impl<PadNode>();
      oss << "Pad:" << static_cast<int>(p.padType()) << ":"
          << reinterpret_cast<uintptr_t>(p.input().get());
      for (const auto& [before, after] : p.padWidths()) {
        oss << ":" << before << "," << after;
      }
      return oss.str();
    }
    case NodeType::Reshape: {
      const auto& r = node->impl<ReshapeNode>();
      oss << "Rsh:" << r.shape().toString() << ":"
          << reinterpret_cast<uintptr_t>(r.input().get());
      return oss.str();
    }
    case NodeType::Roll: {
      const auto& r = node->impl<RollNode>();
      oss << "Roll:" << r.shift() << ":" << r.axis() << ":"
          << reinterpret_cast<uintptr_t>(r.input().get());
      return oss.str();
    }
    case NodeType::Transpose: {
      const auto& t = node->impl<TransposeNode>();
      oss << "Tr:" << reinterpret_cast<uintptr_t>(t.input().get());
      for (int i = 0; i < t.axes().ndim(); i++) {
        oss << ":" << t.axes().dim(i);
      }
      return oss.str();
    }
    case NodeType::Triangular: {
      const auto& t = node->impl<TriangularNode>();
      oss << "Tri:" << static_cast<int>(t.mode()) << ":" << t.diagonal()
          << ":" << reinterpret_cast<uintptr_t>(t.input().get());
      return oss.str();
    }
    case NodeType::Tile: {
      const auto& t = node->impl<TileNode>();
      oss << "Tile:" << reinterpret_cast<uintptr_t>(t.input().get());
      for (int i = 0; i < t.tileDims().ndim(); i++) {
        oss << ":" << t.tileDims().dim(i);
      }
      return oss.str();
    }
    case NodeType::Where: {
      const auto& w = node->impl<WhereNode>();
      oss << "W:" << reinterpret_cast<uintptr_t>(w.condition().get()) << ":"
          << reinterpret_cast<uintptr_t>(w.x().get()) << ":"
          << reinterpret_cast<uintptr_t>(w.y().get());
      return oss.str();
    }
    case NodeType::Cast: {
      const auto& c = node->impl<CastNode>();
      oss << "C:" << static_cast<int>(c.targetDtype()) << ":"
          << reinterpret_cast<uintptr_t>(c.input().get());
      return oss.str();
    }
    // CustomNode: has non-comparable lambda -- can't CSE
    // IndexNode / IndexedUpdateNode: non-comparable Index objects -- can't CSE
    // Conv2dNode: opaque op with many params -- skip CSE for now
    case NodeType::Value:
    case NodeType::MultiValue:
    case NodeType::Custom:
    case NodeType::Index:
    case NodeType::IndexedUpdate:
    case NodeType::Conv2d:
    case NodeType::Projection:
    case NodeType::BatchNorm:
    case NodeType::Pool2d:
    case NodeType::Transfer:
      return {};
  }
  throw std::runtime_error(
      "[CommonSubexpressionElimination] Unknown node type");
}

} // namespace

NodePtr CommonSubexpressionElimination::rewrite(NodePtr node) {
  std::string key = makeKey(node);
  if (key.empty()) {
    return node;
  }

  auto it = cache_.find(key);
  if (it != cache_.end()) {
    node->replaceAllUsesWith(it->second);
    return it->second;
  }

  cache_[key] = node;
  return node;
}

NodePtr CommonSubexpressionElimination::apply(NodePtr root) {
  cache_.clear();
  return BottomUpPass::apply(root);
}

} // namespace fl
