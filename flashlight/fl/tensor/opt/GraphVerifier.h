/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * @file GraphVerifier.h
 *
 * Graph verifier for the JIT IR. Runs after each optimization pass to catch
 * graph invariant violations early.
 *
 * Enabled by default. Define FL_DISABLE_JIT_GRAPH_VERIFY to disable.
 */

#ifndef FL_DISABLE_JIT_GRAPH_VERIFY

#include <stack>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/ShapeInference.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/CastNode.h"
#include "flashlight/fl/tensor/ir/ClipNode.h"
#include "flashlight/fl/tensor/ir/ConcatenateNode.h"
#include "flashlight/fl/tensor/ir/CumsumNode.h"
#include "flashlight/fl/tensor/ir/FlipNode.h"
#include "flashlight/fl/tensor/ir/MatmulNode.h"
#include "flashlight/fl/tensor/ir/Node.h"
#include "flashlight/fl/tensor/ir/OpProperties.h"
#include "flashlight/fl/tensor/ir/PadNode.h"
#include "flashlight/fl/tensor/ir/ProjectionNode.h"
#include "flashlight/fl/tensor/ir/ReductionNode.h"
#include "flashlight/fl/tensor/ir/ReshapeNode.h"
#include "flashlight/fl/tensor/ir/RollNode.h"
#include "flashlight/fl/tensor/ir/TileNode.h"
#include "flashlight/fl/tensor/ir/TransposeNode.h"
#include "flashlight/fl/tensor/ir/TriangularNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/ir/WhereNode.h"

namespace fl {

/**
 * Check that a pass preserved the metadata of the graph root. Every tensor
 * holding the root relies on its shape and dtype, and a rewrite that changes
 * them (e.g. an identity rule that ignores broadcasting) silently produces a
 * differently-shaped result.
 */
inline void verifyRootMetadata(
    const NodePtr& root,
    const Shape& expectedShape,
    dtype expectedType) {
  if (root->shape() != expectedShape || root->dataType() != expectedType) {
    throw std::logic_error(
        "[verifyGraph] Pass changed the root's metadata from " +
        expectedShape.toString() + "/" + dtypeToString(expectedType) + " to " +
        root->shape().toString() + "/" + dtypeToString(root->dataType()));
  }
}

inline void verifyGraph(const NodePtr& root) {
  if (!root)
    return;

  // DFS to collect all reachable nodes, rejecting cycles: a node that is
  // reached again while still on the DFS path is its own (transitive) input.
  std::unordered_set<Node*> reachable;
  std::unordered_set<Node*> onPath;
  std::stack<std::pair<Node*, bool>> worklist;
  worklist.push({root.get(), false});
  while (!worklist.empty()) {
    auto [node, expanded] = worklist.top();
    worklist.pop();
    if (expanded) {
      onPath.erase(node);
      continue;
    }
    if (onPath.contains(node)) {
      throw std::logic_error("[verifyGraph] Cycle in graph");
    }
    if (!reachable.insert(node).second)
      continue;
    onPath.insert(node);
    worklist.push({node, true});
    for (const auto& input : node->inputs()) {
      if (input)
        worklist.push({input.get(), false});
    }
  }

  auto shapeErr = [](const char* kind) {
    throw std::logic_error(
        std::string("[verifyGraph] Shape mismatch on ") + kind + " node");
  };

  for (auto* node : reachable) {
    // Check 1: every Use in node->uses() is backed by the user's input slot.
    for (const auto& use : node->uses()) {
      auto& user = use->user();
      auto idx = use->inputIdx();
      if (idx >= user.inputs().size() || user.inputs()[idx].get() != node) {
        throw std::logic_error("[verifyGraph] Use-list inconsistency");
      }
    }

    // Check 2: shape consistency for nodes with deterministic inference.
    switch (node->type()) {
      case NodeType::Binary: {
        auto& n = node->impl<BinaryNode>();
        if (node->shape() !=
            broadcastShapes(n.lhs()->shape(), n.rhs()->shape()))
          shapeErr("Binary");
        break;
      }
      case NodeType::Unary:
        if (node->shape() != node->impl<UnaryNode>().input()->shape())
          shapeErr("Unary");
        break;
      case NodeType::Cast:
        if (node->shape() != node->impl<CastNode>().input()->shape())
          shapeErr("Cast");
        break;
      case NodeType::Clip: {
        auto& n = node->impl<ClipNode>();
        if (node->shape() != n.input()->shape()) {
          shapeErr("Clip");
        }
        if (node->dataType() != n.input()->dataType()) {
          throw std::logic_error("[verifyGraph] Clip output dtype mismatch");
        }
        break;
      }
      case NodeType::Concatenate: {
        auto& n = node->impl<ConcatenateNode>();
        std::vector<Shape> shapes;
        std::vector<Dim> elements;
        shapes.reserve(n.numInputs());
        elements.reserve(n.numInputs());
        auto outputDtype = n.input(0)->dataType();
        for (unsigned i = 0; i < n.numInputs(); ++i) {
          shapes.push_back(n.input(i)->shape());
          elements.push_back(n.input(i)->shape().elements());
          outputDtype = promoteDtypes(outputDtype, n.input(i)->dataType());
        }
        if (node->dataType() != outputDtype) {
          throw std::logic_error(
              "[verifyGraph] Concatenate output dtype mismatch");
        }
        if (node->shape() !=
            inferConcatenateOutputShape(shapes, elements, n.axis()))
          shapeErr("Concatenate");
        break;
      }
      case NodeType::Cumsum: {
        auto& n = node->impl<CumsumNode>();
        if (n.axis() >= static_cast<unsigned>(n.input()->shape().ndim())) {
          throw std::logic_error("[verifyGraph] Invalid Cumsum axis");
        }
        if (node->shape() != n.input()->shape()) {
          shapeErr("Cumsum");
        }
        break;
      }
      case NodeType::Flip: {
        auto& n = node->impl<FlipNode>();
        if (n.axis() >= static_cast<unsigned>(n.input()->shape().ndim())) {
          throw std::logic_error("[verifyGraph] Invalid Flip axis");
        }
        if (node->shape() != n.input()->shape()) {
          shapeErr("Flip");
        }
        break;
      }
      case NodeType::Reshape:
        if (node->shape().elements() !=
            node->impl<ReshapeNode>().input()->shape().elements())
          shapeErr("Reshape");
        break;
      case NodeType::Reduction: {
        auto& n = node->impl<ReductionNode>();
        if (node->shape() !=
            inferReductionOutputShape(
                n.input()->shape(), n.axes(), n.keepDims()))
          shapeErr("Reduction");
        break;
      }
      case NodeType::Roll: {
        auto& n = node->impl<RollNode>();
        if (n.axis() >= static_cast<unsigned>(n.input()->shape().ndim())) {
          throw std::logic_error("[verifyGraph] Invalid Roll axis");
        }
        if (node->shape() != n.input()->shape()) {
          shapeErr("Roll");
        }
        break;
      }
      case NodeType::Transpose: {
        auto& n = node->impl<TransposeNode>();
        if (node->shape() !=
            inferTransposeOutputShape(n.input()->shape(), n.axes()))
          shapeErr("Transpose");
        break;
      }
      case NodeType::Triangular: {
        auto& n = node->impl<TriangularNode>();
        if (n.input()->shape().ndim() < 2) {
          throw std::logic_error(
              "[verifyGraph] Triangular input rank must be at least 2");
        }
        if (node->shape() != n.input()->shape()) {
          shapeErr("Triangular");
        }
        if (node->dataType() != n.input()->dataType()) {
          throw std::logic_error(
              "[verifyGraph] Triangular output dtype mismatch");
        }
        break;
      }
      case NodeType::Tile: {
        auto& n = node->impl<TileNode>();
        if (node->shape() !=
            inferTileOutputShape(n.input()->shape(), n.tileDims()))
          shapeErr("Tile");
        break;
      }
      case NodeType::Where: {
        auto& n = node->impl<WhereNode>();
        if (n.condition()->dataType() != dtype::b8) {
          throw std::logic_error(
              "[verifyGraph] Where condition must have b8 dtype");
        }
        if (node->dataType() !=
            promoteDtypes(n.x()->dataType(), n.y()->dataType())) {
          throw std::logic_error("[verifyGraph] Where output dtype mismatch");
        }
        if (node->shape() !=
            inferWhereOutputShape(
                n.condition()->shape(), n.x()->shape(), n.y()->shape()))
          shapeErr("Where");
        break;
      }
      case NodeType::Matmul: {
        auto& n = node->impl<MatmulNode>();
        if (node->shape() !=
            inferMatmulOutputShape(
                n.lhs()->shape(), n.rhs()->shape(), n.lhsProp(), n.rhsProp()))
          shapeErr("Matmul");
        break;
      }
      case NodeType::Pad: {
        auto& n = node->impl<PadNode>();
        if (n.padWidths().size() !=
            static_cast<size_t>(n.input()->shape().ndim())) {
          throw std::logic_error(
              "[verifyGraph] Pad widths must be rank-normalized");
        }
        for (const auto& [before, after] : n.padWidths()) {
          if (before < 0 || after < 0) {
            throw std::logic_error(
                "[verifyGraph] Pad widths must be nonnegative");
          }
        }
        if (node->dataType() != n.input()->dataType()) {
          throw std::logic_error("[verifyGraph] Pad output dtype mismatch");
        }
        if (node->shape() !=
            inferPadOutputShape(n.input()->shape(), n.padWidths()))
          shapeErr("Pad");
        break;
      }
      default:
        break;
    }

    // Check 3: reachable nodes with ExternalUses must have internal consumers
    // (or be the root, or be a ProjectionNode of a multi-output parent).
    if (!node->externalUses().empty() && node != root.get()) {
      bool ok = !node->uses().empty();
      if (!ok && node->isProjection()) {
        auto parent = node->impl<ProjectionNode>().parent();
        ok = parent && parent->numOutputs() > 0;
      }
      if (!ok) {
        throw std::logic_error(
            "[verifyGraph] Orphaned ExternalUse: node has ExternalUses "
            "but no internal uses. A fusion pass may have absorbed this "
            "node without redirecting its ExternalUses.");
      }
    }
  }
}

} // namespace fl

#else // FL_DISABLE_JIT_GRAPH_VERIFY

namespace fl {
inline void verifyRootMetadata(const NodePtr&, const Shape&, dtype) {}
inline void verifyGraph(const NodePtr&) {}
} // namespace fl

#endif
