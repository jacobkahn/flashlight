/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * @file FusionUtils.h
 *
 * Utilities for fusion passes that need to absorb multi-use nodes safely.
 *
 * ## Why this exists
 *
 * When `loss.scalar<float>()` triggers evaluation, the JIT evaluator replaces
 * the evaluated subgraph with a ValueNode. Forward and backward then become
 * separate graphs. Autograd holds ExternalUses on forward intermediate nodes
 * that backward needs. If a fusion pass absorbs such a node into a fused
 * CustomNode, the ExternalUse must transfer to a side-output of the fused
 * kernel — otherwise autograd's backward reads stale/garbage data.
 *
 * ## How it works
 *
 * 1. `findExternallyReferencedNodes()` scans absorbed nodes for ExternalUses.
 * 2. `buildOutputDescsForSideOutputs()` creates OutputDescs to append to
 *    the fused node's multi-output descriptor list.
 * 3. `createSideOutputProjections()` creates ProjectionNodes for each
 *    side-output and calls `replaceAllUsesWith()` to redirect ExternalUses.
 *
 * The evaluator's existing `computeLiveOutputMask()` handles dead output
 * elimination — if backward never reads an intermediate, the side-output
 * ProjectionNode has no ExternalUse and is automatically eliminated.
 *
 * ## When to use
 *
 * Passes should call these helpers when absorbing nodes that may have
 * ExternalUses (i.e., when relaxing the `isSingleUse()` check). Passes
 * that reject multi-use nodes via `isSingleUse()` don't need this.
 */

#include <utility>
#include <vector>

#include "flashlight/fl/tensor/ir/Node.h"
#include "flashlight/fl/tensor/ir/OutputDesc.h"
#include "flashlight/fl/tensor/ir/ProjectionNode.h"

namespace fl {

/// Describes a side-output that a fusion pass must emit to preserve an
/// ExternalUse on an absorbed node.
struct FusionSideOutput {
  NodePtr absorbedNode;  // The absorbed node with ExternalUses
  unsigned outputIndex;  // Its index in the fused multi-output node
};

/// Scan a list of absorbed nodes and return those that have non-empty
/// externalUses(). Pure query — no mutation.
inline std::vector<NodePtr> findExternallyReferencedNodes(
    const std::vector<NodePtr>& absorbed) {
  std::vector<NodePtr> result;
  for (const auto& node : absorbed) {
    if (!node->externalUses().empty()) {
      result.push_back(node);
    }
  }
  return result;
}

/// Build OutputDescs for side-outputs starting at `firstOutputIndex`.
///
/// Returns a pair:
///   - OutputDescs to append to the fused node's output descriptor list
///   - FusionSideOutput mapping for use with createSideOutputProjections()
inline std::pair<std::vector<OutputDesc>, std::vector<FusionSideOutput>>
buildOutputDescsForSideOutputs(
    const std::vector<NodePtr>& externallyReferenced,
    unsigned firstOutputIndex) {
  std::vector<OutputDesc> descs;
  std::vector<FusionSideOutput> sideOutputs;
  descs.reserve(externallyReferenced.size());
  sideOutputs.reserve(externallyReferenced.size());

  unsigned idx = firstOutputIndex;
  for (const auto& node : externallyReferenced) {
    descs.push_back({node->shape(), node->dataType()});
    sideOutputs.push_back({node, idx});
    ++idx;
  }
  return {std::move(descs), std::move(sideOutputs)};
}

/// Create ProjectionNodes for each side-output and redirect ExternalUses.
///
/// For each FusionSideOutput: creates a ProjectionNode extracting that output
/// from the fused node, then calls absorbedNode->replaceAllUsesWith(proj) to
/// atomically transfer all internal and external uses.
inline void createSideOutputProjections(
    const NodePtr& fusedNode,
    const std::vector<FusionSideOutput>& sideOutputs) {
  for (const auto& so : sideOutputs) {
    auto proj = ProjectionNode::create(
        fusedNode,
        so.outputIndex,
        so.absorbedNode->shape(),
        so.absorbedNode->dataType(),
        so.absorbedNode->stream());
    so.absorbedNode->replaceAllUsesWith(proj);
  }
}

} // namespace fl
