/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/IndexedUpdateNode.h"

#include <stdexcept>

#include "flashlight/fl/tensor/JitTensorBase.h"

namespace fl {

namespace {

std::vector<NodePtr> computeInputs(
    NodePtr indexedNode,
    NodePtr updateDataNode,
    const std::vector<std::vector<Index>>& indexings) {
  std::vector<NodePtr> inputs{indexedNode, updateDataNode};
  for (const auto& indexing : indexings) {
    for (const auto& idx : indexing) {
      switch (idx.type()) {
        case detail::IndexType::Tensor: {
          const auto& tensorIdx = idx.get<Tensor>();
          const auto tensorIdxNode = toJitTensorBase(tensorIdx).node();
          inputs.push_back(tensorIdxNode);
        }
        default:
          continue;
      }
    }
  }
  return inputs;
}

} // namespace

IndexedUpdateNode::IndexedUpdateNode(
    NodePtr indexedNode,
    const std::vector<std::vector<Index>>& indexings,
    NodePtr updateDataNode,
    std::optional<Shape> preReshapeShape,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait(
          computeInputs(indexedNode, updateDataNode, indexings),
          Shape(indexedNode->shape()),
          indexedNode->dataType(),
          std::move(stream)),
      indexings_(indexings),
      preReshapeShape_(std::move(preReshapeShape)) {}

IndexedUpdateNodePtr IndexedUpdateNode::create(
    NodePtr indexedNode,
    const std::vector<std::vector<Index>>& indexings,
    NodePtr updateDataNode,
    std::optional<Shape> preReshapeShape,
    std::shared_ptr<const Stream> stream) {
  return std::make_shared<IndexedUpdateNode>(indexedNode, indexings, updateDataNode, std::move(preReshapeShape), std::move(stream), PrivateHelper{});
}

NodePtr IndexedUpdateNode::indexedNode() const {
  return getInput(indexedNodeIdx);
}

const std::vector<std::vector<Index>>& IndexedUpdateNode::indexings() const {
  return indexings_;
}

NodePtr IndexedUpdateNode::updateDataNode() const {
  return getInput(updateDataNodeIdx);
}

} // namespace fl
