/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <optional>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/ir/Node.h"

namespace fl {

class IndexedUpdateNode;
using IndexedUpdateNodePtr = std::shared_ptr<IndexedUpdateNode>;

/**
 * A node that represents an indexed update in SSA form.
 * let output = IndexedUpdateNode(indexedNode, indexings, updateDataNode),
 *     indexedTensor = indexedNode.value(),
 *     updateDataTensor = updateDataNode.value(),
 * we have
 *   output == indexedTensor __after__
 *      indexedTensor(indexings[0])(indexings[1])... = updateDataTensor
 *
 * But note that since the graph represents pure computation (SSA, no
 * assignment), this node conceptually creates a new tensor that holds the
 * result of the indexed update, thus the name.
 */
class IndexedUpdateNode : public NodeTrait<IndexedUpdateNode> {
  const std::vector<std::vector<Index>> indexings_;
  const std::optional<Shape> preReshapeShape_;

  // helps indexing into inputs
  static constexpr unsigned indexedNodeIdx = 0;
  static constexpr unsigned updateDataNodeIdx = 1;

  // help control allocation while allowing `std::make_shared`
  struct PrivateHelper{};

 public:
  static constexpr NodeType nodeType = NodeType::IndexedUpdate;
  IndexedUpdateNode(
      NodePtr indexedNode,
      const std::vector<std::vector<Index>>& indexings,
      NodePtr updateDataNode,
      std::optional<Shape> preReshapeShape,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static IndexedUpdateNodePtr create(
      NodePtr indexedNode,
      const std::vector<std::vector<Index>>& indexings,
      NodePtr updateDataNode,
      std::optional<Shape> preReshapeShape = std::nullopt,
      std::shared_ptr<const Stream> stream = nullptr);

  NodePtr indexedNode() const;
  const std::vector<std::vector<Index>>& indexings() const;
  NodePtr updateDataNode() const;
  bool hasPreReshape() const { return preReshapeShape_.has_value(); }
  const Shape& preReshapeShape() const { return *preReshapeShape_; }
};

} // namespace fl
