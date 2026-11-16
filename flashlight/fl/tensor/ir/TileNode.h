/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/fl/tensor/ir/Node.h"

namespace fl {

class TileNode;
using TileNodePtr = std::shared_ptr<TileNode>;

/**
 * A node representing a tile (repeat) operation.
 *
 * Replicates the input tensor along each dimension by the corresponding
 * tile factor. Output shape: `output[i] = input[i] * tileDims[i]`.
 *
 * Input: [0] input tensor
 */
class TileNode : public NodeTrait<TileNode> {
  Shape tileDims_;

  static constexpr unsigned kInputIdx = 0;

  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::Tile;

  TileNode(
      NodePtr input,
      const Shape& tileDims,
      const Shape& outputShape,
      dtype dtype,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static TileNodePtr create(
      NodePtr input,
      const Shape& tileDims,
      std::shared_ptr<const Stream> stream = nullptr);

  NodePtr input() const;
  const Shape& tileDims() const;
};

} // namespace fl
