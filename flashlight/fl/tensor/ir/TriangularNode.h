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

enum class TriangularMode {
  Lower,
  Upper,
};

class TriangularNode;
using TriangularNodePtr = std::shared_ptr<TriangularNode>;

/**
 * A node representing lower/upper triangular masking.
 *
 * The mask is applied independently to submatrices over the last two
 * dimensions. `diagonal = 0` selects the main diagonal, positive offsets move
 * above the main diagonal, and negative offsets move below it.
 *
 * Input: [0] input tensor
 */
class TriangularNode : public NodeTrait<TriangularNode> {
  TriangularMode mode_;
  int diagonal_;

  static constexpr unsigned kInputIdx = 0;

  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::Triangular;

  TriangularNode(
      NodePtr input,
      TriangularMode mode,
      int diagonal,
      const Shape& outputShape,
      dtype dtype,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static TriangularNodePtr create(
      NodePtr input,
      TriangularMode mode,
      int diagonal,
      std::shared_ptr<const Stream> stream = nullptr);

  NodePtr input() const;
  TriangularMode mode() const;
  int diagonal() const;
};

} // namespace fl
