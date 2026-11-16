/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/ir/Node.h"

namespace fl {

class MatmulNode;
using MatmulNodePtr = std::shared_ptr<MatmulNode>;

/**
 * A node that represents matrix multiplication.
 * Captures MatrixProperty for each operand (None or Transpose).
 */
class MatmulNode : public NodeTrait<MatmulNode> {
  const MatrixProperty lhsProp_;
  const MatrixProperty rhsProp_;

  static constexpr unsigned kLhsIdx = 0;
  static constexpr unsigned kRhsIdx = 1;

  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::Matmul;
  MatmulNode(
      NodePtr lhs,
      NodePtr rhs,
      MatrixProperty lhsProp,
      MatrixProperty rhsProp,
      const Shape& shape,
      dtype dtype,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static MatmulNodePtr create(
      NodePtr lhs,
      NodePtr rhs,
      MatrixProperty lhsProp,
      MatrixProperty rhsProp,
      std::shared_ptr<const Stream> stream = nullptr);

  MatrixProperty lhsProp() const;
  MatrixProperty rhsProp() const;
  NodePtr lhs() const;
  NodePtr rhs() const;
};

} // namespace fl
