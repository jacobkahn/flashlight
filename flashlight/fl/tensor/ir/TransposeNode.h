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

class TransposeNode;
using TransposeNodePtr = std::shared_ptr<TransposeNode>;

/**
 * A node that represents a transpose (axis permutation) operation.
 * Output shape is determined by the axis permutation; dtype is preserved.
 */
class TransposeNode : public NodeTrait<TransposeNode> {
  const Shape axes_; // permutation, or empty for reverse-all

  static constexpr unsigned kInputIdx = 0;

  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::Transpose;
  TransposeNode(
      NodePtr input,
      Shape axes,
      const Shape& outputShape,
      dtype dtype,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static TransposeNodePtr create(
      NodePtr input,
      const Shape& axes,
      std::shared_ptr<const Stream> stream = nullptr);

  NodePtr input() const;
  const Shape& axes() const;
};

} // namespace fl
