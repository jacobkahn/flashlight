/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

#include "flashlight/fl/tensor/ir/Node.h"

namespace fl {

class ConcatenateNode;
using ConcatenateNodePtr = std::shared_ptr<ConcatenateNode>;

/**
 * A node representing variadic tensor concatenation along one axis.
 *
 * Input: [0..N) tensors to concatenate, in order.
 */
class ConcatenateNode : public NodeTrait<ConcatenateNode> {
  unsigned axis_;

  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::Concatenate;

  ConcatenateNode(
      std::vector<NodePtr> inputs,
      unsigned axis,
      const Shape& outputShape,
      dtype dtype,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static ConcatenateNodePtr create(
      const std::vector<NodePtr>& inputs,
      unsigned axis,
      std::shared_ptr<const Stream> stream = nullptr);

  NodePtr input(unsigned inputIdx) const;
  unsigned numInputs() const;
  unsigned axis() const;
};

} // namespace fl
