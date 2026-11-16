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

class CastNode;
using CastNodePtr = std::shared_ptr<CastNode>;

/**
 * A node that represents a type cast operation.
 * Output shape equals input shape; dtype changes to the target type.
 */
class CastNode : public NodeTrait<CastNode> {
  const dtype targetDtype_;

  static constexpr unsigned kInputIdx = 0;

  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::Cast;
  CastNode(
      NodePtr input,
      dtype targetDtype,
      const Shape& shape,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  // Returns NodePtr (not CastNodePtr) because it may return the input
  // directly for no-op casts (same dtype).
  static NodePtr create(
      NodePtr input,
      dtype targetDtype,
      std::shared_ptr<const Stream> stream = nullptr);

  NodePtr input() const;
  dtype targetDtype() const;
};

} // namespace fl
