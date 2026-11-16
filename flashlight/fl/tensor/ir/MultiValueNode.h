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

class MultiValueNode;
using MultiValueNodePtr = std::shared_ptr<MultiValueNode>;

/**
 * A leaf node holding multiple pre-computed tensors (multi-output compaction).
 *
 * When the evaluator compacts a multi-output node (e.g. a fused CustomNode or
 * BatchNormNode), it replaces the parent with a single MultiValueNode instead
 * of N independent ValueNodes. ProjectionNodes continue to reference the
 * MultiValueNode, preserving sibling relationships — which is required for
 * optimizations that recognize co-outputs (e.g. FusionSideOutput).
 *
 * Like ValueNode, this is a leaf (no inputs) and requires no evaluation.
 */
class MultiValueNode : public NodeTrait<MultiValueNode> {
 public:
  static constexpr NodeType nodeType = NodeType::MultiValue;

  /// Create a MultiValueNode holding pre-computed tensors.
  /// Shape/dtype come from values[0] (the "primary" output).
  static MultiValueNodePtr create(
      std::vector<Tensor> values,
      std::shared_ptr<const Stream> stream = nullptr);

  unsigned numValues() const { return numOutputs(); }

 private:
  MultiValueNode(
      Shape shape,
      dtype dataType,
      std::vector<Tensor> values,
      std::shared_ptr<const Stream> stream);
};

} // namespace fl
