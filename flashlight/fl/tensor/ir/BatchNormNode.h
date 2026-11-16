/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/ir/Node.h"
#include "flashlight/fl/tensor/ir/OutputDesc.h"

namespace fl {

class BatchNormNode;
using BatchNormNodePtr = std::shared_ptr<BatchNormNode>;

/**
 * A node representing a batch normalization forward operation.
 *
 * This is a multi-output node with 5 outputs:
 *   0: Result        — normalized output
 *   1: SaveMean      — batch mean (for backward)
 *   2: SaveVar       — batch variance (for backward)
 *   3: NewRunningMean — updated running mean
 *   4: NewRunningVar  — updated running variance
 *
 * Use ProjectionNode to extract individual outputs.
 *
 * Inputs (variable count):
 *   [0] input        — always present
 *   [1] weight       — may be empty-value node
 *   [2] bias         — may be empty-value node
 *   [3] runningMean  — may be empty-value node
 *   [4] runningVar   — may be empty-value node
 */
class BatchNormNode : public NodeTrait<BatchNormNode> {
 public:
  static constexpr NodeType nodeType = NodeType::BatchNorm;

  // Output indices for use with ProjectionNode.
  // Non-class enum: implicitly converts to OutputIndex (unsigned).
  enum Out : OutputIndex {
    Result = 0,
    SaveMean = 1,
    SaveVar = 2,
    NewRunningMean = 3,
    NewRunningVar = 4,
  };
  static constexpr unsigned kNumOutputs = 5;

  static BatchNormNodePtr create(
      std::vector<NodePtr> inputs,
      std::vector<int> axes,
      bool train,
      double momentum,
      double epsilon,
      std::vector<OutputDesc> outputDescs,
      std::shared_ptr<const Stream> stream = nullptr);

  const std::vector<int>& axes() const;
  bool train() const;
  double momentum() const;
  double epsilon() const;

  // Input accessors
  NodePtr inputTensor() const;
  NodePtr weight() const;
  NodePtr bias() const;
  NodePtr runningMean() const;
  NodePtr runningVar() const;

  const std::vector<OutputDesc>& outputDescs() const;

  // Returns true when the live output mask indicates that backward-pass
  // state (SaveMean or SaveVar) is needed. Used by the evaluator to
  // decide whether to request state from the backend.
  bool needsState(uint32_t liveMask) const;

  // Cached forward state for backward pass (logical const — see Conv2dNode)
  mutable ForwardBackwardState cachedState_;

 private:
  std::vector<int> axes_;
  bool train_;
  double momentum_;
  double epsilon_;
  std::vector<OutputDesc> outputDescs_;

  struct PrivateHelper {};

 public:
  BatchNormNode(
      std::vector<NodePtr> inputs,
      std::vector<int> axes,
      bool train,
      double momentum,
      double epsilon,
      std::vector<OutputDesc> outputDescs,
      const Shape& shape,
      dtype type,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);
};

} // namespace fl
