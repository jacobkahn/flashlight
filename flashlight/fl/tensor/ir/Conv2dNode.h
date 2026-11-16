/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/ir/Node.h"

namespace fl {

class Conv2dNode;
using Conv2dNodePtr = std::shared_ptr<Conv2dNode>;

/**
 * A node that represents a 2D convolution operation.
 * Captures conv parameters so the JIT optimizer can pattern-match
 * activation(conv2d(...)) and replace it with a fused cuDNN call.
 *
 * Layout is NCHW (row-major). Inputs:
 *   [0] input   — (N, C_in, H, W)
 *   [1] weights — (C_out, C_in/groups, kH, kW)
 *   [2] bias    — (1, C_out, 1, 1) or empty
 */
class Conv2dNode : public NodeTrait<Conv2dNode> {
  int sx_, sy_, px_, py_, dx_, dy_, groups_;
  bool wantState_;

  static constexpr unsigned kInputIdx = 0;
  static constexpr unsigned kWeightsIdx = 1;
  static constexpr unsigned kBiasIdx = 2;

  // Mutable cache for ForwardBackwardState from evaluation
  mutable ForwardBackwardState cachedState_;

  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::Conv2d;

  Conv2dNode(
      NodePtr input,
      NodePtr weights,
      NodePtr bias,
      int sx, int sy, int px, int py, int dx, int dy, int groups,
      bool wantState,
      const Shape& outputShape,
      dtype outputDtype,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static Conv2dNodePtr create(
      NodePtr input,
      NodePtr weights,
      NodePtr bias,
      int sx, int sy, int px, int py, int dx, int dy, int groups,
      bool wantState,
      std::shared_ptr<const Stream> stream = nullptr);

  // Accessors
  NodePtr input() const;
  NodePtr weights() const;
  NodePtr bias() const;
  int sx() const;
  int sy() const;
  int px() const;
  int py() const;
  int dx() const;
  int dy() const;
  int groups() const;
  bool wantState() const;

  // Cached state from evaluation (for backward pass)
  void setCachedState(ForwardBackwardState s) const;
  const ForwardBackwardState& cachedState() const;
};

} // namespace fl
