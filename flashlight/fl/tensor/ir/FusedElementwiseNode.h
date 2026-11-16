/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <variant>
#include <vector>

#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/Node.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"

namespace fl {

/**
 * A single step in a fused elementwise program.
 *
 * Each step reads from indexed sources and produces one output.
 * Source indices 0..numInputs-1 refer to the node's external inputs.
 * Source indices >= numInputs refer to outputs of previous steps
 * (step i's output is at index numInputs + i).
 */
struct FusedOp {
  std::variant<UnaryOp, BinaryOp> op;
  unsigned src0;
  unsigned src1; // only used for BinaryOp

  bool isUnary() const {
    return std::holds_alternative<UnaryOp>(op);
  }
  bool isBinary() const {
    return std::holds_alternative<BinaryOp>(op);
  }
  UnaryOp unaryOp() const {
    return std::get<UnaryOp>(op);
  }
  BinaryOp binaryOp() const {
    return std::get<BinaryOp>(op);
  }
};

class FusedElementwiseNode;
using FusedElementwiseNodePtr = std::shared_ptr<FusedElementwiseNode>;

/**
 * A node representing a fused sequence of elementwise operations.
 *
 * Stores a mini-program of unary and binary ops that are evaluated in sequence.
 * The output is the result of the last step. All intermediate results are
 * temporary and never materialized as separate tensors (when lowered to a
 * backend-specific kernel).
 *
 * This node is created by the ElementwiseFusion pass and can be:
 * - Evaluated sequentially by the default Evaluator (fallback)
 * - Lowered to a fused kernel by backend-specific passes (e.g., NVRTC for CUDA,
 *   eltwise post-ops for OneDNN)
 */
class FusedElementwiseNode : public NodeTrait<FusedElementwiseNode> {
  std::vector<FusedOp> program_;
  unsigned numExternalInputs_;

  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::FusedElementwise;
  FusedElementwiseNode(
      std::vector<NodePtr>&& inputs,
      std::vector<FusedOp> program,
      unsigned numExternalInputs,
      const Shape& shape,
      dtype dtype,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static FusedElementwiseNodePtr create(
      std::vector<NodePtr> inputs,
      std::vector<FusedOp> program,
      unsigned numExternalInputs,
      const Shape& shape,
      std::shared_ptr<const Stream> stream = nullptr);

  const std::vector<FusedOp>& program() const;
  unsigned numExternalInputs() const;

  /**
   * Number of steps (ops) in the fused program.
   */
  unsigned numSteps() const;
};

} // namespace fl
