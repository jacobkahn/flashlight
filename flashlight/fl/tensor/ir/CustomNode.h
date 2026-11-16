/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/ir/Node.h"
#include "flashlight/fl/tensor/ir/OutputDesc.h"

#include <functional>
#include <optional>
#include <vector>
#include <memory>

namespace fl {

class CustomNode;
using CustomNodePtr = std::shared_ptr<CustomNode>;

/**
 * A node that holds customized evaluation logic for things like
 * backend-specific graph rewrite or as a fallback to capture maximal graph.
 *
 * Supports both single-output (EvalFunc) and multi-output
 * (MultiOutputEvalFunc) modes. Multi-output mode stores results in
 * Node::outputs_ and is accessed via ProjectionNode.
 */
class CustomNode : public NodeTrait<CustomNode> {
 public:
  using EvalFunc = std::function<Tensor(const std::vector<const Tensor*>&)>;
  using MultiOutputEvalFunc =
      std::function<std::vector<Tensor>(const std::vector<const Tensor*>&)>;

  // help control allocation while allowing `std::make_shared`
  struct PrivateHelper{};

 private:
  const std::string name_;
  const EvalFunc evalFunc_;
  const std::optional<MultiOutputEvalFunc> multiOutputEvalFunc_;

 public:
  static constexpr NodeType nodeType = NodeType::Custom;

  // Single-output constructor
  CustomNode(
      std::string&& name,
      std::vector<NodePtr>&& inputs,
      const Shape& shape,
      dtype dtype,
      EvalFunc&& evalFunc,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  // Multi-output constructor
  CustomNode(
      std::string&& name,
      std::vector<NodePtr>&& inputs,
      std::vector<OutputDesc> outputDescs,
      MultiOutputEvalFunc&& evalFunc,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static CustomNodePtr create(
      std::string&& debugName,
      std::vector<NodePtr>&& inputs,
      const Shape& shape,
      dtype dtype,
      EvalFunc&& evalFunc,
      std::shared_ptr<const Stream> stream = nullptr);

  static CustomNodePtr createMultiOutput(
      std::string&& debugName,
      std::vector<NodePtr>&& inputs,
      std::vector<OutputDesc> outputDescs,
      MultiOutputEvalFunc&& evalFunc,
      std::shared_ptr<const Stream> stream = nullptr);

  const std::string& name() const;
  const EvalFunc& evalFunc() const;
  bool isMultiOutput() const;
  const MultiOutputEvalFunc& multiOutputEvalFunc() const;
};

} // namespace fl
