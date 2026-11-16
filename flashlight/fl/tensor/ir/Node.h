/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <list>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/ir/NodeType.h"
#include "flashlight/fl/tensor/ir/Use.h"

namespace fl {

// Index type for multi-output node projections. Non-class enums on
// multi-output node types (e.g. BatchNormNode::Out) inherit from this
// so they implicitly convert to unsigned.
using OutputIndex = unsigned;

class Stream;
class ExternalUse;
using ExternalUseList = std::list<ExternalUse*>;
using UseList = std::list<std::unique_ptr<Use>>;
using NodePtr = std::shared_ptr<Node>;

/**
 * A Node represents a step in some Tensor computation.
 *
 * Conceptually, nodes form a computation DAG where the result of the
 * computation is immutable (this must be enforced by all graph rewrites).
 */
class Node {
  std::vector<NodePtr> inputs_;
  std::vector<UseList::iterator> inputUseIters_;
  UseList uses_;
  ExternalUseList externalUses_;
  const Shape shape_;
  const dtype dtype_;
  std::shared_ptr<const Stream> stream_;

  // present if this node has been evaluated
  std::optional<Tensor> result_{std::nullopt};

  // Pre-allocated output buffer for buffer assignment.
  // Set by Evaluator::assignBuffers() or JitTensorBase::device().
  // The evaluator writes the op result into this buffer when possible.
  std::optional<Tensor> outputBuffer_{std::nullopt};

  // set the input at `inputIdx` -- `unlinkInput` should be used to clear the
  // old input at `inputIdx`, if any.
  void linkInput(unsigned inputIdx, NodePtr input);
  void unlinkInput(unsigned inputIdx);

  static void nodeImplTypeCheck(NodeType expect, NodeType actual);

 protected:
  // Multi-output support — only used by multi-output node types.
  // Empty for single-output nodes (result_ used directly).
  // Sized at construction for multi-output nodes (result_ unused).
  std::vector<std::optional<Tensor>> outputs_;

  // A constructor that sets up all the metadata
  Node(
      std::vector<NodePtr>&& inputs,
      const Shape& shape,
      dtype dtype,
      std::shared_ptr<const Stream> stream = nullptr);

  // Help enforce internal consistency.
  NodePtr getInput(unsigned inputIdx) const;

 public:
  virtual ~Node();

  // Inputs
  const std::vector<NodePtr>& inputs() const;
  void setInput(unsigned inputIdx, NodePtr newInput);

  // Shape
  const Shape& shape() const;

  // Dtype (statically inferred at construction time)
  dtype dataType() const;

  // Stream on which this op should execute (nullptr = backend default)
  const std::shared_ptr<const Stream>& stream() const;

  // Uses
  const UseList& uses() const;
  const ExternalUseList& externalUses() const;
  // replaces both internal and external uses
  void replaceAllUsesWith(NodePtr newInput);

  // Useful for lazy eval (single-output nodes)
  const std::optional<Tensor>& getResult() const;
  void setResult(Tensor&& tensor);
  void unsetResult();

  // Output buffer assignment (pre-allocated destination for op result)
  void setOutputBuffer(Tensor buffer);
  const std::optional<Tensor>& getOutputBuffer() const;
  void clearOutputBuffer();

  // Multi-output nodes: store/retrieve individual outputs by index.
  void setOutput(OutputIndex index, Tensor result);
  const Tensor& getOutput(OutputIndex index) const;
  unsigned numOutputs() const;
  bool hasOutput(OutputIndex index) const;

  // Convenient type checks
  bool isBinary() const;
  bool isCast() const;
  bool isClip() const;
  bool isConcatenate() const;
  bool isCustom() const;
  bool isCumsum() const;
  bool isIndex() const;
  bool isScalar() const;
  bool isValue() const;
  bool isIndexedUpdate() const;
  bool isFlip() const;
  bool isRoll() const;
  bool isUnary() const;
  bool isReshape() const;
  bool isReduction() const;
  bool isTranspose() const;
  bool isTriangular() const;
  bool isFusedElementwise() const;
  bool isFusedReduceElementwise() const;
  bool isMatmul() const;
  bool isConv2d() const;
  bool isProjection() const;
  bool isBatchNorm() const;
  bool isPool2d() const;
  bool isMultiValue() const;
  bool isTransfer() const;
  bool isPad() const;
  bool isTile() const;
  bool isWhere() const;

  // Fast & safe casts
  virtual NodeType type() const = 0;

  template <typename T>
  T& impl() {
    nodeImplTypeCheck(T::nodeType, this->type());
    return *static_cast<T*>(this);
  }

  template <typename T>
  const T& impl() const {
    nodeImplTypeCheck(T::nodeType, this->type());
    return *static_cast<const T*>(this);
  }

  /**
   * Cast NodePtr to its actual derived type, useful if caller wishes to obtain
   * a derived NodePtr back.
   *
   * @param node the node to cast from
   * @return node the result of the cast.
   *
   * Example:
   *   NodePtr node = ...;
   *   BinaryNodePtr binaryNode = Node::cast<BinaryNode>(node);
   */
  template <typename T, typename E = typename T::element_type>
  static T cast(NodePtr node) {
    nodeImplTypeCheck(E::nodeType, node->type());
    return std::static_pointer_cast<E>(node);
  }

  friend class ExternalUse;
};

/**
 * A trait for some generic Node functionalities.
 *
 * REQUIRED definition in derived class:
 *   public: static constexpr NodeType nodeType;
 */
template <typename Derived>
class NodeTrait : public Node {
 public:
  NodeTrait(
      std::vector<NodePtr>&& inputs,
      const Shape& shape,
      dtype dtype,
      std::shared_ptr<const Stream> stream = nullptr)
      : Node(std::move(inputs), std::move(shape), dtype, std::move(stream)) {}

  NodeType type() const override {
    return Derived::nodeType;
  }
};

} // namespace fl
