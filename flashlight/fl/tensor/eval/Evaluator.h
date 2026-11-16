/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>

#include "flashlight/fl/runtime/Stream.h"
#include "flashlight/fl/tensor/BackendDispatch.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/ir/BatchNormNode.h"
#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/CastNode.h"
#include "flashlight/fl/tensor/ir/ClipNode.h"
#include "flashlight/fl/tensor/ir/ConcatenateNode.h"
#include "flashlight/fl/tensor/ir/Conv2dNode.h"
#include "flashlight/fl/tensor/ir/CumsumNode.h"
#include "flashlight/fl/tensor/ir/CustomNode.h"
#include "flashlight/fl/tensor/ir/FlipNode.h"
#include "flashlight/fl/tensor/ir/FusedElementwiseNode.h"
#include "flashlight/fl/tensor/ir/FusedReduceElementwiseNode.h"
#include "flashlight/fl/tensor/ir/IndexNode.h"
#include "flashlight/fl/tensor/ir/IndexedUpdateNode.h"
#include "flashlight/fl/tensor/ir/MatmulNode.h"
#include "flashlight/fl/tensor/ir/PadNode.h"
#include "flashlight/fl/tensor/ir/Pool2dNode.h"
#include "flashlight/fl/tensor/ir/ProjectionNode.h"
#include "flashlight/fl/tensor/ir/ReductionNode.h"
#include "flashlight/fl/tensor/ir/ReshapeNode.h"
#include "flashlight/fl/tensor/ir/RollNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/TileNode.h"
#include "flashlight/fl/tensor/ir/TransferNode.h"
#include "flashlight/fl/tensor/ir/TransposeNode.h"
#include "flashlight/fl/tensor/ir/TriangularNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/ir/WhereNode.h"

namespace fl {

/**
 * A JIT tree evaluator. It dispatches to another Tensor Backend for carrying
 * out the computation represented by the JIT tree.
 */
class Evaluator {
 public:
  // takes in the evaluated node and execution stats (empty if profiling is
  // disabled)
  using PostEvalCallback =
      std::function<void(NodePtr, std::unordered_map<NodePtr, float>)>;
  using PostEvalCallbackList = std::list<PostEvalCallback>;
  using PostEvalCallbackHandle = PostEvalCallbackList::iterator;

 private:
  // function-pointer dispatch table for Tensor ops.
  BackendDispatch dispatch_;
  // track time spent on executing a node alone (not its inputs)
  std::unordered_map<NodePtr, float> nodeToTotTimeMs_{};
  bool profilerEnabled_{false};
  PostEvalCallbackList postEvalCallbacks_;

  struct EvalPlan;
  static EvalPlan buildPlan(NodePtr root);
  void assignBuffer(const NodePtr& node);
  void executePlan(EvalPlan& plan);
  void evalNodeDispatch(NodePtr node);
  // profile execution time of `func` and associate it with `nodePtr`
  void profile(std::function<void()> func, NodePtr nodePtr);

  // evaluate and set result without checking for existing result
  // ASSUME inputs have been evaluated
  void evalBinaryNode(BinaryNodePtr node);
  void evalCastNode(CastNodePtr node);
  void evalClipNode(ClipNodePtr node);
  void evalConcatenateNode(ConcatenateNodePtr node);
  void evalCustomNode(CustomNodePtr node);
  void evalCumsumNode(CumsumNodePtr node);
  void evalFlipNode(FlipNodePtr node);
  void evalIndexNode(IndexNodePtr node);
  void evalIndexedUpdateNode(IndexedUpdateNodePtr node);
  // JitTensor in indices becomes the backing tensor
  std::vector<Index> unwrapTensorInIndices(const std::vector<Index>& indices);
  void evalScalarNode(ScalarNodePtr node);
  void evalUnaryNode(UnaryNodePtr node);
  void evalReshapeNode(ReshapeNodePtr node);
  void evalReductionNode(ReductionNodePtr node);
  void evalRollNode(RollNodePtr node);
  void evalTransposeNode(TransposeNodePtr node);
  void evalTriangularNode(TriangularNodePtr node);
  void evalFusedElementwiseNode(FusedElementwiseNodePtr node);
  void evalFusedReduceElementwiseNode(FusedReduceElementwiseNodePtr node);
  void evalMatmulNode(MatmulNodePtr node);
  void evalConv2dNode(Conv2dNodePtr node);
  void evalProjectionNode(ProjectionNodePtr node);
  void evalBatchNormNode(BatchNormNodePtr node);
  void evalPadNode(PadNodePtr node);
  void evalPool2dNode(Pool2dNodePtr node);
  void evalTileNode(TileNodePtr node);
  void evalTransferNode(TransferNodePtr node);
  void evalWhereNode(WhereNodePtr node);

  // helpers that evaluates without setting results
  void
  evalBinaryOp(Tensor& out, BinaryOp op, const Tensor& lhs, const Tensor& rhs);
  void evalUnaryOp(Tensor& out, UnaryOp op, const Tensor& input);
  void evalReductionOp(
      Tensor& out,
      ReductionOp op,
      const Tensor& input,
      const std::vector<int>& axes,
      bool keepDims);
  void evalScalar(Tensor& out, ScalarNodePtr node);

 public:
  /**
   * Creates a JIT graph Evaluator that dispatches via the given function
   * pointers.
   */
  explicit Evaluator(BackendDispatch dispatch);

  // no copy/move
  Evaluator(const Evaluator&) = delete;
  Evaluator(Evaluator&&) = delete;
  Evaluator& operator=(const Evaluator&) = delete;
  Evaluator& operator=(const Evaluator&&) = delete;

  /**
   * Execute the entire computation tree rooted at `node`.
   * 1. no op if result already set
   * 2. set result for intermediate nodes if they have external uses
   */
  void eval(NodePtr node);

  /**
   * TODO document
   */
  void setProfilerState(bool active);
  bool getProfilerState();
  const std::unordered_map<NodePtr, float>& getProfilerStats();
  void clearProfilerStats();

  PostEvalCallbackHandle addPostEvalCallback(PostEvalCallback callback);
  void removePostEvalCallback(PostEvalCallbackHandle handle);
};

} // namespace fl
