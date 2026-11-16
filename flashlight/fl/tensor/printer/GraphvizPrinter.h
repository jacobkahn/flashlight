/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <unordered_set>

#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/TensorBackend.h"
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
#include "flashlight/fl/tensor/ir/TransposeNode.h"
#include "flashlight/fl/tensor/ir/TriangularNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"
#include "flashlight/fl/tensor/ir/WhereNode.h"

namespace fl {

namespace detail {

class NameGenerator {
  unsigned namemCounter_;

 public:
  NameGenerator();
  std::string genName(std::string prefix);
};

class NodeNamer {
  const std::string prefix_;
  NameGenerator nodeNameGenerator_;
  std::unordered_map<NodePtr, std::string> nodeToName_;

 public:
  NodeNamer(const std::string& defaultPrefix);
  const std::string& getName(NodePtr node);
  bool contains(NodePtr node) const;
};

} // namespace detail

/**
 * A printer that prints a computation graph to GraphViz DOT format.
 */
class GraphvizPrinter {
 public:
  enum class Color {
    Black,
    Green,
    Red,
  };

 private:
  std::ostream& os_;
  detail::NameGenerator subgraphNameGenerator_;
  detail::NodeNamer nodeNamer_;
  std::unordered_set<NodePtr> edgesPrinted_;
  // temporary state use when printing a single subgraph
  const std::unordered_map<NodePtr, float>* nodeToTotTimeMs_{};
  float medianTotTime_ = 0;
  float maxTotTime_ = 0;

  Color edgeColor_ = Color::Black;
  Color nodeColor_ = Color::Black;

  std::ostream& os();
  void printBinaryNodeLabels(const BinaryNode& node);
  void printClipNodeLabels(const ClipNode& node);
  void printConcatenateNodeLabels(const ConcatenateNode& node);
  void printCustomNodeLabels(const CustomNode& node);
  void printCumsumNodeLabels(const CumsumNode& node);
  void printFlipNodeLabels(const FlipNode& node);
  void printIndexNodeLabels(const IndexNode& node);
  void printIndexedUpdateNodeLabels(const IndexedUpdateNode& node);
  std::ostream& printIndices(const std::vector<Index>& indices);
  void printRangeIndex(const range& rangeIdx);
  void printScalarNodeLabels(const ScalarNode& node);
  std::ostream& printScalarValue(const ScalarNode& node);
  void printUnaryNodeLabels(const UnaryNode& node);
  void printReshapeNodeLabels(const ReshapeNode& node);
  void printReductionNodeLabels(const ReductionNode& node);
  void printRollNodeLabels(const RollNode& node);
  void printTransposeNodeLabels(const TransposeNode& node);
  void printTriangularNodeLabels(const TriangularNode& node);
  void printCastNodeLabels(const CastNode& node);
  void printFusedElementwiseNodeLabels(const FusedElementwiseNode& node);
  void printFusedReduceElementwiseNodeLabels(
      const FusedReduceElementwiseNode& node);
  void printMatmulNodeLabels(const MatmulNode& node);
  void printConv2dNodeLabels(const Conv2dNode& node);
  void printProjectionNodeLabels(const ProjectionNode& node);
  void printBatchNormNodeLabels(const BatchNormNode& node);
  void printPadNodeLabels(const PadNode& node);
  void printPool2dNodeLabels(const Pool2dNode& node);
  void printTileNodeLabels(const TileNode& node);
  void printWhereNodeLabels(const WhereNode& node);
  void printValueNodeLabels(const ValueNode& node);
  std::ostream& printNodes(NodePtr node);
  std::ostream& printNodeLabels(NodePtr node);
  std::ostream& printNodeColor(float tottime);
  std::ostream& printRelativeColor(float tottime);
  std::ostream& printEdges(NodePtr node);

  GraphvizPrinter& printSubgraphImpl(
      NodePtr node,
      const std::string& namePrefix);

 public:
  // no copy/move
  GraphvizPrinter(std::ostream& os); // wrap
  ~GraphvizPrinter(); // wrap
  GraphvizPrinter(const GraphvizPrinter&) = delete;
  GraphvizPrinter(GraphvizPrinter&&) = delete;
  GraphvizPrinter& operator=(const GraphvizPrinter&) = delete;
  GraphvizPrinter& operator=(GraphvizPrinter&&) = delete;

  /**
   * Applies to all newly printed edges after this function call
   * TODO
   */
  GraphvizPrinter& setEdgeColor(Color newColor);

  /**
   * Applies to all newly printed nodes after this function call
   * TODO
   */
  GraphvizPrinter& setNodeColor(Color newColor);

  /**
   * Print the entire computation tree rooted at `node` to given stream as a
   * `subgraph` DOT construct. This way, user can visually differentiate graphs
   * from different materializations, if they print out a subgraph for each
   * materialization.
   *
   * TODO
   */
  GraphvizPrinter& printSubgraph(NodePtr node, const std::string& namePrefix);

  // color node with execution statistics
  GraphvizPrinter& printSubgraph(
      NodePtr node,
      const std::string& namePrefix,
      const std::unordered_map<NodePtr, float>& nodeToTotTimeMs_);
};

std::ostream& operator<<(std::ostream& os, const GraphvizPrinter::Color& color);

} // namespace fl
