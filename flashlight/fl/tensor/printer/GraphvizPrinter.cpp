/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/printer/GraphvizPrinter.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <string>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/JitTensorBase.h"

namespace fl {

namespace detail {

NameGenerator::NameGenerator() : namemCounter_(0) {}

std::string NameGenerator::genName(std::string prefix) {
  return prefix + std::to_string(namemCounter_++);
}

NodeNamer::NodeNamer(const std::string& prefix) : prefix_(prefix) {}

const std::string& NodeNamer::getName(NodePtr node) {
  if (!nodeToName_.contains(node)) {
    nodeToName_.emplace(node, nodeNameGenerator_.genName(prefix_));
  }
  return nodeToName_.at(node);
}

bool NodeNamer::contains(NodePtr node) const {
  return nodeToName_.contains(node);
}

} // namespace detail

namespace {

const char* binopToStr(const BinaryOp op) {
  switch (op) {
    case BinaryOp::Add:
      return "Add";
    case BinaryOp::Sub:
      return "Sub";
    case BinaryOp::Mul:
      return "Mul";
    case BinaryOp::Div:
      return "Div";
    case BinaryOp::Eq:
      return "Eq";
    case BinaryOp::Neq:
      return "Neq";
    case BinaryOp::Gt:
      return "Gt";
    case BinaryOp::Gte:
      return "Gte";
    case BinaryOp::Lt:
      return "Lt";
    case BinaryOp::Lte:
      return "Lte";
    case BinaryOp::Min:
      return "Min";
    case BinaryOp::Max:
      return "Max";
    case BinaryOp::Pow:
      return "Pow";
    case BinaryOp::Mod:
      return "Mod";
    case BinaryOp::And:
      return "And";
    case BinaryOp::Or:
      return "Or";
    case BinaryOp::Shl:
      return "Shl";
    case BinaryOp::Shr:
      return "Shr";
    case BinaryOp::BitAnd:
      return "BitAnd";
    case BinaryOp::BitOr:
      return "BitOr";
    case BinaryOp::BitXor:
      return "BitXor";
  }
  throw std::runtime_error("Unsupported binary operation type");
}

const char* unaryOpToStr(const UnaryOp op) {
  switch (op) {
    case UnaryOp::Exp:
      return "Exp";
    case UnaryOp::Log:
      return "Log";
    case UnaryOp::Log1p:
      return "Log1p";
    case UnaryOp::Sin:
      return "Sin";
    case UnaryOp::Cos:
      return "Cos";
    case UnaryOp::Sqrt:
      return "Sqrt";
    case UnaryOp::Tanh:
      return "Tanh";
    case UnaryOp::Floor:
      return "Floor";
    case UnaryOp::Ceil:
      return "Ceil";
    case UnaryOp::Rint:
      return "Rint";
    case UnaryOp::Negative:
      return "Negative";
    case UnaryOp::Absolute:
      return "Absolute";
    case UnaryOp::Sigmoid:
      return "Sigmoid";
    case UnaryOp::Erf:
      return "Erf";
    case UnaryOp::LogicalNot:
      return "LogicalNot";
    case UnaryOp::Sign:
      return "Sign";
    case UnaryOp::Isnan:
      return "Isnan";
    case UnaryOp::Isinf:
      return "Isinf";
  }
  throw std::runtime_error("Unsupported unary operation type");
}

const char* reductionOpToStr(const ReductionOp op) {
  switch (op) {
    case ReductionOp::Amin:
      return "Amin";
    case ReductionOp::Amax:
      return "Amax";
    case ReductionOp::Sum:
      return "Sum";
    case ReductionOp::Mean:
      return "Mean";
    case ReductionOp::Median:
      return "Median";
    case ReductionOp::Std:
      return "Std";
    case ReductionOp::CountNonzero:
      return "CountNonzero";
    case ReductionOp::Any:
      return "Any";
    case ReductionOp::All:
      return "All";
  }
  throw std::runtime_error("Unsupported reduction operation type");
}

const char* padTypeToStr(const PadType type) {
  switch (type) {
    case PadType::Constant:
      return "Constant";
    case PadType::Edge:
      return "Edge";
    case PadType::Symmetric:
      return "Symmetric";
  }
  throw std::runtime_error("Unsupported pad type");
}

} // namespace

std::ostream& GraphvizPrinter::os() {
  return os_;
}

GraphvizPrinter& GraphvizPrinter::setEdgeColor(Color newColor) {
  this->edgeColor_ = newColor;
  return *this;
}

GraphvizPrinter& GraphvizPrinter::setNodeColor(Color newColor) {
  this->nodeColor_ = newColor;
  return *this;
}

void GraphvizPrinter::printBinaryNodeLabels(const BinaryNode& node) {
  os() << "BinaryNode"
       << "\\n"
       << "op = " << binopToStr(node.op()) << "\\n"
       << "shape = " << node.shape() << "\\n";
}

void GraphvizPrinter::printClipNodeLabels(const ClipNode& node) {
  os() << "ClipNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n";
}

void GraphvizPrinter::printConcatenateNodeLabels(const ConcatenateNode& node) {
  os() << "ConcatenateNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "axis = " << node.axis() << "\\n"
       << "inputs = " << node.numInputs() << "\\n";
}

void GraphvizPrinter::printCustomNodeLabels(const CustomNode& node) {
  os() << node.name() << "\\n"
       << "shape = " << node.shape() << "\\n";
}

void GraphvizPrinter::printCumsumNodeLabels(const CumsumNode& node) {
  os() << "CumsumNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "axis = " << node.axis() << "\\n";
}

void GraphvizPrinter::printFlipNodeLabels(const FlipNode& node) {
  os() << "FlipNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "axis = " << node.axis() << "\\n";
}

void GraphvizPrinter::printTriangularNodeLabels(const TriangularNode& node) {
  os() << "TriangularNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "mode = "
       << (node.mode() == TriangularMode::Lower ? "lower" : "upper") << "\\n"
       << "diagonal = " << node.diagonal() << "\\n";
}

void GraphvizPrinter::printIndexNodeLabels(const IndexNode& node) {
  os() << "IndexNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "indices = ";
  printIndices(node.indices()) << "\\n";
}

void GraphvizPrinter::printIndexedUpdateNodeLabels(
    const IndexedUpdateNode& node) {
  os() << "IndexedUpdateNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "indices = {\\n";
  for (const auto& indices : node.indexings()) {
    printIndices(indices) << "\\n";
  }
  os() << "}\"";
}

void GraphvizPrinter::printRangeIndex(const range& rangeIdx) {
  os() << rangeIdx.start() << ":";
  if (rangeIdx.end().has_value()) {
    os() << rangeIdx.end().value();
  }
  os() << ":" << rangeIdx.stride();
}

std::ostream& GraphvizPrinter::printIndices(const std::vector<Index>& indices) {
  os() << "[";
  for (unsigned i = 0; i < indices.size(); i++) {
    const auto& idx = indices[i];
    switch (idx.type()) {
      case detail::IndexType::Literal:
        os() << idx.get<Dim>();
        break;
      case detail::IndexType::Span:
        os() << ":";
        break;
      case detail::IndexType::Range:
        printRangeIndex(idx.get<range>());
        break;
      case detail::IndexType::Tensor:
        os() << "<Tensor>";
        break;
    }
    if (i != indices.size() - 1) {
      os() << ", ";
    }
  }
  return os();
}

std::ostream& GraphvizPrinter::printScalarValue(const ScalarNode& node) {
  switch (node.dataType()) {
    case dtype::f16:
    case dtype::bf16:
      throw std::runtime_error(
          "[GraphvizPrinter::printScalarNodeValue] f16/bf16 is unsupported");
    case dtype::f32:
      os() << node.scalar<float>();
      break;
    case dtype::f64:
      os() << node.scalar<double>();
      break;
    case dtype::b8:
      os() << node.scalar<char>();
      break;
    case dtype::s16:
      os() << node.scalar<short>();
      break;
    case dtype::s32:
      os() << node.scalar<int>();
      break;
    case dtype::s64:
      os() << node.scalar<long long>();
      break;
    case dtype::u8:
      os() << node.scalar<unsigned char>();
      break;
    case dtype::u16:
      os() << node.scalar<unsigned short>();
      break;
    case dtype::u32:
      os() << node.scalar<unsigned int>();
      break;
    case dtype::u64:
      os() << node.scalar<unsigned long long>();
      break;
    default:
      throw std::runtime_error("Unknown data type");
  }
  return os();
}

void GraphvizPrinter::printScalarNodeLabels(const ScalarNode& node) {
  os() << "ScalarNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "dtype = " << node.dataType() << "\\n"
       << "value = ";
  printScalarValue(node) << "\\n";
}

void GraphvizPrinter::printUnaryNodeLabels(const UnaryNode& node) {
  os() << "UnaryNode"
       << "\\n"
       << "op = " << unaryOpToStr(node.op()) << "\\n"
       << "shape = " << node.shape() << "\\n";
}

void GraphvizPrinter::printReshapeNodeLabels(const ReshapeNode& node) {
  os() << "ReshapeNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n";
}

void GraphvizPrinter::printTransposeNodeLabels(const TransposeNode& node) {
  os() << "TransposeNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "axes = [";
  for (int i = 0; i < node.axes().ndim(); i++) {
    os() << node.axes().dim(i);
    if (i != node.axes().ndim() - 1) {
      os() << ", ";
    }
  }
  os() << "]\\n";
}

void GraphvizPrinter::printCastNodeLabels(const CastNode& node) {
  os() << "CastNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "targetDtype = " << node.targetDtype() << "\\n";
}

void GraphvizPrinter::printReductionNodeLabels(const ReductionNode& node) {
  os() << "ReductionNode"
       << "\\n"
       << "op = " << reductionOpToStr(node.op()) << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "axes = [";
  for (unsigned i = 0; i < node.axes().size(); i++) {
    os() << node.axes()[i];
    if (i != node.axes().size() - 1) {
      os() << ", ";
    }
  }
  os() << "]\\n"
       << "keepDims = " << (node.keepDims() ? "true" : "false") << "\\n";
}

void GraphvizPrinter::printRollNodeLabels(const RollNode& node) {
  os() << "RollNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "shift = " << node.shift() << "\\n"
       << "axis = " << node.axis() << "\\n";
}

void GraphvizPrinter::printFusedElementwiseNodeLabels(
    const FusedElementwiseNode& node) {
  os() << "FusedElementwise"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "inputs = " << node.numExternalInputs() << "\\n"
       << "steps = [\\n";
  for (unsigned i = 0; i < node.program().size(); i++) {
    const auto& step = node.program()[i];
    if (step.isUnary()) {
      os() << "  " << unaryOpToStr(step.unaryOp()) << "(s" << step.src0 << ")";
    } else {
      os() << "  " << binopToStr(step.binaryOp()) << "(s" << step.src0 << ", s"
           << step.src1 << ")";
    }
    os() << " -> s" << (node.numExternalInputs() + i) << "\\n";
  }
  os() << "]\\n";
}

void GraphvizPrinter::printFusedReduceElementwiseNodeLabels(
    const FusedReduceElementwiseNode& node) {
  os() << "FusedReduceElementwise"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "inputs = " << node.numExternalInputs() << "\\n"
       << "reduction = " << reductionOpToStr(node.reductionOp()) << "\\n"
       << "axes = [";
  for (unsigned i = 0; i < node.axes().size(); i++) {
    os() << node.axes()[i];
    if (i != node.axes().size() - 1) {
      os() << ", ";
    }
  }
  os() << "]\\n"
       << "keepDims = " << (node.keepDims() ? "true" : "false") << "\\n"
       << "steps = [\\n";
  for (unsigned i = 0; i < node.program().size(); i++) {
    const auto& step = node.program()[i];
    if (step.isUnary()) {
      os() << "  " << unaryOpToStr(step.unaryOp()) << "(s" << step.src0 << ")";
    } else {
      os() << "  " << binopToStr(step.binaryOp()) << "(s" << step.src0 << ", s"
           << step.src1 << ")";
    }
    os() << " -> s" << (node.numExternalInputs() + i) << "\\n";
  }
  os() << "]\\n";
}

void GraphvizPrinter::printMatmulNodeLabels(const MatmulNode& node) {
  os() << "MatmulNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "lhsProp = "
       << (node.lhsProp() == MatrixProperty::Transpose ? "T" : "N") << "\\n"
       << "rhsProp = "
       << (node.rhsProp() == MatrixProperty::Transpose ? "T" : "N") << "\\n";
}

void GraphvizPrinter::printConv2dNodeLabels(const Conv2dNode& node) {
  os() << "Conv2dNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "stride = (" << node.sx() << ", " << node.sy() << ")\\n"
       << "padding = (" << node.px() << ", " << node.py() << ")\\n"
       << "dilation = (" << node.dx() << ", " << node.dy() << ")\\n"
       << "groups = " << node.groups() << "\\n";
}

void GraphvizPrinter::printProjectionNodeLabels(const ProjectionNode& node) {
  os() << "Projection"
       << "\\n"
       << "index = " << node.outputIndex() << "\\n"
       << "shape = " << node.shape() << "\\n";
}

void GraphvizPrinter::printBatchNormNodeLabels(const BatchNormNode& node) {
  os() << "BatchNormNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "axes = [";
  for (unsigned i = 0; i < node.axes().size(); i++) {
    os() << node.axes()[i];
    if (i != node.axes().size() - 1) {
      os() << ", ";
    }
  }
  os() << "]\\n"
       << "train = " << (node.train() ? "true" : "false") << "\\n"
       << "momentum = " << node.momentum() << "\\n"
       << "epsilon = " << node.epsilon() << "\\n";
}

void GraphvizPrinter::printPadNodeLabels(const PadNode& node) {
  os() << "PadNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "type = " << padTypeToStr(node.padType()) << "\\n"
       << "padWidths = [";
  for (size_t i = 0; i < node.padWidths().size(); ++i) {
    const auto& [before, after] = node.padWidths()[i];
    os() << "(" << before << ", " << after << ")";
    if (i + 1 < node.padWidths().size()) {
      os() << ", ";
    }
  }
  os() << "]\\n";
}

void GraphvizPrinter::printPool2dNodeLabels(const Pool2dNode& node) {
  os() << "Pool2dNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "window = (" << node.wx() << ", " << node.wy() << ")\\n"
       << "stride = (" << node.sx() << ", " << node.sy() << ")\\n"
       << "padding = (" << node.px() << ", " << node.py() << ")\\n";
}

void GraphvizPrinter::printTileNodeLabels(const TileNode& node) {
  os() << "TileNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "tileDims = " << node.tileDims() << "\\n";
}

void GraphvizPrinter::printWhereNodeLabels(const WhereNode& node) {
  os() << "WhereNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n";
}

void GraphvizPrinter::printValueNodeLabels(const ValueNode& node) {
  os() << "ValueNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n";
}

std::ostream& GraphvizPrinter::printNodes(NodePtr node) {
  if (!nodeNamer_.contains(node)) {
    // roots at bottom
    for (const auto& input : node->inputs()) {
      printNodes(input);
    }
    os() << "    " << nodeNamer_.getName(node);
    os() << "  ["; // node attributes
    os() << "color=\"" << nodeColor_ << "\" ";
    os() << "label=\"";
    printNodeLabels(node);
    const auto iter = nodeToTotTimeMs_->find(node);
    if (iter != nodeToTotTimeMs_->end()) {
      const auto tottime = iter->second;
      os() << "tottime = " << tottime << "ms"
           << "\" ";
      printNodeColor(tottime);
    } else { // just end label string
      os() << "\" ";
    }
    os() << "];" << std::endl;
  }
  return os();
}

std::ostream& GraphvizPrinter::printNodeLabels(NodePtr node) {
  switch (node->type()) {
    case NodeType::Binary:
      printBinaryNodeLabels(node->impl<BinaryNode>());
      break;
    case NodeType::Cast:
      printCastNodeLabels(node->impl<CastNode>());
      break;
    case NodeType::Clip:
      printClipNodeLabels(node->impl<ClipNode>());
      break;
    case NodeType::Concatenate:
      printConcatenateNodeLabels(node->impl<ConcatenateNode>());
      break;
    case NodeType::Custom:
      printCustomNodeLabels(node->impl<CustomNode>());
      break;
    case NodeType::Cumsum:
      printCumsumNodeLabels(node->impl<CumsumNode>());
      break;
    case NodeType::Flip:
      printFlipNodeLabels(node->impl<FlipNode>());
      break;
    case NodeType::Index:
      printIndexNodeLabels(node->impl<IndexNode>());
      break;
    case NodeType::IndexedUpdate:
      printIndexedUpdateNodeLabels(node->impl<IndexedUpdateNode>());
      break;
    case NodeType::Scalar:
      printScalarNodeLabels(node->impl<ScalarNode>());
      break;
    case NodeType::Unary:
      printUnaryNodeLabels(node->impl<UnaryNode>());
      break;
    case NodeType::Reshape:
      printReshapeNodeLabels(node->impl<ReshapeNode>());
      break;
    case NodeType::Reduction:
      printReductionNodeLabels(node->impl<ReductionNode>());
      break;
    case NodeType::Roll:
      printRollNodeLabels(node->impl<RollNode>());
      break;
    case NodeType::Transpose:
      printTransposeNodeLabels(node->impl<TransposeNode>());
      break;
    case NodeType::Triangular:
      printTriangularNodeLabels(node->impl<TriangularNode>());
      break;
    case NodeType::FusedElementwise:
      printFusedElementwiseNodeLabels(node->impl<FusedElementwiseNode>());
      break;
    case NodeType::FusedReduceElementwise:
      printFusedReduceElementwiseNodeLabels(
          node->impl<FusedReduceElementwiseNode>());
      break;
    case NodeType::Matmul:
      printMatmulNodeLabels(node->impl<MatmulNode>());
      break;
    case NodeType::Conv2d:
      printConv2dNodeLabels(node->impl<Conv2dNode>());
      break;
    case NodeType::Projection:
      printProjectionNodeLabels(node->impl<ProjectionNode>());
      break;
    case NodeType::BatchNorm:
      printBatchNormNodeLabels(node->impl<BatchNormNode>());
      break;
    case NodeType::Pad:
      printPadNodeLabels(node->impl<PadNode>());
      break;
    case NodeType::Pool2d:
      printPool2dNodeLabels(node->impl<Pool2dNode>());
      break;
    case NodeType::Tile:
      printTileNodeLabels(node->impl<TileNode>());
      break;
    case NodeType::Where:
      printWhereNodeLabels(node->impl<WhereNode>());
      break;
    case NodeType::Value:
      printValueNodeLabels(node->impl<ValueNode>());
      break;
    case NodeType::MultiValue:
      break; // leaf node with multiple values — no special label
  }
  return os();
}

std::ostream& GraphvizPrinter::printNodeColor(float tottime) {
  os() << " fillcolor=\"";
  printRelativeColor(tottime) << "\""
                              << " style=filled ";
  return os();
}

std::ostream& GraphvizPrinter::printRelativeColor(float tottime) {
  // RGB: (why median -- to avoid all red if all nodes have ~= tottime)
  // max    = (255, 0, 0)     -- all read
  // median = (255, 128, 128) -- in between
  // 0      = (255, 255, 255) -- white
  //
  // i.e.,
  // f(0) = 255, f(median) = 128, f(max) = 0
  // We want to differentiate the hotspot more,
  // so f(x) = a(x - b)^2, and we don't try to fit f(0) = 255 -- we just
  // truncate to 255 if result overflows.
  //
  // --> f(max) = a(max - b)^2 = 0
  // Thus b = max
  //
  // --> f(median) = a(median - max)^2 = 128
  // Thus a = 128/(max - median)^2
  const auto a = 128 / std::pow(maxTotTime_ - medianTotTime_, 2);
  const auto b = maxTotTime_;
  const auto res = std::min(255., a * std::pow(tottime - b, 2));
  const auto normalizedIntensity = static_cast<int>(res);
  std::ios state(nullptr);
  state.copyfmt(os());
  os() << "#" << std::hex << 255 << std::setfill('0') << std::setw(2)
       << normalizedIntensity << std::setfill('0') << std::setw(2)
       << normalizedIntensity;
  os().copyfmt(state);
  return os();
}

std::ostream& GraphvizPrinter::printEdges(NodePtr node) {
  if (!edgesPrinted_.contains(node)) {
    edgesPrinted_.insert(node);
    // root at bottom
    for (const auto& input : node->inputs()) {
      printEdges(input);
    }
    const auto& nodeName = nodeNamer_.getName(node);
    for (const auto& input : node->inputs()) {
      os() << "    " << nodeName << " -> " << nodeNamer_.getName(input);
      os() << " ["; // edge attributes
      os() << "color=\"" << edgeColor_ << "\"";
      os() << " ]" << std::endl;
    }
  }
  return os();
}

GraphvizPrinter::GraphvizPrinter(std::ostream& os) : os_(os), nodeNamer_("n") {
  this->os() << "digraph G {\n"
             << "rankdir=BT\n\n"; // root nodes at bottom
}

GraphvizPrinter::~GraphvizPrinter() {
  this->os() << "}\n";
}

std::ostream& operator<<(
    std::ostream& os,
    const GraphvizPrinter::Color& color) {
  switch (color) {
    case GraphvizPrinter::Color::Black:
      return os << "black";
    case GraphvizPrinter::Color::Green:
      return os << "green";
    case GraphvizPrinter::Color::Red:
      return os << "red";
  }
  throw std::runtime_error(
      "[operator<< for GraphvizPrinter::Color] unknown color");
}

GraphvizPrinter& GraphvizPrinter::printSubgraphImpl(
    NodePtr node,
    const std::string& namePrefix) {
  const auto prefix = "cluster_" + namePrefix; // "cluster_" adds boundary
  os() << "subgraph " << subgraphNameGenerator_.genName(prefix) << " {\n";
  printNodes(node) << std::endl;
  printEdges(node);
  os() << "}\n";
  return *this;
}

GraphvizPrinter& GraphvizPrinter::printSubgraph(
    NodePtr node,
    const std::string& namePrefix) {
  std::unordered_map<NodePtr, float> emptyStats;
  return printSubgraph(node, namePrefix, emptyStats);
}

GraphvizPrinter& GraphvizPrinter::printSubgraph(
    NodePtr node,
    const std::string& namePrefix,
    const std::unordered_map<NodePtr, float>& nodeToTotTimeMs) {
  this->nodeToTotTimeMs_ = &nodeToTotTimeMs;
  if (!nodeToTotTimeMs.empty()) {
    std::vector<float> vals;
    for (const auto& [_, val] : nodeToTotTimeMs) {
      vals.push_back(val);
    }
    std::sort(vals.begin(), vals.end());
    medianTotTime_ = vals[vals.size() / 2];
    maxTotTime_ = vals.back();
  }
  return printSubgraphImpl(node, namePrefix);
}

} // namespace fl
