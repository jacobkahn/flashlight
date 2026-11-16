/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/opt/backends/cuda/CudnnSubgraphLowering.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cudnn_frontend.h>

#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/BatchNormNode.h"
#include "flashlight/fl/tensor/ir/CastNode.h"
#include "flashlight/fl/tensor/ir/Conv2dNode.h"
#include "flashlight/fl/tensor/ir/CustomNode.h"
#include "flashlight/fl/tensor/ir/MatmulNode.h"
#include "flashlight/fl/tensor/ir/Pool2dNode.h"
#include "flashlight/fl/tensor/ir/TransposeNode.h"
#include "flashlight/fl/tensor/ir/ProjectionNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/opt/FusionUtils.h"
#include "flashlight/fl/tensor/BackendRegistry.h"
#include "flashlight/fl/tensor/ops/cudnn/CudnnAutogradOps.h"
#include "flashlight/fl/tensor/ops/cudnn/CudnnNumerics.h"
#include "flashlight/fl/tensor/ops/cudnn/CudnnPlanCache.h"
#include "flashlight/fl/tensor/ops/cudnn/CudnnUtils.h"

namespace fl {

namespace fe = cudnn_frontend;

namespace {

// ============================== Subgraph Analysis ==============================

// Is this node an anchor that has a dedicated cuDNN graph operation?
bool isAnchorNode(const NodePtr& node) {
  return node->isConv2d() || node->isBatchNorm() || node->isPool2d() ||
         node->isMatmul();
}

// Is this node a cuDNN-expressible pointwise operation?
bool isPointwiseNode(const NodePtr& node) {
  if (node->isBinary()) {
    auto op = node->impl<BinaryNode>().op();
    return op == BinaryOp::Add || op == BinaryOp::Sub ||
           op == BinaryOp::Mul || op == BinaryOp::Div ||
           op == BinaryOp::Max || op == BinaryOp::Min ||
           op == BinaryOp::Pow;
  }
  if (node->isUnary()) {
    auto op = node->impl<UnaryNode>().op();
    return op == UnaryOp::Exp || op == UnaryOp::Log ||
           op == UnaryOp::Sqrt || op == UnaryOp::Tanh ||
           op == UnaryOp::Sigmoid || op == UnaryOp::Erf ||
           op == UnaryOp::Negative || op == UnaryOp::Absolute ||
           op == UnaryOp::Cos || op == UnaryOp::Sin ||
           op == UnaryOp::Ceil || op == UnaryOp::Floor;
  }
  if (node->isScalar()) {
    return true;
  }
  if (node->isCast()) {
    return true;
  }
  return false;
}

// Can this node be included in a cuDNN subgraph?
bool isCudnnLowerable(const NodePtr& node) {
  return isAnchorNode(node) || isPointwiseNode(node) ||
         node->isProjection() || node->isTranspose();
}

// Check if a node has exactly one consumer (internal + external uses).
bool isSingleUse(const NodePtr& node) {
  return !isEvaluated(*node) &&
      (node->uses().size() + node->externalUses().size()) <= 1;
}

// A subgraph is a connected set of nodes that can be lowered to cuDNN.
struct CudnnSubgraph {
  // All nodes in the subgraph, topologically ordered (inputs before consumers)
  std::vector<NodePtr> nodes;
  // Nodes that are at the boundary (inputs to the subgraph from outside)
  std::vector<NodePtr> boundaryInputs;
  // Whether the subgraph contains at least one anchor node
  bool hasAnchor = false;
  // The output node of the subgraph
  NodePtr outputNode;

  // Captured anchor nodes for backward state population
  Conv2dNodePtr capturedConv;
  Pool2dNodePtr capturedPool;
  BatchNormNodePtr capturedBatchNorm;
};

// Expand a subgraph starting from root, walking inputs.
// Includes a node if:
// 1. It's cuDNN-lowerable
// 2. It has single use (so we're not duplicating computation)
//    OR it's a scalar (cheap to duplicate)
// 3. It's not already included
//
// Special handling for ProjectionNode: when a projection's parent is an
// anchor node (e.g. BatchNormNode), both the projection and its parent are
// included regardless of the parent's use count (multi-output nodes
// naturally have multiple uses from their projections).
CudnnSubgraph expandSubgraph(const NodePtr& root) {
  CudnnSubgraph sg;
  sg.outputNode = root;

  std::unordered_set<Node*> inSubgraph;
  std::vector<NodePtr> worklist;

  // Start with the root
  worklist.push_back(root);
  inSubgraph.insert(root.get());

  while (!worklist.empty()) {
    auto current = worklist.back();
    worklist.pop_back();

    if (isAnchorNode(current)) {
      sg.hasAnchor = true;
      // Capture anchor nodes for backward
      if (current->isConv2d()) {
        sg.capturedConv = Node::cast<Conv2dNodePtr>(current);
      } else if (current->isPool2d()) {
        sg.capturedPool = Node::cast<Pool2dNodePtr>(current);
      } else if (current->isBatchNorm()) {
        sg.capturedBatchNorm = Node::cast<BatchNormNodePtr>(current);
      }
    }

    // Walk inputs
    for (const auto& input : current->inputs()) {
      if (inSubgraph.count(input.get())) {
        continue; // Already included
      }

      // Special case: ProjectionNode whose parent is an anchor.
      // Include the projection and its anchor parent, regardless of the
      // anchor's use count (a multi-output anchor has N projection uses).
      if (input->isProjection() && isSingleUse(input)) {
        auto parent = input->impl<ProjectionNode>().parent();
        if (isAnchorNode(parent)) {
          inSubgraph.insert(input.get());
          worklist.push_back(input);
          if (!inSubgraph.count(parent.get())) {
            inSubgraph.insert(parent.get());
            worklist.push_back(parent);
          }
          continue;
        }
      }

      // Don't absorb CastNodes that convert from non-float types (e.g.,
      // bool→f32). cuDNN doesn't support boolean boundary inputs — the bool
      // data is misinterpreted via the INT8 type mapping, producing garbage.
      // Keep the Cast outside so the boundary input is the float result.
      if (input->isCast()) {
        auto srcType = input->impl<CastNode>().input()->dataType();
        if (srcType != fl::dtype::f32 && srcType != fl::dtype::f16 &&
            srcType != fl::dtype::bf16 && srcType != fl::dtype::f64) {
          sg.boundaryInputs.push_back(input);
          continue;
        }
      }

      // Normal case: include if lowerable AND (single-use or scalar)
      if (isCudnnLowerable(input) &&
          (isSingleUse(input) || input->isScalar())) {
        inSubgraph.insert(input.get());
        worklist.push_back(input);
      } else {
        // This is a boundary input
        sg.boundaryInputs.push_back(input);
      }
    }
  }

  // Topological sort: inputs before consumers
  // Simple approach: BFS from boundary inputs through the subgraph
  std::unordered_map<Node*, int> indegree;
  for (auto* n : inSubgraph) {
    indegree[n] = 0;
  }
  // Count in-subgraph predecessors
  for (auto* n : inSubgraph) {
    // Find the actual NodePtr from the set
    NodePtr nodePtr;
    if (n == root.get()) {
      nodePtr = root;
    }
    // We need to iterate through inputs to find in-subgraph deps
  }

  // Simpler: just collect all nodes in the subgraph (order doesn't matter
  // for graph construction since cuDNN frontend handles its own ordering)
  // But we need them for our mapping. Do a reverse BFS from root.
  sg.nodes.clear();
  std::vector<NodePtr> reverseOrder;
  std::unordered_set<Node*> visited;

  // DFS post-order (inputs first, then node)
  std::function<void(const NodePtr&)> dfs = [&](const NodePtr& node) {
    if (visited.count(node.get()) || !inSubgraph.count(node.get())) {
      return;
    }
    visited.insert(node.get());
    for (const auto& input : node->inputs()) {
      if (inSubgraph.count(input.get())) {
        dfs(input);
      }
    }
    sg.nodes.push_back(node);
  };
  dfs(root);

  // Deduplicate boundary inputs
  {
    std::unordered_set<Node*> seen;
    std::vector<NodePtr> unique;
    for (auto& inp : sg.boundaryInputs) {
      if (seen.insert(inp.get()).second) {
        unique.push_back(inp);
      }
    }
    sg.boundaryInputs = std::move(unique);
  }

  return sg;
}

// ============================== cuDNN Graph Building ==============================

fe::PointwiseMode_t binaryOpToPointwise(BinaryOp op) {
  switch (op) {
    case BinaryOp::Add: return fe::PointwiseMode_t::ADD;
    case BinaryOp::Sub: return fe::PointwiseMode_t::SUB;
    case BinaryOp::Mul: return fe::PointwiseMode_t::MUL;
    case BinaryOp::Div: return fe::PointwiseMode_t::DIV;
    case BinaryOp::Max: return fe::PointwiseMode_t::MAX;
    case BinaryOp::Min: return fe::PointwiseMode_t::MIN;
    case BinaryOp::Pow: return fe::PointwiseMode_t::POW;
    default:
      throw std::runtime_error("Unsupported binary op for cuDNN pointwise");
  }
}

fe::PointwiseMode_t unaryOpToPointwise(UnaryOp op) {
  switch (op) {
    case UnaryOp::Exp: return fe::PointwiseMode_t::EXP;
    case UnaryOp::Log: return fe::PointwiseMode_t::LOG;
    case UnaryOp::Sqrt: return fe::PointwiseMode_t::SQRT;
    case UnaryOp::Tanh: return fe::PointwiseMode_t::TANH_FWD;
    case UnaryOp::Sigmoid: return fe::PointwiseMode_t::SIGMOID_FWD;
    case UnaryOp::Erf: return fe::PointwiseMode_t::ERF;
    case UnaryOp::Negative: return fe::PointwiseMode_t::NEG;
    case UnaryOp::Absolute: return fe::PointwiseMode_t::ABS;
    case UnaryOp::Cos: return fe::PointwiseMode_t::COS;
    case UnaryOp::Sin: return fe::PointwiseMode_t::SIN;
    case UnaryOp::Ceil: return fe::PointwiseMode_t::CEIL;
    case UnaryOp::Floor: return fe::PointwiseMode_t::FLOOR;
    default:
      throw std::runtime_error("Unsupported unary op for cuDNN pointwise");
  }
}

// Compute contiguous row-major strides for an N-dimensional shape.
std::vector<int64_t> contiguousStrides(const std::vector<int64_t>& dims) {
  int n = dims.size();
  std::vector<int64_t> strides(n);
  strides[n - 1] = 1;
  for (int i = n - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides;
}

// Pad shape to 4D with trailing 1s. Used for conv2d/batchnorm subgraphs
// where cuDNN expects NCHW format: {N,C,H,W} with trailing spatial dims.
// E.g. {N,C} → {N,C,1,1}.
std::vector<int64_t> shapeTo4D(const Shape& s) {
  std::vector<int64_t> dims(4, 1);
  for (unsigned i = 0; i < s.ndim() && i < 4; ++i) {
    dims[i] = s[i];
  }
  return dims;
}

// Pad shape to 3D with leading 1s. Used for matmul subgraphs where cuDNN
// expects {batch, rows, cols} format. The cuDNN backend validates that
// B.getDimA()[1] == K, so the contraction dim must be at index [ndim-2].
// Leading-1 padding ensures this: {M,K} → {1,M,K}, {B,M,K} → {B,M,K}.
//
// All tensors in a matmul subgraph (operands, output, and pointwise
// neighbors) must use the same leading-1 convention so that broadcast
// dims align correctly in pointwise ops.
std::vector<int64_t> shapeTo3D(const Shape& s) {
  if (s.ndim() >= 3) {
    std::vector<int64_t> dims(s.ndim());
    for (unsigned i = 0; i < s.ndim(); i++) {
      dims[i] = s[i];
    }
    return dims;
  }
  std::vector<int64_t> dims(3, 1);
  int offset = 3 - static_cast<int>(s.ndim());
  for (unsigned i = 0; i < s.ndim(); i++) {
    dims[offset + i] = s[i];
  }
  return dims;
}

struct GraphBuildResult {
  std::shared_ptr<fe::graph::Graph> graph;
  std::vector<int64_t> inputUids; // UIDs for boundary inputs, in order
  int64_t outputUid;              // primary fused output

  // Batchnorm auxiliary outputs (empty if no batchnorm in subgraph).
  // Pairs of (BatchNormNode::Out index, UID).
  std::vector<std::pair<unsigned, int64_t>> bnAuxOutputUids;
};

// Build a cuDNN frontend graph from a CudnnSubgraph.
// Returns nullopt if the subgraph can't be expressed as a cuDNN graph.
std::optional<GraphBuildResult> buildCudnnGraph(const CudnnSubgraph& sg) {
  auto graph = std::make_shared<fe::graph::Graph>();

  // Use f32 compute for f16/bf16
  auto outDt = sg.outputNode->dataType();
  auto ioDt = flTypeToCudnn(outDt);
  auto cDt = (outDt == fl::dtype::f16 || outDt == fl::dtype::bf16)
      ? fe::DataType_t::FLOAT : ioDt;
  graph->set_io_data_type(ioDt)
       .set_intermediate_data_type(ioDt)
       .set_compute_data_type(cDt);

  // Map from Node* to cuDNN tensor in the graph
  std::unordered_map<Node*, std::shared_ptr<fe::graph::Tensor_attributes>>
      tensorMap;

  // Assign UIDs: boundary inputs get UIDs 1..N, output gets N+1
  int64_t nextUid = 1;
  std::vector<int64_t> inputUids;

  // Batchnorm auxiliary output tracking
  std::vector<std::pair<unsigned, int64_t>> bnAuxOutputUids;
  // Map from batchnorm aux output index to cuDNN tensor (for ProjectionNode)
  std::unordered_map<unsigned, std::shared_ptr<fe::graph::Tensor_attributes>>
      bnAuxOutputTensors;

  // Determine the dim convention for this subgraph. Matmul subgraphs use
  // 3D with leading-1 padding (cuDNN matmul backend validates B.dim[1]==K).
  // Conv2d/batchnorm subgraphs use 4D with trailing-1 padding (NCHW).
  bool hasMatmul = false;
  for (const auto& node : sg.nodes) {
    if (node->isMatmul()) { hasMatmul = true; break; }
  }
  auto toGraphDims = [hasMatmul](const Shape& s) -> std::vector<int64_t> {
    return hasMatmul ? shapeTo3D(s) : shapeTo4D(s);
  };

  // Pre-scan matmul nodes to find inputs (boundary or scalar) that need
  // transposed dims/strides. cuDNN graph matmul has no transpose flag —
  // transpose is expressed via strides (SBR). We apply the transpose when
  // creating the input tensor so the matmul handler can use it directly.
  std::unordered_set<Node*> transposedInputs;
  for (const auto& node : sg.nodes) {
    if (!node->isMatmul()) continue;
    auto& mm = node->impl<MatmulNode>();
    if (mm.lhsProp() == MatrixProperty::Transpose) {
      transposedInputs.insert(mm.lhs().get());
    }
    if (mm.rhsProp() == MatrixProperty::Transpose) {
      transposedInputs.insert(mm.rhs().get());
    }
  }

  // Pre-scan for absorbed TransposeNodes at the input boundary.
  // When a TransposeNode is absorbed into the subgraph, its input becomes
  // a boundary input. We record the permutation so the boundary input's
  // cuDNN tensor gets the transposed dims with permuted strides — cuDNN
  // reads the physical buffer using those strides (zero-copy transpose).
  struct TransposeInfo {
    Shape axes;        // normalized permutation
    Shape outputShape; // post-transpose logical shape
  };
  std::unordered_map<Node*, TransposeInfo> transposedBoundaryMap;
  for (const auto& node : sg.nodes) {
    if (!node->isTranspose()) {
      continue;
    }
    auto& tr = node->impl<TransposeNode>();
    int ndim = node->shape().ndim();
    auto axes = tr.axes();
    if (axes.ndim() == 0) {
      // Empty axes = reverse all dims
      std::vector<Dim> reversed(ndim);
      for (int i = 0; i < ndim; ++i) {
        reversed[i] = ndim - 1 - i;
      }
      axes = Shape(reversed);
    }
    for (const auto& bi : sg.boundaryInputs) {
      if (bi.get() == tr.input().get()) {
        transposedBoundaryMap[bi.get()] = {axes, node->shape()};
        break;
      }
    }
  }

  // Transposes are realized by giving a boundary input's (or scalar's) cuDNN
  // tensor permuted dims/strides. That only works when the permuted layout is
  // the sole way the subgraph reads the value: an operand computed inside the
  // subgraph has no such tensor, and a value read twice (e.g. x and x^T) would
  // get one layout for both reads. Leave such subgraphs unlowered.
  std::unordered_set<Node*> boundarySet;
  for (const auto& bi : sg.boundaryInputs) {
    boundarySet.insert(bi.get());
  }
  std::unordered_map<Node*, int> readsInSubgraph;
  for (const auto& node : sg.nodes) {
    for (const auto& input : node->inputs()) {
      readsInSubgraph[input.get()]++;
    }
  }
  for (Node* operand : transposedInputs) {
    const bool hasOwnTensor =
        boundarySet.contains(operand) || operand->isScalar();
    if (!hasOwnTensor || readsInSubgraph[operand] != 1) {
      return std::nullopt;
    }
  }
  for (const auto& node : sg.nodes) {
    if (!node->isTranspose()) {
      continue;
    }
    Node* input = node->impl<TransposeNode>().input().get();
    if (!transposedBoundaryMap.contains(input) ||
        readsInSubgraph[input] != 1) {
      return std::nullopt;
    }
  }

  // Create input tensors for boundary inputs
  for (const auto& input : sg.boundaryInputs) {
    auto dims = toGraphDims(input->shape());
    auto strides = contiguousStrides(dims);

    // For matmul operands that need transpose, swap last two dims/strides
    // so the cuDNN matmul reads the contraction dimension correctly.
    if (transposedInputs.count(input.get())) {
      int n = static_cast<int>(dims.size());
      std::swap(dims[n - 2], dims[n - 1]);
      std::swap(strides[n - 2], strides[n - 1]);
    }

    // For absorbed TransposeNode inputs: use the post-transpose logical
    // dims with permuted physical strides. cuDNN interprets the buffer
    // layout via strides (e.g., NHWC data with NCHW dims + NHWC strides).
    auto trIt = transposedBoundaryMap.find(input.get());
    if (trIt != transposedBoundaryMap.end()) {
      auto& info = trIt->second;
      auto physDims = toGraphDims(input->shape());
      auto physStrides = contiguousStrides(physDims);
      dims = toGraphDims(info.outputShape);
      strides.resize(info.axes.ndim());
      for (int i = 0; i < info.axes.ndim(); ++i) {
        strides[i] = physStrides[info.axes[i]];
      }
    }

    auto uid = nextUid++;
    inputUids.push_back(uid);

    auto t = graph->tensor(fe::graph::Tensor_attributes()
        .set_dim(dims)
        .set_stride(strides)
        .set_data_type(flTypeToCudnn(input->dataType()))
        .set_uid(uid));
    tensorMap[input.get()] = t;
  }

  int64_t virtualUid = 1000; // UIDs for virtual (intermediate) tensors


  // Process nodes in topological order
  for (const auto& node : sg.nodes) {
    if (tensorMap.count(node.get())) {
      continue; // Already created (boundary input)
    }

    auto dims = toGraphDims(node->shape());
    auto strides = contiguousStrides(dims);
    auto dt = flTypeToCudnn(node->dataType());
    bool isOutput = (node.get() == sg.outputNode.get());

    if (node->isScalar()) {
      // ScalarNode represents a constant value broadcast to a shape.
      // Use the node's actual shape (via toGraphDims) so cuDNN sees the
      // correct dims for broadcast/matmul compatibility.
      auto uid = nextUid++;
      inputUids.push_back(uid);
      auto scalarDims = toGraphDims(node->shape());
      auto scalarStrides = contiguousStrides(scalarDims);
      // Apply transpose for matmul operands (same as boundary inputs).
      if (transposedInputs.count(node.get())) {
        int n = static_cast<int>(scalarDims.size());
        std::swap(scalarDims[n - 2], scalarDims[n - 1]);
        std::swap(scalarStrides[n - 2], scalarStrides[n - 1]);
      }
      auto t = graph->tensor(fe::graph::Tensor_attributes()
          .set_dim(scalarDims)
          .set_stride(scalarStrides)
          .set_data_type(dt)
          .set_uid(uid));
      tensorMap[node.get()] = t;
      continue;
    }

    if (node->isProjection()) {
      // ProjectionNode: transparent passthrough to parent's output.
      auto& proj = node->impl<ProjectionNode>();
      auto parent = proj.parent();
      if (proj.outputIndex() == 0) {
        // Primary output — map to the parent's Y tensor
        tensorMap[node.get()] = tensorMap.at(parent.get());
      } else {
        // Auxiliary output (batchnorm saveMean, etc.)
        auto it = bnAuxOutputTensors.find(proj.outputIndex());
        if (it != bnAuxOutputTensors.end()) {
          tensorMap[node.get()] = it->second;
        } else {
          return std::nullopt; // Unknown aux output
        }
      }
      continue;
    }

    // Absorbed TransposeNode: the permutation is already encoded in the
    // boundary input's strides. Map this node to its input's cuDNN tensor.
    if (node->isTranspose()) {
      auto& tr = node->impl<TransposeNode>();
      auto it = tensorMap.find(tr.input().get());
      if (it != tensorMap.end()) {
        tensorMap[node.get()] = it->second;
      } else {
        return std::nullopt;
      }
      continue;
    }

    if (node->isBinary()) {
      auto& b = node->impl<BinaryNode>();
      auto lhs = tensorMap.at(b.lhs().get());
      auto rhs = tensorMap.at(b.rhs().get());
      auto mode = binaryOpToPointwise(b.op());
      auto result = graph->pointwise(lhs, rhs,
          fe::graph::Pointwise_attributes().set_mode(mode));

      if (isOutput) {
        result->set_output(true).set_dim(dims).set_stride(strides)
            .set_data_type(dt).set_uid(nextUid);
      } else {
        result->set_dim(dims).set_stride(strides)
            .set_data_type(dt).set_uid(virtualUid++).set_is_virtual(true);
      }
      tensorMap[node.get()] = result;
      continue;
    }

    if (node->isUnary()) {
      auto& u = node->impl<UnaryNode>();
      auto input = tensorMap.at(u.input().get());
      auto mode = unaryOpToPointwise(u.op());
      auto result = graph->pointwise(input,
          fe::graph::Pointwise_attributes().set_mode(mode));

      if (isOutput) {
        result->set_output(true).set_dim(dims).set_stride(strides)
            .set_data_type(dt).set_uid(nextUid);
      } else {
        result->set_dim(dims).set_stride(strides)
            .set_data_type(dt).set_uid(virtualUid++).set_is_virtual(true);
      }
      tensorMap[node.get()] = result;
      continue;
    }

    if (node->isCast()) {
      auto& cast = node->impl<CastNode>();
      auto input = tensorMap.at(cast.input().get());
      auto result = graph->pointwise(input,
          fe::graph::Pointwise_attributes()
              .set_mode(fe::PointwiseMode_t::IDENTITY));

      if (isOutput) {
        result->set_output(true).set_dim(dims).set_stride(strides)
            .set_data_type(dt).set_uid(nextUid);
      } else {
        result->set_dim(dims).set_stride(strides)
            .set_data_type(dt).set_uid(virtualUid++).set_is_virtual(true);
      }
      tensorMap[node.get()] = result;
      continue;
    }

    if (node->isConv2d()) {
      auto& conv = node->impl<Conv2dNode>();
      auto X = tensorMap.at(conv.input().get());
      auto W = tensorMap.at(conv.weights().get());

      auto convAttrs = fe::graph::Conv_fprop_attributes()
          .set_padding({conv.py(), conv.px()})
          .set_stride({conv.sy(), conv.sx()})
          .set_dilation({conv.dy(), conv.dx()});

      auto Y = graph->conv_fprop(X, W, convAttrs);

      // Handle bias if present. Conv2dNode uses ScalarNode(0) as a sentinel
      // for "no bias", so we skip only zero-valued scalars.
      auto biasNode = conv.bias();
      bool isZeroBias = biasNode && biasNode->isScalar() &&
          biasNode->impl<ScalarNode>().scalar<double>() == 0.0;
      bool hasBias = biasNode && !isZeroBias;
      if (hasBias && tensorMap.count(biasNode.get())) {
        auto B = tensorMap.at(biasNode.get());
        Y = graph->pointwise(Y, B,
            fe::graph::Pointwise_attributes().set_mode(
                fe::PointwiseMode_t::ADD));
      }

      if (isOutput) {
        Y->set_output(true).set_dim(dims).set_stride(strides)
            .set_data_type(dt).set_uid(nextUid);
      } else {
        Y->set_dim(dims).set_stride(strides)
            .set_data_type(dt).set_uid(virtualUid++).set_is_virtual(true);
      }
      tensorMap[node.get()] = Y;
      continue;
    }

    if (node->isMatmul()) {
      auto& mm = node->impl<MatmulNode>();

      // Transpose for boundary inputs was already applied when creating
      // their tensors (see transposedBoundaryInputs pre-scan above).
      // For intermediate (virtual) operands that need transpose, create
      // a new tensor with swapped dims/strides.
      auto buildMatmulOperand = [&](
          const NodePtr& operandNode, MatrixProperty prop)
          -> std::shared_ptr<fe::graph::Tensor_attributes> {
        auto existing = tensorMap.at(operandNode.get());
        if (prop != MatrixProperty::Transpose) {
          return existing;
        }
        // If this is a boundary input or scalar, transpose was already applied.
        if (transposedInputs.count(operandNode.get())) {
          return existing;
        }
        // Intermediate node: create a transposed-view virtual tensor.
        auto opDims = toGraphDims(operandNode->shape());
        auto opStrides = contiguousStrides(opDims);
        int n = static_cast<int>(opDims.size());
        std::swap(opDims[n - 2], opDims[n - 1]);
        std::swap(opStrides[n - 2], opStrides[n - 1]);
        return graph->tensor(fe::graph::Tensor_attributes()
            .set_dim(opDims)
            .set_stride(opStrides)
            .set_data_type(flTypeToCudnn(operandNode->dataType()))
            .set_uid(virtualUid++).set_is_virtual(true));
      };

      auto A = buildMatmulOperand(mm.lhs(), mm.lhsProp());
      auto B = buildMatmulOperand(mm.rhs(), mm.rhsProp());

      auto matmulAttrs = fe::graph::Matmul_attributes()
          .set_compute_data_type(cDt);
      auto C = graph->matmul(A, B, matmulAttrs);

      auto outDims = toGraphDims(node->shape());
      auto outStrides = contiguousStrides(outDims);
      if (isOutput) {
        C->set_output(true).set_dim(outDims).set_stride(outStrides)
            .set_data_type(dt).set_uid(nextUid);
      } else {
        C->set_dim(outDims).set_stride(outStrides)
            .set_data_type(dt).set_uid(virtualUid++).set_is_virtual(true);
      }
      tensorMap[node.get()] = C;
      continue;
    }

    if (node->isBatchNorm()) {
      auto& bn = node->impl<BatchNormNode>();
      auto X = tensorMap.at(bn.inputTensor().get());
      auto scale = tensorMap.at(bn.weight().get());
      auto bias = tensorMap.at(bn.bias().get());

      if (bn.train()) {
        // Training batchnorm: 5 outputs
        auto epsilon = graph->tensor(static_cast<float>(bn.epsilon()));
        auto momentum = graph->tensor(static_cast<float>(bn.momentum()));

        auto bnAttrs = fe::graph::Batchnorm_attributes()
            .set_epsilon(epsilon);

        // Set running stats if provided (non-empty inputs)
        auto rmNode = bn.runningMean();
        auto rvNode = bn.runningVar();
        bool hasRunningStats =
            tensorMap.count(rmNode.get()) && tensorMap.count(rvNode.get());
        if (hasRunningStats) {
          bnAttrs.set_previous_running_stats(
              tensorMap.at(rmNode.get()),
              tensorMap.at(rvNode.get()),
              momentum);
        }

        auto [Y, mean, invVar, nextRM, nextRV] =
            graph->batchnorm(X, scale, bias, bnAttrs);

        // Y flows into pointwise ops (virtual unless it's the final output)
        if (isOutput) {
          Y->set_output(true).set_dim(dims).set_stride(strides)
              .set_data_type(dt).set_uid(nextUid);
        } else {
          Y->set_dim(dims).set_stride(strides)
              .set_data_type(dt).set_uid(virtualUid++).set_is_virtual(true);
        }
        tensorMap[node.get()] = Y;

        // Auxiliary outputs are always concrete (non-virtual)
        const auto& outDescs = bn.outputDescs();

        auto makeAuxOutput = [&](
            std::shared_ptr<fe::graph::Tensor_attributes>& tensor,
            unsigned outIdx) {
          auto auxDims = toGraphDims(outDescs[outIdx].shape);
          auto auxStrides = contiguousStrides(auxDims);
          auto auxDt = flTypeToCudnn(outDescs[outIdx].type);
          auto auxUid = nextUid++;
          tensor->set_output(true)
              .set_dim(auxDims)
              .set_stride(auxStrides)
              .set_data_type(auxDt)
              .set_uid(auxUid);
          bnAuxOutputUids.push_back({outIdx, auxUid});
          bnAuxOutputTensors[outIdx] = tensor;
        };

        makeAuxOutput(mean, BatchNormNode::SaveMean);
        makeAuxOutput(invVar, BatchNormNode::SaveVar);
        if (hasRunningStats) {
          makeAuxOutput(nextRM, BatchNormNode::NewRunningMean);
          makeAuxOutput(nextRV, BatchNormNode::NewRunningVar);
        }
      } else {
        // Inference batchnorm: single output
        auto rmTensor = tensorMap.at(bn.runningMean().get());
        auto rvTensor = tensorMap.at(bn.runningVar().get());

        auto Y = graph->batchnorm_inference(
            X, rmTensor, rvTensor, scale, bias,
            fe::graph::Batchnorm_inference_attributes());

        if (isOutput) {
          Y->set_output(true).set_dim(dims).set_stride(strides)
              .set_data_type(dt).set_uid(nextUid);
        } else {
          Y->set_dim(dims).set_stride(strides)
              .set_data_type(dt).set_uid(virtualUid++).set_is_virtual(true);
        }
        tensorMap[node.get()] = Y;
      }
      continue;
    }

    if (node->isPool2d()) {
      auto& pool = node->impl<Pool2dNode>();
      auto X = tensorMap.at(pool.input().get());

      // Map PoolingMode -> cuDNN ResampleMode_t + PaddingMode_t
      fe::ResampleMode_t resampleMode;
      fe::PaddingMode_t padMode;
      switch (pool.mode()) {
        case PoolingMode::MAX:
          resampleMode = fe::ResampleMode_t::MAXPOOL;
          padMode = fe::PaddingMode_t::NEG_INF_PAD;
          break;
        case PoolingMode::AVG_INCLUDE_PADDING:
          resampleMode = fe::ResampleMode_t::AVGPOOL_INCLUDE_PADDING;
          padMode = fe::PaddingMode_t::ZERO_PAD;
          break;
        case PoolingMode::AVG_EXCLUDE_PADDING:
          resampleMode = fe::ResampleMode_t::AVGPOOL_EXCLUDE_PADDING;
          padMode = fe::PaddingMode_t::ZERO_PAD;
          break;
      }

      // Use inference mode: the fused graph computes pool2d forward only.
      // TODO(training max pooling): set_is_inference(false) for max pooling
      // in training mode. This produces an Index output tensor containing
      // argmax positions per window. Two options for backward:
      //   (a) Store Index as a side-output of the fused CustomNode via
      //       multi-output + ProjectionNode. Pool2dNode::setCachedState()
      //       would hold the Index tensor for pool2dBackward to consume.
      //   (b) Recompute: let pool2dBackward run the unfused legacy path
      //       (cudnnPoolingBackward) which recomputes indices internally.
      //       This avoids storing the Index tensor at the cost of redundant
      //       forward work. cuDNN's legacy pooling backward already does this.
      // Current choice: (b) — pool2dBackward runs unfused via the evaluator.
      // The fusion benefit is forward pool2d + adjacent pointwise only.
      auto [Y, Index] = graph->resample(X,
          fe::graph::Resample_attributes()
              .set_is_inference(true)
              .set_resampling_mode(resampleMode)
              .set_padding_mode(padMode)
              .set_window({static_cast<int64_t>(pool.wy()),
                           static_cast<int64_t>(pool.wx())})
              .set_stride({static_cast<int64_t>(pool.sy()),
                           static_cast<int64_t>(pool.sx())})
              .set_pre_padding({static_cast<int64_t>(pool.py()),
                                static_cast<int64_t>(pool.px())})
              .set_post_padding({static_cast<int64_t>(pool.py()),
                                 static_cast<int64_t>(pool.px())}));

      if (isOutput) {
        Y->set_output(true).set_dim(dims).set_stride(strides)
            .set_data_type(dt).set_uid(nextUid);
      } else {
        Y->set_dim(dims).set_stride(strides)
            .set_data_type(dt).set_uid(virtualUid++).set_is_virtual(true);
      }
      tensorMap[node.get()] = Y;
      continue;
    }

    // Unknown node type in subgraph — cannot lower.
    return std::nullopt;
  }

  auto outputUid = nextUid;

  return GraphBuildResult{
      graph, inputUids, outputUid, bnAuxOutputUids};
}

// ============================== CustomNode Building ==============================

// Build a single-output fused CustomNode (conv2d, matmul, pool2d paths).
NodePtr buildFusedCustomNodeSingleOutput(
    const CudnnSubgraph& sg,
    const GraphBuildResult& graphResult) {
  // Collect boundary inputs + any scalar nodes that ended up as inputs
  std::vector<NodePtr> customInputs;

  // Map from boundary input Node* to index in customInputs
  std::unordered_map<Node*, size_t> inputIndexMap;

  for (const auto& input : sg.boundaryInputs) {
    inputIndexMap[input.get()] = customInputs.size();
    customInputs.push_back(input);
  }

  // Also add scalar nodes that are in the subgraph but became graph inputs
  for (const auto& node : sg.nodes) {
    if (node->isScalar() && !inputIndexMap.count(node.get())) {
      inputIndexMap[node.get()] = customInputs.size();
      customInputs.push_back(node);
    }
  }

  auto compiledGraph = graphResult.graph;
  auto inputUids = graphResult.inputUids;
  auto outputUid = graphResult.outputUid;
  auto capturedConv = sg.capturedConv;
  auto capturedPool = sg.capturedPool;

  // Capture output shape/dtype at graph-build time (the compiled graph
  // doesn't expose get_output_tensors()).
  auto outShape = sg.outputNode->shape();
  auto outDtype = sg.outputNode->dataType();

  auto evalFunc = [compiledGraph, inputUids, outputUid,
                   capturedConv, capturedPool, outShape, outDtype](
                      const std::vector<const Tensor*>& ins) -> Tensor {
    auto handle = getCudnnHandle();
    const auto& cudnnStream = getCudnnStream();

    // Build variant pack mapping UIDs to device pointers
    std::unordered_map<int64_t, void*> variantPack;
    std::vector<DevicePtr> devicePtrs;
    devicePtrs.reserve(ins.size());

    // Sync all inputs to cuDNN stream
    std::vector<Tensor> syncTensors;
    for (size_t i = 0; i < ins.size(); ++i) {
      syncTensors.push_back(*ins[i]);
    }

    for (size_t i = 0; i < ins.size(); ++i) {
      auto contiguous = ins[i]->asContiguousTensor();
      devicePtrs.emplace_back(contiguous);
      if (i < inputUids.size()) {
        variantPack[inputUids[i]] = devicePtrs.back().get();
      }
    }

    // Create bare CudaTensor (not JitTensor) — the result must be a raw
    // backend tensor so the evaluator can use it directly without triggering
    // nested JIT evaluation.
    auto output = detail::makeTensor(outShape, outDtype, TensorBackendType::Cuda);
    DevicePtr outPtr(output);
    variantPack[outputUid] = outPtr.get();

    relativeSync(cudnnStream, syncTensors);
    relativeSync(cudnnStream, {output});

    // Allocate workspace
    auto wsSize = compiledGraph->get_workspace_size();
    void* wsPtr = nullptr;
    Tensor wsTensor;
    DevicePtr* wsDev = nullptr;
    if (wsSize > 0) {
      wsTensor = detail::makeTensor(
          Shape{static_cast<long long>(wsSize)}, fl::dtype::b8,
          TensorBackendType::Cuda);
      wsDev = new DevicePtr(wsTensor);
      wsPtr = wsDev->get();
      relativeSync(cudnnStream, {wsTensor});
    }

    auto status = compiledGraph->execute(handle, variantPack, wsPtr);
    delete wsDev;

    if (!status.is_good()) {
      throw std::runtime_error(
          "CudnnSubgraphLowering: execute failed: " +
          std::string(status.get_message()));
    }

    relativeSync({output}, cudnnStream);

    // Populate cached state for backward
    if (capturedConv && capturedConv->wantState()) {
      capturedConv->setCachedState(nullptr);
      capturedConv->setResult(Tensor(output));
    }
    if (capturedPool) {
      capturedPool->setCachedState(nullptr);
    }

    return output;
  };

  return CustomNode::create(
      "cudnnSubgraph",
      std::move(customInputs),
      sg.outputNode->shape(),
      sg.outputNode->dataType(),
      std::move(evalFunc),
      sg.outputNode->stream());
}

// Build a multi-output fused CustomNode for training batchnorm subgraphs.
NodePtr buildFusedCustomNodeMultiOutput(
    const CudnnSubgraph& sg,
    const GraphBuildResult& graphResult) {
  // Collect boundary inputs + any scalar nodes that ended up as inputs
  std::vector<NodePtr> customInputs;
  std::unordered_map<Node*, size_t> inputIndexMap;

  for (const auto& input : sg.boundaryInputs) {
    inputIndexMap[input.get()] = customInputs.size();
    customInputs.push_back(input);
  }

  for (const auto& node : sg.nodes) {
    if (node->isScalar() && !inputIndexMap.count(node.get())) {
      inputIndexMap[node.get()] = customInputs.size();
      customInputs.push_back(node);
    }
  }

  auto compiledGraph = graphResult.graph;
  auto inputUids = graphResult.inputUids;
  auto outputUid = graphResult.outputUid;
  auto bnAuxOutputUids = graphResult.bnAuxOutputUids;
  auto capturedBatchNorm = sg.capturedBatchNorm;

  // Build output descriptions: output[0] = primary fused result,
  // output[1..4] = batchnorm auxiliary outputs (matching BN output indices).
  const auto& bnOutDescs = capturedBatchNorm->outputDescs();
  std::vector<OutputDesc> outputDescs;
  // Output 0: primary fused result (same shape/dtype as subgraph output)
  outputDescs.push_back({sg.outputNode->shape(), sg.outputNode->dataType()});
  // Outputs 1-4: batchnorm auxiliaries
  for (unsigned i = 1; i < BatchNormNode::kNumOutputs; i++) {
    outputDescs.push_back(bnOutDescs[i]);
  }

  auto outShape = sg.outputNode->shape();
  auto outDtype = sg.outputNode->dataType();

  auto evalFunc = [compiledGraph, inputUids, outputUid, bnAuxOutputUids,
                   capturedBatchNorm, outputDescs, outShape,
                   outDtype](
                      const std::vector<const Tensor*>& ins)
      -> std::vector<Tensor> {
    auto handle = getCudnnHandle();
    const auto& cudnnStream = getCudnnStream();

    std::unordered_map<int64_t, void*> variantPack;
    std::vector<DevicePtr> devicePtrs;
    devicePtrs.reserve(ins.size() + 1 + bnAuxOutputUids.size());

    // Sync all inputs to cuDNN stream
    std::vector<Tensor> syncTensors;
    for (size_t i = 0; i < ins.size(); ++i) {
      syncTensors.push_back(*ins[i]);
    }

    for (size_t i = 0; i < ins.size(); ++i) {
      auto contiguous = ins[i]->asContiguousTensor();
      devicePtrs.emplace_back(contiguous);
      if (i < inputUids.size()) {
        variantPack[inputUids[i]] = devicePtrs.back().get();
      }
    }

    // Allocate primary output (bare CudaTensor, not JitTensor)
    auto primaryOutput = detail::makeTensor(
        outShape, outDtype, TensorBackendType::Cuda);
    devicePtrs.emplace_back(primaryOutput);
    variantPack[outputUid] = devicePtrs.back().get();
    syncTensors.push_back(primaryOutput);

    // Allocate auxiliary outputs
    std::vector<Tensor> auxOutputs;
    for (const auto& [outIdx, uid] : bnAuxOutputUids) {
      auto auxTensor = detail::makeTensor(
          outputDescs[outIdx].shape, outputDescs[outIdx].type,
          TensorBackendType::Cuda);
      devicePtrs.emplace_back(auxTensor);
      variantPack[uid] = devicePtrs.back().get();
      syncTensors.push_back(auxTensor);
      auxOutputs.push_back(std::move(auxTensor));
    }

    relativeSync(cudnnStream, syncTensors);

    // Allocate workspace
    auto wsSize = compiledGraph->get_workspace_size();
    void* wsPtr = nullptr;
    Tensor wsTensor;
    DevicePtr* wsDev = nullptr;
    if (wsSize > 0) {
      wsTensor = detail::makeTensor(
          Shape{static_cast<long long>(wsSize)}, fl::dtype::b8,
          TensorBackendType::Cuda);
      wsDev = new DevicePtr(wsTensor);
      wsPtr = wsDev->get();
      relativeSync(cudnnStream, {wsTensor});
    }

    auto status = compiledGraph->execute(handle, variantPack, wsPtr);
    delete wsDev;

    if (!status.is_good()) {
      throw std::runtime_error(
          "CudnnSubgraphLowering: multi-output execute failed: " +
          std::string(status.get_message()));
    }

    // Sync outputs back from cuDNN stream
    std::vector<Tensor> allOutputsForSync = {primaryOutput};
    for (auto& aux : auxOutputs) {
      allOutputsForSync.push_back(aux);
    }
    relativeSync(allOutputsForSync, cudnnStream);

    // Populate cached state for backward
    if (capturedBatchNorm) {
      capturedBatchNorm->cachedState_ = nullptr;
    }

    // Build result vector: [primary, saveMean, saveVar, newRM, newRV]
    std::vector<Tensor> results;
    results.push_back(std::move(primaryOutput));
    // Fill in aux outputs matching indices 1..4
    // bnAuxOutputUids is ordered by BN output index
    unsigned auxIdx = 0;
    for (unsigned i = 1; i < BatchNormNode::kNumOutputs; i++) {
      if (auxIdx < bnAuxOutputUids.size() &&
          bnAuxOutputUids[auxIdx].first == i) {
        results.push_back(std::move(auxOutputs[auxIdx]));
        auxIdx++;
      } else {
        // No running stats — push empty tensor
        results.push_back(Tensor());
      }
    }

    return results;
  };

  return CustomNode::createMultiOutput(
      "cudnnBatchNormSubgraph",
      std::move(customInputs),
      std::move(outputDescs),
      std::move(evalFunc),
      sg.outputNode->stream());
}

NodePtr buildFusedCustomNode(
    const CudnnSubgraph& sg,
    const GraphBuildResult& graphResult) {
  if (sg.capturedBatchNorm && sg.capturedBatchNorm->train()) {
    return buildFusedCustomNodeMultiOutput(sg, graphResult);
  }
  return buildFusedCustomNodeSingleOutput(sg, graphResult);
}

} // namespace

NodePtr CudnnSubgraphLowering::rewrite(NodePtr node) {
  // Only try to fuse if this node is cuDNN-lowerable
  if (!isCudnnLowerable(node)) {
    return node;
  }

  // Expand subgraph from this node
  auto sg = expandSubgraph(node);

  // Only fuse if the subgraph contains at least one anchor AND at least
  // one additional pointwise operation (otherwise no benefit over unfused).
  if (!sg.hasAnchor) {
    return node; // Pure pointwise — leave for NVRTC fusion
  }
  if (sg.nodes.size() <= 1) {
    return node; // Single anchor with no pointwise neighbors — no benefit
  }

  // Try to build a cuDNN graph
  auto graphResult = buildCudnnGraph(sg);
  if (!graphResult.has_value()) {
    return node; // Can't build cuDNN graph, fall back
  }

  // Validate and compile the graph
  auto handle = getCudnnHandle();
  auto& graph = graphResult->graph;

  if (auto s = graph->validate(); !s.is_good()) {
    return node;
  }
  if (auto s = graph->build_operation_graph(handle); !s.is_good()) {
    return node;
  }
  if (auto s = graph->create_execution_plans({fe::HeurMode_t::A}); !s.is_good()) {
    return node;
  }
  // f32 means f32 unless TF32 is allowed (see restrictToStrictMath). If no
  // plan remains, the subgraph simply stays unfused.
  bool hasWideOperand =
      detail::cudnn::isWideFloat(sg.outputNode->dataType());
  for (const auto& input : sg.boundaryInputs) {
    hasWideOperand =
        hasWideOperand || detail::cudnn::isWideFloat(input->dataType());
  }
  detail::cudnn::restrictToStrictMath(*graph, hasWideOperand);
  if (auto s = graph->check_support(handle); !s.is_good()) {
    return node;
  }
  if (auto s = graph->build_plans(handle); !s.is_good()) {
    return node;
  }

  // Build the fused CustomNode
  auto fused = buildFusedCustomNode(sg, graphResult.value());

  // Infrastructure for future ExternalUse-aware fusion.
  // Currently a no-op since isSingleUse() prevents absorption of nodes
  // with ExternalUses. Ready for follow-up that relaxes absorption criteria.
  auto extNodes = findExternallyReferencedNodes(sg.nodes);
  (void)extNodes;

  if (sg.capturedBatchNorm && sg.capturedBatchNorm->train()) {
    // Multi-output: the fused CustomNode has outputs [0..4] matching
    // the BN's output indices. Redirect the subgraph root's users to
    // a projection of output 0 (the primary fused result), and redirect
    // the BN's projections (1-4) to point at the fused node.
    auto fusedResultProj = ProjectionNode::create(
        fused, 0, sg.outputNode->shape(), sg.outputNode->dataType(),
        sg.outputNode->stream());
    node->replaceAllUsesWith(fusedResultProj);
    sg.capturedBatchNorm->replaceAllUsesWith(fused);
  } else {
    node->replaceAllUsesWith(fused);
  }

  return fused;
}

} // namespace fl
