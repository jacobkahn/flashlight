/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/opt/backends/cuda/CublasLtBackwardEpilogueFusion.h"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include <cublasLt.h>
#include <cuda_runtime.h>

#include "flashlight/fl/runtime/CUDAUtils.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/backend/cuda/CudaBackend.h"
#include "flashlight/fl/tensor/backend/cuda/CudaTensor.h"
#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/CastNode.h"
#include "flashlight/fl/tensor/ir/CustomNode.h"
#include "flashlight/fl/tensor/ir/MatmulNode.h"
#include "flashlight/fl/tensor/ir/ProjectionNode.h"
#include "flashlight/fl/tensor/ir/ReductionNode.h"
#include "flashlight/fl/tensor/ir/ReshapeNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/ops/cuda/CudaNumerics.h"

namespace fl {

namespace {

// ==================== Helpers ====================

bool isSingleUse(const NodePtr& node) {
  return !isEvaluated(*node) &&
      (node->uses().size() + node->externalUses().size()) <= 1;
}

// Match the relu mask pattern: CastNode(f32, BinaryNode(Gt, z, Scalar(0)))
// Returns the pre-relu activation 'z' if matched, nullptr otherwise.
NodePtr matchReluMask(const NodePtr& node) {
  if (!node->isCast()) return nullptr;
  auto& cast = node->impl<CastNode>();
  if (cast.targetDtype() != dtype::f32) return nullptr;

  auto gtNode = cast.input();
  if (!gtNode->isBinary()) return nullptr;
  auto& gt = gtNode->impl<BinaryNode>();
  if (gt.op() != BinaryOp::Gt) return nullptr;

  // Check RHS is Scalar(0)
  auto rhs = gt.rhs();
  if (!rhs->isScalar()) return nullptr;
  if (rhs->impl<ScalarNode>().scalar<double>() != 0.0) return nullptr;

  return gt.lhs(); // 'z' — the pre-relu activation
}

// Walk through single-use ReshapeNodes from 'node' to find a MatmulNode.
// Returns {matmulNode, reshapeShapes} where reshapeShapes is the chain of
// output shapes from innermost (closest to matmul) to outermost.
struct PeelResult {
  NodePtr matmulNode;
  std::vector<NodePtr> reshapeChain; // ReshapeNodes from matmul outward
};

PeelResult peelReshapesToMatmul(const NodePtr& node, bool requireSingleUse = true) {
  PeelResult result;
  auto current = node;
  while (current->isReshape()) {
    if (requireSingleUse && !isSingleUse(current)) {
      return {}; // multi-use reshape — can't fuse
    }
    result.reshapeChain.push_back(current);
    current = current->impl<ReshapeNode>().input();
  }
  // The cuBLASLt calls here are plain 2-D GEMMs (no batch count/stride), so
  // only 2-D matmuls can be fused; higher ranks would compute one batch.
  if (current->isMatmul() && current->shape().ndim() == 2 &&
      current->impl<MatmulNode>().lhs()->shape().ndim() == 2 &&
      current->impl<MatmulNode>().rhs()->shape().ndim() == 2) {
    if (!requireSingleUse || isSingleUse(current)) {
      result.matmulNode = current;
    }
  }
  return result;
}

// Verify all nodes in the fused relu pattern are single-use.
bool allSingleUse(
    const NodePtr& castNode,
    const NodePtr& gtNode,
    const PeelResult& peeled) {
  if (!isSingleUse(castNode)) return false;
  if (!isSingleUse(gtNode)) return false;
  if (!peeled.matmulNode) return false;
  for (auto& r : peeled.reshapeChain) {
    if (!isSingleUse(r)) return false;
  }
  return true;
}

// Match dgelu derivative pattern by looking for the Erf landmark.
// The gelu gradFunc emits: cdf = 0.5*(1+erf(x*M_SQRT1_2)), pdf = x*exp(...)*coeff
// and returns gradOutput * (cdf + pdf). The Erf node is unique to dgelu.
//
// Walk the expression tree from 'node' following single-use elementwise nodes
// (Unary, Binary, Scalar). If we find Erf(Mul(z, Scalar(≈0.7071))), return z.
// Returns nullptr if the pattern isn't found.
NodePtr matchDgeluInput(const NodePtr& node) {
  // Walk the single-use elementwise tree under 'node'. The derivative
  // cdf + x * pdf contains both Erf(Mul(z, ≈0.7071)) (from cdf) and an Exp
  // (from pdf); forward gelu has the Erf but no Exp, so requiring both keeps
  // e.g. matmul(a) * gelu(b) from being rewritten to matmul(a) * gelu'(b).
  NodePtr erfInput;
  bool sawExp = false;
  std::vector<NodePtr> stack;
  stack.push_back(node);

  while (!stack.empty()) {
    auto current = stack.back();
    stack.pop_back();

    if (current->isUnary()) {
      auto& unary = current->impl<UnaryNode>();
      if (unary.op() == UnaryOp::Erf) {
        // Erf found — check its input is Mul(z, Scalar(≈0.7071))
        auto mulInput = unary.input();
        if (!erfInput && mulInput->isBinary()) {
          auto& mulNode = mulInput->impl<BinaryNode>();
          if (mulNode.op() == BinaryOp::Mul) {
            // Check for Scalar(≈M_SQRT1_2) on either side
            for (int side = 0; side < 2; ++side) {
              auto scalarSide = (side == 0) ? mulNode.rhs() : mulNode.lhs();
              auto zSide = (side == 0) ? mulNode.lhs() : mulNode.rhs();
              if (scalarSide->isScalar()) {
                double val = scalarSide->impl<ScalarNode>().scalar<double>();
                if (std::abs(val - M_SQRT1_2) < 1e-4) {
                  erfInput = zSide; // 'z' — the pre-gelu activation
                  break;
                }
              }
            }
          }
        }
        continue;
      }
      if (unary.op() == UnaryOp::Exp) {
        sawExp = true;
      }
      // Continue walking down if single-use
      if (isSingleUse(current)) {
        stack.push_back(unary.input());
      }
      continue;
    }

    if (current->isBinary()) {
      auto& bin = current->impl<BinaryNode>();
      // Walk both sides (but each intermediate must be single-use)
      auto lhs = bin.lhs();
      auto rhs = bin.rhs();
      if (isSingleUse(current) || current == node) {
        if (lhs->isUnary() || lhs->isBinary()) {
          stack.push_back(lhs);
        }
        if (rhs->isUnary() || rhs->isBinary()) {
          stack.push_back(rhs);
        }
      }
      continue;
    }

    // Scalar nodes are terminals — skip
  }

  return sawExp ? erfInput : nullptr;
}

// Collect all single-use intermediate nodes in the dgelu chain between
// 'root' and 'z'. These need to be marked visited after fusion.
void collectDgeluIntermediates(
    const NodePtr& root,
    const NodePtr& z,
    std::vector<NodePtr>& intermediates) {
  std::vector<NodePtr> stack;
  stack.push_back(root);

  while (!stack.empty()) {
    auto current = stack.back();
    stack.pop_back();

    if (current == z) continue;

    if (current->isUnary()) {
      intermediates.push_back(current);
      auto& unary = current->impl<UnaryNode>();
      stack.push_back(unary.input());
      continue;
    }

    if (current->isBinary()) {
      intermediates.push_back(current);
      auto& bin = current->impl<BinaryNode>();
      auto lhs = bin.lhs();
      auto rhs = bin.rhs();
      if (lhs != z && (lhs->isUnary() || lhs->isBinary() || lhs->isScalar())) {
        stack.push_back(lhs);
      }
      if (rhs != z && (rhs->isUnary() || rhs->isBinary() || rhs->isScalar())) {
        stack.push_back(rhs);
      }
      continue;
    }

    if (current->isScalar()) {
      intermediates.push_back(current);
    }
  }
}

// ==================== Bitmask packing kernel ====================

__global__ void packReluBitmask(
    const float* __restrict__ z,
    uint8_t* __restrict__ bitmask,
    int64_t rows,
    int64_t cols,
    int64_t ldBits) {
  // z is row-major [rows, cols]. cuBLASLt expects column-major bitmask.
  // bit_idx = row + col * ldBits (column-major bit layout)
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows * cols;
  if (idx >= total) return;

  // Row-major index: row = idx / cols (m-index), col = idx % cols (n-index)
  int64_t row = idx / cols;
  int64_t col = idx % cols;

  // Column-major bit index: the output is col-major [N, M], so
  // element (m, n) maps to col-major position (n, m) → bit = n + m * ldBits
  int64_t bitIdx = col + row * ldBits;
  int bit = (z[idx] > 0.0f) ? 1 : 0;

  // Atomic OR to set the bit (multiple threads target different bits
  // within the same byte, but atomicOr handles this safely)
  atomicOr(
      reinterpret_cast<unsigned int*>(&bitmask[bitIdx / 8 & ~3u]),
      static_cast<unsigned int>(bit) << (bitIdx % 32));
}

// ==================== Epilogue type enum ====================

enum class EpilogueType {
  Drelu,
  Dgelu,
  Bgradb,
  DreluBgrad,
  DgeluBgrad,
};

// ==================== Generalized cuBLASLt matmul with epilogue ====================

// Perform matmul with the specified epilogue.
// Returns 1 tensor for DRELU/DGELU, 2 tensors for *_BGRAD/BGRADB.
// For DRELU: z is pre-relu activation (packs into bitmask).
// For DGELU: z is pre-gelu activation (passed as full-precision AUX).
// For BGRADB: z is unused (can be empty tensor).
std::vector<Tensor> cublasLtMatmulWithEpilogue(
    EpilogueType epilogueType,
    const Tensor& A,
    const Tensor& B,
    const Tensor& z,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp,
    const Shape& matmulOutShape) {
  auto& backend = CudaBackend::getInstance();
  auto stream = backend.cudaStream();
  auto ltHandle = backend.cublasLtHandle();

  bool transA = (lhsProp == MatrixProperty::Transpose);
  bool transB = (rhsProp == MatrixProperty::Transpose);

  // Make inputs contiguous
  auto Ac = A.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? A : A.asContiguousTensor();
  auto Bc = B.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? B : B.asContiguousTensor();

  // Matmul dimensions (row-major semantics):
  //   A is [ARows, ACols], B is [BRows, BCols]
  //   With transA: logical A is [ACols, ARows] → M=ACols, K=ARows
  //   Without:     logical A is [ARows, ACols] → M=ARows, K=ACols
  Dim ARows = Ac.dim(Ac.ndim() - 2);
  Dim ACols = Ac.dim(Ac.ndim() - 1);
  Dim BRows = Bc.dim(Bc.ndim() - 2);
  Dim BCols = Bc.dim(Bc.ndim() - 1);

  Dim M = transA ? ACols : ARows;
  Dim K = transA ? ARows : ACols;
  Dim N = transB ? BRows : BCols;

  // Row-major C[M,N] = A * B
  // cuBLAS column-major trick: C^T[N,M] = B^T[N,K] * A^T[K,M]
  // So for cuBLAS:  m=N, n=M, k=K
  //   cuBLAS "A" ← our B (row-major [BRows,BCols] = col-major storage [BCols,BRows])
  //   cuBLAS "B" ← our A (row-major [ARows,ACols] = col-major storage [ACols,ARows])
  //   cuBLAS "C" ← output (row-major [M,N] = col-major storage [N,M])
  int m = static_cast<int>(N);
  int n = static_cast<int>(M);
  int k = static_cast<int>(K);

  // Row-major [R,C] → column-major: ld = C (number of columns = stride between rows)
  int lda = static_cast<int>(BCols);
  int ldb = static_cast<int>(ACols);
  int ldc = static_cast<int>(N);

  // Transpose flags are swapped due to row→col conversion:
  //   Row-major B with rhsProp → cuBLAS "A" with opA
  //   Row-major A with lhsProp → cuBLAS "B" with opB
  cublasOperation_t opA = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transA ? CUBLAS_OP_T : CUBLAS_OP_N;

  // Allocate output
  auto result = toTensor<CudaTensor>(
      matmulOutShape, dtype::f32, nullptr, MemoryLocation::Host);

  // Set up cublasLt matmul descriptor
  cublasLtMatmulDesc_t matmulDesc;
  // Strict f32 unless TF32 is allowed (fl::cuda::detail::allowTF32()).
  cublasLtMatmulDescCreate(
      &matmulDesc,
      fl::cuda::detail::allowTF32() ? CUBLAS_COMPUTE_32F_FAST_TF32
                                    : CUBLAS_COMPUTE_32F,
      CUDA_R_32F);
  cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
  cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

  // Epilogue-specific setup
  cublasLtEpilogue_t epilogue;
  Tensor bitmaskBuf;    // for DRELU variants
  Tensor biasBuf;       // for BGRAD variants
  int64_t ldBits = 0;

  // Column-major dimensions of the output (for bitmask/AUX layout)
  int64_t colMajRows = static_cast<int64_t>(N);
  int64_t colMajCols = static_cast<int64_t>(M);

  bool needsAux = (epilogueType == EpilogueType::Drelu ||
                   epilogueType == EpilogueType::DreluBgrad);
  bool needsDgeluAux = (epilogueType == EpilogueType::Dgelu ||
                        epilogueType == EpilogueType::DgeluBgrad);
  bool needsBias = (epilogueType == EpilogueType::Bgradb ||
                    epilogueType == EpilogueType::DreluBgrad ||
                    epilogueType == EpilogueType::DgeluBgrad);

  // Set epilogue type
  switch (epilogueType) {
    case EpilogueType::Drelu:
      epilogue = CUBLASLT_EPILOGUE_DRELU;
      break;
    case EpilogueType::Dgelu:
      epilogue = CUBLASLT_EPILOGUE_DGELU;
      break;
    case EpilogueType::Bgradb:
      epilogue = CUBLASLT_EPILOGUE_BGRADB;
      break;
    case EpilogueType::DreluBgrad:
      epilogue = CUBLASLT_EPILOGUE_DRELU_BGRAD;
      break;
    case EpilogueType::DgeluBgrad:
      epilogue = CUBLASLT_EPILOGUE_DGELU_BGRAD;
      break;
  }

  cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));

  // DRELU bitmask AUX setup
  if (needsAux) {
    auto zc = z.getAdapter<CudaTensor>().isContiguousRowMajor()
        ? z : z.asContiguousTensor();

    ldBits = ((colMajRows + 127) / 128) * 128;
    int64_t bitmaskBytes = (ldBits * colMajCols + 7) / 8;

    bitmaskBuf = toTensor<CudaTensor>(
        Shape{bitmaskBytes}, dtype::u8, nullptr, MemoryLocation::Host);
    FL_CUDA_CHECK(cudaMemsetAsync(
        bitmaskBuf.getAdapter<CudaTensor>().devicePtr(), 0, bitmaskBytes, stream));

    // Launch packing kernel
    int64_t totalElements = M * N;
    int threads = 256;
    int blocks = static_cast<int>((totalElements + threads - 1) / threads);
    packReluBitmask<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(zc.getAdapter<CudaTensor>().devicePtr()),
        static_cast<uint8_t*>(bitmaskBuf.getAdapter<CudaTensor>().devicePtr()),
        M, N, ldBits);
    FL_CUDA_CHECK(cudaGetLastError());

    void* auxPtr = bitmaskBuf.getAdapter<CudaTensor>().devicePtr();
    cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
        &auxPtr, sizeof(auxPtr));
    cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
        &ldBits, sizeof(ldBits));
  }

  // DGELU full-precision AUX setup
  if (needsDgeluAux) {
    auto zc = z.getAdapter<CudaTensor>().isContiguousRowMajor()
        ? z : z.asContiguousTensor();

    // For DGELU, AUX is the full-precision pre-gelu activation z.
    // cuBLASLt reads z in column-major layout. Our row-major [M, N] data
    // is column-major [N, M], so AUX_LD = N (= colMajRows).
    int64_t auxLd = colMajRows;

    void* auxPtr = zc.getAdapter<CudaTensor>().devicePtr();
    cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
        &auxPtr, sizeof(auxPtr));
    cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
        &auxLd, sizeof(auxLd));
  }

  // Bias output setup for BGRADB / *_BGRAD
  if (needsBias) {
    // Bias gradient: sum over rows (axis 0 of row-major [M, N]) → shape [1, N].
    // In column-major [N, M], this is a sum over columns → [N, 1], which is
    // what cuBLASLt produces at BIAS_POINTER.
    biasBuf = toTensor<CudaTensor>(
        Shape{1, N}, dtype::f32, nullptr, MemoryLocation::Host);

    void* biasPtr = biasBuf.getAdapter<CudaTensor>().devicePtr();
    cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
        &biasPtr, sizeof(biasPtr));
  }

  // Create matrix layouts.
  // cublasLtMatrixLayoutCreate(type, rows, cols, ld) describes the physical
  // storage layout. Row-major [R,C] → col-major storage: rows=C, cols=R, ld=C.
  cublasLtMatrixLayout_t layoutA, layoutB, layoutC;

  // cuBLAS A = our B: stored row-major [BRows, BCols] → col-major [BCols, BRows]
  cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_32F,
      static_cast<int>(BCols), static_cast<int>(BRows), lda);
  // cuBLAS B = our A: stored row-major [ARows, ACols] → col-major [ACols, ARows]
  cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_32F,
      static_cast<int>(ACols), static_cast<int>(ARows), ldb);
  // cuBLAS C = output: stored row-major [M, N] → col-major [N, M]
  cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, m, n, ldc);

  // Query heuristic
  cublasLtMatmulPreference_t preference;
  cublasLtMatmulPreferenceCreate(&preference);
  size_t workspaceSize = 4 * 1024 * 1024; // 4MB workspace
  cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspaceSize, sizeof(workspaceSize));

  cublasLtMatmulHeuristicResult_t heuristicResult;
  int returnedAlgoCount = 0;
  auto heurStatus = cublasLtMatmulAlgoGetHeuristic(
      ltHandle, matmulDesc, layoutA, layoutB, layoutC, layoutC,
      preference, 1, &heuristicResult, &returnedAlgoCount);

  if (heurStatus != CUBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) {
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatmulDescDestroy(matmulDesc);
    throw std::runtime_error(
        "CublasLtBackwardEpilogueFusion: no algorithm found for epilogue"
        " (m=" + std::to_string(m) + " n=" + std::to_string(n) +
        " k=" + std::to_string(k) + ")");
  }

  // Allocate workspace
  void* workspace = nullptr;
  size_t actualWorkspace = heuristicResult.workspaceSize;
  if (actualWorkspace > 0) {
    FL_CUDA_CHECK(cudaMallocAsync(&workspace, actualWorkspace, stream));
  }

  float alpha = 1.0f, beta = 0.0f;
  auto* APtr = Bc.getAdapter<CudaTensor>().devicePtr(); // cuBLAS A = our B
  auto* BPtr = Ac.getAdapter<CudaTensor>().devicePtr(); // cuBLAS B = our A
  auto* CPtr = result.getAdapter<CudaTensor>().devicePtr();

  auto status = cublasLtMatmul(
      ltHandle, matmulDesc,
      &alpha, APtr, layoutA, BPtr, layoutB,
      &beta, CPtr, layoutC, CPtr, layoutC,
      &heuristicResult.algo, workspace, actualWorkspace, stream);

  // Cleanup
  if (workspace) {
    FL_CUDA_CHECK(cudaFreeAsync(workspace, stream));
  }
  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatrixLayoutDestroy(layoutC);
  cublasLtMatrixLayoutDestroy(layoutB);
  cublasLtMatrixLayoutDestroy(layoutA);
  cublasLtMatmulDescDestroy(matmulDesc);

  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "CublasLtBackwardEpilogueFusion: cublasLtMatmul failed with status "
        + std::to_string(status));
  }

  std::vector<Tensor> outputs;
  outputs.push_back(std::move(result));
  if (needsBias) {
    outputs.push_back(std::move(biasBuf));
  }
  return outputs;
}

// ==================== Pattern matching helpers ====================

// Try to match DRELU pattern from a Mul node.
// Returns {z, matmulSide, castNode, gtNode, scalarNode, peeled} or nullptrs.
struct DreluMatch {
  NodePtr z;
  NodePtr castNode;
  NodePtr gtNode;
  NodePtr scalarNode;
  PeelResult peeled;
};

DreluMatch tryMatchDrelu(const NodePtr& mulNode) {
  auto& bin = mulNode->impl<BinaryNode>();

  for (int order = 0; order < 2; ++order) {
    NodePtr maskSide = (order == 0) ? bin.lhs() : bin.rhs();
    NodePtr matmulSide = (order == 0) ? bin.rhs() : bin.lhs();

    auto z = matchReluMask(maskSide);
    if (!z) continue;

    auto peeled = peelReshapesToMatmul(matmulSide);
    if (!peeled.matmulNode) continue;

    auto gtNode = maskSide->impl<CastNode>().input();
    if (!allSingleUse(maskSide, gtNode, peeled)) continue;

    auto scalarNode = gtNode->impl<BinaryNode>().rhs();

    if (peeled.matmulNode->dataType() != dtype::f32) continue;
    if (z->dataType() != dtype::f32) continue;

    return {z, maskSide, gtNode, scalarNode, std::move(peeled)};
  }

  return {};
}

// Try to match DGELU pattern from a Mul node.
// Returns {z, dgeluSide, peeled, dgeluIntermediates} or nullptrs.
struct DgeluMatch {
  NodePtr z;
  NodePtr dgeluSide;
  PeelResult peeled;
  std::vector<NodePtr> intermediates;
};

DgeluMatch tryMatchDgelu(const NodePtr& mulNode) {
  auto& bin = mulNode->impl<BinaryNode>();

  for (int order = 0; order < 2; ++order) {
    NodePtr dgeluSide = (order == 0) ? bin.lhs() : bin.rhs();
    NodePtr matmulSide = (order == 0) ? bin.rhs() : bin.lhs();

    auto z = matchDgeluInput(dgeluSide);
    if (!z) continue;

    auto peeled = peelReshapesToMatmul(matmulSide);
    if (!peeled.matmulNode) continue;

    if (peeled.matmulNode->dataType() != dtype::f32) continue;
    if (z->dataType() != dtype::f32) continue;
    // cuBLASLt requires the GELU aux leading dimension (the row length of
    // the row-major output) to be a multiple of 8; reject here rather than
    // fail at eval time.
    if (peeled.matmulNode->shape().dim(1) % 8 != 0) continue;

    std::vector<NodePtr> intermediates;
    collectDgeluIntermediates(dgeluSide, z, intermediates);

    return {z, dgeluSide, std::move(peeled), std::move(intermediates)};
  }

  return {};
}

} // namespace

// ==================== Pass implementation ====================

// Check if the sole user of 'node' is a Sum({0}, keepDims=true) reduction.
// If so, we should defer Mul-level fusion so the BGRAD variant can match.
static bool hasBgradReductionUser(const NodePtr& node) {
  if (node->uses().size() != 1 || !node->externalUses().empty()) return false;
  auto& use = *node->uses().front();
  Node& user = use.user();
  if (!user.isReduction()) return false;
  auto& red = user.impl<ReductionNode>();
  return red.op() == ReductionOp::Sum &&
         red.axes() == std::vector<int>{0} &&
         red.keepDims();
}

NodePtr CublasLtBackwardEpilogueFusion::rewrite(NodePtr node) {
  // === Pattern A: Mul node — try DRELU, then DGELU ===
  if (node->isBinary()) {
    auto& bin = node->impl<BinaryNode>();
    if (bin.op() != BinaryOp::Mul) return node;

    // If the Mul's only user is a Sum({0}) reduction, defer to the BGRAD
    // pattern (matched when we visit the Reduction node).
    if (hasBgradReductionUser(node)) return node;

    // --- Try DRELU ---
    {
      auto match = tryMatchDrelu(node);
      if (match.z) {
        auto& mm = match.peeled.matmulNode->impl<MatmulNode>();
        auto mmLhs = mm.lhs();
        auto mmRhs = mm.rhs();
        auto lhsProp = mm.lhsProp();
        auto rhsProp = mm.rhsProp();
        auto matmulOutShape = match.peeled.matmulNode->shape();

        std::vector<Shape> reshapeShapes;
        for (auto& r : match.peeled.reshapeChain) {
          reshapeShapes.push_back(r->shape());
        }

        auto fusedNode = CustomNode::create(
            "CublasLtDrelu",
            {mmLhs, mmRhs, match.z},
            node->shape(),
            dtype::f32,
            [matmulOutShape, lhsProp, rhsProp, reshapeShapes](
                const std::vector<const Tensor*>& inputs) -> Tensor {
              auto results = cublasLtMatmulWithEpilogue(
                  EpilogueType::Drelu,
                  *inputs[0], *inputs[1], *inputs[2],
                  lhsProp, rhsProp, matmulOutShape);
              auto result = std::move(results[0]);
              for (const auto& shape : reshapeShapes) {
                result = fl::reshape(result, shape);
              }
              return result;
            },
            node->stream());

        markVisited(match.castNode.get());
        markVisited(match.gtNode.get());
        markVisited(match.scalarNode.get());
        markVisited(match.peeled.matmulNode.get());
        for (auto& r : match.peeled.reshapeChain) {
          markVisited(r.get());
        }

        return fusedNode;
      }
    }

    // --- Try DGELU ---
    {
      auto match = tryMatchDgelu(node);
      if (match.z) {
        auto& mm = match.peeled.matmulNode->impl<MatmulNode>();
        auto mmLhs = mm.lhs();
        auto mmRhs = mm.rhs();
        auto lhsProp = mm.lhsProp();
        auto rhsProp = mm.rhsProp();
        auto matmulOutShape = match.peeled.matmulNode->shape();

        std::vector<Shape> reshapeShapes;
        for (auto& r : match.peeled.reshapeChain) {
          reshapeShapes.push_back(r->shape());
        }

        auto fusedNode = CustomNode::create(
            "CublasLtDgelu",
            {mmLhs, mmRhs, match.z},
            node->shape(),
            dtype::f32,
            [matmulOutShape, lhsProp, rhsProp, reshapeShapes](
                const std::vector<const Tensor*>& inputs) -> Tensor {
              auto results = cublasLtMatmulWithEpilogue(
                  EpilogueType::Dgelu,
                  *inputs[0], *inputs[1], *inputs[2],
                  lhsProp, rhsProp, matmulOutShape);
              auto result = std::move(results[0]);
              for (const auto& shape : reshapeShapes) {
                result = fl::reshape(result, shape);
              }
              return result;
            },
            node->stream());

        for (auto& intermediate : match.intermediates) {
          markVisited(intermediate.get());
        }
        markVisited(match.peeled.matmulNode.get());
        for (auto& r : match.peeled.reshapeChain) {
          markVisited(r.get());
        }

        return fusedNode;
      }
    }

    return node;
  }

  // === Patterns B/C/D: Reduction(Sum, {0}) — try DRELU_BGRAD, DGELU_BGRAD, BGRADB ===
  if (node->isReduction()) {
    auto& red = node->impl<ReductionNode>();
    if (red.op() != ReductionOp::Sum) return node;
    if (red.axes() != std::vector<int>{0}) return node;
    if (!red.keepDims()) return node;

    auto redInput = red.input();

    // The bias-gradient epilogues sum the 2-D matmul output over its rows, so
    // the reduced tensor must be exactly that output: 2-D, with no reshapes
    // in between (which would change what the reduction covers).
    if (redInput->shape().ndim() != 2) return node;

    // --- Try DRELU_BGRAD: Sum(Mul(relu_mask, Matmul(A, B)), {0}) ---
    if (redInput->isBinary() && redInput->impl<BinaryNode>().op() == BinaryOp::Mul) {
      // Try DRELU_BGRAD first
      auto dreluMatch = tryMatchDrelu(redInput);
      if (dreluMatch.z && dreluMatch.peeled.reshapeChain.empty()) {
        auto& mm = dreluMatch.peeled.matmulNode->impl<MatmulNode>();
        auto mmLhs = mm.lhs();
        auto mmRhs = mm.rhs();
        auto lhsProp = mm.lhsProp();
        auto rhsProp = mm.rhsProp();
        auto matmulOutShape = dreluMatch.peeled.matmulNode->shape();

        std::vector<Shape> reshapeShapes;
        for (auto& r : dreluMatch.peeled.reshapeChain) {
          reshapeShapes.push_back(r->shape());
        }

        auto mulShape = redInput->shape();
        auto biasShape = node->shape();

        // Multi-output: [drelu_result, bias_grad]
        auto fusedNode = CustomNode::createMultiOutput(
            "CublasLtDreluBgrad",
            {mmLhs, mmRhs, dreluMatch.z},
            {{mulShape, dtype::f32}, {biasShape, dtype::f32}},
            [matmulOutShape, lhsProp, rhsProp, reshapeShapes](
                const std::vector<const Tensor*>& inputs) -> std::vector<Tensor> {
              auto results = cublasLtMatmulWithEpilogue(
                  EpilogueType::DreluBgrad,
                  *inputs[0], *inputs[1], *inputs[2],
                  lhsProp, rhsProp, matmulOutShape);
              // results[0] is the drelu result, apply reshapes
              for (const auto& shape : reshapeShapes) {
                results[0] = fl::reshape(results[0], shape);
              }
              return results;
            },
            node->stream());

        auto proj0 = ProjectionNode::create(
            fusedNode, 0, mulShape, dtype::f32, node->stream());
        auto proj1 = ProjectionNode::create(
            fusedNode, 1, biasShape, dtype::f32, node->stream());

        // Redirect Mul's other uses to proj0
        redInput->replaceAllUsesWith(proj0);

        // Mark visited
        markVisited(redInput.get());
        markVisited(dreluMatch.castNode.get());
        markVisited(dreluMatch.gtNode.get());
        markVisited(dreluMatch.scalarNode.get());
        markVisited(dreluMatch.peeled.matmulNode.get());
        for (auto& r : dreluMatch.peeled.reshapeChain) {
          markVisited(r.get());
        }

        return proj1;
      }

      // Try DGELU_BGRAD
      auto dgeluMatch = tryMatchDgelu(redInput);
      if (dgeluMatch.z && dgeluMatch.peeled.reshapeChain.empty()) {
        auto& mm = dgeluMatch.peeled.matmulNode->impl<MatmulNode>();
        auto mmLhs = mm.lhs();
        auto mmRhs = mm.rhs();
        auto lhsProp = mm.lhsProp();
        auto rhsProp = mm.rhsProp();
        auto matmulOutShape = dgeluMatch.peeled.matmulNode->shape();

        std::vector<Shape> reshapeShapes;
        for (auto& r : dgeluMatch.peeled.reshapeChain) {
          reshapeShapes.push_back(r->shape());
        }

        auto mulShape = redInput->shape();
        auto biasShape = node->shape();

        auto fusedNode = CustomNode::createMultiOutput(
            "CublasLtDgeluBgrad",
            {mmLhs, mmRhs, dgeluMatch.z},
            {{mulShape, dtype::f32}, {biasShape, dtype::f32}},
            [matmulOutShape, lhsProp, rhsProp, reshapeShapes](
                const std::vector<const Tensor*>& inputs) -> std::vector<Tensor> {
              auto results = cublasLtMatmulWithEpilogue(
                  EpilogueType::DgeluBgrad,
                  *inputs[0], *inputs[1], *inputs[2],
                  lhsProp, rhsProp, matmulOutShape);
              for (const auto& shape : reshapeShapes) {
                results[0] = fl::reshape(results[0], shape);
              }
              return results;
            },
            node->stream());

        auto proj0 = ProjectionNode::create(
            fusedNode, 0, mulShape, dtype::f32, node->stream());
        auto proj1 = ProjectionNode::create(
            fusedNode, 1, biasShape, dtype::f32, node->stream());

        redInput->replaceAllUsesWith(proj0);

        markVisited(redInput.get());
        for (auto& intermediate : dgeluMatch.intermediates) {
          markVisited(intermediate.get());
        }
        markVisited(dgeluMatch.peeled.matmulNode.get());
        for (auto& r : dgeluMatch.peeled.reshapeChain) {
          markVisited(r.get());
        }

        return proj1;
      }
    }

    // --- BGRADB: disabled ---
    // CUBLASLT_EPILOGUE_BGRADB reduces the GEMM's B operand over k (a vector
    // as long as D's column count), not Sum(D, {0}); the pattern below
    // computed the wrong values and overran its {1, N} output when M > N.
    // Kept for reference until it is re-derived against cublasLt.h.
    constexpr bool kEnableBgradb = false;
    if (kEnableBgradb) {
      auto peeled = peelReshapesToMatmul(redInput, /*requireSingleUse=*/false);
      if (peeled.matmulNode && peeled.matmulNode->isMatmul()) {
        // BGRADB only makes sense if the matmul has OTHER uses (besides the sum).
        // If it only feeds the sum, a regular matmul + reduction is fine.
        auto matmulUseCount = peeled.matmulNode->uses().size() +
                              peeled.matmulNode->externalUses().size();
        if (matmulUseCount <= 1) return node; // single use — don't fuse

        if (peeled.matmulNode->dataType() != dtype::f32) return node;

        // Check reshape chain is single-use
        for (auto& r : peeled.reshapeChain) {
          if (!isSingleUse(r)) return node;
        }

        auto& mm = peeled.matmulNode->impl<MatmulNode>();
        auto mmLhs = mm.lhs();
        auto mmRhs = mm.rhs();
        auto lhsProp = mm.lhsProp();
        auto rhsProp = mm.rhsProp();
        auto matmulOutShape = peeled.matmulNode->shape();

        std::vector<Shape> reshapeShapes;
        for (auto& r : peeled.reshapeChain) {
          reshapeShapes.push_back(r->shape());
        }

        auto matmulResultShape = peeled.reshapeChain.empty()
            ? matmulOutShape : peeled.reshapeChain.back()->shape();
        auto biasShape = node->shape();

        // Multi-output: [matmul_result, bias_grad]
        // We need a dummy z tensor for the API; won't be used.
        auto fusedNode = CustomNode::createMultiOutput(
            "CublasLtBgradb",
            {mmLhs, mmRhs},
            {{matmulResultShape, dtype::f32}, {biasShape, dtype::f32}},
            [matmulOutShape, lhsProp, rhsProp, reshapeShapes](
                const std::vector<const Tensor*>& inputs) -> std::vector<Tensor> {
              // Pass empty tensor for z — BGRADB doesn't use AUX
              Tensor dummyZ;
              auto results = cublasLtMatmulWithEpilogue(
                  EpilogueType::Bgradb,
                  *inputs[0], *inputs[1], dummyZ,
                  lhsProp, rhsProp, matmulOutShape);
              for (const auto& shape : reshapeShapes) {
                results[0] = fl::reshape(results[0], shape);
              }
              return results;
            },
            node->stream());

        auto proj0 = ProjectionNode::create(
            fusedNode, 0, matmulResultShape, dtype::f32, node->stream());
        auto proj1 = ProjectionNode::create(
            fusedNode, 1, biasShape, dtype::f32, node->stream());

        // Redirect matmul (or the outermost reshape) to proj0
        if (peeled.reshapeChain.empty()) {
          peeled.matmulNode->replaceAllUsesWith(proj0);
        } else {
          // The outermost reshape in the chain is the one the reduction reads.
          // But the matmul's other uses may go through the matmul directly.
          // We redirect the matmul itself to proj0.
          peeled.matmulNode->replaceAllUsesWith(proj0);
        }

        for (auto& r : peeled.reshapeChain) {
          markVisited(r.get());
        }

        return proj1;
      }
    }

    return node;
  }

  return node;
}

} // namespace fl
