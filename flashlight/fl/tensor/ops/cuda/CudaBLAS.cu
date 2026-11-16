/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdexcept>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/backend/cuda/CudaTensor.h"

namespace fl::cuda {

namespace {

// Determine the compute dtype for matmul given two input types.
// Promotes to widest float type; integer types promote to f32.
fl::dtype matmulComputeType(fl::dtype a, fl::dtype b) {
  if (a == fl::dtype::f64 || b == fl::dtype::f64) {
    return fl::dtype::f64;
  }
  if (a == fl::dtype::f16 && b == fl::dtype::f16) {
    return fl::dtype::f16;
  }
  if (a == fl::dtype::bf16 && b == fl::dtype::bf16) {
    return fl::dtype::bf16;
  }
  return fl::dtype::f32;
}

} // namespace

void matmul(Tensor& out,
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp,
    cudaStream_t stream,
    cublasHandle_t handle) {
  bool transA = (lhsProp == MatrixProperty::Transpose);
  bool transB = (rhsProp == MatrixProperty::Transpose);

  // Determine compute type and convert inputs
  fl::dtype computeType = matmulComputeType(lhs.type(), rhs.type());
  auto lhsTyped = (lhs.type() == computeType) ? lhs : lhs.astype(computeType);
  auto rhsTyped = (rhs.type() == computeType) ? rhs : rhs.astype(computeType);

  // Handle 1D inputs: treat as row vector (lhs) or column vector (rhs)
  auto lhsExpanded = lhsTyped;
  auto rhsExpanded = rhsTyped;
  bool squeezeLhsRow = false;
  bool squeezeRhsCol = false;
  if (lhsExpanded.ndim() == 1) {
    // 1D lhs → row vector [1, K]
    lhsExpanded = fl::reshape(lhsExpanded, Shape{1, lhsExpanded.dim(0)});
    squeezeLhsRow = true;
  }
  if (rhsExpanded.ndim() == 1) {
    // 1D rhs → column vector [K, 1]
    rhsExpanded = fl::reshape(rhsExpanded, Shape{rhsExpanded.dim(0), 1});
    squeezeRhsCol = true;
  }

  // For >3D tensors, flatten leading batch dims into one.
  // Save original batch shape for restoring output shape.
  Shape origLhsBatchShape, origRhsBatchShape;
  if (lhsExpanded.ndim() > 3) {
    Dim flatBatch = 1;
    std::vector<Dim> batchDims;
    for (int d = 0; d < lhsExpanded.ndim() - 2; ++d) {
      flatBatch *= lhsExpanded.dim(d);
      batchDims.push_back(lhsExpanded.dim(d));
    }
    origLhsBatchShape = Shape(batchDims);
    Dim r = lhsExpanded.dim(lhsExpanded.ndim() - 2);
    Dim c = lhsExpanded.dim(lhsExpanded.ndim() - 1);
    lhsExpanded = fl::reshape(lhsExpanded, Shape{flatBatch, r, c});
  }
  if (rhsExpanded.ndim() > 3) {
    Dim flatBatch = 1;
    std::vector<Dim> batchDims;
    for (int d = 0; d < rhsExpanded.ndim() - 2; ++d) {
      flatBatch *= rhsExpanded.dim(d);
      batchDims.push_back(rhsExpanded.dim(d));
    }
    origRhsBatchShape = Shape(batchDims);
    Dim r = rhsExpanded.dim(rhsExpanded.ndim() - 2);
    Dim c = rhsExpanded.dim(rhsExpanded.ndim() - 1);
    rhsExpanded = fl::reshape(rhsExpanded, Shape{flatBatch, r, c});
  }

  auto lhsC = lhsExpanded.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? lhsExpanded : lhsExpanded.asContiguousTensor();
  auto rhsC = rhsExpanded.getAdapter<CudaTensor>().isContiguousRowMajor()
      ? rhsExpanded : rhsExpanded.asContiguousTensor();

  // Get shapes
  Dim lhsRows = lhsC.dim(lhsC.ndim() - 2);
  Dim lhsCols = lhsC.dim(lhsC.ndim() - 1);
  Dim rhsRows = rhsC.dim(rhsC.ndim() - 2);
  Dim rhsCols = rhsC.dim(rhsC.ndim() - 1);

  Dim M = transA ? lhsCols : lhsRows;
  Dim K = transA ? lhsRows : lhsCols;
  Dim K2 = transB ? rhsCols : rhsRows;
  Dim N = transB ? rhsRows : rhsCols;

  if (K != K2) {
    throw std::invalid_argument(
        "CudaBackend::matmul: inner dimensions mismatch: "
        + std::to_string(K) + " vs " + std::to_string(K2));
  }

  // Batch dimension
  Dim batchSize = 1;
  bool batched = false;
  if (lhsC.ndim() == 3 || rhsC.ndim() == 3) {
    batched = true;
    Dim lhsBatch = lhsC.ndim() == 3 ? lhsC.dim(0) : 1;
    Dim rhsBatch = rhsC.ndim() == 3 ? rhsC.dim(0) : 1;
    if (lhsBatch != rhsBatch && lhsBatch != 1 && rhsBatch != 1) {
      throw std::invalid_argument(
          "CudaBackend::matmul: batch dimensions mismatch");
    }
    batchSize = std::max(lhsBatch, rhsBatch);
  }

  Shape outShape = batched ? Shape{batchSize, M, N} : Shape{M, N};
  // Reuse pre-assigned out buffer when shape and type match (common 2D/3D case).
  // For 1D-squeeze or >3D-restore cases, shapes differ so we allocate fresh.
  Tensor result = (!out.isEmpty() && out.type() == computeType
                   && out.shape() == outShape)
      ? out
      : toTensor<CudaTensor>(outShape, computeType, nullptr, MemoryLocation::Host);
  auto& outAdapter = result.getAdapter<CudaTensor>();

  // cuBLAS assumes column-major. For row-major C = A * B, we compute
  // C^T = B^T * A^T in cuBLAS column-major, which is equivalent.
  // So we swap the operands and swap the transpose flags.
  cublasOperation_t opA = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  int m = static_cast<int>(N);
  int n = static_cast<int>(M);
  int k = static_cast<int>(K);
  int lda = static_cast<int>(rhsCols);
  int ldb = static_cast<int>(lhsCols);
  int ldc = static_cast<int>(N);

  cublasStatus_t status;

  if (computeType == fl::dtype::f64) {
    // Double-precision matmul via cublasDgemm
    double alpha = 1.0, beta = 0.0;
    auto* lhsPtr = static_cast<const double*>(
        lhsC.getAdapter<CudaTensor>().devicePtr());
    auto* rhsPtr = static_cast<const double*>(
        rhsC.getAdapter<CudaTensor>().devicePtr());
    auto* outPtr = static_cast<double*>(outAdapter.devicePtr());

    if (!batched) {
      status = cublasDgemm(
          handle, opA, opB, m, n, k,
          &alpha, rhsPtr, lda, lhsPtr, ldb, &beta, outPtr, ldc);
    } else {
      long long strideA = (rhsC.ndim() == 3) ? rhsRows * rhsCols : 0;
      long long strideB = (lhsC.ndim() == 3) ? lhsRows * lhsCols : 0;
      long long strideC = M * N;
      status = cublasDgemmStridedBatched(
          handle, opA, opB, m, n, k,
          &alpha, rhsPtr, lda, strideA, lhsPtr, ldb, strideB,
          &beta, outPtr, ldc, strideC, static_cast<int>(batchSize));
    }
  } else if (computeType == fl::dtype::f16) {
    // Half-precision matmul via cublasGemmEx (f32 accumulate, tensor cores)
    float alpha = 1.0f, beta = 0.0f;
    auto* lhsPtr = lhsC.getAdapter<CudaTensor>().devicePtr();
    auto* rhsPtr = rhsC.getAdapter<CudaTensor>().devicePtr();
    auto* outPtr = outAdapter.devicePtr();

    if (!batched) {
      status = cublasGemmEx(
          handle, opA, opB, m, n, k,
          &alpha,
          rhsPtr, CUDA_R_16F, lda,
          lhsPtr, CUDA_R_16F, ldb,
          &beta,
          outPtr, CUDA_R_16F, ldc,
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else {
      long long strideA = (rhsC.ndim() == 3) ? rhsRows * rhsCols : 0;
      long long strideB = (lhsC.ndim() == 3) ? lhsRows * lhsCols : 0;
      long long strideC = M * N;
      status = cublasGemmStridedBatchedEx(
          handle, opA, opB, m, n, k,
          &alpha,
          rhsPtr, CUDA_R_16F, lda, strideA,
          lhsPtr, CUDA_R_16F, ldb, strideB,
          &beta,
          outPtr, CUDA_R_16F, ldc, strideC,
          static_cast<int>(batchSize),
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
  } else if (computeType == fl::dtype::bf16) {
    // BFloat16 matmul via cublasGemmEx (f32 accumulate, tensor cores)
    float alpha = 1.0f, beta = 0.0f;
    auto* lhsPtr = lhsC.getAdapter<CudaTensor>().devicePtr();
    auto* rhsPtr = rhsC.getAdapter<CudaTensor>().devicePtr();
    auto* outPtr = outAdapter.devicePtr();

    if (!batched) {
      status = cublasGemmEx(
          handle, opA, opB, m, n, k,
          &alpha,
          rhsPtr, CUDA_R_16BF, lda,
          lhsPtr, CUDA_R_16BF, ldb,
          &beta,
          outPtr, CUDA_R_16BF, ldc,
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else {
      long long strideA = (rhsC.ndim() == 3) ? rhsRows * rhsCols : 0;
      long long strideB = (lhsC.ndim() == 3) ? lhsRows * lhsCols : 0;
      long long strideC = M * N;
      status = cublasGemmStridedBatchedEx(
          handle, opA, opB, m, n, k,
          &alpha,
          rhsPtr, CUDA_R_16BF, lda, strideA,
          lhsPtr, CUDA_R_16BF, ldb, strideB,
          &beta,
          outPtr, CUDA_R_16BF, ldc, strideC,
          static_cast<int>(batchSize),
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
  } else {
    // Single-precision matmul via cublasSgemm
    float alpha = 1.0f, beta = 0.0f;
    auto* lhsPtr = static_cast<const float*>(
        lhsC.getAdapter<CudaTensor>().devicePtr());
    auto* rhsPtr = static_cast<const float*>(
        rhsC.getAdapter<CudaTensor>().devicePtr());
    auto* outPtr = static_cast<float*>(outAdapter.devicePtr());

    if (!batched) {
      status = cublasSgemm(
          handle, opA, opB, m, n, k,
          &alpha, rhsPtr, lda, lhsPtr, ldb, &beta, outPtr, ldc);
    } else {
      long long strideA = (rhsC.ndim() == 3) ? rhsRows * rhsCols : 0;
      long long strideB = (lhsC.ndim() == 3) ? lhsRows * lhsCols : 0;
      long long strideC = M * N;
      status = cublasSgemmStridedBatched(
          handle, opA, opB, m, n, k,
          &alpha, rhsPtr, lda, strideA, lhsPtr, ldb, strideB,
          &beta, outPtr, ldc, strideC, static_cast<int>(batchSize));
    }
  }

  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "CudaBackend::matmul: cuBLAS gemm failed with status "
        + std::to_string(status));
  }

  // Determine the original batch shape for output restoration.
  // Use whichever operand had the larger batch shape (was >3D before flattening).
  Shape origBatchShape;
  if (origLhsBatchShape.ndim() > 0) {
    origBatchShape = origLhsBatchShape;
  } else if (origRhsBatchShape.ndim() > 0) {
    origBatchShape = origRhsBatchShape;
  }

  // Squeeze dimensions that were added for 1D inputs
  if (squeezeLhsRow && squeezeRhsCol) {
    // vector · vector → scalar
    out = fl::reshape(result, Shape{});
  } else if (squeezeLhsRow) {
    // [1,K]×[K,N] → [N] (remove leading 1)
    if (batched) {
      out = fl::reshape(result, Shape{batchSize, N});
    } else {
      out = fl::reshape(result, Shape{N});
    }
  } else if (squeezeRhsCol) {
    // [M,K]×[K,1] → [M] (remove trailing 1)
    if (batched) {
      out = fl::reshape(result, Shape{batchSize, M});
    } else {
      out = fl::reshape(result, Shape{M});
    }
  } else {
    out = result;
  }

  // Restore original batch dims if we flattened >3D inputs
  if (origBatchShape.ndim() > 0 && !squeezeLhsRow && !squeezeRhsCol) {
    std::vector<Dim> finalDims;
    for (int d = 0; d < origBatchShape.ndim(); ++d) {
      finalDims.push_back(origBatchShape[d]);
    }
    finalDims.push_back(M);
    finalDims.push_back(N);
    out = fl::reshape(out, Shape(finalDims));
  }
}

} // namespace fl::cuda
