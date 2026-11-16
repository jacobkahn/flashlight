/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/native/Matmul.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "flashlight/fl/tensor/ops/native/CpuTensorHelpers.h"

extern "C" {
#if FL_USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
}

namespace fl::native {

Tensor matmulF64(
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp) {
  auto lf = lhs.astype(fl::dtype::f64).asContiguousTensor();
  auto rf = rhs.astype(fl::dtype::f64).asContiguousTensor();
  auto lDims = lf.shape().get();
  auto rDims = rf.shape().get();

  // Pad vectors to 2D
  bool lhsVec = lDims.size() <= 1;
  bool rhsVec = rDims.size() <= 1;
  if (lhsVec) lDims.insert(lDims.begin(), 2 - lDims.size(), 1);
  if (rhsVec) rDims.insert(rDims.end(), 2 - rDims.size(), 1);

  // Pad to same ndim for batched case
  int maxNd = std::max(lDims.size(), rDims.size());
  while (static_cast<int>(lDims.size()) < maxNd)
    lDims.insert(lDims.begin(), 1);
  while (static_cast<int>(rDims.size()) < maxNd)
    rDims.insert(rDims.begin(), 1);

  // Apply transpose on inner matrix dims
  if (lhsProp == MatrixProperty::Transpose && !lhsVec)
    std::swap(lDims[maxNd - 2], lDims[maxNd - 1]);
  if (rhsProp == MatrixProperty::Transpose && !rhsVec)
    std::swap(rDims[maxNd - 2], rDims[maxNd - 1]);

  Dim M = lDims[maxNd - 2], K = lDims[maxNd - 1], N = rDims[maxNd - 1];
  if (K != rDims[maxNd - 2]) {
    throw std::invalid_argument(
        "matmulF64: inner dims mismatch " + lhs.shape().toString() +
        " vs " + rhs.shape().toString());
  }

  // Compute batch size
  Dim batch = 1;
  std::vector<Dim> dstDims(maxNd);
  for (int i = 0; i < maxNd - 2; ++i) {
    dstDims[i] = std::max(lDims[i], rDims[i]);
    batch *= dstDims[i];
  }
  dstDims[maxNd - 2] = M;
  dstDims[maxNd - 1] = N;

  auto lhsData = std::vector<double>(lf.elements());
  auto rhsData = std::vector<double>(rf.elements());
  lf.hostSync(lhsData.data());
  rf.hostSync(rhsData.data());

  // Effective strides for transpose
  bool lT = (lhsProp == MatrixProperty::Transpose && !lhsVec);
  bool rT = (rhsProp == MatrixProperty::Transpose && !rhsVec);
  Dim lhsBatchStride = lf.elements() / std::max(batch, static_cast<Dim>(1));
  Dim rhsBatchStride = rf.elements() / std::max(batch, static_cast<Dim>(1));
  // Handle broadcasting: if original had fewer batch elements, stride is 0
  Dim lhsOrigBatch = lf.elements() / (lT ? (K * M) : (M * K));
  Dim rhsOrigBatch = rf.elements() / (rT ? (N * K) : (K * N));
  if (lhsOrigBatch < batch) lhsBatchStride = 0;
  if (rhsOrigBatch < batch) rhsBatchStride = 0;

  std::vector<double> dst(batch * M * N, 0.0);
  for (Dim b = 0; b < batch; ++b) {
    const double* A = lhsData.data() + b * lhsBatchStride;
    const double* B = rhsData.data() + b * rhsBatchStride;
    double* C = dst.data() + b * M * N;

    // cblas_dgemm: C = alpha * op(A) * op(B) + beta * C
    // Row-major layout: lda is the number of columns of the original matrix
    // (before transpose), which is the row stride.
    cblas_dgemm(
        CblasRowMajor,
        lT ? CblasTrans : CblasNoTrans,
        rT ? CblasTrans : CblasNoTrans,
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        1.0,                               // alpha
        A, static_cast<int>(lT ? M : K),   // lda
        B, static_cast<int>(rT ? K : N),   // ldb
        0.0,                               // beta
        C, static_cast<int>(N));           // ldc
  }

  Shape dstShape(dstDims);
  if (lhsVec || rhsVec) {
    Dim elems = batch * M * N;
    // Squeeze vector dims back
    if (lhsVec && rhsVec)
      dstShape = Shape({elems});
    else if (lhsVec)
      dstShape = (maxNd > 2) ? Shape(dstDims) : Shape({N});
    else
      dstShape = (maxNd > 2) ? Shape(dstDims) : Shape({M});
    // Remove unit inner dims for vector results
    if (maxNd <= 2) {
      dstShape = Shape({elems});
    }
  }

  return makeCpuTensor(dstShape, dst.data())
      .copy()
      .astype(fl::dtype::f64);
}

} // namespace fl::native
