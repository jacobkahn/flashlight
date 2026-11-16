/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"

#include <af/blas.h>

#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#include "flashlight/fl/tensor/backend/af/Utils.h"

namespace fl {

void ArrayFireBackend::matmul(
    Tensor& out,
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp,
    std::shared_ptr<const Stream> /* stream */) {
  unsigned numDims = std::max(lhs.ndim(), rhs.ndim());
  if ((lhs.ndim() == 1 || rhs.ndim() == 1) && numDims > 1) {
    numDims -= 1;
  }

  af::array lhsArray = toArray(lhs);
  af::array rhsArray = toArray(rhs);

  if (lhs.ndim() == 1 && rhs.ndim() == 1) {
    // Dot product: 1D arrays have no reversal. Use transpose trick:
    // (1, k) x (k, 1) = (1, 1) --> reshape to (1)
    auto arr = af::matmul(
        lhsArray,
        rhsArray,
        detail::flToAfMatrixProperty(MatrixProperty::Transpose),
        detail::flToAfMatrixProperty(MatrixProperty::None));
    out = toTensor<ArrayFireTensor>(std::move(arr), /* numDims = */ 1);
    return;
  }

  // Row-major to AF col-major: swap operands and their properties.
  // FL: op(L){..., M, K} @ op(R){..., K, N} --> AF: af::matmul(R_af, L_af, rhsProp, lhsProp)
  //
  // With full dimension reversal, FL batch-first {b..., M, K} becomes
  // AF {K, M, ...b_reversed}. AF matmul treats dims 0,1 as the matrix
  // and dims 2+ as batch, so the matrix dims {K, M} are correctly at
  // AF positions 0,1 and batch dims at 2+. The swap handles the row-to-col
  // major matrix transpose.
  //
  // Sparse tensors bypass FL-->AF dimension reversal (their AF dims match FL
  // dims). AF also requires sparse operands as the first matmul argument.

  // Sparse LHS: can't use standard swap (would place sparse second).
  // AF sparse matmul supports Transpose on the sparse (first) operand but
  // requires AF_MAT_NONE on the dense (second) operand, so we apply rhsProp
  // by explicitly transposing the dense array.
  //
  // Derivation (2D): dense arrays are reversed in AF (R_af = R_fl^T), sparse
  // are not (L_af = L_fl). We compute C_af^T = op_L(L_af) @ op_R(R_af^T),
  // then C_af = transpose(C_af^T). AF reads C_af reversed to recover C_fl.
  // op_R(R_af^T) = transpose(R_af) if rhsProp=None, R_af if rhsProp=Transpose.
  if (lhs.isSparse()) {
    af::array denseEff = rhsArray;
    if (rhsProp == MatrixProperty::None) {
      denseEff = af::transpose(rhsArray);
    }
    auto arr = af::matmul(
        lhsArray,
        denseEff,
        detail::flToAfMatrixProperty(lhsProp),
        detail::flToAfMatrixProperty(MatrixProperty::None));
    arr = af::transpose(arr);
    out = toTensor<ArrayFireTensor>(std::move(arr), numDims);
    return;
  }

  // Sparse RHS: after swap, sparse is naturally first AF arg. Toggle the
  // sparse operand's property to compensate for the missing reversal.
  // AF sparse matmul requires AF_MAT_NONE on the dense (second) operand,
  // so apply lhsProp to the dense array explicitly when rhs is sparse.
  auto effectiveRhsProp = rhsProp;
  if (rhs.isSparse()) {
    effectiveRhsProp = (rhsProp == MatrixProperty::Transpose)
        ? MatrixProperty::None
        : MatrixProperty::Transpose;
    if (lhsProp == MatrixProperty::Transpose) {
      lhsArray = af::transpose(lhsArray);
    }
  }

  // Handle 1D arguments for the swapped matmul
  if (rhs.ndim() == 1) {
    rhsArray = af::moddims(rhsArray, {1, rhs.dim(0)});
  }
  if (lhs.ndim() == 1) {
    lhsArray = af::moddims(lhsArray, {lhs.dim(0), 1});
  }

  auto arr = af::matmul(
      rhsArray,
      lhsArray,
      detail::flToAfMatrixProperty(effectiveRhsProp),
      detail::flToAfMatrixProperty(
          rhs.isSparse() ? MatrixProperty::None : lhsProp));

  // Squeeze dimensions for 1D inputs
  if (lhs.ndim() == 1 && rhs.ndim() == 2) {
    arr = af::moddims(arr, arr.dims(0));
  } else if (rhs.ndim() == 1 && lhs.ndim() == 2) {
    arr = af::moddims(arr, arr.dims(1));
  }

  out = toTensor<ArrayFireTensor>(std::move(arr), numDims);
}
} // namespace fl
