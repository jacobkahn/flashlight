/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/ext/integrations/sputnik/SputnikInterface.h"

#include <af/sparse.h>
#include <sputnik/sputnik.h>

#include <iostream>
#include <stdexcept>

#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/common/backend/cuda/CudaUtils.h"

namespace fl {
namespace cuda {

af::array getSparseRowIndices(const af::array& rowOffsets) {
  af::array _a, rowIndices;
  af::sort(_a, rowIndices, -1 * af::diff1(rowOffsets));
  return rowIndices;
}

void denseToSparseCSRt(
    af::array& values,
    af::array& rowIndices,
    af::array& rowOffsets,
    af::array& columnIndices,
    const af::array& input,
    const dim_t dim) {
  if (input.numdims() != 2) {
    throw std::invalid_argument("denseToSparseCSRt accepts only 2D tensors");
  }

  // Nonzero values and indices
  auto _input = dim == 0 ? input : input.T(); // transpose if dim == 1
  values = af::lookup(af::flat(_input), af::where(_input));
  auto zeroMask = (input != 0);

  // Row offsets/indices
  rowOffsets = af::join(
      0,
      af::constant(0, 1, af::dtype::s32),
      af::accum(af::flat(af::sum(zeroMask, dim))).as(af::dtype::s32));
  rowIndices = fl::cuda::getSparseRowIndices(rowOffsets).as(af::dtype::s32);

  // Column indices
  auto nzc = (af::flat(zeroMask) *
              af::flat(
                  af::range(af::dim4(
                      input.dims(dim), zeroMask.dims(dim == 0 ? 1 : 0))) +
                  1))
                 .as(af::dtype::s32);
  columnIndices = af::lookup(nzc, af::where(nzc)) - 1;
}

void sparseToDenseCSRt(
    const af::array& values,
    const af::array& rowOffsets,
    const af::array& columnIndices,
    const af::array& input,
    const dim_t dim) {
  auto out = af::dense(af::sparse(
      input.dims(1), input.dims(0), values, rowOffsets, columnIndices));
  af::print("out", out);
}

// void denseToSparse2(
//     const af::array& input,
//     af::array& values,
//     af::array& rowIndices,
//     af::array& rowOffsets,
//     af::array& columnIndices) {
//   if (input.numdims() != 2) {
//     throw std::invalid_argument(
//         "sputnik denseToSparse operates only on tensors of order 2");
//   }

//   // Nonzero values and indices
//   values = af::lookup(af::flat(input.T()), af::where(input.T()));
//   auto zeroMask = (input != 0);

//   // Row offsets
//   af::print("zeroMask", zeroMask);
//   auto sum = af::sum(zeroMask, /* dim = */ 1);
//   af::print("sum", sum);
//   rowOffsets = af::join(
//       0,
//       af::constant(0, 1, af::dtype::u32),
//       af::accum(af::flat(af::sum(zeroMask, /* dim = */ 1))));

//   // Row indices
//   rowIndices = fl::cuda::getSparseRowIndices(rowOffsets);

//   // Column indices
//   // auto tiled = af::tile((af::range(input.dims(1)) + 1), 1,
//   // zeroMask.dims(0));
//   // auto tiled = af::range(af::dim4(input.dims(1), zeroMask.dims(0))) + 1;
//   auto tiled = af::range(af::dim4(input.dims(1), zeroMask.dims(0)), 0) + 1;
//   af::print("tiled", tiled);
//   auto nzc = (af::flat(zeroMask) * af::flat(tiled));
//   af::print("nzc", nzc);
//   columnIndices = af::lookup(nzc, af::where(nzc)) - 1;
// }

af::array sputnikSparseSoftmax(const af::array& input, const dim_t dim) {
  if (input.numdims() != 2) {
    throw std::invalid_argument(
        "sputnikSparseSoftmax must be a tensor of order 2");
  }
  if (dim != 1 && dim != 0) {
    throw std::invalid_argument("sputnikSparseSoftmax dim can be 0 or 1");
  }

  // ArrayFire is column major, but Sputnik's kernels all expect row-major
  // value ordering. Using CSR with ArrayFire's af::sparse performs a
  // transpose of the entire input; this limits the benefit of using a sparse
  // kernel across the second dim.
  //
  // Using Sputnik with dim 0 (efficient) gives a CSR order but doesn't
  // transpose the input, which is equivalent to doing softmax over dim 1 with
  // a transpose. Using with dim 1 (less efficient) transposes the value order
  // of the input and generates the equivalent of a CSC but with row/column
  // roles flipped (as required by Sputnik's API). Transposing the value order
  // is required since most kernels require contiguity along the dim over
  // which ops are being performed. This is ArrayFire's behavior, so use their
  // API.

  af::array values, rowOffsets, rowIndices, columnIndices;
  // if (dim == 1) {
  //   auto sparse = af::sparse(input, AF_STORAGE_CSR);
  //   af::storage storage;
  //   af::sparseGetInfo(values, rowOffsets, columnIndices, storage, sparse);
  //   rowIndices = getSparseRowIndices(rowOffsets);
  // } else if (dim == 0) {
  fl::cuda::denseToSparseCSRt(
      values, rowIndices, rowOffsets, columnIndices, input, dim);
  // }
  af::print("values", values);
  af::print("rowIndices", rowIndices);
  af::print("rowOffsets", rowOffsets);
  af::print("columnIndices", columnIndices);

  int nnz = columnIndices.dims(0);
  af::array output(nnz, input.type());

  {
    DevicePtr valuesPtr(values);
    DevicePtr rowIndicesPtr(rowIndices);
    DevicePtr rowOffsetsPtr(rowOffsets);
    DevicePtr columnIndicesPtr(columnIndices);
    DevicePtr outputPtr(output);

    FL_CUDA_CHECK(sputnik::SparseSoftmax(
        input.dims(dim == 0 ? 1 : 0) /* m */,
        input.dims(dim == 0 ? 0 : 1) /* n */,
        nnz /* nonzeros */,
        valuesPtr.getAs<float>() /* values */,
        rowIndicesPtr.getAs<int>() /* const int* __restrict__ row_indices */,
        rowOffsetsPtr.getAs<int>() /* const int* __restrict__ row_offsets */,
        columnIndicesPtr
            .getAs<int>() /* const int* __restrict__ column_indices */,
        outputPtr.getAs<float>() /* float* __restrict__ output_values */,
        fl::cuda::getActiveStream()));
  }

  af::sync();

  af::print("outputraw", output);

  sparseToDenseCSRt(values, rowOffsets, columnIndices, input, 0);
  // output = af::moddims(output, input.dims());

  // af::array dense;
  // af::dim4 denseDims(values.dims(0), input.dims(0), input.dims(1));
  // af::array matrix(denseDims, input.type());

  // auto sizes = af::diff1(rowOffsets);
  // auto idxs = af::range(input.dims(0));

  /*
      shape = (self.values.shape[0],) + self.shape
      matrix = torch.zeros(shape, dtype=self.values.dtype,
     device=self.values.device) sizes = torch.diff(self.row_offsets).long()
     idxs = torch.arange(self.shape[0], device=self.values.device) r_idxs =
     torch.repeat_interleave(idxs, sizes) for i, v in enumerate(self.values):
          matrix[i, r_idxs, self.column_indices.long()] = v
          # matrix[r_idxs, self.column_indices.long()] = self.values
      return matrix
  */

  return output;
}

} // namespace cuda
} // namespace fl
