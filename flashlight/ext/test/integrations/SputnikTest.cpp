/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <af/array.h>
#include <af/sparse.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sputnik/sputnik.h>

#include "flashlight/ext/integrations/sputnik/SputnikInterface.h"
#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/common/Init.h"
#include "flashlight/fl/common/backend/cuda/CudaUtils.h"

using namespace fl;

namespace {

/**
 * Reference impl for converting an ArrayFire Array into CSR format
 */
} // namespace

// TEST(SputnikTest, SparseCSRRef) {
//   // Sanity check ArrayFire CSR
//   auto input = af::randu({3, 4});
//   input(1, 1) = 0.;

//   // Reference impl
//   af::array valuesRef, rowIndicesRef, rowOffsetsRef, columnIndicesRef;
//   denseToSparseCSRRef(
//       input.T(), valuesRef, rowIndicesRef, rowOffsetsRef, columnIndicesRef);

//   auto sparse = af::sparse(input, AF_STORAGE_CSR);
//   af::storage storage;
//   af::array values, rowOffsets, columnIndices;
//   af::sparseGetInfo(values, rowOffsets, columnIndices, storage, sparse);

//   ASSERT_EQ(storage, AF_STORAGE_CSR);
//   af::print("valuesRef", valuesRef);
//   af::print("values", values);
//   ASSERT_TRUE(allClose(valuesRef, values));
//   af::print("ref", rowOffsetsRef);
//   af::print("rowOffsets", rowOffsets);

//   ASSERT_TRUE(allClose(rowOffsetsRef, rowOffsets));
//   ASSERT_TRUE(allClose(columnIndicesRef, columnIndices));
// }

TEST(SputnikTest, Basic) {
  // TODO: write an actual test, wrap me with some stuffs
  // For now, check that I build and link - I will segfault
  FL_CUDA_CHECK(sputnik::SparseSoftmax(
      1 /* m */,
      1 /* n */,
      0 /* nonzeros */,
      nullptr /* const float* __restrict__ values */,
      nullptr /* const int* __restrict__ row_indices */,
      nullptr /* const int* __restrict__ row_offsets */,
      nullptr /* const int* __restrict__ column_indices */,
      nullptr /* float* __restrict__ output_values */,
      fl::cuda::getActiveStream()));
}

TEST(SputnikTest, Computation) {
  float threshold = 0.01;
  auto in = af::randu({2, 3});
  in(1, 1) = 0.;
  af::print("in", in);

  auto out = fl::cuda::sputnikSparseSoftmax(in, 0);
  af::print("out", out);

  af::print("manual 0", fl::softmax(Variable(in, false), 0).array());
  af::print("manual 1", fl::softmax(Variable(in.T(), false), 0).array());
  af::print("manual 1 no trans", fl::softmax(Variable(in, false), 1).array());

  // af::array values, rowIndices, rowOffsets, columnIndices;
  // fl::cuda::denseToSparse(in, values, rowIndices, rowOffsets, columnIndices);

  // // TODO: write an actual test, wrap me with some stuffs
  // // For now, check that I build and link - I will segfault
  // FL_CUDA_CHECK(sputnik::SparseSoftmax(
  //     1 /* m */,
  //     1 /* n */,
  //     0 /* nonzeros - unused internally */,
  //     nullptr /* const float* __restrict__ values */,
  //     nullptr /* const int* __restrict__ row_indices */,
  //     nullptr /* const int* __restrict__ row_offsets */,
  //     nullptr /* const int* __restrict__ column_indices */,
  //     nullptr /* float* __restrict__ output_values */,
  //     fl::cuda::getActiveStream()));
}

int main(int argc, char** argv) {
  fl::init();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
