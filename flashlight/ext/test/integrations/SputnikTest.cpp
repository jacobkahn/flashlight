/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <af/array.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sputnik/sputnik.h>

#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/common/backend/cuda/CudaUtils.h"

using namespace fl;

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
