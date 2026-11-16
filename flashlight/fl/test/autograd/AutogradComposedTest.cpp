/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/test/autograd/AutogradTestUtils.h"
#include "flashlight/fl/test/TestInit.h"

using namespace fl;
namespace ag = fl::autograd;

class AutogradComposedTest : public ::testing::Test {};

// Composed matmul chains amplify f32 rounding noise. Larger perturbations keep
// the finite-difference signal above the noise floor. Inputs are biased
// positive (+2) so pre-relu values stay away from the non-differentiable
// point at 0.

// relu(linear(x)) — single layer, gradient wrt input
TEST_F(AutogradComposedTest, ReluLinearJacobianInput) {
  int inDim = 4, outDim = 3;
  auto w = Tensor(fl::randn({outDim, inDim}) + 2.0, true);

  auto func = [&](Tensor& x) -> Tensor {
    return ag::relu(ag::matmul(x, ag::transpose(w)));
  };

  auto input = Tensor(fl::randn({1, inDim}) + 2.0, true);
  EXPECT_TRUE(fl::detail::jacobianTestImpl(func, input, 5e-4, 1e-2));
}

// relu(linear(x)) — gradient wrt weight
TEST_F(AutogradComposedTest, ReluLinearJacobianWeight) {
  int inDim = 4, outDim = 3;
  auto x = Tensor(fl::randn({1, inDim}) + 2.0, false);

  auto func = [&](Tensor& w) -> Tensor {
    return ag::relu(ag::matmul(x, ag::transpose(w)));
  };

  auto weight = Tensor(fl::randn({outDim, inDim}) + 2.0, true);
  // Relaxed precision: cuBLASLt DRELU epilogue packs the relu mask into a
  // 1-bit bitmask applied during the backward matmul, which can introduce
  // small numerical differences vs. the unfused element-wise path.
  EXPECT_TRUE(fl::detail::jacobianTestImpl(func, weight, 5e-3, 1e-2));
}

// relu(linear(relu(linear(x)))) — stacked layers, cross-layer backward fusion
TEST_F(AutogradComposedTest, StackedReluLinearJacobian) {
  int inDim = 4, hidDim = 5, outDim = 3;
  auto w1 = Tensor(fl::randn({hidDim, inDim}) + 2.0, true);
  auto w2 = Tensor(fl::randn({outDim, hidDim}) + 2.0, true);

  auto func = [&](Tensor& x) -> Tensor {
    auto h = ag::relu(ag::matmul(x, ag::transpose(w1)));
    return ag::relu(ag::matmul(h, ag::transpose(w2)));
  };

  auto input = Tensor(fl::randn({1, inDim}) + 2.0, true);
  EXPECT_TRUE(fl::detail::jacobianTestImpl(func, input, 5e-4, 5e-2));
}

// gelu(linear(x)) — future DGELU baseline
TEST_F(AutogradComposedTest, GeluLinearJacobian) {
  int inDim = 4, outDim = 3;
  auto w = Tensor(fl::randn({outDim, inDim}) + 2.0, true);

  auto func = [&](Tensor& x) -> Tensor {
    return ag::gelu(ag::matmul(x, ag::transpose(w)));
  };

  auto input = Tensor(fl::randn({1, inDim}) + 2.0, true);
  EXPECT_TRUE(fl::detail::jacobianTestImpl(func, input, 5e-4, 1e-2));
}

// relu(linear(x)) with batched inputs
TEST_F(AutogradComposedTest, ReluLinearBatchedJacobian) {
  int batchSize = 2, inDim = 4, outDim = 3;
  auto w = Tensor(fl::randn({outDim, inDim}) + 2.0, true);

  auto func = [&](Tensor& x) -> Tensor {
    return ag::relu(ag::matmul(x, ag::transpose(w)));
  };

  auto input = Tensor(fl::randn({batchSize, inDim}) + 2.0, true);
  EXPECT_TRUE(fl::detail::jacobianTestImpl(func, input, 5e-4, 1e-2));
}

// Three-layer stacked relu(linear(x)) — deeper cross-layer pattern
TEST_F(AutogradComposedTest, ThreeLayerStackedReluLinearJacobian) {
  int d1 = 4, d2 = 5, d3 = 4, d4 = 3;
  auto w1 = Tensor(fl::randn({d2, d1}) + 2.0, true);
  auto w2 = Tensor(fl::randn({d3, d2}) + 2.0, true);
  auto w3 = Tensor(fl::randn({d4, d3}) + 2.0, true);

  auto func = [&](Tensor& x) -> Tensor {
    auto h1 = ag::relu(ag::matmul(x, ag::transpose(w1)));
    auto h2 = ag::relu(ag::matmul(h1, ag::transpose(w2)));
    return ag::relu(ag::matmul(h2, ag::transpose(w3)));
  };

  // Three chained matmuls compound f32 rounding; use wider tolerance.
  auto input = Tensor(fl::randn({1, d1}) + 2.0, true);
  EXPECT_TRUE(fl::detail::jacobianTestImpl(func, input, 5e-3, 1e-1));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  flTestInit();
  return RUN_ALL_TESTS();
}
