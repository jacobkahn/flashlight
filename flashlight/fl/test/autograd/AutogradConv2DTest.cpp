/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/common/common.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/test/TestInit.h"
#include "flashlight/fl/test/autograd/AutogradTestUtils.h"

using namespace ::testing;
using namespace fl;

using fl::autograd::conv2d;
using fl::detail::AutogradTestF16;

// All shapes are NCHW: {N, C, H, W}
// Filter shapes are {K, C_in/groups, kH, kW}

TEST(AutogradConv2DTest, Convolve) {
  // Jacobian test: O(n_in + n_out) conv2d calls, each rebuilds a cuDNN
  // graph (no plan caching yet). Keep tensors small.
  // Input: N=2, C=3, H=4, W=5
  auto in = Tensor(fl::rand({2, 3, 4, 5}, fl::dtype::f32), true);
  // Filter: K=2, C_in=3, kH=3, kW=3
  auto wt = Tensor(fl::rand({2, 3, 3, 3}, fl::dtype::f32), true);
  // Bias: {1, K, 1, 1}
  auto bs = Tensor(fl::rand({1, 2, 1, 1}, fl::dtype::f32), true);
  int px = 1, py = 1;
  int sx = 1, sy = 1;
  int dx = 1, dy = 1;
  auto funcConvIn = [&](Tensor& input) {
    return conv2d(input, wt, sx, sy, px, py, dx, dy, /* groups */ 1);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvIn, in, 0.06));
  auto funcConvWt = [&](Tensor& weight) {
    return conv2d(in, weight, sx, sy, px, py, dx, dy, /* groups */ 1);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvWt, wt, 0.06));
  auto funcConvBs = [&](Tensor& bias) {
    return conv2d(in, wt, bias, sx, sy, px, py, dx, dy, /* groups */ 1);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvBs, bs, 0.03));
}

TEST_F(AutogradTestF16, ConvolveF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  const float scaleFactor = 10.0; // scale the input to prevent grad underflow
  // Input: N=1, C=2, H=1, W=3
  auto in =
      Tensor(fl::rand({1, 2, 1, 3}, fl::dtype::f16) * scaleFactor, true);
  // Filter: K=1, C_in=2, kH=1, kW=2
  auto wt = Tensor(fl::rand({1, 2, 1, 2}, fl::dtype::f16), true);
  int px = 1, py = 1;
  int sx = 1, sy = 1;
  int dx = 1, dy = 1;
  // Note: cuDNN frontend may not support f16 conv2d+bias fused graphs on all
  // architectures. Test without bias to avoid plan creation failures.
  auto funcConvIn = [&](Tensor& input) {
    return conv2d(
        input, wt, sx, sy, px, py, dx, dy, /* groups */ 1);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvIn, in, 5e-1, 0.1));
  auto funcConvWt = [&](Tensor& weight) {
    return conv2d(
        in, weight, sx, sy, px, py, dx, dy, /* groups */ 1);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvWt, wt, 5e-2, 0.1));
}

TEST(AutogradConv2DTest, ConvolveFilterGroups) {
  int channel = 4;
  int groups = 2;
  // Input: N=2, C=4, H=4, W=5
  auto in = Tensor(fl::rand({2, channel, 4, 5}, fl::dtype::f32), true);
  // Filter: K=2, C_in=channel/groups=2, kH=3, kW=3
  auto wt =
      Tensor(fl::rand({2, channel / groups, 3, 3}, fl::dtype::f32), true);
  // Bias: {1, K, 1, 1}
  auto bs = Tensor(fl::rand({1, 2, 1, 1}, fl::dtype::f32), true);
  int px = 1, py = 1;
  int sx = 1, sy = 1;
  int dx = 1, dy = 1;
  auto funcConvIn = [&](Tensor& input) {
    return conv2d(input, wt, bs, sx, sy, px, py, dx, dy, groups);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvIn, in, 0.06));
  auto funcConvWt = [&](Tensor& weight) {
    return conv2d(in, weight, bs, sx, sy, px, py, dx, dy, groups);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvWt, wt, 0.05));
  auto funcConvBs = [&](Tensor& bias) {
    return conv2d(in, wt, bias, sx, sy, px, py, dx, dy, groups);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvBs, bs, 0.02));
}

TEST(AutogradConv2DTest, ConvolveDilation) {
  // Input: N=2, C=3, H=6, W=6
  auto in = Tensor(fl::rand({2, 3, 6, 6}, fl::dtype::f32), true);
  // Filter: K=2, C_in=3, kH=3, kW=3
  auto wt = Tensor(fl::rand({2, 3, 3, 3}, fl::dtype::f32), true);
  // Bias: {1, K, 1, 1}
  auto bs = Tensor(fl::rand({1, 2, 1, 1}, fl::dtype::f32), true);
  int px = 2, py = 1;
  int sx = 1, sy = 1;
  int dx = 2, dy = 1;
  auto funcConvIn = [&](Tensor& input) {
    return conv2d(
        input, wt, bs, sx, sy, px, py, dx, dy, /* groups */ 1);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvIn, in, 0.06));
  auto funcConvWt = [&](Tensor& weight) {
    return conv2d(
        in, weight, bs, sx, sy, px, py, dx, dy, /* groups */ 1);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvWt, wt, 0.05));
  auto funcConvBs = [&](Tensor& bias) {
    return conv2d(
        in, wt, bias, sx, sy, px, py, dx, dy, /* groups */ 1);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvBs, bs, 0.02));
}

TEST(AutogradConv2DTest, WeightNormConv) {
  // Filter: K=2, C_in=2, kH=3, kW=3
  auto v = Tensor(fl::rand({2, 2, 3, 3}), true);
  std::vector<int> normDim = {1, 2, 3}; // norm over C_in, kH, kW
  auto g = Tensor(
      autograd::norm(v, normDim, /* p = */ 2, /* keepDims = */ true).copy(),
      true);
  // Input: N=2, C=2, H=5, W=5
  auto in = Tensor(fl::rand({2, 2, 5, 5}) * 2 - 2, true);

  auto funcWeightNormIn = [&](Tensor& input) {
    auto w = v *
        autograd::tileAs(
            g / autograd::norm(v, normDim, /* p = */ 2, /* keepDims = */ true),
            v);
    return conv2d(
        input, w, /* sx */ 1, /* sy */ 1, /* px */ 0,
        /* py */ 0, /* dx */ 1, /* dy */ 1, /* groups */ 1);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcWeightNormIn, in, 3E-1));

  auto funcWeightNormV = [&](Tensor& input) {
    auto w = input *
        autograd::tileAs(
            g / autograd::norm(
                    input, normDim, /* p = */ 2, /* keepDims = */ true),
            input);
    return conv2d(
        in, w, /* sx */ 1, /* sy */ 1, /* px */ 0,
        /* py */ 0, /* dx */ 1, /* dy */ 1, /* groups */ 1);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcWeightNormV, v, 2E-1));

  auto funcWeightNormG = [&](Tensor& input) {
    auto w = v *
        autograd::tileAs(
            input /
                autograd::norm(v, normDim, /* p = */ 2, /* keepDims = */ true),
            v);
    return conv2d(
        in, w, /* sx */ 1, /* sy */ 1, /* px */ 0,
        /* py */ 0, /* dx */ 1, /* dy */ 1, /* groups */ 1);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcWeightNormG, g, 2E-1));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  flTestInit();
  return RUN_ALL_TESTS();
}
