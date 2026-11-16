/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <cmath>

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/test/TestInit.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/test/autograd/AutogradTestUtils.h"

using namespace ::testing;
using namespace fl;

using fl::detail::AutogradTestF16;

TEST(AutogradNormalizationTest, Normalize) {
  auto x = Tensor(fl::rand({5, 3}, fl::dtype::f64), true);
  auto funcNormalize2 = [](Tensor& in) { return autograd::normalize(in, {1}); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcNormalize2, x));
  auto ys = funcNormalize2(x);
  ASSERT_TRUE(allClose(
      fl::sum(ys * ys, {1}),
      fl::full({5}, 1, fl::dtype::f64)));
  auto yb = autograd::normalize(x, {1}, 2, 1);
  ASSERT_TRUE(fl::all(fl::sqrt(fl::sum(yb * yb, {1})) <= 1)
                  .scalarSync<char>());
}

TEST(AutogradNormalizationTest, BatchNormEvalModeOutputSingleAxis) {
  int featDims = 3;
  std::vector<int> featAxes = {1};
  // Row-major input order: NCHW
  auto input = Tensor(fl::rand({16, featDims, 13, 13}), false);
  auto runningMean = Tensor(fl::rand({featDims}, input.type()), false);
  auto runningVar = Tensor(fl::rand({featDims}, input.type()), false);
  auto weight = Tensor(fl::rand({featDims}, input.type()), false);
  auto bias = Tensor(fl::rand({featDims}, input.type()), false);

  auto out = std::get<0>(autograd::batchnorm(
      input,
      weight,
      bias,
      runningMean,
      runningVar,
      featAxes,
      false,
      0.0,
      1E-5));
  for (int i = 0; i < featDims; ++i) {
    std::array<fl::Index, 4> sel = {fl::span, i, fl::span, fl::span};
    auto thisInput = input(sel[0], sel[1], sel[2], sel[3]);
    auto thisMean = runningMean.flatten()(i).scalarSync<float>();
    auto thisVar = runningVar.flatten()(i).scalarSync<float>();
    auto thisWeight = weight.flatten()(i).scalarSync<float>();
    auto thisBias = bias.flatten()(i).scalarSync<float>();

    auto expectedOut = (thisInput - thisMean) / std::sqrt(thisVar + 1E-5);
    expectedOut = expectedOut * thisWeight + thisBias;
    ASSERT_TRUE(allClose(
        out(sel[0], sel[1], sel[2], sel[3]), expectedOut, 1E-5));
  }

  // test on empty weigts and bias
  out = std::get<0>(autograd::batchnorm(
      input,
      Tensor(),
      Tensor(),
      runningMean,
      runningVar,
      featAxes,
      false,
      0.0,
      1E-5));
  for (int i = 0; i < featDims; ++i) {
    std::array<fl::Index, 4> sel = {fl::span, i, fl::span, fl::span};
    auto thisInput = input(sel[0], sel[1], sel[2], sel[3]);
    auto thisMean = runningMean.flatten()(i).scalarSync<float>();
    auto thisVar = runningVar.flatten()(i).scalarSync<float>();

    auto expectedOut = (thisInput - thisMean) / std::sqrt(thisVar + 1E-5);
    ASSERT_TRUE(allClose(
        out(sel[0], sel[1], sel[2], sel[3]), expectedOut, 1E-5));
  }
}

TEST(AutogradNormalizationTest, BatchNormEvalModeOutputMultipleAxis) {
  // Row-major input order: NCHW. Feature axes = {C, H, W}, batch = N.
  std::vector<int> featAxes = {1, 2, 3};
  auto input = Tensor(fl::rand({16, 4, 13, 13}), false);

  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dim(ax);
  }
  auto runningMean = Tensor(fl::rand({nfeatures}, input.type()), false);
  auto runningVar = Tensor(fl::rand({nfeatures}, input.type()), false);
  auto weight = Tensor(fl::rand({nfeatures}, input.type()), false);
  auto bias = Tensor(fl::rand({nfeatures}, input.type()), false);

  auto out = std::get<0>(autograd::batchnorm(
      input,
      weight,
      bias,
      runningMean,
      runningVar,
      featAxes,
      false,
      0.0,
      1E-5));
  for (int i = 0; i < nfeatures; ++i) {
    // Row-major flattening: dim 3 changes fastest, then dim 2, then dim 1
    std::array<fl::Index, 4> sel = {
        fl::span, i / 169, (i / 13) % 13, i % 13};
    auto thisInput = input(sel[0], sel[1], sel[2], sel[3]);
    auto thisMean = runningMean.flatten()(i).scalarSync<float>();
    auto thisVar = runningVar.flatten()(i).scalarSync<float>();
    auto thisWeight = weight.flatten()(i).scalarSync<float>();
    auto thisBias = bias.flatten()(i).scalarSync<float>();

    auto expectedOut = (thisInput - thisMean) / std::sqrt(thisVar + 1e-5);
    expectedOut = expectedOut * thisWeight + thisBias;

    ASSERT_TRUE(allClose(
        out(sel[0], sel[1], sel[2], sel[3]), expectedOut, 1e-4));
  }

  // test on empty weigts and bias
  out = std::get<0>(autograd::batchnorm(
      input,
      Tensor(),
      Tensor(),
      runningMean,
      runningVar,
      featAxes,
      false,
      0.0,
      1E-5));
  for (int i = 0; i < nfeatures; ++i) {
    std::array<fl::Index, 4> sel = {
        fl::span, i / 169, (i / 13) % 13, i % 13};
    auto thisInput = input(sel[0], sel[1], sel[2], sel[3]);
    auto thisMean = runningMean.flatten()(i).scalarSync<float>();
    auto thisVar = runningVar.flatten()(i).scalarSync<float>();

    auto expectedOut = (thisInput - thisMean) / std::sqrt(thisVar + 1e-5);
    ASSERT_TRUE(allClose(
        out(sel[0], sel[1], sel[2], sel[3]), expectedOut, 5e-5));
  }
}

TEST(AutogradNormalizationTest, BatchNormTrainModeOutputSingleAxis) {
  int numFeat = 3;
  std::vector<int> featAxes = {1};
  double epsilon = 1E-5;
  // Row-major: NCHW
  auto input = Tensor(fl::rand({8, numFeat, 13, 13}), true);
  auto weight = Tensor(fl::rand({numFeat}), true);
  auto bias = Tensor(fl::rand({numFeat}), true);
  auto runningMean = Tensor(fl::rand({numFeat}), false);
  auto runningVar = Tensor(fl::rand({numFeat}), false);

  auto out = std::get<0>(autograd::batchnorm(
      input,
      weight,
      bias,
      runningMean,
      runningVar,
      featAxes,
      true,
      0.0,
      epsilon));

  auto todim = Shape({1, numFeat, 1, 1});
  std::vector<int> nrmAxes = {0, 2, 3};
  auto avg = autograd::reshape(autograd::mean(input, nrmAxes), todim);
  auto variance =
      autograd::reshape(autograd::var(input, nrmAxes, false /* population var */), todim);
  auto expectedOut = (input - autograd::tileAs(avg, input)) /
      autograd::sqrt(autograd::tileAs(variance, input) + epsilon);
  expectedOut = expectedOut * autograd::tileAs(autograd::reshape(weight, todim), input) +
      autograd::tileAs(autograd::reshape(bias, todim), input);
  ASSERT_TRUE(allClose(out, expectedOut, 1e-5));
}

TEST(AutogradNormalizationTest, BatchNormTrainModeOutputMultipleAxis) {
  // Row-major: NCHW. Feature axes = {C, H, W}, batch = N.
  std::vector<int> featAxes = {1, 2, 3};
  auto input = Tensor(fl::rand({8, 4, 13, 13}), true);

  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dim(ax);
  }
  auto weight = Tensor(fl::rand({nfeatures}), true);
  auto bias = Tensor(fl::rand({nfeatures}), true);
  auto runningMean = Tensor(fl::rand({nfeatures}), false);
  auto runningVar = Tensor(fl::rand({nfeatures}), false);

  auto out = std::get<0>(autograd::batchnorm(
      input, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));

  auto todim = Shape({nfeatures});
  std::vector<int> nrmAxes = {0};
  auto avg = autograd::reshape(autograd::mean(input, nrmAxes), todim);
  auto variance = autograd::reshape(autograd::var(input, nrmAxes, false), todim);

  for (int i = 0; i < nfeatures; ++i) {
    // Row-major flattening: dim 3 changes fastest, then dim 2, then dim 1
    std::array<fl::Index, 4> sel = {
        fl::span, i / 169, (i / 13) % 13, i % 13};
    auto thisInput = input(sel[0], sel[1], sel[2], sel[3]);
    auto thisMean = avg.flatten()(i).scalarSync<float>();
    auto thisVar = variance.flatten()(i).scalarSync<float>();
    auto thisWeight = weight.flatten()(i).scalarSync<float>();
    auto thisBias = bias.flatten()(i).scalarSync<float>();

    auto expectedOut = (thisInput - thisMean) / std::sqrt(thisVar + 1e-5);
    expectedOut = expectedOut * thisWeight + thisBias;
    ASSERT_TRUE(allClose(
        out(sel[0], sel[1], sel[2], sel[3]), expectedOut, 1e-5));
  }
}

TEST(AutogradNormalizationTest, BatchNormJacobian) {
  // Jacobian Test with trainMode = true;

  int numFeat = 3;
  std::vector<int> featAxes = {1};
  // Row-major: NCHW — smaller on CPU where Jacobian is O(n^2)
#ifdef FL_TEST_BACKEND_CPU
  auto input = Tensor(fl::rand({2, numFeat, 3, 3}, fl::dtype::f32), true);
#else
  auto input = Tensor(fl::rand({16, numFeat, 8, 8}, fl::dtype::f32), true);
#endif
  auto runningMean = Tensor(fl::rand({numFeat}, fl::dtype::f32), false);
  auto runningVar = Tensor(fl::rand({numFeat}, fl::dtype::f32), false);
  auto weight = Tensor(fl::rand({numFeat}, fl::dtype::f32), true);
  auto bias = Tensor(fl::rand({numFeat}, fl::dtype::f32), true);

  auto funcBnIn = [&](Tensor& in) {
    return std::get<0>(autograd::batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnIn, input, 1e-2, 1e-4));

  auto funcBnWt = [&](Tensor& wt) {
    return std::get<0>(autograd::batchnorm(
        input, wt, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnWt, weight, 1e-2, 1e-4));

  auto funcBnBs = [&](Tensor& bs) {
    return std::get<0>(autograd::batchnorm(
        input, weight, bs, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnBs, bias, 1e-2, 1e-4));
}

TEST_F(AutogradTestF16, BatchNormJacobianF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // Jacobian Test with trainMode = true;

  int numFeat = 3;
  std::vector<int> featAxes = {1};
  // Row-major: NCHW
  auto input = Tensor(fl::rand({16, numFeat, 8, 8}, fl::dtype::f16), true);
  auto runningMean = Tensor(fl::rand({numFeat}, fl::dtype::f32), false);
  auto runningVar = Tensor(fl::rand({numFeat}, fl::dtype::f32), false);
  auto weight = Tensor(fl::rand({numFeat}, fl::dtype::f32), true);
  auto bias = Tensor(fl::rand({numFeat}, fl::dtype::f32), true);

  // Use larger perturbations to ensure gradients don't underflow with fp16

  auto funcBnIn = [&](Tensor& in) {
    return std::get<0>(autograd::batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnIn, input, 5e-2, 1e-1));

  auto funcBnWt = [&](Tensor& wt) {
    return std::get<0>(autograd::batchnorm(
        input, wt, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnWt, weight, 5e-2, 1e-1));

  auto funcBnBs = [&](Tensor& bs) {
    return std::get<0>(autograd::batchnorm(
        input, weight, bs, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnBs, bias, 5e-2, 1e-1));
}

TEST(AutogradNormalizationTest, BatchNormJacobianMultipleAxes) {
  // Jacobian Test with  trainMode = true;
  // Row-major: NCHW. Feature axes = {C, H, W}, batch = N.
  // Batch size must be >= 4 to ensure stable variance estimates — with N=2,
  // batch variance is ((x1-x2)/2)^2 which can be near-zero for unlucky
  // random inputs, causing gradient blow-up through 1/sqrt(var+eps).
  std::vector<int> featAxes = {1, 2, 3};
#ifdef FL_TEST_BACKEND_CPU
  auto input = Tensor(fl::rand({4, 2, 3, 3}, fl::dtype::f32), true);
#else
  auto input = Tensor(fl::rand({16, 3, 8, 8}, fl::dtype::f32), true);
#endif
  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dim(ax);
  }
  auto runningMean = Tensor(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto runningVar = Tensor(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto weight = Tensor(fl::rand({nfeatures}, fl::dtype::f32), true);
  auto bias = Tensor(fl::rand({nfeatures}, fl::dtype::f32), true);

  auto funcBnIn = [&](Tensor& in) {
    return std::get<0>(autograd::batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnIn, input, 1e-2, 1e-3));

  auto funcBnWt = [&](Tensor& wt) {
    return std::get<0>(autograd::batchnorm(
        input, wt, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnWt, weight, 1e-2, 1e-3));

  auto funcBnBs = [&](Tensor& bs) {
    return std::get<0>(autograd::batchnorm(
        input, weight, bs, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnBs, bias, 1e-2, 1e-3));
}

TEST_F(AutogradTestF16, BatchNormJacobianMultipleAxesF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // Jacobian Test with trainMode = true;
  // Row-major: NCHW. Feature axes = {C, H, W}, batch = N.
  std::vector<int> featAxes = {1, 2, 3};
  auto input = Tensor(fl::rand({1, 2, 2, 2}, fl::dtype::f16), true);
  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dim(ax);
  }
  auto runningMean = Tensor(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto runningVar = Tensor(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto weight = Tensor(fl::rand({nfeatures}, fl::dtype::f32), true);
  auto bias = Tensor(fl::rand({nfeatures}, fl::dtype::f32), true);

  // Use larger perturbations to ensure gradients don't underflow with fp16

  auto funcBnIn = [&](Tensor& in) {
    return std::get<0>(autograd::batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(
      funcBnIn, input, 5e-2, 1e-1)); // TODO: investigate

  auto funcBnWt = [&](Tensor& wt) {
    return std::get<0>(autograd::batchnorm(
        input, wt, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnWt, weight, 5e-2, 1e-1));

  auto funcBnBs = [&](Tensor& bs) {
    return std::get<0>(autograd::batchnorm(
        input, weight, bs, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnBs, bias, 5e-2, 1e-1));
}

TEST_F(AutogradTestF16, BatchNormBf16) {
  if (!fl::dtypeSupported(fl::dtype::bf16)) {
    GTEST_SKIP() << "bf16 not supported on this device";
  }

  int numFeat = 3;
  std::vector<int> featAxes = {1};
  // Row-major: NCHW
  auto input = Tensor(fl::rand({16, numFeat, 8, 8}, fl::dtype::bf16), true);
  auto runningMean = Tensor(fl::rand({numFeat}, fl::dtype::f32), false);
  auto runningVar = Tensor(fl::rand({numFeat}, fl::dtype::f32), false);
  auto weight = Tensor(fl::rand({numFeat}, fl::dtype::f32), true);
  auto bias = Tensor(fl::rand({numFeat}, fl::dtype::f32), true);

  auto out = std::get<0>(autograd::batchnorm(
      input, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  // Forward: correct dtype
  ASSERT_EQ(out.type(), fl::dtype::bf16);

  // f32 reference comparison
  {
    auto input_f32 = Tensor(input.detach().astype(fl::dtype::f32), false);
    auto weight_f32 = Tensor(weight.detach(), false);
    auto bias_f32 = Tensor(bias.detach(), false);
    auto rm = Tensor(runningMean.copy(), false);
    auto rv = Tensor(runningVar.copy(), false);
    auto out_f32 = std::get<0>(autograd::batchnorm(
        input_f32, weight_f32, bias_f32, rm, rv, featAxes, true, 0.0, 1E-5));
    ASSERT_TRUE(allClose(
        out.detach().astype(fl::dtype::f32), out_f32.detach(), 0.05));
  }

  // Backward: correct gradient dtypes, no NaNs
  autograd::sum(out, {0, 1, 2, 3}).backward();
  ASSERT_EQ(input.grad().type(), fl::dtype::bf16);
  ASSERT_EQ(weight.grad().type(), fl::dtype::f32);
  ASSERT_FALSE(fl::any(fl::isnan(input.grad().detach().astype(fl::dtype::f32))).asScalar<bool>());
  ASSERT_FALSE(fl::any(fl::isnan(weight.grad().detach().astype(fl::dtype::f32))).asScalar<bool>());
}

TEST(AutogradNormalizationTest, LayerNormJacobian) {
  std::vector<int> featAxes = {0, 1, 2, 3};
#ifdef FL_TEST_BACKEND_CPU
  auto input = Tensor(fl::rand({2, 2, 3, 3}), true);
#else
  auto input = Tensor(fl::rand({10, 3, 7, 7}), true);
#endif
  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dim(ax);
  }
  auto runningMean = Tensor(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto runningVar = Tensor(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto weight = Tensor(fl::rand({nfeatures}, fl::dtype::f32), true);
  auto bias = Tensor(fl::rand({nfeatures}, fl::dtype::f32), true);

  auto funcLnIn = [&](Tensor& in) {
    return std::get<0>(autograd::batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };

  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcLnIn, input, 1e-2, 1e-4));
}

TEST_F(AutogradTestF16, LayerNormJacobianF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  std::vector<int> featAxes = {0, 1, 2, 3};
  const float inputScale = 4.0; // scale the input to prevent grad underflow
  auto input =
      Tensor(inputScale * fl::rand({4, 2, 2, 2}, fl::dtype::f16), true);
  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dim(ax);
  }
  auto runningMean = Tensor(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto runningVar = Tensor(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto weight = Tensor(fl::rand({nfeatures}, fl::dtype::f32), true);
  auto bias = Tensor(fl::rand({nfeatures}, fl::dtype::f32), true);

  auto funcLnIn = [&](Tensor& in) {
    return std::get<0>(autograd::batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };

  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcLnIn, input, 1e-4, 1e-2));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  flTestInit();
  return RUN_ALL_TESTS();
}
