/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/common.h"
#include "flashlight/fl/contrib/modules/modules.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/test/TestInit.h"

using namespace fl;

namespace {

class ContribModuleTestF16 : public ::testing::Test {};

} // namespace

TEST(ContribModuleTest, ResidualFwd) {
  auto conv = Conv2D(30, 50, 9, 7, 2, 3, 3, 2);
  // bn is shared between both residual networks
  // Row-major: channel axis is 1 in {N, C, H, W}
  auto bn = std::make_shared<BatchNorm>(1, 50, 0.0);
  auto relu = ReLU();

  int batchsize = 10;
  // Row-major: {N, C, H, W}
  auto input = Tensor(fl::rand({batchsize, 30, 100, 120}), false);

  auto outputConv = conv.forward(input);
  auto outputBn = bn->forward(outputConv);

  Residual resModule1;
  resModule1.add(Conv2D(conv)); // explicit copy
  resModule1.add(bn);
  resModule1.add(ReLU());
  resModule1.addShortcut(1, 3);

  auto output1 = resModule1.forward(input);
  auto output1True = relu.forward(outputBn + outputConv);
  ASSERT_TRUE(allClose(output1, output1True));

  Residual resModule2;
  resModule2.add(std::move(conv));
  resModule2.add(bn);
  resModule2.add(ReLU());
  resModule2.addShortcut(1, 4);
  resModule2.addShortcut(1, 3);
  resModule2.addShortcut(2, 4);

  auto output2 = resModule2.forward(input);
  auto output2True =
      relu.forward(outputBn + outputConv) + outputBn + outputConv;
  ASSERT_TRUE(allClose(output2, output2True));
}

TEST(ContribModuleTest, ResidualFwdWithProjection) {
  const float proj1FwdScale = 0.24;
  const float proj2FwdScale = 0.5;
  const float linFwdScale = 0.3;

  auto linear1 = Linear(12, 8);
  auto relu1 = ReLU();
  auto linear2 = Linear(8, 4);
  auto relu2 = ReLU();
  auto linear3 = Linear(4, 4);
  auto relu3 = ReLU();
  auto projection1 = Linear(8, 4);
  auto projection2 = Linear(12, 4);

  // Row-major: input last dim = inputSize = 12
  auto input = Tensor(fl::rand({4, 3, 10, 12}), false);
  auto output1True = linear1.forward(input);
  auto outputTrue = relu1.forward(output1True);
  outputTrue = linear2.forward(outputTrue * linFwdScale);
  outputTrue = relu2.forward(
      (outputTrue + projection1.forward(output1True)) * proj1FwdScale);
  outputTrue = (outputTrue + projection2.forward(input)) * proj2FwdScale;
  outputTrue = linear3.forward(outputTrue);
  outputTrue = relu3.forward(outputTrue) + outputTrue;

  auto resModule = Residual();
  resModule.add(std::move(linear1));
  resModule.add(std::move(relu1));
  resModule.add(std::move(linear2));
  resModule.addScale(3, linFwdScale);
  resModule.add(std::move(relu2));
  resModule.addShortcut(1, 4, projection1);
  resModule.addScale(4, proj1FwdScale);
  resModule.add(std::move(linear3));
  resModule.addShortcut(0, 5, projection2);
  resModule.addScale(5, proj2FwdScale);
  resModule.add(std::move(relu3));
  resModule.addShortcut(5, 7);

  auto outputRes = resModule.forward(input);
  ASSERT_TRUE(allClose(outputRes, outputTrue));
}

TEST(ContribModuleTest, AsymmetricConv1DFwd) {
  int batchsize = 10;
  int timesteps = 120;
  int c = 32;

  auto conv = AsymmetricConv1D(c, c, 5, 1, -1, 0, 1); // use only past
  // Row-major: {N, C, H, W} where H=1, W=T (wx-->kW on dim 3)
  auto input = Tensor(fl::rand({batchsize, c, 1, timesteps}), false);

  auto output = conv.forward(input);

  ASSERT_EQ(output.dim(0), batchsize);
  ASSERT_EQ(output.dim(1), c);
  ASSERT_EQ(output.dim(3), timesteps);

  auto convFuture = AsymmetricConv1D(c, c, 5, 1, -1, 1, 1); // use only future
  auto outputFuture = convFuture.forward(input);
  ASSERT_EQ(outputFuture.dim(0), batchsize);
  ASSERT_EQ(outputFuture.dim(1), c);
  ASSERT_EQ(outputFuture.dim(3), timesteps);

  ASSERT_FALSE(allClose(output, outputFuture));
}

void transformerPadMaskFwd(bool isfp16) {
  int timesteps = 10;
  int c = 4;
  int nheads = 2;
  auto dtype = isfp16 ? fl::dtype::f16 : fl::dtype::f32;

  auto tr =
      Transformer(c, c / nheads, c, nheads, timesteps, 0, 0, false, false);
  // Row-major: {B, T, C}
  auto input1 = Tensor(fl::rand({/* B = */ 1, timesteps, c}, dtype), false);
  auto input1NoPad = input1(fl::span, fl::range(0, timesteps / 2));
  auto input2 = Tensor(fl::rand({/* B = */ 1, timesteps, c}, dtype), false);
  auto input = autograd::concatenate({input1, input2}, 0); // concat along batch
  // padMask: {B, T}
  auto padMask = fl::full({2, timesteps}, 1);
  padMask(fl::range(0, 1), fl::range(timesteps / 2, timesteps)) = 0;
  auto noPadMask = fl::full({2, timesteps}, 1);

  auto output = tr.forward({input, Tensor(padMask, false)}).front();
  auto outputNoPad = tr.forward({input, Tensor(noPadMask, false)}).front();

  ASSERT_EQ(output.dim(0), 2);
  ASSERT_EQ(output.dim(1), timesteps);
  ASSERT_EQ(output.dim(2), c);

  ASSERT_EQ(outputNoPad.type(), input.type());

  auto output1 = tr.forward({input1NoPad,
                             Tensor(
                                 padMask(fl::range(0, 1))(
                                     fl::span, fl::range(0, timesteps / 2)),
                                 false)})
                     .front();
  auto output2 =
      tr.forward({input2, Tensor(padMask(fl::range(1, 2)), false)})
          .front();
  ASSERT_TRUE(allClose(
      output(fl::range(1, 2)), output2));
  ASSERT_TRUE(allClose(
      outputNoPad(fl::range(1, 2)),
      output2));
  ASSERT_TRUE(allClose(
      output(fl::range(0, 1), fl::range(0, timesteps / 2)),
      output1));
  ASSERT_FALSE(allClose(
      outputNoPad(fl::range(0, 1), fl::range(0, timesteps / 2)),
      output1));
}

TEST(ContribModuleTest, TransformerPadMaskFwd) {
  transformerPadMaskFwd(false);
}

TEST_F(ContribModuleTestF16, TransformerPadMaskFwd16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }
  transformerPadMaskFwd(true);
}

void transformerFwd(bool isfp16) {
  int batchsize = 10;
  int timesteps = 120;
  int c = 32;
  int nheads = 4;
  auto dtype = isfp16 ? fl::dtype::f16 : fl::dtype::f32;

  auto tr =
      Transformer(c, c / nheads, c, nheads, timesteps, 0.2, 0.1, true, false);
  // Row-major: {B, T, C}
  auto input = Tensor(fl::rand({batchsize, timesteps, c}, dtype), false);

  fl::Tensor padMask;
  auto output = tr.forward({input, padMask});
  ASSERT_EQ(output[0].type(), input.type());

  ASSERT_EQ(output[0].dim(0), batchsize);
  ASSERT_EQ(output[0].dim(1), timesteps);
  ASSERT_EQ(output[0].dim(2), c);

  tr.setDropout(0);
  tr.setLayerDropout(0);
  auto output1 = tr.forward({input, padMask}).front();
  auto output2 = tr.forward({input, padMask}).front();
  ASSERT_TRUE(allClose(output1, output2, 1E-7));
}

TEST(ContribModuleTest, TransformerFwd) {
  transformerFwd(false);
}

TEST_F(ContribModuleTestF16, TransformerFwdF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }
  transformerFwd(true);
}

void conformerFwd(bool isfp16) {
  int batchsize = 10;
  int timesteps = 120;
  int c = 32;
  int nheads = 4;
  auto dtype = isfp16 ? fl::dtype::f16 : fl::dtype::f32;

  auto tr = Conformer(c, c / nheads, c, nheads, timesteps, 33, 0.2, 0.1);
  // Row-major: {B, T, C}
  auto input = Tensor(fl::rand({batchsize, timesteps, c}, dtype), false);

  auto output = tr.forward({input, Tensor()});
  ASSERT_EQ(output[0].type(), input.type());

  ASSERT_EQ(output[0].dim(0), batchsize);
  ASSERT_EQ(output[0].dim(1), timesteps);
  ASSERT_EQ(output[0].dim(2), c);
}

TEST(ContribModuleTest, ConformerFwd) {
  conformerFwd(false);
}

TEST_F(ContribModuleTestF16, ConformerFwdF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }
  conformerFwd(true);
}

void positionEmbeddingFwd(bool isfp16) {
  int batchsize = 10;
  int timesteps = 120;
  int csz = 256;
  auto dtype = isfp16 ? fl::dtype::f16 : fl::dtype::f32;

  auto posemb = PositionEmbedding(csz, timesteps, 0.5);
  // Row-major: {B, T, C}
  auto input = Tensor(fl::rand({batchsize, timesteps, csz}, dtype), false);

  auto output = posemb.forward({input});

  ASSERT_EQ(output[0].dim(0), batchsize);
  ASSERT_EQ(output[0].dim(1), timesteps);
  ASSERT_EQ(output[0].dim(2), csz);

  ASSERT_FALSE(allClose(output[0], input));
}

TEST(ContribModuleTest, PositionEmbeddingFwd) {
  positionEmbeddingFwd(false);
}

TEST_F(ContribModuleTestF16, PositionEmbeddingFwdF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }
  positionEmbeddingFwd(true);
}

void sinusoidalPositionEmbeddingFwd(bool isfp16) {
  int batchsize = 10;
  int timesteps = 120;
  int csz = 256;
  auto dtype = isfp16 ? fl::dtype::f16 : fl::dtype::f32;

  auto posemb = SinusoidalPositionEmbedding(csz, /* inputScale = */ 2.);
  // Row-major: {B, T, C}
  auto input =
      Tensor(fl::rand({batchsize, timesteps, csz}, dtype), false) - 0.5;

  auto output = posemb.forward({input});

  ASSERT_EQ(output[0].dim(0), batchsize);
  ASSERT_EQ(output[0].dim(1), timesteps);
  ASSERT_EQ(output[0].dim(2), csz);
  auto castOutput = output[0];
  if (isfp16) {
    castOutput = output[0].astype(fl::dtype::f32);
  }
  ASSERT_TRUE((fl::amax(castOutput, {0})).scalarSync<float>() <= 2);
  ASSERT_TRUE((fl::amin(castOutput, {0})).scalarSync<float>() >= -2);
}

TEST(ContribModuleTest, SinusoidalPositionEmbeddingFwd) {
  sinusoidalPositionEmbeddingFwd(false);
}

TEST_F(ContribModuleTestF16, SinusoidalPositionEmbeddingFwdF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }
  sinusoidalPositionEmbeddingFwd(true);
}

TEST(ContribModuleTest, AdaptiveEmbedding) {
  std::vector<float> values = {1, 4, 6, 2, 12, 7, 4, 21, 22, 18, 3, 23};
  int T = 6, B = 2, dim = 128;
  // Row-major: input is {B, T}
  auto input = Tensor(Tensor::fromVector({B, T}, values), false);
  std::vector<int> cutoff = {5, 10, 25};
  auto emb = AdaptiveEmbedding(dim, cutoff);
  auto output = emb.forward(input);

  ASSERT_EQ(output.dim(0), B);
  ASSERT_EQ(output.dim(1), T);
  ASSERT_EQ(output.dim(2), dim);
}

void tdsFwd(bool isfp16) {
  int batchsize = 10;
  int timesteps = 120;
  int w = 4;
  int c = 10;

  auto tds = TDSBlock(c, 9, w);
  auto dtype = isfp16 ? fl::dtype::f16 : fl::dtype::f32;
  // Row-major: {N, C, H, W} = {B, channels, width, T}
  // wx=kernelSize maps to kW (dim 3 = T)
  auto input = Tensor(fl::rand({batchsize, c, w, timesteps}, dtype), false);

  auto output = tds.forward({input})[0];

  ASSERT_EQ(output.dim(0), batchsize);
  ASSERT_EQ(output.dim(1), c);
  ASSERT_EQ(output.dim(2), w);
  ASSERT_EQ(output.dim(3), timesteps);
  ASSERT_EQ(output.type(), input.type());
}

TEST(ContribModuleTest, TDSFwd) {
  tdsFwd(false);
}

TEST_F(ContribModuleTestF16, TDSFwdF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }
  tdsFwd(true);
}

void streamingTDSFwd(bool isfp16) {
  int batchsize = 10;
  int timesteps = 120;
  int w = 4;
  int c = 10;
  int kw = 9;
  int rPad = 3;

  auto stds =
      TDSBlock(c, kw, w, 0 /* dropout */, 0 /* innerLinearDim */, rPad, true);
  auto dtype = isfp16 ? fl::dtype::f16 : fl::dtype::f32;
  // Row-major: {N, C, H, W} = {B, channels, width, T}
  auto input = Tensor(fl::rand({batchsize, c, w, timesteps}, dtype), false);

  auto output = stds.forward({input})[0];

  ASSERT_EQ(output.dim(0), batchsize);
  ASSERT_EQ(output.dim(1), c);
  ASSERT_EQ(output.dim(2), w);
  ASSERT_EQ(output.dim(3), timesteps);
  ASSERT_EQ(output.type(), input.type());
}

TEST(ContribModuleTest, StreamingTDSFwd) {
  streamingTDSFwd(false);
}

TEST_F(ContribModuleTestF16, StreamingTDSFwdF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }
  streamingTDSFwd(true);
}

TEST(ContribModuleTest, SpecAugmentFwd) {
  SpecAugment specAug(0, 27, 2, 100, 0.2, 2);
  int T = 512, F = 80;
  auto input = Tensor(fl::rand({T, F}), false);

  specAug.eval();
  ASSERT_TRUE(fl::allClose(input, specAug(input)));

  specAug.train();
  auto output = specAug(input);
  ASSERT_FALSE(fl::allClose(input, output));

  // Every value of output is either 0 or input
  for (int t = 0; t < T; ++t) {
    for (int f = 0; f < F; ++f) {
      auto o = output(t, f).scalarSync<float>();
      auto i = input(t, f).scalarSync<float>();
      ASSERT_TRUE(o == i || o == 0);
    }
  }

  // non-zero time frames are masked
  int tZeros = 0;
  for (int t = 0; t < T; ++t) {
    auto curOutSlice = output(t);
    tZeros = fl::all(curOutSlice == 0).asScalar<bool>() ? tZeros + 1 : tZeros;
  }
  ASSERT_GT(tZeros, 0);

  // non-zero frequency channels are masked
  int fZeros = 0;
  for (int f = 0; f < F; ++f) {
    auto curOutSlice = output(fl::span, f);
    fZeros = fl::all(curOutSlice == 0).asScalar<bool>() ? fZeros + 1 : fZeros;
  }
  ASSERT_GT(fZeros, 0);
}

void computeRawWavSpecAug(bool isfp16, float epsilon) {
  // no time, only freq masking
  for (int nmask = 1; nmask < 3; nmask++) {
    RawWavSpecAugment specAug(
        0, 1, nmask, 0, 0, 0, 1, 2000, 6000, 16000, 20000);
    specAug.train();

    int T = 300, C = 3, B = 4;
    auto time = 2 * M_PI * fl::iota({T}) / 16000;
    auto finalWav = fl::sin(time * 500) + fl::sin(time * 1000) +
        fl::sin(time * 7000) + fl::sin(time * 7500);
    auto inputWav = finalWav + fl::sin(time * 3000) + fl::sin(time * 4000) +
        fl::sin(time * 5000);
    // Row-major: input is {B, C, T}
    inputWav = fl::tile(fl::reshape(inputWav, {1, 1, T}), {B, C, 1});
    finalWav = fl::tile(fl::reshape(finalWav, {1, 1, T}), {B, C, 1});
    if (isfp16) {
      inputWav = inputWav.astype(fl::dtype::f16);
      finalWav = finalWav.astype(fl::dtype::f16);
    }

    auto filteredWav = specAug(fl::Tensor(inputWav, false));
    // compare middle of filtered wave to avoid edge artifacts comparison
    int halfKernelWidth = 63;

    // Slice on dim 2 (T)
    ASSERT_TRUE(fl::allClose(
        fl::Tensor(
            finalWav(fl::span, fl::span,
                fl::range(halfKernelWidth, T - halfKernelWidth)),
            false),
        filteredWav(fl::span, fl::span,
            fl::range(halfKernelWidth, T - halfKernelWidth)),
        epsilon));
  }
}

TEST(ContribModuleTest, RawWavSpecAugmentFwd) {
  computeRawWavSpecAug(false, 2e-3);
}

TEST_F(ContribModuleTestF16, RawWavSpecAugmentFwdF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }
  computeRawWavSpecAug(true, 1e-2);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  flTestInit();
  return RUN_ALL_TESTS();
}
