/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// TODO move all other Rnn tests to this file (e.g. from F16 tests)
#include <gtest/gtest.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/test/TestInit.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/test/autograd/AutogradTestUtils.h"

using namespace ::testing;
using namespace fl;

using fl::detail::AutogradTestF16;

namespace {

void testRnnImpl(RnnMode mode, fl::dtype precision = fl::dtype::f64) {
    int numLayers = 2;
    int hiddenSize = 2;
    int inputSize = 2;
    int batchSize = 2;
    int seqLength = 3;
    bool bidirectional = true;
    float expectedPrecision = precision == fl::dtype::f16 ? 5E-2 : 1E-5;
    float perturbation = precision == fl::dtype::f16 ? 1E-1 : 1E-4;

    // Row-major: {seqLength, batchSize, inputSize}
    auto in =
        Tensor(fl::rand({seqLength, batchSize, inputSize}, precision), true);
    size_t nParams;

    switch (mode) {
        case RnnMode::TANH:
        nParams = 56;
        break;
        case RnnMode::LSTM:
        nParams = 224;
        break;
        case RnnMode::GRU:
        nParams = 168;
        break;
        default:
        throw std::invalid_argument("invalid RNN mode for the test");
    }

    auto w =
        Tensor(fl::rand({static_cast<long long>(nParams)}, precision), true);

    auto funcRnnIn = [&](Tensor& input) -> Tensor {
        return std::get<0>(
            autograd::rnn(input,
                Tensor().astype(precision),
                Tensor().astype(precision),
                w,
                hiddenSize,
                numLayers,
                mode,
                bidirectional,
                0.0));
    };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcRnnIn, in, expectedPrecision, perturbation));

    auto funcRnnW = [&](Tensor& weights) -> Tensor {
        return std::get<0>(
            autograd::rnn(in,
                Tensor().astype(precision),
                Tensor().astype(precision),
                weights,
                hiddenSize,
                numLayers,
                mode,
                bidirectional,
                0.0));
    };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcRnnW, w, expectedPrecision, perturbation));

    // We get the correct gradient for hx
    // Row-major: hidden state is {totalLayers, batchSize, hiddenSize}
    auto hx = Tensor(
        fl::rand(
            {numLayers * (1 + bidirectional), batchSize, hiddenSize},
            precision),
        true);
    auto funcRnnHx = [&](Tensor& hiddenState) -> Tensor {
        return std::get<0>(
            autograd::rnn(in,
                hiddenState.astype(precision),
                Tensor().astype(precision),
                w,
                hiddenSize,
                numLayers,
                mode,
                bidirectional,
                0.0));
    };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcRnnHx, hx, expectedPrecision, perturbation));

    // We can compute the gradient w.r.t. hy
    auto funcRnnInDhy = [&](Tensor& input) -> Tensor {
        return std::get<1>(
            autograd::rnn(input,
                Tensor().astype(precision),
                Tensor().astype(precision),
                w,
                hiddenSize,
                numLayers,
                mode,
                bidirectional,
                0.0));
    };
    ASSERT_TRUE(
        fl::detail::jacobianTestImpl(funcRnnInDhy, in, expectedPrecision, perturbation));

    if (mode == RnnMode::LSTM) {
        // We get the correct gradient for cx
        // Row-major: cell state is {totalLayers, batchSize, hiddenSize}
        auto cx = Tensor(
            fl::rand(
                {numLayers * (1 + bidirectional), batchSize, hiddenSize},
                precision),
            true);
        auto funcRnnCx = [&](Tensor& cellState) -> Tensor {
        return std::get<0>(
            autograd::rnn(in,
                Tensor().astype(precision),
                cellState.astype(precision),
                w,
                hiddenSize,
                numLayers,
                mode,
                bidirectional,
                0.0));
        };
        ASSERT_TRUE(
            fl::detail::jacobianTestImpl(funcRnnCx, cx, expectedPrecision, perturbation));

        // We can compute the gradient w.r.t. cy
        auto funcRnnInDcy = [&](Tensor& input) -> Tensor {
        return std::get<2>(
            autograd::rnn(input,
                Tensor().astype(precision),
                Tensor().astype(precision),
                w,
                hiddenSize,
                numLayers,
                mode,
                bidirectional,
                0.0));
        };
        ASSERT_TRUE(
            fl::detail::jacobianTestImpl(funcRnnInDcy, in, expectedPrecision, perturbation));
    }
}

}

TEST(AutogradRnnTest, Rnn) {
  testRnnImpl(RnnMode::TANH);
}

TEST(AutogradRnnTest, Lstm) {
  testRnnImpl(RnnMode::LSTM);
}

TEST(AutogradRnnTest, Gru) {
  testRnnImpl(RnnMode::GRU);
}

TEST_F(AutogradTestF16, RnnF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  testRnnImpl(RnnMode::TANH, fl::dtype::f16);
}

TEST_F(AutogradTestF16, LstmF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  testRnnImpl(RnnMode::LSTM, fl::dtype::f16);
}

TEST_F(AutogradTestF16, GruF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  testRnnImpl(RnnMode::GRU, fl::dtype::f16);
}

TEST_F(AutogradTestF16, LstmBf16) {
  if (!fl::dtypeSupported(fl::dtype::bf16)) {
    GTEST_SKIP() << "bf16 not supported on this device";
  }

  int numLayers = 2;
  int hiddenSize = 2;
  int inputSize = 2;
  int batchSize = 2;
  int seqLength = 3;
  bool bidirectional = true;
  size_t nParams = 224; // LSTM

  // Row-major: {seqLength, batchSize, inputSize}
  auto in =
      Tensor(fl::rand({seqLength, batchSize, inputSize}, fl::dtype::bf16), true);
  auto w =
      Tensor(fl::rand({static_cast<long long>(nParams)}, fl::dtype::bf16), true);

  // OneDNN RNN primitives may not support bf16 on all CPU targets
  try {
    auto [out, hy, cy] = autograd::rnn(
        in, Tensor(), Tensor(), w,
        hiddenSize, numLayers, RnnMode::LSTM, bidirectional, 0.0);

    // Forward: no NaNs (RNN may upcast to f32 internally on some backends)
    ASSERT_FALSE(fl::any(fl::isnan(out.detach().astype(fl::dtype::f32))).asScalar<bool>());

    // Backward: gradients computed and finite
    autograd::sum(out, {0, 1, 2}).backward();
    ASSERT_FALSE(fl::any(fl::isnan(in.grad().detach().astype(fl::dtype::f32))).asScalar<bool>());
    ASSERT_FALSE(fl::any(fl::isnan(w.grad().detach().astype(fl::dtype::f32))).asScalar<bool>());
  } catch (const std::exception&) {
    GTEST_SKIP() << "OneDNN RNN primitives do not support bf16 on this CPU";
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  flTestInit();
  return RUN_ALL_TESTS();
}
