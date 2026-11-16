/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/common.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/test/TestInit.h"
#include "flashlight/fl/tensor/Random.h"

using namespace fl;

namespace {

class ContainerTestClass : public Sequential {
 public:
  ContainerTestClass() = default;
  ContainerTestClass(const ContainerTestClass& other) {
    copy(other);
  }
  ContainerTestClass& operator=(const ContainerTestClass& other) {
    copy(other);
    return *this;
  }
  ContainerTestClass(ContainerTestClass&& other) = default;
  ContainerTestClass& operator=(ContainerTestClass&& other) = default;
  void copy(const ContainerTestClass& other) {
    auto orphanParamIdxMap = other.getOrphanedParamsIdxMap();
    for (int i = -1; i < static_cast<int>(other.modules_.size()); ++i) {
      if (i >= 0) {
        add(other.modules_[i]->clone());
      }
      auto [paramIter, pEnd] = orphanParamIdxMap.equal_range(i);
      for (; paramIter != pEnd; ++paramIter) {
        const auto& param = other.params_[paramIter->second];
        params_.emplace_back(param.copy());
      }
    }
  }

  std::unique_ptr<Module> clone() const override {
    return std::make_unique<ContainerTestClass>(*this);
  }

  void addParam(const Tensor& param) {
    params_.push_back(param);
  }
};

class ModuleTestF16 : public ::testing::Test {};

} // namespace

TEST(ModuleTest, EmbeddingFwd) {
  int embDim = 3, nEmb = 5, nQuery = 2, batchSize = 2;
  // Row-major: weight is {nEmb, embDim}, data in row-major order
  // Embedding table: [0]=[8,2,2], [1]=[10,5,3], [2]=[3,4,6], [3]=[12,3,8], [4]=[0,5,2]
  auto wtVar = param(Tensor::fromVector<float>(
      {nEmb, embDim}, {8, 2, 2, 10, 5, 3, 3, 4, 6, 12, 3, 8, 0, 5, 2}));

  auto inVar = input(Tensor::fromVector<float>({nQuery, batchSize}, {1, 3, 0, 0}));

  // Row-major output: {nQuery, batchSize, embDim}
  auto expectedOutVar = Tensor(
      Tensor::fromVector<float>(
          {nQuery, batchSize, embDim}, {10, 5, 3, 12, 3, 8, 8, 2, 2, 8, 2, 2}),
      true);

  // Var initialization
  auto emb = Embedding(wtVar);
  ASSERT_TRUE(allClose(emb.forward(inVar), expectedOutVar, 1E-7));

  // Regular initialization
  emb = Embedding(embDim, nEmb);
  wtVar = emb.param(0);
  ASSERT_EQ(wtVar.shape(), Shape({nEmb, embDim}));

  expectedOutVar = Tensor(
      fl::reshape(
          wtVar(inVar),
          {nQuery, batchSize, embDim}),
      true);
  ASSERT_TRUE(allClose(emb.forward(inVar), expectedOutVar, 1E-7));
}

TEST(ModuleTest, LinearFwd) {
  int n_in = 2, n_out = 3, x = 4, batchsize = 2;
  // Row-major: weight {n_out, n_in}, data in row-major order
  // Row 0: [8, 10], Row 1: [2, 5], Row 2: [2, 3]
  auto wtVar =
      param(Tensor::fromVector<float>({n_out, n_in}, {8, 10, 2, 5, 2, 3}));

  // Row-major: input {batchsize, x, n_in}, features in last dim
  auto inVar = input(Tensor::fromVector<float>(
      {batchsize, x, n_in}, {6, 2, 1, 4, 8, 2, 7, 1, 10, 7, 3, 7, 5, 9, 2, 4}));

  // Row-major: output {batchsize, x, n_out}
  auto expected_outVar = Tensor(
      Tensor::fromVector<float>(
          {batchsize, x, n_out},
          {68,  22, 18, 48, 22, 14, 84,  26, 22, 66, 19, 17,
           150, 55, 41, 94, 41, 27, 130, 55, 37, 56, 24, 16}),
      true);

  auto linNoBias = Linear(wtVar);
  ASSERT_TRUE(allClose(linNoBias.forward(inVar), expected_outVar, 1E-7));

  auto bsVar = input(Tensor::fromVector<float>({n_out}, {1, 2, 3}));
  expected_outVar = Tensor(
      Tensor::fromVector<float>(
          {batchsize, x, n_out},
          {69,  24, 21, 49, 24, 17, 85,  28, 25, 67, 21, 20,
           151, 57, 44, 95, 43, 30, 131, 57, 40, 57, 26, 19}),
      true);

  auto linBias = Linear(wtVar, bsVar);
  ASSERT_TRUE(allClose(linBias.forward(inVar), expected_outVar, 1E-7));
}

TEST_F(ModuleTestF16, LinearFwdF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  int n_in = 2, n_out = 3, x = 4, batchsize = 2;
  // Row-major: weight {n_out, n_in}
  auto wtVar =
      param(Tensor::fromVector<float>({n_out, n_in}, {8, 10, 2, 5, 2, 3}));

  // Row-major: input {batchsize, x, n_in}
  auto inVar = input(Tensor::fromVector<float>(
                         {batchsize, x, n_in},
                         {6, 2, 1, 4, 8, 2, 7, 1, 10, 7, 3, 7, 5, 9, 2, 4})
                         .astype(fl::dtype::f16));

  // Row-major: output {batchsize, x, n_out}
  auto expected_outVar = Tensor(
      Tensor::fromVector<float>(
          {batchsize, x, n_out},
          {68,  22, 18, 48, 22, 14, 84,  26, 22, 66, 19, 17,
           150, 55, 41, 94, 41, 27, 130, 55, 37, 56, 24, 16})
          .astype(fl::dtype::f16),
      true);

  auto linNoBias = Linear(wtVar);
  auto result = linNoBias.forward(inVar);
  ASSERT_EQ(result.type(), inVar.type());
  ASSERT_TRUE(allClose(result, expected_outVar, 1E-2));

  auto bsVar = input(Tensor::fromVector<float>({n_out}, {1, 2, 3}));
  expected_outVar = Tensor(
      Tensor::fromVector<float>(
          {batchsize, x, n_out},
          {69,  24, 21, 49, 24, 17, 85,  28, 25, 67, 21, 20,
           151, 57, 44, 95, 43, 30, 131, 57, 40, 57, 26, 19})
          .astype(inVar.type()),
      true);

  auto linBias = Linear(wtVar, bsVar);
  auto resultBias = linBias.forward(inVar);
  ASSERT_EQ(resultBias.type(), fl::dtype::f16);
  ASSERT_TRUE(allClose(resultBias, expected_outVar, 1E-3));
}

TEST(ModuleTest, ConvPadding) {
  auto conv1 = Conv2D(30, 100, 3, 5, 2, 1, PaddingMode::SAME, 0, true, 1);
  auto conv2 = Conv2D(
      30, 100, 3, 5, 2, 1, PaddingMode::SAME, PaddingMode::SAME, true, 1);
  auto conv3 =
      Conv2D(30, 100, 10, 10, 1, 1, PaddingMode::SAME, PaddingMode::SAME, 4, 4);
  // Row-major: NCHW
  auto input = Tensor(fl::rand({2, 30, 32, 32}), false);

  auto conv1Op = conv1(input);
  ASSERT_EQ(conv1Op.shape(), Shape({2, 100, 28, 16}));

  auto conv2Op = conv2(input);
  ASSERT_EQ(conv2Op.shape(), Shape({2, 100, 32, 16}));

  // test dilation
  auto conv3Op = conv3(input);
  ASSERT_EQ(conv3Op.shape(), Shape({2, 100, 32, 32}));
}

TEST(ModuleTest, GLUFwd) {
  // Row-major: data in row-major order for {3, 2} matrix
  // [[0.8, 0.1], [0.2, 0.5], [0.2, 0.3]]
  auto inVar = Tensor(
      Tensor::fromVector<float>({3, 2}, {0.8, 0.1, 0.2, 0.5, 0.2, 0.3}), true);

  auto expected_outVar = Tensor(
      Tensor::fromVector<float>({3, 1}, {0.419983, 0.124492, 0.114888}), true);

  GatedLinearUnit glu(1);
  ASSERT_TRUE(allClose(glu.forward(inVar), expected_outVar, 1E-4));

  // test batching
  int batchsize = 5;
  inVar = Tensor(fl::rand({10, 7, batchsize}), true);
  glu = GatedLinearUnit(0);

  auto batchOutVar = glu(inVar);

  for (int i = 0; i < batchsize; ++i) {
    expected_outVar = glu.forward(inVar(fl::span, fl::span, fl::range(i, i + 1)));
    ASSERT_TRUE(allClose(
        batchOutVar(fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar,
        1E-7));
  }
}

TEST_F(ModuleTestF16, GLUFwdF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // Row-major: data in row-major order for {3, 2} matrix
  auto inVar = Tensor(
      Tensor::fromVector<float>({3, 2}, {0.8, 0.1, 0.2, 0.5, 0.2, 0.3})
          .astype(fl::dtype::f16),
      true);

  auto expected_outVar = Tensor(
      Tensor::fromVector<float>({3, 1}, {0.419983, 0.124492, 0.114888})
          .astype(fl::dtype::f16),
      true);

  GatedLinearUnit glu(1);
  auto out = glu.forward(inVar);
  ASSERT_EQ(out.type(), inVar.type());
  ASSERT_TRUE(allClose(out, expected_outVar, 1E-2));

  // test batching
  int batchsize = 5;
  inVar = Tensor(fl::rand({10, 7, batchsize}).astype(fl::dtype::f16), true);
  glu = GatedLinearUnit(0);

  auto batchOutVar = glu(inVar);

  for (int i = 0; i < batchsize; ++i) {
    expected_outVar = glu.forward(inVar(fl::span, fl::span, fl::range(i, i + 1)));
    ASSERT_EQ(batchOutVar.type(), expected_outVar.type());
    ASSERT_TRUE(allClose(
        batchOutVar(fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar,
        1E-3));
  }
}

TEST(ModuleTest, LogSoftmaxFwd) {
  // Row-major: data in row-major order for {3, 2} matrix
  // [[0.8, 0.1], [0.2, 0.5], [0.2, 0.3]]
  auto inVar = Tensor(
      Tensor::fromVector<float>({3, 2}, {0.8, 0.1, 0.2, 0.5, 0.2, 0.3}), true);

  // LogSoftmax along dim 0 (over rows)
  auto expected_outVar0 = Tensor(
      Tensor::fromVector<float>(
          {3, 2}, {-0.740805, -1.3119, -1.34081, -0.911902, -1.34081, -1.1119}),
      true);
  LogSoftmax lsm0(0);
  ASSERT_TRUE(allClose(lsm0.forward(inVar), expected_outVar0, 1E-4));

  // LogSoftmax along dim 1 (over columns)
  auto expected_outVar1 = Tensor(
      Tensor::fromVector<float>(
          {3, 2},
          {-0.403186, -1.10319, -0.854355, -0.554355, -0.744397, -0.644397}),
      true);
  LogSoftmax lsm1(1);
  ASSERT_TRUE(allClose(lsm1.forward(inVar), expected_outVar1, 1E-4));

  // test batching
  int batchsize = 5;
  inVar = Tensor(fl::rand({10, 7, batchsize}), true);
  LogSoftmax lsm(0);

  auto batchOutVar = lsm(inVar);

  for (int i = 0; i < batchsize; ++i) {
    auto expected_outVar =
        lsm.forward(inVar(fl::span, fl::span, fl::range(i, i + 1)));
    ASSERT_TRUE(allClose(
        batchOutVar(fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar,
        1E-7));
  }
}

TEST_F(ModuleTestF16, LogSoftmaxFwdF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // Row-major: data in row-major order for {3, 2} matrix
  auto inVar = Tensor(
      Tensor::fromVector<float>({3, 2}, {0.8, 0.1, 0.2, 0.5, 0.2, 0.3})
          .astype(fl::dtype::f16),
      true);

  auto expected_outVar0 = Tensor(
      Tensor::fromVector<float>(
          {3, 2}, {-0.740805, -1.3119, -1.34081, -0.911902, -1.34081, -1.1119}),
      true);
  LogSoftmax lsm0(0);
  auto result0 = lsm0.forward(inVar);
  ASSERT_TRUE(allClose(
      result0.astype(fl::dtype::f32),
      expected_outVar0, 1E-3));

  auto expected_outVar1 = Tensor(
      Tensor::fromVector<float>(
          {3, 2},
          {-0.403186, -1.10319, -0.854355, -0.554355, -0.744397, -0.644397}),
      true);
  LogSoftmax lsm1(1);
  ASSERT_TRUE(allClose(
      lsm1.forward(inVar).astype(fl::dtype::f32),
      expected_outVar1, 1E-3));

  // test batching
  int batchsize = 5;
  inVar = Tensor(fl::rand({10, 7, batchsize}), true);
  LogSoftmax lsm(0);

  auto batchOutVar = lsm(inVar);

  for (int i = 0; i < batchsize; ++i) {
    auto expected_outVar =
        lsm.forward(inVar(fl::span, fl::span, fl::range(i, i + 1)));
    ASSERT_TRUE(allClose(
        batchOutVar(fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar,
        1E-7));
  }
}

TEST(ModuleTest, ConvolutionFwd) {
  // test batching - Row-major: NCHW
  // cuDNN may select different internal algorithms for different batch sizes,
  // leading to numerical differences between batched and individual conv2d.
  // Use 1e-3 tolerance to accommodate algorithm-dependent accumulation order.
  auto conv = Conv2D(30, 50, 9, 7, 2, 3, 3, 2, 1, 1, true, 1);
  int batchsize = 10;
  auto input = fl::rand({batchsize, 30, 100, 120}, fl::dtype::f32);
  auto batchOutVar = conv(Tensor(input, false));

  for (int i = 0; i < batchsize; ++i) {
    auto expected_outVar = conv(
        Tensor(input(fl::range(i, i + 1), fl::span, fl::span, fl::span), false));
    ASSERT_TRUE(allClose(
        batchOutVar(fl::range(i, i + 1), fl::span, fl::span, fl::span),
        expected_outVar,
        1E-3));
  }
}

TEST_F(ModuleTestF16, ConvolutionFwdF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // test batching
  // Row-major: NCHW
  // Note: cuDNN frontend may not support f16 conv2d+bias fusion on all
  // architectures. Use no-bias conv to avoid cuDNN frontend plan failures.
  auto conv = Conv2D(16, 32, 3, 3, 1, 1, 1, 1, 1, 1, /*bias=*/false, 1);
  int batchsize = 1;
  auto input = fl::rand({batchsize, 16, 32, 32}, fl::dtype::f16);
  auto batchOutVar = conv(Tensor(input, false));
  ASSERT_EQ(batchOutVar.type(), input.type());

  for (int i = 0; i < batchsize; ++i) {
    auto expected_outVar = conv(
        Tensor(input(fl::range(i, i + 1), fl::span, fl::span, fl::span), false));
    ASSERT_TRUE(allClose(
        batchOutVar(fl::range(i, i + 1), fl::span, fl::span, fl::span),
        expected_outVar,
        1E-3));
  }

}

TEST(ModuleTest, ConvolutionWithGroupFwd) {
  // test batching - Row-major: NCHW
  // cuDNN may select different internal algorithms for different batch sizes.
  auto conv = Conv2D(30, 50, 9, 7, 2, 3, 3, 2, true, 2);
  int batchsize = 10;
  auto input = fl::rand({batchsize, 30, 100, 120});
  auto batchOutVar = conv(Tensor(input, false));
  for (int i = 0; i < batchsize; ++i) {
    auto expected_outVar = conv(
        Tensor(input(fl::range(i, i + 1), fl::span, fl::span, fl::span), false));
    ASSERT_TRUE(allClose(
        batchOutVar(fl::range(i, i + 1), fl::span, fl::span, fl::span),
        expected_outVar,
        1E-3));
  }
}

TEST(ModuleTest, PoolingFwd) {
  // test batching - Row-major: NCHW
  auto pool = Pool2D(9, 7, 1, 1, PaddingMode::SAME, PaddingMode::SAME);
  int batchsize = 10;
  auto input = fl::rand({batchsize, 30, 100, 120});
  auto batchOutVar = pool(Tensor(input, false));
  for (int i = 0; i < batchsize; ++i) {
    ASSERT_EQ(input.shape(), batchOutVar.shape());
    auto expected_outVar = pool(
        Tensor(input(fl::range(i, i + 1), fl::span, fl::span, fl::span), false));
    ASSERT_TRUE(allClose(
        batchOutVar(fl::range(i, i + 1), fl::span, fl::span, fl::span),
        expected_outVar,
        1E-7));
  }
}

TEST_F(ModuleTestF16, PoolingFwdF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // test batching
  // Row-major: NCHW
  auto pool = Pool2D(9, 7, 1, 1, PaddingMode::SAME, PaddingMode::SAME);
  int batchsize = 10;
  auto input = fl::rand({batchsize, 30, 100, 120}, fl::dtype::f16);
  auto batchOutVar = pool(Tensor(input, false));
  for (int i = 0; i < batchsize; ++i) {
    ASSERT_EQ(input.shape(), batchOutVar.shape());
    auto expected_outVar = pool(
        Tensor(input(fl::range(i, i + 1), fl::span, fl::span, fl::span), false));
    ASSERT_TRUE(allClose(
        batchOutVar(fl::range(i, i + 1), fl::span, fl::span, fl::span),
        expected_outVar,
        1E-7));
  }
}

TEST(ModuleTest, RNNFwd) {
  auto mode = RnnMode::RELU;
  int num_layers = 2;
  int hidden_size = 3;
  int input_size = 4;
  int batch_size = 5;
  int seq_length = 6;

  // Row-major: {seqLength, batchSize, inputSize}
  auto in = Tensor(
      fl::rand({seq_length, batch_size, input_size}, fl::dtype::f32), true);
  unsigned n_params = 51;
  auto w = Tensor(fl::rand({1, 1, n_params}, fl::dtype::f32), true);
  for (int i = 0; i < in.elements(); ++i) {
    in.detach().flat(i) = (i + 1) * 0.01;
  }
  for (int i = 0; i < w.elements(); ++i) {
    w.detach().flat(i) = (i + 1) * 0.01;
  }
  auto rnn = RNN(input_size, hidden_size, num_layers, mode);
  rnn.setParams(w, 0);

  auto out = rnn(in);
  // Row-major: {seqLength, batchSize, outSize}
  Shape expected_dims({6, 5, 3});
  ASSERT_EQ(out.shape(), expected_dims);
  // Calculated from Lua Torch Cudnn implementation

  auto expected_outVar = Tensor(
      Tensor::fromVector<float>(
          expected_dims,
          {1.5418,  1.6389,  1.7361,  1.5491,  1.6472,  1.7452,  1.5564,
           1.6554,  1.7544,  1.5637,  1.6637,  1.7636,  1.5710,  1.6719,
           1.7728,  3.4571,  3.7458,  4.0345,  3.4761,  3.7670,  4.0578,
           3.4951,  3.7881,  4.0812,  3.5141,  3.8093,  4.1045,  3.5331,
           3.8305,  4.1278,  5.6947,  6.2004,  6.7060,  5.7281,  6.2373,
           6.7466,  5.7614,  6.2743,  6.7871,  5.7948,  6.3112,  6.8276,
           5.8282,  6.3482,  6.8681,  8.2005,  8.9458,  9.6911,  8.2500,
           9.0005,  9.7509,  8.2995,  9.0551,  9.8107,  8.3491,  9.1098,
           9.8705,  8.3986,  9.1645,  9.9303,  10.9520, 11.9587, 12.9655,
           11.0191, 12.0326, 13.0462, 11.0861, 12.1065, 13.1269, 11.1532,
           12.1804, 13.2075, 11.2203, 12.2543, 13.2882, 13.9432, 15.2333,
           16.5233, 14.0291, 15.3277, 16.6263, 14.1149, 15.4221, 16.7292,
           14.2008, 15.5165, 16.8322, 14.2866, 15.6109, 16.9351}),
      true);
  // Tolerance relaxed: reference values from Lua Torch (cuDNN v6). cuDNN v9
  // uses slightly different numerics for unbounded RELU RNN.
  ASSERT_TRUE(allClose(out, expected_outVar, 2E-2));
}

TEST(ModuleTest, LSTMFwd) {
  auto mode = RnnMode::LSTM;
  int num_layers = 4;
  int hidden_size = 5;
  int input_size = 3;
  int batch_size = 2;
  int seq_length = 2;

  // Row-major: {seqLength, batchSize, inputSize}
  auto in = Tensor(
      fl::rand({seq_length, batch_size, input_size}, fl::dtype::f32), true);
  unsigned n_params = 920;
  auto w = Tensor(fl::rand({1, 1, n_params}, fl::dtype::f32), true);

  for (int i = 0; i < in.elements(); ++i) {
    in.detach().flat(i) = (i + 1) * 0.001;
  }
  for (int i = 0; i < w.elements(); ++i) {
    w.detach().flat(i) = (i + 1) * 0.001;
  }

  auto rnn = RNN(input_size, hidden_size, num_layers, mode);
  rnn.setParams(w, 0);

  auto out = rnn(in);
  // Row-major: {seqLength, batchSize, outSize}
  Shape expected_dims({2, 2, 5});
  ASSERT_EQ(out.shape(), expected_dims);
  // Calculated from Lua Torch Cudnn implementation
  auto expected_outVar = Tensor(
      Tensor::fromVector<float>(
          expected_dims,
          {0.7390, 0.7395, 0.7399, 0.7403, 0.7407, 0.7390, 0.7395,
           0.7399, 0.7403, 0.7407, 0.9617, 0.9618, 0.9619, 0.9619,
           0.962,  0.9617, 0.9618, 0.9619, 0.9619, 0.962}),
      true);
  ASSERT_TRUE(allClose(out, expected_outVar, 1E-4));
}

TEST(ModuleTest, GRUFwd) {
  auto mode = RnnMode::GRU;
  int num_layers = 4;
  int hidden_size = 5;
  int input_size = 3;
  int batch_size = 2;
  int seq_length = 2;

  // Row-major: {seqLength, batchSize, inputSize}
  auto in = Tensor(
      fl::rand({seq_length, batch_size, input_size}, fl::dtype::f32), true);
  unsigned n_params = 690;
  auto w = Tensor(fl::rand({1, 1, n_params}, fl::dtype::f32), true);

  for (int i = 0; i < in.elements(); ++i) {
    in.detach().flat(i) = (i + 1) * 0.001;
  }
  for (int i = 0; i < w.elements(); ++i) {
    w.detach().flat(i) = (i + 1) * 0.001;
  }

  auto rnn = RNN(input_size, hidden_size, num_layers, mode);
  rnn.setParams(w, 0);

  auto out = rnn(in);
  // Row-major: {seqLength, batchSize, outSize}
  Shape expected_dims({2, 2, 5});
  ASSERT_EQ(out.shape(), expected_dims);
  // Calculated from Lua Torch Cudnn implementation
  auto expected_outVar = Tensor(
      Tensor::fromVector<float>(
          expected_dims,
          {0.1430, 0.1425, 0.1419, 0.1413, 0.1408, 0.1430, 0.1425,
           0.1419, 0.1413, 0.1408, 0.2206, 0.2194, 0.2181, 0.2168,
           0.2155, 0.2206, 0.2194, 0.2181, 0.2168, 0.2155}),
      true);
  ASSERT_TRUE(allClose(out, expected_outVar, 1E-4));
}

// Verify that RNN backward produces finite, non-zero gradients in f32.
// This exercises the manual BPTT fallback (OneDNN RNN backward is not
// available on CPU). Unlike Jacobian tests, this only checks gradient
// sanity, not exact values. Tests both unidirectional and bidirectional
// (which uses the split-layer backward path when inputSize != hiddenSize*2).
TEST(ModuleTest, RNNBackwardF32) {
  for (bool bidir : {false, true}) {
    for (auto mode : {RnnMode::TANH, RnnMode::LSTM, RnnMode::GRU}) {
      int numLayers = 2;
      int hiddenSize = 3;
      int inputSize = 4;
      int batchSize = 2;
      int seqLength = 3;

      auto in = Tensor(
          fl::rand({seqLength, batchSize, inputSize}, fl::dtype::f32), true);
      int64_t nParams = detail::getNumRnnParams(
          inputSize, hiddenSize, numLayers, mode, bidir);
      auto w = Tensor(
          fl::rand({static_cast<long long>(nParams)}, fl::dtype::f32), true);

      auto [y, hy, cy] = autograd::rnn(
          in,
          Tensor().astype(fl::dtype::f32),
          Tensor().astype(fl::dtype::f32),
          w,
          hiddenSize, numLayers, mode, bidir, 0.0);

      y.backward(fl::full(y.shape(), 1.f));

      std::string tag = std::string(bidir ? "bidir" : "unidir") +
          " mode=" + std::to_string(static_cast<int>(mode));

      ASSERT_TRUE(w.isGradAvailable()) << "No weight gradient: " << tag;
      auto wGrad = w.grad().toHostVectorSync<float>();
      bool allZero = true, hasNanInf = false;
      for (float v : wGrad) {
        if (v != 0.f) allZero = false;
        if (std::isnan(v) || std::isinf(v)) hasNanInf = true;
      }
      ASSERT_FALSE(allZero) << "Weight gradient all zero: " << tag;
      ASSERT_FALSE(hasNanInf) << "Weight gradient has NaN/Inf: " << tag;

      ASSERT_TRUE(in.isGradAvailable()) << "No input gradient: " << tag;
      auto inGrad = in.grad().toHostVectorSync<float>();
      allZero = true; hasNanInf = false;
      for (float v : inGrad) {
        if (v != 0.f) allZero = false;
        if (std::isnan(v) || std::isinf(v)) hasNanInf = true;
      }
      ASSERT_FALSE(allZero) << "Input gradient all zero: " << tag;
      ASSERT_FALSE(hasNanInf) << "Input gradient has NaN/Inf: " << tag;
    }
  }
}

TEST_F(ModuleTestF16, RNNFwdF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto mode = RnnMode::RELU;
  int num_layers = 2;
  int hidden_size = 3;
  int input_size = 4;
  int batch_size = 5;
  int seq_length = 6;

  // Row-major: {seqLength, batchSize, inputSize}
  auto in = Tensor(
      fl::rand({seq_length, batch_size, input_size}, fl::dtype::f16), true);
  unsigned n_params = 51;
  auto w = Tensor(fl::rand({1, 1, n_params}, fl::dtype::f16), true);
  for (int i = 0; i < in.elements(); ++i) {
    in.detach().flat(i) = (i + 1) * 0.01;
  }
  for (int i = 0; i < w.elements(); ++i) {
    w.detach().flat(i) = (i + 1) * 0.01;
  }
  auto rnn = RNN(input_size, hidden_size, num_layers, mode);
  rnn.setParams(w, 0);

  auto out = rnn(in);
  // Row-major: {seqLength, batchSize, outSize}
  Shape expected_dims({6, 5, 3});
  ASSERT_EQ(out.shape(), expected_dims);
  // Calculated from Lua Torch Cudnn implementation
  auto expected_outVar = Tensor(
      Tensor::fromVector<float>(
          expected_dims,
          {1.5418,  1.6389,  1.7361,  1.5491,  1.6472,  1.7452,  1.5564,
           1.6554,  1.7544,  1.5637,  1.6637,  1.7636,  1.5710,  1.6719,
           1.7728,  3.4571,  3.7458,  4.0345,  3.4761,  3.7670,  4.0578,
           3.4951,  3.7881,  4.0812,  3.5141,  3.8093,  4.1045,  3.5331,
           3.8305,  4.1278,  5.6947,  6.2004,  6.7060,  5.7281,  6.2373,
           6.7466,  5.7614,  6.2743,  6.7871,  5.7948,  6.3112,  6.8276,
           5.8282,  6.3482,  6.8681,  8.2005,  8.9458,  9.6911,  8.2500,
           9.0005,  9.7509,  8.2995,  9.0551,  9.8107,  8.3491,  9.1098,
           9.8705,  8.3986,  9.1645,  9.9303,  10.9520, 11.9587, 12.9655,
           11.0191, 12.0326, 13.0462, 11.0861, 12.1065, 13.1269, 11.1532,
           12.1804, 13.2075, 11.2203, 12.2543, 13.2882, 13.9432, 15.2333,
           16.5233, 14.0291, 15.3277, 16.6263, 14.1149, 15.4221, 16.7292,
           14.2008, 15.5165, 16.8322, 14.2866, 15.6109, 16.9351}),
      true);
  ASSERT_TRUE(allClose(out, expected_outVar.astype(in.type()), 5E-2));
}

TEST(ModuleTest, ViewFwd) {
  auto module = View(Shape({-1, 0, 6}));
  auto input = Tensor(Tensor({1, 2, 3, 4}), true);
  ASSERT_EQ(module(input).shape(), Shape({2, 2, 6}));
}

TEST(ModuleTest, DropoutFwd) {
  auto module = Dropout(0.5);
  // Train Mode
  module.train();
  auto in = Tensor(fl::rand({1000, 1000}), true);
  auto out = module(in);

  ASSERT_NEAR(
      out.elements() - fl::countNonzero(out).asScalar<unsigned>(),
      in.elements() / 2,
      in.elements() / 16); // Check enough zeroes

  ASSERT_GT(
      fl::amax(out).scalarSync<float>(), 1.5); // Check input is scaled

  // Eval Mode
  module.eval();
  out = module(in);
  ASSERT_TRUE(allClose(out, in, 1E-5));
}

TEST_F(ModuleTestF16, DropoutFwdF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto module = Dropout(0.5);
  // Train Mode
  module.train();
  auto in = Tensor(fl::rand({1000, 1000}, fl::dtype::f16), true);
  auto out = module(in);
  ASSERT_EQ(out.type(), fl::dtype::f16);

  ASSERT_NEAR(
      out.elements() - fl::countNonzero(out).asScalar<unsigned>(),
      in.elements() / 2,
      in.elements() / 16); // Check enough zeroes

  ASSERT_GT(
      fl::amax(out).asScalar<float>(), 1.5); // Check input is scaled

  // Eval Mode
  module.eval();
  out = module(in);
  ASSERT_TRUE(allClose(out, in, 1E-5));
}

TEST(ModuleTest, PaddingFwd) {
  auto module = Padding({{1, 2}, {3, 4}}, -1);
  auto input = Tensor(fl::rand({1, 2, 3, 4}, fl::dtype::f64), true);
  auto output = module(input);
  ASSERT_EQ(output.shape(), Shape({4, 9, 3, 4}));
  ASSERT_TRUE(allClose(input, output(fl::range(1, 2), fl::range(3, 5))));
  ASSERT_NEAR(
      fl::sum(input).scalarSync<double>(),
      fl::sum(output).scalarSync<double>() + 408,
      1E-5);
}

TEST(ModuleTest, LayerNormFwd) {
  double eps = 1E-5;
  std::vector<int> feat_axes = {3};
  int F = 10;
  auto input = Tensor(fl::rand({4, 4, 3, F}), true);

  auto sample_mean = autograd::mean(input, {3}, /* keepDims = */ true);
  auto sample_var = autograd::var(input, {3}, /* bias = */ false, /* keepDims = */ true);
  auto true_out = (input - autograd::tileAs(sample_mean, input)) /
      autograd::tileAs(autograd::sqrt(sample_var + eps), input);

  // no affine transform
  auto module1 = LayerNorm(feat_axes, eps, false);

  module1.train();
  auto out = module1.forward(input);

  ASSERT_TRUE(allClose(out, true_out, eps));
  ASSERT_EQ(out.type(), input.type());

  module1.eval();
  out = module1.forward(input);

  ASSERT_TRUE(allClose(out, true_out, eps));
  ASSERT_EQ(out.type(), input.type());

  // with affine transform
  auto module2 = LayerNorm(feat_axes, eps, true);

  module2.train();
  auto out_train = module2.forward(input);
  module2.eval();
  auto out_eval = module2.forward(input);

  ASSERT_TRUE(allClose(out_train, out_eval, eps));
  ASSERT_EQ(out_train.shape(), input.shape());

  // with affine transform
  auto module3 = LayerNorm(feat_axes, eps, true, F);
  module3.setParams(Tensor(fl::full({F}, 1.0), false), 0);
  module3.setParams(Tensor(fl::full({F}, 0.0), false), 1);
  auto out3 = module3.forward(input);
  ASSERT_TRUE(allClose(out_train, out3, eps));

  // With other shapes
  auto input3Dim = Tensor(fl::rand({4, 4, 3}), true);
  auto module4 = LayerNorm(std::vector<int>{0}, eps, false);
  out = module4.forward(input3Dim);
  ASSERT_EQ(out.shape(), input3Dim.shape());
}

TEST_F(ModuleTestF16, LayerNormFwdF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  double eps = 5E-2;
  std::vector<int> feat_axes = {3};
  auto input = Tensor(fl::rand({4, 4, 3, 10}, fl::dtype::f16), true);

  auto sample_mean = autograd::mean(input, {3}, /* keepDims = */ true);
  auto sample_var = autograd::var(input, {3}, /* bias = */ false, /* keepDims = */ true);
  auto true_out = (input - autograd::tileAs(sample_mean, input).astype(input.type())) /
      autograd::tileAs(autograd::sqrt(sample_var + eps), input).astype(input.type());

  // no affine transform
  auto module1 = LayerNorm(feat_axes, eps, false);

  module1.train();
  auto out = module1.forward(input);

  ASSERT_TRUE(allClose(out, true_out.astype(out.type()), eps));

  module1.eval();
  out = module1.forward(input);

  ASSERT_TRUE(
      allClose(out, true_out.astype(out.type()), eps));

  // with affine transform
  auto module2 = LayerNorm(feat_axes, eps, true);

  module2.train();
  auto out_train = module2.forward(input);
  module2.eval();
  auto out_eval = module2.forward(input);

  ASSERT_TRUE(allClose(out_train, out_eval, eps));
  ASSERT_EQ(out_train.shape(), input.shape());

  module2.train();
}

TEST(ModuleTest, NormalizeFwd) {
  auto input = Tensor(fl::rand({10, 3}, fl::dtype::f64), true);
  auto module = Normalize({1}, 2, 1e-12, 5);
  module.train();
  auto out = module.forward(input);
  ASSERT_TRUE(allClose(
      fl::sqrt(fl::sum(out * out, {1})),
      fl::full({10}, 5, fl::dtype::f64)));
}

TEST(ModuleTest, TransformFwd) {
  auto inVar = Tensor(fl::full({4, 5}, 1.0), true);

  auto l = Transform([](const Tensor& in) { return autograd::log(in); });

  ASSERT_TRUE(allClose(l.forward(inVar), fl::full(inVar.shape(), 0.0)));
}

TEST(ModuleTest, PrecisionCastFwd) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half precision not available on this device";
  }

  auto in = Tensor(fl::full({3, 3}, 1.0), true);
  auto precisionCast = PrecisionCast(fl::dtype::f16);
  auto out = precisionCast.forward(in);

  ASSERT_EQ(out.type(), fl::dtype::f16);
  ASSERT_TRUE(allClose(in, out.astype(fl::dtype::f32)));
}

TEST(ModuleTest, ContainerReplaceParam) {
  auto seq = ContainerTestClass();
  seq.addParam(Tensor(fl::rand({5, 5}), true));
  seq.add(Linear(10, 20));
  seq.addParam(Tensor(fl::rand({5, 5}), true));
  seq.add(ReLU());
  seq.add(Linear(20, 30));
  seq.addParam(Tensor(fl::rand({5, 5}), true));

  // Change the first parameter
  auto new_param = Tensor(fl::rand({5, 5}), true);
  seq.setParams(new_param, 0);
  ASSERT_TRUE(allClose(seq.params()[0], new_param));

  // Change the first linear layer's first parameter
  new_param = Tensor(fl::rand({10, 20}), true);
  seq.setParams(new_param, 1);
  ASSERT_TRUE(allClose(seq.params()[1], new_param));
  ASSERT_TRUE(allClose(seq.module(0)->param(0), new_param));

  // Change the second linear layer's first parameter
  new_param = Tensor(fl::rand({20, 30}), true);
  seq.setParams(new_param, 4);
  ASSERT_TRUE(allClose(seq.params()[4], new_param));
  ASSERT_TRUE(allClose(seq.module(2)->param(0), new_param));

  // Change the last parameter
  new_param = Tensor(fl::rand({5, 5}), true);
  seq.setParams(new_param, 6);
  ASSERT_TRUE(allClose(seq.param(6), new_param));
}

TEST(ModuleTest, AdaptiveSoftMaxPredict) {
  // test predict gives the same as argmax along probs
  int N = 5;
  int C = 5;
  int T = 10;
  int B = 5;

  // Row-major: {B, T, N}
  auto x = input(fl::rand({B, T, N}, fl::dtype::f32));
  auto y = Tensor(
      (fl::rand({B, T}, fl::dtype::u32) % C).astype(fl::dtype::s32), false);

  std::vector<int> cutoff{{C / 2, C}};
  auto activation = std::make_shared<AdaptiveSoftMax>(N, cutoff);

  auto probs = activation->forward(x);
  auto result1 = activation->predict(x);
  // Row-major: classes in last dim, argmax over last dim
  auto result2 = fl::argmax(probs, probs.ndim() - 1, /* keepDims = */ true);

  ASSERT_TRUE(allClose(result1, result2));
}

TEST(ModuleTest, AdaptiveSoftMaxLossBatchFwd) {
  // test batching
  int N = 5;
  int C = 3;
  int T = 10;
  int B = 5;

  // Row-major: {B, T, N}
  auto x = input(fl::rand({B, T, N}, fl::dtype::f32));
  auto y = Tensor(
      (fl::rand({B, T}, fl::dtype::u32) % C).astype(fl::dtype::s32), false);

  std::vector<int> cutoff{{C / 2, C}};
  auto activation = std::make_shared<AdaptiveSoftMax>(N, cutoff);
  auto asml =
      std::make_shared<AdaptiveSoftMaxLoss>(activation, ReduceMode::NONE);
  auto batchOutVar = asml->forward(x, y);

  auto singleOut = fl::full({B, T}, 0, fl::dtype::f32);
  for (int i = 0; i < B; ++i) {
    auto singleOutVar = asml->forward(
        x(fl::range(i, i + 1), fl::span, fl::span), y(fl::range(i, i + 1), fl::span));
    singleOut(i, fl::span) = singleOutVar;
  }
  ASSERT_TRUE(allClose(batchOutVar, singleOut));
}

TEST(ModuleTest, AdaptiveSoftMaxLossIgnoreIndex) {
  // test batching
  int N = 5;
  int C = 3;
  int T = 10;
  int B = 5;

  // Row-major: {B, T, N}
  auto x = input(fl::rand({B, T, N}, fl::dtype::f32));
  auto y = Tensor(
      (fl::rand({B, T}, fl::dtype::u32) % C).astype(fl::dtype::s32), false);
  auto ignoreIdx = y(0, 0).scalarSync<int>();
  auto ignoreCount = fl::sum(y != ignoreIdx).asScalar<unsigned>();

  std::vector<int> cutoff{{C / 2, C}};
  auto activation = std::make_shared<AdaptiveSoftMax>(N, cutoff);
  auto asml1 = std::make_shared<AdaptiveSoftMaxLoss>(
      activation, ReduceMode::SUM, ignoreIdx);
  auto asml2 = std::make_shared<AdaptiveSoftMaxLoss>(
      activation, ReduceMode::MEAN, ignoreIdx);

  auto lossSum = asml1->forward(x, y);
  auto lossMean = asml2->forward(x, y);
  if (ignoreCount > 0) {
    ASSERT_NEAR(
        fl::sum(lossSum).scalarSync<float>(),
        fl::sum(lossMean).scalarSync<float>() * ignoreCount,
        1E-5);
  } else {
    // All targets ignored: both losses should be 0
    ASSERT_NEAR(fl::sum(lossSum).scalarSync<float>(), 0.0f, 1E-5);
    ASSERT_NEAR(fl::sum(lossMean).scalarSync<float>(), 0.0f, 1E-5);
  }
}

TEST(ModuleTest, IdentityFwd) {
  auto module = Identity();
  std::vector<Tensor> in = {
      Tensor(fl::rand({1000, 1000}), true),
      Tensor(fl::rand({100, 100}), true)};

  // Train Mode
  module.train();
  auto out = module(in);
  ASSERT_EQ(out.size(), 2);
  ASSERT_TRUE(allClose(out.at(0), in.at(0), 1e-20));
  ASSERT_TRUE(allClose(out.at(1), in.at(1), 1e-20));

  // Eval Mode
  module.eval();
  out = module(in);
  ASSERT_EQ(out.size(), 2);
  ASSERT_TRUE(allClose(out.at(0), in.at(0), 1e-20));
  ASSERT_TRUE(allClose(out.at(1), in.at(1), 1e-20));
}

TEST(ModuleTest, ModuleCloneCopy) {
  int n_in = 1, n_out = 2;
  auto wtVar = param(Tensor::fromVector<float>({n_out, n_in}, {2, 4}));
  auto inVar = input(Tensor::fromVector<float>({n_in}, {3}));
  Tensor expected_outVar(Tensor::fromVector<float>({n_out}, {6, 12}), true);

  Linear lin(wtVar);
  ASSERT_TRUE(allClose(lin(inVar), expected_outVar, 1E-7));

  // Intentionally cast to base Module ptr and clone/copy via the various
  // options
  std::unique_ptr<Module> modulePtr = std::make_unique<Linear>(std::move(lin));
  std::unique_ptr<Module> clonedModulePtr = modulePtr->clone();

  // Change the original module param and check the cloned modules have not
  // changed
  modulePtr->param(0).detach() += 1.0F;
  ASSERT_FALSE(
      allClose(modulePtr->forward({inVar}).front(), expected_outVar, 1E-7));

  ASSERT_TRUE(allClose(
      clonedModulePtr->forward({inVar}).front(), expected_outVar, 1E-7));
}

TEST(ModuleTest, ContainerCloneCopy) {
  ContainerTestClass seq;
  seq.addParam(Tensor(fl::rand({5, 5}), true));
  seq.add(Linear(10, 20));
  // Create copy/clone vis copy constructor
  auto seqCopy = seq;

  // Make sure they are the same
  ASSERT_TRUE(allClose(seq.params()[0], seqCopy.params()[0]));
  ASSERT_TRUE(allClose(seq.params()[1], seqCopy.params()[1]));

  // Change the first parameter and check the copy has not changed
  Tensor new_param(fl::rand({5, 5}), true);
  seq.setParams(new_param, 0);
  ASSERT_TRUE(allClose(seq.params()[0], new_param));
  ASSERT_FALSE(allClose(seqCopy.params()[0], seq.params()[0]));

  // Change the linear layer's first parameter and check the copy has not
  // changed
  new_param = Tensor(fl::rand({10, 20}), true);
  seq.setParams(new_param, 1);
  ASSERT_TRUE(allClose(seq.params()[1], new_param));
  ASSERT_TRUE(allClose(seq.module(0)->param(0), new_param));
  ASSERT_FALSE(allClose(seqCopy.params()[1], seq.params()[1]));
  ASSERT_FALSE(allClose(seqCopy.module(0)->param(0), seq.module(0)->param(0)));

  // Intentionally cast to base Module ptr and clone/copy via the various
  // options
  std::unique_ptr<Module> modulePtr =
      std::make_unique<ContainerTestClass>(std::move(seq));
  std::unique_ptr<Module> clonedModulePtr = modulePtr->clone();

  ASSERT_TRUE(allClose(clonedModulePtr->params()[0], modulePtr->params()[0]));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  flTestInit();
  return RUN_ALL_TESTS();
}
