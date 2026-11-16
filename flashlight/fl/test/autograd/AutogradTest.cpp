/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/common.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/test/TestInit.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/test/autograd/AutogradTestUtils.h"

using namespace fl;

using fl::detail::AutogradTestF16;

TEST(AutogradTest, OperatorParenthesis) {
  auto x = Tensor(fl::rand({1, 3, 3}, fl::dtype::f64), true);
  auto y = x(0, 0) + x(0, 1);
  auto funcOperatorParen = [](Tensor& in) { return in(0, 0) + in(0, 1); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcOperatorParen, x));
}

TEST(AutogradTest, AutogradOperatorTypeCompatibility) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto f16 = Tensor(fl::rand({2, 2}, fl::dtype::f16), true);
  auto f32 = Tensor(fl::rand({2, 2}, fl::dtype::f32), true);

  // Tensor-level operators (+, -, *, /, comparisons) delegate to the backend
  // which may handle mixed types via implicit casting. These no longer throw.

  // Autograd-specific functions still enforce type matching
  EXPECT_THROW({ autograd::maximum(f16, f32); }, std::invalid_argument); // maximum
  EXPECT_THROW({ autograd::minimum(f16, f32); }, std::invalid_argument); // minimum
  EXPECT_THROW({ autograd::matmul(f16, f32); }, std::invalid_argument); // matmul
  EXPECT_THROW({ autograd::matmul(f16, f32, MatrixProperty::Transpose); }, std::invalid_argument); // matmul(TN)
  EXPECT_THROW({ autograd::matmul(f16, f32, MatrixProperty::None, MatrixProperty::Transpose); }, std::invalid_argument); // matmul(NT)
  EXPECT_NO_THROW({ autograd::binaryCrossEntropy(f16, f32); });
  EXPECT_NO_THROW({
    autograd::categoricalCrossEntropy(
        Tensor(fl::rand({10, 4, 7}, fl::dtype::f16), true),
        Tensor(
            (fl::rand({10, 4}, fl::dtype::u32) % 7).astype(fl::dtype::s32),
            false));
  });
  EXPECT_NO_THROW({ autograd::pool2d(f16, 1, 1, 1, 1, 1, 1); });
  EXPECT_NO_THROW({ autograd::embedding(f16, f32); }); // lookup is of a different type
  // Ternary operators
  auto f32_2 = Tensor(fl::rand({2, 2}, fl::dtype::f32), true);
  auto f16_2 = Tensor(fl::rand({2, 2}, fl::dtype::f16), true);
  EXPECT_THROW({ autograd::linear(f16, f32, f16_2); }, std::invalid_argument); // linear
  EXPECT_THROW({ autograd::linear(f16, f32, f32_2); }, std::invalid_argument); // linear
  auto w = Tensor(fl::rand({1}, fl::dtype::f32), true);
  auto b = Tensor(fl::rand({1}, fl::dtype::f32), true);
#ifdef FL_TEST_BACKEND_CPU
  // OneDNN batchnorm doesn't support f16 inputs
  EXPECT_THROW(
      { autograd::batchnorm(f16, f32, f32_2, w, b, {1}, true, 0.01, 0.01); },
      std::invalid_argument);
  EXPECT_THROW(
      { autograd::batchnorm(f16, f32, f16_2, w, b, {1}, true, 0.01, 0.01); },
      std::invalid_argument);
#else
  // cuDNN batchnorm accepts f16 input with f32 params
  EXPECT_NO_THROW(
      { autograd::batchnorm(f16, f32, f32_2, w, b, {1}, true, 0.01, 0.01); });
#endif
  EXPECT_THROW(
      { autograd::conv2d(f16, f32, f16_2, 1, 1, 0, 0, 1, 1); }, std::invalid_argument);
  // Quaternary
  auto f16_3 = Tensor(fl::rand({2, 2, 3}, fl::dtype::f16), false);
  auto f16_4 = Tensor(fl::rand({50}, fl::dtype::f16), false);
  EXPECT_THROW(
      {
        autograd::rnn(f16_3,
            Tensor(Tensor(fl::dtype::f32), false),
            Tensor(Tensor(fl::dtype::f32), false),
            f16_4,
            2,
            2,
            RnnMode::LSTM,
            true,
            0.0);
      },
      std::invalid_argument);
  // Variadic operators
  std::vector<Tensor> concatInputs = {f16, f32, f16_2, f32_2};
  EXPECT_THROW({ autograd::concatenate(concatInputs, 0); }, std::invalid_argument);
}

TEST(AutogradTest, CastingAsDifferentGradTypes) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto f32 = Tensor(fl::rand({5, 5}), true);
  auto f16 = Tensor(fl::rand({5, 5}, fl::dtype::f16), true);
  // Tensor-level operators delegate to the backend which handles mixed types
  // via type promotion -- no longer throws
  EXPECT_NO_THROW({ f32 + f16; });
}

TEST(AutogradTest, CastingAs) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto var = Tensor(fl::rand({5, 5}), true);
  auto varF16 = var.astype(fl::dtype::f16);
  ASSERT_EQ(var.type(), fl::dtype::f32);
  ASSERT_EQ(varF16.type(), fl::dtype::f16);
  ASSERT_TRUE(allClose(varF16, var.astype(fl::dtype::f16)));
}

TEST(AutogradTest, CastingAsBackward) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto a = Tensor(fl::rand({4, 4}, fl::dtype::f16), true);
  auto b = Tensor(fl::rand({4, 4}, fl::dtype::f16), false);
  auto c = b + a;
  c.backward();
  ASSERT_EQ(a.grad().type(), fl::dtype::f16);
  ASSERT_EQ(a.grad().type(), fl::dtype::f16);
  a = a.astype(fl::dtype::f32);
  ASSERT_FALSE(a.isGradAvailable());
}

TEST(AutogradTest, CastingAsGrad) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // compare to f32 case
  auto x = Tensor(fl::full({5}, 2.0), true);
  auto y = Tensor(fl::full({5}, 3.0), true);
  auto z = x * x + x * y + y * y;
  auto dz = Tensor(fl::full({5}, 1.0), false);
  z.backward(dz);
  auto dx = x.grad();
  auto dy = y.grad();

  // f16 -- cast gradients in both directions
  auto x32 = Tensor(fl::full({5}, 2.0), true);
  auto y32 = Tensor(fl::full({5}, 3.0), true);
  auto xf16 = x32.astype(fl::dtype::f16);
  auto yf16 = y32.astype(fl::dtype::f16);
  auto zf16 = xf16 * xf16 + xf16 * yf16 + yf16 * yf16;
  auto zf32 = zf16.astype(fl::dtype::f32);
  zf32.backward(dz);

  ASSERT_EQ(xf16.grad().type(), fl::dtype::f16);
  ASSERT_EQ(yf16.grad().type(), fl::dtype::f16);
  ASSERT_EQ(zf16.grad().type(), fl::dtype::f16);
  ASSERT_EQ(x32.grad().type(), fl::dtype::f32);
  ASSERT_EQ(y32.grad().type(), fl::dtype::f32);
  ASSERT_TRUE(
      allClose(dx, xf16.grad().astype(fl::dtype::f32)));
  ASSERT_TRUE(
      allClose(dy, y32.grad().astype(fl::dtype::f32)));
  ASSERT_TRUE(allClose(dx, x32.grad()));
  ASSERT_TRUE(allClose(dy, y32.grad()));
}

TEST(AutogradTest, NoCalcGrad) {
  auto x = Tensor(fl::rand({5}), false);
  auto y = Tensor(fl::rand({5}), true);
  auto z = x * x + x * y + y * y;
  auto dz = Tensor(fl::full({5}, 1.0), false);
  z.backward(dz);
  auto dy = y.grad();
  ASSERT_TRUE(allClose(dy, 2 * y + x));
  ASSERT_THROW(x.grad(), std::logic_error);
}

TEST(AutogradTest, Concatenate) {
  auto x1 = Tensor(fl::rand({2, 3, 1, 2}, fl::dtype::f64), true);
  auto x2 = Tensor(fl::rand({2, 3, 3, 2}, fl::dtype::f64), true);
  auto x3 = Tensor(fl::rand({2, 3, 1, 2}, fl::dtype::f64), true);
  auto x4 = Tensor(fl::rand({2, 3, 7, 2}, fl::dtype::f64), true);
  std::vector<Tensor> inputs = {x1, x2, x3, x4};
  auto output = autograd::concatenate(inputs, 2);

  ASSERT_EQ(output.shape(), Shape({2, 3, 12, 2}));

  auto funcConcatenateT1 = [x2, x3, x4](Tensor& in) {
    return autograd::concatenate({in, x2, x3, x4}, 2);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConcatenateT1, x1));

  auto funcConcatenateT2 = [x1, x2, x4](Tensor& in) {
    return autograd::concatenate({x1, x2, in, x4}, 2);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConcatenateT2, x3));
}

TEST(AutogradTest, Split) {
  // check output
  auto x = Tensor(fl::arange({7, 2}), true);
  auto yVec = autograd::split(x, 1, 0);
  ASSERT_EQ(yVec.size(), 7);
  ASSERT_EQ(yVec[0].shape(), Shape({1, 2}));
  ASSERT_EQ(yVec[2].shape(), Shape({1, 2}));
  ASSERT_TRUE(fl::all(yVec[6] == 6).scalarSync<char>());

  auto a = Tensor(fl::arange({5, 3}, 1), true);
  auto bVec = autograd::split(a, {2, 1}, 1);
  ASSERT_EQ(bVec.size(), 2);
  ASSERT_EQ(bVec[0].shape(), Shape({5, 2}));
  ASSERT_EQ(bVec[1].shape(), Shape({5, 1}));
  ASSERT_TRUE(
      fl::all(bVec[0] == fl::arange({5, 2}, 1)).scalarSync<char>());
  ASSERT_TRUE(fl::all(bVec[1] == 2).scalarSync<char>());

  // check exception handling
  ASSERT_THROW(autograd::split(a, {2, 2}, 0), std::invalid_argument);

  // check gradient
  auto gradFunc = [](Tensor& in) { return autograd::split(in, 2, 1)[0]; };
  auto input = Tensor(fl::rand({2, 3}, fl::dtype::f64), true);
  ASSERT_TRUE(fl::detail::jacobianTestImpl(gradFunc, input));
}

TEST(AutogradTest, Tile) {
  // Right-align: x{3} broadcasts to {6, 3} by tiling dim 0
  auto x = Tensor(fl::rand({3}), true);
  auto y = Tensor(fl::rand({6, 3}), true);
  auto z = y * autograd::tile(x, {6, 1});
  auto dz = Tensor(fl::full({6, 3}, 1.0), false);
  z.backward(dz);
  auto dy = y.grad();
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dy, fl::tile(x, {6, 1})));
  ASSERT_TRUE(allClose(dx, fl::sum(y, {0})));

  // Jacobian
  auto input = Tensor(fl::rand({10, 1, 5}), true);
  auto funcTile = [](Tensor& in) { return autograd::tile(in, {1, 2, 1}); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcTile, input, 1E-4, 1E-3));
}

TEST(AutogradTest, TileAs) {
  // Right-align: x{2} broadcasts to {5, 2} by tiling dim 0
  auto x = Tensor(fl::rand({2}), true);
  auto y = Tensor(fl::rand({5, 2}), true);
  auto z = y * autograd::tileAs(x, y);
  auto dz = Tensor(fl::full({5, 2}, 1.0), false);
  z.backward(dz);
  auto dy = y.grad();
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dy, fl::tile(x, {5, 1})));
  ASSERT_TRUE(allClose(dx, fl::sum(y, {0})));
}

TEST_F(AutogradTestF16, TileAsF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // Right-align: x{2} broadcasts to {5, 2} by tiling dim 0
  auto x = Tensor(fl::rand({2}, fl::dtype::f16), true);
  auto y = Tensor(fl::rand({5, 2}, fl::dtype::f16), true);
  auto z = y * autograd::tileAs(x, y);
  ASSERT_EQ(x.type(), z.type());
  auto dz = Tensor(fl::full({5, 2}, 1.0, fl::dtype::f16), false);
  z.backward(dz);
  auto dy = y.grad();
  auto dx = x.grad();
  ASSERT_TRUE(allClose(
      dy, fl::tile(x, {5, 1}).astype(dx.type()), 1e-2));
  ASSERT_TRUE(
      allClose(dx, fl::sum(y, {0}).astype(dx.type()), 1e-2));
}

TEST(AutogradTest, TileAs2) {
  // Right-align: x{3} tiled to {10, 3}
  auto x = Tensor(fl::rand({3}), true);
  auto z = autograd::tileAs(x, Shape({10, 3}));
  auto dz = Tensor(fl::full({10, 3}, 1.0), false);
  z.backward(dz);
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx, fl::full(x.shape(), 10.0)));
}

TEST(AutogradTest, Indexing) {
  auto x = Tensor(fl::rand({5, 6, 7, 4}, fl::dtype::f64), true);

  auto funcCol = [](Tensor& input) { return input(fl::span, 4); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcCol, x));

  auto funcRow = [](Tensor& input) { return input(4); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcRow, x));

  auto funcSlice = [](Tensor& input) {
    return input(fl::span, fl::span, 4);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcSlice, x));

  auto funcCols = [](Tensor& input) {
    return input(fl::span, fl::range(2, 5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcCols, x));

  auto funcRows = [](Tensor& input) { return input(fl::range(2, 5)); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcRows, x));

  auto funcSlices = [](Tensor& input) {
    return input(fl::span, fl::span, fl::range(2, 5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcSlices, x));
  auto funcFlat = [](Tensor& input) {
    return input.flat(fl::range(4, 100));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcFlat, x));
}

TEST(AutogradTest, Padding) {
  auto in = Tensor(fl::rand({3, 3}, fl::dtype::f32), true);
  auto funcPad = [&](Tensor& input) {
    return autograd::pad(input, {{1, 2}, {0, 1}}, -1);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcPad, in, 1E-3));
}

TEST(AutogradTest, Pooling) {
  // Row-major: {N, C, H, W} = {1, 1, 3, 3}
  auto in = Tensor(fl::rand({1, 1, 3, 3}, fl::dtype::f32), true);
  auto funcPool = [&](Tensor& input) { return autograd::pool2d(input, 2, 2, 1, 1); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcPool, in, 1E-3));
}

TEST_F(AutogradTestF16, PoolingF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  const float inputScale = 2.0; // scale the input to prevent grad underflow
  // Row-major: {N, C, H, W} = {1, 1, 3, 3}
  auto in = Tensor(inputScale * fl::rand({1, 1, 3, 3}, fl::dtype::f16), true);
  auto funcPool = [&](Tensor& input) { return autograd::pool2d(input, 2, 2, 1, 1); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcPool, in, 1e1, 1e-1)); // TODO: investigate
}

TEST_F(AutogradTestF16, PoolingBf16) {
  if (!fl::dtypeSupported(fl::dtype::bf16)) {
    GTEST_SKIP() << "bf16 not supported on this device";
  }

  // Row-major: {N, C, H, W} = {1, 1, 3, 3}
  auto in = Tensor(fl::rand({1, 1, 3, 3}, fl::dtype::bf16), true);

  auto out = autograd::pool2d(in, 2, 2, 1, 1);
  // Forward: correct dtype
  ASSERT_EQ(out.type(), fl::dtype::bf16);

  // f32 reference comparison
  auto in_f32 = Tensor(in.detach().astype(fl::dtype::f32), false);
  auto out_f32 = autograd::pool2d(in_f32, 2, 2, 1, 1);
  ASSERT_TRUE(allClose(out.detach().astype(fl::dtype::f32), out_f32.detach(), 0.05));

  // Backward: correct gradient dtype, no NaNs
  autograd::sum(out, {0, 1, 2, 3}).backward();
  ASSERT_EQ(in.grad().type(), fl::dtype::bf16);
  ASSERT_FALSE(fl::any(fl::isnan(in.grad().detach().astype(fl::dtype::f32))).asScalar<bool>());
}

TEST_F(AutogradTestF16, LinearBf16) {
  if (!fl::dtypeSupported(fl::dtype::bf16)) {
    GTEST_SKIP() << "bf16 not supported on this device";
  }
  int inDim = 4, outDim = 3, batchSize = 2;
  auto x = Tensor(fl::rand({inDim, batchSize}, fl::dtype::bf16) * 0.1, true);
  auto wt = Tensor(fl::rand({outDim, inDim}, fl::dtype::bf16) * 0.1, true);
  auto bs = Tensor(fl::rand({outDim, 1}, fl::dtype::bf16) * 0.1, true);

  auto y = autograd::matmul(wt, x) + autograd::tileAs(bs, Shape({outDim, batchSize}));
  // Forward: correct dtype
  ASSERT_EQ(y.type(), fl::dtype::bf16);

  // f32 reference comparison — compute using the same bf16-quantized inputs
  // cast to f32, so we only measure matmul precision difference
  auto x_f32 = Tensor(x.detach().astype(fl::dtype::f32), false);
  auto wt_f32 = Tensor(wt.detach().astype(fl::dtype::f32), false);
  auto bs_f32 = Tensor(bs.detach().astype(fl::dtype::f32), false);
  auto y_f32 = autograd::matmul(wt_f32, x_f32) +
      autograd::tileAs(bs_f32, Shape({outDim, batchSize}));
  // bf16 matmul uses cublasGemmEx with f32 accumulate and bf16 I/O.
  // Both paths use same bf16-quantized inputs; differences come from
  // bf16 output rounding vs f32 output.
  ASSERT_TRUE(allClose(y.detach().astype(fl::dtype::f32), y_f32.detach(), 0.05));

  // Backward: correct gradient dtypes, no NaNs
  autograd::sum(y, {0, 1}).backward();
  ASSERT_EQ(x.grad().type(), fl::dtype::bf16);
  ASSERT_EQ(wt.grad().type(), fl::dtype::bf16);
  ASSERT_EQ(bs.grad().type(), fl::dtype::bf16);
  ASSERT_FALSE(fl::any(fl::isnan(x.grad().detach().astype(fl::dtype::f32))).asScalar<bool>());
  ASSERT_FALSE(fl::any(fl::isnan(wt.grad().detach().astype(fl::dtype::f32))).asScalar<bool>());
}

TEST(AutogradTest, PoolingAvgIncludePadding) {
  auto input = Tensor(fl::rand({2, 3, 8, 8}), false);
  auto out = autograd::pool2d(
      input, 3, 3, 1, 1, 1, 1, PoolingMode::AVG_INCLUDE_PADDING);
  ASSERT_EQ(out.shape(), Shape({2, 3, 8, 8}));

  // Small known-value case: all-ones input, 2x2 pool, stride 2
  auto small = Tensor(fl::full({1, 1, 4, 4}, 1.0f), false);
  auto pooled = autograd::pool2d(
      small, 2, 2, 2, 2, 0, 0, PoolingMode::AVG_INCLUDE_PADDING);
  ASSERT_EQ(pooled.shape(), Shape({1, 1, 2, 2}));
  // Each 2x2 region of all-ones averages to 1.0
  ASSERT_TRUE(allClose(pooled, fl::full({1, 1, 2, 2}, 1.0f)));
}

TEST(AutogradTest, PoolingAvgJacobian) {
  auto input = Tensor(fl::rand({1, 1, 6, 6}, fl::dtype::f32), true);
  auto func = [](Tensor& in) {
    return autograd::pool2d(
        in, 3, 3, 2, 2, 0, 0, PoolingMode::AVG_INCLUDE_PADDING);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(func, input, 1e-3));
}

TEST(AutogradTest, TransposeMultiDim) {
  auto in = Tensor(fl::rand({3, 1, 4, 1}, fl::dtype::f32) * 2, true);
  auto funcTranspose = [&](Tensor& input) {
    return autograd::transpose(input, {2, 0, 3, 1});
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcTranspose, in, 1E-3));
}

TEST(AutogradTest, Embedding) {
  // 1D input
  {
    auto ip_t = Tensor({3}, fl::dtype::f32);
    ip_t(0) = 1; ip_t(1) = 0; ip_t(2) = 2;
    auto input = Tensor(ip_t, false);
    auto weights = Tensor(fl::randn({3, 2}, fl::dtype::f64), true);

    auto out = autograd::embedding(input, weights);
    // Forward check: out[i, :] = weights[ip[i], :]
    ASSERT_TRUE(allClose(out(0, fl::span), weights(1, fl::span)));
    ASSERT_TRUE(allClose(out(1, fl::span), weights(0, fl::span)));
    ASSERT_TRUE(allClose(out(2, fl::span), weights(2, fl::span)));

    auto funcEmbed = [&](Tensor& w) { return autograd::embedding(input, w); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcEmbed, weights, 1E-5));
  }

  // Multi-dim input
  {
    int nWords = 10;
    auto input =
        Tensor((fl::rand({4, 2}) * nWords).astype(fl::dtype::f32), false);
    auto weights = Tensor(fl::randn({nWords, 4}, fl::dtype::f64), true);
    auto funcEmbed = [&](Tensor& w) { return autograd::embedding(input, w); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcEmbed, weights, 1E-5));
  }
}

TEST(AutogradTest, GetAdvancedIndex) {
#ifdef FL_TEST_BACKEND_CPU
  GTEST_SKIP()
      << "Advanced indexing gradient unsupported on CPU backend";
#endif
#ifdef FL_TEST_BACKEND_CUDA
  GTEST_SKIP()
      << "Advanced indexing gradient has illegal memory access on CUDA JIT backend";
#endif
  if (!FL_BACKEND_CUDA) {
    GTEST_SKIP()
        << "Advanced indexing operator unsupported for non-CUDA backends";
  }
  std::vector<fl::dtype> validIndexTypes = {
      fl::dtype::s32, fl::dtype::s64, fl::dtype::u32, fl::dtype::u64};
  for (const auto& dtype : validIndexTypes) {
    auto x = Tensor(fl::rand({20, 50, 40, 30}, fl::dtype::f32), true);
    Tensor a({6}, dtype);
    a(0) = 0;
    a(1) = 15;
    a(2) = 6;
    a(3) = 1;
    a(4) = 10;
    a(5) = 6;
    Tensor b({3}, dtype);
    b(0) = 5;
    b(1) = 11;
    b(2) = 19;
    auto x2 = x(a, b, fl::span, fl::range(0, 4));
    auto y = autograd::sum(x2 * x2, {0, 1, 2, 3}, /* keepDims = */ true);
    auto res = 2 * autograd::sum(x2, {0, 1, 2, 3}, /* keepDims = */ true);
    y.backward();
    auto grad = autograd::sum(x.grad(), {0, 1, 2, 3}, /* keepDims = */ true);
    ASSERT_TRUE(allClose(grad, res, 1e-3));
  }
}

TEST(AutogradTest, GetAdvancedIndexF16) {
#ifdef FL_TEST_BACKEND_CUDA
  GTEST_SKIP()
      << "Advanced indexing gradient has illegal memory access on CUDA JIT backend";
#endif
  // TODO: remove me
  if (!FL_BACKEND_CUDA) {
    GTEST_SKIP()
        << "Advanced indexing operator unsupported for non-CUDA backends";
  }
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }
  std::vector<fl::dtype> validIndexTypes = {
      fl::dtype::s32, fl::dtype::s64, fl::dtype::u32, fl::dtype::u64};
  for (const auto& dtype : validIndexTypes) {
    auto x = Tensor(fl::rand({20, 50, 40, 30}, fl::dtype::f16), true);
    Tensor a({6}, dtype);
    a(0) = 0;
    a(1) = 15;
    a(2) = 6;
    a(3) = 1;
    a(4) = 10;
    a(5) = 6;
    Tensor b({3}, dtype);
    b(0) = 5;
    b(1) = 11;
    b(2) = 19;
    auto x2 = x(a, b, fl::span, fl::range(0, 4));
    ASSERT_EQ(x2.type(), fl::dtype::f16);
    auto y = autograd::sum(x2 * x2, {0, 1, 2, 3}, /* keepDims = */ true);
    auto res = 2 * autograd::sum(x2, {0, 1, 2, 3}, /* keepDims = */ true);
    y.backward();
    ASSERT_EQ(x.grad().type(), fl::dtype::f16);
    auto grad = autograd::sum(x.grad(), {0, 1, 2, 3}, /* keepDims = */ true);
    ASSERT_TRUE(allClose(grad, res, 1e-3));
  }
}

TEST(AutogradTest, AsContiguous) {
#ifdef FL_TEST_BACKEND_CPU
  // JIT backend always returns contiguous copies from indexing (no strided
  // views), so the non-contiguous precondition cannot be satisfied.
  GTEST_SKIP() << "Strided views not supported on JIT/CPU backend";
#endif
  // Slicing with a step > 1 creates a non-contiguous view
  auto raw = fl::rand({6, 8});
  auto sliced = raw(fl::range(0, 6, 2), fl::span); // every other row
  ASSERT_FALSE(sliced.isContiguous());

  // Use std::move to avoid Tensor copy constructor which deep-clones
  // (making the data contiguous). The by-value parameter of Tensor(Tensor, bool)
  // will use the move constructor instead, preserving the non-contiguous view.
  auto t = Tensor(std::move(sliced), true);
  ASSERT_FALSE(t.isContiguous());
  ASSERT_TRUE(t.requiresGrad());

  auto c = t.asContiguous();
  ASSERT_TRUE(c.isContiguous());
  // autograd metadata is preserved (shared autogradMeta_)
  ASSERT_TRUE(c.requiresGrad());
  // data matches
  ASSERT_TRUE(allClose(c, t));

  // already-contiguous tensor returns itself
  auto x2 = Tensor(fl::rand({3, 3}), true);
  ASSERT_TRUE(x2.isContiguous());
  auto c2 = x2.asContiguous();
  ASSERT_TRUE(c2.isContiguous());
  ASSERT_TRUE(c2.requiresGrad());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  flTestInit();
  return RUN_ALL_TESTS();
}
