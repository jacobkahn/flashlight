/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/test/TestInit.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/test/autograd/AutogradTestUtils.h"

using namespace ::testing;
using namespace fl;

using fl::detail::AutogradTestF16;

TEST(AutogradBinaryOpsTest, BasicOps) {
  using FuncVar = std::function<Tensor(Tensor&, Tensor&)>;
  using FuncScalarL = std::function<Tensor(double, Tensor&)>;
  using FuncScalarR = std::function<Tensor(Tensor&, double)>;
  auto testImpl = [](FuncVar fn1, FuncScalarL fn2, FuncScalarR fn3) {
    auto input = Tensor(fl::rand({3, 4, 5, 6}, fl::dtype::f64) + 1, true);
    auto temp = Tensor(fl::rand({3, 4, 5, 6}, fl::dtype::f64) - 2, false);
    fl::detail::JacobianFunc fnArrL = [&](Tensor& in) { return fn1(in, temp); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(fnArrL, input));
    fl::detail::JacobianFunc fnArrR = [&](Tensor& in) { return fn1(temp, in); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(fnArrR, input));
    fl::detail::JacobianFunc fnScalarL = [&](Tensor& in) { return fn2(1.414, in); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(fnScalarL, input, 1E-5, 1E-7));
    fl::detail::JacobianFunc fnScalarR = [&](Tensor& in) { return fn3(in, 1.732); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(fnScalarR, input, 1E-5, 1E-7));
  };

  FuncVar funcAdd1 = [](Tensor& a, Tensor& b) { return a + b; };
  FuncScalarL funcAdd2 = [](double a, Tensor& b) { return a + b; };
  FuncScalarR funcAdd3 = [](Tensor& a, double b) { return a + b; };
  testImpl(funcAdd1, funcAdd2, funcAdd3);

  FuncVar funcSub1 = [](Tensor& a, Tensor& b) { return a - b; };
  FuncScalarL funcSub2 = [](double a, Tensor& b) { return a - b; };
  FuncScalarR funcSub3 = [](Tensor& a, double b) { return a - b; };
  testImpl(funcSub1, funcSub2, funcSub3);

  FuncVar funcDiv1 = [](Tensor& a, Tensor& b) { return a / b; };
  FuncScalarL funcDiv2 = [](double a, Tensor& b) { return a / b; };
  FuncScalarR funcDiv3 = [](Tensor& a, double b) { return a / b; };
  testImpl(funcDiv1, funcDiv2, funcDiv3);

  FuncVar funcMul1 = [](Tensor& a, Tensor& b) { return a * b; };
  FuncScalarL funcMul2 = [](double a, Tensor& b) { return a * b; };
  FuncScalarR funcMul3 = [](Tensor& a, double b) { return a * b; };
  testImpl(funcMul1, funcMul2, funcMul3);

  FuncVar funcMin1 = [](Tensor& a, Tensor& b) { return autograd::minimum(a, b); };
  FuncScalarL funcMin2 = [](double a, Tensor& b) { return autograd::minimum(a, b); };
  FuncScalarR funcMin3 = [](Tensor& a, double b) { return autograd::minimum(a, b); };
  testImpl(funcMin1, funcMin2, funcMin3);

  FuncVar funcMax1 = [](Tensor& a, Tensor& b) { return autograd::maximum(a, b); };
  FuncScalarL funcMax2 = [](double a, Tensor& b) { return autograd::maximum(a, b); };
  FuncScalarR funcMax3 = [](Tensor& a, double b) { return autograd::maximum(a, b); };
  testImpl(funcMax1, funcMax2, funcMax3);
}

TEST(AutogradBinaryOpsTest, BroadcastedBinaryOps) {
  // Test that autograd correctly reduces gradients when forward ops broadcast.
  // Shape pairs: {lhs, rhs} where broadcasting occurs.
  std::vector<std::pair<Shape, Shape>> shapePairs = {
      {{3, 1}, {3, 4}},       // broadcast dim 1
      {{1, 4}, {3, 4}},       // broadcast dim 0
      {{3, 1, 5}, {3, 4, 5}}, // broadcast middle dim
      {{4}, {3, 4}},          // rank mismatch (right-aligned broadcast)
      {{1}, {3, 4}},          // scalar-like broadcast
  };

  for (auto& [lhsShape, rhsShape] : shapePairs) {
    auto lhs =
        Tensor(fl::rand(lhsShape, fl::dtype::f64) * 2 - 1, true);
    auto rhs =
        Tensor(fl::rand(rhsShape, fl::dtype::f64) * 2 - 1, true);

    // operator+
    auto funcAddLhs = [&](Tensor& in) { return in + rhs; };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcAddLhs, lhs, 1E-5))
        << "add lhs grad failed for shapes " << lhsShape << ", " << rhsShape;
    auto funcAddRhs = [&](Tensor& in) { return lhs + in; };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcAddRhs, rhs, 1E-5))
        << "add rhs grad failed for shapes " << lhsShape << ", " << rhsShape;

    // operator-
    auto funcSubLhs = [&](Tensor& in) { return in - rhs; };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcSubLhs, lhs, 1E-5))
        << "sub lhs grad failed for shapes " << lhsShape << ", " << rhsShape;
    auto funcSubRhs = [&](Tensor& in) { return lhs - in; };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcSubRhs, rhs, 1E-5))
        << "sub rhs grad failed for shapes " << lhsShape << ", " << rhsShape;

    // operator*
    auto funcMulLhs = [&](Tensor& in) { return in * rhs; };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMulLhs, lhs, 1E-5))
        << "mul lhs grad failed for shapes " << lhsShape << ", " << rhsShape;
    auto funcMulRhs = [&](Tensor& in) { return lhs * in; };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMulRhs, rhs, 1E-5))
        << "mul rhs grad failed for shapes " << lhsShape << ", " << rhsShape;

    // operator/ (use positive values to avoid division issues)
    auto lhsPos =
        Tensor(fl::rand(lhsShape, fl::dtype::f64) + 0.5, true);
    auto rhsPos =
        Tensor(fl::rand(rhsShape, fl::dtype::f64) + 0.5, true);
    auto funcDivLhs = [&](Tensor& in) { return in / rhsPos; };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcDivLhs, lhsPos, 1E-5))
        << "div lhs grad failed for shapes " << lhsShape << ", " << rhsShape;
    auto funcDivRhs = [&](Tensor& in) { return lhsPos / in; };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcDivRhs, rhsPos, 1E-5))
        << "div rhs grad failed for shapes " << lhsShape << ", " << rhsShape;
  }
}

TEST(AutogradBinaryOpsTest, BinaryCrossEntropy) {
  auto x = Tensor(fl::rand({10}), true);
  auto y = Tensor(fl::rand({10}), true);
  auto loss = autograd::binaryCrossEntropy(x, y);

  // bce loss should be positive
  ASSERT_TRUE(fl::all(loss > 0).scalarSync<char>());

  // Verify against formula using raw tensor ops: -[y*log(x) + (1-y)*log(1-x)]
  auto xd = x.detach();
  auto yd = y.detach();
  auto expected = -(yd * fl::log(xd) + (1.0f - yd) * fl::log(1.0f - xd));
  ASSERT_TRUE(allClose(loss.detach(), expected, 1e-5));
}

TEST(AutogradBinaryOpsTest, CrossEntropy) {
  // Row-major: input {X1, X2, C}, target {X1, X2}
  // Use smaller tensors on CPU backend where Jacobian is O(n^2) per-element
#ifdef FL_TEST_BACKEND_CPU
  auto x = Tensor(fl::rand({3, 2, 5}, fl::dtype::f64), true);
  auto y = Tensor(
      (fl::rand({3, 2}, fl::dtype::u32) % 5).astype(fl::dtype::s32), false);
  const int numTargetElements = 6; // 3 * 2
#else
  auto x = Tensor(fl::rand({10, 4, 7}, fl::dtype::f64), true);
  auto y = Tensor(
      (fl::rand({10, 4}, fl::dtype::u32) % 7).astype(fl::dtype::s32), false);
  const int numTargetElements = 40; // 10 * 4
#endif
  auto ignoreIdx = y(0, 0).scalarSync<int>();

  std::vector<ReduceMode> modes = {
      ReduceMode::NONE, ReduceMode::SUM, ReduceMode::MEAN};
  for (auto mode : modes) {
    auto func = [&](Tensor& input) {
      return autograd::categoricalCrossEntropy(input, y, mode);
    };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(func, x, 1E-5));
    auto funcIgnore = [&](Tensor& input) {
      return autograd::categoricalCrossEntropy(input, y, mode, ignoreIdx);
    };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcIgnore, x, 1E-5));
  }

  auto lossSum = autograd::categoricalCrossEntropy(x, y, ReduceMode::SUM);
  auto lossMean = autograd::categoricalCrossEntropy(x, y, ReduceMode::MEAN);
  ASSERT_NEAR(
      (lossSum / lossMean).scalarSync<double>(), numTargetElements, 1e-5);

  auto lossSumIgnore =
      autograd::categoricalCrossEntropy(x, y, ReduceMode::SUM, ignoreIdx);
  auto lossMeanIgnore =
      autograd::categoricalCrossEntropy(x, y, ReduceMode::MEAN, ignoreIdx);
  auto ignoreCount = fl::sum(y == ignoreIdx).asScalar<unsigned>();
  if (ignoreCount < static_cast<unsigned>(numTargetElements)) {
    // Ratio of SUM/MEAN should equal number of non-ignored targets
    ASSERT_NEAR(
        (lossSumIgnore / lossMeanIgnore).scalarSync<double>(),
        numTargetElements - ignoreCount,
        1e-5);
  } else {
    // All targets ignored: both losses should be 0
    ASSERT_NEAR(lossSumIgnore.scalarSync<double>(), 0.0, 1e-5);
    ASSERT_NEAR(lossMeanIgnore.scalarSync<double>(), 0.0, 1e-5);
  }

  ASSERT_THROW(
      autograd::categoricalCrossEntropy(
          Tensor(fl::rand({4, 5, 6}), false),
          Tensor(fl::rand({5, 8}), false)),
      std::invalid_argument);

  ASSERT_THROW(
      autograd::categoricalCrossEntropy(
          Tensor(fl::rand({4, 5, 6}), false), Tensor(fl::rand({5}), false)),
      std::invalid_argument);
}


TEST(AutogradBinaryOpsTest, Linear) {
  std::vector<int> batchsizes = {1, 5};
  for (auto b : batchsizes) {
    auto in = Tensor(fl::rand({b, 4, 3}, fl::dtype::f64) * 2 - 1, true);
    auto wt = Tensor(fl::rand({6, 3}, fl::dtype::f64) * 2 - 1, true);
    auto bs = Tensor(fl::rand({6}, fl::dtype::f64) * 2 - 1, true);
    auto funcLinIn = [&](Tensor& input) { return autograd::linear(input, wt, bs); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcLinIn, in, 1E-8));
    auto funcLinWt = [&](Tensor& weight) { return autograd::linear(in, weight, bs); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcLinWt, wt, 1E-8));
    auto funcLinBs = [&](Tensor& bias) { return autograd::linear(in, wt, bias); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcLinBs, bs, 1E-8));
  }
}

TEST_F(AutogradTestF16, LinearF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  std::vector<int> batchsizes = {1, 5};
  const float scale = 4.0; // scale prevent grad underflow
  for (auto b : batchsizes) {
    auto in = Tensor(fl::rand({b, 2, 2}, fl::dtype::f16) * scale, true);
    auto wt = Tensor(fl::rand({2, 2}, fl::dtype::f16) * scale, true);
    auto bs = Tensor(fl::rand({2}, fl::dtype::f16) * scale, true);
    auto funcLinIn = [&](Tensor& input) { return autograd::linear(input, wt, bs); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcLinIn, in, 5E-2, 5E-1));
    auto funcLinWt = [&](Tensor& weight) { return autograd::linear(in, weight, bs); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcLinWt, wt, 5E-2, 5E-1));
    auto funcLinBs = [&](Tensor& bias) { return autograd::linear(in, wt, bias); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcLinBs, bs, 5E-2, 5E-1));
  }
}

TEST(AutogradBinaryOpsTest, Multiply) {
  auto x = Tensor(fl::rand({5}), true);
  auto y = x * x;
  auto dy = Tensor(fl::full({5}, 1.0), false);
  y.backward(dy);
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx, 2 * x));
}

TEST(AutogradBinaryOpsTest, MultiplyAdd) {
  auto x = Tensor(fl::rand({5}), true);
  auto y = Tensor(fl::rand({5}), true);
  auto z = x * x + x * y + y * y;
  auto dz = Tensor(fl::full({5}, 1.0), false);
  z.backward(dz);
  auto dx = x.grad();
  auto dy = y.grad();
  ASSERT_TRUE(allClose(dx, 2 * x + y));
  ASSERT_TRUE(allClose(dy, 2 * y + x));
}

TEST(AutogradBinaryOpsTest, MultiplyAddScalar) {
  auto x = Tensor(fl::rand({5}), true);
  auto y = Tensor(fl::rand({5}), true);
  auto z = 2 * x + x * y + y;
  auto dz = Tensor(fl::full({5}, 1.0), false);
  z.backward(dz);
  auto dx = x.grad();
  auto dy = y.grad();
  ASSERT_TRUE(allClose(dx, (2.0 + y)));
  ASSERT_TRUE(allClose(dy, (1.0 + x)));
}

TEST(AutogradBinaryOpsTest, MultiplySub) {
  auto x = Tensor(fl::rand({5}), true);
  auto y = Tensor(fl::rand({5}), true);
  auto z = x * x - x * y;
  auto dz = Tensor(fl::full({5}, 1.0), false);
  z.backward(dz);
  auto dx = x.grad();
  auto dy = y.grad();
  ASSERT_TRUE(allClose(dx, (2 * x - y)));
  ASSERT_TRUE(allClose(dy, (-x)));
}

TEST(AutogradBinaryOpsTest, DivideAdd) {
  auto x = Tensor(fl::rand({5}, fl::dtype::f64), true);
  auto y = Tensor(fl::rand({5}, fl::dtype::f64), true);
  auto z = x + x / y + y;
  auto dz = Tensor(fl::full({5}, 1.0, fl::dtype::f64), false);
  z.backward(dz);
  auto dx = x.grad();
  auto dy = y.grad();
  ASSERT_EQ(z.type(), fl::dtype::f64);
  ASSERT_TRUE(allClose(dx, (1.0 + 1.0 / y)));
  ASSERT_TRUE(
      allClose(dy, (1.0 - x / (y * y))));
}

TEST(AutogradBinaryOpsTest, matmul) {
  unsigned M = 10;
  unsigned K = 12;
  unsigned N = 14;
  unsigned b2 = 2;
  unsigned b3 = 4;
  auto mk = Shape({M, K});
  auto mkb2 = Shape({b2, M, K}); // 1 batch dim
  auto mkb2b3 = Shape({b2, b3, M, K}); // 2 batch dims
  auto kn = Shape({K, N});
  auto knb2 = Shape({b2, K, N}); // 1 batch dim
  auto knb2b3 = Shape({b2, b3, K, N}); // 2 batch dims

  // lhs, rhs
  std::vector<std::pair<Shape, Shape>> inputs = {
      {mk, kn},
      {mk, knb2},
      {mk, knb2b3},
      {mkb2, kn},
      {mkb2, knb2},
      {mkb2b3, kn},
      {mkb2b3, knb2b3}};

  // Swap the last two (matrix) dims for transpose operations
  auto trMatrixDims = [](const Shape& in) -> Shape {
    Shape out = in;
    int n = in.ndim();
    auto tmp = out[n - 1];
    out[n - 1] = out[n - 2];
    out[n - 2] = tmp;
    return out;
  };

  for (auto& pair : inputs) {
    auto& aShape = pair.first;
    auto& bShape = pair.second;

    auto a = Tensor(fl::rand(aShape, fl::dtype::f64) * 2 - 1, true);
    auto b = Tensor(fl::rand(bShape, fl::dtype::f64) * 2 - 1, true);

    auto aT = Tensor(fl::rand(trMatrixDims(aShape), fl::dtype::f64), true);
    auto bT = Tensor(fl::rand(trMatrixDims(bShape), fl::dtype::f64), true);

    // matmul
    auto funcMatmulLhs = [&](Tensor& input) { return autograd::matmul(input, b); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMatmulLhs, a, 1E-6))
        << "matmul lhs gradient: lhs " << a.shape() << " rhs " << b.shape();
    auto funcMatmulRhs = [&](Tensor& input) { return autograd::matmul(a, input); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMatmulRhs, b, 1E-6))
        << "matmul rhs gradient: lhs " << a.shape() << " rhs " << b.shape();

    // matmul with Transpose on lhs
    auto funcMatmulTNLhs = [&](Tensor& input) { return autograd::matmul(input, b, MatrixProperty::Transpose); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMatmulTNLhs, aT, 1E-6))
        << "matmul(TN) lhs gradient: lhs " << a.shape() << " rhs " << b.shape();
    auto funcMatmulTNRhs = [&](Tensor& input) { return autograd::matmul(aT, input, MatrixProperty::Transpose); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMatmulTNRhs, b, 1E-6))
        << "matmul(TN) rhs gradient: lhs " << a.shape() << " rhs " << b.shape();

    // matmul with Transpose on rhs
    auto funcMatmulNTLhs = [&](Tensor& input) { return autograd::matmul(input, bT, MatrixProperty::None, MatrixProperty::Transpose); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMatmulNTLhs, a, 1E-6))
        << "matmul(NT) lhs gradient: lhs " << a.shape() << " rhs " << b.shape();
    auto funcMatmulNTRhs = [&](Tensor& input) { return autograd::matmul(a, input, MatrixProperty::None, MatrixProperty::Transpose); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMatmulNTRhs, bT, 1E-6))
        << "matmul(NT) rhs gradient: lhs " << a.shape() << " rhs " << b.shape();
  }
}

TEST(AutogradBinaryOpsTest, PowVariableVariable) {
  auto base = Tensor(fl::rand({3, 4}, fl::dtype::f64) + 0.5, true);
  auto exponent = Tensor(fl::rand({3, 4}, fl::dtype::f64) + 0.5, true);

  auto funcBase = [&](Tensor& in) { return autograd::power(in, exponent); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBase, base, 1E-5));

  auto funcExp = [&](Tensor& in) { return autograd::power(base, in); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcExp, exponent, 1E-5));
}

TEST(AutogradBinaryOpsTest, Where) {
  auto x = Tensor(fl::rand({3, 4, 5}, fl::dtype::f64), true);
  auto y = Tensor(fl::rand({3, 4, 5}, fl::dtype::f64), true);
  auto cond = Tensor(fl::rand({3, 4, 5}) > 0.5, false);

  auto funcX = [&](Tensor& in) { return autograd::where(cond, in, y); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcX, x, 1E-5));

  auto funcY = [&](Tensor& in) { return autograd::where(cond, x, in); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcY, y, 1E-5));
}

TEST(AutogradBinaryOpsTest, PowScalarBase) {
  auto exponent = Tensor(fl::rand({3, 4}, fl::dtype::f64) + 0.5, true);
  double base = 2.5;

  auto func = [&](Tensor& in) { return autograd::power(base, in); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(func, exponent, 1E-5));

  // Verify forward value
  auto result = autograd::power(base, exponent);
  ASSERT_TRUE(allClose(result, fl::power(base, exponent)));
}

TEST(AutogradNormalizationTest, WeightNormLinear) {
  auto v = Tensor(fl::rand({3, 2}), true);
  auto normDim = {1};
  // Use keepDims=true so norm output {3, 1} is broadcast-compatible with {3, 2}
  auto g = Tensor(autograd::norm(v, normDim, 2, true), true);
  auto in = Tensor(fl::rand({2, 3}, fl::dtype::f32), true);

  auto funcWeightNormIn = [&](Tensor& input) {
    auto w = v * autograd::tileAs(g / autograd::norm(v, normDim, 2, true), v);
    return autograd::matmul(w, input);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcWeightNormIn, in, 1E-3));

  auto funcWeightNormV = [&](Tensor& input) {
    auto w = input * autograd::tileAs(g / autograd::norm(input, normDim, 2, true), input);
    return autograd::matmul(w, in);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcWeightNormV, v, 1E-2));

  auto funcWeightNormG = [&](Tensor& input) {
    auto w = v * autograd::tileAs(input / autograd::norm(v, normDim, 2, true), v);
    return autograd::matmul(w, in);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcWeightNormG, g, 5E-3));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  flTestInit();
  return RUN_ALL_TESTS();
}
