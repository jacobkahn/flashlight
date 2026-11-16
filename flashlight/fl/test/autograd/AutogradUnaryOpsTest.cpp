/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/test/TestInit.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/test/autograd/AutogradTestUtils.h"

using namespace ::testing;
using namespace fl;

using fl::detail::AutogradTestF16;

TEST(AutogradUnaryOpsTest, Clip) {
  auto input = Tensor(fl::rand({5, 6, 7, 4}, fl::dtype::f64) * 3, true);
  double lo = -1.0, hi = 1.0;
  float perturb = 1E-5;
  // Need to do this as gradient is not continuous when input = lo / hi.
  auto& inarr = input.detach();
  inarr = fl::where(fl::abs(inarr - lo) > perturb, inarr, lo + 10 * perturb);
  inarr = fl::where(fl::abs(inarr - hi) > perturb, inarr, hi + 10 * perturb);

  auto funcCol = [lo, hi](Tensor& in) { return autograd::clip(in, lo, hi); };

  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcCol, input, 1E-10, perturb));
}

TEST(AutogradUnaryOpsTest, Glu) {
  auto in = Tensor(fl::rand({3, 4, 5}, fl::dtype::f64), true);
  auto funcGlu = [&](Tensor& input) { return autograd::gatedlinearunit(input, 1); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcGlu, in, 1E-5));
}

TEST(AutogradUnaryOpsTest, Sigmoid) {
  auto x = Tensor(fl::rand({5}), true);
  auto y = autograd::sigmoid(x);
  auto dy = Tensor(fl::full({5}, 1.0), false);
  y.backward(dy);
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx, (y * (1 - y))));
  ASSERT_TRUE(allClose(
      dx, (fl::sigmoid(x) * (1 - fl::sigmoid(x)))));
}

TEST(AutogradUnaryOpsTest, Erf) {
  auto x = Tensor(fl::rand({5}), true);
  auto y = autograd::erf(x);
  ASSERT_TRUE(allClose(fl::erf(x), y));

  auto dy = Tensor(fl::full({5}, 1.0), false);
  y.backward(dy);
  auto targetGrads = 2 / std::sqrt(M_PI) * autograd::exp(autograd::negative(x * x));
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx, targetGrads));

  auto funcErf = [](Tensor& in) { return autograd::erf(in); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcErf, x, 5e-4, 1e-4));
}

TEST(AutogradUnaryOpsTest, Tanh) {
  auto x = Tensor(fl::rand({5}), true);
  auto y = autograd::tanh(x);
  auto dy = Tensor(fl::full({5}, 1.0), false);
  y.backward(dy);
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx, (1 - y * y)));
  ASSERT_TRUE(allClose(
      dx, (1 + fl::tanh(x)) * (1 - fl::tanh(x))));
}

TEST(AutogradUnaryOpsTest, Transpose) {
  // Use smaller tensors on CPU backend where Jacobian is O(n^2) per-element
#ifdef FL_TEST_BACKEND_CPU
  auto in = Tensor(fl::rand({2, 3, 4, 5}), true);
#else
  auto in = Tensor(fl::rand({5, 6, 7, 8}), true);
#endif
  auto out = autograd::transpose(in, {2, 0, 1, 3});
  out.backward();
  ASSERT_EQ(in.grad().shape(), in.shape());

  auto funcErf = [](Tensor& in) { return autograd::transpose(in, {1, 3, 2, 0}); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcErf, in, 5e-4, 1e-4));

#ifdef FL_TEST_BACKEND_CPU
  auto in2 = Tensor(fl::rand({3, 4, 5, 2}), true);
#else
  auto in2 = Tensor(fl::rand({6, 7, 8, 9}), true);
#endif
  auto out2 = autograd::transpose(in2);
  out2.backward();
  ASSERT_EQ(in2.grad().shape(), in2.shape());

  auto funcErf2 = [](Tensor& in) { return autograd::transpose(in); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcErf2, in2, 5e-4, 1e-4));
}

TEST(AutogradUnaryOpsTest, Exp) {
  auto x = Tensor(fl::rand({5}), true);
  auto y = autograd::exp(x);
  auto dy = Tensor(fl::full({5}, 1.0), false);
  y.backward(dy);
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx, (fl::exp(x))));
}

TEST(AutogradUnaryOpsTest, Log1p) {
  auto x = Tensor(fl::rand({5}), true);
  auto y = autograd::log1p(x);

  auto xCopy = Tensor(x, true);
  auto yExp = autograd::log(1 + xCopy);

  y.backward();
  yExp.backward();

  ASSERT_TRUE(allClose(y, yExp));
  ASSERT_TRUE(allClose(y.grad(), yExp.grad()));
  ASSERT_TRUE(allClose(x.grad(), xCopy.grad()));
}

TEST(AutogradUnaryOpsTest, Softmax) {
  auto in = Tensor(fl::rand({3, 5, 1}, fl::dtype::f64), true);
  auto funcSm = [&](Tensor& input) { return autograd::softmax(input, 0); };

  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcSm, in, 1E-5));
}

TEST_F(AutogradTestF16, SoftmaxF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto in = Tensor(fl::rand({3, 5, 1}, fl::dtype::f16), true);
  auto funcSm = [&](Tensor& input) { return autograd::softmax(input, 0); };

  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcSm, in, 1E-2, 1e-1));
}

TEST(AutogradUnaryOpsTest, LogSoftmax) {
  auto in = Tensor(fl::rand({3, 5, 1}, fl::dtype::f64), true);
  auto funcLsm = [&](Tensor& input) { return autograd::logSoftmax(input, 0); };

  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcLsm, in, 1E-5));
}

TEST_F(AutogradTestF16, LogSoftmaxF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto in = Tensor(fl::rand({3, 5, 1}, fl::dtype::f16), true);
  auto funcLsm = [&](Tensor& input) { return autograd::logSoftmax(input, 0); };

  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcLsm, in, 1E-2, 1e-1));
}

TEST(AutogradUnaryOpsTest, Pow) {
  {
    auto x = Tensor(fl::rand({5}), true);
    auto y = autograd::power(x, 2);
    auto dy = Tensor(fl::full({5}, 2.0), false);
    y.backward(dy);
    auto dx = x.grad();
    ASSERT_TRUE(allClose(dx, (2 * 2 * x)));
  }
  {
    auto x = Tensor(fl::rand({5}), true);
    auto y = autograd::power(x, 3);
    auto dy = Tensor(fl::full({5}, 1.0), false);
    y.backward(dy);
    auto dx = x.grad();
    ASSERT_TRUE(allClose(dx, (3 * fl::power(x, 2))));
  }
}

TEST(AutogradUnaryOpsTest, Sqrt) {
  auto x = Tensor(fl::rand({5, 3}, fl::dtype::f64), true);
  auto funcSqrt = [](Tensor& in) { return autograd::sqrt(in); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcSqrt, x, 1E-3));
}

TEST(AutogradUnaryOpsTest, Flip) {
  auto x = Tensor(fl::rand({5, 3, 2}, fl::dtype::f64), true);
  auto func = [](Tensor& in) { return autograd::flip(in, 0); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(func, x, 1E-5));

  auto func1 = [](Tensor& in) { return autograd::flip(in, 1); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(func1, x, 1E-5));
}

TEST(AutogradUnaryOpsTest, Tril) {
  auto x = Tensor(fl::rand({5, 5}, fl::dtype::f64), true);
  auto func = [](Tensor& in) { return autograd::tril(in); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(func, x, 1E-5));
}

TEST(AutogradUnaryOpsTest, Triu) {
  auto x = Tensor(fl::rand({5, 5}, fl::dtype::f64), true);
  auto func = [](Tensor& in) { return autograd::triu(in); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(func, x, 1E-5));
}

TEST(AutogradUnaryOpsTest, Cumsum) {
  auto x = Tensor(fl::rand({5, 3}, fl::dtype::f64), true);
  auto func = [](Tensor& in) { return autograd::cumsum(in, 0); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(func, x, 1E-5));

  auto func1 = [](Tensor& in) { return autograd::cumsum(in, 1); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(func1, x, 1E-5));
}

TEST(AutogradUnaryOpsTest, Roll) {
  auto x = Tensor(fl::rand({5, 3}, fl::dtype::f64), true);
  auto func = [](Tensor& in) { return autograd::roll(in, 2, 0); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(func, x, 1E-5));

  auto func1 = [](Tensor& in) { return autograd::roll(in, -1, 1); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(func1, x, 1E-5));
}

TEST(AutogradUnaryOpsTest, NonDifferentiableWrappers) {
  auto x = Tensor(fl::rand({5, 3}) * 10 - 5, true);

  // floor
  auto fResult = autograd::floor(x);
  ASSERT_TRUE(allClose(fResult, fl::floor(x)));
  ASSERT_FALSE(fResult.requiresGrad());

  // ceil
  auto cResult = autograd::ceil(x);
  ASSERT_TRUE(allClose(cResult, fl::ceil(x)));
  ASSERT_FALSE(cResult.requiresGrad());

  // rint
  auto rResult = autograd::rint(x);
  ASSERT_TRUE(allClose(rResult, fl::rint(x)));
  ASSERT_FALSE(rResult.requiresGrad());

  // sign
  auto sResult = autograd::sign(x);
  ASSERT_TRUE(allClose(sResult, fl::sign(x)));
  ASSERT_FALSE(sResult.requiresGrad());

  // isnan
  auto nanResult = autograd::isnan(x);
  ASSERT_TRUE(allClose(nanResult, fl::isnan(x)));
  ASSERT_FALSE(nanResult.requiresGrad());

  // isinf
  auto infResult = autograd::isinf(x);
  ASSERT_TRUE(allClose(infResult, fl::isinf(x)));
  ASSERT_FALSE(infResult.requiresGrad());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  flTestInit();
  return RUN_ALL_TESTS();
}
