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

TEST(AutogradReductionTest, Sum) {
  for (const bool keepDims : {false, true}) {
    Shape s = {6};
    if (keepDims) {
      s = {6, 1};
    }

    auto x = Tensor(fl::rand(s), true);
    auto y = Tensor(fl::rand({6, 3}), true);

    auto z = x * autograd::sum(y, {1}, keepDims);
    auto dz = Tensor(fl::full(s, 1.0), false);
    z.backward(dz);

    auto dy = y.grad();
    auto dx = x.grad();
    // Reshape x to {6, 1} so tile broadcasts correctly with right-alignment
    ASSERT_TRUE(allClose(
        dy, fl::tile(fl::reshape(x, {6, 1}), {1, 3})));
    ASSERT_TRUE(allClose(dx, fl::sum(y, {1}, keepDims)));

    // Reduce over 1-dim input
    auto funcMean_0 = [keepDims](const Tensor& in) {
      return autograd::sum(in, {0}, keepDims);
    };
    auto in = Tensor(fl::rand({6}), true);
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMean_0, in, 5E-3));
    // Reduce over scalar input
    auto inScalar = Tensor(fl::fromScalar(3.14), true);
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMean_0, inScalar, 5E-3));
  }

  auto r = Tensor(fl::rand({5, 6, 7, 8}), true);
  auto rOut = autograd::sum(r, {1, 2});
  auto rOutTensor = fl::sum(r, {1, 2});
  ASSERT_TRUE(allClose(rOut, rOutTensor));
}

TEST(AutogradReductionTest, SumAs) {
  // Use right-alignment compatible shapes: {2} broadcasts with {5, 2}
  auto x = Tensor(fl::rand({2}), true);
  auto y = Tensor(fl::rand({5, 2}), true);
  auto z = x * autograd::sumAs(y, x);
  auto dz = Tensor(fl::full({2}, 1.0), false);
  z.backward(dz);
  auto dy = y.grad();
  auto dx = x.grad();
  ASSERT_TRUE(allClose(
      dy, fl::tile(fl::reshape(x, {1, 2}), {5, 1})));
  ASSERT_TRUE(allClose(dx, fl::sum(y, {0})));
}

TEST(AutogradReductionTest, SumAs2) {
  // Use right-alignment compatible target: {2} broadcasts with {5, 2}
  auto y = Tensor(fl::rand({5, 2}), true);
  auto z = autograd::sumAs(y, {2});
  auto dz = Tensor(fl::full({2}, 1.0), false);
  z.backward(dz);
  auto dy = y.grad();
  ASSERT_TRUE(allClose(dy, fl::full({5, 2}, 1.0)));
}

TEST(AutogradReductionTest, Mean) {
  for (const bool keepDims : {false, true}) {
    Shape xShape = keepDims ? Shape({5, 1, 1}) : Shape({5});
    auto x = Tensor(fl::rand(xShape), true);
    auto y = Tensor(fl::rand({5, 3, 2}), true);
    auto varOut = autograd::mean(y, {1, 2}, keepDims);
    auto z = x * autograd::mean(y, {1, 2}, keepDims);
    auto dz = Tensor(fl::full(x.shape(), 1.0), false);
    z.backward(dz);
    auto dy = y.grad();
    auto dx = x.grad();
    // Reshape x to {5, 1, 1} so tile broadcasts correctly with right-alignment
    ASSERT_TRUE(allClose(
        dy,
        fl::tile(fl::reshape(x, {5, 1, 1}), {1, 3, 2}) / 6));
    ASSERT_TRUE(allClose(dx, fl::mean(y, {1, 2}, keepDims)));

    auto a = Tensor(fl::rand({5, 3, 2}, fl::dtype::f64), true);
    auto funcMean = [keepDims](Tensor& in) {
      return autograd::mean(in, {1, 2}, keepDims);
    };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMean, a, 1E-4));

    auto q = Tensor(fl::rand({5, 6, 7, 8}), false);
    auto qOut = autograd::mean(q, {1, 2}, keepDims);
    auto qOutTensor = fl::mean(q, {1, 2}, keepDims);
    ASSERT_TRUE(allClose(qOut, qOutTensor));

    auto funcMean_0 = [keepDims](Tensor& in) {
      return autograd::mean(in, {0}, keepDims);
    };
    // Reduce over 1-dim input
    auto in = Tensor(fl::rand({6}), true);
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMean_0, in, 5E-3));
    // Reduce over scalar input
    auto inScalar = Tensor(fl::fromScalar(3.14), true);
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMean_0, inScalar, 5E-3));
  }
}

TEST(AutogradReductionTest, Variance) {
  std::vector<bool> biased = {true, false};
  for (auto b : biased) {
    for (const bool keepDims : {false, true}) {
      auto x = Tensor(fl::rand({5, 6, 7, 8}, fl::dtype::f64), true);

      auto expectedVar = fl::var(x, {1}, b, keepDims);
      auto calculatedVar = autograd::var(x, {1}, b, keepDims);
      ASSERT_TRUE(allClose(calculatedVar, expectedVar));

      auto funcVar = [b, keepDims](Tensor& in) {
        return autograd::var(in, {1, 2}, b, keepDims);
      };
      ASSERT_TRUE(fl::detail::jacobianTestImpl(funcVar, x, 1E-5, 1E-5));
    }
  }
}

TEST(AutogradReductionTest, Norm) {
  auto x = Tensor(fl::rand({5, 3}, fl::dtype::f64), true);
  for (const bool keepDims : {false, true}) {
    auto funcNorm2 = [keepDims](Tensor& in) {
      return autograd::norm(in, {1}, 2, keepDims);
    };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcNorm2, x, 1E-4));
    auto funcNorm1 = [keepDims](Tensor& in) {
      return autograd::norm(in, {1}, 1, keepDims);
    };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcNorm1, x, 1E-4));
    auto funcNorm3 = [keepDims](Tensor& in) {
      return autograd::norm(in, {1}, 3, keepDims);
    };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcNorm3, x, 1E-4));
  }
}

TEST(AutogradReductionTest, Std) {
  auto x = Tensor(fl::rand({5, 6, 7}, fl::dtype::f64), true);
  for (auto biased : {true, false}) {
    for (const bool keepDims : {false, true}) {
      auto funcStd = [biased, keepDims](Tensor& in) {
        return autograd::std(in, {1}, biased, keepDims);
      };
      ASSERT_TRUE(fl::detail::jacobianTestImpl(funcStd, x, 1E-4));
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  flTestInit();
  return RUN_ALL_TESTS();
}
