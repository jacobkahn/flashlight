/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <vector>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/backend/cuda/CudaTensor.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"

using namespace fl;

class CudaTensorBinaryOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fl::setDefaultJitTensorType<CudaTensor>();
  }
};

TEST_F(CudaTensorBinaryOpsTest, AddSubMulDiv) {
  auto a = fl::full({3}, 6.0f);
  auto b = fl::full({3}, 2.0f);

  auto add_r = a + b;
  auto sub_r = a - b;
  auto mul_r = a * b;
  auto div_r = a / b;

  std::vector<float> ah(3), sh(3), mh(3), dh(3);
  add_r.hostSync(ah.data());
  sub_r.hostSync(sh.data());
  mul_r.hostSync(mh.data());
  div_r.hostSync(dh.data());

  for (int i = 0; i < 3; ++i) {
    ASSERT_FLOAT_EQ(ah[i], 8.0f);
    ASSERT_FLOAT_EQ(sh[i], 4.0f);
    ASSERT_FLOAT_EQ(mh[i], 12.0f);
    ASSERT_FLOAT_EQ(dh[i], 3.0f);
  }
}

TEST_F(CudaTensorBinaryOpsTest, BinaryBroadcast) {
  // [3, 1] + [1, 4] -> [3, 4]
  auto a = Tensor::fromVector(Shape{3, 1}, std::vector<float>{1.0f, 2.0f, 3.0f});
  auto b = Tensor::fromVector(Shape{1, 4}, std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f});
  auto result = a + b;
  ASSERT_EQ(result.shape(), Shape({3, 4}));

  std::vector<float> host(12);
  result.hostSync(host.data());
  // Row-major: result[r][c] = a[r] + b[c]
  ASSERT_FLOAT_EQ(host[0], 11.0f);   // 1+10
  ASSERT_FLOAT_EQ(host[1], 21.0f);   // 1+20
  ASSERT_FLOAT_EQ(host[2], 31.0f);   // 1+30
  ASSERT_FLOAT_EQ(host[3], 41.0f);   // 1+40
  ASSERT_FLOAT_EQ(host[4], 12.0f);   // 2+10
  ASSERT_FLOAT_EQ(host[5], 22.0f);   // 2+20
  ASSERT_FLOAT_EQ(host[8], 13.0f);   // 3+10
  ASSERT_FLOAT_EQ(host[11], 43.0f);  // 3+40
}

TEST_F(CudaTensorBinaryOpsTest, BinaryBroadcastRankMismatch) {
  // [4] + [3, 4] -> [3, 4]
  auto a = Tensor::fromVector(Shape{4}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  auto b = fl::full({3, 4}, 10.0f);
  auto result = a + b;
  ASSERT_EQ(result.shape(), Shape({3, 4}));

  std::vector<float> host(12);
  result.hostSync(host.data());
  // a broadcasts along dim 0: each row gets a + 10
  ASSERT_FLOAT_EQ(host[0], 11.0f);
  ASSERT_FLOAT_EQ(host[1], 12.0f);
  ASSERT_FLOAT_EQ(host[4], 11.0f);
}

TEST_F(CudaTensorBinaryOpsTest, Comparison) {
  auto a = Tensor::fromVector(Shape{4}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  auto b = Tensor::fromVector(Shape{4}, std::vector<float>{2.0f, 2.0f, 2.0f, 2.0f});

  auto lt = a < b;
  auto eq = a == b;
  auto gt = a > b;

  auto ltF32 = lt.astype(fl::dtype::f32);
  auto eqF32 = eq.astype(fl::dtype::f32);
  auto gtF32 = gt.astype(fl::dtype::f32);
  std::vector<float> lh(4), eh(4), gh(4);
  ltF32.hostSync(lh.data());
  eqF32.hostSync(eh.data());
  gtF32.hostSync(gh.data());

  ASSERT_FLOAT_EQ(lh[0], 1.0f);  // 1 < 2
  ASSERT_FLOAT_EQ(lh[1], 0.0f);  // 2 < 2 = false
  ASSERT_FLOAT_EQ(eh[1], 1.0f);  // 2 == 2
  ASSERT_FLOAT_EQ(gh[2], 1.0f);  // 3 > 2
  ASSERT_FLOAT_EQ(gh[3], 1.0f);  // 4 > 2
}

TEST_F(CudaTensorBinaryOpsTest, MinimumMaximum) {
  auto a = Tensor::fromVector(Shape{4}, std::vector<float>{1.0f, 5.0f, 3.0f, 7.0f});
  auto b = Tensor::fromVector(Shape{4}, std::vector<float>{4.0f, 2.0f, 6.0f, 0.0f});

  auto mn = fl::minimum(a, b);
  auto mx = fl::maximum(a, b);
  std::vector<float> mnh(4), mxh(4);
  mn.hostSync(mnh.data());
  mx.hostSync(mxh.data());

  ASSERT_FLOAT_EQ(mnh[0], 1.0f);
  ASSERT_FLOAT_EQ(mnh[1], 2.0f);
  ASSERT_FLOAT_EQ(mxh[0], 4.0f);
  ASSERT_FLOAT_EQ(mxh[1], 5.0f);
}

TEST_F(CudaTensorBinaryOpsTest, Power) {
  auto base = fl::full({3}, 2.0f);
  auto exp = Tensor::fromVector(Shape{3}, std::vector<float>{1.0f, 2.0f, 3.0f});
  auto result = fl::power(base, exp);
  std::vector<float> host(3);
  result.hostSync(host.data());
  ASSERT_NEAR(host[0], 2.0f, 1e-5);
  ASSERT_NEAR(host[1], 4.0f, 1e-5);
  ASSERT_NEAR(host[2], 8.0f, 1e-5);
}

TEST_F(CudaTensorBinaryOpsTest, F64BinaryOps) {
  // f64 binary ops should compute natively in double precision
  auto a = Tensor::fromVector(Shape{3}, std::vector<double>{1.0, 2.0, 3.0});
  auto b = Tensor::fromVector(Shape{3}, std::vector<double>{4.0, 5.0, 6.0});
  auto sum = fl::add(a, b);
  ASSERT_EQ(sum.type(), fl::dtype::f64);
  std::vector<double> host(3);
  sum.hostSync(host.data());
  ASSERT_DOUBLE_EQ(host[0], 5.0);
  ASSERT_DOUBLE_EQ(host[1], 7.0);
  ASSERT_DOUBLE_EQ(host[2], 9.0);
}

TEST_F(CudaTensorBinaryOpsTest, F64BinaryPrecision) {
  // Verify f64 precision in binary ops
  // (1e15 + 1.0) - 1e15 should be exactly 1.0 in f64, but 0.0 in f32
  auto big = Tensor::fromVector(Shape{1}, std::vector<double>{1e15});
  auto one = Tensor::fromVector(Shape{1}, std::vector<double>{1.0});
  auto sum = fl::add(big, one);
  auto diff = fl::sub(sum, big);
  ASSERT_EQ(diff.type(), fl::dtype::f64);
  std::vector<double> host(1);
  diff.hostSync(host.data());
  ASSERT_DOUBLE_EQ(host[0], 1.0);
}

TEST_F(CudaTensorBinaryOpsTest, F64BroadcastBinary) {
  // f64 broadcasting
  auto a = Tensor::fromVector(Shape{2, 3}, std::vector<double>{1, 2, 3, 4, 5, 6});
  auto b = Tensor::fromVector(Shape{1, 3}, std::vector<double>{10, 20, 30});
  auto result = fl::add(a, b);
  ASSERT_EQ(result.type(), fl::dtype::f64);
  ASSERT_EQ(result.shape(), Shape({2, 3}));
  std::vector<double> host(6);
  result.hostSync(host.data());
  ASSERT_DOUBLE_EQ(host[0], 11.0);
  ASSERT_DOUBLE_EQ(host[1], 22.0);
  ASSERT_DOUBLE_EQ(host[2], 33.0);
  ASSERT_DOUBLE_EQ(host[3], 14.0);
  ASSERT_DOUBLE_EQ(host[4], 25.0);
  ASSERT_DOUBLE_EQ(host[5], 36.0);
}

TEST_F(CudaTensorBinaryOpsTest, MixedF32F64Promotion) {
  // f32 + f64 should promote to f64
  auto a = fl::full({3}, 1.0f);
  auto b = Tensor::fromVector(Shape{3}, std::vector<double>{2.0, 3.0, 4.0});
  auto result = fl::add(a, b);
  ASSERT_EQ(result.type(), fl::dtype::f64);
  std::vector<double> host(3);
  result.hostSync(host.data());
  ASSERT_DOUBLE_EQ(host[0], 3.0);
  ASSERT_DOUBLE_EQ(host[1], 4.0);
  ASSERT_DOUBLE_EQ(host[2], 5.0);
}

TEST_F(CudaTensorBinaryOpsTest, Bf16BinaryOps) {
  // bf16 arithmetic: add, sub, mul, div
  auto a = fl::full({4}, 6.0f, fl::dtype::bf16);
  auto b = fl::full({4}, 2.0f, fl::dtype::bf16);

  auto add_r = (a + b).astype(fl::dtype::f32);
  auto sub_r = (a - b).astype(fl::dtype::f32);
  auto mul_r = (a * b).astype(fl::dtype::f32);
  auto div_r = (a / b).astype(fl::dtype::f32);

  std::vector<float> ah(4), sh(4), mh(4), dh(4);
  add_r.hostSync(ah.data());
  sub_r.hostSync(sh.data());
  mul_r.hostSync(mh.data());
  div_r.hostSync(dh.data());

  for (int i = 0; i < 4; ++i) {
    ASSERT_NEAR(ah[i], 8.0f, 0.1f);
    ASSERT_NEAR(sh[i], 4.0f, 0.1f);
    ASSERT_NEAR(mh[i], 12.0f, 0.1f);
    ASSERT_NEAR(dh[i], 3.0f, 0.1f);
  }
}

TEST_F(CudaTensorBinaryOpsTest, MixedBf16F16Compute) {
  // bf16 + f16: the backend upcasts both to f32 for compute.
  // Output type follows backend promotion rules (may be bf16 or f32).
  auto a = fl::full({3}, 1.0f).astype(fl::dtype::bf16);
  auto b = fl::full({3}, 2.0f).astype(fl::dtype::f16);
  auto result = fl::add(a, b);
  // Verify the result values are correct regardless of output type
  auto asF32 = result.astype(fl::dtype::f32);
  std::vector<float> host(3);
  asF32.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 3.0f, 0.1f);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
