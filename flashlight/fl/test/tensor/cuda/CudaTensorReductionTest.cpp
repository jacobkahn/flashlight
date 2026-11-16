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

class CudaTensorReductionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fl::setDefaultJitTensorType<CudaTensor>();
  }
};

TEST_F(CudaTensorReductionTest, Sum) {
  auto t = fl::full({3, 4}, 1.0f);
  auto total = fl::sum(t, {});
  ASSERT_NEAR(total.scalarSync<float>(), 12.0f, 1e-5);

  auto rowSum = fl::sum(t, {1});  // sum along columns -> [3]
  ASSERT_EQ(rowSum.shape(), Shape({3}));
  std::vector<float> host(3);
  rowSum.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 4.0f, 1e-5);
  }
}

TEST_F(CudaTensorReductionTest, SumKeepDims) {
  auto t = fl::full({3, 4}, 2.0f);
  auto result = fl::sum(t, {1}, true);
  ASSERT_EQ(result.shape(), Shape({3, 1}));
  std::vector<float> host(3);
  result.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 8.0f, 1e-5);
  }
}

TEST_F(CudaTensorReductionTest, Mean) {
  auto vals = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto t = Tensor::fromVector(Shape{2, 3}, vals);
  auto total = fl::mean(t, {});
  ASSERT_NEAR(total.scalarSync<float>(), 3.5f, 1e-5);

  auto rowMean = fl::mean(t, {1});  // mean along dim 1
  ASSERT_EQ(rowMean.shape(), Shape({2}));
  std::vector<float> host(2);
  rowMean.hostSync(host.data());
  ASSERT_NEAR(host[0], 2.0f, 1e-5);  // (1+2+3)/3
  ASSERT_NEAR(host[1], 5.0f, 1e-5);  // (4+5+6)/3
}

TEST_F(CudaTensorReductionTest, AminAmax) {
  auto vals = std::vector<float>{3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f};
  auto t = Tensor::fromVector(Shape{2, 3}, vals);

  auto mn = fl::amin(t, {});
  auto mx = fl::amax(t, {});
  ASSERT_NEAR(mn.scalarSync<float>(), 1.0f, 1e-5);
  ASSERT_NEAR(mx.scalarSync<float>(), 9.0f, 1e-5);

  auto rowMin = fl::amin(t, {1});
  std::vector<float> host(2);
  rowMin.hostSync(host.data());
  ASSERT_NEAR(host[0], 1.0f, 1e-5);
  ASSERT_NEAR(host[1], 1.0f, 1e-5);
}

TEST_F(CudaTensorReductionTest, ArgmaxArgmin) {
  auto vals = std::vector<float>{3.0f, 1.0f, 4.0f, 5.0f, 2.0f, 6.0f};
  auto t = Tensor::fromVector(Shape{2, 3}, vals);

  auto amx = fl::argmax(t, 1);
  auto amn = fl::argmin(t, 1);
  ASSERT_EQ(amx.shape(), Shape({2}));
  std::vector<int> mxh(2), mnh(2);
  amx.astype(fl::dtype::s32).hostSync(mxh.data());
  amn.astype(fl::dtype::s32).hostSync(mnh.data());
  ASSERT_EQ(mxh[0], 2);  // index of 4 in [3,1,4]
  ASSERT_EQ(mxh[1], 2);  // index of 6 in [5,2,6]
  ASSERT_EQ(mnh[0], 1);  // index of 1 in [3,1,4]
  ASSERT_EQ(mnh[1], 1);  // index of 2 in [5,2,6]
}

TEST_F(CudaTensorReductionTest, Var) {
  auto vals = std::vector<float>{2.0f, 4.0f, 4.0f, 4.0f, 5.0f, 5.0f, 7.0f, 9.0f};
  auto t = Tensor::fromVector(Shape{8}, vals);
  auto v = fl::var(t, {});
  // Population variance (bias=false): mean=5, var = sum((x-5)^2)/8 = 32/8 = 4
  ASSERT_NEAR(v.scalarSync<float>(), 32.0f / 8.0f, 1e-4);
  // Sample variance (bias=true): divide by n-1
  auto sv = fl::var(t, {}, /*bias=*/true);
  ASSERT_NEAR(sv.scalarSync<float>(), 32.0f / 7.0f, 1e-4);
}

TEST_F(CudaTensorReductionTest, CountNonzero) {
  auto vals = std::vector<float>{0.0f, 1.0f, 0.0f, 3.0f, 0.0f};
  auto t = Tensor::fromVector(Shape{5}, vals);
  auto count = fl::countNonzero(t, {});
  auto asF32 = count.astype(fl::dtype::f32);
  ASSERT_NEAR(asF32.scalarSync<float>(), 2.0f, 1e-5);
}

TEST_F(CudaTensorReductionTest, AnyAll) {
  auto zeros = fl::full({4}, 0.0f);
  auto ones = fl::full({4}, 1.0f);
  auto mixed = Tensor::fromVector(Shape{4}, std::vector<float>{0.0f, 0.0f, 1.0f, 0.0f});

  auto anyZeros = fl::any(zeros, {}).astype(fl::dtype::f32);
  auto anyOnes = fl::any(ones, {}).astype(fl::dtype::f32);
  auto anyMixed = fl::any(mixed, {}).astype(fl::dtype::f32);
  auto allZeros = fl::all(zeros, {}).astype(fl::dtype::f32);
  auto allOnes = fl::all(ones, {}).astype(fl::dtype::f32);
  auto allMixed = fl::all(mixed, {}).astype(fl::dtype::f32);

  ASSERT_FLOAT_EQ(anyZeros.scalarSync<float>(), 0.0f);
  ASSERT_FLOAT_EQ(anyOnes.scalarSync<float>(), 1.0f);
  ASSERT_FLOAT_EQ(anyMixed.scalarSync<float>(), 1.0f);
  ASSERT_FLOAT_EQ(allZeros.scalarSync<float>(), 0.0f);
  ASSERT_FLOAT_EQ(allOnes.scalarSync<float>(), 1.0f);
  ASSERT_FLOAT_EQ(allMixed.scalarSync<float>(), 0.0f);
}

TEST_F(CudaTensorReductionTest, Cumsum) {
  auto vals = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto t = Tensor::fromVector(Shape{2, 3}, vals);
  auto cs = fl::cumsum(t, 1);  // cumsum along dim 1
  ASSERT_EQ(cs.shape(), Shape({2, 3}));
  std::vector<float> host(6);
  cs.hostSync(host.data());
  // Row 0: 1, 1+2=3, 1+2+3=6
  ASSERT_NEAR(host[0], 1.0f, 1e-5);
  ASSERT_NEAR(host[1], 3.0f, 1e-5);
  ASSERT_NEAR(host[2], 6.0f, 1e-5);
  // Row 1: 4, 4+5=9, 4+5+6=15
  ASSERT_NEAR(host[3], 4.0f, 1e-5);
  ASSERT_NEAR(host[4], 9.0f, 1e-5);
  ASSERT_NEAR(host[5], 15.0f, 1e-5);
}

TEST_F(CudaTensorReductionTest, NonContiguousAxisReduction) {
  // Reduce axes {0, 2} of shape {2, 3, 4} -- non-contiguous axes
  auto vals = std::vector<float>(24);
  for (int i = 0; i < 24; ++i) {
    vals[i] = static_cast<float>(i + 1);
  }
  auto t = Tensor::fromVector(Shape{2, 3, 4}, vals);
  auto result = fl::sum(t, {0, 2});
  ASSERT_EQ(result.shape(), Shape({3}));
  std::vector<float> host(3);
  result.hostSync(host.data());
  // dim0=2, dim2=4 reduced, dim1=3 kept
  // result[j] = sum over i=0..1, k=0..3 of input[i][j][k]
  // result[0] = (1+2+3+4) + (13+14+15+16) = 10 + 58 = 68
  // result[1] = (5+6+7+8) + (17+18+19+20) = 26 + 74 = 100
  // result[2] = (9+10+11+12) + (21+22+23+24) = 42 + 90 = 132
  ASSERT_NEAR(host[0], 68.0f, 1e-5);
  ASSERT_NEAR(host[1], 100.0f, 1e-5);
  ASSERT_NEAR(host[2], 132.0f, 1e-5);
}

TEST_F(CudaTensorReductionTest, MedianFull) {
  auto vals = std::vector<float>{5, 1, 3, 2, 4};
  auto t = Tensor::fromVector(Shape{5}, vals);
  auto med = fl::median(t, {});
  ASSERT_NEAR(med.scalarSync<float>(), 3.0f, 1e-5);
}

TEST_F(CudaTensorReductionTest, MedianAlongAxis) {
  // [[3, 1, 4], [1, 5, 9]]  shape [2, 3], median along axis 1
  auto vals = std::vector<float>{3, 1, 4, 1, 5, 9};
  auto t = Tensor::fromVector(Shape{2, 3}, vals);
  auto med = fl::median(t, {1});
  ASSERT_EQ(med.shape(), Shape({2}));
  std::vector<float> host(2);
  med.hostSync(host.data());
  // Row 0 sorted: [1, 3, 4] -> median = 3
  // Row 1 sorted: [1, 5, 9] -> median = 5
  ASSERT_NEAR(host[0], 3.0f, 1e-5);
  ASSERT_NEAR(host[1], 5.0f, 1e-5);
}

TEST_F(CudaTensorReductionTest, MedianMultiAxis) {
  // Shape {2, 3, 4} -- reduce axes {0, 2} (non-contiguous)
  // For each j in [0,3): median over i in [0,2), k in [0,4) = 8 values
  auto vals = std::vector<float>(24);
  for (int i = 0; i < 24; ++i) {
    vals[i] = static_cast<float>(i + 1);
  }
  auto t = Tensor::fromVector(Shape{2, 3, 4}, vals);
  auto med = fl::median(t, {0, 2});
  ASSERT_EQ(med.shape(), Shape({3}));
  std::vector<float> host(3);
  med.hostSync(host.data());
  // j=0: values = {1,2,3,4, 13,14,15,16} sorted = {1..4,13..16}, median = (4+13)/2 = 8.5
  // j=1: values = {5,6,7,8, 17,18,19,20} sorted = {5..8,17..20}, median = (8+17)/2 = 12.5
  // j=2: values = {9,10,11,12, 21,22,23,24} sorted = {9..12,21..24}, median = (12+21)/2 = 16.5
  ASSERT_NEAR(host[0], 8.5f, 1e-5);
  ASSERT_NEAR(host[1], 12.5f, 1e-5);
  ASSERT_NEAR(host[2], 16.5f, 1e-5);
}

TEST_F(CudaTensorReductionTest, MedianMultiAxisContiguous) {
  // Shape {2, 3, 4} -- reduce axes {1, 2} (contiguous)
  auto vals = std::vector<float>(24);
  for (int i = 0; i < 24; ++i) {
    vals[i] = static_cast<float>(i + 1);
  }
  auto t = Tensor::fromVector(Shape{2, 3, 4}, vals);
  auto med = fl::median(t, {1, 2});
  ASSERT_EQ(med.shape(), Shape({2}));
  std::vector<float> host(2);
  med.hostSync(host.data());
  // i=0: values 1..12, median = (6+7)/2 = 6.5
  // i=1: values 13..24, median = (18+19)/2 = 18.5
  ASSERT_NEAR(host[0], 6.5f, 1e-5);
  ASSERT_NEAR(host[1], 18.5f, 1e-5);
}

TEST_F(CudaTensorReductionTest, MedianMultiAxisKeepDims) {
  auto t = fl::full({2, 3, 4}, 1.0f);
  auto med = fl::median(t, {0, 2}, /*keepDims=*/true);
  ASSERT_EQ(med.shape(), Shape({1, 3, 1}));
}

TEST_F(CudaTensorReductionTest, MedianFullMultidim) {
  // Full reduction on a multi-dimensional tensor
  auto vals = std::vector<float>{5, 1, 3, 2, 4, 6};
  auto t = Tensor::fromVector(Shape{2, 3}, vals);
  auto med = fl::median(t, {});
  // sorted: {1,2,3,4,5,6}, median = (3+4)/2 = 3.5
  ASSERT_NEAR(med.scalarSync<float>(), 3.5f, 1e-5);
}

TEST_F(CudaTensorReductionTest, F64Sum) {
  // f64 sum should preserve precision
  auto vals = std::vector<double>{1e15, 1.0, -1e15, 2.0};
  auto t = Tensor::fromVector(Shape{4}, vals);
  auto total = fl::sum(t, {});
  ASSERT_EQ(total.type(), fl::dtype::f64);
  std::vector<double> host(1);
  total.hostSync(host.data());
  // In f32, 1e15 + 1.0 would lose the 1.0. In f64, exact.
  ASSERT_DOUBLE_EQ(host[0], 3.0);
}

TEST_F(CudaTensorReductionTest, F64AxisSum) {
  auto vals = std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  auto t = Tensor::fromVector(Shape{2, 3}, vals);
  auto rowSum = fl::sum(t, {1});
  ASSERT_EQ(rowSum.type(), fl::dtype::f64);
  ASSERT_EQ(rowSum.shape(), Shape({2}));
  std::vector<double> host(2);
  rowSum.hostSync(host.data());
  ASSERT_DOUBLE_EQ(host[0], 6.0);
  ASSERT_DOUBLE_EQ(host[1], 15.0);
}

TEST_F(CudaTensorReductionTest, F64CumsumPrecision) {
  // cumsum with f64 for precision
  auto vals = std::vector<double>{1e15, 1.0, 1.0, 1.0};
  auto t = Tensor::fromVector(Shape{4}, vals);
  auto cs = fl::cumsum(t, 0);
  ASSERT_EQ(cs.type(), fl::dtype::f64);
  std::vector<double> host(4);
  cs.hostSync(host.data());
  ASSERT_DOUBLE_EQ(host[0], 1e15);
  ASSERT_DOUBLE_EQ(host[1], 1e15 + 1.0);
  ASSERT_DOUBLE_EQ(host[2], 1e15 + 2.0);
  ASSERT_DOUBLE_EQ(host[3], 1e15 + 3.0);
}

TEST_F(CudaTensorReductionTest, Bf16Sum) {
  auto t = fl::full({3, 4}, 1.0f, fl::dtype::bf16);
  auto total = fl::sum(t, {});
  // bf16 sum may upcast to f32 for accumulation
  auto totalF32 = total.astype(fl::dtype::f32);
  ASSERT_NEAR(totalF32.scalarSync<float>(), 12.0f, 0.5f);

  auto rowSum = fl::sum(t, {1});
  ASSERT_EQ(rowSum.shape(), Shape({3}));
  auto rowSumF32 = rowSum.astype(fl::dtype::f32);
  std::vector<float> host(3);
  rowSumF32.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 4.0f, 0.5f);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
