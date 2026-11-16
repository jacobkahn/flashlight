/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/backend/cuda/CudaTensor.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"

using namespace fl;

class CudaTensorUnaryOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fl::setDefaultJitTensorType<CudaTensor>();
  }
};

TEST_F(CudaTensorUnaryOpsTest, Exp) {
  auto t = fl::full({2, 3}, 1.0f);
  auto result = fl::exp(t);
  std::vector<float> host(6);
  result.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, std::exp(1.0f), 1e-5);
  }
}

TEST_F(CudaTensorUnaryOpsTest, Log) {
  auto t = fl::full({4}, std::exp(1.0f));
  auto result = fl::log(t);
  std::vector<float> host(4);
  result.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 1.0f, 1e-5);
  }
}

TEST_F(CudaTensorUnaryOpsTest, Sqrt) {
  auto t = fl::full({3}, 9.0f);
  auto result = fl::sqrt(t);
  std::vector<float> host(3);
  result.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 3.0f, 1e-5);
  }
}

TEST_F(CudaTensorUnaryOpsTest, Negative) {
  auto t = fl::full({3}, 5.0f);
  auto result = fl::negative(t);
  std::vector<float> host(3);
  result.hostSync(host.data());
  for (float v : host) {
    ASSERT_FLOAT_EQ(v, -5.0f);
  }
}

TEST_F(CudaTensorUnaryOpsTest, Absolute) {
  auto vals = std::vector<float>{-3.0f, -1.0f, 0.0f, 2.0f, 4.0f};
  auto t = Tensor::fromVector(Shape{5}, vals);
  auto result = fl::abs(t);
  std::vector<float> host(5);
  result.hostSync(host.data());
  ASSERT_FLOAT_EQ(host[0], 3.0f);
  ASSERT_FLOAT_EQ(host[1], 1.0f);
  ASSERT_FLOAT_EQ(host[2], 0.0f);
  ASSERT_FLOAT_EQ(host[3], 2.0f);
  ASSERT_FLOAT_EQ(host[4], 4.0f);
}

TEST_F(CudaTensorUnaryOpsTest, Sigmoid) {
  auto t = fl::full({3}, 0.0f);
  auto result = fl::sigmoid(t);
  std::vector<float> host(3);
  result.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 0.5f, 1e-5);
  }
}

TEST_F(CudaTensorUnaryOpsTest, Tanh) {
  auto t = fl::full({3}, 0.0f);
  auto result = fl::tanh(t);
  std::vector<float> host(3);
  result.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 0.0f, 1e-5);
  }
}

TEST_F(CudaTensorUnaryOpsTest, Sign) {
  auto vals = std::vector<float>{-2.0f, 0.0f, 3.0f};
  auto t = Tensor::fromVector(Shape{3}, vals);
  auto result = fl::sign(t);
  std::vector<float> host(3);
  result.hostSync(host.data());
  ASSERT_FLOAT_EQ(host[0], -1.0f);
  ASSERT_FLOAT_EQ(host[1], 0.0f);
  ASSERT_FLOAT_EQ(host[2], 1.0f);
}

TEST_F(CudaTensorUnaryOpsTest, FloorCeilRint) {
  auto vals = std::vector<float>{1.3f, 2.7f, -0.5f, 3.5f};
  auto t = Tensor::fromVector(Shape{4}, vals);

  auto fl_result = fl::floor(t);
  auto cl_result = fl::ceil(t);
  auto rn_result = fl::rint(t);

  std::vector<float> fh(4), ch(4), rh(4);
  fl_result.hostSync(fh.data());
  cl_result.hostSync(ch.data());
  rn_result.hostSync(rh.data());

  ASSERT_FLOAT_EQ(fh[0], 1.0f);
  ASSERT_FLOAT_EQ(fh[1], 2.0f);
  ASSERT_FLOAT_EQ(fh[2], -1.0f);
  ASSERT_FLOAT_EQ(ch[0], 2.0f);
  ASSERT_FLOAT_EQ(ch[1], 3.0f);
  ASSERT_FLOAT_EQ(ch[2], 0.0f);
}

TEST_F(CudaTensorUnaryOpsTest, IsnanIsinf) {
  auto vals = std::vector<float>{
      1.0f, std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(), 0.0f};
  auto t = Tensor::fromVector(Shape{4}, vals);

  auto nanResult = fl::isnan(t);
  auto infResult = fl::isinf(t);

  auto nanF32 = nanResult.astype(fl::dtype::f32);
  auto infF32 = infResult.astype(fl::dtype::f32);
  std::vector<float> nh(4), ih(4);
  nanF32.hostSync(nh.data());
  infF32.hostSync(ih.data());

  ASSERT_FLOAT_EQ(nh[0], 0.0f);
  ASSERT_FLOAT_EQ(nh[1], 1.0f);
  ASSERT_FLOAT_EQ(nh[2], 0.0f);
  ASSERT_FLOAT_EQ(ih[2], 1.0f);
  ASSERT_FLOAT_EQ(ih[0], 0.0f);
}

TEST_F(CudaTensorUnaryOpsTest, Clip) {
  auto vals = std::vector<float>{-5.0f, 0.0f, 3.0f, 10.0f};
  auto t = Tensor::fromVector(Shape{4}, vals);
  auto result = fl::clip(t, fl::full({}, 0.0f), fl::full({}, 5.0f));
  std::vector<float> host(4);
  result.hostSync(host.data());
  ASSERT_FLOAT_EQ(host[0], 0.0f);
  ASSERT_FLOAT_EQ(host[1], 0.0f);
  ASSERT_FLOAT_EQ(host[2], 3.0f);
  ASSERT_FLOAT_EQ(host[3], 5.0f);
}

TEST_F(CudaTensorUnaryOpsTest, LogicalNot) {
  auto vals = std::vector<float>{0.0f, 1.0f, 0.0f, 5.0f};
  auto t = Tensor::fromVector(Shape{4}, vals);
  auto result = fl::logicalNot(t);
  auto asF32 = result.astype(fl::dtype::f32);
  std::vector<float> host(4);
  asF32.hostSync(host.data());
  ASSERT_FLOAT_EQ(host[0], 1.0f);
  ASSERT_FLOAT_EQ(host[1], 0.0f);
  ASSERT_FLOAT_EQ(host[2], 1.0f);
  ASSERT_FLOAT_EQ(host[3], 0.0f);
}

TEST_F(CudaTensorUnaryOpsTest, Triangular) {
  auto vals = std::vector<float>{
      1, 2, 3, 4, 5, 6, 7, 8, 9,
      10, 11, 12, 13, 14, 15, 16, 17, 18};
  auto t = Tensor::fromVector(Shape{2, 3, 3}, vals);

  auto lower = fl::tril(t, 1);
  std::vector<float> lowerHost(vals.size());
  lower.hostSync(lowerHost.data());
  ASSERT_EQ(
      lowerHost,
      (std::vector<float>{
          1, 2, 0, 4, 5, 6, 7, 8, 9,
          10, 11, 0, 13, 14, 15, 16, 17, 18}));

  auto upper = fl::triu(t, -1);
  std::vector<float> upperHost(vals.size());
  upper.hostSync(upperHost.data());
  ASSERT_EQ(
      upperHost,
      (std::vector<float>{
          1, 2, 3, 4, 5, 6, 0, 8, 9,
          10, 11, 12, 13, 14, 15, 0, 17, 18}));
}

// ===================== Sort =====================

TEST_F(CudaTensorUnaryOpsTest, Sort1D) {
  auto vals = std::vector<float>{5, 1, 3, 2, 4};
  auto t = Tensor::fromVector(Shape{5}, vals);
  auto sorted = fl::sort(t, 0, SortMode::Ascending);
  std::vector<float> host(5);
  sorted.hostSync(host.data());
  ASSERT_FLOAT_EQ(host[0], 1.0f);
  ASSERT_FLOAT_EQ(host[1], 2.0f);
  ASSERT_FLOAT_EQ(host[2], 3.0f);
  ASSERT_FLOAT_EQ(host[3], 4.0f);
  ASSERT_FLOAT_EQ(host[4], 5.0f);
}

TEST_F(CudaTensorUnaryOpsTest, SortDescending) {
  auto vals = std::vector<float>{5, 1, 3, 2, 4};
  auto t = Tensor::fromVector(Shape{5}, vals);
  auto sorted = fl::sort(t, 0, SortMode::Descending);
  std::vector<float> host(5);
  sorted.hostSync(host.data());
  ASSERT_FLOAT_EQ(host[0], 5.0f);
  ASSERT_FLOAT_EQ(host[1], 4.0f);
  ASSERT_FLOAT_EQ(host[2], 3.0f);
  ASSERT_FLOAT_EQ(host[3], 2.0f);
  ASSERT_FLOAT_EQ(host[4], 1.0f);
}

TEST_F(CudaTensorUnaryOpsTest, SortAlongLastAxis) {
  // [[3, 1, 2], [6, 4, 5]]  shape [2, 3], sort along axis 1
  auto vals = std::vector<float>{3, 1, 2, 6, 4, 5};
  auto t = Tensor::fromVector(Shape{2, 3}, vals);
  auto sorted = fl::sort(t, 1, SortMode::Ascending);
  std::vector<float> host(6);
  sorted.hostSync(host.data());
  // Row 0: [1, 2, 3], Row 1: [4, 5, 6]
  ASSERT_FLOAT_EQ(host[0], 1.0f);
  ASSERT_FLOAT_EQ(host[1], 2.0f);
  ASSERT_FLOAT_EQ(host[2], 3.0f);
  ASSERT_FLOAT_EQ(host[3], 4.0f);
  ASSERT_FLOAT_EQ(host[4], 5.0f);
  ASSERT_FLOAT_EQ(host[5], 6.0f);
}

TEST_F(CudaTensorUnaryOpsTest, SortAlongFirstAxis) {
  // [[3, 1], [2, 4]]  shape [2, 2], sort along axis 0
  auto vals = std::vector<float>{3, 1, 2, 4};
  auto t = Tensor::fromVector(Shape{2, 2}, vals);
  auto sorted = fl::sort(t, 0, SortMode::Ascending);
  std::vector<float> host(4);
  sorted.hostSync(host.data());
  // Col 0: [2, 3], Col 1: [1, 4]
  ASSERT_FLOAT_EQ(host[0], 2.0f);
  ASSERT_FLOAT_EQ(host[1], 1.0f);
  ASSERT_FLOAT_EQ(host[2], 3.0f);
  ASSERT_FLOAT_EQ(host[3], 4.0f);
}

TEST_F(CudaTensorUnaryOpsTest, SortWithIndices) {
  auto vals = std::vector<float>{5, 1, 3};
  auto t = Tensor::fromVector(Shape{3}, vals);
  Tensor sortedVals, sortedIdx;
  fl::sort(sortedVals, sortedIdx, t, 0, SortMode::Ascending);
  std::vector<float> vh(3);
  std::vector<int> ih(3);
  sortedVals.hostSync(vh.data());
  sortedIdx.astype(fl::dtype::s32).hostSync(ih.data());
  ASSERT_FLOAT_EQ(vh[0], 1.0f);
  ASSERT_FLOAT_EQ(vh[1], 3.0f);
  ASSERT_FLOAT_EQ(vh[2], 5.0f);
  ASSERT_EQ(ih[0], 1);  // 1 was at index 1
  ASSERT_EQ(ih[1], 2);  // 3 was at index 2
  ASSERT_EQ(ih[2], 0);  // 5 was at index 0
}

TEST_F(CudaTensorUnaryOpsTest, Argsort) {
  auto vals = std::vector<float>{5, 1, 3};
  auto t = Tensor::fromVector(Shape{3}, vals);
  auto idx = fl::argsort(t, 0, SortMode::Ascending);
  std::vector<int> ih(3);
  idx.astype(fl::dtype::s32).hostSync(ih.data());
  ASSERT_EQ(ih[0], 1);
  ASSERT_EQ(ih[1], 2);
  ASSERT_EQ(ih[2], 0);
}

TEST_F(CudaTensorUnaryOpsTest, Topk) {
  auto vals = std::vector<float>{5, 1, 3, 2, 4};
  auto t = Tensor::fromVector(Shape{5}, vals);
  Tensor topkVals, topkIdx;
  fl::topk(topkVals, topkIdx, t, 3, 0, SortMode::Descending);
  ASSERT_EQ(topkVals.shape(), Shape({3}));
  std::vector<float> vh(3);
  std::vector<int> ih(3);
  topkVals.hostSync(vh.data());
  topkIdx.astype(fl::dtype::s32).hostSync(ih.data());
  ASSERT_FLOAT_EQ(vh[0], 5.0f);
  ASSERT_FLOAT_EQ(vh[1], 4.0f);
  ASSERT_FLOAT_EQ(vh[2], 3.0f);
  ASSERT_EQ(ih[0], 0);  // 5 was at index 0
  ASSERT_EQ(ih[1], 4);  // 4 was at index 4
  ASSERT_EQ(ih[2], 2);  // 3 was at index 2
}

TEST_F(CudaTensorUnaryOpsTest, TopkAlongAxis) {
  // [[5, 1, 3], [2, 4, 6]]  shape [2, 3], topk=2 along axis 1 descending
  auto vals = std::vector<float>{5, 1, 3, 2, 4, 6};
  auto t = Tensor::fromVector(Shape{2, 3}, vals);
  Tensor topkVals, topkIdx;
  fl::topk(topkVals, topkIdx, t, 2, 1, SortMode::Descending);
  ASSERT_EQ(topkVals.shape(), Shape({2, 2}));
  std::vector<float> vh(4);
  std::vector<int> ih(4);
  topkVals.hostSync(vh.data());
  topkIdx.astype(fl::dtype::s32).hostSync(ih.data());
  // Row 0: top-2 desc = [5, 3], indices [0, 2]
  ASSERT_FLOAT_EQ(vh[0], 5.0f);
  ASSERT_FLOAT_EQ(vh[1], 3.0f);
  // Row 1: top-2 desc = [6, 4], indices [2, 1]
  ASSERT_FLOAT_EQ(vh[2], 6.0f);
  ASSERT_FLOAT_EQ(vh[3], 4.0f);
  ASSERT_EQ(ih[0], 0);
  ASSERT_EQ(ih[1], 2);
  ASSERT_EQ(ih[2], 2);
  ASSERT_EQ(ih[3], 1);
}

TEST_F(CudaTensorUnaryOpsTest, F64UnaryOps) {
  // Verify f64 unary ops compute natively in double precision
  auto t = fl::full({4}, 1.0f, fl::dtype::f64);
  auto result = fl::exp(t);
  ASSERT_EQ(result.type(), fl::dtype::f64);
  std::vector<double> host(4);
  result.hostSync(host.data());
  for (double v : host) {
    ASSERT_NEAR(v, std::exp(1.0), 1e-12);  // f64 precision
  }

  // sqrt
  auto sq = fl::full({4}, 2.0f, fl::dtype::f64);
  auto sqr = fl::sqrt(sq);
  ASSERT_EQ(sqr.type(), fl::dtype::f64);
  sqr.hostSync(host.data());
  for (double v : host) {
    ASSERT_NEAR(v, std::sqrt(2.0), 1e-12);
  }

  // log
  auto lg = fl::log(sq);
  ASSERT_EQ(lg.type(), fl::dtype::f64);
  lg.hostSync(host.data());
  for (double v : host) {
    ASSERT_NEAR(v, std::log(2.0), 1e-12);
  }
}

TEST_F(CudaTensorUnaryOpsTest, F64Precision) {
  // Verify that f64 preserves precision that f32 would lose
  // exp(30) = 1.0686474581524e+13 -- needs f64 for full mantissa
  auto t = Tensor::fromVector(Shape{1}, std::vector<double>{30.0});
  auto result = fl::exp(t);
  ASSERT_EQ(result.type(), fl::dtype::f64);
  std::vector<double> host(1);
  result.hostSync(host.data());
  double expected = std::exp(30.0);
  // Relative error should be < 1e-14 for f64
  ASSERT_NEAR(host[0] / expected, 1.0, 1e-14);
}

TEST_F(CudaTensorUnaryOpsTest, Bf16UnaryOps) {
  // exp, log, sqrt on bf16 -- upcast to f32 for compute, result in bf16
  auto t = fl::full({4}, 1.0f, fl::dtype::bf16);
  auto expR = fl::exp(t);
  ASSERT_EQ(expR.type(), fl::dtype::bf16);
  auto expF32 = expR.astype(fl::dtype::f32);
  std::vector<float> host(4);
  expF32.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, std::exp(1.0f), 0.05f);
  }

  auto sq = fl::full({4}, 9.0f, fl::dtype::bf16);
  auto sqrtR = fl::sqrt(sq);
  ASSERT_EQ(sqrtR.type(), fl::dtype::bf16);
  auto sqrtF32 = sqrtR.astype(fl::dtype::f32);
  sqrtF32.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 3.0f, 0.05f);
  }

  auto negR = fl::negative(t);
  ASSERT_EQ(negR.type(), fl::dtype::bf16);
  auto negF32 = negR.astype(fl::dtype::f32);
  negF32.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, -1.0f, 0.05f);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
