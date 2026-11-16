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

class CudaTensorBLASTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fl::setDefaultJitTensorType<CudaTensor>();
  }
};

TEST_F(CudaTensorBLASTest, Matmul) {
  // [2, 3] x [3, 4] -> [2, 4]
  auto a = fl::full({2, 3}, 1.0f);
  auto b = fl::full({3, 4}, 2.0f);
  auto c = fl::matmul(a, b);
  ASSERT_EQ(c.shape(), Shape({2, 4}));
  std::vector<float> host(8);
  c.hostSync(host.data());
  // Each element = sum of 3 products of 1*2 = 6
  for (float v : host) {
    ASSERT_NEAR(v, 6.0f, 1e-5);
  }
}

TEST_F(CudaTensorBLASTest, MatmulNonTrivial) {
  // A = [[1, 2, 3], [4, 5, 6]] (2x3)
  // B = [[1, 0], [0, 1], [1, 1]] (3x2)
  // C = A * B = [[1+0+3, 0+2+3], [4+0+6, 0+5+6]] = [[4, 5], [10, 11]]
  auto a = Tensor::fromVector(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  auto b = Tensor::fromVector(Shape{3, 2}, std::vector<float>{1, 0, 0, 1, 1, 1});
  auto c = fl::matmul(a, b);
  ASSERT_EQ(c.shape(), Shape({2, 2}));
  std::vector<float> host(4);
  c.hostSync(host.data());
  ASSERT_NEAR(host[0], 4.0f, 1e-5);
  ASSERT_NEAR(host[1], 5.0f, 1e-5);
  ASSERT_NEAR(host[2], 10.0f, 1e-5);
  ASSERT_NEAR(host[3], 11.0f, 1e-5);
}

TEST_F(CudaTensorBLASTest, MatmulTranspose) {
  // A = [[1, 2], [3, 4]] (2x2), B = [[5, 6], [7, 8]] (2x2)
  // A^T * B = [[1,3],[2,4]] * [[5,6],[7,8]] = [[26,30],[38,44]]
  auto a = Tensor::fromVector(Shape{2, 2}, std::vector<float>{1, 2, 3, 4});
  auto b = Tensor::fromVector(Shape{2, 2}, std::vector<float>{5, 6, 7, 8});
  auto c = fl::matmul(a, b, MatrixProperty::Transpose, MatrixProperty::None);
  ASSERT_EQ(c.shape(), Shape({2, 2}));
  std::vector<float> host(4);
  c.hostSync(host.data());
  ASSERT_NEAR(host[0], 26.0f, 1e-5);
  ASSERT_NEAR(host[1], 30.0f, 1e-5);
  ASSERT_NEAR(host[2], 38.0f, 1e-5);
  ASSERT_NEAR(host[3], 44.0f, 1e-5);
}

TEST_F(CudaTensorBLASTest, MatmulF64) {
  // f64 matmul via cublasDgemm -- precision preservation
  auto a = Tensor::fromVector(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  auto b = Tensor::fromVector(Shape{3, 2}, std::vector<float>{1, 0, 0, 1, 1, 1});
  auto aD = a.astype(fl::dtype::f64);
  auto bD = b.astype(fl::dtype::f64);
  auto c = fl::matmul(aD, bD);
  ASSERT_EQ(c.type(), fl::dtype::f64);
  ASSERT_EQ(c.shape(), Shape({2, 2}));
  auto cF32 = c.astype(fl::dtype::f32);
  std::vector<float> host(4);
  cF32.hostSync(host.data());
  ASSERT_NEAR(host[0], 4.0f, 1e-5);
  ASSERT_NEAR(host[1], 5.0f, 1e-5);
  ASSERT_NEAR(host[2], 10.0f, 1e-5);
  ASSERT_NEAR(host[3], 11.0f, 1e-5);
}

TEST_F(CudaTensorBLASTest, MatmulF64Precision) {
  // Verify f64 preserves precision that f32 would lose.
  // Use values that are exact in f64 but lose precision in f32.
  double bigVal = 1e15;
  double smallVal = 1.0;
  // [1] x [1] matmul (really scalar)
  auto a = fl::full({1, 1}, 0.0f, fl::dtype::f64);
  auto b = fl::full({1, 1}, 0.0f, fl::dtype::f64);
  // Set via fromVector
  auto aVec = Tensor::fromVector(Shape{1, 2}, std::vector<double>{bigVal, smallVal});
  auto bVec = Tensor::fromVector(Shape{2, 1}, std::vector<double>{1.0, 1.0});
  auto c = fl::matmul(aVec, bVec);
  ASSERT_EQ(c.type(), fl::dtype::f64);
  std::vector<double> host(1);
  c.astype(fl::dtype::f64).hostSync(host.data());
  // bigVal + smallVal should be exact in f64
  ASSERT_DOUBLE_EQ(host[0], bigVal + smallVal);
}

TEST_F(CudaTensorBLASTest, MatmulF16) {
  // f16 matmul via cublasGemmEx
  auto a = fl::full({2, 3}, 1.0f, fl::dtype::f16);
  auto b = fl::full({3, 4}, 2.0f, fl::dtype::f16);
  auto c = fl::matmul(a, b);
  ASSERT_EQ(c.type(), fl::dtype::f16);
  ASSERT_EQ(c.shape(), Shape({2, 4}));
  auto cF32 = c.astype(fl::dtype::f32);
  std::vector<float> host(8);
  cF32.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 6.0f, 0.1f);  // f16 has limited precision
  }
}

TEST_F(CudaTensorBLASTest, MatmulMixedTypes) {
  // f32 x f64 should promote to f64
  auto a = fl::full({2, 3}, 1.0f);
  auto b = fl::full({3, 2}, 2.0f, fl::dtype::f64);
  auto c = fl::matmul(a, b);
  ASSERT_EQ(c.type(), fl::dtype::f64);
  auto cF32 = c.astype(fl::dtype::f32);
  std::vector<float> host(4);
  cF32.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 6.0f, 1e-5);
  }
}

TEST_F(CudaTensorBLASTest, MatmulBf16) {
  // bf16 matmul via cublasGemmEx -- cuBLAS accumulates in f32
  auto a = fl::full({2, 3}, 1.0f, fl::dtype::bf16);
  auto b = fl::full({3, 4}, 2.0f, fl::dtype::bf16);
  auto c = fl::matmul(a, b);
  // cuBLAS bf16 matmul outputs f32 (f32 accumulator)
  ASSERT_TRUE(c.type() == fl::dtype::bf16 || c.type() == fl::dtype::f32);
  ASSERT_EQ(c.shape(), Shape({2, 4}));
  auto cF32 = c.astype(fl::dtype::f32);
  std::vector<float> host(8);
  cF32.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 6.0f, 0.1f);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
