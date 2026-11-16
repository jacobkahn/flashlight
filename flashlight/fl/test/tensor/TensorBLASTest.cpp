/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/test/TestInit.h"
#include "flashlight/fl/tensor/Random.h"

using namespace ::testing;
using namespace fl;

TEST(TensorBLASTest, matmul) {
  // TODO: test tensors with order > 2

  // Reference impl - operates on host vectors to avoid JIT overhead from
  // scalar tensor indexing in a tight loop.
  auto matmulRef = [](const Tensor& lhs, const Tensor& rhs) {
    // (M x N) x (N x K) --> (M x K)
    int M = lhs.dim(0);
    int N = lhs.dim(1);
    int K = rhs.dim(1);

    auto lhsData = lhs.toHostVectorSync<float>();
    auto rhsData = rhs.toHostVectorSync<float>();
    std::vector<float> outData(M * K, 0.0f);

    // Row-major: element (i, j) in an MxN matrix is at index i*N + j
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        for (int k = 0; k < N; ++k) {
          outData[i * K + j] += lhsData[i * N + k] * rhsData[k * K + j];
        }
      }
    }
    return Tensor::fromVector<float>({M, K}, outData);
  };

  int i = 10;
  int j = 20;
  int k = 12;

  auto a = fl::rand({i, j});
  auto b = fl::rand({j, k});
  auto ref = matmulRef(a, b);
  ASSERT_TRUE(allClose(fl::matmul(a, b), ref));
  ASSERT_TRUE(allClose(
      fl::matmul(
          a,
          fl::transpose(b),
          fl::MatrixProperty::None,
          fl::MatrixProperty::Transpose),
      ref));
  ASSERT_TRUE(allClose(
      fl::matmul(fl::transpose(a), b, fl::MatrixProperty::Transpose), ref));
}

TEST(TensorBLASTest, matmulShapes) {
  using T = fl::MatrixProperty;
  // Matrix/vector/scalar multiplies
  ASSERT_EQ(fl::matmul(fl::rand({10}), fl::rand({10})).shape(), Shape({1}));
  ASSERT_EQ(
      fl::matmul(fl::rand({10}), fl::rand({10}), T::Transpose).shape(),
      Shape({1}));
  ASSERT_EQ(
      fl::matmul(fl::rand({10}), fl::rand({10}), T::Transpose, T::Transpose)
          .shape(),
      Shape({1}));
  ASSERT_EQ(
      fl::matmul(fl::rand({10}), fl::rand({10}), T::None, T::Transpose).shape(),
      Shape({1}));
  ASSERT_EQ(fl::matmul(fl::rand({1, 10}), fl::rand({10})).shape(), Shape({1}));
  ASSERT_EQ(fl::matmul(fl::rand({1}), fl::rand({1, 10})).shape(), Shape({10}));
  ASSERT_EQ(
      fl::matmul(fl::rand({10}), fl::rand({10}), T::Transpose).shape(),
      Shape({1}));
  ASSERT_EQ(fl::matmul(fl::rand({3, 4}), fl::rand({4})).shape(), Shape({3}));
  ASSERT_EQ(fl::matmul(fl::rand({5}), fl::rand({5, 7})).shape(), Shape({7}));
  ASSERT_THROW(fl::matmul(fl::rand({1}), fl::rand({10})), std::exception);
  ASSERT_THROW(fl::matmul(fl::rand({3}), fl::rand({5, 7})), std::exception);

  // Batch matrix multiply (row-major: batch dims first)
  unsigned M = 10;
  unsigned K = 12;
  unsigned N = 14;
  unsigned b2 = 2;
  unsigned b3 = 4;
  ASSERT_EQ(
      fl::matmul(fl::rand({M, K}), fl::rand({K, N})).shape(), Shape({M, N}));
  ASSERT_EQ(
      fl::matmul(fl::rand({b2, M, K}), fl::rand({b2, K, N})).shape(),
      Shape({b2, M, N}));
  ASSERT_EQ(
      fl::matmul(fl::rand({b2, b3, M, K}), fl::rand({b2, b3, K, N})).shape(),
      Shape({b2, b3, M, N}));
  ASSERT_EQ(
      fl::matmul(fl::rand({b2, b3, M, K}), fl::rand({K, N})).shape(),
      Shape({b2, b3, M, N}));
  ASSERT_EQ(
      fl::matmul(fl::rand({M, K}), fl::rand({b2, b3, K, N})).shape(),
      Shape({b2, b3, M, N}));
  // Batch matrix multiply with transpose
  ASSERT_EQ(
      fl::matmul(fl::rand({K, M}), fl::rand({K, N}), T::Transpose).shape(),
      Shape({M, N}));
  ASSERT_EQ(
      fl::matmul(fl::rand({M, K}), fl::rand({N, K}), T::None, T::Transpose)
          .shape(),
      Shape({M, N}));
  // b2 transpose
  ASSERT_EQ(
      fl::matmul(fl::rand({b2, K, M}), fl::rand({K, N}), T::Transpose).shape(),
      Shape({b2, M, N}));
  ASSERT_EQ(
      fl::matmul(fl::rand({b2, M, K}), fl::rand({N, K}), T::None, T::Transpose)
          .shape(),
      Shape({b2, M, N}));
  ASSERT_EQ(
      fl::matmul(fl::rand({K, M}), fl::rand({b2, K, N}), T::Transpose).shape(),
      Shape({b2, M, N}));
  ASSERT_EQ(
      fl::matmul(fl::rand({M, K}), fl::rand({b2, N, K}), T::None, T::Transpose)
          .shape(),
      Shape({b2, M, N}));
  ASSERT_EQ(
      fl::matmul(fl::rand({b2, K, M}), fl::rand({b2, K, N}), T::Transpose)
          .shape(),
      Shape({b2, M, N}));
  ASSERT_EQ(
      fl::matmul(
          fl::rand({b2, M, K}), fl::rand({b2, N, K}), T::None, T::Transpose)
          .shape(),
      Shape({b2, M, N}));
  // b2, b3 transpose
  ASSERT_EQ(
      fl::matmul(fl::rand({b2, b3, K, M}), fl::rand({K, N}), T::Transpose)
          .shape(),
      Shape({b2, b3, M, N}));
  ASSERT_EQ(
      fl::matmul(
          fl::rand({b2, b3, M, K}), fl::rand({N, K}), T::None, T::Transpose)
          .shape(),
      Shape({b2, b3, M, N}));
  ASSERT_EQ(
      fl::matmul(fl::rand({K, M}), fl::rand({b2, b3, K, N}), T::Transpose)
          .shape(),
      Shape({b2, b3, M, N}));
  ASSERT_EQ(
      fl::matmul(
          fl::rand({M, K}), fl::rand({b2, b3, N, K}), T::None, T::Transpose)
          .shape(),
      Shape({b2, b3, M, N}));
  ASSERT_EQ(
      fl::matmul(
          fl::rand({b2, b3, K, M}), fl::rand({b2, b3, K, N}), T::Transpose)
          .shape(),
      Shape({b2, b3, M, N}));
  ASSERT_EQ(
      fl::matmul(
          fl::rand({b2, b3, M, K}),
          fl::rand({b2, b3, N, K}),
          T::None,
          T::Transpose)
          .shape(),
      Shape({b2, b3, M, N}));

  ASSERT_EQ(
      fl::matmul(
          fl::rand({2, 256, 200}),
          fl::rand({2, 256, 200}),
          T::None,
          T::Transpose)
          .shape(),
      Shape({2, 256, 256}));
}

TEST(TensorBLASTest, matmulValues) {
  using T = fl::MatrixProperty;

  // 2D with transpose flags (hand-computed)
  // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
  auto A = Tensor::fromVector<float>({2, 2}, {1, 2, 3, 4});
  auto B = Tensor::fromVector<float>({2, 2}, {5, 6, 7, 8});
  // A^T * B = [[1*5+3*7, 1*6+3*8], [2*5+4*7, 2*6+4*8]] = [[26,30],[38,44]]
  ASSERT_TRUE(allClose(
      fl::matmul(A, B, T::Transpose),
      Tensor::fromVector<float>({2, 2}, {26, 30, 38, 44})));
  // A * B^T = [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]] = [[17,23],[39,53]]
  ASSERT_TRUE(allClose(
      fl::matmul(A, B, T::None, T::Transpose),
      Tensor::fromVector<float>({2, 2}, {17, 23, 39, 53})));

  // Batch matmul: {2, 2, 2} with known values
  // Batch 0: [[1,2],[3,4]] * I = [[1,2],[3,4]]
  // Batch 1: [[5,6],[7,8]] * 2I = [[10,12],[14,16]]
  auto batchA =
      Tensor::fromVector<float>({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
  auto batchB =
      Tensor::fromVector<float>({2, 2, 2}, {1, 0, 0, 1, 2, 0, 0, 2});
  auto batchC = fl::matmul(batchA, batchB);
  ASSERT_EQ(batchC.shape(), Shape({2, 2, 2}));
  ASSERT_TRUE(allClose(
      batchC,
      Tensor::fromVector<float>({2, 2, 2}, {1, 2, 3, 4, 10, 12, 14, 16})));

  // Vector-matrix: v{3} * M{3,2} -> {2}
  auto v = Tensor::fromVector<float>({3}, {1, 2, 3});
  auto M = Tensor::fromVector<float>({3, 2}, {1, 2, 3, 4, 5, 6});
  // [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
  ASSERT_TRUE(allClose(
      fl::matmul(v, M), Tensor::fromVector<float>({2}, {22, 28})));

  // Matrix-vector: M{3,2} * v{2} -> {3}
  auto v2 = Tensor::fromVector<float>({2}, {1, 2});
  // [1*1+2*2, 3*1+4*2, 5*1+6*2] = [5, 11, 17]
  ASSERT_TRUE(allClose(
      fl::matmul(M, v2), Tensor::fromVector<float>({3}, {5, 11, 17})));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  flTestInit();
  return RUN_ALL_TESTS();
}
