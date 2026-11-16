/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/JitTensorBase.h"
#include "flashlight/fl/tensor/ir/NodeUtils.h"
#include "flashlight/fl/tensor/ir/ValueNode.h"
#include "flashlight/fl/tensor/backend/cuda/CudaTensor.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"

using namespace fl;

class CudaJitEndToEndTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fl::setDefaultJitTensorType<CudaTensor>();
  }
};

// ========================= Elementwise Fusion =========================

TEST_F(CudaJitEndToEndTest, UnaryChainFusion) {
  // exp(log(x)) ≈ x for positive x
  // This chain should be fused into a single NVRTC kernel
  auto vals = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
  auto t = Tensor::fromVector(Shape{4}, vals);
  auto result = fl::exp(fl::log(t));
  std::vector<float> host(4);
  result.hostSync(host.data());
  for (int i = 0; i < 4; i++) {
    ASSERT_NEAR(host[i], vals[i], 1e-5);
  }
}

TEST_F(CudaJitEndToEndTest, BinaryChainFusion) {
  // (a + b) * c -- should be fused into a single NVRTC kernel
  auto a = Tensor::fromVector(Shape{4}, std::vector<float>{1, 2, 3, 4});
  auto b = Tensor::fromVector(Shape{4}, std::vector<float>{10, 20, 30, 40});
  auto c = Tensor::fromVector(Shape{4}, std::vector<float>{2, 2, 2, 2});
  auto result = (a + b) * c;
  std::vector<float> host(4);
  result.hostSync(host.data());
  ASSERT_FLOAT_EQ(host[0], 22.0f);   // (1+10)*2
  ASSERT_FLOAT_EQ(host[1], 44.0f);   // (2+20)*2
  ASSERT_FLOAT_EQ(host[2], 66.0f);   // (3+30)*2
  ASSERT_FLOAT_EQ(host[3], 88.0f);   // (4+40)*2
}

TEST_F(CudaJitEndToEndTest, MixedUnaryBinaryFusion) {
  // exp(a + b) -- unary applied to binary result, should be one fused kernel
  auto a = Tensor::fromVector(Shape{4}, std::vector<float>{0, 0, 0, 0});
  auto b = Tensor::fromVector(Shape{4}, std::vector<float>{1, 2, 3, 4});
  auto result = fl::exp(a + b);
  std::vector<float> host(4);
  result.hostSync(host.data());
  ASSERT_NEAR(host[0], std::exp(1.0f), 1e-4);
  ASSERT_NEAR(host[1], std::exp(2.0f), 1e-4);
  ASSERT_NEAR(host[2], std::exp(3.0f), 1e-4);
  ASSERT_NEAR(host[3], std::exp(4.0f), 1e-4);
}

TEST_F(CudaJitEndToEndTest, LongerChainFusion) {
  // tanh(sqrt(a * b + c)) -- 4-op chain
  auto a = fl::full({1024}, 4.0f);
  auto b = fl::full({1024}, 4.0f);
  auto c = fl::full({1024}, 9.0f);
  auto result = fl::tanh(fl::sqrt(a * b + c));  // tanh(sqrt(16+9)) = tanh(5)
  std::vector<float> host(1024);
  result.hostSync(host.data());
  float expected = std::tanh(5.0f);
  for (int i = 0; i < 1024; i++) {
    ASSERT_NEAR(host[i], expected, 1e-5);
  }
}

TEST_F(CudaJitEndToEndTest, DiamondFusion) {
  // x = input; y = x + x (diamond: two uses of same node)
  // The JIT should handle this correctly
  auto x = Tensor::fromVector(Shape{4}, std::vector<float>{1, 2, 3, 4});
  auto result = x + x;
  std::vector<float> host(4);
  result.hostSync(host.data());
  ASSERT_FLOAT_EQ(host[0], 2.0f);
  ASSERT_FLOAT_EQ(host[1], 4.0f);
  ASSERT_FLOAT_EQ(host[2], 6.0f);
  ASSERT_FLOAT_EQ(host[3], 8.0f);
}

TEST_F(CudaJitEndToEndTest, MultiUserNoFusion) {
  // a + b used by two consumers -- the addition itself shouldn't be
  // fused into either consumer since it has multiple uses
  auto a = fl::full({256}, 1.0f);
  auto b = fl::full({256}, 2.0f);
  auto sum = a + b;  // sum has two users
  auto r1 = fl::exp(sum);
  auto r2 = fl::log(sum);
  std::vector<float> h1(256), h2(256);
  r1.hostSync(h1.data());
  r2.hostSync(h2.data());
  for (int i = 0; i < 256; i++) {
    ASSERT_NEAR(h1[i], std::exp(3.0f), 1e-5);
    ASSERT_NEAR(h2[i], std::log(3.0f), 1e-5);
  }
}

// =================== Reduction of Elementwise Fusion ===================

TEST_F(CudaJitEndToEndTest, SumOfExp) {
  // sum(exp(x)) -- should be fused into a single reduce kernel via NVRTC
  auto vals = std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto t = Tensor::fromVector(Shape{4}, vals);
  auto result = fl::sum(fl::exp(t), {});
  float expected =
      std::exp(0.0f) + std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);
  ASSERT_NEAR(result.scalarSync<float>(), expected, 1e-3);
}

TEST_F(CudaJitEndToEndTest, SumOfAddAlongAxis) {
  // sum(a + b, axis=1) -- binary chain feeding into reduction
  auto a = Tensor::fromVector(
      Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  auto b = Tensor::fromVector(
      Shape{2, 3}, std::vector<float>{10, 20, 30, 40, 50, 60});
  auto result = fl::sum(a + b, {1});
  ASSERT_EQ(result.shape(), Shape({2}));
  std::vector<float> host(2);
  result.hostSync(host.data());
  // Row 0: (1+10) + (2+20) + (3+30) = 11+22+33 = 66
  // Row 1: (4+40) + (5+50) + (6+60) = 44+55+66 = 165
  ASSERT_NEAR(host[0], 66.0f, 1e-4);
  ASSERT_NEAR(host[1], 165.0f, 1e-4);
}

TEST_F(CudaJitEndToEndTest, MeanOfNegative) {
  // mean(negative(x)) = -mean(x) -- reduction of unary
  auto t = fl::full({100}, 5.0f);
  auto result = fl::mean(fl::negative(t), {});
  ASSERT_NEAR(result.scalarSync<float>(), -5.0f, 1e-5);
}

TEST_F(CudaJitEndToEndTest, AmaxOfMul) {
  // amax(a * b) -- max reduction of elementwise
  auto a = Tensor::fromVector(Shape{5}, std::vector<float>{1, -3, 2, -4, 5});
  auto b = Tensor::fromVector(Shape{5}, std::vector<float>{2, 2, 2, 2, 2});
  auto result = fl::amax(a * b, {});
  ASSERT_NEAR(result.scalarSync<float>(), 10.0f, 1e-5);  // 5*2 = 10
}

// ========================= Matmul Epilogue Fusion =========================

TEST_F(CudaJitEndToEndTest, MatmulAddFusion) {
  // matmul(A, B) + C -- should be fused via cuBLASLt with beta=1
  auto a = fl::full({2, 3}, 1.0f);
  auto b = fl::full({3, 4}, 2.0f);
  auto c = fl::full({2, 4}, 10.0f);
  auto result = fl::matmul(a, b) + c;
  ASSERT_EQ(result.shape(), Shape({2, 4}));
  std::vector<float> host(8);
  result.hostSync(host.data());
  // Each matmul element = 1*2*3 = 6, then +10 = 16
  for (float v : host) {
    ASSERT_NEAR(v, 16.0f, 1e-4);
  }
}

TEST_F(CudaJitEndToEndTest, AddMatmulFusion) {
  // C + matmul(A, B) -- same as above but with reversed add operands
  auto a = fl::full({2, 3}, 1.0f);
  auto b = fl::full({3, 4}, 2.0f);
  auto c = fl::full({2, 4}, 10.0f);
  auto result = c + fl::matmul(a, b);
  std::vector<float> host(8);
  result.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 16.0f, 1e-4);
  }
}

TEST_F(CudaJitEndToEndTest, MatmulAddReluFusion) {
  // relu(matmul(A, B) + C) -- should be fused via cuBLASLt EPILOGUE_RELU_BIAS
  auto a = fl::full({2, 3}, 1.0f);
  auto b = fl::full({3, 4}, 2.0f);
  auto c = fl::full({2, 4}, -10.0f);  // negative bias to test relu clipping
  auto matmulResult = fl::matmul(a, b) + c;  // 6 + (-10) = -4
  auto result = fl::maximum(matmulResult, 0.0);
  ASSERT_EQ(result.shape(), Shape({2, 4}));
  std::vector<float> host(8);
  result.hostSync(host.data());
  // matmul element = 1*2*3 = 6, + (-10) = -4, relu(-4) = 0
  for (float v : host) {
    ASSERT_NEAR(v, 0.0f, 1e-5);
  }
}

TEST_F(CudaJitEndToEndTest, MatmulAddReluFusionPositive) {
  // relu(matmul(A, B) + C) with positive result -- relu should pass through
  auto a = fl::full({2, 3}, 1.0f);
  auto b = fl::full({3, 4}, 2.0f);
  auto c = fl::full({2, 4}, 10.0f);
  auto result = fl::maximum(fl::matmul(a, b) + c, 0.0);
  std::vector<float> host(8);
  result.hostSync(host.data());
  // matmul = 6, + 10 = 16, relu(16) = 16
  for (float v : host) {
    ASSERT_NEAR(v, 16.0f, 1e-4);
  }
}

TEST_F(CudaJitEndToEndTest, MatmulAddGeluFusion) {
  // gelu(matmul(A, B) + C) -- should be fused via cuBLASLt EPILOGUE_GELU_BIAS
  auto a = fl::full({2, 3}, 1.0f);
  auto b = fl::full({3, 4}, 1.0f);
  auto c = fl::full({2, 4}, 0.0f);
  auto x = fl::matmul(a, b) + c;  // x = 3.0
  // gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
  auto result = 0.5 * x * (1.0 + fl::erf(x / std::sqrt(2.0)));
  ASSERT_EQ(result.shape(), Shape({2, 4}));
  std::vector<float> host(8);
  result.hostSync(host.data());
  // gelu(3.0) = 0.5 * 3 * (1 + erf(3/sqrt(2))) ≈ 2.9960
  float expected = 0.5f * 3.0f * (1.0f + std::erf(3.0f / std::sqrt(2.0f)));
  for (float v : host) {
    ASSERT_NEAR(v, expected, 1e-3);
  }
}

TEST_F(CudaJitEndToEndTest, MatmulAddGeluFusionNegative) {
  // gelu with negative input -- gelu(-x) should be near 0 for large x
  auto a = fl::full({2, 3}, -1.0f);
  auto b = fl::full({3, 4}, 1.0f);
  auto c = fl::full({2, 4}, 0.0f);
  auto x = fl::matmul(a, b) + c;  // x = -3.0
  auto result = 0.5 * x * (1.0 + fl::erf(x / std::sqrt(2.0)));
  std::vector<float> host(8);
  result.hostSync(host.data());
  // gelu(-3) ≈ -0.00404
  float expected = 0.5f * (-3.0f) * (1.0f + std::erf(-3.0f / std::sqrt(2.0f)));
  for (float v : host) {
    ASSERT_NEAR(v, expected, 1e-3);
  }
}

TEST_F(CudaJitEndToEndTest, MatmulNoFusion) {
  // Standalone matmul without add -- no fusion, just regular cuBLAS
  auto a = Tensor::fromVector(
      Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  auto b = Tensor::fromVector(
      Shape{3, 2}, std::vector<float>{1, 0, 0, 1, 1, 1});
  auto result = fl::matmul(a, b);
  ASSERT_EQ(result.shape(), Shape({2, 2}));
  std::vector<float> host(4);
  result.hostSync(host.data());
  ASSERT_NEAR(host[0], 4.0f, 1e-5);   // 1*1+2*0+3*1
  ASSERT_NEAR(host[1], 5.0f, 1e-5);   // 1*0+2*1+3*1
  ASSERT_NEAR(host[2], 10.0f, 1e-5);  // 4*1+5*0+6*1
  ASSERT_NEAR(host[3], 11.0f, 1e-5);  // 4*0+5*1+6*1
}

// ==================== Batched Matmul Epilogue Fusion ====================

TEST_F(CudaJitEndToEndTest, BatchedMatmulAddFusion) {
  // batched matmul(A, B) + C — 3D matmul with add epilogue
  int B = 4, M = 2, K = 3, N = 5;
  auto a = fl::full({B, M, K}, 1.0f);
  auto b = fl::full({B, K, N}, 2.0f);
  auto c = fl::full({B, M, N}, 10.0f);
  auto result = fl::matmul(a, b) + c;
  ASSERT_EQ(result.shape(), Shape({B, M, N}));
  std::vector<float> host(B * M * N);
  result.hostSync(host.data());
  // Each matmul element = 1*2*K = 6, then +10 = 16
  for (float v : host) {
    ASSERT_NEAR(v, 16.0f, 1e-4);
  }
}

TEST_F(CudaJitEndToEndTest, BatchedMatmulAddReluFusion) {
  // relu(batched matmul(A, B) + C) — 3D matmul with add+relu epilogue
  int B = 4, M = 2, K = 3, N = 5;
  auto a = fl::full({B, M, K}, 1.0f);
  auto b = fl::full({B, K, N}, 2.0f);
  auto c = fl::full({B, M, N}, -10.0f); // negative bias to test relu clipping
  auto result = fl::maximum(fl::matmul(a, b) + c, 0.0);
  ASSERT_EQ(result.shape(), Shape({B, M, N}));
  std::vector<float> host(B * M * N);
  result.hostSync(host.data());
  // matmul = 6, + (-10) = -4, relu(-4) = 0
  for (float v : host) {
    ASSERT_NEAR(v, 0.0f, 1e-5);
  }
}

TEST_F(CudaJitEndToEndTest, BatchedMatmulAddReluFusionPositive) {
  // relu(batched matmul(A, B) + C) with positive result
  int B = 4, M = 2, K = 3, N = 5;
  auto a = fl::full({B, M, K}, 1.0f);
  auto b = fl::full({B, K, N}, 2.0f);
  auto c = fl::full({B, M, N}, 10.0f);
  auto result = fl::maximum(fl::matmul(a, b) + c, 0.0);
  std::vector<float> host(B * M * N);
  result.hostSync(host.data());
  // matmul = 6, + 10 = 16, relu(16) = 16
  for (float v : host) {
    ASSERT_NEAR(v, 16.0f, 1e-4);
  }
}

TEST_F(CudaJitEndToEndTest, BatchedMatmulTransposeRhsFusion) {
  // batched matmul(A, B^T) + C — transpose in batched context
  int B = 4, M = 2, K = 8, N = 6;
  auto A = fl::rand({B, M, K});
  auto Bt = fl::rand({B, N, K}); // transposed: {B, N, K}
  auto C = fl::rand({B, M, N});

  // Reference: eager matmul without fusion
  auto refMm = fl::matmul(A, Bt, MatrixProperty::None, MatrixProperty::Transpose);
  fl::eval(refMm);
  auto ref = refMm + C;
  fl::eval(ref);

  // Fused path
  auto fused = fl::matmul(A, Bt, MatrixProperty::None, MatrixProperty::Transpose) + C;
  ASSERT_EQ(fused.shape(), Shape({B, M, N}));
  std::vector<float> hostRef(B * M * N), hostFused(B * M * N);
  ref.hostSync(hostRef.data());
  fused.hostSync(hostFused.data());
  for (int i = 0; i < B * M * N; i++) {
    ASSERT_NEAR(hostFused[i], hostRef[i], 1e-3)
        << "Mismatch at index " << i;
  }
}

TEST_F(CudaJitEndToEndTest, BatchedMatmulBroadcastBiasFusion) {
  // batched matmul(A, B) + bias — bias broadcasts across batch dim
  int B = 4, M = 2, K = 3, N = 5;
  auto a = fl::full({B, M, K}, 1.0f);
  auto b = fl::full({B, K, N}, 2.0f);
  auto bias = fl::full({1, 1, N}, 10.0f); // broadcasts to {B, M, N}
  auto result = fl::matmul(a, b) + bias;
  ASSERT_EQ(result.shape(), Shape({B, M, N}));
  std::vector<float> host(B * M * N);
  result.hostSync(host.data());
  // matmul = 6, + 10 = 16
  for (float v : host) {
    ASSERT_NEAR(v, 16.0f, 1e-4);
  }
}

TEST_F(CudaJitEndToEndTest, BatchedMatmulNoFusion) {
  // Standalone batched matmul without add — no fusion, regular cuBLAS
  int B = 2, M = 3, K = 4, N = 5;
  auto a = fl::full({B, M, K}, 1.0f);
  auto b = fl::full({B, K, N}, 2.0f);
  auto result = fl::matmul(a, b);
  ASSERT_EQ(result.shape(), Shape({B, M, N}));
  std::vector<float> host(B * M * N);
  result.hostSync(host.data());
  // Each element = 1*2*K = 8
  for (float v : host) {
    ASSERT_NEAR(v, 8.0f, 1e-4);
  }
}

TEST_F(CudaJitEndToEndTest, BatchedMatmulSigmoidFusion) {
  // sigmoid(batched matmul(A, B)) — anchor + unary
  int B = 4, M = 2, K = 3, N = 5;
  auto a = fl::full({B, M, K}, 0.0f);
  auto b = fl::full({B, K, N}, 0.0f);
  auto result = fl::sigmoid(fl::matmul(a, b));
  ASSERT_EQ(result.shape(), Shape({B, M, N}));
  std::vector<float> host(B * M * N);
  result.hostSync(host.data());
  // matmul = 0, sigmoid(0) = 0.5
  for (float v : host) {
    ASSERT_NEAR(v, 0.5f, 1e-5);
  }
}

TEST_F(CudaJitEndToEndTest, BatchedLinearLayerForward) {
  // Simulate a batched linear layer: Y = X @ W + bias
  // X={B, seqLen, inF}, W={inF, outF}, bias={1, 1, outF}
  int B = 8, seqLen = 16, inF = 32, outF = 64;
  auto X = fl::full({B, seqLen, inF}, 0.5f);
  auto W = fl::full({inF, outF}, 0.25f);
  auto bias = fl::full({1, 1, outF}, 1.0f);
  auto Y = fl::matmul(X, W) + bias;
  ASSERT_EQ(Y.shape(), Shape({B, seqLen, outF}));
  std::vector<float> host(B * seqLen * outF);
  Y.hostSync(host.data());
  // Each matmul element = 0.5 * 0.25 * inF = 4.0, + 1.0 = 5.0
  for (float v : host) {
    ASSERT_NEAR(v, 5.0f, 1e-3);
  }
}

// ============== Matmul Transpose Fusion (cuDNN subgraph) ===============

TEST_F(CudaJitEndToEndTest, MatmulTransposeRhsFusion) {
  // matmul(A, B, None, Transpose) + C — RHS transpose is the Linear pattern
  // A={M,K}, B={N,K} (transposed to K,N internally)
  int M = 4, K = 8, N = 6;
  auto A = fl::rand({M, K});
  auto B = fl::rand({N, K});
  auto C = fl::rand({M, N});

  // Reference: eager matmul without fusion
  auto refMm = fl::matmul(A, B, MatrixProperty::None, MatrixProperty::Transpose);
  fl::eval(refMm);
  auto ref = refMm + C;
  fl::eval(ref);

  // Fused path
  auto fused = fl::matmul(A, B, MatrixProperty::None, MatrixProperty::Transpose) + C;
  ASSERT_EQ(fused.shape(), Shape({M, N}));
  std::vector<float> hostRef(M * N), hostFused(M * N);
  ref.hostSync(hostRef.data());
  fused.hostSync(hostFused.data());
  for (int i = 0; i < M * N; i++) {
    ASSERT_NEAR(hostFused[i], hostRef[i], 1e-3)
        << "Mismatch at index " << i;
  }
}

TEST_F(CudaJitEndToEndTest, MatmulTransposeLhsFusion) {
  // matmul(A, B, Transpose, None) + C — LHS transpose
  // A={K,M} (transposed to M,K internally), B={K,N}
  int M = 4, K = 8, N = 6;
  auto A = fl::rand({K, M});
  auto B = fl::rand({K, N});
  auto C = fl::rand({M, N});

  auto refMm = fl::matmul(A, B, MatrixProperty::Transpose, MatrixProperty::None);
  fl::eval(refMm);
  auto ref = refMm + C;
  fl::eval(ref);

  auto fused = fl::matmul(A, B, MatrixProperty::Transpose, MatrixProperty::None) + C;
  ASSERT_EQ(fused.shape(), Shape({M, N}));
  std::vector<float> hostRef(M * N), hostFused(M * N);
  ref.hostSync(hostRef.data());
  fused.hostSync(hostFused.data());
  for (int i = 0; i < M * N; i++) {
    ASSERT_NEAR(hostFused[i], hostRef[i], 1e-3)
        << "Mismatch at index " << i;
  }
}

TEST_F(CudaJitEndToEndTest, MatmulTransposeBothFusion) {
  // matmul(A, B, Transpose, Transpose) + C — both transposed
  // A={K,M}, B={N,K}
  int M = 4, K = 8, N = 6;
  auto A = fl::rand({K, M});
  auto B = fl::rand({N, K});
  auto C = fl::rand({M, N});

  auto refMm = fl::matmul(A, B, MatrixProperty::Transpose, MatrixProperty::Transpose);
  fl::eval(refMm);
  auto ref = refMm + C;
  fl::eval(ref);

  auto fused = fl::matmul(A, B, MatrixProperty::Transpose, MatrixProperty::Transpose) + C;
  ASSERT_EQ(fused.shape(), Shape({M, N}));
  std::vector<float> hostRef(M * N), hostFused(M * N);
  ref.hostSync(hostRef.data());
  fused.hostSync(hostFused.data());
  for (int i = 0; i < M * N; i++) {
    ASSERT_NEAR(hostFused[i], hostRef[i], 1e-3)
        << "Mismatch at index " << i;
  }
}

TEST_F(CudaJitEndToEndTest, LinearLayerTransposeFusion) {
  // The real Linear layer pattern: y = matmul(x, W^T) + bias
  // x={batch, inF}, W={outF, inF}, bias={1, outF}
  int batch = 8, inF = 32, outF = 16;
  auto x = fl::full({batch, inF}, 0.5f);
  auto W = fl::full({outF, inF}, 0.25f);
  auto bias = fl::full({1, outF}, 1.0f);

  auto y = fl::matmul(x, W, MatrixProperty::None, MatrixProperty::Transpose) + bias;
  ASSERT_EQ(y.shape(), Shape({batch, outF}));
  std::vector<float> host(batch * outF);
  y.hostSync(host.data());
  // Each element: 0.5 * 0.25 * inF + 1.0 = 0.5 * 0.25 * 32 + 1.0 = 5.0
  for (float v : host) {
    ASSERT_NEAR(v, 5.0f, 1e-3);
  }
}

TEST_F(CudaJitEndToEndTest, MatmulTransposeReluFusion) {
  // relu(matmul(x, W^T) + b) — Linear + ReLU, the most common fusion target
  int batch = 4, inF = 16, outF = 8;
  auto x = fl::full({batch, inF}, 1.0f);
  auto W = fl::full({outF, inF}, 0.5f);
  auto bias = fl::full({1, outF}, -10.0f);

  auto linear = fl::matmul(x, W, MatrixProperty::None, MatrixProperty::Transpose) + bias;
  auto result = fl::maximum(linear, 0.0);
  ASSERT_EQ(result.shape(), Shape({batch, outF}));
  std::vector<float> host(batch * outF);
  result.hostSync(host.data());
  // matmul element = 1.0 * 0.5 * 16 = 8.0, + (-10.0) = -2.0, relu = 0.0
  for (float v : host) {
    ASSERT_NEAR(v, 0.0f, 1e-5);
  }

  // Now with positive result
  auto bias2 = fl::full({1, outF}, 2.0f);
  auto linear2 = fl::matmul(x, W, MatrixProperty::None, MatrixProperty::Transpose) + bias2;
  auto result2 = fl::maximum(linear2, 0.0);
  std::vector<float> host2(batch * outF);
  result2.hostSync(host2.data());
  // matmul = 8.0, + 2.0 = 10.0, relu(10.0) = 10.0
  for (float v : host2) {
    ASSERT_NEAR(v, 10.0f, 1e-3);
  }
}

// ================ Matmul Epilogue Fusion (bf16/f16/f64) ================

TEST_F(CudaJitEndToEndTest, MatmulAddFusionBf16) {
  // matmul(A, B) + C in bf16 -- should be fused via cuBLASLt with bf16 I/O
  auto a = fl::full({2, 3}, 1.0f, fl::dtype::bf16);
  auto b = fl::full({3, 4}, 2.0f, fl::dtype::bf16);
  auto c = fl::full({2, 4}, 10.0f, fl::dtype::bf16);
  auto result = fl::matmul(a, b) + c;
  ASSERT_EQ(result.type(), fl::dtype::bf16);
  ASSERT_EQ(result.shape(), Shape({2, 4}));
  // Each matmul element = 1*2*3 = 6, then +10 = 16
  auto hostResult = result.astype(fl::dtype::f32);
  std::vector<float> host(8);
  hostResult.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 16.0f, 0.5f);  // bf16 tolerance
  }
}

TEST_F(CudaJitEndToEndTest, MatmulAddFusionF16) {
  // matmul(A, B) + C in f16 -- should be fused via cuBLASLt with f16 I/O
  auto a = fl::full({2, 3}, 1.0f, fl::dtype::f16);
  auto b = fl::full({3, 4}, 2.0f, fl::dtype::f16);
  auto c = fl::full({2, 4}, 10.0f, fl::dtype::f16);
  auto result = fl::matmul(a, b) + c;
  ASSERT_EQ(result.type(), fl::dtype::f16);
  ASSERT_EQ(result.shape(), Shape({2, 4}));
  auto hostResult = result.astype(fl::dtype::f32);
  std::vector<float> host(8);
  hostResult.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 16.0f, 0.1f);
  }
}

TEST_F(CudaJitEndToEndTest, MatmulAddFusionF64) {
  // matmul(A, B) + C in f64 -- should be fused via cuBLASLt with f64 I/O
  auto a = fl::full({2, 3}, 1.0, fl::dtype::f64);
  auto b = fl::full({3, 4}, 2.0, fl::dtype::f64);
  auto c = fl::full({2, 4}, 10.0, fl::dtype::f64);
  auto result = fl::matmul(a, b) + c;
  ASSERT_EQ(result.type(), fl::dtype::f64);
  ASSERT_EQ(result.shape(), Shape({2, 4}));
  auto hostResult = result.astype(fl::dtype::f32);
  std::vector<float> host(8);
  hostResult.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 16.0f, 1e-5);
  }
}

TEST_F(CudaJitEndToEndTest, MatmulAddReluFusionBf16) {
  // relu(matmul(A, B) + C) in bf16
  auto a = fl::full({2, 3}, 1.0f, fl::dtype::bf16);
  auto b = fl::full({3, 4}, 2.0f, fl::dtype::bf16);
  auto c = fl::full({2, 4}, -10.0f, fl::dtype::bf16);
  auto result = fl::maximum(fl::matmul(a, b) + c, 0.0);
  ASSERT_EQ(result.type(), fl::dtype::bf16);
  // matmul = 6, + (-10) = -4, relu(-4) = 0
  auto hostResult = result.astype(fl::dtype::f32);
  std::vector<float> host(8);
  hostResult.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 0.0f, 1e-2);
  }
}

TEST_F(CudaJitEndToEndTest, MatmulAddReluFusionBf16Positive) {
  // relu(matmul(A, B) + C) in bf16 with positive result
  auto a = fl::full({2, 3}, 1.0f, fl::dtype::bf16);
  auto b = fl::full({3, 4}, 2.0f, fl::dtype::bf16);
  auto c = fl::full({2, 4}, 10.0f, fl::dtype::bf16);
  auto result = fl::maximum(fl::matmul(a, b) + c, 0.0);
  ASSERT_EQ(result.type(), fl::dtype::bf16);
  // matmul = 6, + 10 = 16, relu(16) = 16
  auto hostResult = result.astype(fl::dtype::f32);
  std::vector<float> host(8);
  hostResult.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 16.0f, 0.5f);
  }
}

TEST_F(CudaJitEndToEndTest, LinearLayerForwardBf16) {
  // Simulate a bf16 linear layer: y = x @ W + b
  int batchSize = 4, inFeatures = 8, outFeatures = 16;
  auto x = fl::full({batchSize, inFeatures}, 0.5f, fl::dtype::bf16);
  auto W = fl::full({inFeatures, outFeatures}, 0.25f, fl::dtype::bf16);
  auto b = fl::full({1, outFeatures}, 1.0f, fl::dtype::bf16);
  auto y = fl::matmul(x, W) + b;
  ASSERT_EQ(y.type(), fl::dtype::bf16);
  ASSERT_EQ(y.shape(), Shape({batchSize, outFeatures}));
  auto hostResult = y.astype(fl::dtype::f32);
  std::vector<float> host(batchSize * outFeatures);
  hostResult.hostSync(host.data());
  // Each matmul element = 0.5 * 0.25 * 8 = 1.0, then +1 = 2.0
  for (float v : host) {
    ASSERT_NEAR(v, 2.0f, 0.1f);  // bf16 tolerance
  }
}

// ==================== Common Subexpression Elimination ====================

TEST_F(CudaJitEndToEndTest, CSE) {
  // exp(x) + exp(x) -- the two exp(x) should be deduplicated by CSE
  auto x = Tensor::fromVector(Shape{4}, std::vector<float>{0, 1, 2, 3});
  auto result = fl::exp(x) + fl::exp(x);
  std::vector<float> host(4);
  result.hostSync(host.data());
  ASSERT_NEAR(host[0], 2.0f * std::exp(0.0f), 1e-4);
  ASSERT_NEAR(host[1], 2.0f * std::exp(1.0f), 1e-4);
  ASSERT_NEAR(host[2], 2.0f * std::exp(2.0f), 1e-4);
  ASSERT_NEAR(host[3], 2.0f * std::exp(3.0f), 1e-4);
}

// ========================= Constant Folding =========================

TEST_F(CudaJitEndToEndTest, ConstantFolding) {
  // full(3) + full(4) should be constant-folded to full(7)
  auto a = fl::full({256}, 3.0f);
  auto b = fl::full({256}, 4.0f);
  auto result = a + b;
  std::vector<float> host(256);
  result.hostSync(host.data());
  for (float v : host) {
    ASSERT_FLOAT_EQ(v, 7.0f);
  }
}

TEST_F(CudaJitEndToEndTest, ConstantFoldUnary) {
  // exp(full(0)) should be constant-folded to full(1)
  auto t = fl::full({128}, 0.0f);
  auto result = fl::exp(t);
  std::vector<float> host(128);
  result.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 1.0f, 1e-5);
  }
}

// ========================= Mixed Operations =========================

TEST_F(CudaJitEndToEndTest, LinearLayerForward) {
  // Simulate a simple linear layer: y = x @ W + b
  // This should fuse matmul+add via cuBLASLt
  int batchSize = 4, inFeatures = 8, outFeatures = 16;
  auto x = fl::full({batchSize, inFeatures}, 0.5f);
  auto W = fl::full({inFeatures, outFeatures}, 0.25f);
  auto b = fl::full({1, outFeatures}, 1.0f);
  auto y = fl::matmul(x, W) + b;  // broadcast add over batch
  ASSERT_EQ(y.shape(), Shape({batchSize, outFeatures}));
  std::vector<float> host(batchSize * outFeatures);
  y.hostSync(host.data());
  // Each matmul element = 0.5 * 0.25 * 8 = 1.0, then +1 = 2.0
  for (float v : host) {
    ASSERT_NEAR(v, 2.0f, 1e-4);
  }
}

TEST_F(CudaJitEndToEndTest, SoftmaxLikeComputation) {
  // Softmax-like: exp(x - max(x)) / sum(exp(x - max(x)))
  // Using simplified version with known inputs
  auto x = Tensor::fromVector(Shape{4}, std::vector<float>{1, 2, 3, 4});

  // Compute max (this is a reduction, breaks the JIT chain)
  auto maxVal = fl::amax(x, {});

  // Compute exp(x - max) -- this is elementwise and should be fused
  auto shifted = x - maxVal;
  auto expShifted = fl::exp(shifted);

  // Sum the exps (reduction of elementwise -- should be fused)
  auto sumExp = fl::sum(expShifted, {});

  // Normalize
  auto softmax = expShifted / sumExp;

  std::vector<float> host(4);
  softmax.hostSync(host.data());

  // Verify sum is 1
  float sum = 0;
  for (float v : host) {
    sum += v;
    ASSERT_GT(v, 0.0f);
    ASSERT_LT(v, 1.0f);
  }
  ASSERT_NEAR(sum, 1.0f, 1e-4);

  // Verify ordering: softmax(1) < softmax(2) < softmax(3) < softmax(4)
  ASSERT_LT(host[0], host[1]);
  ASSERT_LT(host[1], host[2]);
  ASSERT_LT(host[2], host[3]);
}

TEST_F(CudaJitEndToEndTest, MSELoss) {
  // MSE = mean((pred - target)^2)
  // This is: subtraction --> power --> mean (reduction of elementwise)
  auto pred = Tensor::fromVector(
      Shape{6}, std::vector<float>{1, 2, 3, 4, 5, 6});
  auto target = Tensor::fromVector(
      Shape{6}, std::vector<float>{1.5, 2.5, 3.5, 4.5, 5.5, 6.5});
  auto diff = pred - target;
  auto sq = diff * diff;
  auto mse = fl::mean(sq, {});
  // Each diff = -0.5, sq = 0.25, mean of 6 values of 0.25 = 0.25
  ASSERT_NEAR(mse.scalarSync<float>(), 0.25f, 1e-5);
}

// ================ cuDNN Subgraph Lowering (conv2d fusion) ================

TEST_F(CudaJitEndToEndTest, Conv2dBasic) {
  // Basic conv2d through the JIT pipeline — verifies conv2d evaluation works
  int N = 1, C = 1, H = 4, W = 4;
  int outC = 1, kH = 3, kW = 3;
  auto input = fl::full({N, C, H, W}, 1.0f);
  auto weights = fl::full({outC, C, kH, kW}, 1.0f);
  auto bias = fl::full({1, outC, 1, 1}, 0.0f);
  Tensor output;
  input.backend().conv2dForward(
      output, input, weights, bias,
      /*sx=*/1, /*sy=*/1, /*px=*/0, /*py=*/0,
      /*dx=*/1, /*dy=*/1, /*groups=*/1, /*wantState=*/false);
  // 3x3 kernel of ones on 4x4 input of ones, no padding → 2x2 output
  // Each output element = sum of 3x3 = 9.0
  ASSERT_EQ(output.shape(), Shape({N, outC, 2, 2}));
  std::vector<float> host(4);
  output.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 9.0f, 1e-4);
  }
}

TEST_F(CudaJitEndToEndTest, Conv2dReluFusion) {
  // relu(conv2d(x, w, b)) — should be fused by CudnnSubgraphLowering
  int N = 1, C = 1, H = 4, W = 4;
  int outC = 1, kH = 3, kW = 3;
  auto input = fl::full({N, C, H, W}, 1.0f);
  auto weights = fl::full({outC, C, kH, kW}, 1.0f);
  auto bias = fl::full({1, outC, 1, 1}, -20.0f); // negative bias forces relu clipping
  Tensor output;
  input.backend().conv2dForward(
      output, input, weights, bias,
      /*sx=*/1, /*sy=*/1, /*px=*/0, /*py=*/0,
      /*dx=*/1, /*dy=*/1, /*groups=*/1, /*wantState=*/false);
  // conv output = 9.0 + (-20.0) = -11.0, relu(-11.0) = 0.0
  auto result = fl::maximum(output, 0.0);
  ASSERT_EQ(result.shape(), Shape({N, outC, 2, 2}));
  std::vector<float> host(4);
  result.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 0.0f, 1e-5);
  }
}

TEST_F(CudaJitEndToEndTest, Conv2dReluFusionPositive) {
  // relu(conv2d(x, w, b)) with positive result — relu passes through
  int N = 1, C = 1, H = 4, W = 4;
  int outC = 1, kH = 3, kW = 3;
  auto input = fl::full({N, C, H, W}, 1.0f);
  auto weights = fl::full({outC, C, kH, kW}, 1.0f);
  auto bias = fl::full({1, outC, 1, 1}, 1.0f);
  Tensor output;
  input.backend().conv2dForward(
      output, input, weights, bias,
      /*sx=*/1, /*sy=*/1, /*px=*/0, /*py=*/0,
      /*dx=*/1, /*dy=*/1, /*groups=*/1, /*wantState=*/false);
  // conv output = 9.0 + 1.0 = 10.0, relu(10.0) = 10.0
  auto result = fl::maximum(output, 0.0);
  std::vector<float> host(4);
  result.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 10.0f, 1e-4);
  }
}

TEST_F(CudaJitEndToEndTest, Conv2dSigmoidFusion) {
  // sigmoid(conv2d(x, w, b)) — should be fused by CudnnSubgraphLowering
  int N = 1, C = 1, H = 4, W = 4;
  int outC = 1, kH = 3, kW = 3;
  auto input = fl::full({N, C, H, W}, 0.0f);
  auto weights = fl::full({outC, C, kH, kW}, 0.0f);
  auto bias = fl::full({1, outC, 1, 1}, 0.0f);
  Tensor output;
  input.backend().conv2dForward(
      output, input, weights, bias,
      /*sx=*/1, /*sy=*/1, /*px=*/0, /*py=*/0,
      /*dx=*/1, /*dy=*/1, /*groups=*/1, /*wantState=*/false);
  // conv output = 0.0, sigmoid(0.0) = 0.5
  auto result = fl::sigmoid(output);
  std::vector<float> host(4);
  result.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 0.5f, 1e-5);
  }
}

TEST_F(CudaJitEndToEndTest, Conv2dResidualReluFusion) {
  // relu(conv2d(x, w, b) + residual) — 3-node subgraph for CudnnSubgraphLowering
  int N = 1, C = 1, H = 4, W = 4;
  int outC = 1, kH = 3, kW = 3;
  auto input = fl::full({N, C, H, W}, 1.0f);
  auto weights = fl::full({outC, C, kH, kW}, 1.0f);
  auto bias = fl::full({1, outC, 1, 1}, 0.0f);
  auto residual = fl::full({N, outC, 2, 2}, -5.0f);
  Tensor output;
  input.backend().conv2dForward(
      output, input, weights, bias,
      /*sx=*/1, /*sy=*/1, /*px=*/0, /*py=*/0,
      /*dx=*/1, /*dy=*/1, /*groups=*/1, /*wantState=*/false);
  // conv output = 9.0, + residual(-5.0) = 4.0, relu(4.0) = 4.0
  auto result = fl::maximum(output + residual, 0.0);
  std::vector<float> host(4);
  result.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 4.0f, 1e-4);
  }
}

// ============ cuDNN Subgraph Lowering (conv + BN + residual) ============

TEST_F(CudaJitEndToEndTest, Conv2dResidualReluClipping) {
  // relu(conv2d(x, w, b) + residual) where residual is large negative
  // — verifies relu actually clips to zero in the fused path
  int N = 1, C = 1, H = 4, W = 4;
  int outC = 1, kH = 3, kW = 3;
  auto input = fl::full({N, C, H, W}, 1.0f);
  auto weights = fl::full({outC, C, kH, kW}, 1.0f);
  auto bias = fl::full({1, outC, 1, 1}, 0.0f);
  auto residual = fl::full({N, outC, 2, 2}, -100.0f);
  Tensor output;
  input.backend().conv2dForward(
      output, input, weights, bias,
      /*sx=*/1, /*sy=*/1, /*px=*/0, /*py=*/0,
      /*dx=*/1, /*dy=*/1, /*groups=*/1, /*wantState=*/false);
  // conv output = 9.0, + residual(-100.0) = -91.0, relu(-91.0) = 0.0
  auto result = fl::maximum(output + residual, 0.0);
  std::vector<float> host(4);
  result.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 0.0f, 1e-5);
  }
}

TEST_F(CudaJitEndToEndTest, Conv2dAddOnly) {
  // conv2d(x, w, b) + residual — conv + add without activation
  int N = 1, C = 1, H = 4, W = 4;
  int outC = 1, kH = 3, kW = 3;
  auto input = fl::full({N, C, H, W}, 1.0f);
  auto weights = fl::full({outC, C, kH, kW}, 1.0f);
  auto bias = fl::full({1, outC, 1, 1}, 0.0f);
  auto residual = fl::full({N, outC, 2, 2}, 3.0f);
  Tensor output;
  input.backend().conv2dForward(
      output, input, weights, bias,
      /*sx=*/1, /*sy=*/1, /*px=*/0, /*py=*/0,
      /*dx=*/1, /*dy=*/1, /*groups=*/1, /*wantState=*/false);
  // conv output = 9.0, + 3.0 = 12.0
  auto result = output + residual;
  std::vector<float> host(4);
  result.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 12.0f, 1e-4);
  }
}

TEST_F(CudaJitEndToEndTest, Conv2dBNInferenceRelu) {
  // relu(batchnorm_inference(conv2d(x, w, b))) — two anchors, inference mode
  int N = 1, C = 1, H = 6, W = 6;
  int outC = 2, kH = 3, kW = 3;
  auto input = fl::full({N, C, H, W}, 1.0f);
  auto weights = fl::full({outC, C, kH, kW}, 0.5f);
  auto convBias = fl::full({1, outC, 1, 1}, 0.0f);
  Tensor convOut;
  input.backend().conv2dForward(
      convOut, input, weights, convBias,
      /*sx=*/1, /*sy=*/1, /*px=*/0, /*py=*/0,
      /*dx=*/1, /*dy=*/1, /*groups=*/1, /*wantState=*/false);
  // convOut shape: {1, 2, 4, 4}, value: 0.5*9 = 4.5

  auto bnWeight = fl::full({1, outC, 1, 1}, 1.0f);
  auto bnBias = fl::full({1, outC, 1, 1}, 0.0f);
  auto bnRM = fl::full({1, outC, 1, 1}, 4.5f); // mean matches conv output
  auto bnRV = fl::full({1, outC, 1, 1}, 1.0f);
  auto bnResult = convOut.backend().batchnormForward(
      convOut, bnWeight, bnBias, bnRM, bnRV,
      /*axes=*/{1}, /*train=*/false,
      /*momentum=*/0.1, /*epsilon=*/1e-5, /*wantState=*/false);
  // BN inference: (4.5 - 4.5) / sqrt(1.0 + 1e-5) * 1.0 + 0.0 ≈ 0.0
  // relu(0.0) = 0.0
  auto result = fl::maximum(bnResult.output, 0.0);
  ASSERT_EQ(result.shape(), Shape({N, outC, 4, 4}));
  std::vector<float> host(N * outC * 4 * 4);
  result.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 0.0f, 1e-3);
  }
}

TEST_F(CudaJitEndToEndTest, Conv2dBNInferenceReluNonzero) {
  // relu(batchnorm_inference(conv2d(x, w, b))) with positive output
  int N = 1, C = 1, H = 6, W = 6;
  int outC = 2, kH = 3, kW = 3;
  auto input = fl::full({N, C, H, W}, 1.0f);
  auto weights = fl::full({outC, C, kH, kW}, 0.5f);
  auto convBias = fl::full({1, outC, 1, 1}, 0.0f);
  Tensor convOut;
  input.backend().conv2dForward(
      convOut, input, weights, convBias,
      /*sx=*/1, /*sy=*/1, /*px=*/0, /*py=*/0,
      /*dx=*/1, /*dy=*/1, /*groups=*/1, /*wantState=*/false);
  // convOut shape: {1, 2, 4, 4}, value: 0.5*9 = 4.5

  auto bnWeight = fl::full({1, outC, 1, 1}, 1.0f);
  auto bnBias = fl::full({1, outC, 1, 1}, 2.0f); // bias shifts output positive
  auto bnRM = fl::full({1, outC, 1, 1}, 4.5f);
  auto bnRV = fl::full({1, outC, 1, 1}, 1.0f);
  auto bnResult = convOut.backend().batchnormForward(
      convOut, bnWeight, bnBias, bnRM, bnRV,
      /*axes=*/{1}, /*train=*/false,
      /*momentum=*/0.1, /*epsilon=*/1e-5, /*wantState=*/false);
  // BN inference: (4.5 - 4.5) / sqrt(1.0 + 1e-5) * 1.0 + 2.0 ≈ 2.0
  // relu(2.0) = 2.0
  auto result = fl::maximum(bnResult.output, 0.0);
  std::vector<float> host(N * outC * 4 * 4);
  result.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 2.0f, 1e-3);
  }
}

TEST_F(CudaJitEndToEndTest, Conv2dBNInferenceResidualRelu) {
  // relu(batchnorm_inference(conv2d(x, w, b)) + shortcut)
  // — full ResNet block pattern (inference mode)
  int N = 1, C = 1, H = 6, W = 6;
  int outC = 2, kH = 3, kW = 3;
  auto input = fl::full({N, C, H, W}, 1.0f);
  auto weights = fl::full({outC, C, kH, kW}, 0.5f);
  auto convBias = fl::full({1, outC, 1, 1}, 0.0f);
  Tensor convOut;
  input.backend().conv2dForward(
      convOut, input, weights, convBias,
      /*sx=*/1, /*sy=*/1, /*px=*/0, /*py=*/0,
      /*dx=*/1, /*dy=*/1, /*groups=*/1, /*wantState=*/false);

  auto bnWeight = fl::full({1, outC, 1, 1}, 1.0f);
  auto bnBias = fl::full({1, outC, 1, 1}, 0.0f);
  auto bnRM = fl::full({1, outC, 1, 1}, 4.5f);
  auto bnRV = fl::full({1, outC, 1, 1}, 1.0f);
  auto bnResult = convOut.backend().batchnormForward(
      convOut, bnWeight, bnBias, bnRM, bnRV,
      /*axes=*/{1}, /*train=*/false,
      /*momentum=*/0.1, /*epsilon=*/1e-5, /*wantState=*/false);
  // BN output ≈ 0.0 (mean-centered)
  auto shortcut = fl::full({N, outC, 4, 4}, 5.0f);
  // 0.0 + 5.0 = 5.0, relu(5.0) = 5.0
  auto result = fl::maximum(bnResult.output + shortcut, 0.0);
  std::vector<float> host(N * outC * 4 * 4);
  result.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 5.0f, 1e-3);
  }
}

// ========================= Large Tensors =========================

TEST_F(CudaJitEndToEndTest, LargeTensorElementwiseFusion) {
  // Large tensor to stress-test the NVRTC kernel launch
  int n = 1 << 20;  // ~1M elements
  auto a = fl::full({n}, 1.0f);
  auto b = fl::full({n}, 2.0f);
  auto result = fl::sqrt(a + b);  // sqrt(3) ≈ 1.732
  float expected = std::sqrt(3.0f);
  // Check a few values via host copy
  std::vector<float> host(n);
  result.hostSync(host.data());
  ASSERT_NEAR(host[0], expected, 1e-5);
  ASSERT_NEAR(host[n / 2], expected, 1e-5);
  ASSERT_NEAR(host[n - 1], expected, 1e-5);
}

TEST_F(CudaJitEndToEndTest, LargeTensorReductionFusion) {
  // sum(exp(x)) for a large tensor
  int n = 1 << 16;  // 64K elements
  auto t = fl::full({n}, 0.0f);
  auto result = fl::sum(fl::exp(t), {});
  // exp(0) = 1, sum of n ones = n
  ASSERT_NEAR(
      result.scalarSync<float>(), static_cast<float>(n), static_cast<float>(n) * 1e-5);
}

// ========================= Explicit Eval =========================

TEST_F(CudaJitEndToEndTest, ExplicitEval) {
  // Build a JIT graph, explicitly eval, then use the result
  auto a = fl::full({16}, 3.0f);
  auto b = fl::full({16}, 4.0f);
  auto sum = a + b;

  // Before eval, the JIT graph hasn't been evaluated
  fl::eval(sum);  // Force evaluation

  // Now use the evaluated result in further computation
  auto result = sum * sum;
  std::vector<float> host(16);
  result.hostSync(host.data());
  for (float v : host) {
    ASSERT_FLOAT_EQ(v, 49.0f);  // 7*7
  }
}

TEST_F(CudaJitEndToEndTest, MultipleEvalsOfSameGraph) {
  // Evaluate the same expression multiple times -- should be idempotent
  auto t = fl::full({8}, 2.0f);
  auto result = fl::exp(t);
  float expected = std::exp(2.0f);

  std::vector<float> host1(8), host2(8);
  result.hostSync(host1.data());  // First eval
  result.hostSync(host2.data());  // Second eval (should return cached result)
  for (int i = 0; i < 8; i++) {
    ASSERT_FLOAT_EQ(host1[i], host2[i]);
    ASSERT_NEAR(host1[i], expected, 1e-5);
  }
}

// ========================= 2D Operations =========================

TEST_F(CudaJitEndToEndTest, ElementwiseFusion2D) {
  // 2D tensor elementwise fusion
  auto a = Tensor::fromVector(
      Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  auto b = Tensor::fromVector(
      Shape{2, 3}, std::vector<float>{6, 5, 4, 3, 2, 1});
  auto result = fl::exp(a + b);  // exp(7) for all elements
  ASSERT_EQ(result.shape(), Shape({2, 3}));
  std::vector<float> host(6);
  result.hostSync(host.data());
  float expected = std::exp(7.0f);
  for (float v : host) {
    ASSERT_NEAR(v, expected, expected * 1e-5);
  }
}

TEST_F(CudaJitEndToEndTest, AxisReductionOfFused2D) {
  // sum(a * b, axis=0) on a 2D tensor
  auto a = Tensor::fromVector(
      Shape{3, 4}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  auto b = fl::full({3, 4}, 2.0f);
  auto result = fl::sum(a * b, {0});
  ASSERT_EQ(result.shape(), Shape({4}));
  std::vector<float> host(4);
  result.hostSync(host.data());
  // Column sums of 2*a:
  // col 0: 2*(1+5+9) = 30
  // col 1: 2*(2+6+10) = 36
  // col 2: 2*(3+7+11) = 42
  // col 3: 2*(4+8+12) = 48
  ASSERT_NEAR(host[0], 30.0f, 1e-4);
  ASSERT_NEAR(host[1], 36.0f, 1e-4);
  ASSERT_NEAR(host[2], 42.0f, 1e-4);
  ASSERT_NEAR(host[3], 48.0f, 1e-4);
}

// ========================= Non-fusable Operations =========================

TEST_F(CudaJitEndToEndTest, NonFusableOps) {
  // Operations like reshape, transpose, sort go through CustomNode
  // and should still work correctly through the JIT
  auto t = Tensor::fromVector(
      Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  auto reshaped = fl::reshape(t, Shape{3, 2});
  ASSERT_EQ(reshaped.shape(), Shape({3, 2}));
  std::vector<float> host(6);
  reshaped.hostSync(host.data());
  // Reshape preserves element order
  ASSERT_FLOAT_EQ(host[0], 1.0f);
  ASSERT_FLOAT_EQ(host[1], 2.0f);
  ASSERT_FLOAT_EQ(host[2], 3.0f);
  ASSERT_FLOAT_EQ(host[3], 4.0f);
  ASSERT_FLOAT_EQ(host[4], 5.0f);
  ASSERT_FLOAT_EQ(host[5], 6.0f);
}

TEST_F(CudaJitEndToEndTest, FusionAfterNonFusable) {
  // Reshape then elementwise chain -- elementwise part should be fused
  auto t = Tensor::fromVector(
      Shape{6}, std::vector<float>{1, 2, 3, 4, 5, 6});
  auto reshaped = fl::reshape(t, Shape{2, 3});
  auto result = fl::exp(reshaped + fl::full({2, 3}, 1.0f));
  ASSERT_EQ(result.shape(), Shape({2, 3}));
  std::vector<float> host(6);
  result.hostSync(host.data());
  ASSERT_NEAR(host[0], std::exp(2.0f), 1e-4);
  ASSERT_NEAR(host[5], std::exp(7.0f), std::exp(7.0f) * 1e-5);
}

// ====================== Long JIT Chain Tests ======================

TEST_F(CudaJitEndToEndTest, EvalCompactsGraph) {
  auto t = fl::rand({100});

  // Build a chain of indexed updates
  for (int i = 0; i < 50; i++) {
    t(i) = fl::full({1}, static_cast<float>(i));
  }

  // Before eval: node should be a deep IndexedUpdateNode chain
  auto& jitTensor = toJitTensorBase(t);
  auto preEvalNode = jitTensor.node();
  ASSERT_TRUE(preEvalNode->isIndexedUpdate());

  // Eval should compute the result AND compact the graph
  fl::eval(t);

  // After eval: node should be a ValueNode (graph compacted)
  auto postEvalNode = jitTensor.node();
  ASSERT_TRUE(postEvalNode->isValue())
      << "After eval, node should be a ValueNode but got "
      << nodeTypeToString(postEvalNode->type());
  ASSERT_EQ(countNodes(postEvalNode), 1u);

  // Values should be correct
  ASSERT_EQ(t(0).scalarSync<float>(), 0.0f);
  ASSERT_EQ(t(49).scalarSync<float>(), 49.0f);
}

TEST_F(CudaJitEndToEndTest, RepeatedEvalKeepsGraphShallow) {
  // Mimics the Jacobian test pattern: perturb a few elements, eval, repeat.
  // Graph depth should stay O(1) after each eval.
  auto t = fl::rand({100});
  fl::eval(t);

  for (int iter = 0; iter < 500; iter++) {
    t(iter % 100) = fl::full({1}, static_cast<float>(iter));
    t((iter + 1) % 100) = fl::full({1}, static_cast<float>(iter + 1));
    t((iter + 2) % 100) = fl::full({1}, static_cast<float>(iter + 2));
    fl::eval(t);

    auto& jitTensor = toJitTensorBase(t);
    ASSERT_TRUE(jitTensor.node()->isValue())
        << "After eval at iter " << iter
        << ", node should be ValueNode but got "
        << nodeTypeToString(jitTensor.node()->type());
    ASSERT_EQ(countNodes(jitTensor.node()), 1u);
  }
}

TEST_F(CudaJitEndToEndTest, DeepGraphNoStackOverflow) {
  // Build a very deep graph (5000 indexed updates) and eval in one shot.
  auto t = fl::rand({10});

  for (int i = 0; i < 5000; i++) {
    t(i % 10) = fl::full({1}, static_cast<float>(i));
  }

  fl::eval(t);

  auto& jitTensor = toJitTensorBase(t);
  ASSERT_TRUE(jitTensor.node()->isValue());
  ASSERT_EQ(countNodes(jitTensor.node()), 1u);
  ASSERT_EQ(t(0).scalarSync<float>(), 4990.0f);
  ASSERT_EQ(t(9).scalarSync<float>(), 4999.0f);
}

TEST_F(CudaJitEndToEndTest, LongElementwiseChain) {
  // Build a chain of 200 elementwise ops without intermediate evals.
  // Tests that CUDA stream-ordered memory + graph compaction work together.
  auto t = fl::full({1024}, 1.0f);
  for (int i = 0; i < 100; i++) {
    t = t + fl::full({1024}, 0.01f);
    t = t * fl::full({1024}, 1.0f); // identity mul to deepen graph
  }
  fl::eval(t);

  // 1.0 + 100 * 0.01 = 2.0
  std::vector<float> host(1024);
  t.hostSync(host.data());
  for (int i = 0; i < 1024; i++) {
    ASSERT_NEAR(host[i], 2.0f, 1e-3);
  }
}

TEST_F(CudaJitEndToEndTest, RepeatedEvalMemoryStability) {
  // Run many eval cycles with moderate-sized tensors.
  // Verifies that CUDA stream-ordered allocation doesn't leak memory.
  for (int iter = 0; iter < 200; iter++) {
    auto a = fl::rand({256, 256});
    auto b = fl::rand({256, 256});
    auto c = fl::exp(a + b);
    fl::eval(c);
  }
  // If we get here without OOM, the test passes.
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
