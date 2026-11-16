/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/runtime/Stream.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"

using namespace fl;

class OneDnnJitEndToEndTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fl::setDefaultJitTensorType<CPUTensor>();
  }
};

// ========================= Binary Chain Fusion =========================

TEST_F(OneDnnJitEndToEndTest, BinaryChainFusion) {
  // (a + b) * c -- 2 binary ops fused: primary binary_mul + post-op binary_add
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

// ========================= Eltwise Post-Op Fusion =========================

TEST_F(OneDnnJitEndToEndTest, UnaryEltwisePostOp) {
  // exp(a + b) -- binary add + eltwise_exp post-op
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

// ===================== Mixed Binary + Eltwise Chain =====================

TEST_F(OneDnnJitEndToEndTest, MixedBinaryEltwiseChain) {
  // tanh(a * b + c) -- primary mul, post-op binary_add(c), post-op eltwise_tanh
  auto a = fl::full({1024}, 1.0f);
  auto b = fl::full({1024}, 2.0f);
  auto c = fl::full({1024}, 3.0f);
  auto result = fl::tanh(a * b + c);  // tanh(1*2+3) = tanh(5)
  std::vector<float> host(1024);
  result.hostSync(host.data());
  float expected = std::tanh(5.0f);
  for (int i = 0; i < 1024; i++) {
    ASSERT_NEAR(host[i], expected, 1e-5);
  }
}

// ===================== Commutative Swap Fusion =====================

TEST_F(OneDnnJitEndToEndTest, CommutativeSwapFusion) {
  // c + (a - b) -- add is commutative; fusable chain is on RHS.
  // Should swap: primary sub(a,b), post-op binary_add(c).
  auto a = Tensor::fromVector(Shape{4}, std::vector<float>{10, 20, 30, 40});
  auto b = Tensor::fromVector(Shape{4}, std::vector<float>{1, 2, 3, 4});
  auto c = Tensor::fromVector(Shape{4}, std::vector<float>{100, 200, 300, 400});
  auto result = c + (a - b);
  std::vector<float> host(4);
  result.hostSync(host.data());
  ASSERT_FLOAT_EQ(host[0], 109.0f);   // 100 + (10-1)
  ASSERT_FLOAT_EQ(host[1], 218.0f);   // 200 + (20-2)
  ASSERT_FLOAT_EQ(host[2], 327.0f);   // 300 + (30-3)
  ASSERT_FLOAT_EQ(host[3], 436.0f);   // 400 + (40-4)
}

// ===================== Multi-Use Prevents Fusion =====================

TEST_F(OneDnnJitEndToEndTest, MultiUsePrevents) {
  // s = a + b has two uses --> should NOT be fused into either consumer.
  // Both results must still be numerically correct.
  auto a = fl::full({256}, 1.0f);
  auto b = fl::full({256}, 2.0f);
  auto s = a + b;  // s has two consumers
  auto r1 = s * fl::full({256}, 3.0f);
  auto r2 = s * fl::full({256}, 4.0f);
  std::vector<float> h1(256), h2(256);
  r1.hostSync(h1.data());
  r2.hostSync(h2.data());
  for (int i = 0; i < 256; i++) {
    ASSERT_NEAR(h1[i], 9.0f, 1e-5);    // (1+2)*3
    ASSERT_NEAR(h2[i], 12.0f, 1e-5);   // (1+2)*4
  }
}

// ===================== Sigmoid Post-Op =====================

TEST_F(OneDnnJitEndToEndTest, SigmoidPostOp) {
  // sigmoid(a * b) -- primary mul + eltwise_logistic post-op
  auto a = Tensor::fromVector(Shape{4}, std::vector<float>{1, -1, 2, -2});
  auto b = fl::full({4}, 1.0f);
  auto result = fl::sigmoid(a * b);
  std::vector<float> host(4);
  result.hostSync(host.data());
  for (int i = 0; i < 4; i++) {
    float x = (i < 2) ? (i == 0 ? 1.0f : -1.0f) : (i == 2 ? 2.0f : -2.0f);
    float expected = 1.0f / (1.0f + std::exp(-x));
    ASSERT_NEAR(host[i], expected, 1e-5);
  }
}

// ===================== Negative Post-Op =====================

TEST_F(OneDnnJitEndToEndTest, NegativePostOp) {
  // negative(a + b) -- primary add + eltwise_linear(-1, 0) post-op
  auto a = Tensor::fromVector(Shape{4}, std::vector<float>{1, 2, 3, 4});
  auto b = Tensor::fromVector(Shape{4}, std::vector<float>{10, 20, 30, 40});
  auto result = fl::negative(a + b);
  std::vector<float> host(4);
  result.hostSync(host.data());
  ASSERT_FLOAT_EQ(host[0], -11.0f);
  ASSERT_FLOAT_EQ(host[1], -22.0f);
  ASSERT_FLOAT_EQ(host[2], -33.0f);
  ASSERT_FLOAT_EQ(host[3], -44.0f);
}

// ===================== Longer Chain =====================

TEST_F(OneDnnJitEndToEndTest, LongerChain) {
  // exp(tanh(a * b + c)) -- 4 ops fused:
  //   primary mul, post-op binary_add(c), post-op eltwise_tanh, post-op eltwise_exp
  auto a = fl::full({512}, 1.0f);
  auto b = fl::full({512}, 0.5f);
  auto c = fl::full({512}, 0.5f);
  auto result = fl::exp(fl::tanh(a * b + c));  // exp(tanh(0.5+0.5)) = exp(tanh(1))
  std::vector<float> host(512);
  result.hostSync(host.data());
  float expected = std::exp(std::tanh(1.0f));
  for (int i = 0; i < 512; i++) {
    ASSERT_NEAR(host[i], expected, 1e-5);
  }
}

// ===================== Non-Fusable Unary Breaks Chain =====================

TEST_F(OneDnnJitEndToEndTest, NonFusableUnaryBreaksChain) {
  // sin(a + b) -- sin is NOT mapped to a OneDNN eltwise, so the pass should
  // NOT fuse sin. a + b alone is only 1 op (below threshold). Result should
  // still be numerically correct via eager evaluation.
  auto a = Tensor::fromVector(Shape{4}, std::vector<float>{0, 0.5f, 1.0f, 1.5f});
  auto b = Tensor::fromVector(Shape{4}, std::vector<float>{0, 0.5f, 1.0f, 1.5f});
  auto result = fl::sin(a + b);
  std::vector<float> host(4);
  result.hostSync(host.data());
  ASSERT_NEAR(host[0], std::sin(0.0f), 1e-5);
  ASSERT_NEAR(host[1], std::sin(1.0f), 1e-5);
  ASSERT_NEAR(host[2], std::sin(2.0f), 1e-5);
  ASSERT_NEAR(host[3], std::sin(3.0f), 1e-5);
}

// ===================== Comparison Ops =====================

TEST_F(OneDnnJitEndToEndTest, ComparisonOps) {
  // (a + b) > c -- primary add + post-op binary_gt
  auto a = Tensor::fromVector(Shape{4}, std::vector<float>{1, 5, 3, 7});
  auto b = Tensor::fromVector(Shape{4}, std::vector<float>{2, 2, 2, 2});
  auto c = Tensor::fromVector(Shape{4}, std::vector<float>{4, 4, 4, 4});
  auto result = (a + b) > c;
  // a+b = {3, 7, 5, 9}; compare > 4 --> {false, true, true, true}
  // Comparison returns b8; use asScalar to read as bool
  ASSERT_FALSE(result(0).asScalar<bool>());  // 3 > 4 = false
  ASSERT_TRUE(result(1).asScalar<bool>());   // 7 > 4 = true
  ASSERT_TRUE(result(2).asScalar<bool>());   // 5 > 4 = true
  ASSERT_TRUE(result(3).asScalar<bool>());   // 9 > 4 = true
}

// ===================== Type Inference Tests =====================

TEST_F(OneDnnJitEndToEndTest, TypeDoesNotMaterialize) {
  // type() should return the correct dtype without forcing evaluation
  auto a = Tensor::fromVector(Shape{4}, std::vector<float>{1, 2, 3, 4});
  auto b = Tensor::fromVector(Shape{4}, std::vector<float>{10, 20, 30, 40});
  auto result = a + b;
  // type() should work without materializing
  ASSERT_EQ(result.type(), dtype::f32);
  // Verify the result node is still unevaluated
  ASSERT_FALSE(toJitTensorBase(result).node()->getResult().has_value());
  // Now verify correct values
  std::vector<float> host(4);
  result.hostSync(host.data());
  ASSERT_FLOAT_EQ(host[0], 11.0f);
  ASSERT_FLOAT_EQ(host[1], 22.0f);
  ASSERT_FLOAT_EQ(host[2], 33.0f);
  ASSERT_FLOAT_EQ(host[3], 44.0f);
}

TEST_F(OneDnnJitEndToEndTest, TypeInferenceChain) {
  // Verify type inference through a chain of operations
  auto a = fl::full({4}, 1.0f);
  auto b = fl::full({4}, 2.0f);
  auto sum = a + b;           // f32
  auto cmp = sum > b;         // b8 (comparison)
  auto neg = fl::negative(a); // f32

  ASSERT_EQ(sum.type(), dtype::f32);
  ASSERT_EQ(cmp.type(), dtype::b8);
  ASSERT_EQ(neg.type(), dtype::f32);

  // None should have been evaluated
  ASSERT_FALSE(toJitTensorBase(sum).node()->getResult().has_value());
  ASSERT_FALSE(toJitTensorBase(cmp).node()->getResult().has_value());
  ASSERT_FALSE(toJitTensorBase(neg).node()->getResult().has_value());
}

// ===================== Stream Inference Tests =====================

TEST_F(OneDnnJitEndToEndTest, StreamInferenceNoEval) {
  // a + b has ValueNode leaves; stream() should not force eval
  auto a = Tensor::fromVector(Shape{4}, std::vector<float>{1, 2, 3, 4});
  auto b = Tensor::fromVector(Shape{4}, std::vector<float>{10, 20, 30, 40});
  auto result = a + b;
  // stream() should work without materializing
  const auto& s = result.stream();
  ASSERT_EQ(s.type(), StreamType::AsyncCPU);
  // Verify the result node is still unevaluated
  ASSERT_FALSE(toJitTensorBase(result).node()->getResult().has_value());
  // Now verify correct values
  std::vector<float> host(4);
  result.hostSync(host.data());
  ASSERT_FLOAT_EQ(host[0], 11.0f);
  ASSERT_FLOAT_EQ(host[1], 22.0f);
  ASSERT_FLOAT_EQ(host[2], 33.0f);
  ASSERT_FLOAT_EQ(host[3], 44.0f);
}

TEST_F(OneDnnJitEndToEndTest, StreamInferencePureScalar) {
  // full() + full() creates ScalarNode only (no ValueNode leaves)
  // stream() should return the default AsyncCPU stream without eval
  auto a = fl::full({4}, 1.0f);
  auto b = fl::full({4}, 2.0f);
  auto result = a + b;
  const auto& s = result.stream();
  ASSERT_EQ(s.type(), StreamType::AsyncCPU);
  // Verify the result node is still unevaluated
  ASSERT_FALSE(toJitTensorBase(result).node()->getResult().has_value());
}

TEST_F(OneDnnJitEndToEndTest, LazyAsyncComposition) {
  // Build ((a+b)*2)-1 chain, verify all intermediates stay lazy
  auto a = Tensor::fromVector(Shape{4}, std::vector<float>{1, 2, 3, 4});
  auto b = Tensor::fromVector(Shape{4}, std::vector<float>{10, 20, 30, 40});
  auto sum = a + b;
  auto scaled = sum * fl::full({4}, 2.0f);
  auto result = scaled - fl::full({4}, 1.0f);
  // All intermediate nodes should be unevaluated
  ASSERT_FALSE(toJitTensorBase(sum).node()->getResult().has_value());
  ASSERT_FALSE(toJitTensorBase(scaled).node()->getResult().has_value());
  ASSERT_FALSE(toJitTensorBase(result).node()->getResult().has_value());
  // stream() should work without eval on all of them
  ASSERT_EQ(sum.stream().type(), StreamType::AsyncCPU);
  ASSERT_EQ(scaled.stream().type(), StreamType::AsyncCPU);
  ASSERT_EQ(result.stream().type(), StreamType::AsyncCPU);
  // Still unevaluated after stream() calls
  ASSERT_FALSE(toJitTensorBase(result).node()->getResult().has_value());
  // Verify correct final values: ((a+b)*2)-1
  std::vector<float> host(4);
  result.hostSync(host.data());
  ASSERT_FLOAT_EQ(host[0], 21.0f);  // (1+10)*2-1
  ASSERT_FLOAT_EQ(host[1], 43.0f);  // (2+20)*2-1
  ASSERT_FLOAT_EQ(host[2], 65.0f);  // (3+30)*2-1
  ASSERT_FLOAT_EQ(host[3], 87.0f);  // (4+40)*2-1
}

// ===================== Async Eval Tests =====================

TEST_F(OneDnnJitEndToEndTest, EvalIsNonBlocking) {
  // Build a + b, call fl::eval(result), verify eval just enqueues
  // (the node may or may not have a result yet depending on timing).
  // Then host() triggers sync and gives correct values.
  auto a = Tensor::fromVector(Shape{4}, std::vector<float>{1, 2, 3, 4});
  auto b = Tensor::fromVector(Shape{4}, std::vector<float>{10, 20, 30, 40});
  auto result = a + b;
  fl::eval(result);  // enqueues; should not block
  // Verify correct values after sync via host()
  std::vector<float> host(4);
  result.hostSync(host.data());
  ASSERT_FLOAT_EQ(host[0], 11.0f);
  ASSERT_FLOAT_EQ(host[1], 22.0f);
  ASSERT_FLOAT_EQ(host[2], 33.0f);
  ASSERT_FLOAT_EQ(host[3], 44.0f);
}

TEST_F(OneDnnJitEndToEndTest, EvalThenHost) {
  // Build a deeper chain ((a+b)*c)-d, call fl::eval(), then host().
  // Exercises re-entrant enqueue pattern (graph walk on worker enqueues
  // compute tasks onto the same stream).
  auto a = Tensor::fromVector(Shape{4}, std::vector<float>{1, 2, 3, 4});
  auto b = Tensor::fromVector(Shape{4}, std::vector<float>{10, 20, 30, 40});
  auto c = Tensor::fromVector(Shape{4}, std::vector<float>{2, 2, 2, 2});
  auto d = Tensor::fromVector(Shape{4}, std::vector<float>{1, 1, 1, 1});
  auto result = ((a + b) * c) - d;
  fl::eval(result);
  std::vector<float> host(4);
  result.hostSync(host.data());
  ASSERT_FLOAT_EQ(host[0], 21.0f);   // (1+10)*2-1
  ASSERT_FLOAT_EQ(host[1], 43.0f);   // (2+20)*2-1
  ASSERT_FLOAT_EQ(host[2], 65.0f);   // (3+30)*2-1
  ASSERT_FLOAT_EQ(host[3], 87.0f);   // (4+40)*2-1
}

TEST_F(OneDnnJitEndToEndTest, EvalIdempotent) {
  // Call eval() twice on the same tensor, verify no double-enqueue or crash.
  auto a = Tensor::fromVector(Shape{4}, std::vector<float>{1, 2, 3, 4});
  auto b = Tensor::fromVector(Shape{4}, std::vector<float>{10, 20, 30, 40});
  auto result = a + b;
  fl::eval(result);  // first eval
  fl::eval(result);  // second eval — should be a no-op
  std::vector<float> host(4);
  result.hostSync(host.data());
  ASSERT_FLOAT_EQ(host[0], 11.0f);
  ASSERT_FLOAT_EQ(host[1], 22.0f);
  ASSERT_FLOAT_EQ(host[2], 33.0f);
  ASSERT_FLOAT_EQ(host[3], 44.0f);
}

TEST_F(OneDnnJitEndToEndTest, StreamAfterEval) {
  // Force eval via host(), then verify stream() still works
  auto a = Tensor::fromVector(Shape{4}, std::vector<float>{1, 2, 3, 4});
  auto b = Tensor::fromVector(Shape{4}, std::vector<float>{10, 20, 30, 40});
  auto result = a + b;
  // Force evaluation
  std::vector<float> host(4);
  result.hostSync(host.data());
  // Now the node should have a result
  ASSERT_TRUE(toJitTensorBase(result).node()->getResult().has_value());
  // stream() should still work (fast path)
  const auto& s = result.stream();
  ASSERT_EQ(s.type(), StreamType::AsyncCPU);
}

// ===================== Reshape/Flatten/Cast Lazy Tests =====================

TEST_F(OneDnnJitEndToEndTest, FlattenDoesNotMaterialize) {
  // flatten() should use ReshapeNode and NOT force evaluation
  auto a = Tensor::fromVector(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  auto flat = a.flatten();
  // Node should not be evaluated yet
  ASSERT_FALSE(toJitTensorBase(flat).node()->getResult().has_value());
  ASSERT_EQ(flat.shape(), Shape({6}));
  // Verify correct values
  std::vector<float> host(6);
  flat.hostSync(host.data());
  ASSERT_FLOAT_EQ(host[0], 1.0f);
  ASSERT_FLOAT_EQ(host[5], 6.0f);
}

TEST_F(OneDnnJitEndToEndTest, FlatIndexComposesLazily) {
  // flat(idx) should compose flatten + index lazily
  auto a = Tensor::fromVector(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  auto elem = a.flat(2);
  // Verify correct value
  ASSERT_FLOAT_EQ(elem.scalarSync<float>(), 3.0f);
}

TEST_F(OneDnnJitEndToEndTest, ReshapeStaysLazy) {
  auto a = Tensor::fromVector(Shape{6}, std::vector<float>{1, 2, 3, 4, 5, 6});
  auto b = fl::reshape(a, Shape{2, 3});
  ASSERT_FALSE(toJitTensorBase(b).node()->getResult().has_value());
  ASSERT_EQ(b.shape(), Shape({2, 3}));
  std::vector<float> host(6);
  b.hostSync(host.data());
  ASSERT_FLOAT_EQ(host[0], 1.0f);
  ASSERT_FLOAT_EQ(host[5], 6.0f);
}

TEST_F(OneDnnJitEndToEndTest, TransposeStaysLazy) {
  auto a = Tensor::fromVector(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  auto t = fl::transpose(a, Shape{1, 0});
  ASSERT_FALSE(toJitTensorBase(t).node()->getResult().has_value());
  ASSERT_EQ(t.shape(), Shape({3, 2}));
  std::vector<float> host(6);
  t.hostSync(host.data());
  ASSERT_FLOAT_EQ(host[0], 1.0f);
}

TEST_F(OneDnnJitEndToEndTest, CastStaysLazy) {
  auto a = Tensor::fromVector(Shape{4}, std::vector<float>{1, 2, 3, 4});
  auto c = a.astype(dtype::f64);
  ASSERT_FALSE(toJitTensorBase(c).node()->getResult().has_value());
  ASSERT_EQ(c.type(), dtype::f64);
  ASSERT_EQ(c.shape(), Shape({4}));
}

TEST_F(OneDnnJitEndToEndTest, CastNoOp) {
  auto a = Tensor::fromVector(Shape{4}, std::vector<float>{1, 2, 3, 4});
  auto c = a.astype(dtype::f32);
  // No-op cast: should return the same underlying node
  ASSERT_EQ(toJitTensorBase(c).node(), toJitTensorBase(a).node());
}

// ===================== Conv2d Through JIT =====================

TEST_F(OneDnnJitEndToEndTest, Conv2dForward) {
  // Basic conv2d: 1 batch, 3 input channels, 8 output channels, 3x3 kernel
  // Input: NCHW = {1, 3, 8, 8}
  // Weight: {outC, inC, kH, kW} = {8, 3, 3, 3}
  auto input = fl::rand({1, 3, 8, 8});
  auto weights = fl::rand({8, 3, 3, 3});
  auto result = fl::autograd::conv2d(input, weights);
  // Output shape: {N, outC, outH, outW} = {1, 8, 6, 6}
  ASSERT_EQ(result.shape(), Shape({1, 8, 6, 6}));
  fl::eval(result);
  // Verify no NaN in output
  ASSERT_FALSE(fl::any(fl::isnan(result), {}).asScalar<bool>());
}

TEST_F(OneDnnJitEndToEndTest, Conv2dGrouped) {
  // Grouped convolution: 2 groups, 4 input channels, 8 output channels
  // Input: {1, 4, 8, 8}, Weight: {8, 2, 3, 3} (inC/groups = 2)
  auto input = fl::rand({1, 4, 8, 8});
  auto weights = fl::rand({8, 2, 3, 3});
  auto result = fl::autograd::conv2d(
      input, weights, /*sx=*/1, /*sy=*/1, /*px=*/0, /*py=*/0,
      /*dx=*/1, /*dy=*/1, /*groups=*/2);
  ASSERT_EQ(result.shape(), Shape({1, 8, 6, 6}));
  fl::eval(result);
  ASSERT_FALSE(fl::any(fl::isnan(result), {}).asScalar<bool>());
}

TEST_F(OneDnnJitEndToEndTest, Conv2dStridedDilated) {
  // Strided + dilated convolution
  // Input: {2, 3, 16, 16}, Weight: {4, 3, 3, 3}
  // stride=2, padding=1, dilation=2
  auto input = fl::rand({2, 3, 16, 16});
  auto weights = fl::rand({4, 3, 3, 3});
  auto result = fl::autograd::conv2d(
      input, weights, /*sx=*/2, /*sy=*/2, /*px=*/1, /*py=*/1,
      /*dx=*/2, /*dy=*/2);
  // outH = floor((16 + 2*1 - 2*(3-1) - 1) / 2 + 1) = floor((18-5)/2 + 1) = 7
  ASSERT_EQ(result.shape(), Shape({2, 4, 7, 7}));
  fl::eval(result);
  ASSERT_FALSE(fl::any(fl::isnan(result), {}).asScalar<bool>());
}

// ================ Conv2d + Activation Post-Op Fusion ================

TEST_F(OneDnnJitEndToEndTest, Conv2dReluFusion) {
  // max(conv2d(x, w, b), 0) should be fused into a single OneDNN conv with
  // relu post-op. Verify output matches unfused computation.
  auto input = fl::rand({1, 3, 8, 8});
  auto weights = fl::rand({8, 3, 3, 3});
  auto bias = fl::rand({1, 8, 1, 1});

  // Fused path (JIT will fuse max(..., 0) onto conv2d)
  auto convOut = fl::autograd::conv2d(input, weights, bias);
  auto fused = fl::maximum(convOut, fl::full(convOut.shape(), 0.f));
  fl::eval(fused);

  // Reference: eager conv2d + relu (force eval between steps)
  auto refConv = fl::autograd::conv2d(input, weights, bias);
  fl::eval(refConv);
  auto refRelu = fl::maximum(refConv, fl::full(refConv.shape(), 0.f));
  fl::eval(refRelu);

  ASSERT_EQ(fused.shape(), refRelu.shape());
  ASSERT_TRUE(fl::allClose(fused, refRelu, 1e-5));
}

TEST_F(OneDnnJitEndToEndTest, Conv2dSigmoidFusion) {
  // sigmoid(conv2d(x, w, b)) — UnaryOp activation fused as eltwise post-op
  auto input = fl::rand({2, 3, 8, 8});
  auto weights = fl::rand({4, 3, 3, 3});
  auto bias = fl::rand({1, 4, 1, 1});

  auto fused = fl::sigmoid(fl::autograd::conv2d(input, weights, bias));
  fl::eval(fused);

  auto refConv = fl::autograd::conv2d(input, weights, bias);
  fl::eval(refConv);
  auto ref = fl::sigmoid(refConv);
  fl::eval(ref);

  ASSERT_TRUE(fl::allClose(fused, ref, 1e-5));
}

TEST_F(OneDnnJitEndToEndTest, Conv2dNoFusionMultipleUses) {
  // Conv2d result used by both relu AND another op → no fusion should occur.
  // Both results must still be correct.
  auto input = fl::rand({1, 3, 8, 8});
  auto weights = fl::rand({4, 3, 3, 3});

  auto convOut = fl::autograd::conv2d(input, weights);
  auto reluOut = fl::maximum(convOut, fl::full(convOut.shape(), 0.f));
  auto otherOut = convOut + fl::full(convOut.shape(), 1.f);

  fl::eval(reluOut);
  fl::eval(otherOut);

  // Verify relu is correct
  auto refConv = fl::autograd::conv2d(input, weights);
  fl::eval(refConv);
  auto refRelu = fl::maximum(refConv, fl::full(refConv.shape(), 0.f));
  fl::eval(refRelu);
  ASSERT_TRUE(fl::allClose(reluOut, refRelu, 1e-5));

  // Verify other use is correct
  auto refOther = refConv + fl::full(refConv.shape(), 1.f);
  fl::eval(refOther);
  ASSERT_TRUE(fl::allClose(otherOut, refOther, 1e-5));
}

// ================ Pool2d + Activation Post-Op Fusion ================

TEST_F(OneDnnJitEndToEndTest, Pool2dReluFusion) {
  auto input = fl::rand({1, 3, 8, 8});
  auto poolOut = fl::autograd::pool2d(input, 2, 2, 2, 2);
  auto fused = fl::maximum(poolOut, fl::full(poolOut.shape(), 0.f));
  fl::eval(fused);

  auto refPool = fl::autograd::pool2d(input, 2, 2, 2, 2);
  fl::eval(refPool);
  auto ref = fl::maximum(refPool, fl::full(refPool.shape(), 0.f));
  fl::eval(ref);

  ASSERT_TRUE(fl::allClose(fused, ref, 1e-5));
}

TEST_F(OneDnnJitEndToEndTest, Pool2dTanhFusion) {
  auto input = fl::rand({2, 4, 8, 8});
  auto fused = fl::tanh(fl::autograd::pool2d(
      input, 2, 2, 2, 2, 0, 0, PoolingMode::AVG_INCLUDE_PADDING));
  fl::eval(fused);

  auto refPool = fl::autograd::pool2d(
      input, 2, 2, 2, 2, 0, 0, PoolingMode::AVG_INCLUDE_PADDING);
  fl::eval(refPool);
  auto ref = fl::tanh(refPool);
  fl::eval(ref);

  ASSERT_TRUE(fl::allClose(fused, ref, 1e-5));
}

TEST_F(OneDnnJitEndToEndTest, Conv2dReluBackward) {
  // Forward with fused relu, then backward via autograd.
  // Verify gradients match unfused computation.
  auto input = fl::rand({1, 3, 8, 8});
  input.setRequiresGrad(true);
  auto weights = fl::rand({4, 3, 3, 3});
  weights.setRequiresGrad(true);

  // Fused: relu(conv) — autograd relu uses max(x, 0) which creates
  // max(BinaryNode) in the JIT IR, matching the fusion pattern.
  auto convOut = fl::autograd::conv2d(input, weights);
  auto activated = fl::autograd::relu(convOut);
  fl::autograd::sum(activated, {0, 1, 2, 3}).backward();
  auto fusedInputGrad = input.grad().copy();
  auto fusedWeightGrad = weights.grad().copy();
  fl::eval(fusedInputGrad);
  fl::eval(fusedWeightGrad);

  // Reset grads
  input.zeroGrad();
  weights.zeroGrad();

  // Reference: force eval between conv and relu to prevent fusion
  auto refConv = fl::autograd::conv2d(input, weights);
  fl::eval(refConv);
  auto refActivated = fl::autograd::relu(refConv);
  fl::autograd::sum(refActivated, {0, 1, 2, 3}).backward();
  auto refInputGrad = input.grad().copy();
  auto refWeightGrad = weights.grad().copy();
  fl::eval(refInputGrad);
  fl::eval(refWeightGrad);

  ASSERT_TRUE(fl::allClose(fusedInputGrad, refInputGrad, 1e-4));
  ASSERT_TRUE(fl::allClose(fusedWeightGrad, refWeightGrad, 1e-4));
}

TEST_F(OneDnnJitEndToEndTest, Pool2dReluBackward) {
  auto input = fl::rand({1, 3, 8, 8});
  input.setRequiresGrad(true);

  // Fused
  auto poolOut = fl::autograd::pool2d(input, 2, 2, 2, 2);
  auto activated = fl::autograd::relu(poolOut);
  fl::autograd::sum(activated, {0, 1, 2, 3}).backward();
  auto fusedGrad = input.grad().copy();
  fl::eval(fusedGrad);

  input.zeroGrad();

  // Unfused
  auto refPool = fl::autograd::pool2d(input, 2, 2, 2, 2);
  fl::eval(refPool);
  auto refActivated = fl::autograd::relu(refPool);
  fl::autograd::sum(refActivated, {0, 1, 2, 3}).backward();
  auto refGrad = input.grad().copy();
  fl::eval(refGrad);

  ASSERT_TRUE(fl::allClose(fusedGrad, refGrad, 1e-4));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
