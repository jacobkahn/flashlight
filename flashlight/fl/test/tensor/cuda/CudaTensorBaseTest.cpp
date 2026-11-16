/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/backend/cuda/CudaTensor.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"

using namespace fl;

class CudaTensorBaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fl::setDefaultJitTensorType<CudaTensor>();
  }
};

TEST_F(CudaTensorBaseTest, BackendType) {
  // Default tensors are JIT tensors wrapping the CUDA backend; once evaluated
  // their data lives in a CudaTensor.
  auto t = fl::full({2}, 1.0f);
  ASSERT_EQ(t.backendType(), TensorBackendType::Jit);
  fl::withTensorType<CudaTensor>([]() {
    auto bare = fl::full({2}, 1.0f);
    ASSERT_EQ(bare.backendType(), TensorBackendType::Cuda);
  });
}

TEST_F(CudaTensorBaseTest, Bf16Creation) {
  // Create bf16 tensors via full() and verify shape/type
  auto t = fl::full({3, 4}, 2.5f, fl::dtype::bf16);
  ASSERT_EQ(t.type(), fl::dtype::bf16);
  ASSERT_EQ(t.shape(), Shape({3, 4}));

  // Verify values by casting to f32
  auto tF32 = t.astype(fl::dtype::f32);
  std::vector<float> host(12);
  tF32.hostSync(host.data());
  for (float v : host) {
    ASSERT_NEAR(v, 2.5f, 0.01f);
  }

  // Scalar bf16
  auto s = fl::fromScalar(1.0f, fl::dtype::bf16);
  ASSERT_EQ(s.type(), fl::dtype::bf16);
  ASSERT_EQ(s.shape(), Shape({}));
}

TEST_F(CudaTensorBaseTest, Bf16Astype) {
  // f32 -> bf16 -> f32 round-trip
  auto orig = fl::full({4}, 3.14f);
  auto bf16 = orig.astype(fl::dtype::bf16);
  ASSERT_EQ(bf16.type(), fl::dtype::bf16);
  auto back = bf16.astype(fl::dtype::f32);
  ASSERT_EQ(back.type(), fl::dtype::f32);
  std::vector<float> host(4);
  back.hostSync(host.data());
  for (float v : host) {
    // bf16 has ~7-bit mantissa, so ~1% relative error
    ASSERT_NEAR(v, 3.14f, 0.04f);
  }

  // bf16 -> f64 -> bf16
  auto bf16_2 = fl::full({2}, 1.5f, fl::dtype::bf16);
  auto f64 = bf16_2.astype(fl::dtype::f64);
  ASSERT_EQ(f64.type(), fl::dtype::f64);
  auto bf16_3 = f64.astype(fl::dtype::bf16);
  ASSERT_EQ(bf16_3.type(), fl::dtype::bf16);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
