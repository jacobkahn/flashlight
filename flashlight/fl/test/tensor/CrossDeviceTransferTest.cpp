/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/runtime/DeviceManager.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/JitTensorBase.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/test/TestInit.h"

using namespace fl;

class CrossDeviceTransferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    hostDev_ =
        &DeviceManager::getInstance().getActiveDevice(DeviceType::x64);
    cudaDev_ =
        &DeviceManager::getInstance().getActiveDevice(DeviceType::CUDA);
  }

  Device* hostDev_;
  Device* cudaDev_;
};

// ========================= Device-to-Host =========================

TEST_F(CrossDeviceTransferTest, D2H_EvaluatedSource) {
  // Transfer an already-evaluated CUDA tensor to host.
  auto t = fl::full({4}, 42.0f);
  fl::eval(t);
  t.stream().sync();

  auto [event, hostTensor] = toJitTensorBase(t).to(*hostDev_);
  fl::eval(hostTensor);
  event.wait();

  auto hostVec = hostTensor.toHostVectorSync<float>();
  ASSERT_EQ(hostVec.size(), 4u);
  for (size_t i = 0; i < hostVec.size(); ++i) {
    ASSERT_FLOAT_EQ(hostVec[i], 42.0f);
  }
}

TEST_F(CrossDeviceTransferTest, D2H_LazyGraph) {
  // Transfer a lazy (unevaluated) computation to host. A single eval
  // should evaluate both the computation and the transfer.
  auto t = fl::full({4}, 7.0f) + fl::full({4}, 3.0f);
  ASSERT_FALSE(fl::ready(t));

  auto [event, hostTensor] = toJitTensorBase(t).to(*hostDev_);

  fl::eval(hostTensor);
  event.wait();

  auto hostVec = hostTensor.toHostVectorSync<float>();
  ASSERT_EQ(hostVec.size(), 4u);
  for (size_t i = 0; i < hostVec.size(); ++i) {
    ASSERT_FLOAT_EQ(hostVec[i], 10.0f)
        << "Transfer data wrong at index " << i;
  }
}

// ========================= Host-to-Device =========================

TEST_F(CrossDeviceTransferTest, H2D_EvaluatedSource) {
  // Create a CPU tensor directly, then transfer to CUDA.
  auto cpuTensor = fl::full({4}, 42.0f, dtype::f32, *hostDev_);
  fl::eval(cpuTensor);

  auto [h2dEvent, gpuResult] = toJitTensorBase(cpuTensor).to(*cudaDev_);
  fl::eval(gpuResult);
  h2dEvent.wait();

  auto hostVec = gpuResult.toHostVectorSync<float>();
  ASSERT_EQ(hostVec.size(), 4u);
  for (size_t i = 0; i < hostVec.size(); ++i) {
    ASSERT_FLOAT_EQ(hostVec[i], 42.0f);
  }
}

TEST_F(CrossDeviceTransferTest, H2D_LazyGraph) {
  // Create a lazy CPU computation, then transfer to CUDA. A single eval
  // should evaluate both the CPU computation and the transfer.
  auto cpuA = fl::full({4}, 7.0f, dtype::f32, *hostDev_);
  auto cpuB = fl::full({4}, 3.0f, dtype::f32, *hostDev_);
  auto cpuSum = cpuA + cpuB;
  ASSERT_FALSE(fl::ready(cpuSum));

  auto [h2dEvent, gpuResult] = toJitTensorBase(cpuSum).to(*cudaDev_);

  fl::eval(gpuResult);
  h2dEvent.wait();

  auto hostVec = gpuResult.toHostVectorSync<float>();
  ASSERT_EQ(hostVec.size(), 4u);
  for (size_t i = 0; i < hostVec.size(); ++i) {
    ASSERT_FLOAT_EQ(hostVec[i], 10.0f)
        << "H2D data wrong at index " << i;
  }
}

// ========================= Async Stress =========================

TEST_F(CrossDeviceTransferTest, AsyncRoundTrip_LargeTensor) {
  // Large tensor stresses the async pipeline: compute -> D2H -> H2D with
  // enough data that timing-dependent bugs (race conditions, premature
  // reads) are more likely to surface.
  const Dim N = 1024 * 1024;
  auto gpuOrig = fl::arange({N}, 0, dtype::f32);

  // Build lazy chain: GPU -> D2H -> H2D (no eval yet)
  auto [d2hEvent, cpuTensor] = toJitTensorBase(gpuOrig).to(*hostDev_);
  auto [h2dEvent, gpuResult] = toJitTensorBase(cpuTensor).to(*cudaDev_);

  fl::eval(gpuResult);
  h2dEvent.wait();

  auto resultVec = gpuResult.toHostVectorSync<float>();
  ASSERT_EQ(static_cast<Dim>(resultVec.size()), N);
  for (Dim i = 0; i < N; ++i) {
    ASSERT_FLOAT_EQ(resultVec[i], static_cast<float>(i))
        << "Mismatch at index " << i;
  }
}

TEST_F(CrossDeviceTransferTest, MultiTensorBatch) {
  // Three independent tensors each do GPU -> CPU -> GPU. Evaluating all
  // three verifies that batch evaluation with multiple TransferNode
  // chains works correctly.
  auto g1 = fl::full({100}, 1.0f);
  auto g2 = fl::full({200}, 2.0f);
  auto g3 = fl::full({300}, 3.0f);

  // D2H (lazy)
  auto [d2h1, c1] = toJitTensorBase(g1).to(*hostDev_);
  auto [d2h2, c2] = toJitTensorBase(g2).to(*hostDev_);
  auto [d2h3, c3] = toJitTensorBase(g3).to(*hostDev_);

  // H2D (lazy)
  auto [h2d1, r1] = toJitTensorBase(c1).to(*cudaDev_);
  auto [h2d2, r2] = toJitTensorBase(c2).to(*cudaDev_);
  auto [h2d3, r3] = toJitTensorBase(c3).to(*cudaDev_);

  fl::eval(r1);
  fl::eval(r2);
  fl::eval(r3);
  h2d1.wait();
  h2d2.wait();
  h2d3.wait();

  auto v1 = r1.toHostVectorSync<float>();
  auto v2 = r2.toHostVectorSync<float>();
  auto v3 = r3.toHostVectorSync<float>();

  ASSERT_EQ(v1.size(), 100u);
  ASSERT_EQ(v2.size(), 200u);
  ASSERT_EQ(v3.size(), 300u);

  for (const auto& x : v1) {
    ASSERT_FLOAT_EQ(x, 1.0f);
  }
  for (const auto& x : v2) {
    ASSERT_FLOAT_EQ(x, 2.0f);
  }
  for (const auto& x : v3) {
    ASSERT_FLOAT_EQ(x, 3.0f);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  flTestInit();
  return RUN_ALL_TESTS();
}
