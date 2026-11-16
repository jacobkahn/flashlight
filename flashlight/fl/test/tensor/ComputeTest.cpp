/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <stdexcept>
#include <thread>

#include "flashlight/fl/runtime/Future.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/JitTensorBase.h"
#include "flashlight/fl/test/TestInit.h"
#include "flashlight/fl/tensor/Tensor.h"

TEST(TensorComputeTest, sync) {
  // Testing whether a value is ready isn't meaningful since any function to
  // inspect its state will implicitly synchronize -- this test simply ensures
  // sync runs
  auto t1 = fl::full({10, 10}, 1.);
  auto t2 = fl::full({10, 10}, 2.);
  auto t3 = t1 + t2;
  fl::sync();

  int deviceId = fl::getDevice();
  auto t4 = t1 + t2 + t3;
  fl::sync(deviceId);
}

TEST(TensorComputeTest, eval) {
  // Testing whether a value is ready isn't meaningful since any function to
  // inspect its state will implicitly synchronize -- this test simply ensures
  // eval runs
  auto t1 = fl::full({10, 10}, 3.);
  auto t2 = fl::full({10, 10}, 4.);
  auto t3 = t1 * t2;
  fl::eval(t3);
}

TEST(TensorComputeTest, ready) {
  auto t1 = fl::full({3, 3}, 1.0);
  auto t2 = t1 + t1;

  if (t2.backendType() == fl::TensorBackendType::Jit) {
    // JIT: unevaluated graph nodes are not ready
    ASSERT_FALSE(fl::ready(t1));
    ASSERT_FALSE(fl::ready(t2));

    // After eval, tensor is ready
    fl::eval(t2);
    ASSERT_TRUE(fl::ready(t2));
  } else {
    // Non-JIT: always ready
    ASSERT_TRUE(fl::ready(t1));
    ASSERT_TRUE(fl::ready(t2));
  }

  ASSERT_TRUE(fl::allClose(t2, fl::full({3, 3}, 2.0)));

  // empty() wraps in ValueNode (concrete allocation) — always ready
  auto e = fl::empty({2, 2}, fl::dtype::f32);
  ASSERT_TRUE(fl::ready(e));
}

TEST(TensorComputeTest, scalarReturnsLazyFuture) {
  auto t = fl::full({1}, 3.14f);

  if (t.backendType() == fl::TensorBackendType::Jit) {
    // scalar() returns Future without evaluating
    auto fut = t.scalar<float>();
    ASSERT_TRUE(fut.valid());

    // eval then get
    fl::eval(t);
    float val = fut.get();
    ASSERT_FLOAT_EQ(val, 3.14f);
  } else {
    // Non-JIT: scalar returns immediately-ready Future
    auto fut = t.scalar<float>();
    ASSERT_TRUE(fut.isReady());
    ASSERT_FLOAT_EQ(fut.get(), 3.14f);
  }
}

TEST(TensorComputeTest, scalarSyncEquivalent) {
  auto t = fl::full({1}, 42.0f);
  float sync = t.scalarSync<float>();
  ASSERT_FLOAT_EQ(sync, 42.0f);
}

TEST(TensorComputeTest, scalarGetBlocksUntilEval) {
  auto t1 = fl::full({1}, 2.0f);
  auto t2 = fl::full({1}, 3.0f);
  auto t3 = t1 + t2;

  if (t3.backendType() == fl::TensorBackendType::Jit) {
    auto fut = t3.scalar<float>();

    // Eval from a background thread
    std::thread evalThread([&t3]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      fl::eval(t3);
    });

    // get() should block until eval completes
    float val = fut.get();
    ASSERT_FLOAT_EQ(val, 5.0f);

    evalThread.join();
  }
}

TEST(TensorComputeTest, evalWithoutCompact) {
  if (!std::is_same_v<fl::DefaultTensorType_t, fl::JitTensorBase>) {
    GTEST_SKIP() << "Test requires JIT backend";
  }

  auto t = fl::full({3, 3}, 1.0f) + fl::full({3, 3}, 2.0f);

  fl::eval(t);
  ASSERT_TRUE(fl::ready(t));

  auto result = t.scalarSync<float>();
  ASSERT_FLOAT_EQ(result, 3.0f);
}

TEST(TensorComputeTest, scalarSyncTriggersEval) {
  if (!std::is_same_v<fl::DefaultTensorType_t, fl::JitTensorBase>) {
    GTEST_SKIP() << "Test requires JIT backend";
  }

  auto t = fl::full({1}, 7.0f) + fl::full({1}, 3.0f);
  ASSERT_FALSE(fl::ready(t));

  // scalarSync triggers eval internally
  ASSERT_FLOAT_EQ(t.scalarSync<float>(), 10.0f);
  ASSERT_TRUE(fl::ready(t));
}

TEST(TensorComputeTest, scalarAsyncRequiresEval) {
  if (!std::is_same_v<fl::DefaultTensorType_t, fl::JitTensorBase>) {
    GTEST_SKIP() << "Test requires JIT backend";
  }

  auto t = fl::full({1}, 7.0f) + fl::full({1}, 3.0f);
  auto fut = t.scalar<float>();

  // With true async D2H, future won't be ready until eval.
  // Currently falls back to sync (createReady), so may be immediately ready.
  fl::eval(t);
  ASSERT_FLOAT_EQ(fut.get(), 10.0f);
}

TEST(TensorComputeTest, scalarMultipleFutures) {
  if (!std::is_same_v<fl::DefaultTensorType_t, fl::JitTensorBase>) {
    GTEST_SKIP() << "Test requires JIT backend";
  }

  auto a = fl::full({1}, 2.0f) + fl::full({1}, 3.0f);
  auto b = fl::full({1}, 10.0f) * fl::full({1}, 4.0f);

  auto futA = a.scalar<float>();
  auto futB = b.scalar<float>();

  fl::eval(a);
  fl::eval(b);

  ASSERT_FLOAT_EQ(futA.get(), 5.0f);
  ASSERT_FLOAT_EQ(futB.get(), 40.0f);
}

TEST(TensorComputeTest, scalarCopyFuture) {
  if (!std::is_same_v<fl::DefaultTensorType_t, fl::JitTensorBase>) {
    GTEST_SKIP() << "Test requires JIT backend";
  }

  auto t = fl::full({1}, 42.0f);
  auto fut = t.scalar<float>();

  // Copies share the same state
  auto futCopy = fut;

  fl::eval(t);

  ASSERT_FLOAT_EQ(fut.get(), 42.0f);
  ASSERT_FLOAT_EQ(futCopy.get(), 42.0f);
}

TEST(TensorComputeTest, doubleScalarBeforeEval) {
  if (!std::is_same_v<fl::DefaultTensorType_t, fl::JitTensorBase>) {
    GTEST_SKIP() << "Test requires JIT backend";
  }

  auto t = fl::full({1}, 5.0f) + fl::full({1}, 3.0f);

  // Two scalar() calls before eval — both must resolve
  auto f1 = t.scalar<float>();
  auto f2 = t.scalar<float>();

  fl::eval(t);

  ASSERT_FLOAT_EQ(f1.get(), 8.0f);
  ASSERT_FLOAT_EQ(f2.get(), 8.0f);
}

TEST(TensorComputeTest, scalarOnReduction) {
  auto t = fl::full({4}, 2.5f);
  auto s = fl::sum(t);

  if (s.backendType() == fl::TensorBackendType::Jit) {
    auto fut = s.scalar<float>();
    fl::eval(s);
    ASSERT_FLOAT_EQ(fut.get(), 10.0f);
  } else {
    ASSERT_FLOAT_EQ(s.scalarSync<float>(), 10.0f);
  }
}

TEST(TensorComputeTest, scalarAfterMutation) {
  if (!std::is_same_v<fl::DefaultTensorType_t, fl::JitTensorBase>) {
    GTEST_SKIP() << "Test requires JIT backend";
  }

  auto t = fl::full({1}, 1.0f) + fl::full({1}, 2.0f);

  // First scalar + eval cycle
  auto f1 = t.scalar<float>();
  fl::eval(t);
  ASSERT_FLOAT_EQ(f1.get(), 3.0f);

  // Mutate the tensor — old event should be invalidated
  t += fl::full({1}, 10.0f);

  // Second scalar should get a fresh deferred event, not the stale one.
  // With true async D2H, f2 won't be ready until eval.
  // Currently falls back to sync (createReady), so may be immediately ready.
  auto f2 = t.scalar<float>();
  fl::eval(t);
  ASSERT_FLOAT_EQ(f2.get(), 13.0f);
}

TEST(TensorComputeTest, hostReturnsLazyFuture) {
  auto t = fl::full({4}, 2.5f);

  if (t.backendType() == fl::TensorBackendType::Jit) {
    // host() returns Future without evaluating
    auto fut = t.host<float>();
    ASSERT_TRUE(fut.valid());

    // eval then get
    fl::eval(t);
    float* ptr = fut.get();
    ASSERT_NE(ptr, nullptr);
    for (int i = 0; i < 4; ++i) {
      ASSERT_FLOAT_EQ(ptr[i], 2.5f);
    }
    delete[] reinterpret_cast<char*>(ptr);
  } else {
    // Non-JIT: host returns immediately-ready Future
    auto fut = t.host<float>();
    ASSERT_TRUE(fut.isReady());
    float* ptr = fut.get();
    ASSERT_NE(ptr, nullptr);
    for (int i = 0; i < 4; ++i) {
      ASSERT_FLOAT_EQ(ptr[i], 2.5f);
    }
    delete[] reinterpret_cast<char*>(ptr);
  }
}

TEST(TensorComputeTest, hostSyncEquivalent) {
  auto t = fl::full({4}, 3.0f);
  float* ptr = t.hostSync<float>();
  ASSERT_NE(ptr, nullptr);
  for (int i = 0; i < 4; ++i) {
    ASSERT_FLOAT_EQ(ptr[i], 3.0f);
  }
  delete[] reinterpret_cast<char*>(ptr);
}

TEST(TensorComputeTest, hostAfterMutation) {
  if (!std::is_same_v<fl::DefaultTensorType_t, fl::JitTensorBase>) {
    GTEST_SKIP() << "Test requires JIT backend";
  }

  auto t = fl::full({2}, 1.0f) + fl::full({2}, 2.0f);

  // First host + eval cycle
  auto f1 = t.host<float>();
  fl::eval(t);
  float* p1 = f1.get();
  ASSERT_FLOAT_EQ(p1[0], 3.0f);
  ASSERT_FLOAT_EQ(p1[1], 3.0f);
  delete[] reinterpret_cast<char*>(p1);

  // Mutate the tensor
  t += fl::full({2}, 10.0f);

  // Second host should get a fresh deferred event.
  // Currently falls back to sync (createReady), so may be immediately ready.
  auto f2 = t.host<float>();
  fl::eval(t);
  float* p2 = f2.get();
  ASSERT_FLOAT_EQ(p2[0], 13.0f);
  ASSERT_FLOAT_EQ(p2[1], 13.0f);
  delete[] reinterpret_cast<char*>(p2);
}

TEST(TensorComputeTest, toHostVectorReturnsLazyFuture) {
  auto t = fl::full({4}, 2.5f);

  if (t.backendType() == fl::TensorBackendType::Jit) {
    auto fut = t.toHostVector<float>();
    ASSERT_TRUE(fut.valid());

    fl::eval(t);
    auto vec = fut.get();
    ASSERT_EQ(vec.size(), 4);
    for (int i = 0; i < 4; ++i) {
      ASSERT_FLOAT_EQ(vec[i], 2.5f);
    }
  } else {
    auto fut = t.toHostVector<float>();
    ASSERT_TRUE(fut.isReady());
    auto vec = fut.get();
    ASSERT_EQ(vec.size(), 4);
    for (int i = 0; i < 4; ++i) {
      ASSERT_FLOAT_EQ(vec[i], 2.5f);
    }
  }
}

TEST(TensorComputeTest, toHostVectorSyncEquivalent) {
  auto t = fl::full({4}, 3.0f);
  auto vec = t.toHostVectorSync<float>();
  ASSERT_EQ(vec.size(), 4);
  for (int i = 0; i < 4; ++i) {
    ASSERT_FLOAT_EQ(vec[i], 3.0f);
  }
}

TEST(TensorComputeTest, toHostVectorAfterMutation) {
  if (!std::is_same_v<fl::DefaultTensorType_t, fl::JitTensorBase>) {
    GTEST_SKIP() << "Test requires JIT backend";
  }

  auto t = fl::full({2}, 1.0f) + fl::full({2}, 2.0f);

  auto f1 = t.toHostVector<float>();
  fl::eval(t);
  auto v1 = f1.get();
  ASSERT_FLOAT_EQ(v1[0], 3.0f);
  ASSERT_FLOAT_EQ(v1[1], 3.0f);

  // Mutate — second toHostVector should get a fresh deferred event
  t += fl::full({2}, 10.0f);

  // Currently falls back to sync (createReady), so may be immediately ready.
  auto f2 = t.toHostVector<float>();
  fl::eval(t);
  auto v2 = f2.get();
  ASSERT_FLOAT_EQ(v2[0], 13.0f);
  ASSERT_FLOAT_EQ(v2[1], 13.0f);
}

TEST(TensorComputeTest, toHostVectorEmpty) {
  auto fut = fl::Tensor().toHostVector<float>();
  ASSERT_TRUE(fut.isReady());
  ASSERT_EQ(fut.get().size(), 0);

  ASSERT_EQ(fl::Tensor().toHostVectorSync<float>().size(), 0);
}

TEST(TensorComputeTest, scalarWrongTypeThrows) {
  auto t = fl::full({1}, 1.0f); // float tensor
  ASSERT_THROW(t.scalar<int>(), std::invalid_argument);
  ASSERT_THROW(t.scalarSync<int>(), std::invalid_argument);
}

TEST(TensorComputeTest, deviceReturnsLazyPointer) {
  if (!std::is_same_v<fl::DefaultTensorType_t, fl::JitTensorBase>) {
    GTEST_SKIP() << "Test requires JIT backend";
  }

  auto t = fl::full({4}, 2.5f) + fl::full({4}, 1.5f);

  // device() should return a non-null pointer without triggering eval
  float* ptr = t.device<float>();
  ASSERT_NE(ptr, nullptr);

  // After eval + sync, the data should be valid
  fl::eval(t);
  t.stream().sync();

  // deviceSync gives the same pointer (reconciled or same buffer)
  float* syncPtr = t.deviceSync<float>();
  ASSERT_NE(syncPtr, nullptr);

  // Verify data through host read (device pointers may be GPU memory)
  auto hostVec = t.toHostVectorSync<float>();
  for (int i = 0; i < 4; ++i) {
    ASSERT_FLOAT_EQ(hostVec[i], 4.0f);
  }
  t.unlock();
}

TEST(TensorComputeTest, deviceSyncEquivalent) {
  auto t = fl::full({4}, 3.0f);
  float* ptr = t.deviceSync<float>();
  ASSERT_NE(ptr, nullptr);

  // Verify data is correct (deviceSync is blocking)
  auto hostVec = t.toHostVectorSync<float>();
  for (int i = 0; i < 4; ++i) {
    ASSERT_FLOAT_EQ(hostVec[i], 3.0f);
  }
  t.unlock();
}

TEST(TensorComputeTest, devicePointerStableAfterEval) {
  if (!std::is_same_v<fl::DefaultTensorType_t, fl::JitTensorBase>) {
    GTEST_SKIP() << "Test requires JIT backend";
  }

  auto t = fl::full({4}, 5.0f) + fl::full({4}, 3.0f);

  // Get lazy pointer
  float* lazyPtr = t.device<float>();
  ASSERT_NE(lazyPtr, nullptr);

  // After eval + sync, the pointer should still be valid
  fl::eval(t);
  t.stream().sync();

  // Read data through host copy (device pointer may be on GPU)
  auto hostVec = t.toHostVectorSync<float>();
  for (int i = 0; i < 4; ++i) {
    ASSERT_FLOAT_EQ(hostVec[i], 8.0f);
  }
  t.unlock();
}

TEST(TensorComputeTest, deviceAfterMutation) {
  if (!std::is_same_v<fl::DefaultTensorType_t, fl::JitTensorBase>) {
    GTEST_SKIP() << "Test requires JIT backend";
  }

  auto t = fl::full({2}, 1.0f) + fl::full({2}, 2.0f);

  // Get pointer, eval, verify via host copy (pointer may be on GPU)
  float* p1 = t.deviceSync<float>();
  ASSERT_NE(p1, nullptr);
  {
    auto hv = t.toHostVectorSync<float>();
    ASSERT_FLOAT_EQ(hv[0], 3.0f);
  }
  t.unlock();

  // Mutate — old pointer is invalidated
  t += fl::full({2}, 10.0f);

  // New deviceSync gives correct data (verify via host copy)
  float* p2 = t.deviceSync<float>();
  ASSERT_NE(p2, nullptr);
  {
    auto hv = t.toHostVectorSync<float>();
    ASSERT_FLOAT_EQ(hv[0], 13.0f);
    ASSERT_FLOAT_EQ(hv[1], 13.0f);
  }
  t.unlock();
}

TEST(TensorComputeTest, deviceOnEmpty) {
  fl::Tensor empty;
  ASSERT_EQ(empty.device<float>(), nullptr);
  ASSERT_EQ(empty.deviceSync<float>(), nullptr);
}

TEST(TensorComputeTest, isContiguousWithoutEval) {
  if (!std::is_same_v<fl::DefaultTensorType_t, fl::JitTensorBase>) {
    GTEST_SKIP() << "Test requires JIT backend";
  }

  // Unevaluated binary op — should report contiguous without materializing.
  auto t = fl::full({3, 4}, 1.0f) + fl::full({3, 4}, 2.0f);
  ASSERT_FALSE(fl::ready(t)); // not yet evaluated
  ASSERT_TRUE(t.isContiguous());
  ASSERT_FALSE(fl::ready(t)); // still not evaluated — no materialization

  // After explicit eval, should still be contiguous.
  fl::eval(t);
  ASSERT_TRUE(fl::ready(t));
  ASSERT_TRUE(t.isContiguous());
}

TEST(TensorComputeTest, stridesWithoutEval) {
  if (!std::is_same_v<fl::DefaultTensorType_t, fl::JitTensorBase>) {
    GTEST_SKIP() << "Test requires JIT backend";
  }

  // Unevaluated chain — strides should be row-major without materializing.
  auto t = fl::full({2, 3, 4}, 1.0f) * fl::full({2, 3, 4}, 2.0f);
  ASSERT_FALSE(fl::ready(t));
  fl::Shape strides = t.strides();
  ASSERT_FALSE(fl::ready(t)); // strides() must not trigger eval
  // Row-major strides for (2, 3, 4): (12, 4, 1)
  ASSERT_EQ(strides, (fl::Shape{12, 4, 1}));

  // Scalar tensor — strides should be (1).
  auto s = fl::full({}, 5.0f) + fl::full({}, 3.0f);
  ASSERT_FALSE(fl::ready(s));
  ASSERT_EQ(s.strides(), (fl::Shape{1}));
  ASSERT_FALSE(fl::ready(s));

  // 1D tensor — strides should be (1).
  auto v = fl::full({5}, 1.0f) + fl::full({5}, 2.0f);
  ASSERT_FALSE(fl::ready(v));
  ASSERT_EQ(v.strides(), (fl::Shape{1}));
  ASSERT_FALSE(fl::ready(v));
}

TEST(TensorComputeTest, contiguousStridesUtility) {
  // Test the contiguousStrides() free function directly.
  ASSERT_EQ(fl::contiguousStrides(fl::Shape({4, 3, 2})), (fl::Shape{6, 2, 1}));
  ASSERT_EQ(fl::contiguousStrides(fl::Shape({5})), (fl::Shape{1}));
  ASSERT_EQ(fl::contiguousStrides(fl::Shape({})), (fl::Shape{1})); // scalar
  ASSERT_EQ(
      fl::contiguousStrides(fl::Shape({2, 3})), (fl::Shape{3, 1}));
}

TEST(TensorComputeTest, isContiguousAfterReshapeTranspose) {
  if (!std::is_same_v<fl::DefaultTensorType_t, fl::JitTensorBase>) {
    GTEST_SKIP() << "Test requires JIT backend";
  }

  // Reshape and transpose in the backends always produce contiguous output.
  auto t = fl::reshape(
      fl::full({6}, 1.0f) + fl::full({6}, 2.0f), {2, 3});
  ASSERT_FALSE(fl::ready(t));
  ASSERT_TRUE(t.isContiguous());
  ASSERT_EQ(t.strides(), (fl::Shape{3, 1}));
  ASSERT_FALSE(fl::ready(t)); // no eval triggered

  auto t2 = fl::transpose(fl::full({2, 3}, 1.0f), {1, 0});
  ASSERT_FALSE(fl::ready(t2));
  ASSERT_TRUE(t2.isContiguous());
  ASSERT_EQ(t2.strides(), (fl::Shape{2, 1}));
  ASSERT_FALSE(fl::ready(t2)); // no eval triggered
}

TEST(TensorComputeTest, isContiguousValueNode) {
  // Evaluated tensors (ValueNodes) should delegate to the underlying tensor.
  auto t = fl::full({3, 4}, 1.0f);
  fl::eval(t);
  t.stream().sync();
  ASSERT_TRUE(t.isContiguous());
  ASSERT_EQ(t.strides(), (fl::Shape{4, 1}));
}

TEST(TensorComputeTest, lockUnlockNoOp) {
  // isLocked() returns false; unlock() is a no-op. Buffer lifetime is
  // managed by the Tensor object itself.
  auto t = fl::full({4}, 1.0f) + fl::full({4}, 2.0f);
  ASSERT_FALSE(t.isLocked());

  float* ptr = t.deviceSync<float>();
  ASSERT_NE(ptr, nullptr);
  ASSERT_FALSE(t.isLocked());

  t.unlock();
  ASSERT_FALSE(t.isLocked());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  flTestInit();
  return RUN_ALL_TESTS();
}
