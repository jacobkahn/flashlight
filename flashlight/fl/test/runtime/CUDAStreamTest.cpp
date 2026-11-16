/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "flashlight/fl/runtime/AsyncCPUStream.h"
#include "flashlight/fl/runtime/CUDAStream.h"
#include "flashlight/fl/runtime/CUDAUtils.h"
#include "flashlight/fl/runtime/DeviceManager.h"
#include "flashlight/fl/tensor/Init.h"

#include <cuda_runtime.h>

using fl::AsyncCPUStream;
using fl::CUDAStream;
using fl::DeviceManager;
using fl::DeviceType;
using fl::Stream;
using fl::StreamType;

TEST(CUDAStreamTest, createManaged) {
  auto& manager = DeviceManager::getInstance();
  for (auto cudaDevice : manager.getDevicesOfType(DeviceType::CUDA)) {
    cudaDevice->setActive();
    auto cudaStream = CUDAStream::createManaged();

    ASSERT_EQ(cudaStream->streamType, StreamType::CUDA);
    ASSERT_EQ(&cudaStream->device(), cudaDevice);
    ASSERT_EQ(&cudaStream->impl<CUDAStream>(), cudaStream.get());
  }
}

TEST(CUDAStreamTest, createUnmanaged) {
  auto& manager = DeviceManager::getInstance();
  for (auto cudaDevice : manager.getDevicesOfType(DeviceType::CUDA)) {
    cudaDevice->setActive();
    auto cudaStream = CUDAStream::createUnmanaged();

    ASSERT_EQ(cudaStream->streamType, StreamType::CUDA);
    ASSERT_EQ(&cudaStream->device(), cudaDevice);
    ASSERT_EQ(&cudaStream->impl<CUDAStream>(), cudaStream.get());
    // safe to destroy since underlying stream isn't managed.
    FL_CUDA_CHECK(cudaStreamDestroy(cudaStream->handle()));
  }
}


TEST(CUDAStreamTest, unmanagedWrapper) {
  auto& manager = DeviceManager::getInstance();
  int numCudaDevices = 0;
  FL_CUDA_CHECK(cudaGetDeviceCount(&numCudaDevices));

  for (int id = 0; id < numCudaDevices; id++) {
    FL_CUDA_CHECK(cudaSetDevice(id));
    cudaStream_t nativeStream;
    FL_CUDA_CHECK(cudaStreamCreate(&nativeStream));
    auto& cudaDevice = manager.getDevice(DeviceType::CUDA, id);
    auto cudaStream = CUDAStream::wrapUnmanaged(id, nativeStream);

    ASSERT_EQ(cudaStream->streamType, StreamType::CUDA);
    ASSERT_EQ(&cudaStream->device(), &cudaDevice);
    ASSERT_EQ(cudaStream->handle(), cudaStream->handle());
    ASSERT_EQ(&cudaStream->impl<CUDAStream>(), cudaStream.get());
    // safe to destroy since wrapper won't manage underlying stream by default.
    FL_CUDA_CHECK(cudaStreamDestroy(nativeStream));
  }
}

TEST(CUDAStreamTest, unmanagedWrapperDeviceSwitch) {
  auto& manager = DeviceManager::getInstance();
  if (manager.getDeviceCount(DeviceType::CUDA) > 1) {
    const auto& device0 = manager.getDevice(DeviceType::CUDA, 0);
    const auto& device1 = manager.getDevice(DeviceType::CUDA, 1);

    device0.setActive();
    cudaStream_t nativeStream;
    FL_CUDA_CHECK(cudaStreamCreate(&nativeStream));
    device1.setActive();
    // wrapper will switch to device0, create, and then switch back.
    auto cudaStreamWrapped = CUDAStream::wrapUnmanaged(device0.nativeId(), nativeStream);
    auto cudaStreamCreated = CUDAStream::createManaged();

    // nothing blows up -- event was created and used correctly
    ASSERT_NO_THROW(cudaStreamWrapped->relativeSync(*cudaStreamCreated));
    ASSERT_NO_THROW(cudaStreamCreated->relativeSync(*cudaStreamWrapped));

    ASSERT_EQ(&cudaStreamWrapped->device(), &device0);
    ASSERT_EQ(&cudaStreamCreated->device(), &device1);
    ASSERT_EQ(&manager.getActiveDevice(DeviceType::CUDA), &device1);
  }
}

TEST(CUDAStreamTest, relativeSync) {
  auto cs1 = CUDAStream::createManaged();
  auto cs2 = CUDAStream::createManaged();
  std::shared_ptr<Stream> s1 = cs1;
  std::shared_ptr<Stream> s2 = cs2;
  ASSERT_NO_THROW(s1->relativeSync(*s2));
  ASSERT_NO_THROW(s1->relativeSync(*cs2));
  ASSERT_NO_THROW(cs1->relativeSync(*s2));
  ASSERT_NO_THROW(cs1->relativeSync(*cs2));

  std::unordered_set<const Stream*> streams { s1.get(), s2.get() };
  std::shared_ptr<Stream> s3 = CUDAStream::createManaged();
  ASSERT_NO_THROW(s3->relativeSync(streams));
}

TEST(CUDAStreamTest, sync) {
  auto cs1 = CUDAStream::createManaged();
  ASSERT_NO_THROW(cs1->sync());
}

// Cross-device stream synchronization tests

TEST(CUDAStreamTest, RelativeSyncCUDAAfterAsyncCPU) {
  auto asyncStream = AsyncCPUStream::create();
  auto cudaStream = CUDAStream::createManaged();

  std::atomic<bool> cpuWorkDone{false};
  asyncStream->enqueue([&cpuWorkDone]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    cpuWorkDone.store(true);
  });

  // GPU stream waits for CPU stream. A host callback waits on the Event,
  // blocking the GPU stream until CPU work completes.
  cudaStream->relativeSync(*asyncStream);

  // cudaStreamSynchronize blocks until all GPU stream work completes,
  // which includes the event wait planted by relativeSync. If relativeSync
  // were a no-op, sync() would return immediately before cpuWorkDone is set.
  cudaStream->sync();
  ASSERT_TRUE(cpuWorkDone.load());
}

TEST(CUDAStreamTest, RelativeSyncAsyncCPUAfterCUDA) {
  auto cudaStream = CUDAStream::createManaged();
  auto asyncStream = AsyncCPUStream::create();

  // Put work on the GPU stream
  void* d_ptr;
  FL_CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(int)));
  int val = 42;
  FL_CUDA_CHECK(cudaMemcpyAsync(
      d_ptr, &val, sizeof(int), cudaMemcpyHostToDevice, cudaStream->handle()));

  // CPU stream waits for GPU stream — enqueues cudaEventSynchronize on the
  // CPU worker, which blocks the worker until the GPU memcpy completes.
  asyncStream->relativeSync(*cudaStream);

  // After relativeSync, the GPU memcpy is guaranteed complete.
  std::atomic<int> readBack{0};
  asyncStream->enqueue([&readBack, d_ptr]() {
    int hostVal = 0;
    FL_CUDA_CHECK(cudaMemcpy(
        &hostVal, d_ptr, sizeof(int), cudaMemcpyDeviceToHost));
    readBack.store(hostVal);
  });

  asyncStream->sync();
  ASSERT_EQ(readBack.load(), 42);
  FL_CUDA_CHECK(cudaFree(d_ptr));
}

TEST(CUDAStreamTest, RelativeSyncCUDAAfterSynchronous) {
  auto cudaStream = CUDAStream::createManaged();
  auto& device = DeviceManager::getInstance().getActiveDevice(DeviceType::x64);
  const auto& streams = device.getStreams();

  for (const auto& s : streams) {
    if (s->type() == StreamType::Synchronous) {
      // Synchronous work is already complete — relativeSync is a no-op
      ASSERT_NO_THROW(cudaStream->relativeSync(*s));
      break;
    }
  }
}

TEST(CUDAStreamTest, RelativeSyncSynchronousAfterCUDA) {
  auto cudaStream = CUDAStream::createManaged();
  auto& device = DeviceManager::getInstance().getActiveDevice(DeviceType::x64);
  const auto& streams = device.getStreams();

  for (const auto& s : streams) {
    if (s->type() == StreamType::Synchronous) {
      // SynchronousStream::relativeSync(CUDA) calls waitOn.sync() — blocking
      ASSERT_NO_THROW(s->relativeSync(*cudaStream));
      break;
    }
  }
}

TEST(CUDAStreamTest, CrossTypeRelativeSyncViaBasePointer) {
  auto asyncStream = AsyncCPUStream::create();
  auto cudaStream = CUDAStream::createManaged();

  std::shared_ptr<Stream> cpuBase = asyncStream;
  std::shared_ptr<Stream> gpuBase = cudaStream;

  // Both directions via base pointers
  ASSERT_NO_THROW(gpuBase->relativeSync(*cpuBase));
  ASSERT_NO_THROW(cpuBase->relativeSync(*gpuBase));

  // Drain all pending work
  cudaStream->sync();
  asyncStream->sync();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
