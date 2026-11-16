/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "flashlight/fl/runtime/AsyncCPUStream.h"
#include "flashlight/fl/runtime/DeviceManager.h"
#include "flashlight/fl/runtime/Event.h"
#include "flashlight/fl/tensor/Init.h"

using fl::AsyncCPUStream;
using fl::DeviceManager;
using fl::DeviceType;
using fl::Event;
using fl::Stream;
using fl::StreamType;

TEST(AsyncCPUStreamTest, Create) {
  auto stream = AsyncCPUStream::create();
  ASSERT_EQ(stream->streamType, StreamType::AsyncCPU);
  auto& x64Device =
      DeviceManager::getInstance().getActiveDevice(DeviceType::x64);
  ASSERT_EQ(&stream->device(), &x64Device);
  ASSERT_EQ(&stream->impl<AsyncCPUStream>(), stream.get());
}

TEST(AsyncCPUStreamTest, EnqueueAndSync) {
  auto stream = AsyncCPUStream::create();
  std::atomic<int> value{0};
  stream->enqueue([&value]() { value.store(42); });
  stream->sync();
  ASSERT_EQ(value.load(), 42);
}

TEST(AsyncCPUStreamTest, MultipleTasks) {
  auto stream = AsyncCPUStream::create();
  std::atomic<int> counter{0};
  for (int i = 0; i < 100; i++) {
    stream->enqueue([&counter]() { counter.fetch_add(1); });
  }
  stream->sync();
  ASSERT_EQ(counter.load(), 100);
}

TEST(AsyncCPUStreamTest, TaskExecutionOrder) {
  auto stream = AsyncCPUStream::create();
  std::mutex mtx;
  std::vector<int> order;
  for (int i = 0; i < 10; i++) {
    stream->enqueue([&mtx, &order, i]() {
      std::lock_guard<std::mutex> lock(mtx);
      order.push_back(i);
    });
  }
  stream->sync();
  std::vector<int> expected{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  ASSERT_EQ(order, expected);
}

TEST(AsyncCPUStreamTest, SyncBlocksUntilComplete) {
  auto stream = AsyncCPUStream::create();
  std::atomic<bool> done{false};
  stream->enqueue([&done]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    done.store(true);
  });
  // Task should not be complete immediately
  // (race: it could be, but with 50ms sleep it's very unlikely)
  ASSERT_FALSE(done.load());
  stream->sync();
  ASSERT_TRUE(done.load());
}

TEST(AsyncCPUStreamTest, EnqueueAfterSync) {
  auto stream = AsyncCPUStream::create();
  std::atomic<int> counter{0};

  stream->enqueue([&counter]() { counter.fetch_add(1); });
  stream->sync();
  ASSERT_EQ(counter.load(), 1);

  stream->enqueue([&counter]() { counter.fetch_add(1); });
  stream->sync();
  ASSERT_EQ(counter.load(), 2);
}

TEST(AsyncCPUStreamTest, ErrorPropagation) {
  auto stream = AsyncCPUStream::create();
  stream->enqueue([]() {
    throw std::runtime_error("test error");
  });
  EXPECT_THROW(
      {
        try {
          stream->sync();
        } catch (const std::runtime_error& e) {
          EXPECT_STREQ(e.what(), "test error");
          throw;
        }
      },
      std::runtime_error);
}

TEST(AsyncCPUStreamTest, ErrorStopsSubsequentTasks) {
  auto stream = AsyncCPUStream::create();
  std::atomic<bool> taskAfterErrorRan{false};

  stream->enqueue([]() {
    throw std::runtime_error("fail");
  });
  stream->enqueue([&taskAfterErrorRan]() {
    taskAfterErrorRan.store(true);
  });

  EXPECT_THROW(stream->sync(), std::runtime_error);
  ASSERT_FALSE(taskAfterErrorRan.load());
}

TEST(AsyncCPUStreamTest, StreamUsableAfterError) {
  auto stream = AsyncCPUStream::create();
  stream->enqueue([]() { throw std::runtime_error("fail"); });
  EXPECT_THROW(stream->sync(), std::runtime_error);

  // Once the error has been reported, the stream runs new work again.
  std::atomic<bool> ran{false};
  stream->enqueue([&ran]() { ran.store(true); });
  ASSERT_NO_THROW(stream->sync());
  ASSERT_TRUE(ran.load());
}

TEST(AsyncCPUStreamTest, EventRecordAndWait) {
  auto stream = AsyncCPUStream::create();
  std::atomic<bool> flag{false};

  stream->enqueue([&flag]() { flag.store(true); });

  auto event = stream->createEvent();

  event.wait();
  ASSERT_TRUE(flag.load());
}

TEST(AsyncCPUStreamTest, EventIsReady) {
  auto stream = AsyncCPUStream::create();

  stream->enqueue([]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  });

  auto event = stream->createEvent();

  // Event should not be ready immediately (50ms sleep pending)
  ASSERT_FALSE(event.isReady());

  event.wait();
  ASSERT_TRUE(event.isReady());
}

TEST(AsyncCPUStreamTest, MultipleWaiters) {
  auto stream = AsyncCPUStream::create();

  stream->enqueue([]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
  });

  auto event = stream->createEvent();

  // Launch 3 threads that all wait on the same event
  std::atomic<int> waitCount{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < 3; i++) {
    threads.emplace_back([&event, &waitCount]() {
      event.wait();
      waitCount.fetch_add(1);
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  ASSERT_EQ(waitCount.load(), 3);
  stream->sync();
}

TEST(AsyncCPUStreamTest, SyncDrainsQueue) {
  auto stream = AsyncCPUStream::create();
  std::atomic<int> counter{0};
  for (int i = 0; i < 10; i++) {
    stream->enqueue([&counter]() {
      counter.fetch_add(1);
    });
  }
  stream->sync();
  ASSERT_EQ(counter.load(), 10);
}

TEST(AsyncCPUStreamTest, RelativeSyncTwoAsyncStreams) {
  auto stream1 = AsyncCPUStream::create();
  auto stream2 = AsyncCPUStream::create();
  std::atomic<int> result{0};

  // Stream1: sleep then set result
  stream1->enqueue([&result]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    result.store(42);
  });

  // Stream2 must wait for stream1's work
  stream2->relativeSync(*stream1);

  // Stream2: read result (should be 42 since we waited for stream1)
  std::atomic<int> readValue{0};
  stream2->enqueue([&result, &readValue]() {
    readValue.store(result.load());
  });

  stream2->sync();
  ASSERT_EQ(readValue.load(), 42);
}

TEST(AsyncCPUStreamTest, RelativeSyncViaBasePointer) {
  auto s1 = AsyncCPUStream::create();
  auto s2 = AsyncCPUStream::create();

  std::shared_ptr<Stream> base1 = s1;
  std::shared_ptr<Stream> base2 = s2;

  // Should dispatch correctly through the base class
  ASSERT_NO_THROW(base2->relativeSync(*base1));
  ASSERT_NO_THROW(s2->relativeSync(*base1));
  ASSERT_NO_THROW(base2->relativeSync(*s1));

  // Drain all pending relativeSync tasks before test ends
  s1->sync();
  s2->sync();
}

TEST(AsyncCPUStreamTest, RelativeSyncMultipleStreams) {
  auto s1 = AsyncCPUStream::create();
  auto s2 = AsyncCPUStream::create();
  auto s3 = AsyncCPUStream::create();

  std::atomic<int> v1{0};
  std::atomic<int> v2{0};

  s1->enqueue([&v1]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    v1.store(1);
  });

  s2->enqueue([&v2]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    v2.store(2);
  });

  // s3 waits on both s1 and s2
  std::unordered_set<const Stream*> waitOns{s1.get(), s2.get()};
  s3->relativeSync(waitOns);

  std::atomic<int> sum{0};
  s3->enqueue([&v1, &v2, &sum]() {
    sum.store(v1.load() + v2.load());
  });

  s3->sync();
  ASSERT_EQ(sum.load(), 3);
}

// Cross-type relativeSync tests.
// These iterate device.getStreams() and should run last to avoid
// CUDA driver interaction issues with subsequent thread-heavy tests.

TEST(AsyncCPUStreamTest, RelativeSyncAsyncOnSynchronous) {
  auto asyncStream = AsyncCPUStream::create();

  // Get a SynchronousStream-derived stream if available
  // The SynchronousStream is always "complete" so this should be fast
  auto& device = DeviceManager::getInstance().getActiveDevice(DeviceType::x64);
  const auto& streams = device.getStreams();

  for (const auto& s : streams) {
    if (s->type() == StreamType::Synchronous) {
      ASSERT_NO_THROW(asyncStream->relativeSync(*s));
      break;
    }
  }

  asyncStream->sync();
}

TEST(AsyncCPUStreamTest, RelativeSyncSynchronousOnAsync) {
  auto asyncStream = AsyncCPUStream::create();
  std::atomic<int> result{0};

  asyncStream->enqueue([&result]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    result.store(99);
  });

  auto& device = DeviceManager::getInstance().getActiveDevice(DeviceType::x64);
  const auto& streams = device.getStreams();

  for (const auto& s : streams) {
    if (s->type() == StreamType::Synchronous) {
      // SynchronousStream::relativeSync(AsyncCPU) should block until async
      // work completes
      s->relativeSync(*asyncStream);
      ASSERT_EQ(result.load(), 99);
      break;
    }
  }
}

#if FL_BACKEND_CUDA
#include <cuda_runtime.h>
#include "flashlight/fl/runtime/CUDAStream.h"
#include "flashlight/fl/runtime/CUDAUtils.h"

TEST(AsyncCPUStreamTest, RelativeSyncAsyncCPUAfterCUDA) {
  auto cudaStream = fl::CUDAStream::createManaged();
  auto asyncStream = AsyncCPUStream::create();

  std::atomic<bool> cpuWorkDone{false};

  // CPU stream waits for GPU stream
  asyncStream->relativeSync(*cudaStream);

  // Follow-up task runs only after the relativeSync task completes
  asyncStream->enqueue([&cpuWorkDone]() {
    cpuWorkDone.store(true);
  });

  asyncStream->sync();
  ASSERT_TRUE(cpuWorkDone.load());
}

TEST(AsyncCPUStreamTest, RelativeSyncCUDAAfterAsyncCPU) {
  auto asyncStream = AsyncCPUStream::create();
  auto cudaStream = fl::CUDAStream::createManaged();

  std::atomic<bool> cpuWorkDone{false};
  asyncStream->enqueue([&cpuWorkDone]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    cpuWorkDone.store(true);
  });

  // GPU stream waits for CPU stream
  cudaStream->relativeSync(*asyncStream);

  // sync() blocks until the bridge thread signals the CUDA event
  cudaStream->sync();
  ASSERT_TRUE(cpuWorkDone.load());
}
#endif

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
