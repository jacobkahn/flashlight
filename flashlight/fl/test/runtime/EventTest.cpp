/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <thread>
#include <vector>

#include "flashlight/fl/runtime/AsyncCPUStream.h"
#include "flashlight/fl/runtime/DeviceManager.h"
#include "flashlight/fl/runtime/Event.h"
#include "flashlight/fl/runtime/EventImpl.h"
#include "flashlight/fl/tensor/Init.h"

using fl::AsyncCPUStream;
using fl::DeviceManager;
using fl::DeviceType;
using fl::Event;
using fl::StreamType;

TEST(EventTest, DefaultIsReady) {
  Event e;
  ASSERT_TRUE(e.isReady());
  e.wait(); // should not block
}

TEST(EventTest, SynchronousStreamRecord) {
  auto& device = DeviceManager::getInstance().getActiveDevice(DeviceType::x64);
  const auto& streams = device.getStreams();
  for (const auto& s : streams) {
    if (s->type() == StreamType::Synchronous) {
      Event e;
      e.record(*s);
      ASSERT_TRUE(e.isReady());
      e.wait(); // should not block
      return;
    }
  }
  GTEST_SKIP() << "No SynchronousStream available";
}

TEST(EventTest, AsyncCPURecord) {
  auto stream = AsyncCPUStream::create();

  stream->enqueue([]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  });

  Event e;
  e.record(*stream);

  // Event should not be ready immediately (50ms sleep pending)
  ASSERT_FALSE(e.isReady());

  e.wait();
  ASSERT_TRUE(e.isReady());
}

TEST(EventTest, MultipleWaiters) {
  auto stream = AsyncCPUStream::create();

  stream->enqueue([]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
  });

  Event e;
  e.record(*stream);

  // Launch 3 threads that all wait on the same event
  std::atomic<int> waitCount{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < 3; i++) {
    threads.emplace_back([&e, &waitCount]() {
      e.wait();
      waitCount.fetch_add(1);
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  ASSERT_EQ(waitCount.load(), 3);
  stream->sync();
}

TEST(EventTest, CopySemantics) {
  auto stream = AsyncCPUStream::create();

  stream->enqueue([]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  });

  Event e;
  e.record(*stream);

  Event copy = e; // NOLINT(performance-unnecessary-copy-initialization)

  // Both should see the same state
  ASSERT_FALSE(e.isReady());
  ASSERT_FALSE(copy.isReady());

  e.wait();

  // Both should now be ready (same underlying impl)
  ASSERT_TRUE(e.isReady());
  ASSERT_TRUE(copy.isReady());

  stream->sync();
}

TEST(EventTest, ReRecord) {
  auto stream = AsyncCPUStream::create();

  // First recording
  Event e;
  e.record(*stream);

  Event copy = e; // NOLINT(performance-unnecessary-copy-initialization)

  // Re-record original on a new stream whose pending work the test releases
  // explicitly (a sleep would make the assertions below timing-dependent).
  auto stream2 = AsyncCPUStream::create();
  std::promise<void> release;
  auto released = release.get_future().share();
  stream2->enqueue([released]() { released.wait(); });
  e.record(*stream2);

  // Copy still refers to the old recording, which completes on its own
  copy.wait();
  ASSERT_TRUE(copy.isReady());

  // Original now tracks the new stream, which is blocked
  ASSERT_FALSE(e.isReady());

  release.set_value();
  e.wait();
  ASSERT_TRUE(e.isReady());

  stream->sync();
  stream2->sync();
}

TEST(EventTest, DeferredWaitBlocksUntilRecord) {
  auto event = Event(std::make_shared<fl::detail::DeferredEventImpl>());

  // Deferred event should not be ready before record
  ASSERT_FALSE(event.isReady());

  // Launch a thread that will record the event after a delay
  auto stream = AsyncCPUStream::create();
  std::thread recorder([&event, &stream]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    event.record(*stream);
  });

  // wait() should block until the record happens
  event.wait();
  ASSERT_TRUE(event.isReady());

  recorder.join();
  stream->sync();
}

TEST(EventTest, DeferredIsReadyBeforeAndAfter) {
  auto event = Event(std::make_shared<fl::detail::DeferredEventImpl>());
  ASSERT_FALSE(event.isReady());

  auto stream = AsyncCPUStream::create();
  event.record(*stream);

  // After recording on a synchronous-ish stream (no pending work), should be ready
  event.wait();
  ASSERT_TRUE(event.isReady());
  stream->sync();
}

TEST(EventTest, DeferredMultipleWaiters) {
  auto event = Event(std::make_shared<fl::detail::DeferredEventImpl>());

  std::atomic<int> waitCount{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < 3; i++) {
    threads.emplace_back([&event, &waitCount]() {
      event.wait();
      waitCount.fetch_add(1);
    });
  }

  // Let waiters block, then record
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  ASSERT_EQ(waitCount.load(), 0);

  auto stream = AsyncCPUStream::create();
  event.record(*stream);

  for (auto& t : threads) {
    t.join();
  }

  ASSERT_EQ(waitCount.load(), 3);
  stream->sync();
}

TEST(EventTest, DeferredDoubleFulfill) {
  auto impl = std::make_shared<fl::detail::DeferredEventImpl>();
  auto event = Event(impl);

  auto stream = AsyncCPUStream::create();
  event.record(*stream); // first fulfill
  event.wait();
  ASSERT_TRUE(event.isReady());

  // Second record replaces impl — should not crash or deadlock
  auto stream2 = AsyncCPUStream::create();
  event.record(*stream2);
  event.wait();
  ASSERT_TRUE(event.isReady());

  stream->sync();
  stream2->sync();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
