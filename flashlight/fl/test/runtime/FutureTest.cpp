/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "flashlight/fl/runtime/AsyncCPUStream.h"
#include "flashlight/fl/runtime/Event.h"
#include "flashlight/fl/runtime/EventImpl.h"
#include "flashlight/fl/runtime/Future.h"
#include "flashlight/fl/tensor/Init.h"

using fl::AsyncCPUStream;
using fl::Event;
using fl::Future;

TEST(FutureTest, ReadyFuture) {
  auto f = Future<int>::createReady(42);
  ASSERT_TRUE(f.valid());
  ASSERT_TRUE(f.isReady());
  ASSERT_EQ(f.get(), 42);
}

TEST(FutureTest, ReadyFutureMultipleGet) {
  auto f = Future<float>::createReady(3.14f);
  ASSERT_FLOAT_EQ(f.get(), 3.14f);
  ASSERT_FLOAT_EQ(f.get(), 3.14f); // multiple get() calls should work
}

TEST(FutureTest, EventGatedFuture) {
  auto stream = AsyncCPUStream::create();

  stream->enqueue([]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  });

  Event event;
  event.record(*stream);

  int val = 0;
  // Reader is self-contained: waits on event, then reads.
  Future<int> fut(event, [&val, event]() {
    event.wait();
    return val;
  });

  // Not ready yet (stream work pending)
  ASSERT_FALSE(fut.isReady());

  val = 99;
  ASSERT_EQ(fut.get(), 99);
  ASSERT_TRUE(fut.isReady());

  stream->sync();
}

TEST(FutureTest, DeferredEventFuture) {
  auto event = Event(std::make_shared<fl::detail::DeferredEventImpl>());

  // Reader does its own wait + read.
  Future<int> fut(event, [event]() {
    event.wait();
    return 123;
  });

  // Not ready before record
  ASSERT_FALSE(fut.isReady());

  // Record the event from another thread
  auto stream = AsyncCPUStream::create();
  std::thread recorder([&event, &stream]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    event.record(*stream);
  });

  // get() calls reader which waits on event
  ASSERT_EQ(fut.get(), 123);

  recorder.join();
  stream->sync();
}

TEST(FutureTest, ValidCheck) {
  Future<int> ready = Future<int>::createReady(1);
  ASSERT_TRUE(ready.valid());

  auto event = Event(std::make_shared<fl::detail::DeferredEventImpl>());
  Future<int> pending(event, []() { return 0; });
  ASSERT_TRUE(pending.valid());
}

TEST(FutureTest, CopySharesState) {
  auto event = Event(std::make_shared<fl::detail::DeferredEventImpl>());
  Future<int> original(event, []() { return 77; });

  auto copy = original;

  auto stream = AsyncCPUStream::create();
  event.record(*stream);

  // Both return the same value — shared state, reader runs once
  ASSERT_EQ(original.get(), 77);
  ASSERT_EQ(copy.get(), 77);

  stream->sync();
}

TEST(FutureTest, ConcurrentGet) {
  auto event = Event(std::make_shared<fl::detail::DeferredEventImpl>());
  std::atomic<int> readerCalls{0};

  // Reader is self-contained: waits on event, then reads.
  Future<int> fut(event, [&readerCalls, event]() {
    event.wait();
    readerCalls.fetch_add(1);
    return 42;
  });

  // Share the future across threads
  auto futCopy = fut;

  auto stream = AsyncCPUStream::create();
  stream->enqueue([]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  });
  event.record(*stream);

  // Two threads call get() concurrently
  std::thread t1([&fut]() { ASSERT_EQ(fut.get(), 42); });
  std::thread t2([&futCopy]() { ASSERT_EQ(futCopy.get(), 42); });

  t1.join();
  t2.join();

  // Reader should have been called exactly once (call_once guarantee)
  ASSERT_EQ(readerCalls.load(), 1);

  stream->sync();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
