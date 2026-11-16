/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include "flashlight/fl/runtime/CUDAStream.h"
#include "flashlight/fl/runtime/CUDAUtils.h"
#include "flashlight/fl/runtime/Event.h"
#include "flashlight/fl/tensor/Init.h"

using fl::CUDAStream;
using fl::Event;

TEST(EventCUDATest, CUDARecord) {
  auto stream = CUDAStream::createManaged();

  Event e;
  e.record(*stream);

  e.wait();
  ASSERT_TRUE(e.isReady());
}

TEST(EventCUDATest, CUDAIsReadyPoll) {
  auto stream = CUDAStream::createManaged();

  // Enqueue a host callback that sleeps, to keep the stream busy
  FL_CUDA_CHECK(cudaLaunchHostFunc(
      stream->handle(),
      [](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      },
      nullptr));

  Event e;
  e.record(*stream);

  // Should not be ready immediately
  ASSERT_FALSE(e.isReady());

  e.wait();
  ASSERT_TRUE(e.isReady());
}

TEST(EventCUDATest, CUDACopySemantics) {
  auto stream = CUDAStream::createManaged();

  // Enqueue some work via host callback
  FL_CUDA_CHECK(cudaLaunchHostFunc(
      stream->handle(),
      [](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      },
      nullptr));

  Event e;
  e.record(*stream);

  Event copy = e; // NOLINT(performance-unnecessary-copy-initialization)

  // Both should track the same underlying CUDA event
  ASSERT_FALSE(e.isReady());
  ASSERT_FALSE(copy.isReady());

  e.wait();

  ASSERT_TRUE(e.isReady());
  ASSERT_TRUE(copy.isReady());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
