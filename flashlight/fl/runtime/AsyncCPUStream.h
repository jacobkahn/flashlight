/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include "flashlight/fl/runtime/DeviceManager.h"
#include "flashlight/fl/runtime/Stream.h"

namespace fl {

/**
 * An asynchronous CPU stream backed by a dedicated worker thread.
 *
 * Operations are enqueued as tasks (std::function<void()>) and executed
 * sequentially by the worker thread in FIFO order. The calling thread
 * returns immediately after enqueue.
 *
 * Analogous to cudaStream_t but for CPU compute.
 */
class FL_API AsyncCPUStream : public StreamTrait<AsyncCPUStream> {
 public:
  using StreamTrait<AsyncCPUStream>::relativeSync;

  static constexpr StreamType streamType = StreamType::AsyncCPU;

  /**
   * Create an AsyncCPUStream and register it with the active x64 device.
   */
  static std::shared_ptr<AsyncCPUStream> create();

  ~AsyncCPUStream() override;

  // Non-copyable, non-movable (inherited from Stream)

  X64Device& device() override;
  const X64Device& device() const override;

  /**
   * Block the calling thread until all enqueued tasks have completed.
   * If any task threw an exception, rethrows the first stored exception.
   */
  void sync() const override;

  /**
   * Same-type relative sync: enqueues an Event wait on this stream so
   * future tasks wait for waitOn's current tasks to complete.
   */
  void relativeSync(const AsyncCPUStream& waitOn) const override;

  /**
   * Cross-type relative sync dispatch.
   */
  void relativeSync(const Stream& waitOn) const override;

  /**
   * Enqueue a task for execution on the worker thread.
   * Returns immediately. Tasks execute in FIFO order.
   */
  void enqueue(std::function<void()> task) const;

  Event createEvent() const override;

 protected:
  AsyncCPUStream();

 private:
  X64Device& device_;

  // Worker thread state (mutable for const-qualified Stream interface methods)
  mutable std::thread worker_;
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
  mutable std::condition_variable cvDrain_;
  mutable std::deque<std::function<void()>> queue_;
  mutable bool workerBusy_{false};
  mutable bool shutdown_{false};
  mutable std::exception_ptr pendingException_;

  void workerLoop() const;
  void checkException() const;
};

} // namespace fl
