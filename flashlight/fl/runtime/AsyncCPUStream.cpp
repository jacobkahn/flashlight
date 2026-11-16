/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/runtime/AsyncCPUStream.h"

#include <future>
#include <stdexcept>

#include "flashlight/fl/runtime/Event.h"
#include "flashlight/fl/runtime/EventImpl.h"


namespace fl {

AsyncCPUStream::AsyncCPUStream()
    : device_(DeviceManager::getInstance()
                  .getActiveDevice(DeviceType::x64)
                  .impl<X64Device>()) {
  worker_ = std::thread([this]() { workerLoop(); });
}

std::shared_ptr<AsyncCPUStream> AsyncCPUStream::create() {
  auto* rawPtr = new AsyncCPUStream();
  auto stream = std::shared_ptr<AsyncCPUStream>(rawPtr);
  rawPtr->device_.addStream(stream);
  return stream;
}

AsyncCPUStream::~AsyncCPUStream() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_ = true;
  }
  cv_.notify_one();
  if (worker_.joinable()) {
    if (worker_.get_id() == std::this_thread::get_id()) {
      // Destructor is running on the worker thread itself (e.g. last
      // shared_ptr released during lambda destruction in workerLoop).
      // Self-join would deadlock (EDEADLK), so detach instead.
      worker_.detach();
    } else {
      worker_.join();
    }
  }
}

X64Device& AsyncCPUStream::device() {
  return device_;
}

const X64Device& AsyncCPUStream::device() const {
  return device_;
}

void AsyncCPUStream::enqueue(std::function<void()> task) const {
  // A pending error is reported by the next sync(), not by whichever caller
  // happens to enqueue next; tasks enqueued meanwhile are skipped.
  std::lock_guard<std::mutex> lock(mutex_);
  queue_.push_back(std::move(task));
  cv_.notify_one();
}

void AsyncCPUStream::sync() const {
  std::unique_lock<std::mutex> lock(mutex_);
  cvDrain_.wait(lock, [this]() { return queue_.empty() && !workerBusy_; });
  checkException();
}

void AsyncCPUStream::relativeSync(const AsyncCPUStream& waitOn) const {
  // Same-stream: FIFO ordering already guarantees all prior tasks complete
  // before any future task, so no synchronization is needed.
  if (this == &waitOn) {
    return;
  }
  auto event = waitOn.createEvent();
  enqueue([event]() { event.wait(); });
}

void AsyncCPUStream::relativeSync(const Stream& waitOn) const {
  switch (waitOn.type()) {
    case StreamType::AsyncCPU:
      relativeSync(waitOn.impl<AsyncCPUStream>());
      break;
    case StreamType::Synchronous:
      // Synchronous streams execute inline on the calling thread — all work
      // is already complete by the time we reach this point, so there is
      // nothing to wait on.
      break;
    default: {
      // For any other stream type (e.g. CUDA), record a unified Event and
      // enqueue a wait on this stream's worker thread.
      auto event = waitOn.createEvent();
      enqueue([event]() { event.wait(); });
      break;
    }
  }
}

void AsyncCPUStream::workerLoop() const {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this]() { return !queue_.empty() || shutdown_; });
      if (shutdown_ && queue_.empty()) {
        return;
      }
      task = std::move(queue_.front());
      queue_.pop_front();
      workerBusy_ = true;
    }
    // After a failure the stream is in an error state: later tasks may depend
    // on buffers the failed task never wrote, so they are skipped until sync()
    // reports the error. Destroying a skipped task releases what it captured
    // (e.g. an event's promise, which wakes its waiters instead of hanging
    // them). The worker itself keeps running so the stream stays usable.
    bool skip;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      skip = pendingException_ != nullptr;
    }
    if (skip) {
      task = nullptr;
    } else {
      try {
        task();
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        pendingException_ = std::current_exception();
      }
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      workerBusy_ = false;
      if (queue_.empty()) {
        cvDrain_.notify_all();
      }
    }
  }
}

void AsyncCPUStream::checkException() const {
  if (pendingException_) {
    auto ex = pendingException_;
    pendingException_ = nullptr;
    std::rethrow_exception(ex);
  }
}

Event AsyncCPUStream::createEvent() const {
  auto promise = std::make_shared<std::promise<void>>();
  auto future = promise->get_future().share();
  enqueue([promise]() { promise->set_value(); });
  return Event(std::make_shared<detail::CPUEventImpl>(std::move(future)));
}

} // namespace fl
