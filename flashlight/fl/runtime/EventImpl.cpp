/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/runtime/EventImpl.h"

#include <chrono>

namespace fl {
namespace detail {

EventImpl::~EventImpl() = default;

void NoOpEventImpl::wait() const {}

bool NoOpEventImpl::isReady() const {
  return true;
}

CPUEventImpl::CPUEventImpl(std::shared_future<void> f)
    : future(std::move(f)) {}

void CPUEventImpl::wait() const {
  future.get();
}

bool CPUEventImpl::isReady() const {
  return future.wait_for(std::chrono::seconds(0)) ==
      std::future_status::ready;
}

void DeferredEventImpl::wait() const {
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait(lock, [this] { return recorded_ != nullptr; });
  lock.unlock();
  recorded_->wait();
}

bool DeferredEventImpl::isReady() const {
  std::lock_guard<std::mutex> lock(mu_);
  return recorded_ && recorded_->isReady();
}

void DeferredEventImpl::fulfill(std::shared_ptr<EventImpl> impl) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    recorded_ = std::move(impl);
  }
  cv_.notify_all();
}

} // namespace detail
} // namespace fl
