/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>

namespace fl {
namespace detail {

/**
 * Virtual base for type-erased Event implementations.
 */
struct EventImpl {
  virtual ~EventImpl();
  virtual void wait() const = 0;
  virtual bool isReady() const = 0;
};

/** Always-ready impl for synchronous streams and default-constructed Events. */
struct NoOpEventImpl : EventImpl {
  void wait() const override;
  bool isReady() const override;
};

/** CPU impl backed by a shared_future signaled from an AsyncCPUStream worker. */
struct CPUEventImpl : EventImpl {
  std::shared_future<void> future;
  explicit CPUEventImpl(std::shared_future<void> f);
  void wait() const override;
  bool isReady() const override;
};

/**
 * A deferred event bridges the temporal gap between Future creation (before
 * eval) and stream-side recording (during eval). wait() blocks the caller
 * until fulfill() supplies a concrete, stream-recorded EventImpl.
 */
struct DeferredEventImpl : EventImpl {
  void wait() const override;
  bool isReady() const override;

  /// Supply a concrete, stream-recorded EventImpl. Unblocks all waiters.
  void fulfill(std::shared_ptr<EventImpl> impl);

 private:
  mutable std::mutex mu_;
  mutable std::condition_variable cv_;
  std::shared_ptr<EventImpl> recorded_;
};

} // namespace detail
} // namespace fl
