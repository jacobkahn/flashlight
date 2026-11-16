/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/runtime/Event.h"

#include "flashlight/fl/runtime/Stream.h"
#include "flashlight/fl/runtime/EventImpl.h"

namespace fl {

Event::Event() : impl_(std::make_shared<detail::NoOpEventImpl>()) {}

Event::Event(std::shared_ptr<detail::EventImpl> impl)
    : impl_(std::move(impl)) {}

void Event::record(const Stream& stream) {
  auto newEvent = stream.createEvent();
  // If this was a deferred event, fulfill it so waiters unblock.
  // The deferred impl stays alive via shared_ptr in any Futures holding it.
  if (auto* deferred =
          dynamic_cast<detail::DeferredEventImpl*>(impl_.get())) {
    deferred->fulfill(newEvent.impl_);
  }
  impl_ = newEvent.impl_;
}

void Event::wait() const {
  impl_->wait();
}

bool Event::isReady() const {
  return impl_->isReady();
}

} // namespace fl
