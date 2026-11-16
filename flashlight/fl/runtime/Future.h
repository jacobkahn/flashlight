/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include "flashlight/fl/runtime/Event.h"

namespace fl {

/**
 * An event-gated async result.
 *
 * Created by scalar<T>() / host<T>(). The reader handles evaluation,
 * synchronization, and data read internally.
 *
 * Typical lifecycle:
 *   auto f = tensor.scalar<float>();  // builds transfer chain, no eval
 *   float v = f.get();               // eval + sync + pure data read
 *
 * Calling fl::eval(tensor) before get() is optional — if the source tensor
 * is already evaluated, the internal eval is a fast no-op.
 *
 * Movable and copyable — copies share the same internal state.
 * Users who want async callbacks should manage their own threads:
 *   myPool.submit([f]() mutable { logger.log(f.get()); });
 */
template <typename T>
class Future {
  struct State {
    Event event;
    std::function<T()> reader;
    std::optional<T> value;
    std::once_flag once;
  };

 public:
  /// Wrap an already-computed value (no Event wait needed).
  static Future createReady(T value) {
    Future f;
    f.state_->value = std::move(value);
    return f;
  }

  /// Async: event gates synchronization, reader does pure data read.
  Future(Event event, std::function<T()> reader)
      : state_(std::make_shared<State>()) {
    state_->event = std::move(event);
    state_->reader = std::move(reader);
  }

  /**
   * Block until value is available. First call invokes the reader (which
   * handles evaluation, synchronization, and data read) and caches the
   * result. Subsequent calls return the cached value.
   */
  T get() {
    auto s = state_;
    std::call_once(s->once, [s] {
      if (s->reader) {
        s->value = s->reader();
      }
    });
    return *s->value;
  }

  /// Non-blocking readiness check. Does not trigger evaluation.
  bool isReady() const {
    return state_->value.has_value() || state_->event.isReady();
  }

  bool valid() const {
    return state_->value.has_value() || state_->reader != nullptr;
  }

 private:
  Future() : state_(std::make_shared<State>()) {}

  std::shared_ptr<State> state_;
};

} // namespace fl
