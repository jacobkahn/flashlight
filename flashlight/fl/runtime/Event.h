/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/fl/common/Defines.h"

namespace fl {

class Stream;

namespace detail {
struct EventImpl;
} // namespace detail

/**
 * A unified, type-erased synchronization event.
 *
 * Records a point in a Stream's task queue. When the stream reaches that
 * point the event is signaled. Other threads can wait on or poll the event.
 *
 * Copyable via shared_ptr — copies refer to the same underlying event.
 * Calling record() replaces this object's impl; prior copies keep theirs.
 */
class FL_API Event {
 public:
  /** Default-constructed Event is immediately ready (no-op). */
  Event();

  /** Construct from an EventImpl (used by Stream subclasses). */
  explicit Event(std::shared_ptr<detail::EventImpl> impl);

  /**
   * Record this event on the given stream.
   *
   * Creates a new underlying impl each time. Old copies of the Event keep
   * the old impl — this matches cudaEventRecord semantics.
   *
   * @param[in] stream the stream to record on. Dispatches by stream type:
   *   - Synchronous  -> always-ready (no-op)
   *   - AsyncCPU     -> promise/shared_future on the worker thread
   *   - CUDA         -> cudaEventRecord (compiled only when FL_USE_CUDA)
   */
  void record(const Stream& stream);

  /** Block the calling thread until this event is signaled. */
  void wait() const;

  /** Non-blocking poll. Returns true if the event has been signaled. */
  bool isReady() const;

 private:
  std::shared_ptr<detail::EventImpl> impl_;
};

} // namespace fl
