/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "flashlight/fl/common/Defines.h"

namespace fl {

class Device;
class Event;

enum class StreamType {
  CUDA,
  Synchronous,
  AsyncCPU,
};

/**
 * An abstraction that represents a sequence of computations that must happen
 * synchronously on a specific device. It focuses on synchronization of the
 * computations, while being agnostic to the computations themselves.
 */
class FL_API Stream {
 public:
  Stream() = default;
  virtual ~Stream() = default;

  // no copy/move
  Stream(const Stream&) = delete;
  Stream(Stream&&) = delete;
  Stream& operator=(const Stream&) = delete;
  Stream& operator=(const Stream&&) = delete;

  /**
   * Get the underlying implementation of this stream.
   *
   * Throws invalid_argument if the specified type does not match the actual
   * derived stream type.
   *
   * @return an immutable reference to the specified stream type.
   */
  template <typename T>
  const T& impl() const {
    if (T::streamType != type()) {
      throw std::invalid_argument(
          "[fl::Stream::impl] "
          "specified stream type doesn't match actual stream type.");
    }
    return *(static_cast<const T*>(this));
  }

  /**
   * Returns the type of this stream.
   *
   * @return a enum denoting stream type.
   */
  virtual StreamType type() const = 0;

  /**
   * Return the owner device of this stream.
   *
   * @return a reference to the owner device of this stream.
   */
  virtual Device& device() = 0;

  /**
   * Return the owner device of this stream.
   *
   * @return an immutable reference to the owner device of this stream.
   */
  virtual const Device& device() const = 0;

  /**
   * Block calling thread and synchronize w.r.t. all tasks on this stream.
   */
  virtual void sync() const = 0;

  /**
   * Synchronize future tasks on this stream w.r.t. current tasks on given
   * stream, i.e., the former can only start after the completion of the latter.
   * NOTE this function may or may not block the calling thread.
   *
   * @param[in] waitOn the stream to perform relative synchronization against.
   */
  virtual void relativeSync(const Stream& waitOn) const = 0;

  /**
   * Synchronize future tasks on this stream w.r.t. current tasks on all given
   * stream, i.e., the former can only start after the completion of the latter.
   * NOTE this function may or may not block the calling thread.
   *
   * @param[in] waitOns the streams to perform relative synchronization against.
   */
  virtual void relativeSync(
      const std::unordered_set<const Stream*>& waitOns) const;

  /**
   * Record and return an Event marking the current point in this stream.
   */
  virtual Event createEvent() const = 0;
};

/**
 * A trait for some generic stream functionalities.
 *
 * REQUIRED definition in derived class:
 *   static constexpr StreamType streamType;
 */
template <typename Derived>
class StreamTrait : public Stream {
 public:
  // prevent name hiding
  using Stream::relativeSync;

  // A specialized relativeSync for streams of the same type.
  virtual void relativeSync(const Derived& waitOn) const = 0;

  StreamType type() const override {
    return Derived::streamType;
  }

  virtual void relativeSync(const Stream& waitOn) const override {
    switch (waitOn.type()) {
      case Derived::streamType:
        relativeSync(waitOn.impl<Derived>());
        break;
      default:
        throw std::runtime_error(
            "[Stream::relativeSync] Unsupported for different types of streams");
    }
  }
};

/**
 * Type-checked downcast of a shared_ptr<const Stream> to a concrete stream
 * type. Throws std::invalid_argument if the stream's runtime type doesn't
 * match T::streamType.
 *
 * Example:
 *   auto cudaStream = streamAs<CUDAStream>(someStream);
 *   cudaStream_t h = cudaStream->handle();
 */
template <typename T>
std::shared_ptr<T> streamAs(std::shared_ptr<const Stream> s) {
  if (!s) {
    throw std::invalid_argument(
        "[fl::streamAs] null stream pointer");
  }
  if (s->type() != T::streamType) {
    throw std::invalid_argument(
        "[fl::streamAs] stream type mismatch: expected " +
        std::to_string(static_cast<int>(T::streamType)) +
        " but got " +
        std::to_string(static_cast<int>(s->type())));
  }
  return std::const_pointer_cast<T>(
      std::static_pointer_cast<const T>(std::move(s)));
}

} // namespace fl
