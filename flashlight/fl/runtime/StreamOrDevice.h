/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <variant>

namespace fl {

class Stream;
class Device;

/**
 * A type that can hold a Stream, a Device, or represent "use the backend's
 * default stream". Uses shared_ptr for safe lifetime: the JIT graph keeps
 * streams alive as long as any node references them.
 *
 * Implicit constructors allow natural usage:
 *   auto s = std::make_shared<AsyncCPUStream>(...);
 *   fl::exp(a, s);
 */
class StreamOrDevice {
  std::variant<
      std::monostate,
      std::shared_ptr<const Stream>,
      std::shared_ptr<const Device>> v_;

 public:
  StreamOrDevice() = default;
  /* implicit */ StreamOrDevice(std::shared_ptr<const Stream> s)
      : v_(std::move(s)) {}
  /* implicit */ StreamOrDevice(std::shared_ptr<const Device> d)
      : v_(std::move(d)) {}
  /* implicit */ StreamOrDevice(const Device& d)
      : v_(std::shared_ptr<const Device>(&d, [](const Device*) {})) {}

  bool isDefault() const {
    return std::holds_alternative<std::monostate>(v_);
  }

  bool hasDevice() const {
    return std::holds_alternative<std::shared_ptr<const Device>>(v_);
  }

  /// Returns the Device reference. Caller must check hasDevice() first.
  const Device& device() const {
    return *std::get<std::shared_ptr<const Device>>(v_);
  }

  /// Returns the Stream shared_ptr (empty if default/device).
  std::shared_ptr<const Stream> stream() const {
    if (auto* s = std::get_if<std::shared_ptr<const Stream>>(&v_)) {
      return *s;
    }
    return nullptr;
  }

  /// Move-extract the Stream shared_ptr (avoids refcount bump).
  std::shared_ptr<const Stream> takeStream() {
    if (auto* s = std::get_if<std::shared_ptr<const Stream>>(&v_)) {
      return std::move(*s);
    }
    return nullptr;
  }
};

} // namespace fl
