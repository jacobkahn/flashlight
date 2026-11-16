/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdlib>
#include <utility>

namespace fl {

/**
 * RAII wrapper for 64-byte-aligned memory. The alignment satisfies OneDNN's
 * AVX-512 requirement and is also good for general cache performance.
 */
struct AlignedBuffer {
  void* data{nullptr};
  size_t sizeBytes{0};

  AlignedBuffer() = default;

  explicit AlignedBuffer(size_t bytes) : sizeBytes(bytes) {
    if (bytes > 0) {
      // 64-byte alignment for AVX-512
      if (posix_memalign(&data, 64, bytes) != 0) {
        data = nullptr;
        sizeBytes = 0;
        throw std::bad_alloc();
      }
    }
  }

  ~AlignedBuffer() {
    free(data);
  }

  // Non-copyable
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  // Movable
  AlignedBuffer(AlignedBuffer&& other) noexcept
      : data(other.data), sizeBytes(other.sizeBytes) {
    other.data = nullptr;
    other.sizeBytes = 0;
  }

  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
      free(data);
      data = other.data;
      sizeBytes = other.sizeBytes;
      other.data = nullptr;
      other.sizeBytes = 0;
    }
    return *this;
  }
};

} // namespace fl
