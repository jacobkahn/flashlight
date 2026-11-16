/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdlib>

// Machinery for symbol visibility
#if defined(_WIN32) || defined(_MSC_VER)
#if FL_COMPILE_LIBRARY
// Only annotate if building shared libs/DLLs
#ifdef FL_DLL
#define FL_API __declspec(dllexport)
#else // FL_DLL
#define FL_API __declspec(dllimport)
#endif // FL_DLL
#else // FL_COMPILE_LIBRARY
#define FL_API
#endif // FL_COMPILE_LIBRARY
#else
// Non-MSVC compilers
#define FL_API __attribute__((visibility("default")))
#endif // defined(_WIN32) || defined(_MSC_VER)

// C++14 standard deprecated attribute -- works on all compilers
#define FL_DEPRECATED(msg) [[deprecated(msg)]]


namespace fl {

/**
 * \defgroup common_defines Common constants and definitions
 * @{
 */

/**
 * Reduction mode to used for CrossEntropy, AdaptiveSoftMax etc ...
 */
enum class ReduceMode {
  NONE = 0,
  MEAN = 1,
  SUM = 2,
};

/**
 * Pooling method to be used
 */
enum class PoolingMode {

  /// Use maximum value inside the pooling window
  MAX = 0,

  /// Use average value (including padding) inside the pooling window
  AVG_INCLUDE_PADDING = 1,

  /// Use average value (excluding padding) inside the pooling window// Use
  /// average value (excluding padding) inside the pooling window
  AVG_EXCLUDE_PADDING = 2,
};

/**
 * RNN network type
 */
enum class RnnMode {
  RELU = 0,
  TANH = 1,
  LSTM = 2,
  GRU = 3,
};

enum class PaddingMode {
  /// Use smallest possible padding such that out_size = ceil(in_size/stride)
  SAME = -1,
};

enum class DistributedBackend {
  /// https://github.com/facebookincubator/gloo
  GLOO = 0,
  /// https://developer.nvidia.com/nccl
  NCCL = 1,
  STUB = 2,
};

enum class DistributedInit {
  MPI = 0,
  FILE_SYSTEM = 1,
};

namespace DistributedConstants {
constexpr const char* kMaxDevicePerNode = "MAX_DEVICE_PER_NODE";
constexpr const char* kFilePath = "FILE_PATH";
constexpr const std::size_t kCoalesceCacheSize = ((size_t)(20) << 20); // 20 MB
} // namespace DistributedConstants

constexpr double kAmpMinimumScaleFactorValue = 1e-4;

/** @} */

} // namespace fl
