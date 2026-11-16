/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/common/Utils.h"

namespace fl::cuda::detail {

/**
 * Whether f32 math may use TF32 (tensor cores with a 10-bit mantissa). Strict
 * f32 is the default; set the environment variable FL_ALLOW_TF32=1 to opt in.
 * Read once, on first use. Every CUDA path that chooses an f32 math mode
 * (cuDNN engine selection, cuDNN RNN math type, cuBLAS/cuBLASLt compute type)
 * consults this, so results don't depend on which path an op takes.
 *
 * Internal to the CUDA primitive-library code (cuDNN, cuBLAS, cuBLASLt); not
 * part of the public API. Header-only because the cuDNN ops also build without
 * the native CUDA backend.
 */
inline bool allowTF32() {
  static const bool allowed = fl::getEnvVar("FL_ALLOW_TF32") == "1";
  return allowed;
}

} // namespace fl::cuda::detail
