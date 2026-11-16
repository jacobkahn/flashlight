/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Types.h"

#ifdef FL_TEST_BACKEND_CPU
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/JitTensorBase.h"
#endif
#ifdef FL_TEST_BACKEND_CUDA
#include "flashlight/fl/tensor/backend/cuda/CudaTensor.h"
#include "flashlight/fl/tensor/JitTensorBase.h"
#endif
#ifdef FL_TEST_BACKEND_AF
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#endif

/**
 * Skip the current test if the default tensor backend does not support the
 * given dtype. This is backend-agnostic: on OneDNN it skips for s16, s64, u16,
 * u32, u64 (no OneDNN mapping); on ArrayFire it runs all types.
 */
#define FL_SKIP_IF_DTYPE_UNSUPPORTED(dtype)                                    \
  do {                                                                         \
    if (!fl::defaultTensorBackend().supportsDataType(dtype)) {                 \
      GTEST_SKIP() << "Backend does not support "                              \
                   << fl::dtypeToString(dtype);                                \
    }                                                                          \
  } while (0)

/**
 * Initialize Flashlight for tests.
 *
 * When compiled with -DFL_TEST_BACKEND_CPU, sets JIT+CPUTensor as the default.
 * When compiled with -DFL_TEST_BACKEND_CUDA, sets JIT+CudaTensor as the default.
 * When compiled with -DFL_TEST_BACKEND_AF, sets bare ArrayFireTensor (no JIT)
 * as the default: the ArrayFire baseline the native backends are measured
 * against.
 *
 * Otherwise uses the compile-time default (native CUDA if built, else CPU;
 * see DefaultTensorType.h).
 */
inline void flTestInit() {
#ifdef FL_TEST_BACKEND_CPU
  fl::setDefaultJitTensorType<fl::CPUTensor>();
  fl::initLogging();
#elif defined(FL_TEST_BACKEND_CUDA)
  fl::setDefaultJitTensorType<fl::CudaTensor>();
  fl::initLogging();
#elif defined(FL_TEST_BACKEND_AF)
  fl::setDefaultTensorType<fl::ArrayFireTensor>();
  fl::initLogging();
#else
  fl::init();
#endif
}
