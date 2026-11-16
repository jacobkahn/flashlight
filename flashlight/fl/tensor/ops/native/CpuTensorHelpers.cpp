/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/native/CpuTensorHelpers.h"

#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"

namespace fl {
namespace native {

Tensor makeCpuTensor(const Shape& shape, fl::dtype type, const void* hostPtr) {
  // Stopgap: use CPUTensor as the CPU tensor implementation.
  // TODO(T_CPUTENSOR): Replace with toTensor<CPUTensor>(...) once a proper
  // CPUTensor adapter exists, then delete this shim entirely.
  return toTensor<CPUTensor>(shape, type, hostPtr, Location::Host);
}

} // namespace native
} // namespace fl
