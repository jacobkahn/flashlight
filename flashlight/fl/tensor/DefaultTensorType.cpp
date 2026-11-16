/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/DefaultTensorType.h"

#include "flashlight/fl/tensor/TensorAdapter.h"

namespace fl {

TensorBackend& defaultTensorBackend() {
  return detail::DefaultTensorType::getInstance().getBackend();
}

} // namespace fl
