/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/common/Defines.h"

namespace fl {

FL_API double clipGradNorm(
    const std::vector<Tensor>& parameters,
    double max_norm);

}
