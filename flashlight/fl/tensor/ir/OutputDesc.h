/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Types.h"

namespace fl {

// Describes the shape and dtype of one output of a multi-output node.
struct OutputDesc {
  Shape shape;
  dtype type;
};

} // namespace fl
