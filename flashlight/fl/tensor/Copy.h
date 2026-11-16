/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/fl/common/Defines.h"

namespace fl {

class Tensor;
class Stream;

/// Functional: returns a new tensor on the target stream's device.
/// JIT mode: creates a TransferNode (lazy, no materialization).
/// Non-JIT: allocates dst, then calls imperative copy(dst, src).
/// Source may be strided; backend handles it at eval/copy time.
FL_API Tensor copy(const Tensor& src, std::shared_ptr<const Stream> dstStream);

/// Imperative: copy data from src into pre-allocated dst.
/// Both must have the same shape and dtype. src and dst may be strided.
/// Stream and backend are inferred from src/dst device types.
FL_API void copy(Tensor& dst, const Tensor& src);

} // namespace fl
