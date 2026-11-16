/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/ir/Node.h"

namespace fl {

/// If `node` is max(x, Scalar(0)) or max(Scalar(0), x), return x.
NodePtr tryMatchRelu(const NodePtr& node);

/// If `node` is x * 0.5 * (1 + erf(x * M_SQRT1_2)), return x.
/// All interior nodes must be single-use.
NodePtr tryMatchGeluErf(const NodePtr& node);

/// If `node` is x * sigmoid(x), return x.
NodePtr tryMatchSilu(const NodePtr& node);

} // namespace fl
