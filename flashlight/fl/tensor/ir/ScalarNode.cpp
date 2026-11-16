/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/ScalarNode.h"

namespace fl {

ScalarNode::ScalarNode(
    const Shape& shape,
    const fl::dtype type,
    const ScalarType scalar,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait({}, shape, type, std::move(stream)), scalar_(scalar) {}

} // namespace fl
