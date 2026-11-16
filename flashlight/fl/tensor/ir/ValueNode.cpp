/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/ValueNode.h"

namespace fl {

ValueNode::ValueNode(Tensor&& value, std::shared_ptr<const Stream> stream, PrivateHelper)
    : NodeTrait({}, value.shape(), value.type(), std::move(stream)) {
  setResult(std::move(value));
}

ValueNodePtr ValueNode::create(Tensor&& value, std::shared_ptr<const Stream> stream) {
  return std::make_shared<ValueNode>(std::move(value), std::move(stream), PrivateHelper{});
}

const Tensor& ValueNode::value() const {
  return getResult().value(); // guaranteed to be present (by construction)
}

} // namespace fl
