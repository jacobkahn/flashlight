/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/MultiValueNode.h"

#include <stdexcept>

namespace fl {

MultiValueNodePtr MultiValueNode::create(
    std::vector<Tensor> values,
    std::shared_ptr<const Stream> stream) {
  if (values.empty()) {
    throw std::invalid_argument("MultiValueNode requires at least one value");
  }
  auto shape = values[0].shape();
  auto dt = values[0].type();
  return std::shared_ptr<MultiValueNode>(
      new MultiValueNode(
          std::move(shape), dt, std::move(values), std::move(stream)));
}

MultiValueNode::MultiValueNode(
    Shape shape,
    dtype dataType,
    std::vector<Tensor> values,
    std::shared_ptr<const Stream> stream)
    : NodeTrait<MultiValueNode>(
          /* inputs = */ {},
          std::move(shape),
          dataType,
          std::move(stream)) {
  outputs_.resize(values.size());
  for (unsigned i = 0; i < values.size(); ++i) {
    if (values[i].elements() > 0) { // skip empty slots (dead outputs)
      setOutput(i, std::move(values[i]));
    }
  }
}

} // namespace fl
