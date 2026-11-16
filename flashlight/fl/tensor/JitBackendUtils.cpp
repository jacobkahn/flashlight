/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/JitBackendUtils.h"

namespace fl {
namespace detail {

std::vector<NodePtr> tensorsToNodes(const std::vector<Tensor>& tensors) {
  std::vector<NodePtr> nodes;
  for (const auto& tensor : tensors) {
    nodes.push_back(toJitTensorBase(tensor).node());
  }
  return nodes;
}

const Tensor& materializeSafe(const Tensor& tensor) {
  if (tensor.isEmpty() || tensor.backendType() != TensorBackendType::Jit) {
    return tensor;
  }
  const auto& jitTensor = toJitTensorBase(tensor);
  jitTensor.eval();
  jitTensor.stream().sync();
  return jitTensor.node()->getResult().value();
}

void materializeMutable(Tensor& tensor) {
  if (tensor.isEmpty() || tensor.backendType() != TensorBackendType::Jit) {
    return;
  }
  auto& jitTensor = toJitTensorBase(tensor);
  jitTensor.eval();
  jitTensor.stream().sync();
  tensor = Tensor(jitTensor.node()->getResult().value());
}

} // namespace detail
} // namespace fl
