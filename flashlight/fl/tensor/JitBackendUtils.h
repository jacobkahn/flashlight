/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "flashlight/fl/tensor/JitTensorBase.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/ir/Node.h"

namespace fl {
namespace detail {

std::vector<NodePtr> tensorsToNodes(const std::vector<Tensor>& tensors);

template <typename... T>
std::vector<NodePtr> tensorsToNodes(const T&... tensors) {
  std::vector<NodePtr> nodes;
  for (const auto& tensor : {&tensors...}) {
    nodes.push_back(toJitTensorBase(*tensor).node());
  }
  return nodes;
}

template <>
inline std::vector<NodePtr> tensorsToNodes() {
  return {};
}

// Materialize a JIT tensor, or return non-JIT/empty tensors as-is.
// Evaluates the graph and blocks until complete.
const Tensor& materializeSafe(const Tensor& tensor);

// Materialize a mutable JIT tensor reference. The reference is replaced with
// the underlying concrete tensor so the wrapped backend can read from and write
// to its buffer directly.
void materializeMutable(Tensor& tensor);

} // namespace detail
} // namespace fl
