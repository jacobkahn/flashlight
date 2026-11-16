/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ostream>
#include <string>

namespace fl {

/**
 * A runtime type for various types of jit nodes.
 */
enum class NodeType {
  Binary,
  Cast,
  Clip,
  Concatenate,
  Custom,
  Cumsum,
  Scalar,
  Value,
  Index,
  IndexedUpdate,
  Flip,
  Roll,
  Unary,
  Reshape,
  Reduction,
  Transpose,
  Triangular,
  FusedElementwise,
  FusedReduceElementwise,
  Matmul,
  Conv2d,
  Projection,
  BatchNorm,
  Pool2d,
  MultiValue,
  Transfer,
  Pad,
  Tile,
  Where,
};

/**
 * Return a readable string representation of the given node type.
 *
 * @return a string that represents the given node type.
 */
std::string nodeTypeToString(const NodeType type);

/**
 * Output a string representation of `type` to `os`.
 */
std::ostream& operator<<(std::ostream& os, const NodeType& type);

} // namespace fl
