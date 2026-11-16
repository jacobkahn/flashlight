/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/NodeType.h"

#include <stdexcept>

namespace fl {

std::string nodeTypeToString(const NodeType type) {
  switch (type) {
    case NodeType::Binary:
      return "Binary";
    case NodeType::Cast:
      return "Cast";
    case NodeType::Clip:
      return "Clip";
    case NodeType::Concatenate:
      return "Concatenate";
    case NodeType::Custom:
      return "Custom";
    case NodeType::Cumsum:
      return "Cumsum";
    case NodeType::Scalar:
      return "Scalar";
    case NodeType::Value:
      return "Value";
    case NodeType::Index:
      return "Index";
    case NodeType::IndexedUpdate:
      return "IndexedUpdate";
    case NodeType::Flip:
      return "Flip";
    case NodeType::Roll:
      return "Roll";
    case NodeType::Unary:
      return "Unary";
    case NodeType::Reshape:
      return "Reshape";
    case NodeType::Reduction:
      return "Reduction";
    case NodeType::Transpose:
      return "Transpose";
    case NodeType::Triangular:
      return "Triangular";
    case NodeType::FusedElementwise:
      return "FusedElementwise";
    case NodeType::FusedReduceElementwise:
      return "FusedReduceElementwise";
    case NodeType::Matmul:
      return "Matmul";
    case NodeType::Conv2d:
      return "Conv2d";
    case NodeType::Projection:
      return "Projection";
    case NodeType::BatchNorm:
      return "BatchNorm";
    case NodeType::Pool2d:
      return "Pool2d";
    case NodeType::MultiValue:
      return "MultiValue";
    case NodeType::Transfer:
      return "Transfer";
    case NodeType::Pad:
      return "Pad";
    case NodeType::Tile:
      return "Tile";
    case NodeType::Where:
      return "Where";
  }
  throw std::runtime_error("Unknown node type");
}

std::ostream& operator<<(std::ostream& os, const NodeType& type) {
  return os << nodeTypeToString(type);
}

} // namespace fl
