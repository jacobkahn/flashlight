/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/PadNode.h"

#include <stdexcept>

#include "flashlight/fl/tensor/ShapeInference.h"

namespace fl {

namespace {

void validateNonNegative(const std::vector<std::pair<int, int>>& padWidths) {
  for (const auto& [before, after] : padWidths) {
    if (before < 0 || after < 0) {
      throw std::invalid_argument("PadNode requires nonnegative pad widths");
    }
  }
}

std::vector<std::pair<int, int>> normalizePadWidths(
    const Shape& inputShape,
    const std::vector<std::pair<int, int>>& padWidths) {
  if (padWidths.size() > static_cast<size_t>(inputShape.ndim())) {
    // Use the existing shape inference error text for this case.
    inferPadOutputShape(inputShape, padWidths);
  }
  auto normalized = padWidths;
  normalized.resize(inputShape.ndim(), {0, 0});
  return normalized;
}

} // namespace

PadNode::PadNode(
    NodePtr input,
    std::vector<std::pair<int, int>> padWidths,
    PadType type,
    const Shape& outputShape,
    dtype dtype,
    std::shared_ptr<const Stream> stream,
    PrivateHelper)
    : NodeTrait({input}, outputShape, dtype, std::move(stream)),
      padWidths_(std::move(padWidths)),
      type_(type) {}

PadNodePtr PadNode::create(
    NodePtr input,
    const std::vector<std::pair<int, int>>& padWidths,
    PadType type,
    std::shared_ptr<const Stream> stream) {
  validateNonNegative(padWidths);
  auto normalizedPadWidths = normalizePadWidths(input->shape(), padWidths);
  auto outputShape = inferPadOutputShape(input->shape(), normalizedPadWidths);
  return std::make_shared<PadNode>(
      input,
      std::move(normalizedPadWidths),
      type,
      outputShape,
      input->dataType(),
      std::move(stream),
      PrivateHelper{});
}

NodePtr PadNode::input() const {
  return getInput(kInputIdx);
}

const std::vector<std::pair<int, int>>& PadNode::padWidths() const {
  return padWidths_;
}

PadType PadNode::padType() const {
  return type_;
}

} // namespace fl
