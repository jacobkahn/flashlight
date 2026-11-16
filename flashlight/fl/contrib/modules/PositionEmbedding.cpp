/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/PositionEmbedding.h"

#include <stdexcept>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"
#include "flashlight/fl/tensor/Index.h"

namespace fl {

PositionEmbedding::PositionEmbedding(
    int32_t layerDim,
    int32_t maxLen,
    double dropout)
    : dropout_(dropout) {
  // Row-major: embedding is {maxLen, layerDim}
  auto embeddings = uniform(maxLen, layerDim, -0.1, 0.1, fl::dtype::f32, true);
  params_ = {embeddings};
}

PositionEmbedding::PositionEmbedding(const PositionEmbedding& other)
    : Module(other.copyParams()), dropout_(other.dropout_) {
  train_ = other.train_;
}

PositionEmbedding& PositionEmbedding::operator=(
    const PositionEmbedding& other) {
  params_ = other.copyParams();
  train_ = other.train_;
  dropout_ = other.dropout_;
  return *this;
}

std::vector<Tensor> PositionEmbedding::forward(
    const std::vector<Tensor>& input) {
  if (input[0].ndim() != 3) {
    throw std::invalid_argument(
        "PositionEmbedding::forward - expect a tensor with "
        "3 dimensions - B x T x C");
  }

  // Row-major: input is {B, T, C}, params_ is {maxLen, layerDim}
  int n = input[0].dim(1); // T
  // Slice first n rows: {T, C}. tileAs broadcasts to {B, T, C}
  Tensor posEmb = autograd::tileAs(
      params_[0].astype(input[0].type())(fl::range(0, n)), input[0]);
  if (dropout_ > 0.0 && train_) {
    return {input[0] + autograd::dropout(posEmb, dropout_)};
  } else {
    return {input[0] + posEmb};
  }
}

std::vector<Tensor> PositionEmbedding::operator()(
    const std::vector<Tensor>& input) {
  return forward(input);
}

std::unique_ptr<Module> PositionEmbedding::clone() const {
  return std::make_unique<PositionEmbedding>(*this);
}

std::string PositionEmbedding::prettyString() const {
  return "Position Embedding Layer";
}

PositionEmbedding::PositionEmbedding() = default;

} // namespace fl
