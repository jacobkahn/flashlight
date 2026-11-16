/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/SinusoidalPositionEmbedding.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>
#include <stdexcept>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace fl {

SinusoidalPositionEmbedding::SinusoidalPositionEmbedding(
    int32_t layerDim,
    double inputScale /* = 1. */)
    : layerDim_(layerDim), inputScale_(inputScale) {
  // 'scale_' chosen based on positional embedding from:
  //      Attention is All You Need
  //      Ashish Vaswani, et al. (2017)
  //      https://arxiv.org/pdf/1706.03762.pdf
  // Create an `iota` that looks like `[0,0,1,1,2,2,...]`, use it as the scale.
  scale_ = fl::exp(
      -2 * fl::floor((fl::iota({layerDim_}) / 2)) * std::log(10000) /
      layerDim_);
  // Create a Cosine phase shift that acts on indices like `[0,1,0,1,...]`
  const double sinToCosPhaseShift = M_PI / 2.0;
  cosShifts_ = sinToCosPhaseShift * fl::iota({layerDim_}) % 2;
  // In the forward pass, the even indices of embedding vectors will have the
  // Sine function applied and the odd indices will have the Cosine function
  // applied.
}

SinusoidalPositionEmbedding::SinusoidalPositionEmbedding(
    const SinusoidalPositionEmbedding& other)
    : layerDim_(other.layerDim_),
      inputScale_(other.inputScale_),
      scale_(other.scale_.copy()),
      cosShifts_(other.cosShifts_.copy()) {}

SinusoidalPositionEmbedding& SinusoidalPositionEmbedding::operator=(
    const SinusoidalPositionEmbedding& other) {
  layerDim_ = other.layerDim_;
  inputScale_ = other.inputScale_;
  scale_ = other.scale_.copy();
  cosShifts_ = other.cosShifts_.copy();
  return *this;
}

std::vector<Tensor> SinusoidalPositionEmbedding::forward(
    const std::vector<Tensor>& input) {
  // Row-major: input is {B, T, C} or {B, T, C, 1}
  // C is always at dim 2
  if (input[0].dim(2) != layerDim_) {
    throw std::invalid_argument(
        "Input dimension " + std::to_string(input[0].dim(2)) +
        " and Embedding dimension " + std::to_string(layerDim_) +
        " are different.");
  }
  // Retrieve the number of tokens (positions) and the numeric type.
  const int nPositions = input[0].dim(1); // T
  const auto numType = input[0].type();
  // Generate the tensor of positions {T, C}:
  //   positions[t][c] = t (time index, same across channels)
  Tensor positions = fl::iota({nPositions, 1}, {1, layerDim_}, numType);
  // Generate the embedding transformation with the precomputed scale and shift.
  // scale_ and cosShifts_ are {layerDim_} = {C}. Reshape to {1, C} for
  // broadcasting.
  auto scaleRM = fl::unsqueeze(scale_.astype(numType), 0);
  auto shiftsRM = fl::unsqueeze(cosShifts_.astype(numType), 0);
  positions = fl::sin(
      positions * fl::tile(scaleRM, {nPositions, 1}) +
      fl::tile(shiftsRM, {nPositions, 1}));
  // positions: {T, C}. tileAs broadcasts to input shape (adds batch dim).
  Tensor embeddingsPos = Tensor(positions, false);
  return {input[0] * inputScale_ + autograd::tileAs(embeddingsPos, input[0])};
}

std::vector<Tensor> SinusoidalPositionEmbedding::operator()(
    const std::vector<Tensor>& input) {
  return forward(input);
}

std::unique_ptr<Module> SinusoidalPositionEmbedding::clone() const {
  return std::make_unique<SinusoidalPositionEmbedding>(*this);
}

std::string SinusoidalPositionEmbedding::prettyString() const {
  return std::format(
      "Sinusoidal Position Embedding Layer (embDim: {}), (input scale {})",
      layerDim_, inputScale_);
}

SinusoidalPositionEmbedding::SinusoidalPositionEmbedding() = default;

} // namespace fl
