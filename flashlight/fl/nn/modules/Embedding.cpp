/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Embedding.h"

#include <cmath>
#include <format>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"

namespace fl {

Embedding::Embedding(int embeddingDim, int numEmbeddings)
    : embeddingDim_(embeddingDim), numEmbeddings_(numEmbeddings) {
  initialize();
}

Embedding::Embedding(const Tensor& w)
    : UnaryModule({w}), embeddingDim_(w.dim(1)), numEmbeddings_(w.dim(0)) {}

Embedding::Embedding(const Embedding& other)
    : UnaryModule(other.copyParams()),
      embeddingDim_(other.embeddingDim_),
      numEmbeddings_(other.numEmbeddings_) {
  train_ = other.train_;
}

Embedding& Embedding::operator=(const Embedding& other) {
  params_ = other.copyParams();
  train_ = other.train_;
  embeddingDim_ = other.embeddingDim_;
  numEmbeddings_ = other.numEmbeddings_;
  return *this;
}

void Embedding::initialize() {
  double stdv = std::sqrt(1.0 / static_cast<double>(embeddingDim_));
  auto embeddings =
      uniform(numEmbeddings_, embeddingDim_, -stdv, stdv, fl::dtype::f32, true);
  params_ = {embeddings};
}

Tensor Embedding::forward(const Tensor& input) {
  return autograd::embedding(input, params_[0]);
}

std::unique_ptr<Module> Embedding::clone() const {
  return std::make_unique<Embedding>(*this);
}

std::string Embedding::prettyString() const {
  return std::format(
      "Embedding (embeddings: {}) (dim: {})", numEmbeddings_, embeddingDim_);
}

} // namespace fl
