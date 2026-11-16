/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/AdaptiveEmbedding.h"

#include <cmath>
#include <format>
#include <stdexcept>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/tensor/Index.h"

namespace fl {

AdaptiveEmbedding::AdaptiveEmbedding(
    int embeddingDim,
    std::vector<int> cutoff,
    float divValue /*= 4 */)
    : embeddingDim_(embeddingDim), cutoff_(cutoff), divValue_(divValue) {
  if (cutoff_.empty()) {
    throw std::invalid_argument("Invalid cutoff for AdaptiveEmbedding");
  }
  double stdv = std::sqrt(1.0 / static_cast<double>(embeddingDim_));
  // to be in agreement with the adaptive softmax to simplify
  // tied version of adaptive input and softmax
  auto headEmbedding = fl::normal(cutoff_[0], embeddingDim_, stdv, 0);
  params_.push_back(headEmbedding);
  auto head = fl::glorotUniform(
      {embeddingDim_, embeddingDim_}, embeddingDim_, embeddingDim_);
  params_.push_back(head);

  int denominator = 1;
  for (int tailIdx = 1; tailIdx < cutoff_.size(); tailIdx++) {
    denominator *= divValue_;
    int tailEmbeddingDim = embeddingDim_ / denominator;
    double stdvTail = std::sqrt(1.0 / static_cast<double>(tailEmbeddingDim));
    // to be in agreement with the adaptive softmax to simplify
    // tied version of adaptive input and softmax
    auto tailEmbedding = fl::normal(
        cutoff_[tailIdx] - cutoff_[tailIdx - 1], tailEmbeddingDim, stdvTail, 0);
    params_.push_back(tailEmbedding);
    auto tail = fl::glorotUniform(
        {embeddingDim_, tailEmbeddingDim}, tailEmbeddingDim, embeddingDim_);
    params_.push_back(tail);
  }
}

Tensor AdaptiveEmbedding::forward(const Tensor& input) {
  if (input.ndim() != 2) {
    throw std::invalid_argument(
        "AdaptiveEmbedding::forward - input must "
        "have 2 dimensions - expect B x T");
  }

  auto flatInput = autograd::flatten(input);
  std::vector<Tensor> indices;
  std::vector<Tensor> embeddings;

  // Row-major: embedding table is {numEmbeddings, embeddingDim}
  // params_[0] is already {cutoff_[0], embeddingDim_} - correct for embedding()
  Tensor headMask = flatInput < cutoff_[0];
  if (fl::sum(headMask).asScalar<unsigned>() > 0) {
    // embedding returns {numSelected, embeddingDim_}
    auto headEmbedding = autograd::embedding(flatInput(headMask), params_[0]);
    // Project: {numSelected, embDim} @ {embDim, embDim}^T = {numSelected, embDim}
    headEmbedding = autograd::matmul(headEmbedding, params_[1], MatrixProperty::None, MatrixProperty::Transpose);
    indices.emplace_back(fl::nonzero(headMask), false);
    embeddings.push_back(headEmbedding);
  }

  for (int tailIdx = 1; tailIdx < cutoff_.size(); tailIdx++) {
    Tensor tailMask = flatInput < cutoff_[tailIdx] &&
        flatInput >= cutoff_[tailIdx - 1];
    if (fl::any(tailMask).asScalar<bool>()) {
      // embedding returns {numSelected, tailEmbeddingDim}
      auto tailEmbedding = autograd::embedding(
          flatInput(tailMask) - cutoff_[tailIdx - 1],
          params_[tailIdx * 2]);
      // Project: {numSelected, tailEmbDim} @ {embDim, tailEmbDim}^T
      //        = {numSelected, embDim}
      tailEmbedding = autograd::matmul(tailEmbedding, params_[tailIdx * 2 + 1], MatrixProperty::None, MatrixProperty::Transpose);
      indices.emplace_back(fl::nonzero(tailMask), false);
      embeddings.push_back(tailEmbedding);
    }
  }
  if (embeddings.empty()) {
    throw std::invalid_argument(
        "Invalid input, no positions in the AdaptiveEmbedding layer");
  }

  // Row-major: output is {B, T, embeddingDim_}
  Shape outShape({input.dim(0), input.dim(1), embeddingDim_});
  // Each embedding is {numSelected_i, embeddingDim_}. Concatenate along dim 0.
  auto result = autograd::concatenate(embeddings, 0);
  auto resultIndices = autograd::concatenate(indices, 0);
  Tensor tmpIndices = fl::argsort(resultIndices, 0);
  // Reorder rows by original position, then reshape to {B, T, C}
  return autograd::reshape(result(tmpIndices), outShape);
}

std::unique_ptr<Module> AdaptiveEmbedding::clone() const {
  return std::make_unique<AdaptiveEmbedding>(*this);
}

std::string AdaptiveEmbedding::prettyString() const {
  std::string cutoffs;
  for (size_t i = 0; i < cutoff_.size(); ++i) {
    if (i > 0) cutoffs += ", ";
    cutoffs += std::to_string(cutoff_[i]);
  }
  return std::format(
      "AdaptiveEmbedding (dim: {}), (cutoff: {}), (divValue: {})",
      embeddingDim_, cutoffs, divValue_);
}

} // namespace fl
