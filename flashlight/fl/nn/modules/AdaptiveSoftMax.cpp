/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <format>
#include <stdexcept>
#include <vector>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/modules/AdaptiveSoftMax.h"
#include "flashlight/fl/tensor/Index.h"

namespace fl {

AdaptiveSoftMax::AdaptiveSoftMax(
    int inputSize,
    const std::vector<int>& cutoff,
    float divValue)
    : UnaryModule(), cutoff_(cutoff), divValue_(divValue) {
  if (cutoff_.empty()) {
    throw std::invalid_argument("invalid cutoff for AdaptiveSoftMaxLoss");
  }

  int outputSize = cutoff_[0] + cutoff_.size() - 1;

  auto head = kaimingUniform(
      {outputSize, inputSize}, inputSize /* fanIn */, fl::dtype::f32, true);
  params_.push_back(head);

  int denominator = 1;
  for (int i = 0; i < cutoff_.size() - 1; i++) {
    denominator *= divValue_;
    int hiddenSize = inputSize / denominator;
    auto tail1 = kaimingUniform(
        {hiddenSize, inputSize}, inputSize /* fanIn */, fl::dtype::f32, true);
    auto tail2 = kaimingUniform(
        {cutoff_[i + 1] - cutoff_[i], hiddenSize},
        hiddenSize /* fanIn */,
        fl::dtype::f32,
        true);

    params_.push_back(tail1);
    params_.push_back(tail2);
  }
}

Tensor AdaptiveSoftMax::getFullLogProb(
    const Tensor& inputs,
    const Tensor& headOutput) const {
  // Row-major: inputs {B, C_in}, headOutput {B, headSize}
  auto outputSize = cutoff_[cutoff_.size() - 1];
  auto batchSize = inputs.dim(0);
  Tensor output({batchSize, outputSize}, inputs.type());

  output(
      fl::span,
      fl::range(0, cutoff_[0] + static_cast<long long>(cutoff_.size()) - 1)) =
      headOutput;

  for (int i = cutoff_.size() - 2; i >= 0; i--) {
    auto tailOutput = autograd::matmul(inputs, params_[1 + i * 2], MatrixProperty::None, MatrixProperty::Transpose);
    tailOutput = autograd::matmul(tailOutput, params_[2 + i * 2], MatrixProperty::None, MatrixProperty::Transpose);
    auto idx = i + cutoff_[0];
    tailOutput = autograd::logSoftmax(tailOutput, 1) +
        autograd::tileAs(headOutput(fl::span, fl::range(idx, idx + 1)), tailOutput);
    output(fl::span, fl::range(cutoff_[i], cutoff_[i + 1])) = tailOutput;
  }

  return Tensor(output, false);
}

Tensor AdaptiveSoftMax::forward(const Tensor& inputs) {
  // Row-major: input -- [N, .., C_in], return -- [N, .., C_out]
  auto lastDim = inputs.ndim() - 1;
  auto inputSize = inputs.dim(lastDim);
  if (inputSize != params_[0].dim(1)) {
    throw std::invalid_argument("invalid input dimension for AdaptiveSoftMax");
  }

  auto inputsFlattened = autograd::reshape(inputs, {-1, inputSize});
  auto headOutput = autograd::logSoftmax(
      autograd::matmul(inputsFlattened, params_[0], MatrixProperty::None, MatrixProperty::Transpose), 1);

  auto ret = getFullLogProb(inputsFlattened, headOutput);

  Shape outDims = inputs.shape();
  outDims[lastDim] = ret.dim(1);
  return autograd::reshape(ret, outDims);
}

Tensor AdaptiveSoftMax::predict(const Tensor& inputs) const {
  // Row-major: input -- [N, .., C], return -- [N, .., 1]
  auto lastDim = inputs.ndim() - 1;
  auto inputSize = inputs.dim(lastDim);
  if (inputSize != params_[0].dim(1)) {
    throw std::invalid_argument(
        "invalid input dimension for AdaptiveSoftMaxLoss");
  }

  auto inputsFlattened = autograd::reshape(inputs, {-1, inputSize});
  auto headOutput = autograd::matmul(inputsFlattened, params_[0], MatrixProperty::None, MatrixProperty::Transpose);
  Tensor maxValue, prediction;
  fl::max(maxValue, prediction, headOutput, 1);

  auto notInShortlist = (prediction >= cutoff_[0]);
  Tensor ret = Tensor(prediction, false);
  if (fl::any(notInShortlist).asScalar<bool>()) {
    headOutput = autograd::logSoftmax(headOutput, 1);
    // Use integer row indices instead of boolean indexing, which flattens
    // 2D tensors on non-ArrayFire backends.
    auto rowIndices = fl::nonzero(notInShortlist.flatten());
    auto logProbTailPositions = getFullLogProb(
        inputsFlattened(rowIndices),
        headOutput(rowIndices));
    Tensor maxValueTailPositions, predictionTailPositions;
    fl::max(
        maxValueTailPositions,
        predictionTailPositions,
        logProbTailPositions,
        1);
    ret.detach()(rowIndices) = predictionTailPositions;
  }

  Shape outDims = inputs.shape();
  outDims[lastDim] = 1;
  return autograd::reshape(ret, outDims);
}

std::vector<int> AdaptiveSoftMax::getCutoff() const {
  return cutoff_;
}

std::unique_ptr<Module> AdaptiveSoftMax::clone() const {
  return std::make_unique<AdaptiveSoftMax>(*this);
}

std::string AdaptiveSoftMax::prettyString() const {
  std::string cutoffs;
  for (size_t i = 0; i < cutoff_.size(); ++i) {
    if (i > 0) cutoffs += ", ";
    cutoffs += std::to_string(cutoff_[i]);
  }
  return std::format("Adaptive Softmax ({})", cutoffs);
}

} // namespace fl
