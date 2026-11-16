/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Loss.h"

#include <format>
#include <stdexcept>
#include <vector>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/tensor/Index.h"

namespace fl {

Tensor MeanSquaredError::forward(
    const Tensor& inputs,
    const Tensor& targets) {
  if (inputs.shape() != targets.shape()) {
    throw std::invalid_argument(std::format(
        "MeanSquaredError::forward - inputs and targets are of different"
        " sizes: {{inputs: {}, targets: {}}}",
        inputs.shape().toString(), targets.shape().toString()));
  }

  auto df = inputs - targets;
  auto res = autograd::mean(autograd::flatten(df * df), {0});
  return res;
}

std::unique_ptr<Module> MeanSquaredError::clone() const {
  return std::make_unique<MeanSquaredError>(*this);
}

std::string MeanSquaredError::prettyString() const {
  return "MeanSquaredError";
}

Tensor MeanAbsoluteError::forward(
    const Tensor& inputs,
    const Tensor& targets) {
  if (inputs.shape() != targets.shape()) {
    throw std::invalid_argument(std::format(
        "MeanAbsoluteError::forward - inputs and targets are of different"
        " sizes: {{inputs: {}, targets: {}}}",
        inputs.shape().toString(), targets.shape().toString()));
  }

  auto df = inputs - targets;
  return autograd::mean(autograd::flatten(autograd::abs(df)), {0});
}

std::unique_ptr<Module> MeanAbsoluteError::clone() const {
  return std::make_unique<MeanAbsoluteError>(*this);
}

std::string MeanAbsoluteError::prettyString() const {
  return "MeanAbsoluteError";
}

Tensor BinaryCrossEntropy::forward(
    const Tensor& inputs,
    const Tensor& targets) {
  return autograd::mean(autograd::flatten(autograd::binaryCrossEntropy(inputs, targets)), {0});
}

Tensor BinaryCrossEntropy::forward(
    const Tensor& inputs,
    const Tensor& targets,
    const Tensor& weights) {
  return autograd::mean(autograd::flatten(weights * autograd::binaryCrossEntropy(inputs, targets)), {0});
}

std::unique_ptr<Module> BinaryCrossEntropy::clone() const {
  return std::make_unique<BinaryCrossEntropy>(*this);
}

std::string BinaryCrossEntropy::prettyString() const {
  return "BinaryCrossEntropy";
}

Tensor CategoricalCrossEntropy::forward(
    const Tensor& inputs,
    const Tensor& targets) {
  return autograd::categoricalCrossEntropy(inputs, targets, reduction_, ignoreIndex_);
}

std::unique_ptr<Module> CategoricalCrossEntropy::clone() const {
  return std::make_unique<CategoricalCrossEntropy>(*this);
}

std::string CategoricalCrossEntropy::prettyString() const {
  return "CategoricalCrossEntropy";
}

AdaptiveSoftMaxLoss::AdaptiveSoftMaxLoss(
    std::shared_ptr<AdaptiveSoftMax> activation,
    ReduceMode reduction,
    int ignoreIndex)
    : BinaryModule(),
      activation_(activation),
      reduction_(reduction),
      ignoreIndex_(ignoreIndex) {
  params_ = activation_->params();
}

Tensor AdaptiveSoftMaxLoss::cast(
    const Tensor& input,
    const Shape& outDims,
    const Tensor& indices) {
  if (input.elements() != indices.elements()) {
    throw std::invalid_argument("AdaptiveSoftMaxLoss: input, indices mismatch");
  }
  Tensor output = fl::full(outDims, 0, input.type());
  output(indices) = input.flatten();

  auto inputDims = input.shape();

  auto gradFunc = [indices, inputDims](
                      std::vector<Tensor>& inputs,
                      const Tensor& grad_output) {
    Tensor gradTensor = grad_output(indices);
    auto grad = Tensor(fl::reshape(gradTensor, inputDims), false);
    inputs[0].addGrad(grad);
  };
  return Tensor(output, {input.withoutData()}, gradFunc);
}

Tensor AdaptiveSoftMaxLoss::forward(
    const Tensor& inputs,
    const Tensor& targets) {
  // Row-major: inputs: B x T x N, targets: B x T
  if (inputs.ndim() != 3) {
    throw std::invalid_argument(
        "AdaptiveSoftMaxLoss::forward expects input tensor with "
        "3 dimensions in B x T x N ordering.");
  }
  if (targets.ndim() != 2) {
    throw std::invalid_argument(
        "AdaptiveSoftMaxLoss::forward expects target tensor with "
        "2 dimensions in B x T ordering.");
  }
  if (inputs.dim(0) != targets.dim(0)) {
    throw std::invalid_argument("AdaptiveSoftMaxLoss: batch size mismatch");
  } else if (inputs.dim(1) != targets.dim(1)) {
    throw std::invalid_argument("AdaptiveSoftMaxLoss: length mismatch");
  }

  auto B = inputs.dim(0);
  auto T = inputs.dim(1);
  auto N = inputs.dim(2);
  auto cutoff = activation_->getCutoff();

  auto input = autograd::reshape(inputs, {B * T, N});
  auto target = autograd::reshape(targets, {B * T});

  auto headOutput = autograd::matmul(input, params_[0], MatrixProperty::None, MatrixProperty::Transpose);
  auto headTarget = Tensor(target, false) * (target < cutoff[0]);
  auto res = Tensor(fl::full({B * T}, 0, fl::dtype::f32), true);

  // Tail forward
  for (int i = 0; i < cutoff.size() - 1; i++) {
    auto mask = (target >= cutoff[i]) && (target < cutoff[i + 1]);
    if (!fl::any(mask).scalarSync<char>()) {
      continue;
    }

    auto indicesArray = fl::nonzero(mask);
    headTarget =
        headTarget + (mask * (cutoff[0] + i)).astype(headTarget.type());
    auto tailTarget = target(indicesArray) - cutoff[i];
    auto selectedInput = autograd::embedding(Tensor(indicesArray, false), input);
    auto tailOutput = autograd::matmul(selectedInput, params_[1 + i * 2], MatrixProperty::None, MatrixProperty::Transpose);
    tailOutput = autograd::matmul(tailOutput, params_[2 + i * 2], MatrixProperty::None, MatrixProperty::Transpose);
    auto localLoss = autograd::categoricalCrossEntropy(
        autograd::logSoftmax(tailOutput, 1), tailTarget, ReduceMode::NONE, ignoreIndex_);
    auto castResult = cast(localLoss, res.shape(), indicesArray);
    res = res + castResult;
  }

  // Head forward
  auto headLoss = autograd::categoricalCrossEntropy(
            autograd::logSoftmax(headOutput, 1),
            headTarget,
            ReduceMode::NONE,
            ignoreIndex_);
  res = res + headLoss;

  // Reduce
  if (reduction_ == ReduceMode::NONE) {
    return autograd::reshape(res, targets.shape());
  }
  res = autograd::sum(res, {0});
  if (reduction_ == ReduceMode::MEAN) {
    auto denominator = std::max(
        fl::countNonzero(target != ignoreIndex_).asScalar<unsigned>(), 1u);
    res = res / denominator;
  }
  return res;
}

std::shared_ptr<AdaptiveSoftMax> AdaptiveSoftMaxLoss::getActivation() const {
  return activation_;
};

void AdaptiveSoftMaxLoss::setParams(const Tensor& var, int position) {
  Module::setParams(var, position);
  activation_->setParams(var, position);
}

std::unique_ptr<Module> AdaptiveSoftMaxLoss::clone() const {
  return std::make_unique<AdaptiveSoftMaxLoss>(*this);
}

std::string AdaptiveSoftMaxLoss::prettyString() const {
  auto cutoff = activation_->getCutoff();
  std::string cutoffs;
  for (size_t i = 0; i < cutoff.size(); ++i) {
    if (i > 0) cutoffs += ", ";
    cutoffs += std::to_string(cutoff[i]);
  }
  return std::format("Adaptive Softmax ({})", cutoffs);
}

} // namespace fl
