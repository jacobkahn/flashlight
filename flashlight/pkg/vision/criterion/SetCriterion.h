/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>
#include <utility>

#include "flashlight/pkg/vision/criterion/Hungarian.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl {
namespace pkg {
namespace vision {

class SetCriterion {
 public:
  using LossDict = std::unordered_map<std::string, Tensor>;

  SetCriterion(
      const int numClasses,
      const HungarianMatcher& matcher,
      const std::unordered_map<std::string, float>& weightDict,
      const float eosCoef);

  std::vector<Tensor> match(
      const Tensor& predBoxes,
      const Tensor& predLogits,
      const std::vector<Tensor>& targetBoxes,
      const std::vector<Tensor>& targetClasses);

  LossDict lossLabels(
      const Tensor& predBoxes,
      const Tensor& predLogits,
      const std::vector<Tensor>& targetBoxes,
      const std::vector<Tensor>& targetClasses,
      const std::vector<std::pair<Tensor, Tensor>>& indices,
      const int numBoxes);

  LossDict lossCardinality(
      const Tensor& predBoxes,
      const Tensor& predLogits,
      const std::vector<Tensor>& targetBoxes,
      const std::vector<Tensor>& targetClasses);

  LossDict lossBoxes(
      const Tensor& predBoxes,
      const Tensor& predLogits,
      const std::vector<Tensor>& targetBoxes,
      const std::vector<Tensor>& targetClasses,
      const std::vector<std::pair<Tensor, Tensor>>& indices,
      const int numBoxes);

  LossDict lossMasks(
      const Tensor& predBoxes,
      const Tensor& predLogits,
      const std::vector<Tensor>& targetBoxes,
      const std::vector<Tensor>& targetClasses);

  LossDict forward(
      const Tensor& predBoxes,
      const Tensor& predLogits,
      const std::vector<Tensor>& targetBoxes,
      const std::vector<Tensor>& targetClasses);

  std::unordered_map<std::string, float> getWeightDict();

 private:
  std::pair<Tensor, Tensor> getSrcPermutationIdx(
      const std::vector<std::pair<Tensor, Tensor>>& indices);

  std::pair<Tensor, Tensor> getTgtPermutationIdx(
      const std::vector<std::pair<Tensor, Tensor>>& indices);

  const int numClasses_;
  const HungarianMatcher matcher_;
  const std::unordered_map<std::string, float> weightDict_;
  const float eosCoef_;
};

} // namespace vision
} // namespace pkg
} // namespace fl
