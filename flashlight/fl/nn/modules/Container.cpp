/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Container.h"

#include <format>

#include "flashlight/fl/tensor/Tensor.h"

namespace fl {

Container::Container() = default;

void Container::clear() {
  childParamIdx_.clear();
  modules_.clear();
  params_.clear();
}

std::unordered_multimap<int, int> Container::getOrphanedParamsIdxMap() const {
  // The previous module index which has params
  int prevMidx = -1;
  std::unordered_multimap<int, int> orphanedParamsIdxMap;
  for (size_t i = 0; i < params_.size();) {
    auto paramIdx = childParamIdx_.find(i);
    if (paramIdx != childParamIdx_.end()) {
      const auto [midx, pidx] = paramIdx->second;
      prevMidx = midx;
      const auto& mod = modules_.at(midx);
      i += mod->params().size();
    } else {
      orphanedParamsIdxMap.emplace(prevMidx, i);
      ++i;
    }
  }

  return orphanedParamsIdxMap;
}

ModulePtr Container::module(int id) const {
  return modules_[id];
}

std::vector<ModulePtr> Container::modules() const {
  return modules_;
}

void Container::train() {
  train_ = true;

  for (int i = 0; i < params_.size(); ++i) {
    if (!childParamIdx_.contains(i)) {
      params_[i].setRequiresGrad(true);
    }
  }

  for (auto& module : modules_) {
    module->train();
  }
}

void Container::eval() {
  train_ = false;

  for (int i = 0; i < params_.size(); ++i) {
    if (!childParamIdx_.contains(i)) {
      params_[i].setRequiresGrad(false);
    }
  }

  for (auto& module : modules_) {
    module->eval();
  }
}

void Container::setParams(const Tensor& var, int position) {
  Module::setParams(var, position);
  auto indices = childParamIdx_.find(position);
  if (indices != childParamIdx_.end()) {
    int midx, pidx;
    std::tie(midx, pidx) = indices->second;
    modules_[midx]->setParams(var, pidx);
  }
}

std::string Container::prettyString() const {
  std::string result = " [input";
  for (int i = 0; i < static_cast<int>(modules_.size()); ++i) {
    result += std::format(" -> ({})", i);
  }
  result += " -> output]";
  for (int i = 0; i < static_cast<int>(modules_.size()); ++i) {
    result += std::format("\n\t({}): {}", i, modules_[i]->prettyString());
  }
  return result;
}

Sequential::Sequential() = default;

std::vector<Tensor> Sequential::forward(const std::vector<Tensor>& input) {
  auto output = input;
  for (auto& module : modules_) {
    output = module->forward(output);
  }
  return output;
}

Tensor Sequential::forward(const Tensor& input) {
  std::vector<Tensor> output = {input};
  for (auto& module : modules_) {
    output = module->forward(output);
  }
  if (output.size() != 1) {
    throw std::invalid_argument("Module output size is not 1");
  }
  return output.front();
}

Tensor Sequential::operator()(const Tensor& input) {
  return this->forward(input);
}

std::string Sequential::prettyString() const {
  std::string result = "Sequential [input";
  for (int i = 0; i < static_cast<int>(modules_.size()); ++i) {
    result += std::format(" -> ({})", i);
  }
  result += " -> output]";
  for (int i = 0; i < static_cast<int>(modules_.size()); ++i) {
    result += std::format("\n\t({}): {}", i, modules_[i]->prettyString());
  }
  return result;
}

} // namespace fl
