/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <format>
#include <stdexcept>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/contrib/modules/Residual.h"

namespace fl {

std::unordered_set<int> Residual::getProjectionsIndices() const {
  return projectionsIndices_;
}

void Residual::addScale(int beforeLayer, float scale) {
  int nLayers = modules_.size() - projectionsIndices_.size();
  if (beforeLayer < 1 || beforeLayer > nLayers + 1) {
    throw std::invalid_argument(
        "Residual: invalid layer index " + std::to_string(beforeLayer) +
        " before which apply the scaling");
  }
  if (scales_.contains(beforeLayer - 1)) {
    throw std::invalid_argument(
        "Residual: scaling before layer " + std::to_string(beforeLayer) +
        " was already added; adding only once is allowed");
  }
  scales_[beforeLayer - 1] = scale;
}

void Residual::checkShortcut(int fromLayer, int toLayer) {
  int nLayers = modules_.size() - projectionsIndices_.size();

  if (fromLayer < 0 || fromLayer >= nLayers || toLayer <= 0 ||
      toLayer > nLayers + 2 || toLayer - fromLayer <= 1) {
    throw std::invalid_argument(
        "Residual: invalid skip connection; check fromLayer=" +
        std::to_string(fromLayer) + " and toLayer=" + std::to_string(toLayer) +
        " parameters. They are out of range of added layers");
  }
  if (shortcut_.contains(toLayer - 1) &&
      shortcut_[toLayer - 1].contains(fromLayer)) {
    throw std::invalid_argument(
        "Residual: skip connection for fromLayer " + std::to_string(fromLayer) +
        " to toLayer " + std::to_string(toLayer) + " is already added");
  }
}

void Residual::processShortcut(
    int fromLayer,
    int toLayer,
    int projectionIndex) {
  shortcut_[toLayer - 1].insert({fromLayer, projectionIndex});
}

void Residual::addShortcut(int fromLayer, int toLayer) {
  // fromLayer: 0, .., nLayers_ - 1; toLayer: 1, 2, .., nLayers_ + 1
  // toLayer - fromLayer > 1 (avoid adding skip connection
  // from layer K to layer  K+1)
  checkShortcut(fromLayer, toLayer);
  processShortcut(fromLayer, toLayer, -1);
}

Tensor Residual::applyScale(const Tensor& input, const int layerIndex) {
  float scale =
      scales_.contains(layerIndex) ? scales_[layerIndex] : 1.;
  return input * scale;
}

std::vector<Tensor> Residual::forward(const std::vector<Tensor>& inputs) {
  if (inputs.size() != 1) {
    throw std::invalid_argument("Residual module expects only one input");
  }
  return {forward(inputs[0])};
}

Tensor Residual::forward(const Tensor& input) {
  Tensor output = input;
  int nLayers = modules_.size() - projectionsIndices_.size();
  std::vector<Tensor> outputs(nLayers + 1, Tensor());
  outputs[0] = input;

  int moduleIndex = 0, layerIndex = 0;

  while (layerIndex < nLayers) {
    while (projectionsIndices_.contains(moduleIndex)) {
      moduleIndex++;
    }
    if (shortcut_.contains(layerIndex)) {
      for (const auto& shortcut : shortcut_[layerIndex]) {
        Tensor connectionOut = outputs[shortcut.first];
        if (shortcut.second != -1) {
          connectionOut = modules_[shortcut.second]
                              ->forward({outputs[shortcut.first]})
                              .front();
        }
        output = output + connectionOut.astype(output.type());
      }
    }
    output = modules_[moduleIndex]
                 ->forward({applyScale(output, layerIndex)})
                 .front();
    outputs[layerIndex + 1] = output;
    layerIndex++;
    moduleIndex++;
  }
  if (shortcut_.contains(nLayers)) {
    for (const auto& shortcut : shortcut_[nLayers]) {
      Tensor connectionOut = outputs[shortcut.first];
      if (shortcut.second != -1) {
        connectionOut = modules_[shortcut.second]
                            ->forward({outputs[shortcut.first]})
                            .front();
      }
      output = output + connectionOut.astype(output.type());
    }
  }
  return applyScale(output, nLayers);
}

std::string Residual::prettyString() const {
  // prepare inverted residual skip connection
  std::unordered_map<int, std::unordered_map<int, int>>
      reverseShortcut; // start -> end
  for (const auto& shortcut : shortcut_) {
    for (const auto& value : shortcut.second) {
      reverseShortcut[value.first].insert({shortcut.first, value.second});
    }
  }

  int nLayers = modules_.size() - projectionsIndices_.size();
  int moduleIndex = -1, layerIndex = 0;
  std::unordered_map<int, float>::const_iterator scaleIt;
  std::string result;

  while (layerIndex <= nLayers) {
    result += std::format("\n\tRes({}): ", layerIndex);
    if (layerIndex == 0) {
      result += "Input";
    } else {
      while (projectionsIndices_.find(moduleIndex) !=
             projectionsIndices_.end()) {
        moduleIndex++;
      }
      result += modules_[moduleIndex]->prettyString();
    }

    scaleIt = scales_.find(layerIndex);
    if (scaleIt != scales_.end()) {
      result += std::format(
          " with scale (before layer is applied) {};", scaleIt->second);
    }

    if (reverseShortcut.contains(layerIndex) &&
        !reverseShortcut[layerIndex].empty()) {
      result += "; skip connection to ";
      for (auto shortcut : reverseShortcut[layerIndex]) {
        if (shortcut.first < nLayers) {
          result += std::format("layer Res({})", shortcut.first + 1);
        } else {
          result += "output";
        }
        if (shortcut.second != -1) {
          result += std::format(
              " with transformation: {};",
              modules_[shortcut.second]->prettyString());
        }
        result += " ";
      }
    }
    layerIndex++;
    moduleIndex++;
  }
  result += std::format("\n\tRes({}): Output;", nLayers + 1);
  scaleIt = scales_.find(nLayers + 1);
  if (scaleIt != scales_.end()) {
    result += std::format(
        " with scale (before layer is applied) {};", scaleIt->second);
  }

  return result;
}

} // namespace fl
