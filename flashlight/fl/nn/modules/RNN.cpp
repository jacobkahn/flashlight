/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/RNN.h"

#include <cmath>
#include <format>
#include <stdexcept>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace fl {

RNN::RNN(
    int input_size,
    int hidden_size,
    int num_layers,
    RnnMode mode,
    bool bidirectional /* false */,
    float drop_prob /* = 0.0 */)
    : inputSize_(input_size),
      hiddenSize_(hidden_size),
      numLayers_(num_layers),
      mode_(mode),
      bidirectional_(bidirectional),
      dropProb_(drop_prob) {
  initialize();
}

RNN::RNN(const RNN& other)
    : Module(other.copyParams()),
      inputSize_(other.inputSize_),
      hiddenSize_(other.hiddenSize_),
      numLayers_(other.numLayers_),
      mode_(other.mode_),
      bidirectional_(other.bidirectional_),
      dropProb_(other.dropProb_) {
  train_ = other.train_;
}

RNN& RNN::operator=(const RNN& other) {
  params_ = other.copyParams();
  train_ = other.train_;
  inputSize_ = other.inputSize_;
  hiddenSize_ = other.hiddenSize_;
  numLayers_ = other.numLayers_;
  mode_ = other.mode_;
  bidirectional_ = other.bidirectional_;
  dropProb_ = other.dropProb_;
  return *this;
}

void RNN::initialize() {
  int64_t n_params = detail::getNumRnnParams(
      inputSize_, hiddenSize_, numLayers_, mode_, bidirectional_);

  double stdv = std::sqrt(1.0 / static_cast<double>(hiddenSize_));
  auto w = uniform({n_params}, -stdv, stdv, fl::dtype::f32, true);
  params_ = {w};
}

std::vector<Tensor> RNN::forward(const std::vector<Tensor>& inputs) {
  if (inputs.empty() || inputs.size() > 3) {
    throw std::invalid_argument("Invalid inputs size");
  }

  const auto& input = inputs[0];
  const auto& hiddenState = inputs.size() >= 2 ? inputs[1] : Tensor();
  const auto& cellState = inputs.size() == 3 ? inputs[2] : Tensor();

  float dropProb = train_ ? dropProb_ : 0.0;
  auto rnnRes =
      autograd::rnn(input,
          hiddenState.astype(input.type()),
          cellState.astype(input.type()),
          params_[0].astype(input.type()),
          hiddenSize_,
          numLayers_,
          mode_,
          bidirectional_,
          dropProb);

  std::vector<Tensor> output(1, std::get<0>(rnnRes));
  if (inputs.size() >= 2) {
    output.push_back(std::get<1>(rnnRes));
  }
  if (inputs.size() == 3) {
    output.push_back(std::get<2>(rnnRes));
  }
  return output;
}

Tensor RNN::forward(const Tensor& input) {
  return forward(std::vector<Tensor>{input}).front();
}

Tensor RNN::operator()(const Tensor& input) {
  return forward(input);
}

std::tuple<Tensor, Tensor> RNN::forward(
    const Tensor& input,
    const Tensor& hidden_state) {
  auto res = forward(std::vector<Tensor>{input, hidden_state});
  return std::make_tuple(res[0], res[1]);
}

std::tuple<Tensor, Tensor> RNN::operator()(
    const Tensor& input,
    const Tensor& hidden_state) {
  return forward(input, hidden_state);
}

std::tuple<Tensor, Tensor, Tensor> RNN::forward(
    const Tensor& input,
    const Tensor& hidden_state,
    const Tensor& cell_state) {
  auto res = forward(std::vector<Tensor>{input, hidden_state, cell_state});
  return std::make_tuple(res[0], res[1], res[2]);
}

std::tuple<Tensor, Tensor, Tensor> RNN::operator()(
    const Tensor& input,
    const Tensor& hidden_state,
    const Tensor& cell_state) {
  return forward(input, hidden_state, cell_state);
}

std::unique_ptr<Module> RNN::clone() const {
  return std::make_unique<RNN>(*this);
}

std::string RNN::prettyString() const {
  const char* name = [&] {
    switch (mode_) {
      case RnnMode::RELU: return "RNN (relu)";
      case RnnMode::TANH: return "RNN (tanh)";
      case RnnMode::LSTM: return "LSTM";
      case RnnMode::GRU:  return "GRU";
      default: return "";
    }
  }();
  int output_size = bidirectional_ ? 2 * hiddenSize_ : hiddenSize_;
  std::string result = std::format("{} ({}->{})", name, inputSize_, output_size);
  if (numLayers_ > 1) {
    result += std::format(" ({}-layer)", numLayers_);
  }
  if (bidirectional_) {
    result += " (bidirectional)";
  }
  if (dropProb_ > 0) {
    result += std::format(" (dropout={})", dropProb_);
  }
  return result;
}

} // namespace fl
