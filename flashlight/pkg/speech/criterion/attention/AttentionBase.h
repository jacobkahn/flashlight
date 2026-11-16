/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/flashlight.h"

namespace fl {
namespace pkg {
namespace speech {

/**
 * Attention base class for encoder-decoder: decoder attends to particular
 * encoder part
 */
class AttentionBase : public fl::Container {
 public:
  AttentionBase() {}

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    if (inputs.size() != 3 && inputs.size() != 4 && inputs.size() != 5) {
      throw std::invalid_argument(
          "Attention encoder-decoder: Invalid inputs size, should be 3, 4, or 5 arguments");
    }

    auto logAttnWeight = inputs.size() == 4 ? inputs[3] : Tensor();
    auto xEncodedSizes = inputs.size() == 5 ? inputs[4] : Tensor();
    auto res = forwardBase(
        inputs[0], inputs[1], inputs[2], logAttnWeight, xEncodedSizes);
    return {res.first, res.second};
  }

  std::pair<Tensor, Tensor> forward(
      const Tensor& state,
      const Tensor& xEncoded,
      const Tensor& prevAttn) {
    return forward(
        state,
        xEncoded,
        prevAttn,
        Tensor() /* logAttnWeight */,
        Tensor() /* xEncodedSizes */);
  }

  std::pair<Tensor, Tensor> forward(
      const Tensor& state,
      const Tensor& xEncoded,
      const Tensor& prevAttn,
      const Tensor& logAttnWeight) {
    return forwardBase(
        state,
        xEncoded,
        prevAttn,
        logAttnWeight,
        Tensor() /* xEncodedSizes */);
  }

  virtual std::pair<Tensor, Tensor> forward(
      const Tensor& state,
      const Tensor& xEncoded,
      const Tensor& prevAttn,
      const Tensor& logAttnWeight,
      const Tensor& xEncodedSizes) {
    return forwardBase(state, xEncoded, prevAttn, logAttnWeight, xEncodedSizes);
  }

 protected:
  /**
   * Forward pass
   * @param state current decoder state
   * @param xEncoded encoder output = decoder input
   * @param prevAttn previous attention vector
   * @param logAttnWeight attention weights to add: finalAttn =
   * exp(logAttnWeight + logComputedAttn)
   * @param xEncodedSizes encoder output actual sizes has (1, B) dims
   * Returns <attention vector (sum = 1), summary> of sizes
   * [targetlen, seqlen, batchsize] for attention,
   * [hiddendim, targetlen, batchsize] for summary
   */
  virtual std::pair<Tensor, Tensor> forwardBase(
      const Tensor& state,
      const Tensor& xEncoded,
      const Tensor& prevAttn,
      const Tensor& logAttnWeight,
      const Tensor& xEncodedSizes) = 0;

 private:
  FL_SAVE_LOAD_WITH_BASE(fl::Container)
};
} // namespace speech
} // namespace pkg
} // namespace fl

CEREAL_REGISTER_TYPE(fl::pkg::speech::AttentionBase)
