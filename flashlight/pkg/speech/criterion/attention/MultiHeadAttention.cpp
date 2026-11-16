/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/attention/MultiHeadAttention.h"

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/pkg/speech/criterion/attention/Utils.h"

#include <cmath>
#include <stdexcept>

namespace fl::pkg::speech {

MultiHeadContentAttention::MultiHeadContentAttention(
    int dim,
    int numHeads /* = 8 */,
    bool keyValue /* = false */,
    bool splitInput /* = false */)
    : numHeads_(numHeads), keyValue_(keyValue), splitInput_(splitInput) {
  if (splitInput && dim % numHeads != 0) {
    throw std::invalid_argument("Invalid dimensions");
  }

  if (!splitInput) {
    add(Linear(dim, dim)); // query
    add(Linear(dim, dim)); // key
    add(Linear(dim, dim)); // value
  }
  add(Linear(dim, dim));
}

std::unique_ptr<Module> MultiHeadContentAttention::clone() const {
  throw std::runtime_error(
      "Cloning is unimplemented in Module 'MultiHeadContentAttention'");
}

std::pair<Tensor, Tensor> MultiHeadContentAttention::forwardBase(
    const Tensor& state,
    const Tensor& xEncoded,
    const Tensor& /* unused */,
    const Tensor& logAttnWeight,
    const Tensor& xEncodedSizes) {
  if (state.ndim() != 3) {
    throw std::invalid_argument(
        "MultiHeadContentAttention::forwardBase: "
        "state input must be of shape {H, U, B}");
  }
  int hEncode = xEncoded.dim(0);
  int T = xEncoded.dim(1);
  int hState = state.dim(0);
  int U = state.dim(1);
  int B = state.dim(2);
  auto hiddenDim = hState / numHeads_;
  if (hEncode != (1 + keyValue_) * hState) {
    throw std::invalid_argument("Invalid input encoder dimension");
  }

  auto xEncodedKey = keyValue_
      ? xEncoded(fl::arange(0, hEncode / 2), fl::span, fl::span)
      : xEncoded;
  auto xEncodedValue = keyValue_
      ? xEncoded(fl::arange(hEncode / 2, hEncode), fl::span, fl::span)
      : xEncoded;

  auto query = splitInput_ ? state : module(0)->forward({state})[0];
  auto key = splitInput_ ? xEncodedKey : module(1)->forward({xEncodedKey})[0];
  auto value =
      splitInput_ ? xEncodedValue : module(2)->forward({xEncodedValue})[0];

  query =
      autograd::reshape(autograd::transpose(query, {1, 0, 2}), {U, hiddenDim, B * numHeads_});
  key = autograd::reshape(autograd::transpose(key, {1, 0, 2}), {T, hiddenDim, B * numHeads_});
  value =
      autograd::reshape(autograd::transpose(value, {1, 0, 2}), {T, hiddenDim, B * numHeads_});

  // [U, T, B * numHeads_]
  auto innerProd =
      autograd::matmul(query, key, MatrixProperty::None, MatrixProperty::Transpose) / std::sqrt(static_cast<float>(hiddenDim));

  if (!logAttnWeight.isEmpty()) {
    auto tiledLogAttnWeight = autograd::tile(logAttnWeight, {1, 1, numHeads_});
    if (tiledLogAttnWeight.shape() != innerProd.shape()) {
      throw std::invalid_argument(
          "MultiHeadContentAttention: logAttnWeight has wong dimentions");
    }
    innerProd = innerProd + tiledLogAttnWeight;
  }

  if (!xEncodedSizes.isEmpty()) {
    innerProd = maskAttention(
        innerProd,
        autograd::reshape(autograd::tile(xEncodedSizes, {numHeads_, 1}), {1, B * numHeads_}));
  }

  // [U, T, B * numHeads_]
  auto attention = autograd::softmax(innerProd, 1);
  // [U, hiddendim, B * numHeads_]
  auto summaries = autograd::matmul(attention, value);
  // [hiddendim * numHeads_, U, B];
  summaries = autograd::transpose(autograd::reshape(summaries, {U, hState, B}), {1, 0, 2});

  auto out_summaries = modules().back()->forward({summaries}).front();

  // [U * numHeads_, T, B]
  attention = autograd::reshape(
      autograd::transpose(autograd::reshape(attention, {U, T, numHeads_, B}), {0, 2, 1, 3}),
      {U * numHeads_, T, B});
  return std::make_pair(attention, out_summaries);
}

std::string MultiHeadContentAttention::prettyString() const {
  return "MultiHeadContentAttention";
}
} // namespace fl
