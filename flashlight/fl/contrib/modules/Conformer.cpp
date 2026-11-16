/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/Conformer.h"

#include <cmath>
#include <format>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl {

Conformer::Conformer(
    int32_t modelDim,
    int32_t headDim,
    int32_t mlpDim,
    int32_t nHeads,
    int32_t posEmbContextSize,
    int32_t convKernelSize,
    float pDropout,
    float pLayerDropout /* = 0. */)
    : nHeads_(nHeads),
      posEmbContextSize_(posEmbContextSize),
      convKernelSize_(convKernelSize),
      pDropout_(pDropout),
      pLayerDropout_(pLayerDropout),
      w11_(std::make_shared<Linear>(conformerInitLinear(modelDim, mlpDim))),
      w12_(std::make_shared<Linear>(conformerInitLinear(mlpDim, modelDim))),
      w21_(std::make_shared<Linear>(conformerInitLinear(modelDim, mlpDim))),
      w22_(std::make_shared<Linear>(conformerInitLinear(mlpDim, modelDim))),
      wq_(std::make_shared<Linear>(
          conformerInitLinear(modelDim, headDim * nHeads))),
      wk_(std::make_shared<Linear>(
          conformerInitLinear(modelDim, headDim * nHeads))),
      wv_(std::make_shared<Linear>(
          conformerInitLinear(modelDim, headDim * nHeads))),
      wf_(std::make_shared<Linear>(
          conformerInitLinear(headDim * nHeads, modelDim))),
      conv1_(std::make_shared<Linear>(
          conformerInitLinear(modelDim, modelDim * 2))),
      conv2_(std::make_shared<Linear>(conformerInitLinear(modelDim, modelDim))),
      // Row-major: input is {B, T, C}, normalize over C (axis 2)
      norm1_(std::make_shared<LayerNorm>(
          std::vector<int>({2}),
          1e-5,
          true,
          modelDim)),
      norm2_(std::make_shared<LayerNorm>(
          std::vector<int>({2}),
          1e-5,
          true,
          modelDim)),
      normMhsa_(std::make_shared<LayerNorm>(
          std::vector<int>({2}),
          1e-5,
          true,
          modelDim)),
      normConv1_(std::make_shared<LayerNorm>(
          std::vector<int>({2}),
          1e-5,
          true,
          modelDim)),
      normConv2_(std::make_shared<LayerNorm>(
          std::vector<int>({2}),
          1e-5,
          true,
          modelDim)),
      norm3_(std::make_shared<LayerNorm>(
          std::vector<int>({2}),
          1e-5,
          true,
          modelDim)),
      convDepthWise_(std::make_shared<Conv2D>(
          modelDim,
          modelDim,
          convKernelSize,
          1,
          1,
          1,
          fl::PaddingMode::SAME,
          0,
          1,
          1,
          true,
          modelDim)) {
  if (posEmbContextSize_ > 0) {
    params_.push_back(uniform(2 * posEmbContextSize_ - 1, headDim, -0.1, 0.1));
  }
  createLayers();
}

Conformer::Conformer(const Conformer& other) {
  copy(other);
  createLayers();
}

Conformer& Conformer::operator=(const Conformer& other) {
  clear();
  copy(other);
  createLayers();
  return *this;
}

void Conformer::copy(const Conformer& other) {
  train_ = other.train_;
  nHeads_ = other.nHeads_;
  posEmbContextSize_ = other.posEmbContextSize_;
  convKernelSize_ = other.convKernelSize_;
  pDropout_ = other.pDropout_;
  pLayerDropout_ = other.pLayerDropout_;
  w11_ = std::make_shared<Linear>(*other.w11_);
  w12_ = std::make_shared<Linear>(*other.w12_);
  w21_ = std::make_shared<Linear>(*other.w21_);
  w22_ = std::make_shared<Linear>(*other.w22_);
  wq_ = std::make_shared<Linear>(*other.wq_);
  wk_ = std::make_shared<Linear>(*other.wk_);
  wv_ = std::make_shared<Linear>(*other.wv_);
  wf_ = std::make_shared<Linear>(*other.wf_);
  conv1_ = std::make_shared<Linear>(*other.conv1_);
  conv2_ = std::make_shared<Linear>(*other.conv2_);
  norm1_ = std::make_shared<LayerNorm>(*other.norm1_);
  norm2_ = std::make_shared<LayerNorm>(*other.norm2_);
  normMhsa_ = std::make_shared<LayerNorm>(*other.normMhsa_);
  normConv1_ = std::make_shared<LayerNorm>(*other.normConv1_);
  normConv2_ = std::make_shared<LayerNorm>(*other.normConv2_);
  norm3_ = std::make_shared<LayerNorm>(*other.norm3_);
  convDepthWise_ = std::make_shared<Conv2D>(*other.convDepthWise_);
  if (posEmbContextSize_ > 0) {
    const auto& p = other.param(0);
    params_.emplace_back(p.copy());
  }
}

void Conformer::createLayers() {
  // first feed-forward module
  add(w11_);
  add(w12_);
  add(norm1_);
  // second feed-forward module
  add(w21_);
  add(w22_);
  add(norm2_);
  // multihead attention module
  add(wq_);
  add(wk_);
  add(wv_);
  add(wf_);
  add(normMhsa_);
  // conv module
  add(conv1_);
  add(conv2_);
  add(convDepthWise_);
  add(normConv1_);
  add(normConv2_);
  // final layer norm of conformer block
  add(norm3_);
}

Tensor Conformer::conformerInitLinear(int32_t inDim, int32_t outDim) {
  float std = std::sqrt(1.0 / float(inDim));
  return fl::uniform(outDim, inDim, -std, std);
}

Tensor Conformer::mhsa(const Tensor& input, const Tensor& inputPadMask) {
  float pDropout = train_ ? pDropout_ : 0.0;
  // Row-major: input is {B, T, C}
  int bsz = input.dim(0);

  auto normedInput = (*normMhsa_)(input);
  // Linear projects on last dim (C). multiheadAttention expects {B, T, modelDim}
  auto q = (*wq_)(normedInput);
  auto k = (*wk_)(normedInput);
  auto v = (*wv_)(normedInput);

  Tensor mask, posEmb;
  if (posEmbContextSize_ > 0) {
    // params_[0]: {2*posEmbContextSize-1, headDim} = {L, hD}
    // Need: {nH*bsz, L, hD}
    auto pe = autograd::unsqueeze(params_[0].astype(input.type()), 0);
    posEmb = autograd::tile(pe, {nHeads_ * bsz, 1, 1});
  }

  // padMask: {B, T} in row-major
  fl::Tensor padMask;
  if (!inputPadMask.isEmpty()) {
    auto padMaskArr = inputPadMask;
    Shape newMaskShape = {input.dim(0), input.dim(1)};
    if (padMaskArr.elements() != newMaskShape.elements()) {
      throw std::runtime_error(
          "Conformer::mhsa - pad mask requires resize. "
          "This behavior will be fixed in a future release ");
    }
    padMaskArr = fl::reshape(padMaskArr, newMaskShape);
    padMask = fl::Tensor(fl::log(padMaskArr), false);
  }

  auto result =
      autograd::multiheadAttention(q, k, v, posEmb, mask, padMask, nHeads_, pDropout, 0);
  // multiheadAttention returns {B, T, modelDim}. Linear on last dim.
  result = (*wf_)(result);
  result = autograd::dropout(result, pDropout);
  return result;
}

Tensor Conformer::conv(const Tensor& _input) {
  // Row-major: _input is {B, T, C}
  float pDropout = train_ ? pDropout_ : 0.0;

  // normConv1_ normalizes over C (axis 2), conv1_ operates on last dim (C)
  auto result = autograd::gatedlinearunit(
      (*conv1_)(((*normConv1_)(_input)).astype(_input.type())), 2);
  // result: {B, T, C}

  // Reshape for Conv2D: {B, T, C} --> {B, T, C, 1} --> {B, C, 1, T} = {N, C, H, W}
  // wx=convKernelSize-->kW operates on dim 3 (T)
  result = autograd::unsqueeze(result, 3);
  result = autograd::transpose(result, {0, 2, 3, 1});
  // apply depthwise separable convolution
  result = (*convDepthWise_)(result);
  // {B, C, 1, T} --> {B, T, C, 1} --> {B, T, C}
  result = autograd::transpose(result, {0, 3, 1, 2});
  result = autograd::squeeze(result, 3);

  result = autograd::swish(((*normConv2_)(result)).astype(_input.type()), 1.);
  // apply second pointwise conv
  result = autograd::dropout((*conv2_)(result), pDropout);
  return result;
}

std::vector<Tensor> Conformer::forward(const std::vector<Tensor>& input) {
  if (input.size() != 2) {
    throw std::invalid_argument(
        "Invalid inputs for conformer block: there should be input "
        "and paddding mask (can be empty Tensor)");
  }

  auto x = input[0];

  if (x.ndim() != 3) {
    throw std::invalid_argument(
        "Conformer::forward - input should be of 3 dimensions "
        "expects an input of size B x T x C - see documentation.");
  }

  float pDropout = train_ ? pDropout_ : 0.0;
  float f = 1.0;
  if (train_ && (fl::rand({1}).scalarSync<float>() < pLayerDropout_)) {
    f = 0.0;
  }
  // apply first feed-forward module
  auto ffn1 = autograd::dropout(
      (*w12_)(autograd::dropout(
          autograd::swish((*w11_)(((*norm1_)(x)).astype(x.type())), 1.), pDropout)),
      pDropout);
  x = x + f * 0.5 * ffn1;
  // apply multihead attention module
  x = x + f * mhsa(x, input[1]);
  // apply conv module
  x = x + f * conv(x);
  // apply second feed-forward module
  auto ffn2 = autograd::dropout(
      (*w22_)(autograd::dropout(
          autograd::swish((*w21_)(((*norm2_)(x)).astype(x.type())), 1.), pDropout)),
      pDropout);
  x = x + f * 0.5 * ffn2;
  x = ((*norm3_)(x)).astype(x.type());
  return {x};
}

std::unique_ptr<Module> Conformer::clone() const {
  return std::make_unique<Conformer>(*this);
}

std::string Conformer::prettyString() const {
  return std::format(
      "Conformer (modelDim: {}), (mlpDim: {}), (nHeads: {}), "
      "(pDropout: {}), (pLayerDropout: {}), "
      "(posEmbContextSize: {}), (convKernelSize: {}) ",
      params_[1].dim(1), params_[1].dim(0), nHeads_,
      pDropout_, pLayerDropout_, posEmbContextSize_, convKernelSize_);
}

} // namespace fl
