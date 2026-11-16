/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/nn/Transformer.h"

#include <cmath>


using namespace fl;

namespace {

std::shared_ptr<fl::Linear>
makeTransformerLinear(int inDim, int outDim, float gain = 1.0f) {
  int fanIn = inDim;
  int fanOut = outDim;
  float std = gain * std::sqrt(2.0 / (fanIn + fanOut));
  float bound = std::sqrt(3.0) * std;
  auto w = fl::uniform(outDim, inDim, -bound, bound, fl::dtype::f32, true);
  bound = std::sqrt(1.0 / fanIn);
  auto b = fl::uniform({outDim}, -bound, bound, fl::dtype::f32, true);
  return std::make_shared<fl::Linear>(w, b);
}

std::shared_ptr<fl::Linear>
makeMultiheadedAttentionLinear(int inDim, int outDim, int fanOutMult = 1) {
  int fanIn = inDim;
  int fanOut = outDim * fanOutMult;
  float gain = 1.0;
  float std = gain * std::sqrt(2.0 / (fanIn + fanOut));
  float bound = std::sqrt(3.0) * std;
  auto w = fl::uniform(outDim, inDim, -bound, bound, fl::dtype::f32, true);
  auto b = fl::param(fl::full({outDim}, 0));
  return std::make_shared<fl::Linear>(w, b);
}

} // namespace

namespace fl::pkg::vision {

// query [ E X B X L ]
// values and keys [ E X B X S ]
fl::Tensor transformerMultiheadAttention(
    const fl::Tensor& query,
    const fl::Tensor& key,
    const fl::Tensor& value,
    const fl::Tensor& keyPaddingMask,
    const int32_t nHead,
    const double pDropout) {
  int32_t bsz = query.dim(1);
  int32_t modelDim = query.dim(0);
  int32_t headDim = modelDim / nHead;
  int32_t tgtLen = query.dim(2);
  int32_t srcLen = key.dim(2);

  auto q = autograd::reshape(query, {headDim, nHead, bsz, tgtLen});
  auto v = autograd::reshape(value, {headDim, nHead, bsz, srcLen});
  auto k = autograd::reshape(key, {headDim, nHead, bsz, srcLen});
  // Reorder so that the "Sequence" is along the first dimension,
  // the embedding is along the zeroth dimension
  q = autograd::transpose(q, {0, 3, 1, 2});
  v = autograd::transpose(v, {0, 3, 1, 2});
  k = autograd::transpose(k, {0, 3, 1, 2});

  auto scores = autograd::matmul(q, k, MatrixProperty::Transpose);

  if (!keyPaddingMask.isEmpty()) {
    scores = scores +
        autograd::tileAs(autograd::reshape(autograd::log(keyPaddingMask), {1, srcLen, 1, bsz}), scores);
  }

  auto attn = autograd::dropout(autograd::softmax(scores, 1), pDropout);
  auto result = autograd::matmul(attn.astype(v.type()), v, MatrixProperty::None, MatrixProperty::Transpose);
  result = autograd::reshape(result, {tgtLen, modelDim, bsz});
  result = autograd::transpose(result, {1, 2, 0});
  return result;
}

MultiheadAttention::MultiheadAttention(
    int32_t modelDim,
    int32_t headDim,
    int32_t numHeads,
    float pDropout)
    : pDropout_(pDropout), numHeads_(numHeads) {
  wq_ = makeMultiheadedAttentionLinear(modelDim, headDim * numHeads, 3);
  wk_ = makeMultiheadedAttentionLinear(modelDim, headDim * numHeads, 3);
  wv_ = makeMultiheadedAttentionLinear(modelDim, headDim * numHeads, 3);
  wf_ = makeMultiheadedAttentionLinear(headDim * numHeads, modelDim);
  createLayers();
}

MultiheadAttention::MultiheadAttention(const MultiheadAttention& other) {
  copy(other);
  createLayers();
}
MultiheadAttention& MultiheadAttention::operator=(
    const MultiheadAttention& other) {
  clear();
  copy(other);
  createLayers();
  return *this;
}

void MultiheadAttention::copy(const MultiheadAttention& other) {
  train_ = other.train_;
  pDropout_ = other.pDropout_;
  numHeads_ = other.numHeads_;
  wq_ = std::make_shared<Linear>(*other.wq_);
  wk_ = std::make_shared<Linear>(*other.wk_);
  wv_ = std::make_shared<Linear>(*other.wv_);
  wf_ = std::make_shared<Linear>(*other.wf_);
}

void MultiheadAttention::createLayers() {
  add(wq_);
  add(wk_);
  add(wv_);
  add(wf_);
}

std::unique_ptr<Module> MultiheadAttention::clone() const {
  return std::make_unique<MultiheadAttention>(*this);
}

std::vector<Tensor> MultiheadAttention::forward(
    const Tensor& queries,
    const Tensor& keys,
    const Tensor& values,
    const Tensor& keyPaddingMask) {
  assert(queries.dim(0) == keys.dim(0));
  assert(queries.dim(0) == values.dim(0));
  assert(queries.dim(1) == keys.dim(1));
  assert(queries.dim(1) == values.dim(1));
  assert(values.dim(2) == keys.dim(2));

  int32_t modelDim = queries.dim(0);
  int32_t headDim = modelDim / numHeads_;

  if (!keyPaddingMask.isEmpty()) {
    assert(keyPaddingMask.dim(0) == keys.dim(2));
    assert(keyPaddingMask.dim(1) == keys.dim(1));
  }

  auto q = wq_->forward(queries);
  auto k = wk_->forward(keys);
  auto v = wv_->forward(values);

  q = q / std::sqrt(float(headDim));
  float dropout = train_ ? pDropout_ : 0.0f;
  auto result = transformerMultiheadAttention(
      q, k, v, keyPaddingMask, numHeads_, dropout);
  result = (*wf_)(result);

  assert(result.shape() == queries.shape());
  std::vector<Tensor> results = {result};
  return results;
};

std::vector<Tensor> MultiheadAttention::forward(
    const std::vector<Tensor>& input) {
  assert(input.size() == 4);
  return this->forward(input[0], input[1], input[2], input[3]);
}

std::string MultiheadAttention::prettyString() const {
  std::ostringstream ss;
  ss << "MultiheadAttention";
  ss << Container::prettyString();
  return ss.str();
}

TransformerBaseLayer::TransformerBaseLayer(
    int32_t modelDim,
    int32_t mlpDim,
    int32_t nHeads,
    float pDropout)
    : self_attn_(std::make_shared<MultiheadAttention>(
          modelDim,
          modelDim / nHeads,
          nHeads,
          pDropout)),
      w1_(makeTransformerLinear(modelDim, mlpDim)),
      w2_(makeTransformerLinear(mlpDim, modelDim)),
      norm1_(std::make_shared<LayerNorm>(
          std::vector<int>{0},
          1e-5,
          true,
          modelDim)),
      norm2_(std::make_shared<LayerNorm>(
          std::vector<int>{0},
          1e-5,
          true,
          modelDim)),
      pDropout_(pDropout) {
  createLayers();
}

TransformerBaseLayer::TransformerBaseLayer(const TransformerBaseLayer& other) {
  copy(other);
  createLayers();
}
TransformerBaseLayer& TransformerBaseLayer::operator=(
    const TransformerBaseLayer& other) {
  clear();
  copy(other);
  createLayers();
  return *this;
}

void TransformerBaseLayer::copy(const TransformerBaseLayer& other) {
  train_ = other.train_;
  pDropout_ = other.pDropout_;
  self_attn_ = std::make_shared<MultiheadAttention>(*other.self_attn_);
  w1_ = std::make_shared<Linear>(*other.w1_);
  w2_ = std::make_shared<Linear>(*other.w2_);
  norm1_ = std::make_shared<LayerNorm>(*other.norm1_);
  norm2_ = std::make_shared<LayerNorm>(*other.norm2_);
}

void TransformerBaseLayer::createLayers() {
  add(self_attn_);
  add(w1_);
  add(w2_);
  add(norm1_);
  add(norm2_);
};

Tensor TransformerBaseLayer::mlp(const Tensor& in) {
  float pDropout = train_ ? pDropout_ : 0.0;
  return (*w2_)(autograd::dropout(autograd::relu((*w1_)(in)), pDropout));
}

Tensor TransformerBaseLayer::withPosEmbed(
    const Tensor& input,
    const Tensor& pos) {
  if (pos.isEmpty()) {
    return input;
  }
  return input + pos;
}

Tensor TransformerBaseLayer::selfAttention(
    const Tensor& input,
    const Tensor& pos,
    const Tensor& keyPaddingMask) {
  auto k = withPosEmbed(input, pos);
  auto q = withPosEmbed(input, pos);
  return self_attn_->forward(q, k, input, keyPaddingMask)[0];
}

TransformerEncoderLayer::TransformerEncoderLayer(
    int32_t modelDim,
    int32_t mlpDim,
    int32_t nHeads,
    float pDropout)
    : TransformerBaseLayer(modelDim, mlpDim, nHeads, pDropout){};

std::unique_ptr<Module> TransformerEncoderLayer::clone() const {
  return std::make_unique<TransformerEncoderLayer>(*this);
}

std::vector<Tensor> TransformerEncoderLayer::forward(
    const std::vector<Tensor>& input) {
  auto src = input[0];
  auto mask = input[1];
  auto pos = input[2];

  float pDropout = train_ ? pDropout_ : 0.0f;

  auto src2 = this->selfAttention(src, pos, mask);
  src = src + autograd::dropout(src2, pDropout);
  src = (*norm1_)(src);
  src2 = mlp(src);
  src = src + autograd::dropout(src2, pDropout);
  src = (*norm2_)(src);

  return {src, mask, pos};
}

std::string TransformerEncoderLayer::prettyString() const {
  std::ostringstream ss;
  ss << "TransformerEncoderLayer";
  ss << Container::prettyString();
  return ss.str();
}

TransformerDecoderLayer::TransformerDecoderLayer(
    int32_t modelDim,
    int32_t mlpDim,
    int32_t nHeads,
    float pDropout)
    : self_attn_(std::make_shared<MultiheadAttention>(
          modelDim,
          modelDim / nHeads,
          nHeads,
          pDropout)),
      encoder_attn_(std::make_shared<MultiheadAttention>(
          modelDim,
          modelDim / nHeads,
          nHeads,
          pDropout)),
      w1_(makeTransformerLinear(modelDim, mlpDim)),
      w2_(makeTransformerLinear(mlpDim, modelDim)),
      norm1_(std::make_shared<LayerNorm>(
          std::vector<int>{0},
          1e-5,
          true,
          modelDim)),
      norm2_(std::make_shared<LayerNorm>(
          std::vector<int>{0},
          1e-5,
          true,
          modelDim)),
      norm3_(std::make_shared<LayerNorm>(
          std::vector<int>{0},
          1e-5,
          true,
          modelDim)),
      pDropout_(pDropout) {
  createLayers();
}

TransformerDecoderLayer::TransformerDecoderLayer(
    const TransformerDecoderLayer& other) {
  copy(other);
  createLayers();
}
TransformerDecoderLayer& TransformerDecoderLayer::operator=(
    const TransformerDecoderLayer& other) {
  clear();
  copy(other);
  createLayers();
  return *this;
}

void TransformerDecoderLayer::copy(const TransformerDecoderLayer& other) {
  train_ = other.train_;
  pDropout_ = other.pDropout_;
  self_attn_ = std::make_shared<MultiheadAttention>(*other.self_attn_);
  encoder_attn_ = std::make_shared<MultiheadAttention>(*other.encoder_attn_);
  w1_ = std::make_shared<Linear>(*other.w1_);
  w2_ = std::make_shared<Linear>(*other.w2_);
  norm1_ = std::make_shared<LayerNorm>(*other.norm1_);
  norm2_ = std::make_shared<LayerNorm>(*other.norm2_);
  norm3_ = std::make_shared<LayerNorm>(*other.norm3_);
}

void TransformerDecoderLayer::createLayers() {
  add(self_attn_);
  add(encoder_attn_);
  add(w1_);
  add(w2_);
  add(norm1_);
  add(norm2_);
  add(norm3_);
}

std::unique_ptr<Module> TransformerDecoderLayer::clone() const {
  return std::make_unique<TransformerDecoderLayer>(*this);
}

Tensor TransformerDecoderLayer::mlp(const Tensor& in) {
  float pDropout = train_ ? pDropout_ : 0.0;
  return (*w2_)(autograd::dropout(autograd::relu((*w1_)(in)), pDropout));
}

Tensor TransformerDecoderLayer::withPosEmbed(
    const Tensor& input,
    const Tensor& pos) {
  if (pos.isEmpty()) {
    return input;
  }
  return input + pos;
}

Tensor TransformerDecoderLayer::selfAttention(
    const Tensor& input,
    const Tensor& pos,
    const Tensor& keyPaddingMask /* = Tensor() */) {
  auto k = withPosEmbed(input, pos);
  auto q = withPosEmbed(input, pos);
  return self_attn_->forward(q, k, input, keyPaddingMask)[0];
}

std::vector<Tensor> TransformerDecoderLayer::forward(
    const std::vector<Tensor>& input) {
  auto tgt = input[0];
  auto memory = input[1];
  auto pos = (input.size() > 2) ? input[2] : Tensor();
  auto queryPos = (input.size() > 3) ? input[3] : Tensor();
  auto memoryKeyPaddingMask = (input.size() > 4) ? input[4] : Tensor();

  float pDropout = train_ ? pDropout_ : 0.0f;

  auto tgt2 = this->selfAttention(tgt, queryPos);
  tgt = tgt + autograd::dropout(tgt2, pDropout);
  tgt = (*norm1_)(tgt);
  tgt2 = encoder_attn_->forward({
      this->withPosEmbed(tgt, queryPos), // queries
      this->withPosEmbed(memory, pos), // keys
      memory, // values
      memoryKeyPaddingMask // mask
  })[0];
  tgt = tgt + autograd::dropout(tgt2, pDropout);
  tgt = (*norm2_)(tgt);
  tgt2 = mlp(tgt);
  tgt = tgt + autograd::dropout(tgt2, pDropout);
  tgt = (*norm3_)(tgt);
  return {tgt};
}

std::string TransformerDecoderLayer::prettyString() const {
  std::ostringstream ss;
  ss << "TransformerDecoderLayer";
  ss << Container::prettyString();
  return ss.str();
}

TransformerDecoder::TransformerDecoder(
    int32_t modelDim,
    int32_t mlpDim,
    int32_t nHeads,
    int32_t layers,
    float pDropout) {
  // TODO add norm
  for (int i = 0; i < layers; i++) {
    add(TransformerDecoderLayer(modelDim, mlpDim, nHeads, pDropout));
  }
  add(LayerNorm(std::vector<int>{0}, 1e-5, true, modelDim));
}

std::vector<Tensor> TransformerDecoder::forward(
    const std::vector<Tensor>& input) {
  auto tgt = input[0];
  auto memory = input[1];
  auto pos = (input.size() > 2) ? input[2] : Tensor();
  auto query_pos = (input.size() > 3) ? input[3] : Tensor();
  auto mask = (input.size() > 4) ? input[4] : Tensor();

  fl::Tensor output = tgt;
  auto mods = modules();

  std::vector<Tensor> intermediate;
  for (int i = 0; i < mods.size() - 1; i++) {
    output = mods[i]->forward({output, memory, pos, query_pos, mask})[0];
    intermediate.push_back(
        autograd::reshape(mods.back()->forward({output})[0], {0, 0, 0, 1}));
  }
  return {autograd::concatenate(intermediate, 3)};
}
std::string TransformerDecoder::prettyString() const {
  std::ostringstream ss;
  ss << "TransformerDecoder";
  ss << Container::prettyString();
  return ss.str();
}

TransformerEncoder::TransformerEncoder(
    int32_t modelDim,
    int32_t mlpDim,
    int32_t nHeads,
    int32_t layers,
    float pDropout) {
  for (int i = 0; i < layers; i++) {
    add(TransformerEncoderLayer(modelDim, mlpDim, nHeads, pDropout));
  }
}

std::vector<Tensor> TransformerEncoder::forward(
    const std::vector<Tensor>& input) {
  std::vector<Tensor> output = input;
  auto mods = modules();
  for (int i = 0; i < mods.size(); i++) {
    output = mods[i]->forward(output);
  }
  return output;
}

std::string TransformerEncoder::prettyString() const {
  std::ostringstream ss;
  ss << "TransformerDecoder";
  ss << Container::prettyString();
  return ss.str();
}

Transformer::Transformer(
    int32_t modelDim,
    int32_t numHeads,
    int32_t numEncoderLayers,
    int32_t numDecoderLayers,
    int32_t mlpDim,
    float pDropout)
    : encoder_(std::make_shared<TransformerEncoder>(
          modelDim,
          mlpDim,
          numHeads,
          numEncoderLayers,
          pDropout)),
      decoder_(std::make_shared<TransformerDecoder>(
          modelDim,
          mlpDim,
          numHeads,
          numDecoderLayers,
          pDropout)) {
  createLayers();
}

Transformer::Transformer(const Transformer& other) {
  copy(other);
  createLayers();
}

Transformer& Transformer::operator=(const Transformer& other) {
  clear();
  copy(other);
  createLayers();
  return *this;
}

void Transformer::copy(const Transformer& other) {
  train_ = other.train_;
  encoder_ = std::make_shared<TransformerEncoder>(*other.encoder_);
  decoder_ = std::make_shared<TransformerDecoder>(*other.decoder_);
}

void Transformer::createLayers() {
  add(encoder_);
  add(decoder_);
};

std::unique_ptr<Module> Transformer::clone() const {
  return std::make_unique<Transformer>(*this);
}

std::vector<Tensor> Transformer::forward(
    Tensor src,
    Tensor mask,
    Tensor queryEmbed,
    Tensor posEmbed) {
  if (src.ndim() != 4) {
    throw std::invalid_argument(
        "vision::Transformer::forward - "
        "expect src to be of shape (W, H, C, B).");
  }
  assert(src.dim(2) == queryEmbed.dim(0));

  int B = src.dim(3);
  // Reshape from [ W X H X C X B ] to [ WH X C X B ]

  src = flatten(src, 0, 1);
  // Flatten to C x B x WH x 1
  src = autograd::transpose(src, {1, 2, 0, 3});
  // Squeeze to C x B x WH
  src = autograd::reshape(src, {0, 0, 0});

  posEmbed = flatten(posEmbed, 0, 1);
  posEmbed = autograd::transpose(posEmbed, {1, 2, 0, 3});
  posEmbed = autograd::reshape(posEmbed, {0, 0, 0});

  mask = flatten(mask, 0, 2);

  // Tile object queries for each batch
  Shape unsqueeze = {queryEmbed.dim(0), 1, queryEmbed.dim(1)};
  queryEmbed = autograd::reshape(queryEmbed, unsqueeze);
  queryEmbed = autograd::tile(queryEmbed, {1, B, 1});
  assert(queryEmbed.dim(1) == src.dim(1));
  assert(queryEmbed.dim(0) == src.dim(0));

  auto tgt = fl::Tensor(fl::full(queryEmbed.shape(), 0, src.type()), false);

  auto memory = encoder_->forward({src, mask, posEmbed});
  auto hs = decoder_->forward({tgt, memory[0], posEmbed, queryEmbed, mask})[0];

  auto reordered = autograd::transpose(hs, {0, 2, 1, 3});
  return {reordered};
}

std::vector<Tensor> Transformer::forward(const std::vector<Tensor>& input) {
  assert(input.size() > 3);
  auto src = input[0];
  auto mask = (input.size() > 1) ? input[1] : Tensor();
  auto query_embed = (input.size() > 2) ? input[2] : Tensor();
  auto pos_embed = (input.size() > 3) ? input[3] : Tensor();
  return forward(src, mask, query_embed, pos_embed);
}

std::string Transformer::prettyString() const {
  std::ostringstream ss;
  ss << "Transformer";
  ss << Container::prettyString();
  return ss.str();
}

} // namespace fl
