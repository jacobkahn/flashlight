/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/modules.h"
#include "flashlight/fl/flashlight.h"

namespace slimIPL {
class ctc_letters_st3_ls100h_baseline : public fl::Container {
 public:
  ctc_letters_st3_ls100h_baseline(int64_t nFeature, int64_t nLabel)
      : convFrontend_(std::make_shared<fl::Sequential>()) {
    convFrontend_->add(
        std::make_shared<fl::View>(af::dim4(-1, 1, nFeature, 0)));
    // Time x 1 x nFeature x Batch
    std::vector<int> lnDims = {0, 1, 2};
    convFrontend_->add(std::make_shared<fl::LayerNorm>(lnDims));
    convFrontend_->add(
        std::make_shared<fl::Conv2D>(nFeature, 1536, 7, 1, 3, 1, -1, 0, 1, 1));
    convFrontend_->add(std::make_shared<fl::GatedLinearUnit>(2));
    convFrontend_->add(std::make_shared<fl::Dropout>(0.3));
    convFrontend_->add(std::make_shared<fl::Reorder>(2, 0, 3, 1));
    // nFeature x Time x Batch x 1
    add(convFrontend_);
    for (int trIdx = 0; trIdx < 36; trIdx++) {
      auto layer = std::make_shared<fl::Transformer>(
          768, 192, 3072, 4, 920, 0.3, 0.3, false, false);
      transformers_.push_back(layer);
      add(layer);
    }
    linear_ = std::make_shared<fl::Linear>(768, nLabel);
    add(linear_);
  }

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& input) override {
    auto out = input[0];
    auto xSizes = input[1].array();
    // expected input dims T x C x 1 x B
    int T = out.dims(0), B = out.dims(3);
    auto inputMaxSize = af::tile(af::max(xSizes), 1, B);
    af::array inputNotPaddedSize = af::ceil(xSizes * T / inputMaxSize);
    auto padMask = af::iota(af::dim4(T, 1), af::dim4(1, B)) <
        af::tile(inputNotPaddedSize, T, 1);
    out = convFrontend_->forward(out);
    for (int trIdx = 0; trIdx < 36; trIdx++) {
      out = transformers_[trIdx]->forward({out, fl::noGrad(padMask)}).front();
    }
    out = linear_->forward(out);
    return {out};
  }

  std::string prettyString() const {
    std::ostringstream ss;
    ss << "Model ctc_letters_st3_ls100h_baseline: ";
    for (int trIdx = 0; trIdx < 36; trIdx++) {
      ss << transformers_[trIdx]->prettyString() << "\n";
    }
    ss << linear_->prettyString() << "\n";
  }

 private:
  std::shared_ptr<fl::Sequential> convFrontend_;
  std::vector<std::shared_ptr<fl::Transformer>> transformers_;
  std::shared_ptr<fl::Linear> linear_;

  FL_SAVE_LOAD_WITH_BASE(fl::Container, convFrontend_, transformers_, linear_)

  ctc_letters_st3_ls100h_baseline();
};
} // namespace slimIPL

extern "C" fl::Module* createModule(int64_t nFeature, int64_t nLabel) {
  return new slimIPL::ctc_letters_st3_ls100h_baseline(nFeature, nLabel);
}

CEREAL_REGISTER_TYPE(slimIPL::ctc_letters_st3_ls100h_baseline);
