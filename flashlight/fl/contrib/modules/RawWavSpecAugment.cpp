/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <format>
#include <numeric>
#include <stdexcept>

#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/contrib/modules/RawWavSpecAugment.h"
#include "flashlight/fl/tensor/Index.h"

namespace fl {

RawWavSpecAugment::RawWavSpecAugment(
    int tWarpW,
    int fMaskF,
    int nFMask,
    int tMaskT,
    float tMaskP,
    int nTMask,
    int nMels /* 80 */,
    int lowFreqHz /* 0 */,
    int highFreqHz /* 8000 */,
    int sampleRate /* 16000 */,
    int maxKernelSize /* 20000 */,
    MaskingStrategy mStrategy /* = MaskingStrategy::ZERO */)
    : timeWarpW_(tWarpW),
      freqMaskF_(fMaskF),
      numFreqMask_(nFMask),
      timeMaskT_(tMaskT),
      timeMaskP_(tMaskP),
      numTimeMask_(nTMask),
      maskStrategy_(mStrategy),
      rawWavNMels_(nMels),
      rawWavLowFreqHz_(lowFreqHz),
      rawWavHighFreqHz_(highFreqHz),
      rawWavSampleRate_(sampleRate),
      maxKernelSize_(maxKernelSize) {
  if (numFreqMask_ > 0 && freqMaskF_ <= 0) {
    throw std::invalid_argument("invalid arguments for frequency masking.");
  }
  if (numTimeMask_ > 0 && timeMaskT_ <= 0) {
    throw std::invalid_argument("invalid arguments for time masking.");
  }
  if (numTimeMask_ > 0 && (timeMaskP_ <= 0 || timeMaskP_ > 1.0)) {
    throw std::invalid_argument("invalid arguments for time masking.");
  }
  if (rawWavLowFreqHz_ < 0 || rawWavHighFreqHz_ < 0 ||
      rawWavLowFreqHz_ >= rawWavHighFreqHz_) {
    throw std::invalid_argument(
        "invalid arguments for raw Wav high and low frequencies.");
  }
  if (rawWavNMels_ <= 0) {
    throw std::invalid_argument("invalid arguments for raw Wav nMels.");
  }
  precomputeFilters();
}

void RawWavSpecAugment::precomputeFilters() {
  if (!lowPassFilters_.empty()) {
    return;
  }
  auto mel2hz = [](float mel) {
    return 700.0 * (std::pow(10, (mel / 2595.0)) - 1.0);
  };
  auto hz2mel = [](float hz) { return 2595.0 * std::log10(1.0 + hz / 700.0); };
  float minMel = hz2mel(rawWavLowFreqHz_), maxMel = hz2mel(rawWavHighFreqHz_);
  // nMels intervals and nMels + 1 points
  float delta = (maxMel - minMel) / rawWavNMels_;
  float currentMel = minMel;
  // set transition band as half of lowest bin frequency size (left bin)
  // for lowest frequency set it to half of the right bin
  // cutoff frequency and transmision band are stored from 0 to 0.5 of sampling
  // rate
  std::vector<float> transBandKhz(rawWavNMels_ + 1);
  for (int index = 0; index <= rawWavNMels_; index++) {
    cutoff_.push_back(mel2hz(currentMel) / rawWavSampleRate_);
    currentMel += delta;
    if (index > 0) {
      transBandKhz[index] = cutoff_[index - 1] / 4.;
    }
  }
  transBandKhz[0] = transBandKhz[1];
  ignoredLowPassFilters_ = 0;
  // compute filters for each frequency point, nMel + 1 low pass filters
  for (int fidx = 0; fidx < cutoff_.size(); fidx++) {
    int width = 2. / (1e-6 + transBandKhz[fidx]);
    if (width * 2 + 1 > maxKernelSize_) {
      FL_LOG(fl::LogLevel::INFO)
          << "RawWavSpecAugment raw wave: frequency " << cutoff_[fidx]
          << " will be skipped for eval, too large kernel";
      lowPassFilters_.push_back(nullptr);
      ignoredLowPassFilters_++;
      continue;
    }
    Tensor indexArr = fl::iota({2 * width + 1});
    Tensor blackmanWindow = 0.42 - 0.5 * fl::cos(M_PI * indexArr / width) +
        0.08 * fl::cos(2 * M_PI * indexArr / width);
    Tensor denom = indexArr - width;
    // compute sinc with proper process for index = width
    Tensor kernel = fl::sin(2 * M_PI * cutoff_[fidx] * (indexArr - width));
    kernel(denom != 0) = kernel(denom != 0) / denom(denom != 0);
    kernel(denom == 0) = 2 * M_PI * cutoff_[fidx];
    kernel = kernel * blackmanWindow;
    // normalize kernel
    kernel = kernel / fl::tile(fl::sum(kernel, {0}), {2 * width + 1});
    // create low pass filter
    // Row-major: Conv2D weight is {Cout, Cin/groups, kH, kW}
    // kH = kernelSize operates on dim 2 (H = T).
    // py = SAME for H dimension, px = 0 for W dimension.
    auto filter = std::make_shared<Conv2D>(
        Tensor(fl::reshape(kernel, {1, 1, kernel.dim(0), 1}), false),
        1,
        1,
        0,
        PaddingMode::SAME);
    filter->eval();
    lowPassFilters_.push_back(filter);
  }
  if (ignoredLowPassFilters_ >= lowPassFilters_.size()) {
    throw std::invalid_argument(
        "All low pass filters are ignored, too huge kernel for all frequencies");
  }
}

Tensor RawWavSpecAugment::forward(const Tensor& input) {
  if (input.requiresGrad()) {
    throw std::invalid_argument(
        "input gradient calculation is not supported for RawWavSpecAugment.");
  }
  if (lowPassFilters_.empty()) {
    throw std::invalid_argument("invalid RawWavSpecAugment, filters are empty");
  }

  auto output = Tensor(input, false);
  if (!train_) {
    return output;
  }

  if (input.ndim() != 3) {
    throw std::invalid_argument(
        "RawWavSpecAugment::forward - invalid input shape: "
        "input is expected to be B x C x T");
  }

  // Row-major: input is {B, C, T} (mostly C=1)
  const Shape& inShape = input.shape();
  // Conv2D input: {N, Cin, H, W}. Reshape to {B*C, 1, T, 1} for lowpass filter.
  // N=B*C keeps each waveform contiguous, H=T is where kH filter operates.
  Shape timeView = {inShape[0] * inShape[1], 1, inShape[2], 1};
  for (int i = 0; i < numFreqMask_; ++i) {
    auto low = generateRandomInt(ignoredLowPassFilters_, rawWavNMels_);
    auto high =
        generateRandomInt(low, std::min(rawWavNMels_, low + freqMaskF_) + 1);
    if (high > low) {
      auto inputForFilter = autograd::reshape(output, timeView);
      auto midLowWav = lowPassFilters_[high]->forward(inputForFilter);
      auto lowWav = lowPassFilters_[low]->forward(inputForFilter);
      output = output - autograd::reshape(midLowWav - lowWav, input.shape());
    }
  }

  double replaceVal = (maskStrategy_ == MaskingStrategy::GLOBAL_MEAN)
      ? fl::mean(input).asScalar<double>()
      : 0.0;

  auto& opArr = output.detach();
  auto numTimeSteps = input.dim(2); // T is dim 2 in row-major
  // an upper bound on the time mask
  int T = std::min(timeMaskT_, static_cast<int>(numTimeSteps * timeMaskP_));
  if (T > 0) {
    for (int i = 0; i < numTimeMask_; ++i) {
      auto t = generateRandomInt(0, T);
      auto t0 = generateRandomInt(0, numTimeSteps - t);
      // Mask on dim 2 (T)
      opArr(fl::span, fl::span, fl::range(t0, t0 + t + 1)) = replaceVal;
    }
  }
  return output;
}

int RawWavSpecAugment::generateRandomInt(int low, int high) {
  std::uniform_int_distribution<int> uniformDist(low, high - 1);
  return uniformDist(eng_);
}

std::unique_ptr<Module> RawWavSpecAugment::clone() const {
  return std::make_unique<RawWavSpecAugment>(*this);
}

std::string RawWavSpecAugment::prettyString() const {
  return std::format(
      "RawWavSpecAugment ( W: {}, F: {}, mF: {}, T: {}, p: {}, mT: {}, "
      "rawWavNMels: {}, rawWavLowFreqHz: {}, rawWavHighFreqHz: {}, "
      "rawWavSampleRate: {}, maxKernelSize: {},  )",
      timeWarpW_, freqMaskF_, numFreqMask_, timeMaskT_, timeMaskP_,
      numTimeMask_, rawWavNMels_, rawWavLowFreqHz_, rawWavHighFreqHz_,
      rawWavSampleRate_, maxKernelSize_);
}
} // namespace fl
