/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/attention/Utils.h"
#include "flashlight/pkg/speech/criterion/attention/Defines.h"

#include "flashlight/fl/flashlight.h"
#include "flashlight/fl/tensor/Index.h"

namespace fl::pkg::speech {

Tensor maskAttention(const Tensor& input, const Tensor& sizes) {
  int B = input.dim(2);
  int T = input.dim(1);
  // xEncodedSizes is (1, B) size
  Tensor inputNotPaddedSize =
      fl::ceil(sizes / fl::amax(sizes).asScalar<float>() * T);
  Tensor padMask =
      fl::iota({T, 1}, {1, B}) >= fl::tile(inputNotPaddedSize, {T, 1});
  padMask = fl::tile(fl::reshape(padMask, {1, T, B}), {input.dim(0), 1, 1});

  Tensor output = input;
  output(padMask) = kAttentionMaskValue;

  auto gradFunc =
      [padMask](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
        Tensor gradArray = gradOutput;
        gradArray(padMask) = 0.;
        inputs[0].addGrad(Tensor(gradArray, false));
      };
  return Tensor(output, {input.withoutData()}, gradFunc);
}

} // namespace fl
