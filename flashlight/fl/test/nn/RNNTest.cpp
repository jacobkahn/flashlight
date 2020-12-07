/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/common.h"
#include "flashlight/fl/nn/nn.h"

using namespace fl;

TEST(ModuleTest, RNNFwd) {
  auto mode = RnnMode::RELU;
  int num_layers = 2;
  int hidden_size = 2;
  int input_size = 2;
  int batch_size = 1;
  int seq_length = 1;

  auto in = Variable(
      af::randu(input_size, batch_size, seq_length, af::dtype::f32), true);

  size_t n_params = 24;
  auto w = Variable(af::randu(1, 1, n_params, af::dtype::f32), true);

  for (int i = 0; i < in.elements(); ++i) {
    in.array()(i) = (i + 1);
  }
  for (int i = 0; i < w.elements(); ++i) {
    w.array()(i) = (i + 1);
    // w.array()(i) =
    //     i > n_params - 2 * hidden_size * num_layers - 1 ? 0 : (i + 1);
  }

  af::print("in", in.array());
  af::print("w", w.array());

  auto rnn = RNN(input_size, hidden_size, num_layers, mode);
  rnn.setParams(w, 0);

  auto out = rnn(in);

  af::print("out", out.array());

  af::dim4 expected_dims(3, 5, 6);
  ASSERT_EQ(out.dims(), expected_dims);
  // Calculated from Lua Torch Cudnn implementation
  std::array<double, 90> expected_out = {
      1.5418,  1.6389,  1.7361,  1.5491,  1.6472,  1.7452,  1.5564,  1.6554,
      1.7544,  1.5637,  1.6637,  1.7636,  1.5710,  1.6719,  1.7728,  3.4571,
      3.7458,  4.0345,  3.4761,  3.7670,  4.0578,  3.4951,  3.7881,  4.0812,
      3.5141,  3.8093,  4.1045,  3.5331,  3.8305,  4.1278,  5.6947,  6.2004,
      6.7060,  5.7281,  6.2373,  6.7466,  5.7614,  6.2743,  6.7871,  5.7948,
      6.3112,  6.8276,  5.8282,  6.3482,  6.8681,  8.2005,  8.9458,  9.6911,
      8.2500,  9.0005,  9.7509,  8.2995,  9.0551,  9.8107,  8.3491,  9.1098,
      9.8705,  8.3986,  9.1645,  9.9303,  10.9520, 11.9587, 12.9655, 11.0191,
      12.0326, 13.0462, 11.0861, 12.1065, 13.1269, 11.1532, 12.1804, 13.2075,
      11.2203, 12.2543, 13.2882, 13.9432, 15.2333, 16.5233, 14.0291, 15.3277,
      16.6263, 14.1149, 15.4221, 16.7292, 14.2008, 15.5165, 16.8322, 14.2866,
      15.6109, 16.9351};

  auto expected_outVar =
      Variable(af::array(expected_dims, expected_out.data()), true);
  ASSERT_TRUE(allClose(out, expected_outVar, 1E-4));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
