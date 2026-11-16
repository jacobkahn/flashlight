/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cmath>

#include "flashlight/fl/common/common.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/fl/optim/optim.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/test/TestInit.h"

using namespace fl;

TEST(OptimTest, GradNorm) {
  std::vector<Tensor> parameters;
  for (int i = 0; i < 5; i++) {
    auto v = Tensor(fl::randn({10, 10, 10}), true);
    v = v.astype(fl::dtype::f64);
    v.addGrad(Tensor(fl::randn({10, 10, 10}, fl::dtype::f64), false));
    parameters.push_back(v);
  }
  double max_norm = 5.0;
  clipGradNorm(parameters, max_norm);

  double clipped = 0.0;
  for (auto& v : parameters) {
    auto& g = v.grad();
    clipped += fl::sum(g * g).asScalar<double>();
  }
  clipped = std::sqrt(clipped);
  ASSERT_TRUE(allClose(fl::full({1}, max_norm), fl::full({1}, clipped)));
}

TEST(OptimTest, GradNormF16) {
  if (!fl::dtypeSupported(fl::dtype::f16)) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  std::vector<Tensor> parameters;
  for (int i = 0; i < 5; i++) {
    auto v = Tensor(fl::randn({10, 10, 10}), true);
    v = v.astype(fl::dtype::f16);
    v.addGrad(Tensor(fl::randn({10, 10, 10}, fl::dtype::f16), false));
    parameters.push_back(v);
  }
  double max_norm = 5.0;
  clipGradNorm(parameters, max_norm);

  double clipped = 0.0;
  for (auto& v : parameters) {
    auto& g = v.grad();
    clipped += fl::sum(g * g).asScalar<double>();
  }
  clipped = std::sqrt(clipped);
  ASSERT_TRUE(allClose(fl::full({1}, max_norm), fl::full({1}, clipped), 1e-2));
}

TEST(SerializationTest, OptimizerSerialize) {
  const fs::path path = fs::temp_directory_path() / "optmizer.bin";

  std::vector<Tensor> parameters;
  for (int i = 0; i < 5; i++) {
    auto v = Tensor(fl::randn({10, 10, 10}, fl::dtype::f64), true);
    v.addGrad(Tensor(fl::randn({10, 10, 10}, fl::dtype::f64), false));
    parameters.push_back(v);
  }

  std::shared_ptr<FirstOrderOptimizer> opt;
  opt = std::make_shared<AdamOptimizer>(parameters, 0.0001);
  opt->step();

  save(
      path, parameters, static_cast<std::shared_ptr<FirstOrderOptimizer>>(opt));

  std::vector<Tensor> parameters2;
  std::shared_ptr<FirstOrderOptimizer> opt2;
  load(path, parameters2, opt2);

  for (int i = 0; i < 5; i++) {
    parameters2[i].addGrad(Tensor(parameters[i].grad(), false));
  }

  opt->step();
  opt2->step();

  for (int i = 0; i < 5; i++) {
    ASSERT_TRUE(allClose(parameters[i], parameters2[i]));
  }

  opt = std::make_shared<NovogradOptimizer>(parameters, 0.01);
  opt->step();

  save(
      path, parameters, static_cast<std::shared_ptr<FirstOrderOptimizer>>(opt));
  load(path, parameters2, opt2);

  for (int i = 0; i < 5; i++) {
    parameters2[i].addGrad(Tensor(parameters[i].grad(), false));
  }

  opt->step();
  opt2->step();

  for (int i = 0; i < 5; i++) {
    ASSERT_TRUE(allClose(parameters[i], parameters2[i]));
  }
}

// Optimizer steps must update the module's own parameters. Disabled: today
// Tensor copies get separate storage (sharing only autograd metadata), and
// optimizers hold copies that `data = data - lr * grad` rebinds, so the module
// never changes (see TODO.md "Review + test status (audit 2026-10-26)").
// Enable once Tensor value-vs-alias semantics for parameters are decided.
TEST(OptimTest, DISABLED_StepUpdatesModuleParams) {
  Linear model(4, 3);
  SGDOptimizer opt(model.params(), /* lr = */ 0.1);
  auto x = fl::rand({5, 4});
  auto loss = [&]() {
    return fl::sum(model.forward(x), {0, 1})
        .astype(dtype::f64)
        .scalarSync<double>();
  };
  double prev = loss();
  for (int step = 0; step < 2; ++step) {
    opt.zeroGrad();
    auto out = fl::sum(model.forward(x), {0, 1});
    out.backward();
    opt.step();
    double cur = loss();
    ASSERT_LT(cur, prev) << "step " << step << " did not change the model";
    prev = cur;
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  flTestInit();
  return RUN_ALL_TESTS();
}
