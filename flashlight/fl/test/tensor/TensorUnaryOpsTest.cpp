/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/JitBackend.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/test/TestInit.h"

#include <limits>
#include <utility>
#include <vector>

using namespace ::testing;
using namespace fl;

namespace {

TensorBackendType concreteDefaultBackendType() {
  auto& backend = fl::defaultTensorBackend();
  if (backend.backendType() != TensorBackendType::Jit) {
    return backend.backendType();
  }
  return dynamic_cast<JitBackend&>(backend).wrappedBackend().backendType();
}

bool defaultBackendSupportsGeneralTriangularOps() {
  const auto backendType = concreteDefaultBackendType();
  return backendType == TensorBackendType::CPU ||
      backendType == TensorBackendType::Cuda;
}

} // namespace

TEST(TensorUnaryOpsTest, negative) {
  auto a = fl::full({3, 3}, 1);
  auto b = fl::full({3, 3}, 2);
  auto c = a - b;
  ASSERT_TRUE(allClose(c, -a));
  ASSERT_TRUE(allClose(c, negative(a)));
}

TEST(TensorUnaryOpsTest, logicalNot) {
  ASSERT_TRUE(allClose(
      !fl::full({3, 3}, true), fl::full({3, 3}, false).astype(dtype::b8)));
}

TEST(TensorUnaryOpsTest, clip) {
  float h = 3.;
  float l = 2.;
  Shape s = {3, 3};
  auto high = fl::full(s, h);
  auto low = fl::full(s, l);
  ASSERT_TRUE(allClose(fl::clip(fl::full({3, 3}, 4.), low, high), high));
  ASSERT_TRUE(allClose(fl::clip(fl::full({3, 3}, 4.), l, high), high));
  ASSERT_TRUE(allClose(fl::clip(fl::full({3, 3}, 4.), low, h), high));
  ASSERT_TRUE(allClose(fl::clip(fl::full({3, 3}, 4.), l, h), high));
}

TEST(TensorUnaryOpsTest, clipElementwiseBounds) {
  // Each element is clipped against its own bounds, not the bounds' first
  // element.
  auto x = Tensor::fromVector<float>({4}, {0, 1, 2, 3});
  auto low = Tensor::fromVector<float>({4}, {1, 0, 3, 0});
  auto high = Tensor::fromVector<float>({4}, {2, 0.5, 4, 2});
  auto expected = Tensor::fromVector<float>({4}, {1, 0.5, 3, 2});
  ASSERT_TRUE(allClose(fl::clip(x, low, high), expected));
}

TEST(TensorUnaryOpsTest, mathOpsNonF32Types) {
  // floor/ceil/sin/cos/log1p/erf/isnan must handle every dtype, not just
  // f32/f64, and keep the input dtype (isnan returns b8).
  auto i = Tensor::fromVector<int>({3}, {-2, 0, 5});
  ASSERT_EQ(fl::floor(i).type(), dtype::s32);
  ASSERT_TRUE(allClose(fl::floor(i), i));
  ASSERT_TRUE(allClose(fl::ceil(i), i));
  auto zeros = fl::full({3}, 0, dtype::s32);
  ASSERT_EQ(fl::cos(zeros).type(), dtype::s32);
  ASSERT_TRUE(allClose(fl::cos(zeros), fl::full({3}, 1, dtype::s32)));
  ASSERT_TRUE(allClose(fl::sin(zeros), zeros));
  ASSERT_FALSE(fl::any(fl::isnan(i)).scalarSync<char>());

  for (auto type : {dtype::f16, dtype::bf16}) {
    if (!fl::defaultTensorBackend().supportsDataType(type)) {
      continue;
    }
    auto xf = Tensor::fromVector<float>({3}, {0.25, 1.5, -2.5});
    auto x = xf.astype(type);
    ASSERT_EQ(fl::floor(x).type(), type);
    ASSERT_TRUE(allClose(
        fl::floor(x).astype(dtype::f32),
        Tensor::fromVector<float>({3}, {0, 1, -3})));
    ASSERT_TRUE(allClose(
        fl::ceil(x).astype(dtype::f32),
        Tensor::fromVector<float>({3}, {1, 2, -2})));
    for (auto [got, want] :
         {std::pair{fl::sin(x), fl::sin(xf)},
          std::pair{fl::cos(x), fl::cos(xf)},
          std::pair{fl::log1p(fl::abs(x)), fl::log1p(fl::abs(xf))},
          std::pair{fl::erf(x), fl::erf(xf)}}) {
      ASSERT_EQ(got.type(), type);
      ASSERT_TRUE(allClose(got.astype(dtype::f32), want, 2e-2));
    }
    auto nan =
        fl::full({2}, std::numeric_limits<double>::quiet_NaN()).astype(type);
    ASSERT_TRUE(fl::all(fl::isnan(nan)).scalarSync<char>());
  }
}

TEST(TensorUnaryOpsTest, roll) {
  auto t = fl::full({5, 5}, 4.);
  ASSERT_TRUE(allClose(t, fl::roll(t, /* shift = */ 3, /* axis = */ 1)));

  Shape dims({4, 5});
  auto r = fl::arange(dims);
  auto result = fl::roll(r, /* shift = */ 1, /* axis = */ 0);
  ASSERT_EQ(r.shape(), result.shape());
  ASSERT_TRUE(allClose(result(0), fl::full({dims[1]}, dims[0] - 1, r.type())));
  ASSERT_TRUE(allClose(
      result(fl::range(1, fl::end)),
      fl::arange({dims[0] - 1, dims[1]}, /* seqDim = */ 0, r.type())));
}

TEST(TensorUnaryOpsTest, isnan) {
  Shape s = {3, 3};
  ASSERT_TRUE(allClose(
      fl::isnan(fl::full(s, 1.) / 3),
      fl::full(s, false).astype(fl::dtype::b8)));
}

TEST(TensorUnaryOpsTest, isinf) {
  Shape s = {3, 3};
  ASSERT_TRUE(allClose(
      fl::isinf(fl::full(s, 1.) / 3),
      fl::full(s, false).astype(fl::dtype::b8)));
  ASSERT_TRUE(allClose(
      fl::isinf(fl::full(s, 1.) / 0.),
      fl::full(s, true).astype(fl::dtype::b8)));
}

TEST(TensorUnaryOpsTest, sign) {
  auto vals = fl::rand({5, 5}) - 0.5;
  vals(2, 2) = 0.;
  auto signs = fl::sign(vals);
  vals(vals > 0) = 1;
  vals(vals == 0) = 0;
  vals(vals < 0) = -1;
  ASSERT_TRUE(allClose(signs, vals));
}

TEST(TensorUnaryOpsTest, tril) {
  auto checkTril = [](const Shape& shape,
                      const Tensor& res,
                      const Tensor& in,
                      int diagonal = 0) {
    ASSERT_GE(shape.ndim(), 2);
    const auto rows = shape.dim(shape.ndim() - 2);
    const auto cols = shape.dim(shape.ndim() - 1);
    const auto batch = shape.elements() / (rows * cols);
    std::vector<float> resHost(static_cast<size_t>(shape.elements()));
    std::vector<float> inHost(static_cast<size_t>(shape.elements()));
    res.hostSync(resHost.data());
    in.hostSync(inHost.data());

    for (Dim b = 0; b < batch; ++b) {
      for (Dim i = 0; i < rows; ++i) {
        for (Dim j = 0; j < cols; ++j) {
          const auto idx = static_cast<size_t>((b * rows + i) * cols + j);
          if (j - i > diagonal) {
            ASSERT_FLOAT_EQ(resHost[idx], 0.);
          } else {
            ASSERT_FLOAT_EQ(resHost[idx], inHost[idx]);
          }
        }
      }
    }
  };
  if (!defaultBackendSupportsGeneralTriangularOps()) {
    GTEST_SKIP() << concreteDefaultBackendType()
                 << " backend does not yet support generalized tril";
  }

  Dim dim = 10;
  auto t = fl::rand({dim, dim}, dtype::f32);
  auto out = fl::tril(t);
  checkTril(t.shape(), out, t);

  auto diagonalOut = fl::tril(t, 1);
  checkTril(t.shape(), diagonalOut, t, 1);

  // > 2 dims: triangular submatrices are over the last two dimensions.
  auto t2 = fl::rand({2, 3, 4}, dtype::f32);
  auto out2 = fl::tril(t2);
  checkTril(t2.shape(), out2, t2);
}

TEST(TensorUnaryOpsTest, triu) {
  auto checkTriu = [](const Shape& shape,
                      const Tensor& res,
                      const Tensor& in,
                      int diagonal = 0) {
    ASSERT_GE(shape.ndim(), 2);
    const auto rows = shape.dim(shape.ndim() - 2);
    const auto cols = shape.dim(shape.ndim() - 1);
    const auto batch = shape.elements() / (rows * cols);
    std::vector<float> resHost(static_cast<size_t>(shape.elements()));
    std::vector<float> inHost(static_cast<size_t>(shape.elements()));
    res.hostSync(resHost.data());
    in.hostSync(inHost.data());

    for (Dim b = 0; b < batch; ++b) {
      for (Dim i = 0; i < rows; ++i) {
        for (Dim j = 0; j < cols; ++j) {
          const auto idx = static_cast<size_t>((b * rows + i) * cols + j);
          if (j - i < diagonal) {
            ASSERT_FLOAT_EQ(resHost[idx], 0.);
          } else {
            ASSERT_FLOAT_EQ(resHost[idx], inHost[idx]);
          }
        }
      }
    }
  };

  if (!defaultBackendSupportsGeneralTriangularOps()) {
    GTEST_SKIP() << concreteDefaultBackendType()
                 << " backend does not yet support generalized triu";
  }

  int dim = 10;
  auto t = fl::rand({dim, dim}, dtype::f32);
  auto out = fl::triu(t);
  checkTriu(t.shape(), out, t);

  auto diagonalOut = fl::triu(t, -1);
  checkTriu(t.shape(), diagonalOut, t, -1);

  // > 2 dims: triangular submatrices are over the last two dimensions.
  auto t2 = fl::rand({2, 3, 4}, dtype::f32);
  auto out2 = fl::triu(t2);
  checkTriu(t2.shape(), out2, t2);
}

TEST(TensorUnaryOpsTest, floor) {
  auto a = fl::rand({10, 10}) + 0.5;
  ASSERT_TRUE(allClose((a >= 1.).astype(fl::dtype::f32), fl::floor(a)));
}

TEST(TensorUnaryOpsTest, ceil) {
  auto a = fl::rand({10, 10}) + 0.5;
  ASSERT_TRUE(allClose((a >= 1).astype(fl::dtype::f32), fl::ceil(a) - 1));
}

TEST(TensorUnaryOpsTest, rint) {
  Shape s = {10, 10};
  auto a = fl::rand(s) - 0.5;
  ASSERT_TRUE(allClose(fl::rint(a), fl::full(s, 0.)));
  auto b = fl::rand(s) + 0.5;
  ASSERT_TRUE(allClose(fl::rint(b), fl::full(s, 1.)));
}

TEST(TensorUnaryOpsTest, sigmoid) {
  auto a = fl::rand({10, 10});
  ASSERT_TRUE(allClose(1 / (1 + fl::exp(-a)), fl::sigmoid(a)));
}

TEST(TensorUnaryOpsTest, flip) {
  const unsigned high = 10;
  auto a = fl::arange({high});
  auto flipped = fl::flip(a, /* dim = */ 0);
  a *= -1;
  a += (high - 1);
  ASSERT_TRUE(allClose(a, flipped));

  auto b = fl::arange({high, high}, /* seqDim = */ 0);
  ASSERT_TRUE(allClose(fl::flip(b, 1), b));
  auto c = fl::arange({high, high}, /* seqDim = */ 1);
  ASSERT_TRUE(allClose(fl::flip(c, 0), c));
}

TEST(TensorUnaryOpsTest, where) {
  // 1 0
  // 0 1
  auto cond = fl::Tensor::fromVector<char>({2, 2}, {1, 0, 0, 1});
  // 0 2
  // 1 3
  auto x = fl::Tensor::fromVector<int>({2, 2}, {0, 1, 2, 3});
  // 4 6
  // 5 7
  auto y = fl::Tensor::fromVector<int>({2, 2}, {4, 5, 6, 7});

  // 0 6
  // 5 3
  ASSERT_TRUE(allClose(
      fl::where(cond, x, y),
      fl::Tensor::fromVector<int>({2, 2}, {0, 5, 6, 3})));
  // 0 1
  // 1 3
  ASSERT_TRUE(allClose(
      fl::where(cond, x, 1.0),
      fl::Tensor::fromVector<int>({2, 2}, {0, 1, 1, 3})));
  // 2 6
  // 5 2
  ASSERT_TRUE(allClose(
      fl::where(cond, 2.0, y),
      fl::Tensor::fromVector<int>({2, 2}, {2, 5, 6, 2})));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  flTestInit();
  return RUN_ALL_TESTS();
}
