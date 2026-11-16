/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/backend/cpu/CPUBackend.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"

using fl::CPUTensor;

namespace {

void assertRawDataConstructor(
    CPUTensor& tensor,
    const fl::Shape& shape,
    const fl::dtype type,
    const fl::Location location,
    const fl::Shape& strides,
    const int scalar) {
  ASSERT_EQ(tensor.shape(), shape);
  ASSERT_EQ(tensor.backendType(), fl::TensorBackendType::CPU);
  ASSERT_EQ(tensor.type(), type);
  ASSERT_EQ(tensor.location(), location);
  ASSERT_FALSE(tensor.isSparse());
  ASSERT_TRUE(tensor.isContiguous());
  ASSERT_EQ(tensor.strides(), strides);
  int scalarVar = 0;
  ASSERT_NO_THROW(tensor.scalarSync(&scalarVar));
  ASSERT_EQ(scalarVar, scalar);
}

template <typename T>
static CPUTensor fromVector(const fl::Shape& s, const std::vector<T>& v) {
  assert(s.elements() == v.size());
  return CPUTensor(
      s, fl::dtype_traits<T>::fl_type, v.data(), fl::Location::Host);
}

void assertCPUTensorEq(fl::Tensor& lhs, fl::Tensor&& rhs) {
  auto& oneDnnLhs = lhs.getAdapter<CPUTensor>();
  auto& oneDnnRhs = rhs.getAdapter<CPUTensor>();
  ASSERT_TRUE(oneDnnLhs.equals(std::move(oneDnnRhs)))
      << "lhs:\n"
      << oneDnnLhs.toString() << "rhs:\n"
      << oneDnnRhs.toString();
}

void assertCPUTensorEq(fl::Tensor&& lhs, fl::Tensor&& rhs) {
  // we know it's safe, because the reference parameter won't be stored anywhere
  fl::Tensor& lhsRef = lhs;
  assertCPUTensorEq(lhsRef, std::move(rhs));
}

// usually we just need to specify output type, like mapFunc<float>(...);
template <typename Out, typename In>
std::vector<Out> mapFunc(const std::vector<In>& inputs, Out (*func)(In)) {
  return mapFunc<Out>(inputs, [func](In x) { return func(x); });
}

template <typename Out, typename In, typename Func>
std::vector<Out> mapFunc(const std::vector<In>& inputs, Func func) {
  std::vector<Out> outputs;
  for (const auto& input : inputs) {
    outputs.emplace_back(func(input));
  }
  return outputs;
}

} // namespace

TEST(CPUTensorTest, emptyConstructor) {
  CPUTensor tensor;
  ASSERT_EQ(tensor.shape().elements(), 0);
  ASSERT_EQ(tensor.backendType(), fl::TensorBackendType::CPU);
  ASSERT_EQ(tensor.type(), fl::dtype::f32);
  ASSERT_EQ(tensor.location(), fl::Location::Host);
  ASSERT_FALSE(tensor.isSparse());
  ASSERT_TRUE(tensor.isContiguous());
  ASSERT_EQ(tensor.strides(), fl::Shape({1}));
  int scalar = 0;
  ASSERT_THROW(tensor.scalarSync(&scalar), std::invalid_argument);
  ASSERT_EQ(tensor.toString(), "[]\n");
}

TEST(CPUTensorTest, rawDataConstructor0D) {
  const fl::Shape shape{};
  const std::vector<int> data{42};
  const fl::dtype type = fl::dtype::s32;
  const fl::Location location = fl::Location::Host;
  const fl::Shape strides{1};
  CPUTensor tensor(shape, type, data.data(), location);
  assertRawDataConstructor(tensor, shape, type, location, strides, data[0]);
}

TEST(CPUTensorTest, rawDataConstructor1D) {
  const fl::Shape shape{2};
  const std::vector<int> data{1, 2};
  const fl::dtype type = fl::dtype::s32;
  const fl::Location location = fl::Location::Host;
  const fl::Shape strides{1};
  CPUTensor tensor(shape, type, data.data(), location);
  assertRawDataConstructor(tensor, shape, type, location, strides, data[0]);
}

TEST(CPUTensorTest, rawDataConstructor2D) {
  const fl::Shape shape{2, 3};
  const std::vector<int> data{2, 3, 4, 5, 6, 7};
  const fl::dtype type = fl::dtype::s32;
  const fl::Location location = fl::Location::Host;
  const fl::Shape strides{3, 1}; // row-major
  CPUTensor tensor(shape, type, data.data(), location);
  assertRawDataConstructor(tensor, shape, type, location, strides, data[0]);
}

TEST(CPUTensorTest, rawDataConstructor3D) {
  const fl::Shape shape{2, 3, 4};
  const std::vector<int> data(shape.elements());
  const fl::dtype type = fl::dtype::s32;
  const fl::Location location = fl::Location::Host;
  const fl::Shape strides{12, 4, 1}; // row-major
  CPUTensor tensor(shape, type, data.data(), location);
  assertRawDataConstructor(tensor, shape, type, location, strides, data[0]);
}

TEST(CPUTensorTest, rawDataConstructor4D) {
  const fl::Shape shape{2, 3, 4, 5};
  const std::vector<int> data(shape.elements());
  const fl::dtype type = fl::dtype::s32;
  const fl::Location location = fl::Location::Host;
  const fl::Shape strides{60, 20, 5, 1}; // row-major
  CPUTensor tensor(shape, type, data.data(), location);
  assertRawDataConstructor(tensor, shape, type, location, strides, data[0]);
}

TEST(CPUTensorTest, toString) {
  // NOTE using `char` to make sure we don't print out bytes as ascii chars
  // empty
  ASSERT_EQ(fromVector<char>({0}, {}).toString(), "[]\n");
  ASSERT_EQ(fromVector<char>({0, 0}, {}).toString(), "[]\n");

  // 1D
  ASSERT_EQ(fromVector<char>({}, {0}).toString(), "0\n");

  // 1D
  ASSERT_EQ(
      fromVector<char>({4}, {0, 1, 2, 3}).toString(),
      "[0, 1, 2, 3]\n");

  // 2D — shape {3, 2}: 3 rows, 2 cols, row-major data
  ASSERT_EQ(
      fromVector<char>({3, 2}, {0, 1, 2, 3, 4, 5}).toString(),
      "[[0, 1],\n"
      " [2, 3],\n"
      " [4, 5]]\n");

  // 3D — shape {2, 3, 2}: row-major data
  ASSERT_EQ(
      fromVector<char>({2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})
          .toString(),
      "[[[0, 1],\n"
      "  [2, 3],\n"
      "  [4, 5]],\n"
      " [[6, 7],\n"
      "  [8, 9],\n"
      "  [10, 11]]]\n");
}

TEST(CPUTensorTest, equals) {
  const fl::Shape shape{2, 3, 4};
  const fl::Location location = fl::Location::Host;

  {
    std::vector<int> data(shape.elements());
    const fl::dtype type = fl::dtype::s32;
    std::generate(data.begin(), data.end(), std::rand);
    CPUTensor t1 = CPUTensor(shape, type, data.data(), location);
    ASSERT_TRUE(t1.equals(CPUTensor(shape, type, data.data(), location)));
    data[0]++;
    ASSERT_FALSE(t1.equals(CPUTensor(shape, type, data.data(), location)));
  }

  {
    std::vector<float> data(shape.elements());
    const fl::dtype type = fl::dtype::f32;
    std::generate(data.begin(), data.end(), std::rand);
    CPUTensor t1 = CPUTensor(shape, type, data.data(), location);
    ASSERT_TRUE(t1.equals(CPUTensor(shape, type, data.data(), location)));
    data[0] = (data[0] + 1) * (data[0] + 1); // make sure it's different enough
    ASSERT_FALSE(t1.equals(CPUTensor(shape, type, data.data(), location)));
  }

  {
    std::vector<char> data(shape.elements());
    const fl::dtype type = fl::dtype::u8;
    std::generate(data.begin(), data.end(), std::rand);
    CPUTensor t1 = CPUTensor(shape, type, data.data(), location);
    ASSERT_TRUE(t1.equals(CPUTensor(shape, type, data.data(), location)));
    data[0]++;
    ASSERT_FALSE(t1.equals(CPUTensor(shape, type, data.data(), location)));
  }
}

TEST(CPUTensorTest, transpose) {
  // [[[0, 1],
  //   [2, 3],
  //   [4, 5]],
  //  [[6, 7],
  //   [8, 9],
  //   [10, 11]]]
  auto t1 = fl::Tensor::fromVector<int>(
      {2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  assertCPUTensorEq(t1, fl::transpose(fl::transpose(t1)));

  // Default transpose reverses the axes: t2[a][b][c] = t1[c][b][a].
  // [[[0, 6],
  //   [2, 8],
  //   [4, 10]],
  //  [[1, 7],
  //   [3, 9],
  //   [5, 11]]]
  auto t2 = fl::Tensor::fromVector<int>(
      {2, 3, 2}, {0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11});
  assertCPUTensorEq(t2, fl::transpose(t1));

  // Row-major: t1[i][j][k] = 6i + 2j + k, and transpose(t1, {2, 0, 1})[p][q][r]
  // = t1[q][r][p]:
  // [[[0, 2, 4],
  //   [6, 8, 10]],
  //  [[1, 3, 5],
  //   [7, 9, 11]]]
  auto t3 = fl::Tensor::fromVector<int>(
      {2, 2, 3}, {0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11});
  ASSERT_EQ(fl::Shape({2, 2, 3}), fl::transpose(t1, {2, 0, 1}).shape());
  assertCPUTensorEq(t3, fl::transpose(t1, {2, 0, 1}));
}

TEST(CPUTensorTest, full) {
  const fl::Shape shape{2, 2, 2};
  assertCPUTensorEq(
      fl::Tensor::fromVector(shape, std::vector<float>(shape.elements(), 40.7)),
      fl::full(shape, 40.7, fl::dtype::f32));
  assertCPUTensorEq(
      fl::Tensor::fromVector(shape, std::vector<int>(shape.elements(), 42)),
      fl::full(shape, 42, fl::dtype::s32));
}

TEST(CPUTensorTest, astype) {
  // some casts like u8 -> f16, aren't supported in OneDNN on certain platforms.
  auto tInt = fl::full({2, 2, 2}, 40, fl::dtype::s32);
  auto tFloat = fl::full({2, 2, 2}, 40.0f, fl::dtype::f32);
  assertCPUTensorEq(tInt, tFloat.astype(fl::dtype::s32));
  assertCPUTensorEq(tFloat, tInt.astype(fl::dtype::f32));
}

TEST(CPUTensorTest, host) {
  const std::vector<int> data{0, 1, 2, 3};
  std::vector<int> temp(4, 0);
  auto t = fromVector({2, 2}, data);

  // check temp.data() is propagated with tensor data
  ASSERT_NE(data, temp);
  t.hostSync(temp.data());
  ASSERT_EQ(data, temp);

  // check temp.data() isn't "mirrored" to tensor data
  temp.data()[0] = 42;
  ASSERT_TRUE(t.equals(fromVector({2, 2}, data)));
}

TEST(CPUTensorTest, device) {
  auto t = fromVector<int>({2, 2}, {0, 1, 2, 3});
  void* devicePtr = nullptr;

  t.device(&devicePtr);
  ASSERT_NE(devicePtr, nullptr);
  ASSERT_FALSE(t.isLocked());
}

TEST(CPUTensorTest, arithmetics) {
  auto t1 = fl::Tensor::fromVector<float>({2, 2}, {0, 1, 2, 3});
  auto t2 = fl::Tensor::fromVector<int>({2, 2}, {1, 2, 3, 4});
  auto t3 = fl::Tensor::fromVector<int>({2, 2}, {3, 5, 7, 9});

  assertCPUTensorEq( // implicit casting
      t1 + t2,
      fl::Tensor::fromVector<float>({2, 2}, {1, 3, 5, 7}));

  // literal with casting
  auto t = fl::Tensor::fromVector<float>({2, 2}, {1, 2, 3, 4});
  double oneDouble = 1.0f;
  double oneInt = 1;
  assertCPUTensorEq(t, t1 + oneDouble);
  assertCPUTensorEq(t, oneDouble + t1);
  assertCPUTensorEq(t, t1 + oneInt);
  assertCPUTensorEq(t, oneInt + t1);

  assertCPUTensorEq(
      t3 - t2, fl::Tensor::fromVector<int>({2, 2}, {2, 3, 4, 5}));

  assertCPUTensorEq(
      t1 * t2, fl::Tensor::fromVector<float>({2, 2}, {0, 2, 6, 12}));

  assertCPUTensorEq(
      t3 / t2, fl::Tensor::fromVector<int>({2, 2}, {3, 2, 2, 2}));
}

TEST(CPUTensorTest, comparison) {
  auto t1 = fl::Tensor::fromVector<float>({2, 2}, {0, 1, 2, 3});
  auto t2 = fl::Tensor::fromVector<float>({2, 2}, {0, 2, 2, 4});

  assertCPUTensorEq(
      t1 == t2, fl::Tensor::fromVector<char>({2, 2}, {1, 0, 1, 0}));

  // literals with casting
  auto t = fl::Tensor::fromVector<char>({2, 2}, {0, 1, 1, 0});
  double twoDouble = 2.0f;
  double twoInt = 2;
  assertCPUTensorEq(t, t2 == twoDouble);
  assertCPUTensorEq(t, twoDouble == t2);
  assertCPUTensorEq(t, t2 == twoInt);
  assertCPUTensorEq(t, twoInt == t2);

  assertCPUTensorEq(
      t1 != t2, fl::Tensor::fromVector<char>({2, 2}, {0, 1, 0, 1}));

  assertCPUTensorEq(
      t1 < t2, fl::Tensor::fromVector<char>({2, 2}, {0, 1, 0, 1}));

  assertCPUTensorEq(
      t1 <= t2, fl::Tensor::fromVector<char>({2, 2}, {1, 1, 1, 1}));

  assertCPUTensorEq(
      t1 > t2, fl::Tensor::fromVector<char>({2, 2}, {0, 0, 0, 0}));

  assertCPUTensorEq(
      t1 >= t2, fl::Tensor::fromVector<char>({2, 2}, {1, 0, 1, 0}));
}

TEST(CPUTensorTest, minimumMaximum) {
  auto t1 = fl::Tensor::fromVector<float>({2, 2}, {0, 1, 2, 3});
  auto t2 = fl::Tensor::fromVector<float>({2, 2}, {0, 2, 2, 4});

  assertCPUTensorEq(t2, fl::maximum(t1, t2));
  assertCPUTensorEq(
      fl::maximum(t1, 2), fl::Tensor::fromVector<float>({2, 2}, {2, 2, 2, 3}));
  assertCPUTensorEq(
      fl::maximum(2, t1), fl::Tensor::fromVector<float>({2, 2}, {2, 2, 2, 3}));

  assertCPUTensorEq(t1, fl::minimum(t1, t2));
  assertCPUTensorEq(
      fl::minimum(t1, 1), fl::Tensor::fromVector<float>({2, 2}, {0, 1, 1, 1}));
  assertCPUTensorEq(
      fl::minimum(1, t1), fl::Tensor::fromVector<float>({2, 2}, {0, 1, 1, 1}));
}

TEST(CPUTensorTest, logicalBinops) {
  auto t1 = fl::Tensor::fromVector<char>({2, 2}, {0, 3, 0, 5});
  auto t2 = fl::Tensor::fromVector<int>({2, 2}, {0, 2, -1, 0});
  auto t3 = fl::Tensor::fromVector<int>({2}, {0, 2});
  auto t4 = fl::Tensor::fromVector<int>({2, 1}, {0, 2});
  auto t5 = fl::Tensor::fromVector<int>({3}, {0, 2, -1});

  assertCPUTensorEq(
      t1 && t2, fl::Tensor::fromVector<char>({2, 2}, {0, 1, 0, 0}));
  assertCPUTensorEq(
      t1 || t2, fl::Tensor::fromVector<char>({2, 2}, {0, 1, 1, 1}));
  // NumPy broadcasting: {2} and {2, 1} broadcast against {2, 2}.
  assertCPUTensorEq(
      t1 && t3, fl::Tensor::fromVector<char>({2, 2}, {0, 1, 0, 1}));
  assertCPUTensorEq(
      t4 || t1, fl::Tensor::fromVector<char>({2, 2}, {0, 1, 1, 1}));
  ASSERT_THROW(t1 && t5, std::invalid_argument);
}

TEST(CPUTensorTest, assign) {
  const auto type = fl::dtype::f32;
  auto t1 = fl::full({2, 2}, 40.7, type);
  auto t2 = fl::full({2, 2}, 23, type);

  // ensure it's not a shallow copy
  t2 = t1;
  t1(0) = fl::full({2}, 0, type);
  // row-major: t1(0) sets first row to 0
  assertCPUTensorEq(
      t1, fl::Tensor::fromVector<float>({2, 2}, {0, 0, 40.7, 40.7}, type));
  assertCPUTensorEq(t2, fl::full({2, 2}, 40.7, type));
}

TEST(CPUTensorTest, copy) {
  const auto type = fl::dtype::f32;
  auto t1 = fl::full({2, 2}, 40.7, type);
  assertCPUTensorEq(t1, t1.copy());

  // ensure it's not a shallow copy
  auto t2 = t1.copy();
  t1(0) = fl::full({2}, 0, type);
  // row-major: t1(0) sets first row to 0
  assertCPUTensorEq(
      t1, fl::Tensor::fromVector<float>({2, 2}, {0, 0, 40.7, 40.7}, type));
  assertCPUTensorEq(t2, fl::full({2, 2}, 40.7, type));
}

TEST(CPUTensorTest, rand) {
  fl::Shape shape{2, 2, 2};
  auto type = fl::dtype::f32;
  auto t1 = fl::CPUBackend::getInstance().randn(shape, type);
  ASSERT_EQ(t1.shape(), shape);
  ASSERT_EQ(t1.type(), type);
}

TEST(CPUTensorTest, eltwiseCompute) {
  std::vector<float> t1Data = {1, 2, 3, 4};
  std::vector<float> t2Data = {0, 2, 2, 0};
  auto t1 = fl::Tensor::fromVector({2, 2}, t1Data);
  auto t2 = fl::Tensor::fromVector({2, 2}, t2Data);

  assertCPUTensorEq(
      fl::exp(t1),
      fl::Tensor::fromVector({2, 2}, mapFunc<float>(t1Data, std::exp)));

  assertCPUTensorEq(
      fl::log(t1),
      fl::Tensor::fromVector({2, 2}, mapFunc<float>(t1Data, std::log)));

  assertCPUTensorEq(
      fl::sqrt(t1),
      fl::Tensor::fromVector({2, 2}, mapFunc<float>(t1Data, std::sqrt)));

  assertCPUTensorEq(
      fl::tanh(t1),
      fl::Tensor::fromVector({2, 2}, mapFunc<float>(t1Data, std::tanh)));

  assertCPUTensorEq(
      fl::rint(fl::Tensor::fromVector<float>({2, 2}, {0.1, 1.4, 2.5, 3.9})),
      fl::Tensor::fromVector<float>({2, 2}, {0, 1, 2, 4}));

  assertCPUTensorEq(
      fl::absolute(fl::Tensor::fromVector<float>({2, 2}, {-2, -2.8, 3.4, 0})),
      fl::Tensor::fromVector<float>({2, 2}, {2, 2.8, 3.4, 0}));

  assertCPUTensorEq(
      fl::logicalNot(t2), fl::Tensor::fromVector<char>({2, 2}, {1, 0, 0, 1}));

  assertCPUTensorEq(
      fl::clip(
          fl::Tensor::fromVector<float>({4}, {1.4, 0.1, 2.5, 3.9}), 1.2, 2.7),
      fl::Tensor::fromVector<float>({4}, {1.4, 1.2, 2.5, 2.7}));

  auto powThree = [](auto x) { return std::pow(x, 3); };
  assertCPUTensorEq(
      fl::power(t1, 3),
      fl::Tensor::fromVector(t1.shape(), mapFunc<float>(t1Data, powThree)));

  assertCPUTensorEq(
      fl::erf(t1),
      fl::Tensor::fromVector({2, 2}, mapFunc<float>(t1Data, std::erf)));
}

TEST(CPUTensorTest, eltwiseLogical) {
  std::vector<float> t1Data = {
      std::numeric_limits<float>::infinity(),
      2,
      0.2f,
      -std::numeric_limits<float>::infinity(),
      -2,
      0};
  auto t1 = fl::Tensor::fromVector({2, 3}, t1Data);

  assertCPUTensorEq(
      fl::isinf(t1),
      fl::Tensor::fromVector(t1.shape(), mapFunc<char>(t1Data, [](auto x) {
                               return std::isinf(x);
                             })));

  auto sign = [](auto x) {
    if (x > 0) {
      return 1;
    } else if (x == 0) {
      return 0;
    } else {
      return -1;
    }
  };
  // sign keeps the input's dtype.
  assertCPUTensorEq(
      fl::sign(t1),
      fl::Tensor::fromVector(t1.shape(), mapFunc<float>(t1Data, sign)));
}

TEST(CPUTensorTest, reduction) {
  std::vector<float> data = {-1, 0, 33, 0};
  auto t0 = fl::Tensor::fromVector({2, 2}, data);
  auto t1 = fl::full({2, 3, 4}, 42, fl::dtype::f32);

  assertCPUTensorEq(fl::sum(t1, {1}), fl::full({2, 4}, 126, fl::dtype::f32));
  // row-major data [-1, 0, 33, 0]: row 0 = [-1, 0], row 1 = [33, 0]
  // amax along axis 1: [max(-1, 0), max(33, 0)] = [0, 33]
  assertCPUTensorEq(
      fl::amax(t0, {1}), fl::Tensor::fromVector<float>({2}, {0, 33}));
  assertCPUTensorEq(fl::amin(t0), fl::Tensor::fromVector<float>({}, {-1}));
  assertCPUTensorEq(
      fl::countNonzero(t0), fl::Tensor::fromVector<unsigned>({}, {2}));
  assertCPUTensorEq(
      fl::sum(t1, {}, true),
      fl::full({1, 1, 1}, 42 * 2 * 3 * 4, fl::dtype::f32));
}

TEST(CPUTensorTest, matmul) {
  using MP = fl::MatrixProperty;
  auto& backend = fl::CPUBackend::getInstance();
  // 1 2 3  X  2 5  =  20 38
  // 4 5 6     3 6     47 92
  //           4 7
  // row-major data ordering
  auto t1 = fl::Tensor::fromVector<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  auto t2 = fl::Tensor::fromVector<float>({3, 2}, {2, 5, 3, 6, 4, 7});
  fl::Tensor res1;
  backend.matmul(res1, t1, t2, MP::None, MP::None);
  assertCPUTensorEq(
      res1, fl::Tensor::fromVector<float>({2, 2}, {20, 38, 47, 92}));

  // 1 4  X  2 3 4  = 22 27 32
  // 2 5     5 6 7    29 36 43
  // 3 6              36 45 54
  fl::Tensor res2;
  backend.matmul(res2, t1, t2, MP::Transpose, MP::Transpose);
  assertCPUTensorEq(
      res2,
      fl::Tensor::fromVector<float>(
          {3, 3}, {22, 27, 32, 29, 36, 43, 36, 45, 54}));
}

TEST(CPUTensorTest, matmulShapes) {
  using MP = fl::MatrixProperty;
  auto& backend = fl::CPUBackend::getInstance();
  struct Input {
    fl::Shape lhsShape;
    fl::Shape rhsShape;
    fl::MatrixProperty lhsMP;
    fl::MatrixProperty rhsMP;
    std::optional<fl::Shape> dstShape;
  };
  const std::vector<Input> inputs = {
      // scalar/vector
      {{}, {}, MP::None, MP::None, {{1}}},
      {{2}, {}, MP::None, MP::None, std::nullopt},
      {{}, {3}, MP::None, MP::None, std::nullopt},
      {{4}, {4}, MP::None, MP::None, {{1}}},
      {{2}, {3}, MP::None, MP::None, std::nullopt},
      {{2}, {2, 3}, MP::None, MP::None, {{3}}},
      {{2, 3}, {3}, MP::None, MP::None, {{2}}},
      // matrix
      {{3, 2}, {2, 3}, MP::None, MP::None, {{3, 3}}},
      {{3, 2}, {2, 3}, MP::Transpose, MP::Transpose, {{2, 2}}},
      {{2, 3}, {2, 3}, MP::Transpose, MP::None, {{3, 3}}},
      {{2, 3}, {2, 3}, MP::None, MP::Transpose, {{2, 2}}},
      {{2, 3}, {2, 3}, MP::None, MP::Transpose, {{2, 2}}},
      // batch matrix (batch dims lead: {B, M, K} x {B, N, K}^T = {B, M, N})
      {{2, 3, 42}, {2, 3, 42}, MP::None, MP::Transpose, {{2, 3, 3}}},
      {{2, 3, 41}, {2, 3, 42}, MP::None, MP::Transpose, std::nullopt},
      // TODO support broadcast
      {{2, 3, 1}, {2, 3, 42}, MP::None, MP::Transpose, std::nullopt},
      {{2, 3}, {2, 3, 42}, MP::None, MP::Transpose, std::nullopt},
  };
  for (const auto& input : inputs) {
    const auto lhs = backend.rand(input.lhsShape, fl::dtype::f32);
    const auto rhs = backend.rand(input.rhsShape, fl::dtype::f32);
    if (input.dstShape.has_value()) {
      fl::Tensor res;
      ASSERT_NO_THROW(backend.matmul(res, lhs, rhs, input.lhsMP, input.rhsMP));
      backend.matmul(res, lhs, rhs, input.lhsMP, input.rhsMP);
      ASSERT_EQ(res.shape(), input.dstShape.value());
    } else {
      fl::Tensor res;
      ASSERT_THROW(
          backend.matmul(res, lhs, rhs, input.lhsMP, input.rhsMP),
          std::invalid_argument);
    }
  }
}

TEST(CPUTensorTest, max) {
  using fl::Shape;
  using fl::Tensor;
  // Row-major layout: {1, 2, 4, 3, 5, 6} with shape {2, 3}
  // Row 0: 1 2 4
  // Row 1: 3 5 6
  Tensor in = Tensor::fromVector<float>({2, 3}, {1, 2, 4, 3, 5, 6});
  Tensor values, indices;

  fl::max(values, indices, in, 0);
  assertCPUTensorEq(indices, Tensor::fromVector<unsigned>({3}, {1, 1, 1}));
  assertCPUTensorEq(values, Tensor::fromVector<float>({3}, {3, 5, 6}));

  fl::max(values, indices, in, 1);
  assertCPUTensorEq(indices, Tensor::fromVector<unsigned>({2}, {2, 2}));
  assertCPUTensorEq(values, Tensor::fromVector<float>({2}, {4, 6}));

  fl::max(values, indices, in, 0, /* keepDims = */ true);
  assertCPUTensorEq(indices, Tensor::fromVector<unsigned>({1, 3}, {1, 1, 1}));
  assertCPUTensorEq(values, Tensor::fromVector<float>({1, 3}, {3, 5, 6}));

  fl::max(values, indices, in, 1, /* keepDims = */ true);
  assertCPUTensorEq(indices, Tensor::fromVector<unsigned>({2, 1}, {2, 2}));
  assertCPUTensorEq(values, Tensor::fromVector<float>({2, 1}, {4, 6}));
}

TEST(CPUTensorTest, min) {
  using fl::Shape;
  using fl::Tensor;
  // Row-major layout: {1, 2, 4, 3, 5, 6} with shape {2, 3}
  // Row 0: 1 2 4
  // Row 1: 3 5 6
  Tensor in = Tensor::fromVector<float>({2, 3}, {1, 2, 4, 3, 5, 6});
  Tensor values, indices;

  fl::min(values, indices, in, 0);
  assertCPUTensorEq(indices, Tensor::fromVector<unsigned>({3}, {0, 0, 0}));
  assertCPUTensorEq(values, Tensor::fromVector<float>({3}, {1, 2, 4}));

  fl::min(values, indices, in, 1);
  assertCPUTensorEq(indices, Tensor::fromVector<unsigned>({2}, {0, 0}));
  assertCPUTensorEq(values, Tensor::fromVector<float>({2}, {1, 3}));

  fl::min(values, indices, in, 0, /* keepDims = */ true);
  assertCPUTensorEq(indices, Tensor::fromVector<unsigned>({1, 3}, {0, 0, 0}));
  assertCPUTensorEq(values, Tensor::fromVector<float>({1, 3}, {1, 2, 4}));

  fl::min(values, indices, in, 1, /* keepDims = */ true);
  assertCPUTensorEq(indices, Tensor::fromVector<unsigned>({2, 1}, {0, 0}));
  assertCPUTensorEq(values, Tensor::fromVector<float>({2, 1}, {1, 3}));
}

TEST(CPUTensorTest, reshape) {
  auto a = fl::full({4, 4}, 3.);
  auto b = fl::reshape(a, fl::Shape({8, 2}));
  assertCPUTensorEq(b, fl::full({8, 2}, 3.));
  assertCPUTensorEq(a, fl::reshape(b, {4, 4}));
  assertCPUTensorEq(fl::reshape(a, fl::Shape({16})), fl::full({16}, 3.));

  ASSERT_THROW(fl::reshape(a, {}), std::invalid_argument);
  ASSERT_THROW(fl::reshape(a, {4}), std::invalid_argument);
  ASSERT_THROW(fl::reshape(a, {4, 5}), std::invalid_argument);
  ASSERT_NO_THROW(fl::reshape(a, {4, 4, 1}));
  ASSERT_NO_THROW(fl::reshape(a, {1, 4, 4}));
  ASSERT_NO_THROW(fl::reshape(a, {4, 1, 4}));
  ASSERT_NO_THROW(fl::reshape(a, {1, 4, 4, 1}));
}

TEST(CPUTensorTest, index) {
  // row-major: data {1, 2, 3, 4} with shape {2, 2} = [[1, 2], [3, 4]]
  auto a = fl::Tensor::fromVector<float>({2, 2}, {1, 2, 3, 4});
  // indexing for read: a(0) = first row = [1, 2], a(1) = second row = [3, 4]
  assertCPUTensorEq(a(0), fl::Tensor::fromVector<float>({2}, {1, 2}));
  assertCPUTensorEq(a(1), fl::Tensor::fromVector<float>({2}, {3, 4}));
  // a(0)=[1,2], a(1)=[3,4], sum=[4,6] — sub-view binary ops apply offsets
  assertCPUTensorEq(
      a(0) + a(1),
      fl::Tensor::fromVector<float>({2}, {4, 6}));

  // indexing for write: a(0) sets first row
  a(0) = fl::Tensor::fromVector<float>({2}, {0, 1});
  assertCPUTensorEq(a, fl::Tensor::fromVector<float>({2, 2}, {0, 1, 3, 4}));

  // indexing composability: a(0)(1) = second element of first row
  a(0)(1) = fl::Tensor::fromVector<float>({}, {42});
  assertCPUTensorEq(a, fl::Tensor::fromVector<float>({2, 2}, {0, 42, 3, 4}));
}

TEST(CPUTensorTest, tile) {
  // row-major: {1, 2, 3, 4} with shape {2, 2} = [[1, 2], [3, 4]]
  auto a = fl::Tensor::fromVector<float>({2, 2}, {1, 2, 3, 4});
  // tile along dim 0: [[1,2],[3,4],[1,2],[3,4]]
  assertCPUTensorEq(
      fl::tile(a, {2}),
      fl::Tensor::fromVector<float>({4, 2}, {1, 2, 3, 4, 1, 2, 3, 4}));
  // a(span, 1) = second column = [2, 4], tiled = [2, 4, 2, 4]
  assertCPUTensorEq(
      fl::tile(a(fl::span, 1), {2}),
      fl::Tensor::fromVector<float>({4}, {2, 4, 2, 4}));
}

TEST(CPUTensorTest, arange) {
  // Range/step overload
  assertCPUTensorEq(
      fl::arange(2, 10, 2), fl::Tensor::fromVector<int>({2, 4, 6, 8}));
  assertCPUTensorEq(
      fl::arange(0, 6), fl::Tensor::fromVector<int>({0, 1, 2, 3, 4, 5}));

  // Shape overload
  assertCPUTensorEq(
      fl::arange({4}), fl::Tensor::fromVector<float>({0, 1, 2, 3}));
  // row-major: arange along axis 1 with shape {2, 2} = [[0, 1], [0, 1]]
  assertCPUTensorEq(
      fl::arange({2, 2}, 1),
      fl::Tensor::fromVector<float>({2, 2}, {0, 1, 0, 1}));
}

// Async chain tests: exercise multi-op pipelines where many operations
// are enqueued before any data read forces a sync. Correctness depends
// on FIFO ordering of the AsyncCPUStream worker thread.

TEST(CPUTensorTest, asyncChainArithmetic) {
  // Long chain of arithmetic ops — all enqueued before the final read
  auto a = fl::Tensor::fromVector<float>({2, 2}, {1, 2, 3, 4});
  auto b = fl::Tensor::fromVector<float>({2, 2}, {5, 6, 7, 8});

  // ((a + b) * a - b) / 2.0
  // a+b = {6,8,10,12}, *a = {6,16,30,48}, -b = {1,10,23,40}, /2 = {0.5,5,11.5,20}
  auto result = (((a + b) * a) - b) / 2.0;
  assertCPUTensorEq(
      result,
      fl::Tensor::fromVector<float>({2, 2}, {0.5, 5, 11.5, 20}));
}

TEST(CPUTensorTest, asyncChainEltwiseOps) {
  // Chain of unary element-wise ops
  auto a = fl::Tensor::fromVector<float>({4}, {1, 4, 9, 16});

  // sqrt(a) -> abs(result - 3) -> result * result
  // sqrt = {1,2,3,4}, -3 = {-2,-1,0,1}, abs = {2,1,0,1}, *self = {4,1,0,1}
  auto s = fl::sqrt(a);
  auto diff = s - 3.0;
  auto absDiff = fl::absolute(diff);
  auto result = absDiff * absDiff;
  assertCPUTensorEq(
      result, fl::Tensor::fromVector<float>({4}, {4, 1, 0, 1}));
}

TEST(CPUTensorTest, asyncChainShapeAndArithmetic) {
  // Mix shape ops (transpose, reshape) with arithmetic
  auto a = fl::Tensor::fromVector<float>({2, 3}, {1, 2, 3, 4, 5, 6});

  // transpose -> reshape -> add scalar -> transpose back
  auto t = fl::transpose(a);                      // {3, 2}
  auto r = fl::reshape(t, {6});                    // {6}
  auto added = r + 10.0;                           // {6}
  auto r2 = fl::reshape(added, {3, 2});            // {3, 2}
  auto result = fl::transpose(r2);                 // {2, 3}

  // transpose of {1,2,3,4,5,6} in {2,3} row-major is {1,4,2,5,3,6} in {3,2}
  // reshape to {6} = {1,4,2,5,3,6}
  // +10 = {11,14,12,15,13,16}
  // reshape to {3,2} = {11,14,12,15,13,16}
  // transpose to {2,3} = {11,12,13,14,15,16}
  assertCPUTensorEq(
      result,
      fl::Tensor::fromVector<float>({2, 3}, {11, 12, 13, 14, 15, 16}));
}

TEST(CPUTensorTest, asyncChainWithAstype) {
  // Chain with type conversions
  auto a = fl::Tensor::fromVector<int>({4}, {10, 20, 30, 40});

  // astype to float -> divide by 3.0 -> astype back to int (truncates)
  auto flt = a.astype(fl::dtype::f32);
  auto divided = flt / 3.0;
  auto result = divided.astype(fl::dtype::s32);

  // float->int conversion truncates toward zero
  // 10/3=3.33->3, 20/3=6.66->6, 30/3=10->10, 40/3=13.33->13
  assertCPUTensorEq(
      result, fl::Tensor::fromVector<int>({4}, {3, 6, 10, 13}));
}

TEST(CPUTensorTest, asyncChainWithReduction) {
  // Arithmetic chain followed by reduction — reduction reads data
  auto a = fl::Tensor::fromVector<float>({3, 3},
      {1, 2, 3, 4, 5, 6, 7, 8, 9});

  // (a * 2 + 1), then sum along axis 1
  auto scaled = a * 2.0 + 1.0;
  auto result = fl::sum(scaled, {1});

  // a*2+1 = {3,5,7,9,11,13,15,17,19} in {3,3} row-major
  // rows: [3,5,7], [9,11,13], [15,17,19]
  // sum along axis 1: [15, 33, 51]
  assertCPUTensorEq(
      result, fl::Tensor::fromVector<float>({3}, {15, 33, 51}));
}

TEST(CPUTensorTest, asyncChainWithClone) {
  // Verify clone produces independent data after an async chain
  auto a = fl::Tensor::fromVector<float>({2, 2}, {1, 2, 3, 4});
  auto b = (a + 10.0) * 2.0;  // {22, 24, 26, 28}
  auto c = b.copy();           // clone enqueues a reorder

  // Mutate b, c should be unaffected
  b(0) = fl::full({2}, 0.0f);
  assertCPUTensorEq(
      c, fl::Tensor::fromVector<float>({2, 2}, {22, 24, 26, 28}));
}

TEST(CPUTensorTest, asyncChainWithIndex) {
  // Sub-view from an async chain, then more ops on the view
  auto a = fl::Tensor::fromVector<float>({3, 4},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

  auto scaled = a * 2.0;       // all async
  auto row1 = scaled(1);       // view of second row: {10, 12, 14, 16}
  auto result = row1 + 100.0;  // {110, 112, 114, 116}

  assertCPUTensorEq(
      result, fl::Tensor::fromVector<float>({4}, {110, 112, 114, 116}));
}

TEST(CPUTensorTest, asyncChainWithMatmul) {
  // Arithmetic -> matmul -> arithmetic
  // Row-major: {1,0,0,0,1,0} with shape {2,3} = [[1,0,0],[0,1,0]]
  auto a = fl::Tensor::fromVector<float>({2, 3}, {1, 0, 0, 0, 1, 0});
  auto b = fl::Tensor::fromVector<float>({3, 2}, {1, 2, 3, 4, 5, 6});

  // Scale inputs, matmul, then scale output
  auto scaledA = a * 2.0;      // [[2,0,0],[0,2,0]]
  auto scaledB = b + 1.0;      // [[2,3],[4,5],[6,7]]
  auto& backend = fl::CPUBackend::getInstance();
  fl::Tensor m;
  backend.matmul(
      m, scaledA, scaledB, fl::MatrixProperty::None, fl::MatrixProperty::None);
  auto result = m - 1.0;

  // scaledA = [[2,0,0],[0,2,0]]
  // scaledB = [[2,3],[4,5],[6,7]]
  // matmul = [[2*2+0*4+0*6, 2*3+0*5+0*7],[0*2+2*4+0*6, 0*3+2*5+0*7]]
  //        = [[4, 6], [8, 10]]
  // - 1 = [[3, 5], [7, 9]]
  assertCPUTensorEq(
      result, fl::Tensor::fromVector<float>({2, 2}, {3, 5, 7, 9}));
}

TEST(CPUTensorTest, asyncNativeFlip) {
  auto a = fl::Tensor::fromVector<float>({4}, {1, 2, 3, 4});
  auto scaled = a * 2.0;         // async: {2, 4, 6, 8}
  auto flipped = fl::flip(scaled, 0);  // async native flip
  auto result = flipped + 10.0;  // async: {18, 16, 14, 12}
  assertCPUTensorEq(
      result, fl::Tensor::fromVector<float>({4}, {18, 16, 14, 12}));
}

TEST(CPUTensorTest, asyncNativeRoll) {
  auto a = fl::Tensor::fromVector<float>({4}, {1, 2, 3, 4});
  auto scaled = a + 10.0;         // async: {11, 12, 13, 14}
  auto rolled = fl::roll(scaled, 1, 0);  // async native roll: {14, 11, 12, 13}
  auto result = rolled * 2.0;     // async: {28, 22, 24, 26}
  assertCPUTensorEq(
      result, fl::Tensor::fromVector<float>({4}, {28, 22, 24, 26}));
}

TEST(CPUTensorTest, asyncNativeTrilTriu) {
  auto a = fl::full({3, 3}, 1.0f);
  auto scaled = a * 5.0;          // async: all 5s
  auto lower = fl::tril(scaled);  // async native tril
  auto upper = fl::triu(scaled);  // async native triu
  // tril should zero the upper triangle
  // triu should zero the lower triangle
  // Their sum should give 5s on diagonal, 5 elsewhere = original
  auto result = lower + upper - fl::identity(3) * 5.0;
  assertCPUTensorEq(result, fl::full({3, 3}, 5.0f));
}

TEST(CPUTensorTest, asyncNativeCumsum) {
  auto a = fl::Tensor::fromVector<float>({4}, {1, 2, 3, 4});
  auto scaled = a * 2.0;          // async: {2, 4, 6, 8}
  auto cs = fl::cumsum(scaled, 0); // async native cumsum: {2, 6, 12, 20}
  auto result = cs - 1.0;          // async: {1, 5, 11, 19}
  assertCPUTensorEq(
      result, fl::Tensor::fromVector<float>({4}, {1, 5, 11, 19}));
}

TEST(CPUTensorTest, asyncNativeSort) {
  auto a = fl::Tensor::fromVector<float>({4}, {3, 1, 4, 2});
  auto scaled = a * 2.0;         // async: {6, 2, 8, 4}
  auto sorted = fl::sort(scaled, 0, fl::SortMode::Ascending);  // {2, 4, 6, 8}
  auto result = sorted + 100.0;  // {102, 104, 106, 108}
  assertCPUTensorEq(
      result, fl::Tensor::fromVector<float>({4}, {102, 104, 106, 108}));
}

TEST(CPUTensorTest, asyncNativePad) {
  auto a = fl::Tensor::fromVector<float>({3}, {1, 2, 3});
  auto scaled = a * 2.0;          // async: {2, 4, 6}
  auto padded = fl::pad(scaled, {{1, 1}}, fl::PadType::Constant);
  // async: {0, 2, 4, 6, 0}
  auto result = padded + 1.0;     // async: {1, 3, 5, 7, 1}
  assertCPUTensorEq(
      result, fl::Tensor::fromVector<float>({5}, {1, 3, 5, 7, 1}));
}

TEST(CPUTensorTest, asyncNativeIsnan) {
  auto a = fl::Tensor::fromVector<float>({4}, {1, 2, 3, 4});
  auto denom = fl::Tensor::fromVector<float>({4}, {1, 0, 1, 0});
  auto divided = a / denom;       // async: {1, inf, 3, inf}
  auto zero = fl::Tensor::fromVector<float>({4}, {0, 0, 0, 0});
  auto nans = divided * zero;     // {0, nan, 0, nan}
  auto result = fl::isnan(nans);  // async native: {0, 1, 0, 1}
  auto resultF32 = result.astype(fl::dtype::f32);
  assertCPUTensorEq(
      resultF32, fl::Tensor::fromVector<float>({4}, {0, 1, 0, 1}));
}

TEST(CPUTensorTest, asyncNativeSortWithIndices) {
  auto a = fl::Tensor::fromVector<float>({4}, {3, 1, 4, 2});
  fl::Tensor values, indices;
  fl::sort(values, indices, a, 0, fl::SortMode::Ascending);
  // values: {1, 2, 3, 4}, indices: {1, 3, 0, 2}
  auto valResult = values + 10.0;  // {11, 12, 13, 14}
  assertCPUTensorEq(
      valResult, fl::Tensor::fromVector<float>({4}, {11, 12, 13, 14}));
  assertCPUTensorEq(
      indices, fl::Tensor::fromVector<int>({4}, {1, 3, 0, 2}));
}

TEST(CPUTensorTest, asyncNativeFloorCeil) {
  auto a = fl::Tensor::fromVector<float>({4}, {1.3, 2.7, -0.5, 3.9});
  auto scaled = a + 0.0;  // force through async pipeline
  auto floored = fl::floor(scaled);
  auto ceiled = fl::ceil(scaled);
  assertCPUTensorEq(
      floored, fl::Tensor::fromVector<float>({4}, {1, 2, -1, 3}));
  assertCPUTensorEq(
      ceiled, fl::Tensor::fromVector<float>({4}, {2, 3, 0, 4}));
}

TEST(CPUTensorTest, asyncNativeSinCosLog1p) {
  // Verify sin, cos, log1p chain through async pipeline correctly
  auto a = fl::full({4}, 0.0f);
  auto sinA = fl::sin(a);   // sin(0) = 0
  auto cosA = fl::cos(a);   // cos(0) = 1
  auto result = sinA + cosA; // 0 + 1 = 1
  assertCPUTensorEq(
      result, fl::full({4}, 1.0f));

  auto b = fl::full({4}, 0.0f);
  auto log1pB = fl::log1p(b); // log1p(0) = 0
  assertCPUTensorEq(log1pB, fl::full({4}, 0.0f));
}

TEST(CPUTensorTest, Bf16Creation) {
  auto t = fl::full({3, 3}, 2.5f, fl::dtype::bf16);
  ASSERT_EQ(t.type(), fl::dtype::bf16);
  ASSERT_TRUE(fl::allClose(
      t.astype(fl::dtype::f32), fl::full({3, 3}, 2.5f, fl::dtype::f32), 0.01));

  auto s = fl::fromScalar(3.0f, fl::dtype::bf16);
  ASSERT_EQ(s.type(), fl::dtype::bf16);
  ASSERT_TRUE(fl::allClose(
      s.astype(fl::dtype::f32), fl::fromScalar(3.0f, fl::dtype::f32), 0.01));
}

TEST(CPUTensorTest, Bf16Astype) {
  auto f32 = fl::rand({4, 4});
  auto bf = f32.astype(fl::dtype::bf16);
  ASSERT_EQ(bf.type(), fl::dtype::bf16);
  auto back = bf.astype(fl::dtype::f32);
  ASSERT_TRUE(fl::allClose(f32, back, 0.01));
}

TEST(CPUTensorTest, Bf16BinaryOps) {
  auto a = fl::full({3, 3}, 2.0f, fl::dtype::bf16);
  auto b = fl::full({3, 3}, 3.0f, fl::dtype::bf16);

  auto sum = a + b;
  ASSERT_EQ(sum.type(), fl::dtype::bf16);
  ASSERT_TRUE(fl::allClose(
      sum.astype(fl::dtype::f32), fl::full({3, 3}, 5.0f), 0.01));

  auto diff = b - a;
  ASSERT_TRUE(fl::allClose(
      diff.astype(fl::dtype::f32), fl::full({3, 3}, 1.0f), 0.01));

  auto prod = a * b;
  ASSERT_TRUE(fl::allClose(
      prod.astype(fl::dtype::f32), fl::full({3, 3}, 6.0f), 0.01));

  auto quot = b / a;
  ASSERT_TRUE(fl::allClose(
      quot.astype(fl::dtype::f32), fl::full({3, 3}, 1.5f), 0.01));
}

TEST(CPUTensorTest, Bf16UnaryOps) {
  auto a = fl::full({3, 3}, 4.0f, fl::dtype::bf16);
  auto s = fl::sqrt(a);
  ASSERT_EQ(s.type(), fl::dtype::bf16);
  ASSERT_TRUE(fl::allClose(
      s.astype(fl::dtype::f32), fl::full({3, 3}, 2.0f), 0.01));

  auto e = fl::exp(fl::full({3, 3}, 0.0f, fl::dtype::bf16));
  ASSERT_TRUE(fl::allClose(
      e.astype(fl::dtype::f32), fl::full({3, 3}, 1.0f), 0.01));

  auto neg = fl::negative(a);
  ASSERT_TRUE(fl::allClose(
      neg.astype(fl::dtype::f32), fl::full({3, 3}, -4.0f), 0.01));
}

TEST(CPUTensorTest, Bf16MatmulReduction) {
  auto a = fl::full({4, 3}, 1.0f, fl::dtype::bf16);
  auto b = fl::full({3, 2}, 1.0f, fl::dtype::bf16);
  auto c = fl::matmul(a, b);
  ASSERT_TRUE(fl::allClose(
      c.astype(fl::dtype::f32), fl::full({4, 2}, 3.0f), 0.01));

  auto s = fl::sum(a, {1});
  ASSERT_TRUE(fl::allClose(
      s.astype(fl::dtype::f32), fl::full({4}, 3.0f), 0.01));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  fl::setDefaultTensorType<CPUTensor>();
  return RUN_ALL_TESTS();
}
