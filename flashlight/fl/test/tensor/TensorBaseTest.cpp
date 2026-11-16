/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/test/TestInit.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/Tensor.h"

using namespace ::testing;
using namespace fl;

TEST(TensorBaseTest, DefaultConstruction) {
  Tensor t;
  ASSERT_EQ(t.shape(), Shape({0}));
  ASSERT_EQ(t.type(), fl::dtype::f32);

  Tensor u({1, 2, 3});
  ASSERT_EQ(u.shape(), Shape({1, 2, 3}));
  ASSERT_EQ(u.type(), fl::dtype::f32);
  Tensor x({0, 3});
  ASSERT_EQ(x.shape(), Shape({0, 3}));

  Tensor q(fl::dtype::f64);
  ASSERT_EQ(q.shape(), Shape({0}));
  ASSERT_EQ(q.type(), fl::dtype::f64);

  if (fl::defaultTensorBackend().supportsDataType(fl::dtype::u64)) {
    Tensor v({4, 5, 6}, fl::dtype::u64);
    ASSERT_EQ(v.shape(), Shape({4, 5, 6}));
    ASSERT_EQ(v.type(), fl::dtype::u64);
  }
}

TEST(TensorBaseTest, CopyConstruction) {
  Shape shape{2, 2};
  auto x = fl::full(shape, 0);
  auto y = x; // actual copy (implementation may be CoW)

  ASSERT_TRUE(allClose(x, fl::full(shape, 0)));
  ASSERT_TRUE(allClose(y, fl::full(shape, 0)));
  x += 23; // affects both tensors
  ASSERT_TRUE(allClose(x, fl::full(shape, 23)));
  ASSERT_TRUE(allClose(y, fl::full(shape, 0)));
}

TEST(TensorBaseTest, MoveConstruction) {
  Shape shape{2, 2};
  auto x = fl::full(shape, 0);
  auto y = x(span, span); // view of x

  auto z = std::move(x); // `z` takes over `x`'s data
  // TODO the following line (or any read to `y`, as it seems) promotes view to
  // copy; to avoid this, we must update impl of `assign`
  // ASSERT_TRUE(allClose(y, fl::full(shape, 0)));
  ASSERT_TRUE(allClose(z, fl::full(shape, 0)));

  z += 42; // `y` is now a view of `z`, so it's affected
  ASSERT_TRUE(allClose(y, fl::full(shape, 42)));
  ASSERT_TRUE(allClose(z, fl::full(shape, 42)));
}

TEST(TensorBaseTest, AssignmentOperatorLvalueWithRvalue) {
  Shape shape{2, 2};
  auto x = fl::full({2, 2}, 0);
  auto y = x(span, span);

  // view as a lvalue cannot be used to update original tensor
  y = fl::full({2, 2}, 42); // `x` isn't affected
  y += 1; // `x` isn't affected
  ASSERT_TRUE(allClose(x, fl::full(shape, 0)));
  ASSERT_TRUE(allClose(y, fl::full(shape, 43)));
}

TEST(TensorBaseTest, AssignmentOperatorLvalueWithLvalue) {
  Shape shape{2, 2};
  auto x = fl::full({2, 2}, 0);
  auto y = x(span, span);
  auto z = fl::full({2, 2}, 1);

  y = z; // `x` is a copy of `z` now (impl may be CoW)
  y += 1; // `z` isn't affected
  ASSERT_TRUE(allClose(x, fl::full(shape, 0)));
  ASSERT_TRUE(allClose(y, fl::full(shape, 2)));
  ASSERT_TRUE(allClose(z, fl::full(shape, 1)));
}

TEST(TensorBaseTest, AssignmentOperatorRvalueWithRvalue) {
  Shape shape{2, 2};
  auto type = dtype::f32;
  auto x = fl::full({2, 2}, 0, type);
  auto y = x(span, span);

  x(0, span) = fl::full({2}, 1); // `x` is updated by copying from rhs data
  // Row-major: x(0, span) sets row 0 to [1,1]; data = {1,1,0,0}
  auto res = fl::Tensor::fromVector<float>(shape, {1, 1, 0, 0}, type);
  ASSERT_TRUE(allClose(x, res));
  ASSERT_TRUE(allClose(y, res));
}

TEST(TensorBaseTest, AssignmentOperatorRvalueWithLvalue) {
  Shape shape{2, 2};
  auto type = dtype::f32;
  auto x = fl::full(shape, 0, type);
  auto y = x(span, span); // view of `x`
  auto z = fl::full({2}, 1, type);

  x(span, 1) = z; // `x` is updated by copying from `z`'s data
  x += 1; // `z` isn't affected
  // Row-major: x(span,1) sets col 1 to [1,1]; after +=1: data = {1,2,1,2}
  auto res = fl::Tensor::fromVector<float>(shape, {1, 2, 1, 2}, type);
  ASSERT_TRUE(allClose(x, res));
  ASSERT_TRUE(allClose(y, res));
  ASSERT_TRUE(allClose(z, fl::full({2}, 1, type)));
}

TEST(TensorBaseTest, Metadata) {
  int s = 9;
  auto t = fl::rand({s, s});
  ASSERT_EQ(t.elements(), s * s);
  ASSERT_FALSE(t.isEmpty());
  ASSERT_EQ(t.bytes(), s * s * sizeof(float));

  Tensor e;
  ASSERT_EQ(e.elements(), 0);
  ASSERT_TRUE(e.isEmpty());
  ASSERT_FALSE(e.isSparse());
  ASSERT_FALSE(e.isLocked());
}

TEST(TensorBaseTest, fromScalar) {
  Tensor a = fromScalar(3.14, fl::dtype::f32);
  ASSERT_EQ(a.elements(), 1);
  ASSERT_EQ(a.ndim(), 0);
  ASSERT_FALSE(a.isEmpty());
  ASSERT_EQ(a.shape(), Shape({}));
}

TEST(TensorBaseTest, Empty) {
  auto a = fl::empty({3, 4}, fl::dtype::f32);
  ASSERT_EQ(a.shape(), Shape({3, 4}));
  ASSERT_EQ(a.type(), fl::dtype::f32);
  ASSERT_EQ(a.elements(), 12);
  ASSERT_FALSE(a.isEmpty());

  auto b = fl::empty({5}, fl::dtype::s32);
  ASSERT_EQ(b.shape(), Shape({5}));
  ASSERT_EQ(b.type(), fl::dtype::s32);
  ASSERT_EQ(b.elements(), 5);

  auto c = fl::empty({2, 3, 4}, fl::dtype::f32);
  ASSERT_EQ(c.shape(), Shape({2, 3, 4}));
  ASSERT_EQ(c.elements(), 24);

  // Writing to empty tensor should work
  c = 1.0;
  ASSERT_TRUE(allClose(c, fl::full({2, 3, 4}, 1.0)));

  // Scalar shape
  auto d = fl::empty({}, fl::dtype::f32);
  ASSERT_EQ(d.shape(), Shape({}));
  ASSERT_EQ(d.elements(), 1);
  ASSERT_EQ(d.ndim(), 0);
}

TEST(TensorBaseTest, string) {
  // Different backends might print tensors differently - check for consistency
  // across two identical tensors
  auto a = fl::full({3, 4, 5}, 6.);
  auto b = fl::full({3, 4, 5}, 6.);
  ASSERT_EQ(a.toString(), b.toString());

  std::stringstream ssa, ssb;
  ssa << a;
  ssb << b;
  ASSERT_EQ(ssa.str(), ssb.str());
}

TEST(TensorBaseTest, AssignmentOperators) {
  auto a = fl::full({3, 3}, 1.);
  a += 2;
  ASSERT_TRUE(allClose(a, fl::full({3, 3}, 3.)));
  a -= 1;
  ASSERT_TRUE(allClose(a, fl::full({3, 3}, 2.)));
  a *= 8;
  ASSERT_TRUE(allClose(a, fl::full({3, 3}, 16.)));
  a /= 4;
  ASSERT_TRUE(allClose(a, fl::full({3, 3}, 4.)));

  a = fl::full({4, 4}, 7.);
  ASSERT_TRUE(allClose(a, fl::full({4, 4}, 7.)));
  auto b = a;
  ASSERT_TRUE(allClose(b, fl::full({4, 4}, 7.)));
  a = 6.;
  ASSERT_TRUE(allClose(a, fl::full({4, 4}, 6.)));

  a = fl::full({5, 6, 7}, 8.);
  ASSERT_TRUE(allClose(a, fl::full({5, 6, 7}, 8.)));
}

TEST(TensorBaseTest, CopyOperators) {
  auto a = fl::full({3, 3}, 1.);
  auto b = a;
  a += 1;
  ASSERT_TRUE(allClose(b, fl::full({3, 3}, 1.)));
  ASSERT_TRUE(allClose(a, fl::full({3, 3}, 2.)));

  auto c = a.copy();
  a += 1;
  ASSERT_TRUE(allClose(a, fl::full({3, 3}, 3.)));
  ASSERT_TRUE(allClose(c, fl::full({3, 3}, 2.)));
}

TEST(TensorBaseTest, ConstructFromData) {
  // Tensor::fromVector
  float val = 3.;
  std::vector<float> vec(100, val);
  fl::Shape s = {10, 10};
  ASSERT_TRUE(allClose(fl::Tensor::fromVector(s, vec), fl::full(s, val)));

  ASSERT_TRUE(allClose(
      fl::Tensor::fromBuffer(s, vec.data(), fl::MemoryLocation::Host),
      fl::full(s, val)));

  std::vector<float> ascending = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  auto t = fl::Tensor::fromVector({3, 4}, ascending);
  ASSERT_EQ(t.type(), fl::dtype::f32);
  // Row-major: shape {3, 4} stores data row-by-row, so element i is at (i/4, i%4)
  for (int i = 0; i < ascending.size(); ++i) {
    ASSERT_FLOAT_EQ(t(i / 4, i % 4).scalarSync<float>(), ascending[i]);
  }

  // TODO: add fixtures/check stuff
  std::vector<int> intV = {1, 2, 3};
  ASSERT_EQ(fl::Tensor::fromVector({3}, intV).type(), fl::dtype::s32);
  ASSERT_EQ(
      fl::Tensor::fromVector<float>({5}, {0., 1., 2., 3., 4.}).type(),
      fl::dtype::f32);

  std::vector<float> flat = {0, 1, 2, 3, 4, 5, 6, 7};
  unsigned size = flat.size();
  ASSERT_EQ(fl::Tensor::fromVector(flat).shape(), Shape({size}));

  // Tensor::fromArray
  constexpr unsigned arrFSize = 5;
  std::array<float, arrFSize> arrF = {1, 2, 3, 4, 5};
  auto tArrF = Tensor::fromArray(arrF);
  ASSERT_EQ(tArrF.type(), fl::dtype::f32);
  ASSERT_EQ(tArrF.shape(), Shape({arrFSize}));
  auto tArrD = Tensor::fromArray({arrFSize}, arrF, fl::dtype::f64);
  ASSERT_EQ(tArrD.type(), fl::dtype::f64);

  if (fl::defaultTensorBackend().supportsDataType(fl::dtype::u32)) {
    constexpr unsigned arrISize = 8;
    std::array<unsigned, arrISize> arrI = {1, 2, 3, 4, 5, 6, 7, 8};
    auto tArrI = Tensor::fromArray(arrI);
    ASSERT_EQ(tArrI.type(), fl::dtype::u32);
    ASSERT_EQ(tArrI.shape(), Shape({arrISize}));
    auto tArrIs = Tensor::fromArray({2, 4}, arrI);
    ASSERT_EQ(tArrIs.shape(), Shape({2, 4}));
  }
}

TEST(TensorBaseTest, reshape) {
  auto a = fl::full({4, 4}, 3.);
  auto b = fl::reshape(a, Shape({8, 2}));
  ASSERT_EQ(b.shape(), Shape({8, 2}));
  ASSERT_TRUE(allClose(a, fl::reshape(b, {4, 4})));

  ASSERT_THROW(fl::reshape(a, {}), std::exception);
}

TEST(TensorBaseTest, squeeze) {
  auto a = fl::rand({3, 1, 4, 1});
  ASSERT_EQ(fl::squeeze(a).shape(), Shape({3, 4}));
  ASSERT_EQ(fl::squeeze(a, 1).shape(), Shape({3, 4, 1}));
  ASSERT_EQ(fl::squeeze(a, 3).shape(), Shape({3, 1, 4}));
  ASSERT_THROW(fl::squeeze(a, 0), std::exception); // dim 0 != 1
  ASSERT_THROW(fl::squeeze(a, 5), std::exception); // out of range

  auto b = fl::rand({1, 1, 1});
  ASSERT_EQ(fl::squeeze(b).shape(), Shape({})); // scalar

  auto c = fl::rand({5, 3});
  ASSERT_EQ(fl::squeeze(c).shape(), Shape({5, 3})); // no size-1 dims

  // Value preservation: squeeze is a reshape
  auto sq_in = fl::rand({3, 1, 4});
  auto squeezed = fl::squeeze(sq_in, 1);
  ASSERT_EQ(squeezed.shape(), Shape({3, 4}));
  ASSERT_TRUE(allClose(squeezed, fl::reshape(sq_in, {3, 4})));
}

TEST(TensorBaseTest, unsqueeze) {
  auto a = fl::rand({3, 4});
  ASSERT_EQ(fl::unsqueeze(a, 0).shape(), Shape({1, 3, 4}));
  ASSERT_EQ(fl::unsqueeze(a, 1).shape(), Shape({3, 1, 4}));
  ASSERT_EQ(fl::unsqueeze(a, 2).shape(), Shape({3, 4, 1}));
  ASSERT_THROW(fl::unsqueeze(a, 3), std::exception); // out of range

  auto b = fl::rand({5});
  ASSERT_EQ(fl::unsqueeze(b, 0).shape(), Shape({1, 5}));
  ASSERT_EQ(fl::unsqueeze(b, 1).shape(), Shape({5, 1}));

  // Value preservation: unsqueeze + squeeze = identity
  auto usq_in = fl::rand({3, 4});
  auto expanded = fl::unsqueeze(usq_in, 1);
  ASSERT_EQ(expanded.shape(), Shape({3, 1, 4}));
  ASSERT_TRUE(allClose(fl::squeeze(expanded, 1), usq_in));
}

TEST(TensorBaseTest, transpose) {
  // 2D transpose with element verification
  auto a2d = fl::iota({3, 4});
  auto a2dt = fl::transpose(a2d);
  ASSERT_EQ(a2dt.shape(), Shape({4, 3}));
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      ASSERT_FLOAT_EQ(a2d(i, j).scalarSync<float>(), a2dt(j, i).scalarSync<float>());

  // 4D transpose with element verification
  auto b = fl::rand({4, 5, 6, 7});
  auto bt = fl::transpose(b, {2, 0, 1, 3});
  ASSERT_EQ(bt.shape(), Shape({6, 4, 5, 7}));
  // Spot check: b(i,j,k,l) == bt(k,i,j,l)
  ASSERT_FLOAT_EQ(
      b(0, 0, 0, 0).scalarSync<float>(), bt(0, 0, 0, 0).scalarSync<float>());
  ASSERT_FLOAT_EQ(
      b(1, 2, 3, 4).scalarSync<float>(), bt(3, 1, 2, 4).scalarSync<float>());
  ASSERT_FLOAT_EQ(
      b(3, 4, 5, 6).scalarSync<float>(), bt(5, 3, 4, 6).scalarSync<float>());

  ASSERT_THROW(fl::transpose(fl::rand({3, 4, 5}), {0, 1}), std::exception);
  ASSERT_THROW(
      fl::transpose(fl::rand({2, 4, 6, 8}), {1, 0, 2}), std::exception);
  ASSERT_THROW(
      fl::transpose(fl::rand({2, 4, 6, 8}), {1, 0, 2, 4}), std::exception);

  auto a = fl::rand({4});
  ASSERT_TRUE(allClose(fl::transpose(a), a));

  ASSERT_EQ(fl::transpose(fl::rand({5, 6, 7})).shape(), Shape({7, 6, 5}));
  ASSERT_EQ(fl::transpose(fl::rand({5, 6, 1, 7})).shape(), Shape({7, 1, 6, 5}));
  ASSERT_EQ(fl::transpose(fl::rand({1, 1})).shape(), Shape({1, 1}));
  ASSERT_EQ(
      fl::transpose(fl::rand({7, 2, 1, 3}), {0, 2, 1, 3}).shape(),
      Shape({7, 1, 2, 3}));
}

TEST(TensorBaseTest, tile) {
  auto a = fl::full({4, 4}, 3.);
  auto tiled = fl::tile(a, {2, 2});
  ASSERT_EQ(tiled.shape(), Shape({8, 8}));
  ASSERT_TRUE(allClose(tiled, fl::full({8, 8}, 3.)));
  ASSERT_EQ(fl::tile(a, {}).shape(), a.shape());

  auto s = fl::fromScalar(3.14);
  ASSERT_EQ(fl::tile(s, {3, 3}).shape(), Shape({3, 3}));
  ASSERT_EQ(fl::tile(s, {}).shape(), s.shape());
}

TEST(TensorBaseTest, concatenate) {
  auto a = fl::full({3, 3}, 1.);
  auto b = fl::full({3, 3}, 2.);
  auto c = fl::full({3, 3}, 3.);
  ASSERT_TRUE(
      allClose(fl::concatenate(0, a, b, c), fl::concatenate({a, b, c})));
  auto out = fl::concatenate(0, a, b, c);
  ASSERT_EQ(out.shape(), Shape({9, 3}));

  // Empty tensors
  ASSERT_EQ(fl::concatenate(0, Tensor(), Tensor()).shape(), Shape({0}));
  // Row-major: concatenating empty 1D tensors along axis 2 extends to 3D
  ASSERT_EQ(fl::concatenate(2, Tensor(), Tensor()).shape(), Shape({1, 1, 0}));

  // More than 4 tensors (tests iterative concatenation)
  const float val = 3.;
  const int axis = 0;
  std::vector<Tensor> many;
  for (int i = 0; i < 6; i++) {
    many.push_back(fl::full({4, 2}, val));
  }
  auto t = fl::concatenate(many, axis);
  ASSERT_EQ(t.shape(), Shape({24, 2}));
  ASSERT_TRUE(allClose(t, fl::full({24, 2}, val)));

  // Even more tensors
  std::vector<Tensor> lots;
  for (int i = 0; i < 10; i++) {
    lots.push_back(fl::full({2, 3}, static_cast<float>(i)));
  }
  auto u = fl::concatenate(lots, 0);
  ASSERT_EQ(u.shape(), Shape({20, 3}));
}

TEST(TensorBaseTest, concatenateEdgeCases) {
  // Single tensor: identity
  auto a = fl::full({3, 4}, 5.f);
  ASSERT_TRUE(allClose(fl::concatenate({a}, 0), a));

  // Concatenate along axis 1
  auto x = fl::full({2, 3}, 1.f);
  auto y = fl::full({2, 5}, 2.f);
  auto cat1 = fl::concatenate(1, x, y);
  ASSERT_EQ(cat1.shape(), Shape({2, 8}));
  ASSERT_TRUE(allClose(cat1(fl::span, fl::range(3)), x));
  ASSERT_TRUE(allClose(cat1(fl::span, fl::range(3, 8)), y));

  // Concatenate 1-D tensors with value verification
  auto p = Tensor::fromVector<float>({3}, {1, 2, 3});
  auto q = Tensor::fromVector<float>({2}, {4, 5});
  auto cat1d = fl::concatenate(0, p, q);
  ASSERT_EQ(cat1d.shape(), Shape({5}));
  ASSERT_TRUE(allClose(cat1d, Tensor::fromVector<float>({5}, {1, 2, 3, 4, 5})));

  // Concatenate tensors with different values along axis 0, verify content
  auto t1 = fl::full({2, 2}, 10.f);
  auto t2 = fl::full({3, 2}, 20.f);
  auto catv = fl::concatenate(0, t1, t2);
  ASSERT_EQ(catv.shape(), Shape({5, 2}));
  ASSERT_TRUE(allClose(catv(fl::range(0, 2), fl::span), t1));
  ASSERT_TRUE(allClose(catv(fl::range(2, 5), fl::span), t2));

  // Concatenate along higher axis
  auto m = fl::full({2, 3, 4}, 1.f);
  auto n = fl::full({2, 3, 2}, 2.f);
  auto cat2 = fl::concatenate(2, m, n);
  ASSERT_EQ(cat2.shape(), Shape({2, 3, 6}));
  ASSERT_TRUE(allClose(cat2(fl::span, fl::span, fl::range(4)), m));
  ASSERT_TRUE(allClose(cat2(fl::span, fl::span, fl::range(4, 6)), n));
}

TEST(TensorBaseTest, nonzero) {
  std::vector<int> idxs = {0, 1, 4, 9, 11, 23, 55, 82, 91};
  auto a = fl::full({10, 10}, 1, fl::dtype::s32);
  for (const auto idx : idxs) {
    a(idx / 10, idx % 10) = 0;
  }
  auto indices = fl::nonzero(a);
  int nnz = a.elements() - idxs.size();
  ASSERT_EQ(indices.shape(), Shape({nnz}));
  ASSERT_TRUE(
      allClose(a.flatten()(indices), fl::full({nnz}, 1, fl::dtype::s32)));
}

TEST(TensorBaseTest, flatten) {
  unsigned s = 6;
  auto a = fl::full({s, s, s}, 2.);
  // Member function
  auto flat = a.flatten();
  ASSERT_EQ(flat.shape(), Shape({s * s * s}));
  ASSERT_TRUE(allClose(flat, fl::full({s * s * s}, 2.)));
  // Free function
  auto flatFree = fl::flatten(a);
  ASSERT_EQ(flatFree.shape(), Shape({s * s * s}));
  ASSERT_TRUE(allClose(flatFree, flat));
}

TEST(TensorBaseTest, pad) {
  auto t = fl::rand({5, 2});
  auto zeroPadded = fl::pad(t, {{1, 2}, {3, 4}});
  auto zeroTest = fl::concatenate(
      1,
      fl::full({8, 3}, 0.),
      fl::concatenate(0, fl::full({1, 2}, 0.), t, fl::full({2, 2}, 0.)),
      fl::full({8, 4}, 0.));
  ASSERT_TRUE(allClose(zeroPadded, zeroTest));

  auto edgePadded = fl::pad(t, {{1, 1}, {2, 2}}, PadType::Edge);
  auto vertTiled = fl::concatenate(
      0,
      fl::reshape(t(0, fl::span), {1, 2}),
      t,
      fl::reshape(t(t.dim(0) - 1, fl::span), {1, 2}));
  // Keep as 2D columns for axis-1 concatenation
  auto nRows = vertTiled.dim(0);
  auto vTiled0 = fl::reshape(vertTiled(fl::span, 0), {nRows, 1});
  auto vTiled1 = fl::reshape(vertTiled(fl::span, 1), {nRows, 1});
  ASSERT_TRUE(allClose(
      edgePadded,
      fl::concatenate(
          1,
          fl::tile(vTiled0, {1, 3}),
          fl::tile(vTiled1, {1, 3}))));

  auto symmetricPadded = fl::pad(t, {{1, 1}, {2, 2}}, PadType::Symmetric);
  auto symRef = fl::concatenate(
          1,
          vTiled1,
          vTiled0,
          vTiled0,
          fl::concatenate(1, vTiled1, vTiled1, vTiled0));
  ASSERT_TRUE(allClose(symmetricPadded, symRef));
}

TEST(TensorBaseTest, padSymmetricSmall) {
  // Small known tensor for symmetric (numpy "symmetric") padding.
  // The edge element IS repeated in the reflection:
  //   pattern: ..., b, a, | a, b, c, d, | d, c, ...
  auto t = Tensor::fromVector<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  auto padded = fl::pad(t, {{1, 1}, {1, 1}}, PadType::Symmetric);
  ASSERT_EQ(padded.shape(), Shape({4, 5}));
  // For n=2 rows, pad 1 is indistinguishable from edge. Use a larger tensor
  // to verify the symmetric (edge-included) reflection pattern is distinct.
  // With pad {{1,1},{1,1}}: symmetric pattern = edge pattern when pad <= 1
  // and dim = 2, so use a simpler assertion here.
  auto expected = Tensor::fromVector<float>(
      {4, 5}, {1, 1, 2, 3, 3, 1, 1, 2, 3, 3, 4, 4, 5, 6, 6, 4, 4, 5, 6, 6});
  ASSERT_TRUE(allClose(padded, expected));

  // Larger tensor (3x4) with pad {{2,2},{2,2}} distinguishes symmetric from edge.
  // Input: [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
  // Symmetric (numpy "symmetric") mirrors INCLUDING the edge element:
  //   ..., b, a, | a, b, c, d, | d, c, ...
  auto t2 = Tensor::fromVector<float>({3, 4},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  auto padded2 = fl::pad(t2, {{2, 2}, {2, 2}}, PadType::Symmetric);
  ASSERT_EQ(padded2.shape(), Shape({7, 8}));
  auto expected2 = Tensor::fromVector<float>({7, 8}, {
      6,  5,  5,  6,  7,  8,  8,  7,
      2,  1,  1,  2,  3,  4,  4,  3,
      2,  1,  1,  2,  3,  4,  4,  3,
      6,  5,  5,  6,  7,  8,  8,  7,
      10, 9,  9,  10, 11, 12, 12, 11,
      10, 9,  9,  10, 11, 12, 12, 11,
      6,  5,  5,  6,  7,  8,  8,  7});
  ASSERT_TRUE(allClose(padded2, expected2));
}

TEST(TensorBaseTest, padEdgeSmall) {
  // Small known tensor for edge padding.
  // Input (2x3): [[1,2,3],[4,5,6]]
  // Edge pad {{1,1},{1,1}} → 4x5:
  //   [[1, 1, 2, 3, 3],
  //    [1, 1, 2, 3, 3],
  //    [4, 4, 5, 6, 6],
  //    [4, 4, 5, 6, 6]]
  auto t = Tensor::fromVector<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  auto padded = fl::pad(t, {{1, 1}, {1, 1}}, PadType::Edge);
  ASSERT_EQ(padded.shape(), Shape({4, 5}));
  auto expected = Tensor::fromVector<float>(
      {4, 5}, {1, 1, 2, 3, 3, 1, 1, 2, 3, 3, 4, 4, 5, 6, 6, 4, 4, 5, 6, 6});
  ASSERT_TRUE(allClose(padded, expected));
}

TEST(TensorBaseTest, astype) {
  auto a = fl::rand({3, 3});
  ASSERT_EQ(a.type(), dtype::f32);
  ASSERT_EQ(a.astype(dtype::f64).type(), dtype::f64);
}

TEST(TensorBaseTest, astypeConversions) {
  // f32 -> s32 truncates
  auto t = fl::full({4}, 3.7f, fl::dtype::f32);
  auto ti = t.astype(fl::dtype::s32);
  ASSERT_EQ(ti.type(), fl::dtype::s32);
  ASSERT_EQ(ti.scalarSync<int>(), 3);

  // s32 -> f32
  auto tf = ti.astype(fl::dtype::f32);
  ASSERT_EQ(tf.type(), fl::dtype::f32);
  ASSERT_FLOAT_EQ(tf.scalarSync<float>(), 3.0f);

  // f32 -> f64 -> f32 roundtrip
  auto td = t.astype(fl::dtype::f64);
  ASSERT_EQ(td.type(), fl::dtype::f64);
  auto tdf = td.astype(fl::dtype::f32);
  ASSERT_NEAR(tdf.scalarSync<float>(), 3.7f, 1e-5);

  // f32 -> b8 (nonzero -> 1)
  auto tb = t.astype(fl::dtype::b8);
  ASSERT_EQ(tb.type(), fl::dtype::b8);
  auto tbf = tb.astype(fl::dtype::f32);
  ASSERT_FLOAT_EQ(tbf.scalarSync<float>(), 1.0f);
}

TEST(TensorBaseTest, opsOnNonF32) {
  auto ti = fl::full({4}, 3.7f).astype(fl::dtype::s32); // [3, 3, 3, 3]
  auto result = ti + ti;
  ASSERT_TRUE(allClose(result.astype(fl::dtype::f32), fl::full({4}, 6.0f)));
}

TEST(TensorBaseTest, where) {
  auto a = Tensor::fromVector<int>({2, 5}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto out = fl::where(a < 5, a, a * 10);
  a(a >= 5) *= 10;
  ASSERT_TRUE(allClose(out, a));
  auto outC = fl::where(a < 5, a, 3);
  a(a >= 5) = 3;
  ASSERT_TRUE(allClose(outC, a));
  auto outC2 = fl::where(a < 5, 3, a);
  a(a < 5) = 3;
  ASSERT_TRUE(allClose(outC2, a));

  // non b8-type vector throws
  EXPECT_THROW(
      fl::where((a < 5).astype(fl::dtype::f32), a, a * 10), std::exception);
}

TEST(TensorBaseTest, topk) {
  // AF topk only works on AF axis 0, which is the last FL dim in row-major.
  // Use arange along the last dim (axis=1) so topk along axis=1 works.
  auto a = fl::arange({2, 10}, /* seqDim = */ 1);
  Tensor values;
  Tensor indices;
  fl::topk(values, indices, a, /* k = */ 3, /* axis = */ 1); // descending sort
  // Row-major: each row has [0..9], top 3 descending = [9, 8, 7]
  ASSERT_TRUE(
      allClose(values, Tensor::fromVector<float>({2, 3}, {9, 8, 7, 9, 8, 7})));

  fl::topk(
      values,
      indices,
      a,
      /* k = */ 4,
      /* axis = */ 1,
      fl::SortMode::Ascending);
  ASSERT_TRUE(allClose(
      values, Tensor::fromVector<float>({2, 4}, {0, 1, 2, 3, 0, 1, 2, 3})));
}

TEST(TensorBaseTest, sort) {
  Shape dims({10, 2});
  auto a = fl::arange(dims);
  auto sorted = fl::sort(a, /* axis = */ 0, SortMode::Descending);

  // Build expected descending values [9, 8, ..., 0] without indexed assignment
  auto expected = fl::full({dims[0]}, static_cast<float>(dims[0] - 1), a.type())
      - fl::arange({dims[0]}, 0, a.type());
  auto tiled = fl::tile(fl::reshape(expected, {dims[0], 1}), {1, 2});
  ASSERT_TRUE(allClose(sorted, tiled));

  ASSERT_TRUE(allClose(a, fl::sort(tiled, 0, SortMode::Ascending)));

  auto b = fl::rand({10});
  Tensor values, indices;
  fl::sort(values, indices, b, /* axis = */ 0, SortMode::Descending);
  ASSERT_TRUE(
      allClose(values, fl::sort(b, /* axis = */ 0, SortMode::Descending)));
  ASSERT_TRUE(
      allClose(fl::argsort(b, /* axis = */ 0, SortMode::Descending), indices));
}

TEST(TensorBaseTest, argsort) {
  Shape dims({10, 2});
  auto a = fl::arange(dims);
  auto sorted = fl::argsort(a, /* axis = */ 0, SortMode::Descending);

  // Use the same index dtype that argsort returns (u32 on AF, s32 on OneDNN)
  auto idxType = sorted.type();
  // Build expected descending indices [9, 8, ..., 0] without indexed assignment
  auto expected = fl::full({dims[0]}, static_cast<float>(dims[0] - 1), fl::dtype::f32)
      - fl::arange({dims[0]}, 0, fl::dtype::f32);
  expected = expected.astype(idxType);
  auto tiled = fl::tile(fl::reshape(expected, {dims[0], 1}), {1, 2});
  ASSERT_TRUE(allClose(sorted, tiled));

  ASSERT_TRUE(allClose(tiled, fl::argsort(tiled, 0, SortMode::Ascending)));
}

template <typename ScalarArgType>
void assertScalarBehavior(fl::dtype type) {
  ScalarArgType scalar = 42; // small enough for any scalar type
  auto one = fl::full({1}, scalar, type);

  if (dtype_traits<ScalarArgType>::fl_type != type) {
    ASSERT_THROW(one.template scalarSync<ScalarArgType>(), std::invalid_argument)
        << "dtype: " << type
        << ", ScalarArgType: " << dtype_traits<ScalarArgType>::getName();
    return;
  }

  if ((type == fl::dtype::f16) || (type == fl::dtype::f32) ||
      (type == fl::dtype::f64)) {
    ASSERT_FLOAT_EQ(one.template scalarSync<ScalarArgType>(), scalar)
        << "dtype: " << type
        << ", ScalarArgType: " << dtype_traits<ScalarArgType>::getName();
  } else {
    ASSERT_EQ(one.template scalarSync<ScalarArgType>(), scalar)
        << "dtype: " << type
        << ", ScalarArgType: " << dtype_traits<ScalarArgType>::getName();
  }

  auto a = fl::rand({5, 6}, type);
  ASSERT_TRUE(allClose(fl::fromScalar(a.scalarSync<ScalarArgType>(), type), a(0, 0)))
      << "dtype: " << type
      << ", ScalarArgType: " << dtype_traits<ScalarArgType>::getName();
}

TEST(TensorBaseTest, scalar) {
  auto types = {
      fl::dtype::b8,
      fl::dtype::u8,
      fl::dtype::s16,
      fl::dtype::u16,
      fl::dtype::s32,
      fl::dtype::u32,
      fl::dtype::s64,
      fl::dtype::u64,
      fl::dtype::f16,
      fl::dtype::f32,
      fl::dtype::f64};
  for (auto type : types) {
    if (!fl::defaultTensorBackend().supportsDataType(type)) {
      continue;
    }
    assertScalarBehavior<char>(type);
    assertScalarBehavior<unsigned char>(type);
    assertScalarBehavior<short>(type);
    assertScalarBehavior<unsigned short>(type);
    assertScalarBehavior<int>(type);
    assertScalarBehavior<unsigned int>(type);
    assertScalarBehavior<long>(type);
    assertScalarBehavior<unsigned long>(type);
    assertScalarBehavior<long long>(type);
    assertScalarBehavior<unsigned long long>(type);
    assertScalarBehavior<float>(type);
    assertScalarBehavior<double>(type);
  }
}

TEST(TensorBaseTest, isContiguous) {
  // Contiguous by default
  auto a = fl::rand({10, 10});
  ASSERT_TRUE(a.isContiguous());
}

TEST(TensorBaseTest, strides) {
  auto t = fl::rand({10, 10});
  // Row-major: dim 0 (outermost) has stride 10, dim 1 (innermost) has stride 1
  ASSERT_EQ(t.strides(), Shape({10, 1}));
}

TEST(TensorBaseTest, stream) {
  auto t1 = fl::rand({10, 10});
  auto t2 = -t1;
  auto t3 = t1 + t2;
  ASSERT_EQ(&t1.stream(), &t2.stream());
  ASSERT_EQ(&t1.stream(), &t3.stream());
}

TEST(TensorBaseTest, asContiguousTensor) {
  auto t = fl::rand({5, 6, 7, 8});
  auto indexed =
      t(fl::range(1, 4, 2),
        fl::range(0, 6, 2),
        fl::range(0, 6, 3),
        fl::range(0, 5, 3));

  auto contiguous = indexed.asContiguousTensor();
  // Row-major strides: last dim has stride 1, each dim to the left has
  // stride = product of all dimensions to its right.
  std::vector<Dim> strides(contiguous.ndim());
  Dim stride = 1;
  for (int i = contiguous.ndim() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= contiguous.dim(i);
  }
  ASSERT_EQ(contiguous.strides(), Shape(strides));
}

TEST(TensorBaseTest, host) {
  auto a = fl::rand({10, 10});

  float* ptr = a.hostSync<float>();
  for (int i = 0; i < a.elements(); ++i) {
    ASSERT_EQ(ptr[i], a.flatten()(i).scalarSync<float>());
  }

  float* existingBuffer = new float[100];
  a.hostSync(existingBuffer);
  for (int i = 0; i < a.elements(); ++i) {
    ASSERT_EQ(existingBuffer[i], a.flatten()(i).scalarSync<float>());
  }

  ASSERT_EQ(Tensor().hostSync<void>(), nullptr);
}

TEST(TensorBaseTest, toHostVector) {
  auto a = fl::rand({10, 10});
  auto vec = a.toHostVectorSync<float>();

  for (int i = 0; i < a.elements(); ++i) {
    ASSERT_EQ(vec[i], a.flatten()(i).scalarSync<float>());
  }

  ASSERT_EQ(Tensor().toHostVectorSync<float>().size(), 0);
}

TEST(TensorBaseTest, arange) {
  // Range/step overload
  ASSERT_TRUE(
      allClose(fl::arange(2, 10, 2), Tensor::fromVector<int>({2, 4, 6, 8})));
  ASSERT_TRUE(
      allClose(fl::arange(0, 6), Tensor::fromVector<int>({0, 1, 2, 3, 4, 5})));
  ASSERT_TRUE(allClose(
      fl::arange(0., 1.22, 0.25),
      Tensor::fromVector<float>({0., 0.25, 0.5, 0.75})));
  ASSERT_TRUE(allClose(
      fl::arange(0., 4.1), Tensor::fromVector<float>({0., 1., 2., 3.})));

  // Shape overload
  auto v = Tensor::fromVector<float>({0., 1., 2., 3.});
  ASSERT_TRUE(allClose(fl::arange({4}), v));

  ASSERT_TRUE(
      allClose(fl::arange({4, 5}), fl::tile(fl::reshape(v, {4, 1}), {1, 5})));
  ASSERT_EQ(fl::arange({4, 5}, 1).shape(), Shape({4, 5}));
  ASSERT_TRUE(allClose(
      fl::arange({4, 5}, 1),
      fl::tile(
          fl::reshape(Tensor::fromVector<float>({0., 1., 2., 3., 4.}), {1, 5}),
          {4, 1})));
  ASSERT_EQ(fl::arange({2, 6}, 0, fl::dtype::f64).type(), fl::dtype::f64);
}

TEST(TensorBaseTest, iota) {
  ASSERT_TRUE(allClose(
      fl::iota({5, 3}, {1, 2}),
      fl::tile(fl::reshape(fl::arange({15}), {5, 3}), {1, 2})));
  ASSERT_EQ(fl::iota({2, 2}, {2, 2}, fl::dtype::f64).type(), fl::dtype::f64);
  ASSERT_EQ(fl::iota({1, 10}, {5}).shape(), Shape({5, 10}));
}

TEST(TensorBaseTest, identity) {
  auto eye = fl::identity(3);
  ASSERT_EQ(eye.shape(), Shape({3, 3}));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      float expected = (i == j) ? 1.0f : 0.0f;
      ASSERT_FLOAT_EQ(eye(i, j).scalarSync<float>(), expected);
    }
  }
  ASSERT_EQ(fl::identity(4, fl::dtype::f64).type(), fl::dtype::f64);
}

TEST(TensorBaseTest, rand) {
  auto t = fl::rand({100});
  ASSERT_EQ(t.shape(), Shape({100}));
  // All values in [0, 1]
  ASSERT_TRUE(fl::all(t >= 0.).scalarSync<char>());
  ASSERT_TRUE(fl::all(t <= 1.).scalarSync<char>());
}

TEST(TensorBaseTest, randn) {
  auto t = fl::randn({1000});
  ASSERT_EQ(t.shape(), Shape({1000}));
  ASSERT_NEAR(fl::mean(t, {}).scalarSync<float>(), 0.0f, 0.2f);
}

// =================== Tensor Autograd Tests ===================

TEST(TensorBaseTest, AutogradBasicRequiresGrad) {
  auto t = fl::full({3, 4}, 2.0f);
  ASSERT_FALSE(t.requiresGrad());

  Tensor tGrad(t, /*requiresGrad=*/true);
  ASSERT_TRUE(tGrad.requiresGrad());
  ASSERT_FALSE(tGrad.isGradAvailable());

  // Copy shares autograd identity
  Tensor copy = tGrad;
  ASSERT_TRUE(copy.requiresGrad());
}

TEST(TensorBaseTest, AutogradSetRequiresGrad) {
  auto t = fl::full({2, 3}, 1.0f);
  ASSERT_FALSE(t.requiresGrad());

  t.setRequiresGrad(true);
  ASSERT_TRUE(t.requiresGrad());

  t.setRequiresGrad(false);
  ASSERT_FALSE(t.requiresGrad());
}

TEST(TensorBaseTest, AutogradAddGrad) {
  Tensor t(fl::full({2, 2}, 1.0f), true);
  auto grad = fl::full({2, 2}, 3.0f);

  t.addGrad(grad);
  ASSERT_TRUE(t.isGradAvailable());
  ASSERT_TRUE(allClose(t.grad(), fl::full({2, 2}, 3.0f)));

  // Accumulate
  t.addGrad(grad);
  ASSERT_TRUE(allClose(t.grad(), fl::full({2, 2}, 6.0f)));

  // Zero grad
  t.zeroGrad();
  ASSERT_FALSE(t.isGradAvailable());
}

TEST(TensorBaseTest, AutogradWithoutData) {
  Tensor t(fl::full({3, 4}, 5.0f), true);
  Tensor noData = t.withoutData();

  // Shares autograd identity
  ASSERT_TRUE(noData.requiresGrad());
  ASSERT_EQ(noData.shape(), t.shape());
  ASSERT_EQ(noData.type(), t.type());

  // addGrad on noData affects original
  auto grad = fl::full({3, 4}, 1.0f);
  noData.addGrad(grad);
  ASSERT_TRUE(t.isGradAvailable());
  ASSERT_TRUE(allClose(t.grad(), fl::full({3, 4}, 1.0f)));
}

TEST(TensorBaseTest, AutogradSimpleBackward) {
  // f(x) = 2*x, df/dx = 2
  Tensor x(fl::full({3}, 3.0f), true);
  auto two = fl::full({3}, 2.0f);
  auto result = x * two; // uses Tensor operator*, no autograd

  // Manual autograd: create result with gradFunc
  auto gradFunc = [](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
    auto grad = gradOutput * fl::full(gradOutput.shape(), 2.0f);
    inputs[0].addGrad(grad);
  };
  Tensor y(x * two, {x.withoutData()}, gradFunc);
  ASSERT_TRUE(y.requiresGrad());

  y.backward();
  ASSERT_TRUE(x.isGradAvailable());
  ASSERT_TRUE(allClose(x.grad(), fl::full({3}, 2.0f)));
}

TEST(TensorBaseTest, AutogradChainBackward) {
  // f(x) = (x + 1) * 3, df/dx = 3
  Tensor x(fl::full({2}, 2.0f), true);

  // Step 1: y = x + 1
  auto addGradFunc = [](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
    inputs[0].addGrad(gradOutput); // d(x+c)/dx = 1
  };
  Tensor y(x + fl::full({2}, 1.0f), {x.withoutData()}, addGradFunc);

  // Step 2: z = y * 3
  auto mulGradFunc = [](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
    auto grad = gradOutput * fl::full(gradOutput.shape(), 3.0f);
    inputs[0].addGrad(grad); // d(y*c)/dy = c
  };
  Tensor z(y * fl::full({2}, 3.0f), {y.withoutData()}, mulGradFunc);

  z.backward();
  ASSERT_TRUE(x.isGradAvailable());
  // df/dx = df/dy * dy/dx = 3 * 1 = 3
  ASSERT_TRUE(allClose(x.grad(), fl::full({2}, 3.0f)));
}

TEST(TensorBaseTest, AutogradGradHook) {
  Tensor x(fl::full({2}, 1.0f), true);
  bool hookCalled = false;
  x.registerGradHook([&hookCalled](Tensor& grad) {
    hookCalled = true;
    grad = grad * fl::full(grad.shape(), 2.0f); // double the gradient
  });

  auto gradFunc = [](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
    inputs[0].addGrad(gradOutput);
  };
  Tensor y(x * fl::full({2}, 5.0f), {x.withoutData()}, gradFunc);
  y.backward();

  ASSERT_TRUE(hookCalled);
  ASSERT_TRUE(allClose(x.grad(), fl::full({2}, 2.0f))); // 1.0 * 2 from hook
}

TEST(TensorBaseTest, AutogradNoGradInput) {
  Tensor x(fl::full({2}, 1.0f), false); // no grad
  Tensor y(fl::full({2}, 2.0f), true);

  // Only y requires grad
  auto gradFunc = [](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
    inputs[0].addGrad(gradOutput);
    inputs[1].addGrad(gradOutput);
  };
  Tensor z(x + y, {x, y.withoutData()}, gradFunc);
  ASSERT_TRUE(z.requiresGrad()); // at least one input requires grad

  z.backward();
  ASSERT_FALSE(x.isGradAvailable()); // x didn't request grad
  ASSERT_TRUE(y.isGradAvailable());  // y did
}

TEST(TensorBaseTest, AutogradNoInputsNoGrad) {
  Tensor x(fl::full({2}, 1.0f), false);
  Tensor y(fl::full({2}, 2.0f), false);

  auto gradFunc = [](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
    inputs[0].addGrad(gradOutput);
  };
  // Neither input requires grad -- output should NOT require grad
  Tensor z(x + y, {x, y}, gradFunc);
  ASSERT_FALSE(z.requiresGrad());
}

TEST(TensorTypesTest, ScalarDtype) {
  ASSERT_EQ(fl::scalarDtype(3.0), fl::dtype::s32);
  ASSERT_EQ(fl::scalarDtype(-1.0), fl::dtype::s32);
  ASSERT_EQ(fl::scalarDtype(0.0), fl::dtype::s32);
  ASSERT_EQ(fl::scalarDtype(3.14), fl::dtype::f32);
  ASSERT_EQ(fl::scalarDtype(-0.5), fl::dtype::f32);
  ASSERT_EQ(fl::scalarDtype(1e20), fl::dtype::f32); // too large for int
}

TEST(TensorTypesTest, PromoteDtypes) {
  // float > int
  ASSERT_EQ(fl::promoteDtypes(fl::dtype::s32, fl::dtype::f32), fl::dtype::f32);
  ASSERT_EQ(fl::promoteDtypes(fl::dtype::f32, fl::dtype::s32), fl::dtype::f32);
  // same type
  ASSERT_EQ(fl::promoteDtypes(fl::dtype::s32, fl::dtype::s32), fl::dtype::s32);
  ASSERT_EQ(fl::promoteDtypes(fl::dtype::f32, fl::dtype::f32), fl::dtype::f32);
  // wider wins
  ASSERT_EQ(fl::promoteDtypes(fl::dtype::f32, fl::dtype::f64), fl::dtype::f64);
  ASSERT_EQ(fl::promoteDtypes(fl::dtype::s32, fl::dtype::s64), fl::dtype::s64);
  // f16 > int
  ASSERT_EQ(fl::promoteDtypes(fl::dtype::s32, fl::dtype::f16), fl::dtype::f16);
  // bf16 promotion rules
  ASSERT_EQ(fl::promoteDtypes(fl::dtype::bf16, fl::dtype::bf16), fl::dtype::bf16);
  ASSERT_EQ(fl::promoteDtypes(fl::dtype::bf16, fl::dtype::f32), fl::dtype::f32);
  ASSERT_EQ(fl::promoteDtypes(fl::dtype::f32, fl::dtype::bf16), fl::dtype::f32);
  ASSERT_EQ(fl::promoteDtypes(fl::dtype::bf16, fl::dtype::f16), fl::dtype::f32);
  ASSERT_EQ(fl::promoteDtypes(fl::dtype::f16, fl::dtype::bf16), fl::dtype::f32);
}

TEST(TensorTypesTest, IsFloatingPoint) {
  ASSERT_TRUE(fl::isFloatingPoint(fl::dtype::f16));
  ASSERT_TRUE(fl::isFloatingPoint(fl::dtype::f32));
  ASSERT_TRUE(fl::isFloatingPoint(fl::dtype::f64));
  ASSERT_TRUE(fl::isFloatingPoint(fl::dtype::bf16));
  ASSERT_FALSE(fl::isFloatingPoint(fl::dtype::s32));
  ASSERT_FALSE(fl::isFloatingPoint(fl::dtype::u8));
  ASSERT_FALSE(fl::isFloatingPoint(fl::dtype::b8));
}

TEST(TensorTypesTest, Bf16TypeProperties) {
  ASSERT_EQ(fl::getTypeSize(fl::dtype::bf16), 2u);
  ASSERT_EQ(fl::dtypeToString(fl::dtype::bf16), "bf16");
  ASSERT_EQ(fl::stringToDtype("bf16"), fl::dtype::bf16);
}

TEST(TensorBaseTest, IntegerTensorIndexAssign) {
  // Test: output(indices) = values — scatter values into specific positions
  auto output = fl::full({10}, 0.0f);
  auto indices = fl::Tensor::fromBuffer(
      {3}, std::vector<int>({2, 5, 7}).data(), fl::MemoryLocation::Host);
  auto values = fl::Tensor::fromBuffer(
      {3}, std::vector<float>({10.f, 20.f, 30.f}).data(), fl::MemoryLocation::Host);

  output(indices) = values;

  std::vector<float> result(10);
  output.hostSync(result.data());
  ASSERT_FLOAT_EQ(result[0], 0.f);
  ASSERT_FLOAT_EQ(result[1], 0.f);
  ASSERT_FLOAT_EQ(result[2], 10.f);
  ASSERT_FLOAT_EQ(result[3], 0.f);
  ASSERT_FLOAT_EQ(result[4], 0.f);
  ASSERT_FLOAT_EQ(result[5], 20.f);
  ASSERT_FLOAT_EQ(result[6], 0.f);
  ASSERT_FLOAT_EQ(result[7], 30.f);
  ASSERT_FLOAT_EQ(result[8], 0.f);
  ASSERT_FLOAT_EQ(result[9], 0.f);
}

TEST(TensorBaseTest, IntegerTensorIndexAssignThenAdd) {
  // Test: cast pattern from AdaptiveSoftMaxLoss
  // Create output, scatter values into it, then use result in addition
  auto output = fl::full({10}, 0.0f);
  auto indices = fl::Tensor::fromBuffer(
      {3}, std::vector<int>({2, 5, 7}).data(), fl::MemoryLocation::Host);
  auto values = fl::Tensor::fromBuffer(
      {3}, std::vector<float>({10.f, 20.f, 30.f}).data(), fl::MemoryLocation::Host);

  output(indices) = values;

  // Add to another tensor (simulate res = res + cast_result)
  auto base = fl::full({10}, 1.0f);
  auto result = base + output;

  std::vector<float> resultData(10);
  result.hostSync(resultData.data());
  ASSERT_FLOAT_EQ(resultData[0], 1.f);
  ASSERT_FLOAT_EQ(resultData[2], 11.f);
  ASSERT_FLOAT_EQ(resultData[5], 21.f);
  ASSERT_FLOAT_EQ(resultData[7], 31.f);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  flTestInit();
  return RUN_ALL_TESTS();
}
