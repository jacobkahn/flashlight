/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/backend/cuda/CudaTensor.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"

using namespace fl;

class CudaTensorIndexTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fl::setDefaultJitTensorType<CudaTensor>();
  }
};

TEST_F(CudaTensorIndexTest, FlatReadWriteRoundtrip) {
  auto t = fl::full({2, 3}, 0.0f);
  for (int i = 0; i < t.elements(); ++i) {
    t.flat(i) = static_cast<float>(i + 1);
  }
  for (int i = 0; i < t.elements(); ++i) {
    ASSERT_FLOAT_EQ(t.flat(i).scalarSync<float>(), static_cast<float>(i + 1))
        << "Mismatch at flat(" << i << ")";
  }
}

TEST_F(CudaTensorIndexTest, FlatPerturbRestore) {
  auto t = fl::rand({3, 4}, fl::dtype::f32);
  for (int i = 0; i < t.elements(); ++i) {
    Tensor orig = t.flatten()(i);
    float origF = orig.scalarSync<float>();
    t.flat(i) = orig - 0.01f;
    float perturbed = t.flat(i).scalarSync<float>();
    ASSERT_NEAR(perturbed, origF - 0.01f, 1e-6)
        << "Perturb failed at i=" << i;
    t.flat(i) = orig;
    float restored = t.flat(i).scalarSync<float>();
    ASSERT_NEAR(restored, origF, 1e-6)
        << "Restore failed at i=" << i;
  }
}

TEST_F(CudaTensorIndexTest, ColumnSliceAssign) {
  auto m = fl::full({3, 4}, 0.0f);
  auto col = fl::full({3}, 42.0f);
  m(fl::span, 1) = col;
  for (int r = 0; r < 3; ++r) {
    ASSERT_FLOAT_EQ(m(r, 1).scalarSync<float>(), 42.0f);
  }
  for (int r = 0; r < 3; ++r) {
    ASSERT_FLOAT_EQ(m(r, 0).scalarSync<float>(), 0.0f);
    ASSERT_FLOAT_EQ(m(r, 2).scalarSync<float>(), 0.0f);
  }
}

// The pointer-identity tests below inspect CudaTensor views directly, so they
// build bare CudaTensors (the fixture's default is a JIT tensor, whose adapter
// is a JitTensorBase).
TEST_F(CudaTensorIndexTest, ViewSharesBufferPointer) {
  fl::withTensorType<CudaTensor>([]() {
    auto t = fl::rand({4, 5});
    auto& tAdapter = t.getAdapter<CudaTensor>();
    void* origBase = tAdapter.devicePtr();

    auto row0 = t(0);
    auto& r0Adapter = row0.getAdapter<CudaTensor>();
    void* row0Ptr = r0Adapter.devicePtr();
    ASSERT_EQ(row0Ptr, origBase) << "Row 0 view should point to same base";

    auto row2 = t(2);
    auto& r2Adapter = row2.getAdapter<CudaTensor>();
    void* row2Ptr = r2Adapter.devicePtr();
    ptrdiff_t diff = static_cast<char*>(row2Ptr) - static_cast<char*>(origBase);
    ASSERT_EQ(diff, 2 * 5 * static_cast<ptrdiff_t>(sizeof(float)))
        << "Row 2 view offset incorrect";
  });
}

TEST_F(CudaTensorIndexTest, ViewWriteThrough) {
  auto t = fl::full({3, 4}, 1.0f);
  float before = t(1, 2).scalarSync<float>();
  ASSERT_FLOAT_EQ(before, 1.0f);

  auto row1 = t(1);
  row1(2) = 99.0f;

  float after = t(1, 2).scalarSync<float>();
  ASSERT_FLOAT_EQ(after, 99.0f) << "Write through view should modify original";

  ASSERT_FLOAT_EQ(t(0, 0).scalarSync<float>(), 1.0f);
  ASSERT_FLOAT_EQ(t(2, 3).scalarSync<float>(), 1.0f);
}

TEST_F(CudaTensorIndexTest, FlatViewWriteThrough) {
  auto t = fl::full({2, 3}, 0.0f);
  t.flat(4) = 77.0f;

  float val = t(1, 1).scalarSync<float>();
  ASSERT_FLOAT_EQ(val, 77.0f) << "flat() write should modify original";
}

TEST_F(CudaTensorIndexTest, FlattenReturnsIndependentCopy) {
  auto t = fl::full({2, 3}, 5.0f);
  auto flat = t.flatten();

  flat(0) = 999.0f;

  ASSERT_FLOAT_EQ(t(0, 0).scalarSync<float>(), 5.0f)
      << "flatten() should return independent copy";
  ASSERT_FLOAT_EQ(flat(0).scalarSync<float>(), 999.0f);
}

TEST_F(CudaTensorIndexTest, CopyReturnsIndependentTensor) {
  auto t = fl::rand({3, 3});
  float origVal = t(1, 1).scalarSync<float>();

  auto c = t.copy();
  c(1, 1) = 12345.0f;

  ASSERT_FLOAT_EQ(t(1, 1).scalarSync<float>(), origVal)
      << "copy() should return independent tensor";
  ASSERT_FLOAT_EQ(c(1, 1).scalarSync<float>(), 12345.0f);
}

TEST_F(CudaTensorIndexTest, ViewStridesAreCorrect) {
  auto t = fl::rand({3, 4});
  auto col1 = t(fl::span, 1);

  ASSERT_EQ(col1.shape(), Shape({3}));
  auto strides = col1.strides();
  ASSERT_EQ(strides[0], 4)
      << "Column view stride should be row length";
}

TEST_F(CudaTensorIndexTest, NonContiguousViewIsNotContiguous) {
  auto t = fl::rand({3, 4});
  auto col = t(fl::span, 2);

  ASSERT_FALSE(col.isContiguous())
      << "Column view should be non-contiguous";

  auto row = t(1);
  ASSERT_TRUE(row.isContiguous())
      << "Row view should be contiguous";
}

TEST_F(CudaTensorIndexTest, ChainedViewsAccumulateOffset) {
  fl::withTensorType<CudaTensor>([]() {
    auto t = fl::rand({4, 5, 3});
    auto& tAdapter = t.getAdapter<CudaTensor>();
    void* base = tAdapter.devicePtr();

    auto sub = t(1);
    auto& subAdapter = sub.getAdapter<CudaTensor>();
    ptrdiff_t diff1 = static_cast<char*>(subAdapter.devicePtr()) -
                      static_cast<char*>(base);
    ASSERT_EQ(diff1, 15 * static_cast<ptrdiff_t>(sizeof(float)));

    auto subsub = sub(2);
    auto& ssubAdapter = subsub.getAdapter<CudaTensor>();
    ptrdiff_t diff2 = static_cast<char*>(ssubAdapter.devicePtr()) -
                      static_cast<char*>(base);
    ASSERT_EQ(diff2, 21 * static_cast<ptrdiff_t>(sizeof(float)));

    float viaChain = subsub(0).scalarSync<float>();
    float viaDirect = t(1, 2, 0).scalarSync<float>();
    ASSERT_FLOAT_EQ(viaChain, viaDirect);
  });
}

TEST_F(CudaTensorIndexTest, ScalarViewReadWrite) {
  auto t = fl::full({3, 4}, 7.0f);

  auto s = t(1, 2);
  ASSERT_EQ(s.shape(), Shape({}));
  ASSERT_FLOAT_EQ(s.scalarSync<float>(), 7.0f);

  s = 42.0f;
  ASSERT_FLOAT_EQ(t(1, 2).scalarSync<float>(), 42.0f)
      << "Scalar view write should modify original";
}

TEST_F(CudaTensorIndexTest, RangeViewCorrectness) {
  auto t = fl::full({10}, 0.0f);
  for (int i = 0; i < 10; ++i) {
    t.flat(i) = static_cast<float>(i);
  }

  auto sub = t(fl::range(2, 8));
  ASSERT_EQ(sub.shape(), Shape({6}));
  ASSERT_TRUE(sub.isContiguous());
  for (int i = 0; i < 6; ++i) {
    ASSERT_FLOAT_EQ(sub(i).scalarSync<float>(), static_cast<float>(i + 2));
  }

  sub(0) = 99.0f;
  ASSERT_FLOAT_EQ(t(2).scalarSync<float>(), 99.0f)
      << "Range view write should modify original";
}

TEST_F(CudaTensorIndexTest, NonContiguousAssignColumn) {
  auto t = fl::full({3, 4}, 0.0f);

  ASSERT_FALSE(t(fl::span, 2).isContiguous());

  auto vals = fl::full({3}, 10.0f);
  t(fl::span, 2) = vals;

  for (int r = 0; r < 3; ++r) {
    ASSERT_FLOAT_EQ(t(r, 2).scalarSync<float>(), 10.0f);
  }
  ASSERT_FLOAT_EQ(t(0, 0).scalarSync<float>(), 0.0f);
  ASSERT_FLOAT_EQ(t(0, 1).scalarSync<float>(), 0.0f);
  ASSERT_FLOAT_EQ(t(0, 3).scalarSync<float>(), 0.0f);
}

TEST_F(CudaTensorIndexTest, LvalueAssignReplacesView) {
  auto t = fl::full({3, 3}, 1.0f);
  auto view = t(0);

  auto newData = fl::full({3}, 99.0f);
  view = newData;

  ASSERT_FLOAT_EQ(t(0, 0).scalarSync<float>(), 1.0f)
      << "Lvalue assignment should replace, not write through";
  ASSERT_FLOAT_EQ(view(0).scalarSync<float>(), 99.0f);
}

TEST_F(CudaTensorIndexTest, SpanIndexSharesBuffer) {
  fl::withTensorType<CudaTensor>([]() {
    auto t = fl::rand({3, 3});
    auto view = t(fl::span, fl::span);

    auto& tA = t.getAdapter<CudaTensor>();
    auto& vA = view.getAdapter<CudaTensor>();
    ASSERT_EQ(tA.devicePtr(), vA.devicePtr());
    ASSERT_EQ(view.shape(), t.shape());

    view(0, 0) = 123.0f;
    ASSERT_FLOAT_EQ(t(0, 0).scalarSync<float>(), 123.0f)
        << "Span view should share buffer";
  });
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
