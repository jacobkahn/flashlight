/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/Tensor.h"

using namespace fl;

TEST(DevicePtrTest, Null) {
  Tensor x;
  DevicePtr xp(x);
  EXPECT_EQ(xp.get(), nullptr);
}

TEST(DevicePtrTest, NoCopy) {
  Tensor a = fl::full({3, 3}, 5.);

  void* devicePtrLoc;
  {
    DevicePtr p(a);
    devicePtrLoc = p.get();
  }
  EXPECT_EQ(devicePtrLoc, a.deviceSync<void>());
}

TEST(DevicePtrTest, Locking) {
  // isLocked() returns false — buffer lifetime is managed by the Tensor
  // object, not by explicit lock/unlock.
  Tensor x({3, 3});
  EXPECT_FALSE(x.isLocked());
  {
    DevicePtr xp(x);
    EXPECT_FALSE(x.isLocked());
  }
  EXPECT_FALSE(x.isLocked());
}

TEST(DevicePtrTest, Move) {
  Tensor x({3, 3});
  Tensor y({4, 4});

  DevicePtr yp(y);
  EXPECT_FALSE(x.isLocked());
  EXPECT_FALSE(y.isLocked());

  yp = DevicePtr(x);
  EXPECT_FALSE(x.isLocked());
  EXPECT_FALSE(y.isLocked());

  yp = DevicePtr();
  EXPECT_FALSE(x.isLocked());
  EXPECT_FALSE(y.isLocked());
}

TEST(DevicePtrTest, NonContiguous) {
  // Slicing with a step > 1 creates a non-contiguous view
  auto a = fl::rand({6, 8});
  auto sliced = a(fl::range(0, 6, 2), fl::span); // every other row
  ASSERT_FALSE(sliced.isContiguous());

  // DevicePtr should accept non-contiguous tensors without throwing
  DevicePtr dp(sliced);
  EXPECT_NE(dp.get(), nullptr);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
