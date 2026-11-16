/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Copy.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/JitBackend.h"
#include "flashlight/fl/tensor/JitTensorBase.h"
#include "flashlight/fl/tensor/BackendRegistry.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/ir/TransferNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/runtime/Stream.h"
#include "flashlight/fl/runtime/Device.h"
#include "flashlight/fl/test/TestInit.h"

using namespace fl;

namespace {

// Backend type and stream for the active test backend (CPU or CUDA).
#ifdef FL_TEST_BACKEND_CUDA
constexpr TensorBackendType kTestBackendType = TensorBackendType::Cuda;
#else
constexpr TensorBackendType kTestBackendType = TensorBackendType::CPU;
#endif

std::shared_ptr<const Stream> backendStream() {
  return fl::getBackend(kTestBackendType).jitStream();
}

} // namespace

// --- Functional copy tests ---

TEST(CopyTest, FunctionalCopy_SameDevice) {
  auto src = fl::full({3, 4}, 42.0f);
  auto dst = fl::copy(src, backendStream());
  ASSERT_TRUE(allClose(src, dst));
}

TEST(CopyTest, FunctionalCopy_NoOpElimination) {
  auto stream = backendStream();
  auto scalarNode = ScalarNode::create({2, 3}, dtype::f32, 1.0, stream);
  auto transferNode = TransferNode::create(
      scalarNode,
      stream->device().type(),
      kTestBackendType,
      stream);

  // Same device type -> returns input directly (no-op)
  ASSERT_EQ(transferNode, scalarNode);
}

TEST(CopyTest, FunctionalCopy_StridedSrc) {
  auto orig = fl::rand({4, 4});
  auto transposed = fl::transpose(orig);
  auto copied = fl::copy(transposed, backendStream());
  ASSERT_TRUE(allClose(transposed, copied));
}

TEST(CopyTest, FunctionalCopy_Empty) {
  auto src = fl::full({0}, 0.0f);
  auto dst = fl::copy(src, backendStream());
  ASSERT_EQ(dst.elements(), 0);
}

// --- Imperative copy tests ---

TEST(CopyTest, ImperativeCopy_SameBackend) {
  auto src = fl::full({3, 4}, 7.0f);
  auto dst = fl::full({3, 4}, 0.0f);
  fl::copy(dst, src);
  ASSERT_TRUE(allClose(src, dst));
}

TEST(CopyTest, ImperativeCopy_StridedSrc) {
  auto orig = fl::rand({4, 4});
  auto transposed = fl::transpose(orig);
  auto dst = fl::full({4, 4}, 0.0f);
  fl::copy(dst, transposed);
  ASSERT_TRUE(allClose(transposed, dst));
}

TEST(CopyTest, ImperativeCopy_ShapeMismatch) {
  auto src = fl::full({3, 4}, 1.0f);
  auto dst = fl::full({4, 3}, 0.0f);
  EXPECT_THROW(fl::copy(dst, src), std::invalid_argument);
}

TEST(CopyTest, ImperativeCopy_DtypeMismatch) {
  auto src = fl::full({3, 4}, 1.0f);
  auto dst = fl::full({3, 4}, 0.0, fl::dtype::f64);
  EXPECT_THROW(fl::copy(dst, src), std::invalid_argument);
}

// --- TransferNode unit tests ---

TEST(CopyTest, TransferNode_Create) {
  auto stream = backendStream();
  auto dstDevType = stream->device().type();
  auto inputNode = ScalarNode::create({2, 3}, dtype::f32, 1.0);
  auto node = TransferNode::create(
      inputNode,
      dstDevType,
      kTestBackendType,
      stream);

  // Input has no stream, so no-op elimination won't trigger
  ASSERT_NE(node, inputNode);
  ASSERT_TRUE(node->isTransfer());
  ASSERT_EQ(node->shape(), Shape({2, 3}));
  ASSERT_EQ(node->dataType(), dtype::f32);
  auto& tn = node->impl<TransferNode>();
  ASSERT_EQ(tn.dstDeviceType(), dstDevType);
  ASSERT_EQ(tn.dstBackendType(), kTestBackendType);
}

TEST(CopyTest, TransferNode_NoOpSameDevice) {
  auto stream = backendStream();
  auto inputNode = ScalarNode::create({2, 3}, dtype::f32, 1.0, stream);
  auto node = TransferNode::create(
      inputNode,
      stream->device().type(),
      kTestBackendType,
      stream);

  // Same device type -> returns input directly
  ASSERT_EQ(node, inputNode);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  flTestInit();
  return RUN_ALL_TESTS();
}
