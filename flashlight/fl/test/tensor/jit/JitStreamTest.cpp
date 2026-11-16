/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <memory>

#include "flashlight/fl/runtime/Stream.h"
#include "flashlight/fl/runtime/StreamOrDevice.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/JitBackend.h"
#include "flashlight/fl/tensor/JitTensorBase.h"
#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"
#include "flashlight/fl/tensor/opt/passes/ScalarFolding.h"

using namespace fl;

namespace {

JitBackend& jitBackend() {
  return dynamic_cast<JitBackend&>(defaultTensorBackend());
}

std::shared_ptr<const Stream> defaultStreamPtr() {
  return jitBackend().defaultStreamShared();
}

} // namespace

TEST(JitStreamTest, defaultStreamIsNull) {
  auto a = fl::rand({10, 10});
  auto b = fl::exp(a);
  ASSERT_EQ(toJitTensorBase(b).node()->stream(), nullptr);
}

TEST(JitStreamTest, streamAnnotationOnNode) {
  auto stream = defaultStreamPtr();
  auto a = fl::rand({10, 10});
  auto b = fl::exp(a, stream);
  ASSERT_EQ(toJitTensorBase(b).node()->stream(), stream);
}

TEST(JitStreamTest, binaryOpStreamAnnotation) {
  auto stream = defaultStreamPtr();
  auto a = fl::rand({10, 10});
  auto b = fl::rand({10, 10});
  auto c = fl::add(a, b, stream);
  ASSERT_EQ(toJitTensorBase(c).node()->stream(), stream);
}

TEST(JitStreamTest, operatorsHaveNullStream) {
  auto a = fl::rand({10, 10});
  auto b = fl::rand({10, 10});
  auto c = a + b;
  ASSERT_EQ(toJitTensorBase(c).node()->stream(), nullptr);
}

TEST(JitStreamTest, scalarFoldingPreservesStream) {
  auto stream = defaultStreamPtr();

  Shape shape({2, 2});
  auto c1 = ScalarNode::create(shape, dtype::f32, 3.0);
  auto c2 = ScalarNode::create(shape, dtype::f32, 4.0);
  auto add = BinaryNode::create(c1, c2, BinaryOp::Add, stream);

  ScalarFolding folder;
  auto result = folder.apply(add);

  // The folded scalar should inherit the binary node's stream
  ASSERT_NE(result, add); // folding happened
  ASSERT_EQ(result->stream(), stream);
}

TEST(JitStreamTest, streamFlowsToCorrectResult) {
  // Verify that computing with a stream-annotated op produces
  // the correct numerical result
  auto a = fl::full({3, 3}, 2.0f);
  auto b = fl::exp(a);
  auto result = b.scalarSync<float>();
  ASSERT_NEAR(result, std::exp(2.0f), 1e-5);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
