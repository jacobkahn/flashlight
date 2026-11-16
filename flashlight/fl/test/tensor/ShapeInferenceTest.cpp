/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <stdexcept>

#include "flashlight/fl/tensor/ShapeInference.h"
#include "flashlight/fl/tensor/ir/OpProperties.h"

using namespace fl;

// ======================== inferReductionOutputShape ========================

TEST(ShapeInferenceTest, ReductionScalar) {
  Shape scalar;
  ASSERT_EQ(inferReductionOutputShape(scalar, {}, false), Shape());
  ASSERT_EQ(inferReductionOutputShape(scalar, {}, true), Shape());
}

TEST(ShapeInferenceTest, ReductionAllAxes) {
  ASSERT_EQ(inferReductionOutputShape({3, 4, 5}, {}, false), Shape());
  ASSERT_EQ(inferReductionOutputShape({3, 4, 5}, {}, true), Shape({1, 1, 1}));
}

TEST(ShapeInferenceTest, ReductionSingleAxis) {
  ASSERT_EQ(inferReductionOutputShape({3, 4, 5}, {0}, false), Shape({4, 5}));
  ASSERT_EQ(inferReductionOutputShape({3, 4, 5}, {1}, false), Shape({3, 5}));
  ASSERT_EQ(inferReductionOutputShape({3, 4, 5}, {2}, false), Shape({3, 4}));
}

TEST(ShapeInferenceTest, ReductionSingleAxisKeepDims) {
  ASSERT_EQ(
      inferReductionOutputShape({3, 4, 5}, {0}, true), Shape({1, 4, 5}));
  ASSERT_EQ(
      inferReductionOutputShape({3, 4, 5}, {1}, true), Shape({3, 1, 5}));
  ASSERT_EQ(
      inferReductionOutputShape({3, 4, 5}, {2}, true), Shape({3, 4, 1}));
}

TEST(ShapeInferenceTest, ReductionMultipleAxes) {
  ASSERT_EQ(
      inferReductionOutputShape({3, 4, 5}, {0, 2}, false), Shape({4}));
  ASSERT_EQ(
      inferReductionOutputShape({3, 4, 5}, {0, 2}, true), Shape({1, 4, 1}));
  ASSERT_EQ(
      inferReductionOutputShape({3, 4, 5}, {0, 1, 2}, false), Shape());
}

TEST(ShapeInferenceTest, ReductionInvalidAxis) {
  EXPECT_THROW(
      inferReductionOutputShape({3, 4}, {2}, false), std::invalid_argument);
  EXPECT_THROW(
      inferReductionOutputShape({3, 4}, {-1}, false), std::invalid_argument);
}

TEST(ShapeInferenceTest, Reduction1D) {
  ASSERT_EQ(inferReductionOutputShape({5}, {0}, false), Shape());
  ASSERT_EQ(inferReductionOutputShape({5}, {0}, true), Shape({1}));
}

// ======================== inferTransposeOutputShape ========================

TEST(ShapeInferenceTest, TransposeDefaultReverse) {
  ASSERT_EQ(inferTransposeOutputShape({3, 4}, {}), Shape({4, 3}));
  ASSERT_EQ(inferTransposeOutputShape({2, 3, 4}, {}), Shape({4, 3, 2}));
  ASSERT_EQ(
      inferTransposeOutputShape({2, 3, 4, 5}, {}), Shape({5, 4, 3, 2}));
}

TEST(ShapeInferenceTest, TransposeExplicitAxes) {
  ASSERT_EQ(inferTransposeOutputShape({3, 4}, {1, 0}), Shape({4, 3}));
  ASSERT_EQ(
      inferTransposeOutputShape({2, 3, 4}, {2, 0, 1}), Shape({4, 2, 3}));
  ASSERT_EQ(
      inferTransposeOutputShape({2, 3, 4}, {0, 1, 2}), Shape({2, 3, 4}));
}

TEST(ShapeInferenceTest, TransposeInvalidAxes) {
  EXPECT_THROW(
      inferTransposeOutputShape({2, 3, 4}, {0, 1}), std::runtime_error);
}

TEST(ShapeInferenceTest, Transpose1D) {
  ASSERT_EQ(inferTransposeOutputShape({5}, {0}), Shape({5}));
  ASSERT_EQ(inferTransposeOutputShape({5}, {}), Shape({5}));
}

// ======================== inferTileOutputShape ========================

TEST(ShapeInferenceTest, TileBasic) {
  ASSERT_EQ(inferTileOutputShape({3, 4}, {2, 3}), Shape({6, 12}));
  ASSERT_EQ(inferTileOutputShape({3, 4}, {1, 1}), Shape({3, 4}));
}

TEST(ShapeInferenceTest, TileDimsMismatch) {
  // Tile rank > input rank: input is padded with leading 1s
  ASSERT_EQ(inferTileOutputShape({3}, {2, 3}), Shape({2, 9}));
  // Input rank > tile rank: tile is padded with trailing 1s
  // tileDims {2} → {2, 1}, so dim 0 is tiled by 2, dim 1 untouched
  ASSERT_EQ(inferTileOutputShape({3, 4}, {2}), Shape({6, 4}));
}

TEST(ShapeInferenceTest, Tile1D) {
  ASSERT_EQ(inferTileOutputShape({5}, {3}), Shape({15}));
}

TEST(ShapeInferenceTest, TileHigherRank) {
  ASSERT_EQ(
      inferTileOutputShape({2, 3, 4}, {1, 2, 3}), Shape({2, 6, 12}));
}

// ======================== broadcastShapesToMaxRank ========================

TEST(ShapeInferenceTest, BroadcastSameRank) {
  auto result = broadcastShapesToMaxRank(
      {Shape({3, 4}), Shape({5, 6})}, 0);
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result[0], Shape({3, 4}));
  ASSERT_EQ(result[1], Shape({5, 6}));
}

TEST(ShapeInferenceTest, BroadcastDifferentRank) {
  auto result = broadcastShapesToMaxRank(
      {Shape({3}), Shape({5, 6})}, 0);
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result[0], Shape({3, 1})); // padded with trailing 1
  ASSERT_EQ(result[1], Shape({5, 6}));
}

TEST(ShapeInferenceTest, BroadcastMinOutputRank) {
  // minOutputRank forces padding even when all inputs are lower rank
  auto result = broadcastShapesToMaxRank(
      {Shape({3}), Shape({5})}, 3);
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result[0], Shape({3, 1, 1}));
  ASSERT_EQ(result[1], Shape({5, 1, 1}));
}

// ======================== inferConcatenateOutputShape ========================

TEST(ShapeInferenceTest, ConcatenateSameShape) {
  ASSERT_EQ(
      inferConcatenateOutputShape(
          {Shape({3, 4}), Shape({3, 4})}, {12, 12}, 0),
      Shape({6, 4}));
  ASSERT_EQ(
      inferConcatenateOutputShape(
          {Shape({3, 4}), Shape({3, 4})}, {12, 12}, 1),
      Shape({3, 8}));
}

TEST(ShapeInferenceTest, ConcatenateDifferentAlongAxis) {
  ASSERT_EQ(
      inferConcatenateOutputShape(
          {Shape({3, 4}), Shape({5, 4})}, {12, 20}, 0),
      Shape({8, 4}));
}

TEST(ShapeInferenceTest, ConcatenateRankPromotion) {
  // 1D inputs concatenated along axis 1 get promoted to 2D
  ASSERT_EQ(
      inferConcatenateOutputShape(
          {Shape({3}), Shape({3})}, {3, 3}, 1),
      Shape({3, 2}));
}

TEST(ShapeInferenceTest, ConcatenateEmpty) {
  EXPECT_THROW(
      inferConcatenateOutputShape({}, {}, 0), std::runtime_error);
}

TEST(ShapeInferenceTest, ConcatenateWithEmptyInput) {
  // Empty inputs (0 elements) contribute 0 along the concat axis
  ASSERT_EQ(
      inferConcatenateOutputShape(
          {Shape({3, 4}), Shape({0})}, {12, 0}, 0),
      Shape({3, 4}));
  ASSERT_EQ(
      inferConcatenateOutputShape(
          {Shape({0}), Shape({3, 4})}, {0, 12}, 0),
      Shape({3, 4}));
}

TEST(ShapeInferenceTest, ConcatenateThreeInputs) {
  ASSERT_EQ(
      inferConcatenateOutputShape(
          {Shape({2, 4}), Shape({3, 4}), Shape({5, 4})}, {8, 12, 20}, 0),
      Shape({10, 4}));
}

// ======================== inferPadOutputShape ========================

TEST(ShapeInferenceTest, PadBasic) {
  ASSERT_EQ(
      inferPadOutputShape({3, 4}, {{1, 2}, {3, 4}}), Shape({6, 11}));
}

TEST(ShapeInferenceTest, PadPartialAxes) {
  ASSERT_EQ(inferPadOutputShape({3, 4}, {{1, 1}}), Shape({5, 4}));
}

TEST(ShapeInferenceTest, PadZero) {
  ASSERT_EQ(inferPadOutputShape({3, 4}, {{0, 0}, {0, 0}}), Shape({3, 4}));
}

TEST(ShapeInferenceTest, PadTooManyAxes) {
  EXPECT_THROW(
      inferPadOutputShape({3, 4}, {{1, 1}, {1, 1}, {1, 1}}),
      std::runtime_error);
}

TEST(ShapeInferenceTest, Pad3D) {
  ASSERT_EQ(
      inferPadOutputShape({2, 3, 4}, {{1, 1}, {2, 2}, {0, 3}}),
      Shape({4, 7, 7}));
}

// ======================== inferMatmulOutputShape ========================

TEST(ShapeInferenceTest, MatmulBasic2D) {
  ASSERT_EQ(
      inferMatmulOutputShape(
          {3, 4}, {4, 5}, MatrixProperty::None, MatrixProperty::None),
      Shape({3, 5}));
}

TEST(ShapeInferenceTest, MatmulTransposeLhs) {
  ASSERT_EQ(
      inferMatmulOutputShape(
          {4, 3}, {4, 5}, MatrixProperty::Transpose, MatrixProperty::None),
      Shape({3, 5}));
}

TEST(ShapeInferenceTest, MatmulTransposeRhs) {
  ASSERT_EQ(
      inferMatmulOutputShape(
          {3, 4}, {5, 4}, MatrixProperty::None, MatrixProperty::Transpose),
      Shape({3, 5}));
}

TEST(ShapeInferenceTest, MatmulTransposeBoth) {
  ASSERT_EQ(
      inferMatmulOutputShape(
          {4, 3}, {5, 4},
          MatrixProperty::Transpose, MatrixProperty::Transpose),
      Shape({3, 5}));
}

TEST(ShapeInferenceTest, MatmulVectorTimesMatrix) {
  ASSERT_EQ(
      inferMatmulOutputShape(
          {4}, {4, 5}, MatrixProperty::None, MatrixProperty::None),
      Shape({5}));
}

TEST(ShapeInferenceTest, MatmulMatrixTimesVector) {
  ASSERT_EQ(
      inferMatmulOutputShape(
          {3, 4}, {4}, MatrixProperty::None, MatrixProperty::None),
      Shape({3}));
}

TEST(ShapeInferenceTest, MatmulVectorDot) {
  ASSERT_EQ(
      inferMatmulOutputShape(
          {4}, {4}, MatrixProperty::None, MatrixProperty::None),
      Shape({1}));
}

TEST(ShapeInferenceTest, MatmulBatched) {
  ASSERT_EQ(
      inferMatmulOutputShape(
          {2, 3, 4}, {2, 4, 5}, MatrixProperty::None, MatrixProperty::None),
      Shape({2, 3, 5}));
}

TEST(ShapeInferenceTest, MatmulBatchedVsUnbatched) {
  ASSERT_EQ(
      inferMatmulOutputShape(
          {2, 3, 4}, {4, 5}, MatrixProperty::None, MatrixProperty::None),
      Shape({2, 3, 5}));
  ASSERT_EQ(
      inferMatmulOutputShape(
          {3, 4}, {2, 4, 5}, MatrixProperty::None, MatrixProperty::None),
      Shape({2, 3, 5}));
}

TEST(ShapeInferenceTest, MatmulIncompatibleK) {
  EXPECT_THROW(
      inferMatmulOutputShape(
          {3, 4}, {5, 6}, MatrixProperty::None, MatrixProperty::None),
      std::invalid_argument);
}

TEST(ShapeInferenceTest, MatmulIncompatibleBatch) {
  EXPECT_THROW(
      inferMatmulOutputShape(
          {2, 3, 4}, {3, 4, 5}, MatrixProperty::None, MatrixProperty::None),
      std::invalid_argument);
}

TEST(ShapeInferenceTest, MatmulSquare) {
  ASSERT_EQ(
      inferMatmulOutputShape(
          {4, 4}, {4, 4}, MatrixProperty::None, MatrixProperty::None),
      Shape({4, 4}));
}

// ======================== inferOutputDtype — UnaryOp ========================

TEST(TypeInferenceTest, UnaryPreservesType) {
  // Arithmetic unary ops preserve input dtype
  for (auto op : {
      UnaryOp::Exp, UnaryOp::Log, UnaryOp::Log1p,
      UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Sqrt,
      UnaryOp::Tanh, UnaryOp::Floor, UnaryOp::Ceil,
      UnaryOp::Rint, UnaryOp::Negative, UnaryOp::Absolute,
      UnaryOp::Sigmoid, UnaryOp::Erf, UnaryOp::Sign}) {
    ASSERT_EQ(inferOutputDtype(op, dtype::f32), dtype::f32);
    ASSERT_EQ(inferOutputDtype(op, dtype::f64), dtype::f64);
    ASSERT_EQ(inferOutputDtype(op, dtype::s32), dtype::s32);
  }
}

TEST(TypeInferenceTest, UnaryPredicateReturnsB8) {
  for (auto op : {UnaryOp::LogicalNot, UnaryOp::Isnan, UnaryOp::Isinf}) {
    ASSERT_EQ(inferOutputDtype(op, dtype::f32), dtype::b8);
    ASSERT_EQ(inferOutputDtype(op, dtype::s32), dtype::b8);
    ASSERT_EQ(inferOutputDtype(op, dtype::b8), dtype::b8);
  }
}

// ======================== inferOutputDtype — BinaryOp =======================

TEST(TypeInferenceTest, BinaryComparisonReturnsB8) {
  for (auto op : {
      BinaryOp::Eq, BinaryOp::Neq, BinaryOp::Gt,
      BinaryOp::Gte, BinaryOp::Lt, BinaryOp::Lte,
      BinaryOp::And, BinaryOp::Or}) {
    ASSERT_EQ(inferOutputDtype(op, dtype::f32, dtype::f32), dtype::b8);
    ASSERT_EQ(inferOutputDtype(op, dtype::s32, dtype::f64), dtype::b8);
  }
}

TEST(TypeInferenceTest, BinaryArithmeticPromotes) {
  for (auto op : {
      BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div}) {
    // Same type: preserved
    ASSERT_EQ(inferOutputDtype(op, dtype::f32, dtype::f32), dtype::f32);
    ASSERT_EQ(inferOutputDtype(op, dtype::s32, dtype::s32), dtype::s32);
    // Mixed: wider/float wins
    ASSERT_EQ(inferOutputDtype(op, dtype::s32, dtype::f32), dtype::f32);
    ASSERT_EQ(inferOutputDtype(op, dtype::f32, dtype::f64), dtype::f64);
  }
}

TEST(TypeInferenceTest, BinaryMinMaxPromotes) {
  ASSERT_EQ(
      inferOutputDtype(BinaryOp::Min, dtype::f32, dtype::f32), dtype::f32);
  ASSERT_EQ(
      inferOutputDtype(BinaryOp::Max, dtype::s32, dtype::f32), dtype::f32);
}

TEST(TypeInferenceTest, BinaryBitwisePromotes) {
  for (auto op : {
      BinaryOp::Shl, BinaryOp::Shr,
      BinaryOp::BitAnd, BinaryOp::BitOr, BinaryOp::BitXor}) {
    ASSERT_EQ(inferOutputDtype(op, dtype::s32, dtype::s32), dtype::s32);
    ASSERT_EQ(inferOutputDtype(op, dtype::u8, dtype::s32), dtype::s32);
  }
}

// ======================== inferOutputDtype — ReductionOp ====================

TEST(TypeInferenceTest, ReductionPredicateReturnsB8) {
  for (auto op : {ReductionOp::Any, ReductionOp::All}) {
    ASSERT_EQ(inferOutputDtype(op, dtype::f32), dtype::b8);
    ASSERT_EQ(inferOutputDtype(op, dtype::s32), dtype::b8);
    ASSERT_EQ(inferOutputDtype(op, dtype::b8), dtype::b8);
  }
}

TEST(TypeInferenceTest, ReductionCountNonzeroReturnsU32) {
  ASSERT_EQ(inferOutputDtype(ReductionOp::CountNonzero, dtype::f32), dtype::u32);
  ASSERT_EQ(inferOutputDtype(ReductionOp::CountNonzero, dtype::s32), dtype::u32);
  ASSERT_EQ(inferOutputDtype(ReductionOp::CountNonzero, dtype::b8), dtype::u32);
}

TEST(TypeInferenceTest, ReductionMeanStdMedianPromoteIntToF32) {
  for (auto op : {ReductionOp::Mean, ReductionOp::Std, ReductionOp::Median}) {
    // Float preserved
    ASSERT_EQ(inferOutputDtype(op, dtype::f32), dtype::f32);
    ASSERT_EQ(inferOutputDtype(op, dtype::f64), dtype::f64);
    // Integer promoted to f32
    ASSERT_EQ(inferOutputDtype(op, dtype::s32), dtype::f32);
    ASSERT_EQ(inferOutputDtype(op, dtype::u8), dtype::f32);
    ASSERT_EQ(inferOutputDtype(op, dtype::b8), dtype::f32);
  }
}

TEST(TypeInferenceTest, ReductionSumPreservesType) {
  ASSERT_EQ(inferOutputDtype(ReductionOp::Sum, dtype::f32), dtype::f32);
  ASSERT_EQ(inferOutputDtype(ReductionOp::Sum, dtype::f64), dtype::f64);
  ASSERT_EQ(inferOutputDtype(ReductionOp::Sum, dtype::s32), dtype::s32);
}

TEST(TypeInferenceTest, ReductionSumBoolToU32) {
  ASSERT_EQ(inferOutputDtype(ReductionOp::Sum, dtype::b8), dtype::u32);
}

TEST(TypeInferenceTest, ReductionAminAmaxPreservesType) {
  for (auto op : {ReductionOp::Amin, ReductionOp::Amax}) {
    ASSERT_EQ(inferOutputDtype(op, dtype::f32), dtype::f32);
    ASSERT_EQ(inferOutputDtype(op, dtype::s32), dtype::s32);
    ASSERT_EQ(inferOutputDtype(op, dtype::u8), dtype::u8);
  }
}

// ======================== isElementwise / isFusable / isCommutative =========

TEST(TypeInferenceTest, AllUnaryOpsElementwise) {
  for (auto op : {
      UnaryOp::Exp, UnaryOp::Log, UnaryOp::Log1p,
      UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Sqrt,
      UnaryOp::Tanh, UnaryOp::Floor, UnaryOp::Ceil,
      UnaryOp::Rint, UnaryOp::Negative, UnaryOp::Absolute,
      UnaryOp::Sigmoid, UnaryOp::Erf, UnaryOp::LogicalNot,
      UnaryOp::Sign, UnaryOp::Isnan, UnaryOp::Isinf}) {
    ASSERT_TRUE(isElementwise(op));
    ASSERT_TRUE(isFusableElementwise(op));
  }
}

TEST(TypeInferenceTest, AllBinaryOpsElementwise) {
  for (auto op : {
      BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div,
      BinaryOp::Eq, BinaryOp::Neq, BinaryOp::Gt, BinaryOp::Gte,
      BinaryOp::Lt, BinaryOp::Lte, BinaryOp::Max, BinaryOp::Min,
      BinaryOp::Pow, BinaryOp::Mod, BinaryOp::And, BinaryOp::Or,
      BinaryOp::Shl, BinaryOp::Shr,
      BinaryOp::BitAnd, BinaryOp::BitOr, BinaryOp::BitXor}) {
    ASSERT_TRUE(isElementwise(op));
    ASSERT_TRUE(isFusableElementwise(op));
  }
}

TEST(TypeInferenceTest, CommutativeOps) {
  for (auto op : {
      BinaryOp::Add, BinaryOp::Mul, BinaryOp::Min, BinaryOp::Max,
      BinaryOp::Eq, BinaryOp::Neq, BinaryOp::And, BinaryOp::Or,
      BinaryOp::BitAnd, BinaryOp::BitOr, BinaryOp::BitXor}) {
    ASSERT_TRUE(isCommutative(op));
  }
}

TEST(TypeInferenceTest, NonCommutativeOps) {
  for (auto op : {
      BinaryOp::Sub, BinaryOp::Div, BinaryOp::Pow, BinaryOp::Mod,
      BinaryOp::Gt, BinaryOp::Gte, BinaryOp::Lt, BinaryOp::Lte,
      BinaryOp::Shl, BinaryOp::Shr}) {
    ASSERT_FALSE(isCommutative(op));
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
