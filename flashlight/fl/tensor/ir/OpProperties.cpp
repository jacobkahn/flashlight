/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/OpProperties.h"

#include <stdexcept>
#include <vector>

namespace fl {

dtype inferOutputDtype(UnaryOp op, dtype inputType) {
  switch (op) {
    // Predicate ops always return boolean
    case UnaryOp::LogicalNot:
    case UnaryOp::Isnan:
    case UnaryOp::Isinf:
      return dtype::b8;
    // All other unary ops preserve dtype
    case UnaryOp::Exp:
    case UnaryOp::Log:
    case UnaryOp::Log1p:
    case UnaryOp::Sin:
    case UnaryOp::Cos:
    case UnaryOp::Sqrt:
    case UnaryOp::Tanh:
    case UnaryOp::Floor:
    case UnaryOp::Ceil:
    case UnaryOp::Rint:
    case UnaryOp::Negative:
    case UnaryOp::Absolute:
    case UnaryOp::Sigmoid:
    case UnaryOp::Erf:
    case UnaryOp::Sign:
      return inputType;
  }
  throw std::runtime_error("[inferOutputDtype(UnaryOp)] Unknown op");
}

dtype inferOutputDtype(BinaryOp op, dtype lhsType, dtype rhsType) {
  switch (op) {
    // Comparison and logical ops always return boolean
    case BinaryOp::Eq:
    case BinaryOp::Neq:
    case BinaryOp::Gt:
    case BinaryOp::Gte:
    case BinaryOp::Lt:
    case BinaryOp::Lte:
    case BinaryOp::And:
    case BinaryOp::Or:
      return dtype::b8;
    // Arithmetic, bitwise, and min/max use type promotion
    case BinaryOp::Add:
    case BinaryOp::Sub:
    case BinaryOp::Mul:
    case BinaryOp::Div:
    case BinaryOp::Pow:
    case BinaryOp::Mod:
    case BinaryOp::Min:
    case BinaryOp::Max:
    case BinaryOp::Shl:
    case BinaryOp::Shr:
    case BinaryOp::BitAnd:
    case BinaryOp::BitOr:
    case BinaryOp::BitXor:
      return promoteDtypes(lhsType, rhsType);
  }
  throw std::runtime_error("[inferOutputDtype(BinaryOp)] Unknown op");
}

dtype inferOutputDtype(ReductionOp op, dtype inputType) {
  switch (op) {
    // Predicate reductions return boolean
    case ReductionOp::Any:
    case ReductionOp::All:
      return dtype::b8;
    // Count returns unsigned integer
    case ReductionOp::CountNonzero:
      return dtype::u32;
    // Mean/std/median promote integer inputs to f32
    case ReductionOp::Mean:
    case ReductionOp::Std:
    case ReductionOp::Median:
      return isFloatingPoint(inputType) ? inputType : dtype::f32;
    // Sum of boolean promotes to u32 (matching CPU backend behavior)
    case ReductionOp::Sum:
      return inputType == dtype::b8 ? dtype::u32 : inputType;
    // These preserve dtype
    case ReductionOp::Amin:
    case ReductionOp::Amax:
      return inputType;
  }
  throw std::runtime_error("[inferOutputDtype(ReductionOp)] Unknown op");
}

bool isElementwise(UnaryOp /* op */) {
  // All unary ops in the enum are elementwise
  return true;
}

bool isElementwise(BinaryOp /* op */) {
  // All binary ops in the enum are elementwise (with broadcasting)
  return true;
}

bool isFusableElementwise(UnaryOp op) {
  // All UnaryOps in our enum are pointwise-fusable
  // (shape/indexing ops like tril/triu/flip are first-class nodes outside
  // UnaryOp because they are not same-index pointwise operations)
  (void)op;
  return true;
}

bool isFusableElementwise(BinaryOp op) {
  // All BinaryOps are pointwise-fusable
  (void)op;
  return true;
}

bool isCommutative(BinaryOp op) {
  switch (op) {
    case BinaryOp::Add:
    case BinaryOp::Mul:
    case BinaryOp::Min:
    case BinaryOp::Max:
    case BinaryOp::Eq:
    case BinaryOp::Neq:
    case BinaryOp::And:
    case BinaryOp::Or:
    case BinaryOp::BitAnd:
    case BinaryOp::BitOr:
    case BinaryOp::BitXor:
      return true;
    default:
      return false;
  }
}

dtype inferFusedProgramOutputDtype(
    const std::vector<NodePtr>& inputs,
    const std::vector<FusedOp>& program,
    unsigned numExternalInputs) {
  std::vector<dtype> slotDtypes;
  slotDtypes.reserve(numExternalInputs + program.size());
  for (unsigned i = 0; i < numExternalInputs; ++i) {
    slotDtypes.push_back(inputs[i]->dataType());
  }
  for (const auto& step : program) {
    if (step.isUnary()) {
      slotDtypes.push_back(
          inferOutputDtype(step.unaryOp(), slotDtypes[step.src0]));
    } else {
      slotDtypes.push_back(
          inferOutputDtype(step.binaryOp(), slotDtypes[step.src0], slotDtypes[step.src1]));
    }
  }
  return slotDtypes.back();
}

} // namespace fl
