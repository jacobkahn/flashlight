/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace fl {

/**
 * Types of binary operations.
 */
enum class BinaryOp {
  Add,
  Sub,
  Mul,
  Div,
  Eq,
  Neq,
  Gt,
  Gte,
  Lt,
  Lte,
  Max,
  Min,
  Pow,
  Mod,
  And,
  Or,
  Shl,
  Shr,
  BitAnd,
  BitOr,
  BitXor,
};

/**
 * Types of unary operations.
 */
enum class UnaryOp {
  Exp,
  Log,
  Log1p,
  Sin,
  Cos,
  Sqrt,
  Tanh,
  Floor,
  Ceil,
  Rint,
  Negative,
  Absolute,
  Sigmoid,
  Erf,
  LogicalNot,
  Sign,
  Isnan,
  Isinf,
};

/**
 * Types of reduction operations.
 */
enum class ReductionOp {
  Amin,
  Amax,
  Sum,
  Mean,
  Median,
  Std,
  CountNonzero,
  Any,
  All,
};

} // namespace fl
