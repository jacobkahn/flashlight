/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/opt/passes/ScalarFolding.h"

#include "flashlight/fl/tensor/ir/BinaryNode.h"
#include "flashlight/fl/tensor/ir/ScalarNode.h"
#include "flashlight/fl/tensor/ir/UnaryNode.h"

#include <cmath>
#include <limits>
#include <optional>
#include <type_traits>

namespace fl {

namespace {

template <typename T>
std::optional<T> foldScalars(const T lhs, const T rhs, const BinaryOp op) {
  switch (op) {
    case BinaryOp::Add:
      return lhs + rhs;
    case BinaryOp::Sub:
      return lhs - rhs;
    case BinaryOp::Mul:
      return lhs * rhs;
    case BinaryOp::Div:
      // Integer division by zero (or MIN / -1) is undefined behavior; leave
      // it for the backend rather than trapping inside the optimizer.
      if constexpr (std::is_integral_v<T>) {
        if (rhs == 0 ||
            (std::is_signed_v<T> && rhs == T(-1) &&
             lhs == std::numeric_limits<T>::min())) {
          return std::nullopt;
        }
      }
      return lhs / rhs;
    case BinaryOp::Eq:
      return lhs == rhs;
    case BinaryOp::Neq:
      return lhs != rhs;
    case BinaryOp::Gt:
      return lhs > rhs;
    case BinaryOp::Gte:
      return lhs >= rhs;
    case BinaryOp::Lt:
      return lhs < rhs;
    case BinaryOp::Lte:
      return lhs <= rhs;
    case BinaryOp::Min:
      return std::min(lhs, rhs);
    case BinaryOp::Max:
      return std::max(lhs, rhs);
    case BinaryOp::Pow:
      return std::pow(lhs, rhs);
    // TODO the following support integral scalars, specialize template to
    // support them if it becomes profitable.
    case BinaryOp::And:
    case BinaryOp::Or:
    case BinaryOp::Mod:
    case BinaryOp::Shl:
    case BinaryOp::Shr:
    case BinaryOp::BitAnd:
    case BinaryOp::BitOr:
    case BinaryOp::BitXor:
      return std::nullopt;
  }
  throw std::runtime_error("[foldScalars] Unknown binary operation type");
}

// `resultType` is the binary node's dtype (e.g. b8 for comparisons), which
// differs from the operands' dtype `T`.
template <typename T>
std::optional<ScalarNodePtr> foldScalarNodes(
    const ScalarNode& lhs,
    const ScalarNode& rhs,
    const BinaryOp op,
    const dtype resultType,
    std::shared_ptr<const Stream> stream = nullptr) {
  T lhsVal = lhs.scalar<T>();
  T rhsVal = rhs.scalar<T>();
  std::optional<T> resVal = foldScalars(lhsVal, rhsVal, op);
  if (resVal.has_value()) {
    return ScalarNode::create(
        Shape(lhs.shape()), resultType, resVal.value(), std::move(stream));
  }
  return std::nullopt;
}

std::optional<ScalarNodePtr> foldScalarNodes(
    const ScalarNode& lhs,
    const ScalarNode& rhs,
    const BinaryOp op,
    const dtype resultType,
    std::shared_ptr<const Stream> stream = nullptr) {
  // TODO shape doesn't matter for scalars, support broadcast and add test
  if (lhs.shape() != rhs.shape() || lhs.dataType() != rhs.dataType()) {
    return std::nullopt;
  }
  const auto type = lhs.dataType();
  switch (type) {
    case dtype::f16:
    case dtype::bf16:
      return std::nullopt;
    case dtype::f32:
      return foldScalarNodes<float>(lhs, rhs, op, resultType, stream);
    case dtype::f64:
      return foldScalarNodes<double>(lhs, rhs, op, resultType, stream);
    case dtype::b8:
      return foldScalarNodes<char>(lhs, rhs, op, resultType, stream);
    case dtype::s16:
      return foldScalarNodes<short>(lhs, rhs, op, resultType, stream);
    case dtype::s32:
      return foldScalarNodes<int>(lhs, rhs, op, resultType, stream);
    case dtype::s64:
      return foldScalarNodes<long long>(lhs, rhs, op, resultType, stream);
    case dtype::u8:
      return foldScalarNodes<unsigned char>(lhs, rhs, op, resultType, stream);
    case dtype::u16:
      return foldScalarNodes<unsigned short>(lhs, rhs, op, resultType, stream);
    case dtype::u32:
      return foldScalarNodes<unsigned int>(lhs, rhs, op, resultType, stream);
    case dtype::u64:
      return foldScalarNodes<unsigned long long>(lhs, rhs, op, resultType, stream);
  }
  throw std::runtime_error("[foldScalarNodes] Unknown data type");
}

NodePtr foldScalarsInBinaryNode(BinaryNodePtr node) {
  const auto binop = node->op();
  const auto lhs = node->lhs();
  const auto rhs = node->rhs();
  if (lhs->isScalar() && rhs->isScalar()) {
    const auto& lhsScalar = lhs->impl<ScalarNode>();
    const auto& rhsScalar = rhs->impl<ScalarNode>();
    const auto optFoldedScalarNode =
        foldScalarNodes(
            lhsScalar, rhsScalar, binop, node->dataType(), node->stream());
    if (optFoldedScalarNode.has_value()) {
      const auto foldedScalarNode = optFoldedScalarNode.value();
      node->replaceAllUsesWith(foldedScalarNode);
      return foldedScalarNode;
    }
  }
  return node;
}

// Evaluate a unary op on a double-precision scalar value.
// Returns nullopt for type-changing ops (LogicalNot, Isnan, Isinf).
std::optional<double> foldUnaryScalar(double val, UnaryOp op) {
  switch (op) {
    case UnaryOp::Exp:
      return std::exp(val);
    case UnaryOp::Log:
      return std::log(val);
    case UnaryOp::Log1p:
      return std::log1p(val);
    case UnaryOp::Sin:
      return std::sin(val);
    case UnaryOp::Cos:
      return std::cos(val);
    case UnaryOp::Sqrt:
      return std::sqrt(val);
    case UnaryOp::Tanh:
      return std::tanh(val);
    case UnaryOp::Floor:
      return std::floor(val);
    case UnaryOp::Ceil:
      return std::ceil(val);
    case UnaryOp::Rint:
      return std::rint(val);
    case UnaryOp::Negative:
      return -val;
    case UnaryOp::Absolute:
      return std::abs(val);
    case UnaryOp::Sigmoid:
      return 1.0 / (1.0 + std::exp(-val));
    case UnaryOp::Erf:
      return std::erf(val);
    case UnaryOp::Sign:
      return (val > 0) ? 1.0 : (val < 0 ? -1.0 : 0.0);
    // Type-changing ops -- output dtype differs from input
    case UnaryOp::LogicalNot:
    case UnaryOp::Isnan:
    case UnaryOp::Isinf:
      return std::nullopt;
  }
  throw std::runtime_error("[foldUnaryScalar] Unknown unary operation type");
}

NodePtr foldScalarsInUnaryNode(UnaryNodePtr node) {
  if (!node->input()->isScalar()) {
    return node;
  }
  const auto& scalar = node->input()->impl<ScalarNode>();
  double val = scalar.scalar<double>();
  auto result = foldUnaryScalar(val, node->op());
  if (!result.has_value()) {
    return node;
  }
  auto folded = ScalarNode::create(
      node->shape(), scalar.dataType(), result.value(), node->stream());
  node->replaceAllUsesWith(folded);
  return folded;
}

} // namespace

NodePtr ScalarFolding::rewrite(NodePtr node) {
  switch (node->type()) {
    case NodeType::Binary:
      return foldScalarsInBinaryNode(Node::cast<BinaryNodePtr>(node));
    case NodeType::Unary:
      return foldScalarsInUnaryNode(Node::cast<UnaryNodePtr>(node));
    default:
      return node;
  }
}

} // namespace fl
