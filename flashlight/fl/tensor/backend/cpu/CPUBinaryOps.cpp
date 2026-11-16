/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/cpu/CPUBackend.h"

#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/ops/native/Binary.h"
#include "flashlight/fl/tensor/ops/native/CpuTensorHelpers.h"
#include "flashlight/fl/tensor/ops/onednn/OneDnnOps.h"

namespace fl {

/************************** Binary Operators ***************************/
#define FL_CPU_BINARY_OP_DEF(FUNC, ALG)                           \
  Tensor CPUBackend::FUNC(const Tensor& lhs, const Tensor& rhs,   \
      std::shared_ptr<const Stream> /* stream */) {                    \
    Tensor out;                                                        \
    onednnOps_->applyBinop(out, lhs, rhs, ALG, nativeStream(), asyncStream_); \
    return out;                                                        \
  }

#define FL_CPU_BINARY_LOGICAL_OP_DEF(FUNC, ALG)                   \
  Tensor CPUBackend::FUNC(const Tensor& lhs, const Tensor& rhs,   \
      std::shared_ptr<const Stream> /* stream */) {                    \
    Tensor out;                                                        \
    onednnOps_->applyBinop(out, lhs, rhs, ALG, nativeStream(), asyncStream_, \
        dnnl::memory::data_type::s8);                                 \
    return out;                                                        \
  }

FL_CPU_BINARY_OP_DEF(add, dnnl::algorithm::binary_add);
FL_CPU_BINARY_OP_DEF(sub, dnnl::algorithm::binary_sub);
FL_CPU_BINARY_OP_DEF(mul, dnnl::algorithm::binary_mul);
FL_CPU_BINARY_OP_DEF(div, dnnl::algorithm::binary_div);
FL_CPU_BINARY_LOGICAL_OP_DEF(eq, dnnl::algorithm::binary_eq);
FL_CPU_BINARY_LOGICAL_OP_DEF(neq, dnnl::algorithm::binary_ne);
FL_CPU_BINARY_LOGICAL_OP_DEF(lessThan, dnnl::algorithm::binary_lt);
FL_CPU_BINARY_LOGICAL_OP_DEF(lessThanEqual, dnnl::algorithm::binary_le);
FL_CPU_BINARY_LOGICAL_OP_DEF(greaterThan, dnnl::algorithm::binary_gt);
FL_CPU_BINARY_LOGICAL_OP_DEF(greaterThanEqual, dnnl::algorithm::binary_ge);
#undef FL_CPU_BINARY_OP_DEF
#undef FL_CPU_BINARY_LOGICAL_OP_DEF

Tensor CPUBackend::mod(const Tensor& lhs, const Tensor& rhs, std::shared_ptr<const Stream> /* stream */) {
  auto commonType = promoteDtypes(lhs.type(), rhs.type());
  auto lhsP = (lhs.type() == commonType) ? lhs : lhs.astype(commonType);
  auto rhsP = (rhs.type() == commonType) ? rhs : rhs.astype(commonType);
  return fl::native::dispatchByType(commonType, [&](auto typeTag) -> Tensor {
    using T = decltype(typeTag);
    return fl::native::cpuBinaryOpWithBroadcast<T>(lhsP, rhsP, [](T a, T b) -> T {
      if constexpr (std::is_floating_point_v<T>) {
        return std::fmod(a, b);
      } else {
        return a % b;
      }
    });
  });
}

Tensor CPUBackend::bitwiseAnd(const Tensor& lhs, const Tensor& rhs, std::shared_ptr<const Stream> /* stream */) {
  auto commonType = promoteDtypes(lhs.type(), rhs.type());
  auto lhsP = (lhs.type() == commonType) ? lhs : lhs.astype(commonType);
  auto rhsP = (rhs.type() == commonType) ? rhs : rhs.astype(commonType);
  return fl::native::dispatchByType(commonType, [&](auto typeTag) -> Tensor {
    using T = decltype(typeTag);
    if constexpr (std::is_integral_v<T>) {
      return fl::native::cpuBinaryOpWithBroadcast<T>(
          lhsP, rhsP, [](T a, T b) -> T { return a & b; });
    } else {
      throw std::invalid_argument(
          "[CPUBackend::bitwiseAnd] not supported for floating point types");
    }
  });
}

Tensor CPUBackend::bitwiseOr(const Tensor& lhs, const Tensor& rhs, std::shared_ptr<const Stream> /* stream */) {
  auto commonType = promoteDtypes(lhs.type(), rhs.type());
  auto lhsP = (lhs.type() == commonType) ? lhs : lhs.astype(commonType);
  auto rhsP = (rhs.type() == commonType) ? rhs : rhs.astype(commonType);
  return fl::native::dispatchByType(commonType, [&](auto typeTag) -> Tensor {
    using T = decltype(typeTag);
    if constexpr (std::is_integral_v<T>) {
      return fl::native::cpuBinaryOpWithBroadcast<T>(
          lhsP, rhsP, [](T a, T b) -> T { return a | b; });
    } else {
      throw std::invalid_argument(
          "[CPUBackend::bitwiseOr] not supported for floating point types");
    }
  });
}

Tensor CPUBackend::bitwiseXor(const Tensor& lhs, const Tensor& rhs, std::shared_ptr<const Stream> /* stream */) {
  auto commonType = promoteDtypes(lhs.type(), rhs.type());
  auto lhsP = (lhs.type() == commonType) ? lhs : lhs.astype(commonType);
  auto rhsP = (rhs.type() == commonType) ? rhs : rhs.astype(commonType);
  return fl::native::dispatchByType(commonType, [&](auto typeTag) -> Tensor {
    using T = decltype(typeTag);
    if constexpr (std::is_integral_v<T>) {
      return fl::native::cpuBinaryOpWithBroadcast<T>(
          lhsP, rhsP, [](T a, T b) -> T { return a ^ b; });
    } else {
      throw std::invalid_argument(
          "[CPUBackend::bitwiseXor] not supported for floating point types");
    }
  });
}

Tensor CPUBackend::lShift(const Tensor& lhs, const Tensor& rhs, std::shared_ptr<const Stream> /* stream */) {
  auto commonType = promoteDtypes(lhs.type(), rhs.type());
  auto lhsP = (lhs.type() == commonType) ? lhs : lhs.astype(commonType);
  auto rhsP = (rhs.type() == commonType) ? rhs : rhs.astype(commonType);
  return fl::native::dispatchByType(commonType, [&](auto typeTag) -> Tensor {
    using T = decltype(typeTag);
    if constexpr (std::is_integral_v<T>) {
      return fl::native::cpuBinaryOpWithBroadcast<T>(
          lhsP, rhsP, [](T a, T b) -> T { return a << b; });
    } else {
      throw std::invalid_argument(
          "[CPUBackend::lShift] not supported for floating point types");
    }
  });
}

Tensor CPUBackend::rShift(const Tensor& lhs, const Tensor& rhs, std::shared_ptr<const Stream> /* stream */) {
  auto commonType = promoteDtypes(lhs.type(), rhs.type());
  auto lhsP = (lhs.type() == commonType) ? lhs : lhs.astype(commonType);
  auto rhsP = (rhs.type() == commonType) ? rhs : rhs.astype(commonType);
  return fl::native::dispatchByType(commonType, [&](auto typeTag) -> Tensor {
    using T = decltype(typeTag);
    if constexpr (std::is_integral_v<T>) {
      return fl::native::cpuBinaryOpWithBroadcast<T>(
          lhsP, rhsP, [](T a, T b) -> T { return a >> b; });
    } else {
      throw std::invalid_argument(
          "[CPUBackend::rShift] not supported for floating point types");
    }
  });
}

Tensor CPUBackend::logicalAnd(const Tensor& lhs, const Tensor& rhs, std::shared_ptr<const Stream> /* stream */) {
  return fl::native::logicalAndCpu(lhs, rhs);
}

Tensor CPUBackend::logicalOr(const Tensor& lhs, const Tensor& rhs, std::shared_ptr<const Stream> /* stream */) {
  return fl::native::logicalOrCpu(lhs, rhs);
}

Tensor CPUBackend::minimum(const Tensor& lhs, const Tensor& rhs, std::shared_ptr<const Stream> /* stream */) {
  Tensor out;
  onednnOps_->applyBinop(out, lhs, rhs, dnnl::algorithm::binary_min, nativeStream(), asyncStream_);
  return out;
}

// scalar-double overloads: use base class defaults (TensorBackend)
// which handle type promotion correctly via promoteScalarDtype/promoteDtypes.

Tensor CPUBackend::maximum(const Tensor& lhs, const Tensor& rhs, std::shared_ptr<const Stream> /* stream */) {
  Tensor out;
  onednnOps_->applyBinop(out, lhs, rhs, dnnl::algorithm::binary_max, nativeStream(), asyncStream_);
  return out;
}

Tensor CPUBackend::power(const Tensor& lhs, const Tensor& rhs, std::shared_ptr<const Stream> /* stream */) {
  auto commonType = promoteDtypes(lhs.type(), rhs.type());
  auto lhsP = (lhs.type() == commonType) ? lhs : lhs.astype(commonType);
  auto rhsP = (rhs.type() == commonType) ? rhs : rhs.astype(commonType);
  return fl::native::dispatchByType(commonType, [&](auto typeTag) {
    using T = decltype(typeTag);
    return fl::native::cpuBinaryOpWithBroadcast<T>(lhsP, rhsP, [](T a, T b) -> T {
      return static_cast<T>(
          std::pow(static_cast<double>(a), static_cast<double>(b)));
    });
  });
}

/************************** BLAS ***************************/

void CPUBackend::matmul(
    Tensor& out,
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp,
    std::shared_ptr<const Stream> /* stream */) {
  onednnOps_->matmul(out, lhs, rhs, lhsProp, rhsProp, nativeStream(), asyncStream_);
}

} // namespace fl
