/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/cpu/CPUBackend.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/backend/cpu/CPUBackendUtils.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"
#include "flashlight/fl/tensor/ops/native/Unary.h"
#include "flashlight/fl/tensor/ops/onednn/OneDnnOps.h"

namespace fl {

using detail::enqueueNativeOp;

namespace {

// The native loops below (log1p, sin, cos, floor, ceil, erf, isnan) have f32
// and f64 paths only. Other dtypes compute in f32 and cast back, since the IR
// keeps the input dtype for these ops (matching the CUDA backend).
bool hasNativeFloatLoop(dtype type) {
  return type == dtype::f32 || type == dtype::f64;
}

} // namespace

/************************** Unary Operators ***************************/

Tensor CPUBackend::exp(const Tensor& tensor, std::shared_ptr<const Stream> /* stream */) {
  Tensor out;
  onednnOps_->applyEltwiseOp(out, tensor, dnnl::algorithm::eltwise_exp, nativeStream(), asyncStream_);
  return out;
}

Tensor CPUBackend::log(const Tensor& tensor, std::shared_ptr<const Stream> /* stream */) {
  Tensor out;
  onednnOps_->applyEltwiseOp(out, tensor, dnnl::algorithm::eltwise_log, nativeStream(), asyncStream_);
  return out;
}

Tensor CPUBackend::negative(const Tensor& tensor, std::shared_ptr<const Stream> /* stream */) {
  Tensor out;
  onednnOps_->applyEltwiseOp(out, tensor, dnnl::algorithm::eltwise_linear, nativeStream(), asyncStream_, -1);
  return out;
}

Tensor CPUBackend::logicalNot(const Tensor& tensor, std::shared_ptr<const Stream> /* stream */) {
  return tensor == 0;
}

Tensor CPUBackend::log1p(const Tensor& tensor, std::shared_ptr<const Stream> stream) {
  if (!hasNativeFloatLoop(tensor.type())) {
    return log1p(tensor.astype(dtype::f32), std::move(stream))
        .astype(tensor.type());
  }
  return enqueueNativeOp(tensor, asyncStream_,
      [](Tensor& out, const Tensor& in) {
        size_t n = in.shape().elements();
        if (in.type() == dtype::f32) {
          auto* d = static_cast<float*>(toCPUTensor(out).bufferPtr());
          const auto* s = static_cast<const float*>(toCPUTensor(in).bufferPtr());
          for (size_t i = 0; i < n; i++)
            d[i] = std::log1p(s[i]);
        } else if (in.type() == dtype::f64) {
          auto* d = static_cast<double*>(toCPUTensor(out).bufferPtr());
          const auto* s = static_cast<const double*>(toCPUTensor(in).bufferPtr());
          for (size_t i = 0; i < n; i++)
            d[i] = std::log1p(s[i]);
        }
      });
}

Tensor CPUBackend::sin(const Tensor& tensor, std::shared_ptr<const Stream> stream) {
  if (!hasNativeFloatLoop(tensor.type())) {
    return sin(tensor.astype(dtype::f32), std::move(stream))
        .astype(tensor.type());
  }
  return enqueueNativeOp(tensor, asyncStream_,
      [](Tensor& out, const Tensor& in) {
        size_t n = in.shape().elements();
        if (in.type() == dtype::f32) {
          auto* d = static_cast<float*>(toCPUTensor(out).bufferPtr());
          const auto* s = static_cast<const float*>(toCPUTensor(in).bufferPtr());
          for (size_t i = 0; i < n; i++)
            d[i] = std::sin(s[i]);
        } else if (in.type() == dtype::f64) {
          auto* d = static_cast<double*>(toCPUTensor(out).bufferPtr());
          const auto* s = static_cast<const double*>(toCPUTensor(in).bufferPtr());
          for (size_t i = 0; i < n; i++)
            d[i] = std::sin(s[i]);
        }
      });
}

Tensor CPUBackend::cos(const Tensor& tensor, std::shared_ptr<const Stream> stream) {
  if (!hasNativeFloatLoop(tensor.type())) {
    return cos(tensor.astype(dtype::f32), std::move(stream))
        .astype(tensor.type());
  }
  return enqueueNativeOp(tensor, asyncStream_,
      [](Tensor& out, const Tensor& in) {
        size_t n = in.shape().elements();
        if (in.type() == dtype::f32) {
          auto* d = static_cast<float*>(toCPUTensor(out).bufferPtr());
          const auto* s = static_cast<const float*>(toCPUTensor(in).bufferPtr());
          for (size_t i = 0; i < n; i++)
            d[i] = std::cos(s[i]);
        } else if (in.type() == dtype::f64) {
          auto* d = static_cast<double*>(toCPUTensor(out).bufferPtr());
          const auto* s = static_cast<const double*>(toCPUTensor(in).bufferPtr());
          for (size_t i = 0; i < n; i++)
            d[i] = std::cos(s[i]);
        }
      });
}

Tensor CPUBackend::sqrt(const Tensor& tensor, std::shared_ptr<const Stream> /* stream */) {
  Tensor out;
  onednnOps_->applyEltwiseOp(out, tensor, dnnl::algorithm::eltwise_sqrt, nativeStream(), asyncStream_);
  return out;
}

Tensor CPUBackend::tanh(const Tensor& tensor, std::shared_ptr<const Stream> /* stream */) {
  Tensor out;
  onednnOps_->applyEltwiseOp(out, tensor, dnnl::algorithm::eltwise_tanh, nativeStream(), asyncStream_);
  return out;
}

Tensor CPUBackend::floor(const Tensor& tensor, std::shared_ptr<const Stream> stream) {
  if (!isFloatingPoint(tensor.type())) {
    return tensor.copy(); // integral values are already whole
  }
  if (!hasNativeFloatLoop(tensor.type())) {
    return floor(tensor.astype(dtype::f32), std::move(stream))
        .astype(tensor.type());
  }
  return enqueueNativeOp(tensor, asyncStream_,
      [](Tensor& out, const Tensor& in) {
        size_t n = in.shape().elements();
        if (in.type() == dtype::f32) {
          auto* d = static_cast<float*>(toCPUTensor(out).bufferPtr());
          const auto* s = static_cast<const float*>(toCPUTensor(in).bufferPtr());
          for (size_t i = 0; i < n; i++)
            d[i] = std::floor(s[i]);
        } else if (in.type() == dtype::f64) {
          auto* d = static_cast<double*>(toCPUTensor(out).bufferPtr());
          const auto* s = static_cast<const double*>(toCPUTensor(in).bufferPtr());
          for (size_t i = 0; i < n; i++)
            d[i] = std::floor(s[i]);
        }
      });
}

Tensor CPUBackend::ceil(const Tensor& tensor, std::shared_ptr<const Stream> stream) {
  if (!isFloatingPoint(tensor.type())) {
    return tensor.copy(); // integral values are already whole
  }
  if (!hasNativeFloatLoop(tensor.type())) {
    return ceil(tensor.astype(dtype::f32), std::move(stream))
        .astype(tensor.type());
  }
  return enqueueNativeOp(tensor, asyncStream_,
      [](Tensor& out, const Tensor& in) {
        size_t n = in.shape().elements();
        if (in.type() == dtype::f32) {
          auto* d = static_cast<float*>(toCPUTensor(out).bufferPtr());
          const auto* s = static_cast<const float*>(toCPUTensor(in).bufferPtr());
          for (size_t i = 0; i < n; i++)
            d[i] = std::ceil(s[i]);
        } else if (in.type() == dtype::f64) {
          auto* d = static_cast<double*>(toCPUTensor(out).bufferPtr());
          const auto* s = static_cast<const double*>(toCPUTensor(in).bufferPtr());
          for (size_t i = 0; i < n; i++)
            d[i] = std::ceil(s[i]);
        }
      });
}

Tensor CPUBackend::rint(const Tensor& tensor, std::shared_ptr<const Stream> /* stream */) {
  Tensor out;
  onednnOps_->applyEltwiseOp(out, tensor, dnnl::algorithm::eltwise_round, nativeStream(), asyncStream_);
  return out;
}

Tensor CPUBackend::absolute(const Tensor& tensor, std::shared_ptr<const Stream> /* stream */) {
  // OneDNN eltwise_abs only supports float types (f32, f64, f16, bf16).
  // For integer types, use the native CPU fallback.
  switch (tensor.type()) {
    case dtype::f32:
    case dtype::f64:
    case dtype::f16: {
      Tensor out;
      onednnOps_->applyEltwiseOp(out, tensor, dnnl::algorithm::eltwise_abs, nativeStream(), asyncStream_);
      return out;
    }
    case dtype::s32:
    case dtype::b8:
    case dtype::u8:
    case dtype::u32:
      return enqueueNativeOp(tensor, asyncStream_,
          [](Tensor& out, const Tensor& in) {
            native::absoluteIntImpl(out, in);
          });
    default: {
      Tensor out;
      onednnOps_->applyEltwiseOp(out, tensor, dnnl::algorithm::eltwise_abs, nativeStream(), asyncStream_);
      return out;
    }
  }
}

Tensor CPUBackend::sigmoid(const Tensor& tensor, std::shared_ptr<const Stream> /* stream */) {
  Tensor out;
  onednnOps_->applyEltwiseOp(out, tensor, dnnl::algorithm::eltwise_logistic, nativeStream(), asyncStream_);
  return out;
}

Tensor CPUBackend::erf(const Tensor& tensor, std::shared_ptr<const Stream> stream) {
  if (!hasNativeFloatLoop(tensor.type())) {
    return erf(tensor.astype(dtype::f32), std::move(stream))
        .astype(tensor.type());
  }
  return enqueueNativeOp(tensor, asyncStream_,
      [](Tensor& out, const Tensor& in) {
        size_t n = in.shape().elements();
        if (in.type() == dtype::f32) {
          auto* d = static_cast<float*>(toCPUTensor(out).bufferPtr());
          const auto* s = static_cast<const float*>(toCPUTensor(in).bufferPtr());
          for (size_t i = 0; i < n; i++)
            d[i] = std::erff(s[i]);
        } else if (in.type() == dtype::f64) {
          auto* d = static_cast<double*>(toCPUTensor(out).bufferPtr());
          const auto* s = static_cast<const double*>(toCPUTensor(in).bufferPtr());
          for (size_t i = 0; i < n; i++)
            d[i] = std::erf(s[i]);
        }
      });
}

Tensor CPUBackend::isnan(const Tensor& tensor, std::shared_ptr<const Stream> stream) {
  if (isFloatingPoint(tensor.type()) && !hasNativeFloatLoop(tensor.type())) {
    return isnan(tensor.astype(dtype::f32), std::move(stream));
  }
  return enqueueNativeOp(tensor, tensor.shape(), dtype::b8, asyncStream_,
      [](Tensor& out, const Tensor& in) {
        size_t n = in.shape().elements();
        auto* d = static_cast<char*>(toCPUTensor(out).bufferPtr());
        if (in.type() == dtype::f32) {
          const auto* s = static_cast<const float*>(toCPUTensor(in).bufferPtr());
          for (size_t i = 0; i < n; i++)
            d[i] = std::isnan(s[i]) ? 1 : 0;
        } else if (in.type() == dtype::f64) {
          const auto* s = static_cast<const double*>(toCPUTensor(in).bufferPtr());
          for (size_t i = 0; i < n; i++)
            d[i] = std::isnan(s[i]) ? 1 : 0;
        } else {
          std::memset(d, 0, n); // integral types are never NaN
        }
      });
}

Tensor CPUBackend::isinf(const Tensor& tensor, std::shared_ptr<const Stream> /* stream */) {
  return tensor == std::numeric_limits<float>::infinity() ||
      tensor == (-1 * std::numeric_limits<float>::infinity());
}

Tensor CPUBackend::sign(const Tensor& tensor, std::shared_ptr<const Stream> /* stream */) {
  auto pos = (tensor > 0).astype(tensor.type());
  auto neg = (tensor < 0).astype(tensor.type());
  return pos - neg;
}

} // namespace fl
