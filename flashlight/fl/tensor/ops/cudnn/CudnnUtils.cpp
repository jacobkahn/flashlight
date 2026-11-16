/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/cudnn/CudnnUtils.h"

#include <array>
#include <stdexcept>
#include <unordered_map>

#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/runtime/DeviceManager.h"
#include "flashlight/fl/tensor/ops/cudnn/CudnnTensorHelpers.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/ops/cuda/CudaNumerics.h"

namespace {

struct DeviceHandle {
  cudnnHandle_t cudnnHandle;
  std::shared_ptr<fl::CUDAStream> stream;

  explicit DeviceHandle(std::shared_ptr<fl::CUDAStream> _stream)
    : cudnnHandle(nullptr), stream(_stream) {
    CUDNN_CHECK_ERR(cudnnCreate(&cudnnHandle));
    CUDNN_CHECK_ERR(cudnnSetStream(cudnnHandle, stream->handle()));
  }

  ~DeviceHandle() {
    if (cudnnHandle) {
// See https://git.io/fNQnM - sometimes, at exit, the CUDA context
// (or something) is already destroyed by the time a handle gets destroyed
// because of an issue with the destruction order.
#ifdef NO_CUDNN_DESTROY_HANDLE
#else
      CUDNN_CHECK_ERR(cudnnDestroy(cudnnHandle));
#endif
    }
  }
};

const float kFloatZero = 0.0;
const float kFloatOne = 1.0;

const double kDoubleZero = 0.0;
const double kDoubleOne = 1.0;

std::unordered_map<int, DeviceHandle> handles;

const DeviceHandle& getActiveDeviceHandle() {
  auto& manager = fl::DeviceManager::getInstance();
  auto& cudaDevice =
    manager.getActiveDevice(fl::DeviceType::CUDA).impl<fl::CUDADevice>();
  int id = cudaDevice.nativeId();
  // lazily initialize cuda stream for cudnn
  if (!handles.contains(id)) {
#ifdef NO_CUDNN_DESTROY_HANDLE
    // NOTE unmanaged so to avoid CUDA driver shut down prior to stream
    // destruction. This is safe because this object is always part of a global
    // map -- the resource won't be relased until program shutdown anyway.
    auto stream = fl::CUDAStream::createUnmanaged();
#else
    auto stream = fl::CUDAStream::createManaged();
#endif
    handles.emplace(id, DeviceHandle(stream));
  }
  return handles.at(id);
}

} // namespace

namespace fl {

void cudnnCheckErr(cudnnStatus_t status) {
  if (status == CUDNN_STATUS_SUCCESS) {
    return;
  }
  const char* err = cudnnGetErrorString(status);
  switch (status) {
    case CUDNN_STATUS_BAD_PARAM:
      throw std::invalid_argument(err);
    default:
      throw std::runtime_error(err);
  }
}

cudnnDataType_t cudnnMapToType(const fl::dtype& t) {
  switch (t) {
    case fl::dtype::f16:
      return CUDNN_DATA_HALF;
    case fl::dtype::f32:
      return CUDNN_DATA_FLOAT;
    case fl::dtype::f64:
      return CUDNN_DATA_DOUBLE;
    case fl::dtype::bf16:
      return CUDNN_DATA_BFLOAT16;
    default:
      throw std::invalid_argument("unsupported data type for cuDNN");
  }
}

cudnnPoolingMode_t cudnnMapToPoolingMode(const PoolingMode mode) {
  switch (mode) {
    case PoolingMode::MAX:
      return CUDNN_POOLING_MAX;
    case PoolingMode::AVG_INCLUDE_PADDING:
      return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    case PoolingMode::AVG_EXCLUDE_PADDING:
      return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    default:
      throw std::invalid_argument("unsupported pooling mode for cuDNN");
  }
}

cudnnRNNMode_t cudnnMapToRNNMode(const RnnMode mode) {
  switch (mode) {
    case RnnMode::RELU:
      return CUDNN_RNN_RELU;
    case RnnMode::TANH:
      return CUDNN_RNN_TANH;
    case RnnMode::LSTM:
      return CUDNN_LSTM;
    case RnnMode::GRU:
      return CUDNN_GRU;
    default:
      throw std::invalid_argument("unsupported RNN mode for cuDNN");
  }
}

TensorDescriptor::TensorDescriptor(const fl::dtype type, const Shape& flDims) {
  CUDNN_CHECK_ERR(cudnnCreateTensorDescriptor(&descriptor));
  cudnnDataType_t cudnntype = cudnnMapToType(type);

  std::array<int, 4> dims = {1, 1, 1, 1};
  for (unsigned i = 0; i < flDims.ndim(); ++i) {
    dims[i] = flDims[i];
  }

  // Contiguous NCHW strides
  std::vector<int> r_strides = {1};
  for (auto it = dims.rbegin(); it != dims.rend() - 1; ++it) {
    r_strides.push_back(r_strides.back() * (*it));
  }
  std::vector<int> strides(r_strides.rbegin(), r_strides.rend());

  CUDNN_CHECK_ERR(cudnnSetTensorNdDescriptor(
      descriptor, cudnntype, dims.size(), dims.data(), strides.data()));
}

TensorDescriptor::TensorDescriptor(
    const fl::dtype type,
    const Shape& flDims,
    const std::vector<int>& explicitStrides) {
  CUDNN_CHECK_ERR(cudnnCreateTensorDescriptor(&descriptor));
  cudnnDataType_t cudnntype = cudnnMapToType(type);

  std::array<int, 4> dims = {1, 1, 1, 1};
  for (unsigned i = 0; i < flDims.ndim(); ++i) {
    dims[i] = flDims[i];
  }

  std::array<int, 4> strides = {1, 1, 1, 1};
  for (unsigned i = 0; i < explicitStrides.size(); ++i) {
    strides[i] = explicitStrides[i];
  }

  CUDNN_CHECK_ERR(cudnnSetTensorNdDescriptor(
      descriptor, cudnntype, dims.size(), dims.data(), strides.data()));
}

TensorDescriptor::TensorDescriptor(const Tensor& input) {
  CUDNN_CHECK_ERR(cudnnCreateTensorDescriptor(&descriptor));
  cudnnDataType_t cudnntype = cudnnMapToType(input.type());

  auto flStrides = input.strides();
  auto flDims = input.shape();

  std::array<int, 4> strides = {1, 1, 1, 1};
  for (unsigned i = 0; i < flStrides.ndim(); ++i) {
    strides[i] = flStrides[i];
  }

  std::array<int, 4> dims = {1, 1, 1, 1};
  for (unsigned i = 0; i < flDims.ndim(); ++i) {
    dims[i] = flDims[i];
  }

  CUDNN_CHECK_ERR(cudnnSetTensorNdDescriptor(
      descriptor /* descriptor handle */,
      cudnntype /* = dataType */,
      4,
      dims.data(),
      strides.data()));
}

TensorDescriptor::~TensorDescriptor() {
  CUDNN_CHECK_ERR(cudnnDestroyTensorDescriptor(descriptor));
}

PoolingDescriptor::PoolingDescriptor(
    int wx,
    int wy,
    int sx,
    int sy,
    int px,
    int py,
    PoolingMode mode) {
  CUDNN_CHECK_ERR(cudnnCreatePoolingDescriptor(&descriptor));
  std::array<int, 2> window = {static_cast<int>(wy), static_cast<int>(wx)};
  std::array<int, 2> padding = {static_cast<int>(py), static_cast<int>(px)};
  std::array<int, 2> stride = {static_cast<int>(sy), static_cast<int>(sx)};

  auto cudnnpoolingmode = cudnnMapToPoolingMode(mode);
  CUDNN_CHECK_ERR(cudnnSetPoolingNdDescriptor(
      descriptor,
      cudnnpoolingmode,
      CUDNN_PROPAGATE_NAN,
      2,
      window.data(),
      padding.data(),
      stride.data()));
}

PoolingDescriptor::~PoolingDescriptor() {
  CUDNN_CHECK_ERR(cudnnDestroyPoolingDescriptor(descriptor));
}

DropoutDescriptor::DropoutDescriptor(float drop_prob, const Tensor& like) {
  CUDNN_CHECK_ERR(cudnnCreateDropoutDescriptor(&descriptor));
  auto cudnnHandle = getCudnnHandle();
  unsigned long long seed = 0;
  size_t state_size;
  CUDNN_CHECK_ERR(cudnnDropoutGetStatesSize(cudnnHandle, &state_size));
  auto& dropout_states = getDropoutStates();
  if (dropout_states.isEmpty()) {
    dropout_states =
        detail::makeTensorLike(
            like, {static_cast<long long>(state_size)}, fl::dtype::b8);
    DevicePtr statesraw(dropout_states);
    CUDNN_CHECK_ERR(cudnnSetDropoutDescriptor(
        descriptor, cudnnHandle, drop_prob, statesraw.get(), state_size, seed));
  } else {
    DevicePtr statesraw(dropout_states);
    CUDNN_CHECK_ERR(cudnnRestoreDropoutDescriptor(
        descriptor, cudnnHandle, drop_prob, statesraw.get(), state_size, seed));
  }
}

DropoutDescriptor::~DropoutDescriptor() {
  CUDNN_CHECK_ERR(cudnnDestroyDropoutDescriptor(descriptor));
}

Tensor& DropoutDescriptor::getDropoutStates() {
  thread_local Tensor dropout_states;
  return dropout_states;
}

RNNDescriptor::RNNDescriptor(
    fl::dtype type,
    int input_size,
    int hidden_size,
    int num_layers,
    RnnMode mode,
    bool bidirectional,
    DropoutDescriptor& dropout) {
  CUDNN_CHECK_ERR(cudnnCreateRNNDescriptor(&descriptor));

  cudnnRNNInputMode_t in_mode = CUDNN_LINEAR_INPUT;

  cudnnDirectionMode_t dir =
      bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;

  cudnnRNNMode_t cell = cudnnMapToRNNMode(mode);
  cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD;
  cudnnDataType_t cudnntype = cudnnMapToType(type);

  // Math type: use tensor ops with conversion for f16/bf16. For f32/f64,
  // CUDNN_DEFAULT_MATH would allow TF32, so it is only used when TF32 is
  // allowed (fl::cuda::detail::allowTF32()); otherwise restrict to FMA kernels.
  cudnnMathType_t mathType =
      (type == fl::dtype::f16 || type == fl::dtype::bf16)
      ? CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION
      : (fl::cuda::detail::allowTF32() ? CUDNN_DEFAULT_MATH : CUDNN_FMA_MATH);

  CUDNN_CHECK_ERR(cudnnSetRNNDescriptor_v8(
      descriptor,
      algo,
      cell,
      CUDNN_RNN_DOUBLE_BIAS,
      dir,
      in_mode,
      cudnntype,
      cudnntype,
      mathType,
      input_size,
      hidden_size,
      hidden_size, /* projSize = hiddenSize (no projection) */
      num_layers,
      dropout.descriptor,
      CUDNN_RNN_PADDED_IO_DISABLED));
}

RNNDescriptor::~RNNDescriptor() {
  CUDNN_CHECK_ERR(cudnnDestroyRNNDescriptor(descriptor));
}

cudnnHandle_t getCudnnHandle() {
  return getActiveDeviceHandle().cudnnHandle;
}

const CUDAStream& getCudnnStream() {
  return *getActiveDeviceHandle().stream;
}

const void* kOne(const fl::dtype t) {
  switch (t) {
    case fl::dtype::f16:
    case fl::dtype::bf16:
    case fl::dtype::f32:
      return &kFloatOne;
    case fl::dtype::f64:
      return &kDoubleOne;
    default:
      throw std::invalid_argument("unsupported data type for cuDNN");
  }
}

const void* kZero(const fl::dtype t) {
  switch (t) {
    case fl::dtype::f16:
    case fl::dtype::bf16:
    case fl::dtype::f32:
      return &kFloatZero;
    case fl::dtype::f64:
      return &kDoubleZero;
    default:
      throw std::invalid_argument("unsupported data type for cuDNN");
  }
}

} // namespace fl
