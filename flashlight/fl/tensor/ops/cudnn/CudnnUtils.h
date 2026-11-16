/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cudnn.h>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/runtime/CUDAStream.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl {

class TensorDescriptor {
 public:
  explicit TensorDescriptor(const Tensor& a);

  TensorDescriptor(const fl::dtype type, const Shape& dims);

  // Constructor with explicit strides for synthetic descriptors
  TensorDescriptor(
      const fl::dtype type,
      const Shape& dims,
      const std::vector<int>& strides);

  cudnnTensorDescriptor_t descriptor;
  ~TensorDescriptor();
};

class PoolingDescriptor {
 public:
  PoolingDescriptor(
      int wx,
      int wy,
      int sx,
      int sy,
      int px,
      int py,
      PoolingMode mode);
  cudnnPoolingDescriptor_t descriptor;
  ~PoolingDescriptor();
};

class DropoutDescriptor {
 public:
  /// `like` selects the backend that allocates the dropout state buffer (the
  /// backend of the op's input; see detail::makeTensorLike).
  DropoutDescriptor(float drop_prob, const Tensor& like);
  cudnnDropoutDescriptor_t descriptor;
  ~DropoutDescriptor();

  Tensor& getDropoutStates();
};

class RNNDescriptor {
 public:
  RNNDescriptor(
      fl::dtype type,
      int input_size,
      int hidden_size,
      int num_layers,
      RnnMode mode,
      bool bidirectional,
      DropoutDescriptor& dropout);
  cudnnRNNDescriptor_t descriptor;
  ~RNNDescriptor();
};

#define CUDNN_CHECK_ERR(expr) ::fl::cudnnCheckErr((expr))

void cudnnCheckErr(cudnnStatus_t status);

cudnnDataType_t cudnnMapToType(const fl::dtype& t);

const void* kOne(const fl::dtype t);

const void* kZero(const fl::dtype t);

cudnnHandle_t getCudnnHandle();
const CUDAStream& getCudnnStream();

} // namespace fl
