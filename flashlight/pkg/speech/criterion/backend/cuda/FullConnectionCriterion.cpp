/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/FullConnectionCriterion.h"

#include <stdexcept>

#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/runtime/CUDAStream.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/pkg/speech/criterion/CriterionUtils.h"

#include <flashlight/lib/sequence/criterion/cuda/FullConnectionCriterion.cuh>

using FCC = fl::lib::cuda::FullConnectionCriterion<float>;

namespace fl::pkg::speech {

static void backward(
    std::vector<Tensor>& inputs,
    const Tensor& gradVar,
    int B,
    int T,
    int N,
    const Tensor& trans,
    Tensor& workspace) {
  if (gradVar.type() != fl::dtype::f32) {
    throw std::invalid_argument("FCC: grad must be float32");
  }
  if (inputs.size() != 2) {
    throw std::invalid_argument(
        "FullConnectionCriterion backward expects two input args");
  }

  const auto& grad = gradVar;
  Tensor inputGrad({N, T, B}, fl::dtype::f32);
  Tensor transGrad({N, N}, fl::dtype::f32);

  {
    // CUDA kernels treat buffers as flat contiguous
    auto contiguousTrans = trans.asContiguousTensor();
    auto contiguousGrad = grad.asContiguousTensor();
    fl::DevicePtr transRaw(contiguousTrans);
    fl::DevicePtr gradRaw(contiguousGrad);
    fl::DevicePtr inputGradRaw(inputGrad);
    fl::DevicePtr transGradRaw(transGrad);
    fl::DevicePtr workspaceRaw(workspace);
    FCC::backward(
        B,
        T,
        N,
        static_cast<const float*>(transRaw.get()),
        static_cast<const float*>(gradRaw.get()),
        static_cast<float*>(inputGradRaw.get()),
        static_cast<float*>(transGradRaw.get()),
        workspaceRaw.get(),
        inputs[0].stream().impl<CUDAStream>().handle());
  }

  inputs[0].addGrad(Tensor(inputGrad, false));
  inputs[1].addGrad(Tensor(transGrad, false));
}

Tensor FullConnectionCriterion::forward(
    const Tensor& inputVar,
    const Tensor& targetVar) {
  if (inputVar.ndim() != 3) {
    throw std::invalid_argument(
        "FullConnectionCriterion::forward: "
        "expects input with dimensions {N, T, B}");
  }
  if (targetVar.ndim() != 2) {
    throw std::invalid_argument(
        "FullConnectionCriterion::forward: "
        "expects target with dimensions {B, L}");
  }

  const auto& transVar = param(0);
  int B = inputVar.dim(2);
  int T = inputVar.dim(1);
  int N = inputVar.dim(0);

  if (N != transVar.dim(0)) {
    throw std::invalid_argument("FCC: input dim doesn't match N");
  } else if (inputVar.type() != fl::dtype::f32) {
    throw std::invalid_argument("FCC: input must be float32");
  } else if (targetVar.type() != fl::dtype::s32) {
    throw std::invalid_argument("FCC: target must be int32");
  }

  const auto& input = inputVar;
  const auto& target = targetVar;
  const auto& targetSize = getTargetSizeArray(target, T);
  const auto& trans = transVar;
  Tensor loss({B}, fl::dtype::f32);
  Tensor workspace(
      {static_cast<long long>(FCC::getWorkspaceSize(B, T, N))}, fl::dtype::u8);

  {
    // CUDA kernels treat buffers as flat contiguous
    auto contiguousInput = input.asContiguousTensor();
    auto contiguousTrans = trans.asContiguousTensor();
    fl::DevicePtr inputRaw(contiguousInput);
    fl::DevicePtr targetSizeRaw(targetSize);
    fl::DevicePtr transRaw(contiguousTrans);
    fl::DevicePtr lossRaw(loss);
    fl::DevicePtr workspaceRaw(workspace);

    FCC::forward(
        B,
        T,
        N,
        scaleMode_,
        static_cast<const float*>(inputRaw.get()),
        static_cast<const int*>(targetSizeRaw.get()),
        static_cast<const float*>(transRaw.get()),
        static_cast<float*>(lossRaw.get()),
        workspaceRaw.get(),
        input.stream().impl<CUDAStream>().handle());
  }

  return Tensor(
      loss,
      {inputVar.withoutData(), transVar.withoutData()},
      [=](std::vector<Tensor>& inputs, const Tensor& gradVar) mutable {
        backward(inputs, gradVar, B, T, N, trans, workspace);
      });
}
} // namespace fl
