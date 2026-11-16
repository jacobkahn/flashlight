/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Separate .cpp (not .cu) because CPUTensor.h triggers NVCC internal errors.

#include "flashlight/fl/tensor/backend/cuda/CudaBackend.h"

#include <stdexcept>

#include <cuda_runtime.h>

#include "flashlight/fl/runtime/CUDAStream.h"
#include "flashlight/fl/runtime/CUDAUtils.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"

namespace fl {

void CudaBackend::copy(
    Tensor& dst, const Tensor& src, const Stream& stream) {
  // If dst is empty, allocate the destination buffer.
  if (dst.isEmpty()) {
    auto srcLoc = src.location();
    if (srcLoc == Location::Device) {
      // D2H: pinned host memory for truly async cudaMemcpyAsync.
      // TODO: replace with a pooled pinned allocator to avoid per-transfer
      // cudaMallocHost/cudaFreeHost overhead.
      size_t bytes = src.shape().elements() * fl::getTypeSize(src.type());
      void* ptr = nullptr;
      FL_CUDA_CHECK(cudaMallocHost(&ptr, bytes));
      auto owner = std::shared_ptr<void>(
          ptr, [](void* p) { cudaFreeHost(p); });
      dst = toTensor<CPUTensor>(src.shape(), src.type(), ptr, std::move(owner));
    } else {
      dst = empty(src.shape(), src.type());
    }
  }

  auto& cudaStr = stream.impl<CUDAStream>();
  auto srcLoc = src.location();
  auto dstLoc = dst.location();

  cudaMemcpyKind kind;
  if (srcLoc == Location::Host && dstLoc == Location::Device)
    kind = cudaMemcpyHostToDevice;
  else if (srcLoc == Location::Device && dstLoc == Location::Host)
    kind = cudaMemcpyDeviceToHost;
  else if (srcLoc == Location::Device && dstLoc == Location::Device)
    kind = cudaMemcpyDeviceToDevice;
  else
    throw std::invalid_argument(
        "[CudaBackend::copy] expected at least one Device tensor");

  // Handle strided src: make contiguous staging copy
  Tensor contigSrc = src.isContiguous() ? src : src.asContiguousTensor();

  void* dstPtr;
  dst.device(&dstPtr);
  void* srcPtr;
  contigSrc.device(&srcPtr);
  size_t bytes = dst.elements() * fl::getTypeSize(dst.type());

  FL_CUDA_CHECK(cudaMemcpyAsync(
      dstPtr, srcPtr, bytes, kind, cudaStr.handle()));
}

} // namespace fl
