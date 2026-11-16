/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/opt/backends/cuda/CudaFusedKernelLowering.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

#include "flashlight/fl/tensor/BackendRegistry.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/backend/cuda/CudaBackend.h"
#include "flashlight/fl/tensor/backend/cuda/CudaTensor.h"
#include "flashlight/fl/tensor/ir/CustomNode.h"
#include "flashlight/fl/tensor/ir/FusedElementwiseNode.h"
#include "flashlight/fl/tensor/ir/FusedReduceElementwiseNode.h"
#include "flashlight/fl/tensor/opt/backends/cuda/NvrtcKernelCompiler.h"

namespace fl {

namespace {

constexpr int kBlockSize = 256;

} // namespace

NodePtr CudaFusedKernelLowering::lowerFusedNode(NodePtr node) {
  const auto& fused = node->impl<FusedElementwiseNode>();
  const auto& program = fused.program();
  const unsigned numInputs = fused.numExternalInputs();
  const Shape outputShape = fused.shape();

  // Compile the kernel (or get from cache)
  CUfunction func =
      NvrtcKernelCompiler::getInstance().getOrCompile(program, numInputs);

  // Create a CustomNode that launches the compiled kernel
  auto outputDtype = node->dataType();
  auto evalFunc = [func, numInputs, outputShape, outputDtype](
                      const std::vector<const Tensor*>& inputs) -> Tensor {
    // Compute total element count
    Dim totalElements = 1;
    for (int i = 0; i < outputShape.ndim(); i++) {
      totalElements *= outputShape.dim(i);
    }
    int n = static_cast<int>(totalElements);

    // Cast non-f32 inputs to f32 and ensure contiguity.
    // The NVRTC kernel reads by linear index, so inputs must be contiguous.
    std::vector<Tensor> castInputs;
    castInputs.reserve(numInputs);
    for (unsigned i = 0; i < numInputs; i++) {
      auto inp = *inputs[i];
      if (inp.type() != fl::dtype::f32) {
        inp = inp.astype(fl::dtype::f32);
      }
      if (!inp.getAdapter<CudaTensor>().isContiguousRowMajor()) {
        inp = inp.asContiguousTensor();
      }
      castInputs.push_back(std::move(inp));
    }

    // Allocate output tensor
    auto result = detail::makeTensor(
        outputShape, fl::dtype::f32, TensorBackendType::Cuda);
    auto& outAdapter = result.getAdapter<CudaTensor>();

    // Get the CUDA stream from the backend
    auto stream = CudaBackend::getInstance().cudaStream();

    // Build kernel arguments: input pointers + output pointer + n
    std::vector<void*> inputPtrs(numInputs);
    for (unsigned i = 0; i < numInputs; i++) {
      auto& inAdapter = castInputs[i].getAdapter<CudaTensor>();
      inputPtrs[i] = inAdapter.devicePtr();
    }
    void* outPtr = outAdapter.devicePtr();

    // Build argument array for cuLaunchKernel
    std::vector<void*> args;
    args.reserve(numInputs + 2);
    for (unsigned i = 0; i < numInputs; i++) {
      args.push_back(&inputPtrs[i]);
    }
    args.push_back(&outPtr);
    args.push_back(&n);

    // Launch kernel
    int gridSize = (n + kBlockSize - 1) / kBlockSize;

    CUresult res = cuLaunchKernel(
        func,
        gridSize, 1, 1,    // grid dims
        kBlockSize, 1, 1,  // block dims
        0,                 // shared mem bytes
        stream,            // CUDA stream
        args.data(),       // kernel args
        nullptr);          // extra

    if (res != CUDA_SUCCESS) {
      const char* errStr = nullptr;
      cuGetErrorString(res, &errStr);
      throw std::runtime_error(
          std::string("[CudaFusedKernelLowering] cuLaunchKernel failed: ") +
          (errStr ? errStr : "unknown"));
    }

    // Cast back to the expected output dtype
    if (outputDtype != fl::dtype::f32) {
      return result.astype(outputDtype);
    }
    return result;
  };

  // Collect input nodes
  std::vector<NodePtr> inputNodes(
      node->inputs().begin(), node->inputs().end());

  auto customNode = CustomNode::create(
      "CudaFusedElementwise",
      std::move(inputNodes),
      outputShape,
      node->dataType(),
      std::move(evalFunc),
      node->stream());

  node->replaceAllUsesWith(customNode);
  return customNode;
}


NodePtr CudaFusedKernelLowering::lowerFusedReduceNode(NodePtr node) {
  const auto& fused = node->impl<FusedReduceElementwiseNode>();
  const auto& program = fused.program();
  const unsigned numInputs = fused.numExternalInputs();
  const Shape outputShape = fused.shape();
  const auto reductionOp = fused.reductionOp();
  const auto& axes = fused.axes();
  const bool keepDims = fused.keepDims();

  // Only fuse Sum, Mean, Amin, Amax, CountNonzero, Any, All
  // Median and Std need more complex kernels -- let the fallback handle them
  if (reductionOp == ReductionOp::Median ||
      reductionOp == ReductionOp::Std) {
    return node;
  }

  // The fused reduce kernel requires contiguous reduction axes.
  // If axes are non-contiguous (e.g. {0, 2, 3}), fall back to the unfused
  // evaluator which handles them step-by-step.
  {
    auto sortedAxes = axes;
    std::sort(sortedAxes.begin(), sortedAxes.end());
    for (size_t i = 1; i < sortedAxes.size(); i++) {
      if (sortedAxes[i] != sortedAxes[i-1] + 1) {
        return node;
      }
    }
  }

  // Compile the kernel
  CUfunction func =
      NvrtcKernelCompiler::getInstance().getOrCompileReduce(
          program, numInputs, reductionOp);

  // Capture the input shape for computing reduce_size and inner_size at runtime
  // We need the elementwise output shape (= the shape before reduction)
  // This is the shape of the first input (for unary) or broadcast shape
  // Since FusedReduceElementwise stores the reduction output shape,
  // we need to reconstruct the pre-reduction shape from the inputs
  // The elementwise output shape == first input's shape (all elementwise ops
  // are shape-preserving or broadcasting, and the inputs are already broadcast)

  auto outputDtype = node->dataType();
  auto evalFunc = [func, numInputs, outputShape, axes, keepDims, outputDtype](
                      const std::vector<const Tensor*>& inputs) -> Tensor {
    // Cast non-f32 inputs to f32 and ensure contiguity.
    // The NVRTC kernel reads by linear index, so inputs must be contiguous.
    std::vector<Tensor> castInputs;
    castInputs.reserve(numInputs);
    for (unsigned i = 0; i < numInputs; i++) {
      auto inp = *inputs[i];
      if (inp.type() != fl::dtype::f32) {
        inp = inp.astype(fl::dtype::f32);
      }
      if (!inp.getAdapter<CudaTensor>().isContiguousRowMajor()) {
        inp = inp.asContiguousTensor();
      }
      castInputs.push_back(std::move(inp));
    }

    // The elementwise intermediate shape is the input shape
    const Shape& elemShape = castInputs[0].shape();

    // Compute reduce_size and inner_size for the kernel
    // The kernel assumes contiguous layout: [outer, reduce, inner]
    // For single-axis reduction, outer = prod(dims before axis),
    //   reduce = dim(axis), inner = prod(dims after axis)
    // For multi-axis or all-axis, we collapse them
    std::vector<int> reduceAxes = axes;
    if (reduceAxes.empty()) {
      // All axes
      for (int i = 0; i < elemShape.ndim(); i++) {
        reduceAxes.push_back(i);
      }
    }

    // Sort axes
    std::sort(reduceAxes.begin(), reduceAxes.end());

    // Check if axes are contiguous
    bool contiguous = true;
    for (size_t i = 1; i < reduceAxes.size(); i++) {
      if (reduceAxes[i] != reduceAxes[i-1] + 1) {
        contiguous = false;
        break;
      }
    }

    if (!contiguous) {
      // Should not reach here -- non-contiguous axes are rejected at
      // optimization time. But as a safety net, fall back to the unfused
      // evaluator path by not producing a result.
      throw std::runtime_error(
          "[CudaFusedReduceKernel] Non-contiguous reduction axes "
          "should have been rejected at optimization time");
    }

    int firstReduceAxis = reduceAxes.front();
    int lastReduceAxis = reduceAxes.back();

    Dim innerSize = 1;
    for (int d = lastReduceAxis + 1; d < elemShape.ndim(); d++) {
      innerSize *= elemShape.dim(d);
    }

    Dim reduceSize = 1;
    for (int ax : reduceAxes) {
      reduceSize *= elemShape.dim(ax);
    }

    Dim numOutputElements = 1;
    for (int i = 0; i < outputShape.ndim(); i++) {
      numOutputElements *= outputShape.dim(i);
    }

    int nOut = static_cast<int>(numOutputElements);
    int nReduce = static_cast<int>(reduceSize);
    int nInner = static_cast<int>(innerSize);

    // Allocate output tensor
    auto result = detail::makeTensor(
        outputShape, fl::dtype::f32, TensorBackendType::Cuda);
    auto& outAdapter = result.getAdapter<CudaTensor>();

    auto stream = CudaBackend::getInstance().cudaStream();

    // Build kernel arguments
    std::vector<void*> inputPtrs(numInputs);
    for (unsigned i = 0; i < numInputs; i++) {
      auto& inAdapter = castInputs[i].getAdapter<CudaTensor>();
      inputPtrs[i] = inAdapter.devicePtr();
    }
    void* outPtr = outAdapter.devicePtr();

    std::vector<void*> args;
    args.reserve(numInputs + 4);
    for (unsigned i = 0; i < numInputs; i++) {
      args.push_back(&inputPtrs[i]);
    }
    args.push_back(&outPtr);
    args.push_back(&nOut);
    args.push_back(&nReduce);
    args.push_back(&nInner);

    // Cooperative reduction: one block per output element.
    // Each block of kBlockSize threads cooperatively reduces the reduction
    // dimension using warp shuffles + shared memory.
    int gridSize = nOut;
    // Shared memory for cross-warp reduction: one float per warp
    unsigned sharedMemBytes =
        (kBlockSize / 32) * static_cast<unsigned>(sizeof(float));
    CUresult res = cuLaunchKernel(
        func,
        gridSize, 1, 1,
        kBlockSize, 1, 1,
        sharedMemBytes,
        stream,
        args.data(),
        nullptr);

    if (res != CUDA_SUCCESS) {
      const char* errStr = nullptr;
      cuGetErrorString(res, &errStr);
      throw std::runtime_error(
          std::string("[CudaFusedReduceKernel] cuLaunchKernel failed: ") +
          (errStr ? errStr : "unknown"));
    }

    // Cast back to the expected output dtype
    if (outputDtype != fl::dtype::f32) {
      return result.astype(outputDtype);
    }
    return result;
  };

  std::vector<NodePtr> inputNodes(
      node->inputs().begin(), node->inputs().end());

  auto customNode = CustomNode::create(
      "CudaFusedReduce",
      std::move(inputNodes),
      outputShape,
      node->dataType(),
      std::move(evalFunc),
      node->stream());

  node->replaceAllUsesWith(customNode);
  return customNode;
}

NodePtr CudaFusedKernelLowering::rewrite(NodePtr node) {
  // Fused kernels compute in f32. That is exact for f16/bf16/b8 values and
  // standard for f32, but loses precision for f64 and is wrong for integers
  // (values above 2^24, integer division, bitwise ops). Leave other dtypes to
  // the unfused evaluator path, which dispatches per op and dtype.
  auto computesInF32 = [](fl::dtype t) {
    return t == fl::dtype::f32 || t == fl::dtype::f16 ||
        t == fl::dtype::bf16 || t == fl::dtype::b8;
  };
  if (!computesInF32(node->dataType())) {
    return node;
  }
  for (const auto& input : node->inputs()) {
    if (!computesInF32(input->dataType())) {
      return node;
    }
  }
  if (node->isFusedElementwise()) {
    return lowerFusedNode(node);
  }
  if (node->isFusedReduceElementwise()) {
    return lowerFusedReduceNode(node);
  }
  return node;
}

} // namespace fl
