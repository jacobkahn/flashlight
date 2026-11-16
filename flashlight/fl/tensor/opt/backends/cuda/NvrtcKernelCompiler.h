/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <mutex>
#include <string>
#include <unordered_map>

#include <cuda.h>

#include "flashlight/fl/tensor/ir/FusedElementwiseNode.h"
#include "flashlight/fl/tensor/ir/ReductionNode.h"

namespace fl {

/**
 * A compiled CUDA kernel.
 */
struct CompiledKernel {
  CUmodule module{nullptr};
  CUfunction function{nullptr};

  ~CompiledKernel();

  // no copy
  CompiledKernel(const CompiledKernel&) = delete;
  CompiledKernel& operator=(const CompiledKernel&) = delete;
  CompiledKernel(CompiledKernel&& other) noexcept;
  CompiledKernel& operator=(CompiledKernel&& other) noexcept;
  CompiledKernel() = default;
};

/**
 * NVRTC-based kernel compiler for fused elementwise operations.
 *
 * Takes a FusedElementwiseNode's program and generates a CUDA kernel that
 * applies all operations in a single pass over the data. Compiled kernels
 * are cached by a key derived from the program structure.
 *
 * Thread-safe: multiple threads may compile/retrieve kernels concurrently.
 */
class NvrtcKernelCompiler {
 public:
  static NvrtcKernelCompiler& getInstance();

  /**
   * Get or compile a kernel for the given fused elementwise program.
   * Returns a CUfunction ready for launch.
   */
  CUfunction getOrCompile(
      const std::vector<FusedOp>& program,
      unsigned numExternalInputs);

  /**
   * Get or compile a fused reduce kernel. The kernel applies the elementwise
   * program to each input element inline and reduces along the specified axis.
   *
   * Currently supports single-axis reduction. The generated kernel assigns one
   * thread per output element, iterating over the reduction dimension.
   */
  CUfunction getOrCompileReduce(
      const std::vector<FusedOp>& program,
      unsigned numExternalInputs,
      ReductionOp reductionOp);

 private:
  NvrtcKernelCompiler() = default;

  static std::string makeKey(
      const std::vector<FusedOp>& program,
      unsigned numExternalInputs);

  static std::string makeReduceKey(
      const std::vector<FusedOp>& program,
      unsigned numExternalInputs,
      ReductionOp reductionOp);

  static std::string generateKernelSource(
      const std::vector<FusedOp>& program,
      unsigned numExternalInputs);

  static std::string generateReduceKernelSource(
      const std::vector<FusedOp>& program,
      unsigned numExternalInputs,
      ReductionOp reductionOp);

  static CompiledKernel compile(
      const std::string& source,
      const std::string& kernelName = "fused_elementwise");

  std::mutex mutex_;
  std::unordered_map<std::string, CompiledKernel> cache_;
};

} // namespace fl