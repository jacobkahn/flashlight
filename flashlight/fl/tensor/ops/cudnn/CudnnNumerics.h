/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cudnn_frontend.h>

#include "flashlight/fl/tensor/ops/cuda/CudaNumerics.h"
#include "flashlight/fl/tensor/Types.h"

namespace fl::detail::cudnn {

/**
 * Restrict a cuDNN frontend graph's candidate plans to strict f32/f64 math:
 * drops engines that compute on tensor cores (TF32 for f32 operands) or
 * down-convert inputs, unless TF32 is allowed
 * (`fl::cuda::detail::allowTF32()`).
 * `NVIDIA_TF32_OVERRIDE` does not affect graph-API engine selection, so this
 * filter is what keeps f32 strict. Call between `create_execution_plans()` and
 * `check_support()`; if no plan survives, `check_support()` fails.
 *
 * @param[in] hasWideOperand whether the graph reads or writes f32/f64 data.
 *   Half-precision graphs keep tensor-core engines.
 */
inline void restrictToStrictMath(
    cudnn_frontend::graph::Graph& graph,
    bool hasWideOperand) {
  if (!hasWideOperand || fl::cuda::detail::allowTF32()) {
    return;
  }
  graph.deselect_numeric_notes(
      {cudnn_frontend::NumericalNote_t::TENSOR_CORE,
       cudnn_frontend::NumericalNote_t::DOWN_CONVERT_INPUTS});
}

/// Whether `type` is f32 or f64, i.e. subject to strict-math engine filtering.
inline bool isWideFloat(fl::dtype type) {
  return type == fl::dtype::f32 || type == fl::dtype::f64;
}

} // namespace fl::detail::cudnn
