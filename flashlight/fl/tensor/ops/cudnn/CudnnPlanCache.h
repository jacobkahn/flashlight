/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <cudnn_frontend.h>

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Types.h"

namespace fl {

namespace fe = cudnn_frontend;

/**
 * Maps fl::dtype to cudnn_frontend DataType_t.
 */
fe::DataType_t flTypeToCudnn(fl::dtype type);

/**
 * Build a cudnn_frontend tensor descriptor from shape, dtype, and an
 * integer UID. Assumes contiguous row-major (NCHW) strides.
 *
 * @param uid Unique identifier within a cuDNN graph for this tensor.
 * @param shape Tensor shape (up to 4D, padded to 4D with 1s).
 * @param type Flashlight dtype.
 * @param isVirtual If true, the tensor is an intermediate (no backing memory).
 * @return A shared_ptr to the created cudnn_frontend Tensor_attributes.
 */
std::shared_ptr<fe::graph::Tensor_attributes> buildCudnnTensor(
    int64_t uid,
    const Shape& shape,
    fl::dtype type,
    bool isVirtual = false);

/**
 * Thread-safe cache for compiled cuDNN execution plans.
 *
 * Keys are the graph topology hash returned by graph->key(). On miss,
 * the caller builds + compiles the graph and stores it. Plans are valid
 * for the lifetime of the cuDNN handle (same device, same driver).
 */
class CudnnPlanCache {
 public:
  struct CachedPlan {
    std::shared_ptr<fe::graph::Graph> graph;
    // Tensor UIDs for mapping JIT inputs to cuDNN variant pack
    std::vector<int64_t> inputUids;
    int64_t outputUid;
    size_t workspaceSize;
  };

  static CudnnPlanCache& getInstance();

  // Returns nullptr if not found.
  const CachedPlan* get(size_t key) const;

  // Stores a plan. Returns a pointer to the stored plan.
  const CachedPlan* put(size_t key, CachedPlan plan);

  void clear();

 private:
  CudnnPlanCache() = default;
  mutable std::mutex mu_;
  std::unordered_map<size_t, CachedPlan> cache_;
};

} // namespace fl
