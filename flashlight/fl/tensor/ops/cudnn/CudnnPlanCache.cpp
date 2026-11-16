/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/cudnn/CudnnPlanCache.h"

#include <stdexcept>

namespace fl {

namespace fe = cudnn_frontend;

fe::DataType_t flTypeToCudnn(fl::dtype type) {
  switch (type) {
    case fl::dtype::f16:
      return fe::DataType_t::HALF;
    case fl::dtype::bf16:
      return fe::DataType_t::BFLOAT16;
    case fl::dtype::f32:
      return fe::DataType_t::FLOAT;
    case fl::dtype::f64:
      return fe::DataType_t::DOUBLE;
    case fl::dtype::s32:
      return fe::DataType_t::INT32;
    case fl::dtype::b8:
      return fe::DataType_t::INT8;
    default:
      throw std::invalid_argument(
          "[CudnnPlanCache] unsupported dtype for cuDNN frontend");
  }
}

std::shared_ptr<fe::graph::Tensor_attributes> buildCudnnTensor(
    int64_t uid,
    const Shape& shape,
    fl::dtype type,
    bool isVirtual) {
  // Pad shape to 4D
  std::vector<int64_t> dims(4, 1);
  for (unsigned i = 0; i < shape.ndim() && i < 4; ++i) {
    dims[i] = shape[i];
  }

  // Compute contiguous row-major (NCHW) strides
  std::vector<int64_t> strides(4);
  strides[3] = 1;
  for (int i = 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }

  auto attrs = fe::graph::Tensor_attributes()
      .set_dim(dims)
      .set_stride(strides)
      .set_data_type(flTypeToCudnn(type))
      .set_uid(uid)
      .set_is_virtual(isVirtual);

  return std::make_shared<fe::graph::Tensor_attributes>(std::move(attrs));
}

CudnnPlanCache& CudnnPlanCache::getInstance() {
  static CudnnPlanCache instance;
  return instance;
}

const CudnnPlanCache::CachedPlan* CudnnPlanCache::get(size_t key) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = cache_.find(key);
  if (it == cache_.end()) {
    return nullptr;
  }
  return &it->second;
}

const CudnnPlanCache::CachedPlan* CudnnPlanCache::put(
    size_t key,
    CachedPlan plan) {
  std::lock_guard<std::mutex> lock(mu_);
  auto [it, _] = cache_.emplace(key, std::move(plan));
  return &it->second;
}

void CudnnPlanCache::clear() {
  std::lock_guard<std::mutex> lock(mu_);
  cache_.clear();
}

} // namespace fl
