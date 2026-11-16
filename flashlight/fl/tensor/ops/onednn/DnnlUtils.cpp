/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/onednn/DnnlUtils.h"

#include <stdexcept>
#include <utility>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/runtime/AsyncCPUStream.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/ops/onednn/Utils.h"

namespace fl::detail {

DnnlMemoryWrapper::DnnlMemoryWrapper(
    const Tensor& tensor,
    dnnl::memory::dims dims,
    dnnl::memory::format_tag format,
    const dnnl::engine& engine) {
  devicePtr_ = fl::DevicePtr(tensor);
  void* buffer = devicePtr_.get();
  descriptor_ =
      dnnl::memory::desc({dims}, detail::flToOneDnnType(tensor.type()), format);
  memory_ = dnnl::memory(descriptor_, engine, buffer);
}

DnnlMemoryWrapper& DnnlMemoryWrapper::operator=(DnnlMemoryWrapper&& other) {
  devicePtr_ = std::move(other.devicePtr_);
  memory_ = std::move(other.memory_);
  descriptor_ = std::move(other.descriptor_);
  return *this;
}

dnnl::memory DnnlMemoryWrapper::getMemory() const {
  return memory_;
}

dnnl::memory::desc DnnlMemoryWrapper::getDescriptor() const {
  return descriptor_;
}

dnnl::memory dnnlAlignOrdering(
    std::vector<dnnl::primitive>& net,
    std::vector<std::unordered_map<int, dnnl::memory>>& netArgs,
    const dnnl::memory& memory,
    const dnnl::memory::desc& desc,
    const dnnl::engine& engine) {
  auto memoryOut = memory;
  if (memory.get_desc() != desc) {
    // use the ordering requested by the descriptor
    memoryOut = dnnl::memory(desc, engine);
    net.push_back(dnnl::reorder(memory, memoryOut));
    netArgs.push_back({{DNNL_ARG_FROM, memory}, {DNNL_ARG_TO, memoryOut}});
  }
  return memoryOut;
}

void executeNetwork(
    std::vector<dnnl::primitive> net,
    std::vector<std::unordered_map<int, dnnl::memory>> netArgs,
    AsyncCPUStream& stream,
    dnnl::stream& nativeStream,
    std::vector<std::shared_ptr<CPUTensor::SharedData>> keepAlive) {
  if (net.size() != netArgs.size()) {
    throw std::invalid_argument(
        "executeNetwork - given different size nets and netArgs");
  }
  // Sync first to drain all pending async tasks (e.g. input.flat(i) = val)
  // so input data is ready, then execute synchronously on the calling thread.
  // We cannot enqueue DNNL primitive execution on the async worker thread
  // because DNNL primitives use thread-local scratchpad/workspace memory
  // that must match the thread of execution. Heavyweight operations
  // (conv2d, pool2d, batchnorm, rnn) have negligible sync overhead.
  //
  stream.sync();
  for (size_t i = 0; i < net.size(); ++i) {
    net.at(i).execute(nativeStream, netArgs.at(i));
  }
  nativeStream.wait();
}

dnnl::algorithm dnnlMapToPoolingMode(const PoolingMode mode) {
  switch (mode) {
    case PoolingMode::MAX:
      return dnnl::algorithm::pooling_max;
    case PoolingMode::AVG_INCLUDE_PADDING:
      return dnnl::algorithm::pooling_avg_include_padding;
    case PoolingMode::AVG_EXCLUDE_PADDING:
      return dnnl::algorithm::pooling_avg_exclude_padding;
    default:
      throw std::invalid_argument("unsupported pooling mode for OneDNN");
  }
}

} // namespace fl
