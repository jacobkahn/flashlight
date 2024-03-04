/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/distributed/DistributedApi.h"

namespace fl::detail {

class GlooBackend : public DistributedCommGroup {
  std::shared_ptr<gloo::mpi::Context> glooContext_;

  const gloo::mpi::Context& globalContext() const;

  // TODO: move this out of the header so we don't need to have
  // gloo.h in the header
  template <typename T>
  inline void allreduceGloo(T* ptr, size_t s) {
    auto key = detail::makeHashKey(ptr, s, "allreduceCpu");
    auto algorithm = glooCache_.get(key);
    if (algorithm == nullptr) {
      using Allreduce = gloo::AllreduceHalvingDoubling<T>;
      algorithm = glooCache_.put(
          key,
          std::make_unique<Allreduce>(
              globalContext(),
              std::vector<T*>({ptr}),
              s,
              gloo::ReductionFunction<T>::sum));
    }
    algorithm->run();
  }

 public:
  GlooBackend();
  ~GlooBackend();
  // TODO: should these methods be const this*?
  bool isInitialized() override;
  DistributedInfo rendezvous() override;
  void allBroadcast(void* data, size_t bytes, int rootRank) override;
  void allReduce(Tensor& input, bool async) override;
  void allReduceMultiple(std::vector<Tensor*> arrs, bool async, bool contiguous)
      override;
  void sendData(const Device& device, void* data, size_t bytes) const override;
  void recvData(const Device& device, void* data, size_t bytes) const override;
  void allBarrier() override;
};

} // namespace fl::detail
