/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/distributed/DistributedApi.h"

namespace fl {
namespace detail {

void mpiCheck(int ec);

class MPIBackend : public DistributedInterface {
 public:
  MPIBackend();
  ~MPIBackend();
  bool isInitialized() override;
  DistributedInfo rendezvous() override;
  void allBroadcast(void* data, size_t bytes, int rootRank) override;
  void allReduce(Tensor& input, bool async) override;
  void allReduceMulitple(std::vector<Tensor*> arrs, bool async, bool contiguous)
      override;
  void allBarrier() override;
};

} // namespace detail
} // namespace fl

#define FL_MPI_CHECK(expr) ::fl::detail::mpiCheck((expr))
