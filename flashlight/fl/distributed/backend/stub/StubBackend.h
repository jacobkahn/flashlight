/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/distributed/DistributedApi.h"

#include "flashlight/fl/common/Defines.h"

namespace fl {

class FL_API StubBackend : public DistributedCommGroup {
 public:
  StubBackend();
  ~StubBackend();
  // TODO: should these methods be const this*?
  bool isInitialized() const override;
  DistributedInitInfo rendezvous() override;
  void allBroadcast(void* data, size_t bytes, int rootRank) const override;
  void allReduce(Tensor& input, bool async) const override;
  void allReduceMultiple(std::vector<Tensor*> arrs, bool async, bool contiguous)
      const override;
  void allBarrier() const override;
};

} // namespace fl
