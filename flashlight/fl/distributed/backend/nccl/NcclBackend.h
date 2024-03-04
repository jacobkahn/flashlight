/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/distributed/DistributedApi.h"

#include <nccl.h>

#include "flashlight/fl/tensor/Types.h"

namespace fl::detail {

void ncclCheck(ncclResult_t r);

ncclDataType_t getNcclType(fl::dtype type);

ncclDataType_t getNcclType(const Tensor& tensor);

class NcclBackend : public DistributedCommGroup {
  // TODO: to be stored here:
  /*
    - coalesceBuffer
    - allocBuffer_
    - reductionStream_
    - workerStream_
    - ncclComm_t comm_
   */

 public:
  NcclBackend();
  ~NcclBackend();
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

#define FL_NCCL_CHECK(expr) ::fl::detail::ncclCheck((expr))
