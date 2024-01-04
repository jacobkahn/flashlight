/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

namespace fl {

enum class DistributedBackend {
  /// https://github.com/facebookincubator/gloo
  GLOO = 0,
  /// https://developer.nvidia.com/nccl
  NCCL = 1,
  STUB = 2,
  /// https://www.open-mpi.org/
  MPI = 3,
};

enum class DistributedInit {
  MPI = 0,
  FILE_SYSTEM = 1,
};

/**
 * Information returned from rendezvous/distributed initialization.
 */
struct DistributedInitInfo {
  int worldSize;
  int worldRank;
};

class Tensor;

/**
 * An instance of a communicator, initialized with state to represent a quorum
 * of processes that communicate. arbitrary data.
 */
class DistributedCommGroup {
 public:
  DistributedCommGroup();
  virtual ~DistributedCommGroup();
  virtual bool isInitialized() const = 0;
  virtual DistributedInitInfo rendezvous() = 0;
  virtual void allBroadcast(void* data, size_t bytes, int rootRank) const = 0;
  virtual void allReduce(Tensor& input, bool async) const = 0;
  // TODO: finish these
  // void sendData(const Device& device, void* data, size_t bytes) const = 0;
  // void recvData(const Device& device, void* data, size_t bytes) const = 0;
  virtual void allReduceMultiple(
      std::vector<Tensor*> arrs,
      bool async,
      bool contiguous) const = 0;
  virtual void allBarrier() const = 0;
};

} // namespace fl
