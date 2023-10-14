/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/distributed/backend/mpi/MPIBackend.h"

#include <stdexcept>

#include <mpi.h>

namespace fl {
namespace detail {

void mpiCheck(int ec) {
  if (ec == MPI_SUCCESS) {
    return;
  } else {
    char buf[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(ec, buf, &resultlen);
    throw std::runtime_error(buf);
  }
}

MPIBackend::MPIBackend() {
  FL_MPI_CHECK(MPI_Init(nullptr, nullptr));
}

MPIBackend::~MPIBackend() {
  FL_MPI_CHECK(MPI_Finalize());
}

bool MPIBackend::isInitialized() {
  // Always initialized in the ctor
  return true;
}

// TODO: this is a function on an interface
DistributedInitInfo MPIBackend::rendezvous() {
  DistributedInitInfo info;
  FL_MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &info.worldRank));
  FL_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &info.worldSize));
  return info;
}

void MPIBackend::allBroadcast(void* data, size_t sizeInBytes, int rootRank) {
  FL_MPI_CHECK(
      MPI_Bcast(data, sizeInBytes, MPI_BYTE, rootRank, MPI_COMM_WORLD));
}

void MPIBackend::allReduce(Tensor& input, bool async) {
  throw std::runtime_error("MPIBackend::allReduce - not implemented");
}

void MPIBackend::allReduceMulitple(
    std::vector<Tensor*> arrs,
    bool async,
    bool contiguous) {
  throw std::runtime_error("MPIBackend::allReduceMultiple - not implemented");
}

void MPIBackend::allBarrier() {
  FL_MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
};

} // namespace detail
} // namespace fl
