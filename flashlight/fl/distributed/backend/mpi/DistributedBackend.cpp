/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#if FL_USE_MPI
#include <mpi.h>
#endif // FL_USE_MPI

void mpiCheck(int ec) {
#if FL_USE_MPI
  if (ec == MPI_SUCCESS) {
    return;
  } else {
    char buf[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(ec, buf, &resultlen);
    throw std::runtime_error(buf);
  }
#else // FL_USE_MPI
  throw std::runtime_error(
      "mpiCheck: Flashlight not built with MPI support set "
      "FL_USE_MPI to be on");
#endif // FL_USE_MPI
}

#define MPICHECK(expr) ::fl::detail::mpiCheck((expr))

// TODO: move this to a common interface
struct DistributedInitInfo {
  int worldSize;
  int worldRank;
};

// TODO: this is a function on an interface
DistributedInitInfo initWithMPI() {
  DistributedInitInfo info;

  // initializing MPI
  MPICHECK(MPI_Init(nullptr, nullptr));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &info.worldRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &info.worldSize));

  return info;
}

void broadcast(void* data, size_t sizeInBytes int rootRank) {
  MPICHECK(MPI_Bcast(data, sizeInBytes, MPI_BYTE, rootRank, MPI_COMM_WORLD));
}
