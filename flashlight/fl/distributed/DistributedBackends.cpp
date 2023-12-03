/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/distributed/DistributedBackends.h"

#if (                                                                         \
    !defined(FL_USE_NCCL) || !defined(FL_USE_GLOO) || !defined(FL_USE_MPI) || \
    !defined(FL_USE_STUB))
#error \
    "Missing FL_USE_NCCL, FL_USE_GLOO, FL_USE_MPI, FL_USE_STUB, or other "\
    "distributed preproc definitions. Cannot compile the distributed API."
#endif

#if FL_USE_NCCL
#include "flashlight/fl/distributed/backend/nccl/NcclBackend.h"
#endif

#if FL_USE_GLOO
#include "flashlight/fl/distributed/backend/gloo/GlooBackend.h"
#endif

#if FL_USE_MPI
#include "flashlight/fl/distributed/backend/mpi/MpiBackend.h"
#endif

// Always available
#include "flashlight/fl/distributed/backend/stub/StubBackend.h"

namespace fl {
namespace {

std::shared_ptr<DistributedCommGroup> commGroupFromBackendType(
    DistributedBackend backend) {
  switch (backend) {
#if FL_USE_NCCL
    case DistributedBackend::NCCL:
      return std::make_shared<NcclBackend>();
#endif
#if FL_USE_GLOO
    case DistributedBackend::GLOO:
      return std::make_shared<GlooBackend>();
#endif
#if FL_USE_MPI
    case DistributedBackend::MPI:
      commGroup = std::make_shared<MPIBackend>();
#endif
    case DistributedBackend::STUB:
      return std::make_shared<StubBackend>();
    default:
      throw std::runtime_error(
          "commGroupFromBackendType: unreachable state; backend not included");
  }
}

const std::unordered_map<DistributedBackend, bool>
    kDistributedBackendCompileOptions = {
        {DistributedBackend::MPI, FL_USE_MPI},
        {DistributedBackend::NCCL, FL_USE_NCCL},
        {DistributedBackend::GLOO, FL_USE_GLOO},
        {DistributedBackend::STUB, FL_USE_STUB}};

} // namespace

bool distributedBackendAvailable(DistributedBackend backendType) {
  auto iter = kDistributedBackendCompileOptions.find(backendType);
  if (iter == kDistributedBackendCompileOptions.end()) {
    throw std::invalid_argument("distributedBackendAvailable: invalid backend");
  }
  return static_cast<bool>(iter->second);
}

std::string toString(DistributedBackend backend) {
  switch (backend) {
    case DistributedBackend::NCCL:
      return "NCCL";
    case DistributedBackend::GLOO:
      return "GLOO";
    case DistributedBackend::STUB:
      return "STUB";
    default:
      throw std::runtime_error("toString: invalid DistributedBackend type");
  }
}

std::string toString(DistributedInit initMethod) {
  switch (initMethod) {
    case DistributedInit::MPI:
      return "MPI";
    case DistributedInit::FILE_SYSTEM:
      return "FILE_SYSTEM";
    default:
      throw std::runtime_error("toString: invalid DistributedInit type");
  }
}

namespace detail {

/*  static */ DistributedInfo& DistributedInfo::getInstance() {
  static DistributedInfo dinfo;
  return dinfo;
}

const DistributedCommGroup& DistributedInfo::getInterface(
    DistributedBackend backend) const {
  if (!defaultCommGroupExists(backend)) {
    throw std::invalid_argument("" + toString(backend));
  }
  return *defaultCommGroups_.at(backend);
}

bool DistributedInfo::defaultCommGroupExists(DistributedBackend backend) const {
  return defaultCommGroups_.find(backend) != defaultCommGroups_.end();
}

bool DistributedInfo::setDefaultCommGroup(
    DistributedBackend backend,
    std::shared_ptr<DistributedCommGroup> commGroup) {
  // Check if a default already exists for this backend type
  if (defaultCommGroupExists(backend)) {
    return false;
  } else {
    defaultCommGroups_.emplace(backend, std::move(commGroup));
    return true;
  }
}

} // namespace detail

std::shared_ptr<DistributedCommGroup> createDistributedCommGroup(
    DistributedInit rendezvousMethod,
    DistributedBackend backendType,
    std::optional<int> worldRank,
    std::optional<int> worldSize,
    const std::unordered_map<std::string, std::string>& params /* = {} */) {
  if (rendezvousMethod == DistributedInit::FILE_SYSTEM &&
      (!worldRank.has_value() || !worldSize.has_value())) {
    throw std::runtime_error(
        "distributedInit: FILE_SYSTEM rendezvous requires passing "
        "worldRank and worldSize arguments.");
  }

  if (rendezvousMethod == DistributedInit::MPI &&
      distributedBackendAvailable(backendType)) {
    throw std::runtime_error(
        "distributedInit: cannot use MPI as rendezvous method; "
        "Flashlight not built with MPI support. Set FL_USE_MPI=ON");
  }

  if (!distributedBackendAvailable(backendType)) {
    throw std::runtime_error(
        "distributedInit: cannot use " + toString(backendType) +
        " as distributed backend; Flashlight not built with " +
        toString(backendType) + " support. Set FL_USE_" +
        toString(backendType) + "=ON");
  }

  return commGroupFromBackendType(backendType);
}

} // namespace fl
