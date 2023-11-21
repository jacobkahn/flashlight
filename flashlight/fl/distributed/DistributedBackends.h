/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "flashlight/fl/distributed/DistributedInterface.h"

namespace fl {

/**
 * Returns of a particular distributed is available based on how Flashlight was
 * compiled.
 *
 * @param[in] backendType the DistributedBackend type to check
 *
 * @return true if Flashlight was built with the specified backend, false
 * otherwise
 */
bool distributedBackendAvailable(DistributedBackend backendType);

/**
 * Returns a string representation of the distributed backend.
 *
 * @param[in] backend the distributed backend
 * @return a string representation of the distributed backend
 */
std::string toString(DistributedBackend backend);

/**
 * Returns a string representation of the distributed initialization method.
 *
 * @param[in] initMethod the distributed initialization method
 * @return a string representation of the distributed initialization method
 */
std::string toString(DistributedInit initMethod);

namespace detail {

/**
 *
 */
class DistributedInfo {
 public:
  static DistributedInfo& getInstance();

  // The default initialization method
  DistributedInit initMethod_;
  // The default distributed backend
  DistributedBackend backend_;

  /**
   * Get the default DistributedCommGroup for a given backend.
   *
   * @param[in] backend the DistributedBackend type
   *
   * @return a reference to the default communication group for the given
   * backend
   */
  const DistributedCommGroup& getInterface(DistributedBackend backend);

  /**
   * Check if a default communication group exists for a given backend.
   *
   * @param[in] backend the DistributedBackend type on which to query
   *
   * @return true if a default exists, false otherwise
   */
  bool defaultCommGroupExists(DistributedBackend backend);

  /**
   * Set the default communicator for a given backend type.
   *
   * @param[in] backend the DistributedBackend type
   * @param[in] commGroup the DistributedCommGroup to make the default
   */
  bool setDefaultCommGroup(
      DistributedBackend backend,
      std::shared_ptr<DistributedCommGroup> commGroup);

 private:
  DistributedInfo() = default;

  // An instance of the distributed comm group interfaces per backend that will
  // be used by default if no group is specified
  std::unordered_map<DistributedBackend, std::shared_ptr<DistributedCommGroup>>
      defaultCommGroups_;
};

} // namespace detail

/**
 * Create a distributed communicator with the given parameters.
 *
 * \note `worldSize`, `worldRank` are ignored if `DistributedInit::MPI` is used
 * as the rendezvousMethod.
 *
 * @param[in] rendezvousMethod Initialization method used for setting up the
 * rendezvous.
 * @param[in] backendType the type of communicator to create, based on backend
 * @param[in] worldSize Total number of processes in the communication group
 * @param[in] worldRank 0-indexed rank of the current process
 * @param[in] params Additional parameters (if any) needed for initialization
 *
 * @return a rank-local instance of a DistributedCommGroup initialized and
 * rendezvous'ed with the given parameters.
 */
std::shared_ptr<DistributedCommGroup> createDistributedCommGroup(
    DistributedInit rendezvousMethod,
    DistributedBackend backendType,
    std::optional<int> worldRank,
    std::optional<int> worldSize,
    const std::unordered_map<std::string, std::string>& params /* = {} */);

} // namespace fl
