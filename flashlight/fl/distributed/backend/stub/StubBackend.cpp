/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/distributed/backend/stub/StubBackend.h"

#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace fl {

void distributedInit(
    DistributedInit /* initMethod */,
    int worldRank,
    int worldSize,
    const std::unordered_map<std::string, std::string>& /* params = {} */) {
  if (worldSize > 1 || worldRank > 0) {
    throw std::runtime_error("worldSize must be 1 with distributed stub");
  }
  // TODO(jacobkahn) -- implement this if anything, might just be an empty
  // constructor call to set the singleton
  // detail::DistributedInfo::getInstance().backend_ = DistributedBackend::STUB;
  // detail::DistributedInfo::getInstance().isInitialized_ = true;
}

// TODO: have this and the other funcs just return the tensor?
void StubBackend::allReduce(Tensor& arr, bool async /* = false */) const {
  throw std::runtime_error("allReduce not supported for stub backend");
}

// Not yet supported
void StubBackend::allReduceMultiple(
    std::vector<Tensor*> arrs,
    bool async /* = false */,
    bool contiguous /* = false */) const {
  throw std::runtime_error(
      "allReduceMultiple not supported for distributed stub backend");
}

bool StubBackend::isInitialized() const {
  return true;
}

DistributedInitInfo StubBackend::rendezvous() {
  // Supports a trivial single-proc rendezvous
  DistributedInitInfo info;
  info.worldSize = 1;
  info.worldRank = 0;
  return info;
}

void StubBackend::allBroadcast(void* data, size_t bytes, int rootRank) const {
  throw std::runtime_error(
      "allBroadcast not supported for distributed stub backend");
}

void StubBackend::allBarrier() const {
  // TODO: make this function a no-op passthrough?
  throw std::runtime_error(
      "allBarrier not supported for distributed stub backend");
}

void syncDistributed() {
  throw std::runtime_error(
      "Asynchronous allReduce not supported for distributed stub backend");
}

int getWorldRank() {
  return 0;
}

int getWorldSize() {
  return 1;
}
} // namespace fl
