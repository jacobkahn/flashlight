/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/distributed/DistributedApi.h"

#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/distributed/DistributedBackends.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

void distributedInit(
    DistributedInit rendezvousMethod,
    DistributedBackend backend,
    std::optional<int> worldRank,
    std::optional<int> worldSize,
    const std::unordered_map<std::string, std::string>& params /* = {} */) {
  auto commGroup = createDistributedCommGroup(
      rendezvousMethod, backend, worldRank, worldSize, params);

  auto& distributedInfo = detail::DistributedInfo::getInstance();

  if (distributedInfo.defaultCommGroupExists(backend)) {
    throw std::runtime_error(
        "setDefaultCommGroup: a default communicator already exists for the " +
        toString(backend) + "backend type.");
  }

  // Set this communicator as the default for this DistributedBackend type
  detail::DistributedInfo::getInstance().setDefaultCommGroup(
      backend, commGroup);
}

bool isDistributedInit(DistributedBackend backend) {
  return detail::DistributedInfo::getInstance().defaultCommGroupExists(backend);
}

const DistributedCommGroup& getDefaultDistributedCommGroupForTensor(
    const Tensor& tensor) {
  const auto& distributedInfo = detail::DistributedInfo::getInstance();
  switch (tensor.backendType()) {
    case TensorBackendType::Jit:
      throw std::invalid_argument(
          "getDefaultDistributedCommGroupForTensor: "
          "JIT tensors not yet supported");
    // static_cast<JitBackend>
    // // get the wrapped backend's backend type, recurse
    // return getDistributedCommGroupForTensorBackendType(
    //     tensor.backend().backendType());
    default:
      return getDefaultDistributedCommGroupForTensorBackendType(
          tensor.backendType());
  }
}

const DistributedCommGroup& getDefaultDistributedCommGroupForTensorBackendType(
    TensorBackendType type) {
  const auto& distributedInfo = detail::DistributedInfo::getInstance();
  switch (type) {
    case TensorBackendType::Stub:
      return distributedInfo.getInterface(DistributedBackend::STUB);
    case TensorBackendType::ArrayFire:
      return distributedInfo.getInterface(DistributedBackend::NCCL);
    case TensorBackendType::OneDnn:
      return distributedInfo.getInterface(DistributedBackend::GLOO);
    case TensorBackendType::Jit:
      throw std::invalid_argument(
          "getDefaultDistributedCommGroupForTensorBackendType: "
          "can't convert Jit backend type to distributed backend type - use "
          "getDefaultDistributedCommGroupForTensor with a tensor from the "
          "respective JIT backend.");
    default:
      throw std::invalid_argument(
          "getDefaultDistributedCommGroupForTensorBackendType: "
          "impossible state - no known distributed interface for given tensor "
          "type. Developers: please add a case and/or informative exception");
  }
}

DistributedBackend distributedBackend() {
  return detail::DistributedInfo::getInstance().backend_;
}

void allReduce(
    Variable& var,
    double scale /* = 1.0 */,
    bool async /* = false */) {
  const auto& group = getDefaultDistributedCommGroupForTensor(var.tensor());
  if (getWorldSize() > 1) {
    group.allReduce(var.tensor(), async);
  }
  var.tensor() *= scale;
}

void allReduce(
    DistributedCommGroup& group,
    Variable& var,
    double scale /* = 1.0 */,
    bool async /* = false */) {
  allReduce(group, var.tensor(), scale, async);
  if (getWorldSize() > 1) {
    group.allReduce(var.tensor(), async);
  }
  var.tensor() *= scale;
}

void allReduce(
    Tensor& tensor,
    double scale /* = 1.0 */,
    bool async /* = false */) {
  const auto& group = getDefaultDistributedCommGroupForTensor(tensor);
  if (getWorldSize() > 1) {
    group.allReduce(tensor, async);
  }
  tensor *= scale;
}

void allReduce(
    DistributedCommGroup& group,
    Tensor& tensor,
    double scale /* = 1.0 */,
    bool async /* = false */) {
  if (getWorldSize() > 1) {
    group.allReduce(tensor, async);
  }
  tensor *= scale;
}

void allReduceMultiple(
    DistributedCommGroup& group,
    std::vector<Variable> vars,
    double scale /* = 1.0 */,
    bool async /* = false */,
    bool contiguous /* = false */) {
  if (vars.empty()) {
    return;
  }
  const auto& group =
      getDefaultDistributedCommGroupForTensor(vars.front().tensor());
  alLReduceMultiple(group, vars, scale, async, contiguous);
}

void allReduceMultiple(
    std::vector<Variable> vars,
    double scale /* = 1.0 */,
    bool async /* = false */,
    bool contiguous /* = false */) {
  if (vars.empty()) {
    return;
  }

  const auto& group = getDefaultDistributedCommGroupForTensor(var.tensor());
  // use a vector of pointers to thoroughly avoid copying
  std::vector<Tensor*> arrs;
  for (auto& var : vars) {
    arrs.push_back(&var.tensor());
  }
  if (getWorldSize() > 1) {
    group.allReduceMultiple(arrs, async, contiguous);
  }
  for (auto& var : vars) {
    var.tensor() *= scale;
  }
}

void barrier() {
  auto tensor = Tensor::fromVector<int>({0});
  // blocks for the default tensor backend
  const auto& group = getDefaultDistributedCommGroupForTensor(tensor);
  group.allReduce(tensor, false);

  // TODO: re-implement this based on primtiives if they exist
  // This hack is to make sure `tensor` will not be optimized away by a
  // JIT during allreduce().
  fl::sum(tensor).asScalar<float>();
}

} // namespace fl
