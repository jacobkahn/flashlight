/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/distributed/DistributedInterface.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

/**
 * \defgroup distributed_api Distributed API
 * @{
 */

// TODO: remove this namespace
namespace DistributedConstants {

constexpr const char* kMaxDevicePerNode = "MAX_DEVICE_PER_NODE";
constexpr const char* kFilePath = "FILE_PATH";
constexpr const std::size_t kCoalesceCacheSize = ((size_t)(20) << 20); // 20 MB

} // namespace DistributedConstants

/**
 * Initialize the distributed environment, setting the communicator created with
 * the given settings as the default for the given DistributedBackend type.
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
 */
FL_API void distributedInit(
    DistributedInit rendezvousMethod,
    DistributedBackend backendType,
    std::optional<int> worldRank,
    std::optional<int> worldSize,
    const std::unordered_map<std::string, std::string>& params = {});

/**
 * Returns whether the distributed environment has been initialized
 */
FL_API bool isDistributedInit();

/**
 * Returns the default DistributedCommGroup for a given a tensor.
 *
 * @param[in] tensor a tensor for which to get the default distributed comm
 * group
 * @return a reference tot he distributed comm group corresponding to this
 * tensor.
 * TODO: this is not a very good return type, make this better.
 */
FL_API const DistributedCommGroup& getDefaultDistributedCommGroupForTensor(
    const Tensor& tensor);

/**
 * Returns the default DistributedCommGroup for a given a tensor.
 *
 * @param[in] type a tensor backend type for which to get a distributed comm
 * group
 * @return a reference to the distributed comm group corresponding to this
 * backend.
 * TODO: this is not a very good return type, make this better.
 */
FL_API const DistributedCommGroup&
getDefaultDistributedCommGroupForTensorBackendType(TensorBackendType type);

/**
 * Returns the backend used for distributed setup
 */
FL_API DistributedBackend distributedBackend();

/**
 * Returns rank of the current process in the communication group
 * (zero-based). Returns 0 if distributed environment is not initialized
 */
FL_API int getWorldRank();

/**
 * Returns total process in the communication group
 * Returns 1 if distributed environment is not initialized
 */
FL_API int getWorldSize();

/**
 * Synchronizes a the array wrapped by the Variable with allreduce.
 *
 * @param[in] var a variable whose array will be synchronized
 * @param[in] scale scale the Variable after allreduce by this factor
 * @param[in] async perform the allReduce operation asynchronously in a
 * separate compute stream to the Flashlight compute stream. NB: if true,
 * ``syncDistributed`` *must* be called in order to ensure the Flashlight CUDA
 * stream waits until ``allReduce`` is complete and uses updated values.
 */
FL_API void allReduce(Variable& var, double scale = 1.0, bool async = false);

FL_API void allReduce(
    DistributedCommGroup& com,
    Variable& var,
    double scale /* = 1.0 */,
    bool async /* = false */);

/**
 * Synchronizes a single Flashlight array with allreduce.
 *
 * @param arr an array which will be synchronized
 * @param[in] async perform the allReduce operation asynchronously in a
 * separate compute stream to the Flashlight compute stream. NB: if used,
 * ``syncDistributed`` *must* be called in order to ensure asynchrnous
 * reduction and worker streams wait until ``allReduce`` is complete and uses
 * updated values.
 */
FL_API void allReduce(Tensor& arr, bool async = false);

/**
 * Synchronizes a the arrays wrapped by a vector of Variables with allreduce.
 *
 * @param[in] vars `Variable`s whose arrays will be synchronized
 * @param[in] scale scale the Variable after allreduce by this factor
 * @param[in] async perform the allReduce operation asynchronously in a
 * separate compute stream to the Flashlight compute stream. NB: if used,
 * ``syncDistributed`` *must* be called in order to ensure asynchrnous
 * reduction and worker streams wait until ``allReduce`` is complete and uses
 * updated values.
 * @param[in] contiguous copy data for each Variable into a contiguous buffer
 * before performing the allReduce operation
 */
FL_API void allReduceMultiple(
    std::vector<Variable> vars,
    double scale = 1.0,
    bool async = false,
    bool contiguous = false);

FL_API void allReduceMultiple(
    DistributedCommGroup& comm,
    std::vector<Variable> vars,
    double scale /* = 1.0 */,
    bool async /* = false */,
    bool contiguous /* = false */);

/**
 * Synchronizes a vector of pointers to arrays with allreduce.
 *
 * @param[in] arrs a vector of pointers to arrays which will be synchronized
 * @param[in] async perform the allReduce operation asynchronously in a
 * separate compute stream to the Flashlight compute stream. NB: if used,
 * ``syncDistributed`` *must* be called in order to ensure asynchrnous
 * reduction and worker streams wait until ``allReduce`` is complete and uses
 * updated values.
 * @param[in] contiguous copy data for each Variable into a contiguous buffer
 * before performing the allReduce operation
 */
FL_API void allReduceMultiple(
    std::vector<Tensor*> arrs,
    bool async = false,
    bool contiguous = false);

/**
 * Synchronizes operations in the Flashlight compute stream with operations in
 * the distributed compute stream, if applicable. That is, all operations in
 * the Flashlight compute stream will not be executed until operations
 * currently enqueued on the distributed compute stream are finished
 * executing.
 *
 * Note that if asynchronous allReduce is not used, this operation will be a
 * no-op, since no operations will be enqueued on the distributed compute
 * stream.
 */
FL_API void syncDistributed();

/**
 * Blocks until all CPU processes have reached this routine.
 */
FL_API void barrier();

/** @} */

} // namespace fl
