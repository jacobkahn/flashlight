/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/fl/runtime/DeviceType.h"
#include "flashlight/fl/tensor/Tensor.h"
#include "flashlight/fl/tensor/ir/Node.h"

namespace fl {

class TransferNode;
using TransferNodePtr = std::shared_ptr<TransferNode>;

/// A node representing data transfer from one device to another.
/// Shape and dtype are preserved. The node's stream is the stream on which
/// the copy work is enqueued (for sync edge correctness).
class TransferNode : public NodeTrait<TransferNode> {
  DeviceType dstDeviceType_;
  TensorBackendType dstBackendType_;

  static constexpr unsigned kInputIdx = 0;
  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::Transfer;

  TransferNode(
      NodePtr input,
      DeviceType dstDeviceType,
      TensorBackendType dstBackendType,
      std::shared_ptr<const Stream> copyStream,
      PrivateHelper);

  /// Create a TransferNode. Returns the input directly if src and dst
  /// are on the same device type (no-op elimination).
  static NodePtr create(
      NodePtr input,
      DeviceType dstDeviceType,
      TensorBackendType dstBackendType,
      std::shared_ptr<const Stream> copyStream);

  NodePtr input() const;
  DeviceType dstDeviceType() const;
  TensorBackendType dstBackendType() const;
};

} // namespace fl
