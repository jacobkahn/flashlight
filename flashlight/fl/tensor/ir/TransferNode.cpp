/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/TransferNode.h"

#include "flashlight/fl/runtime/Stream.h"
#include "flashlight/fl/runtime/Device.h"

namespace fl {

TransferNode::TransferNode(
    NodePtr input,
    DeviceType dstDeviceType,
    TensorBackendType dstBackendType,
    std::shared_ptr<const Stream> copyStream,
    PrivateHelper)
    : NodeTrait({input}, input->shape(), input->dataType(),
                std::move(copyStream)),
      dstDeviceType_(dstDeviceType),
      dstBackendType_(dstBackendType) {}

NodePtr TransferNode::create(
    NodePtr input,
    DeviceType dstDeviceType,
    TensorBackendType dstBackendType,
    std::shared_ptr<const Stream> copyStream) {
  // No-op: if src device type matches dst, return input directly
  if (input->stream() &&
      input->stream()->device().type() == dstDeviceType) {
    return input;
  }
  return std::make_shared<TransferNode>(
      input, dstDeviceType, dstBackendType,
      std::move(copyStream), PrivateHelper{});
}

NodePtr TransferNode::input() const {
  return getInput(kInputIdx);
}

DeviceType TransferNode::dstDeviceType() const {
  return dstDeviceType_;
}

TensorBackendType TransferNode::dstBackendType() const {
  return dstBackendType_;
}

} // namespace fl
