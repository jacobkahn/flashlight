/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/TensorExtension.h"

#include <format>
#include <stdexcept>

namespace fl::detail {

bool TensorExtensionRegistrar::registerTensorExtension(
    TensorBackendType backend,
    TensorExtensionType extensionType,
    TensorExtensionCallback&& creationFunc) {
  auto& _extensions = (*extensions_
                            .try_emplace(
                                backend,
                                std::unordered_map<
                                    TensorExtensionType,
                                    TensorExtensionCallback>())
                            .first)
                          .second;

  // Add extension to registry
  _extensions.try_emplace(extensionType, std::move(creationFunc));
  return true;
}

bool TensorExtensionRegistrar::isTensorExtensionRegistered(
      TensorBackendType backend,
      TensorExtensionType extensionType) {
  return extensions_.contains(backend) &&
    extensions_[backend].contains(extensionType);
}

TensorExtensionCallback&
TensorExtensionRegistrar::getTensorExtensionCreationFunc(
    TensorBackendType backend,
    TensorExtensionType extensionType) {
  if (!extensions_.contains(backend)) {
    throw std::invalid_argument(
        "TensorExtensionRegistrar::getTensorExtensionCreationFunc: "
        "no tensor extensions registered for given backend.");
  }
  auto& _extensions = extensions_[backend];
  if (!_extensions.contains(extensionType)) {
    throw std::invalid_argument(std::format(
        "TensorExtensionRegistrar::getTensorExtensionCreationFunc: "
        "no tensor extensions registered for backend {}",
        tensorBackendTypeToString(backend)));
  }
  return _extensions[extensionType];
}

TensorExtensionRegistrar& TensorExtensionRegistrar::getInstance() {
  static TensorExtensionRegistrar instance;
  return instance;
}

} // namespace fl
