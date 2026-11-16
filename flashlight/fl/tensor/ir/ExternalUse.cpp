/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ir/ExternalUse.h"

namespace fl {

void ExternalUse::unlinkWithCurrentUsee() {
  usee_->externalUses_.erase(useIter_);
}

void ExternalUse::linkWithUsee(NodePtr usee) {
  usee_ = usee;
  useIter_ = usee->externalUses_.insert(usee->externalUses_.end(), this);
}

ExternalUse::ExternalUse(NodePtr usee) {
  linkWithUsee(usee);
}

ExternalUse::~ExternalUse() {
  if (usee_) {
    unlinkWithCurrentUsee();
  }
}

ExternalUse::ExternalUse(ExternalUse&& other) noexcept
    : usee_(std::move(other.usee_)), useIter_(other.useIter_) {
  // Update the pointer stored in the node's external use list to point to us.
  if (usee_) {
    *useIter_ = this;
  }
  other.usee_ = nullptr;
}

ExternalUse& ExternalUse::operator=(ExternalUse&& other) noexcept {
  if (this != &other) {
    if (usee_) {
      unlinkWithCurrentUsee();
    }
    usee_ = std::move(other.usee_);
    useIter_ = other.useIter_;
    if (usee_) {
      *useIter_ = this;
    }
    other.usee_ = nullptr;
  }
  return *this;
}

NodePtr ExternalUse::usee() const {
  return usee_;
}

void ExternalUse::setUsee(NodePtr newUsee) {
  unlinkWithCurrentUsee();
  linkWithUsee(newUsee);
}

} // namespace fl
