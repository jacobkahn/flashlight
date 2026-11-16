/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Autograd method implementations for fl::Tensor.
// These are member functions of Tensor but live here to keep tensor code
// (fl/tensor/) cleanly separated from autograd concerns (fl/autograd/).

#include "flashlight/fl/tensor/Tensor.h"

#include <algorithm>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "flashlight/fl/autograd/AutogradMeta.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/TensorAdapter.h"

namespace fl {

Tensor::Tensor(Tensor data, bool requiresGrad) {
  impl_ = std::move(data.impl_);
  if (requiresGrad) {
    autogradMeta_ = std::make_shared<AutogradMeta>();
    autogradMeta_->requiresGrad = true;
  }
}

Tensor::Tensor(
    Tensor data,
    std::vector<Tensor> inputs,
    AutogradMeta::GradFunc gradFunc) {
  impl_ = std::move(data.impl_);
  if (std::any_of(inputs.begin(), inputs.end(), [](const Tensor& input) {
        return input.requiresGrad();
      })) {
    autogradMeta_ = std::make_shared<AutogradMeta>();
    autogradMeta_->requiresGrad = true;
    autogradMeta_->inputs = std::move(inputs);
    autogradMeta_->gradFunc = std::move(gradFunc);
  }
}

bool Tensor::requiresGrad() const {
  return autogradMeta_ && autogradMeta_->requiresGrad;
}

bool Tensor::isGradAvailable() const {
  if (!requiresGrad()) {
    return false;
  }
  return autogradMeta_->grad != nullptr;
}

Tensor& Tensor::grad() const {
  if (!requiresGrad()) {
    throw std::logic_error("gradient computation disabled for this Tensor");
  }
  if (!autogradMeta_->grad) {
    throw std::logic_error("gradient not computed yet for this Tensor");
  }
  return *autogradMeta_->grad;
}

void Tensor::addGrad(const Tensor& childGrad) {
  if (requiresGrad()) {
    if (childGrad.type() != this->type()) {
      std::ostringstream ss;
      ss << "Tensor::addGrad: gradient type " << childGrad.type()
         << " doesn't match tensor type " << this->type();
      throw std::invalid_argument(ss.str());
    }
    if (childGrad.shape() != this->shape()) {
      std::ostringstream ss;
      ss << "Tensor::addGrad: gradient shape " << childGrad.shape()
         << " doesn't match tensor shape " << this->shape();
      throw std::invalid_argument(ss.str());
    }
    if (autogradMeta_->grad) {
      autogradMeta_->grad =
          std::make_unique<Tensor>(*autogradMeta_->grad + childGrad);
    } else {
      autogradMeta_->grad = std::make_unique<Tensor>(childGrad);
    }
  }
}

void Tensor::setRequiresGrad(bool requiresGrad) {
  if (requiresGrad) {
    if (!autogradMeta_) {
      autogradMeta_ = std::make_shared<AutogradMeta>();
    }
    autogradMeta_->requiresGrad = true;
  } else if (autogradMeta_) {
    autogradMeta_->requiresGrad = false;
    autogradMeta_->gradFunc = nullptr;
    autogradMeta_->inputs.clear();
    autogradMeta_->grad.reset();
  }
}

void Tensor::zeroGrad() {
  if (autogradMeta_) {
    autogradMeta_->grad.reset();
    autogradMeta_->detachedViews.clear();
  }
}

void Tensor::registerGradHook(const AutogradMeta::GradHook& hook) {
  if (!autogradMeta_) {
    autogradMeta_ = std::make_shared<AutogradMeta>();
  }
  autogradMeta_->onGradAvailable = hook;
}

void Tensor::clearGradHook() {
  if (autogradMeta_) {
    autogradMeta_->onGradAvailable = nullptr;
  }
}

void Tensor::backward(const Tensor& grad, bool retainGraph) {
  addGrad(grad);
  auto dag = buildAutogradDAG();
  for (auto iter = dag.rbegin(); iter != dag.rend(); iter++) {
    iter->calcGradInputs(retainGraph);
    iter->applyGradHook();
    if (!retainGraph) {
      *iter = Tensor();
    }
  }
}

void Tensor::backward(bool retainGraph) {
  auto ones = fl::full(shape(), 1, this->type());
  backward(ones, retainGraph);
}

Tensor Tensor::withoutData() const {
  // A lazy constant with matching shape and type on this tensor's backend. It
  // records the metadata gradient functions need without allocating: on the
  // JIT backend it is a scalar node that is only materialized if read
  // (a default-adapter tensor would allocate an uninitialized buffer of the
  // full size for every recorded input).
  Tensor other = backend().full(shape(), 0.0, type());
  other.autogradMeta_ = autogradMeta_;
  return other;
}

AutogradMeta* Tensor::autogradMeta() const {
  return autogradMeta_.get();
}

std::vector<Tensor> Tensor::buildAutogradDAG() const {
  std::unordered_set<AutogradMeta*> cache;
  std::vector<Tensor> dag;
  std::function<void(const Tensor&)> recurse;

  recurse = [&](const Tensor& t) {
    if (!t.autogradMeta_) {
      return;
    }
    auto id = t.autogradMeta_.get();
    if (cache.contains(id)) {
      return;
    }
    for (const auto& input : t.autogradMeta_->inputs) {
      recurse(input);
    }
    cache.insert(id);
    dag.push_back(t);
  };

  recurse(*this);
  return dag;
}

void Tensor::calcGradInputs(bool retainGraph) {
  if (autogradMeta_ && autogradMeta_->gradFunc) {
    if (!autogradMeta_->grad) {
      throw std::logic_error("gradient was not propagated to this Tensor");
    }
    autogradMeta_->gradFunc(autogradMeta_->inputs, *autogradMeta_->grad);
  }
  if (!retainGraph && autogradMeta_) {
    autogradMeta_->inputs.clear();
  }
}

void Tensor::applyGradHook() {
  if (autogradMeta_ && autogradMeta_->onGradAvailable) {
    if (autogradMeta_->grad) {
      autogradMeta_->onGradAvailable(*autogradMeta_->grad);
    }
  }
}

Tensor Tensor::asContiguous() const {
  if (isEmpty() || isContiguous()) {
    return *this;
  }
  auto contiguous = asContiguousTensor();
  // Preserve autograd metadata by sharing the same autogradMeta_.
  Tensor result;
  result.impl_ = std::move(contiguous.impl_);
  result.autogradMeta_ = autogradMeta_;
  return result;
}

Tensor& Tensor::detach() const {
  if (!autogradMeta_) {
    return const_cast<Tensor&>(*this);
  }
  // Return a detached view that shares underlying data but has no autograd
  // metadata. The returned Tensor provides raw access to the underlying data
  // without triggering autograd paths in operator(), flat(), etc.
  //
  // shallowCopy() shares the backend's data handle (e.g. ArrayFire's
  // shared_ptr<af::array>) so writes through the detached view (including
  // through lazy views like flat(i)) propagate to this tensor.
  //
  // We append to a vector (rather than overwriting a single shared_ptr) so
  // that multiple calls in the same expression (e.g. x.detach() * x.detach())
  // don't invalidate earlier references.
  autogradMeta_->detachedViews.push_back(
      std::make_shared<Tensor>(shallowCopy()));
  return *autogradMeta_->detachedViews.back();
}

} // namespace fl
