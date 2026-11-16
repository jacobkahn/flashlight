/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <vector>

namespace fl {

class Tensor;

/**
 * Autograd metadata attached to a Tensor.
 *
 * When a Tensor participates in differentiable computation, it carries
 * an AutogradMeta that records the computation graph for backpropagation.
 * This struct is stored via shared_ptr so that Tensor copies share the same
 * autograd identity (gradient accumulation point).
 *
 * Tensors without autograd have a null autogradMeta_ pointer (zero overhead).
 */
struct AutogradMeta {
  using GradFunc = std::function<
      void(std::vector<Tensor>& inputs, const Tensor& grad_output)>;
  using GradHook = std::function<void(Tensor& grad)>;

  /// Whether the gradient should be computed for this Tensor
  bool requiresGrad{false};
  /// Inputs of the operation that produced this Tensor
  std::vector<Tensor> inputs;
  /// Gradient with respect to this Tensor
  std::unique_ptr<Tensor> grad;
  /// Function for computing the gradient of the input Tensors
  GradFunc gradFunc;
  /// Function applied to gradient after it's computed during backward pass
  GradHook onGradAvailable;

  /// Pool of detached views for .detach().
  /// Each call appends a view that shares the owning Tensor's data but has no
  /// autograd metadata.  Using a vector avoids the dangling-reference problem
  /// when .detach() is called multiple times in the same expression (e.g.
  /// `x.detach() * x.detach()`).
  mutable std::vector<std::shared_ptr<Tensor>> detachedViews;
};

} // namespace fl
