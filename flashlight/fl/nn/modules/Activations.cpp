/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Activations.h"

#include <format>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace fl {

Sigmoid::Sigmoid() = default;

Tensor Sigmoid::forward(const Tensor& input) {
  return autograd::sigmoid(input);
}

std::unique_ptr<Module> Sigmoid::clone() const {
  return std::make_unique<Sigmoid>(*this);
}

std::string Sigmoid::prettyString() const {
  return "Sigmoid";
}

Log::Log() = default;

Tensor Log::forward(const Tensor& input) {
  return autograd::log(input);
}

std::unique_ptr<Module> Log::clone() const {
  return std::make_unique<Log>(*this);
}

std::string Log::prettyString() const {
  return "Log";
}

Tanh::Tanh() = default;

Tensor Tanh::forward(const Tensor& input) {
  return autograd::tanh(input);
}

std::unique_ptr<Module> Tanh::clone() const {
  return std::make_unique<Tanh>(*this);
}

std::string Tanh::prettyString() const {
  return "Tanh";
}

HardTanh::HardTanh() = default;

Tensor HardTanh::forward(const Tensor& input) {
  return autograd::clip(input, -1.0, 1.0);
}

std::unique_ptr<Module> HardTanh::clone() const {
  return std::make_unique<HardTanh>(*this);
}

std::string HardTanh::prettyString() const {
  return "HardTanh";
}

ReLU::ReLU() = default;

Tensor ReLU::forward(const Tensor& input) {
  return autograd::maximum(input, 0.0);
}

std::unique_ptr<Module> ReLU::clone() const {
  return std::make_unique<ReLU>(*this);
}

std::string ReLU::prettyString() const {
  return "ReLU";
}

ReLU6::ReLU6() = default;

Tensor ReLU6::forward(const Tensor& input) {
  return autograd::clip(input, 0.0, 6.0);
}

std::unique_ptr<Module> ReLU6::clone() const {
  return std::make_unique<ReLU6>(*this);
}

std::string ReLU6::prettyString() const {
  return "ReLU6";
}

LeakyReLU::LeakyReLU(double slope) : mSlope_(slope) {}

Tensor LeakyReLU::forward(const Tensor& input) {
  return autograd::maximum(input, mSlope_ * input);
}

std::unique_ptr<Module> LeakyReLU::clone() const {
  return std::make_unique<LeakyReLU>(*this);
}

std::string LeakyReLU::prettyString() const {
  return std::format("LeakyReLU ({})", mSlope_);
}

PReLU::PReLU(const Tensor& w) : UnaryModule({w}) {}

PReLU::PReLU(int size, double value) {
  auto w = constant(value, size, 1);
  params_ = {w};
}

Tensor PReLU::forward(const Tensor& input) {
  auto mask = input >= 0.0;
  return (input * mask) + (input * !mask * autograd::tileAs(params_[0], input));
}

std::unique_ptr<Module> PReLU::clone() const {
  return std::make_unique<PReLU>(*this);
}

std::string PReLU::prettyString() const {
  return "PReLU";
}

ELU::ELU(double alpha) : mAlpha_(alpha) {}

Tensor ELU::forward(const Tensor& input) {
  auto mask = input >= 0.0;
  return (mask * input) + (!mask * mAlpha_ * (autograd::exp(input) - 1));
}

std::unique_ptr<Module> ELU::clone() const {
  return std::make_unique<ELU>(*this);
}

std::string ELU::prettyString() const {
  return std::format("ELU ({})", mAlpha_);
}

ThresholdReLU::ThresholdReLU(double threshold) : mThreshold_(threshold) {}

Tensor ThresholdReLU::forward(const Tensor& input) {
  auto mask = input >= mThreshold_;
  return input * mask;
}

std::unique_ptr<Module> ThresholdReLU::clone() const {
  return std::make_unique<ThresholdReLU>(*this);
}

std::string ThresholdReLU::prettyString() const {
  return std::format("ThresholdReLU ({})", mThreshold_);
}

GatedLinearUnit::GatedLinearUnit(int dim) : dim_(dim) {}

Tensor GatedLinearUnit::forward(const Tensor& input) {
  return autograd::gatedlinearunit(input, dim_);
}

std::unique_ptr<Module> GatedLinearUnit::clone() const {
  return std::make_unique<GatedLinearUnit>(*this);
}

std::string GatedLinearUnit::prettyString() const {
  return std::format("GatedLinearUnit ({})", dim_);
}

LogSoftmax::LogSoftmax(int dim /* = 0 */) : dim_(dim) {}

Tensor LogSoftmax::forward(const Tensor& input) {
  return autograd::logSoftmax(input, dim_);
}

std::unique_ptr<Module> LogSoftmax::clone() const {
  return std::make_unique<LogSoftmax>(*this);
}

std::string LogSoftmax::prettyString() const {
  return std::format("LogSoftmax ({})", dim_);
}

Swish::Swish(double beta /* = 1.0 */) : beta_(beta) {}

Tensor Swish::forward(const Tensor& input) {
  return autograd::swish(input, beta_);
}

std::unique_ptr<Module> Swish::clone() const {
  return std::make_unique<Swish>(*this);
}

std::string Swish::prettyString() const {
  return std::format("Swish ({})", beta_);
}

} // namespace fl
