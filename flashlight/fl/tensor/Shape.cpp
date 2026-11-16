/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Shape.h"

#include <algorithm>
#include <format>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace fl {

Shape::Shape(std::vector<Dim> d) : dims_(std::move(d)) {}
Shape::Shape(std::initializer_list<Dim> d) : Shape(std::vector<Dim>(d)) {}

const Dim kEmptyShapeNumberOfElements = 1;

void Shape::checkDimsOrThrow(const size_t dim) const {
  if (dim > ndim() - 1) {
    throw std::invalid_argument(std::format(
        "Shape index {} out of bounds for shape with {} dimensions.",
        dim, dims_.size()));
  }
}

Dim Shape::elements() const {
  if (dims_.empty()) {
    return kEmptyShapeNumberOfElements;
  }
  return std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<Dim>());
}

int Shape::ndim() const {
  return dims_.size();
}

Dim Shape::dim(const size_t dim) const {
  checkDimsOrThrow(dim);
  return dims_[dim];
}

Dim& Shape::operator[](const size_t dim) {
  checkDimsOrThrow(dim);
  return dims_[dim];
}

const Dim& Shape::operator[](const size_t dim) const {
  checkDimsOrThrow(dim);
  return dims_[dim];
}

bool Shape::operator==(const Shape& other) const {
  return dims_ == other.dims_;
}

bool Shape::operator!=(const Shape& other) const {
  return !(this->operator==(other));
}

bool Shape::operator==(const std::initializer_list<Dim>& other) const {
  return dims_.size() == other.size() &&
      std::equal(std::begin(dims_), std::end(dims_), std::begin(other));
}

bool Shape::operator!=(const std::initializer_list<Dim>& other) const {
  return !(this->operator==(other));
}

const std::vector<Dim>& Shape::get() const {
  return dims_;
}

std::vector<Dim>& Shape::get() {
  return dims_;
};

std::string Shape::toString() const {
  std::string result = "(";
  for (size_t i = 0; i < ndim(); ++i) {
    if (i > 0) result += ", ";
    result += std::to_string(dim(i));
  }
  result += ")";
  return result;
}

std::ostream& operator<<(std::ostream& ostr, const Shape& s) {
  ostr << s.toString();
  return ostr;
}

Shape broadcastShapes(const Shape& lhs, const Shape& rhs) {
  const auto lhsRank = lhs.ndim();
  const auto rhsRank = rhs.ndim();
  const auto maxRank = std::max(lhsRank, rhsRank);
  std::vector<Dim> out(maxRank);
  // Right-align dimensions (numpy/PyTorch broadcasting semantics):
  // shorter shape is treated as having leading 1s.
  for (int i = maxRank - 1; i >= 0; --i) {
    const int li = i - (maxRank - lhsRank);
    const int ri = i - (maxRank - rhsRank);
    const Dim l = li >= 0 ? lhs.dim(li) : 1;
    const Dim r = ri >= 0 ? rhs.dim(ri) : 1;
    if (l != r && l != 1 && r != 1) {
      throw std::invalid_argument(std::format(
          "broadcastShapes: cannot broadcast shapes {} and {}"
          " - mismatch at dimension {} ({} vs {}).",
          lhs.toString(), rhs.toString(), i, l, r));
    }
    out[i] = std::max(l, r);
  }
  return Shape(out);
}

Shape contiguousStrides(const Shape& shape) {
  if (shape.ndim() == 0) {
    return Shape({1}); // scalar
  }
  std::vector<Dim> strides(shape.ndim());
  strides[shape.ndim() - 1] = 1;
  for (int i = shape.ndim() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape.dim(i + 1);
  }
  return Shape(std::move(strides));
}

} // namespace fl
