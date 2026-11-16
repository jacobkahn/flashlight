/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Types.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace fl {

const std::unordered_map<dtype, std::string> kTypeToString = {
    {dtype::f16, "f16"},
    {dtype::f32, "f32"},
    {dtype::f64, "f64"},
    {dtype::b8, "b8"},
    {dtype::s16, "s16"},
    {dtype::s32, "s32"},
    {dtype::s64, "s64"},
    {dtype::u8, "u8"},
    {dtype::u16, "u16"},
    {dtype::u32, "u32"},
    {dtype::u64, "u64"},
    {dtype::bf16, "bf16"},
};

const std::unordered_map<std::string, dtype> kStringToType = {
    {"f16", dtype::f16},
    {"f32", dtype::f32},
    {"f64", dtype::f64},
    {"b8", dtype::b8},
    {"s16", dtype::s16},
    {"s32", dtype::s32},
    {"s64", dtype::s64},
    {"u8", dtype::u8},
    {"u16", dtype::u16},
    {"u32", dtype::u32},
    {"u64", dtype::u64},
    {"bf16", dtype::bf16},
};

size_t getTypeSize(dtype type) {
  switch (type) {
    case dtype::f16:
      return sizeof(float) / 2;
    case dtype::f32:
      return sizeof(float);
    case dtype::f64:
      return sizeof(double);
    case dtype::b8:
      return sizeof(unsigned char);
    case dtype::s16:
      return sizeof(short);
    case dtype::s64:
      return sizeof(long long);
    case dtype::s32:
      return sizeof(int);
    case dtype::u8:
      return sizeof(unsigned char);
    case dtype::u16:
      return sizeof(unsigned short);
    case dtype::u32:
      return sizeof(unsigned);
    case dtype::u64:
      return sizeof(unsigned long long);
    case dtype::bf16:
      return 2;
    default:
      throw std::invalid_argument("getTypeSize - invalid type queried.");
  }
}

const std::string& dtypeToString(dtype type) {
  return kTypeToString.at(type);
}

fl::dtype stringToDtype(const std::string& string) {
  if (kStringToType.contains(string)) {
    return kStringToType.at(string);
  }
  throw std::invalid_argument("stringToDtype: Invalid input type: " + string);
}

std::ostream& operator<<(std::ostream& ostr, const dtype& s) {
  ostr << dtypeToString(s);
  return ostr;
}

bool isFloatingPoint(dtype type) {
  return type == dtype::f16 || type == dtype::bf16 || type == dtype::f32 ||
      type == dtype::f64;
}

namespace {

// Rank for type promotion: higher rank wins.
// Float types outrank int types; wider outranks narrower.
int dtypeRank(dtype t) {
  switch (t) {
    case dtype::b8:
      return 0;
    case dtype::u8:
      return 1;
    case dtype::s16:
      return 2;
    case dtype::u16:
      return 3;
    case dtype::s32:
      return 4;
    case dtype::u32:
      return 5;
    case dtype::s64:
      return 6;
    case dtype::u64:
      return 7;
    case dtype::f16:
      return 8;
    case dtype::bf16:
      return 8;
    case dtype::f32:
      return 9;
    case dtype::f64:
      return 10;
  }
  throw std::invalid_argument("[dtypeRank] Unknown dtype");
}

} // namespace

dtype promoteDtypes(dtype a, dtype b) {
  if (a == b) {
    return a;
  }
  // Mixed f16/bf16 promotes to f32 (different 16-bit float encodings)
  if ((a == dtype::f16 && b == dtype::bf16) ||
      (a == dtype::bf16 && b == dtype::f16)) {
    return dtype::f32;
  }
  // Float always wins over int
  bool aFloat = isFloatingPoint(a);
  bool bFloat = isFloatingPoint(b);
  if (aFloat && !bFloat) {
    return a;
  }
  if (bFloat && !aFloat) {
    return b;
  }
  // Both float or both int: wider wins
  return dtypeRank(a) >= dtypeRank(b) ? a : b;
}

dtype scalarDtype(double value) {
  // If the value is integral and fits in s32, treat as integer
  auto intVal = static_cast<int>(value);
  if (value == static_cast<double>(intVal) &&
      value >= std::numeric_limits<int>::min() &&
      value <= std::numeric_limits<int>::max()) {
    return dtype::s32;
  }
  return dtype::f32;
}

dtype promoteScalarDtype(dtype tensorType, double scalarValue) {
  auto sd = scalarDtype(scalarValue);
  // If the tensor is already floating-point, a scalar never widens it
  // (e.g., f16_tensor / 2.828 stays f16).
  if (isFloatingPoint(tensorType)) {
    return tensorType;
  }
  // Integer tensor: promote normally (e.g., s32 / 0.5 --> f32)
  return promoteDtypes(tensorType, sd);
}

} // namespace fl
