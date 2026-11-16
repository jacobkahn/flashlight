/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/cpu/CPUTensor.h"

#include <atomic>
#include <cassert>
#include <cstring>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/backend/cpu/CPUBackend.h"
#include "flashlight/fl/tensor/ops/onednn/OneDnnOps.h"

#define FL_CPU_TENSOR_UNIMPLEMENTED \
  throw std::invalid_argument(      \
      "CPUTensor::" + std::string(__func__) + " - unimplemented.");

namespace fl {

namespace {

constexpr float kfloatEqualTolerance = 1e-5;

// Row-major linear index from multi-dimensional coordinates.
inline size_t linearIndex(
    const std::vector<Dim>& coords,
    const std::vector<Dim>& dims) {
  size_t idx = 0;
  size_t stride = 1;
  for (int d = static_cast<int>(dims.size()) - 1; d >= 0; d--) {
    idx += coords[d] * stride;
    stride *= dims[d];
  }
  return idx;
}

// Decompose a linear index into multi-dimensional coordinates (row-major).
inline std::vector<Dim> delinearize(size_t idx, const std::vector<Dim>& dims) {
  std::vector<Dim> coords(dims.size());
  for (int d = static_cast<int>(dims.size()) - 1; d >= 0; d--) {
    coords[d] = idx % dims[d];
    idx /= dims[d];
  }
  return coords;
}

template <typename T>
void copyScalar(void* out, const void* data) {
  *(static_cast<T*>(out)) = *(static_cast<const T*>(data));
}

bool floatsEqual(const void* lhs, const void* rhs, unsigned numFloats) {
  auto lhsFloats = static_cast<const float*>(lhs);
  auto rhsFloats = static_cast<const float*>(rhs);
  // TODO consider adding loop parallelism
  for (auto i = 0; i < numFloats; i++) {
    if (std::abs(lhsFloats[i] - rhsFloats[i]) >= kfloatEqualTolerance) {
      return false;
    }
  }
  return true;
}

bool bytesEqual(const void* lhs, const void* rhs, unsigned numBytes) {
  return std::memcmp(lhs, rhs, numBytes) == 0;
}

// Canonicalize all integers or fl::end in the index (except for Tensor) into
// non-negative integers, e.g., range(0, -1) --> range(0, axisDim - 1).
// TODO consider moving this to a general util when we have other backends that
// reuse this logic.
Index canonicalizeIndex(const Index& idx, const Dim axisDim) {
  auto canonicalizeDim = [axisDim](const std::optional<Dim>& optDim) {
    if (!optDim.has_value()) {
      return axisDim;
    }
    const auto dim = optDim.value();
    if (dim < -axisDim || axisDim < dim) { // end is exclusive
      std::ostringstream oss;
      oss << "[canonicalizeIndexByShape] dim out of range: dim = " << dim
          << ", axisDim = " << axisDim;
      throw std::invalid_argument(oss.str());
    }
    return dim < 0 ? axisDim + dim : dim;
  };
  switch (idx.type()) {
    case detail::IndexType::Range: {
      const auto& rangeIdx = idx.get<range>();
      auto start = canonicalizeDim(rangeIdx.start());
      auto stride = rangeIdx.stride();
      auto end = canonicalizeDim(rangeIdx.end());
      if (start >= end) {
        std::ostringstream oss;
        oss << "[canonicalizeIndexByShape] end must be larger than start: end = "
            << end << ", start: " << start;
        throw std::invalid_argument(oss.str());
      }
      return range(start, end, stride);
    }
    case detail::IndexType::Literal: {
      // canonicalizeDim admits axisDim (an exclusive range end), which is
      // out of bounds for a literal.
      auto literal = canonicalizeDim(idx.get<Dim>());
      if (literal >= axisDim) {
        std::ostringstream oss;
        oss << "[canonicalizeIndexByShape] literal index too large: literal = "
            << literal << ", axisDim = " << axisDim;
        throw std::invalid_argument(oss.str());
      }
      return literal;
    }
    case detail::IndexType::Span: {
      return idx;
    }
    case detail::IndexType::Tensor: {
      return idx;
    }
  }
  throw std::runtime_error("Unexpected IndexType");
}

} // namespace

void* CPUTensor::bufferPtr() {
  return sharedData_->bufferPtr;
}

const void* CPUTensor::bufferPtr() const {
  return sharedData_->bufferPtr;
}

const std::vector<Dim>& CPUTensor::stridesVec() const {
  return strides_;
}

namespace {

// Compute contiguous row-major strides for the given shape.
std::vector<Dim> computeRowMajorStrides(const Shape& shape) {
  if (shape.ndim() == 0) {
    return {1}; // scalar has stride 1
  }
  std::vector<Dim> strides(shape.ndim());
  strides[shape.ndim() - 1] = 1;
  for (int i = shape.ndim() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape.dim(i + 1);
  }
  return strides;
}

// Check if strides represent a contiguous (row-major) layout for the given shape.
bool isContigStrides(const std::vector<Dim>& strides, const Shape& shape) {
  if (shape.ndim() == 0) return true;
  Dim expected = 1;
  for (int d = static_cast<int>(strides.size()) - 1; d >= 0; --d) {
    if (shape.dim(d) > 1 && strides[d] != expected) return false;
    expected *= shape.dim(d);
  }
  return true;
}

} // namespace

void CPUTensor::scatterToParent() {
  if (!parentIndexState_) {
    return;
  }
  auto& state = *parentIndexState_;
  const size_t typeSize = fl::getTypeSize(dtype_);
  const auto n = shape().elements();

  // Read this tensor's current data
  std::vector<char> selfData(n * typeSize);
  hostSync(selfData.data());

  // Write each element to the parent at the mapped position
  void* parentHandle = state.parent->bufferPtr;
  for (Dim i = 0; i < n; i++) {
    std::memcpy(
        static_cast<char*>(parentHandle) + state.indices[i] * typeSize,
        selfData.data() + i * typeSize,
        typeSize);
  }
}

CPUTensor::CPUTensor(
    std::shared_ptr<SharedData> sharedData,
    const Shape& shape)
    : sharedData_(std::move(sharedData)),
      shape_(shape),
      stream_(CPUBackend::getInstance().asyncStream()) {
  // dtype_ and strides_ must be set by the caller after construction.
}

CPUTensor::CPUTensor(
    std::shared_ptr<SharedData> sharedData,
    const Shape& shape,
    fl::dtype dtype,
    std::shared_ptr<AsyncCPUStream> stream)
    : sharedData_(std::move(sharedData)),
      shape_(shape),
      dtype_(dtype),
      strides_(computeRowMajorStrides(shape)),
      stream_(stream ? std::move(stream) : CPUBackend::getInstance().asyncStream()) {
}

void* CPUTensor::getOrEvalDataHandle() {
  if (!sharedData_->isDataReady) {
    stream().sync();
    sharedData_->isDataReady = true;
  }
  return sharedData_->bufferPtr;
}

size_t CPUTensor::getSizeInBytes() const {
  return shape_.elements() * fl::getTypeSize(dtype_);
}

CPUTensor::CPUTensor(
    const Shape& shape,
    fl::dtype type,
    void* bufferPtr,
    std::shared_ptr<void> dataOwner,
    std::shared_ptr<AsyncCPUStream> stream)
    : shape_(shape),
      dtype_(type),
      strides_(computeRowMajorStrides(shape)),
      stream_(stream ? std::move(stream) : CPUBackend::getInstance().asyncStream()) {
  sharedData_ = std::make_shared<SharedData>();
  sharedData_->bufferPtr = bufferPtr;
  sharedData_->bufferOwner = std::move(dataOwner);
}

CPUTensor::CPUTensor()
    : CPUTensor({0}, fl::dtype::f32, nullptr, Location::Host) {}

CPUTensor::CPUTensor(
    const Shape& shape,
    fl::dtype type,
    const void* ptr,
    Location memoryLocation)
    : shape_(shape),
      dtype_(type),
      strides_(computeRowMajorStrides(shape)),
      stream_(CPUBackend::getInstance().asyncStream()) {
  if (memoryLocation != Location::Host) {
    throw std::invalid_argument(
        "[CPUTensor] initialization data must be on host.");
  }

  const size_t numDataBytes =
      static_cast<size_t>(shape.elements()) * fl::getTypeSize(type);

  sharedData_ = std::make_shared<SharedData>();
  // Use AlignedBuffer for 64-byte aligned allocation (required for OneDNN
  // AVX-512 and good for cache performance).
  sharedData_->buffer =
      std::make_shared<AlignedBuffer>(std::max<size_t>(numDataBytes, 1));
  sharedData_->bufferPtr = sharedData_->buffer->data;
  if (ptr != nullptr) {
    std::memcpy(sharedData_->bufferPtr, ptr, numDataBytes);
  }
}

CPUTensor::CPUTensor(
    const Dim /* nRows */,
    const Dim /* nCols */,
    const Tensor& /* values */,
    const Tensor& /* rowIdx */,
    const Tensor& /* colIdx */,
    StorageType /* storageType */) {
  throw std::runtime_error(
      "CPUTensor currently doesn't support sparse tensor");
}

std::unique_ptr<TensorAdapterBase> CPUTensor::clone() const {
  // Sync to ensure source data is ready before copying
  stream_->sync();

  const auto typeSize = fl::getTypeSize(dtype_);
  const auto* srcBase = static_cast<const char*>(sharedData_->bufferPtr);
  const Dim totalElements = shape_.elements();
  const size_t totalBytes = totalElements * typeSize;

  // Allocate destination buffer
  auto dstBuffer = std::make_shared<AlignedBuffer>(std::max<size_t>(totalBytes, 1));
  auto* dstBase = static_cast<char*>(dstBuffer->data);

  if (isContigStrides(strides_, shape_)) {
    std::memcpy(dstBase, srcBase, totalBytes);
  } else {
    // Non-contiguous: copy element by element using strides
    const int ndim = static_cast<int>(strides_.size());
    const auto& dims = shape_.get();
    for (Dim i = 0; i < totalElements; ++i) {
      Dim rem = i;
      Dim srcOffset = 0;
      for (int d = ndim - 1; d >= 0; --d) {
        Dim coord = rem % dims[d];
        rem /= dims[d];
        srcOffset += coord * strides_[d];
      }
      std::memcpy(
          dstBase + i * typeSize,
          srcBase + srcOffset * typeSize,
          typeSize);
    }
  }

  auto result = std::make_unique<CPUTensor>(
      std::make_shared<SharedData>(), shape_);
  result->sharedData_->buffer = std::move(dstBuffer);
  result->sharedData_->bufferPtr = result->sharedData_->buffer->data;
  result->dtype_ = dtype_;
  result->strides_ = computeRowMajorStrides(shape_);
  result->stream_ = stream_;
  return result;
}

Tensor CPUTensor::copy() {
  return Tensor(clone());
}

Tensor CPUTensor::shallowCopy() {
  auto t = std::make_unique<CPUTensor>(sharedData_, shape_);
  t->stream_ = stream_;
  t->dtype_ = dtype_;
  t->strides_ = strides_;
  return Tensor(std::move(t));
}

TensorBackendType CPUTensor::backendType() const {
  return backend().backendType();
}

CPUBackend& CPUTensor::backend() const {
  return CPUBackend::getInstance();
}

const Shape& CPUTensor::shape() const {
  return shape_;
}

fl::dtype CPUTensor::type() const {
  return dtype_;
}

bool CPUTensor::isSparse() const {
  return false;
}

Location CPUTensor::location() const {
  // CPUTensor always lives on the host
  return Location::Host;
}

void CPUTensor::scalarSync(void* out) {
  if (shape().elements() == 0) {
    throw std::invalid_argument("Cannot call scalar on empty CPUTensor");
  }
  // Sync to ensure all enqueued work on this tensor's stream is complete
  stream_->sync();

  // Direct memcpy from the buffer pointer — always on host for CPUTensor
  const auto typeSize = fl::getTypeSize(dtype_);
  std::memcpy(out, sharedData_->bufferPtr, typeSize);
}

void CPUTensor::device(void** out) {
  *out = sharedData_->bufferPtr;
}

void CPUTensor::hostSync(void* out) {
  if (!isContiguous()) {
    asContiguousTensor().hostSync(out);
  } else {
    // Sync to ensure all enqueued work is complete before reading data
    stream_->sync();
    std::memcpy(out, sharedData_->bufferPtr, getSizeInBytes());
  }
}

void CPUTensor::unlock() {
  // No-op: buffer lifetime managed by shared_ptr.
}

bool CPUTensor::isLocked() {
  return false;
}

bool CPUTensor::isContiguous() const {
  const auto& shape = this->shape();
  if (shape.ndim() == 0) { // scalar
    return true;
  }
  const auto& dims = shape.get();
  Dim expected = 1;
  for (int d = static_cast<int>(strides_.size()) - 1; d >= 0; --d) {
    if (dims[d] > 1 && strides_[d] != expected) {
      return false;
    }
    expected *= dims[d];
  }
  return true;
}

Shape CPUTensor::strides() const {
  return Shape(strides_);
}

const Stream& CPUTensor::stream() const {
  return *stream_;
}

Tensor CPUTensor::astype(const dtype type) {
  if (type == this->type()) {
    // Return a view sharing the same data — no copy needed.
    auto viewSharedData = std::make_shared<SharedData>();
    viewSharedData->bufferPtr = sharedData_->bufferPtr;
    viewSharedData->buffer = sharedData_->buffer;
    viewSharedData->bufferOwner = sharedData_->bufferOwner;
    viewSharedData->parent = sharedData_;
    viewSharedData->isDataReady = sharedData_->isDataReady;
    auto view = std::make_unique<CPUTensor>(viewSharedData, shape_);
    view->stream_ = stream_;
    view->dtype_ = dtype_;
    view->strides_ = strides_;
    return Tensor(std::move(view));
  }

  // OneDNN reorder path: try using OneDNN for type conversion.
  // Skip float->int because OneDNN uses round-to-nearest-even while
  // C++ convention (and test expectations) use truncation.
  const auto srcType = this->type();
  bool srcIsFloat = (srcType == dtype::f32 || srcType == dtype::f64 ||
                     srcType == dtype::f16 || srcType == dtype::bf16);
  bool dstIsInt = (type == dtype::s32 || type == dtype::s16 ||
                   type == dtype::s64 || type == dtype::u8 ||
                   type == dtype::u16 || type == dtype::u32 ||
                   type == dtype::u64 || type == dtype::b8);

  if (!(srcIsFloat && dstIsInt)) {
    auto result = backend().reorderAstype(*this, type);
    if (result.elements() > 0) {
      return result;
    }
    // Empty result means OneDNN reorder failed — fall through to CPU fallback
  }

  // CPU fallback: read source as doubles, static_cast to target type.
  // Use the FL dtype (not OneDNN storage type) to interpret data correctly.
  const Dim n = shape().elements();
  std::vector<double> srcDoubles(n);

  // Helper to read source data as doubles using the actual FL dtype
  auto readSourceAsDoubles = [&]() {
    switch (srcType) {
      case dtype::f64: hostSync(srcDoubles.data()); break;
      case dtype::f32: {
        std::vector<float> f32(n);
        hostSync(f32.data());
        std::copy(f32.begin(), f32.end(), srcDoubles.begin());
        break;
      }
      case dtype::s32: {
        std::vector<int32_t> ints(n);
        asContiguousTensor().hostSync(ints.data());
        for (Dim i = 0; i < n; ++i) srcDoubles[i] = static_cast<double>(ints[i]);
        break;
      }
      case dtype::u32: {
        // u32 is stored as s32 (same 4 bytes), reinterpret
        std::vector<uint32_t> ints(n);
        asContiguousTensor().hostSync(ints.data());
        for (Dim i = 0; i < n; ++i) srcDoubles[i] = static_cast<double>(ints[i]);
        break;
      }
      case dtype::s64: {
        std::vector<int64_t> ints(n);
        asContiguousTensor().hostSync(ints.data());
        for (Dim i = 0; i < n; ++i) srcDoubles[i] = static_cast<double>(ints[i]);
        break;
      }
      case dtype::u64: {
        std::vector<uint64_t> ints(n);
        asContiguousTensor().hostSync(ints.data());
        for (Dim i = 0; i < n; ++i) srcDoubles[i] = static_cast<double>(ints[i]);
        break;
      }
      case dtype::s16: {
        std::vector<int16_t> ints(n);
        asContiguousTensor().hostSync(ints.data());
        for (Dim i = 0; i < n; ++i) srcDoubles[i] = static_cast<double>(ints[i]);
        break;
      }
      case dtype::u16: {
        std::vector<uint16_t> ints(n);
        asContiguousTensor().hostSync(ints.data());
        for (Dim i = 0; i < n; ++i) srcDoubles[i] = static_cast<double>(ints[i]);
        break;
      }
      case dtype::b8:
      case dtype::u8: {
        std::vector<uint8_t> ints(n);
        asContiguousTensor().hostSync(ints.data());
        for (Dim i = 0; i < n; ++i) srcDoubles[i] = static_cast<double>(ints[i]);
        break;
      }
      case dtype::bf16: {
        // bf16 has same exponent as f32, just 16 fewer mantissa bits.
        // Convert by shifting the 16-bit value into the upper half of f32.
        std::vector<uint16_t> raw(n);
        asContiguousTensor().hostSync(raw.data());
        for (Dim i = 0; i < n; ++i) {
          uint32_t bits = static_cast<uint32_t>(raw[i]) << 16;
          float f;
          std::memcpy(&f, &bits, sizeof(f));
          srcDoubles[i] = static_cast<double>(f);
        }
        break;
      }
      default:
        throw std::runtime_error(
            "CPUTensor::astype: unsupported source dtype for CPU fallback");
    }
  };
  readSourceAsDoubles();
  auto convertAndCreate = [&](auto dstTypeTag) {
    using DstT = decltype(dstTypeTag);
    std::vector<DstT> dst(n);
    for (Dim i = 0; i < n; ++i) {
      dst[i] = static_cast<DstT>(srcDoubles[i]);
    }
    return toTensor<CPUTensor>(shape(), type, dst.data(), Location::Host);
  };
  switch (type) {
    case dtype::f32:
      return convertAndCreate(float{});
    case dtype::f64:
      return convertAndCreate(double{});
    case dtype::s32:
      return convertAndCreate(int{});
    case dtype::b8: {
      // Boolean: nonzero --> 1, zero --> 0
      std::vector<char> dst(n);
      for (Dim i = 0; i < n; ++i) {
        dst[i] = srcDoubles[i] != 0.0 ? 1 : 0;
      }
      return toTensor<CPUTensor>(shape(), type, dst.data(), Location::Host);
    }
    case dtype::u8:
      return convertAndCreate(static_cast<unsigned char>(0));
    case dtype::s16:
      return convertAndCreate(short{});
    case dtype::s64:
      return convertAndCreate(static_cast<long long>(0));
    case dtype::u16:
      return convertAndCreate(static_cast<unsigned short>(0));
    case dtype::u32:
      return convertAndCreate(static_cast<unsigned int>(0));
    case dtype::u64:
      return convertAndCreate(static_cast<unsigned long long>(0));
    default:
      throw std::invalid_argument(
          "[CPUTensor::astype] unsupported target dtype for CPU fallback");
  }
}

Tensor CPUTensor::index(const std::vector<Index>& indices) {
  const auto& shape = this->shape();
  // allow indexing scalar with empty indices
  if (indices.size() > shape.ndim() ||
      (indices.size() == 0 && shape.ndim() > 0)) {
    std::ostringstream oss;
    oss << "[CPUTensor::index] Invalid number of indices: " << indices.size()
        << " for tensor of ndim = " << shape.ndim();
    throw std::invalid_argument(oss.str());
  }

  // If there are Tensor indices mixed with other non-span indices, or multiple
  // Tensor indices, process Tensor indices one at a time. The gather code
  // returns a new tensor, so we apply one Tensor index per pass with spans
  // elsewhere, then apply the remaining indices on the result.
  //
  // Also handle boolean masks in multi-dimensional indexing: boolean mask
  // indexing flattens the source, so when used alongside other indices
  // (even spans) on a multi-dim tensor, convert to integer indices first.
  {
    int tensorCount = 0;
    int nonSpanCount = 0;
    int firstTensorAxis = -1;
    bool hasBoolMaskInMultiDimIndex = false;
    for (int i = 0; i < static_cast<int>(indices.size()); i++) {
      if (indices[i].type() == detail::IndexType::Tensor) {
        tensorCount++;
        if (firstTensorAxis == -1) firstTensorAxis = i;
        // Boolean mask indexing does flat gathering, which is wrong when
        // there are additional axes. Convert to integer indices.
        if (indices[i].get<Tensor>().type() == dtype::b8 &&
            indices.size() > 1) {
          hasBoolMaskInMultiDimIndex = true;
        }
      }
      if (indices[i].type() != detail::IndexType::Span) {
        nonSpanCount++;
      }
    }
    // Convert boolean masks to integer indices for multi-dim indexing
    if (hasBoolMaskInMultiDimIndex) {
      std::vector<Index> converted;
      converted.reserve(indices.size());
      for (int i = 0; i < static_cast<int>(indices.size()); i++) {
        if (indices[i].type() == detail::IndexType::Tensor &&
            indices[i].get<Tensor>().type() == dtype::b8) {
          converted.push_back(fl::nonzero(indices[i].get<Tensor>()));
        } else {
          converted.push_back(indices[i]);
        }
      }
      return this->index(converted);
    }
    if (tensorCount >= 1 && nonSpanCount > 1) {
      // Apply just the first Tensor index with spans for everything else
      std::vector<Index> pass1(indices.size(), fl::span);
      pass1[firstTensorAxis] = indices[firstTensorAxis];
      auto intermediate = this->index(pass1);
      // The Tensor index may reduce dimensionality (scalar index removes axis).
      // Build remaining indices without the Tensor index we just applied.
      const auto& idxTensor = indices[firstTensorAxis].get<Tensor>();
      bool scalarIdx = (idxTensor.type() != dtype::b8 && idxTensor.ndim() == 0);
      std::vector<Index> remaining;
      for (int i = 0; i < static_cast<int>(indices.size()); i++) {
        if (i == firstTensorAxis) {
          if (!scalarIdx) {
            remaining.push_back(fl::span); // axis still exists, just gathered
          }
          // If scalar, axis was removed — don't add anything
        } else {
          remaining.push_back(indices[i]);
        }
      }
      if (!remaining.empty()) {
        return intermediate(remaining);
      }
      return intermediate;
    }
  }

  // by default, assume all indices are spans
  std::vector<Dim> dims(shape.get().begin(), shape.get().end());
  std::vector<Dim> offsets(shape.ndim(), 0);
  std::vector<int> dimsAxesWithLiteralIndex;
  // Track axes that have stride > 1 for strided copy
  std::vector<std::pair<int, Dim>> stridedAxes; // (axis, stride)
  for (int axis = 0; axis < indices.size(); axis++) {
    const auto idx = canonicalizeIndex(indices[axis], shape[axis]);
    switch (idx.type()) {
      case detail::IndexType::Range: {
        const auto& rangeIdx = idx.get<range>();
        offsets[axis] = rangeIdx.start();
        Dim stride = rangeIdx.stride();
        Dim len = rangeIdx.endVal() - rangeIdx.start();
        dims[axis] = (len + stride - 1) / stride; // ceil division
        if (stride > 1) {
          stridedAxes.push_back({axis, stride});
        }
        continue;
      }
      case detail::IndexType::Span: {
        continue;
      }
      case detail::IndexType::Literal: {
        offsets[axis] = idx.get<Dim>();
        dims[axis] = 1;
        dimsAxesWithLiteralIndex.push_back(axis);
        continue;
      }
      case detail::IndexType::Tensor: {
        // Tensor-as-index: gather elements.
        // Ensure the index tensor's data is ready relative to this tensor's
        // stream. The host() / toHostVector() calls below will sync our own
        // stream before reading.
        // TODO: The JIT should handle this more elegantly via lazy
        // materialization — index tensors would be materialized on demand,
        // avoiding any explicit sync here.
        const auto& idxTensor = idx.get<Tensor>();
        fl::relativeSync(*stream_, {idxTensor});
        const auto srcType = this->type();
        const auto srcShape = this->shape();
        const auto srcDims = srcShape.get();
        const Dim nSrc = srcShape.elements();
        const size_t typeSize = fl::getTypeSize(srcType);

        if (idxTensor.type() == dtype::b8) {
          // Boolean mask indexing: gather elements where mask is nonzero.
          // The mask covers all elements (flat), regardless of shape.
          // Result is always 1D with shape {numTrue}.
          auto mask = idxTensor.toHostVectorSync<char>();
          std::vector<int> flatIndices;
          for (size_t i = 0; i < mask.size(); i++) {
            if (mask[i]) {
              flatIndices.push_back(static_cast<int>(i));
            }
          }

          if (flatIndices.empty()) {
            return toTensor<CPUTensor>(
                Shape({0}), srcType, nullptr, Location::Host);
          }

          const Dim numIdx = static_cast<Dim>(flatIndices.size());
          Shape outShape({numIdx});

          std::vector<char> srcBytes(nSrc * typeSize);
          hostSync(srcBytes.data());
          std::vector<char> dstBytes(numIdx * typeSize);

          for (Dim i = 0; i < numIdx; i++) {
            std::memcpy(
                dstBytes.data() + i * typeSize,
                srcBytes.data() + flatIndices[i] * typeSize,
                typeSize);
          }

          auto result = toTensor<CPUTensor>(
              outShape, srcType, dstBytes.data(), Location::Host);
          // Store parent reference and flat index map for write-back
          auto& resultTensor = toCPUTensor(result);
          resultTensor.parentIndexState_ = ParentIndexState{
              sharedData_, std::move(flatIndices)};
          return result;
        } else {
          // Integer tensor indexing: gather along the given axis
          auto idxData = idxTensor.astype(dtype::s32).toHostVectorSync<int>();

          if (idxData.empty()) {
            return toTensor<CPUTensor>(
                Shape({0}), srcType, nullptr, Location::Host);
          }

          const Dim axisDim = srcShape[axis];
          size_t innerSize = 1;
          for (int d = srcShape.ndim() - 1; d > axis; d--) {
            innerSize *= srcDims[d];
          }
          size_t outerSize = 1;
          for (int d = 0; d < axis; d++) {
            outerSize *= srcDims[d];
          }

          const Dim numIdx = static_cast<Dim>(idxData.size());
          std::vector<Dim> outDims = srcDims;
          outDims[axis] = numIdx;
          // When index is a scalar (0-dim), remove the axis dimension
          // to match NumPy/PyTorch semantics (scalar index --> dim removed)
          const bool scalarIndex = (idxTensor.ndim() == 0);
          if (scalarIndex) {
            outDims.erase(outDims.begin() + axis);
          }
          Shape outShape(outDims);

          std::vector<char> srcBytes(nSrc * typeSize);
          hostSync(srcBytes.data());
          const Dim outElements = outShape.elements();
          std::vector<char> dstBytes(outElements * typeSize);
          std::vector<int> flatIndexMap(outElements);

          Dim dstIdx = 0;
          for (size_t outer = 0; outer < outerSize; outer++) {
            for (size_t gi = 0; gi < idxData.size(); gi++) {
              int srcIdx = idxData[gi];
              for (size_t inner = 0; inner < innerSize; inner++) {
                size_t srcLinear = outer * axisDim * innerSize +
                    srcIdx * innerSize + inner;
                std::memcpy(
                    dstBytes.data() + dstIdx * typeSize,
                    srcBytes.data() + srcLinear * typeSize,
                    typeSize);
                flatIndexMap[dstIdx] = static_cast<int>(srcLinear);
                dstIdx++;
              }
            }
          }

          auto result = toTensor<CPUTensor>(
              outShape, srcType, dstBytes.data(), Location::Host);
          auto& resultTensor = toCPUTensor(result);
          resultTensor.parentIndexState_ = ParentIndexState{
              sharedData_, std::move(flatIndexMap)};
          return result;
        }
      }
    }
    throw std::runtime_error("Unexpected IndexType");
  }

  // Compute element offset into the parent buffer using the current strides.
  const auto typeSize = fl::getTypeSize(dtype_);
  Dim elemOffset = 0;
  for (int i = 0; i < static_cast<int>(offsets.size()); i++) {
    elemOffset += offsets[i] * strides_[i];
  }

  // Remove literal-indexed dimensions to get the result shape.
  std::vector<Dim> condensedDims;
  {
    std::unordered_set<int> literalSet(
        dimsAxesWithLiteralIndex.begin(), dimsAxesWithLiteralIndex.end());
    for (int idx = 0; idx < static_cast<int>(dims.size()); idx++) {
      if (!literalSet.count(idx)) {
        condensedDims.push_back(dims[idx]);
      }
    }
  }
  const Shape indexedShape(condensedDims);

  // Compute effective strides: for axes with index stride > 1, the effective
  // stride is parentStride * indexStride.
  std::vector<Dim> effectiveStrides = strides_;
  for (auto& [axis, idxStride] : stridedAxes) {
    effectiveStrides[axis] = strides_[axis] * idxStride;
  }

  // Compute the condensed strides by removing literal-indexed axes.
  std::vector<Dim> condensedStrides;
  for (int axis = 0; axis < static_cast<int>(dims.size()); axis++) {
    bool isLiteral = false;
    for (int li : dimsAxesWithLiteralIndex) {
      if (li == axis) { isLiteral = true; break; }
    }
    if (!isLiteral) {
      condensedStrides.push_back(effectiveStrides[axis]);
    }
  }

  // Handle scalar result (no remaining dimensions)
  if (condensedStrides.empty()) {
    condensedStrides = {1};
  }

  void* offsetDataHandle =
      static_cast<char*>(sharedData_->bufferPtr) + elemOffset * typeSize;
  auto viewSharedData = std::make_shared<SharedData>();
  viewSharedData->bufferPtr = offsetDataHandle;
  viewSharedData->parent = sharedData_; // keep parent buffer alive
  viewSharedData->isDataReady = sharedData_->isDataReady;
  auto view = std::make_unique<CPUTensor>(viewSharedData, indexedShape);
  view->stream_ = stream_;
  view->dtype_ = dtype_;
  view->strides_ = std::move(condensedStrides);
  return Tensor(std::move(view));
}

Tensor CPUTensor::flatten() const {
  const Dim n = shape_.elements();
  const Shape flatShape({n});

  if (isContigStrides(strides_, shape_)) {
    // Contiguous: return a 1D view sharing the same buffer
    auto viewSharedData = std::make_shared<SharedData>();
    viewSharedData->bufferPtr = sharedData_->bufferPtr;
    viewSharedData->buffer = sharedData_->buffer;
    viewSharedData->bufferOwner = sharedData_->bufferOwner;
    viewSharedData->parent = sharedData_;
    viewSharedData->isDataReady = sharedData_->isDataReady;
    auto view = std::make_unique<CPUTensor>(viewSharedData, flatShape);
    view->stream_ = stream_;
    view->dtype_ = dtype_;
    view->strides_ = {1};
    return Tensor(std::move(view));
  }

  // Non-contiguous: copy to contiguous layout, then reshape to 1D
  auto copyResult = Tensor(clone());
  auto& copyImpl = toCPUTensor(copyResult);
  auto viewSharedData = std::make_shared<SharedData>();
  viewSharedData->bufferPtr = copyImpl.sharedData_->bufferPtr;
  viewSharedData->buffer = copyImpl.sharedData_->buffer;
  viewSharedData->bufferOwner = copyImpl.sharedData_->bufferOwner;
  viewSharedData->parent = copyImpl.sharedData_;
  viewSharedData->isDataReady = copyImpl.sharedData_->isDataReady;
  auto view = std::make_unique<CPUTensor>(viewSharedData, flatShape);
  view->stream_ = stream_;
  view->dtype_ = dtype_;
  view->strides_ = {1};
  return Tensor(std::move(view));
}

Tensor CPUTensor::flat(const Index& idx) const {
  const auto elemSize = fl::getTypeSize(dtype_);
  const auto& dims = shape_.get();

  if (idx.type() == detail::IndexType::Literal) {
    const auto i = idx.get<Dim>();

    // Compute the physical offset for flat element i
    Dim offset = 0;
    if (isContigStrides(strides_, shape_)) {
      offset = i;
    } else {
      // Convert flat index to multi-dim coords, then to strided offset
      Dim rem = i;
      for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
        offset += (rem % dims[d]) * strides_[d];
        rem /= dims[d];
      }
    }

    void* offsetDataHandle =
        static_cast<char*>(sharedData_->bufferPtr) + offset * elemSize;
    auto viewSharedData = std::make_shared<SharedData>();
    viewSharedData->bufferPtr = offsetDataHandle;
    viewSharedData->parent = sharedData_;
    viewSharedData->isDataReady = sharedData_->isDataReady;
    auto view = std::make_unique<CPUTensor>(viewSharedData, Shape({}));
    view->stream_ = stream_;
    view->dtype_ = dtype_;
    view->strides_ = {1};
    return Tensor(std::move(view));
  }
  // For non-literal indices (ranges, tensors), create a 1D view of the
  // original buffer and index into it.
  if (isContigStrides(strides_, shape_)) {
    const Dim n = shape_.elements();
    const Shape flatShape({n});
    auto viewSharedData = std::make_shared<SharedData>();
    viewSharedData->bufferPtr = sharedData_->bufferPtr;
    viewSharedData->buffer = sharedData_->buffer;
    viewSharedData->bufferOwner = sharedData_->bufferOwner;
    viewSharedData->parent = sharedData_;
    viewSharedData->isDataReady = sharedData_->isDataReady;
    auto view = std::make_unique<CPUTensor>(viewSharedData, flatShape);
    view->stream_ = stream_;
    view->dtype_ = dtype_;
    view->strides_ = {1};
    return Tensor(std::move(view))({idx});
  }
  // Non-contiguous: copy to contiguous, then flatten the copy.
  auto contiguousCopy = Tensor(clone());
  return contiguousCopy.flatten()({idx});
}

Tensor CPUTensor::asContiguousTensor() {
  if (isContiguous()) {
    // Return a view sharing the same buffer — no copy needed.
    auto viewSharedData = std::make_shared<SharedData>();
    viewSharedData->bufferPtr = sharedData_->bufferPtr;
    viewSharedData->buffer = sharedData_->buffer;
    viewSharedData->bufferOwner = sharedData_->bufferOwner;
    viewSharedData->parent = sharedData_;
    viewSharedData->isDataReady = sharedData_->isDataReady;
    auto view = std::make_unique<CPUTensor>(viewSharedData, shape_);
    view->stream_ = stream_;
    view->dtype_ = dtype_;
    view->strides_ = strides_;
    return Tensor(std::move(view));
  }
  return this->copy();
}

void CPUTensor::setContext(void* /* context */) {
  // no-op
}

void* CPUTensor::getContext() {
  return nullptr;
}

template <typename T>
void printData(std::ostringstream& oss, const T& element) {
  oss << element;
}

// specialization to print out element as number instead of ascii character
template <>
void printData(std::ostringstream& oss, const unsigned char& element) {
  printData<unsigned>(oss, element); // safe cast w/o precision loss
}

// specialization to print out element as number instead of ascii character
template <>
void printData(std::ostringstream& oss, const char& element) {
  printData<int>(oss, element); // safe cast w/o precision loss
}

// Print a 1D array of elements as [e0, e1, ..., eN].
// Returns pointer past the last element consumed.
template <typename T>
const T*
printData1D(std::ostringstream& oss, const T* elements, const fl::Dim count) {
  oss << '[';
  for (Dim i = 0; i < count; i++) {
    if (i != 0) {
      oss << ", ";
    }
    printData(oss, elements[i]);
  }
  oss << ']';
  return elements + count;
}

// Recursively print a row-major tensor starting at dimension dimIdx.
// Data is laid out so that the first dimension is outermost (slowest-varying)
// and the last dimension is innermost (fastest-varying).
// Returns pointer past the last element consumed.
template <typename T>
const T* printDataRowMajor(
    std::ostringstream& oss,
    const T* elements,
    const std::vector<fl::Dim>& dims,
    const unsigned dimIdx) {
  const unsigned ndim = dims.size();
  if (dimIdx >= ndim) {
    // Scalar: print single element
    printData(oss, *elements);
    return elements + 1;
  }
  if (dimIdx == ndim - 1) {
    // Last dimension: print as 1D list
    return printData1D(oss, elements, dims[dimIdx]);
  }
  // Intermediate dimension: print as [slice0, slice1, ...]
  const auto size = dims[dimIdx];
  oss << '[';
  const T* nextStart = elements;
  for (Dim i = 0; i < size; i++) {
    if (i != 0) {
      oss << ',' << std::endl;
      oss << std::string(dimIdx + 1, ' ');
    }
    nextStart = printDataRowMajor(oss, nextStart, dims, dimIdx + 1);
  }
  oss << ']';
  return nextStart;
}

template <typename T>
std::string dataToString(const void* data, const Shape& shape) {
  std::ostringstream oss;
  if (shape.ndim() == 0) {
    printData(oss, *static_cast<const T*>(data));
  } else {
    printDataRowMajor(
        oss, static_cast<const T*>(data), shape.get(), 0);
  }
  oss << std::endl;
  return oss.str();
}

std::string CPUTensor::toString() {
  // TODO lift this up into a util method: Tensor -> std::string
  std::vector<char> vec(getSizeInBytes());
  void* data = vec.data();
  this->hostSync(data);
  const auto& shape = this->shape();
  switch (type()) {
    case fl::dtype::f16:
    case fl::dtype::bf16:
      throw std::runtime_error("CPUTensor::toString doesn't support f16/bf16");
    case fl::dtype::f32:
      return dataToString<float>(data, shape);
    case fl::dtype::f64:
      return dataToString<double>(data, shape);
    case fl::dtype::b8:
      return dataToString<char>(data, shape);
    case fl::dtype::s16:
      return dataToString<short>(data, shape);
    case fl::dtype::s32:
      return dataToString<int>(data, shape);
    case fl::dtype::s64:
      return dataToString<long long>(data, shape);
    case fl::dtype::u8:
      return dataToString<unsigned char>(data, shape);
    case fl::dtype::u16:
      return dataToString<unsigned short>(data, shape);
    case fl::dtype::u32:
      return dataToString<unsigned int>(data, shape);
    case fl::dtype::u64:
      return dataToString<unsigned long long>(data, shape);
  }
  throw std::invalid_argument("CPUTensor::toString: unsupported dtype");
}

std::ostream& CPUTensor::operator<<(std::ostream& ostr) {
  return ostr << toString();
}

/******************** Assignment Operators ********************/



// Scalar assign: fill all elements with the given value.
#define FL_CPU_TENSOR_ASSIGN_SCALAR_IMPL(TYPE)                              \
  void CPUTensor::assign(const TYPE& val) {                                 \
    backend().fillInPlace(*this, static_cast<double>(val));         \
    scatterToParent();                                                    \
  }                                                                            \
  void CPUTensor::inPlaceAdd(const TYPE& val) {                             \
    backend().inPlaceAdd(*this, static_cast<double>(val));          \
    scatterToParent();                                                    \
  }                                                                            \
  void CPUTensor::inPlaceSubtract(const TYPE& val) {                        \
    backend().inPlaceSub(*this, static_cast<double>(val));          \
    scatterToParent();                                                    \
  }                                                                            \
  void CPUTensor::inPlaceMultiply(const TYPE& val) {                        \
    backend().inPlaceMul(*this, static_cast<double>(val));          \
    scatterToParent();                                                    \
  }                                                                            \
  void CPUTensor::inPlaceDivide(const TYPE& val) {                          \
    backend().inPlaceDiv(*this, static_cast<double>(val));          \
    scatterToParent();                                                    \
  }
FL_CPU_TENSOR_ASSIGN_SCALAR_IMPL(double)
FL_CPU_TENSOR_ASSIGN_SCALAR_IMPL(float)
FL_CPU_TENSOR_ASSIGN_SCALAR_IMPL(int)
FL_CPU_TENSOR_ASSIGN_SCALAR_IMPL(unsigned)
FL_CPU_TENSOR_ASSIGN_SCALAR_IMPL(bool)
FL_CPU_TENSOR_ASSIGN_SCALAR_IMPL(char)
FL_CPU_TENSOR_ASSIGN_SCALAR_IMPL(unsigned char)
FL_CPU_TENSOR_ASSIGN_SCALAR_IMPL(short)
FL_CPU_TENSOR_ASSIGN_SCALAR_IMPL(unsigned short)
FL_CPU_TENSOR_ASSIGN_SCALAR_IMPL(long)
FL_CPU_TENSOR_ASSIGN_SCALAR_IMPL(unsigned long)
FL_CPU_TENSOR_ASSIGN_SCALAR_IMPL(long long)
FL_CPU_TENSOR_ASSIGN_SCALAR_IMPL(unsigned long long)
#undef FL_CPU_TENSOR_ASSIGN_SCALAR_IMPL

// Tensor-Tensor in-place ops using OneDNN binary in-place.
void CPUTensor::inPlaceAdd(const Tensor& other) {
  backend().inPlaceAdd(*this, other);
  scatterToParent();
}
void CPUTensor::inPlaceSubtract(const Tensor& other) {
  backend().inPlaceSub(*this, other);
  scatterToParent();
}
void CPUTensor::inPlaceMultiply(const Tensor& other) {
  backend().inPlaceMul(*this, other);
  scatterToParent();
}
void CPUTensor::inPlaceDivide(const Tensor& other) {
  backend().inPlaceDiv(*this, other);
  scatterToParent();
}

void CPUTensor::assign(const Tensor& tensor) {
  auto& other = toCPUTensor(tensor);
  if (this->sharedData_ == other.sharedData_) {
    return;
  }

  if (this->shape() != other.shape()) {
    // Allow assignment when shapes differ only by singleton dimensions
    // (e.g., assigning {1, 10} to {10} or vice versa). This matches
    // NumPy/PyTorch semantics where indexing assignment broadcasts.
    if (this->shape().elements() == other.shape().elements()) {
      auto reshaped = fl::reshape(tensor, this->shape());
      this->assign(reshaped);
      return;
    }
    std::ostringstream oss;
    oss << "Cannot update OneDNN tensor to different shape: lhs="
        << this->shape() << " rhs=" << other.shape();
    throw std::invalid_argument(oss.str());
  }

  backend().assignTensor(*this, tensor);
  this->sharedData_->isDataReady = false;
  scatterToParent();
}

bool CPUTensor::equals(CPUTensor&& other) {
  // TODO lift this up into a util method: (Tensor, Tensor) -> std::string
  if (this->sharedData_ == other.sharedData_) {
    return true;
  }
  if (this->shape() != other.shape()) {
    return false;
  }
  if (this->type() != other.type()) {
    return false;
  }
  // TODO investigate ways to speed up this on non-CPU platform.
  std::vector<char> lhsVec(this->getSizeInBytes());
  std::vector<char> rhsVec(other.getSizeInBytes());
  void* lhsData = lhsVec.data();
  void* rhsData = rhsVec.data();
  this->hostSync(lhsData);
  other.hostSync(rhsData);
  return this->type() == fl::dtype::f32
      ? floatsEqual(lhsData, rhsData, this->shape().elements())
      : bytesEqual(lhsData, rhsData, getSizeInBytes());
}

CPUTensor& toCPUTensor(const Tensor& tensor) {
  auto type = tensor.backendType();
  if (type != TensorBackendType::CPU) {
    std::ostringstream oss;
    oss << "[toCPUTensor] expected CPU-backed tensor, got " << type;
    throw std::invalid_argument(oss.str());
  }
  return tensor.getAdapter<CPUTensor>();
}

CPUTensor& toCPUTensor(Tensor& tensor) {
  return toCPUTensor(static_cast<const Tensor&>(tensor));
}

} // namespace fl
