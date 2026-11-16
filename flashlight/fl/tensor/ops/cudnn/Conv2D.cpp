/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ops/cudnn/CudnnAutogradOps.h"

#include <array>
#include <stdexcept>
#include <vector>

#include <cudnn_frontend.h>

#include "flashlight/fl/tensor/ops/cudnn/CudnnNumerics.h"
#include "flashlight/fl/tensor/ops/cudnn/CudnnPlanCache.h"
#include "flashlight/fl/tensor/ops/cudnn/CudnnTensorHelpers.h"
#include "flashlight/fl/tensor/ops/cudnn/CudnnUtils.h"
#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/tensor/Compute.h"

namespace fl {
namespace detail {
namespace cudnn {

namespace fe = cudnn_frontend;

namespace {

// Compute contiguous row-major strides for a 4D shape.
std::vector<int64_t> contiguousStrides(const std::vector<int64_t>& dims) {
  std::vector<int64_t> strides(4);
  strides[3] = 1;
  for (int i = 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides;
}

// Convert fl::Shape to 4D int64_t vector, padding with 1s.
std::vector<int64_t> shapeTo4D(const Shape& s) {
  std::vector<int64_t> dims(4, 1);
  for (unsigned i = 0; i < s.ndim() && i < 4; ++i) {
    dims[i] = s[i];
  }
  return dims;
}

// Compute conv2d output dimensions: NCHW layout.
Shape computeConvOutputShape(
    const Shape& input,
    const Shape& weights,
    int sx, int sy, int px, int py, int dx, int dy, int groups) {
  auto N = input[0];
  auto C_out = weights[0];
  auto H_in = input[2];
  auto W_in = input[3];
  auto kH = weights[2];
  auto kW = weights[3];
  auto H_out = (H_in + 2 * py - dy * (kH - 1) - 1) / sy + 1;
  auto W_out = (W_in + 2 * px - dx * (kW - 1) - 1) / sx + 1;
  return Shape{N, C_out, H_out, W_out};
}

// Compute type for cuDNN: f16/bf16 accumulate in f32.
fe::DataType_t computeType(fl::dtype dt) {
  return (dt == fl::dtype::f16 || dt == fl::dtype::bf16)
      ? fe::DataType_t::FLOAT : flTypeToCudnn(dt);
}

} // namespace

ForwardBackwardState conv2dForward(
    Tensor& out,
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    bool /*wantState*/) {
  if (input.ndim() != 4) {
    throw std::invalid_argument(
        "conv2d: expects input tensor to be 4 dimensions: "
        "in NCHW ordering. Given tensor has " +
        std::to_string(input.ndim()) + " dimensions.");
  }

  auto hasBias = bias.elements() > 0;
  auto outputShape = computeConvOutputShape(
      input.shape(), weights.shape(), sx, sy, px, py, dx, dy, groups);
  auto dt = flTypeToCudnn(input.type());
  auto cType = computeType(input.type());

  // UID assignments: 1=input, 2=weights, 3=bias, 4=conv_out, 5=output
  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(dt)
       .set_intermediate_data_type(dt)
       .set_compute_data_type(cType);

  auto xDims = shapeTo4D(input.shape());
  auto wDims = shapeTo4D(weights.shape());
  auto yDims = shapeTo4D(outputShape);

  auto X = graph->tensor(fe::graph::Tensor_attributes()
      .set_dim(xDims).set_stride(contiguousStrides(xDims))
      .set_data_type(dt).set_uid(1));
  auto W = graph->tensor(fe::graph::Tensor_attributes()
      .set_dim(wDims).set_stride(contiguousStrides(wDims))
      .set_data_type(dt).set_uid(2));

  // Groups are implicit in weight tensor dimensions: W has shape
  // {outC, inC/groups, kH, kW}. The graph API infers groups from the
  // dimension mismatch between X (channels) and W (channels).
  auto convAttrs = fe::graph::Conv_fprop_attributes()
      .set_padding({py, px})
      .set_stride({sy, sx})
      .set_dilation({dy, dx});

  auto Y = graph->conv_fprop(X, W, convAttrs);

  // Add bias if present
  std::shared_ptr<fe::graph::Tensor_attributes> finalOut;
  if (hasBias) {
    auto bDims = shapeTo4D(bias.shape());
    auto B = graph->tensor(fe::graph::Tensor_attributes()
        .set_dim(bDims).set_stride(contiguousStrides(bDims))
        .set_data_type(dt).set_uid(3));
    finalOut = graph->pointwise(Y, B,
        fe::graph::Pointwise_attributes().set_mode(
            fe::PointwiseMode_t::ADD));
  } else {
    finalOut = Y;
  }

  finalOut->set_output(true)
      .set_dim(yDims).set_stride(contiguousStrides(yDims))
      .set_data_type(dt).set_uid(hasBias ? 5 : 4);

  auto handle = getCudnnHandle();
  auto status = graph->validate();
  if (!status.is_good()) {
    throw std::runtime_error(
        "conv2dForward: graph validation failed: " + std::string(status.get_message()));
  }
  status = graph->build_operation_graph(handle);
  if (!status.is_good()) {
    throw std::runtime_error(
        "conv2dForward: build_operation_graph failed: " + std::string(status.get_message()));
  }
  status = graph->create_execution_plans({fe::HeurMode_t::A});
  if (!status.is_good()) {
    throw std::runtime_error(
        "conv2dForward: create_execution_plans failed: " + std::string(status.get_message()));
  }
  detail::cudnn::restrictToStrictMath(
      *graph, detail::cudnn::isWideFloat(input.type()));
  status = graph->check_support(handle);
  if (!status.is_good()) {
    throw std::runtime_error(
        "conv2dForward: check_support failed: " + std::string(status.get_message()));
  }
  status = graph->build_plans(handle);
  if (!status.is_good()) {
    throw std::runtime_error(
        "conv2dForward: build_plans failed: " + std::string(status.get_message()));
  }

  auto output = (!out.isEmpty() && out.shape() == outputShape && out.type() == input.type())
      ? out : makeTensorLike(input, outputShape, input.type());
  auto wsSize = graph->get_workspace_size();
  Tensor wspace;
  if (wsSize > 0) {
    wspace = makeTensorLike(
        input, Shape{static_cast<long long>(wsSize)}, fl::dtype::b8);
  }

  const auto& cudnnStream = getCudnnStream();

  auto contiguousWeights = weights.asContiguousTensor();
  DevicePtr inPtr(input);
  DevicePtr wtPtr(contiguousWeights);
  DevicePtr outPtr(output);

  std::unordered_map<int64_t, void*> variantPack;
  variantPack[1] = inPtr.get();
  variantPack[2] = wtPtr.get();

  if (hasBias) {
    // bias DevicePtr must stay alive until execution completes
    auto biasPtr = std::make_unique<DevicePtr>(bias);
    variantPack[3] = biasPtr->get();
    variantPack[hasBias ? 5 : 4] = outPtr.get();
    relativeSync(cudnnStream, {input, contiguousWeights, bias, output});
    if (wsSize > 0) {
      DevicePtr wsPtr(wspace);
      relativeSync(cudnnStream, {wspace});
      status = graph->execute(handle, variantPack, wsPtr.get());
    } else {
      status = graph->execute(handle, variantPack, nullptr);
    }
  } else {
    variantPack[4] = outPtr.get();
    relativeSync(cudnnStream, {input, contiguousWeights, output});
    if (wsSize > 0) {
      DevicePtr wsPtr(wspace);
      relativeSync(cudnnStream, {wspace});
      status = graph->execute(handle, variantPack, wsPtr.get());
    } else {
      status = graph->execute(handle, variantPack, nullptr);
    }
  }

  if (!status.is_good()) {
    throw std::runtime_error(
        "conv2dForward: execute failed: " + std::string(status.get_message()));
  }

  relativeSync({output}, cudnnStream);
  out = std::move(output);
  return nullptr;
}

Tensor conv2dBackwardData(
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& weight,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const ForwardBackwardState& /*state*/) {
  auto dt = flTypeToCudnn(input.type());
  auto cType = computeType(input.type());
  auto handle = getCudnnHandle();
  const auto& cudnnStream = getCudnnStream();

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(dt)
       .set_intermediate_data_type(dt)
       .set_compute_data_type(cType);

  auto dyDims = shapeTo4D(gradOutput.shape());
  auto wDims = shapeTo4D(weight.shape());
  auto dxDims = shapeTo4D(input.shape());

  auto DY = graph->tensor(fe::graph::Tensor_attributes()
      .set_dim(dyDims).set_stride(contiguousStrides(dyDims))
      .set_data_type(dt).set_uid(1));
  auto W = graph->tensor(fe::graph::Tensor_attributes()
      .set_dim(wDims).set_stride(contiguousStrides(wDims))
      .set_data_type(dt).set_uid(2));

  auto dgradAttrs = fe::graph::Conv_dgrad_attributes()
      .set_padding({py, px})
      .set_stride({sy, sx})
      .set_dilation({dy, dx});

  auto DX = graph->conv_dgrad(DY, W, dgradAttrs);
  DX->set_output(true)
      .set_dim(dxDims).set_stride(contiguousStrides(dxDims))
      .set_data_type(dt).set_uid(3);

  auto status = graph->validate();
  if (!status.is_good()) {
    throw std::runtime_error(
        "conv2dBackwardData: validate failed: " + std::string(status.get_message()));
  }
  status = graph->build_operation_graph(handle);
  if (!status.is_good()) {
    throw std::runtime_error(
        "conv2dBackwardData: build_operation_graph failed: " + std::string(status.get_message()));
  }
  status = graph->create_execution_plans({fe::HeurMode_t::A});
  if (!status.is_good()) {
    throw std::runtime_error(
        "conv2dBackwardData: create_execution_plans failed: " + std::string(status.get_message()));
  }
  detail::cudnn::restrictToStrictMath(
      *graph, detail::cudnn::isWideFloat(input.type()));
  status = graph->check_support(handle);
  if (!status.is_good()) {
    throw std::runtime_error(
        "conv2dBackwardData: check_support failed: " + std::string(status.get_message()));
  }
  status = graph->build_plans(handle);
  if (!status.is_good()) {
    throw std::runtime_error(
        "conv2dBackwardData: build_plans failed: " + std::string(status.get_message()));
  }

  auto gradInput = makeTensorLike(input, input.shape(), input.type());
  auto wsSize = graph->get_workspace_size();
  Tensor ws;
  if (wsSize > 0) {
    ws = makeTensorLike(
        input, Shape{static_cast<long long>(wsSize)}, fl::dtype::b8);
  }

  auto contiguousWeight = weight.asContiguousTensor();
  DevicePtr dyPtr(gradOutput);
  DevicePtr wPtr(contiguousWeight);
  DevicePtr dxPtr(gradInput);

  std::unordered_map<int64_t, void*> variantPack;
  variantPack[1] = dyPtr.get();
  variantPack[2] = wPtr.get();
  variantPack[3] = dxPtr.get();

  relativeSync(cudnnStream, {gradOutput, contiguousWeight, gradInput});
  if (wsSize > 0) {
    DevicePtr wsPtr(ws);
    relativeSync(cudnnStream, {ws});
    status = graph->execute(handle, variantPack, wsPtr.get());
  } else {
    status = graph->execute(handle, variantPack, nullptr);
  }
  if (!status.is_good()) {
    throw std::runtime_error(
        "conv2dBackwardData: execute failed: " + std::string(status.get_message()));
  }

  relativeSync({gradInput}, cudnnStream);
  return gradInput;
}

std::pair<Tensor, Tensor> conv2dBackwardFilterBias(
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    const ForwardBackwardState& /*state*/) {
  auto dt = flTypeToCudnn(input.type());
  auto cType = computeType(input.type());
  auto handle = getCudnnHandle();
  const auto& cudnnStream = getCudnnStream();

  // --- Filter gradient ---
  {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(dt)
         .set_intermediate_data_type(dt)
         .set_compute_data_type(cType);

    auto xDims = shapeTo4D(input.shape());
    auto dyDims = shapeTo4D(gradOutput.shape());
    auto dwDims = shapeTo4D(weight.shape());

    auto X = graph->tensor(fe::graph::Tensor_attributes()
        .set_dim(xDims).set_stride(contiguousStrides(xDims))
        .set_data_type(dt).set_uid(1));
    auto DY = graph->tensor(fe::graph::Tensor_attributes()
        .set_dim(dyDims).set_stride(contiguousStrides(dyDims))
        .set_data_type(dt).set_uid(2));

    auto wgradAttrs = fe::graph::Conv_wgrad_attributes()
        .set_padding({py, px})
        .set_stride({sy, sx})
        .set_dilation({dy, dx});

    auto DW = graph->conv_wgrad(DY, X, wgradAttrs);
    DW->set_output(true)
        .set_dim(dwDims).set_stride(contiguousStrides(dwDims))
        .set_data_type(dt).set_uid(3);

    auto status = graph->validate();
    if (!status.is_good()) {
      throw std::runtime_error(
          "conv2dBackwardFilter: validate failed: " + std::string(status.get_message()));
    }
    status = graph->build_operation_graph(handle);
    if (!status.is_good()) {
      throw std::runtime_error(
          "conv2dBackwardFilter: build_operation_graph failed: " + std::string(status.get_message()));
    }
    status = graph->create_execution_plans({fe::HeurMode_t::A});
    if (!status.is_good()) {
      throw std::runtime_error(
          "conv2dBackwardFilter: create_execution_plans failed: " + std::string(status.get_message()));
    }
    detail::cudnn::restrictToStrictMath(
        *graph, detail::cudnn::isWideFloat(input.type()));
    status = graph->check_support(handle);
    if (!status.is_good()) {
      throw std::runtime_error(
          "conv2dBackwardFilter: check_support failed: " + std::string(status.get_message()));
    }
    status = graph->build_plans(handle);
    if (!status.is_good()) {
      throw std::runtime_error(
          "conv2dBackwardFilter: build_plans failed: " + std::string(status.get_message()));
    }

    auto gradWeight = makeTensorLike(input, weight.shape(), weight.type());
    auto wsSize = graph->get_workspace_size();
    Tensor ws;
    if (wsSize > 0) {
      ws = makeTensorLike(
          input, Shape{static_cast<long long>(wsSize)}, fl::dtype::b8);
    }

    auto contiguousInput = input.asContiguousTensor();
    DevicePtr xPtr(contiguousInput);
    DevicePtr dyPtr(gradOutput);
    DevicePtr dwPtr(gradWeight);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = xPtr.get();
    variantPack[2] = dyPtr.get();
    variantPack[3] = dwPtr.get();

    relativeSync(cudnnStream, {contiguousInput, gradOutput, gradWeight});
    if (wsSize > 0) {
      DevicePtr wsPtr(ws);
      relativeSync(cudnnStream, {ws});
      status = graph->execute(handle, variantPack, wsPtr.get());
    } else {
      status = graph->execute(handle, variantPack, nullptr);
    }
    if (!status.is_good()) {
      throw std::runtime_error(
          "conv2dBackwardFilter: execute failed: " + std::string(status.get_message()));
    }
    relativeSync({gradWeight}, cudnnStream);

    // --- Bias gradient (legacy API -- simple reduction) ---
    Tensor biasGradOut;
    if (!bias.isEmpty()) {
      auto scalarsType =
          (input.type() == fl::dtype::f16 || input.type() == fl::dtype::bf16)
          ? fl::dtype::f32 : input.type();
      const void* oneg = kOne(scalarsType);
      const void* zerog = kZero(scalarsType);

      biasGradOut = makeTensorLike(input, bias.shape(), bias.type());
      auto oDesc = TensorDescriptor(gradOutput);
      auto bDesc = TensorDescriptor(bias);
      {
        DevicePtr gradBiasPtr(biasGradOut);
        DevicePtr gradResultPtr(gradOutput);
        relativeSync(cudnnStream, {gradOutput, biasGradOut});
        CUDNN_CHECK_ERR(cudnnConvolutionBackwardBias(
            handle,
            oneg,
            oDesc.descriptor,
            gradResultPtr.get(),
            zerog,
            bDesc.descriptor,
            gradBiasPtr.get()));
      }
      relativeSync({biasGradOut}, cudnnStream);
    }

    return {gradWeight, biasGradOut};
  }
}

} // namespace cudnn
} // namespace detail
} // namespace fl

