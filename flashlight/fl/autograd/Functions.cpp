/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/tensor/AutogradOps.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl {
namespace autograd {
namespace detail {

// Accumulates gradients from the 3 RNN outputs (y, hy, cy).
struct RNNGradData {
  fl::Tensor dy;
  fl::Tensor dhy;
  fl::Tensor dcy;
};

bool areDtypesEqual(const Tensor& a, const Tensor& b) {
  return a.type() == b.type();
}

// --- Autograd wrappers called by Tensor-level operators/methods ---
// These keep gradient logic in fl/autograd/ rather than fl/tensor/.
// Inside gradFunc lambdas we use ONLY tensor-level named functions (fl::add,
// fl::mul, fl::negative, fl::detail::sumAs, etc.) to avoid building nested
// autograd graphs.

Tensor autogradIndex(
    const Tensor& input,
    const std::vector<Index>& indices,
    const Tensor& forwardResult) {
  auto inDims = input.shape();
  auto inType = input.type();
  auto gradFunc = [indices, inDims, inType](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    if (!inputs[0].isGradAvailable()) {
      inputs[0].addGrad(Tensor(fl::full(inDims, 0.0, inType), false));
    }
    auto& g = inputs[0].grad();

    // Find Tensor-type indices (advanced indexing). When Tensor indices
    // contain repeated values, g(indices) += gradOutput silently drops
    // duplicates (overwrites instead of accumulating). We handle this by
    // looping over Tensor index elements one at a time so each += targets
    // a unique scalar position.
    std::vector<size_t> tensorPos;
    for (size_t i = 0; i < indices.size(); ++i) {
      if (indices[i].type() == fl::detail::IndexType::Tensor) {
        tensorPos.push_back(i);
      }
    }

    if (tensorPos.empty()) {
      // No Tensor indices — no repeated-index risk
      g(indices) += gradOutput;
    } else {
      // Map each index position to a gradOutput dimension.
      // Literal (scalar) indices squeeze a dimension, so they don't create
      // a dimension in gradOutput.
      std::vector<int> idxToGradDim(indices.size(), -1);
      int gd = 0;
      for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i].type() != fl::detail::IndexType::Literal) {
          idxToGradDim[i] = gd++;
        }
      }

      // Compute Tensor index sizes and total iteration count
      std::vector<Dim> sizes;
      for (auto p : tensorPos) {
        sizes.push_back(indices[p].get<Tensor>().elements());
      }
      Dim total = 1;
      for (auto s : sizes) {
        total *= s;
      }

      // Pre-flatten and cast Tensor indices to s32 for scalar extraction
      std::vector<Tensor> flatIdx;
      for (auto p : tensorPos) {
        flatIdx.push_back(
            indices[p].get<Tensor>().astype(fl::dtype::s32).flatten());
      }

      for (Dim iter = 0; iter < total; ++iter) {
        // Decompose iter into per-Tensor offsets
        std::vector<Dim> offsets(tensorPos.size());
        Dim rem = iter;
        for (int d = static_cast<int>(tensorPos.size()) - 1; d >= 0; --d) {
          offsets[d] = rem % sizes[d];
          rem /= sizes[d];
        }

        // Build gIndices: replace each Tensor index with its scalar value
        std::vector<Index> gIdx(indices.begin(), indices.end());
        for (size_t d = 0; d < tensorPos.size(); ++d) {
          gIdx[tensorPos[d]] = static_cast<Dim>(
              flatIdx[d](offsets[d]).scalarSync<int>());
        }

        // Build goIndices: select the corresponding slice from gradOutput
        std::vector<Index> goIdx(gradOutput.ndim(), fl::span);
        for (size_t d = 0; d < tensorPos.size(); ++d) {
          goIdx[idxToGradDim[tensorPos[d]]] = offsets[d];
        }

        g(gIdx) += gradOutput(goIdx);
      }
    }
  };
  return Tensor(forwardResult, {input.withoutData()}, gradFunc);
}

Tensor autogradFlat(
    const Tensor& input,
    const Index& idx,
    const Tensor& forwardResult) {
  auto inDims = input.shape();
  auto inType = input.type();
  auto gradFunc = [idx, inDims, inType](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    if (!inputs[0].isGradAvailable()) {
      inputs[0].addGrad(Tensor(fl::full(inDims, 0.0, inType), false));
    }
    auto& g = inputs[0].grad();
    g.flat(idx) += gradOutput;
  };
  return Tensor(forwardResult, {input.withoutData()}, gradFunc);
}

Tensor autogradAstype(
    const Tensor& input,
    fl::dtype /*newType*/,
    const Tensor& forwardResult) {
  auto gradFunc = [](std::vector<Tensor>& inputs,
                     const Tensor& gradOutput) {
    auto& inp = inputs[0];
    inp.addGrad(Tensor(gradOutput.astype(inp.type()), false));
  };
  return Tensor(forwardResult, {input.withoutData()}, gradFunc);
}

Tensor autogradAdd(const Tensor& lhs, const Tensor& rhs) {
  auto result = fl::add(lhs, rhs);
  auto lhsShape = lhs.shape();
  auto rhsShape = rhs.shape();
  auto gradFunc = [lhsShape, rhsShape](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    if (inputs[0].requiresGrad()) {
      inputs[0].addGrad(
          Tensor(fl::detail::sumAs(gradOutput, lhsShape), false));
    }
    if (inputs[1].requiresGrad()) {
      inputs[1].addGrad(
          Tensor(fl::detail::sumAs(gradOutput, rhsShape), false));
    }
  };
  return Tensor(
      result, {lhs.withoutData(), rhs.withoutData()}, gradFunc);
}

Tensor autogradAdd(const Tensor& lhs, double rhs) {
  auto result = fl::add(lhs, rhs);
  auto gradFunc = [](std::vector<Tensor>& inputs,
                     const Tensor& gradOutput) {
    inputs[0].addGrad(Tensor(gradOutput, false));
  };
  return Tensor(result, {lhs.withoutData()}, gradFunc);
}

Tensor autogradSub(const Tensor& lhs, const Tensor& rhs) {
  auto result = fl::sub(lhs, rhs);
  auto lhsShape = lhs.shape();
  auto rhsShape = rhs.shape();
  auto gradFunc = [lhsShape, rhsShape](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    if (inputs[0].requiresGrad()) {
      inputs[0].addGrad(
          Tensor(fl::detail::sumAs(gradOutput, lhsShape), false));
    }
    if (inputs[1].requiresGrad()) {
      inputs[1].addGrad(Tensor(
          fl::detail::sumAs(fl::negative(gradOutput), rhsShape), false));
    }
  };
  return Tensor(
      result, {lhs.withoutData(), rhs.withoutData()}, gradFunc);
}

Tensor autogradSub(const Tensor& lhs, double rhs) {
  auto result = fl::sub(lhs, rhs);
  auto gradFunc = [](std::vector<Tensor>& inputs,
                     const Tensor& gradOutput) {
    inputs[0].addGrad(Tensor(gradOutput, false));
  };
  return Tensor(result, {lhs.withoutData()}, gradFunc);
}

Tensor autogradSub(double lhs, const Tensor& rhs) {
  auto result = fl::sub(lhs, rhs);
  auto gradFunc = [](std::vector<Tensor>& inputs,
                     const Tensor& gradOutput) {
    inputs[0].addGrad(Tensor(fl::negative(gradOutput), false));
  };
  return Tensor(result, {rhs.withoutData()}, gradFunc);
}

Tensor autogradMul(const Tensor& lhs, const Tensor& rhs) {
  auto result = fl::mul(lhs, rhs);
  auto lhsShape = lhs.shape();
  auto rhsShape = rhs.shape();
  auto gradFunc = [lhsShape, rhsShape](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    if (inputs[0].requiresGrad()) {
      inputs[0].addGrad(Tensor(
          fl::detail::sumAs(fl::mul(gradOutput, inputs[1]), lhsShape), false));
    }
    if (inputs[1].requiresGrad()) {
      inputs[1].addGrad(Tensor(
          fl::detail::sumAs(fl::mul(gradOutput, inputs[0]), rhsShape), false));
    }
  };
  return Tensor(
      result,
      {rhs.requiresGrad() ? lhs : lhs.withoutData(),
       lhs.requiresGrad() ? rhs : rhs.withoutData()},
      gradFunc);
}

Tensor autogradMul(const Tensor& lhs, double rhs) {
  auto result = fl::mul(lhs, rhs);
  auto gradFunc = [rhs](std::vector<Tensor>& inputs,
                        const Tensor& gradOutput) {
    inputs[0].addGrad(Tensor(fl::mul(gradOutput, rhs), false));
  };
  return Tensor(result, {lhs.withoutData()}, gradFunc);
}

Tensor autogradDiv(const Tensor& lhs, const Tensor& rhs) {
  auto result = fl::div(lhs, rhs);
  auto lhsShape = lhs.shape();
  auto rhsShape = rhs.shape();
  auto gradFunc = [lhsShape, rhsShape](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    // d/dlhs = grad / rhs
    if (inputs[0].requiresGrad()) {
      inputs[0].addGrad(Tensor(
          fl::detail::sumAs(fl::div(gradOutput, inputs[1]), lhsShape), false));
    }
    if (inputs[1].requiresGrad()) {
      // d/drhs = -grad * lhs / rhs^2
      inputs[1].addGrad(Tensor(
          fl::detail::sumAs(
              fl::negative(fl::div(
                  fl::mul(gradOutput, inputs[0]),
                  fl::mul(inputs[1], inputs[1]))),
              rhsShape),
          false));
    }
  };
  return Tensor(
      result,
      {rhs.requiresGrad() ? lhs : lhs.withoutData(), rhs},
      gradFunc);
}

Tensor autogradDiv(const Tensor& lhs, double rhs) {
  auto result = fl::div(lhs, rhs);
  auto gradFunc = [rhs](std::vector<Tensor>& inputs,
                        const Tensor& gradOutput) {
    inputs[0].addGrad(Tensor(fl::div(gradOutput, rhs), false));
  };
  return Tensor(result, {lhs.withoutData()}, gradFunc);
}

Tensor autogradDiv(double lhs, const Tensor& rhs) {
  auto result = fl::div(lhs, rhs);
  auto gradFunc = [lhs](std::vector<Tensor>& inputs,
                        const Tensor& gradOutput) {
    // d/drhs = -lhs / rhs^2 * grad
    inputs[0].addGrad(Tensor(
        fl::negative(fl::div(
            fl::mul(gradOutput, lhs), fl::mul(inputs[0], inputs[0]))),
        false));
  };
  return Tensor(result, {rhs}, gradFunc);
}

} // namespace detail

Tensor maximum(const Tensor& lhs, const Tensor& rhs) {
  FL_DTYPES_MATCH_CHECK(lhs, rhs);
  auto result = fl::maximum(lhs, rhs);
  // Pre-compute mask during forward; re-reading inputs during backward via
  // .detach() can produce incorrect results on JIT backends.
  auto mask = Tensor((lhs > rhs).astype(lhs.type()), false);
  auto gradFunc = [mask](std::vector<Tensor>& inputs,
                     const Tensor& gradOutput) {
    inputs[0].addGrad(Tensor(mask * gradOutput, false));
    inputs[1].addGrad(Tensor(!mask * gradOutput, false));
  };
  return Tensor(result, {lhs.withoutData(), rhs.withoutData()}, gradFunc);
}

Tensor maximum(const Tensor& lhs, const double& rhsVal) {
  auto result = fl::maximum(lhs, rhsVal).astype(lhs.type());
  auto gradFunc =
      [rhsVal](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
        auto mask = (inputs[0] > rhsVal).astype(gradOutput.type());
        inputs[0].addGrad(Tensor(mask * gradOutput, false));
      };
  return Tensor(result, {lhs}, gradFunc);
}

Tensor maximum(const double& lhsVal, const Tensor& rhs) {
  return fl::autograd::maximum(rhs, lhsVal);
}

Tensor minimum(const Tensor& lhs, const Tensor& rhs) {
  FL_DTYPES_MATCH_CHECK(lhs, rhs);
  auto result = fl::minimum(lhs, rhs);
  auto mask = Tensor((lhs < rhs).astype(lhs.type()), false);
  auto gradFunc = [mask](std::vector<Tensor>& inputs,
                     const Tensor& gradOutput) {
    inputs[0].addGrad(Tensor(mask * gradOutput, false));
    inputs[1].addGrad(Tensor(!mask * gradOutput, false));
  };
  return Tensor(result, {lhs.withoutData(), rhs.withoutData()}, gradFunc);
}

Tensor minimum(const Tensor& lhs, const double& rhsVal) {
  auto result = fl::minimum(lhs, rhsVal).astype(lhs.type());
  auto gradFunc =
      [rhsVal](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
        auto mask = (inputs[0] < rhsVal).astype(gradOutput.type());
        inputs[0].addGrad(Tensor(mask * gradOutput, false));
      };
  return Tensor(result, {lhs}, gradFunc);
}

Tensor minimum(const double& lhsVal, const Tensor& rhs) {
  return fl::autograd::minimum(rhs, lhsVal);
}

Tensor negative(const Tensor& input) {
  auto result = (0.0 - input.detach()).astype(input.type());
  auto gradFunc = [](std::vector<Tensor>& inputs,
                     const Tensor& gradOutput) {
    inputs[0].addGrad(Tensor((0.0 - gradOutput), false));
  };
  return Tensor(result, {input.withoutData()}, gradFunc);
}

Tensor reciprocal(const Tensor& input) {
  auto result = 1.0 / input.detach();
  auto gradFunc = [](std::vector<Tensor>& inputs,
                     const Tensor& gradOutput) {
    auto res = 1.0 / inputs[0].detach();
    inputs[0].addGrad(
        Tensor(((0.0 - gradOutput) * res * res), false));
  };
  return Tensor(result, {input}, gradFunc);
}

Tensor exp(const Tensor& input) {
  auto result = fl::exp(input);
  auto gradFunc = [](std::vector<Tensor>& inputs,
                     const Tensor& gradOutput) {
    inputs[0].addGrad(
        Tensor(gradOutput * fl::exp(inputs[0]), false));
  };
  return Tensor(result, {input}, gradFunc);
}

Tensor log(const Tensor& input) {
  auto result = fl::log(input);
  auto gradFunc = [](std::vector<Tensor>& inputs,
                     const Tensor& gradOutput) {
    inputs[0].addGrad(
        Tensor((gradOutput / inputs[0].detach()), false));
  };
  return Tensor(result, {input}, gradFunc);
}

Tensor log1p(const Tensor& input) {
  auto result = fl::log1p(input);
  auto gradFunc = [](std::vector<Tensor>& inputs,
                     const Tensor& gradOutput) {
    inputs[0].addGrad(
        Tensor((gradOutput / (1.0 + inputs[0].detach())), false));
  };
  return Tensor(result, {input}, gradFunc);
}

Tensor power(const Tensor& input, double p) {
  auto result = fl::power(input, p);
  auto gradFunc = [p](std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    Tensor grad =
        p * fl::power(inputs[0], p - 1) * gradOutput;
    inputs[0].addGrad(Tensor(grad, false));
  };
  return Tensor(result, {input}, gradFunc);
}

Tensor power(const Tensor& base, const Tensor& exp) {
  FL_DTYPES_MATCH_CHECK(base, exp);
  auto result = fl::power(base, exp);
  auto gradFunc = [](std::vector<Tensor>& inputs,
                     const Tensor& gradOutput) {
    auto& b = inputs[0];
    auto& e = inputs[1];
    if (b.requiresGrad()) {
      // d/db (b^e) = e * b^(e-1)
      auto grad = e.detach() * fl::power(b, e.detach() - 1) *
          gradOutput;
      b.addGrad(Tensor(grad, false));
    }
    if (e.requiresGrad()) {
      // d/de (b^e) = b^e * ln(b)
      auto grad =
          fl::power(b, e) * fl::log(b) *
          gradOutput;
      e.addGrad(Tensor(grad, false));
    }
  };
  return Tensor(result, {base, exp}, gradFunc);
}

Tensor power(const double& base, const Tensor& exp) {
  auto result = fl::power(base, exp);
  auto gradFunc = [base](std::vector<Tensor>& inputs,
                         const Tensor& gradOutput) {
    // d/de (c^e) = c^e * ln(c)
    auto grad = fl::power(base, inputs[0]) * std::log(base) *
        gradOutput;
    inputs[0].addGrad(Tensor(grad, false));
  };
  return Tensor(result, {exp}, gradFunc);
}

Tensor sin(const Tensor& input) {
  auto result = fl::sin(input);
  auto gradFunc = [](std::vector<Tensor>& inputs,
                     const Tensor& gradOutput) {
    inputs[0].addGrad(
        Tensor((gradOutput * fl::cos(inputs[0])), false));
  };
  return Tensor(result, {input}, gradFunc);
}

Tensor cos(const Tensor& input) {
  auto result = fl::cos(input);
  auto gradFunc = [](std::vector<Tensor>& inputs,
                     const Tensor& gradOutput) {
    inputs[0].addGrad(Tensor(
        (gradOutput * (0.0 - fl::sin(inputs[0]))), false));
  };
  return Tensor(result, {input}, gradFunc);
}

Tensor tanh(const Tensor& input) {
  auto result = fl::tanh(input);
  auto gradFunc =
      [result](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
        auto grad =
            Tensor((1.0 - result * result) * gradOutput, false);
        inputs[0].addGrad(Tensor(grad, false));
      };
  return Tensor(result, {input.withoutData()}, gradFunc);
}

Tensor clip(const Tensor& input, const double lo, const double hi) {
  auto result = fl::clip(input, lo, hi);
  auto gradFunc = [lo, hi, result](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    Tensor gradMask = gradOutput;
    gradMask = fl::where((result > lo) && (result < hi), gradMask, 0);
    inputs[0].addGrad(Tensor(gradMask, false));
  };
  return Tensor(result, {input.withoutData()}, gradFunc);
}

Tensor sqrt(const Tensor& input) {
  auto result = fl::sqrt(input);
  auto gradFunc = [result](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    auto output = Tensor(result, false);
    inputs[0].addGrad(Tensor(gradOutput / (2 * output), false));
  };
  return Tensor(result, {input.withoutData()}, gradFunc);
}

Tensor sigmoid(const Tensor& input) {
  auto result = fl::sigmoid(input);
  auto gradFunc =
      [result](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
        auto grad = gradOutput * result * (1 - result);
        inputs[0].addGrad(Tensor(grad, false));
      };
  return Tensor(result, {input.withoutData()}, gradFunc);
}

Tensor swish(const Tensor& input, double beta) {
  return input * fl::autograd::sigmoid(beta * input);
}

Tensor erf(const Tensor& input) {
  auto result = fl::erf(input);
  auto gradFunc = [](std::vector<Tensor>& inputs,
                     const Tensor& gradOutput) {
    auto x = inputs[0].detach();
    auto grad = gradOutput * 2 / std::sqrt(M_PI) * fl::exp(-(x * x));
    inputs[0].addGrad(Tensor(grad, false));
  };
  return Tensor(result, {input}, gradFunc);
}

Tensor transpose(const Tensor& input, const Shape& dims /* = {} */) {
  auto result = fl::transpose(input, dims);
  auto gradFunc = [inputDims = input.shape(), ndim = input.ndim(), dims](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    Shape reverseShape = dims;

    if (dims.ndim()) {
      // Reverse vec if transposing all dims (empty arg)
      auto dVec = dims.get();
      std::reverse(dVec.begin(), dVec.end());
      reverseShape = Shape(dVec);
    }

    for (unsigned i = 0; i < reverseShape.ndim(); ++i) {
      reverseShape[dims[i]] = i;
    }

    inputs[0].addGrad(
        Tensor(fl::transpose(gradOutput, reverseShape), false));
  };
  return Tensor(result, {input.withoutData()}, gradFunc);
}

Tensor tileAs(const Tensor& input, const Shape& rdims) {
  auto result = fl::detail::tileAs(input, rdims);

  Shape inDims = input.shape();
  auto gradFunc = [inDims](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    inputs[0].addGrad(Tensor(
        fl::detail::sumAs(gradOutput, inDims).astype(inputs[0].type()), false));
  };
  return Tensor(result, {input.withoutData()}, gradFunc);
}

Tensor tileAs(const Tensor& input, const Tensor& reference) {
  return fl::autograd::tileAs(input, reference.shape());
}

Tensor sumAs(const Tensor& input, const Shape& rdims) {
  auto result = fl::detail::sumAs(input, rdims);
  auto idims = input.shape();
  auto gradFunc =
      [idims](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
        inputs[0].addGrad(Tensor(fl::detail::tileAs(gradOutput, idims), false));
      };
  return Tensor(result, {input.withoutData()}, gradFunc);
}

Tensor sumAs(const Tensor& input, const Tensor& reference) {
  return fl::autograd::sumAs(input, reference.shape());
}

Tensor concatenate(const std::vector<Tensor>& concatInputs, unsigned axis) {
  if (concatInputs.empty()) {
    throw std::invalid_argument("cannot concatenate zero variables");
  }

  if (concatInputs.size() == 1) {
    return concatInputs[0];
  }
  // All Variables must be of the same type
  fl::dtype type = concatInputs[0].type();
  for (auto& var : concatInputs) {
    if (var.type() != type) {
      throw std::invalid_argument(
          "concatenate: all input Variables must be of the same type");
    }
  }
  // All Variables must have the same number of dims
  unsigned numDims = concatInputs[0].ndim();
  for (auto& var : concatInputs) {
    if (numDims != var.ndim()) {
      throw std::invalid_argument(
          "concatenate: all input Variables must "
          "have the same number of dimensions");
    }
  }

  // All Variables must have the same size when indexed along the axis not being
  // concatenated along
  auto dims = concatInputs[0].shape();
  int concatSize = dims[axis];
  for (size_t i = 1; i < concatInputs.size(); i++) {
    concatSize += concatInputs[i].dim(axis);
    for (unsigned d = 0; d < numDims; d++) {
      if (axis != d && concatInputs[i].dim(d) != dims[d]) {
        throw std::invalid_argument(
            "mismatch in dimension not being concatenated");
      }
    }
  }
  dims[axis] = concatSize;
  Tensor result(dims, concatInputs[0].type());
  std::vector<fl::Index> slice(numDims, fl::span);
  int start = 0;
  for (const auto& input : concatInputs) {
    slice[axis] = fl::range({start, start + input.dim(axis)});
    result(slice) = input.detach();
    start += input.dim(axis);
  }

  std::vector<Tensor> inputsNoData;
  std::vector<Shape> inDims;

  for (const auto& in : concatInputs) {
    inputsNoData.push_back(in.withoutData());
    inDims.push_back(in.shape());
  }

  auto gradFunc = [axis, inDims, numDims](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    std::vector<fl::Index> sx(numDims, fl::span);
    int s = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      sx[axis] = fl::range(s, s + inDims[i][axis]);
      inputs[i].addGrad(Tensor(gradOutput(sx), false));
      s += inDims[i][axis];
    }
  };

  return Tensor(result, inputsNoData, gradFunc);
}

std::vector<Tensor> split(const Tensor& input, long splitSize, int dim) {
  if (splitSize <= 0) {
    throw std::invalid_argument("split size must be a positive integer");
  }
  auto dimSize = input.dim(dim);
  std::vector<long> splitSizes(dimSize / splitSize, splitSize);

  if (dimSize % splitSize > 0) {
    splitSizes.push_back(dimSize % splitSize);
  }
  return fl::autograd::split(input, splitSizes, dim);
}

std::vector<Tensor>
split(const Tensor& input, const std::vector<long>& splitSizes, int dim) {
  if (dim >= input.ndim()) {
    throw std::invalid_argument(
        "split: passed dim is larger than the number of dimensions "
        "of the input.");
  }
  auto dimSize = input.dim(dim);
  auto N = splitSizes.size();

  std::vector<Tensor> outputs(N);
  std::vector<fl::Index> sel(input.ndim(), fl::span);
  int start = 0;
  for (int i = 0; i < N; ++i) {
    if (splitSizes[i] <= 0) {
      throw std::invalid_argument("elements in split sizes has to be positive");
    }
    int end = start + splitSizes[i];
    sel[dim] = fl::range(start, end);
    outputs[i] = input(sel);
    start = end;
  }
  if (start != dimSize) {
    throw std::invalid_argument("sum of split sizes must match split dim");
  }
  return outputs;
}

Tensor tile(const Tensor& input, const Shape& dims) {
  Tensor result = fl::tile(input, dims);
  Shape idims = input.shape();
  auto gradFunc =
      [idims](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
        inputs[0].addGrad(Tensor(
            fl::detail::sumAs(gradOutput, idims).astype(inputs[0].type()), false));
      };
  return Tensor(result, {input.withoutData()}, gradFunc);
}

Tensor sum(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims /* = false*/) {
  auto result = fl::sum(input, axes, keepDims);

  Shape indims = input.shape();
  auto gradFunc = [indims, axes, keepDims](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    inputs[0].addGrad(Tensor(
        fl::detail::tileAs(
            fl::detail::expandFromReduction(gradOutput, axes, keepDims),
            indims),
        false));
  };
  return Tensor(result.astype(input.type()), {input.withoutData()}, gradFunc);
}

Tensor mean(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims /* = false*/) {
  auto result = fl::mean(input, axes, keepDims);

  Shape idims = input.shape();
  auto gradFunc = [idims, axes, keepDims](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    Dim count = 1;
    for (auto ax : axes) {
      if (ax < idims.ndim()) {
        count *= idims[ax];
      }
    }
    inputs[0].addGrad(Tensor(
        fl::detail::tileAs(
            fl::detail::expandFromReduction(gradOutput, axes, keepDims),
            idims) /
            count,
        false));
  };

  return Tensor(result, {input.withoutData()}, gradFunc);
}

Tensor var(
    const Tensor& in,
    const std::vector<int>& axes,
    const bool bias /* = false */,
    bool keepDims /* = false*/) {
  const auto& input = in.detach();
  auto result = fl::sum(input * input, axes, keepDims);

  auto avg = fl::mean(input, axes, keepDims);
  auto n = 1;
  for (auto ax : axes) {
    n *= input.dim(ax);
  }
  if (bias && n == 1) {
    throw std::invalid_argument(
        "cannot compute sample variance with only one sample");
  }
  auto val = 1.0 / (bias ? n - 1 : n);
  result = val * (result - n * avg * avg);

  auto gradFunc =
      [val, axes](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
        Shape expandedDims = inputs[0].shape();
        Shape tileDims = inputs[0].shape();
        for (auto ax : axes) {
          tileDims[ax] = inputs[0].dim(ax);
          expandedDims[ax] = 1;
        }

        inputs[0].addGrad(Tensor(
            ((2 * val * fl::detail::tileAs(fl::reshape(gradOutput, expandedDims), tileDims)) *
             (inputs[0].detach() -
              fl::detail::tileAs(fl::reshape(fl::mean(inputs[0], axes), expandedDims), tileDims))),
            false));
      };
  return Tensor(result, {in}, gradFunc);
}

Tensor std(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool bias /* = false */,
    bool keepDims /* = false*/) {
  // std = sqrt(var)
  return fl::autograd::sqrt(fl::autograd::var(input, axes, bias, keepDims));
}

Tensor norm(
    const Tensor& input,
    const std::vector<int>& axes,
    double p /* = 2 */,
    bool keepDims /* = false */) {
  if (p <= 0) {
    throw std::out_of_range("Lp norm: p must be > 0");
  }
  auto result = fl::power(fl::abs(input), p);
  result = fl::sum(result, axes, /* keepDims = */ keepDims);

  Tensor sumap = fl::detail::expandFromReduction(result, axes, keepDims);
  result = fl::power(result, 1 / p);
  fl::eval(result);

  auto gradFunc = [sumap, p, axes, keepDims](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    // correct, but less precise: auto gvar = Tensor(fl::power(result, p - 1),
    // false);
    auto gvar = Tensor(fl::power(sumap, 1 - 1 / p), false);
    auto normGrad =
        (inputs[0].detach() * fl::power(fl::abs(inputs[0]), p - 2) *
         fl::detail::tileAs(
             fl::detail::expandFromReduction(gradOutput, axes, keepDims) /
                 gvar,
             inputs[0].shape()));
    inputs[0].addGrad(Tensor(normGrad, false));
  };
  return Tensor(result, {input}, gradFunc);
}

Tensor normalize(
    const Tensor& in,
    const std::vector<int>& axes,
    double p /* = 2 */,
    double eps /* = 1e-12 */) {
  Tensor normVal = fl::autograd::norm(in, axes, p, /* keepDims = */ true);
  Tensor invscale = fl::autograd::maximum(normVal, eps);
  return in / fl::autograd::tileAs(invscale, in);
}

Tensor matmul(
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp /* = MatrixProperty::None */,
    MatrixProperty rhsProp /* = MatrixProperty::None */) {
  FL_DTYPES_MATCH_CHECK(lhs, rhs);
  auto result = fl::matmul(lhs, rhs, lhsProp, rhsProp);

  bool isNN = lhsProp == MatrixProperty::None && rhsProp == MatrixProperty::None;
  auto gradFunc = [lhsProp, rhsProp, isNN](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    if (lhsProp == MatrixProperty::None && rhsProp == MatrixProperty::None) {
      // matmul(A, B): dA = grad @ B^T, dB = A^T @ grad
      if (inputs[0].requiresGrad()) {
        Tensor _lhs = gradOutput;
        if (_lhs.ndim() == 1) {
          _lhs = fl::unsqueeze(_lhs, 0);
        }
        Tensor _rhs = inputs[1].detach();
        if (_rhs.ndim() == 1) {
          _rhs = fl::unsqueeze(_rhs, 1);
        }
        auto val = fl::matmul(_lhs, _rhs, MatrixProperty::None, MatrixProperty::Transpose);
        inputs[0].addGrad(Tensor(fl::detail::sumAs(val, inputs[0].shape()), false));
      }
      if (inputs[1].requiresGrad()) {
        Tensor _lhs = inputs[0].detach();
        if (_lhs.ndim() == 1) {
          _lhs = fl::unsqueeze(_lhs, 0);
        }
        Tensor _rhs = gradOutput;
        if (_rhs.ndim() == 1) {
          _rhs = fl::unsqueeze(_rhs, 1);
        }
        auto val = fl::matmul(_lhs, _rhs, MatrixProperty::Transpose);
        inputs[1].addGrad(Tensor(fl::detail::sumAs(val, inputs[1].shape()), false));
      }
    } else if (lhsProp == MatrixProperty::Transpose && rhsProp == MatrixProperty::None) {
      // matmul(A^T, B): dA = B @ grad^T, dB = A @ grad
      if (inputs[0].requiresGrad()) {
        auto val = fl::matmul(
            inputs[1], gradOutput,
            MatrixProperty::None, MatrixProperty::Transpose);
        inputs[0].addGrad(Tensor(fl::detail::sumAs(val, inputs[0].shape()), false));
      }
      if (inputs[1].requiresGrad()) {
        auto val = fl::matmul(inputs[0], gradOutput);
        inputs[1].addGrad(Tensor(fl::detail::sumAs(val, inputs[1].shape()), false));
      }
    } else if (lhsProp == MatrixProperty::None && rhsProp == MatrixProperty::Transpose) {
      // matmul(A, B^T): dA = grad @ B, dB = grad^T @ A
      if (inputs[0].requiresGrad()) {
        auto val = fl::matmul(gradOutput, inputs[1]);
        inputs[0].addGrad(Tensor(fl::detail::sumAs(val, inputs[0].shape()), false));
      }
      if (inputs[1].requiresGrad()) {
        auto val = fl::matmul(
            gradOutput, inputs[0],
            MatrixProperty::Transpose);
        inputs[1].addGrad(Tensor(fl::detail::sumAs(val, inputs[1].shape()), false));
      }
    } else {
      throw std::invalid_argument(
          "matmul: Transpose+Transpose not supported for autograd");
    }
  };
  return Tensor(result, {lhs, rhs}, gradFunc);
}

Tensor abs(const Tensor& input) {
  auto result = fl::abs(input);
  auto gradFunc = [](std::vector<Tensor>& inputs,
                     const Tensor& gradOutput) {
    // Convert it into -1, 0, 1
    auto sign = fl::sign(inputs[0]);
    inputs[0].addGrad(Tensor((sign * gradOutput), false));
  };
  return Tensor(result, {input}, gradFunc);
}

Tensor where(
    const Tensor& condition,
    const Tensor& x,
    const Tensor& y) {
  auto result = fl::where(condition, x, y);
  auto condTensor = condition.detach();
  auto gradFunc = [condTensor](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    if (inputs[1].requiresGrad()) {
      inputs[1].addGrad(Tensor(
          fl::where(condTensor, gradOutput, 0), false));
    }
    if (inputs[2].requiresGrad()) {
      inputs[2].addGrad(Tensor(
          fl::where(condTensor, 0, gradOutput), false));
    }
  };
  return Tensor(
      result, {condition.withoutData(), x.withoutData(), y.withoutData()},
      gradFunc);
}

Tensor flip(const Tensor& input, const unsigned dim) {
  auto result = fl::flip(input, dim);
  auto gradFunc = [dim](std::vector<Tensor>& inputs,
                        const Tensor& gradOutput) {
    inputs[0].addGrad(Tensor(fl::flip(gradOutput, dim), false));
  };
  return Tensor(result, {input.withoutData()}, gradFunc);
}

Tensor tril(const Tensor& input, int diagonal) {
  auto result = fl::tril(input, diagonal);
  auto gradFunc = [diagonal](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    inputs[0].addGrad(Tensor(fl::tril(gradOutput, diagonal), false));
  };
  return Tensor(result, {input.withoutData()}, gradFunc);
}

Tensor triu(const Tensor& input, int diagonal) {
  auto result = fl::triu(input, diagonal);
  auto gradFunc = [diagonal](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    inputs[0].addGrad(Tensor(fl::triu(gradOutput, diagonal), false));
  };
  return Tensor(result, {input.withoutData()}, gradFunc);
}

Tensor cumsum(const Tensor& input, const unsigned axis) {
  auto result = fl::cumsum(input, axis);
  auto gradFunc = [axis](std::vector<Tensor>& inputs,
                         const Tensor& gradOutput) {
    // Gradient of cumsum is reverse cumsum: flip, cumsum, flip
    auto grad = fl::flip(
        fl::cumsum(fl::flip(gradOutput, axis), axis), axis);
    inputs[0].addGrad(Tensor(grad, false));
  };
  return Tensor(result, {input.withoutData()}, gradFunc);
}

Tensor roll(const Tensor& input, int shift, unsigned axis) {
  auto result = fl::roll(input, shift, axis);
  auto gradFunc = [shift, axis](std::vector<Tensor>& inputs,
                                const Tensor& gradOutput) {
    inputs[0].addGrad(
        Tensor(fl::roll(gradOutput, -shift, axis), false));
  };
  return Tensor(result, {input.withoutData()}, gradFunc);
}

Tensor floor(const Tensor& input) {
  return Tensor(fl::floor(input), false);
}

Tensor ceil(const Tensor& input) {
  return Tensor(fl::ceil(input), false);
}

Tensor rint(const Tensor& input) {
  return Tensor(fl::rint(input), false);
}

Tensor sign(const Tensor& input) {
  return Tensor(fl::sign(input), false);
}

Tensor isnan(const Tensor& input) {
  return Tensor(fl::isnan(input), false);
}

Tensor isinf(const Tensor& input) {
  return Tensor(fl::isinf(input), false);
}

Tensor flatten(const Tensor& input) {
  auto result = input.flatten();
  Shape idims = input.shape();
  auto gradFunc =
      [idims](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
        inputs[0].addGrad(Tensor(fl::reshape(gradOutput, idims), false));
      };
  return Tensor(result, {input.withoutData()}, gradFunc);
}

Tensor reshape(const Tensor& input, const Shape& dims) {
  if (input.ndim() == 0) {
    return input;
  }
  Shape inferDims = dims;
  unsigned maxNDims =
      std::max(static_cast<unsigned>(input.ndim()), static_cast<unsigned>(dims.ndim()));

  // Check for inferred dims that are beyond the input's number of dims
  for (int i = 0; i < maxNDims; ++i) {
    if (i >= input.ndim() && inferDims[i] == 0) {
      throw std::invalid_argument(
          "reshape: tried to infer dimension " + std::to_string(i) +
          " which exceeds the number of dimensions of the input.");
    }
  }

  // Infer any 0 dim
  for (int i = 0; i < maxNDims; ++i) {
    if (i < inferDims.ndim() && inferDims[i] == 0) {
      inferDims[i] = input.dim(i);
    }
  }

  // Infer any -1 dim
  int nInfer = 0;
  for (int i = 0; i < maxNDims; i++) {
    if (i < inferDims.ndim() && inferDims[i] == -1) {
      nInfer++;
      // Cast to signed Dim to avoid unsigned/signed division mismatch
      inferDims[i] = -(static_cast<Dim>(input.elements()) / inferDims.elements());
    }
  }

  if (nInfer > 1) {
    throw std::invalid_argument("reshape: too many dimensions infer");
  }

  if (static_cast<size_t>(inferDims.elements()) != input.elements()) {
    throw std::invalid_argument("reshape: mismatched # of elements");
  }

  auto result = fl::reshape(input, inferDims);

  Shape inDims = input.shape();
  auto gradFunc = [inDims](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    inputs[0].addGrad(Tensor(fl::reshape(gradOutput, inDims), false));
  };
  return Tensor(result, {input.withoutData()}, gradFunc);
}

Tensor squeeze(const Tensor& input) {
  return fl::autograd::reshape(input, fl::squeeze(input).shape());
}

Tensor squeeze(const Tensor& input, const unsigned axis) {
  return fl::autograd::reshape(input, fl::squeeze(input, axis).shape());
}

Tensor unsqueeze(const Tensor& input, const unsigned axis) {
  return fl::autograd::reshape(input, fl::unsqueeze(input, axis).shape());
}

Tensor softmax(const Tensor& input, const int dim) {
  auto maxvals = fl::amax(input, {dim}, /* keepDims = */ true);
  Shape tiledims(std::vector<Dim>(input.ndim(), 1));
  tiledims[dim] = input.dim(dim);

  auto expInput = fl::exp(input.detach() - fl::tile(maxvals, tiledims));
  auto result = expInput /
      fl::tile(fl::sum(expInput, {dim}, /* keepDims = */ true), tiledims);

  fl::eval(result);
  auto gradFunc = [dim, tiledims, result](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    auto rbyg = gradOutput * result;
    auto gradSm = rbyg -
        result *
            fl::tile(fl::sum(rbyg, {dim}, /* keepDims = */ true), tiledims);
    inputs[0].addGrad(
        Tensor(gradSm.astype(inputs[0].type()), false));
  };
  return Tensor(result, {input.withoutData()}, gradFunc);
}

Tensor logSoftmax(const Tensor& input, const int dim) {
  auto maxvals = fl::amax(input, {dim}, /* keepDims = */ true);
  Shape tiledims(std::vector<Dim>(input.ndim(), 1));
  tiledims[dim] = input.dim(dim);
  auto result = input.detach() -
      fl::tile(fl::log(fl::sum(
                   fl::exp(input.detach() - fl::tile(maxvals, tiledims)),
                   {dim},
                   /* keepDims = */ true)) +
                   maxvals,
               tiledims);

  fl::eval(result);
  auto gradFunc = [dim, tiledims, result](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    auto gradLsm = gradOutput -
        fl::exp(result) *
            fl::tile(
                fl::sum(gradOutput, {dim}, /* keepDims = */ true),
                tiledims);
    inputs[0].addGrad(
        Tensor(gradLsm.astype(inputs[0].type()), false));
  };
  return Tensor(result, {input.withoutData()}, gradFunc);
}

Tensor binaryCrossEntropy(const Tensor& inputs, const Tensor& targets) {
  auto targetsTyped = targets.astype(inputs.type());
  return fl::autograd::negative(
      targetsTyped * fl::autograd::log(inputs) + (1 - targetsTyped) * fl::autograd::log(1 - inputs));
}

Tensor categoricalCrossEntropy(
    const Tensor& input,
    const Tensor& targets,
    ReduceMode reduction /* =ReduceMode::MEAN */,
    int ignoreIndex /* = -1 */) {
  // Row-major: input -- [X1, X2, X3, C], target -- [X1, X2, X3]
  if (input.ndim() != targets.ndim() + 1) {
    throw std::invalid_argument(
        "dimension mismatch in categorical cross entropy: "
        "target must have one fewer dimension than input");
  }
  for (int i = 0; i < input.ndim() - 1; i++) {
    if (input.dim(i) != targets.dim(i)) {
      throw std::invalid_argument(
          "dimension mismatch in categorical cross entropy");
    }
  }

  int C = input.dim(input.ndim() - 1);
  int X = targets.elements();
  if (fl::any(
          ((targets.detach() < 0) || (targets.detach() >= C)) &&
          (targets.detach() != ignoreIndex))
          .scalarSync<char>()) {
    throw std::invalid_argument(
        "target contains elements out of valid range [0, num_categories) "
        "in categorical cross entropy");
  }

  auto x = fl::reshape(input, Shape({X, C}));
  auto y = fl::reshape(targets, Shape({X, 1}));

  auto A = fl::arange(Shape({X, C}), 1);
  auto B = fl::tile(y, Shape({1, C})).astype(A.type());
  // Cast comparison result to input type to avoid mixed-type binary ops
  // (OneDNN binary primitives require same-type sources)
  auto eqMask = (A == B).astype(x.type()); // [X, C], 0.0 or 1.0

  auto result = eqMask * x; // picks logprob of correct class

  auto ignoreMask = (y == ignoreIndex).flatten(); // [X]
  result = -(fl::sum(result, {1}).flatten()); // [X], negate for cross entropy
  result(ignoreMask) = 0.;

  Tensor denominator;
  if (reduction == ReduceMode::NONE) {
    result = fl::reshape(result, targets.shape()); // [X1 X2 X3]
  } else if (reduction == ReduceMode::MEAN) {
    denominator = fl::sum((!ignoreMask).astype(fl::dtype::s32), {0});
    // Clamp denominator to at least 1 to avoid division by zero
    // when all targets match ignoreIndex
    denominator = fl::maximum(denominator, fl::full({1}, 1, fl::dtype::s32));
    result = fl::sum(result, {0}) / denominator; // [1]
  } else if (reduction == ReduceMode::SUM) {
    result = fl::sum(result, {0}); // [1]
  } else {
    throw std::invalid_argument(
        "unknown reduction method for categorical cross entropy");
  }

  auto inputDims = input.shape();
  auto gradFunc = [C, X, eqMask, ignoreMask, denominator, reduction, inputDims](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    Tensor grad = gradOutput;
    if (reduction == ReduceMode::NONE) {
      grad = fl::reshape(grad, {X});
    } else if (reduction == ReduceMode::MEAN) {
      grad = fl::tile(grad / denominator, {X});
    } else if (reduction == ReduceMode::SUM) {
      grad = fl::tile(grad, {X});
    }
    // [X]
    grad(ignoreMask) = 0.;
    grad = fl::unsqueeze(grad, 1);
    // eqMask is 1 at correct class, 0 elsewhere; negate for cross entropy grad
    grad = fl::tile(grad, {1, C}) * (-eqMask);
    inputs[0].addGrad(Tensor(fl::reshape(grad, inputDims), false));
  };

  return Tensor(result, {input.withoutData(), targets}, gradFunc);
}

Tensor weightedCategoricalCrossEntropy(
    const Tensor& input,
    const Tensor& targets,
    const Tensor& weight,
    int ignoreIndex /* = -1 */) {
  // Row-major: input -- [X1, X2, X3, C], target -- [X1, X2, X3]
  if (input.ndim() < targets.ndim() - 1) {
    throw std::invalid_argument(
        "weightedCategoricalCrossEntropy: input must have one more than the "
        "number of target dimensions minus 1");
  }

  for (int i = 0; i < targets.ndim() - 2; i++) {
    if (input.dim(i) != targets.dim(i)) {
      throw std::invalid_argument(
          "weightedCategoricalCrossEntropy: dimension mismatch in categorical cross entropy");
    }
  }
  int C = input.dim(input.ndim() - 1);
  if (weight.dim(0) != C) {
    throw std::invalid_argument(
        "weightedCategoricalCrossEntropy: dimension mismatch in categorical cross entropy");
  }

  int X = targets.elements();
  if (fl::any((targets.detach() < 0) || (targets.detach() >= C))
          .scalarSync<char>()) {
    throw std::invalid_argument(
        "weightedCategoricalCrossEntropy: target contains elements out of valid range "
        "[0, num_categories) in categorical cross entropy");
  }

  auto x = fl::reshape(input, {X, C});
  auto y = fl::reshape(targets, {X, 1});

  auto A = fl::arange({X, C}, 1);
  auto B = fl::tile(y, {1, C});
  auto mask = -(A == B); // [X, C]

  auto weightSum = (-mask) * fl::tile(weight, {X, 1});
  Tensor denominator = {fl::sum(weightSum, {0, 1}), false};

  auto result = mask * x;
  result = result * weight.detach();

  auto ignoreMask = (y != ignoreIndex).astype(fl::dtype::s32); // [X, 1]
  result = ignoreMask * fl::sum(result, {1}, /* keepDims = */ true); // [X, 1]
  result = fl::sum(result, {0}, /* keepDims = */ true) / denominator.detach();

  auto inputDims = input.shape();
  auto gradFunc = [C, X, mask, ignoreMask, denominator, inputDims](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    auto grad = gradOutput;
    grad = fl::tile(grad / denominator.detach(), {X, 1});

    auto weightTensor = inputs[2].detach();
    grad *= ignoreMask;
    grad = fl::tile(grad, {1, C}) * mask;
    grad = fl::reshape(grad, inputDims);
    grad = grad * weightTensor;
    inputs[0].addGrad(Tensor(fl::reshape(grad, inputDims), false));
  };

  return Tensor(result, {input.withoutData(), targets, weight}, gradFunc);
}

Tensor linear(const Tensor& input, const Tensor& weight) {
  auto dummyBias = Tensor(Tensor().astype(input.type()), false);
  return fl::autograd::linear(input, weight, dummyBias);
}

Tensor linear(const Tensor& in, const Tensor& wt, const Tensor& bs) {
  FL_DTYPES_MATCH_CHECK(in, wt, bs);
  // In row-major convention, the last dim is the features/input dimension
  int lastDim = in.ndim() - 1;
  auto nIn = in.dim(lastDim);
  Dim nframes = static_cast<Dim>(in.elements()) / nIn;
  Shape to2d({nframes, nIn});
  auto toNd = in.shape();
  toNd[lastDim] = wt.dim(0);

  // input_2d: {nframes, nIn}, weight: {nOut, nIn}
  // output = input_2d @ weight^T = {nframes, nOut}
  auto output =
      fl::reshape(fl::matmul(fl::reshape(in, to2d), wt,
                         MatrixProperty::None, MatrixProperty::Transpose), toNd);

  auto hasBias = bs.elements() > 0;
  if (hasBias) {
    output = output + bs;
  }

  auto gradFunc = [hasBias](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    auto& in = inputs[0];
    auto& wt = inputs[1];
    Tensor wtTensor = wt.detach();

    int lastDim = in.ndim() - 1;
    auto nIn = in.dim(lastDim);
    Dim nframes = static_cast<Dim>(in.elements()) / nIn;

    if (hasBias && inputs[2].requiresGrad()) {
      auto& bs = inputs[2];
      auto biasGrad = fl::detail::sumAs(gradOutput, bs.shape());
      bs.addGrad(Tensor(biasGrad, false));
    }
    if (in.requiresGrad()) {
      // gradOutput_2d: {nframes, nOut}, weight: {nOut, nIn}
      // inGrad = gradOutput_2d @ weight = {nframes, nIn}
      Shape to2dout({nframes, wtTensor.dim(0)});
      auto inGrad =
          fl::reshape(fl::matmul(fl::reshape(gradOutput, to2dout), wt), in.shape());
      in.addGrad(Tensor(inGrad, false));
    }
    if (wt.requiresGrad()) {
      // gradOutput_2d: {nframes, nOut}, input_2d: {nframes, nIn}
      // wtGrad = gradOutput_2d^T @ input_2d = {nOut, nIn}
      Shape to2din({nframes, wtTensor.dim(1)});
      Shape to2dout({nframes, wtTensor.dim(0)});
      auto wtGrad =
          fl::matmul(fl::reshape(gradOutput, to2dout), fl::reshape(in, to2din), MatrixProperty::Transpose);
      wt.addGrad(Tensor(wtGrad, false));
    }
  };
  if (hasBias) {
    return Tensor(output, {in, wt, bs}, gradFunc);
  }
  return Tensor(output, {in, wt}, gradFunc);
}

Tensor conv2d(
    const Tensor& input,
    const Tensor& weights,
    int sx,
    int sy,
    int px,
    int py,
    int dx,
    int dy,
    int groups) {
  auto dummyBias = Tensor(Tensor(input.type()), false);
  return fl::autograd::conv2d(input, weights, dummyBias, sx, sy, px, py, dx, dy, groups);
}

Tensor conv2d(
    const Tensor& in,
    const Tensor& wt,
    const Tensor& bs,
    int sx,
    int sy,
    int px,
    int py,
    int dx,
    int dy,
    int groups) {
  FL_DTYPES_MATCH_CHECK(in, wt, bs);

  bool hasBias = !bs.isEmpty();
  bool wantState =
      in.requiresGrad() || wt.requiresGrad() || (hasBias && bs.requiresGrad());

  auto [output, state] = fl::detail::conv2d(
      in, wt, bs,
      sx, sy, px, py, dx, dy, groups, wantState);

  auto gradFunc =
      [sx, sy, px, py, dx, dy, hasBias, groups, state](
          std::vector<Tensor>& inputs, const Tensor& gradOutput) {
        auto& in = inputs[0];
        auto& wt = inputs[1];
        Tensor bs;
        const bool computeBiasGrad =
            inputs.size() > 2 && inputs[2].requiresGrad();
        if (hasBias && computeBiasGrad) {
          bs = inputs[2].detach();
        }

        // Data (input) gradients
        if (inputs[0].requiresGrad()) {
          auto dataGrad = fl::detail::conv2dBackwardData(
              gradOutput, in, wt,
              sx, sy, px, py, dx, dy, groups, state);
          inputs[0].addGrad(Tensor(dataGrad, false));
        }

        // Filter (weight) and bias gradients
        if (inputs[1].requiresGrad() || computeBiasGrad) {
          auto [filterGrad, biasGrad] = fl::detail::conv2dBackwardFilterBias(
              gradOutput, in, wt, bs,
              sx, sy, px, py, dx, dy, groups, state);
          if (inputs[1].requiresGrad()) {
            inputs[1].addGrad(Tensor(filterGrad, false));
          }
          if (computeBiasGrad) {
            inputs[2].addGrad(Tensor(biasGrad, false));
          }
        }
      };
  if (hasBias) {
    return Tensor(output, {in, wt, bs}, gradFunc);
  }
  return Tensor(output, {in, wt}, gradFunc);
}

Tensor pool2d(
    const Tensor& input,
    int wx,
    int wy,
    int sx,
    int sy,
    int px,
    int py,
    PoolingMode mode /* = PoolingMode::MAX */) {
  bool wantState = input.requiresGrad();

  auto [output, state] = fl::detail::pool2d(
      input, wx, wy, sx, sy, px, py, mode, wantState);

  auto gradFunc = [wx, wy, sx, sy, px, py, mode, output, state](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    auto& in = inputs[0];
    if (!in.requiresGrad()) {
      return;
    }

    in.addGrad(Tensor(
        fl::detail::pool2dBackward(
            gradOutput, in, output,
            wx, wy, sx, sy, px, py, mode, state),
        false));
  };
  return Tensor(output, {input}, gradFunc);
}

std::tuple<Tensor, Tensor, Tensor> batchnorm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& runningMean,
    const Tensor& runningVar,
    const std::vector<int>& axes,
    bool train,
    double momentum,
    double epsilon) {
  bool wantState =
      input.requiresGrad() || weight.requiresGrad() || bias.requiresGrad();

  auto result = fl::detail::batchnorm(
      input, weight, bias,
      runningMean.detach(), runningVar.detach(),
      axes, train, momentum, epsilon, wantState);

  auto gradFunc =
      [saveMean = std::move(result.saveMean),
       saveVar = std::move(result.saveVar),
       train,
       axes,
       epsilon,
       state = result.state](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
        auto& in = inputs[0];
        auto& wt = inputs[1];
        auto& bs = inputs[2];

        if (!in.requiresGrad() && !wt.requiresGrad() && !bs.requiresGrad()) {
          return;
        }

        auto [gradIn, gradWt, gradBs] = fl::detail::batchnormBackward(
            gradOutput, saveMean, saveVar,
            in, wt,
            axes, train, epsilon, state);

        in.addGrad(Tensor(gradIn.astype(in.type()), false));
        wt.addGrad(Tensor(gradWt.astype(wt.type()), false));
        if (!bs.isEmpty()) {
          bs.addGrad(Tensor(gradBs.astype(bs.type()), false));
        }
      };
  auto output = Tensor(result.output, {input, weight, bias}, gradFunc);
  return {std::move(output),
          std::move(result.newRunningMean),
          std::move(result.newRunningVar)};
}

Tensor gatedlinearunit(const Tensor& input, const int dim) {
  if (dim >= input.ndim()) {
    throw std::invalid_argument(
        "gatedlinearunit - passed dim is great than the "
        "number of dimensions of the input.");
  }

  auto inDims = input.shape();
  auto inType = input.type();
  auto inSize = inDims[dim];
  if (inSize % 2 == 1) {
    throw std::invalid_argument("halving dimension must be even for GLU");
  }

  std::vector<fl::Index> fhalf(input.ndim(), fl::span);
  std::vector<fl::Index> shalf(input.ndim(), fl::span);
  fhalf[dim] = fl::range(inSize / 2);
  shalf[dim] = fl::range(inSize / 2, inSize);

  Tensor fhalfout = input.detach()(fhalf);
  Tensor shalfout = input.detach()(shalf);

  // Temporary workaround for indexing bug present in ArrayFire 3.6.1.
  fhalfout = fl::reshape(fhalfout, fhalfout.shape());
  shalfout = fl::reshape(shalfout, shalfout.shape());
  shalfout = fl::sigmoid(shalfout);

  auto gradFunc = [fhalf, shalf, fhalfout, shalfout, inDims, inType](
                      std::vector<Tensor>& inputs,
                      const Tensor& gradOutput) {
    auto gradGlu = Tensor(inDims, inType);
    gradGlu(fhalf) = shalfout * gradOutput;
    gradGlu(shalf) =
        shalfout * (1.0 - shalfout) * fhalfout * gradOutput;
    inputs[0].addGrad(Tensor(gradGlu, false));
  };
  return Tensor(fhalfout * shalfout, {input.withoutData()}, gradFunc);
}

std::tuple<Tensor, Tensor, Tensor> rnn(
    const Tensor& input,
    const Tensor& hiddenState,
    const Tensor& cellState,
    const Tensor& weights,
    int hiddenSize,
    int numLayers,
    RnnMode mode,
    bool bidirectional,
    float dropProb) {
  bool wantState = input.requiresGrad() || hiddenState.requiresGrad() ||
      cellState.requiresGrad() || weights.requiresGrad();

  auto [output, hiddenOut, cellStateOut, state] = fl::detail::rnn(
      input, hiddenState,
      cellState, weights,
      hiddenSize, numLayers, mode, bidirectional, dropProb, wantState);

  auto gradData = std::make_shared<detail::RNNGradData>();

  auto gradFunc = [output,
                   numLayers,
                   hiddenSize,
                   mode,
                   bidirectional,
                   dropProb,
                   gradData,
                   state](
                      std::vector<Tensor>& inputs,
                      const Tensor& /* gradOutput */) {
    auto& input = inputs[0];
    auto& hiddenState = inputs[1];
    auto& cellState = inputs[2];
    auto& weights = inputs[3];

    if (!(input.requiresGrad() || hiddenState.requiresGrad() ||
          cellState.requiresGrad() || weights.requiresGrad())) {
      return;
    }

    auto [dx, dhx, dcx, dweights] = fl::detail::rnnBackward(
        input, hiddenState,
        cellState, weights,
        gradData->dy, gradData->dhy, gradData->dcy,
        output, numLayers, hiddenSize,
        mode, bidirectional, dropProb, state);

    input.addGrad(Tensor(dx.astype(input.type()), false));
    hiddenState.addGrad(Tensor(dhx.astype(hiddenState.type()), false));
    cellState.addGrad(Tensor(dcx.astype(cellState.type()), false));
    weights.addGrad(Tensor(dweights.astype(weights.type()), false));
  };

  Tensor dummy(Tensor(), {input, hiddenState, cellState, weights}, gradFunc);

  auto dyGradFunc =
      [gradData](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
        if (!inputs[0].isGradAvailable()) {
          inputs[0].addGrad(Tensor(Tensor(), false));
        }
        gradData->dy = gradOutput.asContiguousTensor();
      };

  auto dhyGradFunc =
      [gradData](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
        if (!inputs[0].isGradAvailable()) {
          inputs[0].addGrad(Tensor(Tensor(), false));
        }
        gradData->dhy = gradOutput.asContiguousTensor();
      };

  auto dcyGradFunc =
      [gradData](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
        if (!inputs[0].isGradAvailable()) {
          inputs[0].addGrad(Tensor(Tensor(), false));
        }
        gradData->dcy = gradOutput.asContiguousTensor();
      };

  Tensor yv(output, {dummy}, dyGradFunc); // output
  Tensor hyv(hiddenOut, {dummy}, dhyGradFunc); // hidden state output
  Tensor cyv(cellStateOut, {dummy}, dcyGradFunc); // cell state output
  return std::make_tuple(yv, hyv, cyv);
}

Tensor embedding(const Tensor& input, const Tensor& embeddings) {
  // TODO{fl::Tensor}{4-dims} - relax this
  if (input.ndim() >= 4) {
    throw std::invalid_argument("embedding input must have 3 or fewer dims");
  }

  auto idxs = input.detach().flatten();
  auto inDims = input.shape();
  // In row-major convention, embedding table is {numEmbeddings, embeddingDim}
  // Result shape is {input_shape..., embeddingDim}
  std::vector<Dim> rDims(input.ndim() + 1);
  for (unsigned i = 0; i < input.ndim(); i++) {
    rDims[i] = inDims[i];
  }
  rDims[input.ndim()] = embeddings.dim(1);
  Shape resultDims(rDims);
  Tensor result = fl::reshape(embeddings.detach()(idxs), resultDims);

  auto gradFunc = [](std::vector<Tensor>& inputs,
                     const Tensor& gradOutput) {
    auto& w = inputs[1];
    if (!w.requiresGrad()) {
      return;
    }

    auto ip = inputs[0].detach().flatten();
    unsigned size = ip.elements();
    // gradOutput reshaped to {size, embeddingDim}
    auto deltas = fl::reshape(gradOutput, {size, w.dim(1)});

    // Accumulate: grad_w[ip[i]] += deltas[i] for each i.
    // Uses indexAdd which correctly handles repeated indices.
    auto grad = fl::full(w.shape(), 0, deltas.type());
    fl::indexAdd(grad, 0, ip, deltas);
    w.addGrad(Tensor(grad, false));
  };

  return Tensor(result, {input, embeddings}, gradFunc);
}

Tensor pad(
    const Tensor& input,
    std::vector<std::pair<int, int>> paddings,
    double val) {
  if (paddings.size() > input.ndim()) {
    throw std::invalid_argument(
        "pad: number of padding dimensions exceeds number "
        "of input dimensions");
  }

  Shape opDims = input.shape();
  std::vector<fl::Index> inSeq(input.ndim(), fl::span);
  for (int i = 0; i < paddings.size(); ++i) {
    opDims[i] += (paddings[i].first + paddings[i].second);
    inSeq[i] = fl::range(paddings[i].first, opDims[i] - paddings[i].second);
  }
  Tensor result = fl::full(opDims, val, input.type());
  result(inSeq) = input.detach();

  auto gradFunc =
      [inSeq](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
        inputs[0].addGrad(Tensor(gradOutput(inSeq), false));
      };
  return Tensor(result, {input.withoutData()}, gradFunc);
}

Tensor dropout(const Tensor& input, double p) {
  if (p > 0.0) {
    auto mask = Tensor(
        (fl::rand(input.shape(), input.type()) > p).astype(input.type()), false);
    return 1.0 / (1.0 - p) * mask * input;
  } else {
    return input;
  }
}

Tensor relu(const Tensor& input) {
  return fl::autograd::maximum(input, 0.0);
}

Tensor gelu(const Tensor& input) {
  // GELU(x) = x * CDF(x) where CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
  // Exact erf form (matches PyTorch, JAX, TF defaults).
  auto result = input * (0.5 * (1.0 + fl::erf(input * M_SQRT1_2)));

  auto gradFunc = [](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
    auto& x = inputs[0];
    // d/dx GELU(x) = CDF(x) + x * PDF(x)
    //   CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
    //   PDF(x) = exp(-x^2 / 2) / sqrt(2 * pi)
    auto cdf = 0.5 * (1.0 + fl::erf(x * M_SQRT1_2));
    auto pdf = x * fl::exp(-0.5 * x * x) * (M_2_SQRTPI * M_SQRT1_2 * 0.5);
    inputs[0].addGrad(Tensor(gradOutput * (cdf + pdf), false));
  };
  return Tensor(result, {input}, gradFunc);
}

fl::Tensor relativePositionEmbeddingRotate(const fl::Tensor& input) {
  if (input.ndim() != 3) {
    throw std::invalid_argument(
        "relativePositionEmbeddingRotate - "
        "input tensor must have 3 dimensions");
  }

  // Row-major: input is {batch, L, T}
  // Output: {batch, L+T-1, T} (skew/rotate L dim for relative position)
  auto data = input.detach();
  int d0 = data.dim(0); // batch
  int d1 = data.dim(1); // L
  int d2 = data.dim(2); // T
  // Transpose to {batch, T, L} so row-major flatten gives column-major order
  data = fl::transpose(data, {0, 2, 1});
  // Pad columns: {batch, T, L+T}
  data = fl::concatenate(2, data, fl::full({d0, d2, d2}, 0.0, data.type()));
  // Flatten last two dims: {batch, T*(L+T)}
  data = fl::reshape(data, {d0, d2 * (d1 + d2), 1});
  // Slice: keep T*(L+T-1) elements
  data = data(fl::span, fl::range(0, d2 * (d1 + d2 - 1)));
  // Reshape: {batch, T, L+T-1}
  data = fl::reshape(data, {d0, d2, d1 + d2 - 1});
  // Transpose back: {batch, L+T-1, T}
  data = fl::transpose(data, {0, 2, 1});

  auto gradFunc = [d0, d1, d2](
                      std::vector<fl::Tensor>& inputs,
                      const fl::Tensor& gradOutput) {
    // Reverse: {batch, L+T-1, T} --> transpose --> {batch, T, L+T-1}
    auto gradData = fl::transpose(gradOutput, {0, 2, 1});
    // Flatten: {batch, T*(L+T-1)}
    gradData = fl::reshape(gradData, {d0, d2 * (d1 + d2 - 1), 1});
    // Pad: add T*1 zeros --> {batch, T*(L+T)}
    gradData = fl::concatenate(
        1, gradData, fl::full({d0, d2, 1}, 0.0, gradData.type()));
    // Reshape: {batch, T, L+T}
    gradData = fl::reshape(gradData, {d0, d2, d1 + d2});
    // Slice: keep first L columns --> {batch, T, L}
    gradData = Tensor(gradData, false)(
        fl::span, fl::span, fl::range(0, d1));
    // Transpose back: {batch, L, T}
    gradData = fl::transpose(gradData, {0, 2, 1});
    inputs[0].addGrad(fl::Tensor(gradData, false));
  };
  return fl::Tensor(data, {input}, gradFunc);
}

fl::Tensor multiheadAttention(
    const fl::Tensor& query,
    const fl::Tensor& key,
    const fl::Tensor& value,
    const fl::Tensor& posEmb,
    const fl::Tensor& mask,
    const fl::Tensor& padMask,
    const int32_t nHeads,
    const double pDropout,
    const int32_t offset /* = 0 */) {
  // Row-major: query/key/value are {B, Time, modelDim}
  if (query.ndim() != 3) {
    throw std::invalid_argument(
        "multiheadAttention - query input tensor should be 3 dimensions: "
        "B x Time x (nHeads * headDim)");
  }
  if (key.ndim() != 3) {
    throw std::invalid_argument(
        "multiheadAttention - key input tensor should be 3 dimensions: "
        "B x Time x (nHeads * headDim)");
  }
  if (value.ndim() != 3) {
    throw std::invalid_argument(
        "multiheadAttention - value input tensor should be 3 dimensions: "
        "B x Time x (nHeads * headDim)");
  }

  int32_t bsz = query.dim(0);       // B
  int32_t modelDim = query.dim(2);   // nHeads * headDim (last dim)
  int32_t headDim = modelDim / nHeads;

  // Reshape {B, T, nH*hD} --> {B, T, nH, hD} --> reorder --> {B, nH, T, hD}
  //   --> reshape --> {B*nH, T, hD}
  auto reshapeForHeads = [&](const Tensor& x) {
    auto x4d = fl::autograd::reshape(x, {bsz, -1, nHeads, headDim});
    x4d = fl::autograd::transpose(x4d, {0, 2, 1, 3});
    return fl::autograd::reshape(x4d, {nHeads * bsz, -1, headDim});
  };

  auto q = reshapeForHeads(query);  // {B*nH, T, hD}
  auto k = reshapeForHeads(key);    // {B*nH, T_k, hD}
  auto v = reshapeForHeads(value);  // {B*nH, T_k, hD}

  q = q / std::sqrt(float(headDim));
  // matmul(NT): {B*nH, T, hD} @ {B*nH, T_k, hD}^T = {B*nH, T, T_k}
  auto scores = fl::autograd::matmul(q, k, MatrixProperty::None, MatrixProperty::Transpose);

  if (!posEmb.isEmpty()) {
    // posEmb: {nH*B, L, hD} where L = 2*bptt-1
    // matmul(NT): {nH*B, L, hD} @ {nH*B, T, hD}^T = {nH*B, L, T}
    // relativePositionEmbeddingRotate: {nH*B, L, T} --> {nH*B, L+T-1, T}
    auto pscores =
        fl::autograd::relativePositionEmbeddingRotate(fl::autograd::matmul(posEmb.astype(q.type()), q, MatrixProperty::None, MatrixProperty::Transpose));
    int n = posEmb.dim(1) / 2 - offset;
    // Slice dim 1: {nH*B, T_k, T}. Transpose last two dims --> {nH*B, T, T_k}
    scores =
        scores + fl::autograd::transpose(
            pscores(fl::span, fl::range(n, n + k.dim(1)), fl::span),
            {0, 2, 1});
  }
  if (!mask.isEmpty()) {
    scores = scores + fl::autograd::tileAs(mask.astype(scores.type()), scores);
  }
  if (!padMask.isEmpty()) {
    // padMask: {B, T} where T matches key time
    if (padMask.dim(1) != key.dim(1)) {
      throw std::invalid_argument(
          "multiheadAttention: invalid padding mask size");
    }
    int T = padMask.dim(1);
    // {B, T} --> {B, 1, 1, T} --> tile to {B, nH, T, T} --> {B*nH, T, T}
    auto padMaskTile = fl::autograd::unsqueeze(fl::autograd::unsqueeze(padMask, 1), 2);
    padMaskTile = fl::autograd::tileAs(padMaskTile, {bsz, nHeads, T, T});
    scores = scores +
        fl::autograd::reshape(padMaskTile.astype(scores.type()), {nHeads * bsz, T, T});
  }

  // Softmax over key dimension (last dim)
  auto attn = fl::autograd::dropout(fl::autograd::softmax(scores, 2), pDropout);
  // matmul: {B*nH, T, T_k} @ {B*nH, T_k, hD} = {B*nH, T, hD}
  auto result = fl::autograd::matmul(attn.astype(v.type()), v);

  // Reshape back: {B*nH, T, hD} --> {B, nH, T, hD} --> reorder --> {B, T, nH, hD}
  //   --> reshape --> {B, T, modelDim}
  result = fl::autograd::reshape(result, {bsz, nHeads, -1, headDim});
  result = fl::autograd::transpose(result, {0, 2, 1, 3});
  result = fl::autograd::reshape(result, {bsz, -1, headDim * nHeads});
  return result;
}

} // namespace autograd
} // namespace fl
