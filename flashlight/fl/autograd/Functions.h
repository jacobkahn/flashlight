/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/common/Utils.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Tensor.h"

namespace fl::autograd {


namespace detail {

FL_API bool areDtypesEqual(const Tensor& a, const Tensor& b);

template <typename... Args>
bool areDtypesEqual(
    const Tensor& a,
    const Tensor& b,
    const Args&... args) {
  return areDtypesEqual(a, b) && areDtypesEqual(a, args...) &&
      areDtypesEqual(b, args...);
}

/**
 * Autograd-aware wrappers for Tensor-level operations.
 * Called by Tensor's operator(), flat(), astype(), and arithmetic operators
 * when requiresGrad() is true, so gradient logic lives in fl/autograd/ rather
 * than in fl/tensor/.
 */
FL_API Tensor autogradIndex(
    const Tensor& input,
    const std::vector<Index>& indices,
    const Tensor& forwardResult);
FL_API Tensor
autogradFlat(const Tensor& input, const Index& idx, const Tensor& fwdResult);
FL_API Tensor autogradAstype(
    const Tensor& input,
    fl::dtype newType,
    const Tensor& forwardResult);

FL_API Tensor autogradAdd(const Tensor& lhs, const Tensor& rhs);
FL_API Tensor autogradAdd(const Tensor& lhs, double rhs);
FL_API Tensor autogradSub(const Tensor& lhs, const Tensor& rhs);
FL_API Tensor autogradSub(const Tensor& lhs, double rhs);
FL_API Tensor autogradSub(double lhs, const Tensor& rhs);
FL_API Tensor autogradMul(const Tensor& lhs, const Tensor& rhs);
FL_API Tensor autogradMul(const Tensor& lhs, double rhs);
FL_API Tensor autogradDiv(const Tensor& lhs, const Tensor& rhs);
FL_API Tensor autogradDiv(const Tensor& lhs, double rhs);
FL_API Tensor autogradDiv(double lhs, const Tensor& rhs);

} // namespace detail

/**
 * Checks if a variadic number of Tensors have the same types.
 */
#define FL_DTYPES_MATCH_CHECK(...)              \
  if (!detail::areDtypesEqual(__VA_ARGS__)) {     \
    throw std::invalid_argument(                         \
        std::string(__func__) +                          \
        " doesn't support binary "                       \
        "operations with Tensors of different types"); \
  }

/**
 * \defgroup autograd_functions Autograd Functions
 * @{
 */

// Arithmetic operators (+, -, *, /) are defined on Tensor in
// fl/tensor/Tensor.cpp but dispatch to autograd::detail::autograd*()
// (below) when either operand requires grad. Comparison and logical operators
// are non-differentiable and defined directly on Tensor.

/**
 * Computes negative of each element in a Tensor.
 * \f[ out_i = -var_i \f]
 */
FL_API Tensor negative(const Tensor& input);

/**
 * Computes reciprocal of each element in a Tensor.
 * \f[ out_i = \frac{1}{var_i} \f]
 */
FL_API Tensor reciprocal(const Tensor& input);

/**
 * Computes exponential of each element in a Tensor.
 * \f[ out_i = e^{var_i} \f]
 */
FL_API Tensor exp(const Tensor& input);

/**
 * Computes natural logarithm of each element in a Tensor.
 * \f[ out_i = log(var_i) \f]
 */
FL_API Tensor log(const Tensor& input);

/**
 * Computes power of each element in a Tensor.
 * \f[ out_i = var_i^p \f]
 */
FL_API Tensor power(const Tensor& input, double p);

/**
 * Computes element-wise power of two Variables.
 * \f[ out_i = base_i^{exp_i} \f]
 */
FL_API Tensor power(const Tensor& base, const Tensor& exp);

/**
 * Computes element-wise power of a scalar base raised to Tensor exponents.
 * \f[ out_i = base^{exp_i} \f]
 */
FL_API Tensor power(const double& base, const Tensor& exp);

/**
 * Computes natural logarithm of (1 + element) for each element in a Tensor.
 * \f[ out_i = log(1.0 + var_i) \f]
 */
FL_API Tensor log1p(const Tensor& input);

/**
 * Computes sine of each element in a Tensor.
 * \f[ out_i = sin(var_i) \f]
 */
FL_API Tensor sin(const Tensor& input);

/**
 * Computes cosine of each element in a Tensor.
 * \f[ out_i = cos(var_i) \f]
 */
FL_API Tensor cos(const Tensor& input);

/**
 * Computes square root of each element in a Tensor.
 * \f[ out_i = \sqrt{var_i} \f]
 */
FL_API Tensor sqrt(const Tensor& input);

/**
 * Computes hyperbolic tangent of each element in a Tensor.
 * \f[ out_i = \frac{\exp(var_i) - \exp(-var_i)}{\exp(var_i) + \exp(-var_i)} \f]
 */
FL_API Tensor tanh(const Tensor& input);

/**
 * Clamps all elements in input into the range [ `min`, `max` ] and return a
 * resulting tensor:
 * \f[ \begin{split}y_i = \begin{cases}
 *     \text{min} & \text{if } x_i < \text{min} \\
 *     x_i & \text{if } \text{min} \leq x_i \leq \text{max} \\
 *     \text{max} & \text{if } x_i > \text{max}
 *     \end{cases}\end{split} \f]
 */
FL_API Tensor
clip(const Tensor& input, const double min, const double max);

/**
 * Computes sigmoid of each element in a Tensor.
 * \f[ out_i = \frac{1}{1 + \exp(-var_i)} \f]
 */
FL_API Tensor sigmoid(const Tensor& input);

/**
 * Computes swish of each element in a Tensor
 * from [Ramachandran et al (2013)](
 * https://arxiv.org/abs/1710.05941), _Searching for Activation Functions_.
 * \f[ Swish(x) = x \cdot sigmoid(\beta x) \f]
 * where \f$\beta\f$ is a constant, often is 1.
 */
FL_API Tensor swish(const Tensor& input, double beta);

/**
 * Computes the error function of each element in a Tensor
 * from as in [here](https://en.wikipedia.org/wiki/Error_function)
 */
FL_API Tensor erf(const Tensor& input);

/**
 * Returns element-wise maximum value of two Variables.
 * \f[ out = maximum(var_1, var_2) \f]
 */
FL_API Tensor maximum(const Tensor& lhs, const Tensor& rhs);

/**
 * Returns maximum value of a scalar and each element in a Tensor.
 * \f[ out_i = maximum(var_i, value) \f]
 */
FL_API Tensor maximum(const Tensor& lhs, const double& rhs);

/**
 * Returns maximum value of a scalar and each element in a Tensor.
 * \f[ out_i = maximum(value, var_i) \f]
 */
FL_API Tensor maximum(const double& lhs, const Tensor& rhs);

/**
 * Returns element-wise minimum value of two Variables.
 * \f[ out = minimum(var_1, var_2) \f]
 */
FL_API Tensor minimum(const Tensor& lhs, const Tensor& rhs);

/**
 * Returns minimum value of a scalar and each element in a Tensor.
 * \f[ out_i = minimum(var_i, value) \f]
 */
FL_API Tensor minimum(const Tensor& lhs, const double& rhs);

/**
 * Returns minimum value of a scalar and each element in a Tensor.
 * \f[ out_i = minimum(value, var_i) \f]
 */
FL_API Tensor minimum(const double& lhs, const Tensor& rhs);

/**
 * Returns a tensor that is a transposed version of a Tensor. Reorders the
 * input based on the shape of the input dimensions. If left empty, transposes
 * over all dimensinos (reverses all dimensions).
 */
FL_API Tensor transpose(const Tensor& input, const Shape& dims = {});

/**
 * Repeats the tensor `input` along certain dimensions so as to match the shape
 * of `reference`. The dimensions to be repeated along are automatically
 * inferred.
 */
FL_API Tensor tileAs(const Tensor& input, const Tensor& reference);

/**
 * Repeats the tensor `input` along certain dimensions so as to match the shape
 * in the descriptor `rdims`. The dimensions to be repeated along are
 * automatically inferred.
 */
FL_API Tensor tileAs(const Tensor& input, const Shape& rdims);

/**
 * Sums up the tensor `input` along certain dimensions so as to match the shape
 * of `reference`. The dimensions to be summed along are automatically inferred.
 * Note that after summation, the shape of those dimensions will be 1.
 */
FL_API Tensor sumAs(const Tensor& input, const Tensor& reference);

/**
 * Sums up the tensor `input` along certain dimensions so as to match the shape
 * in the descriptor `rdims`. The dimensions to be summed along are
 * automatically inferred. Note that after summation, the shape of those
 * dimensions will be 1.
 */
FL_API Tensor sumAs(const Tensor& input, const Shape& rdims);

/**
 * Concatenates Variables along a specific axis. The shape of input
 * Variables should be identical except the axis to concatenate.
 */
FL_API Tensor
concatenate(const std::vector<Tensor>& concatInputs, unsigned axis = 0);

/**
 * Splits a Tensor into equally sized chunks (if possible)
 *
 * @param input a Tensor to split
 * @param splitSize size of each split. If input dimension is not evenly
 * divisible, last chunk of smaller splitSize will be included.
 * @param dim dimension along which to split the Tensor
 */
FL_API std::vector<Tensor>
split(const Tensor& input, long splitSize, int dim);

/**
 * Splits a Tensor into smaller chunks.
 *
 * @param input a Tensor to split
 * @param splitSizes vector of integers specifying the sizes for each split
 * @param dim dimension along which to split the Tensor
 */
FL_API std::vector<Tensor>
split(const Tensor& input, const std::vector<long>& splitSizes, int dim);

/**
 * Repeats the tensor `input` along specific dimensions. The number of
 * repetition along each dimension is specified in descriptor `dims`.
 */
FL_API Tensor tile(const Tensor& input, const Shape& dims);

/**
 * Sums up the tensors `input` along dimensions specified in descriptor `axes`.
 * If `axes` has size greater than 1, reduce over all of them.
 */
FL_API Tensor
sum(const Tensor& input, const std::vector<int>& axes = {}, bool keepDims = false);

/**
 * Computes the mean of the tensor `input` along dimensions specified in
 * descriptor `axes`. If `axes` has size greater than 1, reduce over all of
 * them.
 */
FL_API Tensor mean(
    const Tensor& input,
    const std::vector<int>& axes = {},
    bool keepDims = false);

/**
 * Lp-norm computation, reduced over specified dimensions.
 *
 * @param input tensor on which the Lp norm is going to be computed.
 * @param p the p value of the Lp norm.
 * @param axes dimensions over which the reduction is performed.
 */
FL_API Tensor norm(
    const Tensor& input,
    const std::vector<int>& axes = {},
    double p = 2,
    bool keepDims = false);

/**
 * Lp norm normalization of values across the given dimensions.
 *
 * @param input the tensor to be normalized.
 * @param axes dimensions over which values are normalized.
 * @param p the p value of the Lp norm.
 * @param eps clamping value to avoid overflows.
 */
FL_API Tensor normalize(
    const Tensor& input,
    const std::vector<int>& axes,
    double p = 2,
    double eps = 1e-12);

/**
 * Computes variance of the tensor `input` along dimensions specified in
 * descriptor `axes`. If `axes` has size greater than 1, reduce over all of
 * them. If `bias` is `true`, applies Bessel's correction (divides by N-1,
 * sample variance). Otherwise divides by N (population variance).
 */
FL_API Tensor
var(const Tensor& input,
    const std::vector<int>& axes = {},
    const bool bias = false,
    bool keepDims = false);

/**
 * Computes standard deviation of the tensor `input` along dimensions
 * specified in descriptor `axes`. Equivalent to `sqrt(var(...))`.
 */
FL_API Tensor
std(const Tensor& input,
    const std::vector<int>& axes = {},
    const bool bias = false,
    bool keepDims = false);

/**
 * Conducts matrix-matrix multiplication on two Variables, optionally
 * transposing either input. This is a batched function if \f$B_1\f$ or
 * \f$B_2\f$ is greater than 1.

 * @param lhs the left-hand side Tensor
 * @param rhs the right-hand side Tensor
 * @param lhsProp the MatrixProperty to apply to lhs (None or Transpose)
 * @param rhsProp the MatrixProperty to apply to rhs (None or Transpose)
 * @return the result of the matrix multiplication
 */
FL_API Tensor matmul(
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp = MatrixProperty::None,
    MatrixProperty rhsProp = MatrixProperty::None);

/**
 * Returns the absolute values of each element in a Tensor.
 * \f[ out_i = |var_i| \f]
 */
FL_API Tensor abs(const Tensor& input);

/**
 * Flattens the input to a single dimension.
 */
FL_API Tensor flatten(const Tensor& input);

/**
 * Reshapes a Tensor without changing the data order. The shape of
 * the output Tensor is specified in descriptor `dims`.
 */
FL_API Tensor reshape(const Tensor& input, const Shape& dims);

/**
 * Remove all dimensions of size 1 from a Tensor, or a specific size-1
 * dimension.
 */
FL_API Tensor squeeze(const Tensor& input);
FL_API Tensor squeeze(const Tensor& input, const unsigned axis);

/**
 * Insert a dimension of size 1 at the specified axis.
 */
FL_API Tensor unsqueeze(const Tensor& input, const unsigned axis);

/**
 * Selects elements from `x` or `y` based on `condition`.
 * \f[ out_i = condition_i ? x_i : y_i \f]
 * Gradient flows through the selected branch.
 */
FL_API Tensor
where(const Tensor& condition, const Tensor& x, const Tensor& y);

/**
 * Reverses the elements of a Tensor along a dimension.
 */
FL_API Tensor flip(const Tensor& input, const unsigned dim);

/**
 * Extracts the lower triangular part of the last two dimensions.
 */
FL_API Tensor tril(const Tensor& input, int diagonal = 0);

/**
 * Extracts the upper triangular part of the last two dimensions.
 */
FL_API Tensor triu(const Tensor& input, int diagonal = 0);

/**
 * Computes the cumulative sum of elements along a dimension.
 */
FL_API Tensor cumsum(const Tensor& input, const unsigned axis);

/**
 * Shifts elements of a Tensor along a given axis by a given shift amount,
 * wrapping around the end.
 */
FL_API Tensor roll(const Tensor& input, int shift, unsigned axis);

/**
 * [Non-differentiable] Returns the element-wise floor of the input.
 */
FL_API Tensor floor(const Tensor& input);

/**
 * [Non-differentiable] Returns the element-wise ceiling of the input.
 */
FL_API Tensor ceil(const Tensor& input);

/**
 * [Non-differentiable] Returns the element-wise rounding of the input.
 */
FL_API Tensor rint(const Tensor& input);

/**
 * [Non-differentiable] Returns the element-wise sign of the input.
 */
FL_API Tensor sign(const Tensor& input);

/**
 * [Non-differentiable] Returns whether each element is NaN.
 */
FL_API Tensor isnan(const Tensor& input);

/**
 * [Non-differentiable] Returns whether each element is infinite.
 */
FL_API Tensor isinf(const Tensor& input);


/**
 * Applies a linear transformation to the input Tensor:
 * \f[
 *    y = Ax
 * \f]
 * @param input a Tensor with shape [\f$N\f$, \f$M\f$, \f$B_1\f$, \f$B_2\f$]
 * @param weight a Tensor with shape [\f$K\f$, \f$N\f$]
 * @return a Tensor with shape [\f$K\f$, \f$M\f$, \f$B_1\f$, \f$B_2\f$]
 */
FL_API Tensor linear(const Tensor& input, const Tensor& weight);

/**
 * Applies a linear transformation to the input Tensor:
 * \f[
 *    y = Ax + b
 * \f]
 * @param input a Tensor with shape [\f$N\f$, \f$M\f$, \f$B_1\f$, \f$B_2\f$]
 * @param weight a Tensor with shape [\f$K\f$, \f$N\f$]
 * @param bias a Tensor with shape [\f$K\f$]
 * @return a Tensor with shape [\f$K\f$, \f$M\f$, \f$B_1\f$, \f$B_2\f$]
 */
FL_API Tensor
linear(const Tensor& input, const Tensor& weight, const Tensor& bias);

/**
 * Applies a 2D convolution over an input signal given filter weights. In the
 * simplest case, the output with shape [\f$X_{out}\f$, \f$Y_{out}\f$,
 * \f$C_{out}\f$, \f$N\f$] of the convolution with input [\f$X_{in}\f$,
 * \f$Y_{in}\f$, \f$C_{in}\f$, \f$N\f$] and weight [\f$K_x\f$, \f$K_y\f$,
 * \f$C_{in}\f$, \f$C_{out}\f$] can be precisely described as:
 * \f[
      \text{out}(C_{out_j}, N_i) =
          \sum_{k = 0}^{C_{in} - 1} \text{weight}(k, C_{out_j}) \star
          \text{input}(k, N_i)
 * \f]

 * @param input a Tensor with shape [\f$X_{in}\f$, \f$Y_{in}\f$, \f$C_{in}\f$,
 * \f$N\f$]
 * @param weights a Tensor with shape [\f$K_x\f$, \f$K_y\f$, \f$C_{in}\f$,
 * \f$C_{out}\f$]
 * @param sx stride in the first dimension
 * @param sy stride in the second dimension
 * @param px number of positions of zero-padding on both sides in the first
 * dimension
 * @param py number of positions of zero-padding on both sides in the second
 * dimension
 * @param dx dilation along the first kernel dimension. A dilation of 1
 * is equivalent to a standard convolution along this axis.
 * @param dy dilation along the second kernel dimension. A dilation of 1
 * is equivalent to a standard convolution along this axis.
 * @param groups number of filter groups
 * @return a Tensor with shape [\f$X_{out}\f$, \f$Y_{out}\f$, \f$C_{out}\f$,
 * \f$N\f$]]
 */
FL_API Tensor conv2d(
    const Tensor& input,
    const Tensor& weights,
    int sx = 1,
    int sy = 1,
    int px = 0,
    int py = 0,
    int dx = 1,
    int dy = 1,
    int groups = 1);

/**
 * Applies a 2D convolution over an input signal given filter weights and
 * biases.
 * @param bias a Tensor with shape [\f$C_{out}\f$]
 */
FL_API Tensor conv2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    int sx = 1,
    int sy = 1,
    int px = 0,
    int py = 0,
    int dx = 1,
    int dy = 1,
    int groups = 1);

/**
 * Applies a 2D pooling over an input signal composed of several input planes.
 * @param input a Tensor with shape [\f$X_{in}\f$, \f$Y_{in}\f$, \f$C\f$,
 *   \f$N\f$]
 * @param wx pooling window size in the first dimension
 * @param wy pooling window size in the second dimension
 * @param sx stride in the first dimension
 * @param sy stride in the second dimension
 * @param px number of positions of zero-padding on both sides in the first
 * dimension
 * @param py number of positions of zero-padding on both sides in the second
 * dimension
 * @param mode pooling mode, which supports:
 * - MAX
 * - AVG_INCLUDE_PADDING
 * - AVG_EXCLUDE_PADDING
 */
FL_API Tensor pool2d(
    const Tensor& input,
    int wx,
    int wy,
    int sx = 1,
    int sy = 1,
    int px = 0,
    int py = 0,
    PoolingMode mode = PoolingMode::MAX);

/**
 * Applies a softmax function on Tensor `input` along dimension `dim`, so that
 * the elements of the dimensional `dim` in output lie in the range (0,1) and
 * sum to 1.
 * \f[ out(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)} \f]
 */
FL_API Tensor softmax(const Tensor& input, const int dim);

/**
 * Applies a log(softmax(x)) function on Tensor `input` along dimension `dim`
 * \f[
 *    out(x_{i}) = log \Big( \frac{exp(x_i)}{\sum_j exp(x_j)} \Big)
 * \f]
 */
FL_API Tensor logSoftmax(const Tensor& input, const int dim);

/**
 * Computes the binary cross entropy loss between an input tensor \f$x\f$ and a
 * target tensor \f$y\f$. The binary cross entropy loss is:
 * \f[
      B(x, y) = \frac{1}{n} \sum_{i = 0}^n
      -\left( y_i \times \log(x_i) + (1 - y_i) \times \log(1 - x_i) \right)
   \f]
 * Both the inputs and the targets are expected to be between 0 and 1.
 *
 * @param inputs a tensor with the predicted values
 * @param targets a tensor with the target values
 */
FL_API Tensor
binaryCrossEntropy(const Tensor& inputs, const Tensor& targets);

/**
 * Computes the categorical cross entropy loss. The input is expected to
 * contain log-probabilities for each class. The targets should be the
 * index of the ground truth class for each input example.
 * \f[
 *   \begin{split}\ell(x, y) = \begin{cases}
 *      \frac{1}{N} \sum_{n=1}^N -x_{n,y_n}, & \text{if}\;
 *      \text{reduction} = \text{MEAN},\\
 *      \sum_{n=1}^N -x_{n,y_n},  & \text{if}\;
 *      \text{reduction} = \text{SUM}, \\
 *      \{ -x_{1,y_1}, ..., -x_{N,y_N} \},  & \text{if}\;
 *      \text{reduction} = \text{NONE}.
 *   \end{cases}\end{split}
 * \f]

 * @param input a `Tensor` with shape [\f$C\f$, \f$B_1\f$, \f$B_2\f$,
 * \f$B_3\f$] where \f$C\f$ is the number of classes.
 * @param targets an integer `Tensor` with shape [\f$B_1\f$, \f$B_2\f$,
 * \f$B_3\f$]. The values must be in \f$[0, C - 1]\f$
 * @param reduction reduction mode, which supports:
 * - NONE
 * - MEAN
 * - SUM
 * @param ignoreIndex a target value that is ignored and does not contribute
 * to the loss or the input gradient. If `reduce` is MEAN, the loss is
 * averaged over non-ignored targets. Only indicies in \f$[0, C - 1]\f$ are
 * considered to be valid.
 * @return a `Tensor` of loss value with shape scalar by default. If `reduce`
 * is NONE, then [\f$B_1\f$, \f$B_2\f$, \f$B_3\f$].
 */
FL_API Tensor categoricalCrossEntropy(
    const Tensor& input,
    const Tensor& targets,
    ReduceMode reduction = ReduceMode::MEAN,
    int ignoreIndex = -1);

/**
 * Computes the weighted cross entropy loss. The input is expected to
 * contain log-probabilities for each class. The targets should be the
 * index of the ground truth class for each input example.
 * \f[
 *   \ell(x, y) = \frac{\sum_{n=1}^N weight[y_n] * -x_{n,y_n}}\;
 *   {\sum_{n=1}^N weight[y_n]
 * \f]

 * @param input a `Tensor` with shape [\f$C\f$, \f$B_1\f$, \f$B_2\f$,
 * \f$B_3\f$] where \f$C\f$ is the number of classes.
 * @param targets an integer `Tensor` with shape [\f$B_1\f$, \f$B_2\f$,
 * The values must be in \f$[0, C - 1]\f$.
 * @param weights an `Tensor` with shape f$[0, C - 1]\f.
 * @param ignoreIndex a target value that is ignored and does not contribute
 * to the loss or the input gradient. If `reduce` is MEAN, the loss is
 * averaged over non-ignored targets. Only indicies in \f$[0, C - 1]\f$ are
 * considered to be valid.
 * @return a `Tensor` of loss value with shape scalar by default.  */
FL_API Tensor weightedCategoricalCrossEntropy(
    const Tensor& input,
    const Tensor& targets,
    const Tensor& weight,
    int ignoreIndex);

/**
 * The gated linear unit.
 * \f[
        H = A \times \sigma(B)
 * \f]
 * where `input` is split in half along `dim` to form `A` and `B`.
 * See [Language Modeling with Gated Convolutional Networks]
 * (https://arxiv.org/abs/1612.08083).

 * @param input input Tensor
 * @param dim dimension on which to split the input
 */
FL_API Tensor gatedlinearunit(const Tensor& input, const int dim);

/**
 * Applies an RNN unit to an input sequence.
 * A general RNN operator can be expressed as following:
 * \f[
      (h_t, c_t) = f_W(x_t, h_{t-1}, c_{t-1})
 * \f]
 * where \f$h_t\f$, \f$c_t\f$ are the hidden/cell state at time \f$t\f$,
 * \f$x_t\f$ is the input at time \f$t\f$
 *
 * \note{cuDNN and oneDNN RNN weights are incompatible since the structure of
 * the computation is different for each. There is no mapping between weights
 * from each of those backends.}
 *
 * @param input Tensor of input with shape [input size, batch size, sequence
 * length]
 * @param hiddenState Tensor of hidden state with shape [hidden size, batch
 * size, total layers]
 * @param cellState [LSTM only] Tensor of cell state with same shape as
 * hidden state
 * @param weights Learnable parameters in the RNN unit
 * @param hiddenSize number of features in the hidden state
 * @param numLayers number of recurrent layers
 * @param mode defines the type of RNN unit
 *  - RELU
 *  - TANH
 *  - LSTM
 *  - GRU
 * @param bidirectional if `True`, becomes a bidirectional RNN, unidirectional
 otherwise
 * @param dropout if non-zero, introduces a `Dropout` layer on the outputs of
 * each RNN layer except the last one, with dropout probability equal to dropout

 * @return a tuple of three Variables:
 * - `y`: input with shape [input size, batch size, sequence length *
 * directions]
 * - `hiddenState`: hidden state for the current time step
 * - `cellState`: cell state for the current time step
 */
FL_API std::tuple<Tensor, Tensor, Tensor> rnn(
    const Tensor& input,
    const Tensor& hiddenState,
    const Tensor& cellState,
    const Tensor& weights,
    int hiddenSize,
    int numLayers,
    RnnMode mode,
    bool bidirectional,
    float dropout);

/**
 * Looks up embeddings in a fixed dictionary and size.
 * @param input a Tensor of a list of indices with shape [\f$B_1\f$,
 * \f$B_2\f$, \f$B_3\f$]
 * @param embeddings a Tensor of an embedding matrix with shape [\f$D\f$,
 * \f$N\f$], where \f$N\f$ is the number of items and \f$D\f$ is the embedding
 * size.
 * @return a Tensor of embeddings with shape [\f$D\f$, \f$B_1\f$, \f$B_2\f$,
 * \f$B_3\f$]
 */
FL_API Tensor embedding(const Tensor& input, const Tensor& embeddings);

/**
 * Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with
 * additional channel dimension) as described in the paper
 * [Batch Normalization: Accelerating Deep Network Training by Reducing Internal
 * Covariate Shift] (https://arxiv.org/abs/1502.03167) .
 * \f[
 *   y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma +
 * \beta
 * \f]
 * The mean and standard-deviation are calculated per-dimension over the
 * mini-batches and \f$\gamma\f$ and \f$\beta\f$ are learnable parameter vectors
 * of size \f$C\f$, the input size. By default, during training this layer keeps
 * running estimates of its computed mean and variance, which are then used for
 * normalization during evaluation.

 * @param input a Tensor with size [\f$H\f$, \f$W\f$, \f$C\f$, \f$N\f$]
 * @param weight a Tensor with size [\f$C\f$] for \f$\gamma\f$
 * @param bias a Tensor with size [\f$C\f$] for \f$\beta\f$
 * @param runningMean a buffer storing intermediate means during training
 * @param runningVar a buffer storing intermediate variances during training
 * @param axes dimensions to perform normalization on. If having size greater
 * than one, reduce over all of them.
 * @param train a flag indicating if running in training mode
 * @param momentum value of momentum
 * @param epsilon value of \f$\epsilon\f$

 * @return a Tensor with same shape as `input`
 */
FL_API std::tuple<Tensor, Tensor, Tensor> batchnorm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& runningMean,
    const Tensor& runningVar,
    const std::vector<int>& axes,
    bool train,
    double momentum,
    double epsilon);

/**
 * Applies asymmetric padding on a Tensor `input`.
 * @param input input Tensor
 * @param paddings a list of integer pairs specifying the positions we want to
 * pad on both sides for each dimension
 * @param val padding value
 * @return a padded Tensor
 */
FL_API Tensor pad(
    const Tensor& input,
    std::vector<std::pair<int, int>> paddings,
    double val);

/**
 * Applies dropout on a Tensor `input`.
 * @param input input Tensor
 * @param p the probability of dropout
 * @return a droped out Tensor
 */
FL_API Tensor dropout(const Tensor& input, double p);

/**
 * Applies the [rectified linear
 * unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) function
 * element-wise to a `Tensor`:
 * \f[ ReLU(x) = \max(0, x) \f]
 */
FL_API Tensor relu(const Tensor& input);

/**
 * Applies the [Gaussian Error linear
 * Unit](https://arxiv.org/abs/1606.08415) function
 * element-wise to a `Tensor`
 */
FL_API Tensor gelu(const Tensor& input);

/**
 * Relative positional embedding for the multihead attention
 * Implementation partially follows https://arxiv.org/pdf/1803.02155.pdf
 */
FL_API Tensor relativePositionalEmbeddingRotate(const Tensor& input);

/**
 * Multihead Attention function
 * For details, see [Vaswani et al (2017)](https://arxiv.org/abs/1706.03762).
 * @param query query Tensor of size T x nHeads * headDim x B
 * @param key key Tensor of size Time x nHeads * headDim x B
 * @param value value Tensor of size Time x nHeads * headDim x B
 * @param posEmb if non empty then compute relative
 * positional embedding in additon to standard computations
 * @param mask mask or not future in the computations T x T
 * if non-empty then don't use future (for example for autoregressive language
 * models or for decoder part in the encoder-decoder transformer models)
 * @param padMask mask which is 1 for positions where pad token is,
 * don't attend to the pad-positions, of size T x B
 * @param nHeads number of heads
 * @param pDropout dropout probability
 * @param offset size of the current output from the decoder used now as input
 */
FL_API Tensor multiheadAttention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& posEmb,
    const Tensor& mask,
    const Tensor& padMask,
    const int32_t nHeads,
    const double pDropout,
    const int32_t offset = 0);

/** @} */

} // namespace fl::autograd
