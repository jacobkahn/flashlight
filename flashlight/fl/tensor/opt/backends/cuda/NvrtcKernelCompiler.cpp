/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/opt/backends/cuda/NvrtcKernelCompiler.h"

#include <cuda.h>
#include <nvrtc.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace fl {

namespace {

#define NVRTC_CHECK(expr)                                               \
  do {                                                                  \
    nvrtcResult _res = (expr);                                          \
    if (_res != NVRTC_SUCCESS) {                                        \
      std::ostringstream oss;                                           \
      oss << "[NvrtcKernelCompiler] " #expr " failed: "                 \
          << nvrtcGetErrorString(_res);                                  \
      throw std::runtime_error(oss.str());                              \
    }                                                                   \
  } while (0)

#define CU_CHECK(expr)                                                  \
  do {                                                                  \
    CUresult _res = (expr);                                             \
    if (_res != CUDA_SUCCESS) {                                         \
      const char* errStr = nullptr;                                     \
      cuGetErrorString(_res, &errStr);                                  \
      std::ostringstream oss;                                           \
      oss << "[NvrtcKernelCompiler] " #expr " failed: "                 \
          << (errStr ? errStr : "unknown");                              \
      throw std::runtime_error(oss.str());                              \
    }                                                                   \
  } while (0)

const char* unaryOpToExpr(UnaryOp op) {
  switch (op) {
    case UnaryOp::Exp: return "expf(x)";
    case UnaryOp::Log: return "logf(x)";
    case UnaryOp::Log1p: return "log1pf(x)";
    case UnaryOp::Sin: return "sinf(x)";
    case UnaryOp::Cos: return "cosf(x)";
    case UnaryOp::Sqrt: return "sqrtf(x)";
    case UnaryOp::Tanh: return "tanhf(x)";
    case UnaryOp::Floor: return "floorf(x)";
    case UnaryOp::Ceil: return "ceilf(x)";
    case UnaryOp::Rint: return "rintf(x)";
    case UnaryOp::Negative: return "(-x)";
    case UnaryOp::Absolute: return "fabsf(x)";
    case UnaryOp::Sigmoid: return "(1.0f / (1.0f + expf(-x)))";
    case UnaryOp::Erf: return "erff(x)";
    case UnaryOp::LogicalNot: return "(x == 0.0f ? 1.0f : 0.0f)";
    case UnaryOp::Sign: return "(x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f))";
    case UnaryOp::Isnan: return "(isnan(x) ? 1.0f : 0.0f)";
    case UnaryOp::Isinf: return "(isinf(x) ? 1.0f : 0.0f)";
  }
  return "x"; // unreachable
}

const char* binaryOpToExpr(BinaryOp op) {
  switch (op) {
    case BinaryOp::Add: return "(a + b)";
    case BinaryOp::Sub: return "(a - b)";
    case BinaryOp::Mul: return "(a * b)";
    case BinaryOp::Div: return "(a / b)";
    case BinaryOp::Eq: return "(a == b ? 1.0f : 0.0f)";
    case BinaryOp::Neq: return "(a != b ? 1.0f : 0.0f)";
    case BinaryOp::Gt: return "(a > b ? 1.0f : 0.0f)";
    case BinaryOp::Gte: return "(a >= b ? 1.0f : 0.0f)";
    case BinaryOp::Lt: return "(a < b ? 1.0f : 0.0f)";
    case BinaryOp::Lte: return "(a <= b ? 1.0f : 0.0f)";
    case BinaryOp::Min: return "fminf(a, b)";
    case BinaryOp::Max: return "fmaxf(a, b)";
    case BinaryOp::Pow: return "powf(a, b)";
    case BinaryOp::Mod: return "fmodf(a, b)";
    case BinaryOp::And: return "((a != 0.0f && b != 0.0f) ? 1.0f : 0.0f)";
    case BinaryOp::Or: return "((a != 0.0f || b != 0.0f) ? 1.0f : 0.0f)";
    case BinaryOp::Shl: return "((float)((int)a << (int)b))";
    case BinaryOp::Shr: return "((float)((int)a >> (int)b))";
    case BinaryOp::BitAnd: return "((float)((int)a & (int)b))";
    case BinaryOp::BitOr: return "((float)((int)a | (int)b))";
    case BinaryOp::BitXor: return "((float)((int)a ^ (int)b))";
  }
  return "(a + b)"; // unreachable
}

char unaryOpKey(UnaryOp op) {
  return 'A' + static_cast<int>(op);
}

char binaryOpKey(BinaryOp op) {
  return 'a' + static_cast<int>(op);
}

} // namespace

// CompiledKernel
CompiledKernel::~CompiledKernel() {
  if (module) {
    cuModuleUnload(module);
  }
}

CompiledKernel::CompiledKernel(CompiledKernel&& other) noexcept
    : module(other.module), function(other.function) {
  other.module = nullptr;
  other.function = nullptr;
}

CompiledKernel& CompiledKernel::operator=(CompiledKernel&& other) noexcept {
  if (this != &other) {
    if (module) {
      cuModuleUnload(module);
    }
    module = other.module;
    function = other.function;
    other.module = nullptr;
    other.function = nullptr;
  }
  return *this;
}

// NvrtcKernelCompiler
NvrtcKernelCompiler& NvrtcKernelCompiler::getInstance() {
  static NvrtcKernelCompiler instance;
  return instance;
}

std::string NvrtcKernelCompiler::makeKey(
    const std::vector<FusedOp>& program,
    unsigned numExternalInputs) {
  std::string key;
  key += std::to_string(numExternalInputs) + "_";
  for (const auto& step : program) {
    if (step.isUnary()) {
      key += 'U';
      key += unaryOpKey(step.unaryOp());
      key += std::to_string(step.src0);
    } else {
      key += 'B';
      key += binaryOpKey(step.binaryOp());
      key += std::to_string(step.src0);
      key += '_';
      key += std::to_string(step.src1);
    }
    key += ',';
  }
  return key;
}

std::string NvrtcKernelCompiler::generateKernelSource(
    const std::vector<FusedOp>& program,
    unsigned numExternalInputs) {
  std::ostringstream oss;

  // Kernel signature: takes N float* inputs + float* output + int n
  // Note: CUDA device math functions (expf, logf, etc.) are builtins --
  // no #include needed for NVRTC
  oss << "extern \"C\" __global__\n";
  oss << "void fused_elementwise(\n";
  for (unsigned i = 0; i < numExternalInputs; i++) {
    oss << "    const float* __restrict__ in" << i << ",\n";
  }
  oss << "    float* __restrict__ out,\n";
  oss << "    int n) {\n";
  oss << "  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
  oss << "  if (idx >= n) return;\n\n";

  // Load external inputs
  for (unsigned i = 0; i < numExternalInputs; i++) {
    oss << "  float s" << i << " = in" << i << "[idx];\n";
  }

  // Execute program steps
  for (unsigned i = 0; i < program.size(); i++) {
    unsigned slotIdx = numExternalInputs + i;
    const auto& step = program[i];
    if (step.isUnary()) {
      // Generate: float sN = <unary_expr(s_src0)>;
      oss << "  float s" << slotIdx << " = ";
      // Replace 'x' in expression with actual source variable
      std::string expr = unaryOpToExpr(step.unaryOp());
      std::string srcVar = "s" + std::to_string(step.src0);
      // Simple replacement of 'x' with source variable name
      std::string result;
      for (size_t j = 0; j < expr.size(); j++) {
        if (expr[j] == 'x' &&
            (j == 0 || !std::isalnum(expr[j-1])) &&
            (j+1 >= expr.size() || !std::isalnum(expr[j+1]))) {
          result += srcVar;
        } else {
          result += expr[j];
        }
      }
      oss << result << ";\n";
    } else {
      // Generate: float sN = <binary_expr(s_src0, s_src1)>;
      oss << "  float s" << slotIdx << " = ";
      std::string expr = binaryOpToExpr(step.binaryOp());
      std::string srcA = "s" + std::to_string(step.src0);
      std::string srcB = "s" + std::to_string(step.src1);
      // Replace 'a' and 'b' in expression
      std::string result;
      for (size_t j = 0; j < expr.size(); j++) {
        if (expr[j] == 'a' &&
            (j == 0 || !std::isalnum(expr[j-1])) &&
            (j+1 >= expr.size() || !std::isalnum(expr[j+1]))) {
          result += srcA;
        } else if (expr[j] == 'b' &&
                   (j == 0 || !std::isalnum(expr[j-1])) &&
                   (j+1 >= expr.size() || !std::isalnum(expr[j+1]))) {
          result += srcB;
        } else {
          result += expr[j];
        }
      }
      oss << result << ";\n";
    }
  }

  // Write output (last slot)
  unsigned lastSlot = numExternalInputs + program.size() - 1;
  oss << "\n  out[idx] = s" << lastSlot << ";\n";
  oss << "}\n";

  return oss.str();
}

CompiledKernel NvrtcKernelCompiler::compile(
    const std::string& source,
    const std::string& kernelName) {
  nvrtcProgram prog;
  NVRTC_CHECK(nvrtcCreateProgram(
      &prog, source.c_str(), "fused_elementwise.cu", 0, nullptr, nullptr));

  // Detect GPU compute capability at runtime (driver API — no cuda_runtime.h)
  CUdevice device;
  cuCtxGetDevice(&device);
  int major = 0, minor = 0;
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  std::string archFlag =
      "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
  const char* opts[] = {archFlag.c_str(), "--use_fast_math"};
  nvrtcResult compileResult = nvrtcCompileProgram(prog, 2, opts);

  if (compileResult != NVRTC_SUCCESS) {
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    std::string log(logSize, '\0');
    nvrtcGetProgramLog(prog, log.data());
    nvrtcDestroyProgram(&prog);
    throw std::runtime_error(
        "[NvrtcKernelCompiler::compile] Compilation failed:\n" + log);
  }

  size_t ptxSize;
  NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
  std::string ptx(ptxSize, '\0');
  NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
  NVRTC_CHECK(nvrtcDestroyProgram(&prog));

  CompiledKernel kernel;
  CU_CHECK(cuModuleLoadDataEx(&kernel.module, ptx.data(), 0, nullptr, nullptr));
  CU_CHECK(cuModuleGetFunction(
      &kernel.function, kernel.module, kernelName.c_str()));

  return kernel;
}

CUfunction NvrtcKernelCompiler::getOrCompile(
    const std::vector<FusedOp>& program,
    unsigned numExternalInputs) {
  std::string key = makeKey(program, numExternalInputs);

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = cache_.find(key);
  if (it != cache_.end()) {
    return it->second.function;
  }

  std::string source = generateKernelSource(program, numExternalInputs);
  auto kernel = compile(source, "fused_elementwise");
  CUfunction func = kernel.function;
  cache_.emplace(key, std::move(kernel));
  return func;
}



// --- Fused Reduce Kernel Support ---

namespace {

const char* reductionIdentity(ReductionOp op) {
  switch (op) {
    case ReductionOp::Sum:
    case ReductionOp::Mean:
    case ReductionOp::CountNonzero:
      return "0.0f";
    case ReductionOp::Amin:
      return "INFINITY";
    case ReductionOp::Amax:
      return "-INFINITY";
    case ReductionOp::Any:
      return "0.0f";
    case ReductionOp::All:
      return "1.0f";
    case ReductionOp::Median:
    case ReductionOp::Std:
      return "0.0f"; // not directly fusable, fallback handles
  }
  return "0.0f";
}

const char* reductionAccumulate(ReductionOp op) {
  switch (op) {
    case ReductionOp::Sum:
    case ReductionOp::Mean:
      return "acc += val;";
    case ReductionOp::Amin:
      return "acc = fminf(acc, val);";
    case ReductionOp::Amax:
      return "acc = fmaxf(acc, val);";
    case ReductionOp::CountNonzero:
      return "acc += (val != 0.0f ? 1.0f : 0.0f);";
    case ReductionOp::Any:
      return "acc = (acc != 0.0f || val != 0.0f) ? 1.0f : 0.0f;";
    case ReductionOp::All:
      return "acc = (acc != 0.0f && val != 0.0f) ? 1.0f : 0.0f;";
    case ReductionOp::Median:
    case ReductionOp::Std:
      return "acc += val;"; // not directly fusable
  }
  return "acc += val;";
}

// Binary combine expression for warp shuffle reduction: reduce_combine(a, b).
const char* reductionCombine(ReductionOp op) {
  switch (op) {
    case ReductionOp::Sum:
    case ReductionOp::Mean:
    case ReductionOp::CountNonzero:
      return "(a + b)";
    case ReductionOp::Amin:
      return "fminf(a, b)";
    case ReductionOp::Amax:
      return "fmaxf(a, b)";
    case ReductionOp::Any:
      return "((a != 0.0f || b != 0.0f) ? 1.0f : 0.0f)";
    case ReductionOp::All:
      return "((a != 0.0f && b != 0.0f) ? 1.0f : 0.0f)";
    case ReductionOp::Median:
    case ReductionOp::Std:
      return "(a + b)"; // not directly fusable
  }
  return "(a + b)";
}

char reductionOpKey(ReductionOp op) {
  return '0' + static_cast<int>(op);
}

} // namespace

std::string NvrtcKernelCompiler::makeReduceKey(
    const std::vector<FusedOp>& program,
    unsigned numExternalInputs,
    ReductionOp reductionOp) {
  return "R" + std::string(1, reductionOpKey(reductionOp)) + "_" +
         makeKey(program, numExternalInputs);
}

std::string NvrtcKernelCompiler::generateReduceKernelSource(
    const std::vector<FusedOp>& program,
    unsigned numExternalInputs,
    ReductionOp reductionOp) {
  std::ostringstream oss;

  // Cooperative warp-shuffle reduction kernel.
  //
  // Each block handles one output element. Threads within the block
  // cooperatively reduce the reduction dimension:
  //   1. Each thread accumulates a subset of reduction elements (strided)
  //   2. Warp-level reduction via __shfl_down_sync
  //   3. Cross-warp reduction via shared memory
  //
  // This replaces the old per-thread sequential loop with parallel reduction,
  // giving ~32x speedup for large reduction dimensions.
  //
  // Kernel signature unchanged:
  //   fused_reduce(in0, ..., inN, out, num_output_elements,
  //                reduce_size, inner_size)
  // Launch config: grid = num_output_elements, block = BLOCK_SIZE

  const char* combineExpr = reductionCombine(reductionOp);

  oss << "#define INFINITY __int_as_float(0x7f800000)\n";
  oss << "#define BLOCK_SIZE 256\n";
  oss << "#define WARP_SIZE 32\n\n";

  // Generate the inline reduce_combine function
  oss << "__device__ __forceinline__ float reduce_combine(float a, float b) {\n";
  oss << "  return " << combineExpr << ";\n";
  oss << "}\n\n";

  oss << "extern \"C\" __global__\n";
  oss << "void fused_reduce(\n";
  for (unsigned i = 0; i < numExternalInputs; i++) {
    oss << "    const float* __restrict__ in" << i << ",\n";
  }
  oss << "    float* __restrict__ out,\n";
  oss << "    int num_output_elements,\n";
  oss << "    int reduce_size,\n";
  oss << "    int inner_size) {\n";

  // One block per output element
  oss << "  int out_idx = blockIdx.x;\n";
  oss << "  if (out_idx >= num_output_elements) return;\n\n";
  oss << "  int outer = out_idx / inner_size;\n";
  oss << "  int inner = out_idx % inner_size;\n";
  oss << "  int base = outer * reduce_size * inner_size + inner;\n\n";

  // Phase 1: Each thread accumulates over a strided subset of the reduction dim
  oss << "  float acc = " << reductionIdentity(reductionOp) << ";\n";
  oss << "  for (int r = threadIdx.x; r < reduce_size; r += BLOCK_SIZE) {\n";
  oss << "    int input_idx = base + r * inner_size;\n";

  // Load external inputs at input_idx
  for (unsigned i = 0; i < numExternalInputs; i++) {
    oss << "    float s" << i << " = in" << i << "[input_idx];\n";
  }

  // Execute elementwise program steps
  for (unsigned i = 0; i < program.size(); i++) {
    unsigned slotIdx = numExternalInputs + i;
    const auto& step = program[i];
    if (step.isUnary()) {
      std::string expr = unaryOpToExpr(step.unaryOp());
      std::string srcVar = "s" + std::to_string(step.src0);
      std::string result;
      for (size_t j = 0; j < expr.size(); j++) {
        if (expr[j] == 'x' &&
            (j == 0 || !std::isalnum(expr[j-1])) &&
            (j+1 >= expr.size() || !std::isalnum(expr[j+1]))) {
          result += srcVar;
        } else {
          result += expr[j];
        }
      }
      oss << "    float s" << slotIdx << " = " << result << ";\n";
    } else {
      std::string expr = binaryOpToExpr(step.binaryOp());
      std::string srcA = "s" + std::to_string(step.src0);
      std::string srcB = "s" + std::to_string(step.src1);
      std::string result;
      for (size_t j = 0; j < expr.size(); j++) {
        if (expr[j] == 'a' &&
            (j == 0 || !std::isalnum(expr[j-1])) &&
            (j+1 >= expr.size() || !std::isalnum(expr[j+1]))) {
          result += srcA;
        } else if (expr[j] == 'b' &&
                   (j == 0 || !std::isalnum(expr[j-1])) &&
                   (j+1 >= expr.size() || !std::isalnum(expr[j+1]))) {
          result += srcB;
        } else {
          result += expr[j];
        }
      }
      oss << "    float s" << slotIdx << " = " << result << ";\n";
    }
  }

  // Accumulate the last slot value
  unsigned lastSlot = numExternalInputs + program.size() - 1;
  oss << "    float val = s" << lastSlot << ";\n";
  oss << "    " << reductionAccumulate(reductionOp) << "\n";
  oss << "  }\n\n";

  // Phase 2: Warp-level reduction via __shfl_down_sync
  oss << "  // Warp-level reduction\n";
  oss << "  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {\n";
  oss << "    acc = reduce_combine(acc, __shfl_down_sync(0xffffffff, acc, offset));\n";
  oss << "  }\n\n";

  // Phase 3: Cross-warp reduction via shared memory
  oss << "  // Cross-warp reduction via shared memory\n";
  oss << "  __shared__ float warp_results[BLOCK_SIZE / WARP_SIZE];\n";
  oss << "  int lane = threadIdx.x % WARP_SIZE;\n";
  oss << "  int warp_id = threadIdx.x / WARP_SIZE;\n";
  oss << "  int num_warps = BLOCK_SIZE / WARP_SIZE;\n\n";
  oss << "  if (lane == 0) {\n";
  oss << "    warp_results[warp_id] = acc;\n";
  oss << "  }\n";
  oss << "  __syncthreads();\n\n";

  // First warp reduces across all warp results
  oss << "  if (warp_id == 0) {\n";
  oss << "    acc = (lane < num_warps) ? warp_results[lane] : "
      << reductionIdentity(reductionOp) << ";\n";
  oss << "    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {\n";
  oss << "      acc = reduce_combine(acc, __shfl_down_sync(0xffffffff, acc, offset));\n";
  oss << "    }\n";
  oss << "    if (lane == 0) {\n";

  // Finalize (e.g., mean divides by reduce_size)
  if (reductionOp == ReductionOp::Mean) {
    oss << "      out[out_idx] = acc / (float)reduce_size;\n";
  } else {
    oss << "      out[out_idx] = acc;\n";
  }

  oss << "    }\n";
  oss << "  }\n";
  oss << "}\n";

  return oss.str();
}

CUfunction NvrtcKernelCompiler::getOrCompileReduce(
    const std::vector<FusedOp>& program,
    unsigned numExternalInputs,
    ReductionOp reductionOp) {
  std::string key = makeReduceKey(program, numExternalInputs, reductionOp);

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = cache_.find(key);
  if (it != cache_.end()) {
    return it->second.function;
  }

  std::string source =
      generateReduceKernelSource(program, numExternalInputs, reductionOp);
  auto kernel = compile(source, "fused_reduce");
  CUfunction func = kernel.function;
  cache_.emplace(key, std::move(kernel));
  return func;
}

} // namespace fl
