# Flashlight modernization — architecture notes

How the modernized tensor stack fits together, and the invariants that are easy to break.
Detailed maps with file:line references (JIT core flow, optimizer pipeline, op dispatch
and library map, forced-evaluation catalog, files to touch when adding things) are in
[research/review-2026-10-26.md §6](research/review-2026-10-26.md). Design rationale is in
`TODO.md` ("Design Document: JIT-Only Execution Model", "Hardware Backend Architecture",
"Stream Semantics and Async Execution").

---

## 1. Layers

```
fl::Tensor API (Tensor.h: free functions, operators, indexing, autograd hooks)
   │
   ▼
TensorAdapter ── JitTensorBase (default)  ──►  JitBackend ── builds IR nodes
   │              (lazy; SharedData holds a node        │
   │               + an indexing/view chain)            ▼
   │                                              IR (flashlight/fl/tensor/ir/)
   │                                                    │ fl::eval
   │                                                    ▼
   │                                              Optimizer (opt/): generic passes,
   │                                              backend compound-op passes,
   │                                              fusion, backend lowering passes
   │                                                    │
   │                                                    ▼
   │                                              Evaluator (eval/): plan, allocate per
   │                                              node, dispatch, free dead values,
   │                                              compact root to Value/MultiValueNode
   │                                                    │ BackendDispatch (function table)
   ▼                                                    ▼
bare adapters ── CPUTensor / CudaTensor / ArrayFireTensor / StubTensor
                 TensorBackend implementations:
                   CPUBackend  → ops/onednn (OneDnnOps), ops/native
                   CudaBackend → ops/cuda (Thrust/CUB/custom), ops/cudnn, cuBLAS(Lt), NVRTC
                   ArrayFireBackend (opt-in baseline) → ArrayFire, ops/cudnn or ops/onednn
runtime (flashlight/fl/runtime/): Device, Stream (AsyncCPUStream, CUDAStream,
SynchronousStream), Event, Future, fl::copy
autograd (flashlight/fl/autograd/): AutogradMeta on Tensor, autograd:: functions
(Functions.cpp), Tensor-level hooks (TensorAutograd.cpp)
```

- **Default tensor type** (`DefaultTensorType.h`, thread-local): JIT over the
  compile-time default backend, in order CUDA > CPU > ArrayFire > Stub, resolved lazily.
  Tests select a backend with `setDefaultJitTensorType<CPUTensor|CudaTensor>()` or, for
  the ArrayFire baseline, bare `setDefaultTensorType<ArrayFireTensor>()` (`TestInit.h`).
- **IR node types** (`ir/NodeType.h`): Binary, Unary, Cast, Clip, Concatenate, Cumsum,
  Custom, Scalar, Value, Index, IndexedUpdate, Flip, Roll, Reshape, Reduction,
  Transpose, Triangular, Tile, Pad, Where, Matmul, Conv2d, Pool2d, BatchNorm,
  Projection, MultiValue, Transfer, FusedElementwise, FusedReduceElementwise.
  Conv2d/Pool2d/BatchNorm are first-class; their backward ops are `CustomNode`s; RNN
  forward/backward still materialize (barrier).
- **Optimizer** (`opt/Optimizer.cpp`, once per eval, no fixed point):
  1. generic simplification — `ConvBatchNormFolding`, `TransposeSimplification`,
     `ScalarFolding`, `AlgebraicSimplification`, `CommonSubexpressionElimination`;
  2. backend compound-op passes (CUDA: `CudnnSubgraphLowering`,
     `CublasLtBackwardEpilogueFusion`; oneDNN: `OneDnnPrimitivePostOpFusion`, `OneDnnOpFusion`);
  3. generic fusion — `ElementwiseFusion`, `ReductionOfElementwiseFusion`;
  4. backend lowering (CUDA: `CudaFusedKernelLowering` → NVRTC).
  Backend passes come from a `JitOptimizerExtension` registered per backend type; a
  backend without one (ArrayFire) gets only the generic passes, and fused elementwise
  nodes are then evaluated op by op. After every pass the optimizer verifies the root's
  shape/dtype and the graph (cycles).
- **Evaluator** (`eval/Evaluator.cpp`): builds a plan, allocates each node's output
  buffer just before dispatch (views, projections, transfers, custom, indexed-update and
  scalar nodes get none), dispatches through `BackendDispatch`, frees values by use
  count; `JitTensorBase::eval` compacts the root to a `ValueNode`/`MultiValueNode`.

## 2. Invariants and pitfalls

**JIT and IR**
- The optimizer rewrites shared nodes **in place** during `fl::eval`, and the root is
  compacted afterwards: the forward graph does not survive evaluation. Any VJP design
  must account for this (P-2).
- Passes treat already-evaluated nodes as leaves (`isEvaluated` in `opt/Pass.h`);
  otherwise evaluated history is re-optimized and recomputed (including `rand`).
- Inside evaluation code and `CustomNode` lambdas, never create `Tensor(shape, dtype)`
  — that makes a JIT tensor and causes nested evaluation. Use bare helpers:
  `makeCPUTensor`/`fullCPU` (`ops/onednn/Utils.h`), `toTensor<CudaTensor>(…)`, or
  `detail::makeTensorLike`/`fullLike` (`ops/cudnn/CudnnTensorHelpers.h`).
- At the `JitBackend` dispatch boundary, inputs are materialized with
  `materializeSafe()` (const inputs) or `materializeMutable()` (non-const references).
- One `JitBackend` per backend type is shared process-wide; evaluation is **not**
  thread-safe today (P-4).
- `Tensor` copies are clones that share `AutogradMeta` but not data (P-1).

**Streams**
- Streams are stored per node, but `BackendDispatch` takes no stream: CPU ops run on the
  left operand's `AsyncCPUStream`; CUDA ops run on the `CudaBackend` singleton stream
  (the node stream only affects CUDA buffer allocation — a known hazard).
- All `CudaTensor`s share `CudaBackend`'s stream (`getDefaultCudaStream()`); per-tensor
  streams leaked memory (`cudaFreeAsync` on the wrong stream).
- `AsyncCPUStream` errors are sticky: the worker survives an exception, skips queued
  tasks, and `sync()` reports the error; the stream is usable afterwards. Async tasks
  must own every buffer they touch, including pre-assigned outputs.

**CUDA libraries**
- TF32 is controlled by one internal switch, `fl::cuda::detail::allowTF32()`
  (`ops/cuda/CudaNumerics.h`); graph-API engine selection ignores
  `NVIDIA_TF32_OVERRIDE`, so `restrictToStrictMath` filters engines explicitly.
- `ops/cudnn` is shared by `CudaBackend` and ArrayFire-CUDA: allocate on the input's
  backend (`makeTensorLike`), never unconditionally as `CudaTensor`.
- cuDNN subgraph lowering conventions: matmul transposes are expressed through strides
  (swap the last two dims/strides) for boundary inputs and in-subgraph scalars alike;
  matmul subgraphs use 3-D shapes with leading-1 padding (`shapeTo3D`), conv/batchnorm
  4-D with trailing-1 padding (`shapeTo4D`); `fl::full` creates `ScalarNode`s that end
  up inside subgraphs and must be handled there; the zero-bias sentinel must check
  `scalar == 0`, not "is a scalar"; boolean boundary inputs must not be absorbed (b8 maps
  to INT8 and corrupts data), so casts from non-float types stay outside subgraphs.

**Layout and dtypes**
- Row-major NCHW; layout by strides, not tags.
- Binary ops promote with `promoteDtypes` on every backend, matching the IR.
- Unary math ops keep the input dtype (`isnan`/`isinf`/`logicalNot` return b8);
  backends without a native kernel for a dtype compute in f32 and cast back (D-12).

## 3. Directory map (tensor stack)

| Path | Contents |
|---|---|
| `flashlight/fl/tensor/Tensor.{h,cpp}` | public tensor API, free functions, indexing, autograd hooks |
| `flashlight/fl/tensor/TensorBackend.h`, `BackendDispatch*.{h,cpp}` | backend interface, evaluator dispatch table |
| `flashlight/fl/tensor/DefaultTensorType.h`, `TensorAdapter.{h,cpp}` | default type selection, adapters |
| `flashlight/fl/tensor/JitBackend*.cpp`, `JitTensorBase.{h,cpp}` | JIT frontend: node construction, views, materialization |
| `flashlight/fl/tensor/ir/` | IR nodes |
| `flashlight/fl/tensor/opt/` | optimizer, passes, `GraphVerifier.h`, backend extensions under `opt/backends/{cuda,onednn}` |
| `flashlight/fl/tensor/eval/` | evaluator |
| `flashlight/fl/tensor/backend/{cpu,cuda,af,stub}/` | backend + tensor adapters |
| `flashlight/fl/tensor/ops/{onednn,native,cuda,cudnn}/` | library ops (free functions) |
| `flashlight/fl/runtime/` | devices, streams, events, futures, copy |
| `flashlight/fl/autograd/` | autograd metadata and functions |
| `flashlight/fl/test/` | tests; `TestInit.h` selects the backend per lane |
