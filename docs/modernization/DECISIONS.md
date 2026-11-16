# Flashlight modernization — decision records

Each record states the context, the decision, its consequences, and where the details
live. Status is one of **Accepted**, **Reverted/Rejected**, **Deferred**, or **Pending**.
Earlier design decisions (2026-03 and before) are summarized from `TODO.md`; their full
reasoning stays there.

Index: D-01…D-12 decided · P-1…P-9 pending/open.

---

## Decided

### D-01 — JIT-only execution model *(Accepted, before 2026-10)*
- **Decision:** there is no eager backend. Every tensor is a lazy `JitTensorBase` over an
  IR node; `fl::eval` optimizes and evaluates; data accessors (`scalar`, `host`,
  `device`, `toString`) materialize. The host thread plans; streams execute. Per-op
  streams are captured at graph construction. Library ops write into planner-provided
  output buffers. `CustomNode` is the JIT ↔ library interface for lowered/backward ops.
- **Details:** `TODO.md`, "Design Document: JIT-Only Execution Model" (its "Decisions
  Summary" table lists ~20 sub-decisions).
- **Later corrections (2026-10-26):** buffers are now allocated per node just before
  dispatch rather than all up front; passes treat evaluated nodes as leaves.

### D-02 — Row-major NCHW everywhere *(Accepted, Phase 2)*
Tensors use row-major strides and NCHW ordering; the ArrayFire backend reverses
dimensions internally. Layout is expressed by strides, not layout tags (`TODO.md`,
"Tensor Layout Philosophy: Strides, Not Tags").

### D-03 — Merge `Variable` into `Tensor` ("Option A") *(Accepted, Phase 2.5)*
Gradients live in `AutogradMeta` attached to `Tensor`.
**Open consequence:** `Tensor` copies share `AutogradMeta` but not data, so optimizers
rebind their own copies and training is a no-op. Tracked as P-1.

### D-04 — Backends stratified by hardware; ArrayFire is legacy *(Accepted, before 2026-10)*
One `TensorBackend` per hardware target that dispatches internally to the best library
per op (CUDA: cuBLAS/cuDNN/Thrust/NVRTC; CPU: oneDNN + native). Do not design new
architecture around ArrayFire. (`TODO.md`, "Hardware Backend Architecture".) Refined by
D-09.

### D-05 — Async materialization and data transfer *(Accepted, 2026-03-27/31)*
Unified `Event`; minimal `fl::Future<T>` (`get()` evaluates as a fallback, `isReady()`
never blocks, no callbacks); `fl::copy` lives in the runtime, not in `BackendDispatch`;
`TransferNode` is the IR node for data movement (transport chosen at runtime);
`scalar<T>()` = transfer + host read; explicit `fl::eval`, no global step; collectives
are separate node types. FusionSideOutput is deferred until functional VJP removes
external uses. Record: [research/async-materialization-design.md](research/async-materialization-design.md).

### D-06 — Working process *(Accepted, 2026-10-26)*
- `TODO.md` is append/annotate-only: dated inline corrections; past records and open
  items are never deleted.
- Show a plan and get agreement before anything beyond a local bug fix — public headers,
  general APIs, IR node fields, the evaluator, pass infrastructure, semantics (dtype
  rules, error behavior, allocation timing), or test infrastructure. Flag such changes
  explicitly instead of folding them into "bug fixes".
- Backend-specific properties stay inside backend/library components.

### D-07 — CPU thread count comes from the environment *(Accepted, 2026-10-26)*
- **Context:** OpenMP defaults to every logical CPU; tiny CPU ops became ~100× slower on a
  384-thread machine.
- **Decision:** Flashlight sets no thread count; `OMP_NUM_THREADS` and other OpenMP
  variables decide. Tests get an opt-in CMake cache option `FL_TEST_NUM_THREADS`
  (default empty) that sets `OMP_NUM_THREADS` for ctest runs (`cmake/TestUtils.cmake`;
  `README.md`). Running a binary directly inherits the caller's environment.
- **Follow-up:** a size-aware policy (few threads for small ops) belongs to the future
  execution/scheduling layer, which knows op sizes. Note: OpenMP settings are per
  thread, so any future policy must apply on the `AsyncCPUStream` worker.

### D-08 — Strict f32 on CUDA; TF32 is opt-in and hidden in CUDA internals *(Accepted, 2026-10-26)*
- **Context:** cuDNN graph-API engines ran f32 matmuls as TF32 (10-bit mantissa),
  breaking gradient checks; `NVIDIA_TF32_OVERRIDE` does not affect graph-API engine
  selection.
- **Decision:** strict f32 by default; `FL_ALLOW_TF32=1` (read once) opts in. One
  internal switch, `fl::cuda::detail::allowTF32()` in `ops/cuda/CudaNumerics.h`, drives
  cuDNN engine filtering (`ops/cudnn/CudnnNumerics.h::restrictToStrictMath`), the cuDNN
  RNN math type, the cuBLAS handle math mode, and the cuBLASLt compute type.
- **Rejected:** a `MathMode` field on `MatmulNode`/`Conv2dNode` with an evaluator
  wrapper (implemented, then reverted): backend math knobs must not appear in generic IR,
  the evaluator, or `fl::` APIs. Any programmatic control would live on `CudaBackend`.
- **Deferred:** per-op / per-dtype runtime control (P-6).

### D-09 — Native backends are the defaults; ArrayFire is opt-in and the baseline *(Accepted, 2026-10-26)*
- **Decision:** `FL_USE_ARRAYFIRE` defaults OFF; `FL_USE_CPU` defaults ON;
  `FL_USE_NATIVE_CUDA` defaults ON when `FL_USE_CUDA=ON`. Compile-time default order:
  CUDA > CPU > ArrayFire > Stub (`DefaultTensorType.h`). Gloo defaults on only for
  CPU-only builds. ArrayFire code is kept and builds with `-DFL_USE_ARRAYFIRE=ON`.
- **Baseline role:** ArrayFire is the support/performance bar for the native compiler
  path, so development builds keep it enabled (not default). The baseline runs bare
  ArrayFire tensors (no JIT) — autograd works without the JIT — via the `_AF` test
  variant. If ArrayFire is ever wrapped in the JIT, the JIT must act as a passthrough
  (generic simplifications only); ArrayFire's own JIT is the bar.
- **Consequences:** `pkg/vision` tensor extensions, `pkg/halide`, the ArrayFire memory
  manager tests and OpenCL/macOS GPU support are opt-in with ArrayFire. Open follow-ups:
  benchmark backend selector; measure the evaluator's wasted per-node preallocation on
  the ArrayFire path.

### D-10 — Shared cuDNN ops allocate on the caller's backend *(Accepted, 2026-10-26)*
`ops/cudnn` is shared by `CudaBackend` and ArrayFire-CUDA. Its outputs and workspaces are
allocated with `detail::makeTensorLike(input, …)` / `fullLike(input, …)`
(`ops/cudnn/CudnnTensorHelpers`), which use the backend of the op's input and throw on
JIT tensors (cuDNN ops must only see materialized backend tensors).
`DropoutDescriptor(prob, like)` takes the reference tensor for its state buffer.

### D-11 — Indexing on two primitives: `gather` and `scatter` *(Accepted, 2026-11-02; implementation pending)*
- **Decision:**
  - `gather(input, index, axis)` — take-along-axis semantics: index has input's rank, its
    other axes broadcast against input's; output shape = that broadcast shape.
  - `scatter(input, index, source, axis, mode)` with `mode ∈ {Set, Add}` — returns a copy
    with `source` written/accumulated at `index`; Add accumulates repeated indices, Set
    keeps one (unspecified which).
  - These replace `TensorBackend::indexAdd` and the tensor-index code in every
    backend's `index()`; `IndexNode`/`IndexedUpdateNode` become static-only.
  - `x(...)` with one tensor index follows NumPy/PyTorch (index shape kept; a 0-d index
    drops its axis; no shape-based "flat" guess — use `flat()`).
  - Boolean masks: reads = `nonzero` + `gather`; writes = `where`.
  - Two or more tensor indices throw (semantics deferred, P-7).
  - `x(idx) += v` keeps NumPy's no-accumulate semantics; accumulation is explicit
    `scatter(..., Add)`.
  - ArrayFire implements the same primitives, so the baseline follows the new semantics.
- **Rationale:** smallest operator set that covers every in-repo use (embedding,
  adaptive softmax/embedding, cross-entropy, DETR, masked fill) with static output shapes
  and simple gradients (gather ↔ scatter-add).
- **Plan:** [plans/indexing-gather-scatter-plan.md](plans/indexing-gather-scatter-plan.md).
  Supersedes in scope the broader `scatterReduce`/`indexPut` design in `TODO.md`
  ("Generalized Advanced Indexing Operations").

### D-12 — Low-precision math fallbacks compute in f32 *(Accepted, 2026-10-27)*
When a backend has no native kernel for a dtype (e.g. CPU `sin` on bf16), it computes in
f32 and casts back to the dtype the IR expects. This is a per-backend, per-op fallback,
not a policy: backends with native low-precision kernels should use them, and the choice
never appears in the IR. Follow-up TODO: a single-pass convert-compute-convert loop (or
JIT fusion of the surrounding casts) instead of two extra full-tensor passes.

---

## Pending / open

### P-1 — Tensor value vs alias semantics (training is a no-op) *(Pending, highest priority)*
Copies of a `Tensor` share gradients but not data, so optimizer updates never reach
module parameters; `detach()` returns a reference into a side list; shared-meta
surprises. Needs an explicit semantics decision (review §4.1) and a test that
parameters change across steps (`OptimTest.DISABLED_StepUpdatesModuleParams`).

### P-2 — VJP graph-retention model *(Pending)*
Evaluation optimizes in place and compacts the forward graph, so a VJP walk after eval
sees fused/lowered/compacted nodes. Options: functional `valueAndGrad(f)` that builds
forward + backward before eval (JAX/MLX); retaining recipes through eval; or tape + VJP
rules first. Review §5; research report §1.

### P-3 — Compaction policy for long-lived holders *(Pending)*
BatchNorm running-statistics chains and evaluated history kept alive by user tensors.

### P-4 — JIT thread safety *(Pending)*
One shared `JitBackend` per backend type; passes hold per-run state as members; the
optimizer rewrites shared nodes in place. Options: serialize evaluation per backend;
per-thread optimizer/evaluator state plus a graph-level lock; per-thread JIT backends.
Repro: `DatasetTest.FileBlobDataset`.

### P-5 — Simplification / deletion scope *(Pending)*
Review §4 lists ~1.2k lines deletable now (dispatch trampolines, forwarding wrappers,
unused named CPU ops, dead f64 fallbacks, an unused `CudnnPlanCache` or wire it up) and
the modifiability cost of adding an IR node (~14 files) or backend op (~22 files).

### P-6 — Per-op math-mode control *(Deferred)*
If needed, it must live inside CUDA library components or on `CudaBackend`, never on
generic IR, the evaluator, or `fl::` APIs.

### P-7 — Multi-tensor indexing semantics *(Deferred)*
NumPy/PyTorch pointwise semantics is the likely target (DETR already needs it);
punted until gather/scatter land.

### P-8 — Size-aware CPU thread policy *(Deferred)*
Belongs in the future scheduling layer (see D-07).

### P-9 — Small open items awaiting a yes/no
- Add a `_CUDA` variant for the Contrib model tests (coverage gap; one CMake line).
- Backend selector for `app/benchmark` and recorded ArrayFire numbers.
- Whether to keep `.claude/plans/` copies once these docs are checked in (they are now
  duplicated here).
