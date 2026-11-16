# Flashlight modernization — work log

A chronological record of the modernization effort: what was done, what was found,
what was decided, and what is still open. Written so the work can be recovered and
continued from the repository alone.

- Current state and next steps: [STATUS.md](STATUS.md)
- Decision records (accepted, rejected, pending): [DECISIONS.md](DECISIONS.md)
- How the modernized stack fits together: [ARCHITECTURE.md](ARCHITECTURE.md)
- Building, testing, debugging: [DEVELOPMENT.md](DEVELOPMENT.md)
- Itemized roadmap and task list (the working checklist): [`TODO.md`](../../TODO.md)
- Research reports and the full 2026-10 review: [research/](research/README.md)

Conventions used in this log: dates are when work happened; "the working copy" means
the uncommitted modernization changes on top of base commit `f5f717f3`; file paths are
relative to the repository root; test names are `Suite.Test` and test binaries carry
a backend suffix (`_CPU`, `_CUDA`, `_AF`, or none for the compile-time default).

---

## 0. Goals and principles

From the project brief at the top of `TODO.md` and the Flashlight paper
(arXiv:2201.12465; LaTeX sources were kept locally as `flashlight-paper/`):

- Bring Flashlight into the modern era after years of inactivity, in the spirit of the
  original project: a small, readable, modifiable C++ codebase with low overhead and an
  expressive, minimal API ("total internal modifiability").
- Target large-scale training with complex schedules. People and models should both be
  able to read the code and build abstractions and computation schedules on top of it.
- Do not diverge from the original philosophy. Keep the operator set and abstractions
  small; prefer explicit, predictable behavior.

---

## 1. Work before 2026-10 (summary)

Detailed checklists for these phases are in `TODO.md` (Phase 0 through Phase 6 and the
design documents that follow them). In brief:

**Phase 0 — defaults.** CMake 3.24, C++20 (CUDA C++20), `CMAKE_CUDA_ARCHITECTURES`
defaulting to `native`; dead version branches and options removed; cuDNN v9 RNN API
migration; full test pass.

**Phase 1 / 1.5 — `fl::Tensor` and backend cleanup.** Broadcasting audit; scalar
overload cleanup; autograd-only backend ops (conv2d, pool2d, batchnorm, rnn) moved
into `TensorBackend`; `AutogradPayload` replaced by an opaque `ForwardBackwardState`.

**Phase 2 — row-major migration.** The whole stack moved from ArrayFire's column-major
WHCN convention to row-major NCHW (cuDNN descriptors, matmul, autograd functions, NN and
contrib modules, tests); the ArrayFire backend reverses dimensions internally.

**Phase 2.5 — API cleanup, Variable → Tensor.** Autograd `Variable` ops brought to parity
with tensor ops and then merged into `Tensor` ("Option A"): gradients live in
`AutogradMeta` on `Tensor` (`flashlight/fl/autograd/AutogradMeta.h`,
`TensorAutograd.cpp`). Explicit dtype promotion rules; `TensorBase.h` renamed to
`Tensor.h`.

**Phase 3 — dispatcher.** Thread-local `DefaultTensorType` (`setDefaultTensorType`,
`withTensorType`, `setDefaultJitTensorType`); backend registry
(`TensorBackendType → TensorBackend&`).

**Phase 4 — backend and ops refactor.** Library code organized as
`flashlight/fl/tensor/ops/{onednn,native,cuda,cudnn}/`; a generic `CPUTensor` holding
a raw aligned buffer (oneDNN memory constructed on the fly); `OneDnnOps` owns the
oneDNN engine/stream; host fallbacks consolidated in `ops/native/`.

**Phase 5a — native CUDA backend.** `CudaTensor` (stream-ordered allocation, row-major
strides, offset views) and `CudaBackend` (cuBLAS, cuBLASLt, cuRAND, Thrust/CUB kernels,
cuDNN for conv/pool/batchnorm/RNN), with GPU-native reductions, sort, cumsum and
multi-dtype support.

**Phase 5b — JIT IR and fusion; JIT-only execution.** The JIT became the only execution
model (design document "JIT-Only Execution Model" in `TODO.md`): every tensor is a
`JitTensorBase` over an IR node; `fl::eval` runs the optimizer and evaluator; an
evaluator dispatches through a function-pointer table (`BackendDispatch`). Generic
passes (`ConvBatchNormFolding`, `TransposeSimplification`, `ScalarFolding`,
`AlgebraicSimplification`, `CommonSubexpressionElimination`, `ElementwiseFusion`,
`ReductionOfElementwiseFusion`) plus backend passes (CUDA: `CudnnSubgraphLowering`,
`CudaFusedKernelLowering` via NVRTC, `CublasLtBackwardEpilogueFusion`; oneDNN:
`OneDnnOpFusion`, `OneDnnPrimitivePostOpFusion`). Many former `CustomNode` ops became
first-class IR nodes (Tile, Where, Clip, Flip, Triangular, Cumsum, Concatenate, Pad,
Roll; Conv2d, Pool2d, BatchNorm with `ProjectionNode` outputs). `MultiValueNode`
compaction for multi-output evaluation.

**Async execution and data movement (designed 2026-03-27/31).** `AsyncCPUStream`, a
unified `Event`, a minimal `fl::Future<T>` (`get()`, `isReady()`), runtime-level
`fl::copy`, and a `TransferNode` IR node; `scalar<T>()` is a transfer plus a host read.
Design record: [research/async-materialization-design.md](research/async-materialization-design.md).

**Research by 2026-05.** VJP rule design across JAX/PyTorch/MLX/tinygrad/TVM
([research/vjp-design-research.md](research/vjp-design-research.md)); how production IRs
treat tile and broadcast ([research/tile-broadcast-research.md](research/tile-broadcast-research.md)).

**2026-06-17 triage.** Full `_CPU`/`_CUDA` run; recorded failures (CPU gelu hang,
rank-≥2 broadcast, `fl::copy` "Unknown dtype", CUDA Jacobian OOM, cuBLASLt DRELU
Jacobians, CPU GLU). The record is preserved in `TODO.md` with 2026-10-26 corrections
inline: several root causes had been misattributed.

---

## 2. 2026-10-26 — research, full review, bug fixes

### 2.1 Research and review
- Re-read the paper, `TODO.md`, and prior work; ran two external surveys:
  autodiff over lazy graphs, effects/aliasing, memory planning, execution, dynamic
  shapes, pass infrastructure
  ([research/research-2026-10-autodiff-compilers.md](research/research-2026-10-autodiff-compilers.md));
  distributed/communication in IR, pipeline schedules, MoE, kernel authoring,
  attention libraries, low precision
  ([research/research-2026-10-distributed-kernels.md](research/research-2026-10-distributed-kernels.md)).
  A digest is in `TODO.md` ("External research digest").
- Full re-run of the test suite plus four parallel static reviews (JIT core/runtime,
  optimizer passes, autograd + VJP readiness, CPU/CUDA backends), with the top claims
  reproduced by scratch probes. Report, with file:line evidence, verification tags and
  architecture maps: [research/review-2026-10-26.md](research/review-2026-10-26.md).

**Headline findings** (review §1):
1. **Training is a no-op.** Optimizers never update module parameters: `Tensor`
   copies are clones that share `AutogradMeta` but not data, so optimizers rebind their
   own copies. Tests passed vacuously. Needs a Tensor value-vs-alias decision (still
   open; see DECISIONS P-1).
2. **CPU async lifetime race → heap corruption** (enqueued oneDNN tasks did not own
   their pre-assigned output buffers; concat freed its contiguous copies early).
3. **`AlgebraicSimplification` identity rules ignored shape/dtype** — the real cause of
   the rank-≥2 broadcast failure (`e + full({2,3,4}, 0)` collapsed to `e`).
4. **`fl::copy` same-device bug** (null node streams defeated the `TransferNode`
   no-op; `CudaBackend::copy` allocated pinned host memory for CUDA→CUDA).
5. **CPU gelu hang** — a self-cycle created by `OneDnnPrimitivePostOpFusion`, turned
   into a spin by missing visited-marking in `OneDnnOpFusion`.
6. **ArrayFire initialized in every `_CPU` test; the ArrayFire default path
   segfaulted** (`jitStream()` returned nullptr).
7. CPU GLU (strided view reaching oneDNN eltwise), CUDA dtype drift vs IR metadata,
   cuBLASLt/cuDNN Jacobian failures, CUDA Jacobian OOM/slowness.

### 2.2 Bug-fix pass (no design changes)
All in the working copy; itemized with file references in `TODO.md` ("Fixes applied
2026-10-26"). Summary:
- **Optimizer passes:** identity rules require same shape+dtype; the optimizer checks
  root shape/dtype after every pass (`verifyRootMetadata`); `GraphVerifier` detects
  cycles; gelu self-cycle fixed (`findAnchorChain`, shared-anchor guard, rewrite
  guard, f64 demotion skips bias/residual); `OneDnnOpFusion` visits each node once and
  resets state per apply; `TransposeSimplification` folds into reductions only when
  axis order survives; CSE keys scalars exactly; `ScalarFolding` gives comparisons b8
  and skips integer x/0 and MIN/−1; `ConvBatchNormFolding` builds defaults on the bare
  backend; cuDNN lowering refuses transposed operands it cannot represent; cuBLASLt
  backward epilogues restricted to 2-D (BGRADB disabled behind a flag, DGELU requires
  the derivative's Exp and aux width % 8); NVRTC lowering only for f32/f16/bf16/b8.
- **Pass infrastructure:** passes treat evaluated nodes as leaves (`isEvaluated` in
  `opt/Pass.h`), so evaluated history is not re-optimized or recomputed.
- **Evaluator/JIT core:** output buffers allocated just before dispatch (not all up
  front); Custom/IndexedUpdate/Scalar nodes get none; `buildPlan` no longer reads a
  popped entry; `JitTensorBase::stream()` BFS has a visited set; lazy `device()` on a
  view materializes it and shares the node's buffer; `getAdapter<T>()` asserts its type
  in debug builds; same-device `fl::copy` returns the source node and device
  destinations are allocated on the target backend.
- **Runtime:** `AsyncCPUStream` has sticky error semantics (worker survives
  exceptions, skips tasks until `sync()` reports, stream usable afterwards);
  `DefaultTensorType` resolves the compile-time default lazily (`defaultJitPending`);
  `ArrayFireBackend::jitStream()` returns the wrapped device stream.
- **CPU backend:** async tasks keep pre-assigned outputs alive
  (`keepAliveIfPreassigned`); concat keeps its copies alive; strided inputs made
  contiguous for eltwise and native ops; `where` promotes like the IR; literal index
  bounds; `nonzero`/`var` produce bare tensors inside eval; empty-input guards; CPU
  BatchNorm training returns batch statistics and supports momentum (cuDNN convention).
- **CUDA backend:** arithmetic/bitwise ops return `promoteDtypes(lhs, rhs)`; `where`
  dtype fix; flip/roll element-size dispatch (no overrun); `empty()` allocates on the
  backend stream.
- **Autograd:** `withoutData()` is a lazy constant (no full-size allocation per
  recorded input).
- **TF32 discovery:** f32 matmuls in cuDNN graph-API subgraphs ran on TF32 engines
  (`NVIDIA_TF32_OVERRIDE` does not reach graph-API engine selection) — the real cause of
  the stacked relu Jacobian failures. Led to decision D-08.
- **Tests:** Jacobian helper rewritten (host-built perturbations and cotangents, one
  transfer each, inputs never mutated); stale tests updated to row-major/broadcast
  semantics; new regression tests for every fix; `OptimTest.DISABLED_StepUpdatesModuleParams`
  documents the training bug; `EventTest.ReRecord` made deterministic.
- **Verification at the time:** 1,482 tests passing across 85 binaries.
  *Correction (2026-10-26, later):* that runner only reached binaries whose gtest
  suite name matched the binary name, and the unsuffixed binaries were then running on
  the ArrayFire default, where shared cuDNN ops were broken. Both were found and fixed
  later the same day (§3.4, §3.6).

### 2.3 Process rules agreed
- `TODO.md` is never pruned: corrections are added inline as dated notes; past records
  and open items stay.
- Show a plan before any change beyond a local bug fix (public headers, IR, evaluator,
  pass infrastructure, semantics, test infrastructure), and keep backend-specific knobs
  inside backend internals. See DECISIONS D-06.
- A friction log of slowdowns and painful workflows was started (`TODO.md`,
  "Friction & slowdowns log"): OpenMP oversubscription, CPU `relativeSync` blocking the
  host per op, CPU deep copies, Thrust `par.on` synchronizing, cuDNN plans rebuilt per
  eval, slow Jacobian tests, missing debug switches (graph dump, pass disabling,
  synchronous CPU stream), no ASAN/TSAN configs.

---

## 3. 2026-10-26 — decisions, ArrayFire, parity

Decisions were taken one at a time, least consequential first.

### 3.1 OpenMP threads (D-07)
Measured: with OpenMP's default of every logical CPU (384 on the development machine),
a 64² CPU matmul took 11.3 ms vs 0.1 ms at 16 threads; `AutogradUnaryOpsTest_CPU.Clip`
took 244 s vs 0.6 s. Decision: the library sets no thread count; tests get an opt-in
CMake cache option `FL_TEST_NUM_THREADS` that sets `OMP_NUM_THREADS` for ctest runs
(`cmake/TestUtils.cmake`, documented in `README.md`). A size-aware policy belongs to the
future scheduling layer.

### 3.2 TF32 (D-08)
Strict f32 by default on CUDA; `FL_ALLOW_TF32=1` opts in (read once via
`fl::getEnvVar`). One switch drives cuDNN engine filtering
(`ops/cudnn/CudnnNumerics.h::restrictToStrictMath`, used by `CudnnSubgraphLowering` and
the three conv plan builds), the cuDNN RNN math type, the cuBLAS handle math mode and the
cuBLASLt compute type. Verified: relu Jacobians 5/5 strict, 1/5 with TF32.
- A follow-up attempt at per-op runtime control put a `MathMode` field on
  `MatmulNode`/`Conv2dNode` and a math-mode wrapper in the evaluator. **Reverted** at the
  maintainer's request: backend math knobs must not appear in generic IR, the evaluator
  or `fl::` APIs. Per-op control is deferred (P-6).
- The switch then moved out of the public `runtime/CUDAUtils.h` into the internal,
  header-only `ops/cuda/CudaNumerics.h` as `fl::cuda::detail::allowTF32()`
  (header-only because `ops/cudnn` also builds for ArrayFire-CUDA without the native
  backend).

### 3.3 ArrayFire's role (D-09)
Facts gathered: a stock build defaulted to ArrayFire (`FL_USE_ARRAYFIRE` ON, default
order AF > CPU > Stub, `FL_USE_NATIVE_CUDA` OFF and `CudaTensor` not selectable as the
default); the top-level `FL_USE_CPU` OFF declaration shadowed the tensor-level ON
default; ArrayFire was a hard configure dependency; the README documented a nonexistent
`FL_BUILD_ARRAYFIRE`. Options: keep default / native default with AF built / native
default with AF opt-in / remove. **Chosen: native backends are the defaults; ArrayFire is
opt-in but kept as the reference baseline.**
- CMake: `FL_USE_ARRAYFIRE` OFF; `FL_USE_CPU` ON (duplicate top-level declaration
  removed); `FL_USE_NATIVE_CUDA` defaults ON whenever `FL_USE_CUDA=ON`
  (`cmake_dependent_option`); Gloo defaults on only for CPU-only builds (CPU+CUDA would
  otherwise default both NCCL and Gloo on). Default order in `DefaultTensorType.h`:
  CUDA > CPU > ArrayFire > Stub. README build docs updated.
- Verified: fresh stock (CPU-only) and `-DFL_USE_CUDA=ON` configures never look for
  ArrayFire; the stock CPU-only build compiles and its default-backend suites pass.
- Full run with ArrayFire off and CUDA as default: 101 binaries, 1,663 tests passing,
  48 skipped, 2 problem binaries (`AutogradTest`: known CUDA advanced-index crash plus
  sticky-error fallout; `DatasetTest`: JIT thread-safety race, §3.7).

### 3.4 ArrayFire as the baseline (D-09, D-10)
The maintainer clarified that ArrayFire's main value now is as the bar the native
compiler path must match for support and performance, and that wrapping it in the JIT is
acceptable only as a passthrough.
- Findings: ArrayFire registers no backend optimizer extension, so JIT-over-ArrayFire
  only gets generic passes; generic elementwise fusion is evaluated op by op, so
  ArrayFire's own JIT still sees every op. Distortion found: the evaluator pre-allocates
  an output per node and ArrayFire ops discard it (to be measured).
- Probe: autograd works on bare ArrayFire tensors (no JIT). So the baseline lane is
  ArrayFire **without** the JIT.
- **Shared cuDNN ops allocated `CudaTensor`s unconditionally**, so on ArrayFire the
  conv/batchnorm/RNN results silently switched to the native CUDA backend (bare mode) or
  threw "tensor is not ArrayFire-backed" (under the JIT). Fixed (D-10):
  `detail::makeTensorLike` / `fullLike` in `ops/cudnn/CudnnTensorHelpers` allocate on
  the backend of the op's `input` (and throw on JIT tensors);
  `DropoutDescriptor(prob, like)`. The new guard exposed two tests with wrong setup
  (`JitDeadOutputEliminationTest` fed JIT tensors to a bare evaluator;
  `ArrayFireTensorBaseTest` assumed ArrayFire was the default); each now sets its tensor
  type in `main()`.
- New `_AF` test variant: the generic suites built with `FL_TEST_BACKEND_AF`
  (`TestInit.h` selects bare `ArrayFireTensor`), when `FL_USE_ARRAYFIRE` is on.
- Bug fixed: `ArrayFireBackend::supportsDataType(f16)` tested `FL_BACKEND_CPU` (set
  whenever the native CPU backend is built) instead of `FL_ARRAYFIRE_USE_CPU`, so
  ArrayFire-CUDA reported no f16 and the `_AF` lane skipped every f16 test.

### 3.5 Parity: per-test and per-operation
- Per-test comparison across `_AF` / `_CPU` / `_CUDA` (generic suites):
  `_AF` 334 pass / 14 fail / 6 skip; `_CPU` 313 / 0 / 41; `_CUDA` 321 / 0 / 4
  (no `_CUDA` Contrib variant exists — a coverage gap). Differences are listed in
  `TODO.md` under the ArrayFire decision and in [STATUS.md](STATUS.md).
- Per-operation support matrix (CPU vs CUDA vs ArrayFire, from reading every
  `TensorBackend` override): [research/backend-support-matrix-2026-10-26.md](research/backend-support-matrix-2026-10-26.md).
  Three entries were confirmed by hand and became bugs (§4).

### 3.6 Test-runner corrections
Two mistakes in ad-hoc test running were found and corrected: binaries were selected by
gtest suite name (missed 21 binaries whose suite names differ — all pass on
CPU/CUDA/default once run); and the earlier "all green" picture predates switching the
unsuffixed binaries off the ArrayFire default. The lesson is recorded in
[DEVELOPMENT.md](DEVELOPMENT.md): enumerate binaries from `CTestTestfile.cmake`.

### 3.7 New problems found
- **JIT thread safety (P-4):** `DatasetTest.FileBlobDataset` (multi-threaded writes)
  corrupts the heap. One process-wide `JitBackend` per backend type means all threads
  share one optimizer and evaluator; passes keep per-run state as members
  (`visited_` sets), and the optimizer rewrites shared nodes in place.
- **CUDA tensor indexing** gathered along dim 0 regardless of the index's axis,
  dropped other indices, and mis-addressed write-back for strided views — the cause of
  the `AutogradTest.GetAdvancedIndex` crash (skipped in `_CUDA`).

---

## 4. 2026-10-27 — native-backend bug fixes

- **CPU `log1p`/`sin`/`cos`/`floor`/`ceil`/`erf` wrote nothing for dtypes other than
  f32/f64** (output left uninitialized, no error); `isnan` returned all-false for
  f16/bf16. Fixed: other dtypes compute in f32 and cast back (the IR keeps the input
  dtype); integer `floor`/`ceil` are exact copies. Test
  `TensorUnaryOpsTest.mathOpsNonF32Types`. On `_AF` the test fails because ArrayFire's
  `floor(s32)` returns f32 — kept as a recorded semantic difference.
- **CUDA `clip(t, lowTensor, highTensor)` used only the first element of each bound**
  (and synchronized the host). Fixed with the broadcasting `maximum`/`minimum` kernels.
  Test `TensorUnaryOpsTest.clipElementwiseBounds` (passes on CPU/CUDA/AF/default).
- **CUDA tensor indexing:** a fix (index map through real strides, per-axis semantics)
  was drafted, then **parked**: the maintainer asked whether a general advanced-indexing
  path was the right direction, which led to the indexing redesign (§5). The draft was
  not merged, and its local copy was later lost when scratch space was cleared.
- **Question raised: does computing low-precision math in f32 preclude native bf16?**
  Answer recorded (D-12): for these ops the host C++ math library only works in
  float/double, x86 has no bf16 elementwise math instructions, and oneDNN/PyTorch/XLA all
  compute such ops in f32 internally; the f32 path is a per-backend fallback, not a
  policy. The cost is two extra full-tensor passes; a single-pass convert-on-read /
  convert-on-write loop is a recorded TODO.
- **Incident (process lesson):** a formatter run reformatted three whole files in
  `ops/cudnn/` because a multi-argument variable was not word-split by the shell. The
  files were intent-to-add, so git held no copy; they were restored from an earlier full
  read and the intended edits re-applied, verified by formatting both versions and
  comparing. Lesson in [DEVELOPMENT.md](DEVELOPMENT.md).

---

## 5. 2026-11-02 — indexing redesign (D-11)

- **Survey of in-repo tensor-indexing uses:** embedding (gather rows; backward
  scatter-add via `indexAdd`, whose base implementation round-trips through the host);
  adaptive embedding / adaptive softmax (mask → `nonzero` → gather rows; argsort
  reorder; scatter losses back; assign through gathered rows); cross-entropy (target
  class picked with a dense one-hot multiply; masked fill for the ignore index); speech
  criteria (masked fill); the DETR set criterion in `pkg/vision` — the "advanced indexing
  in a gradient" case — which implements NumPy-style **pointwise** multi-index (pairs of
  query/batch indices, a diagonal) by raveling to linear indices, with a gradient that
  overwrites repeated indices instead of adding. Transformer attention uses additive log
  masks and static reshapes only. Legacy per-axis ("orthogonal") multi-index semantics
  has no real user besides one test and autograd's index backward (a per-combination
  host loop).
- **How other frameworks do it:** NumPy/PyTorch/JAX pointwise ("vectorized") indexing
  with complex multi-index rules; all lower to a few static-shape primitives (index
  select / take-along-axis gather, scatter with add); gradients are scatter-add; boolean
  masks are data-dependent and avoided under compilation (JAX/MLX refuse them).
  ArrayFire (Flashlight's ancestry) uses per-axis indexing.
- **Flashlight artifacts identified:** the "index tensor with the tensor's shape means
  flat" guess; index tensors flattened; gathered copies that write back on later
  in-place ops; a device sync inside shape inference to count mask entries; per-element
  host loops in the index gradient.
- **Decision (D-11):** two primitives, `gather` and `scatter(mode = Set | Add)` along an
  axis (take-along-axis semantics with broadcasting), replacing `indexAdd` and the
  tensor-index code in every backend's `index()`; `x(...)` with one tensor index follows
  NumPy/PyTorch; mask reads = `nonzero` + `gather`, mask writes = `where`; two or more
  tensor indices throw for now; `x(idx) += v` keeps NumPy's no-accumulate semantics.
  ArrayFire implements the same primitives (lookup / flat assign / an atomic-add kernel),
  so the baseline follows the new semantics.
- **Plan:** [plans/indexing-gather-scatter-plan.md](plans/indexing-gather-scatter-plan.md)
  (three phases: primitives; lower `x(...)` onto them; migrate users and delete legacy).
  An earlier, broader design in `TODO.md` ("Generalized Advanced Indexing Operations",
  `scatterReduce` with five reduction ops plus `indexPut`) is superseded in scope by this
  two-primitive plan; its motivation and performance notes still apply. **Status: the
  plan awaits the go-ahead for Phase 1.**

---

## 6. 2026-11-16 — notes consolidated

This directory was created to hold everything needed to continue the work from the
repository: this log, the decision records, current status, architecture and
development guides, and scrubbed copies of the research reports and plans that had lived
in local, untracked tool directories. The maintainer then planned to check in the whole
working copy on a fork and continue from other machines, so the notes from every earlier
AI-assisted session were recovered into [history/](history/README.md): the maintainer's
standing directives, digests of ~45 Claude Code sessions (prompts only; transcripts were
not retained) and the Codex sessions (full transcripts), and a cleaned log of every
maintainer prompt. [CONTINUING.md](CONTINUING.md) is the handoff guide.
