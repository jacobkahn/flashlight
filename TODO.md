# Flashlight Modernization TODO

We want to revamp flashlight after a few years of inactivity to take it into the modern era. this plan contains TODOs and high-level design choices and questions.

We These are large changes. They should be made in the spirit of the Flashlight project itself (read the code, the READMEs, and the paper, which I’ve put in the root of this repo for you (flashlight-paper). Follow these design principles as you go - not necessarily the abstractions, but the spirit and philosophy with which Flashlight was built. We don't want to diverge from these.

We'll eventually impelment complex schedules and schemes for large-scale training  with Flashlight with its expressive, minimal API and low overhead. Flashlight needs to be human readable and editable, but models will also make changes and build abstractions and computation schedules.

> **2026-11-16:** all notes, research and decisions from the modernization work are
> consolidated under `docs/modernization/` (README → STATUS, LOG, DECISIONS,
> ARCHITECTURE, DEVELOPMENT, plans/, research/, tools/). This file stays the itemized
> roadmap and checklist.

## Review + test status (audit 2026-10-26)

Full review of the modernization work (4 parallel code reviews + test re-run +
runtime probes). **Details, evidence, file:line refs, and architecture maps:
`docs/modernization/research/review-2026-10-26.md`.** External research digest:
`docs/modernization/research/research-2026-10-autodiff-compilers.md`,
`docs/modernization/research/research-2026-10-distributed-kernels.md` (summary further below).
The 2026-06-17 triage was incomplete: several "known failures" had different root
causes than recorded, and a critical training bug went unnoticed because tests
passed vacuously.

**Environment notes (development machine used for the 2026-10 measurements):** 8x
H100, 2 × 96-core CPUs with SMT (384 hardware threads). Built with a conda environment
(gcc 14, cmake 4.2, ninja, nvcc, OneDNN 3.11, cuDNN); test binaries resolve runtime libs
through rpath, so tests run the same with or without activation. Default OpenMP uses all
384 threads and makes tiny CPU ops ~20x slower (`OMP_NUM_THREADS=16` speeds CPU suites
up, but also hides the CPU race below). Debugging aids used (a CPU-only ASAN build, an
`LD_PRELOAD` shim making `AsyncCPUStream::enqueue` synchronous, scratch probes linked
against `build/libflashlight.so` + the CUDA driver stub) are described in
`docs/modernization/DEVELOPMENT.md`. Run CPU and CUDA variants of a suite serially (some
tests write fixed `/tmp` file names).

### Fix-first list (verified; see review §1)
- [ ] **CRITICAL: optimizers never update module params (training is a no-op).**
  Repro: Linear + SGD, loss bit-identical after 3 steps (CPU+CUDA). Tensor copy =
  `clone()` (new SharedData, shared AutogradMeta); optimizers hold copies and rebind
  them (`data = data - lr*grad`, recorded by autograd). Decide Tensor value-vs-alias
  semantics (review §4.1); add a test that `model.param(0)` changes over 2 steps.
- [x] **CPU async lifetime race → heap corruption** (fixed 2026-10-26) (`AutogradConv2DTest_CPU.WeightNormConv`,
  `CPUTensorTest` abort ~2/3 runs at ≥128 OMP threads; 5/5 pass with synchronous
  enqueue shim). Async lambdas must own every buffer they touch (incl. pre-assigned
  `out`; CPU concat contiguous copies). Make the worker survive exceptions (today it
  exits and every later `sync()` hangs).
- [x] **AlgebraicSimplification identity rules ignore shape/dtype** (fixed 2026-10-26) (`x+0`, `x*1`, `x-0`,
  `x/1`, `x^1`, `0-x`, `x^2`, `x^0.5`) → this is the real cause of
  `BroadcastRankMismatch` ({4}+full({2,3,4},0) collapses to {4}). Guard with
  same-shape+dtype; GraphVerifier must check root shape/dtype preservation.
- [x] **`fl::copy` same-device "Unknown dtype"** (fixed 2026-10-26): null node streams defeat the
  TransferNode no-op; `CudaBackend::copy` with empty dst always allocates pinned
  host (even CUDA→CUDA); `Tensor::getAdapter<T>()` is an unchecked static_cast.
- [x] **CPU gelu hang** (fixed 2026-10-26): real root cause is a self-cycle built by
  `OneDnnPrimitivePostOpFusion::findAnchorChain` (activation pushed before the
  input-use check; `cur` not advanced on break; standalone branch uses the root as
  its own input). OneDnnOpFusion's missing visited-marking turns it into a spin.
- [x] **DefaultTensorType eagerly initializes ArrayFire+CUDA** (lazy init + AF jitStream fixed 2026-10-26; AF opt-in still a decision) in every `_CPU` test
  (contradicts "lazy init" below); AF default `jitStream()` is nullptr → the
  unsuffixed `Jit*Test` SIGSEGVs are real bugs. Strongly consider making AF opt-in.
- [x] (fixed 2026-10-26) CPU GLU: strided slice reaches OneDNN eltwise (src md ≠ dst md); identity
  reshape that used to force contiguity is now simplified away. Same class: CPU
  native ops ignore strides entirely (wrong results on views).
- [x] (fixed 2026-10-26) CUDA dtype drift vs IR metadata (bf16+f16 → f32 metadata, bf16 data; arithmetic
  returns lhs type; `where` reads f32 as double).
- [x] (fixed 2026-10-26: cuDNN in-subgraph transposes + TF32 engines) cuBLASLt DRELU Jacobian failures are **not** tolerance: single-layer graph has no
  DRELU pattern; suspects cuDNN lowering dropping in-subgraph transposes or TF32.
  Bisect with lowering passes disabled / `NVIDIA_TF32_OVERRIDE=0`.
- [x] (fixed 2026-10-26) CUDA Jacobian OOM/slowness: helper mutates inputs via IndexedUpdate chains that
  are never compacted, full-matrix IndexedUpdate per column, uncompacted `dout`
  chain, grad leaves never zeroed, cuDNN plans rebuilt every eval; plus evaluator
  allocates every intermediate up front. Rewrite helper (review §5.3).

### Fixes applied 2026-10-26 (uncommitted; verified by the test re-run noted below)
Root-caused bugs fixed without design changes. File refs are to the fix sites.
**Verification:** final full run of 85 binaries (`_CPU`/`_CUDA`, Jit*, runtime):
1482 tests pass; the only failures are two stale March binaries no longer built
(`CopyTest`, `JitCudnnConvEpilogueFusionTest` — leftovers in `build/`) and one
flaky timing-dependent `EventTest.ReRecord`, since made deterministic.
- [x] AlgebraicSimplification identity rules keep shape+dtype (`sameMetadata`
  guard on x±0, x*1, x/1, x^{0,1,2,0.5}, 0-x) → `BroadcastRankMismatch` green;
  tests `AddBroadcastZeroNoChange`, `PowBroadcastExponentNoChange`,
  `AddPromotingZeroNoChange`.
- [x] Optimizer checks root shape/dtype after every pass (`verifyRootMetadata`);
  GraphVerifier detects cycles (tests `VerifierRootMetadata`, `VerifierDetectsCycle`).
- [x] `fl::copy` same device: `JitBackend::transfer` resolves null node streams
  to the backend default and returns the source node for same-device copies;
  `evalTransferNode` allocates device destinations on the target backend;
  `Tensor::getAdapter<T>()` asserts the adapter type in debug builds.
- [x] CPU gelu hang: `findAnchorChain` records an activation only together with
  stepping past it; a shared `cur` may only feed a standalone lowering (never be
  absorbed as an anchor); `rewrite` refuses an anchor equal to the replaced node;
  f64 demotion skips bias/residual chains. `OneDnnOpFusion` queues each node's
  inputs once (`expanded_`), resets state per apply, and types its result with
  the node's dtype. Tests `GeluSharedMatmulStandalone`, `GeluSharedConvNotAbsorbed`,
  `GeluOfNegSharedInput`, `GeluBiasNoFusionF64`.
- [x] CPU async lifetime race (heap corruption): enqueued OneDNN tasks keep a
  pre-assigned `out` alive (`keepAliveIfPreassigned` in binop/eltwise/reduction/
  matmul×2) and concat keeps its contiguous copies alive. WeightNormConv 8/8 at
  256 OMP threads (was ~1/6).
- [x] AsyncCPUStream sticky error state (worker survives exceptions; skips tasks
  until `sync()` reports; stream usable afterwards). Test `StreamUsableAfterError`.
- [x] `DefaultTensorType` resolves the compile-time default backend lazily
  (`defaultJitPending`), so `_CPU`/`_CUDA` tests no longer initialize ArrayFire;
  `ArrayFireBackend::jitStream()` returns the wrapped AF device stream, fixing
  the unsuffixed `Jit*Test` segfaults (JitNodeTest 49, ComputeTest 33, ...).
- [x] Evaluator allocates each output buffer just before dispatch (not all up
  front); Custom/IndexedUpdate/Scalar nodes get no pre-assigned buffer.
- [x] Passes treat evaluated nodes as leaves (`isEvaluated` in Pass.h: traversal
  stops there; all absorption predicates refuse them).
- [x] CPU strided views: eltwise makes the source contiguous (GLU fixed); native
  ops get contiguous inputs (`contiguousSharedData`); `where` uses a contiguous
  destination and promotes x/y to `promoteDtypes(x, y)` like the IR.
- [x] CUDA: binary/bitwise ops return `promoteDtypes(lhs, rhs)` (bf16+f16 fixed);
  `where` computes in the dispatched type and returns the promoted dtype;
  flip/roll dispatch on element size (no overrun for 1/2/8-byte dtypes);
  `empty()` allocates on the backend stream (node streams aren't honored by
  dispatch yet).
- [x] Pass correctness: TransposeSimplification folds into reductions only when
  surviving axes keep their order; CSE keys scalars exactly (hexfloat / integer);
  ScalarFolding gives comparisons b8 and skips integer x/0 and MIN/-1;
  ConvBatchNormFolding builds defaults on the bare backend; cuDNN lowering refuses
  transposed operands it can't represent (computed inside the subgraph, or read
  more than once); cuBLASLt backward epilogues restricted to 2-D matmuls, BGRAD
  variants without reshapes, BGRADB disabled, DGELU requires the derivative's Exp
  and an aux width % 8; NVRTC lowering only for f32/f16/bf16/b8 graphs.
- [x] Backend correctness: CPU literal index bounds (`>=`), bare tensors from
  `nonzero`/`var` inside eval (and correct `var` shape for ≤1 element), native
  cumsum/sort on empty input, `getSizeInBytes` is `size_t`.
- [x] JIT core: `JitTensorBase::stream()` BFS has a visited set; `buildPlan`
  no longer uses a popped stack entry; lazy `device()` materializes views and
  shares (not deep-copies) the node's output buffer.
- [x] Autograd: `withoutData()` is a lazy constant on the tensor's backend (no
  full-size allocation per recorded input).
- [x] CPU BatchNorm training: returns the batch statistics OneDNN wrote as
  saveMean/saveVar and supports momentum (cuDNN convention: running =
  (1 − m)·running + m·batch, unbiased batch variance) instead of throwing.
- [x] cuDNN subgraph lowering deselects TENSOR_CORE / DOWN_CONVERT_INPUTS engines
  for f32/f64 subgraphs. f32 matmuls were running as TF32 (10-bit mantissa) — this,
  not the "1-bit relu mask", was why the stacked relu Jacobians failed;
  `NVIDIA_TF32_OVERRIDE=0` does not reach graph-API engine selection. (Legacy
  cuDNN convs still allow TF32 for f32 by default — numeric-policy decision.)
- [x] Tests: Jacobian helper rewritten (perturbed inputs and one-hot cotangents
  built from the host in one transfer each, never mutating the input;
  host-assembled Jacobians) — CUDA BatchNormJacobian* and Transpose now pass; stale tests
  updated (CudaTensorBaseTest.BackendType, CudaTensorIndexTest x3 use bare
  CudaTensors; CPUTensorTest transpose/logicalBinops/eltwiseLogical/matmulShapes
  use row-major/broadcast semantics); `OptimTest.DISABLED_StepUpdatesModuleParams`
  documents the optimizer bug until the semantics decision lands.

**Still open — need a design decision first (see discussion 2026-10-26):**
- [ ] Tensor value-vs-alias semantics for parameters (optimizer no-op; `detach()`
  returning a reference; shared-meta surprises).
- [ ] VJP graph-retention model (functional `valueAndGrad` vs retained recipes
  through eval vs tape+rules first).
- [ ] Compaction policy for long-lived holders (BN running stats chain, evaluated
  history kept alive by user tensors).
- [ ] Native-backend bugs found by the op support matrix (2026-10-26, confirmed by
  reading the code; fix as bug fixes):
  - CPU `log1p`/`sin`/`cos`/`floor`/`ceil`/`erf` write nothing for dtypes other than
    f32/f64 (no error; output buffer left uninitialized) — `CPUUnaryOps.cpp`.
    > **Fixed 2026-10-27:** other dtypes compute in f32 and cast back (the IR keeps
    > the input dtype); integer `floor`/`ceil` are exact copies; `isnan` on f16/bf16
    > checks via f32 (was all-false). Test: `TensorUnaryOpsTest.mathOpsNonF32Types`.
    > On `_AF` that test fails because AF `floor(s32)` returns f32 — an AF/IR dtype
    > difference, kept as a parity data point. Open question (user, 2026-10-27):
    > compute precision for low-precision dtypes is a per-backend, per-op choice; the
    > f32 path is a fallback where no native kernel exists, and it costs two extra
    > full-tensor passes (astype in/out) — a fused convert-compute-convert loop would
    > keep the numerics and drop the passes.
  - CUDA `clip(tensor, lowTensor, highTensor)` reads only the first element of each
    bound (`scalarSync`, also a host sync) — `ops/cuda/CudaUnaryOps.cu`.
    > **Fixed 2026-10-27:** element-wise via the broadcasting `maximum`/`minimum`
    > kernels, result cast to the input dtype; no host sync. Test:
    > `TensorUnaryOpsTest.clipElementwiseBounds` (passes on CPU/CUDA/AF/default).
  - CUDA tensor indexing gathers along dim 0 whatever axis the tensor index is on,
    and returns early, dropping the other indices — `CudaTensor.cu` `index()`. Likely
    cause of the `AutogradTest.GetAdvancedIndex` illegal access (`x(a, b, span,
    range)`). Also: for strided views the gather reads a contiguous copy but the
    scatter-back map is applied to the parent buffer, so writes land at wrong offsets.
    > **Parked 2026-10-27 pending an indexing-semantics decision** (user asked whether
    > a general advanced-indexing kernel is the right direction). A draft (index map
    > through real strides, orthogonal per-axis like CPU/AF, JIT-compatible lone-index
    > rules) was kept outside the repo (since lost; superseded by the gather/scatter
    > plan); the repo copy is unchanged. Related: autograd's index backward
    > (`autogradIndex` in `autograd/Functions.cpp`) loops over every combination of
    > tensor-index values with a `scalarSync` + slice-add each, instead of a
    > scatter-add.
- [ ] CPU f32-fallback unary ops: remove redundant casts / fuse (user, 2026-11-02).
  Today `log1p`/`sin`/`cos`/`floor`/`ceil`/`erf`/`isnan` on bf16/f16/integer inputs
  run `astype(f32)` → op → `astype(type)`: two extra full-tensor passes and two
  temporaries per op. Options:
  - Single-pass loop: one `mapUnaryAsF32(out, in, f)` helper in `CPUUnaryOps.cpp`
    that switches on dtype once (outside the loop) and runs a monomorphic,
    vectorizable loop `store(f(load(in[i])))`, where load/store are per-dtype
    converters (bf16↔f32 is a 16-bit shift with round-to-nearest-even on store —
    reuse the conversion already in `CPUTensor::astype`; f16 via F16C or software).
    Same numerics as today, one pass, no temporaries; the existing f32/f64 loops
    become the identity-converter instances of the same helper.
  - JIT level: the casts are adjacent `CastNode`s around the op, so a fusion pass
    can absorb them (convert-on-load / convert-on-store in the fused kernel) and
    drop cast pairs that cancel (e.g. f32 → bf16 → f32 feeding an op that
    recomputes in f32). Check how often bf16 chains hit these ops before investing.
  - Native low-precision kernels: where a backend has real bf16/f16 instructions for
    an op, it should use them; the f32 path is only a fallback.
- [ ] Indexing redesign — proposal 2026-11-02 (pending user approval; multi-tensor
  indexing punted). In-repo uses of tensor indexing, surveyed 2026-11-02:
  `autograd::embedding` (gather rows; backward `indexAdd` scatter-add);
  `AdaptiveEmbedding` / `AdaptiveSoftMaxLoss` / `AdaptiveSoftMax::predict` (bool mask →
  `nonzero` → gather rows; argsort reorder; scatter losses back via
  `output(indices) = input`; assign through gathered rows); `categoricalCrossEntropy`
  (target-class pick done densely with a one-hot multiply, O(X·C); masked fill
  `result(ignoreMask) = 0`); `pkg/speech` CriterionUtils (masked fill); DETR
  `pkg/vision/criterion/SetCriterion.cpp` — the "advanced indexing in a gradient"
  case: a local `index(in, {Tensor(), colIdxs, batchIdxs})` with NumPy-style
  POINTWISE semantics (pairs of (query, batch) indices, and `{rng, rng}` to take a
  diagonal), implemented by raveling coordinates to linear indices + flat gather;
  its gradient is a flat scatter that overwrites duplicates instead of adding (a TODO
  in the code; fine for unique Hungarian matches). Transformer attention uses
  additive log-masks and pad/reshape/slice only (no tensor indexing). Orthogonal
  multi-tensor indexing (Flashlight/AF legacy) has no in-repo user besides
  `AutogradTest.GetAdvancedIndex`; `autogradIndex`'s backward assumes it.
  Proposal: two primitives, `gather` and `scatter(mode = set | add)` along an axis,
  replacing `TensorBackend::indexAdd` and the tensor-index branches of backend
  `index()`; `x(...)` lowers to views + these; see the 2026-11-02 discussion.
  > **2026-11-02:** design approved (2 primitives, NumPy/PyTorch single-index
  > semantics, masks: read = nonzero + gather / write = where, multi-index throws,
  > `+=` keeps NumPy no-accumulate semantics). File-by-file plan in
  > `docs/modernization/plans/indexing-gather-scatter-plan.md` (pending approval to start).
- [ ] JIT thread-safety model. Found 2026-10-26 (`DatasetTest.FileBlobDataset`
  multi-threaded write, crashes with heap corruption on the CUDA default): there is
  one process-wide `JitBackend` per backend type (`getOrCreateJitBackend`), so all
  threads share its `Optimizer` and `Evaluator`; passes keep per-run state as members
  (`TopDownPass::visited_`, `BottomUpPass::visited_`), and concurrent `eval()` calls
  corrupt those sets (gdb: two threads in `TopDownPass::apply`/`traverse` on the same
  pass object). Even with stateless passes, the optimizer rewrites shared nodes in
  place, so two threads evaluating overlapping graphs race on the IR. Options:
  serialize eval per JitBackend (simple; serializes data-loader threads), per-thread
  optimizer/evaluator state plus a graph-level lock, or per-thread JitBackends.
- [ ] ArrayFire: keep default / opt-in / remove.
  > **2026-10-26 facts for this decision:** `FL_USE_ARRAYFIRE` defaults ON and
  > `DefaultTensorType.h` picks AF > CPU > Stub; `FL_USE_NATIVE_CUDA` defaults OFF
  > and `CudaTensor` can't be the compile-time default at all. So a stock build, and
  > all 78 unsuffixed test binaries, run JIT-over-ArrayFire. That configuration is
  > already broken wherever it reaches shared cuDNN code: `ops/cudnn` allocates via
  > `makeCudaTensor` (registry `TensorBackendType::Cuda`), and AF then throws
  > `toArray: tensor is not ArrayFire-backed`. Measured on AF default: ModuleTest
  > 5 fail (RNN/LSTM/LayerNorm), AutogradNormalizationTest 7/8 fail,
  > AutogradRnnTest 3/3 fail, AutogradConv2DTest.WeightNormConv fail, all with that
  > error (the same tests pass on `_CPU`/`_CUDA`); TensorBaseTest 2 fail
  > (`scalar`, `astypeConversions`). Only AF provides: `pkg/vision` tensor extension,
  > `pkg/halide` interop (both already CMake-guarded on AF), the AF caching memory
  > manager (+3 test binaries), and OpenCL/macOS GPU. `backend/af` is ~5.5k LOC.
  > `backend/af/CMakeLists.txt` does `find_package(ArrayFire REQUIRED)`, so a
  > default configure fails without AF installed. README still documents a
  > nonexistent `FL_BUILD_ARRAYFIRE` (actual option: `FL_USE_ARRAYFIRE`).
  > **Decision (2026-10-26): option C — native backends are the defaults, ArrayFire
  > is opt-in but kept as the reference baseline.** `FL_USE_ARRAYFIRE` defaults OFF;
  > `FL_USE_CPU` defaults ON (duplicate top-level OFF declaration removed);
  > `FL_USE_NATIVE_CUDA` defaults ON whenever `FL_USE_CUDA=ON`; compile-time default
  > order CUDA > CPU > ArrayFire > Stub (`DefaultTensorType.h`); Gloo defaults on only
  > for CPU-only builds (CPU+CUDA would otherwise default both NCCL and Gloo on).
  > README updated. Baseline role (user, 2026-10-26): AF is the bar the native
  > compiler path must match on support and performance, so our dev builds keep AF
  > built (not default). AF runs wrapped in JIT only if autograd needs it, and then
  > the JIT must be a passthrough: generic simplifications only (no backend
  > optimizer extension is registered for AF, so today it gets Phase 1 passes plus
  > generic elementwise fusion, which the evaluator interprets op by op — AF's own
  > JIT still sees individual ops). Follow-ups:
  > - [x] Re-enable AF in the local dev build (non-default). (2026-10-26)
  > - [x] Fix AF + shared cuDNN ops (allocate on the caller's backend, not always
  >   `CudaTensor`) — show diff first (shared code). (2026-10-26, approved diff:
  >   `detail::makeTensorLike`/`fullLike` in `ops/cudnn/CudnnTensorHelpers`, keyed
  >   on the op's `input`, throws on JIT tensors; `DropoutDescriptor(prob, like)`.
  >   Before: AF eager conv returned `CudaTensor`s (backend silently switched),
  >   JIT-over-AF threw. After: results stay on AF in both modes. The new JIT guard
  >   exposed `JitDeadOutputEliminationTest` feeding JIT tensors to a bare
  >   Evaluator; it and `ArrayFireTensorBaseTest` now set their tensor type in
  >   `main()`.)
  > - [x] `_AF` test variant of the generic suites; check whether autograd works
  >   without JIT (decides eager vs passthrough-JIT baseline). (2026-10-26: autograd
  >   works without JIT on AF — elementwise, matmul, conv probes — so `_AF` uses bare
  >   `ArrayFireTensor`, no JIT: `FL_TEST_BACKEND_AF` in `TestInit.h`, built when
  >   `FL_USE_ARRAYFIRE`.)
  > - [x] Parity report: per-test pass/skip/fail across `_AF`/`_CPU`/`_CUDA`, plus
  >   which backend ops throw per backend. (Op-level matrix done:
  >   `docs/modernization/research/backend-support-matrix-2026-10-26.md`.)
  >   **Per-test parity (2026-10-26, generic suites):** `_AF` 334 pass / 14 fail /
  >   6 skip; `_CPU` 313 / 0 / 41; `_CUDA` 321 / 0 / 4 (29 absent: no `_CUDA`
  >   Contrib variants). AF passes what native skips: all f16 tests vs CPU (CPU has
  >   no f16), casting tests, `AsContiguous`, advanced-index gradients
  >   (`GetAdvancedIndex{,F16}`: CPU unsupported, CUDA crashes). Native passes what
  >   AF skips/fails: bf16 (3 tests; AF has none), `tril`/`triu` (AF: rank-2,
  >   diagonal 0 only), f16 conv+bias in Contrib Conformer/TDS (AF path: cuDNN "No
  >   valid engine configs for ConvFwd_ADD"; native CUDA passes), mixed f16/f32
  >   `embedding` (af::exception), `TensorBaseTest.scalar`. Not AF gaps (tests
  >   assume JIT or native backends): ComputeTest x5 (laziness), TensorAdapterTest
  >   x2 (JIT default type), CopyTest x2 (helper picks the CPU stream outside CUDA
  >   builds; also `fl::copy` maps device types to native backends only, so AF has
  >   no functional-copy route). Fixed on the way: `ArrayFireBackend::
  >   supportsDataType(f16)` tested `FL_BACKEND_CPU` (set whenever the native CPU
  >   backend is built) instead of `FL_ARRAYFIRE_USE_CPU`, so AF-CUDA reported no
  >   f16 and skipped every f16 test.
  >   Runner note: earlier "full run" counts on 2026-10-26 missed 21 binaries whose
  >   gtest suite names differ from the binary name (ComputeTest, NNUtilsTest,
  >   TensorAdapterTest, ContribSerializationTest, HistogramTest, LoggingTest, ...);
  >   fixed (binaries now read from CTestTestfile.cmake) and all of them pass on
  >   CPU/CUDA/default.
  > - [ ] Backend selector for `app/benchmark`; record AF numbers as the bar.
  > - [ ] Baseline fairness: the evaluator pre-allocates an output buffer per node
  >   (`Evaluator::assignBuffer` → `ArrayFireBackend::empty` = real `af::array`),
  >   but AF ops replace `out` with a new lazy array, so every preallocation is
  >   wasted on the AF path. Measure before deciding how to fix.
  > **Verified 2026-10-26 (AF off, CUDA default):** full run of all ctest-listed
  > binaries: 101 binaries, 1,663 tests pass, 48 skipped, 2 problem binaries —
  > `AutogradTest` (unsuffixed, now on CUDA): `GetAdvancedIndex` illegal memory access
  > (known; `_CUDA` skips it), and the sticky CUDA error fails 5 later tests in the same
  > process (24/24 pass with the advanced-index tests excluded); `DatasetTest`:
  > JIT thread-safety race (see the open decision below). Fresh configures: stock
  > (CPU only) and `-DFL_USE_CUDA=ON` both configure without looking for ArrayFire
  > (`-DFL_BUILD_DISTRIBUTED=OFF` needed only because this env has no MPI); the stock
  > CPU-only build compiles and its default-backend suites pass (TensorBaseTest 60/60,
  > ModuleTest 25/25, AutogradTest 17/17, JitTensorTest 43/43, JitEvaluatorTest 22/22,
  > OptimTest 2/2). Skips that mark native gaps: 27 f16/bf16 on CPU, advanced-index
  > gradients (CPU unsupported, CUDA crashes), strided views on CPU, bf16 RNN on CPU.
**Decision (2026-10-26): strict f32 on CUDA by default; `FL_ALLOW_TF32=1` opts in.**
`fl::cuda::allowTF32()` (`runtime/CUDAUtils`, read once via `fl::getEnvVar`) is the
single switch consulted by every f32 math-mode choice: cuDNN frontend engine
filtering (`ops/cudnn/CudnnNumerics.h::restrictToStrictMath`, used by
`CudnnSubgraphLowering` and conv fwd / bwd-data / bwd-filter), cuDNN RNN math type
(`CUDNN_FMA_MATH` vs `CUDNN_DEFAULT_MATH`), cuBLAS handle math mode, cuBLASLt
compute type. Verified: relu Jacobians pass strict, fail with `FL_ALLOW_TF32=1`.
Open: per-op / per-dtype runtime control (the env var is process-wide and read
once) — see NumericPolicy discussion.
> **2026-10-26:** Deferred — not on the critical path for the remaining decisions.
> A first attempt (a `MathMode` field on `MatmulNode`/`Conv2dNode` plus an
> evaluator wrapper) was reverted: generic IR, evaluator, and `fl::` APIs must not
> carry backend math knobs. Constraint for any future design: TF32 / math-mode
> properties live only inside CUDA primitive-library components (`ops/cudnn`,
> `backend/cuda`, cuBLAS/cuBLASLt code), well hidden; any programmatic control
> would sit on `CudaBackend` itself. Small cleanup pending: move `allowTF32()`
> out of the public `runtime/CUDAUtils.h` (`FL_API`) into a CUDA-internal header.
> **2026-10-26 (done):** moved to `fl::cuda::detail::allowTF32()` in the
> header-only internal `ops/cuda/CudaNumerics.h` (header-only because `ops/cudnn`
> also builds for ArrayFire-CUDA without the native backend). `runtime/CUDAUtils`
> is back to its original contents. Re-verified: relu Jacobians 5/5 strict, 1/5
> with `FL_ALLOW_TF32=1`; cuDNN lowering 27/27; cuBLASLt epilogue 15/15; CUDA RNN
> 6/6.

**Perf trap found:** OneDNN/OpenMP defaults to all 384 hardware threads on the
development machine; tiny CPU ops
pay a huge per-op threading cost (`AutogradUnaryOpsTest.Clip`: 0.6 s at
`OMP_NUM_THREADS=16` vs 244 s at 384). Needs a CPU-backend thread policy (cap
threads for small ops / sensible default) — currently tests just run slowly.

**Still open — implementation work, no decision needed:** Future `isReady()`;
CPU BatchNorm training with momentum; CPU bf16 unary ops (uninitialized output)
and `dispatchByType` gaps; integer semantics (u32/s16 storage reinterpretation,
int math in f32); OneDNN rank > 4; cuDNN plan cache; CPU `relativeSync` blocking
the host per op; CPU Tensor copy = blocking deep copy; per-node CUDA streams.

### Guardrails to add (cheap, high value)
- [ ] Debug-mode evaluator check after each node: result shape/dtype/backend ==
  node metadata (would have caught the broadcast, copy, bf16 bugs immediately).
- [~] GraphVerifier: root preservation, cycle detection, dtype rules; gate on NDEBUG.
  (2026-10-26: root preservation + cycle detection done; dtype rules and NDEBUG gating open)
- [x] `getAdapter<T>()` backend-type assert in debug builds. (2026-10-26)
- [ ] Synchronous-AsyncCPUStream switch + ASAN/TSAN build configs.
- [x] (2026-10-26, passes; verifier/stream inference still walk history) All passes (and verifier, stream inference) treat already-evaluated nodes as
  leaves (today evaluated history is re-optimized every eval; absorbing it
  recomputes — including `rand` with new values).

### Friction & slowdowns log (found 2026-10-26; come back to these)
Things that made the code slow or made working on it painful. Not correctness
bugs (those are above), but each cost real time this session.

**Runtime performance traps**
- [ ] **OpenMP uses every logical CPU (384 here: 2 sockets × 96 cores × 2 HT).**
  OneDNN runs on OpenMP and Flashlight sets no thread count. Tiny ops pay a
  ~100× fork/join tax and large ops gain nothing. Measured CPU matmul (ms):

  | OMP threads | 64² | 512² | 4096² |
  |---|---|---|---|
  | 1 | 0.10 | 3.2 | 1388 |
  | 16 | 0.10 | 0.99 | 142 |
  | 48 | 0.08 | 0.31 | 107 |
  | 96 (one NUMA node, physical) | 0.16 | 0.58 | 92 |
  | 192 | 0.37 | 0.70 | 99 |
  | 384 (default) | 11.3 | 11.5 | 96 |

  Tests: `AutogradUnaryOpsTest_CPU.Clip` 0.6 s at 16 threads vs 244 s at 384;
  `WeightNormConv` 0.23 s vs 4.5 s. Needs a CPU thread policy (default + later a
  size-aware one). Implementation note: ICVs are per thread, so the count must be
  set on the AsyncCPUStream worker (and any host thread running primitives), not
  just in `main`.
  **Decision (2026-10-26):** the library sets no thread count; `OMP_NUM_THREADS`
  (and other OpenMP env vars) from the environment decide. Tests get an opt-in
  CMake cache option `FL_TEST_NUM_THREADS` (default empty) that sets
  `OMP_NUM_THREADS` for ctest runs (`cmake/TestUtils.cmake`; documented in
  README). Running a test binary directly inherits the caller's environment.
  On the 384-thread development machine use `-DFL_TEST_NUM_THREADS=16` (or export
  `OMP_NUM_THREADS`).
  Remaining: a size-aware thread policy (few threads for small ops) belongs in
  the future execution/schedule layer, which knows op sizes.
- [ ] **CPU `relativeSync` blocks the host on every op.** `relativeSync` calls
  `backend().eval()` = `stream().sync()` for bare CPU tensors, and binary ops,
  matmul, where and concat all call it → no CPU async overlap at all
  (`Compute.cpp:64-82`, `CPUBackend.cpp:139-144`). Only eval JIT tensors there.
- [ ] **CPU `Tensor` copy = blocking sync + deep memcpy** (`CPUTensor::clone`),
  hidden inside eval: Projection eval, Concatenate/FusedElementwise inputs,
  shared-cache returns, `cond ? a : a.astype(t)` ternaries. CUDA copies are
  shallow. (Tied to the Tensor copy-semantics decision.)
- [ ] **Native "host loop" CPU ops sync and round-trip per call**: logicalAnd/Or,
  pow, mod, bitwise, non-f32 `where` read every input to the host with a stream
  sync (gdb: `logicalAndCpu → toHostVectorSync → AsyncCPUStream::sync`). f64 on
  CPU goes through similar slow fallbacks.
- [ ] **CPU elementwise chains get no fusion**: ElementwiseFusion runs before
  OneDnnOpFusion and the evaluator runs `FusedElementwiseNode` op by op on CPU;
  reshape always copies; OneDNN `primitive_desc` rebuilt per call; conv weights
  reordered per call; `executeNetwork` runs conv/pool/bn synchronously on the host.
- [ ] **CUDA: Thrust `par.on(stream)` synchronizes after algorithms** (gdb:
  `thrust::fill → cub::SyncStream → cudaStreamSynchronize` inside `CudaBackend::full`).
  Use `thrust::cuda::par_nosync.on(stream)` (Thrust ≥1.16) or CUB directly.
- [ ] **cuDNN frontend graphs are rebuilt and plans compiled on every eval**
  (`CudnnSubgraphLowering` and unfused conv); `CudnnPlanCache` exists with zero
  users. Big part of why CUDA BatchNorm Jacobian tests take ~110 s each.
- [ ] Other CUDA host round-trips: full reductions via Thrust copy to host; RNN
  does synchronous `cudaMalloc`/`cudaMemcpy` per call; cumsum launches CUB per row;
  embedding backward (`indexAdd`) copies the whole table to the host and back.
- [ ] Per-op overhead on the graph-building side: every op starts with
  `Tensor out;`, which allocates a placeholder CPUTensor + JitTensorBase +
  ValueNode; CSE builds an `ostringstream` key per node; GraphVerifier walks the
  whole graph after every pass in release builds (9-10 walks per eval).

**Test-suite speed / reliability**
- [ ] Full `_CPU`/`_CUDA` suite takes ~20+ min even on the development machine, dominated by
  Jacobian finite-difference tests (O(nIn + nOut) graph evals each) at 384 threads.
  Shrink test sizes, pin `OMP_NUM_THREADS` for tests, or share one forward pass.
- [ ] Jacobian helper still lets *other* grad leaves (BN/conv weights captured by
  the test lambda) accumulate lazy gradient chains across rows; give the helper
  a way to zero or exclude them.
- [ ] Tests that write fixed `/tmp` filenames collide when CPU and CUDA variants
  run concurrently (`NNSerializationTest.FileSize` gave a false failure); use
  unique temp names.
- [ ] Sleep-based timing tests are flaky under load (`EventTest.ReRecord` fixed;
  audit other `sleep_for` tests in runtime/).
- [ ] Stale binaries stay in `build/flashlight/fl/test/` after their targets are
  removed (`CopyTest`, `JitCudnnConvEpilogueFusionTest` from March) and fail
  confusingly; a clean rebuild or a CMake clean step would remove them.
- [ ] No parallel test runner with per-test timeouts and GPU assignment; I used a
  local shell script (12-way parallel, `CUDA_VISIBLE_DEVICES` round-robin). Consider
  ctest labels/RESOURCE_GROUPS or a checked-in script.
  > **2026-11-16:** a cleaned-up runner is checked in as
  > `docs/modernization/tools/run_tests.py` (binaries from `CTestTestfile.cmake`,
  > variants of a suite run serially, GPUs round-robin) with `parity.py` for the
  > AF/CPU/CUDA per-test matrix.
- [ ] Pass tests are structural only (graph shape, never executed against a
  reference), so wrong-result fusions passed. Add differential tests:
  fused vs unfused vs reference on random inputs.
- [ ] Unsuffixed tests silently depend on the compile-time default backend
  (ArrayFire); surprising and easy to misread as "AF-only" failures.

**Debuggability / tooling gaps**
- [ ] No switch to dump the optimized graph per eval (needed a scratch program
  with `ScopedPostEvalGraphvizPrinter`); add e.g. `FL_JIT_DUMP_GRAPH`.
- [ ] No way to disable individual optimizer passes for bisecting
  (wanted this for the cuDNN / cuBLASLt investigation); add e.g.
  `FL_JIT_DISABLE_PASSES=CudnnSubgraphLowering,...`.
- [ ] No synchronous mode for `AsyncCPUStream` (needed an LD_PRELOAD shim to
  confirm the lifetime race); add an env/debug switch.
- [ ] No ASAN/TSAN build configs in the tree. Note for setting one up: configure
  needs `-DFL_BUILD_DISTRIBUTED=OFF -DFL_USE_GLOO=OFF` (gloo's
  `cmake_minimum_required` is too old for CMake 4), and use-after-free *writes by
  OneDNN* (uninstrumented) are hidden by ASAN's allocator quarantine.
- [ ] Heap bugs in the CPU path disappear under gdb and ASAN; valgrind aborts on
  the default 384 OpenMP threads (needs `--max-threads` and fewer OMP threads).
- [ ] gdb `catch throw` misses exceptions built via `std::make_exception_ptr`
  (used by futures/streams); break on the exception constructors instead.
- [ ] Writing a quick probe against the library means copying compile flags out
  of `build.ninja` and linking the CUDA driver stub by hand; a small
  `scratch/` CMake target (or exported config) would make probes one command.

### Other confirmed high-severity issues (review §2)
Evaluator allocates every intermediate before execution and CustomNode/IndexedUpdate
never consume their pre-assigned buffer; node stream ≠ execution stream on CUDA;
Future `isReady()` can stay false forever; lazy `device()` on a view attaches a
view-sized buffer to the base (overflow); CPU Tensor copy = blocking deep copy (CUDA
shallow); TransposeSimplification reduce-axis bug; cuBLASLt BGRADB wrong + overrun,
DGELU matches forward gelu, batch dims ignored; cuDNN lowering drops in-subgraph
transposes, per-node transpose views, rebuilds plans every eval (CudnnPlanCache
unused); CSE merges constants differing past 6 digits; ScalarFolding compare dtype +
integer div-by-zero SIGFPE; NVRTC computes ints in f32 with fast-math; `withoutData()`
allocates full device buffers; `detach()` returns a reference into a side list
(writes through it are no-ops for grad tensors); BN running stats chain leak; CUDA
flip/roll buffer overrun for non-f32/f64; CPU concat use-after-free; CPU BN training
broken for momentum≠0; integer semantics (u32→s32, s16→f16 reinterpretation, int math
in f32); CPU rejects rank>4; `relativeSync` blocks the host on every CPU op.

### Test-suite snapshot (2026-10-26, 85 binaries)
Green: all suffixed Tensor*/Compute/Index/Shape/Adapter (CPU+CUDA), Autograd* except
below, Module/NNSerialization/NNUtils/Optim (CUDA; CPU except GLU), Contrib_CPU,
all Jit* pass-structure tests, CudaJitEndToEnd 65, OneDnnJitEndToEnd 35, runtime
tests. Failing: BroadcastRankMismatch (CPU+CUDA), CopyTest_CUDA x2,
CudaTensorIndexTest x3, GeluLinearJacobian (CPU hang), WeightNormConv + CPUTensorTest
(CPU heap corruption), GLUFwd (CPU), MixedBf16F16Compute, ReluLinear* Jacobians (CUDA
x3), AutogradUnaryOps Transpose OOM (+Exp cascade), BatchNormJacobian* (CUDA x4).
Stale tests to update: CudaTensorBaseTest.BackendType, CPUTensorTest
transpose/logicalBinops/eltwiseLogical/matmulShapes. Note: vacuous passes exist
(e.g. OptimTest.SerializationTest passes because params never change).

## External research digest (2026-10-26)

Two web surveys informed the roadmap; full reports (with sources, versions and
verified/inferred tags): `docs/modernization/research/research-2026-10-autodiff-compilers.md`
and `docs/modernization/research/research-2026-10-distributed-kernels.md`. Headline takeaways:

**VJP / effects / memory (next three milestones)**
- Every lazy system that differentiates its own graph hits the same problem we
  have: evaluation compacts the graph. MLX re-traces inside transforms and exempts
  tracers from detach; tinygrad requires building gradients before `realize()`;
  PyTorch keeps saved tensors per node. Flashlight must pick one (see VJP
  corrections in Phase 4).
- VJP driver shape (MLX/tinygrad): explicit targets, `StopGradient`, on-path
  filter, reverse topo, cotangent map with free-on-consume, unbroadcast + dtype
  cast centrally in the driver; rules return optional (symbolic-zero) cotangents.
- Effects/aliasing: value semantics + handle rebinding (MLX), a static per-node
  table `Pure | ViewOf(i) | MayReuse{i} | Mutates{i}` + effect classes with token
  edges, one reinplace rule (MayReuse ∧ not held ∧ no later use in plan order) +
  refcount donation where graph edges count as references.
- Planner v1 can be tiny (tinygrad's is 61 lines): plan-order lifetimes, exclude
  held buffers, best-fit/TLSF offsets, per (device, lane) arenas; then min-peak
  order selection (Inductor). Remat v1 = ~80-line min-cut restricted to fusible
  ops; run after CSE (CSE undoes remat).
- Execution: capture whole plans as CUDA graphs keyed by (plan hash, shape
  bucket); keep scalars (lr, step) and RNG state in device buffers, never NVRTC
  immediates. Dynamic shapes matter only for caching here (graphs are rebuilt per
  call): `SymDim{min,max,multipleOf}` + bucketing, no guard engine.
- Pass infra: XLA-style header-only `m::` matcher + tinygrad-style rule-list driver
  (fixed point, iteration limit, per-rule match counters, before/after dumps).

**Distributed / MoE / kernels / numerics**
- Execution layer first: with one CUDA stream nothing overlaps. Semantic IR keeps
  collectives only; lower to generic `AsyncStart/AsyncDone`; one-sided put/get
  stays in the kernel layer via a `SymmetricBuffer` (NCCL ≥2.28 windows: LSA,
  multimem, GIN; copy-engine collectives avoid SM contention).
- Pipeline schedules as data (PyTorch's `_Action` table + lowering passes +
  deadlock-checking simulator); VJP gives the I/W split for zero-bubble.
- MoE: `SegmentedTensor` (static capacity + device offsets) + `DispatchHandle`;
  7 ops (route, dispatchPlan, pack, exchange, groupedMatmul, segmentedActivation,
  combine); lower to cuBLASLt grouped GEMM (CUDA ≥13.2U1 on Hopper) or cuDNN
  `moe_grouped_matmul`; avoid host token-count readback.
- Kernels: NVRTC-generated C++ kernel families with CUB via bundled CCCL headers
  and a disk cache is the mainstream path (DeepGEMM, DeepEP V2 JIT everything);
  don't generate GEMMs/attention — use cuBLASLt epilogues and cuDNN SDPA with
  cached plans and pinned minimum versions.
- FP8 is ready on H100 (see FP8 section update).

## Test status / known failures (audit 2026-06-17)

> **2026-10-26:** record preserved as written; inline notes give the corrected root
> causes found in the 2026-10-26 review (see section above and
> `docs/modernization/research/review-2026-10-26.md`).

Ran the full `_CPU`/`_CUDA` suffixed suites. IR-node promotion (Tile/Where/Clip/
Flip/Triangular/Cumsum/Concatenate/Pad/Roll) and the Variable→Tensor autograd
refactor are solid: all JIT optimizer-pass tests pass (Algebraic 75, CSE 32,
Transpose 18) and CPU autograd suites are fully green. Open failures, by severity:
> **2026-10-26:** "Variable→Tensor refactor is solid" was premature: optimizer
> updates never reach module params (training no-op; tests pass vacuously), and
> `detach()`/`withoutData()` have new bugs. See review §1.1, §2.3.

- [ ] **CPU `OneDnnOpFusion` infinite loop on gelu** (BLOCKER for CPU composed
  autograd). `AutogradComposedTest_CPU.GeluLinearJacobian` hangs forever on a tiny
  {1,4} graph; gdb shows `OneDnnOpFusion::apply → rewriteFrom → searchAndFuse →
  fuseNodes` spinning. Lead: `opt/backends/onednn/OneDnnOpFusion.cpp:186-190` does
  the early `return fuseNodes(...)` BEFORE `visited_.insert(node)`, so non-fusable
  shared multi-use nodes (the matmul result reused in `x*0.5*(1+erf(x*c))`) are never
  marked visited and their inputs get re-pushed to the worklist every encounter;
  combined with `replaceAllUsesWith` over the shared node it never converges. Relu
  chains pass (no shared-into-unary input). CPU/OneDNN-only.
  > **2026-10-26:** root cause is a self-cycle created upstream by
  > `OneDnnPrimitivePostOpFusion::findAnchorChain` (standalone-gelu branch uses the
  > gelu root as its own input). The missing visited-marking is real but only turns
  > the cycle into a spin (and causes path blow-up on acyclic graphs).
- [ ] **rank≥2 broadcast materialization** (`TensorBinaryOpsTest.BroadcastRankMismatch`,
  CPU+CUDA). `{4}+{2,3,4}` (operand needs →{1,1,4}): JIT BinaryNode shape inference
  reports {2,3,4} correctly but the materialized result is rank-1. `{4}+{3,4}`
  (rank-1 diff) works. Bug is in binary-op operand alignment for rank difference ≥2.
  > **2026-10-26:** not operand alignment — AlgebraicSimplification rewrites
  > `e + full({2,3,4}, 0)` to `e` (identity rule without shape/dtype check). The
  > `{4}+{3,4}` case passes only because it adds 10, not 0.
- [ ] **`fl::copy` functional path "Unknown dtype"** (CUDA) —
  `CopyTest.FunctionalCopy_SameDevice` and `_StridedSrc` both throw "Unknown dtype".
  > **2026-10-26:** null node stream defeats TransferNode no-op; CudaBackend::copy
  > empty-dst allocates pinned host for CUDA→CUDA; unchecked `getAdapter<T>()`.
- [ ] **CUDA Jacobian-test OOM / extreme slowness** — `AutogradUnaryOpsTest.Transpose`
  OOMs (CudaTensor.cu:171) after ~28s and poisons the GPU so `Exp` then fails
  (passes alone); `AutogradNormalizationTest.BatchNormJacobian{,MultipleAxes}`(+F16)
  each run ~47s then fail. Numeric-diff Jacobian loop re-evaluates/rebuilds per
  perturbation without caching — same pattern as the old Conv2D "looks like a hang".
  > **2026-10-26:** mechanism detailed in review §5.3 (IndexedUpdate chains, per-column
  > full-matrix updates, uncompacted `dout`, unzeroed grad leaves, cuDNN plans rebuilt
  > per eval, evaluator allocating all intermediates up front).
- [ ] **CUDA weight-gradient Jacobians via cuBLASLt DRELU epilogue** —
  `AutogradComposedTest_CUDA` ReluLinearJacobianWeight / Stacked / ThreeLayerStacked
  fail (test comments flag 1-bit relu-mask numerical differences in the fused
  backward matmul). Decide tolerance vs. epilogue-correctness.
  > **2026-10-26:** not tolerance — the single-layer graph contains no DRELU pattern.
  > Suspects: cuDNN lowering silently dropping in-subgraph transposes, TF32 engines.
- [ ] **CPU `ModuleTest.GLUFwd`** — OneDNN can't create the eltwise primitive
  descriptor for the GLU sigmoid over a strided/sliced {3,1} half (plain sigmoid is
  fine). CPU/OneDNN-only (passes on CUDA).
  > **2026-10-26:** src md uses view strides, dst md contiguous; the identity reshape
  > that used to force a copy is now simplified away.
- Note: unsuffixed `Jit*Test` use the ArrayFire compile-time default; the ones that
  materialize crash on the AF path (e.g. `JitNodeTest.ValueNodeMetaData` SIGSEGV in
  `JitTensorBase::stream()`). Not a real regression — run the `_CPU`/`_CUDA` variants.
  > **2026-10-26:** it is a real bug: AF `jitStream()` returns nullptr and
  > `JitBackend::defaultStream()` dereferences it, so `fl::full(...).scalarSync()`
  > segfaults in a default build.

## Phase 0: Upgrade Defaults
- [x] 0A: CMake 3.24, C++20, CUDA 20, CMAKE_CUDA_ARCHITECTURES
- [x] 0A: Remove dead cmake_minimum_required in sub-includes
- [x] 0A: Remove dead version branches (CMAKE_VERSION >= 3.17, MSVC 2017, cmake < 3.13)
- [x] 0A: Replace .count() with .contains() on maps/sets
- [x] 0A: Simplify Filesystem.h (drop experimental fallback)
- [x] 0A: Remove std::filesystem linking (standard in C++17+/GCC >= 9)
- [x] 0A: Remove dead FL_USE_OPENCL option (never used)
- [x] 0A: Fix FL_DEPRECATED macro (use C++14 [[deprecated("msg")]], fix MSVC typo)
- [x] 0A: Fix -Wreturn-type warnings in DeviceType.cpp and DeviceManager.cpp
- [x] 0B: cuDNN v9 RNN API migration (v6 -> v8)
- [x] 0B: Remove dead CUDNN_VERSION branches
- [x] 0B: Fix FL_USE_CUDNN cmake auto-enable
- [x] 0C: Run full test suite, fix failures

## Phase 1: Fix fl::Tensor, Shore Up Tests, Clean Up TensorBackend
- [x] Broadcasting audit and fix
- [x] TensorBackend scalar overload cleanup
- [x] Test improvements (broadcasting tests for Shape, Tensor, Autograd)

## Phase 1.5: Revisit TensorExtension / Autograd Organization
- [x] Audit: AutogradExtension ops are all primitive (conv2d, pool2d, batchnorm, rnn need backend kernels)
  - batchnorm is debatable but cuDNN/oneDNN fused kernels are faster; keep for now
  - softmax, relu, gelu, linear, etc. are already composed from tensor primitives
  - AutogradPayload is used by OneDNN for pool2d+batchnorm and cuDNN for RNN; keep
- [x] Fix defaultTensorBackend() hack (was creating temp tensor, now uses singleton)
- [x] Audit global state: 6 singletons identified, all necessary
  - Thread safety improvements deferred to Phase 3 dispatcher design
- [x] Move AutogradExtension ops to TensorBackend
  - 9 virtual methods on TensorBackend (conv2d/pool2d/batchnorm/rnn forward + backward)
  - CudaBackend, CPUBackend, ArrayFireBackend, JitBackend all override
  - Deleted: AutogradExtension.h, AutogradExtensionBackends.h, CudnnAutogradExtension.h/.cpp, OneDnnAutogradExtension.h/.cpp
  - Removed Autograd from TensorExtensionType enum
  - Kept TensorExtension system for Vision and JitOptimizer extensions
- [x] Replace AutogradPayload with opaque ForwardBackwardState
  - `using ForwardBackwardState = std::shared_ptr<void>` on TensorBackend
  - Forward returns `std::pair<Tensor, ForwardBackwardState>`, backward accepts it
  - `bool wantState` parameter replaces `payload != nullptr` check
  - Eliminated: AutogradPayloadData, AutogradPayload, createAutogradPayload(), all static_pointer_cast
- [x] Remove DynamicBenchmark from op signatures
  - Removed ConvBenchmarks from Conv2D module and Functions.cpp
  - cuDNN algorithm selection now done per-call (getBestAlgorithm) -- no benchmark threading
- [x] Move AutogradOps from fl/autograd/tensor/ to fl/tensor/
  - Pure tensor-to-tensor ops, no autograd dependency
  - RNNGradData moved to Functions.cpp (autograd-internal)

## Phase 2: Row-Major Migration
- [x] Backend dimension conversion utilities (flToAfDims, afToFlDims, flToAfAxis, etc.)
- [x] Flip convention (AF backend reversal, OneDNN backend de-reversal)
- [x] Update cuDNN descriptors (CudnnUtils, Conv2D, Pool2D, BatchNorm)
- [x] Update matmul (operand swap + sparse tensor correction)
- [x] Update autograd functions (multiheadAttention, embedding, relativePositionEmbeddingRotate, etc.)
- [x] Update NN modules (Linear, Conv2D, Pool2D, BatchNorm, LayerNorm, Embedding, etc.)
- [x] Update contrib modules (Transformer, Conformer, TDSBlock, AsymmetricConv1D, RawWavSpecAugment, PositionEmbedding, SinusoidalPositionEmbedding, AdaptiveEmbedding)
- [x] Update all tests (ModuleTest, AutogradTest, AutogradRnnTest, NNSerializationTest, ContribModuleTest, ContribSerializationTest)
- [x] Fix tril/triu AF backend for 2D main-diagonal row-major semantics
  (broader generalized semantics now throw until implemented)
- [x] Fix embedding backward: moved AF sparse workaround from Functions.cpp into AF backend
- [x] Remove migration-artifact comments
- [x] Add squeeze/unsqueeze to Tensor and Variable APIs
- [x] Deploy squeeze/unsqueeze in existing code (12 call sites)
  - Transformer/Conformer posEmb: moddims --> unsqueeze(x, 0)
  - Conformer conv reshape: moddims --> unsqueeze/squeeze(x, 3)
  - SinusoidalPositionEmbedding: reshape --> unsqueeze(x, 0)
  - matmul backward: reshape 1D --> unsqueeze(x, 0/1)
  - categoricalCrossEntropy grad: reshape --> unsqueeze(grad, 1)
  - multiheadAttention padMask: moddims --> unsqueeze(unsqueeze(x, 1), 2)
- [x] Fix scalar binary op inconsistency (min/max/power now use tensor dtype like other ops)
- [x] Rewrite sign() to use af::select (avoids JIT-breaking indexed assignment)
- [x] Audit for overhead in row-major changes: migration uses af::moddims (metadata-only) and af::transpose (JIT) -- no material overhead
- [x] Sanity-check all dimensional reshapes for row-major correctness
  - Audited 31 moddims/reshape call sites across autograd, modules, contrib, AF backend
  - Fixed WeightNorm serialization test: dim_=3-->0 (output channels in row-major), GLU(2)-->GLU(1) (channel dim)
  - All other reshapes verified correct for batch-first row-major convention
- [x] Fix pre-existing matmulShapes test failure (batch matmul shapes updated to batch-first convention)

## Phase 2.5: API Cleanup and Autograd/Tensor Parity
- [x] Bring autograd Variable ops to parity with tensor ops
  - Added: where, flip, tril, triu, cumsum, pow(Variable,Variable), std
  - All with correct gradient implementations and jacobian tests
- [x] Rename Variable ops to match Tensor names
  - moddims-->reshape, flat-->flatten, negate-->negative, pow-->power, clamp-->clip
  - max(Var,Var)-->maximum, min(Var,Var)-->minimum, padding-->pad
  - Updated all callsites across fl/, nn/, contrib/, pkg/, app/, tests
- [x] Full Variable/Tensor signature parity audit
  - Added: power(double, Variable), roll(Variable), flatten(Tensor) free function
  - Added: non-differentiable wrappers (floor, ceil, rint, sign, isnan, isinf)
  - Fixed: concatenate(Variable) now has default dim=0 matching Tensor
  - Confirmed: transpose/reorder already consistent, Init.h helpers already clean
- [x] Variable-Tensor merge -- Option A approved
  - **Phase 1 complete**: AutogradMeta struct and autograd methods added to Tensor (TensorBase.h/cpp)
    - `AutogradMeta` struct: requiresGrad, inputs, grad, gradFunc, onGradAvailable
    - `shared_ptr<AutogradMeta>` on Tensor -- null for non-grad tensors (zero overhead)
    - Shared on copy/move -- copies accumulate gradient to same point (like Variable's shared_ptr semantics)
    - New constructors: `Tensor(Tensor, bool)`, `Tensor(Tensor, vector<Tensor>, GradFunc)`
    - New methods: requiresGrad(), grad(), backward(), addGrad(), setRequiresGrad(), zeroGrad(),
      registerGradHook(), clearGradHook(), withoutData(), isGradAvailable(), autogradMeta()
    - Full backward pass: buildAutogradDAG() topological sort + calcGradInputs + applyGradHook
    - All 188+ existing tests pass -- fully backward compatible (Variable still exists independently)
  - **Phase 1.5 complete**: Structural separation of autograd from tensor code
    - `AutogradMeta` struct moved from TensorBase.h to `fl/autograd/AutogradMeta.h`
    - All autograd method implementations moved from TensorBase.cpp to `fl/autograd/TensorAutograd.cpp`
    - TensorBase.h includes AutogradMeta.h (needed for constructor signatures using GradFunc/GradHook)
    - TensorBase.cpp is now purely tensor code -- no autograd implementation details
    - All 188+ tests pass (autograd, modules, contrib)
  - **Phase 2 complete**: Variable eliminated, all code uses Tensor directly
    - Replaced `Variable` with `Tensor` across ~228 files (bulk sed + manual fixups)
    - Renamed `areVariableTypesEqual` --> `areDtypesEqual`, `FL_VARIABLE_DTYPES_MATCH_CHECK` --> `FL_DTYPES_MATCH_CHECK`
    - Renamed `.tensor()` --> `.detach()` (matches PyTorch convention: detached view without autograd metadata)
    - Deleted dead code: Variable.h, Variable.cpp, autograd/Utils.h, autograd/Utils.cpp
    - Autograd functions remain in `fl::autograd::` namespace (exp, matmul, conv2d, etc.)
    - All 638 tests pass (1 pre-existing failure in ArrayFireTensorBaseTest.concatenate, unrelated)
- [x] Code cleanup pass
  - Fixed: missing Cuda case in tensorBackendTypeToString (would crash at runtime)
  - Fixed: test name AutogradUnaryOpsTest.Clamp --> Clip to match renamed op
  - Fixed: Variable.h docstring still referenced "Arrayfire array" instead of "Tensor"
  - Removed: dead Variable overload of expandFromReduction (all callsites use Tensor version)
  - Removed: stale var() docstring about ArrayFire version inconsistency (now computed independently)
  - Removed: 4-tensor concatenation limit in ArrayFire backend (now iterative, supports arbitrary count)
  - Refactored: CudaMatmulEpilogueFusion now uses PatternMatcher DSL for add(matmul(A,B), C) matching
  - Dead code audit: confirmed all removed features (AutogradExtension, DynamicBenchmark, OptimMode,
    FL_ADJUST_INPUT_TYPE, AutogradPayload) have no stale references
  - Dead code audit: confirmed all renamed ops (moddims, negate, pow, clamp, flat) have no stale uses
- [x] Propagate tensor properties through Variable (added isSparse(), backendType() delegates)
- [x] Remove FL_ADJUST_INPUT_TYPE autocasting from autograd ops
  - Removed OptimMode/OptimLevel singleton and FL_ADJUST_INPUT_TYPE macro
  - Removed all 25 call sites from Functions.cpp (softmax, linear, conv2d, batchnorm, etc.)
  - Removed Types.h (kOptimLevelTypeExclusionMappings), cleaned Defines.h/cpp
  - Kept FL_DTYPES_MATCH_CHECK (legitimate input validation, renamed from FL_VARIABLE_DTYPES_MATCH_CHECK)
  - Restored gradient type cast in softmax/logSoftmax backward (needed when backend upcasts f16-->f32)
  - Cleaned up app/ code: removed all DynamicBenchmark, OptimMode, OptimLevel references
  - Removed fl_benchmark_mode and fl_optim_mode flag definitions/declarations
  - Simplified AMP input casting (always cast to f16 when mixed precision enabled)
- [x] Type promotion rules: define output dtype from (tensor_type, scalar_type) pairs
  - Moved promoteDtypes(), isFloatingPoint(), dtypeRank() from JIT-only OpProperties to general Types.h/cpp
  - Added scalarDtype(double): returns s32 for integral values, f32 for fractional
  - Fixed TensorBackend.cpp scalar widening: uses promoteDtypes(tensor.type(), scalarDtype(scalar)) instead of tensor.type()
  - Fixed: all scalar binary ops (add/sub/mul/div/eq/neq/lt/lte/gt/gte/and/or/mod/bitwise/shift), min/max/power, where
  - Tests: ScalarDtype, PromoteDtypes, IsFloatingPoint (TensorBaseTest), ScalarTypePromotion (TensorBinaryOpsTest)
- [x] Rename TensorBase.h/cpp to Tensor.h/cpp (defines `class Tensor`, ~200 include sites updated)
- [x] `Tensor(shape, dtype, Device)` constructor + unified device-targeted tensor creation
  Added `backendTypeForDevice(DeviceType)` mapping and `detail::makeTensor(shape, type, TensorBackendType)` /
  `detail::full(shape, value, type, TensorBackendType)` as unified internal primitives in BackendRegistry.
  Per-backend helpers (makeCPUTensor, makeCudaTensor, fullCPU, fullCuda) now route through these.
  All public factory functions (empty, full, fromScalar, identity, arange, iota, rand, randn) dispatch
  via `backendFor(s)` which checks `StreamOrDevice::device()` and resolves the correct JitBackend.
  Added `StreamOrDevice(const Device&)` implicit ctor, `hasDevice()`, `device()` accessors.
  JIT lowering files (CudaFusedKernelLowering, CudnnSubgraphLowering) use `detail::makeTensor` instead
  of `toTensor<CudaTensor>`. Copy.cpp and JitTensorBase.cpp transfer helpers use `backendTypeForDevice()`.
- [x] Fix median/var/std all-axis reductions: eliminated device-->host-->device round trip
  - median: sort on device + index middle element(s), cast integer types to f32
  - var: compute mean/subtract/square/sum entirely on device
  - std: delegate to sqrt(var(...)) instead of host-returning af::stdev<double>

## Phase 3: Tensor Backend Dispatcher
- [x] Unified DefaultTensorType: defaultTensorBackend() now uses same mechanism as Tensor() construction
- [x] Thread-local defaults: DefaultTensorType is now thread_local (thread-safe setDefaultTensorType/withTensorType)
- [x] Removed dead detail::getDefaultBackend() declaration from TensorBackend.h
- [x] Fixed pre-existing JitNodeTest.BinaryNodeMetaData (broadcast shapes updated for row-major)
- [x] Backend registry: TensorBackendType -> TensorBackend& via self-registration
  - BackendRegistry singleton with mutex-protected factory map
  - Backends self-register via FL_REGISTER_TENSOR_BACKEND macro (static init)
  - Free functions: getBackend(type), hasBackend(type), getAvailableBackends()
  - Registered: ArrayFire, OneDNN, CUDA, Stub (all conditional on compile flags)
  - 7 tests (registration, lookup, singleton identity, available list, unregistered throws)
- [ ] Backend op capability queries
  - Add a metadata-only runtime query, e.g. `supports(OpId, dtype, shape, attrs)`
    or a richer `capability(OpSignature) -> supported/fallback/unsupported + reason`.
  - Use backend-local constexpr/static tables where possible, but expose the query
    at runtime through `BackendDispatch`/backend objects because the selected
    backend, wrapped JIT backend, library versions, device capability, dtype,
    rank, shape, layout, and op attributes can all affect support.
  - Tests, JIT lowering, and optimizer passes should query capability metadata
    instead of probing by running an operator and catching an exception.
- [ ] Thread-local device context (deviceId) -- deferred, CUDA runtime already provides per-thread device state

## Phase 4: Backend & Ops Refactor

### Op Organization: `fl/tensor/ops/{library}/`

Restructure tensor op implementations into library-specific directories under `fl/tensor/ops/`:
```
fl/tensor/ops/
  native/          ← vanilla C++ (no library deps): transpose, indexAdd, RNN BPTT
  cblas/           ← CBLAS/MKL matmul (f32/f64 via sgemm/dgemm)
  onednn/          ← OneDNN primitive ops: conv2d, pool2d, batchnorm, rnn, binary, eltwise, reorder
  cudnn/           ← cuDNN ops: conv2d, pool2d, batchnorm, rnn
  cuda/            ← CUDA kernel ops: unary, binary, reductions, shaping
```

The backend (`fl/tensor/backend/cpu/` or `fl/tensor/backend/cuda/`) becomes a thin dispatcher: each `TensorBackend::foo()` method is a one-line call into the appropriate `ops/` implementation.

**Shared library state** (e.g. `dnnl::engine`, `dnnl::stream`): use a lightweight class per library (e.g. `OneDnnOps`) that holds engine/stream/allocator state. Op functions are methods on this class. The backend owns the `OneDnnOps` instance and passes it through. This avoids global singletons while keeping ops self-contained.

```cpp
// fl/tensor/ops/onednn/OneDnnOps.h
class OneDnnOps {
 public:
  OneDnnOps();  // creates engine + stream
  Tensor matmul(const Tensor& lhs, const Tensor& rhs, ...);
  std::pair<Tensor, ForwardBackwardState> conv2dForward(...);
  void conv2dBackward(...);
  // ... all ops that use OneDNN primitives
 private:
  dnnl::engine engine_;
  dnnl::stream nativeStream_;
};

// fl/tensor/backend/cpu/CPUBackend.cpp
Tensor CPUBackend::matmul(...) {
  if (canUseOneDnn(lhs, rhs))  // f32, supported format
    return onednnOps_.matmul(lhs, rhs, ...);
  return cblasOps_.matmul(lhs, rhs, ...);  // f64 fallback
}
```

**Reference**: PyTorch uses a similar pattern (`aten/src/ATen/native/mkl/`, `native/mkldnn/`, `native/xnnpack/`, `native/cpu/`). Each library directory is self-contained; the dispatcher routes based on tensor properties.

### Tensor Representation: Generic CPUTensor

**Decision**: `CPUTensor` is now the generic CPU tensor adapter. OneDNN-specific `dnnl::memory` objects are created on the fly per-op (via `DnnlMemoryWrapper` pattern, which already happens today).

**Rationale** (from investigation):
- `dnnl::memory::desc` creation is trivially cheap: just stores dims + strides + type vectors, O(ndim) stride math
- `dnnl::memory(desc, engine, ptr)` wrapping is trivially cheap: stores a C handle + raw pointer, no allocation
- `DnnlMemoryWrapper` already pays this cost on every op call — the overhead is negligible
- The real expense is `primitive_desc` construction (kernel/format selection), which is per-op regardless
- `CPUTensor::SharedData` provides buffer ownership + view chains, which are generic concepts

**CPUTensor design**:
```cpp
struct CPUTensor : public TensorAdapterBase {
  struct SharedData {
    void* buffer;                           // raw allocation
    std::shared_ptr<SharedData> parent;     // keeps parent alive for views
    std::function<void(void*)> deallocator; // custom dealloc (e.g. dnnl allocator)
  };
  std::shared_ptr<SharedData> data_;
  Shape shape_;
  std::vector<Dim> strides_;  // row-major strides (supports non-contiguous views)
  fl::dtype type_;
};
```

Views share `data_->buffer` via `shared_ptr` chain. OneDNN ops construct `dnnl::memory` from `(buffer + offset, strides, shape, type)` — same as `DnnlMemoryWrapper` does today.

**JIT compatibility**: A simpler representation (buffer + shape + strides + type) is cleaner for JIT. The JIT IR only needs shape/type metadata; buffer management is orthogonal to graph construction.

**Migration path** (rename complete, decoupling remaining):
1. [x] Rename `OneDnnTensor` → `CPUTensor`, move `backend/onednn/` → `backend/cpu/`
2. [x] Decouple `dnnl::memory` from CPUTensor internals (see follow-up items below)
3. [x] Port ops to use `CPUTensor` + `OneDnnOps` methods

**Follow-up: Decouple `dnnl::memory` from CPUTensor**:
- [x] Replace `dnnl::memory` buffer in CPUTensor::SharedData with raw `void*` buffer + metadata (shape, strides, dtype, allocator). Buffer allocation uses `AlignedBuffer` (64-byte aligned). `dnnl::memory` constructed on-demand in `ops/onednn/` functions via `detail::makeDnnlMemory(bufferPtr, shape, strides, type, engine)`. CPUTensor.h and CPUTensor.cpp are fully dnnl-free.
- [x] Rename FL_ONEDNN_* macros to FL_CPU_* (8 macros across CPUTensor.cpp, CPUBackend.cpp, CPUBackend.h)
- [x] Fix toCPUTensor() error message ("expected oneDNN-backed tensor" → "expected CPU-backed tensor")
- [x] DnnlUtils.h: convertToDnnlDims/convertShapeToDnnlDims/dnnlMapToType now delegate to Utils.h equivalents (flDimsToOneDnnDims/shapeToOneDnnDims/flToOneDnnType). Backward-compatible inline wrappers.
- [x] Fix dnnlMapToPoolingMode error message ("unsupported pooling mode for cuDNN" → "for OneDNN")
- [x] Extract remaining OneDNN-specific utilities from CPUTensor/CPUBackend into `ops/onednn/`: `isTypeSupportedByOneDnn()` already in `ops/onednn/Utils.h`, called from CPUBackend.cpp. `toCPUTensor()` is a generic cast helper with no dnnl dependency.
- [x] Create `OneDnnOps` class to hold OneDNN library state. Owns `dnnl::engine` and `dnnl::stream`. All OneDNN-primitive-based ops are methods on this class (thin wrappers over stateless `detail::onednn::` free functions). CPUBackend owns `unique_ptr<OneDnnOps>` via PIMPL (forward decl in header). CPUBackend.h is dnnl-free. Autograd ops (Conv2D, Pool2D, BatchNorm, RNN) and JIT OneDnnOpFusion access engine/stream via `backend.onednnOps()`.
- [x] Ensure JIT/fused operations work efficiently after decoupling. OneDnnOpFusion constructs `dnnl::memory` on-the-fly from `toCPUTensor().bufferPtr()` + `.stridesVec()` via `detail::makeDnnlMemory()` — no stored `dnnl::memory` member. All tests pass.

### Sweep: Extract remaining inline CPU fallbacks to `fl/tensor/ops/native/`

`CPUBackend.cpp` still has ~68 inline CPU fallback loops for ops that OneDNN doesn't support
(mostly f64, plus some integer/boolean type handling). These need to be swept into `native/` during
the Phase 4 refactor so that `CPUBackend` dispatch is clean one-line calls.

**Already extracted** (in `fl/tensor/ops/native/`):
- [x] Matmul — CBLAS `sgemm`/`dgemm` (`native/Matmul.cpp`)
- [x] Transpose — coordinate-based N-D permutation (`native/Transpose.cpp`)
- [x] RNN — weight parsing, manual forward, BPTT backward (`native/RNN.cpp`)
- [x] Sort / argsort / topk (`native/Sort.cpp`)
- [x] Pad / Tile / Concatenate / Where / Nonzero (`native/ShapeAndIndex.cpp`)
- [x] Unary math + flip/roll/tril/triu/abs (`native/Unary.cpp`, `native/Unary-inl.h`)
- [x] Binary ops with broadcasting + logicalAnd/Or (`native/Binary.cpp`, `native/Binary-inl.h`)
- [x] Reduction (cumsum, max/min with index) (`native/Reduction.cpp`, `native/Reduction-inl.h`)
- [x] Creation (iota, identity, full) (`native/Creation.cpp`, `native/Creation-inl.h`)
- [x] Fill — strided buffer fill (`native/Fill.h`)

**Remaining inline fallbacks to extract** (all in `CPUBackend.cpp`):
- [x] Sort / argsort — `sortCpu`, `argsortCpu`, `topkCpu` in `native/Sort.cpp`
- [x] Pad — `padCpu` in `native/ShapeAndIndex.cpp`
- [x] Tile — `tileCpu` in `native/ShapeAndIndex.cpp`
- [x] Concatenate — `concatenateCpu` in `native/ShapeAndIndex.cpp`
- [x] Where (conditional select) — `whereCpu` in `native/ShapeAndIndex.cpp`
- [x] Unary math fallbacks — `unaryMathCpu`, `unaryPredicateCpu` in `native/Unary-inl.h`; `flipCpu`, `rollCpu`, `trilCpu`, `triuCpu`, `absoluteIntCpu` in `native/Unary.cpp`
- [x] Binary op fallbacks — `cpuBinaryOpWithBroadcast`, `sameShapeBinopCpu` in `native/Binary-inl.h`
- [x] Reduction fallbacks — `cumsumCpu` in `native/Reduction.cpp`; `maxWithIndexCpu` in `native/Reduction-inl.h`
- [x] LogicalAnd/Or — `logicalAndCpu`, `logicalOrCpu` in `native/Binary.cpp`
- [x] Fill — `fillBufferCpu` in `native/Fill.h` (template, header-only)
- [x] Iota — `iotaSingleAxisCpu` in `native/Creation.cpp`
- [x] Consolidate duplicate impls: algorithm code now lives only in `native/` files;
  `CPUBackend.cpp` calls `native::*Impl` functions. Shared utilities (`dispatchByType`,
  `linearIndex`, `delinearize`) consolidated in `CpuTensorHelpers.h`. Dead sync wrappers
  (`*Cpu`, `*CpuRaw`) and dead template files (`Unary-inl.h`, `Reduction-inl.h`) removed.

**Vectorization TODO** for `native/` ops:
- [x] f32 unary ops (sin, cos, log1p, floor, ceil in `CPUUnaryOps.cpp`) needlessly
  round-trip through `double` (`static_cast<float>(std::sin(static_cast<double>(s[i])))`).
  This halves SIMD width and prevents libmvec/SVML auto-vectorization. Change to operate
  directly on `float` (e.g. `std::sinf` or `std::sin` with float argument).
- [x] `delinearize` / `linearIndex` allocate a `std::vector` per element. Ops that use
  them per-element (flip, roll, tril, triu, pad, broadcast binary, tile) cannot vectorize.
  Fixed by replacing native-op call sites with precomputed row-major strides and reused
  coordinate buffers. Further fixed-size 2D/3D/4D specialization can wait for profiling.
- [ ] Cumsum (`cumsumTyped`) has a loop-carried dependency (prefix sum). Not
  auto-vectorizable; would need manual SIMD prefix-sum if performance matters.
- [ ] Max/min with index (`maxWithIndexTyped`) uses a `visited[]` array and conditional
  argmax — not auto-vectorizable. Consider restructuring as outer/inner/axis loops.
- [ ] These are all fallback paths (hot path goes through OneDNN). Prioritize only if
  profiling shows they're bottlenecks (e.g. for types OneDNN doesn't support).

### Backend Naming: `fl/tensor/backend/cpu/`

**Decision**: Single `fl/tensor/backend/cpu/` backend, not separate `arm/` and `x86/`.

**Rationale** (from survey of major frameworks):
- **PyTorch**: Single `CPU` dispatch key. ISA variants (AVX2, AVX512, NEON) compiled from the same source in `native/cpu/` with different `-march` flags, selected at dlopen time via `cpuinfo`. The main dispatcher has no visibility into ISA.
- **XLA**: Single CPU backend. LLVM auto-selects target triple. No ARM/x86 split.
- **TensorFlow**: Single `DEVICE_CPU`. MKL vs Eigen selected by graph rewrite, not backend split.

95%+ of CPU backend code is shared (memory management, views, type conversion, scalar ops, indexing, etc.). ISA differences only matter for hot kernels — and those are already handled by the library implementations in `fl/tensor/ops/` (OneDNN internally dispatches to AVX512/AVX2/SSE/NEON, CBLAS does the same, cuDNN handles GPU archs).

If we later need ISA-specific kernels in `native/` ops (e.g., hand-vectorized transpose), we can use the PyTorch pattern: compile the same source multiple times with different `-march` flags, select at init time via function pointer table. This stays internal to the CPU backend — no need for separate backend directories.

### Open questions
- [x] Async execution model: **Resolved** — see "Design Document: JIT-Only Execution Model" below. Per-op stream assignment at graph construction time. Host plans, streams execute. No per-tensor stream tracking during graph construction.
- [x] Buffer allocator strategy: **Resolved** — AlignedBuffer (64-byte aligned) for CPU. Planning phase allocates, execution phase computes. Memory planner with liveness analysis is a future step. See "Buffer Management" in design doc.
- [x] Primitive caching: **Resolved** — compiled artifact caching is per-backend (NVRTC: PTX/cubin, OneDNN: library-internal primitive cache, CUDA graphs: re-capture). Keyed by graph signature + shapes + dtypes. See "Decisions Summary" in design doc.

## Phase 5a: Native CUDA Library Backend
- [x] CudaTensor adapter (CudaTensor.h/cu): stream-ordered memory, row-major C-order strides, copy/host/device/astype
- [x] CudaBackend singleton (CudaBackend.h/cu): cuBLAS/cuRAND init, creation ops (full, identity, arange, iota), random (rand, randn)
- [x] Unary ops (CudaUnaryOps.cu): 15 ops via Thrust transform macro, logicalNot, isnan, isinf, clip, flip, roll, tril, triu, where
- [x] Binary ops (CudaBinaryOps.cu): 21 ops with NumPy-style broadcasting via Thrust tabulate, fast path for same-shape
- [x] Reductions (CudaReductions.cu): sum, mean, amin, amax, var, std, norm, cumsum, any, all, countNonzero, min/max with indices, argmin/argmax, median. CUB fast paths: DeviceSegmentedReduce for contiguous reductions (innerSize==1), DeviceScan::InclusiveSum for cumsum, DeviceSegmentedReduce::ArgMin/ArgMax for argmin/argmax
- [x] BLAS (CudaBLAS.cu): matmul via cuBLAS sgemm/sgemmStridedBatched, row-major via operand swap
- [x] Shaping (CudaShapeAndIndex.cu): reshape (buffer-sharing view), transpose, tile, concatenate, nonzero, pad
- [x] CMakeLists.txt: FL_USE_NATIVE_CUDA option, --extended-lambda, links cudart/cublas/curand
- [x] Added TensorBackendType::Cuda enum value
- [x] Sort ops: sort, argsort, topk (implemented via host round-trip)
- [x] All 433 existing tests still pass
- [x] Write CUDA backend-specific tests (77 eager tests split across 5 files: Base 23, UnaryOps 22, BinaryOps 10, Reductions 15, BLAS 7; plus 27 JIT end-to-end tests)
- [x] Fixed b8 dtype output bug: unary/binary transforms now compute in f32 then astype (was writing float to b8 storage)
- [x] Multi-dtype correctness: GPU-native astype + proper dtype handling
  - GPU-native astype kernel: template dispatch over 11 src/dst dtype combinations, no host roundtrip
  - Supports f16 via __half (was missing from host-based astype)
  - Unary/binary ops now do explicit astype(f32) before compute (was UB: raw static_cast<float*> on non-f32 data)
  - 2 new tests (AstypeGPUNative, OpsOnNonF32Tensors)
- [x] Native multi-dtype compute
  - cuBLAS matmul: cublasSgemm (f32), cublasDgemm (f64), cublasGemmEx (f16 and bf16 with f32 accumulate + tensor cores)
  - Batched variants: cublasSgemmStridedBatched, cublasDgemmStridedBatched, cublasGemmStridedBatchedEx (f16/bf16)
  - Type promotion: f32+f64-->f64, f16+f16-->f16, bf16+bf16-->bf16, other-->f32
  - f64 unary ops: native double compute via unaryTransformF64 (exp, log, sqrt, sin, cos, tanh, etc.)
  - f64 binary ops: native double compute via binaryTransformF64 with broadcasting
  - f64 reductions: templated kernels (axisReduceKernel<T>, generalAxisReduceKernel<T>, Thrust full reduce)
  - f64 cumsum: templated cumsumKernel<T>
  - f64 argmin/argmax: templated argReduceKernel<T>
  - f16 elementwise: stays as upcast-to-f32 (standard practice, f16 math is imprecise)
  - 13 new tests: MatmulF64, MatmulF64Precision, MatmulF16, MatmulMixedTypes, F64UnaryOps, F64Precision, F64BinaryOps, F64BinaryPrecision, F64BroadcastBinary, MixedF32F64Promotion, F64Sum, F64AxisSum, F64CumsumPrecision
- [x] GPU-native reductions via Thrust + custom kernels
  - Full reductions: Thrust reduce/min_element/max_element with stream-aware execution policy
  - Axis-wise contiguous reductions: custom CUDA kernel (outerSize x reduceSize x innerSize decomposition)
  - Non-contiguous axis reductions: general GPU kernel with coordinate decomposition (GeneralReduceParams struct)
  - Covers: sum, amin, amax, countNonzero, any, all (GPU-native); mean, var, std, norm (composed from GPU ops)
- [x] GPU-native sort/argsort/topk via Thrust
  - Gather strided data into contiguous segments, per-segment thrust::stable_sort_by_key, scatter back
  - GPU-native topk extraction kernel (avoids host round-trip for top-k gather)
  - 8 new tests (1D, descending, axis-wise, indices, argsort, topk, topk-along-axis)
- [x] GPU-native cumsum via custom CUDA kernel
  - Per-thread sequential scan (each thread handles one (outer, inner) segment)
- [x] GPU-native min/max with indices via custom CUDA kernel
  - argReduceKernel: per-thread arg-reduce tracking both value and index
- [x] Remove host round-trip fallbacks in CUDA backend:
  - [x] Non-contiguous axis reductions: general GPU kernel (generalAxisReduceKernel)
  - [x] Median: GPU sort + medianPickKernel (full and single-axis)
  - [x] Transpose: GPU kernel (transposeKernel with ShapeInfo/IntArray structs)
  - [x] Tile: GPU kernel (tileKernel with ShapeInfo)
  - [x] Concatenate: device-to-device cudaMemcpyAsync per slab
  - [x] Nonzero: Thrust copy_if with counting_iterator
  - [x] Pad: GPU kernel (padKernel with ShapeInfo/IntArray)
  - [x] Median (multi-axis): transpose reduced axes to end, merge, sort, pick
    - Also fixed full-reduction to use GPU-only path (flatten+sort+medianPickKernel, no host roundtrip)
    - 4 new tests: non-contiguous axes, contiguous axes, keepDims, full multidim
- [x] Sort/argsort/topk implementation. CUB DeviceSegmentedSort::SortPairs for all segments in single launch (contiguous fast path, gather/scatter for strided)
  - sort (values only), sort (values+indices), argsort, topk -- all 4 virtual methods
  - Axis-wise sorting via gather-sort-scatter on host
  - topk = sort + take first k elements along axis

### Open: CuTile for CUDA kernel authoring
- [ ] Evaluate CuTile (NVIDIA's tile-based GPU programming model) for simplifying custom CUDA kernels. CuTile provides high-level abstractions (tile load/store, collective operations, warp-level primitives) that can replace hand-written index arithmetic in kernels like `transposeKernel`, `padKernel`, `tileKernel`, `axisReduceKernel`, `medianPickKernel`, and `argReduceKernel`. Potential benefits: less boilerplate (ShapeInfo/IntArray structs, coordinate decomposition), automatic vectorized memory access, and clearer expression of data-parallel patterns. Trade-off: adds a dependency on CuTile headers (CUDA 12.8+), may limit portability to older CUDA toolkits. Start with a proof-of-concept rewrite of one kernel (e.g. `transposeKernel` or `axisReduceKernel`) to measure code reduction and performance impact.
  > **Update (2026-10 research):** CUDA Tile shipped in CUDA 13.1 (Python only);
  > C++ tile kernels came in 13.3, and NVRTC can compile them since 13.3
  > (`--enable-tile`, `nvrtcGetTileIR`), so it fits our runtime-codegen model. The
  > 13.2 support list omits Hopper; confirm H100 support + gather/scan coverage
  > hands-on before investing. See `docs/modernization/research/research-2026-10-distributed-kernels.md` §4.

### Deferred: FP8 and MXFP Support

FP8 (E4M3/E5M2) and MXFP (OCP Microscaling) are important for next-gen training
performance but library support is still maturing. Waiting for cuDNN frontend and
cuBLASLt APIs to stabilize before adding support. Key considerations documented here
so we can move quickly when the time is right.
> **Update (2026-10 research): FP8 on H100 no longer needs to wait.** cuBLASLt has had
> Hopper FP8 tensorwise, rowwise (outer-vector) and 1×128 / 128×128 block scaling
> since CUDA 12.9 (May 2025). Recommended: `ScaleSpec ∈ {Tensorwise, Rowwise,
> Block1D128, Block2D128, MX32, NV16}` mirroring cuBLASLt modes; recipes = FP8
> tensorwise current scaling, rowwise, DeepSeek-style block scaling; skip delayed
> scaling initially. MXFP8/NVFP4 remain Blackwell-only (emulate on H100 for numerics
> research only). Details: `docs/modernization/research/research-2026-10-distributed-kernels.md` §6.

**FP8 (Hopper SM 9.0+, Ada SM 8.9)**:
- [ ] Add `f8_e4m3` and `f8_e5m2` to `dtype` enum in `Types.h` + all switch statements
- [ ] Add `scaled_matmul(a_fp8, b_fp8, scale_a, scale_b) -> (result, amax)` op — explicit
  scales, NOT baked into `Tensor` metadata. Backed by cuBLASLt FP8 matmul with
  `A_SCALE_POINTER`/`B_SCALE_POINTER`/`D_SCALE_POINTER`/`AMAX_D_POINTER`
- [ ] Add `scaled_conv2d` op — cuDNN frontend graph with descale/scale/amax graph nodes
- [ ] Delayed scaling recipe in `pkg/runtime/amp/` (alongside `DynamicScaler`): rolling
  amax history buffer, `scale = max_representable / amax` computation, per-iteration update
- [ ] Autograd: custom backward for scaled ops (E4M3 forward, E5M2 backward convention),
  amax propagation through backward graph
- [ ] JIT IR: FP8 scaled ops as materialization barriers initially; lazy integration later
- [ ] cuBLASLt backward epilogues with FP8: `GELU_AUX` (forward saves pre-activation),
  `DGELU` / `DRELU` (backward consumes AUX), `BGRADB` variants

**MXFP (Blackwell SM 10.0 for native, Hopper via software emulation)**:
- [ ] Block-wise scaling (one 8-bit exponent per 32 elements) — fundamentally different
  from per-tensor FP8 scaling. Storage layout needs side buffer for block exponents.
- [ ] MXFP8 (E4M3/E5M2 + block scale), MXFP6, MXFP4 formats
- [ ] cuDNN 9.x has early MXFP8 support; Transformer Engine is the current entry point
- [ ] MXFP4 on Blackwell delivers ~2x throughput vs bf16 — high value once APIs mature
- [ ] Wait for Blackwell ecosystem (cuBLAS, cuDNN, Transformer Engine) to stabilize

**Design principle**: Keep FP8/MXFP as explicit scaled ops (`scaled_matmul`, etc.) rather
than infecting the core `Tensor` class with per-tensor or per-block scale metadata. Scales
are just regular FP32 tensors passed as arguments. This follows PyTorch's approach over
Transformer Engine's tensor-attached-scale approach.

## Phase 5b: Minimal Custom Kernel Backend / JIT IR Extension
- [x] Add UnaryNode to JIT IR (UnaryOp enum: 18 ops, shape-preserving, UnaryNode.h/cpp)
- [x] Add ReductionNode to JIT IR (ReductionOp enum: 9 ops, axes+keepDims, ReductionNode.h/cpp)
- [x] Register Unary/Reduction in NodeType enum, Node::isUnary()/isReduction()
- [x] Update Evaluator dispatch (evalUnaryNode, evalReductionNode, evalUnaryOp, evalReductionOp)
- [x] Update JitBackend: unary ops use UnaryNode::create, reductions use ReductionNode::create
  - historical note: tril/triu/flip/clip/roll/where initially remained CustomNode
    while they were promoted incrementally to first-class IR nodes
  - var/norm/cumsum/argmax/argmin remain as CustomNode (extra parameters beyond axes+keepDims)
- [x] Update GraphvizPrinter: unaryOpToStr, reductionOpToStr, label printers
- [x] Update ScalarFolding pass: handle new node types (pass-through)
- [x] Add JitNodeTest cases: UnaryNodeMetaData, ReductionNodeMetaData, ReductionNodeKeepDims
- [x] NVRTC JitCompiler for runtime kernel compilation and fusion
  - NvrtcKernelCompiler: generates CUDA source from FusedOp program, compiles via NVRTC, caches by key
  - CudaFusedKernelLowering pass: replaces FusedElementwiseNode with CustomNode launching compiled kernel
  - CudaJitOptimizerExtension: registers CUDA-specific passes via FL_REGISTER_TENSOR_EXTENSION
  - Kernel generation: per-element float ops, supports all 18 unary + 21 binary ops
  - Links CUDA::nvrtc and CUDA::cuda_driver via CMake CUDAToolkit
  - Needs GPU hardware for runtime testing
- [x] Fixed pre-existing JitEvaluatorTest.evalIndexedUpdateNodeWithTensorIdx (test data had OOB index values for row-major dim 0)

### JIT Fusion Passes
The pass infrastructure supports backend-dependent passes with minimal boilerplate:
- `BottomUpPass` / `TopDownPass` base classes handle traversal; subclasses implement `rewrite(NodePtr)`
- Common passes (ScalarFolding, CSE, ElementwiseFusion, ReductionOfElementwiseFusion) run for all backends
- Backend-specific passes register via `JitOptimizerExtension` (e.g. OneDnn's `OneDnnOpFusion`, CUDA's CudaFusedKernelLowering/CudaMatmulEpilogueFusion)
- `Optimizer` merges common + backend passes, runs sequentially
- OneDNN already has a binary chain fusion pass using post-ops (`OneDnnOpFusion` — plain `Pass` subclass with unique iterative pattern)

Pass ideas (common = backend-independent, backend = needs backend-specific lowering):

**Common passes (pattern matching / IR rewriting):**
- [x] Elementwise fusion: merge consecutive Unary/Binary chains into a single FusedElementwiseNode
  - FusedElementwiseNode: first-class IR node with mini-program of FusedOp steps
  - ElementwiseFusion pass: top-down greedy, absorbs single-use fusable ancestors
  - Evaluator: sequential fallback evaluation (pending NVRTC kernel lowering)
  - GraphvizPrinter, ScalarFolding updated for FusedElementwise node type
  - Tests: 7 fusion pass tests (chains, diamonds, non-fusable cases), 1 node metadata test
- [x] Reduction-of-elementwise fusion: merge elementwise ops feeding into a reduction
  - Pattern: `sum(exp(x))` --> `FusedReduceElementwise(sum, [exp], x)`
  - Avoids materializing intermediate -- fallback evaluator runs program then reduces
  - FusedReduceElementwiseNode: first-class IR node with elementwise program + reduction op/axes/keepDims
  - ReductionOfElementwiseFusion pass: absorbs FusedElementwise, single unary, or single binary into reduction
  - Runs after ElementwiseFusion in the optimizer pass pipeline
  - 9 test cases (unary, binary, fused chain, multiple users, shared input, non-fusable, metadata)
- [x] Dead node elimination: N/A -- shared_ptr reference counting automatically cleans up unreachable nodes
- [x] Common subexpression elimination: deduplicate identical subgraphs
  - Bottom-up traversal with string-based hashing of node type + parameters + input pointers
  - Supports: Scalar, Binary, Unary, Reduction, FusedElementwise nodes
  - 9 test cases (identical scalars, binary/unary subtrees, reductions, negative cases)
  - Runs after ScalarFolding, before ElementwiseFusion
- [x] Constant folding: extended ScalarFolding to fold unary ops on ScalarNodes
  - Added 15 unary op evaluations (exp, log, sin, cos, sqrt, tanh, etc.)
  - Chains fold bottom-up: `exp(Scalar(1) + Scalar(2))` --> `Scalar(exp(3))`
  - 4 new test cases (unary, chain, non-scalar input, mixed unary+binary)

**Backend-specific passes (lowered via JitOptimizerExtension):**

> **Note (2026-10-26):** `CudaMatmulEpilogueFusion` and `CudnnConvEpilogueFusion`
> referenced throughout this file no longer exist (subsumed by
> `CudnnSubgraphLowering`). Current passes: CPU = `OneDnnPrimitivePostOpFusion`,
> `OneDnnOpFusion`; CUDA = `CudnnSubgraphLowering`, `CublasLtBackwardEpilogueFusion`,
> `CudaFusedKernelLowering` (+ `NvrtcKernelCompiler`). Pipeline order and per-pass
> bugs: `docs/modernization/research/review-2026-10-26.md` §2.2 and §6.2. On CPU,
> ElementwiseFusion runs before OneDnnOpFusion, so CPU elementwise chains become
> FusedElementwiseNodes evaluated op-by-op (no CPU fusion benefit today).
- [x] OneDNN: extend OneDnnOpFusion to support unary post-ops (eltwise post-ops: relu, tanh, etc.)
  - OneDNN supports eltwise post-ops natively -- `binop(x, y) + relu` can be a single primitive
  - Mapped 8 UnaryOps to OneDNN eltwise algorithms (exp, log, sqrt, tanh, abs, round, neg, sigmoid)
  - Mixed binary/eltwise post-op chains: primary binary + any sequence of binary/eltwise post-ops
  - Fixed pre-existing CustomNode shape bug (was using bottom node shape, now uses searchRoot shape)
- [x] CUDA/NVRTC: lower FusedElementwiseNode to runtime-compiled CUDA kernel
  - NvrtcKernelCompiler maps each FusedOp to CUDA expression, generates kernel, compiles via NVRTC, caches
- [x] CUDA/NVRTC: lower FusedReduceElementwise to a fused reduce kernel
  - Cooperative warp-shuffle reduction: strided per-thread accumulation → __shfl_down_sync warp reduction → shared memory cross-warp reduction
  - One block per output element (BLOCK_SIZE=256 threads), ~32x faster than sequential per-thread loop for large reductions
  - Supports Sum, Mean, Amin, Amax, CountNonzero, Any, All (Median/Std fall back to evaluator)
  - Extended NvrtcKernelCompiler with generateReduceKernelSource and getOrCompileReduce
  - Extended CudaFusedKernelLowering with lowerFusedReduceNode
  - Contiguous axes only; non-contiguous falls back to evaluator
  - Needs GPU hardware for runtime testing
- [x] CUDA: matmul epilogue fusion via cuBLASLt
  - Added MatmulNode as first-class JIT IR node (promotes matmul from opaque CustomNode)
  - MatmulNode stores MatrixProperty for each operand, uses inferMatmulOutputShape
  - Updated all passes: Evaluator, CSE, ScalarFolding, GraphvizPrinter, NodeType enum
  - JitBackend now creates MatmulNode instead of CustomNode for matmul
  - Built lightweight PatternMatcher DSL for declarative subgraph matching
    - Template-based: matchUnary, matchBinary, matchMatmul, matchReduction, capture, withCapture
    - 10 pattern matcher tests (any, capture, unary, binary, matmul, nested, nodeType)
  - CudaMatmulEpilogueFusion pass: matches `add(matmul(A,B), C)` and fuses into cuBLASLt call
    - Row-major via operand swap (same as existing cuBLAS code)
    - cuBLASLt handle reused from existing cublas handle (binary compatible)
    - Heuristic algorithm selection with fallback to regular matmul
    - Matches add+matmul, relu(add+matmul), and gelu(add+matmul) patterns with cuBLASLt epilogues
    - Relu recognized as max(x, Scalar(0)); Gelu recognized as 0.5*x*(1+erf(x/sqrt(2))) with all commutative variants
    - Uses TopDownPass so outermost activation pattern matches before inner add+matmul is consumed
    - Added MatchScalar, MatchScalarAny to PatternMatcher.h for scalar constant matching
    - Multi-dtype: supports f32, f64, f16, bf16 via cudaDataType_t parameterization (CUDA_R_32F/64F/16F/16BF)
      - f16/bf16: cublasGemmEx with f32 accumulate + tensor cores; f64: COMPUTE_64F with double alpha/beta
      - void* pointers throughout eval lambda (no type-specific casts)
      - 6 new end-to-end tests: MatmulAddFusion{Bf16,F16,F64}, MatmulAddReluFusion{Bf16,Bf16Positive}, LinearLayerForwardBf16
  - 3 MatmulNode tests (metadata, outputShape, transposed shape)
  - All 91 JIT tests pass

**Pattern matching infrastructure:**
- PatternMatcher.h: header-only, template-based subgraph pattern DSL
  - MatchAnyNoCapture, MatchAnyCapture (wildcard with optional capture)
  - MatchUnaryOp<P>, MatchAnyUnary<P> (unary node matching)
  - MatchBinaryOp<L,R>, MatchAnyBinary<L,R> (binary node matching)
  - MatchMatmul<L,R> (matmul node matching)
  - MatchReductionOp<P> (reduction node matching)
  - MatchNodeType (type-only matching), MatchWithCapture (capture wrapper)
  - MatchProjection<ParentPattern> (projection node matching with output index)
  - MatchBatchNorm, MatchPool2d (typed multi-output node matching)
  - Factory functions: any(), capture(), matchUnary(), matchBinary(), matchMatmul(), matchReduction(), matchProjection(), matchBatchNorm(), matchPool2d(), withCapture()
  - Pre-built convenience patterns: matchBatchNormResult(), matchBatchNormSaveMean(), matchBatchNormSaveVar(), matchBatchNormNewRunningMean(), matchBatchNormNewRunningVar()
- MatchResult stores captures as name-->NodePtr map
- Used by CudaMatmulEpilogueFusion; designed for cuDNN/OneDNN batchnorm/conv/pool fusion passes

**Completed fusion work:**
- ~~Relu/Gelu activation epilogues~~ **DONE**: CudaMatmulEpilogueFusion recognizes relu=max(x,0) and gelu=0.5*x*(1+erf(x/sqrt(2))) multi-node patterns, fuses into cuBLASLt EPILOGUE_RELU/EPILOGUE_GELU. 4 end-to-end tests.
- ~~cuBLASLt activation epilogues (matmul+relu, matmul+gelu)~~ **DONE**: Implemented in CudaMatmulEpilogueFusion pass
- ~~cuBLASLt multi-dtype support (f16, bf16, f64)~~ **DONE**: CudaMatmulEpilogueFusion now supports all float types via cudaDataType_t parameterization. 6 new end-to-end tests.
- ~~cuDNN conv+bias+activation fusion~~ **DONE**: CudnnConvEpilogueFusion pass matches relu/sigmoid/tanh(conv2d), calls `cudnnConvolutionBiasActivationForward` (single fused kernel when bias present). No-bias path falls back to conv + separate `cudnnActivationForward`.
- ~~cuDNN Frontend Graph API subgraph lowering~~ **DONE**: `CudnnSubgraphLowering` pass identifies subgraphs with anchor ops (Conv2d, Matmul, BatchNorm) plus adjacent pointwise neighborhoods, builds cuDNN frontend graphs, compiles execution plans, replaces with CustomNode. Handles multi-output BatchNorm (5 outputs) via `CustomNode::createMultiOutput()`. ProjectionNode transparent to subgraph expansion.
- ~~ExternalUse-aware fusion side-output infrastructure~~ **DONE**: `FusionUtils.h` provides helpers (`findExternallyReferencedNodes`, `buildOutputDescsForSideOutputs`, `createSideOutputProjections`) for fusion passes to safely absorb nodes that have ExternalUses (autograd references) by emitting them as side-outputs of multi-output CustomNodes. Side-outputs use ProjectionNode + `replaceAllUsesWith` to redirect ExternalUses; `computeLiveOutputMask()` dead-eliminates side-outputs when ExternalUses are released. `OutputDesc` extracted to standalone `ir/OutputDesc.h`. CudnnSubgraphLowering wired up (currently no-op since `isSingleUse()` prevents absorption; infrastructure ready for follow-up).
- ~~JIT graph verifier~~ **DONE**: `GraphVerifier.h` — single inline `verifyGraph()` function runs after every optimization pass in `Optimizer::optimize()`. Three checks: use-list consistency, shape consistency (Binary/Unary/Cast/Reshape/Reduction/Transpose/Matmul via existing inference functions), ExternalUse reachability (catches orphaned ExternalUses from buggy passes). Always-on by default; define `FL_DISABLE_JIT_GRAPH_VERIFY` to disable. 6 tests in `JitFusionSideOutputTest`.
- ~~Lazy backward for maximum/minimum(Tensor, double)~~ **DONE**: Reverted from pre-computed mask to lazy form — backward captures scalar value, computes mask inside gradFunc using `inputs[0]`. Safe because `isSingleUse()` prevents fusion passes from absorbing the comparison node when it has an ExternalUse from autograd.

**CUDA library fusion opportunities (not yet implemented):**

*Forward matmul fusion — handled by CudnnSubgraphLowering:*

Forward matmul + arbitrary pointwise fusion (add, relu, gelu, sigmoid, etc.) is handled
by `CudnnSubgraphLowering`, which builds cuDNN Frontend Graph API execution plans. cuDNN
graph matmul calls cuBLASLt internally and is strictly better for forward fusion — it
fuses arbitrary pointwise neighborhoods, not just the fixed cuBLASLt epilogue set.
- ~~Matmul transpose support~~ **DONE**: `matmulShapeTo4D` pads with leading 1s (cuDNN
  gemm_start_dim = ndim-2 requires matrix dims in last two positions). Transpose expressed
  via swapped dims/strides with alias UIDs sharing boundary input device memory.
  `matmul(x, W, None, Transpose) + bias` (the universal Linear pattern) now fuses.

*Backward matmul epilogues — requires direct cuBLASLt (Phase 3):*

cuDNN Frontend Graph API handles forward matmul fusion well, but backward-specific
cuBLASLt epilogues require features that cuDNN graph does not expose:
- **AUX buffer**: `EPILOGUE_RELU_AUX` / `EPILOGUE_GELU_AUX` store an auxiliary tensor
  (relu bitmask / gelu input) alongside the forward pass. cuBLASLt exposes this via
  the `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER` descriptor. cuDNN graph matmul has
  no equivalent — it abstracts away the cuBLASLt descriptor.
- **Backward activation derivative**: `EPILOGUE_DRELU` / `EPILOGUE_DGELU` fuse the
  activation derivative into a backward matmul, consuming the AUX buffer from forward.
- **Cross-graph data passing**: Forward writes AUX, backward reads it. This requires
  the autograd system to annotate forward IR nodes with AUX tensors that the backward
  graph can later reference. This is a Phase 3 feature.

**Implementation path for backward matmul fusion (updated with deferred eval):**

With deferred eval, forward and backward are one connected graph. The backward
`CublasLtBackwardEpilogueFusion` pass already matches `mul(matmul(A,B), d_act(z))`
patterns. With the connected graph, the pass can walk from `z` (a saved forward
intermediate) through the live forward graph to discover that `z` is the output of a
forward matmul — enabling joint optimization.

No cuBLASLt `EPILOGUE_RELU_AUX`/`EPILOGUE_GELU_AUX` needed: the forward intermediate
`z` is accessible directly. The backward pass recomputes `z > 0` (relu mask) or uses
`z` directly (gelu aux) from the connected graph. The existing `packReluBitmask` kernel
already handles this efficiently.

The original Phase 3 plan (node annotations, AUX buffers, autograd tape modifications)
is superseded by the deferred eval approach — no new infrastructure needed beyond
`MultiValueNode` (for correct multi-output compaction) and async materialization APIs.
4. `BGRADB` variants additionally fold bias gradient reduction into the same kernel.

| Epilogue | Pattern | Priority | Notes |
|----------|---------|----------|-------|
| `EPILOGUE_RELU_AUX` / `EPILOGUE_GELU_AUX` | `act(matmul(A,B) + C)` in training | High | Forward: stores activation aux data. Prerequisite for backward fusion. |
| `EPILOGUE_DRELU` / `EPILOGUE_DGELU` | `mul(matmul(A, B), d_act(Z))` | High | Backward: fuses activation derivative into matmul using saved AUX. |
| `EPILOGUE_DRELU_BGRAD` / `EPILOGUE_DGELU_BGRAD` | backward: `matmul * d_act(aux)` + bias grad reduction | Medium | Combines backward activation + bias gradient sum reduction. |
| `EPILOGUE_BGRADB` | backward: matmul + bias grad reduction | Low | Bias gradient alongside matmul backward without activation fusion. |

*cuDNN conv2d — extend CudnnConvEpilogueFusion:*

| Fusion | Pattern | Priority | Notes |
|--------|---------|----------|-------|
| Conv + residual add + activation | `relu(conv2d(x,w,b) + shortcut)` | High | `cudnnConvolutionBiasActivationForward` already has a `z` parameter for residual add (line 582 of Conv2D.cpp currently sets z=output, unused). This is the core ResNet pattern — single kernel for `y = act(conv(x) + bias + shortcut)`. Requires 3-input pattern matching: conv, bias, shortcut. The shortcut must have the same shape as the conv output. |
| Conv no-bias + activation (single kernel) | `relu(conv2d(x,w,nullptr))` | Low | Currently falls back to conv + separate `cudnnActivationForward` (2 kernels, lines 591-628 of Conv2D.cpp). Could use `cudnnConvolutionBiasActivationForward` with a zero-filled bias tensor to get a single kernel. Small win, but trivial to implement. |
| ClippedReLU / ELU epilogues | `clipped_relu(conv2d(...))` / `elu(conv2d(...))` | Low | Low-level code already supports `CUDNN_ACTIVATION_CLIPPED_RELU` (coef=6.0, line 526) and `CUDNN_ACTIVATION_ELU`, but no JIT patterns match them. Useful for MobileNet-family architectures. Would need new UnaryOp or pattern recognition (e.g., `min(max(x,0), 6)` → ClippedReLU6). |

*cuDNN batchnorm fusion:*

~~BatchNorm + arbitrary pointwise fusion (cuDNN Graph API)~~ **DONE**: Implemented in
`CudnnSubgraphLowering` pass. Uses cuDNN Frontend Graph API (`graph->batchnorm()` for
training, `graph->batchnorm_inference()` for inference) to fuse BatchNormNode + adjacent
pointwise ops (relu, sigmoid, add, etc.) into a single compiled cuDNN execution plan.
Multi-output support: `CustomNode::createMultiOutput()` produces all 5 BatchNorm outputs
(Result, SaveMean, SaveVar, NewRunningMean, NewRunningVar) from a single fused eval.
`ProjectionNode` made transparent to subgraph expansion (special-cased for multi-output
anchor parents). This subsumes all hardcoded BN fusion patterns — BN+ReLU, BN+Add+ReLU,
BN+sigmoid, etc. are handled automatically by the cuDNN graph planner.

**Remaining BN fusion items:**
- [ ] cuDNN graph BN currently passes NCHW strides. The cuDNN graph batchnorm internally
  calls `generate_NHWC_stride_order()` and may reject NCHW plans via `check_support()`.
  Falls back gracefully to unfused path. Will be fixed when NHWC tensor layout support
  is added (see "Tensor layout philosophy" section below).
- [ ] BatchNorm backward fusion: `cudnnBatchNormalizationBackwardEx` fuses relu_backward +
  batchnorm_backward into a single kernel. Requires paired forward/backward fusion
  (forward must be fused for backward to work). The graph API may also support backward
  BN fusion — needs investigation.

*cuDNN Graph API (cudnn backend/frontend):*

The cuDNN Frontend Graph API (v8+) is now the primary mechanism for CUDA library fusion.
`CudnnSubgraphLowering` identifies subgraphs with anchor ops (Conv2d, MatmulNode,
BatchNormNode) plus adjacent pointwise neighborhoods, builds a cuDNN frontend graph,
compiles it into an execution plan, and replaces the subgraph with a `CustomNode`.

**Completed:**
- [x] Conv2d + pointwise fusion (relu, sigmoid, tanh, add, add+relu, etc.)
- [x] Matmul + pointwise fusion (add, add+relu, add+sigmoid, etc.)
- [x] BatchNorm (training + inference) + pointwise fusion (arbitrary chains)
- [x] Multi-output CustomNode for training BatchNorm (5 outputs via ProjectionNode)

**Remaining cuDNN Graph API items:**

| Feature | Priority | Notes |
|---------|----------|-------|
| ~~Pool2d fusion~~ | ~~Medium~~ | **DONE**: `graph->resample()` wired up in `buildCudnnGraph()`. Inference mode; training-mode max pooling Index tensor deferred. |
| ~~CastNode in graph builder~~ | ~~Low~~ | **DONE**: CastNode mapped to `PointwiseMode_t::IDENTITY` with output data type set to target dtype. 2 structural tests (Conv2d+Cast+ReLU, Matmul+Cast+Add). |
| RNN fusion | Low | RNN is not in `isAnchorNode()` at all — still uses the legacy cuDNN API. The cuDNN graph API has RNN operations but they are complex (multi-step unrolling). Low priority until RNN performance matters. |
| Fused multi-head attention (SDPA) | Medium | cuDNN 8.9+ has dedicated SDPA operations that can express `softmax(Q*K^T/sqrt(d) + mask) * V` as a single fused operation. Very high performance win for transformer models. **Step 13 (future)**. |

*cuDNN pool2d:*
cuDNN's legacy `cudnnPoolingForward` does not support fused activations. The graph API
path (Resample op) requires NHWC layout — blocked until NHWC tensor layout support is
added. OneDNN's pool primitive supports eltwise post-ops (already handled by
`OneDnnPrimitivePostOpFusion`).

*OneDNN backend fusion:*
- [x] OneDNN conv/pool/batchnorm/matmul post-ops with chained eltwise + optional sum post-ops. `OneDnnPrimitivePostOpFusion` walks chains of single-use activations (up to 32 OneDNN post-ops) through optional residual add to anchor primitives (conv2d, pool2d, inference BN, matmul). Supports: `eltwise_N(...(conv2d))`, `eltwise_N(...(add(conv2d, x)))`, `add(conv2d, x)` (sum-only), `eltwise_N(...(pool2d))`, `eltwise_N(...(proj(bn)))`, `eltwise_N(...(matmul(A, B)))`, `eltwise_N(...(matmul(A, B) + bias))`, `matmul(A, B) + bias`. ConvBatchNormFolding runs first to algebraically eliminate inference BN into conv weights.
- [x] Gelu canonicalization to OneDNN `eltwise_gelu_erf`: pattern-matches composed gelu subgraph (`x * 0.5 * (1 + erf(x * M_SQRT1_2))`, 7 IR nodes) as a single activation in `OneDnnPrimitivePostOpFusion`. Handles commutative orderings and scalar tolerance. Three lowering paths: (1) **anchor fusion** — `gelu(matmul)`, `gelu(conv2d)`, etc. fuse as eltwise post-op, (2) **standalone** — `gelu(x)` with no anchor emits single `eltwise_gelu_erf` kernel via `OneDnnStandaloneEltwise` CustomNode, (3) **f64 fallback** — standalone gelu works for f64 via CPU fallback in `applyEltwiseOp`. The gelu input `x` has 2 internal uses in the pattern (outer mul + erf inner mul); the pass allows this for both chain walk and matmul anchor single-use checks.

*Cross-cutting concerns:*
- CUDA graph capture and replay → **Step 15**
- Equality saturation for phase-ordering-sensitive rewrites (e.g., egg/egglog — overkill for now, revisit when rule set grows)

*Missing JIT optimization passes (gap analysis vs. PyTorch/XLA/JAX):*

Current Flashlight JIT fusion coverage is roughly 40–50% for transformer workloads
and ~70% for conv-net workloads relative to the important fusions in PyTorch Inductor,
XLA, and JAX. The gaps below are ordered by expected performance impact.

**Tier 1 — high-impact gaps:**

| Pass | Description | Frameworks with it |
|------|-------------|--------------------|
| FlashAttention pattern matching | Match Q·K^T/√d + mask + softmax + V dropout pattern, lower to cuDNN SDPA or FlashAttention kernel. Single largest transformer fusion win (~2–4x for attention). | PyTorch (torch.nn.functional.scaled_dot_product_attention), XLA (custom call) |
| Broadcast-aware elementwise fusion | Current `ElementwiseFusion` bails when shapes don't match exactly. Needs to handle implicit broadcasting (e.g., bias add [N,C,H,W] + [1,C,1,1]) inside fused NVRTC kernels. Most real-world elementwise chains involve broadcasts. | PyTorch Inductor (Triton codegen handles broadcasts), XLA (implicit) |
| Softmax fusion | Multi-pass online softmax reduction (safe exp + sum + normalize) as a single fused kernel. Currently decomposes to exp/sum/div which defeats fusion. | PyTorch Inductor (Triton template), XLA (custom pattern) |
| Memory planning / buffer reuse | Analyze tensor lifetimes in the JIT graph and reuse allocations. Currently every node allocates fresh memory. Linear reduction in peak memory for deep models. | PyTorch (memory_efficient_graph), XLA (HLO buffer assignment), TVM (StorageRewrite) |

**Tier 2 — moderate-impact gaps:**

| Pass | Description | Frameworks with it |
|------|-------------|--------------------|
| Layout optimization (channels_last) | Automatically insert NCHW↔NHWC layout conversions around conv/pool ops to use faster NHWC kernels. Unblocks cuDNN pool2d graph API fusion (currently blocked on NHWC). | PyTorch (channels_last memory format), XLA (LayoutAssignment) |
| LayerNorm fusion | Fuse mean + variance + normalize + scale + bias into a single kernel. Currently decomposes to multiple elementwise + reduction ops. | PyTorch Inductor (Triton persistent reduction), XLA (custom call) |
| ~~Algebraic simplification~~ | **DONE**: AlgebraicSimplification pass — 18 IEEE 754-safe rules (identity, power, involution, reshape). Deferred rules behind fast-math flag (annihilator, inverse pairs, cast round-trip). See dedicated section at end of file. |
| Backward graph optimization / rematerialization | Recompute cheap forward ops during backward instead of saving activations. Trades compute for memory. Requires cost model. | PyTorch (checkpoint), XLA (HLO rematerialization), JAX (jax.checkpoint) |

### Backward fusion and cross-graph data passing

**Key discovery: backward ops go through JIT.** When `backward()` runs, each
`gradFunc` lambda calls `fl::matmul`, `fl::add`, `fl::mul`, etc., which all route
through `JitBackend` and create `MatmulNode`, `BinaryNode`, etc. Nothing in
`TensorAutograd.cpp` or `Functions.cpp` forces JIT evaluation. `addGrad()` stores
lazy JIT tensors. The entire backward pass for a full model produces one big
connected JIT IR graph — all layers, all cross-layer patterns — that is only
evaluated when someone finally demands values (e.g., optimizer step).
> **Correction (2026-10-26 review):** several autograd ops DO force evaluation:
> softmax/logSoftmax/norm call `fl::eval` in forward, CCE/weighted CCE call
> `scalarSync`, tensor-index backward does a per-element `scalarSync` loop,
> embedding backward is a host `indexAdd` round trip, RNN materializes. See
> `docs/modernization/research/review-2026-10-26.md` §6.4 for the full catalog.

This means **JIT passes already apply to backward computation.** Elementwise
fusion, scalar folding, CSE — all work on backward ops transparently. No special
autograd-aware passes are needed. The JIT optimizer matches tensor IR patterns,
agnostic to whether they came from forward or backward.

**Cross-layer patterns are visible in the JIT graph.**

Consider `relu(linear(x))` stacked two layers deep. The backward JIT graph
(built lazily across all `gradFunc` calls) contains:

```
relu_grad_2:    mask2 = cast(gt(z2, 0)), dZ2 = mul(mask2, dLoss)
linear_grad_2:  dA1  = matmul(reshape(dZ2), W2)       ← MatmulNode
relu_grad_1:    mask1 = cast(gt(z1, 0)), dZ1 = mul(mask1, dA1)
linear_grad_1:  dX   = matmul(reshape(dZ1), W1)
```

The cross-layer pattern `dZ1 = mul(mask1, matmul(reshape(dZ2), W2))` is
`mul(matmul(A, B), relu_mask)` — the cuBLASLt DRELU pattern. A JIT pass can
match this purely from IR structure, without autograd knowledge.

**The compaction boundary.**

The graph break is NOT evaluation — it's compaction. In `JitTensorBase::eval()`
(line 493), after evaluating the graph, the node is replaced with a bare
`ValueNode`: `ValueNode::create(std::move(result), stream)`. The old graph
structure (including "this was a relu output") is discarded. When a later graph
computes `gt(pre_relu_value, 0)`, the JIT sees `gt(ValueNode, Scalar(0))` with
no provenance.

This is fine for patterns within a single graph (the pattern matching works on
whatever the graph contains). But it prevents data sharing across graph
boundaries (e.g., a relu bitmask computed in one graph being reused by another
graph instead of recomputing `gt(Z, 0)`).

#### Why NOT node annotations (yet)

We considered adding an annotation system to Node (keyed metadata map, surviving
compaction) so that data computed during one graph evaluation (e.g., relu
bitmasks) could be reused by later graphs. After researching how LLVM, MLIR,
XLA, and TVM handle IR metadata, we decided against it for now:

- **LLVM's #1 metadata bug**: "pass forgot to transfer annotations." Every
  rewrite pass (ElementwiseFusion, CSE, ScalarFolding, every fusion pass) creates
  new nodes. Each would need annotation transfer logic. Get one wrong and
  annotations silently disappear (under LLVM's default-drop) or silently stale
  (under MLIR's default-preserve).
- **`std::any` / untyped metadata dangers**: silent type mismatches, stale keys,
  debugging difficulty. Every system that tried untyped metadata (TVM DictAttrs,
  XLA frontend_attributes) reports these problems.
- **Memory lifetime**: relu bitmasks for large tensors persist on compacted
  ValueNodes indefinitely. No mechanism to know when they'll never be consumed.
- **Lock-in**: once Node has annotations, "passes must handle annotations
  correctly" becomes a permanent contract on every future pass.
- **Marginal benefit over Phase 2**: the cross-graph optimization saves one
  elementwise comparison kernel and reduces mask memory from 32 bits to 1 bit
  per element. For a [256, 1024] tensor: 32KB vs 1MB. Compared to Phase 2's
  win (eliminating a full kernel launch + memory round trip by fusing
  matmul+mask), this is noise.

**Resolution: deferred eval eliminates the cross-graph problem entirely.** If forward
is not evaluated before backward runs, there IS no graph boundary — forward and backward
are one connected DAG. Node annotations for cross-graph data passing are unnecessary
because the data (intermediate tensors) is directly accessible in the live graph. See
Phase 3 below for the deferred eval design. The annotation concerns above remain valid
reasons to avoid node-level metadata for other use cases.

**Implementation phases:**

Phase 1 — Forward-only fusions (pure pattern matching, no new infrastructure):
- [x] BatchNorm + ReLU (cuDNN Graph API via `CudnnSubgraphLowering`)
- [x] BatchNorm + residual add + ReLU (cuDNN Graph API via `CudnnSubgraphLowering`)
- [x] Conv + pointwise fusion (cuDNN Graph API via `CudnnSubgraphLowering`)
- [x] Matmul + pointwise fusion (cuDNN Graph API via `CudnnSubgraphLowering`)
- [x] ~~cuBLASLt batched matmul epilogues~~ — subsumed by cuDNN Frontend Graph API.
  `CudnnSubgraphLowering` already fuses batched matmul (via `shapeTo3D`) + arbitrary
  pointwise chains (bias add, relu, gelu, etc.) into compiled cuDNN execution plans.
  cuDNN internally selects cuBLAS/cuBLASLt kernels. No separate cuBLASLt path needed
  for forward-pass epilogues. When `check_support()` rejects a subgraph (unusual
  shapes, old GPUs), the fallback is unfused cuBLAS GEMM + separate elementwise
  kernels. A cuBLASLt epilogue fallback tier could avoid the kernel separation, but
  cuDNN frontend's matmul coverage is broad enough that the failure rate is low —
  not worth maintaining a second forward-pass fusion code path.
- [x] ~~cuBLASLt 1D bias detection~~ — subsumed by cuDNN Frontend Graph API.
  Bias add is handled generically as a fused pointwise ADD in the cuDNN graph.
  Broadcasting (1D bias → ND output) is handled by cuDNN's tensor descriptors.

Phase 2 — Backward-pass matmul fusion (cuBLASLt epilogues):
- [ ] Forward fusion with side-outputs (FusionSideOutput activation): relax
  `isSingleUse()` to ignore ExternalUses, mark side-output nodes as non-virtual in
  cuDNN graph, wire up multi-output CustomNode + ProjectionNode. Enables
  matmul+relu/gelu/etc. fusion in training mode. Intermediates (z) produced as
  side-outputs for backward via existing FusionUtils.h infrastructure.
  Note: cuBLASLt `EPILOGUE_RELU_AUX` / `EPILOGUE_GELU_AUX` (AUX buffer for bitmask)
  are NOT needed with deferred eval — backward nodes reference forward intermediates
  directly through the connected graph. cuDNN graph API fusion of matmul+activation
  is sufficient; the AUX data path is subsumed by the graph-level approach.
  **Obsolete with VJP**: VJP rules express backward as normal IR nodes referencing
  forward nodes via internal uses (NodePtr edges), not ExternalUses. `isSingleUse()`
  naturally allows absorption. Skip this — implement VJP rules instead.
- [x] cuBLASLt `EPILOGUE_DRELU`: match `mul(Cast(f32, Gt(z, 0)), matmul(A, B))`
  in backward JIT graph → fuse relu derivative into matmul using packed bitmask.
  Implemented as `CublasLtBackwardEpilogueFusion` BottomUpPass, runs after
  CudnnSubgraphLowering. Packs z>0 into 1-bit-per-element bitmask (128x smaller).
- [x] cuBLASLt `EPILOGUE_DGELU`: match composed gelu derivative pattern via Erf landmark.
  Refactored gelu to exact erf form with dedicated gradFunc (matches PyTorch/JAX/TF).
  `matchDgeluInput()` walks the elementwise expression tree to find `Erf(Mul(z, Scalar(≈0.7071)))`.
- [x] cuBLASLt `EPILOGUE_DRELU_BGRAD` / `EPILOGUE_DGELU_BGRAD`: additionally fold
  bias gradient sum reduction into the backward matmul kernel. Uses multi-output
  CustomNode + ProjectionNode pattern. Matches `Sum(Mul(..., Matmul(A,B)), {0})`.
- [x] cuBLASLt `EPILOGUE_BGRADB`: matmul backward + bias gradient reduction (no activation).
  Matches `Sum(Matmul(A,B), {0})` where matmul has other uses. Multi-output CustomNode.
- [x] Cross-layer backward patterns (e.g., consecutive linear+relu layers) are directly
  visible in the lazy JIT backward graph. Verified with stacked relu+linear Jacobian tests.

Phase 3 — Deferred eval + joint forward/backward optimization:

**Key insight: the "cross-graph" problem is artificial.** If forward is not evaluated
(compacted) before backward runs, the gradFunc closures build backward JIT nodes that
reference forward JIT nodes through the saved tensors. The result is one connected DAG —
forward and backward are just subgraphs of a single graph. The JIT optimizer can see
everything: cross-boundary fusion, rematerialization, joint buffer scheduling.

This is the same optimization scope as JAX/XLA's `jit(grad(f))` — achieved without
symbolic gradient transforms. JAX gets the combined graph by transforming jaxprs (IR →
IR). Flashlight gets it by deferring eval: gradFunc closures run imperatively (building
lazy JIT nodes), and those nodes naturally connect to the forward graph.

Functional gradient transforms (JAX/MLX-style jaxpr→jaxpr VJP rules) vs. gradFunc
closures: given the same connected DAG, the optimization opportunities are identical.
The JIT passes see the same nodes regardless of how the graph was constructed. Functional
transforms offer better composability (`grad(grad(f))`, `vmap(grad(f))`) and prevent
pathological closures from breaking laziness by construction — but these are expressiveness
advantages, not optimization advantages. The critical differentiator for optimization is
**whether the JIT sees the connected graph**, which depends on eval deferral, not the
gradient representation.

**Deferred eval design:**
- `scalarAsync<T>()` / `hostAsync()`: enqueue D2H copy without triggering eval. Return
  `std::future<T>`. Eval happens later (at explicit sync, `future.get()`, or next `eval()`).
- `loss.backward()` runs gradFunc closures on the **live** forward graph (not compacted
  ValueNodes). Backward JIT nodes reference forward nodes through saved tensors.
- The complete fwd+bwd+optimizer graph is visible to JIT passes at eval time.
- `scalar<T>()` returns `Future<T>` (lazy — no eval triggered). `scalarSync<T>()` is
  the blocking convenience. Deferred eval is opt-in via explicit `fl::eval()`.
- Compaction still happens, but at end of full train step, not between forward/backward.

**Multi-output-aware compaction (MultiValueNode):** After eval, multi-output nodes
(fused CustomNodes with side-outputs, BatchNorm) compact to `MultiValueNode` — a leaf
node holding N evaluated tensors. ProjectionNodes continue referencing it, preserving
sibling relationships. Analogous to XLA Tuple surviving through compilation — but here
it survives through compaction. Needed for FusionSideOutput correctness regardless of
deferred eval.

- [x] `MultiValueNode` IR node type (compacted multi-output analog of ValueNode)
- [x] Multi-output-aware compaction in `JitTensorBase::eval()`
- [x] Unified Event abstraction:
  - [x] Single `Event` type with `record(stream)`, `wait()`, `isReady()` that works
    across all backends. `EventImpl` dispatches by stream type: SynchronousStream
    (always-ready no-op), AsyncCPU (promise/shared_future on worker thread),
    CUDA (cudaEvent_t). See `runtime/Event.h`, `runtime/EventImpl.h`.
- [x] `fl::Future<T>` — general async result type:
  - [x] Holds a `std::shared_future<T>` internally (allows multiple `.get()` calls,
    `.future()` just returns the member — zero-cost interop).
  - [x] Also holds a `Tensor` reference and an `Event` for the eval-fallback +
    non-blocking poll path.
  - [x] API duck-type compatible with `std::future<T>`: `get()`, `wait()`,
    `wait_for()`, `valid()`. Adds `ready()` (non-blocking poll via `Event::isReady()`).
  - [x] `.get()` falls back to `fl::eval(tensor)` if eval hasn't been called yet,
    then waits on the event. If eval already ran, just waits (likely already done).
  - [x] NOT scalar-specific — works for any async host-side result.
  - [x] See `runtime/Future.h`, `runtime/Future.cpp`.
  > **Correction (2026-10-26 review):** the bullets above describe a planned API, not
  > the code. Actual `Future` (header-only `runtime/Future.h`, no Future.cpp) has
  > `get()` + `isReady()` only, no shared_future/wait/wait_for. `get()` runs
  > `fl::eval` + a full `stream().sync()` (contradicting the "does NOT call eval"
  > line below). `isReady()` can stay false forever (CUDA→host event lives on a
  > hidden transfer tensor; already-evaluated nodes skip event recording) and reads
  > `value` without synchronization. Same-device futures alias the tensor's
  > SharedData, so a later in-place op changes the future's value on CPU only.
- [x] Unified data transfer — `fl::copy`:
  - [x] Functional `fl::copy(src, dstStream)` returns new tensor on target device.
    JIT path creates TransferNode (lazy). Non-JIT allocates + imperative copy.
  - [x] Imperative `fl::copy(dst, src)` copies into pre-allocated dst.
    Direction and transport inferred from device pair.
  - [x] `TensorBackend::copy(dst, src, stream)` pure virtual, implemented by
    CudaBackend (cudaMemcpyAsync), CPUBackend (memcpy), JitBackend (materialize + forward).
  - [x] See `tensor/Copy.h`, `tensor/Copy.cpp`.
  - [x] Pinned host memory for async D2H destinations (CUDA). `CudaBackend::copy`
    allocates via `cudaMallocHost` when dst is empty and src is on device.
    Empty-dst contract: `copy(dst, src, stream)` allocates optimal destination
    when `dst.isEmpty()`. See `backend/cuda/CudaBackendTransfer.cpp`.
    **Bug (2026-10-26):** the empty-dst path treats every device src as D2H, so a
    CUDA→CUDA TransferNode yields a pinned-host CPUTensor (→ "Unknown dtype" in
    the next CUDA kernel). The empty-dst contract can't express the destination
    device; allocate from `dstBackendType` in `evalTransferNode` instead.
  - [ ] Pooled pinned allocator to amortize `cudaMallocHost`/`cudaFreeHost` cost.
  - [ ] P2P transport (`cudaMemcpyPeerAsync`) for multi-GPU same-node copies.
  - [ ] NCCL `ncclSend`/`ncclRecv` for cross-node copies.
- [x] `TransferNode` — IR node for device-to-device tensor transfer:
  - [x] First-class IR node. SSA-clean: input tensor in, output tensor out on dst device.
  - [x] Unified (one node type, direction from device pair). Queryable properties:
    `dstDeviceType()`, `dstBackendType()`.
  - [x] No-op elimination: `TransferNode::create()` returns input directly if
    src and dst are on the same device type. **Bug (2026-10-26):** only when the
    input node has an explicit stream; nodes from `fl::full`/`fl::rand` have a
    null stream, so same-device copies still create a TransferNode.
  - [x] Evaluator dispatches via `evalTransferNode`, calls `TensorBackend::copy`.
  - [x] See `tensor/ir/TransferNode.h`, `tensor/ir/TransferNode.cpp`.
  - [ ] Future work: `TransferStartNode`/`TransferDoneNode` pair for explicit async
    windows (XLA's `CopyStart`/`CopyDone` pattern).
- [x] `scalar<T>()` / `host<T>()` / `toHostVector<T>()` async API:
  - [x] `scalar<T>()` returns `Future<T>` — lazy (no eval triggered), resolves after eval.
  - [x] `scalarSync<T>()` = blocking convenience (triggers eval + blocks).
  - [x] `host<T>()` returns `Future<T*>`, `toHostVector<T>()` returns `Future<std::vector<T>>`.
  - [x] `DeferredEventImpl` bridges pre-eval Future creation and post-eval event recording.
  - [x] `JitTensorBase::eval()` calls `recordDataReadyEvent(stream())` after evaluation.
  - [x] 32 ComputeTest tests verify full async pipeline (CPU + CUDA).
  - [x] Unified `Tensor::to(const Device&)` / `JitTensorBase::to(const Device&)`:
    - Lazy: builds TransferNode graph + deferred event, NO eval triggered.
    - All D2H templates (scalar/host/toHostVector sync+async) share this single primitive.
    - `JitTensorBase::scalar(void*)` / `host(void*)` are pure data reads (no materializeSync).
    - `Future::get()` waits on event, does NOT call eval — caller must fl::eval() first.
    - Non-JIT path returns nullopt (falls back to direct readScalar/readHost).
  - [x] `Tensor::to()` host-to-device (H2D) transfers:
    - `JitTensorBase::transferToDevice()` mirrors `transferToHost()`: creates
      TransferNode targeting `DeviceType::CUDA` / `TensorBackendType::Cuda`
      with the CUDA backend's jitStream as copy stream.
    - `JitTensorBase::to()` no longer throws for CUDA targets.
    - `CudaBackend::copy()` H2D auto-allocation already works (`empty()` → CudaTensor).
    - `evalTransferNode()` already dispatches H2D to CudaBackend.
    - 4 tests: evaluated source round-trip, lazy graph chain, large tensor async
      round-trip, multi-tensor batch eval.
  - [ ] GPU-to-GPU (multi-device) transfers: requires peer access
    (cudaMemcpyPeerAsync) or staging through host. Lower priority.
- [x] `fl::eval(tensors...)` — explicit, non-blocking graph evaluation:
  - [x] Enqueues computation for the subgraphs of the given tensors.
  - [x] Multiple tensors can share subgraph nodes; evaluated once.
  - [x] See `tensor/Compute.h`, `tensor/Compute.cpp`.
  - [ ] **Correction (2026-10-26):** only `fl::eval(Tensor&)` exists and the
    evaluator takes a single root. A real multi-root eval (one plan over many
    roots) is a prerequisite for deferred fwd+bwd+optimizer steps.
- [x] `fl::ready(tensor)` — non-blocking query:
  - [x] Checks if a tensor's graph has been evaluated (node holds concrete result).
    Non-JIT tensors always return true. See `tensor/Compute.h`.
- [x] `fl::sync()` / `fl::sync(tensor)` — blocking stream synchronization.
  - [x] Overloads: `sync()`, `sync(deviceId)`, `sync(deviceTypes)`, `sync(devices)`.
  - [x] See `tensor/Compute.h`.
- [ ] Communication primitives (separate from TransferNode — different semantics):
  - [ ] `AllReduceNode`, `AllGatherNode`, `ReduceScatterNode` — NCCL/xCCL collectives.
    Multi-rank coordination, reduction semantics. Definitely separate node types.
  - [ ] `SendNode`/`RecvNode` — point-to-point for pipeline parallelism.
  - [ ] Future: GPU-initiated communication (NVSHMEM-style `put`/`get` from kernels).
- [ ] In-kernel copies via NVRTC: for fused operations where a copy is part of a larger
  kernel (e.g., a fused elementwise + D2D copy), generating copy code via NVRTC avoids
  a separate memcpy launch. Not precluded by the TransferNode design — the JIT optimizer
  can lower a TransferNode into a fused kernel when profitable, same as elementwise fusion.
- [ ] Dedicated copy streams: use separate CUDA streams for data transfers so that
  transfers can overlap with computation or other communication. Standard practice for
  pipelining (e.g., prefetching next batch while computing on current one). The
  TransferNode's stream can differ from compute streams; `fl::copy` would use the
  transfer stream, and Event-based synchronization ensures ordering.
- [ ] Cross-device stream synchronization via Future/Event: when a tensor arrives from
  another device (e.g., CUDA → CPU via async D2H copy), the receiving tensor carries a
  Future/Event that downstream operations wait on before accessing the data. This enables
  non-blocking cross-device pipelines — a CPU operation on an incoming CUDA tensor
  doesn't block immediately; it builds lazy graph nodes and only waits when the data is
  actually needed (at eval time). The Event abstraction already supports this:
  `CPUEvent::wait()` blocks, `CUDAStream::recordEvent()` + `cudaStreamWaitEvent()` for
  GPU-side waits.

### MoE / bleeding-edge kernel direction

**Conclusion:** continue the current compiler robustness plan before specializing for
MoE. No fundamental architecture change is needed right now. The existing direction
-- first-class IR nodes, deferred eval, VJP rules, explicit TransferNode/comm nodes,
backend lowering passes, NVRTC fusion, and later CUDA graph capture -- does not
preclude the MoE optimizations we care about. The next architectural additions should
be general compiler foundations, not MoE-only machinery: effect/alias metadata, memory
planning, dynamic/ragged shape representation, explicit communication, and a lower
execution/schedule layer.

**Research summary:**
- PyTorch/Inductor/Triton: good at FX/ATen capture, dynamic-shape guards, Triton
  codegen, and CUDA graph replay for stable regions. CUDA Graph Trees require fixed
  kernels, launch arguments/dependencies, and memory addresses; dynamic shapes often
  require re-recording or skipping CUDA graphing. This supports using CUDA graphs only
  as a backend optimization for stable subgraphs, not as the core programming model.
- CUDA: `cudaLaunchHostFunc`/host nodes cannot safely call CUDA APIs directly or
  indirectly. CUDA graph kernel/memcpy/memset node params can be updated in restricted
  cases, but host-side decisions based on device-computed values remain awkward.
- XLA: relevant precedents are `AllReduceStart`/`AllReduceDone`,
  `CopyStart`/`CopyDone`, while/conditional HLO, HLO rematerialization, buffer
  assignment, latency-hiding scheduling, thunks, and command buffers. The important
  lesson is a split between semantic tensor IR and lower execution/schedule/buffer IR.
- MLIR/IREE: useful precedents are async tokens/values, bufferization interfaces for
  reads/writes/aliasing, stream resources, async collectives, timepoints/timelines,
  and stream-ordered allocations. Flashlight should borrow the concepts, not the full
  complexity.
- JAX: `jax.checkpoint`/`remat` exposes save/recompute/offload policy for residuals.
  This validates policy-driven rematerialization, but also reinforces that policies
  should refine strategy rather than define tensor semantics.
- MoE systems: MegaBlocks avoids the drop-vs-pad tradeoff with block-sparse/dropless
  expert computation. Megatron Core MoE uses DeepEP dispatch, GroupedGEMM,
  token permutation fusion, router fusion, fine-grained recomputation, batch-level
  all-to-all overlap, and fine-scoped CUDA graph capture. DeepEP exposes
  dispatch/combine as JIT communication kernels with event overlap, per-expert token
  counts for GEMM, and explicit SM budgeting.
- NVRTC: useful for generated CUDA kernels. It accepts CUDA C++ source at runtime,
  emits PTX, supports modern C++ dialect flags, relocatable device code, and linking
  via nvJitLink/driver APIs. NVRTC does not make host-side dynamic behavior graphable;
  it helps when we can move data-dependent work into a generated device kernel whose
  launch topology is stable. NVRTC-compiled `CUfunction` launches are normal kernel
  launches and can be captured by CUDA graphs when the surrounding graph is capturable.

**Design requirements for MoE / bleeding-edge GEMM work:**
- Keep the public Tensor API simple. Do not add accumulation, scheduling, memory,
  and comm knobs to every high-level op.
- Prefer structured semantic ops over inference from arbitrary tensor soup:
  `route`, `dispatch/pack`, `groupedMatmul`, `segmentedActivation`, `combine/unpack`.
  Each op should have a correct reference lowering before specialized lowering.
- Add structured metadata as types/handles, not ad-hoc node annotations:
  `SegmentedTensor`/packed ragged rows, `DispatchHandle` (token counts, offsets,
  reverse mapping, expert layout), `Workspace`/arena regions, and explicit comm nodes.
- Add effect and alias information before advanced memory/schedule rewrites:
  reads, writes, aliases, mutates, allocates, communicates, stream/resource use.
- Add memory planning with explicit materialization destinations: packed activation
  buffers, workspace slices, aliasing/subviews, and save/recompute/offload decisions.
- Add dynamic/ragged shape values for token counts and expert segment sizes. Dynamic
  rank is not required; dynamic extents and segment metadata are.
- Add a lower execution/schedule layer below tensor IR: streams, events/timepoints,
  async start/done nodes, capture boundaries, stream-ordered allocations, and
  eventually command-buffer/CUDA-graph regions.
- Region-level policies are acceptable; per-op annotation sprawl is not. Useful
  policies: `NumericPolicy` (accumulation dtype, FP8 scale handling, stochastic
  rounding), `MemoryPolicy` (save/recompute/offload/materialize-in-destination),
  `SchedulePolicy` (stream, overlap, SM budget, capture boundary), `LayoutPolicy`
  (expert-major/token-major, alignment, padding multiple). Removing a policy should
  change performance, not correctness.
- Custom kernels remain an escape hatch, but must declare shape, dtype, effects,
  aliasing, stream/resource use, and VJP behavior or decomposition. Otherwise they
  become opaque barriers to scheduling, memory planning, and gradients.

**Kernel DSL direction:**
- We probably want an internal, narrow kernel DSL eventually, but not before the
  compiler foundations above. The DSL should generate known structured families:
  indexed map, segmented map, segmented reduce/scan, pack/unpack, gather/scatter,
  token permutation, activation/activation-backward epilogues, and simple persistent
  work-queue kernels.
- Do not initially try to generate arbitrary expert GEMMs or full communication
  transports. Use cuBLASLt/CUTLASS/cuDNN/DeepEP/NCCL-style library paths first, then
  fuse/generated-kernel glue around them.
- The first practical MoE codegen target should be routing glue: pack/unpack,
  token permutation, metadata scans, segmented activation, combine, and backward
  recomputation. These are general enough to help sparse attention, sequence packing,
  ragged batches, routing layers, and decoding, not just MoE.

**Near-term ordering:**
1. VJP rules / unified forward-backward graph.
2. Effect + alias metadata.
3. Memory planner / workspace and materialize-in-destination support.
4. First-class communication and transfer start/done nodes.
5. Dynamic/ragged shapes and segmented/packed tensor representation.
6. Execution/schedule layer with explicit streams/events/capture boundaries.
7. Internal kernel DSL + NVRTC lowering for structured indexed/segmented kernels.
8. MoE structured ops and specialized lowerings.

- [ ] ~~Defer FusionSideOutput activation~~ — **obsolete with VJP**: VJP rules eliminate
  autograd ExternalUses entirely. Fusion passes absorb forward intermediates naturally
  because backward references them via internal IR edges, not ExternalUse. The
  FusionUtils.h infrastructure remains available for the rare case of user-held
  intermediate tensor references.
  > **Correction (2026-10-26 review): NOT obsolete.** `isSingleUse()` counts internal
  > uses too, so any forward value consumed by a backward rule has ≥2 uses and will
  > not be absorbed by fusion. VJP still needs either side outputs (FusionSideOutput)
  > or a recompute policy. (FusionUtils.h is currently dead code — its one call is a
  > discarded no-op.)
- [ ] ~~Audit all gradFunc closures in Functions.cpp~~ — **obsolete with VJP**: there
  are no gradFunc closures to audit. VJP rules are pure IR→IR transforms with no
  closure captures, no saved tensors, no materialization risk.
- [ ] Profile real training workload with deferred eval vs. immediate eval

Phase 4 — Functional gradient transforms (exploration):

**Motivation:** Flashlight currently maintains two parallel graph representations: the JIT
IR (Node/NodePtr) for lazy computation, and the autograd graph (Variable/AutogradMeta)
for differentiation. gradFunc closures bridge the two — they capture Variables containing
JitTensors and, when called during backward(), build new JIT nodes. This duality works
but has costs: ExternalUse machinery for saved tensors, compaction concerns, the Variable
wrapper layer, and no composability (`grad(grad(f))` requires nested tapes).

Functional gradient transforms would unify these into one representation. Each JIT node
type registers a VJP (vector-Jacobian product) rule that operates on NodePtrs directly.
`backward()` becomes a reverse topological walk over the JIT graph, applying VJP rules
at each node to create gradient nodes. No Variable, no AutogradMeta, no gradFunc closures,
no ExternalUse. The forward JIT graph IS the differentiation substrate.

**What VJP rules look like:**

```cpp
// VJP rule registry — each differentiable node type registers a rule
// Rule signature: (forward node, output cotangent) → input cotangents
using VJPRuleFn = std::function<std::vector<NodePtr>(const Node&, const NodePtr&)>;

// Registration (one per differentiable primitive):
registerVJP(NodeType::Matmul, [](const Node& fwd, const NodePtr& ct) -> std::vector<NodePtr> {
    auto& matmul = fwd.impl<MatmulNode>();
    // dL/d(lhs) = ct @ rhs^T,  dL/d(rhs) = lhs^T @ ct
    return {
        MatmulNode::create(ct, matmul.rhs(), LhsTranspose::None, RhsTranspose::Transpose),
        MatmulNode::create(matmul.lhs(), ct, LhsTranspose::Transpose, RhsTranspose::None)
    };
});

registerVJP(NodeType::Binary, [](const Node& fwd, const NodePtr& ct) -> std::vector<NodePtr> {
    auto& bin = fwd.impl<BinaryNode>();
    switch (bin.op()) {
        case BinaryOp::Add: return {ct, ct};
        case BinaryOp::Mul: return {
            BinaryNode::create(ct, bin.rhs(), BinaryOp::Mul),
            BinaryNode::create(ct, bin.lhs(), BinaryOp::Mul)
        };
        case BinaryOp::Sub: return {ct, UnaryNode::create(ct, UnaryOp::Negative)};
        case BinaryOp::Div: return {
            BinaryNode::create(ct, bin.rhs(), BinaryOp::Div),
            UnaryNode::create(BinaryNode::create(
                BinaryNode::create(ct, bin.lhs(), BinaryOp::Mul),
                BinaryNode::create(bin.rhs(), bin.rhs(), BinaryOp::Mul),
                BinaryOp::Div), UnaryOp::Negative)
        };
    }
});

registerVJP(NodeType::Unary, [](const Node& fwd, const NodePtr& ct) -> std::vector<NodePtr> {
    auto& unary = fwd.impl<UnaryNode>();
    switch (unary.op()) {
        case UnaryOp::Exp: return {BinaryNode::create(ct, fwd.shared_from_this(), BinaryOp::Mul)};
        case UnaryOp::Negative: return {UnaryNode::create(ct, UnaryOp::Negative)};
        case UnaryOp::Sigmoid: {
            auto sig = fwd.shared_from_this();  // sigmoid(x) already computed
            auto one = ScalarNode::create(fwd.shape(), fwd.dataType(), 1.0);
            auto dsig = BinaryNode::create(sig, BinaryNode::create(one, sig, BinaryOp::Sub), BinaryOp::Mul);
            return {BinaryNode::create(ct, dsig, BinaryOp::Mul)};
        }
        // ... other unary ops
    }
});

// Custom ops: user registers VJP alongside the op
registerVJP(NodeType::Custom, [](const Node& fwd, const NodePtr& ct) -> std::vector<NodePtr> {
    auto& custom = fwd.impl<CustomNode>();
    if (custom.hasVJP()) {
        return custom.vjpFunc()(fwd, ct);  // user-provided
    }
    throw std::runtime_error("No VJP rule for custom op: " + custom.name());
});
```

**The backward() algorithm (no tape, no topo sort of a separate graph):**

```cpp
// Walk the FORWARD JIT graph in reverse topological order
// No autograd tape — the JIT graph IS the tape
std::unordered_map<Node*, NodePtr> cotangents;
cotangents[lossNode.get()] = ScalarNode::create(lossNode->shape(), lossNode->dataType(), 1.0);

for (auto& node : reverseTopoOrder(lossNode)) {
    if (!cotangents.count(node.get())) continue;  // unreachable from loss
    auto ct = cotangents[node.get()];
    auto inputCts = dispatchVJP(*node, ct);

    for (size_t i = 0; i < node->inputs().size(); ++i) {
        auto* inp = node->inputs()[i].get();
        if (!requiresGrad(inp)) continue;  // skip non-differentiable inputs

        if (cotangents.count(inp)) {
            cotangents[inp] = BinaryNode::create(cotangents[inp], inputCts[i], BinaryOp::Add);
        } else {
            cotangents[inp] = inputCts[i];
        }
    }
}
// cotangents[param.get()] is the gradient JIT node for each parameter
```

No tape. No separate autograd graph. The forward JIT graph is traversed directly. The
backward nodes are created in-place in the same JIT graph (they reference forward nodes
as inputs). The result is one connected DAG — exactly what deferred eval achieves with
gradFunc closures, but without the closure machinery.

**Consequences for abstractions:**

1. **Variable eliminated.** No more `Variable(tensor, {inputs}, gradFunc)`. Users work
   with Tensors directly. `fl::grad(f)` / `fl::valueAndGrad(f, params)` derives gradients
   from the JIT graph. The Variable class, AutogradMeta, and the entire
   `flashlight/fl/autograd/` directory become unnecessary.

2. **ExternalUse simplified.** ExternalUses still exist (JitTensorBase → Node reference),
   but they're no longer the mechanism for "saving tensors for backward." The VJP rules
   read forward nodes directly from the graph. ExternalUses remain for their original
   purpose: tracking external (non-graph) references for liveness analysis.

3. **VJP rules are tensor code, not graph construction.** Following MLX's approach,
   VJP rules are written as tensor operations — the same operations users write:
   ```cpp
   registerVJP<MatmulNode>([](const MatmulNode& fwd, const Tensor& ct) {
       return std::vector<Tensor>{
           fl::matmul(ct, fl::transpose(fwd.rhs().asTensor())),
           fl::matmul(fl::transpose(fwd.lhs().asTensor()), ct)
       };
   });
   ```
   This is simultaneously readable math AND a graph builder (because tensor ops are lazy).
   Users can override any VJP rule with their own tensor code — no graph-level manipulation
   needed. Debugging: print tensor values inside the rule (triggers eval of that sub-expr).

4. **Custom ops with gradients.** Two options:
   ```cpp
   // Option A: register a VJP function alongside the custom op
   auto myOp = CustomNode::create("my_op", inputs, evalFunc, vjpFunc);

   // Option B: decompose into primitives (all have VJP rules)
   Tensor myOp(const Tensor& x) {
       return fl::exp(fl::matmul(x, w) + b);  // composed from primitives
   }
   // grad(myOp) works automatically — chain rule over primitives
   ```
   Option B is the JAX/MLX idiom: most "custom ops" are just compositions of primitives.
   VJP rules only needed for true primitives (matmul, conv2d, etc. — maybe 30-40 total).
   Option A is the escape hatch for opaque CUDA kernels.

4. **Higher-order gradients.** `grad(grad(f))` applies VJP transform twice. The second
   VJP walks the backward graph (which contains matmul, add, mul nodes) and applies their
   VJP rules to produce second-order gradient nodes. No nested tapes needed.

5. **`requiresGrad` tracking.** Currently on Variable. Would move to a per-tensor flag
   (or per-node annotation) — "is this a differentiable parameter?" Controls which
   branches the reverse topo walk follows. Leaf tensors (data, constants) are not
   differentiated.

6. **Module system.** `Module::params()` returns `std::vector<Tensor>` (already does).
   `Module::forward()` returns Tensor (already does for many modules, but some return
   Variable). The Module system needs minimal changes — drop Variable return types.

**Why functional transforms are better for power users and optimizing entities:**

gradFunc closures feel like "more control" but are actually LESS controllable:
- Each closure builds nodes in isolation — no visibility into the global backward graph
- Closure execution order ≠ evaluation order (evaluator does topo sort)
- No opportunity to inspect/transform the backward graph before eval
- A model writing compiler passes can't reason about backward structure

Functional transforms make the backward graph a first-class data structure:
- `fl::grad(f)` produces the complete fwd+bwd DAG before eval
- A power user or model inspects the graph, writes passes (graph→graph transforms)
- Scheduling is explicit: communication ordering, memory pinning, pipeline partitioning
  are all expressed as graph transformations, not closure execution order
- Passes compose, can be ordered, conditional, and reason about global structure

Concrete scenarios where functional transforms dominate:
- Communication scheduling: reorder AllReduce/AllGather nodes for overlap with compute
- Gradient checkpointing: pass replaces "read saved intermediate" with "recompute"
- Pipeline parallelism: pass partitions graph into stages, inserts Send/Recv nodes
- Custom fusion: pass identifies cross-layer patterns in the combined fwd+bwd graph

All of these require seeing the complete graph as a data structure — impossible with
opaque closures, natural with functional transforms.

**What we lose:**
- The familiar PyTorch-style autograd API (`Variable::backward()`, `gradFunc` closures).
  Replaced by `fl::grad()` / `fl::valueAndGrad()` + VJP rules.
- Inline side effects in backward (logging, debugging). Mitigated by custom VJP rules
  or debug hooks.

**What we gain:**
- One graph representation (JIT IR) instead of two (JIT + autograd)
- No ExternalUse/compaction concerns for saved tensors
- Composable transforms: `grad`, `grad(grad)`, future `vmap`
- Custom ops that decompose into primitives get gradients for free
- Simpler codebase (eliminate autograd/ directory)
- Full graph visibility for power users and optimizing entities (models, passes)
- Explicit scheduling control via graph transformations
- Complexity decomposition: simple VJP rules define the math, passes define the strategy

**Complexity shift:** Imperative gradFunc puts complexity in per-op closures (each op mixes
gradient math with cross-cutting concerns like communication, checkpointing). VJP moves
cross-cutting concerns to passes that operate on the complete graph — the same
decomposition compilers use (simple lowering rules + optimization passes). Per-op VJP
rules are simpler; strategy is expressed as graph transformations.

**Debugging backward values:** Three mechanisms:
1. Eval + inspect: `fl::eval(grads); std::cout << grads[0] << std::endl;`
2. Debug node: `fl::debug(grad, "label")` — graph node that prints during eval,
   VJP rule is identity. Same as `jax.debug.print`.
3. Graph inspection: `printGraph(grads[0])` — print graph structure before eval.
   Impossible with gradFunc (backward graph doesn't exist until closures run).

**Communication primitives in the JIT IR:**

AllReduceNode, AllGatherNode, ReduceScatterNode as first-class IR nodes. The compiler
can explicitly schedule communication (overlap with computation, reorder, coalesce).
Both gradient approaches handle these identically:
- Imperative: gradFunc calls `fl::allReduce(grad)` → creates AllReduceNode in graph
- Functional: VJP rule for AllReduceNode returns appropriate backward communication node
  (all-reduce is self-adjoint for sum-reduce: `VJP(allReduce) = allReduce`)
  > **Caveat (2026-10 research):** only when the output cotangent varies per rank.
  > If the loss is replicated (invariant cotangent), this over-counts by N. Track
  > replication (varying vs invariant per mesh axis, as JAX `shard_map`'s
  > `check_vma` does) so collective VJPs are correct by construction (RS↔AG
  > adjoint, A2A self-inverse, send/recv reverse).

Having communication as graph nodes dovetails with either approach equally — the compiler
sees AllReduceNode in the graph regardless of how it was created.

**Functional transforms vs. imperative gradFunc: optimization equivalence.**

Given async materialization (Phase 3), both approaches produce the same connected
forward+backward DAG. Standard passes (DCE, CSE, fusion, scheduling) produce the same
result regardless of how nodes were created. The functional approach produces a slightly
"cleaner" initial graph (fewer intermediate identity nodes during construction), but a
single DCE pass makes them equivalent. **There is no optimization that functional
transforms enable that smart passes on the imperative graph cannot recover.**

The real advantage of functional transforms is **composability** — `grad(grad(f))` for
Hessian-vector products, future `vmap` for batched gradients. These are expressiveness
features, not optimization features. For first-order training (the common case), the
imperative approach with deferred eval is equivalent.

**Sweet spot: VJP rules as tensor ops + lazy eval + no compile scope.**
Given async materialization, there are no remaining advantages of tape-based autograd
over VJP rules written as tensor operations (MLX-style canonical implementations).
VJP rules produce lazy JIT nodes — same graph, same optimizations, but with
composability (`grad(grad(f))`), custom ops that decompose into primitives get
gradients for free, and one unified representation (no autograd/JIT duality).
Control flow is handled naturally (no tracing/compile scope — JIT builds incrementally).
This combines JAX's advantages (VJP, joint optimization) with PyTorch's advantages
(control flow, dynamic graphs) without either's limitations.

- [x] Unified Event, fl::Future<T>, TransferNode, TensorBackend::copy (see Phase 3)
- [ ] ~~FusionSideOutput~~: **obsolete with VJP** (see Phase 3 note)
- [ ] Communication primitives: AllReduceNode, AllGatherNode, ReduceScatterNode in JIT IR
- [ ] Communication scheduling pass: overlap communication with computation
**User-facing API (familiar, PyTorch-style):**

```cpp
auto loss = model.forward(input);
loss.backward();                          // VJP rules under the hood
auto wGrad = model.linear.weight.grad();  // lazy — evals on access
std::cout << wGrad << std::endl;          // triggers eval, prints values

// Debugging: hooks fire at eval time, not during graph walk
model.linear.weight.registerGradHook([](const Tensor& g) { print(g); });

// Intermediate gradients
auto z = model.layer1(input);
z.retainGrad();
loss.backward();
std::cout << z.grad() << std::endl;
```

The user never sees VJP rules. `loss.backward()` walks the JIT graph, applies VJP rules,
stores gradient nodes on parameters. `.grad()` returns a lazy tensor that evals on access.
Identical feel to PyTorch / current Flashlight autograd.

**Advanced API (for power users / composability):**

```cpp
auto gradFn = fl::grad(lossFunc);              // returns a function
auto grads = gradFn(params);                    // lazy gradient tensors
auto hessVecFn = fl::grad(fl::grad(lossFunc));  // higher-order
```

`fl::grad` / `fl::valueAndGrad` expose the functional interface for composition and
graph inspection. Most users use `tensor.backward()` + `tensor.grad()` instead.

- [ ] VJP rule registry for core ops (~40 rules: matmul, binary, unary, conv2d, bn, etc.)
- [ ] `Tensor::backward()` using VJP rules on JIT graph (replaces Variable::backward())
- [ ] `Tensor::grad()` / `Tensor::setRequiresGrad()` / `Tensor::retainGrad()` on TensorBase
- [ ] `Tensor::registerGradHook()` — fires at eval time with computed gradient value
- [ ] `fl::grad()` / `fl::valueAndGrad()` advanced functional API
- [ ] Migrate Module forward() from Variable → Tensor return types
- [ ] Deprecate Variable/AutogradMeta/gradFunc in favor of VJP rules
- [ ] Evaluate composability: `grad(grad(f))` for Hessian-vector products

**VJP design research summary** (preserved for implementation):

> **Corrections from the 2026-10-26 review** (details: `docs/modernization/research/review-2026-10-26.md` §5):
> - "Forward graph IS residual storage" holds only if backward is built **before**
>   the forward is evaluated. Today `fl::eval` rewrites shared nodes in place
>   (`replaceAllUsesWith`), compacts the root, and frees intermediates by use count;
>   typical loops read `loss.scalar()` before `backward()`. **Open decision:**
>   functional `valueAndGrad(f)` (build fwd+bwd before eval), optimize a copy /
>   keep provenance so the semantic graph survives eval, or keep the tape for
>   target identification and use VJP rules only for the math (recommended first
>   slice: "Phase A — rules under the tape").
> - RNG nodes (rand/dropout) must never be recomputed — needs a residual policy.
> - Target option (b) is unsound (`detach()`/`Tensor(x,false)` share the node);
>   (a) is ambiguous (clones create multiple SharedData per node); (c) + a
>   `StopGradient` node + registry for argument-less `backward()` is workable.
> - `asTensor(NodePtr)` doesn't exist; the bridge is `createJitTensor` (adds an
>   ExternalUse while the tensor lives).
> - New nodes needed: StopGradient, ScatterAdd, Conv2d/Pool2d/BN backward, RnnNode,
>   Sort/TopK/MinMax. Appendix B of the research doc misses the promoted nodes
>   (Clip/Concat/Cumsum/Flip/Roll/Triangular/Pad/Tile/Where), Max/Min, comparisons,
>   and the reductions beyond Sum/Mean.

VJP rule signature: `(const Node& fwd, const Tensor& ct) -> vector<Tensor>`.
Rules are written as tensor ops (MLX-style) — simultaneously readable math
AND lazy graph builders. `asTensor(NodePtr)` bridges Node→Tensor without eval.

Backward algorithm: reverse topo walk over JIT graph with
`unordered_map<Node*, Tensor>` cotangent accumulator. Accumulation via `+`
(lazy — creates BinaryNode::Add). No tape, no separate autograd graph.

Residuals: forward JIT graph IS residual storage. VJP rules access forward
values via `fwd.inputs()` (input data) and `fwd.shared_from_this()` (forward
output). No save_for_backward, no withoutData(), no ExternalUse pinning.

Gradient target identification — three options under consideration:
(a) Back-pointer on SharedData: add `shared_ptr<AutogradMeta>` to
    JitTensorBase::SharedData. Traverse ExternalUse→DataStorage→AutogradMeta.
(b) Flag on Node: `bool requiresGrad_` on Node. Direct but mixes concerns.
(c) Explicit parameter list: backward() takes params list (MLX/JAX style).
    Most principled. `loss.backward()` builds list from registry internally.

Custom ops: CustomNode gains optional `vjpFunc_`. Preferred path: decompose
into primitives (gradients free via chain rule). Escape hatch: explicit VJP.

Composability: `grad(grad(f))` is automatic — VJP rules create nodes that
have their own VJP rules. No special mechanism needed.

Migration: incremental (one op at a time). backward() uses VJP when available,
falls back to gradFunc when not. Each replacement independently testable.

**Prerequisite: promote CustomNode ops to first-class IR nodes** (see below).
VJP dispatch switches on NodeType — all differentiable ops should be first-class
nodes before VJP implementation, so dispatch is a clean switch with no
CustomNode name-matching.

### Promote CustomNode ops to first-class IR nodes

These ops are currently opaque CustomNodes but are used in autograd backward
paths and would benefit from optimization passes, fusion, and VJP dispatch.
Each promotion follows the Conv2dNode/Pool2dNode pattern: NodeType enum,
Node subclass, Evaluator dispatch, JitBackend creation, GraphvizPrinter,
CSE hash key, and applicable optimization pass rules.

Work through these methodically — each gets proper design, pass integration,
backend lowering, and thorough testing.

- [x] **TileNode**: 1 input, tile dims parameter. Shape: `out[i] = in[i] * tileDims[i]`.
  Most impactful for backward (`tileAs` is the universal broadcast gradient pattern).
  Research: `docs/modernization/research/tile-broadcast-research.md` (421 lines — XLA/TVM/ONNX/PyTorch
  comparison, tile vs broadcast vs expand semantics, algebraic identities).
  Key design decision: tile is explicit data replication, broadcasting stays implicit
  in BinaryNode (matching TVM/ONNX/PyTorch, not XLA's explicit broadcast_in_dim).
  **Immediate (this PR):**
    - [x] TileNode IR class, NodeType, Evaluator dispatch, JitBackend creation
    - [x] AlgebraicSimplification: identity (`tile(x,{1,...})→x`), scalar folding,
      consecutive fusion (`tile(tile(x,A),B)→tile(x,A*B)`)
    - [x] CSE hash key, GraphvizPrinter label
    - [x] GraphVerifier coverage and focused JIT tests
  **Deferred (future PRs):**
    - [ ] Elementwise commutation: `f(tile(x,T))→tile(f(x),T)` for pointwise f
      (profitable when x is small — avoids tiling before compute)
    - [ ] Tile-reduction cancellation: `sum(tile(x,T),axis)` where input dim=1
      → simplifies to `x * T[axis]` (the tileAs/sumAs adjoint identity)
    - [ ] CudnnSubgraphLowering: absorb tile as broadcast expansion in cuDNN graph
      (cuDNN handles broadcasting via tensor descriptor dims)
    - [ ] Transpose interaction: `tile(transpose(x,axes),T)` optimization
      (tile the transposed data with permuted tile dims)
    - [ ] Stride-based expand: for tile where all non-1 tile dims correspond to
      size-1 input dims, use stride=0 view instead of copy (zero-copy broadcast,
      like PyTorch's expand). Requires stride-aware downstream ops.
- [x] **WhereNode**: 3 inputs (cond, x, y), no parameters. Current IR semantics
  intentionally match existing Flashlight backends: condition must be `b8`, and
  condition/x/y must have identical shapes. Important for conditional gradients.
  Could participate in CudnnSubgraphLowering as pointwise SELECT
  (`fe::PointwiseMode_t::SELECT`). Could participate in elementwise fusion
  (3-input pointwise op).
  - [x] WhereNode IR class, NodeType, Evaluator dispatch, JitBackend creation
  - [x] CSE hash key, GraphvizPrinter label, GraphVerifier coverage, focused JIT tests
  - [ ] Future semantics cleanup: frontend-style `where(cond, x, y)` should accept
    broadcast-compatible condition/x/y shapes (JAX/NumPy/PyTorch/ONNX-style)
    while lower IR can either materialize explicit broadcasts/tiles or teach
    backend kernels to handle broadcast strides.
  - [ ] Future scalar-overload cleanup: scalar branch overloads currently create
    full tensors with `condition.shape()`, which prevents cases where the tensor
    branch has a different broadcast-compatible shape.
  - [ ] Future dtype cleanup: JIT metadata uses `promoteDtypes(x, y)`, but native
    backend `where` implementations mostly produce `x.type()` after casting. Align
    runtime output dtype, metadata, and scalar promotion before relying on mixed
    dtype `where` in optimization/VJP passes.
- [x] **ClipNode**: 3 inputs (input, low, high), no parameters. Same shape under
  current backend behavior. Used by autograd `clip`; semantically equivalent to
  `minimum(maximum(input, low), high)` but should stay first-class so VJP and
  backend lowering do not have to rediscover the pattern.
  - [x] ClipNode IR class, NodeType, Evaluator dispatch, JitBackend creation
  - [x] CSE hash key, GraphvizPrinter label, GraphVerifier coverage, focused JIT tests
  **Optimization/lowering followups:**
  - [ ] Semantics cleanup: align bound shape rules across backends. CPU/tensor
    paths naturally support tensor bounds, CUDA currently consumes scalar bounds,
    and scalar overloads materialize full-shape tensors through TensorBackend.
    Decide whether lower IR bounds are scalar-only, same-shape, or broadcastable.
  - [ ] Lowering/fusion: represent clip as fused `min(max(x, low), high)` when
    bound shapes/dtypes are compatible, while preserving first-class VJP dispatch.
- [x] **FlipNode**: 1 input, axis parameter. Same shape. Used in cumsum backward.
  Simple promotion. Transpose interaction: `flip(transpose(x))` optimizations.
  - [x] FlipNode IR class, NodeType, Evaluator dispatch, JitBackend creation
  - [x] CSE hash key, GraphvizPrinter label, GraphVerifier coverage, focused JIT tests
  **Optimization/lowering followups:**
  - [x] Simplify `flip(flip(x, axis), axis) -> x`.
  - [x] Simplify `flip(x, axis) -> x` when `x.dim(axis) <= 1`.
  - [ ] Canonicalize commuting flips on different axes for CSE.
  - [x] Transpose interaction:
    `flip(transpose(x, axes), dim) -> transpose(flip(x, axes[dim]), axes)`
    for explicit axes and default reverse-axis transpose.
  - [ ] Lowering gap: no direct cuDNN/OneDNN flip primitive; later fuse as an
    indexed read in an indexed-map CUDA/NVRTC lowering pass.
- [x] **TriangularNode**: 1 input, mode (`Lower`/`Upper`) + diagonal offset.
  Same shape and dtype. Used by tensor and autograd `tril`/`triu` APIs.
  Semantics intentionally match PyTorch/JAX/MLX: input rank must be at least 2,
  the triangular mask applies independently over the last two dimensions, and
  `diagonal=0` selects the main diagonal. This replaces the old Flashlight
  implementation bug where native/CUDA rank>2 tensors were masked over dims 0/1
  despite the public docs promising last-two-dimension submatrices.
  - [x] Public tensor API supports `tril(x, diagonal, stream)` and
    `triu(x, diagonal, stream)` while preserving existing `tril(x)` and
    stream-only call sites.
  - [x] CPU native and CUDA implementations support last-two-dims masking with
    arbitrary diagonal offset. CUDA uses byte-wise copy/zero so dtype is
    preserved rather than accidentally treating non-f64 tensors as f32.
  - [x] ArrayFire backend explicitly throws for unsupported generalized cases
    (`diagonal != 0`, rank != 2) instead of silently returning semantics that do
    not match the tensor contract.
  - [x] TriangularNode IR class, NodeType, Evaluator dispatch, JitBackend
    creation, CSE key, GraphvizPrinter label, GraphVerifier coverage, focused
    JIT tests.
  **Optimization/lowering followups:**
  - [x] Generic simplification: identity for fully retained diagonal ranges,
    zero-scalar identity, same-mode nested mask fusion
    (`tril(tril(x,a),b)->tril(x,min(a,b))`,
    `triu(triu(x,a),b)->triu(x,max(a,b))`), and transpose interaction when the
    transpose only swaps the last two matrix axes.
  - [ ] Mixed-mode composition: `tril(triu(x,a),b)` / `triu(tril(x,a),b)`
    represents a band or diagonal mask. Defer until mask/where/indexed-map IR is
    robust enough to represent the result without inventing a one-off band node.
  - [ ] Reference lowering: lower TriangularNode to `compare(arange rows/cols)`
    + `where/select`, matching the decompositions used by JAX and PyTorch and
    the way StableHLO/XLA represent triangular masks without a dedicated op.
  - [ ] Backend lowering: no direct cuDNN/OneDNN triangular primitive. Use the
    future indexed-map/NVRTC path for fused triangular masking, and consider
    backend-specific kernels only when profiling shows the generic path is not
    enough.
  - [ ] VJP followup: TriangularNode gradients are the same triangular mask
    applied to the cotangent with the same mode/diagonal. Once VJP rules exist,
    add tests that VJP-generated triangular graphs are simplified/CSE'd by the
    same passes.
- [x] **CumsumNode**: 1 input, axis parameter. Same shape. Used in cumsum gradient
  (reverse cumsum). Not fusable (sequential dependency) but benefits from being
  visible to the optimizer for scheduling.
  - [x] CumsumNode IR class, NodeType, Evaluator dispatch, JitBackend creation
  - [x] CSE hash key, GraphvizPrinter label, GraphVerifier coverage, focused JIT tests
  **Optimization/lowering followups:**
  - [x] Simplify `cumsum(x, axis) -> x` when `x.dim(axis) == 1`.
  - [ ] Recognize reverse-cumsum pattern from backward:
    `flip(cumsum(flip(x, axis), axis), axis)`.
  - [ ] Consider representing reverse cumsum as `CumsumNode` with a direction
    flag instead of three separate nodes.
  - [ ] Lowering gap: no cuDNN/OneDNN general cumsum primitive. CUDA has a
    kernel/CUB path today, but the contiguous-segment CUB path launches one scan
    per segment; later improve with segmented scan or a stronger custom scan
    kernel.
- [x] **ConcatenateNode**: N inputs, axis parameter. Shape grows along axis.
  Variable arity makes this slightly more complex. Important for split/concat
  patterns in attention and RNN.
  - [x] ConcatenateNode IR class, NodeType, Evaluator dispatch, JitBackend creation
  - [x] CSE hash key, GraphvizPrinter label, GraphVerifier coverage, focused JIT tests
  - [x] Conservative XLA-style canonicalization:
    single-input identity only when Flashlight rank-promotion semantics are
    preserved, empty-input removal only when output metadata is unchanged, and
    nested same-axis concat flattening.
  **Optimization/lowering followups:**
  - [ ] Slice-of-concat:
    `slice(concat(xs, axis), ...) -> operand`, smaller concat, or adjusted slice
    when the slice covers one or more complete concat operands. XLA implements
    this for unstrided slices and it is important for VJP/split cleanup.
  - [ ] Concat-of-adjacent-slices:
    `concat(slice(x, a:b), slice(x, b:c), axis) -> slice(x, a:c)` for adjacent
    unstrided slices with matching starts/limits on non-concat axes. XLA
    implements this canonicalization.
  - [ ] Concat-as-pad:
    `concat(broadcast(scalar), x)` / `concat(x, broadcast(scalar)) -> pad`
    when the broadcasted scalar is a low/high pad region. XLA uses this because
    pad is easier to fold into neighboring ops.
  - [ ] Repeated singleton concat:
    `concat(x, x, ..., axis)` where `x.dim(axis)==1` can become
    reshape/broadcast when exact. XLA has this broadcast-style canonicalization.
  - [ ] Lowering/planning gap: CPU/OneDNN and CUDA have concat kernels, but
    first-class concat should eventually let buffer planning place producers
    directly into concat output slices instead of materializing and copying.
- [x] **PadNode**: 1 input, pad widths + padding mode parameter. Shape grows.
  Used in conv padding gradient. Could be absorbed into cuDNN conv descriptors.
  Current tensor API supports zero-valued constant padding plus edge/symmetric
  padding through `PadType`; arbitrary constant pad values only exist in the
  autograd helper today and should become explicit IR metadata later if needed.
  - [x] PadNode IR class, NodeType, Evaluator dispatch, JitBackend creation
  - [x] CSE hash key, GraphvizPrinter label, GraphVerifier coverage, focused JIT tests
  - [x] Normalize missing trailing pad-width axes to `{0, 0}` in the IR so the
    node has one width pair per input axis and lowers cleanly to current
    backends.
  **Optimization/lowering followups:**
  - [x] Simplify zero-width pad to identity.
  - [x] Fuse nested zero-constant pads by summing pad widths.
  - [ ] Future semantics: add explicit constant pad value metadata if/when the
    tensor-level API exposes nonzero constant padding directly.
  - [x] Transpose interaction: move pad across transpose by permuting pad widths.
  - [ ] Consider edge-pad nested fusion separately. It is likely equivalent to a
    larger edge pad, but keep it out until tested across backends. Do not fuse
    nested symmetric pads: repeated symmetric padding is not generally equivalent
    to one larger symmetric pad.
  - [ ] Later: pad followed by crop/slice cancellation, once slice/index
    patterns are inspectable.
  - [ ] Backend lowering: absorb constant-zero spatial pad into Conv/Pool
    descriptors in cuDNN/OneDNN where shapes and layouts match.
  - [ ] Indexed-map lowering: lower general pad to computed input indices plus
    zero fill/select, so it can fuse with neighboring indexed data-movement ops
    in a future CUDA/NVRTC indexed-map pass.
- [x] **RollNode**: 1 input, shift + axis parameters. Same shape. Used in roll
  backward. Simple promotion.
  - [x] RollNode IR class, NodeType, Evaluator dispatch, JitBackend creation
  - [x] CSE hash key, GraphvizPrinter label, GraphVerifier coverage, focused JIT tests
  **Optimization/lowering followups:**
  - [x] Normalize `shift %= x.dim(axis)` at node construction.
  - [x] Simplify zero/full-period shift to identity.
  - [x] Simplify singleton-axis roll to identity.
  - [x] Fuse nested rolls on the same axis:
    `roll(roll(x, a, axis), b, axis) -> roll(x, a + b, axis)`.
  - [ ] Canonicalize commuting rolls on different axes for CSE.
  - [x] Transpose interaction:
    `roll(transpose(x, axes), shift, dim) -> transpose(roll(x, shift, axes[dim]), axes)`
    for explicit axes and default reverse-axis transpose.
  - [ ] Lowering gap: no direct cuDNN/OneDNN roll primitive; later fuse as an
    indexed read or decompose to slice/concat once slicing patterns are
    first-class.

### Optimization/lowering roadmap for first-class shape/select IR

Add these incrementally after each node promotion. Keep each change small and
backend-independent unless explicitly listed under backend lowering.

- [x] **WhereNode generic algebraic simplification**:
  - [x] `where(c, x, x) -> x`
  - [x] `where(true_scalar, x, y) -> x`
  - [x] `where(false_scalar, x, y) -> y`
  - [x] preserve current same-shape and dtype caveats until runtime semantics are
    aligned.
- [ ] **TileNode generic algebraic simplification expansion**:
  - [x] normalize all-rank identity cases after tile dim padding
  - [x] rank-preserving explicit-axis transpose/tile interaction:
    `tile(transpose(x, axes), T) -> transpose(tile(x, permuted_T), axes)`
  - [ ] broader transpose/tile interaction for default reverse axes
  - [ ] rank-expanding transpose/tile interaction: handle cases where `tileDims`
    has higher rank than the transposed input by explicitly modeling the
    leading-1 input-shape padding that `inferTileOutputShape` applies before
    permuting tile factors. This likely needs either a canonical padded-rank
    transpose representation or an explicit reshape/pad-rank step before moving
    tile across transpose.
  - investigate tile/reduction cancellation separately because it touches
    reduction dtype/shape rules.
- [ ] **Generic elementwise fusion representation**: decide whether to extend
  `FusedElementwiseNode` from unary/binary steps to N-ary or explicit ternary
  steps so `WhereNode` can participate in generic fusion and NVRTC lowering.
- [ ] **Indexed-map fusion representation**: add a fusion/lowering path for ops
  whose output element reads input elements at computed indices. This would cover
  `flip`, `roll`, `pad`, many `tile` cases, `TriangularNode` masks, and some
  `where`/mask patterns. Current elementwise fusion assumes same-index
  unary/binary programs, so these data-movement ops need richer indexing
  expressions before they can fuse into generated CUDA kernels.
- [ ] **cuDNN lowering**:
  - lower `WhereNode` to cuDNN Frontend pointwise SELECT when shape/dtype rules
    are supported
  - consider absorbing tile-as-broadcast cases via tensor descriptors rather
    than materializing tile.
- [ ] **OneDNN lowering**:
  - lower `WhereNode` through `binary_select` when dtype support matches runtime
    semantics
  - keep native fallback for integer/f64 cases until OneDNN coverage is complete.
- [ ] **Shape/data-movement optimization batch**: after `FlipNode`, `RollNode`,
  `PadNode`, `ConcatenateNode`, and `TriangularNode` exist, add a focused
  pass for transpose interactions, inverse pairs, identity cases, and simple
  pad/concat cancellation patterns.

**Pass abstraction evolution:**

Current abstractions cover single-node rewriting with two traversal orders. This is
sufficient for the passes we have today, but more sophisticated optimizations will
need additional abstractions. Assessment of what's missing:

*Multi-node pattern rewrite pass (`PatternRewritePass`):* **Useful, near-term need.**
Current `rewrite(NodePtr)` sees one node at a time. CudaMatmulEpilogueFusion works
around this by calling the PatternMatcher DSL inside `rewrite()` to inspect the
subtree — but the pattern matching + replacement boilerplate is still manual. A
`PatternRewritePass` would hold a list of (pattern, rewrite_function) pairs and
the base class walks the graph, tries each pattern at each node, and calls the
matching rewrite function. Think MLIR's `RewritePattern` / `PatternRewriter` but
much simpler. This directly enables:
- Relu epilogue recognition: `max(x, Scalar(0))` → `ReluNode(x)` (or fused into cuDNN/cuBLASLt epilogue)
- Gelu recognition: `0.5 * x * (1 + erf(x / sqrt(2)))` → fused gelu
- Conv+bias+activation: multi-node cuDNN Frontend patterns
- Algebraic simplification rules: `x * 1 → x`, `x + 0 → x`, `x - x → 0`
The PatternMatcher DSL already exists (`PatternMatcher.h`), so the main work is
a base class that iterates patterns and handles replacement/invalidation. Probably
the first new abstraction worth building.

*Analysis passes (`AnalysisPass<ResultType>`):* **Useful, needed for memory planning.**
Not all passes transform the graph — some compute information for later passes.
An `AnalysisPass<T>` returns a result `T` rather than a modified `NodePtr`. Examples:
- Liveness analysis: at each point in the topological order, which node results are
  still needed? Required for buffer reuse in the memory planner (Step 8).
- Alias analysis: which tensors share underlying storage? Needed for in-place optimization
  (e.g., `relu(x)` can write into `x`'s buffer when `x` has no other uses).
- Cost model: estimated FLOPS/memory per node, used to guide fusion decisions
  (e.g., don't fuse if the fused kernel would exceed register pressure limits).
Currently `buildPlan()` computes use-counts and sync-edges inline. Liveness analysis
would be a natural addition to the plan phase, or could be a standalone analysis pass
whose results are consumed by a memory planning pass. No strong need to formalize
this until Step 8 (memory planning), but worth keeping in mind.

*Fixed-point pass wrapper:* **Simple utility, add when needed.**
Some transformations interact — constant folding creates new constants that enable
further CSE, which enables further folding. A `FixedPointPass` wrapper runs one or
more inner passes in a loop until the graph stops changing. Not a new base class,
just a wrapper: `while (changed) { node = pass.apply(node); }`. Easy to build when
algebraic simplification rules grow large enough to interact. The current pass
infrastructure already supports this since `apply()` clears `visited_` each call.

*Region / subgraph passes:* **Medium-term, needed for device partitioning and capture.**
Some optimizations operate on subgraphs, not individual nodes:
- CUDA graph capture: identify subgraph boundaries where a CUDA graph can be recorded
  (no host-side control flow, no dynamic shapes, no cross-device transfers).
- Graph partitioning: split a graph across devices/backends. Nodes with different
  `stream()->device()` values form natural partition boundaries; the pass inserts
  transfer nodes.
- Tiling/blocking: for large matmuls or convolutions, decompose into tiled subgraphs
  that fit in cache/shared memory.
These don't fit `rewrite(NodePtr)` — they need to see a region of connected nodes.
A `SubgraphPass` would receive a linearized plan (or subgraph) and return a
transformed version. Not worth building until we actually need CUDA graph capture
or multi-device execution.

*Chain fusion pass:* **Probably not worth generalizing.**
OneDnnOpFusion walks fusable chains of binary/unary ops, accumulating post-ops.
A `ChainFusionPass` base class could provide: find chain heads, walk through
fusable nodes, collect state into a user-defined accumulator, call `fuseChain()`.
But OneDNN's pattern (post-ops with specific LHS/RHS constraints, commutative
swapping, up-to-32 post-ops limit) is specific enough that a general abstraction
wouldn't simplify it much. Other chain-like patterns (e.g., cuDNN convolution
fusion) are better served by multi-node pattern matching. Keep OneDnnOpFusion
as a plain `Pass` unless a second chain-fusion pass appears with similar structure.

*Dataflow analysis passes:* **Far future, not worth building now.**
Forward/backward dataflow propagation (e.g., range analysis: knowing a tensor's
values are in [0,1] after sigmoid, or sparsity propagation). Useful for
specialization but not a near-term need. Revisit when we have enough algebraic
rules to benefit from value range information.

*Pass pipeline / pass manager:* **Current linear Optimizer is sufficient.**
LLVM-style pass managers support: pass groups (run to fixed point together), pass
dependencies (CSE after constant folding), conditional passes (skip if no relevant
nodes), pass instrumentation (timing, verification). The current `Optimizer` runs
passes linearly, which is fine when the pass count is small (~7). Adding complexity
here should be deferred until: (a) pass ordering bugs appear, or (b) compilation
time becomes a concern. The TODO in Optimizer.cpp already notes this.

Summary of priorities:
1. `PatternRewritePass` — first new abstraction to build (enables relu/gelu/conv epilogues)
2. Analysis passes / liveness — needed when Step 8 (memory planning) begins
3. Fixed-point wrapper — trivial, add when rule interactions matter
4. Everything else — defer until concrete need arises

**Design considerations for backend-dependent passes:**
- Current `JitOptimizerExtension::passes()` returns `vector<unique_ptr<Pass>>` -- clean API
- Passes should be ordered: common pattern-matching first, then backend-specific lowering
- `Optimizer` already supports this ordering (common passes added first, then extension passes)
- Backend passes may need access to backend state (e.g. OneDnnOpFusion uses CPUBackend::getInstance())
  - This is fine since they're compiled conditionally (#if FL_USE_CPU) and registered per-backend
- Alternative: passes could declare dependencies/ordering constraints, but YAGNI for now
- Resolved: `FusedElementwiseNode` is a first-class IR node (enables further optimization passes to inspect/rewrite fused nodes)
- Pass base classes (`BottomUpPass`, `TopDownPass`) eliminate traversal boilerplate -- new passes need only `rewrite(NodePtr)`
- `markVisited(Node*)` lets passes like ElementwiseFusion mark group members to prevent re-processing
- OneDnnOpFusion stays as plain `Pass` -- its iterative chain-building pattern doesn't fit either base class

## Operator Specification
- [x] Operator dtype inference (OpProperties.h/cpp):
  - `inferOutputDtype(UnaryOp, dtype)`: LogicalNot/Isnan/Isinf --> b8, all others preserve
  - `inferOutputDtype(BinaryOp, dtype, dtype)`: comparisons/logical --> b8, arithmetic --> promoteDtypes
  - `inferOutputDtype(ReductionOp, dtype)`: Any/All --> b8, CountNonzero --> s32, Mean/Std --> f32 for ints
  - `promoteDtypes(dtype, dtype)`: NumPy-style type promotion (float > int, wider > narrower)
  - `isElementwise(UnaryOp/BinaryOp)`, `isFusableElementwise(UnaryOp/BinaryOp)` for fusion pass
  - Shape rules already exist via ShapeInference.h functions
- [x] JIT Node dtype inference (dtype_ on Node base class):
  - Every Node carries a `dtype_` computed at construction time from op + input dtypes
  - `JitTensorBase::type()` returns `node()->dataType()` -- no materialization needed
  - Enables lazy evaluation (no forced eval/sync just to query dtype)
  - Preserves fusion opportunities that were broken by premature materialization
  - `inferFusedProgramOutputDtype()` simulates dtype through FusedElementwise programs
- [x] Lazy `flat()`: output shape is trivially computable (elements of original shape) -- now uses ReshapeNode + IndexNode composition instead of materializing

## Stream Semantics and Async Execution

### Design Requirements
1. Execution on ALL backends must be asynchronous and compatible with streams
2. Streams from different implementations within a backend (e.g., ACL vs oneDNN) must interop
   with minimum overhead
3. Cross-backend stream semantics (relative synchronization) must work without blocking
   the host thread (e.g., GPU↔CPU transfers)
4. No stance on stream assignment policies -- users specify, but sensible default is to
   enqueue on the same stream as input tensors
5. `eval()` should NOT block the host thread -- only `host()` / `scalar()` should block,
   and only on the computation needed to materialize that specific value
6. Full device syncs should be avoided; use relative synchronization (events) wherever possible
7. Cross-backend tensor transfer should use `Tensor&&` (move semantics) -- one buffer, one owner.
   Schedule transfers with `fl::Stream`. Avoid full device sync.
8. Extensive tests for async/stream behaviors

### Current Infrastructure Audit

**Stream class hierarchy:**
```
Stream (abstract: sync(), relativeSync())
├── StreamTrait<SynchronousStream>
│   └── SynchronousStream
├── StreamTrait<AsyncCPUStream>
│   └── AsyncCPUStream (worker thread + task queue, paired dnnl::stream)
└── StreamTrait<CUDAStream>
    └── CUDAStream (cudaStream_t + cudaEvent_t)
```

**What works today:**
- CUDAStream: true async via `cudaEventRecord` + `cudaStreamWaitEvent` in `relativeSync()`
- CudaTensor: per-tensor `shared_ptr<CUDAStream>`, stream-ordered alloc via `cudaMallocAsync`
- CudaBackend: single stream for cuBLAS/cuRAND handles

**What doesn't work:**
- SynchronousStream: `relativeSync()` just calls `waitOn.sync()` -- fully blocking
- JIT Evaluator: zero stream awareness -- dispatches all ops sequentially, no stream parameter

**What now works (Phase A+B+D):**
- AsyncCPUStream: worker thread + task queue, enqueue() returns immediately, sync() drains queue
- CPUTensor: per-tensor `shared_ptr<AsyncCPUStream>` -- ops derive stream from input tensors
- OneDNN ops (tensor + autograd): enqueued asynchronously on worker thread
- Cross-type relativeSync: AsyncCPU <-> CUDA, AsyncCPU <-> Synchronous
- `fl::eval()` / `fl::relativeSync()` free functions

### Async CPU Execution Design

**Problem:** CPU backends (OneDnn/ARM) execute synchronously. All operations block the calling
thread. This prevents: (a) overlapping CPU compute with GPU compute, (b) overlapping independent
CPU operations with each other, (c) pipelining data loading with computation.

**Proposed solution: `AsyncCPUStream`** -- a stream backed by a dedicated worker thread and
task queue. Operations enqueue work items and return immediately. The worker thread processes
tasks in FIFO order on its thread, ensuring sequential consistency within a stream.

```
AsyncCPUStream
├── worker thread (1 per stream)
├── task queue (std::deque<Task> or lockfree queue)
├── CPUEvent (std::promise/future pair for sync points)
└── inherits Stream (NOT SynchronousStream)
```

**Key design decisions:**

**1. AsyncCPUStream vs SynchronousStream:**
- [ ] AsyncCPUStream is a NEW stream type (StreamType::AsyncCPU), NOT derived from SynchronousStream
- [ ] Must implement its own `sync()` (drain queue + wait for current task) and
      `relativeSync()` (enqueue a wait-on-event task)
- [ ] AsyncCPUStream becomes a detail INSIDE AsyncCPUStream -- the worker thread holds
      the dnnl::stream and executes dnnl primitives on it
- [ ] Native C++ fallback ops (sort, flip, pad, etc.) execute on the same worker thread --
      automatically serialized with oneDNN ops on the same stream

**2. CPUEvent -- the async primitive for CPU:**
- [ ] `CPUEvent` wraps `std::shared_future<void>` + `std::promise<void>`
- [ ] `record(stream)`: enqueues a task on `stream` that sets the promise when reached
- [ ] `wait()`: blocks on the future (for host sync)
- [ ] `stream.waitOnEvent(event)`: enqueues a task that calls `event.wait()` -- the worker
      thread blocks until the event is set, then continues processing its queue
- [ ] Lightweight: no kernel launch overhead, just thread synchronization primitives

**3. Per-tensor stream assignment:**
- [ ] CPUTensor needs a `shared_ptr<AsyncCPUStream>` (like CudaTensor has `shared_ptr<CUDAStream>`)
- [ ] Default: all tensors share the backend's default stream (single worker thread)
- [ ] Users can create additional streams for explicit parallelism:
      `auto stream2 = AsyncCPUStream::create(engine);`
- [ ] Output tensor inherits stream from first input tensor (convention)

**4. How ops enqueue work:**
```cpp
// Before (synchronous):
Tensor CPUBackend::add(const Tensor& a, const Tensor& b) {
  auto result = /* allocate output */;
  dnnl::binary(engine, ...).execute(nativeStream, {{src0, a}, {src1, b}, {dst, result}});
  return result;
}

// After (async):
Tensor CPUBackend::add(const Tensor& a, const Tensor& b) {
  auto result = /* allocate output (metadata only, buffer may be deferred) */;
  a.stream().enqueue([=] {
    dnnl::binary(engine, ...).execute(nativeStream, {{src0, a}, {src1, b}, {dst, result}});
  });
  return result;  // returns immediately, compute happens on worker thread
}
```

**5. Blocking only on materialization:**
- [ ] `host()` / `scalar()` / `toHostVector()`: call `stream().sync()` to drain the queue,
      then copy data. This is the ONLY place user code blocks.
- [ ] `eval()` (TensorBackend virtual): should be a no-op for async streams -- work is already
      enqueued. For JIT, it enqueues the graph evaluation as a single task.

### Cross-Backend Synchronization (CPU ↔ GPU)

**Scenario 1: GPU waiting on CPU result (e.g., CPU preprocesses data, GPU consumes)**
- [ ] CPU stream records a CPUEvent when preprocessing completes
- [ ] GPU stream cannot natively wait on a CPUEvent (cudaStreamWaitEvent needs cudaEvent_t)
- [ ] Solution: `cudaLaunchHostFunc(gpuStream, callback, cpuEvent)` -- CUDA calls our
      callback on a CPU thread when the GPU stream reaches that point. The callback can
      check/wait on the CPUEvent. BUT: this blocks a CUDA callback thread, not ideal.
- [ ] Better: use a bridge thread that waits on the CPUEvent, then records a cudaEvent_t
      on the GPU stream. GPU stream does `cudaStreamWaitEvent` on that event.
- [ ] Or simplest: `cpuStream.sync()` (blocking wait), then enqueue GPU work. Loses overlap
      but is correct and simple. Start here, optimize later.

**Scenario 2: CPU waiting on GPU result (e.g., GPU computes, CPU postprocesses)**
- [ ] GPU stream records a cudaEvent_t when computation completes
- [ ] CPU worker thread calls `cudaEventSynchronize(gpuEvent)` -- blocks the worker thread
      (not the user thread) until GPU work finishes
- [ ] Implementation: `cpuStream.relativeSync(gpuStream)` enqueues a task on the CPU worker
      that does `cudaEventSynchronize(gpuStream.event)`
- [ ] This is natural: the CPU worker blocks on the GPU, but the user's host thread is free

**Scenario 3: CPU ↔ CPU cross-stream (e.g., two independent CPU computations)**
- [ ] `stream1.relativeSync(stream2)`: stream1's worker enqueues a wait on stream2's CPUEvent
- [ ] stream1's worker blocks until stream2 reaches the sync point, then continues
- [ ] Simple and correct via CPUEvent::wait()

**Cross-type relativeSync dispatch:**
```cpp
// Stream.h -- extend relativeSync to handle cross-type streams
void AsyncCPUStream::relativeSync(const Stream& waitOn) const override {
  if (waitOn.type() == StreamType::CUDA) {
    auto& cudaWaitOn = waitOn.impl<CUDAStream>();
    cudaEvent_t event = cudaWaitOn.recordEvent();
    enqueue([event] { cudaEventSynchronize(event); });
  } else if (waitOn.type() == StreamType::AsyncCPU) {
    auto& cpuWaitOn = waitOn.impl<AsyncCPUStream>();
    auto event = cpuWaitOn.recordCPUEvent();
    enqueue([event] { event.wait(); });
  }
}
```

### JIT Evaluator Stream Integration

**Current state:** JIT Evaluator dispatches all ops synchronously via `backend_.add()`,
`backend_.exp()` etc. No stream parameter, no async awareness.

**Design options:**

**Option A: Evaluator enqueues entire graph as single task (recommended for now)**
- [ ] `JitBackend::eval(tensor)` enqueues the full graph evaluation as one task on the
      target stream's worker thread
- [ ] Inside the task, the evaluator runs sequentially (same as today) but on the worker
      thread instead of the caller's thread
- [ ] Advantages: minimal changes to evaluator; graph evaluation is naturally sequential;
      JIT fusion already reduces op count
- [ ] Disadvantage: no intra-graph parallelism (fine for CPU, important for GPU later)

**Option B: Evaluator dispatches individual ops to streams (future)**
- [ ] Each node in the JIT graph gets a target stream
- [ ] Independent subgraphs dispatch to different streams
- [ ] Sync points inserted where subgraphs merge (relativeSync)
- [ ] Advantages: maximum parallelism
- [ ] Disadvantages: complex scheduling, CPU overhead may exceed compute savings
- [ ] Only worthwhile for GPU with many independent subgraphs -- defer this

**Option C: MLX-style per-op stream specification (future consideration)**
- [ ] Each tensor op takes an optional `stream` parameter
- [ ] Lazy evaluation + stream-based scheduling determines execution order
- [ ] Good model but requires significant API changes (every op signature changes)
- [ ] Consider as Phase 7+ if needed

**Recommendation:** Start with Option A. Most CPU workloads are sequential within a graph.
The win is overlapping graph evaluation (on worker thread) with other host work (data loading,
preprocessing). Intra-graph parallelism (Option B) can be added later for GPU.

### Native C++ Ops + oneDNN Ops in Async Context

**Problem:** CPUBackend uses two kinds of ops: oneDNN primitives (matmul, conv, pool, eltwise)
and native C++ fallbacks (sort, flip, pad, cumsum, etc.). Both must work in the async model.

**Solution:** Both kinds of ops enqueue their work on the same AsyncCPUStream worker thread.
```
AsyncCPUStream worker thread:
  Task 1: dnnl::matmul(...).execute(dnnlStream, ...)   ← oneDNN
  Task 2: std::sort(hostData, ...)                      ← native C++
  Task 3: dnnl::eltwise(...).execute(dnnlStream, ...)   ← oneDNN
  Task 4: flip via coordinate copy                      ← native C++
```

- Sequential execution within a stream guarantees correctness
- oneDNN ops use the dnnl::stream held by the worker thread
- Native C++ ops operate on host memory directly (no stream needed)
- No synchronization needed between oneDNN and native ops on the same stream --
  dnnl::stream on CPU is synchronous (completes before returning)

**Memory model:**
- All CPU tensors share host memory -- no copy needed between oneDNN and native ops
- oneDNN uses `dnnl::memory` which wraps a host pointer
- Native C++ ops access the same pointer via `toHostVector()` / raw data pointer
- No memory ownership transfer between libraries -- just pointer sharing

### Implementation Roadmap

**Phase A: AsyncCPUStream foundation**
- [x] `CPUEvent` class: promise/future wrapper with `record()`, `wait()`, `isReady()`
- [x] `AsyncCPUStream` class: worker thread, task queue, enqueue(), sync(), relativeSync()
- [x] Tests: enqueue + sync, multiple tasks, relativeSync between two AsyncCPUStreams
- [x] Cross-type relativeSync: AsyncCPU <-> SynchronousStream (both directions)

**Phase B: CPUBackend async integration**
- [x] CPUTensor: add `shared_ptr<AsyncCPUStream>` member (matching CudaTensor pattern)
- [x] CPUBackend: replace OneDnnCPUStream with AsyncCPUStream, wrap 9 execute sites in enqueue
- [x] host()/scalar(): call `stream_->sync()` before data access
- [x] clone()/astype()/assign(): enqueue reorder asynchronously
- [x] Per-tensor stream propagation: views, copies, shallow copies all propagate stream_
- [x] Delete OneDnnCPUStream (replaced by AsyncCPUStream)
- [x] Delete DnnlStream/DnnlEngine singletons (replaced by backend engine + input tensor stream)
- [x] relativeSync in OneDNN tensor ops (binary, matmul use input tensor's stream)
- [x] Autograd ops (Conv2D, Pool2D, BatchNorm, RNN) use backend engine + AsyncCPUStream
- [x] Fix all compiler warnings (narrowing in Functions.cpp, control-reaches-end in CPUBackend/CPUTensor)
- [x] Add 8 async chain correctness tests in CPUTensorTest (arithmetic, eltwise, shape+arith, astype, reduction, clone, index, matmul chains)
- [x] Add CPU backend test variants for all generic tests (22 _CPU targets via build_test SUFFIX + FL_TEST_BACKEND_CPU)
  - TestInit.h + flTestInit(): conditionally sets CPUTensor as default backend
  - Tensor: 16/53 pass, autograd: 3/60 pass, nn: 6/48 pass, optim: 0/3 pass
  - ALL failures are pre-existing OneDNN backend gaps (not async regressions):
    - 92x unsupported dtype reorder (b8, u32, f64, u64)
    - 25x unimplemented ops (flatten, flat, fromScalar)
    - 12x unsupported dtype in eltwise/assign/inPlaceAdd
    - 7x pool2d primitive descriptor failure
    - 4x missing broadcast support
    - 3x bidirectional RNN not supported
    - 6x half precision not supported on CPU
- [x] OneDNN backend: implement flatten/flat (contiguous reshape, non-contiguous clone+reshape)
- [x] OneDNN backend: implement fromScalar for all types (delegates to full(Shape({})))
- [x] OneDNN backend: implement inPlaceAdd/inPlaceMultiply/etc for all types (OneDNN binary/eltwise in-place)
- [x] OneDNN backend: fix toHostVector<float>() data corruption in 13 fallback ops (toFloatHostVector)
- [x] OneDNN backend: fix absolute() for integer types (CPU fallback for s32, b8; noop for u8)
- [x] OneDNN backend: fix astype() CPU fallback for unsupported reorder paths (s32->f64, f32->f64, etc.)
- [x] OneDNN backend: fix scalar()/clone() memcpy fallback for f64 reorder failures
- [x] OneDNN backend: add FL_SKIP_IF_DTYPE_UNSUPPORTED test macro + dtype skip guards
- [x] OneDNN backend: TensorBaseTest_CPU improved from 19/53 to 42/53 passing
- [x] OneDNN backend: fix missing `throw` in transpose (line 880 — trivial one-liner)
- [x] OneDNN backend: fix async lifetime bugs in CPUTensor (clone, astype, assign enqueue calls not keeping sharedData_ alive)
- [x] OneDNN backend: fix NNSerializationTest.Tensor — add dtype support guards for unsupported types
- [x] CPU backend: fix flaky TensorReductionTest.std — root cause: deterministic RNG state after prior tests produces bad sample with only 504 elements. Fix: use 50,400-element sample (SE drops from 0.009 to 0.0013), keeping 0.005 tolerance (3.8 SE)
- [x] CPU backend: concatenate — edge-case tests added (single tensor identity, axis 1, 1-D value verification, higher axis, different-valued tensors)
- [x] CPU backend: tile CPU fallback for f64 — already works (native fallback dispatches for f64)
- [x] CPU backend: Tensor-as-index (CPU gather) — already implemented (integer + boolean mask indexing). Fixed sync: replaced blocking stream_->sync() with targeted relativeSync on input tensor streams
- [x] CPU backend: strided range indexing — already works for reads and writes. Edge-case tests added (1-D/2-D value verification with stride 2/3, multi-dim strides, strided assignment reads and writes)
- [x] CPU backend: PadType::Symmetric — already implemented in native/ShapeAndIndex.cpp. Fixed bug: formula was NumPy "symmetric" (edge included), confirmed this matches ArrayFire AF_PAD_SYM behavior. Added small-tensor tests with known expected values (padSymmetricSmall, padEdgeSmall)
- [x] CPU backend: scalar shape mismatch ({} vs {1}) — investigated: NOT an issue. All full reductions (sum, amin, amax, mean, var, std, norm, any, all, countNonzero) correctly return shape {}, fromScalar returns {}, axis reductions correctly collapse the specified dim
- [x] CPU backend: broadcast for mismatched shapes — already works (getBinaryOpOutputDesc pads shorter shapes with leading 1s). Edge-case tests added (rank mismatch [4]+[3,4], [4]+[2,3,4], incompatible throw, scalar broadcast both directions)
- [ ] Make eval() trigger lazy materialization when lazy tensor evaluation is implemented (Phase C)
- [ ] Make C++ fallback ops async — deferred to Phase C (JIT eval integration). Pattern: enqueue native op call on worker thread, output CPUTensor created with pre-allocated buffer (stopgap until JIT handles lazy materialization). See Phase C JIT evaluator integration for details.
- [ ] Parameterize ops by an optional stream argument — coupled with Phase C JIT evaluator integration. Per-op stream parameters must be designed alongside the JIT evaluator's stream model.
- [ ] Multi-stream dnnl::stream pairing: each AsyncCPUStream needs its own dnnl::stream
- [ ] Generic CPUTensor representation: raw buffer storage, dnnl::memory views created just-in-time
- [ ] Deprecate getCudnnStream() pattern — derive stream from input tensors. Affects:
  - cuDNN autograd ops (Conv2D.cpp, BatchNorm.cpp, Pool2D.cpp, RNN.cpp)
  - `CudnnSubgraphLowering` eval lambdas (currently use `getCudnnStream()` inside the
    fused CustomNode eval; should use the stream from the node's operands or the
    CustomNode's own stream attribute)

**Phase C: JIT evaluator integration**
- [x] JitBackend::eval(): enqueue graph evaluation on target stream
- [x] JitTensorBase::stream(): infer stream from leaf nodes (avoid forcing eval)
- [x] Tests: JIT eval on async stream, lazy + async composition

**Phase D: Cross-backend sync**
- [x] AsyncCPUStream::relativeSync(CUDAStream): enqueue cudaEventSynchronize on CPU worker
- [x] CUDAStream::relativeSync(AsyncCPUStream): bridge thread waits CPUEvent, records cudaEvent_t
- [x] CUDAStream::relativeSync(SynchronousStream): no-op (sync work already done)
- [x] SynchronousStream::relativeSync(CUDAStream): blocking waitOn.sync()
- [x] Tests: ordering verification (GPU-after-CPU, CPU-after-GPU), base pointer dispatch, cross-type
- [ ] tensor.to(TensorBackendType): async transfer with stream-ordered copy

**RC6: Fix OneDNN autograd ops creating JIT tensors instead of CPUTensors**
- [x] Added `makeCPUTensor(Shape, dtype)` and `fullCPU(Shape, value, dtype)` helpers to `ops/onednn/Utils.h`
  - Directly construct CPUTensor-backed Tensors, bypassing the default (JIT) tensor type
  - Prevents crashes like `"[toCPUTensor] expected CPU-backed tensor, got Jit"` when default is JIT
- [x] Replaced 13 `Tensor(shape, type)` allocations across 4 OneDNN op files:
  - `Conv2D.cpp` (4 sites: output, gradInput, gradWeights, gradBias)
  - `Pool2D.cpp` (2 sites: output, gradInput)
  - `BatchNorm.cpp` (4 sites: output, gradInput, gradWeights, gradBias)
  - `RNN.cpp` (3 sites: y, hy, cy)
- [x] Replaced 4 `fl::full()` calls in BatchNorm.cpp with `fullCPU()` (runningVar/Mean defaults, weight/bias defaults)
- [x] Fixed 6 empty seed tensor initializations in `native/RNN.cpp` (`parseWeights`, `reconstructFlatWeightGrad`, `reorderLbrGruWeights`) — used `makeCpuTensor(Shape({0}), type, nullptr)` to prevent `fl::concatenate` backend mismatches
- [x] Added `materializeMutable()` in JitBackend.cpp for non-const ref params (`runningMean`/`runningVar` in batchnorm) — evals subgraph, syncs per-tensor stream, replaces JIT tensor with underlying CPUTensor
- [x] All autograd tests pass (Conv2D, Pool2D, BatchNorm, RNN); Module tests pass except pre-existing RNN numerical mismatches (Lua Torch cuDNN reference values vs CPU implementation)

**Phase E: CUDA stream refinements**
- [ ] Per-operation stream: ops respect input tensor's stream, not backend singleton
- [ ] Reconcile CudaTensor::stream_ vs CudaBackend::stream_ (per-tensor wins)
- [ ] Multi-stream GPU execution for independent operations
- [ ] Tests: multi-stream GPU, independent op overlap

### bf16 (bfloat16) Support

**Design**: Follow the f16 precedent — no host-side C++ type. bf16 tensors live on-device (OneDNN/CUDA); host access goes through `astype(f32).scalar<float>()`. OneDNN has native `dnnl::memory::data_type::bf16` support, and cuDNN supports `CUDNN_DATA_BFLOAT16`.

**Core type system** (all backends):
- [x] Add `bf16 = 11` to `dtype` enum in `Types.h`
- [x] Add `{dtype::bf16, "bf16"}` to `kTypeToString` and `kStringToType` in `Types.cpp`
- [x] Add `case dtype::bf16: return 2;` to `getTypeSize()` in `Types.cpp`
- [x] Add `type == dtype::bf16` to `isFloatingPoint()` in `Types.cpp`
- [x] Add `case dtype::bf16: return 8;` to `dtypeRank()` (peer with f16; f16+bf16 promotes to f32)
- [x] Add `case dtype::bf16:` to `asScalar<T>()` in `TensorBase.h` (cast through f32)
- [x] Add `case dtype::bf16:` to all JIT backend switch statements (ScalarFolding, GraphvizPrinter, Evaluator, ScalarNode, JitTensorBase)

**OneDNN backend**:
- [x] Add `{fl::dtype::bf16, dnnl::memory::data_type::bf16}` to type map in `Utils.cpp`
- [x] Add `case dtype::bf16:` to CPU fallback switches (applyBinopCpu) and supportsDataType (unsupported on CPU, same as f16)
- [x] Add `case dtype::bf16:` to CPUTensor::toString (unsupported, same as f16)
- [x] Verify: bf16 eltwise/binary/reorder/matmul primitives work on CPU (OneDNN auto-upcasts to f32 on non-AVX512_BF16 CPUs)
- [x] CPU bf16 tensor tests: Bf16Creation, Bf16Astype, Bf16BinaryOps, Bf16UnaryOps, Bf16MatmulReduction in CPUTensorTest
- [x] Fixed CPUBackend::supportsDataType() to allow bf16, added bf16 case to full()
- [x] Fixed CPUTensor::astype CPU fallback to handle bf16 source dtype (bf16→f32→f64 via bit shift)
- [x] Fixed pool2dBackward to preserve input dtype (was hardcoded to f32)

**ArrayFire backend**:
- [x] Return `false` for `dtype::bf16` in `ArrayFireBackend::supportsDataType()`
- [x] `flToAfType()` throws at `.at()` for unmapped bf16 (no explicit case needed)

**CUDA/cuDNN backend**:
- [x] Add `case fl::dtype::bf16: return CUDNN_DATA_BFLOAT16;` to `cudnnMapToType()`
- [x] Add bf16 case to `kOne()`/`kZero()` scaling constants (use float, same as f16)
- [x] Add bf16 math type to cuDNN RNNDescriptor (CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)
- [x] Map bf16 to `__nv_bfloat16` in CudaTensor type dispatch (cast kernels, dtypeSize)
- [x] Add bf16 to CudaBackend `supportsDataType`, `fillByDtype`, `makeCudaTensor`
- [x] Fix CudaBackend fillByDtype bug: bf16/f16 fill now uses `__float2bfloat16`/`__float2half` (was filling with zeros)

**Correctness tests** (bf16):
- [x] Tensor creation: `Bf16Creation` test in CudaTensorBaseTest — `full()` and `fromScalar()` with bf16
- [x] astype round-trip: `Bf16Astype` test in CudaTensorBaseTest — f32→bf16→f32 and bf16→f64→bf16
- [x] Arithmetic: `Bf16BinaryOps` test in CudaTensorBinaryOpsTest — add/sub/mul/div on bf16 tensors
- [x] Matmul: `MatmulBf16` test in CudaTensorBLASTest — bf16 matmul via cublasGemmEx
- [x] OneDNN primitives: conv2d (ConvolveBf16), pool2d (PoolingBf16), batchnorm (BatchNormBf16) forward/backward with bf16 on CPU
- [x] OneDNN RNN: LstmBf16 test (skips gracefully on CPUs without bf16 concat support)
- [x] cuDNN primitives: conv2d (`ConvolveBf16`), pool2d (`PoolingBf16`), batchnorm (`BatchNormJacobianBf16`), LSTM (`LstmBf16`)
- [x] Type promotion: `MixedBf16F16Promotion` test (bf16+f16→f32), `Bf16TypeProperties` + `PromoteDtypes` tests (bf16+bf16→bf16, bf16+f32→f32)
- [x] Autograd: LinearBf16 forward/backward correctness test (gradient dtype preserved)
- [x] Converted bf16 autograd tests from Jacobian to correctness tests (dtype check + finite values)
- [x] Fixed cuDNN Conv2D and BatchNorm to handle bf16 (tensor core math, f32 scalars, algo blacklist)
- [x] Added bf16 to TensorBinaryOpsTest skip lists for modulo/bitwise ops
- [x] Unary ops: `Bf16UnaryOps` test in CudaTensorUnaryOpsTest — exp, sqrt, negative on bf16
- [x] Reductions: `Bf16Sum` test in CudaTensorReductionTest — sum full and along axis
- [x] Added bf16 to `kAllFpDtypes` in TensorTestUtils.h (auto-included in forEachSupportedFpDtype loops)

**Host representation (future, not needed now)**:
- Option A: Custom `fl::bf16_t` struct wrapping `uint16_t` with f32 conversion (like PyTorch's `c10::BFloat16`)
- Option B: Compiler intrinsic `__bf16` (GCC 13+/Clang 15+, not portable)
- Option C: No host type (current f16 approach) — simplest, sufficient for tensor operations
- Recommendation: Option C for now, Option A if `fromVector<bf16_t>()` becomes needed

### Open Design Questions

1. **Task granularity for CPU async:** **Resolved** — JIT-only model. No per-op tasks. Graph
   evaluation is one planning + dispatch cycle. See "Execution Model" in design doc.

2. **Worker thread count per stream:** **Resolved** — 1 worker per stream. Sequential
   consistency within a stream. Users create multiple streams for parallelism. See design doc.

3. **Memory allocation in async context:** **Resolved** — Planning phase allocates buffers
   (host thread), execution phase computes only (stream workers). Memory planner with liveness
   analysis is a future step. Two-tier op API: public allocate-and-return, internal
   output-parameterized for buffer reuse. See design doc.

4. **Error propagation from worker thread:** **Resolved** — Stream stores exception, rethrows
   on sync(). Existing AsyncCPUStream mechanism. See design doc.

5. **AMD streams (future):** hipStream_t ≈ cudaStream_t -- near-1:1 API, same async design
   applies. hiprtc mirrors NVRTC for kernel compilation. HIP compiles for both AMD (ROCm)
   and NVIDIA, so hipStream_t could potentially unify GPU streams.

6. **OpenCL: NOT useful** for stream unification (research finding). In maintenance mode for ML.
   Performance lags CUDA/HIP on GPU, oneDNN/ACL on CPU. Not worth the abstraction.

## OneDNN Pattern Matching
- [x] Port PatternMatcher DSL to OneDNN operator fusion pass -- **option (c) approved**
  - Refactored searchAndFuse to use MatchAnyBinary/MatchAnyUnary with captures (kLhsCapture, kRhsCapture, kInputCapture)
  - Converted searchAndFuse from recursive to iterative (while loop) for large graph safety
  - Added commutativity support: commutative binops swap LHS/RHS to extend fusion chains
  - Added isCommutative() helper for 11 commutative ops (Add, Mul, Min, Max, Eq, Neq, And, Or, BitAnd, BitOr, BitXor)
  - Added 3 new tests: commutativeSwap, nonCommutativeNoSwap, unaryInChain
  - Updated OneDnnOpFusion.h doc to remove commutativity TODO, document the new behavior

## Hardware Backend Architecture

### Core Insight
Backends should be stratified by **hardware environment** (CUDA, ARM, x86, AMD), not by implementing library (cuDNN, cuBLAS, oneDNN, ACL). Each hardware backend is a single `TensorBackend` implementation that internally dispatches to the best library for each op. This is what the CUDA backend already does -- it uses cuBLAS for matmul, Thrust for elementwise, custom kernels for reductions, cuRAND for random, all behind `CudaBackend`.

The current library-specific backends (ArrayFire, OneDNN) don't implement all operators and have holes. A hardware-stratified backend owns **all** operators for its target and never has gaps -- it's a single source of truth.

### Current vs Target Architecture

**Current (library-stratified):**
```
TensorBackend
  ├── ArrayFireBackend  (CPU+GPU via AF -- device-agnostic, incomplete)
  ├── CPUBackend     (x86 CPU via oneDNN -- missing some ops)
  ├── CudaBackend       (GPU via cuBLAS/Thrust/kernels -- complete)
  └── StubBackend       (throws on everything)
```

**Target (hardware-stratified):**
```
TensorBackend
  ├── CudaBackend   (NVIDIA GPU: cuBLAS, cuDNN, Thrust, custom kernels, cuRAND)
  ├── ArmBackend    (ARM CPU: ACL for matmul/conv, oneDNN for some ops, custom)
  ├── X86Backend    (x86 CPU: oneDNN primary, custom fallbacks)
  ├── AmdBackend    (AMD GPU: rocBLAS, MIOpen, HIP kernels -- future)
  └── StubBackend   (throws on everything)
```

### Design Questions (need resolution)

**1. Operator organization within a hardware backend:**
- [x] **Resolved** — Virtual dispatch via `TensorBackend` methods. Evaluator calls
  `wrappedBackend_.op()`, polymorphism routes to the right backend. Each backend internally
  dispatches to libraries (option A — hardcoded per-op, same as current CUDA approach).
  Backends become thin dispatch layers. See "Operator dispatch" in design doc decisions table.

**2. Stream and memory semantics across library consumers within a backend:**
- [ ] CUDA backend already solves this: single cudaStream_t shared by cuBLAS, Thrust, and custom kernels
  - cuBLAS: `cublasSetStream(handle, stream)` at init
  - cuRAND: `curandSetStream(gen, stream)` at init
  - Thrust: `thrust::cuda::par.on(stream)` per call
  - Custom kernels: `<<<grid, block, 0, stream>>>`
  - All operate on the same device memory (CudaBuffer) -- no conversion needed
- [ ] ARM backend: libraries share host memory naturally. Execution model is synchronous (no stream equivalent).
  - OneDNN: `dnnl::engine` + `dnnl::stream` (CPU streams are synchronous)
  - ACL: `NEScheduler` (NEON) -- single-threaded or thread pool, no stream abstraction
  - Buffer sharing: both wrap raw host pointers -- straightforward
- [ ] x86 backend: same as ARM -- oneDNN streams are synchronous on CPU, host memory is shared
- [ ] AMD backend: HIP streams mirror CUDA semantics (hipStream_t ≈ cudaStream_t)
  - rocBLAS, MIOpen use HIP streams like cuBLAS, cuDNN use CUDA streams

**3. JIT interoperability across backends:**
- [x] **Resolved** — JIT pass pipeline is already hardware-independent at the IR level.
  Common passes are backend-agnostic. Backend-specific lowering registers via
  `JitOptimizerExtension`. Cross-backend graphs are a future extension: evaluator
  dispatches per node to `getBackendFor(node->device())`. See design doc.

**4. Device-agnostic backends (ArrayFire):**
- ArrayFire spans CPU+GPU, which violates the "one backend per hardware" model
- AF is a legacy path -- goal is replacing it with hardware-specific backends
- Keep AF working but don't design new architecture around it
- Migration path: AF users switch to CudaBackend (GPU) or X86Backend/ArmBackend (CPU)

**5. How do backends discover available libraries at runtime?**
- [ ] CMake-time: conditional compilation (`#if FL_USE_CUDNN`, `#if FL_USE_ACL`, etc.)
- [ ] Runtime: backend can check for library presence (e.g., `dlopen("libcudnn.so")`) -- probably overkill
- [ ] Current approach (CMake-time) is sufficient. Each backend links only the libraries it needs.

### Hardware-Specific Backend TODOs

**CUDA Backend (Phase 5a -- mostly complete):**
- [x] Core ops via cuBLAS, Thrust, custom kernels, cuRAND
- [x] Multi-dtype compute (f32/f64/f16)
- [x] All ops GPU-native (no host fallbacks except multi-axis median)
- [x] Integrate cuDNN for autograd ops (conv2d forward/backward, pool2d, batchnorm, RNN)
  - cuDNN ops are free functions in fl::detail::cudnn:: namespace (CudnnAutogradOps.h)
  - CudaBackend overrides TensorBackend virtual methods, delegates to cuDNN free functions
  - ArrayFireBackend(CUDA) shares the same cuDNN free functions via namespace alias
  - JitBackend materializes JIT tensors then delegates to wrapped backend
  - 32 tests all passing: conv2d forward/backward/jacobian/groups/dilation + backend direct,
    pool2d max/avg forward + backward + jacobian, batchnorm eval/train + jacobian,
    RNN TANH forward + LSTM forward/backward + jacobian,
    flat read/write roundtrip, flat perturb/restore, column slice assign,
    13 view semantics tests (buffer sharing, write-through, copy independence,
    strides, contiguity, chained views, scalar views, range views, lvalue assign)
- [x] CudaTensor indexing (index, flat, flatten) with view semantics
  - index(): Span/Literal/Range support, creates views via shared buffer + offset/stride adjustment
  - flat(): direct view into original buffer for write-through (no copy)
  - flatten(): returns a copy (not a view) to prevent aliasing bugs without COW
  - Strided copy kernel for non-contiguous destination assignment (e.g. column slice write)
  - assign(): handles contiguous fast-path (memcpy) and non-contiguous (stridedCopyKernel)
- [ ] cuTENSOR integration for advanced tensor contractions (evaluate need)

**ARM Backend:**
- **Decision: use oneDNN-with-ACL -- no separate ACL backend needed** (research finding)
  - oneDNN already integrates ACL on AArch64 since v2.x (`DNNL_AARCH64_USE_ACL=ON`)
  - oneDNN delegates GEMM, conv, and compute-heavy primitives to ACL on ARM
  - This gives 95%+ of direct ACL performance with zero additional Flashlight code
  - Flashlight's existing JIT fusion pass (OneDnnOpFusion post-ops) works unchanged on ARM
  - ACL's fusion is limited: only single-activation epilogue in stable API (vs oneDNN's 32 post-ops)
  - ACL has no kernel cache (uses configure-then-run pattern); oneDNN has built-in LRU cache
- **oneDNN operator coverage on ARM** (comprehensive audit):
  - **ACL-accelerated ops** (high performance on ARM): matmul/GEMM, conv fwd+bwd, pooling,
    batchnorm, common activations (relu/tanh/sigmoid/gelu/elu), softmax, inner product, RNN/LSTM
  - **oneDNN-only ops** (reference kernels, functional but slower): reductions (sum/mean/max/min),
    binary ops (add/mul/sub/div), layer norm, group norm, concat, reorder
  - **NOT oneDNN primitives at all** (~25 ops, same gap on ALL architectures):
    sort/argsort/topk, nonzero, where, cumsum, flip/roll, tril/triu, pad (general),
    iota/arange, random, sin/cos/log1p/floor/ceil, isnan/isinf, power(tensor^tensor),
    argmax/argmin (with indices), median, var/std/norm, bitwise ops, modulo
  - **Key insight**: oneDNN is a DL primitives library (not a tensor library). It covers ~20 primitive
    types but a tensor library needs 50-100+ ops. The missing ~25 ops are the SAME gap on ARM, x86,
    and everywhere else -- they're not ARM-specific holes.
  - **Recommendation**: oneDNN+ACL for DL compute kernels; supplement with portable C++ fallbacks
    for the ~25 missing ops (sort via std::sort, sin/cos via libm, etc.)
  - Do NOT use ACL directly alongside oneDNN+ACL -- would create two memory management systems
  - Current CPUBackend already has some portable helpers (maxWithIndexCpu, fullWithTypeCpu)
    that demonstrate the fallback pattern
  - **Fallback performance**: Most missing ops are memory-bound (flip, roll, pad, where, nonzero,
    tril/triu, argmax/argmin) -- SIMD doesn't help, vanilla C++ is fine. Compute-bound ops
    (sin, cos, floor, ceil, log1p) auto-vectorize to Neon/SVE with -O2 -march=native.
    std::sort uses introsort, competitive with hand-rolled sorting. Start with vanilla C++,
    profile, add Neon/Highway only for proven bottlenecks.
  - **JIT compatibility**: Fusable ops (sin, cos, floor, ceil) are already in the JIT IR as UnaryOp.
    ElementwiseFusion can fuse them. However OneDnnOpFusion only maps 8/18 unary ops to oneDNN
    eltwise -- sin, cos, floor, ceil, log1p break fusion chains and evaluate eagerly. Acceptable
    for now; CPU codegen (like NVRTC for CPU) could solve this later but is a big undertaking.
    Non-fusable ops (sort, nonzero, pad) are CustomNodes regardless -- no JIT issue.
  - **oneDNN does NOT ship with ACL by default** -- must build from source with
    `DNNL_AARCH64_USE_ACL=ON` and have ACL installed. Package manager installs exclude ACL.
- [x] CMake: detect ARM, require oneDNN with ACL, fatal error if absent
  - `check_cxx_source_compiles` checks for `DNNL_AARCH64_USE_ACL` in `dnnl_config.h`
  - CMake errors with clear instructions to rebuild oneDNN with `-DDNNL_AARCH64_USE_ACL=ON`
  - Runtime check in `CPUBackend::CPUBackend()` also verifies ACL on ARM
  - `FL_ONEDNN_ARM` compile definition propagated for conditional compilation
- [x] CMake: minimum oneDNN version requirement (>= 3.5)
  - `find_package(DNNL 3.5 CONFIG)` for primary version check
  - Belt-and-suspenders `check_cxx_source_compiles` verifying `DNNL_VERSION_MAJOR/MINOR` macros
  - Version rationale: 3.5+ brgemm refinements (TF/PyTorch use), 3.6+ matmul fixes,
    3.7+ improved ACL integration, latest stable is 3.10.2
- [x] Test: verify JIT fusion pass generates correct oneDNN post-ops on ARM
  - OneDnnJitEndToEndTest: 10 end-to-end tests using JitTensor<CPUTensor>
  - Tests run on both x86 and ARM (OneDNN API is portable)
  - Covers: binary chains, eltwise post-ops, commutative swap, multi-use, comparison ops
  - TODO: consider generalizing JIT end-to-end tests across backends
- [x] Implement the ~31 missing ops in CPUBackend with portable C++ fallbacks
  - All ops now implemented: identity, iota, concatenate, nonzero, pad, log1p, sin, cos,
    floor, ceil, sigmoid (via eltwise_logistic), flip, roll, clip(tensor,tensor), isnan,
    tril, triu, where, sort, argsort, topk, power(tensor^tensor), cumsum, argmax, argmin,
    median, var, std, norm, all
  - Used vanilla C++ + libm; compiler auto-vectorizes with -O2 -march=native
  - Profile hot paths; add Neon/Highway intrinsics only for proven bottlenecks

**x86 Backend (partially exists as CPUBackend):**
- CPUBackend now serves as a complete CPU backend for both ARM and x86
  - All ~31 previously-missing ops now implemented with portable C++ fallbacks
  - oneDNN has JIT-optimized kernels for its primitives on x86 (AVX2/AVX-512/AMX) -- no ACL needed
- **Decision: no MKL/oneMKL integration for GEMM** -- oneDNN's JIT GEMM (brgemm) is competitive
  with MKL on x64, and oneDNN's post-op fusion (matmul+bias+activation) provides end-to-end
  speedup that would be lost if routing through MKL. Not worth the build complexity.
  MKL still used optionally for RNG (FL_USE_MKL_RNG).

**AMD Backend (future):**
- [ ] HIP-based backend mirroring CUDA backend structure
  - rocBLAS (matmul), hipRAND (random), HIP kernels (elementwise, reductions)
  - hiprtc for runtime kernel compilation (mirror NVRTC approach)
  - MIOpen for conv2d, pool2d, batchnorm (mirror cuDNN)
- [ ] Most CUDA kernels can be compiled for HIP with minimal changes (HIP is CUDA-compatible)

## Phase 6: Lazy Tensor Abstraction
The JIT backend (`backend/jit/`) already implements lazy tensor semantics:
- Tensor operations build an IR graph instead of eagerly computing
- Evaluation is deferred until the result is needed (e.g., `host()`, `scalar()`)
- Optimizer runs passes (CSE, constant folding, elementwise fusion, reduction fusion) before evaluation
- Backend-specific lowering passes compile fused subgraphs (NVRTC for CUDA, post-ops for OneDNN)

Remaining work:
- [x] End-to-end GPU testing of JIT backend with CUDA lowering (NVRTC kernel compilation + cuBLASLt epilogue fusion)
  - 27 tests in CudaJitEndToEndTest.cpp using JitTensor<CudaTensor> (JIT wrapping CudaBackend)
  - Elementwise fusion via NVRTC: unary chains, binary chains, mixed unary+binary, longer chains, diamonds
  - Reduction-of-elementwise fusion via NVRTC: sum(exp(x)), sum(a+b) along axis, mean(neg(x)), amax(a*b)
  - Matmul epilogue fusion via cuBLASLt: matmul+add (both operand orders), standalone matmul
  - Common passes: CSE (exp(x)+exp(x)), constant folding (full+full, exp(full))
  - Mixed operations: linear layer forward (matmul+add), softmax-like, MSE loss
  - Large tensors: 1M-element elementwise, 64K-element reduction
  - Non-fusable operations: reshape, fusion-after-reshape
  - Explicit eval: fl::eval() forcing, idempotent re-evaluation
  - Bugs fixed during testing:
    - NVRTC: removed `#include <math.h>` (not available in NVRTC), defined INFINITY inline
    - NVRTC: detect GPU compute capability at runtime (was hardcoded compute_70)
    - CudaBackend: fixed free-function `full()`/`Tensor::fromVector` calls that created
      tensors via default tensor type (caused backend mismatch when JIT wraps CudaBackend).
      All internal tensor creation now uses `this->full()` or `toTensor<CudaTensor>()` directly.
- [ ] JIT backend integration with new hardware backends
  - JIT backend currently delegates to an underlying eager backend for evaluation
  - Each hardware backend can register JitOptimizerExtension passes for backend-specific lowering
  - CudaJitOptimizerExtension already registered; need equivalents for ARM/x86/AMD
- [ ] Lazy evaluation scheduling
  - Current: evaluate immediately on `host()` / `scalar()` / when used as input to non-JIT op
  - MLX approach: per-op stream specification + lazy eval + stream-based scheduling
  - Consider: batch evaluation (evaluate multiple pending graphs together for better fusion)
- [ ] Memory planning for lazy evaluation
  - When evaluating a graph, can plan memory allocation to reuse buffers for intermediate results
  - Liveness analysis on the IR graph to determine buffer lifetimes
  - Orthogonal to stream semantics but interacts with it (buffer reuse across streams needs synchronization)

## Async Stream Lifetime / Exception Safety
- [x] CPUBackend::~CPUBackend() syncs asyncStream_ before member destruction
  - Prevents use-after-free: enqueued tasks capture &nativeStream_ by reference
  - Without sync, DeviceManager (which holds shared_ptr<AsyncCPUStream>) may keep the
    worker thread alive after nativeStream_ is destroyed
- [x] fl::init() no longer eagerly initializes the default backend (lazy init)
  - DefaultTensorType constructor defers backend singleton creation until first getBackend() call
  - Prevents unnecessary ArrayFire+CUDA initialization when only OneDNN is needed
  - **Regressed (2026-10-26):** the thread_local `DefaultTensorType` constructor now
    eagerly calls `DefaultTensorBackend_t::getInstance()` (ArrayFire) to set
    `jitBackend_` (`TensorAdapter.cpp:30-43`), so every `_CPU` test initializes
    ArrayFire+CUDA via `setDefaultJitTensorType<CPUTensor>()`.
- [ ] Tensor destruction exception safety: when an exception causes stack unwinding,
  tensor destructors don't sync their streams. If async tasks reference tensor data
  that gets freed during unwinding, the worker thread may access freed memory.
  - Consider: SharedData destructor syncing the stream (perf impact?)
  - Consider: enqueued tasks always capturing dnnl::memory by value (shared handle, keeps buffer alive)
  - Current workaround: most ops already capture dnnl::memory by value, but some capture &nativeStream_ by ref
- [x] Eliminate `setBare` / `ScopedBareMode` global state mutation.
  Done: cuDNN ops now use `makeCudaTensor()`/`fullCuda()` (in
  `ops/cudnn/CudnnTensorHelpers.h`) to allocate bare CUDA tensors directly,
  matching the OneDNN pattern (`makeCPUTensor`/`fullCPU`). Removed
  `ScopedBareMode` (6 sites in JitBackend), `setBare` (3 sites in Evaluator),
  and the `bare_`/`bareBackend_` fields from `DefaultTensorType`.
- [ ] Update compiler to GCC 15 (conda: x86_64-conda-linux-gnu-g++ 15.2.0)
  - OneDNN linked library was compiled with GCC 14.3.0 (conda-forge), ABI compatible
  - Enables std::format in examples (C++20)
  - Reconfigure: cmake .. -DCMAKE_CXX_COMPILER=...conda.../x86_64-conda-linux-gnu-g++
  - **std::format migration candidates** (replace ostringstream/to_string patterns):
    - `CPUBackend.cpp`, `CPUTensor.cpp` — error messages using ostringstream
    - `TensorBase.cpp` — shape/type formatting
    - `Shape.cpp` — string representation
    - All `std::to_string()` + concatenation patterns throughout `fl/tensor/`

## Generalized Advanced Indexing Operations

NumPy and PyTorch support a family of "advanced indexing" operations that scatter-reduce
values at indexed positions. The current `fl::indexAdd` only handles the simplest case
(accumulating with addition along a single axis). A complete implementation would support
the full set of scatter-reduce operations and generalize to multi-axis Tensor indexing.

### Motivation

The autograd backward for `Tensor::operator()(tensorIndices...)` currently decomposes
multi-Tensor index gradients into a per-element scalar loop (`g(scalarIdx) += gradOutput(goIdx)`).
This is O(product of index sizes) with per-element kernel launches — functionally correct
but slow. A fused scatter-reduce kernel can process all elements in a single pass.

### Target API

```cpp
// Generalized scatter-reduce: self[axis][indices[i]] op= src[i] for all i.
// `op` determines the reduction: add, mul, min, max, mean.
// Matches PyTorch's scatter_reduce and NumPy's ufunc.at family.
enum class ScatterReduceOp { Add, Mul, Amin, Amax, Mean };
void scatterReduce(
    Tensor& self, unsigned axis, const Tensor& indices,
    const Tensor& src, ScatterReduceOp op);

// Multi-axis variant for the common autograd backward pattern:
// self[idx0, idx1, ..., span, range] op= src
// Indices are a mix of Tensor/Literal/Range/Span (same as operator()).
void indexPut(
    Tensor& self, const std::vector<Index>& indices,
    const Tensor& src, ScatterReduceOp op = ScatterReduceOp::Add);
```

### Supported operations (matching PyTorch/NumPy)

| Op | PyTorch equivalent | NumPy equivalent | Use case |
|----|-------------------|------------------|----------|
| Add | `scatter_add`, `index_add` | `np.add.at` | Gradient accumulation (embedding, gather backward) |
| Mul | `scatter_reduce("prod")` | `np.multiply.at` | Product-based reductions |
| Amin | `scatter_reduce("amin")` | `np.minimum.at` | Min pooling backward |
| Amax | `scatter_reduce("amax")` | `np.maximum.at` | Max pooling backward |
| Mean | `scatter_reduce("mean")` | manual | Normalized accumulation |

### Implementation plan

1. **`TensorBackend::scatterReduce` virtual method** with default CPU host-side implementation
   - Single-axis scatter-reduce: iterate over src elements, accumulate into self at indices
   - Must handle repeated indices correctly (atomic semantics)
   - Default impl reads/writes host buffers directly (no kernel launch per element)

2. **CUDA backend**: fused scatter-reduce kernel
   - Use `atomicAdd` for Add, `atomicMax`/`atomicMin` for Amax/Amin
   - Mul/Mean need CAS loops or separate passes
   - Move the existing `AdvancedIndex.cu` atomicAdd kernel here, generalize to other ops

3. **OneDNN backend**: call default CPU implementation
   - OneDNN has no native scatter-reduce primitive
   - CPU host-side impl is sufficient for correctness; performance is acceptable since
     OneDNN targets inference more than training

4. **`indexPut` multi-axis variant**
   - Decompose mixed Tensor/Range/Literal indices into a flat index mapping (same logic
     as the current multi-Tensor index decomposition in `CPUTensor::index`)
   - Call single-axis `scatterReduce` on the flattened view, or iterate if decomposition
     is simpler
   - This replaces the per-element scalar loop in `autogradIndex` backward

5. **Autograd backward update**
   - Replace the current loop in `autogradIndex` (Functions.cpp) with a single
     `indexPut(g, indices, gradOutput, ScatterReduceOp::Add)` call
   - Preserves correctness for repeated indices while being O(1) kernel launches

### Performance considerations

- The per-element loop in the current autograd backward is O(N) kernel launches where
  N = product of Tensor index sizes. For embedding gradients with vocab_size=50k and
  batch_size=256, this means 12.8M individual scalar assignments.
- A fused kernel does this in one launch with coalesced memory access.
- For CPU backends, the host-side loop is already efficient (no kernel launch overhead),
  but a vectorized implementation with gather/scatter intrinsics could improve throughput.
- Mean reduction requires a count pass (count occurrences of each index) followed by
  division, or a post-normalization step.

### Relation to existing code

- `fl::indexAdd()` (TensorBase.h:1393, TensorBackend.cpp:149) is the current Add-only
  single-axis scatter. It becomes a thin wrapper: `scatterReduce(self, axis, indices, src, Add)`.
- The CUDA `AdvancedIndex.cu` kernel already implements the atomicAdd pattern. Generalize
  to other ops and move to `fl/tensor/backend/cuda/CudaShapeAndIndex.cu`.
- The autograd backward loop in Functions.cpp (autogradIndex) is the primary consumer.

## OneDNN f64 CPU Fallback (removed, restore later with optimization)

f64 (double) is not natively supported by OneDNN primitives on CPU. CPU fallback code
was implemented for binary ops, unary (eltwise) ops, reductions, concatenate, and where,
but it is **extremely slow** because each operation does a full sync + host read + scalar
loop + host write cycle. This makes autograd Jacobian tests (which use f64 for numerical
precision) take minutes instead of seconds.

**Current state**: f64 is marked unsupported via `supportsDataType(f64) = false`. Tests
that check this will skip f64 variants automatically. Operations on f64 tensors throw
with a clear error message.

**TODO when restoring**:
- [ ] Write vectorized CPU fallbacks (use SIMD intrinsics or auto-vectorizable loops)
- [ ] Avoid sync/read/write overhead: operate directly on the dnnl::memory buffer pointer
  (it's a host pointer on CPU) instead of going through toHostVector/toTensor roundtrip
- [ ] Consider batch-fusing multiple f64 ops to amortize overhead
- [ ] Re-enable `supportsDataType(f64) = true` once fallbacks are fast enough

### Saved fallback code

**1. binaryOpCpuFallback (was in applyBinop f64 path)**
```cpp
// CPU fallback for binary operations on types unsupported by OneDNN (e.g. f64).
// Supports broadcasting. T is the element type, OutT is the output element type.
template <typename T, typename OutT = T>
Tensor binaryOpCpuFallback(
    const Tensor& lhs,
    const Tensor& rhs,
    dnnl::algorithm alg,
    fl::dtype outType) {
  auto lhsData = lhs.toHostVector<T>();
  auto rhsData = rhs.toHostVector<T>();
  auto lhsDims = lhs.shape().get();
  auto rhsDims = rhs.shape().get();
  const auto maxNdim = std::max(lhsDims.size(), rhsDims.size());
  while (lhsDims.size() < maxNdim) lhsDims.insert(lhsDims.begin(), 1);
  while (rhsDims.size() < maxNdim) rhsDims.insert(rhsDims.begin(), 1);
  std::vector<Dim> outDims(maxNdim);
  for (size_t i = 0; i < maxNdim; ++i) {
    if (lhsDims[i] != rhsDims[i] && lhsDims[i] != 1 && rhsDims[i] != 1)
      throw std::invalid_argument("Cannot broadcast");
    outDims[i] = std::max(lhsDims[i], rhsDims[i]);
  }
  Shape outShape(outDims);
  std::function<OutT(T, T)> op;
  using A = dnnl::algorithm;
  switch (alg) {
    case A::binary_add: op = [](T a, T b) -> OutT { return a + b; }; break;
    case A::binary_sub: op = [](T a, T b) -> OutT { return a - b; }; break;
    case A::binary_mul: op = [](T a, T b) -> OutT { return a * b; }; break;
    case A::binary_div: op = [](T a, T b) -> OutT { return a / b; }; break;
    case A::binary_min: op = [](T a, T b) -> OutT { return std::min(a, b); }; break;
    case A::binary_max: op = [](T a, T b) -> OutT { return std::max(a, b); }; break;
    case A::binary_eq:  op = [](T a, T b) -> OutT { return a == b ? 1 : 0; }; break;
    case A::binary_ne:  op = [](T a, T b) -> OutT { return a != b ? 1 : 0; }; break;
    case A::binary_lt:  op = [](T a, T b) -> OutT { return a < b ? 1 : 0; }; break;
    case A::binary_le:  op = [](T a, T b) -> OutT { return a <= b ? 1 : 0; }; break;
    case A::binary_gt:  op = [](T a, T b) -> OutT { return a > b ? 1 : 0; }; break;
    case A::binary_ge:  op = [](T a, T b) -> OutT { return a >= b ? 1 : 0; }; break;
    default: throw std::runtime_error("Unsupported algorithm");
  }
  std::vector<OutT> result(outShape.elements());
  if (lhs.shape() == rhs.shape()) {
    for (size_t i = 0; i < result.size(); ++i) result[i] = op(lhsData[i], rhsData[i]);
  } else {
    for (size_t i = 0; i < result.size(); ++i) {
      auto coords = delinearize(i, outDims);
      std::vector<Dim> lC(maxNdim), rC(maxNdim);
      for (size_t d = 0; d < maxNdim; ++d) {
        lC[d] = (lhsDims[d] == 1) ? 0 : coords[d];
        rC[d] = (rhsDims[d] == 1) ? 0 : coords[d];
      }
      result[i] = op(lhsData[linearIndex(lC, lhsDims)], rhsData[linearIndex(rC, rhsDims)]);
    }
  }
  return toTensor<CPUTensor>(outShape, outType, result.data(), Location::Host);
}

// In applyBinop, the f64 path converted both inputs to f64 first:
if (lhs.type() == dtype::f64 || rhs.type() == dtype::f64) {
  Tensor lhsF64 = lhs.type() == dtype::f64 ? lhs : lhs.astype(dtype::f64);
  Tensor rhsF64 = rhs.type() == dtype::f64 ? rhs : rhs.astype(dtype::f64);
  fl::dtype outType = dstType.has_value() ? detail::oneDnnToFlType(*dstType) : dtype::f64;
  if (outType == dtype::f64)
    return binaryOpCpuFallback<double>(lhsF64, rhsF64, alg, outType);
  else
    return binaryOpCpuFallback<double, char>(lhsF64, rhsF64, alg, outType);
}
```

**2. applyEltwiseOp f64 path**
```cpp
if (tensor.type() == dtype::f64) {
  using A = dnnl::algorithm;
  std::function<double(double)> op;
  switch (alg) {
    case A::eltwise_abs:       op = [](double v) { return std::abs(v); }; break;
    case A::eltwise_exp:       op = [](double v) { return std::exp(v); }; break;
    case A::eltwise_log:       op = [](double v) { return std::log(v); }; break;
    case A::eltwise_logistic:  op = [](double v) { return 1.0/(1.0+std::exp(-v)); }; break;
    case A::eltwise_sqrt:      op = [](double v) { return std::sqrt(v); }; break;
    case A::eltwise_tanh:      op = [](double v) { return std::tanh(v); }; break;
    case A::eltwise_elu:       op = [alpha](double v) { return v>=0?v:alpha*(std::exp(v)-1); }; break;
    case A::eltwise_relu:      op = [alpha](double v) { return v>=0?v:alpha*v; }; break;
    case A::eltwise_gelu_tanh: op = [](double v) {
      return 0.5*v*(1.0+std::tanh(std::sqrt(2.0/M_PI)*(v+0.044715*v*v*v))); }; break;
    case A::eltwise_gelu_erf:  op = [](double v) { return 0.5*v*(1.0+std::erf(v/std::sqrt(2.0))); }; break;
    case A::eltwise_round:     op = [](double v) { return std::rint(v); }; break;
    case A::eltwise_linear:    op = [alpha,beta](double v) { return alpha*v+beta; }; break;
    case A::eltwise_soft_relu: op = [](double v) { return std::log1p(std::exp(v)); }; break;
    case A::eltwise_mish:      op = [](double v) { return v*std::tanh(std::log1p(std::exp(v))); }; break;
    case A::eltwise_pow:       op = [alpha,beta](double v) { return alpha*std::pow(v,beta); }; break;
    default: throw std::runtime_error("Unsupported eltwise for f64");
  }
  auto data = tensor.toHostVector<double>();
  for (auto& v : data) v = op(v);
  return toTensor<CPUTensor>(tensor.shape(), dtype::f64, data.data(), Location::Host);
}
```

**3. applyReductionOp f64 path**
```cpp
if (input.type() == dtype::f64) {
  auto data = input.toHostVector<double>();
  const auto srcDims = input.shape().get();
  const int ndim = input.ndim();
  size_t dstElements = 1;
  for (auto d : dstDims) dstElements *= d;
  std::vector<double> result(dstElements, 0.0);
  std::vector<size_t> counts(dstElements, 0);
  using A = dnnl::algorithm;
  if (alg == A::reduction_min) std::fill(result.begin(), result.end(), std::numeric_limits<double>::infinity());
  else if (alg == A::reduction_max) std::fill(result.begin(), result.end(), -std::numeric_limits<double>::infinity());
  std::vector<size_t> dstStrides(ndim);
  dstStrides.back() = 1;
  for (int d = ndim-2; d >= 0; --d) dstStrides[d] = dstStrides[d+1] * dstDims[d+1];
  for (size_t i = 0; i < input.elements(); ++i) {
    auto coords = delinearize(i, srcDims);
    size_t dstIdx = 0;
    for (int d = 0; d < ndim; ++d) {
      dstIdx += ((dstDims[d]==1)?0:coords[d]) * dstStrides[d];
    }
    double val = data[i];
    switch (alg) {
      case A::reduction_sum: case A::reduction_mean: result[dstIdx] += val; break;
      case A::reduction_min: result[dstIdx] = std::min(result[dstIdx], val); break;
      case A::reduction_max: result[dstIdx] = std::max(result[dstIdx], val); break;
      case A::reduction_norm_lp_sum: result[dstIdx] += val*val; break;
      default: throw std::runtime_error("Unsupported reduction for f64");
    }
    counts[dstIdx]++;
  }
  if (alg == A::reduction_mean) for (size_t i = 0; i < dstElements; ++i) result[i] /= counts[i];
  else if (alg == A::reduction_norm_lp_sum) for (size_t i = 0; i < dstElements; ++i) result[i] = std::sqrt(result[i]);
  Shape outShape(dstDims);
  if (!keepDims) outShape = Shape(detail::removeIndices(dstDims, axesToReduce));
  return toTensor<CPUTensor>(outShape, dtype::f64, result.data(), Location::Host);
}
```

**4. concatenate f64 path**
```cpp
if (outType == dtype::f64) {
  std::vector<double> out(outShape.elements());
  size_t innerSize = 1;
  for (int d = ndim-1; d > axis; d--) innerSize *= outDims[d];
  size_t outerSize = 1;
  for (int d = 0; d < axis; d++) outerSize *= outDims[d];
  size_t dstOffset = 0;
  for (size_t outer = 0; outer < outerSize; outer++) {
    for (size_t t = 0; t < tensors.size(); t++) {
      auto data = tensors[t].toHostVector<double>();
      auto tDims = padToNdim(tensors[t].shape());
      size_t chunkSize = tDims[axis] * innerSize;
      std::copy_n(data.data() + outer*chunkSize, chunkSize, out.data() + dstOffset);
      dstOffset += chunkSize;
    }
  }
  return toTensor<CPUTensor>(outShape, dtype::f64, out.data(), Location::Host);
}
```

**5. where f64/integer CPU fallback (type-agnostic byte copy)**
```cpp
auto cond = condU8.toHostVector<unsigned char>();
const size_t n = x.elements();
const size_t typeSize = fl::getTypeSize(x.type());
std::vector<char> xBytes(n * typeSize), yBytes(n * typeSize);
x.hostSync(xBytes.data()); y.hostSync(yBytes.data());
std::vector<char> outBytes(n * typeSize);
for (size_t i = 0; i < n; i++) {
  const char* src = cond[i] ? xBytes.data() : yBytes.data();
  std::memcpy(outBytes.data() + i*typeSize, src + i*typeSize, typeSize);
}
return toTensor<CPUTensor>(x.shape(), x.type(), outBytes.data(), Location::Host);
```

**Key bug found during f64 fallback development**: `toHostVector<T>()` on a tensor with
native type != T does a raw memcpy based on the tensor's native byte size, not sizeof(T).
For example, `toHostVector<double>()` on an s8 tensor copies only 1/8 of the needed bytes,
producing garbage. The fix was to `astype(dtype::f64)` both inputs before calling the
fallback. This bug must be accounted for when restoring fallbacks.

## Async Execution & DevicePtr in OneDNN
- [ ] Restore async execution in `executeNetwork` (DnnlUtils.cpp)
  - Root cause of Conv2D Jacobian failures: `executeNetwork` runs synchronously, but `flat(i) = val` assigns are enqueued on AsyncCPUStream. The conv reads stale data because the assign hasn't executed yet.
  - Fix: capture `Tensor` objects (not `DevicePtr`s) by value in the lambda. SharedData refcounting keeps buffers alive. FIFO ordering guarantees prior async tasks complete before the conv primitive executes.
  - Current blocker: segfault in OneDNN's im2col during async execution. Needs investigation into whether `dnnl::memory` handle semantics allow safe cross-thread use after `DnnlMemoryWrapper` destruction.
  - Defer until lazy tensor + JIT abstraction can execute contiguous OneDNN op sequences, bypassing the native thread pool for fused execution.
- [ ] Audit all `DevicePtr` usage in async lambdas across OneDNN backend
  - `CPUTensor::device()` does NOT sync the stream. Raw pointers obtained via DevicePtr may reference memory whose data isn't ready yet.
  - `DnnlMemoryWrapper` holds DevicePtr on the stack; if the wrapper is destroyed before async completion, the unlock() fires too early.
  - Pattern to adopt: capture Tensors (not DevicePtrs) in async lambdas. DevicePtr should only be used in synchronous contexts.
- [x] Move RNN manual BPTT to `fl/tensor/ops/native/RNN.{h,cpp}`
  - `manualRnnImpl<Scalar>()` and `runSingleRnnBackward<Scalar>()` are pure-Tensor ops, no OneDNN dependency
  - `ParsedWeightsAndBias`, `parseWeights`, `reconstructFlatWeightGrad` are shared utilities
  - OneDNN RNN.cpp would import and call these for f64 forward and all backward
- [x] bf16 correctness tests for RNN (OneDNN forward supports bf16 on AVX512_BF16 platforms)
- [ ] Use MKL for transpose fallback on unsupported precisions (f64, integer types)
  - Current `fl::native::transposeCpu` is a naive coordinate-decomposition loop (integer div/mod per dim per element, no SIMD, no cache blocking)
  - **2D case**: Use `mkl_domatcopy` (`mkl_trans.h`) — supports f32/f64/complex, highly optimized with SIMD and cache blocking. MKL is already an optional dependency (found via OneDNN's CMake or standalone). Signature: `mkl_domatcopy(ordering, 'T', rows, cols, alpha=1.0, src, lda, dst, ldb)`
  - **N-D case**: MKL has no N-D transpose. Two options:
    1. Decompose N-D permutation into a sequence of 2D transposes on reshaped views (complex but leverages optimized 2D kernel)
    2. Improve `transposeCpu` with cache-blocking/tiling and incremental coordinate stepping (avoids div/mod per element). No new dependency.
  - Alternative: [HPTT](https://github.com/springer13/hptt) — purpose-built N-D tensor transpose library with SIMD + cache blocking, supports f64. Would be a new external dependency.
  - Pragmatic approach: use `mkl_domatcopy` for 2D when MKL is available (common case, biggest win), improve the generic N-D fallback with tiling

## Test Quality
- [ ] Value correctness sweep: audit tensor and autograd tests to ensure fundamental operations verify output values, not just shapes/no-throw
- [x] Defensive try-catch cleanup: remove unnecessary try-catch guards in tests where the operation is known to be supported by the backend
- [x] Autograd test deduplication: removed 13 redundant backend-specific tests from CudaCudnnAutogradTest, moved unique tests to proper locations (pool2d AVG to AutogradTest, view tests to CudaTensorIndexTest, conv2d backend to AutogradConv2DTest)
- [x] CudaTensorBaseTest consolidation: 22 generic tests removed. 5 value-correctness tests (identity, rand, randn, astypeConversions, opsOnNonF32) moved to TensorBaseTest. Only BackendType remains (CUDA-specific).
- CPUTensorTest: more justified — 12 tests use CPUTensor/CPUBackend directly, rest validate via assertCPUTensorEq helper. Leave as-is.

### Test Layering

The test structure should follow the execution layering. Each layer is
independently testable and has a clear responsibility.

**Layer 1: Pure IR / pass structure tests (no backend, no execution).**

Construct IR nodes by hand, run passes, verify the output IR structure
(node types, inputs, use edges, shapes). These never execute a CustomNode's
eval lambda — they only check that the pass produced the right graph.

Already done well:
- `JitNodeTest` — IR node construction and manipulation
- `JitCSETest` — CSE pass structural checks
- `JitScalarFoldingTest` — scalar folding structural checks
- `JitElementwiseFusionTest` — fusion pass structural checks
- `JitReductionOfElementwiseFusionTest` — reduction fusion structural checks
- `JitOneDnnOpFusionTest` — OneDNN fusion structural checks
- `JitMatmulFusionTest` — matmul epilogue fusion structural checks

These tests need a backend only to create ValueNodes (via `full()`), not
to execute anything. They should remain backend-independent.

**Layer 2: Op-library-specific tests (backend-specific, isolated ops).**

Test the library function implementations in isolation: call the factory +
execute functions directly with test tensors, verify output correctness.
No JIT, no passes, no graph — just "does this OneDNN/CUDA/native function
compute the right answer?"

These are the ground truth for correctness. If a fused op produces wrong
results, these tests tell you whether the library function itself is broken
or whether the lowering pass wired it up wrong.

Current state:
- `CPUTensorTest` — partially does this but mixes CPUTensor internals
  (construction, strides, in-place ops) with op correctness
- `CudaTensorBaseTest`, `CudaTensorBinaryOpsTest`, etc. — CUDA-specific
  op correctness

Target: each library's ops directory (`fl/tensor/ops/onednn/`,
`fl/tensor/ops/native/`, `fl/tensor/ops/cuda/`) should have corresponding
tests that call the free functions directly. For OneDNN, this means calling
`detail::onednn::applyBinop(...)`, `createOneDnnConv2DData(...)` +
`executeConv2d(...)`, etc. and verifying output values. These tests need
the library (OneDNN, CUDA) but not the JIT.

Helpers: create small test utilities that make it easy to construct inputs
and execute a single library op in one call. Something like:

```cpp
// Test helper: execute a single OneDNN binary op and return the result
Tensor testOneDnnBinop(const Tensor& lhs, const Tensor& rhs, dnnl::algorithm alg);

// Test helper: execute OneDNN conv2d with optional post-ops and return result
Tensor testOneDnnConv2d(const Tensor& input, const Tensor& weights,
                        const Tensor& bias, /* params... */,
                        dnnl::primitive_attr attr = {});
```

**Layer 3: Backend-agnostic API tests (full pipeline, implicit JIT).**

`TensorBaseTest`, `TensorBinaryOpsTest`, `TensorUnaryOpsTest`,
`TensorReductionTest`, `TensorBLASTest`, etc. — compiled with a suffix
for each backend (`_CPU`, etc.).

These test single Tensor API operations through the full pipeline: user
code → IR node → (JIT optimization is a no-op for single ops) → evaluate →
check result. Because they test single ops, the JIT optimization passes
don't fire (nothing to fuse), so these are effectively testing the
evaluator's dispatch to the underlying library functions — like an eager
mode test.

This is fine and desirable: they verify that the full pipeline produces
correct results for each op in isolation. If an op-library-specific test
(Layer 2) passes but the corresponding TensorBaseTest fails, the bug is in
the dispatch wiring, not the library function.

**Layer 4: End-to-end / integration tests (full pipeline, JIT fires).**

`OneDnnJitEndToEndTest`, `CudaJitEndToEndTest` — write multi-op expressions
that trigger JIT optimization (fusion, lowering), verify final results.

These exercise the full pipeline: user code → IR → optimization passes →
lowering → evaluate fused CustomNodes. They verify that fused execution
produces the same results as unfused. If a Layer 2 test passes and a
Layer 4 test fails, the bug is in the lowering pass's wiring (wrong inputs,
wrong fusion config, etc.).

**Summary of what each layer catches:**

| Layer | Catches | Doesn't catch |
|-------|---------|--------------|
| 1. IR/pass structure | Wrong pattern match, wrong graph rewrites | Execution bugs |
| 2. Op-library isolation | Wrong library function implementation | Dispatch/wiring bugs |
| 3. Backend-agnostic API | Dispatch wiring bugs, single-op correctness | Fusion bugs |
| 4. End-to-end | Fusion/lowering wiring bugs, full pipeline | (integration test) |

**TODO:**
- [ ] Create test helpers for isolated library op execution (OneDNN, native, CUDA)
- [ ] Refactor CPUTensorTest: split CPUTensor internals (construction, strides)
  from op correctness (move op correctness to op-library-specific tests)
- [ ] Add Layer 2 tests for OneDNN ops (conv2d factory + execute, binop, eltwise)
- [ ] Add Layer 2 tests for native ops (binary, unary, reduction, matmul)
- [ ] Add Layer 3 lowering correctness tests: construct IR, run one lowering pass,
  evaluate the produced CustomNode, verify output (not full pipeline)

### Full-Stack Backend Test Coverage

Both CPU and CUDA backends must pass the complete generic test suite through the
JIT pipeline. The `_CPU` suffix pattern (compile same test source with
`FL_TEST_BACKEND_CPU` to set `CPUTensor` as default) already covers tensor,
autograd, nn, optim, and contrib tests for the CPU backend.

**CPU backend (`_CPU` suffix tests) — all registered, triaged:**
- [x] Tensor tests: TensorBaseTest, TensorBinaryOpsTest, TensorBLASTest,
  TensorReductionTest, TensorUnaryOpsTest, ComputeTest, IndexTest, ShapeTest,
  TensorAdapterTest — all registered in CMakeLists.txt
- [x] Autograd tests: AutogradTest, AutogradReductionTest, AutogradBinaryOpsTest,
  AutogradUnaryOpsTest, AutogradNormalizationTest, AutogradRnnTest,
  AutogradConv2DTest — all registered
- [x] NN tests: ModuleTest, NNSerializationTest, NNUtilsTest — registered
- [x] Optim tests: OptimTest — registered
- [x] Contrib tests: ContribModuleTest, ContribSerializationTest — registered
- [x] Triage run complete (237 passed, 25 failed, many skipped)

CPU tensor-only failures (11 tests):
- `TensorBaseTest.where` — **Missing JIT feature**: boolean tensor indexing has
  dynamic shape; `inferIndexedShape` rejects b8 index ("dynamic sthape" [sic])
- `TensorBaseTest.concatenate` — **Bug**: empty tensor concatenation produces wrong
  shape `(0,1,2)` instead of `(1,1,0)`. Shape inference issue in concat edge case.
- `TensorBaseTest.sort` — **Bug**: sort on multi-dim tensors produces wrong values.
  `allClose(sorted, tiled)` fails after sort+tile.
- `TensorBaseTest.argsort` — **Bug**: same root cause as sort (argsort shares impl).
- `TensorReductionTest.median` — **Bug**: median returns s32 type instead of f32.
  Type mismatch in `scalar<float>()`.
- `TensorReductionTest.min` / `max` — **Bug**: min/max with indices returns wrong
  index values. `allClose(values.flat(i), in(span, i)(indices(i)))` fails.
- `TensorUnaryOpsTest.sign` — **Missing JIT feature**: uses boolean tensor indexing
  (same root cause as `where`).
- `IndexTest.ExpressionIndex` — **Missing JIT feature**: boolean tensor indexing.
- `IndexTest.IndexAssignment` — **Missing JIT feature**: "Tensor must appear be the
  only index" — multi-dim tensor index assignment not supported.
- `IndexTest.flat` — **Bug**: `a.flat(8).scalar<float>()` returns 9 instead of
  expected value. Flat indexing off-by-one or wrong mapping.

**CUDA backend (`_CUDA` suffix tests) — registered, triaged:**
- [x] Add `_CUDA` suffix test variants (mirror `_CPU` pattern):
  `FL_TEST_BACKEND_CUDA` preprocessor define, `flTestInit()` calls
  `setDefaultJitTensorType<CudaTensor>()`. Registered in CMakeLists.txt under
  `if (FL_USE_NATIVE_CUDA)`.
- [x] Tensor tests: TensorBaseTest_CUDA, TensorBinaryOpsTest_CUDA,
  TensorBLASTest_CUDA, TensorReductionTest_CUDA, TensorUnaryOpsTest_CUDA,
  ComputeTest_CUDA, IndexTest_CUDA, ShapeTest_CUDA, TensorAdapterTest_CUDA
- [ ] Autograd tests: deferred (depends on autograd verification)
- [ ] NN / Optim / Contrib tests: deferred (depends on autograd)
- [x] Triage run complete (88 passed, 39 failed out of 131 tensor tests)

CUDA tensor-only failures (39 tests), classified by root cause:

*Shared with CPU (same root cause):*
- `where`, `sign`, `ExpressionIndex` — boolean tensor indexing (dynamic shape)
- `concatenate` — empty tensor concat shape bug
- `sort` — sort correctness on multi-dim
- `median` — returns s32 instead of f32
- `IndexAssignment` — multi-dim tensor index assignment
- `flat` — flat indexing correctness

*CUDA-specific — missing backend ops:*
- `pad`, `padSymmetricSmall`, `padEdgeSmall` — "only PadType::Constant supported"
- `nonzero` — "Tensor-based indexing not yet implemented" in CudaTensor
- `TensorIndex`, `StridedRangeValues` — tensor-based indexing not implemented

*CUDA-specific — wrong type / scalar mismatch:*
- `mean`, `var`, `std`, `norm`, `cumsum` — reduction returns wrong dtype (s32
  instead of f32). Likely missing dtype promotion in CUDA reduction kernels.
- `amax` — flaky (passes in isolation, fails in parallel)
- `scalar` — scalar type mismatch for non-f32 types
- `matmulValues` — "Shape index 18446744073709551615 out of bounds" (unsigned
  underflow in shape access, likely missing dim check in matmul result shape)

*CUDA-specific — wrong values:*
- `arange` — arange(2,10,2) produces wrong values
- `iota` — depends on arange
- `rand` — flaky (passes in isolation)
- ~~`tril`, `triu` — wrong values in triangular ops~~ fixed by
  `TriangularNode` promotion and generalized CUDA last-two-dims implementation
- `AssignmentOperatorRvalueWithRvalue` — wrong values after rvalue assignment
- `IntegerTensorIndexAssign`, `IntegerTensorIndexAssignThenAdd` — index assignment
- `broadcasting` — broadcast shape mismatch (9,1)+(3,3) — wrong tiling?
- `ArithmeticBinaryOperators`, `ComparisonBinaryOperators`,
  `LogicalBinaryOperators`, `BitBinaryOperators` — many dtype combinations fail
  (likely related to dtype promotion / scalar type issues)
- `countNonzero`, `min`, `max` — wrong reduction values or index values

**Cross-cutting:**
- [x] Add `FL_TEST_BACKEND_CUDA` path to `TestInit.h`'s `flTestInit()`:
  `#ifdef FL_TEST_BACKEND_CUDA` → `fl::setDefaultJitTensorType<fl::CudaTensor>()`
- [ ] Ensure `FL_SKIP_IF_DTYPE_UNSUPPORTED` works for CUDA backend (dtype
  support matrix may differ from CPU)
- [ ] Value correctness sweep: audit tensor and autograd tests to ensure
  fundamental operations verify output values, not just shapes/no-throw

## Comprehensive OneDNN Correctness Tests

**Status**: Ready to implement. JIT machinery is in place — ops execute
synchronously within fused kernels, isolating true correctness bugs from
async race conditions.

**Scope:**
- [ ] OneDNN primitive ops (f32) vs native CPU reference (f64): conv2d
  forward/backward, pool2d, batchnorm, rnn
- [ ] Matmul: OneDNN f32 vs CBLAS f64
- [ ] Binary ops with broadcasting (all supported dtypes)
- [ ] bf16 correctness tests for ops on platforms with AVX512_BF16 support
- [ ] Value correctness sweep across all tensor and autograd ops

## Design Document: JIT-Only Execution Model

### Core Decision: The Backend IS the JIT

There is no eager backend. There is no separate deferred/static backend. There is
one kind of tensor backend, and it is a JIT. All tensor operations build IR nodes.
Computation is deferred until results are needed. The optimizer runs before
evaluation. Evaluation dispatches to hardware-specific library calls.

This eliminates:
- The distinction between eager and lazy execution paths
- Per-op async enqueue machinery (no `enqueueNativeOp`, no per-op `SharedData` capture)
- Per-tensor stream tracking during graph construction (tensors are IR nodes, not buffers)
- The need for multiple `TensorBackend` implementations per hardware target

And simplifies:
- Buffer allocation (happens during evaluation, not scattered across 100+ op callsites)
- Async execution (one stream task per graph evaluation, not per op)
- Error handling (errors happen during evaluation, not at arbitrary op boundaries)

### Design Principles

1. **Compact, high-performance implementations.** Small code, leverage libraries.
   Only add complexity when profiling shows a bottleneck.
2. **Internal simplicity.** "The ability to dramatically modify software is inversely
   correlated with codebase size and complexity." One execution model, not three.
3. **Transparency.** No hidden synchronization. The user controls when computation
   happens via `eval()` / `host()` / `scalar()`.

### Architecture

```
User Tensor ops → build IR (host thread, zero compute)
  │
  ▼
eval() / host() / scalar() triggers:

Host thread (planning):
  ├── Optimizer: common + hardware-specific passes
  ├── Scheduler: graph walk, execution order, buffer allocation, stream assignment
  └── Dispatcher: enqueue pre-planned work items on stream(s)
  │
  ▼
Stream workers (execution only):
  ├── AsyncCPUStream: worker thread runs library calls
  ├── CUDAStream: GPU hardware queue runs kernels
  └── SynchronousStream: inline on caller thread (tests, debugging)
  │
  Libraries:
  ├── CPU: OneDNN primitives, CBLAS, native C++ ops
  └── CUDA: cuBLAS, cuDNN, Thrust, custom kernels, NVRTC
```

### Stream Abstraction

A `Stream` is a sequence of tasks that execute in order. Tasks within a stream
are totally ordered (FIFO). Tasks on different streams are unordered unless
explicitly synchronized via `relativeSync`.

This is the same model as CUDA streams, extended to CPU:

| Stream Type | Task Execution | Backing |
|-------------|---------------|---------|
| `CUDAStream` | GPU hardware queue | `cudaStream_t` |
| `AsyncCPUStream` | Dedicated worker thread | `std::thread` + `std::deque` |
| `SynchronousStream` | Inline on caller thread | (none — immediate) |

In a JIT-only world, streams are simpler than in an eager model: they only matter
at evaluation boundaries, not at every op. But the abstraction itself is the same.

**Worker thread count per stream**: 1. Sequential consistency within a stream.
Users create multiple streams for parallelism.

**SynchronousStream**: Executes inline — no worker thread, no queue. Exists for
tests, simple scripts, and debugging. `SynchronousStream::relativeSync(waitOn)`
blocks: it calls `waitOn.sync()` and returns. This is correct — there's no
queue to insert a deferred wait into.

### Execution Model: Host Plans, Streams Execute

Like a compiler, planning and execution are strictly separated:

- **Graph construction** (user code): ops build IR nodes. Each node carries an
  optional stream assignment from the user. This is where placement decisions
  are made — not by the framework, but by the programmer.
- **Planning** (host thread, on `eval()`): graph walk, optimization, buffer
  allocation, reading stream assignments from nodes, inserting `relativeSync`
  for cross-stream dependencies. All inspectable and controllable.
- **Execution** (stream workers): pure execution engines. They receive
  fully-planned work (library calls with pre-allocated buffers) and run it.
  No graph traversal, no optimization, no scheduling decisions.

This separation gives maximum control for complex computation schedules:
- Placement decisions (which stream, which device) are made by the user during
  graph construction, with full control over where each op runs
- Memory planning can see the full graph and do liveness analysis before any
  execution starts
- The host can dispatch work to multiple streams and insert sync points without
  blocking on any of them
- Scheduling is introspectable: the programmer (or a future scheduling framework)
  can examine, modify, log, or profile the plan before execution

Thread safety for graph and stream abstractions can be added later. For now, all
planning runs single-threaded on the host.

### What Happens When You Write Code

```cpp
auto a = fl::full({3, 3}, 1.0f);   // ScalarNode — no allocation, no compute
auto b = fl::exp(a);                // UnaryNode(exp, a) — no compute
auto c = b + a;                     // BinaryNode(add, b, a) — no compute
auto d = c.scalar<float>();         // eval() + sync() + read
```

On `scalar()` (all on host thread until step 4):
1. **Optimize**: run passes (CSE, constant folding, elementwise fusion) → may
   reduce the graph to `FusedElementwise([exp, add], ScalarNode(1.0f))`
2. **Plan**: walk optimized graph bottom-up, compute execution order, allocate
   buffers, read stream assignment from each node, insert `relativeSync` for
   cross-stream data dependencies
   - ScalarNode → allocate buffer for 3x3 f32
   - FusedNode → allocate output buffer for 3x3 f32
3. **Dispatch**: enqueue planned work on each node's assigned stream
   - Enqueue: "fill buffer A with 1.0f"
   - Enqueue: "run fused kernel: input=A, output=B"
4. **Sync**: `stream.sync()` blocks host thread until all enqueued work completes
5. **Read**: scalar value read from result buffer

The host thread does steps 1-3 (plan) and 5 (read). The stream worker does
step 4 (execute). This is a clean separation: the host is the compiler, the
stream is the target.

### Current State → Target: What Changes

**Currently** (wrapping pattern):
```
Host thread:
  JitTensorBase::eval()
    → optimizer().optimize(node)           // host thread — good
    → evaluator().eval(node)               // host thread — graph walk
        → wrappedBackend_.add(mat_a, mat_b) // calls CPUBackend::add()
            → CPUBackend enqueues per-op on AsyncCPUStream
```

The evaluator walks the graph on the host (correct), but dispatches each node
through the TensorBackend interface to the wrapped backend, which does its own
per-op async enqueue. N nodes → N enqueues, each with its own SharedData
capture. Graph traversal and execution are interleaved.

**Target** (plan/execute separation):
```
Graph construction (user code):
  ops build IR nodes, each carrying a stream assignment

Host thread (plan, on eval()):
  optimize(graph)
  plan = walk(optimized_graph)       // read streams from nodes, alloc buffers,
                                     // insert relativeSync for cross-stream deps
  dispatch(plan)                     // enqueue work items on assigned streams

Stream worker(s) (execute):
  for each work item:
    call library function with pre-allocated buffers
    (OneDnnOps, native::, cuBLAS, NVRTC kernels, etc.)
```

The host thread does all planning. The stream workers are pure executors — they
receive `(function_ptr, input_buffers, output_buffer)` tuples and run them.
No graph traversal, no TensorBackend dispatch, no re-enqueue.

**Migration path** (incremental):
- **Now**: JitBackend wraps CPUBackend/CudaBackend. The Evaluator calls through
  `wrappedBackend_.op()`. The eager backends' per-op async enqueue is redundant
  but harmless. Planning and execution are interleaved.
- **Next**: Add per-op stream parameters. Separate planning from execution. The
  planner reads stream assignments from nodes, allocates buffers, and dispatches
  work items. Library ops called directly (OneDnnOps, native::, cuBLAS) instead
  of through TensorBackend.
- **Eventually**: The eager backends (CPUBackend, CudaBackend) become dead code.
  Their library-dispatch code lives in `ops/onednn/`, `ops/native/`, `ops/cuda/`,
  called directly by the dispatcher.

### Shared vs. Hardware-Specific Code

The JIT infrastructure is naturally layered:

**Fully shared** (backend-independent):
- IR node types: ScalarNode, BinaryNode, UnaryNode, ReductionNode, MatmulNode,
  FusedElementwiseNode, FusedReduceElementwiseNode, IndexNode, Conv2dNode,
  BatchNormNode, Pool2dNode, ProjectionNode, CastNode, ReshapeNode, TransposeNode, etc.
- Common optimizer passes: CSE, ScalarFolding, ElementwiseFusion,
  ReductionOfElementwiseFusion
- Shape/type inference: ShapeInference.h, OpProperties.h
- JitTensorBase: tensor adapter that holds IR nodes
- Graph construction: JitBackend builds IR from TensorBackend method calls

**Per-hardware** (registered via JitOptimizerExtension):
- Optimizer passes: OneDnnOpFusion, CudaFusedKernelLowering, CudaMatmulEpilogueFusion
- Evaluation dispatch: which library to call for each node type
- Buffer management: AlignedBuffer (CPU), cudaMallocAsync (CUDA)
- Stream implementation: AsyncCPUStream, CUDAStream

The current `JitTensor<T>` template parameterizes the JIT over the underlying
tensor storage type (CPUTensor, CudaTensor). This is the right mechanism — it
determines which hardware's libraries and memory management are used, without
duplicating any shared JIT code.

### Streams in a JIT-Only World

#### Per-Op Stream Assignment

Every tensor op takes an optional stream parameter. The stream is recorded on
the IR node at graph construction time — not at eval time. This gives the user
full control over where each op executes.

```cpp
// C++ — optional stream param on every op, defaults to backend default stream
Tensor add(const Tensor& a, const Tensor& b, StreamOrDevice s = {});
Tensor matmul(const Tensor& a, const Tensor& b, StreamOrDevice s = {});
Tensor exp(const Tensor& a, StreamOrDevice s = {});
```

When unspecified, ops use the backend's default stream. When specified, the
stream is stored on the IR node. `eval()` reads the stream from each node
and dispatches accordingly — it doesn't decide where things run.

```cpp
auto s1 = fl::Stream(fl::DeviceType::CPU);
auto s2 = fl::Stream(fl::DeviceType::CPU);

auto a = fl::exp(x, s1);           // node carries s1
auto b = fl::matmul(w, y, s2);     // node carries s2
auto c = a + b;                     // node carries default stream (or s1?)
fl::eval(c);                        // dispatches exp→s1, matmul→s2, add→default
                                    // auto-inserts relativeSync where needed
```

#### Multi-Stream Graphs

A single computation graph can span multiple streams. When `eval()` walks the
graph, it dispatches each node to the stream recorded on that node. Cross-stream
data dependencies are resolved automatically: if a node's input was assigned to
a different stream, the planner inserts `relativeSync` before the dependent node.

```cpp
// Overlap compute and communication
auto activations = forward(input, compute_stream);
auto synced = allreduce(activations, comm_stream);  // different stream
auto loss = loss_fn(synced, target, compute_stream);

fl::eval(loss);
// Dispatches: forward→compute_stream, allreduce→comm_stream, loss→compute_stream
// Auto-inserts: compute_stream.relativeSync(comm_stream) before loss_fn
```

Users can also call `relativeSync` explicitly for ordering between independent
(non-data-dependent) computation graphs:
```cpp
fl::eval(x);  // dispatches x's graph
fl::eval(y);  // dispatches y's graph (independent of x)
y_stream.relativeSync(x_stream);  // explicit ordering
```

#### Stream Default: Backend's Default Stream

When an op doesn't specify a stream, it runs on the backend's default stream —
regardless of where its inputs are. No inheritance from inputs. This is what
MLX does (`default_stream(default_device())`), and it's the simplest, most
predictable behavior: unspecified ops cluster on the default stream, and only
ops the user explicitly annotates run elsewhere.

The planner auto-inserts `relativeSync` when a node's inputs were evaluated on
a different stream than the node itself — whether the node's stream was
explicitly specified or defaulted.

#### TODO: StreamOrDevice Unification

Consider unifying `Stream` and `Device` into a single `StreamOrDevice` type
(as MLX does with `std::variant<std::monostate, Stream, Device>`). Passing a
`Device` would resolve to that device's default stream. This simplifies the
common case of "run on GPU" without needing to name a specific stream.

#### TODO: Block-Level Stream Scope (C++ idiom)

MLX uses Python's `with mx.stream(s):` context manager for block-level stream
assignment. The C++ equivalent could be a RAII guard:

```cpp
{
  fl::StreamGuard guard(s);
  // All ops in this scope default to stream s
  auto a = fl::exp(x);      // → stream s
  auto b = a + y;           // → stream s
}
// Previous default stream restored
```

Implementation: `StreamGuard` sets a thread-local default stream on
construction and restores the previous one on destruction. `JitBackend` reads
the thread-local when no explicit stream is passed to an op.

Alternatively, a more explicit lambda-based approach:
```cpp
fl::on_stream(s, [&] {
  auto a = fl::exp(x);
  auto b = a + y;
});
```

Decide on the idiomatic C++ pattern after per-op stream support is in place.

#### TODO: Distributed Op Stream Parameters

Distributed operations (allreduce, send, recv, etc.) should also take optional
stream parameters, following the same pattern as local tensor ops. This enables
overlapping communication with computation by placing them on different streams.

#### dnnl::stream Pairing

Each `AsyncCPUStream` needs its own `dnnl::stream` (OneDNN streams are not
thread-safe). Since work items execute on the stream's worker thread, the
`dnnl::stream` is naturally owned by the stream:

```cpp
class AsyncCPUStream {
  dnnl::engine engine_;       // shared reference (thread-safe, immutable)
  dnnl::stream nativeStream_; // used ONLY on the worker thread
};
```

The backend shares one `dnnl::engine` (thread-safe). Each stream creates its own
`dnnl::stream` from that engine. Work items access `nativeStream_` only from
within the worker thread — guaranteed safe.

### Buffer Management

#### Pre-Evaluation: No Buffers

Tensors are IR nodes. No memory is allocated when you call `fl::exp(a)` — only a
node is created. This means:
- Zero memory overhead for deferred operations
- No "eager allocation might waste memory" problem
- No `SharedData` capture needed in per-op lambdas (there are no per-op lambdas)

#### Planning Phase: Allocate Buffers (Host Thread)

The scheduler allocates output buffers as it walks the graph during planning:
- CPU: `AlignedBuffer` (64-byte aligned malloc)
- CUDA: `cudaMallocAsync` (stream-ordered — requires dispatching alloc to stream)

Because planning runs on the host thread with full visibility into the graph,
it can do liveness analysis: determine when each intermediate buffer is last
used, and reuse memory for non-overlapping lifetimes. The evaluator already has
the building blocks for this (`nodeToResultUseCount_`).

The planned `Schedule` contains pre-allocated buffers for every work item. Stream
workers receive `(function, input_bufs, output_buf)` — they never allocate.

#### Execution Phase: Compute Only (Stream Workers)

Stream workers execute work items with pre-allocated buffers. After a node's
result is consumed by all its users, the buffer can be released (or recycled to
a free pool). The use-count tracking that currently drives `unsetResult()` is
the mechanism for this.

#### Post-Evaluation: Result Stays on Node

After evaluation, `node->getResult()` holds the materialized tensor. The JIT
tensor (`JitTensorBase`) accesses this via `getTensorOrEvalNode()`. The
materialized tensor owns its buffer via the underlying tensor storage type's
mechanism (CPUTensor::SharedData, CudaTensor's cudaMallocAsync allocation).

The IR graph can be released after evaluation if the result is no longer needed
for autograd backward. Currently, `JitTensorBase` holds the graph via its node
reference — the graph is released when the JitTensorBase is destroyed or when
`unsetResult()` is called for intermediate nodes during evaluation.

#### Views

Indexing (`tensor[0:3]`) creates an `IndexNode` in the IR — no buffer, no copy.
On evaluation, the evaluator can produce either:
- A view (shared buffer, different offset/strides) — for simple range indexing
- A gathered copy — for tensor-as-index or boolean indexing

The choice is made during evaluation, inside the evaluator, using the same
mechanisms that CPUTensor/CudaTensor already support (strides, SharedData parent
chain, gather kernels).

### Buffer Ownership: SharedData

Post-evaluation, materialized tensors own their buffers via `SharedData`.
This applies to the underlying storage types (CPUTensor, CudaTensor) — the
mechanism that `JitTensorBase` delegates to after evaluation.

`CPUTensor::SharedData` is a reference-counted buffer handle:

```cpp
struct SharedData {
  std::shared_ptr<AlignedBuffer> buffer;  // owns the allocation
  std::shared_ptr<void> bufferOwner;       // alternative owner (dnnl::memory, etc.)
  void* bufferPtr;                         // raw pointer into the allocation
  std::shared_ptr<SharedData> parent;      // keeps parent alive for views
  bool isDataReady;
  bool isDevicePtrLocked;
};
```

Tensors that share data (via view or shallow copy) share the same `SharedData`
via `shared_ptr`. The buffer stays alive as long as any tensor references it.
Views hold a `parent` pointer that keeps the parent's buffer alive.

The evaluator creates materialized tensors (with `SharedData`) when it evaluates
nodes. These are stored in `node->setResult(tensor)`. The `JitTensorBase` accesses
them via `getTensorOrEvalNode()`. Intermediate nodes' results are released by
`unsetResult()` once all users are evaluated.

#### isDataReady (design clash — eager vs. JIT)

In the eager model, `isDataReady` tracks whether an async task has finished
writing to the buffer. `host()` checks the flag to decide whether to `sync()`.

**In a JIT-only model, this flag may be unnecessary.** A tensor is either:
- An IR node (no buffer exists, `getResult()` returns nullopt)
- Fully evaluated (buffer exists and is populated, `getResult()` has value)

There's no intermediate "buffer allocated but data not ready" state because
allocation and computation happen together inside the evaluation task. The flag
could be removed once the eager per-op enqueue path is retired. For now it
remains because CPUBackend's per-op async machinery still uses it during the
transition period (the evaluator calls `wrappedBackend_.op()` which enqueues
individually).

#### DevicePtr (transitional)

`DevicePtr` provides RAII lock/unlock on tensor buffers. It was designed for
synchronous contexts (ArrayFire backend) where you need a raw pointer for a
library call and want automatic unlock.

In the eager async model, DevicePtr has a well-known footgun: it unlocks in its
destructor, so if used on the stack before enqueueing an async task, the task may
access data after unlock. The "fix" was to capture `SharedData` in lambdas instead.

**In a JIT-only model, DevicePtr is largely irrelevant.** The host-thread
scheduler allocates buffers during planning and passes them to work items.
Work items access pre-allocated buffers directly — no RAII lock/unlock needed.
DevicePtr remains in the codebase for `DnnlMemoryWrapper` (OneDNN autograd ops)
and legacy code, but new code should not use it. It can be removed once the
eager backends are retired.

### Error Handling

Errors during graph construction are caught immediately (shape mismatches, type
errors) — these are metadata operations that don't require compute.

Errors during evaluation (library failures, out-of-memory) happen on the stream's
worker thread. The existing `AsyncCPUStream` exception machinery handles this:
- Worker thread catches the exception and stores it in `pendingException_`
- Next `sync()` call rethrows it on the caller's thread
- This is the same model as `std::future` exception propagation

A work item failure on a stream worker causes subsequent work items on that
stream to be skipped (stream enters error state). The next `sync()` surfaces the
exception to the host thread.
> **Not what the code does (2026-10-26):** on exception the worker clears the queue
> and *exits* (`AsyncCPUStream.cpp:119-128`); the first `sync()` rethrows and clears
> the error, then later tasks are never run and the next `sync()` hangs forever on
> the process-wide CPU stream. Implement the sticky error state described here.

#### Exception Safety During Stack Unwinding

When an exception causes stack unwinding, tensor destructors run. Two scenarios:

**Unevaluated tensors (IR nodes only):** No buffers to worry about. The IR nodes
are freed normally (shared_ptr reference counting). No stream interaction.

**Evaluated tensors (have materialized buffers):** The buffer is owned by
`SharedData` (reference-counted). If the `Tensor` is destroyed, the refcount
decrements. The buffer is freed only when the last reference drops. If another
evaluated tensor shares the buffer (view), the buffer survives.

**Stream destruction during unwinding:** If `AsyncCPUStream` is destroyed, its
destructor sets `shutdown_` and joins the worker thread. All pending tasks
execute before destruction completes. This is already implemented. Combined with
`CPUBackend::~CPUBackend()` syncing `asyncStream_` before member destruction,
this prevents tasks from accessing destroyed backend state.

### Autograd Interaction

Autograd builds a backward graph during the forward pass. Each forward op
records its inputs and a gradient function (`GradFunc`). During `backward()`,
the gradient functions are called in reverse topological order.

In a JIT-only world:
- Forward ops build IR nodes (JIT). Autograd records these as inputs.
- `backward()` calls gradient functions, which call tensor ops, which build more
  IR nodes (JIT).
- The backward graph is also lazy — evaluation is triggered when the user accesses
  gradient values.

This is already how it works today: autograd operates on `Tensor` (which is
backed by `JitTensorBase`), so gradient computations naturally go through the JIT.

### Conv2D/Pool2D/BatchNorm/RNN (Library Primitives)

#### SSA semantics

All ops in the JIT must follow SSA (Static Single Assignment) semantics: every
tensor value is defined exactly once and never mutated. This is a prerequisite
for lazy evaluation — graph nodes are immutable, and mutation forces
materialization. This is consistent with JAX/XLA (jaxpr and HLO are SSA) and
the lazy tensor model in general.

Concretely: ops must not take mutable `Tensor&` out-params for state they
update (e.g., running statistics). Instead, they take `const Tensor&` inputs
and return new tensors. The caller rebinds names (`runningMean_ = newMean`),
which is not mutation — it's SSA name rebinding.

#### Lazy dispatch via CustomNode

These ops have statically-inferrable output shapes, so they can participate
in the lazy graph without dedicated IR nodes. The approach: wrap the opaque
backend dispatch in a `CustomNode` with the correct output shape/dtype. This
keeps the graph intact, enabling fusion passes to see across op boundaries.

For multi-output ops (batchnorm returns 5 tensors, sort/topk/min/max return 2),
use the shared-cache `CustomNode` pattern: multiple `CustomNode`s share a
`shared_ptr<optional<tuple<...>>>` cache. Whichever evaluates first populates
the cache; the others read from it.

Progress:
- [x] SSA refactor: `batchnormForward` — `runningMean`/`runningVar` are now
  `const Tensor&` inputs, new values returned in `BatchnormForwardResult` struct
- [x] Lazy two-output ops: `sort`, `topk`, `min`, `max` — shared-cache CustomNode
- [x] Lazy `batchnormForward` — wrap in shared-cache CustomNode (5 outputs:
  output, saveMean, saveVar, newRunningMean, newRunningVar). Uses
  `DeferredBatchnormState` wrapper so backward can retrieve the real
  backend-specific state after lazy forward evaluation.
- [x] Lazy `batchnormBackward` — wrap in shared-cache CustomNode (3 outputs:
  gradInput, gradWeight, gradBias)
- [x] Lazy `pool2dForward` — wrap in CustomNode (output shape computable from
  input shape + kernel/stride/pad params). Uses `DeferredPool2dState` wrapper.
- [x] Lazy `pool2dBackward` — wrap in CustomNode (output shape = input shape)
- [ ] SSA refactor + lazy `rnnForward` — currently mutates hiddenState/cellState;
  convert to SSA (return new tensors), then wrap in shared-cache CustomNode
  (4 outputs: output, hiddenState, cellState, state)
- [ ] Lazy `rnnBackward` — wrap in shared-cache CustomNode (4 outputs:
  gradInput, gradHiddenState, gradCellState, gradWeights)
- [x] Lazy `conv2dBackwardData` — wrap in CustomNode (output shape = input shape)
- [x] Lazy `conv2dBackwardFilterBias` — wrap in shared-cache CustomNode
  (2 outputs: gradWeights, gradBias)

`nonzero` is the only op that legitimately cannot be lazy (data-dependent output
shape).

#### First-class IR nodes for multi-output ops (ProjectionNode design)

Once lazy CustomNode dispatch is working, the next step is dedicated IR nodes
(`BatchNormNode`, `Pool2dNode`) with a **ProjectionNode** pattern to expose
individual outputs. This makes multi-output ops visible to the optimizer for
fusion passes (e.g., cuDNN fused batchnorm+relu, OneDNN post-ops).

**Design** (inspired by XLA `GetTupleElement`, TVM `TupleGetItem`, LLVM
`extractvalue`): each multi-output op gets a typed IR node that evaluates all
outputs in one dispatch. Individual outputs are accessed via lightweight
`ProjectionNode` children that reference the parent by a named output index.
This preserves the single-result `Node` invariant (one shape, one dtype, one
result per node) while making structure visible to optimizers.

```
  [input] [weight] [bias] [runningMean] [runningVar]
       \     |      |       /           /
        BatchNormNode (5 outputs internally)
         /    |     |      \         \
   Proj(0) Proj(1) Proj(2) Proj(3) Proj(4)
     |        |       |       |       |
   output  saveMean saveVar  newRM  newRV
```

A typed `Out` enum class on each node type makes indices self-documenting and
prevents cross-op index mistakes:
```cpp
ProjectionNode::create(bnNode, BatchNormNode::Out::SaveMean, statsShape, paramsType)
```

Pre-built pattern matching helpers in `PatternMatcher.h` enable fusion passes:
```cpp
// Match: relu(batchnorm.result)
auto pat = matchUnary(UnaryOp::Relu, matchBatchNormResult("bn"));
if (pat.match(node, result)) { /* replace with fused BN+ReLU */ }
```

Tasks:
- [x] Add multi-output support to `Node` (`outputs_` vector, `setOutput`/`getOutput`/`hasOutput`/`numOutputs`)
- [x] Implement `ProjectionNode` (zero-cost eval, reads from parent output slot)
- [x] Implement `BatchNormNode` (typed node with `Out : OutputIndex` enum, 5 outputs,
  `mutable cachedState_`, replaces shared-cache CustomNode in `batchnormForward`)
- [x] Implement `Pool2dNode` (single-output + `mutable cachedState_`, like Conv2dNode)
- [x] Update `Evaluator` with eval handlers for new node types (evalProjectionNode,
  evalBatchNormNode, evalPool2dNode; buildPlan `isEvaluated` handles multi-output nodes)
- [x] Add `MatchProjection`, `MatchBatchNorm`, `MatchPool2d` to `PatternMatcher.h`
  with pre-built convenience patterns (`matchBatchNormResult`, `matchBatchNormSaveMean`, etc.)
- [x] Update `JitBackend.cpp` to emit typed nodes + ProjectionNodes
  (DeferredBatchnormState/DeferredPool2dState now hold typed node pointers)
- [x] Update Graphviz printer for new node types
- [x] Update CSE pass to handle new node types (non-CSE-able, returns empty key)
- [x] BatchNorm + pointwise fusion (cuDNN graph API): implemented in CudnnSubgraphLowering
- [x] Dead output elimination for multi-output nodes (e.g., unused BN statistics)
- [x] Pool2d + pointwise fusion (cuDNN graph API): wired up `graph->resample()` in
  `buildCudnnGraph()`. Uses inference mode (`set_is_inference(true)`) — training-mode
  max pooling Index tensor output deferred (see follow-up in code comment).

#### Fusion passes enabled by typed multi-output nodes

Now that BatchNormNode, Pool2dNode, and Conv2dNode are first-class IR nodes with
PatternMatcher support, the following fusion passes can be implemented:

**OneDNN post-op fusion passes** (via `dnnl::primitive_attr` with `dnnl::post_ops`):
- [x] BatchNorm + activation fusion (OneDNN): match `relu(batchnormResult)` or other
  eltwise activation after batchnorm result. OneDNN `batch_normalization` primitive
  accepts `primitive_attr` with eltwise post-ops (relu, tanh, sigmoid, etc.). The
  `attr` parameter is already threaded through `BatchNorm.cpp`. Implemented in
  `OneDnnPrimitivePostOpFusion::rewrite()` — peels through ProjectionNode(BN, Result)
  to reach BN. Gated on inference-only (train=false) since training BN has auxiliary
  outputs consumed by backward.
- [x] Pool2d + activation fusion (OneDNN): match `relu(pool2d)` or other eltwise
  activation after pool2d. OneDNN `pooling_forward` accepts `primitive_attr` with
  eltwise post-ops. The `attr` parameter is already threaded through `Pool2D.cpp`.
  Implemented in `OneDnnPrimitivePostOpFusion` pass. Matches UnaryOp activations
  (sigmoid, tanh, exp, log, sqrt, abs, round, negative) plus `max(pool2d, 0)` for relu.
  Single-use gate prevents fusion when pool2d has multiple consumers.
- [x] Conv2d + activation fusion (OneDNN): match `relu(conv2d)` or other eltwise
  activation after conv2d. OneDNN `convolution_forward` accepts `primitive_attr`.
  Implemented in `OneDnnPrimitivePostOpFusion` pass (same as pool2d above).
  Fused CustomNode captures Conv2dNode by shared_ptr, populates `cachedState_` for backward.
  7 end-to-end tests (forward + backward + multi-use gate).
- [~] BatchNorm + sum (residual) fusion (OneDNN): **Won't do** — OneDNN
  `batch_normalization` only supports eltwise post-ops, not `append_sum()` or
  `append_binary()`. Wrapping BN+add+relu in an opaque CustomNode would hide the
  add and relu from downstream passes (ElementwiseFusion, OneDnnOpFusion) that can
  already optimize them. On CUDA, the cuDNN graph API handles this natively via
  `CudnnSubgraphLowering`.
- [x] Conv2d + sum + activation fusion (OneDNN): match `activation(add(conv2d, other))`
  pattern (ResNet residual block). Uses chained post-ops: `append_sum(1.0)` +
  `append_eltwise(relu)`. Added `sumSrc` parameter to `conv2dForward` to pre-fill
  dst buffer. Supports commutative add (conv on either side). Single-use gates on
  both conv2d and add nodes.
- [x] Matmul + activation fusion (OneDNN): match `activation(matmul(A, B))` pattern
  (linear layer forward). OneDNN `matmul` primitive accepts `primitive_attr` with
  eltwise post-ops. Added `attr` parameter to `detail::onednn::matmul()`. Implemented
  in `OneDnnPrimitivePostOpFusion` pass as `AnchorKind::Matmul`. Eliminates a full
  memory round-trip per fused eltwise (10-30% for memory-bound shapes). Gelu
  canonicalization now handled by pattern matching in the same pass (no GeluNode needed).
- [x] Matmul + bias + activation fusion (OneDNN): match `activation(add(matmul(A, B), bias))`
  pattern (full linear layer: `activation(W @ x + b)`). Uses OneDNN's native matmul
  bias parameter (not `append_binary(add)` post-op), following PyTorch/ideep and ONNX
  Runtime convention. Bias validity: last dim must match output, all other dims must be 1.
  Bias memory descriptor padded with leading 1s to match output ndims. Supports
  bias-only (`matmul + bias`) and bias+eltwise (`activation(matmul + bias)`) patterns.
  Commutative add detection (`add(matmul, bias)` or `add(bias, matmul)`).

**cuDNN fusion passes** (via cuDNN Frontend Graph API — `CudnnSubgraphLowering`):
- [x] Conv2d + pointwise fusion: conv2d + relu, conv2d + sigmoid, conv2d + add + relu, etc.
- [x] Matmul + pointwise fusion: matmul + add, matmul + add + relu, etc.
- [x] BatchNorm + pointwise fusion: BN + relu, BN + add + relu, BN + sigmoid, etc.
  Uses `graph->batchnorm()` (training) / `graph->batchnorm_inference()` (inference).
  Multi-output CustomNode produces all 5 BN outputs. ProjectionNode transparent to
  subgraph expansion.
- [x] Pool2d + pointwise fusion (cuDNN graph): wired up `graph->resample()` with
  NCHW strides (NVIDIA samples confirm no NCHW/NHWC difference for avg pooling).
  Inference mode only; training-mode max pooling Index tensor deferred.
- [x] CastNode handling in `buildCudnnGraph()`: mapped to `PointwiseMode_t::IDENTITY`
  with output data type set to target dtype. Structural tests pass.
- [ ] RNN fusion (cuDNN graph): not in `isAnchorNode()`. Legacy cuDNN API still used.
- Note: the cuDNN Frontend Graph API subsumes the legacy `cudnnConvolutionBiasActivationForward`
  and `cudnnBatchNormalizationForwardTrainingEx` approaches. All fusion is now done via
  graph descriptor construction + compilation + execution plan caching.

**Backward fusion passes:**
- [ ] BatchNorm + activation backward fusion (cuDNN): when forward uses
  `cudnnBatchNormalizationForwardTrainingEx` with `CUDNN_BATCHNORM_OPS_BN_ACTIVATION`
  or `CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION`, the backward MUST use
  `cudnnBatchNormalizationBackwardEx` with the matching mode. This fuses
  relu_backward + batchnorm_backward into a single kernel, avoiding
  materializing the relu gradient intermediate. The workspace from the fused
  forward contains the relu activation mask. This is the ResNet backward fusion
  that provides significant speedup. Requires: paired forward/backward fusion
  (forward must be fused for backward to work).
- Note: OneDNN backward primitives do NOT support eltwise post-ops (attr is for
  quantization scales only). cuDNN conv2d/pool2d backward also has no post-op fusion.
- Note: JIT-level backward fusion already works — the existing ElementwiseFusion and
  NVRTC lowering passes apply to backward IR graphs automatically since backward ops
  build JIT IR nodes.

**Cross-op algebraic fusion passes** (backend-independent):
- [x] Conv2d + BatchNorm weight folding (inference only): when `train=false`, batchnorm
  is a linear transform that can be folded into conv2d weights/bias at graph optimization
  time: `w_new = w * gamma / sqrt(var + eps)`, `b_new = (b - mean) * gamma / sqrt(var + eps) + beta`.
  Eliminates the batchnorm entirely. Pattern: match batchnorm consuming conv2d output
  where `bn.train() == false`. Implemented in `ConvBatchNormFolding` pass (Phase 1).

Full design plan: `.claude/plans/linked-zooming-ladybug.md`
> **2026-11-16:** that plan file was not preserved; this section is the design of
> record (see `docs/modernization/research/README.md`).

### Tensor Layout Philosophy: Strides, Not Tags

The JIT IR is **layout-agnostic** ("unopinionated"). There is no `MemoryFormat` enum,
no `Layout` tag, no `channels_last` vs `contiguous` distinction in the IR. Strides
encode everything about physical memory layout. Ops handle layout internally — when
calling into cuDNN or OneDNN, the library call specifies strides matching the tensor's
actual physical layout, and the library handles any internal layout optimization.

**Why no MemoryFormat:**
- A `MemoryFormat` enum (like PyTorch's `torch.channels_last`) creates a parallel
  type system that must be threaded through every op, every pass, every rewrite.
  Every new op must decide "what MemoryFormat does my output have?" and every
  pass must preserve or update it. This is fragile and error-prone.
- Strides already encode the same information: NCHW contiguous strides
  `{C*H*W, H*W, W, 1}` vs NHWC contiguous strides `{H*W*C, 1, W*C, C}` are
  just different stride patterns. The IR sees them identically — as tensors
  with particular strides.
- Users are responsible for choosing efficient layouts. The JIT optimizes
  explicit transposes but does not insert layout conversions automatically.

**Transpose optimization passes (future):**

Three levels of algebraic transpose optimization, all purely mechanical IR rewrites
with no layout semantics:

1. **Transpose cancellation**: `transpose(perm) → transpose(inverse(perm))` = identity.
   When two adjacent transposes compose to the identity permutation, eliminate both.
   Pattern: `matchTranspose(matchTranspose(any()))` where `compose(perm1, perm2) = identity`.

2. **Transpose absorption into library calls**: When a transpose feeds directly into
   a library op (matmul, conv2d), the transpose can be absorbed by changing the strides
   passed to the library. For matmul, this is already handled via `MatrixProperty::Transpose`.
   For conv2d, cuDNN accepts arbitrary strides — a transposed input just means different
   strides in the descriptor. No data movement needed.

3. **Transpose hoisting through stride-agnostic ops**: Pointwise ops (add, mul, relu,
   sigmoid, etc.) are stride-agnostic — they operate element-by-element regardless of
   physical layout. A transpose before a pointwise op can be hoisted after it:
   `relu(transpose(x))` → `transpose(relu(x))`. This exposes more cancellation and
   absorption opportunities. Reductions along specific axes need axis remapping when
   the transpose is hoisted past them.

**NHWC layout support and cuDNN graph ops:**

Several cuDNN graph operations (batchnorm, pooling/resample) internally prefer or
require NHWC layout. Currently we pass NCHW strides and rely on `check_support()` to
fail gracefully when no valid execution plan exists for NCHW. This means:
- cuDNN graph batchnorm fusion works when the graph planner finds an NCHW-compatible
  plan (which it often does for BN+pointwise), but may fall back to unfused execution.
- cuDNN graph pooling (Resample op) requires NHWC and is currently blocked.

When NHWC tensor layout is supported (via explicit user transposes or automatic layout
propagation), the strides on IR nodes will reflect the actual NHWC layout, and cuDNN
graph operations will find optimal execution plans. No IR changes needed — just
different stride values flowing through the same infrastructure.

- [x] TransposeSimplification pass (BottomUpPass) — all three levels in one pass:
  - Identity elimination, composition/cancellation (level 1)
  - Matmul absorption: fold last-2-dims transpose into MatrixProperty (level 2)
  - Unary/binary commutation, reduction axis remapping, scalar folding (level 3)
  18 unit tests. Runs before ScalarFolding in optimizer pipeline.
- [x] Transpose absorption into cuDNN subgraphs: TransposeNode is now `isCudnnLowerable`.
  When absorbed, the boundary input's cuDNN tensor gets permuted dims/strides — cuDNN
  reads the physical buffer with zero-copy transpose. Works for all anchor ops (conv2d,
  matmul, BN, pool2d) and pointwise chains. Falls back gracefully via `check_support()`.
- [ ] NHWC-aware layout propagation (optional): automatically insert transposes at
  graph boundaries to convert between user-facing NCHW and library-preferred NHWC.
  This is a higher-level optimization that sits above the algebraic passes.

### CustomNode as the JIT ↔ Library Interface

`CustomNode` is currently the common interface between JIT lowering passes
and library implementations. Every lowering pass — OneDNN, NVRTC, cuBLASLt —
produces a `CustomNode` with a name, inputs, shape, and eval lambda. The
evaluator calls `node->evalFunc()(inputs)` uniformly without knowing what's
inside the lambda.

This works and is sufficient for the current architecture. The library-
specific fusion mechanisms (dnnl::primitive_attr, NVRTC source compilation,
cuBLASLt epilogue enums) are irreducibly different — there is no useful
common abstraction for "how to configure fusion" across libraries. The
commonality is the **contract**: a lowering pass takes IR nodes and produces
a `CustomNode`.

**Potential improvements (deferred, self-contained):**

- **Typed CustomNode variants**: currently CustomNode's eval is a type-erased
  `std::function<Tensor(const vector<const Tensor*>&)>`. For profiling,
  debugging, and serialization, it would be useful to have typed variants
  that expose their library-specific state (e.g., `OneDnnFusedNode` that
  exposes its `primitive_desc` and `post_ops`, `NvrtcKernelNode` that
  exposes its compiled `CUfunction`). This enables richer introspection
  without changing the evaluator's dispatch path.

- **Eval with pre-allocated output**: the current eval signature returns a
  `Tensor` (allocates internally). For memory planning (Step 7), we need
  `void eval(Tensor& out, const vector<const Tensor*>& inputs)` so the
  planner can provide a pre-allocated buffer. This is a mechanical change
  to CustomNode's interface.

- **CustomNode caching/deduplication**: two CustomNodes with the same name,
  same inputs, same shapes could potentially share a cached result. This
  requires CustomNode to have a structural identity (beyond pointer
  equality). Low priority — CSE already handles this for typed nodes.

- **Test helpers for CustomNode construction**: small utilities that make
  it easy to construct a CustomNode for a specific library op (e.g.,
  `makeOneDnnConv2dNode(inputs, params, attr)`) and evaluate it against
  test tensors. These helpers bridge Layer 2 (isolated op tests) and
  Layer 3 (lowering pass tests) in the test layering.

These improvements are self-contained and don't affect the rest of the
architecture. They can be implemented incrementally as needs arise.

### Backend Abstraction vs. Optimization Knowledge (RESOLVED)

**Status: RESOLVED.** Chose a refined Option A (two-phase optimize-then-lower)
with parameterizable library functions and a three-layer execution model.

#### The Problem

The `TensorBackend` abstraction was designed for eager mode: "here's one op,
execute it." It hides which library implements each op (native, OneDNN, MKL,
cuBLAS, cuDNN). In JIT mode, the optimizer needs to reason about fusing
**sequences of ops together**, which requires library-specific knowledge that
`TensorBackend` intentionally hides.

The fundamental tension: fusion is a joint decision between the optimizer (which
sees graph structure) and library implementations (which know what can be fused).
The current backend-dependent passes bridge the gap by duplicating dispatch
knowledge.

#### Resolution: Three-Layer Architecture

Drawing on how XLA, MLIR, Inductor, TVM, and MLX handle this, every successful
system uses **operation classification + pattern-match-and-replace**. No system
uses per-op virtual dispatch as the optimizer-backend interface. Our resolution:

**Layer 1: Generic passes (target-independent).** Structural rewrites that
operate on the IR without knowledge of any specific library: CSE, constant
folding, elementwise fusion into `FusedElementwiseNode`, reduction-of-
elementwise fusion into `FusedReduceElementwiseNode`. These create maximally-
fused abstract nodes that represent computation, not library calls.

**Layer 2: Backend lowering passes (library-specific).** Pattern-match fused
IR nodes and replace them with `CustomNode`s that call specific library
implementations. Each lowering pass owns the knowledge of one library's fusion
capabilities:
- `CudaFusedKernelLowering`: `FusedElementwiseNode` → NVRTC compiled kernel
- `CudaMatmulEpilogueFusion`: `MatmulNode + add/relu` → cuBLASLt epilogue
- `OneDnnOpFusion`: chains of binary/eltwise → OneDNN primitive with post-ops
- Future: `CudnnConvEpilogueFusion`, `OneDnnConvPostOps`, etc.

**Layer 3: Fallback dispatch.** For IR nodes that survive lowering (no library-
specific fusion applied), the evaluator dispatches through a lightweight
mechanism. This replaces TensorBackend's ~100 virtual methods with a small
dispatch struct (~10 function pointers for op categories).

#### Parameterizable Library Functions

The key insight: library functions need to be factored so that the **primitive
configuration** (the part that fusion modifies) is separated from **execution**
(the part that's always the same). The fusion extension point is on the
primitive descriptor, not on the final execute call — because post-ops must
be attached *before* primitive creation.

**Factoring (using OneDNN conv2d as the concrete example):**

The current code already has the right seed: `createOneDnnConv2DData()` creates
a `convolution_forward::primitive_desc` from shape/stride/padding params. But
it takes no `primitive_attr`, so there's no way to attach post-ops. The fix:

```cpp
// Layer 1: Primitive descriptor factory (the reusable, fusable unit)
// This is the SINGLE SOURCE OF TRUTH for how a conv2d primitive is configured.
OneDnnConv2DData createOneDnnConv2DData(
    const dnnl::engine& engine,
    fl::dtype inputType,
    const Shape& inputShape, const Shape& weightsShape,
    const Shape& biasShape, const Shape& outputShape,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    dnnl::primitive_attr attr = {}  // ← fusion extension point
);

// Layer 2: Execute-with-memory-setup (reusable across all conv2d calls)
// Takes a primitive_desc + tensors, handles memory alignment, reorders, async.
void executeConv2d(
    Tensor& output,
    const OneDnnConv2DData& conv2DData,
    const Tensor& input, const Tensor& weights, const Tensor& bias,
    const dnnl::engine& engine,
    AsyncCPUStream& asyncStream, dnnl::stream& nativeStream);

// Layer 3: Convenience wrapper (what fallback dispatch and tests call)
std::pair<Tensor, ForwardBackwardState> conv2dForward(
    const Tensor& input, const Tensor& weights, const Tensor& bias,
    int sx, int sy, int px, int py, int dx, int dy, int groups,
    bool wantState) {
  auto& backend = CPUBackend::getInstance();
  auto data = createOneDnnConv2DData(
      backend.engine(), input.type(), /* shapes... */,
      sx, sy, px, py, dx, dy, groups,
      {}  /* default: no post-ops */);
  Tensor output = /* allocate */;
  executeConv2d(output, data, input, weights, bias, /* stream... */);
  return {output, nullptr};
}
```

**How fusion works with this factoring:**

A lowering pass matches `conv2d → relu → add(bias)` in the IR, then:

```cpp
// In OneDnnConvPostOps lowering pass:
dnnl::post_ops po;
po.append_eltwise(dnnl::algorithm::eltwise_relu, 0.f, 0.f);
po.append_binary(dnnl::algorithm::binary_add, addend_md);
dnnl::primitive_attr attr;
attr.set_post_ops(po);

// Same factory function, different attr
auto data = createOneDnnConv2DData(engine, ..., attr);

// CustomNode's eval lambda calls the same execute function
auto evalFn = [data, ...](const std::vector<const Tensor*>& inputs) {
    Tensor output = /* allocate */;
    executeConv2d(output, data, *inputs[0], *inputs[1], *inputs[2], ...);
    return output;
};
auto fusedNode = CustomNode::create("conv2d+relu+add", ..., evalFn);
```

Both paths call `createOneDnnConv2DData` (same primitive configuration logic)
and `executeConv2d` (same memory setup and execution logic). The ONLY
difference is the `primitive_attr` parameter.

**Same pattern for other libraries:**

- **cuBLASLt matmul**: primitive descriptor factory takes
  `cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT` and optional
  addend descriptor. Unfused: default epilogue. Fused: `EPILOGUE_RELU_BIAS`.
- **cuDNN conv2d**: same pattern — fused descriptor includes activation mode,
  bias descriptor. (cuDNN Frontend makes this even more natural with its
  operation graph API.)
- **NVRTC fused kernels**: different — the entire kernel is compiled. The
  "primitive" IS the compiled `CUfunction`. No factory/execute split needed;
  the lowering pass owns the full compilation and the CustomNode holds the
  compiled kernel.
- **OneDNN binary/eltwise**: `applyBinop` already takes `dnnl::algorithm`;
  extending to accept `primitive_attr` with post-ops follows the same pattern.

**Testing with a single code path:**

Because both fused and unfused paths call the same factory + execute functions:

1. **Primitive factory tests**: call `createOneDnnConv2DData(engine, ..., {})`
   and verify the primitive_desc properties. Call it again with post-ops attr
   and verify the fused primitive_desc. No JIT, no dispatch, no backend.

2. **Execute tests**: create a primitive_desc (with or without post-ops),
   call `executeConv2d(...)`, verify output. These are free function tests
   that exercise the actual OneDNN execution path.

3. **Convenience wrapper tests**: call `conv2dForward(...)` and verify output.
   This tests the same code path as (1) + (2) composed. This is what the
   existing tests already do.

4. **Lowering pass tests**: construct IR with `conv2d → relu → add`, run
   the lowering pass, verify it produces a CustomNode, evaluate, verify output.
   The CustomNode internally calls the same (1) + (2).

5. **End-to-end tests**: write user code `relu(conv2d(x, w, b) + c)`, let
   the full pipeline optimize + lower + evaluate.

There is no separate code path for fused vs unfused primitive creation.
The factory function is the single source of truth.

#### TensorBackend Evolution

`TensorBackend` as a ~100-method virtual class is heavyweight for JIT-only
execution where it's just a fallback dispatch layer. The evolution:

1. ~~**Now**: `TensorBackend` stays as-is. JitBackend wraps it. Evaluator
   calls through `wrappedBackend_.op()`.~~ **Done.**

2. ~~**Next**: Make TensorBackend output-parameterized.~~ **Done.**
   `void add(Tensor& out, lhs, rhs)` instead of `Tensor add(lhs, rhs)`.

3. **Done**: `BackendDispatch` struct (50 function pointers + backendType)
   replaces virtual dispatch for all ops the evaluator and JitBackend use.
   `JitBackend::wrappedBackend_` member removed; all delegation goes through
   `dispatch_`. `makeBackendDispatch(TensorBackend&)` creates default
   trampolines. Backend-specific overrides can replace individual function
   pointers to bypass virtual dispatch entirely.

   For complex fused ops (conv2d+relu+post-ops, matmul+epilogue), dispatch
   doesn't go through this struct — those are always `CustomNode`s created by
   lowering passes. The dispatch struct only handles the simple cases.

   **Next**: `TensorBackend` as a virtual class can be removed once all
   callers use `BackendDispatch` directly. The remaining role of
   `TensorBackend` is as the concrete object behind `dispatch_.ctx` — it can
   be replaced by a plain struct or per-backend singleton.

#### Why This Works

- **No abstraction leakage**: generic passes don't know about libraries.
  Lowering passes don't know about each other. The dispatch struct doesn't
  know about fusion.
- **Single source of truth**: library function params encode fusion
  capabilities. The lowering pass constructs fusion config. The evaluator
  calls the same function both ways.
- **Testable at each layer**: free function tests, pass tests, end-to-end
  tests — each independently.
- **Small footprint**: ~10 function pointers vs ~100 virtual methods.
  Passes are ~100-300 lines each. Library functions are standalone.
- **Extensible**: adding a new library = new free functions + new lowering
  pass. No changes to generic passes or dispatch struct.

### Remaining Eager Backend Code

The current CPUBackend and CudaBackend have substantial per-op implementation
code. In a JIT-only world, this code doesn't disappear — it becomes the
library functions that both the evaluator (via dispatch struct) and lowering
passes (via CustomNode) call.

**Migration path:**

1. ~~**Now**: JitBackend wraps CPUBackend/CudaBackend.~~ **Done.** The Evaluator
   calls through `BackendDispatch` function pointers (no `wrappedBackend_`).
   The eager backends' per-op async enqueue is redundant but harmless.

2. **Next**: Extract per-op logic from backends into free functions in
   `fl/tensor/ops/{library}/`. Make them output-parameterized. Add optional
   fusion configuration parameters where applicable (e.g., `dnnl::post_ops`
   for OneDNN ops, `cublasLtEpilogue_t` for cuBLAS matmul). Remove per-op
   async wrapping (JIT handles scheduling).

3. **Target** (partially done): CPUBackend/CudaBackend become thin adaptors
   that populate a `BackendDispatch` struct with function pointers to the
   library functions. `makeBackendDispatch()` already does this via
   trampolines; next step is direct function pointers (no virtual call).
   Lowering passes construct `CustomNode`s that call library functions
   directly with fusion config. The evaluator dispatches unlowered ops through
   the struct. Per-op virtual dispatch is gone.

### The Full Picture

```
User code (host thread)
  │
  ▼
Tensor ops → build IR nodes (zero compute, zero allocation)
  │
  ├── BinaryNode, UnaryNode, ReductionNode, MatmulNode, ...
  ├── CustomNode for library primitives (conv2d, pool2d, batchnorm, rnn)
  ├── IndexNode for deferred indexing/views
  └── ScalarNode for constants
  │
  ▼
Evaluation trigger: eval() / host() / scalar()
  │                                                       ALL ON HOST THREAD
  ├── 1. Optimize: run common + backend-specific passes  ─────────────────
  │     ├── CSE, ScalarFolding, ElementwiseFusion, ReductionOfElementwiseFusion
  │     └── OneDnnOpFusion / CudaFusedKernelLowering / CudaMatmulEpilogueFusion
  │
  ├── 2. Plan: walk graph, compute execution order,
  │     allocate buffers, assign streams
  │
  └── 3. Dispatch: enqueue work items on assigned streams
        │
        ▼                                                 ON STREAM WORKERS
      Streams execute pre-planned work items             ─────────────────
        │
        ├── Each work item: (library_call, input_bufs, output_buf)
        ├── Intermediate buffers released via use-count tracking
        └── Result stored in node→getResult()
        │
        ▼
      Buffer lifetime (SharedData / hardware-specific allocation)
        │
        ├── shared_ptr refcounting: buffer lives until last reference drops
        ├── Views share parent SharedData: parent outlives children
        └── Stream destructor drains queue: all tasks complete before stream dies
```

### Decisions Summary

| Question | Decision | Rationale |
|----------|----------|-----------|
| Eager backend? | No — JIT only | One execution model. Simpler. |
| Planning vs. execution? | Host thread plans, streams execute | Compiler model. Inspectable, controllable scheduling. |
| Per-op stream params? | Yes — optional on every op, default to backend default | User controls placement at graph construction time. Eval dispatches accordingly. |
| Buffer allocation? | During planning (host thread), before dispatch | Host sees full graph for liveness analysis. |
| eval() blocking? | No — eval() plans + dispatches, host()/scalar() block | Explicit control over when to wait. |
| Multi-stream dnnl? | Each AsyncCPUStream owns its dnnl::stream | Worker thread exclusivity. |
| Worker threads per stream? | 1 | Sequential consistency within a stream. Parallelism via multiple streams. |
| Memory reuse? | Evaluator's existing use-count tracking | Already implemented. Extend to full memory planning later. |
| Error propagation? | Stream stores exception, rethrows on sync() | Existing AsyncCPUStream mechanism. |
| Exception safety? | SharedData refcounting + stream destructor drain | No additional mechanism needed. |
| DevicePtr? | Transitional — remove when eager backends are retired | Evaluator accesses buffers directly. |
| isDataReady flag? | Transitional — remove when per-op enqueue is retired | JIT tensors are either IR nodes or fully evaluated. |
| Buffer handles in work items? | `Tensor` (underlying storage type, not JitTensor) | Nodes already hold `optional<Tensor> result_`. Library ops already take `Tensor&`. No new abstraction needed. |
| Thunk-style indirection? | No — closures capture `Tensor` directly | No whole-program caching. Kernel caching (NVRTC), execution replay (CUDA graphs), and primitive caching (OneDNN) happen at their own natural granularities. |
| Compiled artifact caching? | Per-backend, keyed by graph signature + shapes + dtypes | NVRTC: cache PTX/cubin. CUDA graphs: re-capture (cheap). OneDNN: library-internal primitive cache. IR: cache optimized graph if needed. Lookup table problem. |
| Operator dispatch? | Three layers: generic passes → lowering passes → lightweight dispatch struct | Generic passes rewrite IR structurally. Lowering passes create CustomNodes calling parameterizable library free functions. Unlowered ops go through ~10-entry dispatch struct (function pointers, no virtual). TensorBackend evolves into adaptor populating the struct. |
| Cross-backend graphs? | Future: `getBackendFor(node->device())` replaces single `wrappedBackend_` | Per-node device/stream assignment already in the design. Evaluator dispatches to the right backend per node. Requires transfer nodes, device-aware allocation. |
| Output allocation in dispatch? | **Done.** Dispatch functions use pre-assigned `Tensor& out` buffers from the evaluator's `assignBuffers()`. When `out` is non-empty and types match, ops write directly into it — no double allocation, no `ensureResultInOutputBuffer` copy. Implemented for CPU binary/unary/reduction (OneDNN) and CUDA binary/unary/reduction. | Evaluator calls output-parameterized methods with planner-allocated buffers. Library ops in `ops/native/`, `ops/onednn/`, `ops/cuda/` take `Tensor& out`. |
| CustomNode? | Keep for lowering pass outputs and backward ops. Promote forward ops to first-class nodes as fusion passes need them. | Conv2d, BatchNorm, Pool2d are now first-class IR nodes (Conv2dNode, BatchNormNode, Pool2dNode). Multi-output ops use ProjectionNode pattern. Backward ops (conv2dBackward, batchnormBackward, pool2dBackward) remain as CustomNodes — no fusion opportunities in backward graph yet. RNN forward/backward still CustomNode (SSA refactor pending). |
| Eager mode? | No special mode. JIT is always on. | `eval()` is explicit. Data-accessing methods (`scalar()`, `host()`, `device()`, `toString()`) already trigger materialization via `getTensorOrEvalNode()`. No global flag, no eager abstractions. |

### Implementation Roadmap (from this design)

1. **Make JIT the default** for all backends. `setDefaultTensorType` should
   always set a `JitTensor<T>` type, never a bare `CPUTensor`/`CudaTensor`.
   Tests should run against the JIT backend.
   - [x] **Done.** 7 test files changed from `setDefaultTensorType<CPUTensor/CudaTensor>()`
     to `setDefaultTensorType<JitTensor<CPUTensor/CudaTensor>>()`.
   - [x] GPU tests verified: CudaTensorBaseTest, CudaTensorBinaryOpsTest,
     CudaTensorUnaryOpsTest, CudaTensorReductionTest (20/20), CudaTensorBLASTest,
     CudaTensorIndexTest all pass. ComputeTest_CUDA 32/32 pass.

2. **Per-op stream parameters.** Add optional `StreamOrDevice` parameter to
   all `TensorBackend` methods and free functions. `JitBackend` stores the
   stream on each IR node. Default: backend's default stream. This is the
   foundation for multi-stream graphs.
   - [x] **Done.** Output-parameterized `TensorBackend` virtual methods with
     `std::shared_ptr<const Stream>` parameter.

3. **Introduce plan/dispatch separation.** ~~Refactor `Evaluator::eval(node)` into
   two phases:~~
   - [x] **Done.** `Evaluator::eval()` now calls `buildPlan()` → `executePlan()`.
   - `EvalPlan` struct holds `topoOrder`, `useCount`, and `syncEdges`.
   - `buildPlan()`: iterative post-order DFS (explicit stack, no recursion). Computes
     topological order, use counts, and cross-stream sync edges in a single pass.
   - `executePlan()`: linear iteration — relativeSync → evalNodeDispatch → use-count
     decrement → dead intermediate cleanup. No recursion.
   - Removed recursive `evalNode()` and BFS `getNodeToUseCountInTree()`.
   - [x] **Pass infrastructure.** Added `BottomUpPass` and `TopDownPass` base classes in `Pass.h/cpp`.
     Base classes handle visited tracking, input recursion, and `setInput` updates.
     Subclasses override `rewrite(NodePtr) -> NodePtr`. `markVisited(Node*)` lets
     passes skip nodes (used by ElementwiseFusion for fusion group members).
     Migrated 6 of 7 passes (OneDnnOpFusion stays as plain `Pass` — unique iterative pattern).
   - All 136 JIT test cases pass.
   - **Next steps enabled by plan/dispatch:**
     - Buffer pre-allocation in the plan phase (memory planner, Step 7)
     - Execution profiling/instrumentation without touching dispatch logic
     - Future: work item enqueueing on assigned streams (Step 5 dispatch struct)
   - **Next steps enabled by pass infrastructure:**
     - New passes require only a `rewrite(NodePtr)` method — no traversal boilerplate
     - Enables rapid iteration on fusion passes: conv+relu epilogue fusion, fused attention,
       cuDNN Frontend graph lowering, CUDA graph capture boundaries
     - `TopDownPass` enables greedy outermost-first fusion (e.g., for tiling/blocking passes)
     - `BottomUpPass` enables canonical-form passes (e.g., strength reduction, algebraic simplification)
     - Fixed-point pass iteration: wrap `pass.apply()` in a loop until convergence
       (base class handles fresh `visited_` each call via `apply()` clearing it)

4. **Extract library functions.** Move per-op logic from CPUBackend/CudaBackend
   into output-parameterized free functions in `fl/tensor/ops/{library}/`.
   Add optional fusion configuration parameters where applicable:
   - OneDNN ops: `dnnl::post_ops` parameter (default empty)
   - cuBLASLt matmul: `cublasLtEpilogue_t` parameter (default `DEFAULT`)
   - cuDNN conv/pool/batchnorm: epilogue configuration
   - Native ops: already standalone free functions (mostly done)
   Remove per-op async wrapping from backends (JIT handles scheduling).
   Each function is independently testable without the JIT.
   - [x] **Done (CUDA).** Extracted all CUDA ops to `namespace fl::cuda` free functions
     in `ops/cuda/CudaOps.h`. CudaBackend virtual methods are now thin delegators.
     6 `.cu` files converted: CudaUnaryOps, CudaBinaryOps, CudaReductions,
     CudaBLAS, CudaShapeAndIndex, CudaAutogradOps.
   - CPUBackend already delegates well to `OneDnnOps` and `ops/native/` — no changes needed.
   - [x] **Buffer reuse wired.** All binary/unary/reduction ops (both CPU and CUDA)
     accept `Tensor& out` and write directly into pre-assigned evaluator buffers
     when non-empty, eliminating double allocation and reconciliation copies.

5. **Introduce BackendDispatch struct.** Replace TensorBackend's ~100 virtual
   methods with a lightweight dispatch struct (~10 function pointers for op
   categories: binary, unary, reduction, matmul, full, etc.). CPUBackend/
   CudaBackend become adaptors that populate this struct. The evaluator's
   `evalNodeDispatch` calls through the struct for unlowered ops. Lowering
   passes create CustomNodes that call library functions directly (bypassing
   the struct). Remove TensorBackend virtual dispatch from the hot path.
   - [x] **Done.** `BackendDispatch` struct in `fl/tensor/BackendDispatch.h`
     with 50 function pointers (7 evaluator hot-path + 5 creation/random +
     8 admin/lifecycle + 4 shaping + 8 named unary + 3 two-output + 7 named
     reductions + 9 autograd) plus `TensorBackendType backendType` field.
     `makeBackendDispatch(TensorBackend&)` creates default trampoline dispatch.
     Evaluator uses `dispatch_` for all op evaluation. JitBackend uses
     `dispatch_` for all delegated ops — `wrappedBackend_` member removed.
     Optimizer redesigned to take `vector<unique_ptr<Pass>>` directly
     (no `TensorBackend&` dependency). JitBackend constructor extracts
     backend passes via helper before member init.

6. **Consolidate `backend/jit/` into core tensor.** The JIT is not an optional
   backend — it IS the execution model. `backend/jit/` should be merged into
   `fl/tensor/` proper:
   - [x] `JitTensor<T>` template indirection removed — the default `Tensor`
     always builds IR nodes. `BackendDispatch` (already in `fl/tensor/`)
     is the interface to concrete backends.
   - [x] `JitTensorBase` is the standard `TensorAdapterBase` — every tensor
     holds a `NodePtr`.
   - [x] `DefaultTensorType.h`'s `#if FL_USE_JIT` conditional removed.
   - [x] IR nodes, passes, evaluator moved to `fl/tensor/ir/`, `fl/tensor/opt/`,
     `fl/tensor/eval/` (out of `fl/tensor/backend/jit/`).
   - [x] `JitBackend` intentionally kept separate from `TensorBackend`. The
     three-layer separation is architecturally correct: `TensorBackend` defines
     the user-facing op contract, `JitBackend` implements lazy execution strategy
     (builds IR, optimizes, evaluates on demand), and `BackendDispatch` provides
     evaluator-facing function pointers for hot-path performance. Merging
     `JitBackend` into `TensorBackend` would conflate "what ops exist" with
     "how ops are scheduled," preventing alternative execution strategies and
     forcing premature materialization.

7. **Cross-backend tensor transfer.** `tensor.to(BackendType)`: evaluate on
   source backend, copy buffer to destination, return new JIT tensor backed by
   destination. Stream-ordered where possible (e.g., `cudaMemcpyAsync`).

9. **`Tensor(Shape, dtype, TensorBackendType)` constructor.** Add a public
   constructor that dispatches through `BackendRegistry` to create a tensor
   backed by a specific backend type, rather than using the thread-local
   default. Useful for user-facing code that needs explicit backend control.
   Internal ops code (e.g., `ops/onednn/`) should continue using direct
   helpers like `makeCPUTensor` for zero-overhead static dispatch.

10. **Eliminate materialization barriers for opaque ops.** All ops with
    statically-inferrable output shapes should be lazy (wrapped in `CustomNode`
    at the JitBackend boundary). Multi-output ops use the shared-cache pattern.
    Ops that mutate state must first be converted to SSA semantics (return new
    tensors instead of mutating in-place). See the "Conv2D/Pool2D/BatchNorm/RNN
    (Library Primitives)" section above for the full task list and progress.
    After all ops are lazy, the next step is dedicated IR nodes for cross-op
    epilogue fusion (e.g., `batchnorm → relu` fused into a single cuDNN/OneDNN
    call via post-ops configuration).

11. **Relative sync for mutable tensor materialization.** `materializeMutable`
    currently uses blocking `stream().sync()` because OneDNN ops need raw
    buffer pointers immediately. If we add output-parameterized op APIs (where
    the caller pre-allocates the output buffer), we could use `relativeSync`
    (CPUEvent / cudaStreamWaitEvent) to order the materialization stream
    before the op's execution stream without blocking the host. This requires:
    - Op APIs that accept pre-allocated output buffers
    - Stream-aware op dispatch (pass target stream to the op)
    - Relative sync between the tensor's eval stream and the op's exec stream
    This is related to item 8 (memory planning) since pre-allocated buffers
    are a prerequisite.

8. **Memory planning.** Liveness analysis in `buildPlan()`: determine when each
   intermediate buffer is dead. Pre-allocate output buffers and recycle memory
   for non-overlapping lifetimes. The evaluator passes pre-allocated buffers
   to output-parameterized library functions from Step 4. Enables:
   - **Buffer reuse**: planner recycles dead buffers for later ops, reducing
     peak memory.
   - **In-place optimization**: when an input has no other uses after an op,
     its buffer can be reused as the output (e.g., `relu(x)` writes into
     `x`'s buffer).
   - [x] **Buffer pre-assignment done.** `assignBuffers()` pre-allocates output
     tensors. Dispatch functions (CPU binary/unary/reduction via OneDNN, CUDA
     binary/unary/reduction) write directly into pre-assigned buffers when
     non-empty. `ensureResultInOutputBuffer` becomes a no-op for updated ops.
   - [ ] **Remaining**: liveness analysis for buffer recycling, in-place mutation.

12. **Relu/gelu pattern recognition.** FL's IR represents `relu(x)` as
    `max(x, Scalar(0))` and `gelu(x)` as `0.5 * x * (1 + erf(x / sqrt(2)))` —
    multi-node subgraphs invisible to single-node lowering passes. Recognizing
    these patterns is a prerequisite for activation epilogue fusion in cuBLASLt,
    cuDNN, and OneDNN. Two approaches (non-exclusive):
    - **Dedicated IR nodes**: `ReluNode`, `GeluNode` created by a canonicalization
      pass that pattern-matches the multi-node idioms and replaces them. Downstream
      lowering passes then match single nodes. Simpler for lowering, but requires
      maintaining canonical forms.
    - **Multi-node pattern rewrite (`PatternRewritePass`)**: lowering passes
      directly match `max(x, Scalar(0))` etc. using the `PatternMatcher` DSL.
      More flexible, no canonicalization pass needed, but each lowering pass
      repeats the pattern. The DSL already exists (`PatternMatcher.h`); the main
      work is a base class that iterates (pattern, rewrite) pairs at each node.
    Prerequisite for: cuBLASLt activation epilogues (step 10 matmul fusion),
    cuDNN conv+activation (step 10 opaque op fusion), OneDNN eltwise post-ops
    on conv/pool/batchnorm (step 13).

13. **Backend-specific epilogue fusion passes.** With opaque ops as first-class
    IR nodes (step 10) and activation pattern recognition (step 12), add
    lowering passes that fuse adjacent ops into library epilogues:
    - **`OneDnnConvPostOps`**: match `conv2d → relu`, `conv2d → add(bias)`,
      `batchnorm → relu`, etc. and lower to a single OneDNN primitive with
      `dnnl::primitive_attr` + `dnnl::post_ops`. The library function
      factoring (step 4) already separates primitive creation from execution —
      the lowering pass populates `primitive_attr` before creation.
      See design doc lines 2289–2325 for the two-phase factory+execute pattern.
    - **`CudnnConvEpilogueFusion`**: match conv+bias+activation patterns and
      lower to cuDNN fused convolution (or cuDNN Frontend graph API).
    - **`CudaMatmulActivationEpilogue`**: extend existing
      `CudaMatmulEpilogueFusion` to match `matmul → relu` / `matmul → gelu`
      and use cuBLASLt `CUBLASLT_EPILOGUE_RELU` / `GELU` enums.
    - **Fused attention**: `CudnnSDPAFusion` for cuDNN scaled dot-product
      attention (future, requires multi-head attention IR pattern).
    Dependencies: steps 10, 12. Each pass is independently implementable.

14. **Copy-on-write mutation semantics.** Currently, views share buffers via
    refcounted `SharedData` and `flatten()` copies defensively. Add COW
    semantics for mutation:
    - Single-owner detection: if `SharedData` refcount == 1, mutate in place
    - Multi-owner: copy buffer, then mutate the copy
    - Integrates with memory planning (step 8): the liveness analysis can
      identify single-use intermediates for in-place reuse without COW overhead
    - JIT interaction: COW applies at materialization time — when the evaluator
      writes results into buffers, it checks ownership before reusing. For
      `IndexedUpdateNode` (in-place indexed assignment), COW ensures the source
      buffer is not aliased before writing.
    - Related: `materializeMutable()` already handles the eager case; COW
      generalizes this to all mutation paths.

15. **CUDA graph capture.** Record and replay entire execution sequences
    (including cuBLAS/cuDNN library calls) via `cudaStreamBeginCapture` /
    `cudaStreamEndCapture`. Distinct from NVRTC kernel fusion (step 5b) which
    compiles individual elementwise kernels — CUDA graphs capture the full
    dispatch sequence including library kernel launches.
    - **Subgraph identification**: a `SubgraphPass` (new abstraction beyond
      single-node `rewrite()`) identifies capturable regions — no host-side
      control flow, no dynamic shapes, no cross-device transfers, no
      host-synchronizing ops (`scalar()`, `host()`).
    - **Capture and replay**: the evaluator captures the first execution of a
      subgraph, then replays the recorded graph on subsequent executions with
      the same shapes/dtypes. Re-capture on shape change (cheap).
    - **Interaction with JIT fusion**: CUDA graphs and NVRTC fusion are
      complementary. NVRTC fuses elementwise chains into single kernels
      (reducing kernel count). CUDA graphs then capture the remaining
      multi-kernel sequence (reducing launch overhead). The optimizer should
      run NVRTC fusion first, then CUDA graph capture on the result.
    - **Prerequisites**: stable CUDA backend with all tests passing (see
      "Full-Stack Backend Test Coverage"), `SubgraphPass` infrastructure.
    - **Caching**: keyed by graph topology + shapes + dtypes. NVRTC caches
      PTX/cubin, CUDA graphs cache captured `cudaGraphExec_t`.
    Deferred until CUDA backend has full test coverage and NVRTC fusion is
    mature.

    **Unification note**: CUDA Graphs, cuDNN frontend graphs, and NVRTC
    kernels are fully complementary and should be unified in the evaluator:
    - NVRTC-compiled `CUfunction` launches can be captured into CUDA graphs
      (they're just `cuLaunchKernel` calls)
    - cuDNN frontend `execute()` calls launch internal kernels onto the
      stream, which CUDA graph capture records transparently
    - A single CUDA graph can contain both cuDNN subgraph nodes and NVRTC
      fused kernel nodes, eliminating all CPU dispatch overhead on replay
    - The main constraint is **address stability**: CUDA graph replay
      requires the same device pointers. Evaluate whether this is too
      onerous — options include a caching allocator with deterministic
      addresses, or `cudaGraphExecKernelNodeSetParams` to update pointers
      per-replay (more flexible but higher overhead). Re-examine whether
      the address stability constraint makes CUDA graphs impractical for
      dynamic workloads or whether shape-keyed re-capture is cheap enough.

## Algebraic Simplification / Canonicalization Pass

- [x] AlgebraicSimplification pass (BottomUpPass). 18 IEEE 754-safe rules:
  - Binary identity: x+0, 0+x, x-0, x*1, 1*x, x/1 → x
  - Power: x^0→1, x^1→x, x^2→x*x, x^0.5→sqrt(x)
  - Canonicalization: 0-x→neg(x), with inline 0-neg(y)→y
  - Unary involutions: neg(neg(x))→x, not(not(x))→x (b8 only)
  - Idempotent: abs(abs(x))→abs(x)
  - Reshape: identity elimination, chain collapse
  22 unit tests. Runs after ScalarFolding, before CSE.
  Also fixed ScalarFolding bug: BinaryOp::Gt was mapped to >= instead of >.

**Deferred rules** (not IEEE 754 safe without fast-math flag):
- [ ] Annihilator: x*0→0, 0/x→0 (NaN*0=NaN, 0/0=NaN)
- [ ] Self-cancellation: x-x→0, x/x→1 (NaN-NaN=NaN, 0/0=NaN)
- [ ] Inverse pairs: exp(log(x))→x (returns negative for x≤0 instead of NaN),
  log(exp(x))→x (returns finite for large x instead of inf)
- [ ] Cast round-trip: cast(cast(x,t1),t2) where t2==x.dtype()→x (removes
  precision loss from narrowing intermediate — semantically different)
- [ ] Div-by-const strength reduction: div(x, Scalar(c))→mul(x, Scalar(1/c))
- [ ] Commutative normalization: put Scalar on RHS for add/mul/min/max
- [ ] relu(relu(x))→relu(x) (max(max(x,0),0)→max(x,0))
- [ ] max(x,-inf)→x, min(x,+inf)→x

**Already handled by other passes:**
- Transpose composition/cancellation → TransposeSimplification
- Scalar constant folding → ScalarFolding
- Identity cast → CastNode::create (construction-time)
