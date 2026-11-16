# Flashlight modernization — current status

*As of 2026-11-16.* What exists, what passes, what is broken, what is in flight, and what
comes next. History is in [LOG.md](LOG.md); decisions in [DECISIONS.md](DECISIONS.md);
the itemized checklist in [`TODO.md`](../../TODO.md).

---

## 1. Where the code is

- All modernization work is **uncommitted in the working copy** on top of base commit
  `f5f717f3`: ~400 modified files, ~253 added (intent-to-add), ~99 deleted; roughly
  +68k / −22k lines versus the base.
- Execution model: JIT-only (D-01). Backends: native CPU (oneDNN + native loops) and
  native CUDA (cuBLAS/cuBLASLt/cuDNN/Thrust/CUB/NVRTC) are the defaults; ArrayFire is
  opt-in and serves as the baseline (D-09).
- Build defaults: `FL_USE_CPU=ON`, `FL_USE_CUDA=OFF` (turn on for GPU; it enables
  `FL_USE_NATIVE_CUDA` and cuDNN when found), `FL_USE_ARRAYFIRE=OFF`. Default tensor
  type: native CUDA if built, else CPU, else ArrayFire, else Stub.
- Numerics: strict f32 on CUDA; `FL_ALLOW_TF32=1` opts in (D-08). CPU thread count from
  `OMP_NUM_THREADS`; `-DFL_TEST_NUM_THREADS=N` for ctest runs (D-07).

## 2. Test status

Test lanes: `_CPU` (JIT over CPU), `_CUDA` (JIT over CUDA), unsuffixed (compile-time
default; CUDA in a CUDA build), `_AF` (bare ArrayFire, built when ArrayFire is enabled).

**Last full run (2026-10-26, CPU + CUDA + ArrayFire enabled, every ctest-registered
binary):**
- CPU, CUDA and default lanes: everything passes except two known problem binaries:
  - `AutogradTest` (default lane): `GetAdvancedIndex` illegal memory access on CUDA
    (skipped in `_CUDA`); the sticky CUDA error then fails 5 later tests in that process
    (they pass when run alone). Root cause: CUDA tensor indexing (fixed by D-11 Phase 2).
  - `DatasetTest.FileBlobDataset`: JIT thread-safety race (P-4).
- `_AF` lane: 334 pass / 14 fail / 6 skip. Of the 14 failures, 9 are tests that assume
  JIT behavior or native backends (`ComputeTest` laziness checks, `TensorAdapterTest`
  default-type checks, `CopyTest` helper choosing the CPU stream); the rest are real
  ArrayFire differences (below).

**Since then (2026-10-27):** fixes for CPU unary dtypes and CUDA `clip` with two new
tests; targeted re-runs pass on CPU/CUDA/default (`TensorUnaryOpsTest`,
`AutogradUnaryOpsTest`, `ModuleTest`, `OptimTest`, `TensorBaseTest`, `AutogradTest`,
`ContribModuleTest_CPU`, `NNUtilsTest_CUDA`). `_AF` fails the new
`mathOpsNonF32Types` (ArrayFire `floor(s32)` returns f32). **A full re-run has not been
done since these fixes.**

### Backend parity (generic suites, 2026-10-26)
| Lane | Pass | Fail | Skip |
|---|---|---|---|
| `_AF` (bare ArrayFire) | 334 | 14 | 6 |
| `_CPU` | 313 | 0 | 41 |
| `_CUDA` | 321 | 0 | 4 |

`_CUDA` has no Contrib-model variant (Transformer, Conformer, TDS…), so 29 tests are
absent there; only the default lane (CUDA) covers them.

- ArrayFire passes, native does not: all f16 tests vs CPU (CPU has no f16), casting
  tests, `AsContiguous`, advanced-index gradients (CPU unsupported, CUDA crashes).
- Native passes, ArrayFire does not: bf16 (ArrayFire has none), `tril`/`triu` (rank-2,
  diagonal 0 only on ArrayFire), f16 conv+bias in Conformer/TDS (cuDNN finds no engine
  on the ArrayFire path), mixed f16/f32 embedding, `TensorBaseTest.scalar`, integer
  `floor` dtype.
- Per-operation matrix: [research/backend-support-matrix-2026-10-26.md](research/backend-support-matrix-2026-10-26.md).

## 3. Known open problems

Priority order; details and file references in `TODO.md` and the review.

1. **Training is a no-op** — optimizer updates never reach module parameters (P-1).
2. **JIT is not thread-safe** (P-4).
3. **CUDA tensor indexing** — wrong axis, dropped indices, strided write-back; autograd's
   index backward is a per-combination host loop (both addressed by D-11).
4. **CPU execution is effectively synchronous** — `relativeSync` blocks the host per op;
   CPU `Tensor` copy is a blocking deep copy; host-loop native ops round-trip per call.
5. **CUDA overheads** — Thrust `par.on(stream)` synchronizes after algorithms; cuDNN
   graphs and plans rebuilt every eval (`CudnnPlanCache` unused); several host
   round-trips (full reductions, RNN allocation, `indexAdd`).
6. **Integer semantics** — CUDA computes non-f64 ops in f32 (s32 above 2^24 and all s64
   lose precision); NVRTC computes ints in f32 with fast-math; u32/s16 storage
   reinterpretation issues; CPU rejects rank > 4 in oneDNN paths.
7. **Runtime gaps** — `Future::isReady()` can stay false; node stream ≠ execution stream
   on CUDA; BatchNorm running-stats chain leak; `detach()` returns a reference into a
   side list.
8. **ArrayFire baseline gaps** — evaluator's per-node preallocation is wasted on the
   ArrayFire path (measure); `fl::copy` has no ArrayFire route; f16 conv+bias engine
   failure on the ArrayFire path.
9. **Test infrastructure** — Jacobian tests are slow (finite differences, cuDNN plan
   rebuilds); some tests write fixed `/tmp` names (collide in parallel runs); stale
   binaries survive in the build tree; pass tests are structural only (no differential
   execution against a reference); no checked-in parallel runner.

## 4. In flight

- **Indexing redesign (D-11):** plan written
  ([plans/indexing-gather-scatter-plan.md](plans/indexing-gather-scatter-plan.md));
  **waiting for the go-ahead to start Phase 1** (add `gather`/`scatter` to all backends,
  the JIT and autograd, with tests; no behavior change to `x(...)` yet).
- A draft CUDA indexing fix from 2026-10-27 was kept only in local scratch space and
  has since been lost; it is superseded by D-11 Phase 2 and was never part of the
  working copy.

## 5. Next steps (proposed order)

1. D-11 Phase 1 → review → Phase 2 → Phase 3.
2. Decide P-1 (Tensor value/alias semantics) and fix training; enable
   `OptimTest.StepUpdatesModuleParams`.
3. Decide P-4 (JIT thread safety).
4. VJP slice and graph-retention model (P-2), then compaction policy (P-3).
5. Execution-layer performance: CPU async overlap (`relativeSync`), Thrust `par_nosync`,
   cuDNN plan caching, benchmark backend selector with ArrayFire numbers.
6. Small items: `_CUDA` Contrib variant; single-pass f32-fallback loop for CPU unary ops;
   debug switches (graph dump, pass disabling, synchronous CPU stream); ASAN/TSAN
   configs; unique temp names in tests.

---

## 6. Check-in checklist (for committing this working copy)

**Notes to check in**
- `docs/modernization/` (this directory) — the consolidated notes, including
  `history/` (session digests, maintainer directives, prompt log) and `CONTINUING.md`.
- `TODO.md` — the working roadmap; references now point into `docs/modernization/`.
- `README.md` — updated build options (ArrayFire opt-in, `FL_USE_CPU`/`FL_USE_CUDA`/
  `FL_USE_NATIVE_CUDA`, `FL_TEST_NUM_THREADS`).

**Source files that must be added (the build needs them)**
- 253 intent-to-add files (`git status` shows ` A`): git holds no content for them
  until they are added.
- Untracked new sources: `flashlight/fl/tensor/ir/{Clip,Concatenate,Cumsum,Flip,Pad,Roll,Tile,Triangular,Where}Node.{h,cpp}`,
  `flashlight/fl/tensor/ops/cuda/CudaNumerics.h`,
  `flashlight/fl/tensor/ops/cudnn/CudnnNumerics.h`.

**Do not check in**
- `Testing/` (ctest scratch output) and `build/`.
- `flashlight/fl/tensor/opt/backends/cuda/impl` (an empty stray file, unreferenced).
- `2201.12465` (the paper's arXiv source tarball) and `flashlight-paper/` (its extracted
  LaTeX) — keep out unless the maintainers want the paper in-tree; link arXiv instead.
- `.claude/` (local tool state; its `plans/` are now copied into
  `docs/modernization/research/` and `plans/`). Consider adding `.claude/` to
  `.gitignore`.
