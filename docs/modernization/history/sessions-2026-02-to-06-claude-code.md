# Claude Code sessions, 2026-02 → 2026-06 (digest)

About 45 sessions. Their transcripts were not retained; this digest is reconstructed
from the maintainer's prompts ([prompts-log.md](prompts-log.md)) and from what landed in
`TODO.md` and the code. Each entry: what was asked or decided, and what came of it.
Session IDs are the 8-character prefixes used in the prompt log.

## 2026-02 — getting started
- **02-12** (`786d9ef2`, `1b1a5922`): clone, build, and get every test in
  `flashlight/fl/test/tensor` passing on a GPU machine without adding dependencies. The
  earliest modernization sessions (Phases 0–2 in `TODO.md`) followed; their transcripts
  were lost even before March ("previous Claude Code sessions that have been lost").

## 2026-03-14 → 03-18 — fresh environment, tests, laziness
- **03-14** (`5c6ca034`): new environment. Build ArrayFire's CUDA backend from source as
  the reference implementation (GCC 14, ccache; CUDA 12.8 package-pinning issues), then
  Flashlight; run tests; read the project; fix failure categories. "Use the TODO list, 1
  file" and keep the maintainer tightly in the loop.
- **03-14/15** (`57f8a07a`): asked for a `Tensor` constructor taking a tensor type instead
  of `makeCpuTensor`-style closures; materialize only what ops need and sync only the
  relevant stream; build shared libraries. Settled that `TensorBackend` mirrors the user
  API, `JitBackend` is the default implementation, and `BackendDispatch` is the op-level
  API — with no design sacrifices for eager mode.
- **03-15** (`32240d31`, `b43d3d8a`): inventory of backend fusion work (oneDNN post-ops,
  copy-on-write, NVRTC vs CUDA graphs) into TODO; goal of full JIT test coverage on CPU and
  CUDA. CMake cleanup of `_CPU`/`_CUDA` test variants; persistent failing-test list;
  ShapeInference tests, splitting shape-only functions from tensor ones; type-inference
  contract tests.
- **03-16/17** (`94d11f1f`, `ff8bcfc6`, `086a4b89`, `85896254`, `e9023044`): JIT indexed
  assignment; CUDA strided views; `where` on CUDA with Thrust; arange/iota/scalar/padding
  gaps using CUDA libraries; arange fix, scalar extraction dtype, rvalue assignment;
  laziness check on "eager validation"; CuTile evaluation and library-first kernels (things
  NVRTC can inline or CUDA graphs can capture); `PatternRewritePass` with relu/gelu
  recognition; CUB replacements; cuDNN epilogue passes; algebraic-simplification TODOs.
- **03-17/18** (`469c89c9`, `09ac5967`, `5fe7cd21`, `ac686f9c`, `d21a825f`): no forced
  evaluation of forward/backward state (Jacobian tests call `fl::eval` instead); graph
  compaction tests and very long JIT chains; CMake loops for CPU/CUDA test variants;
  per-optimizer-pass tests; CPU failures; remove unnecessary materialization (sort etc.);
  SSA for BatchNorm.

## 2026-03-19 → 03-26 — multi-output IR, fusion, forward/backward
- **03-19/22** (`d21a825f`): BatchNorm and pooling as distinct nodes; multi-output IR
  designed after XLA-style precedents — `ProjectionNode` with named output indices, plus
  pattern-matching typedefs documented next to them. Dislike of the `setBare` singleton
  (global mutable state).
- **03-22** (`a95f409d`, `43a67133`): post-op fusion passes for oneDNN/cuDNN; tests for
  the passes themselves; training vs inference fusions in cuDNN; attention-library TODO;
  correctness audits; CPU/CUDA failures, LinearBf16; cuBLAS fusion opportunities; removing
  unnecessary casts; research fusion with CUDA library primitives.
- **03-23** (`43a67133`): forward + backward fusion requires seeing both graphs; graph
  breaks from `loss.scalar()`. Direction set: a purely functional approach where tensors
  know nothing about autograd; maximize laziness; provenance tracking only as a thin
  layer if needed; keep it simple even at a small host-side cost.
- **03-23/24** (`8c7cec68`, `5b55ba9e`): early/late pass stratification called a hack;
  compiler warnings; Phase 5 cleanup; checked suppressed NVCC warnings; `setBare`/
  `ScopedBareMode` eliminated in favor of bare-tensor factories (`makeCPUTensor`,
  `makeCudaTensor`) that the JIT wraps when needed. cuDNN graph API vs legacy BatchNorm;
  NHWC discussion concluded with an unopinionated JIT (no `memory_format`), transpose as
  reorder, users responsible for sensible layouts, and passes removing redundant
  transposes.
- **03-24** (`3a8fe3ac`, `4bb7122b`, `ef556a5a`, `e5f2d048`, `391d0dd0`): transpose-pass
  and per-operand cuDNN stream TODOs; reverted an opaque oneDNN fusion; pass-test
  organization; maximal-subgraph fusion; evaluator comments; BatchNorm-specific code
  flagged as a leak in `Evaluator.cpp`; `dnnl::graph` exploration; dead-output
  elimination pass; cuBLASLt extensions; "you didn't run the tests" — run CUDA tests; audit
  fusion fast paths; note to unify NVRTC and CUDA graphs.
- **03-25** (`531bdd09`, `6f34b97c`, `afd0406d`, `77c6571c`, `bc42916b`): tests for fixes;
  `ConvBatchNormFolding`; maximize graph-walk fusion with few bespoke components;
  cuBLASLt as fallback when cuDNN can't lower; FP8/MXFP deferred until libraries mature;
  backward matmul epilogues as JIT passes (autograd unaware); tolerances must not be
  loosened; relu-mask/forward-backward saving discussion → ExternalUse-aware fusion side
  outputs (`FusionSideOutput`); GraphVerifier simplified, no `NDEBUG` switches; oneDNN
  matmul+eltwise fusion; no `GeluNode`.
- **03-26** (`316e23a9`): `MultiValueNode`/`ProjectionNode` rewrite sketches; SSA vs
  compaction; the maintainer challenged the assumed constraints: if the loss isn't
  materialized mid-step, forward + backward is one DAG. Plan: TODOs for functional
  transformations, the multi-value rewrite, and async materialization; outline of VJP
  rules.

## 2026-03-27 → 04-03 — async materialization and transfers
- **03-27** (`760021a1`): skip activating `FusionSideOutput`; design async
  materialization — no `cudaLaunchHostFunc`, explicit eval, no `fl::step`, a
  `std::future`-interoperable `Future`, unify on `Event`, one `TransferNode` with
  transport chosen by device pair (NCCL P2P included), executed by the runtime. Later
  notes: copies inside generated kernels, dedicated copy streams, non-blocking
  cross-device consumption. Record: research/async-materialization-design.md.
- **03-27** (`f7e2c792`, `8e815861`): `Event` pimpl without `#if FL_BACKEND_CUDA`;
  `createEvent` vs `recordEvent`; keep `relativeSync` as a concept; plan `fl::copy`;
  backend-agnostic `CopyTest`; laziest barrier for copy; `fl::empty` (uninitialized
  allocation).
- **03-28** (`8e815861`, `840b162b`, `4282e0e4`): audit `TransferNode`/copy; empty in the
  IR; split slow files; `JitBackendUtils.cpp`; make all materializing APIs return futures.
  Discussion of where `Event` and stream live (data readiness of tensors vs op
  execution). Cleanups: `DeferredEventImpl`, header-only `Future`, `dataReadyEvent_`;
  async `host()`; `device()` returns a pointer without evaluation;
  `toHostVector`/`toHostVectorSync`.
- **03-29** (`c1e3bd78`, `1ac1dd6d`): stream-type checks; `isContiguous`/`strides`
  without materializing; remove lock/unlock semantics and ArrayFire memory-manager
  functions from the generic tensor interface; dispatch functions write into assigned
  output buffers; conv2d/pool2d output buffers; `ForwardBackwardState` typed properly (no
  `std::shared_ptr<void>`).
- **03-30** (`1ac1dd6d`, `8c5a79eb`): reduction buffer reuse; long-standing `var` test;
  async scalar/host wiring; JIT indexed assignment; relu/gelu recognized at tensor level,
  not as IR nodes; canonical arithmetic rewrite research → TODO.
- **03-30 → 04-03** (`b78535d6`, start): pinned host memory opaque to the evaluator;
  const-correctness sweep; `scalar<T>()`/`host()` via `TransferNode`s; CPU host-to-host
  as shallow copy; no `materializeSync` in async paths; `Future::get()` never evaluates;
  `.then()` removed in favor of event synchronization and user-managed callbacks;
  `hostSync` renames; async H2D path; "no preprocessor guards in tests; braces; Flashlight
  style".

## 2026-04 → 06 — API cleanup, passes, research (session `b78535d6`, continued)
- **04-06 → 04-16:** `Tensor` constructor/creation API cleanup and deduplication;
  `FusionSideOutput` work limited to what functional gradients won't obsolete; items made
  obsolete by VJP marked; `TensorBase.h` renamed to `Tensor.h`; `Pool2dNode`.
- **04-16 → 04-27:** transpose passes researched against mature compilers and
  implemented; conv2d transpose absorption; redundant-transpose tests; algebraic
  simplification planned with research (canonical rewrites), implemented with tests, and
  the rest recorded as TODOs.
- **05-04:** inventory; VJP design discussion started (effects on streams, copy streams,
  communication), then deliberately punted in favor of node promotions and passes that
  remain valid; research saved to files (research/vjp-design-research.md,
  research/tile-broadcast-research.md); TileNode research. Work then moved to Codex
  ([sessions-2026-05-codex.md](sessions-2026-05-codex.md)).
- **06-17:** audit of the Codex changes; builds and the test triage recorded in
  `TODO.md` ("Test status / known failures (audit 2026-06-17)").
