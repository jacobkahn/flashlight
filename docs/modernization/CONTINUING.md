# Continuing the modernization on a new machine

The whole modernization working copy is being checked in as one large change on a fork,
to be split into reviewable commits later. This page is the handoff: how to get going
again without any of the original sessions.

## 1. Read, in this order
1. [STATUS.md](STATUS.md) — where things stand and what is next.
2. [history/MAINTAINER-DIRECTIVES.md](history/MAINTAINER-DIRECTIVES.md) — how the
   maintainer wants the work done (design principles, style, process).
3. [DECISIONS.md](DECISIONS.md) — what is decided and what is open.
4. [ARCHITECTURE.md](ARCHITECTURE.md) and [DEVELOPMENT.md](DEVELOPMENT.md).
5. [LOG.md](LOG.md) and the session digests in [history/](history/README.md) for
   context, then `TODO.md` (top sections first) for the itemized roadmap.

## 2. Get a working build
Follow [DEVELOPMENT.md](DEVELOPMENT.md) §1 (CPU + CUDA, shared libraries; add
`-DFL_USE_ARRAYFIRE=ON` for the baseline lane). Then run
`python3 docs/modernization/tools/run_tests.py build <log-dir>` and compare against
[STATUS.md](STATUS.md) §2. Expected problems: `AutogradTest` (default lane, CUDA
advanced indexing) and `DatasetTest` (JIT thread safety); `_AF` differences as listed.
Anything else is new and worth investigating before changing code.

## 3. Where the work stopped
- **Next action:** the gather/scatter indexing plan
  ([plans/indexing-gather-scatter-plan.md](plans/indexing-gather-scatter-plan.md)) is
  approved in design and awaiting the go-ahead to start Phase 1.
- **Queued decisions:** Tensor copy semantics for parameters (P-1, the training no-op —
  highest priority), then the VJP graph-retention model (P-2); also JIT thread safety
  (P-4) and simplification scope (P-5).
- **Small open items:** `_CUDA` Contrib test variant; benchmark backend selector;
  single-pass f32-fallback loop for CPU unary ops; debug switches.

## 4. Working with an AI assistant
Assistants should follow the directives file. The essentials:
- Never use git; the maintainer handles version control.
- Run commands freely, but propose a plan and get approval before edits beyond local bug
  fixes. New abstractions need code-level plans.
- Keep `TODO.md` append/annotate-only; record decisions in `DECISIONS.md`, work in
  `LOG.md`, and keep `STATUS.md` true.
- Run the relevant tests (CPU and CUDA) after changes; skip known-slow suites unless
  they're affected.
- Keep backend-specific knobs inside backends and the IR small.

A starting prompt that worked well:

> Read `docs/modernization/CONTINUING.md` and the files it lists, then `TODO.md`'s top
> sections. Build the project and run the test runner to confirm the state matches
> `STATUS.md`. Don't use git. Before any edit beyond a local bug fix, propose a plan.
> Then let's continue with the next action in `CONTINUING.md` §3.

## 5. Later: splitting the change set into commits
The change set is large (~700 files versus base commit `f5f717f3`). The phases in
`TODO.md` and the log are the natural seams. A dependency-ordered first cut:
1. Build system and defaults (Phase 0: CMake 3.24, C++20, CUDA settings, option defaults
   including D-09).
2. Tensor API and test cleanups (Phase 1, 1.5): broadcasting, scalar overloads, autograd
   backend ops into `TensorBackend`.
3. Row-major migration (Phase 2), including ArrayFire dimension reversal.
4. `Variable` → `Tensor` merge and dtype promotion (Phase 2.5); `TensorBase.h` →
   `Tensor.h`.
5. Dispatcher: thread-local default type, backend registry (Phase 3).
6. Backend/ops reorganization: `ops/{onednn,native,cuda,cudnn}`, generic `CPUTensor`,
   `OneDnnOps` (Phase 4).
7. Native CUDA backend (Phase 5a).
8. Runtime: streams, `AsyncCPUStream`, `Event`, `Future`, `fl::copy` (async design).
9. JIT core: IR, `JitBackend`, evaluator, `BackendDispatch`, `MultiValueNode`/
   `ProjectionNode` (JIT-only design).
10. Generic optimizer passes, one per commit with its tests.
11. Backend optimizer extensions: oneDNN fusion, cuDNN subgraph lowering, cuBLASLt
    epilogues, NVRTC.
12. First-class node promotions, one per node with its rewrites.
13. The 2026-10 bug-fix pass, TF32 policy, ArrayFire opt-in/baseline, `_AF` lane.
14. Documentation (`docs/modernization/`, `TODO.md`).

Expect to reorder: some later fixes touch code introduced earlier, and several
intermediate states may not build without small adjustments. Splitting by these phases
first, then by feature within a phase, keeps each commit understandable.
