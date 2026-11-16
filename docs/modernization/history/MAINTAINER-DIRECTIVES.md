# Maintainer directives and design principles

The standing instructions and design positions the maintainer gave across every AI-assisted
session (Claude Code 2026-02 → 2026-11, Codex 2026-05/06), consolidated by theme. They
are the best guide to "what would the maintainer want" when continuing the work. Dates
point into [prompts-log.md](prompts-log.md) for the original wording. Where a later
statement refines an earlier one, the later one wins.

---

## 1. Purpose

- Build a framework flexible enough for the largest-scale training workloads, small
  enough for people to understand, and unopinionated enough to express new ways of
  computing. Agents should eventually be able to build on top of it to discover new ways
  to train at scale and help automate research. *(2026-05-04)*
- Follow the spirit of the original Flashlight (paper, READMEs): readable, editable,
  minimal API, low overhead, internal modifiability — even though the goals have grown.
  *(TODO.md brief; 2026-05-04)*
- Write a compiler that can eventually do what hand-written MoE "monokernels" do, without
  breaking the programming model, simplicity, or existing abstractions. Keep making the
  general compiler robust first; specialize later. Keep future requirements (dynamic and
  ragged shapes, explicit communication, memory planning, rematerialization, scheduling,
  generated kernels) in view so today's abstractions don't preclude them. *(2026-05-05 →
  2026-05-11; the conclusions are in TODO.md "MoE / bleeding-edge kernel direction")*

## 2. Architecture and IR

- **JIT only; no design sacrifices for eager ("bare") mode.** `TensorBackend` defines the
  user-facing op interface; `JitBackend` is the default implementation; `BackendDispatch`
  is the op-level API underneath. *(2026-03-15)*
- **Keep the IR small and unopinionated.**
  - No activation nodes such as `GeluNode`: express activations with tensor primitives
    and pattern-match them in backend passes. *(2026-03-25, 2026-03-30)*
  - No layout tags or `memory_format` in the IR. Layout is a contract at the op/library
    level: users prepare data for an op's contract, ops reorder internally if needed,
    and passes remove redundant transposes. A transpose is just a reorder. *(2026-03-24)*
  - Keep the operator set very small; prefer a couple of general primitives (e.g. only
    `gather` and `scatter`) over many specialized ones. *(2026-11-02)*
  - Do promote ops that the compiler must reason about (tile, where, flip, cumsum, roll,
    clip, concatenate, pad, triangular, conv/pool/batchnorm) to first-class nodes rather
    than opaque `CustomNode`s. *(2026-05-04 → 05-12)*
- **Generalize interfaces to mature conventions**, and let backends that can't support
  the general form throw instead of narrowing the interface (e.g. `tril/triu(x,
  diagonal)` over the last two dims). *(2026-05-12)*
- **Backend-specific details stay in backends.** No backend code in the evaluator
  (pinned host memory must be opaque to it; no BatchNorm-specific logic in
  `Evaluator.cpp`); numeric knobs such as TF32 are hidden in CUDA library components,
  never on generic IR nodes, the evaluator, or `fl::` APIs. *(2026-03-24, 2026-03-30,
  2026-10-26)*
- **Don't add API surface for one use.** Question specialized additions (e.g.
  `emptyCopyDst`); reuse existing facilities first. *(2026-03-30)*
- **ArrayFire keeps working but must not shape the Tensor API**; remove ArrayFire-specific
  functions and comments from generic code. ArrayFire is now the opt-in baseline to beat,
  not the default. *(2026-03-29, 2026-10-26)*
- **Prefer standard C++ types** for interoperability (e.g. a future that interoperates
  with `std::future`/`std::shared_future`). *(2026-03-27)*

## 3. Laziness, materialization, and async

- Be as lazy and deferred as possible. Materialize only for implementations that truly
  need memory or pointers, and then only the tensors involved; synchronize the relevant
  stream (relative sync), never the whole JIT stream. *(2026-03-14)*
- Never add materialization to make tests pass; tests call `fl::eval` themselves.
  *(2026-03-17)*
- Prefer SSA-style dataflow; ops that write outputs (BatchNorm etc.) should be SSA where
  possible. *(2026-03-18)*
- Data-accessing APIs are **async by default**; blocking versions carry a `Sync` suffix
  (`scalar` / `scalarSync`, `toHostVector` / `toHostVectorSync`, `hostSync(T*)`).
  Internal API names should be just as clear. *(2026-03-27, 2026-03-28, 2026-04-02)*
- **The user triggers evaluation explicitly.** `fl::eval(tensors...)` starts work;
  `Future::get()` waits on an event and must not call `eval` or have other side effects;
  no global `fl::step()`. No `.then()`: users build their own callback patterns (thread
  pools) so they control the overhead; avoid `cudaLaunchHostFunc`. *(2026-03-27 →
  2026-04-02; see research/async-materialization-design.md)*
- Data movement is one `TransferNode` whose transport depends on the device pair (same
  device, P2P, staged through host, NCCL, host↔device); `scalar<T>()` is a transfer plus
  a host read, not special machinery. Keep room for dedicated copy streams, copies inside
  generated kernels, and non-blocking cross-device consumption. *(2026-03-27)*
- `device()` should hand out pointers without forcing evaluation, so library calls and
  kernels can be wired up lazily (and eventually captured in CUDA graphs).
  *(2026-03-28)*
- Host-to-host transfers are shallow copies (copy-on-write), not stream work.
  *(2026-03-31)*

## 4. Autograd

- Avoid PyTorch-style autograd complexity. Aim for tensors that know nothing about
  autograd, with gradients as functional transformations of the graph; forward and
  backward should be one graph that can be optimized jointly. *(2026-03-23, 2026-03-26)*
- Autograd stays unaware of backend optimizations; forward→backward fusions are JIT
  passes. *(2026-03-25)*
- If the loss is not materialized mid-step, forward + backward is a single DAG: make
  materialization async so logging doesn't break the graph. *(2026-03-26)*
- VJP design is deliberately deferred; first do work that stays valid whatever VJP looks
  like (node promotions, passes, lowerings), and avoid work that functional gradients
  would make obsolete. *(2026-04-16, 2026-05-04)*

## 5. Performance and kernels

- Prefer the fastest library implementations (cuBLAS/cuBLASLt, cuDNN, Thrust, CUB,
  cuTile, oneDNN) over custom kernels, favoring paths that can be inlined into NVRTC
  kernels or captured in CUDA graphs. *(2026-03-16, 2026-03-17)*
- Maximize fusion by graph walking (maximal subgraphs, e.g. cuDNN/oneDNN graph APIs) and
  add as few bespoke components to the JIT as possible; reject opaque fusions that hide
  global optimization opportunities. *(2026-03-24, 2026-03-25)*
- Make sure fast paths are actually hit (verify with documentation, research, standalone
  checks). *(2026-03-24)*
- Ground optimization passes in what mature compilers do (XLA, MLIR, TVM, PyTorch
  Inductor, …); record the rewrites we don't implement yet as TODOs; note optimizations
  that need VJP. Graph-level optimizations that work without VJP will work with it too.
  *(2026-04-16, 2026-04-22, 2026-05-11, 2026-05-12)*
- Wait for library support for FP8/MXFP rather than building it ourselves *(2026-03-25;
  revised 2026-10 research: FP8 is ready on H100)*.

## 6. Code style

- Follow existing Flashlight style; braces around every loop and condition; compact,
  informative comments (Flashlight-style, not narrative). *(2026-03-24, 2026-04-03)*
- **No preprocessor guards in tests**; use runtime backend checks and skip, build
  multi-backend tests conditionally in CMake, and construct tensors canonically
  (`fl::rand`, not `backend.rand`). *(2026-04-03, 2026-05-12)*
- No `NDEBUG`-style preprocessor switches added for features like the graph verifier.
  *(2026-03-25)*
- Prefer simple code: single closures over elaborate helpers, inline trivial helpers.
  *(2026-03-25)*
- No type erasure via `std::shared_ptr<void>`; type state properly. *(2026-03-29)*
- No unnecessary `const_cast`s; methods that can be const should be. *(2026-03-30)*
- No unnecessary casts (pass a dtype to `fl::full` instead of `.astype`). *(2026-03-23)*
- Fix warnings properly instead of suppressing them. *(2026-03-24)*
- Avoid overload explosions; keep the API nice. *(2026-03-29)*
- Split slow-to-compile files; keep implementations out of headers unless templates
  require it. *(2026-03-28)*

## 7. Tests

- Every fix gets a test; tests must test something meaningful (question structural
  "builds X" tests that only check wiring). *(2026-03-25, 2026-05-05)*
- Test optimizer passes individually, not only through end-to-end API tests.
  *(2026-03-18)*
- **Always run the relevant tests after changes — including CUDA tests.** *(2026-03-24,
  2026-03-29)*
- Don't run very slow suites (f64 native fallbacks, CPU autograd/Jacobian) unless the
  change touches them; check test parallelism; kill stuck tests. *(2026-03-15,
  2026-03-18, 2026-03-28, 2026-04-02, 2026-05-04)*
- Don't loosen tolerances to hide problems; investigate (fused kernels should not change
  results materially). *(2026-03-25)*
- Keep the list of failing tests in a persistent place so work can resume after context
  loss. *(2026-03-15)*

## 8. Process

- The maintainer owns version control: never use git (no commits, staging, pushes). The
  working copy accumulates the whole refactor, which will later be split into many small
  commits (dozens to hundreds), with assistant help. *(2026-05-04; 2026-03-25; 2026-10-26)*
- Run commands freely, but **ask before editing**, and make edits thematic. *(2026-05-04)*
- Propose concrete plans first; for new abstractions include code-level detail (classes,
  invariants, validation, lowering, verifier, CSE keys, rewrite rules with guards, tests
  by file). Approve → implement → test → summarize → propose next. *(2026-05-05,
  2026-05-11, 2026-10-26)*
- Ask what to work on next; work in explicit, chunked tasks; prefer high-impact items
  that other work depends on. *(2026-05-04, 2026-05-11)*
- Keep `TODO.md` current as the single roadmap: add items for anything deferred, mark
  completed work, and never delete past records (annotate inline with dated
  corrections). *(2026-03-14, 2026-10-26)*
- Work with the maintainer tightly in the loop on design; pause on fundamental design
  decisions and present options with a recommendation. *(2026-03-14, 2026-10-26)*
- Build shared libraries for faster iteration. *(2026-03-14)*
- Save research findings to files so they don't need to be redone. *(2026-05-04)*
- Explain designs plainly when asked; show examples. *(2026-04-24, 2026-11-02)*
