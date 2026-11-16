# Codex sessions, 2026-05 → 2026-06 (digest)

One long session (2026-05-04 → 05-13; 48 maintainer prompts) plus short sessions that
were interrupted (05-04) or resumed it (06-11). Summarized from the full transcripts.
The maintainer's prompts are in [prompts-log.md](prompts-log.md).

## Working agreement set at the start
- Never use git; keep everything as uncommitted changes. The refactor will later be split
  into many discrete commits.
- Commands may run freely; **ask before any edit**; keep edits thematic.
- Propose concrete plans; ask what to work on next; prefer work that is invariant to later
  design changes (especially VJP).
- For new abstractions, give code-level plans (classes, invariants, validation, lowering,
  verifier checks, CSE keys, rewrite guards, tests by file).
- Ground passes in what mature compilers do.

## Work completed
- **CPU native helpers:** f32 unary loops call float overloads directly; shared row-major
  stride/index helpers in `ops/native/CpuTensorHelpers.h` replaced per-element vector
  allocation in flip, roll, tril, triu, pad, tile and broadcast binary fallbacks.
- **Priority analysis** (05-04): promote opaque `CustomNode` ops to first-class IR nodes
  first (VJP wants clean node types); then test layering; tensor correctness gaps; a
  scatter-reduce primitive; memory planning; communication nodes; higher-level fusions.
  VJP itself deliberately later.
- **First-class IR nodes**, each with node class, `NodeType`, `JitBackend` construction,
  evaluator dispatch, `GraphVerifier` shape/dtype checks, Graphviz labels, CSE keys and
  focused tests:
  - `TileNode` — fixed rank-aware `tile(tile(x, A), B)` composition; identity only when
    rank and shape are preserved; `tile(transpose(x), T) → transpose(tile(x, T'), axes)`
    for rank-preserving tiles with explicit axes and a single-use transpose (the
    rank-expanding case is a TODO).
  - `WhereNode` — modeled as the lower-level ternary select (same-shape operands, b8
    condition, `promoteDtypes(x, y)` result), matching JAX `lax.select` / XLA `Select`.
    Documented gaps: broadcasting `where`, scalar-overload shapes, mixed-dtype
    runtime/metadata agreement. Simplifications: `where(c, x, x) → x`; constant-condition
    folds only when the selected branch dtype matches.
  - `FlipNode` — `flip(flip(x, a), a) → x`; identity on size-≤1 axes; flip/transpose
    commute (including default reverse axes).
  - `CumsumNode`, `RollNode` (shift normalized at construction; nested-roll fusion;
    identities), `ClipNode` (three-input form; scalar bounds are expanded by the
    `TensorBackend` overloads before reaching the node).
  - `ConcatenateNode` — XLA/StableHLO-style canonicalizations: single-input identity,
    empty-operand removal (only when Flashlight's rank-promotion semantics are preserved),
    nested same-axis flattening; TODOs for slice-of-concat, concat of adjacent slices,
    concat-as-pad, repeated singleton concat, direct buffer placement.
  - `PadNode` — normalized per-rank widths and `PadType`; zero-width identity; nested
    zero-constant pad fusion; pad/transpose commute; TODOs for non-zero constants, edge
    pads, crop cancellation, absorbing pads into conv/pool descriptors.
  - `TriangularNode(mode, diagonal)` — at the maintainer's direction, `tril`/`triu` were
    generalized to PyTorch/JAX/MLX semantics (rank ≥ 2, last two dims, diagonal offset)
    across the public API, autograd, CPU and CUDA; ArrayFire throws for unsupported cases.
    Simplifications: identity ranges, zero input, nested same-mode fusion, last-two-dims
    transpose swaps lower/upper.
- **TODO structure** for per-node optimizations and lowerings (flip, roll, cumsum — reverse
  cumsum, CUDA segmented scan —, concat, pad) and a cross-cutting **indexed-map fusion**
  representation (output index → input index expression) for flip/roll/pad/tile/tril/triu
  and masks.
- **Test cleanups:** removed preprocessor backend guards from tests in favor of runtime
  checks; generic tests use canonical `fl::rand` and skip based on the concrete backend
  behind the JIT.

## Explanations worth keeping
- **Why `transpose(tile(x))` beats `tile(transpose(x))`:** transpose is often a view or
  absorbed by backend lowering, while tile replicates data. Moving tile inward puts the
  replication factors on the original semantic axes and leaves the transpose outermost,
  where transpose simplification and layout absorption already work. Example:
  `tile(transpose(x, {2,0,1}), {5,6,7}) = transpose(tile(x, {6,7,5}), {2,0,1})`.
  Rank-expanding tiles complicate this because the synthetic leading axes must be placed
  relative to the permutation.
- **Simplification vs. lowering timing:** add invariant infrastructure for every promoted
  node immediately; add only local, semantics-stable rewrites at promotion time; build
  pattern-rewrite ergonomics; then add cross-op and backend lowerings in thematic batches.
  `FusedElementwiseNode` is unary/binary only, so `where` in generated kernels needs an
  N-ary extension.
- **Backend capability checks** (the question the session ended on, answered 06-11):
  support depends on the runtime-selected backend, JIT wrapping, library and device
  capabilities, dtype, rank, shape, layout and op attributes, so it must be a
  metadata-only *runtime* capability query; constexpr traits can only help populate
  backend-local tables. Recorded as a TODO in the Phase 3 backend-registry section.

## MoE kernels and compiler direction (design memo, 05-05 → 05-11)
Prompted by a pasted discussion of MoE "monokernels" (omitted here). The analysis — now
recorded in `TODO.md` ("MoE / bleeding-edge kernel direction") — concluded:
- Systems that handle this well split a semantic tensor graph from a lower
  execution/schedule/buffer layer (XLA: async start/done collectives, while loops,
  rematerialization, latency-hiding scheduling, buffer assignment, thunks and command
  buffers; MLIR/IREE: async tokens, bufferization, stream resources). CUDA host functions
  must not call CUDA APIs; CUDA graphs need fixed kernels, arguments and addresses, so
  dynamic shapes need re-recording or scoped capture. MegaBlocks, Megatron-Core MoE and
  DeepEP show the production toolkit: dispatch/combine kernels, grouped GEMM, recompute,
  overlap, scoped graphs.
- Don't put MoE knobs on tensor ops. Keep the public API semantic; grow the IR (control
  flow, symbolic/ragged shapes, communication and transfer start/done nodes, effects and
  aliasing, workspaces); attach region-level policies (numerics, memory/recompute,
  schedule, layout) that refine lowering without changing semantics; lower recognized
  regions to library calls, fused kernels or persistent kernels; keep `CustomNode` as an
  escape hatch with declared shapes, aliasing, effects, streams and VJP.
- Realistic generation targets are structured kernel families — routing, permutation,
  pack/unpack, segmented maps/reductions/scans, activation epilogues — generated from a
  small internal kernel DSL and compiled with NVRTC (a full CUDA C++ runtime compiler that
  cannot call cuBLAS/NCCL from kernels). Use libraries for GEMMs and transport first.
- Dependency order: VJP/joint forward-backward graph → effect + alias metadata → memory
  planner with workspaces → async communication nodes → dynamic/ragged shapes → schedule
  IR → an MoE reference pattern. No fundamental architecture change is needed now; keep
  checking that new abstractions don't erase information these steps will need.
- Note: the later rejection of a `MathMode` field on generic IR nodes (2026-10-26) is
  consistent with this memo: policies belong at the region/lowering level, never on
  generic node fields or in the evaluator.
