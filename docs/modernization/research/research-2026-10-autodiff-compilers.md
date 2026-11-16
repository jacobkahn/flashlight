# Research: autodiff, effects/aliasing, memory planning, execution, dynamic shapes, pass infra (2026-10-26)

Web survey done for the Flashlight roadmap (VJP rules → effect/alias metadata →
memory planner → execution layer). Tags: **[V]** verified against linked
doc/source this session; **[R]** recalled, not re-checked; **[I]** inference /
recommendation. Versions at time of writing: MLX 0.32.4, JAX 0.12.0.1
(2026-10-23), PyTorch 2.14.1, tinygrad 0.14.0, Burn 0.22.0 (2026-10-06),
CUDA Programming Guide v13.4.2.

---

## 1. Autodiff over lazy graphs

### How systems do it
- **MLX** ([transforms.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/transforms.cpp)) [V]:
  targets via `argnums`; `vjp()` wraps primals as tracers (materialized leaf →
  buffer-sharing new array; else `copy`), re-runs the function. Tape = post-order
  DFS keeping a `calc_grad` id set (node taped iff an input is in it;
  `StopGradient` early-returns). Reverse sweep: seed `astype(cotan, out.dtype)`;
  skip nodes whose outputs (incl. siblings) have no cotangent; unused sibling
  outputs get `zeros_like`; `Primitive::vjp(inputs, cotangents, argnums,
  outputs)`; fan-out summed with `add`; consumed cotangents `extract`ed (early
  free); unreached primals get `zeros_like`. After eval MLX `detach()`es nodes
  (clears primitive/inputs/siblings) **except tracers**; `RetainGraph` keeps the
  graph inside `vjp` — because transforms re-run the function, compaction never
  destroys a graph a transform needs. `async_eval` is experimental and rejected
  inside transforms. `mx.compile` traces with placeholders; `custom_function`
  can override vjp/jvp/vmap (captured vars are constants). In-place
  `x[idx] = v` is a scatter; gradient at overwritten index is 0; duplicate-index
  writes nondeterministic; `x.at[idx].add(y)` accumulates.
- **JAX** ([autodidax](https://docs.jax.dev/en/latest/autodidax.html)) [V]:
  `linearize` = JVP under partial eval; `vjp` transposes the linear jaxpr (only
  linear prims need transpose rules). `custom_vjp` = `fwd -> (out, res)`,
  `bwd(res, g)`; `None` = zero cotangent; `symbolic_zeros=True`. Remat policies
  return saveable-or-not per equation (`dots_with_no_batch_dims_saveable`,
  `save_only_these_names`, `save_and_offload_only_these_names`); "a value is only
  saved if actually needed"; `print_saved_residuals`. 0.12 adds
  `custom_vjp.defremat` (flags `jax_custom_vjp3`/`jax_remat3`), removes
  `jax.custom_remat`, errors where vmap+custom_vjp used to overcount.
- **tinygrad** ([gradient.py](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/mixin/gradient.py)) [V]:
  `pm_gradient` PatternMatcher — one rule per op returning per-source grads or
  `None` (MUL swaps; WHERE masks & gives cond `None`; comparisons/BITCAST
  `None`; movement ops invert: PAD↔SHRINK, PERMUTE via argsort; EXPAND passes).
  `compute_gradient` walks only target paths (skips DETACH), errors on missing
  rules, **unbroadcasts generically in a wider accumulation dtype**, sums shared
  nodes. `CALL` nodes get a `needed` set. Max ties split 0.5/0.5; reduce-max
  divides by tie count. `requires_grad` removed in 0.13; `Tensor.gradient(*targets)`
  is the core API; `backward()` fills `.grad` for live float tensors found via a
  weakref registry. `realize()` replaces realized nodes with buffers in every
  live Tensor's graph → gradient must be built **before** realize [I from source].
- **PyTorch** [V/R]: tape + version counters; AOTAutograd traces joint fwd+bwd,
  functionalizes, partitions. Default min-cut partitioner
  ([Chillee 2022](https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467)):
  node split into in/out with edge 2×bytes (1× if already in memory), banned ops
  (matmul, RNG, unfusible, >4× shrinking reductions) get source edges; ~80 LOC.
  Compiled autograd (2.4+) captures the real `loss.backward()`. 2.10:
  `torch.cond` autograd; 2.14: experimental complex support in compile.
- **Lightning Thunder** [V]: `register_augmented_forward` + `register_backward`;
  remat = min-cut (igraph) per fusion region.
- **Burn 0.22** [V]: autodiff is device runtime state ("permits graph
  recording; doesn't make every tensor require grad"); custom backward ops;
  dedicated BatchNorm training op with closed-form backward; checkpointing marks
  ops ComputeBound/MemoryBound (memory-bound recomputed via RetroForward).
- **Luminal** [V]: 15 primitive ops, lazy DAG, symbolic dims, egglog search;
  autograd "modeled ahead of time".

### Pitfalls
- **Residual lifetime when evaluated nodes are compacted** — the single biggest
  risk for Flashlight (tinygrad loses the graph on realize; MLX avoids it by
  re-tracing and exempting tracers; PyTorch keeps saved tensors per node).
- Bitwise-consistency assumptions (JAX 0.11.2 changed `lax.min/max` JVP); in a
  fusing JIT, recomputed `x == max(x)` masks can disagree with forward.
- Complex conjugation bugs recur (MLX 0.32.x, JAX 0.12). Ties/degenerate cases
  (max ties, reduce-mul zeros, ldexp at 0). Shared outputs over-counting
  cotangents. Closures get no gradient. Seed/cotangent dtype. Non-differentiable
  dtypes. Cost of zeros (`zeros_like` vs symbolic zeros) — prune zeros in autodiff,
  not via IEEE-unsafe `x*0 → 0`.

### Recommendation for Flashlight [I]
1. Driver `vjp(outputs, cotangents, targets)`: on-path filter (MLX `calc_grad` /
   tinygrad), stop at `StopGradient`, reverse topo, cotangent map with
   drop-on-consume; **unbroadcast + dtype cast once in the driver**.
2. Rule signature `rule(node, outs, span<optional<Tensor>> cots, needed mask) ->
   vector<optional<Tensor>>` (`nullopt` = symbolic zero / non-differentiable);
   rules use only `fl` ops → `grad(grad(f))` free.
3. **Fix the compaction conflict**: when `eval` compacts a node on a
   differentiable path, keep its recipe (kind, attrs, input handles) — MLX
   tracer exemption / PyTorch saved-tensor semantics — so `loss.scalar()`
   mid-step doesn't destroy the backward graph. Add that regression test.
4. Explicit target list as the core API; `requires_grad` only as sugar to build
   the default target set (tinygrad 0.13, Burn 0.22).
5. Opaque ops (Custom, cuDNN backward, barriers) need a VJP rule or an explicit
   "not twice-differentiable" mark; BN as a dedicated node with closed-form
   backward (Burn).
6. Max/min/argmax masks come from the same forward node (never recomputed) and
   that node is must-save under remat.
7. `Index` VJP is scatter-add (duplicates accumulate); `IndexedUpdate(set)`:
   base gets cotangent with overwritten region zeroed; document duplicate-index
   gradient as undefined.
Don't copy: JAX's JVP+partial-eval+transpose stack, AOTAutograd/compiled
autograd (they bridge eager Python), vmap (not needed yet).

---

## 2. Effects, aliasing, mutation in functional IRs

- **PyTorch** [V]: `functionalize` rewrites in-place to out-of-place + view
  replay (no autograd inside, no `resize_` on views, no overlapping
  `as_strided`). `custom_op` `mutates_args` "MUST be accurate"; four kinds
  (functional / in-place returning arg / `out=` / mutating returning nothing
  aliased); "arbitrary aliasing not supported" (2.9: aliased outputs UB under
  compile; 2.13 deprecation). `auto_functionalized(_v2)` clones mutable args;
  Inductor's reinplace pass undoes clones when no later use of a view of the
  mutated arg exists.
- **MLIR One-Shot Bufferize** [V]: destination-passing style; whole-function
  analysis then greedy per-operand copy decisions; `materialize_in_destination`;
  dealloc separate. "Keep a single linear use-def chain."
- **XLA** [V]: `kMustAlias` (user) vs `kMayAlias` (compiler); separate buffer
  donor config. **JAX** [V]: `donate_argnums` (warns "Some donated buffers were
  not usable"); Refs: impure ops sequenced in program order, no returned refs,
  no double-passing, gradient accumulation via `vjp(...).with_refs(grad_ref)`.
- **MLX** [V]: value semantics, slices are copies; `b = a; b[2] = 0` rebinds the
  shared handle. Donation at kernel time when input has one handle & one data
  ref, MLX-owned, itemsize matches, buffer ≤ out+16KB; **graph edges count as
  refs → backward-needed values are protected automatically**.
- **tinygrad** [V]: `assign` builds STORE (void effect) + AFTER (sequenced value);
  view-assign rewrites every live tensor sharing storage.
- Pitfalls: aliasing outputs of custom ops; functionalization's view replay and
  "synthetic bases"; donation silently failing when an extra handle exists (in
  C++ any extra shared_ptr copy) — add counters; in-place on backward-needed
  values (PyTorch version counters; value semantics + refcount donation is safe
  but forces a copy).

### Minimal model for Flashlight [I]
- IR stays SSA with value semantics; user in-place ops rebind the handle to a
  new node (MLX); no user-visible aliasing views initially (planner may make
  pure-metadata ops views).
- Per-node-kind static table (next to VJP rules): `Pure | ViewOf(i) |
  MayReuse({i}) | Mutates({i})` (Custom only, PyTorch's 4-kind contract) +
  `Effect(class)` (HostRead/HostWrite/Collective/RNG/Opaque) sequenced by token
  edges (tinygrad STORE/AFTER, JAX ordered effects).
- Buffer-level: owned/external, held (reachable from live handle/param/output),
  persistent (params/optimizer state, stable addresses), donated (hint).
- **Reinplace rule**: reuse input i iff `i ∈ MayReuse`, not held, owned,
  size-compatible, no later use of i or alias in plan order (plans are totally
  ordered — no dominance machinery needed).
- Optimizer state `p = p - lr*g` reuses p's buffer when the old handle is
  rebound; add `donate()` + log donation failures.
Don't copy: functionalization/view replay, MLIR interface framework, JAX Refs (yet).

---

## 3. Memory planning and rematerialization

- **tinygrad** 61-line planner ([memory.py](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/schedule/memory.py)) [V]:
  first/last appearance over linear schedule; excludes held buffers; sizes by
  symbolic max; TLSF allocator per (device, lane) — copies and compute in
  separate lanes; buffers become arena views; logs "memory reduced X→Y MB".
- **XLA** [V]: `GlobalDecreasingSizeBestFitHeap` over live intervals; HLO remat
  strategies recompute/compress/host_offload; "CSE will undo remat — don't run
  CSE after".
- **Inductor** [V]: `inplace_buffers`, `allow_buffer_reuse` on;
  `reorder_for_peak_memory` tries LPMF/BFS/DFS orders, keeps lowest peak;
  fusion gated on peak-memory delta. SAC `policy_fn -> MUST_SAVE/PREFER_RECOMPUTE`,
  `activation_memory_budget` knapsack; "50% memory reduction by recomputing only
  pointwise ops".
- **MLX** [V]: no static planner; runtime donation + caching allocator; eval
  back-pressure (>10 active tasks or over memory limit → wait).
- Research: Checkmate (MILP) [R], Moccasin (CP, O(n) vars) [V], Coop (window
  eviction, <5% fragmentation) [V], MONeT/Rockmate/OLLA/POFO [R].
- Pitfalls: CSE undoes remat; over-recompute; scalar-ifying ops break
  partitioners; cross-stream address reuse creates false deps; online eviction
  fragments.

### Simple first planner [I]
v1: lifetimes over existing plan order (reuse use counts); exclude held set;
reinplace rule; best-fit-decreasing (XLA) or TLSF (tinygrad) offsets; one arena
per (device, stream/lane) cached by plan hash; library workspaces as `[i,i]`
intervals; log planned peak vs naive sum. v1.5: pick lowest-peak order among
current/DFS/BFS/LPMF. v2 remat: min-cut over joint fwd+bwd restricted to
fusible/cheap ops (ban matmul/conv/RNG/large reductions), after CSE or with CSE
barriers on clones; region `MemoryPolicy` save/recompute/offload; offload as a
remat strategy. Don't copy: ILP solvers, DTR/Coop online eviction, Inductor pool modes.

---

## 4. Execution layer

- **CUDA graphs** ([guide v13.4.2](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)) [V]:
  `cudaStreamBeginCaptureToGraph` (12.3), IF/WHILE conditional nodes (12.4),
  IF/ELSE + SWITCH (12.8); conditional bodies: kernel/memcpy/memset/child/
  conditional, single device; child-graph ownership move (12.9). Device graph
  launch [R 12.0]. `cudaGraphExecUpdate` needs topologically identical graph;
  per-node `cudaGraphExecKernelNodeSetParams`; `cudaGraphNodeSetEnabled`.
  Memory nodes: fixed VAs for graph lifetime, non-overlapping allocations may share.
- **XLA command buffers** [V]: records device addresses per allocation index;
  submits without update if unchanged, else updates only changed allocations;
  warmup through normal thunks (NCCL host APIs); global eviction of instantiated
  graphs; falls back under profiling.
- **XLA latency-hiding scheduler** [V]: async start/done pairing, resource type
  per collective/copy with overlap limit 1, memory limit + pressure, list
  scheduling with latency estimator.
- **PyTorch** [V]: stream capture, private pool, static input copies; capture
  ~100ms, replay ~10µs; CUDAGraph Trees share one pool, checkpoint between
  recordings; Inductor `graph_partition` on by default, `cudagraph_capture_sizes`
  (bucketing), rerecord limits; 2.12 `torch.cond` capturable, 2.14
  `torch.while_loop` + `torch.switch`.
- **NVIDIA guidance** [V]: pad first, then bucket; CPU scalars get baked in —
  keep them as device tensors; RNG seed/offset through pointers; "don't force
  CUDA graphs where they don't fit".
- **MLX streams** [V]: per-op stream; cross-stream within eval via Fences,
  across evals via per-array Events; on exception signal all events (0.32.x fixed
  eval/stream deadlocks when a primitive throws).

### Recommendation [I]
Address stability falls out of the planner (plan + fixed arena; only external
I/O and params vary). Capture whole plans via stream capture (cuDNN/cuBLASLt
launch internally), cache by (plan structural hash, shape bucket), patch only
changed external addresses on replay. **Flashlight-specific**: NVRTC kernels
must read ScalarNode values (`fl::full`, learning rate, step count) from device
buffers or include them in the cache key, else values freeze into graphs.
Start/done pairs for transfers & collectives; scheduler hoists starts, sinks
dones under the memory budget, per-resource overlap limit 1. Signal all events
on error. Don't copy: CUDAGraph Trees/pool checkpointing, device launch,
conditional nodes (until a real decode-loop case).

---

## 5. Dynamic shapes

- **PyTorch 2.14** [V]: static by default + automatic dynamic on recompile;
  `mark_dynamic`, `mark_unbacked` ("unlikely to be what you need"); 0/1
  specialize; no rank change; Python-only `ShapeEnv`; 2.14 `@dynamic_spec`.
- **JAX shape_poly** [V]: dim vars ≥ 1, polynomials/floordiv/mod, "unsound"
  equality, `InconclusiveDimensionOperation`, vars bound only from inputs
  (linear univariate) with generated assertions; `symbolic_dim_bounds` (0.11.2);
  `ragged_dot(lhs[m,k], rhs[g,k,n], group_sizes[g])`.
- **tinygrad** [V]: `UOp.variable(name, min, max, multiple_of)`, `bind`,
  `unbind_all()` → structure-only graph + var values; buffers sized by vmax;
  TinyJit cache keyed by variable *set*, not values.
- **MLX** [V]: `shapeless=True` re-infers shapes, no symbols.
- **PyTorch NJT** [V]: values + offsets(B+1) (+lengths); ragged dim tied to one
  offsets object; one ragged dim; "not under active development"; 2.10 moved
  flex_attention to `varlen_attn()`.

### Recommendation: dynamic extents, not dynamic rank [I]
Flashlight rebuilds graphs per call with concrete shapes, so guards aren't
needed for correctness — dynamic extents only matter for **caching** (NVRTC
kernels, cuDNN plans, memory plans, CUDA graphs). `SymDim{id, min, max,
multipleOf}` optionally attached to a dim; codegen passes marked extents as
kernel args; cache keys = unbound structure + bucket; planner sizes by max or
per bucket. Ragged = segmented value (values + offsets + cached max bound),
offsets checked by identity; segmented ops (ragged_dot/grouped GEMM, segment
reductions) as first-class nodes (aligns with MoE). Don't copy: ShapeEnv/sympy
guards, dynamic rank, NJT subclassing, unbacked data-dependent shapes.

---

## 6. Pass infrastructure for small IRs

- **tinygrad** [V]: `PatternMatcher` = list of `(UPat, fn)` indexed by op, with
  `early_reject` source-op sets; three `graph_rewrite` drivers; immutable
  hash-consed UOps (implicit CSE); `REWRITE_STACK_LIMIT`; `TRACK_MATCH_STATS`,
  `VIZ=1`. Gradients, symbolic simplification, scheduling and memory planning are
  all rewrites.
- **XLA pattern_matcher.h** [V]: header-only `m::` combinators with captures,
  `AnyOrder`, `AllOf/AnyOf`; capture-free pass first then re-run with capture;
  explain-on-failure (`MatchAndLogIfFailed`).
- **MLIR PDLL** [V]: own language/tooling, still evolving.
- **E-graphs**: Willsey (SIGPLAN blog 2026-10-01) — equality saturation is
  best-effort, expect white-box use with scheduling; Cranelift moved to
  restricted ægraphs; start with egglog. Luminal uses egglog; EqiForge
  (arXiv 2609.12330) 1.32× geomean [V]. CloudRift's 5k-line compiler: rules
  return lists of legal options + ~200-LOC MCTS [V].

### Recommendation [I]
Header-only C++ matcher (XLA `m::` style: captures, commutative variants,
explain-on-failure) + tinygrad-style driver (per-NodeKind dispatch, early
reject, worklist to fixed point, iteration limit, per-rule name + match counter,
before/after dumps). Rules as data `{name, pattern, rewrite}` in composable
vectors — easy for humans and agents to add/test one at a time. Possibly express
the VJP table in the same registry. For fusion/kernel choice: rules return
alternatives + small cost/measured search before considering e-graphs. Don't
copy: PDLL/DRR/TableGen, MLIR dialects, e-graphs as main rewriter.

---

## Ten takeaways for the next three milestones (VJP, effect/alias, planner) [I]
1. VJP driver: on-path set from explicit targets, StopGradient, reverse topo,
   cotangent map with free-on-consume, central unbroadcast + dtype cast.
2. Rule signature with optional cotangents/grads; rules in `fl` ops; prune zeros
   in autodiff, not via IEEE-unsafe simplification.
3. Keep differentiable nodes' recipes through eval compaction; test
   `loss.scalar()`-before-`backward()`.
4. Gradient test matrix with JIT optimizer on/off: fan-out, max/min ties (mask
   from forward node, must-save), scatter duplicates, overwritten-index zeros,
   int/bool outputs, unused siblings, broadcasting, finite differences.
5. Value semantics + rebinding for user in-place ops; SSA IR; views only as
   planner-introduced metadata.
6. Static per-node-kind effect/alias table (Pure/ViewOf/MayReuse/Mutates +
   Effect classes with token edges).
7. One reinplace rule + refcount donation (graph edges count as refs); `donate()`
   hint with failure logging.
8. Planner v1: plan-order lifetimes, exclude held, in-place reuse, best-fit or
   TLSF, per (device, lane) arenas, workspaces as `[i,i]`, log planned vs naive;
   then min-peak order selection.
9. Remat v1: min-cut over fwd+bwd restricted to fusible ops, after CSE / CSE
   barriers; region MemoryPolicy save/recompute/offload.
10. Make planner output capture-ready: stable slot IDs, fixed arena per plan
    hash, scalars/RNG in device buffers (not NVRTC immediates); per-plan CUDA
    graphs keyed by (plan hash, shape bucket) become a small follow-on.
