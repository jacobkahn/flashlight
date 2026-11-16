# Claude Code session, 2026-10-26 → 2026-11-16 (digest)

One long session. The detailed record is [../LOG.md](../LOG.md) §2–§6 and
[../DECISIONS.md](../DECISIONS.md); this file keeps the parts not captured there: the
overall assessment of the project given on 2026-10-26, and how the decisions were framed.

## Assessment of the project (2026-10-26, after the review and bug-fix pass)

**The direction is right.** A lazy-by-default C++ runtime where a whole training step is
a graph that people and AI agents can transform is the right niche. Explicit eval,
semantic IR nodes lowered to vendor libraries, region policies instead of per-op knobs,
and collectives as IR nodes are bets the field has converged on. The design documents
are unusually good: decisions come with their rationale, which is what agents need.

**The execution has outrun the foundations, in four ways.**
1. *Correctness wasn't verified as features landed.* Training was a no-op and a test
   passed only because nothing changed; several fusion passes computed wrong results
   while their structure-only tests passed; the June triage misattributed root causes.
2. *It is no longer simple.* `fl/tensor` grew from ~19k to ~52k lines; the optimizer is
   ~8k lines with overlapping fusion paths; one `a + b` passes through ~10 layers; adding
   a node type touches ~14 files. Much fusion work (backward cuBLASLt epilogues, the large
   cuDNN lowering) was built before the foundations it depends on.
3. *"JIT-only" is half-migrated.* Graphs are lazy, but execution still goes through the
   old eager per-op backends: CPU async tasks plus a blocking sync on every op, and deep
   copies whenever a CPU tensor is copied. That is where the race lived, and CPU gets no
   async overlap.
4. *Core semantics are unspecified:* value vs alias for tensor copies, mutation, the
   dtype contract between graph and backends, stream behavior. Gradients, memory
   planning and scheduling depend on these.

**Recommended order:**
1. Settle the semantics and add an end-to-end "loss decreases" training test on CPU and
   CUDA.
2. Correctness tooling: differential tests for fusion passes (fused vs unfused vs
   reference on random inputs); a debug-build check after each node that results match
   node shape/dtype; ASAN/TSAN configs; a CPU thread policy.
3. A simplification pass before new features (a concrete deletion list for approval).
4. Then VJP, with an explicit answer for how the forward graph survives evaluation.
5. Then the execution layer (streams, events, async start/done) — with one CUDA stream
   nothing overlaps, so this precedes communication work. FP8 on H100 can start any
   time.

**Recommendation for the training no-op (P-1, still open):** keep value semantics for
tensors, add an explicit `alias()` for parameters (used by modules and optimizers), have
optimizers update in place, and stop sharing autograd state on plain copies. The
alternative is making every copy an alias, as the old `Variable` did. The `detach()` bug
follows from whichever is chosen. A disabled test checks that two SGD steps change the
model.

## How decisions were taken
The maintainer asked to take open decisions one at a time, least consequential first,
each presented as options with a recommendation:
1. CPU thread count → environment only, plus an opt-in test option (D-07).
2. TF32 → strict by default, env opt-in, hidden in CUDA internals (D-08); a per-op
   `MathMode` attempt was reverted.
3. ArrayFire → native defaults, ArrayFire opt-in and kept as the baseline (D-09, D-10).
4. Indexing → two primitives, gather and scatter (D-11), after a survey of in-repo uses
   and other frameworks.

Still queued, in the maintainer's order: Tensor copy semantics for parameters (P-1),
then the VJP graph-retention model (P-2); also simplification scope (P-5) and JIT thread
safety (P-4).

## Requests that shaped the notes
- 2026-11-16: write an extensive log so the work can be recovered in the open-source
  setting → `docs/modernization/`.
- 2026-11-16: the working copy will be checked in wholesale and pushed to a fork,
  development continues on other machines without this session, and the blob will later
  be split into per-feature commits → this `history/` directory and
  [../CONTINUING.md](../CONTINUING.md).
