# Research reports and reviews

Background research and reviews produced during the modernization. Several were written
in AI-assisted research and code-review sessions; where a report makes claims about this
codebase it tags them — `[REPRO]` reproduced at runtime, `[READ]` traced by reading the
code, `[SUSPECT]` plausible but unverified, `[I]` inference/recommendation. Treat
untagged statements about external projects as of the report's date.

| Report | Date | What it covers | How it is used |
|---|---|---|---|
| [review-2026-10-26.md](review-2026-10-26.md) | 2026-10-26 | Full review of the modernization work: verified critical issues, high-severity issues by subsystem, test state, cross-cutting themes, VJP readiness, **architecture maps** (§6) | Source of the 2026-10-26 fix list and several pending decisions (P-1…P-5). Many §1–§2 items are fixed; see `TODO.md` "Fixes applied 2026-10-26". |
| [research-2026-10-autodiff-compilers.md](research-2026-10-autodiff-compilers.md) | 2026-10-26 | Autodiff over lazy graphs, effects/aliasing, memory planning and rematerialization, execution layer, dynamic shapes, pass infrastructure | Input to P-1 (alias semantics), P-2 (VJP retention), planner/execution roadmap. Digest in `TODO.md`. |
| [research-2026-10-distributed-kernels.md](research-2026-10-distributed-kernels.md) | 2026-10-26 | Communication in the IR and overlap, pipeline schedules as data, MoE systems, kernel authoring from C++, attention/fused ops via libraries, low precision (FP8) | Distributed/MoE/kernels roadmap; FP8 and CuTile notes in `TODO.md`. |
| [backend-support-matrix-2026-10-26.md](backend-support-matrix-2026-10-26.md) | 2026-10-26 | Op-by-op support of every `TensorBackend` method on CPU vs CUDA vs ArrayFire (OK / partial / host / throws), dtype and semantic differences | Parity work (D-09); three entries confirmed and fixed or planned (CPU unary dtypes, CUDA `clip`, CUDA indexing). Unverified entries remain code readings. |
| [async-materialization-design.md](async-materialization-design.md) | 2026-03-27/31 | Design record: `Event`, `fl::Future`, `fl::copy`, `TransferNode`, `scalar<T>()`, explicit eval, collectives; survey of TVM/PyTorch/XLA/MLX/MLIR/CUDA | Decision D-05. |
| [vjp-design-research.md](vjp-design-research.md) | by 2026-05 | How JAX, PyTorch, MLX, tinygrad and TVM implement VJP rules, organized by design decision; recommends an MLX-like design | Input to P-2; corrected in review §5.1. |
| [tile-broadcast-research.md](tile-broadcast-research.md) | by 2026-05 | How XLA, TVM, ONNX, PyTorch and others represent tile and broadcast in their IRs | Informed `TileNode` and broadcast handling (Phase 5b). |

Plans for approved work live in [../plans/](../plans/):
[indexing-gather-scatter-plan.md](../plans/indexing-gather-scatter-plan.md) (D-11).

A conv/library-primitives design plan referenced from `TODO.md`
(`linked-zooming-ladybug.md`) was not preserved; the corresponding `TODO.md` section
("Conv2D/Pool2D/BatchNorm/RNN (Library Primitives)") holds the design.
