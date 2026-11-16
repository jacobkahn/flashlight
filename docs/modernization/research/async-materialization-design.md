# Async Materialization & Data Transfer Design

> Design record from 2026-03-27 / 2026-03-31 (decisions agreed by the project
> maintainer). Implementation status as of 2026-10-26: `Event`, `fl::Future`,
> `fl::copy`, `TransferNode`, `MultiValueNode` exist; see TODO.md and
> `docs/modernization/research/review-2026-10-26.md` for corrections found later
> (e.g. `Future::isReady()` can stay false; `TransferNode` same-device no-op fixed).

## Design decisions (confirmed 2026-03-27)

### FusionSideOutput: DEFERRED
- FusionSideOutput activation is unnecessary once functional autograd (VJP rules as
  tensor ops) eliminates ExternalUse. FusionUtils.h infrastructure remains as-is.
- MultiValueNode (implemented) is still needed for multi-output compaction regardless.

### Unified Event abstraction
- Single `Event` type with `record(stream)`, `wait()`, `isReady()` across all backends.
- Already have `CPUEvent` (promise/future-based) and `CUDAStream::recordEvent()`
  (returns raw cudaEvent_t). Need to unify under one interface.
- CUDA: wraps cudaEvent_t. CPU: wraps existing CPUEvent. SynchronousStream: no-op.

### fl::Future<T> — minimal async result (updated 2026-03-31)
- Two methods only: `get()` and `isReady()`. No `then()`, no callbacks.
- Internally: Event (for isReady), reader function (for get), cached value.
- Reader is self-contained: handles eval + sync + read. `get()` just calls reader.
- If fl::eval() was called first, eval inside reader is no-op, sync is instant.
- If fl::eval() was NOT called, `get()` triggers eval as fallback (prevents deadlock).
- `isReady()` uses `cudaEventQuery` — non-blocking, no side effects.
- Users manage their own async callbacks externally (thread pool, etc.).
- Rationale for no `then()`: passive firing requires cudaLaunchHostFunc (incompatible
  with CUDA graph capture, silent failure on stream errors) or background thread pool.
  Both add complexity. Externalizing to user gives full control over overhead.
  Pattern: `myPool.submit([f]() mutable { logger.log(f.get()); });`
- No dependent eval mechanism needed — graph dependency tracking handles everything.
  `get()` evals the transfer tensor, evaluator walks inputs, evaluates original.

### fl::copy — runtime-level unified transfer
- Lives in flashlight/fl/runtime/, NOT in BackendDispatch.
- BackendDispatch = compute ops (matmul, conv → cuDNN, oneDNN, BLAS).
- Runtime = data movement (copy, sync, events → CUDA runtime, NCCL, C++ stdlib).
- Transport routing by device pair (one function, branches internally):
  - Same GPU: cudaMemcpyAsync D2D
  - Same node, NVLink/P2P: cudaMemcpyPeerAsync
  - Same node, no P2P: stage through host
  - Cross-node: NCCL ncclSend/ncclRecv
  - CPU ↔ GPU: cudaMemcpyAsync H2D/D2H
  - CPU ↔ CPU: std::memcpy
- IR doesn't know transport — TransferNode just says "device A → device B."
- Pinned host memory pool for async D2H.

### TransferNode — IR node for data movement
- Tensor::to(device) creates TransferNode. First-class IR node.
- SSA-clean: input in, output out on destination device. Original stays valid.
- Unified (one node type). Queryable: srcDevice(), dstDevice().
- Passes inspect device pair for scheduling (same power as XLA's separate opcodes).
- Evaluator calls fl::copy directly (runtime), NOT through BackendDispatch.
- Future work: TransferStartNode/TransferDoneNode for explicit async windows.

### scalar<T>() — special case of transfer, not special infrastructure
- Inserts TransferNode(tensor, cpu_host) into graph. Does NOT trigger eval.
- Returns fl::Future<T> that reads from CPU tensor after evaluation.
- scalarSync<T>() = scalar<T>().get().
- No DataRequest, no HostCopy, no special annotations. Just to(cpu) + host read.

### fl::eval(tensors...) — explicit, non-blocking
- Enqueues computation for given tensors' subgraphs. User controls what's evaluated.
- No global state (rejected fl::step() as too opaque/framework-level).
- Multiple tensors can share subgraph nodes; evaluated once.

### fl::ready(tensor) — non-blocking query
- Checks if tensor's graph has been evaluated (node->getResult().has_value()).

### Communication primitives — separate from TransferNode
- AllReduceNode, AllGatherNode, ReduceScatterNode — different semantics (reduction,
  multi-rank). Definitely separate node types.
- SendNode/RecvNode — point-to-point for pipeline parallelism.
- NCCL point-to-point (ncclSend/ncclRecv) is just a TRANSPORT for TransferNode
  when devices are cross-node. Collectives are semantically different.

## Key design principles established
1. No special-purpose abstractions — scalar extraction uses same transfer mechanism
   as any other device-to-device copy.
2. Runtime vs compute separation — fl::copy is runtime, BackendDispatch is compute.
3. Transport is an implementation detail — IR is transport-agnostic.
4. Explicit eval, no global state — user says what to evaluate.
5. .get() on Future triggers eval as fallback — prevents deadlock if user forgets
   fl::eval(). Precedented by C++ std::launch::deferred and Rust futures.
6. Events for synchronization — cudaLaunchHostFunc rejected (incompatible with
   CUDA graph capture, silent failure on stream errors). cudaEventSynchronize
   with cudaEventBlockingSync is the right primitive.
7. Future is minimal — get() + isReady() only. No then(), no callbacks.
   User externalizes async callback patterns via their own thread pool.

## Research findings (async compute frameworks)
- TVM: cleanest unified model — one CopyDataFromTo virtual method, branches by device.
- PyTorch: unified copy_() for tensors, but .item() is separate (scalar extraction).
- XLA: deliberately separated D2H/H2D/D2D at every level (HLO opcodes, PjRt API, thunks).
  Rationale: each direction has different scheduling properties for compiler passes.
- MLX: unified memory eliminates transfers; Copy is for layout transforms only.
- MLIR: gpu.memcpy unified across address spaces, async tokens for synchronization.
- CUDA: cuMemcpy (driver) is unified; cudaMemcpy (runtime) needs direction kind.
- Key insight: unified node with queryable properties = same optimization power as
  separate opcodes. Can split later if passes need it.
