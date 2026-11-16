# Gather/scatter indexing — implementation plan (2026-11-02, pending approval)

## Decisions (user, 2026-11-02)
- Two primitives, `gather` and `scatter` (mode `Set` | `Add`), along an axis, with
  NumPy `take_along_axis` broadcasting. Keep the operator set minimal.
- `x(...)` with ONE tensor index follows NumPy/PyTorch semantics. Boolean-mask reads =
  `nonzero` + `gather`; boolean-mask writes = `where`. Two or more tensor indices throw
  (punted; NumPy-style pointwise semantics is the likely future, see DETR).
- `x(idx) += v` keeps NumPy/PyTorch semantics (read, add, write — repeated indices do
  NOT accumulate; that is how the JIT already builds it). Accumulation is explicit:
  `scatter(..., Add)` (NumPy `np.add.at`, PyTorch `index_put_(accumulate=True)`).

## Semantics
```cpp
// out[..., i, ...] = input[..., index[..., i, ...], ...] along `axis`.
// index: integral dtype, same rank as input; on every axis but `axis` its size
// broadcasts against input's (equal, or 1 on either side). Output shape = that
// broadcast shape, with index.dim(axis) on `axis`. Output dtype = input dtype.
Tensor gather(const Tensor& input, const Tensor& index, int axis);

enum class ScatterMode { Set, Add };
// Returns a copy of `input` where, for every position p of the (broadcast) index,
// out[p with axis coordinate := index[p]] = / += source[p]. `source` broadcasts to the
// index's broadcast shape and is cast to input's dtype. Add accumulates repeated
// indices; Set with repeats keeps one of them (unspecified which, as in PyTorch/JAX).
Tensor scatter(const Tensor& input, const Tensor& index, const Tensor& source,
               int axis, ScatterMode mode);
```
- Index values must lie in `[0, input.dim(axis))` (no negative wrap, like
  `torch.gather`). CPU checks and throws; CUDA/AF kernels do not check (documented).
- Gradients: `gather` → `dInput = scatter(zeros, index, dOut, axis, Add)`.
  `scatter(Add)` → `dInput = dOut`, `dSource = gather(dOut, index, axis)` (summed over
  source's broadcast axes). `scatter(Set)` → `dInput = scatter(dOut, index, 0, axis,
  Set)`, `dSource = gather(dOut, index, axis)` (summed over broadcast axes).

## Operator count
TensorBackend: +`gather`, +`scatter`, −`indexAdd`. IR: +`GatherNode`, +`ScatterNode`;
`IndexNode`/`IndexedUpdateNode` become static-only (spans/ranges/literals), which also
removes the mask-count device sync from shape inference. `nonzero` stays the only
data-dependent-shape op.

## Phase 1 — primitives (no change to `x(...)` behavior yet)
1. `tensor/TensorBackend.h/.cpp`: pure virtual `gather`/`scatter` (out-param style like
   the other ops); `ScatterMode` enum next to them. Remove `indexAdd` (base impl is a
   host round trip, axis 0, 3 dtypes).
2. `tensor/Tensor.h/.cpp`: free functions `fl::gather`, `fl::scatter`; remove
   `fl::indexAdd` (one caller: `autograd::embedding`, migrated in Phase 3 — until then
   it calls `fl::scatter(..., Add)`).
3. `tensor/BackendDispatch.h/.cpp/-inl.h`: two dispatch entries.
4. CPU (`backend/cpu/CPUShapeAndIndex.cpp`, `CPUBackend.h`): native loops through
   `enqueueNativeOp`; element-size dispatch for gather/Set, typed dispatch for Add
   (bf16/f16 accumulate via f32); s32/s64/u32/u64 indices.
5. CUDA (`ops/cuda/CudaShapeAndIndex.cu`, `CudaOps.h`, `backend/cuda/CudaBackend.{h,cu}`):
   one gather kernel (element-size templated, broadcast via strides — no index
   materialization) and one scatter kernel: Set = element copy; Add = `atomicAdd` for
   f32/f64/s32/u32/s64-as-u64; f16 (sm_70+) and bf16 (sm_80+) native atomics.
6. ArrayFire (`backend/af/ArrayFireShapeAndIndex.cpp`, `ArrayFireBackend.h`): gather via
   linear indices + `af::lookup` on the flattened array (or `af::lookup` along the axis
   when the index varies only along it); scatter Set via flat assignment; scatter Add
   via a linear-index `atomicAdd` kernel next to the existing `AdvancedIndex.cu`
   (AF-CUDA), host fallback on AF-CPU.
7. Stub backend: throws.
8. JIT: `ir/GatherNode.{h,cpp}`, `ir/ScatterNode.{h,cpp}` (shape/dtype computed in
   `create`), `ir/NodeType.{h,cpp}`, `ir/Node.cpp`, `ir/CMakeLists.txt`;
   `JitBackend.{h,cpp}` builds the nodes; `eval/Evaluator.{h,cpp}` evaluates them via
   dispatch (scatter writes into a copy of `input`; reusing a single-use input buffer is
   a later optimization); `opt/GraphVerifier.h` (arity/shape checks),
   `opt/passes/CommonSubexpressionElimination.cpp` (keys),
   `printer/GraphvizPrinter.cpp` (labels). Other passes leave unknown node types alone
   (verify).
9. Autograd: `autograd/Functions.{h,cpp}`: `autograd::gather`, `autograd::scatter`.
10. Tests: `test/tensor/TensorGatherScatterTest.cpp` in `_BACKEND_TEST_SRCS` (runs on
    CPU/CUDA/AF/default): shapes, broadcasting, every dtype, repeated indices (Add
    accumulates, Set keeps one), empty index, strided inputs; JIT tests (node
    construction, CSE, graph verifier); Jacobian tests for both gradients.

## Phase 2 — `x(...)` lowers to views + gather/scatter
1. New internal helper `tensor/IndexLowering.{h,cpp}`: given indices + shape, returns the
   static indices (tensor index replaced by span), the tensor index's axis in the view,
   the index reshaped for broadcasting (`[1, .., m, .., 1]`), and the NumPy result shape
   (index shape inserted; 0-d index removes the axis). Two or more tensor indices throw.
   Boolean masks: a lone mask with `x`'s shape → flat gather of `nonzero(mask)`; a 1-D
   mask on one axis → `nonzero` positions on that axis.
2. JIT (`JitTensorBase.cpp`): `SharedData::updateViewNodeIfNeeded` builds
   `IndexNode(static) → GatherNode → ReshapeNode`; `updateDataNode` builds
   `ScatterNode(Set)` (or `where` for a boolean-mask write whose value broadcasts to `x`);
   `flat(tensorIndex)` lowers the same way; the ">1 tensor index → materialize" path
   becomes a throw. `ir/IndexNode.cpp` / `IndexedUpdateNode.cpp`: tensor-index branches
   removed (and the `countNonzero().scalarSync()` in shape inference with them).
3. Bare backends (`CPUTensor::index`, `CudaTensor::index`): the tensor-index branch uses
   the helper: result = `gather(view, ...)`; the write-back map = `gather(offsetGrid(view),
   ...)` — the same primitive applied to the parent's element-offset grid — stored in the
   existing `parentIndexState_` (CPU host map / CUDA device map), so `x(idx) = v` keeps
   writing through. Deletes CPU's multi-pass orthogonal code and CUDA's old index code
   (this also fixes the CUDA dim-0 / dropped-index / strided write-back bugs; supersedes
   a local, unmerged draft written on 2026-10-27, since lost). ArrayFire `index()`: tensor index via
   `gather`; writes via AF assignment on linear indices; AF's "same size → flat" guess
   limited to boolean masks; multi-dimensional index tensors with writes in bare AF mode
   throw (punted).
4. `eval/Evaluator.cpp`: `evalIndexNode` / `evalIndexedUpdateNode` drop tensor-index
   unwrapping.
5. Tests: update `IndexTest`/`TensorBaseTest`/`CudaTensorIndexTest`/
   `ArrayFireTensorBaseTest` where semantics change (0-d index, multi-dim index tensors,
   integer index shaped like `x`); rewrite `AutogradTest.GetAdvancedIndex{,F16}` as
   single-index gradient tests plus a "multi-index throws" test.

## Phase 3 — migrate users, delete legacy
- `autograd::embedding`: forward `gather`, backward `scatter(Add)`.
- `autogradIndex` / `autogradFlat` backward: `scatter(Add)` (replaces the
  per-combination `scalarSync` loop).
- `AdaptiveSoftMaxLoss::cast`: `scatter(Set)` (gradient `gather`) — or keep `x(idx) = v`,
  which now lowers to it.
- `pkg/vision` `SetCriterion` `index()`: flat `gather` + `scatter(Add)` gradient (fixes
  the overwrite-on-duplicates gradient).
- Delete: `TensorBackend::indexAdd`, AF `advancedIndex` once AF scatter Add covers it,
  leftover tensor-index code in backends and IR.
- Optional, separate proposal: `categoricalCrossEntropy` picks the target class with
  `gather` (O(rows)) instead of a one-hot multiply (O(rows × classes)).

## Verification
- Full suite on CPU / CUDA / AF / default backend plus the new tests and Jacobians.
- Perf sanity on CUDA: embedding forward/backward before vs after (backward today goes
  through `indexAdd`'s host round trip, so expect a large win).

## Risks / open
- CUDA scatter Add on floats is non-deterministic in summation order (as in PyTorch);
  a deterministic mode can come later.
- f16/bf16 atomics need sm_70 / sm_80 (fine on H100; fall back to f32 accumulation
  elsewhere).
- Bare-AF writes through multi-dimensional index tensors: throw for now.
- Rough size: Phase 1 ≈ 1.2–1.6k lines incl. tests; Phase 2 ≈ 0.6–0.9k (net deletions in
  backends); Phase 3 ≈ net negative.
