# Backend op support matrix: CPU vs CUDA vs ArrayFire (2026-10-26)

Produced by a read-only code-search pass over `TensorBackend.h` and the three backends
(nothing executed). Spot-checked by hand on 2026-10-26 and confirmed: C4 (CPU
`log1p`/`sin`/`cos`/`floor`/`ceil`/`erf` silently leave the output unwritten for
non-f32/f64 dtypes), U6 (CUDA `clip` reads only the first element of tensor bounds via
`scalarSync`), U19 (CUDA tensor indexing gathers along dim 0 regardless of the index's
axis and returns early, dropping other indices). Other entries are unverified code
readings. Per-test parity across `_AF`/`_CPU`/`_CUDA` is in TODO.md (ArrayFire decision).

## Direct answer

All three backends override most of the `TensorBackend` interface. The gaps that remain are the same on every backend: `indexAdd` and most of the scalar `clip`/`where` overloads fall through to base-class defaults. ArrayFire is the most uniform: it runs every op on device for every dtype except bf16. It is limited to 4-D, its `tril`/`triu`/`topk`/`roll` are restricted, and its advanced-index gradient works only in the CUDA build. CUDA accepts all 12 dtypes but computes most non-f64 ops in f32. The CPU backend has the most gaps: its memory-manager methods throw, it reports only 6 dtypes as supported, its oneDNN-backed ops are limited to rank ≤4, and several f64 paths fall back to host loops.

**Scope:** `TensorBackend.h` virtuals, the three backend implementations, and the op files they call (`ops/onednn`, `ops/native`, `ops/cuda`, `ops/cudnn`, `backend/af`). The JIT backend and `Tensor`-level wrappers were not analyzed. Nothing was built or run; every classification comes from reading the code.

**Legend:** OK = native implementation for the dtypes shown. PARTIAL = throws, falls back, or silently differs for some dtype or argument. HOST = copies to host memory or runs a host loop inside the op. THROWS = unconditional throw. BASE = not overridden. IDs (C#, U#, A#) refer to section 2.

## 1. Support matrix

| Method(s) | CPU | CUDA | ArrayFire |
|---|---|---|---|
| backendType, eval, jitStream | OK | OK | OK |
| createDispatch | OK (direct) | OK (direct) | BASE (trampoline, A12) |
| supportsDataType | PARTIAL C2 | OK (all 12) | PARTIAL A1 |
| getMemMgrInfo, setMemMgrLogStream, setMemMgrLoggingEnabled, setMemMgrFlushInterval | THROWS C1 | PARTIAL U1 (no-op) | PARTIAL A4 |
| isDataTypeSupported (final) | BASE (ANDs `supportsDataType` with extensions) | BASE | BASE |
| setSeed | OK | OK | OK |
| randn | OK | OK | PARTIAL A2 |
| rand | OK | PARTIAL U2 | PARTIAL A2 |
| fromScalar, full, empty | OK | OK | PARTIAL A2 |
| identity | OK | HOST U3 | PARTIAL A2 |
| arange | PARTIAL C14 | OK | PARTIAL A2 |
| iota | OK | OK | PARTIAL A2 |
| reshape | PARTIAL C10 | OK | PARTIAL A5 |
| transpose | PARTIAL C10 | PARTIAL U4, U9 | PARTIAL A5 |
| tile | PARTIAL C7, C10 | PARTIAL U4, U9 | PARTIAL A5 |
| concatenate | PARTIAL C7, C10 | PARTIAL U4 | PARTIAL A5 |
| nonzero | HOST C11 | PARTIAL U4 | OK |
| pad | OK | PARTIAL U4, U10 | PARTIAL A6 |
| flip | OK | OK | OK |
| roll | OK | OK | PARTIAL A9 |
| tril, triu | OK | OK | PARTIAL A8 |
| clip(T,T,T,T) | HOST C7 (f64) | PARTIAL U4, U6 | OK |
| clip(T,T,d,d) | PARTIAL C8 | BASE | BASE |
| clip(T,T,T,d), clip(T,T,d,T) | BASE C20 | BASE | BASE |
| where(T,T,T,T) | PARTIAL C9 | PARTIAL U4 | OK |
| where(T,T,T,d), where(T,T,d,T) | BASE C20 | BASE | BASE |
| topk | PARTIAL C3 | PARTIAL U4, U7 | PARTIAL A7 |
| sort (values), sort (values+indices), argsort | PARTIAL C3 | PARTIAL U4, U7 | OK |
| indexAdd | BASE C20 | BASE | BASE |
| binary (21 enumerators) | PARTIAL C3, C5, C6, C7 | PARTIAL U4, U5 | OK |
| unary (18 enumerators) | PARTIAL C4, C7 | PARTIAL U4 | OK |
| reduction (9 enumerators) | PARTIAL C3, C7, C12 | PARTIAL U4, U8 | OK |
| matmul | PARTIAL C7, C10 | PARTIAL U4, U20 | OK |
| min, max, argmin, argmax | PARTIAL C3 | PARTIAL U4 | OK |
| cumsum | PARTIAL C3 | PARTIAL U4 | OK |
| var | OK | OK | OK |
| norm | PARTIAL C13 | OK | OK |
| conv2dForward | PARTIAL C15 | PARTIAL U11, U12 | A13 |
| conv2dBackwardData, conv2dBackwardFilterBias | OK | PARTIAL U11 | A13 |
| pool2dForward | OK | PARTIAL U11 | A13 |
| pool2dBackward | PARTIAL C16 | PARTIAL U11 | A13 |
| batchnormForward | PARTIAL C17 | PARTIAL U11, U13 | A13 |
| batchnormBackward | PARTIAL C17 | PARTIAL U13 | A13 |
| rnnForward | PARTIAL C18 | PARTIAL U11, U14 | A13 |
| rnnBackward | HOST C19 | PARTIAL U11, U14 | A13 |
| copy | OK | PARTIAL U16 | OK |
| print | OK | HOST U17 | OK |

Only the CPU memory-manager methods throw unconditionally. The `clip`/`where` overloads that each backend does not override (CPU's `clip(T,T,T,d)`/`clip(T,T,d,T)`; CUDA's and AF's double-bound `clip` and scalar `where` overloads) reach the base implementations. Because `CPUBackend`, `CudaBackend`, and `ArrayFireBackend` each declare their own `clip`/`where` overloads, calls made through the concrete type do not see the inherited ones (a C++ name-hiding rule); calls through `TensorBackend&` do.

## 2. Conditions behind PARTIAL / HOST / THROWS / BASE

### CPU

`flashlight/fl/tensor/backend/cpu/CPUBackend.cpp`
- **C1 THROWS:** all four memory-manager methods throw `runtime_error` (lines 170-191).
- **C2 PARTIAL:** `supportsDataType` returns false for f16 and f64 (152-155) and s16, u16, s64, u64 (159-162). True for f32, bf16, s32, u32, u8, b8.

`flashlight/fl/tensor/backend/cpu/CPUBinaryOps.cpp`
- **C3 PARTIAL:** `mod` (52-66), bitwise, shift, and `power` (68-182) dispatch on dtype through `dispatchByType` and throw for bf16, s64, and u16.
- **C5 PARTIAL:** bitwise and shift throw `invalid_argument` for floats (78-79, 94-95, 110-111, 126-127, 142-143).
- **C6 HOST:** `mod`, `power`, bitwise, shift, and `logicalAnd/Or` call `cpuBinaryOpWithBroadcast`, which runs `toHostVectorSync` on both inputs (`flashlight/fl/tensor/ops/native/Binary-inl.h:115-163`, sync at 120-121). `logicalAnd/Or` are at `flashlight/fl/tensor/ops/native/Binary.cpp:13-27`.
- **C7 HOST (f64 fallbacks):**
  - `flashlight/fl/tensor/ops/onednn/OneDnnOps.cpp` 497-575 (binary; comparisons read to host at 527-528), 675-702 (eltwise; bounds converted to double at 693), 796-856 (reduction; host read at 798). The reduction's default branch at 811-813 throws for any algorithm other than sum, mean, max, min, or mul.
  - 1063-1065 (tile → `flashlight/fl/tensor/ops/native/ShapeAndIndex.cpp:118-152`).
  - 1171-1206 (concatenate, host reads at 1187 and 1196).
  - 1281-1284 (matmul → `flashlight/fl/tensor/ops/native/Matmul.cpp:27-131`, `hostSync` at 75-76, CBLAS on host).
  - f64 is marked unsupported by `supportsDataType` (C2) yet these fallbacks exist.
- **C8 PARTIAL:** `clip(T,T,d,d)` passes double bounds to `applyEltwiseOp`, whose `alpha`/`beta` are `float` (`flashlight/fl/tensor/ops/onednn/OneDnnOps.h:175-182`). Bounds are narrowed to float32 for every dtype, including f64 (`flashlight/fl/tensor/backend/cpu/CPUShapeAndIndex.cpp:92-99`).
- **C9 PARTIAL/HOST:** `where(T,T,T,T)` uses oneDNN `binary_select` only for f32 and f16 operands (`flashlight/fl/tensor/ops/onednn/OneDnnOps.cpp:1537-1540`). Other dtypes go to `whereCpu` (1583-1584), which copies to host (`flashlight/fl/tensor/ops/native/ShapeAndIndex.cpp:100-116`, reads at 106-107) and throws for bf16, s64, and u16 through `dispatchByType`. The condition must be b8 or u8 (1516-1519).

`flashlight/fl/tensor/backend/cpu/CPUUnaryOps.cpp`
- **C4 PARTIAL:**
  - `log1p`, `sin`, `cos`, `floor`, `ceil`, `erf` write output only for f32 and f64 (48-64, 66-82, 84-100, 114-130, 132-148, 189-205). Other dtypes leave the output buffer unwritten, with no error.
  - `isnan` writes zeros for other dtypes (207-224, memset at 220-222).
  - `absolute` integer path handles only s32, b8, u8, u32 (156-181). `absoluteIntImpl` throws for anything else (`flashlight/fl/tensor/ops/native/Unary.cpp:103-128`, throw at 124-126).

`flashlight/fl/tensor/backend/cpu/CPUShapeAndIndex.cpp`
- **C3 PARTIAL:** `topk` (150-164), `sort` (166-176), `sort` with indices (178-201), and `argsort` (203-226) all enqueue sort kernels that call `dispatchByType` (`flashlight/fl/tensor/ops/native/Sort.cpp:96-121`). They throw for bf16, s64, u64, and u16 inside the async worker. How that error surfaces is not verified.
- **C10 PARTIAL:** `reshape`, `transpose`, `tile`, `concatenate` (lines 28-50), and batched `matmul` throw for rank >4 through `shapeToOneDnnStrides` (`flashlight/fl/tensor/ops/onednn/Utils.cpp:93-100`). Call sites: `OneDnnOps.cpp` 935-936 (reshape), 1005-1006 (transpose), 1086-1087 (tile), 1234-1235 (concatenate), 1345-1346 (matmul batch).

`flashlight/fl/tensor/backend/cpu/CPUReductions.cpp`
- **C3 PARTIAL:** `min`/`max` (51-113) enqueue `dispatchByType` kernels. `cumsum` (134-139) and `argmax`/`argmin` (141-161) follow the same path.
- **C12 HOST + PARTIAL (median):** full-reduction median (only when `axes` is empty, 195-200) copies to host with `toHostVectorSync`. It returns an f32 result even for f64 input (203-210). Single-axis odd-length median keeps the input dtype (214-227), so integer input stays integer on that path.
- **C13 THROWS (conditional):** `norm` with f64 input and p == 2 on a non-scalar throws. The call at 347-351 reaches the default throw at `OneDnnOps.cpp:811-813`.

`flashlight/fl/tensor/backend/cpu/CPUCreation.cpp`
- **C14 PARTIAL:** `arange` for bf16 throws (lines 74-89, via `flashlight/fl/tensor/ops/native/Creation.cpp:31-66`, throw at 65).

`flashlight/fl/tensor/backend/cpu/CPUAutogradOps.cpp` (delegates to `flashlight/fl/tensor/ops/onednn/`)
- **C15 PARTIAL:** `conv2dForward` throws for f16 (`Conv2D.cpp:139-141`).
- **C16 PARTIAL:** `pool2dBackward` throws unless the forward ran with `wantState=true` (`Pool2D.cpp:177-180`).
- **C17 PARTIAL:** `batchnormForward` throws for f16 (`BatchNorm.cpp:93-95`) and requires contiguous axes (112-115). `batchnormBackward` requires state (232-235).
- **C18 PARTIAL/HOST:** `rnnForward` throws for dropout > 0 (`RNN.cpp:280-282`). f64 forward uses the manual host implementation (307-328; `flashlight/fl/tensor/ops/native/RNN.cpp:307-471`, `readHost` at 330-336).
- **C19 HOST:** `rnnBackward` always runs host BPTT for every dtype (`flashlight/fl/tensor/ops/onednn/RNN.cpp:436-439, 464-476`; `flashlight/fl/tensor/ops/native/RNN.cpp:473-`, `readHost` at 502-508).
- Conv, pool, batchnorm, and RNN run synchronously. `executeNetwork` calls `stream.sync()` and then executes on the calling thread (`flashlight/fl/tensor/ops/onednn/DnnlUtils.cpp:63-85`).

`flashlight/fl/tensor/backend/cpu/CPUTensor.cpp`
- **C21 THROWS (sparse):** the sparse constructor throws (276-285). `isSparse()` returns false (359-361).
- **C22 HOST (indexing):** tensor-valued indices copy to host. Boolean masks at 697-742 (`toHostVectorSync` at 709, `hostSync` at 726). Integer gathers at 745-801 (`hostSync` at 774). Span, range, and literal indices are views (808 onward).

`flashlight/fl/tensor/backend/cpu/CPUBackend.cpp` / `CPUReductions.cpp` / `CPUCreation.cpp`
- `copy` (`CPUBackend.cpp:312-328`) memcpys without a shape or dtype check. It is listed as OK.

### CUDA

`flashlight/fl/tensor/backend/cuda/CudaBackend.cu`
- **U1 PARTIAL:** memory-manager methods are empty stubs (197-200).
- **U2 PARTIAL:** `rand` for integer dtypes generates f32 uniform values and casts them (222-233). The CPU backend's comment (`flashlight/fl/tensor/backend/cpu/CPUBackend.cpp:220-223`) says that [0,1) floats truncate to 0 for integers. cuRAND's range is not in the repo.
- **U3 HOST:** `identity` builds a host vector and copies it (261-272).
- **U17 HOST:** `print` converts to f32 and copies to host (633-646).

`flashlight/fl/tensor/ops/cuda/CudaUnaryOps.cu` (unary, sort, topk, where)
- **U4 PARTIAL (f32 compute):** for any non-f64 input the op casts to f32, computes, and casts back (`unaryTransformF32`, 36-69). Integers above 2^24 lose precision, and s64/u64 are always affected. Specifically:
  - `logicalNot`, `isnan`, `isinf` are always f32, even for f64 input (355-368). An f64 value above f32 range reports `isinf` true.
  - `where` computes in f32 unless either operand is f64 (549-609; condition cast at 560-561).
- **U6 PARTIAL:** `clip(T,T,T,T)` reads the bounds with `scalarSync`, which takes the first element only and synchronizes (370-387; `flashlight/fl/tensor/backend/cuda/CudaTensor.cu:392-400`). A non-constant bound tensor silently yields wrong results.
- **U7 PARTIAL:** `sort`, `sort` with indices, `argsort`, and `topk` cast to f32 before sorting (615-752, casts at 617, 646, 671, 695). f64 values are ordered at f32 precision.

`flashlight/fl/tensor/ops/cuda/CudaBinaryOps.cu`
- **U4 PARTIAL:** arithmetic uses `binaryTransformF32` (95-157). Comparisons always use f32, even when an operand is f64 (256-272). Bitwise and shift ops always use f32 with int casts (274-319; `bitwiseAnd` at 276-283).
- **U5 PARTIAL:** broadcast to more than 8 dimensions throws (`makeBroadcastInfo`, 43-50, `kMaxBcastDims` = 8).

`flashlight/fl/tensor/ops/cuda/CudaReductions.cu`
- **U4 PARTIAL:** `gpuReduce` (644-696) selects f32 unless the input is f64. This applies to `sum`, `amin`/`amax`, `any`/`all`/count (966-985), `cumsum` (1012-1093), and `min`/`max` (1153-1293).
- **U8 PARTIAL:** median computes in f32 (769-888; casts at 783-787 and 851-855). For f64 input the result is f64 but carries f32 precision (804-806).

`flashlight/fl/tensor/ops/cuda/CudaShapeAndIndex.cu`
- **U4 PARTIAL:** `transpose` (116-146), `tile` (214-244), `concatenate` (272-360), `nonzero` (363-399, f32 compare), and `pad` (456-484) cast to f32 for non-f64 input.
- **U9 PARTIAL:** `transpose` (148-185) and `tile` use `ShapeInfo` with `kMaxDims` = 8 (29-37) and have no rank check.
- **U10 PARTIAL:** symmetric pad for widths larger than the dimension differs from CPU. CUDA reflects once and then clamps (`padKernel` 440-448). CPU reflects periodically (`flashlight/fl/tensor/ops/native/ShapeAndIndex.cpp:60-80`).

`flashlight/fl/tensor/ops/cuda/CudaBLAS.cu`
- **U20 PARTIAL:** `matmulComputeType` maps integer inputs to f32 (25-36). The result is f32 (138-141), not the integer input type.

`flashlight/fl/tensor/backend/cuda/CudaBackendTransfer.cpp`
- **U16 PARTIAL:** `copy` throws `invalid_argument` when neither side is on device (49-58).

`flashlight/fl/tensor/ops/cudnn/` (CUDA autograd delegates here)
- **U11 PARTIAL (dtype):** pool, batchnorm, and RNN use `cudnnMapToType` (`flashlight/fl/tensor/ops/cudnn/CudnnUtils.cpp:92-105`), which accepts only f16, f32, f64, and bf16. `kOne`/`kZero` (320-344) match. Conv uses `flTypeToCudnn` (`flashlight/fl/tensor/ops/cudnn/CudnnPlanCache.cpp:16-34`), which accepts f16, bf16, f32, f64, s32, and b8 (mapped to INT8), and throws for other types.
- **U12 PARTIAL:** `conv2dForward` requires a 4-D input (`Conv2D.cpp:81-86`). The `groups` argument is not read; the comment at 111-113 says groups are inferred from weight shape.
- **U13 PARTIAL:** batchnorm forward with f16 or bf16 input requires f32 weight and bias (`BatchNorm.cpp:101-105`). Backward throws `logic_error` unless `train` is true (229-231).
- **U14 PARTIAL:** RNN state is returned only when `wantState` is true (`RNN.cpp:102, 243-245`). Backward throws without state (264-267).
- pool2dBackward does not need state (`Pool2D.cpp:64-69`), which is different from CPU (C16).

`flashlight/fl/tensor/backend/cuda/CudaTensor.cu`
- **U18 THROWS (sparse):** constructor throws (236-237). `isSparse()` returns false (384-386).
- **U19 PARTIAL (indexing):** when a tensor index is present the loop returns early (742-743, 762). It gathers along dimension 0 or flat only (shape check 657-660, result shape 663-672), and it ignores all other indices in the list. The code does not check which axis the tensor index is on.

### ArrayFire

`flashlight/fl/tensor/backend/af/ArrayFireBackend.cpp`
- **A1 PARTIAL:** `supportsDataType` returns false for bf16 (161-163). f16 requires `af::isHalfAvailable` and a non-CPU build (157-160).
- **A2 PARTIAL:** `flToAfType` looks up with `.at()` (`flashlight/fl/tensor/backend/af/Utils.cpp:18-33`, `.at` at 32). bf16 throws `std::out_of_range` from every factory (236-273) and `randn`/`rand` (219-231).
- **A4 PARTIAL:** the memory manager is installed only when `FL_BACKEND_CUDA` is set (73-82). Otherwise the methods do nothing (169-211). A null `ostream` throws (174-176).
- **A7 PARTIAL:** `topk` requires the innermost axis (316-321).
- **A12 BASE:** `createDispatch` is not declared in `flashlight/fl/tensor/backend/af/ArrayFireBackend.h` (29-52), so the base trampoline is used (`flashlight/fl/tensor/TensorBackend.cpp:19-21`). Clip double-bound and scalar where overloads and `indexAdd` are BASE (C20).

`flashlight/fl/tensor/backend/af/Utils.cpp`
- **A3 PARTIAL:** `fromFlData` throws for f16 (switch default at 280-282). Any tensor built from a host buffer in f16 fails. Fill-based factories are not affected.
- **A5 PARTIAL:** `flToAfDims` throws for rank >4 (93-97). Transpose with more than 4 axes throws (`ArrayFireShapeAndIndex.cpp:52-56`). `concatenate` with `axis >= 4` can exceed the 4-D limit (`ArrayFireShapeAndIndex.cpp:117-123`).

`flashlight/fl/tensor/backend/af/ArrayFireShapeAndIndex.cpp`
- **A6 PARTIAL:** `pad` throws for more than 4 pad dimensions (182-185). The result's `numDims` is marked `TODO: check` (204-205).

`flashlight/fl/tensor/backend/af/ArrayFireUnaryOps.cpp`
- **A8 PARTIAL:** `tril` and `triu` throw unless `diagonal == 0` (150-153, 170-173) and the input is rank 2 (159-162, 179-182).
- **A9 PARTIAL:** `roll` checks the FL axis against `AF_MAX_DIMS` (119-122), not against the rank. An out-of-range axis indexes the shift array out of bounds (`shifts[afAxis]`, 125).

`flashlight/fl/tensor/backend/af/ArrayFireTensor.cpp`
- **A10 PARTIAL:** `index` throws for more than 4 indices (271-276). A code comment says multi-tensor indexing is untested (278-279).

`flashlight/fl/tensor/backend/af/AdvancedIndex.cpp`
- **A11 THROWS (for index gradients):** `advancedIndex` throws `"gradAdvancedIndex not implemented for cpu"` (16-24). The CUDA version is `AdvancedIndex.cu`. It is compiled only when `FL_ARRAYFIRE_USE_CUDA` is set (`flashlight/fl/tensor/backend/af/CMakeLists.txt:115-127`). It accepts f32 and f16 only (94-99) and integer indices s32, s64, u32, u64 (115-119).

`flashlight/fl/tensor/backend/af/ArrayFireAutogradOps.cpp`
- **A13 (delegated):** conv, pool, batchnorm, and RNN forward to `detail::cudnn::*` in a CUDA build (10-24) and to `detail::onednn::*` in CPU and OpenCL builds. Use the CUDA or CPU row accordingly.

### Base class (`flashlight/fl/tensor/TensorBackend.cpp`)
- **C20 / BASE:** `clip(T,T,T,d)`, `clip(T,T,d,T)` (and `clip(T,T,d,d)` on CUDA and AF) call `full(...)` to build a bound tensor of the input's shape and dtype, then call the tensor-bound `clip` (39-72). `where(T,T,T,d)` and `where(T,T,d,T)` do the same with `promoteScalarDtype` (74-96).
- **C20 / BASE `indexAdd`:** axis must be 0 (106-109). Supports f32, f64, and s32 accumulation only (135-158). Round-trips `self`, `src`, and `indices` through host (117-127, 162-166). Other dtypes throw.

## 3. Sparse tensors, strided views, advanced indexing, gradients

- **Sparse:** no `TensorBackend` virtual is sparse-specific. Sparse exists only as an adapter constructor and `isSparse()`. AF is the only backend that implements it (`flashlight/fl/tensor/backend/af/ArrayFireTensor.cpp:77-93`, 2-D only; `isSparse` 200-202). AF's sparse matmul is at `flashlight/fl/tensor/backend/af/ArrayFireBLAS.cpp:65-92`. CPU and CUDA throw (U18, C21).
- **Strided views:** CPU stores row-major strides and an offset-free buffer (`CPUTensor.cpp:808` onward). Native kernels copy non-contiguous input first (`flashlight/fl/tensor/backend/cpu/CPUBackendUtils.h:30-37`). CUDA stores element strides plus an offset (`flashlight/fl/tensor/backend/cuda/CudaTensor.h:56-60`). Most kernels call `asContiguousTensor`, and `copy` uses a strided gather for rank ≤8 (`CudaTensor.cu:331-364`). AF wraps `af::array` in lazy proxies that promote on first access (`ArrayFireTensor.cpp:141-161`).
- **Advanced indexing:**
  - CPU: range, span, and literal indices are views. Tensor indices (masks and integer gathers) copy to host (C22). Writes go back through `scatterToParent` (`CPUTensor.cpp:175-195`). Mixed indices are handled in passes (599-657).
  - CUDA: a single tensor index is gathered on device (U19). Mixed indices are dropped.
  - AF: native `af::index` with lazy writes through the proxy (`ArrayFireTensor.cpp:271-345`). Multi-tensor indexing is marked as untested (A10).
- **Gradients:** the only indexing-gradient kernel is AF-CUDA (A11). It is scatter-add with `atomicAdd` (`AdvancedIndex.cu:77`), restricted to f32 and f16. AF-CPU/OpenCL throws. CPU and CUDA have no backend-level indexing-gradient op in `TensorBackend`. I did not locate the autograd call sites of `detail::advancedIndex`, so I cannot say which path a native backend takes for an index gradient.

## 4. Cross-backend differences in dtype or semantics

- **Median:** CPU full-reduction returns f32 for f64 input (CPUReductions.cpp:203-210); CUDA returns f64 (CudaReductions.cu:804-806). CPU odd-length axis median keeps integer dtype (214-227); CUDA and AF return f32 for integers (CudaReductions.cu:883-885; `flashlight/fl/tensor/backend/af/ArrayFireReductions.cpp:233-235`).
- **Integer matmul:** CUDA returns f32 (U20). CPU's oneDNN output type is `getTypeWithLargerRange` of the inputs (`OneDnnOps.cpp:1343-1346`). Whether oneDNN accepts s32 operands is not verified.
- **Index dtypes:** CPU and CUDA return s32 indices for sort, argsort, topk, and nonzero (`ops/native/Sort.cpp`, `CudaUnaryOps.cu:625-627`, `CudaShapeAndIndex.cu:386-387`). CPU and CUDA return u32 indices for min, max, argmin, and argmax (`CPUReductions.cpp:67-68`; `CudaReductions.cu:1194, 1265`). AF returns whatever ArrayFire produces; no conversion is done in the backend (ArrayFireShapeAndIndex.cpp:170-174; ArrayFireReductions.cpp:112-144).
- **Random integers:** CPU generates full-range integers (`CPUBackend.cpp:224-275`) and claims to match AF's `randu`. CUDA truncates f32 values (U2).
- **Clip bounds:** CPU narrows double bounds to float32 (C8). CUDA reads only the first bound element (U6). AF uses element-wise `clamp` (`ArrayFireUnaryOps.cpp:103-111`).
- **Symmetric pad:** CUDA and CPU differ when pad width exceeds the dimension (U10).
- **RNN with integer input:** CPU casts to f32 and returns f32 (`flashlight/fl/tensor/ops/onednn/RNN.cpp:330-334`). CUDA throws (U11).
- **Copy:** AF rebinds the `af::array` handle (`ArrayFireBackend.cpp:394-400`) rather than writing into the destination's existing device buffer. Raw pointers taken earlier would not see the new data. This is a code-reading observation only.

## 5. Summary: what each side has that the other lacks

**ArrayFire has, native lacks**
- Sparse storage and sparse matmul (CPU and CUDA throw; C21, U18).
- Integer arithmetic in native integer types. CUDA computes non-f64 ops in f32, so int32 above 2^24 and all 64-bit integers lose precision (U4).
- Element-wise tensor bounds for `clip` (CUDA reads only the first element; U6).
- Advanced tensor indexing on any axis (CUDA handles dimension 0 or flat only; U19).
- On-device median and norm for f64 (CPU runs median on host, and norm with f64 and p == 2 throws; C7, C13).

**Native has, ArrayFire lacks**
- bf16 on every op (AF: no bf16; A1, A2). CPU's bf16 is also partial (C3, C4, C9, C14).
- f16 on CUDA with cuDNN conv, pool, batchnorm, and RNN, and cuBLAS f16 GEMM with f32 accumulation (`CudaBLAS.cu:180-196`). The AF build uses the same cuDNN code, but it is limited to rank ≤4 (A5) and needs a half-capable device (A1).
- Rank above 4 for element-wise ops on CUDA: broadcast to 8 dimensions (U5). CPU and CUDA also run `flip`, `roll`, `tril`, `triu`, `pad`, `cumsum`, and `sort` at any rank (C4, U4).
- `tril`/`triu` for any rank ≥2 and any diagonal (CPU `CPUShapeAndIndex.cpp:113-139`; CUDA `CudaUnaryOps.cu:475-547`). AF is limited (A8).
- `topk` on any axis (CPU via sort-and-slice, CUDA via full sort and gather). AF requires the innermost axis (A7).
- pool2dBackward without saved state on CUDA (U11 note).
- RNN dropout on CUDA (`CudnnUtils.cpp:233-252`). CPU RNN dropout throws (C18).

**Shared gaps:** `indexAdd` is BASE on all three backends and runs through host (C20). The scalar `clip`/`where` overloads are BASE on CUDA and AF.

## 6. Unverified or ambiguous

- Nothing was executed. Every claim comes from reading code.
- AF: whether bitwise and shift on float arrays throw (`ArrayFireBinaryOps.cpp:97-101`), the dtype of AF index outputs, `af::replace` argument semantics for `where`, `af::count` returning u32, and whether AF `copy` writes through to existing views.
- CPU: whether oneDNN accepts s32 or s8 for matmul, eltwise `clip`, and eltwise `abs`/`exp`. These paths have no guard, so an uncaught `dnnl::error` would propagate. The CPU f16 and f64 behavior of conv, pool, and batchnorm is not guarded explicitly; `dnnlMapToType` was not read.
- How exceptions in enqueued CPU tasks (C3) reach the caller.
- cuRAND's output range (I relied on the CPU comment, not NVIDIA docs).
- The autograd call sites of `detail::advancedIndex` and any native index-gradient path.
- Whether the `Tensor` layer gates unsupported dtypes before calling backend methods. I did not read the `Tensor`-level dispatch.