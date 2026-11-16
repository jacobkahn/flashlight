# Research: communication, pipeline schedules, MoE, kernel authoring, attention, low precision (2026-10-26)

Web survey for Flashlight's large-scale-training roadmap. Tags: **[V]**
verified against linked source this session; **[B]** background knowledge, not
re-checked; **[I]** inference / recommendation. Versions: NCCL 2.32.3 · CUDA 13.4
U2 (cuBLAS 13.8) · cuDNN 9.27 · cudnn-frontend 1.32 nightlies · CUTLASS 4.8.1
(2026-10-15) · JAX 0.12.0.1 · PyTorch 2.14.

**TODO.md items flagged as outdated:**
- CuTile note ("CUDA 12.8+"): CUDA Tile shipped 13.1 (Python), C++ tile kernels
  13.3, NVRTC can compile them since 13.3 (`--enable-tile`, `nvrtcGetTileIR`);
  13.2 support list omits Hopper. [V]
- FP8 "wait for APIs to stabilize": cuBLASLt has Hopper FP8 tensorwise, rowwise
  and 1×128 / 128×128 block scaling since 12.9 (May 2025). [V]
- "VJP(allReduce) = allReduce" is under-specified: it needs replication typing
  (varying vs invariant per mesh axis); with an invariant cotangent the gradient
  is over-counted N×. [B mechanism; V `check_vma` exists]

---

## 1. Communication in the IR and compute/comm overlap

### How leading systems do it
- **XLA/JAX** [V]: generic `async-start/async-update/async-done` HLO wrapping
  any instruction (start returns aliased operands, result, context; one user per
  start/update), plus dedicated pairs (copy-start/done, collective-permute).
  Latency-hiding scheduler (off by default; `-O1` enables it with collective
  pipelining, while-loop double buffering, analytical SoL estimator on
  Hopper/Blackwell); combine thresholds (≥ one layer of weights per AG/RS);
  windowed-einsum collective matmul; reruns under tighter memory limit on OOM;
  PGLE measured-latency feedback. PP via collective-permute; `psend/precv`
  deadlock on cycles. Feb 2026: `CollectiveBackendAssigner` routes
  collective-permute and small all-reduces (<~16 MB) to NVSHMEM inside NVLink
  domains (ring attention 256K on GB200 +30-36%). Shardy default since 0.7.0;
  `shard_map` stable 0.8.0; explicit mesh axes default; 0.12 adds
  `jax.sharding.no_collectives()`; `check_vma` tracks replication.
  ([async ops](https://openxla.org/xla/async_ops), [flags](https://openxla.org/xla/flags_guidance),
  [GPU tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html))
- **PyTorch** [V]: async TP (Sep 2024) splits AG→matmul into per-shard copy-engine
  `cudaMemcpyAsync` + sub-matmuls on two alternating streams (SM-using NCCL
  kernels push cuBLAS into extra waves); up to +29% fwd (8B), ~+8% e2e; needs
  NVSwitch, intra-node. **SymmetricMemory** (alpha, 2.11): `empty` +
  collective `rendezvous` (same order on all ranks), handle exposes peer
  pointers, multicast pointer, signal pads; backends NVSHMEM/CUDA/NCCL; ops
  `multimem_all_reduce_`, `one_shot_all_reduce`, `all_to_all_vdev(_2d)` for MoE;
  copy-engine collectives need NCCL ≥2.28. **SimpleFSDP**: FSDP from
  parametrizations + SAC + DTensor, fully traceable, Inductor bucket/reorder
  passes; up to −28.5% memory, +68.7% throughput vs FSDP2(+others).
  **TorchTitan update (2026-10-22)**: FlexShard (per-bucket placements;
  `BlockShard` aligned with FP8 blocks → +20% on 405B/256 H100s; `Owned` → cheap
  distributed Muon), opt-in NCCL backend with one-sided RMA windows,
  symmetric-memory FSDP buffers → copy-engine AG, CUDA-graphable deterministic
  PP rewrite, bitwise-matching "GraphTrainer".
- **Megatron-Core** [V]: DP RS/AG chunked per layer (128 MB buckets),
  `fp8_param_gather`; TP bulk + ring-exchange P2P via userbuffers ("require
  expert knowledge"); PP P2P overlap only in 1F1B steady state. MoE report
  (arXiv 2603.07685): EP A2A hidden by pairing one microbatch's forward with
  another's backward on separate streams + W/D split of backward MLP → 93%
  overlap, EP comm 30-40% → <5% (DeepSeek-V3/H100); DeepEP's 20 SMs cost ~20%
  GEMM efficiency.
- **NCCL 2.32.3** [V]: device API (2.28): LSA, multimem (NVLS), GIN (2.28.7);
  windows via `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)`; only
  building blocks (no device AllReduce); 2.29 host getters for raw peer
  pointers usable by any kernel; GIN kernels must be rebuilt per NCCL version;
  2.29.7 Apache-2.0 + suspend/resume; 2.30.3 TMA-offloaded symmetric kernels,
  graph-capturable `ncclPutSignal/WaitSignal`; 2.32.3 hierarchical copy-engine
  AllGather, socket GIN. One-sided host API on windows; AlltoAll/Gather/Scatter
  built in. **cuBLASMp** fuses AG+GEMM, GEMM+RS, GEMM+AR (multicast on
  Hopper/NVSwitch).
- **In-kernel comm / megakernels** [V]: DeepEP V2 (Apr 2026) moved NVSHMEM→NCCL
  GIN, all kernels JIT-compiled at runtime (CUDA 13.1+, C++20), SMs 24→4-6;
  Hazy megakernel (TP Llama-70B on 8×H100, no NCCL, atomic work queue, research
  code); MPK (persistent SM-level task graph, inference only); Triton-distributed.

### Pitfalls
SM contention (wave quantization) → prefer copy engines/TMA; decomposed matmuls
lose efficiency unless tiles survive; overlap raises peak memory (needs a memory
cap); static schedules need real costs (PGLE); collective issue order must match
across ranks (also symmetric allocations); collective gradients need replication
typing; NCCL GIN version coupling; Flashlight's single CUDA stream makes overlap
impossible until a comm stream + events exist.

### Recommendation [I]
Two levels: **semantic IR has collectives only** (AllReduce, AllGather,
ReduceScatter, AllToAll, RaggedAllToAll, CollectivePermute, Send, Recv — each
naming a mesh axis/group); a lowering pass turns them into generic
`AsyncStart/AsyncDone` pairs (token-producing start, value-producing done).
One-sided put/get + signals live only in the kernel/library layer via a
first-class `SymmetricBuffer` resource (NCCL window + peer/multicast pointers).
Lowering order: (1) NCCL host collectives on a dedicated comm stream with
events; (2) symmetric windows + zero-CTA copy-engine collectives for intra-node
FSDP AG/RS; (3) collective-matmul decomposition pass (~100-line reference)
and/or cuBLASMp library node; (4) later NVRTC-generated NCCL device-API kernels.
List scheduler hoisting starts / sinking dones under memory cap + SM budget
(`SchedulePolicy`), analytical cost model refined by measured `Event` timings,
one bucketing threshold. Replication typing per mesh axis so collective
gradients are correct by construction (RS ↔ AG adjoint, A2A self-inverse,
send/recv reverse). Don't copy: XLA's flag zoo, Megatron userbuffers configs,
GSPMD/Shardy propagation, DTensor per-op rules, a separate NVSHMEM dependency.

---

## 2. Pipeline-parallel schedules as data

- **PyTorch** `torch/distributed/pipelining/schedules.py` [V]:
  `_Action(stage, computation_type, microbatch, sub_actions)` printed like `2F0`;
  types F, I (input grad), W (weight grad), B, UNSHARD/RESHARD,
  SEND_F/RECV_F/SEND_B/RECV_B, WAIT_SEND_*, OVERLAP_F_B (DualPipeV container),
  REDUCE_GRAD. Compute-only per-rank lists lowered by passes:
  `_add_unshard_reshard`, `_add_send_recv` (round-robin simulation;
  sender places receiver's recv → no hangs; errors on malformed schedules),
  `_merge_bw`, `_add_reduce_grad`, `_add_wait_send`(+budget),
  `_defer_recv_ops`. Built-ins: GPipe, 1F1B, Interleaved1F1B, LoopedBFS,
  InterleavedZeroBubble, ZBVZeroBubble, DualPipeV.
- **Zero Bubble** (arXiv 2401.10241) [V]: B/W split + automatic search under a
  memory budget; up to +23% over 1F1B at equal memory. **DualPipe/V** [V]:
  bubble (PP/2−1)(F&B+B−3W); 2× params, PP+1 activations; hand-written
  `overlapped_forward_backward` per module. **Megatron** prefers 1F1B + paired
  F/B A2A overlap with W/D split.
- Pitfalls: send/recv deadlock; W/I split requires autograd that can emit
  weight grads separately (Flashlight's VJP graph provides this); schedule memory
  profiles differ; zero-bubble assumes equal F/B/W costs; fully unrolling every
  microbatch blows up graph size.

### Recommendation [I]
Copy PyTorch's design almost exactly in C++: `struct Action { int stage; Op op;
int mb; std::vector<Action> sub; }`, schedule = per-rank `vector<vector<Action>>`,
30-80-line generators (1F1B, interleaved, ZB-H1, ZB-V, DualPipeV), lowering
passes (Send/Recv as async start/done, Unshard/Reshard as AllGather nodes,
ReduceGrad) and a simulator for deadlock detection + memory/time estimates from
measured costs. A partition pass splits the fwd+bwd graph by a stage region
policy into F_s/I_s/W_s templates (I/W split from VJP); executor instantiates
templates per microbatch; OVERLAP_F_B runs two templates on two streams;
optional fully-unrolled mode for small research cases.

---

## 3. MoE systems

- **Megatron MoE** [V]: dispatchers allgather/alltoall/flex (DeepEP/HybridEP);
  3-stage permutation fusion (row-ID map once, permute, unpermute with FP32
  accumulation); fused router (3 kernels); grouped GEMM via multi-stream
  cuBLASLt, CUTLASS grouped, cuBLASLt grouped with device shapes
  (graph-capturable), CuTe DSL on Blackwell; main open problem = host readback of
  per-expert token counts; CUDA graphs only per layer.
- **DeepEP V2.5** [V]: `EPBuffer(...)`; `dispatch(x, topk_idx, topk_weights,
  num_experts, …, expert_alignment, do_cpu_sync, defer_epilogue) -> (recv_x,
  recv_topk_idx, recv_topk_weights, handle)`; `combine(x, handle, bias,
  topk_weights)`; handle reused in backward (combine-bwd = dispatch); FP8
  dispatch carries scales; analytic SM count.
- **Hybrid-EP** (Feb 2026) [V]: TMA for NVLink, IBGDA for NIC; ~8 SMs saturate
  DGX-H100 NVLink, ~4 a 400G NIC; worst-case registered buffer; host sync
  unless predefined. **NCCL EP** (arXiv 2603.13606) [V]: `ncclEpDispatch/Combine`
  on the device API, low-latency and high-throughput modes.
- **cuBLASLt grouped GEMM** [V]: experimental 13.1 (Blackwell), Hopper since 13.2
  U1 (BF16/FP16/FP8 tensorwide + VEC128/BLK128x128); device-side shapes/pointers;
  default epilogue only; known Hopper issues (workspace-reuse hang open; host
  pointer mode >2 waves fixed 13.4 U1; k=0 with per-batch scales IMA).
- **cuDNN `moe_grouped_matmul`** (9.18+) [V]: token, weight `[E,K,N]`,
  `first_token_offset[B*E+1]`, `token_index`, `token_ks`; modes NONE / GATHER
  (fold pack into matmul) / SCATTER / COMBINE (atomics, nondeterministic); FP8
  fwd; fusions (SwiGLU, AMAX) 9.21; bwd FP16/BF16 only (dW needs 9.22 + cuBLASLt
  13.5); Hopper support not stated.
- **DeepGEMM** [V]: m-grouped contiguous (M-block padded), masked (decode),
  k-grouped (weight grads); SM90 FP32 scales, NT only; runtime NVCC JIT.
  **CUTLASS 4.x** [V]: ragged contiguous grouped GEMM + `MoEProblemShape` (4.3),
  SM90 grouped EVT epilogues (4.6), device problem shapes in DSL (4.4).
- **Ragged representations** [V]: PyTorch `F.grouped_mm(a, b, offs)` (2D×3D,
  3D×3D, 2D×2D); JAX `ragged_dot(..., group_sizes, group_offset)`,
  `ragged_all_to_all(operand, output, input_offsets, send_sizes, output_offsets,
  recv_sizes)` (receiver-view offsets; gradient assumes non-overlapping slices).
- Pitfalls: host round-trip for token counts (blocks overlap + graphs);
  per-expert alignment (M-block, FP8 dims %16, 16-B pointers); top-k duplicates
  summed in FP32, atomic combine nondeterministic; EP kernels compete for SMs;
  grouped GEMM on Hopper is new with open bugs.

### Minimal op set [I]
Data model `SegmentedTensor` = dense `[capacity, H]` buffer (static capacity),
device `offsets[G+1]`, optional host upper bound — maps 1:1 to cuDNN
`first_token_offset`, PyTorch `offs`, JAX `group_sizes`; CUDA-graph friendly.
Ops: (1) `route(logits, k, scoreFn) -> (topkIdx, topkW)` (generated fused
kernel; aux losses separate); (2) `dispatchPlan(topkIdx, E, align, epAxis) ->
DispatchHandle` (counts, exclusive scan, row-ID map + reverse, all on device;
capacity from `LayoutPolicy`); (3) `pack(x, h) -> SegmentedTensor` (optional
fused FP8 quantize emitting scales); (4) `exchange(seg, h, axis)` semantic
ragged all-to-all (NCCL A2A first, DeepEP-V2/NCCL-EP/symmetric kernel later);
(5) `groupedMatmul(seg, W[E,K,N], epilogue)` (cuBLASLt grouped ≥13.2U1 or cuDNN
GATHER mode, reference loop over cuBLASLt); (6) `segmentedActivation` (SwiGLU,
masked past last offset); (7) `combine(y, h, topkW)` (FP32 accumulation,
deterministic gather reference). Gradients: pack/combine adjoint, exchange
reverse, dX = groupedMatmul(dY, Wᵀ), dW = k-grouped 2D×2D. Don't copy:
MegaBlocks block-sparse format, Megatron's three dispatchers, DeepEP's
Python buffer management.

---

## 4. Kernel authoring options (driving from C++ at runtime)

| Option | From C++ at runtime? | Hopper | Notes |
|---|---|---|---|
| NVRTC CUDA C++ (13.4) | yes | yes | C++20/23; bundled toolkit + CCCL headers (`--use-bundled-headers`, 13.3); PCH (12.8+), `-minimal`, LTO-IR + nvJitLink |
| CUDA Tile / cuTile | yes: NVRTC `--enable-tile`, `nvrtcGetTileIR` (13.3+), Tile IR via `cuModuleLoad` | not in 13.2 list; 13.3 per press (partly verified) | `__tile_global__`, `cuda::tiles`; Tile IR MLIR dialect Apache-2 |
| CUTLASS 4.8.1 | CuTe C++ via nvcc (NVRTC [B]); CuTe DSL Python with AOT `export_to_c` (4.4) | yes | grouped + block-scaled GEMMs |
| Triton / Gluon | needs Python to compile | yes | Gluon exposes layouts/smem/warp specialization |
| Helion (beta 2025-10) | no | yes | compiles to Triton, ~10-min autotune |
| TileLang | Python on TVM; NVRTC backend | yes | DeepSeek TileKernels (Apr 2026) |
| ThunderKittens 2.0 | C++ headers via nvcc | yes | production at Cursor/Together |
| Mojo 1.0 | no C++ embedding | partial | not a fit |

Runtime C++ codegen is mainstream (DeepGEMM, DeepEP V2 compile everything at
runtime). Pitfalls: 0.1-2 s compile latency → disk cache keyed by (source hash,
arch, flags, toolkit) + PCH + `-minimal`; `sm_90a` needed for wgmma/TMA/
setmaxnreg (not forward-compatible); Python DSLs drag Python into build/runtime.

### Recommendation [I]
Primary: a narrow kernel-family generator in C++ emitting CUDA C++ for NVRTC —
indexed/segmented map, segmented reduce/scan (CUB block/warp via bundled CCCL),
pack/unpack, gather/scatter, permutation, quantize-with-scales,
activation(+backward) epilogues, persistent atomic-queue kernels; persistent
disk cache. Secondary (evaluate): cuTile C++ through NVRTC for tile-shaped glue
once Hopper + gather/reduce coverage confirmed. Don't generate tensor-core
GEMMs/attention; vendor a CuTe-DSL `export_to_c` artifact if one top kernel is
ever needed.

---

## 5. Attention & fused ops via libraries

- **cuDNN SDPA** [V]: Hopper FP8/FP16/BF16 prefill, decode, backward (d ≤ 256,
  192/128 MLA), deterministic backward; layouts BHSD/BSHD/padded/ragged THD
  (`set_ragged_offset`); causal, sliding window, ALiBi, softcap, bias, 128×128
  block masks (UNIFIED engine); sinks (9.13+); `score_mod`; paged KV (fwd).
  FP8 fwd limited to causal/padding masks, bwd causal-only without dropout.
  Traps: ragged bwd needs `max_total_seq_len` covering full span ("corrupt
  gradients"); base-2 softmax stats not accepted by bwd (9.27); decode+GQA+causal
  NaNs; avoid 9.10.0/9.10.1.
- **FlashAttention** [V]: FA3 beta, Hopper, FP8 fwd only; FA4 (CuTe DSL)
  targets Blackwell (up to 1.3× vs cuDNN 9.13 on B200); both Python-only.
- **cuBLASLt FP8 epilogues on Hopper** [V]: A/B/C/D scales, amax D/Aux, bias,
  ReLU, GELU(+aux). Grouped GEMM: default epilogue only (13.1).
- Pitfalls: cuDNN graph build/plan time is significant → **cache plans** [B];
  version-specific bugs → pin minimum versions; FP8 SDPA scale/descale/amax
  plumbing.

### Recommendation [I]
`SDPANode` lowering to cuDNN (ragged from `SegmentedTensor` offsets; window,
sinks, GQA as attributes; FP8 later with explicit scale/amax tensors matching
`scaled_matmul`); minimum cuDNN ~9.19+. cuBLASLt epilogues for dense layers;
other epilogues as NVRTC kernels. Don't build attention kernels or port FA3/FA4.

---

## 6. Low precision

- **H100 cuBLASLt** [V]: tensorwide FP8 (11.8+); 12.9: outer-vector
  (rowwise/channelwise), 1×128 1D and 128×128 2D block scaling (A/B may mix);
  up to 1.75× vs BF16 (H200).
- **Blackwell** [V]: NVFP4 (E2M1, 16-elem blocks, UE4M3 scales), MXFP8 (32-elem,
  UE8M0), fused output quantization; experimental MN×K4 scale layouts (13.4).
  No MXFP8/NVFP4 hardware path on H100 [V by omission / I].
- **Transformer Engine recipes** [V]: DelayedScaling (amax history);
  Float8CurrentScaling (stateless, "simplest"; SP needs amax all-reduce; Ada/
  Hopper need physical transpose for backward); Float8BlockScaling (SM90+;
  1D act/grad, 2D weights; FP32 power-of-two scales; Hopper block GEMMs
  software-emulated; compact vs swizzled layouts); MXFP8 (Blackwell only);
  NVFP4 training (SM100/103; 2D 16×16 weight blocks, stochastic rounding for
  grads, 16×16 RHT for wgrad, keep last layers high precision).
- **torchao** (Llama3-8B, 8×H100) [V]: tensorwise + FP8 all-gather +25%,
  rowwise +10%. **Megatron**: FP8 10-25% e2e; FP8 saved activations −12%.
- Pitfalls: wgrad transpose (Hopper TN only; 1D blocks not transposable →
  requantize from high precision); alignment (dims %16, 16-B pointers); scales
  multiply (reciprocals); amax sync across TP/SP; delayed scaling staleness;
  cuBLAS bugs to pin around (Hopper FP8 hang beta≠0 & scale_C=0 fixed 13.2;
  NVFP4 ignoring D_SCALE 13.2U1).

### Recommendation (now, not deferred) [I]
Dtypes `f8e4m3`, `f8e5m2` (later `e8m0`, packed `fp4e2m1`). Ops
`quantize(x, ScaleSpec) -> (q, scale)`, `scaled_matmul(aq, as, bq, bs,
out_dtype, out_scale?, bias?, epilogue?) -> (y, amax?)`, `dequantize`,
`swizzleScales` (explicit). `ScaleSpec ∈ {Tensorwise, Rowwise, Block1D128,
Block2D128, MX32, NV16}` mirroring cuBLASLt modes; scales are ordinary tensors.
Recipes as `NumericPolicy`: BF16; FP8 tensorwise current scaling; FP8 rowwise;
DeepSeek-style block scaling (1×128 act/grad, 128×128 weights, E4M3, pow-2
scales); skip delayed scaling initially; Blackwell recipes later; emulate MX/NVFP4
on H100 via quantize→dequantize for numerics research only. Quantize both
orientations from high precision as explicit nodes (fusable into producer
epilogues). FP8 param all-gather with FSDP shards aligned to quantization
blocks (FlexShard); FP8 EP dispatch carrying scales. Don't copy: TE's
Float8Tensor with amax registries and global autocast, torchao subclasses.

---

## Ten takeaways (priority order) [I]
1. Build the execution layer first: comm stream, events, generic
   `AsyncStart/AsyncDone` lowering (single CUDA stream ⇒ no overlap possible).
2. Semantic comm IR = collectives only (incl. RaggedAllToAll,
   CollectivePermute); one-sided put/get only in kernel/library layer via a
   `SymmetricBuffer` resource (NCCL windows).
3. Replication typing (varying/invariant per mesh axis) so collective gradients
   are correct (fixes the AllReduce backward note).
4. Overlap scheduler: list scheduling under memory cap + SM budget, analytical
   cost model refined by measured Event timings, one bucketing threshold.
5. Pipeline schedules as data: per-rank Action table (F/I/W/B, Send/Recv,
   Unshard/Reshard, ReduceGrad, OverlapFB), small generators, lowering passes,
   deadlock-checking simulator; I/W split from VJP.
6. MoE data model: `SegmentedTensor` (static capacity, device offsets) +
   `DispatchHandle`; no host round-trip on the hot path; 7-op set with gradient
   rules and deterministic references.
7. MoE lowering through libraries (cuBLASLt grouped GEMM ≥13.2U1 or cuDNN
   GATHER; NCCL A2A then GIN/LSA kernels); generate only glue.
8. FP8 now: `scaled_matmul` + `ScaleSpec` mirroring cuBLASLt; recipes
   tensorwise-current, rowwise, DeepSeek block; block-aligned FSDP shards.
9. Kernel DSL: NVRTC-generated C++ families + CUB via bundled CCCL + disk cache;
   spike cuTile-via-NVRTC once Hopper support confirmed; no Python DSL dependency.
10. Attention/dense fusions from cuDNN SDPA + cuBLASLt epilogues with cached
    plans and pinned versions; copy-engine collectives + cuBLASMp/decomposition
    for collective matmul; megakernels out of scope.

Unverified (check hands-on): PyTorch pipelining CSV helpers; cuTile Hopper +
gather/scan coverage; NCCL device headers and CUTLASS under NVRTC; cuDNN
`moe_grouped_matmul` on SM90; cuBLASLt grouped-GEMM epilogues on Hopper.
