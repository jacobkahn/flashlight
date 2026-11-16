# Flashlight modernization — building, testing, debugging

Practical notes collected during the modernization work. General build documentation
(dependencies, all CMake options) is in the top-level `README.md`.

---

## 1. Building

**Dependencies for the native backends:** a C++20 compiler (gcc 14 was used), CMake ≥
3.24, Ninja; oneDNN ≥ 3.5 (CPU backend, built by default); for CUDA: the CUDA toolkit
(cuBLAS, cuBLASLt, cuRAND, NVRTC), cuDNN (cudnn-frontend v1.9.0 is fetched by CMake).
Optional: ArrayFire ≥ 3.7.3 (baseline backend); MPI plus NCCL or Gloo for distributed
training; googletest and cereal are fetched when missing (`FL_BUILD_STANDALONE`). A
conda/micromamba environment holding all of these worked well; test binaries resolve
libraries through rpath, so activation is not needed to run them.

**Typical configurations** (shared libraries make rebuilds much faster):
```sh
# CPU only (stock defaults)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=ON

# CPU + CUDA (native CUDA becomes the default backend)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=ON \
  -DFL_USE_CUDA=ON

# Development build with the ArrayFire baseline lane (_AF tests); CUDA stays default
cmake ... -DFL_USE_CUDA=ON -DFL_USE_ARRAYFIRE=ON

# Without MPI installed
cmake ... -DFL_BUILD_DISTRIBUTED=OFF
ninja -C build -j"$(nproc)"
```

## 2. Tests

**Lanes.** The generic suites (`_BACKEND_TEST_SRCS` in `flashlight/fl/test/CMakeLists.txt`)
are built once per backend; `TestInit.h::flTestInit()` picks the backend:

| Suffix | Backend | Built when |
|---|---|---|
| `_CPU` | JIT over `CPUTensor` | `FL_USE_CPU` |
| `_CUDA` | JIT over `CudaTensor` | `FL_USE_NATIVE_CUDA` |
| `_AF` | bare `ArrayFireTensor` (no JIT) — the baseline | `FL_USE_ARRAYFIRE` |
| none | compile-time default (CUDA > CPU > ArrayFire) | always |

Skips guarded by `FL_TEST_BACKEND_CPU` / `_CUDA` apply only to suffixed lanes, so the
unsuffixed lane can expose bugs the suffixed lanes skip.

**Running.**
```sh
ctest --test-dir build -R '^_CPU\.'     # one lane
ctest --test-dir build -R '^_CUDA\.'
ctest --test-dir build -R '^_AF\.'
python3 docs/modernization/tools/run_tests.py build /tmp/fl-test-logs   # everything, in parallel
python3 docs/modernization/tools/parity.py /tmp/fl-test-logs            # AF vs CPU vs CUDA matrix
```

**Thread count.** OpenMP defaults to every logical CPU, which makes tiny CPU ops ~100×
slower on large machines (measured on 2 × 96 cores with SMT: 64² matmul 11.3 ms at 384
threads vs 0.1 ms at 16). Configure with `-DFL_TEST_NUM_THREADS=16` (sets
`OMP_NUM_THREADS` for ctest) or export `OMP_NUM_THREADS` when running binaries directly.

**Pitfalls**
- Enumerate binaries from `CTestTestfile.cmake`, not from gtest suite names: many
  binaries' suites have different names (`ComputeTest` → `TensorComputeTest`,
  `TensorAdapterTest` → `TensorBaseTest`, `ContribSerializationTest` →
  `SerializationTest`, …). A runner that derived binaries from suite names silently
  skipped 21 binaries.
- Run the variants of one suite serially: some tests write fixed `/tmp` file names and
  collide when CPU and CUDA variants run at once (e.g. `NNSerializationTest.FileSize`).
- A CUDA error such as an illegal address is sticky: every later test in the same
  process fails. Re-run the failing test alone before concluding.
- Binaries for removed targets stay in `build/flashlight/fl/test/` and fail confusingly;
  only run what ctest registers, or clean the build tree.
- Jacobian (finite-difference) tests dominate runtime, especially CUDA BatchNorm (~70 s
  each), partly because cuDNN plans are rebuilt every evaluation.

## 3. Runtime switches

| Variable | Effect |
|---|---|
| `OMP_NUM_THREADS` | CPU (oneDNN/OpenMP) thread count; Flashlight never sets it |
| `FL_ALLOW_TF32=1` | allow TF32 for f32 on CUDA (cuDNN engines, cuDNN RNN, cuBLAS, cuBLASLt); default is strict f32 |
| `NVIDIA_TF32_OVERRIDE=0` | affects cuBLAS/legacy cuDNN only — **not** cuDNN graph-API engine selection |
| `CUDA_VISIBLE_DEVICES` | spread parallel test runs over GPUs |

## 4. Debugging techniques that worked

- **Synchronous CPU stream.** Heap corruption from async lifetime bugs disappeared
  when tasks ran inline. A small `LD_PRELOAD` shim that interposes
  `fl::AsyncCPUStream::enqueue` and runs the task immediately confirmed the race. Not
  kept; a proper debug switch is a TODO (friction log in `TODO.md`).
- **Thread-count bisection.** The CPU race only reproduced at ≥ 128 OpenMP threads
  (larger timing window); vary `OMP_NUM_THREADS` before deeper tooling.
- **ASAN.** A CPU-only ASAN build needs `-DFL_BUILD_DISTRIBUTED=OFF -DFL_USE_GLOO=OFF`
  (Gloo's minimum CMake version is too old for CMake 4). ASAN's allocator quarantine can
  hide use-after-free *writes* by uninstrumented oneDNN code.
- **valgrind** aborts with hundreds of OpenMP threads; lower `OMP_NUM_THREADS` and pass
  `--max-threads`.
- **gdb:** `catch throw` misses exceptions created with `std::make_exception_ptr`
  (futures, streams); break on the exception type's constructor instead.
- **Graph dumps:** wrap code in `ScopedPostEvalGraphvizPrinter`
  (`flashlight/fl/tensor/printer/`) to see the optimized graph per eval. Env switches
  for dumping graphs and disabling individual passes are TODOs.
- **Pass bisection:** temporarily drop passes from `Optimizer::Optimizer` (or a
  backend's `JitOptimizerExtension`) to find which pass breaks a result — this is how
  TF32 engines in `CudnnSubgraphLowering` were found.
- **Scratch probes:** get a test's compile/link commands with
  `ninja -C build -t commands flashlight/fl/test/<Binary>`, swap in your source file,
  drop gtest, and link `libflashlight.so` plus the CUDA driver stub
  (`<cuda>/targets/x86_64-linux/lib/stubs/libcuda.so`). A checked-in scratch target
  would make this one command (TODO).

## 5. Working practices learned the hard way

- Many modernization files are *intent-to-add* (`git status` shows ` A`): git has no
  copy of their contents. Back a file up before any mechanical or scripted edit, and run
  formatters only on explicit line ranges (`clang-format --lines=N:M`). Once, a shell
  variable holding several `--lines` flags was passed as a single argument (zsh does not
  word-split unquoted variables); `clang-format -i` then reformatted three whole files.
  They were recovered only from an earlier full read of the files.
- Put multi-argument shell logic in a bash script or use arrays.
- Scratch state under `/tmp` does not survive reboots: probes, shims, alternative
  builds, and parked drafts from the 2026-10 work were lost that way. Keep anything worth
  keeping in the repository (for example under `docs/modernization/tools/`).
- Verify claims from code reading with a run before relying on them, and say which
  claims are verified (the review uses `[REPRO]` / `[READ]` / `[SUSPECT]` tags).
