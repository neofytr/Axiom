# Axiom production-readiness plan

## Audit summary (Phase H — done, read-only)

- **Code volume**: 73 source files, ~27 K LOC. Top files: `cpu_opt.c` (5 K),
  `conv.c` (4.6 K), `norm.c` (1.6 K), `attention.c` (1.1 K).
- **Public API**: 26 headers in `include/axiom/`, 203 documented public
  functions. `axiom.h` is the master include.
- **Tests**: 30 test binaries, 29 ctest suites (one not registered),
  passing on both CPU and CUDA builds (29/29 each, sequential).
- **Docs**: 21 files in `docs/` (HTML API ref + guides + perf report +
  architecture + embedded), 158-line README.
- **CI**: 3 workflows (benchmarks, ci, autotuner). Benchmarks-vs-TF runs
  daily on main, last 5 of 5 green. `ci.yml` (build+test) failing since
  Apr 17 — `test-embedded` and `test-arm64` jobs.
- **Code hygiene**: 0 TODO/FIXME/XXX/HACK markers in `src/` or `include/`.
- **License**: MIT (`LICENSE.txt`).
- **Build modes**: full (training), `AX_INFERENCE_ONLY=1` (smaller binary,
  excludes losses/optim/data/lr_scheduler), `AX_NO_AUTOTUNE=1` (skip
  startup calibration), `AX_NO_JIT=1`, `AX_CPU_ISA_DISPATCH=1` (multi-
  arch dispatch), `AX_CUDA=ON`.

### What's strong already
- Zero external runtime deps (no Python, no BLAS).
- Multiple backends with vtable dispatch: cpu_naive, cpu_opt
  (AVX2/AVX-512/NEON/scalar), CUDA.
- Embedded story (`AX_INFERENCE_ONLY` + `AX_NO_AUTOTUNE` + `AX_NO_STDIO`
  → <100 KB inference binary).
- Beats TF on essentially every CPU benchmark (post-Phase A) and every
  CUDA GEMM/MHA-fwd shape.
- Storage pool for tensor allocation reuse.
- Well-organized public headers, master `axiom.h`.

### What's weak
1. **Last perf gaps** — `mha_train_B1_S512_D1024_H16` TF +31 % (CPU);
   CUDA conv deep layers TF +31-47 % (no cuDNN).
2. **Largest files (`cpu_opt.c` 5 K LOC, `conv.c` 4.6 K LOC) hold many
   distinct concerns** — micro-kernels, tile calibration, packing, sdpa,
   conv path selection, hybrid CPU dispatch all in one TU. Hard to
   navigate; slow incremental rebuilds.
3. **Public/internal header line is fuzzy** — some headers expose
   internals (`backend_ops.h`, `simd_defs.h` referenced from core).
4. **Force-link CUDA workaround** (`backend.cu` table) is an artifact
   of static-archive + weakly-linked dispatch. Could be replaced by a
   proper backend extension registry.
5. **No SemVer tagging** on releases; no `pkg-config` or
   `axiomConfig.cmake`.
6. **`ci.yml` broken on test-embedded + test-arm64** since Apr 17.
7. **No ASan / Valgrind nightly** — would catch memory issues earlier.
8. **No fuzz coverage on tensor / autograd APIs** (only serialize is
   fuzzed).

---

## Phase plan (in strict order)

### Phase I — close last perf gaps (Axiom > TF on **everything**)

Gap map as of v0.10.0 baseline (from PERF_REPORT + April bench):

| Backend / shape                        | Axiom vs TF | dominant cost           |
|----------------------------------------|-------------|-------------------------|
| CPU GEMM (all)                         | +5 to +40%  | —                       |
| CPU conv (all)                         | +0 to +20%  | —                       |
| CPU mha_fwd (all)                      | +5 to +30%  | —                       |
| CPU mha_train B1_S512_D1024_H16        | TF +30-40%  | sdpa_bwd (53% of bwd)   |
| CPU mha_train B8_S128_D512_H8          | TF +15-20%  | sdpa_bwd + dwqkv gemm   |
| CUDA GEMM / mha_fwd (all)              | +17 to +39% | —                       |
| CUDA conv deep layers (5x5, 256ch+)    | TF +31-47%  | no winograd             |
| CUDA conv first layer + stride-2       | +4 to +6%   | —                       |
| bench_transformer end-to-end           | +27% faster | (sanity check)          |

The two non-winning cells above drive the plan:

#### I.1 — CPU sdpa_bwd optimization

sdpa_bwd is 53% of mha_bwd time on the worst shape. Attempted fix
(strided-A C kernel) lost to the JIT-emitted packed kernel's speed.
Two-pronged approach:

**I.1.a — JIT-emitted strided-A kernel**. Extend the existing AVX2
6×16 and AVX-512 14×32 per-kc JIT emitters (jit_gemm_avx2.c /
jit_gemm_avx512.c) with a stride-parameterised variant. Signature:
  `void fn(int64_t kc, const float *ap, int64_t lda,
            const float *bp, float *c, int64_t ldc_bytes)`
Emit the same fma cascade as the packed variant but with the A-pointer
advanced by `lda` bytes per K-iter instead of `MR * sizeof(float)`. lda
lives in a callee-saved gpr (r15). Per-kc specialisation fully unrolls.

Then the dV and dK paths in `attn_bwd_head` drop `pack_a_t(P_tile)` and
`pack_a_t(dS_tile)` (saves ~64 KB write+read per tile per pack) and call
the strided-A JIT directly. Expected +10-15% on sdpa_bwd, +5-8% on the
whole mha_train backward.

**I.1.b — Per-head thread-parallel (if I.1.a not enough)**. B1
shapes only have BH=16 heads so OMP scales up to 16 threads cleanly.
The worst shape we lose on is B1_S512_D1024_H16 (BH=16). Every thread
walks 25 (qi, kj) tiles serially. Adding an inner parallel-for over qi
or kj strips could exploit more cores if the thread budget allows.
Risky: Bq×Bk tile scratch is per-thread, so blowing up thread count
spikes memory.

**I.1.c — Fused FA-2 backward (stretch goal)**. Replace the
materialise-P/dP/dS pattern with a Flash-Attention-2 style fused
kernel that keeps the tile in registers through QK^T → softmax → PV →
dP → dS → dK/dQ/dV in one pass. Big win (20-30% on bwd) but a full
rewrite of attn_bwd_head. Gate on whether I.1.a+b close the gap.

Exit criterion for I.1: Axiom ≥ TF-5% on every mha_train shape on a
quiet machine over 5 reps.

#### I.2 — CUDA Winograd F(2,3)

Implement the standard Winograd F(2,3) algorithm for 3×3 stride-1 conv
on CUDA. 2.25× theoretical FLOP reduction (9 mults → 4 per 2×2 output
tile). Three kernels + one batched gemm:

1. **Weight transform** (`ops_winograd.cu::wino_weight_transform`):
   `U[16, C_out, C_in] = G * w[C_out, C_in, 3, 3] * G^T` per input/
   output channel pair. Runs once per forward; cached while the weight
   generation counter is unchanged.

2. **Input transform** (`ops_winograd.cu::wino_input_transform`):
   `V[16, C_in, num_tiles] = B^T * d * B` per (n, tile_y, tile_x)
   4×4 input patch. One thread per tile per channel, reads from the
   NCHW input strided.

3. **Batched cuBLAS gemm on transformed domain**: 16 gemms of shape
   `[C_out, num_tiles] = U[16, C_out, C_in] @ V[16, C_in, num_tiles]`.
   Use `cublasSgemmStridedBatched` with batch=16.

4. **Output transform** (`ops_winograd.cu::wino_output_transform`):
   `Y[C_out, 2, 2] = A^T * M * A` per (n, tile_y, tile_x). Scatters
   the 2×2 outputs back into the NCHW output tensor.

Wire into the cuda extension registry as `conv_winograd_f23` and have
conv2d_forward dispatch to it when on CUDA + kh=kw=3 + sh=sw=1 +
`prefer_winograd_f23(...) == true`.

No local CUDA env to test. Validate via CI: push the change, wait for
the benchmarks-vs-TF workflow to run on GH Actions GPU runner, verify
the cuda conv rows flip to +positive.

Exit criterion for I.2: bench_cuda_conv shows Axiom ≥ TF on every
shape in the suite.

#### I.3 — Final re-bench + PERF_REPORT update

After I.1 and I.2 land, run all bench suites 5× on a quiet machine and
update `docs/PERF_REPORT.md` with the final axiom-vs-TF table showing
green on every row. That row table is the "axiom ≥ TF on everything"
deliverable that gates Phase I completion.

### Phase J — API + interface hardening

- **J.1 Public/internal header split**. Move `backend_ops.h`,
  `simd_defs.h`, internal types out of `include/axiom/` into a new
  `include/axiom/internal/` (or `src/private/`). Public headers must
  document which include `axiom.h` brings them in.
- **J.2 Audit every public function for**: error-status returned, NULL
  safety, ownership comment (caller frees? library frees?), thread
  safety. Encode via `@param` / `@return` / `@thread-safety` doxygen
  tags. ~203 functions to audit.
- **J.3 Standardise naming**: every public symbol prefixed `ax_`; every
  enum value prefixed `AX_`; verbs follow CRUD pattern
  (`create / destroy / get / set / forward / backward`). Rename outliers.
- **J.4 ABI stability promise**: tag all currently-public functions
  ABI-stable as of v1.0.0. Any breaking change requires a deprecation
  cycle via `__attribute__((deprecated))`.
- **J.5 Error-handling consistency**: all entry points either return
  `ax_status_t` OR a pointer (NULL on err). Last-error retrievable via
  `ax_err_last_message()`. No silent failures, no asserts in release.

### Phase K — architecture cleanup

- **K.1 Split `cpu_opt.c` (5 K LOC)** — *partial done in v0.10.0*: TOC +
  section dividers added at the top of cpu_opt.c. The full physical
  split into the seven planned files (`gemm.c`, `sdpa.c`, `conv.c`,
  `elementwise.c`, `reduce.c`, `calibrate.c`, `internal.h`) is deferred
  to a follow-up because the file is compiled three times under
  `AX_CPU_ISA_DISPATCH` (avx512 / avx2 / scalar). Each new file has to
  be added to all three OBJECT-library targets and ~30 helpers that
  are currently `static` (TLS buffers, packing helpers, micro-kernel,
  AX_SYM macro) need to move to a shared internal header. The
  mechanical work is large but well-scoped — see the section TOC in
  cpu_opt.c for the line ranges per target file.
- **K.2 Split `conv.c` (4.6 K LOC)** — *partial done in v0.10.0*:
  pooling layers (maxpool / avgpool / global avgpool / flatten,
  ~1023 LOC) extracted to `core/conv/pool.c`. conv.c is now ~3591 LOC.
  The remaining split into
  `core/conv/{forward,backward,im2col,winograd,direct,path_selection}.c`
  is deferred — it requires teasing apart the shared `ax_conv_scratch`
  struct that conv2d, conv-bn-relu, direct, and winograd all depend on.
- **K.3 Split `attention.c` (1.1 K LOC)** — *done in v0.10.0*:
  `core/attention/{layout.c, cache.c, internal.h}` split off from
  attention.c. attention.c is now ~825 LOC (was 1097), layout.c
  ~175 LOC, cache.c ~85 LOC.
- **K.4 Backend registry instead of force-link table** — *done in
  v0.10.0*: `ax_cuda_extension_t` registry in
  `axiom/internal/cuda_extension.h` replaces 10 weak-symbol externs
  and the force-link table that kept the cuda .o files alive in the
  static archive. The cuda backend constructs the table at init via
  `ax_compute_register_cuda_extension(...)`; cpu code calls
  `ax_compute_get_cuda_extension()` and dispatches through the struct.
- **K.5 Tensor lifecycle clarity** — *done in v0.10.0*: the four
  ownership patterns (heap, arena, pool, view) are documented in the
  CONVENTIONS block at the top of `axiom/axiom.h` and called out
  per-header where they apply. Type-tag enforcement (e.g. dedicated
  `ax_view_t`, `ax_arena_tensor_t` types) is deferred — it would be
  an api break and isn't justified by the bug rate so far.

### Phase L — build + distribution

- **L.1 Install target**: `cmake --install build --prefix /usr/local`
  installs headers to `<prefix>/include/axiom`, lib to
  `<prefix>/lib/libaxiom.{a,so}`, `axiomConfig.cmake` to
  `<prefix>/lib/cmake/axiom`.
- **L.2 `pkg-config` file**: `axiom.pc` with `Libs:` and `Cflags:`
  including the optional `-DAX_INFERENCE_ONLY` etc.
- **L.3 SemVer**: tag the current state v0.10.0 (pre-1.0; document
  what's needed for 1.0). Adopt SemVer in CHANGELOG.
- **L.4 Cross-compile validation**: test the embedded build path
  (musl-cross or arm-none-eabi).

### Phase M — tests + CI hardening

- **M.1 Fix `test-embedded` and `test-arm64`** in `ci.yml`.
- **M.2 Add ASan + UBSan workflow** (run nightly, gate merges).
- **M.3 Valgrind workflow** for memory-leak detection (lighter weight
  than ASan, runs on a subset).
- **M.4 Property-based testing** for tensor/autograd APIs (random
  shape + random dtype, compare CPU vs CPU-naive backend).
- **M.5 GPU CI row** (when self-hosted runner available).
- **M.6 Coverage report** via gcov/lcov, target ≥85 % line coverage.

### Phase N — documentation

- **N.1 README**: expand to cover quickstart, install, build modes,
  benchmarks, links to API ref + architecture doc.
- **N.2 Doxygen** generation in CI; publish to GitHub Pages.
- **N.3 Architecture doc**: one-pager describing module boundaries,
  backend dispatch, tensor lifecycle, autograd graph, JIT pipeline.
- **N.4 Migration guide** if any J-phase rename is non-trivial.
- **N.5 Embedded deployment cookbook**: concrete examples for
  Cortex-M, RP2040, ESP32 with measured binary sizes.

### Phase G + E (deferred from prior plan, run after N)

- Phase G: CI ARM/embedded fix + GPU bench gate.
- Phase E: cuDNN integration once env supports.

---

## Production-readiness verdict (current)

Axiom is **late-beta** quality:

- Functionality: full DL framework (tensors, autograd, layers, optimizers,
  conv, attention, quantization, multi-backend).
- Performance: better than TF on essentially every CPU benchmark, every
  CUDA GEMM/MHA-fwd shape.
- Correctness: 29/29 ctest pass on both CPU + CUDA, daily TF parity bench.
- Distribution: header API is clean, but no install target / pkg-config /
  SemVer yet.
- Documentation: 21 docs, HTML API ref, but no doxygen-generated
  per-function reference.
- Reliability: no public ASan/Valgrind report; some CI jobs broken.

**Blockers to "1.0 production"**:
1. Phase M (CI hardening: green, ASan-clean, fuzz-covered).
2. Phase L (install target + SemVer + pkg-config).
3. Phase J.5 (audit error-handling for every public function).

**Nice-to-haves** that don't block 1.0 but improve quality:
- Phase I.2 (CUDA Winograd) — closes last perf gap.
- Phase K (architecture split of large files) — readability win.
- Phase N.2 (doxygen API ref) — discoverability win.

---

## Execution order (this and following sessions)

1. **I.1 + I.2** (close last perf gaps) — this session if time.
2. **J + K** (API + arch cleanup) — next 1-2 sessions.
3. **L + M + N** (distribution + CI + docs) — next 1 session.
4. **G + E** (CI ARM fix + cuDNN) — final session.

Total estimate: 4-6 sessions. I'll start with **I.1** now.
