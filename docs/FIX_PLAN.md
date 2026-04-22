# Fix + optimization plan — close every remaining gap

Goal: from the state documented in `docs/PERF_REPORT.md`, drive Axiom
to parity or lead vs TensorFlow on every benchmark for both CPU and
CUDA, and fix every known correctness bug. Execute the phases **in
order** — later phases depend on earlier ones.

## Architecture

```
  ┌─────────────────────────────────────────────────────────────────┐
  │ CPU backend                                                      │
  │   [A] SDPA backward pack-reuse ── closes mha_train_B1_S512_...   │
  └─────────────────────────────────────────────────────────────────┘
                   │
  ┌────────────────┴────────────────────────────────────────────────┐
  │ CUDA backend                                                     │
  │                                                                  │
  │   [B] CUDA layout transforms (head_interleave / deinterleave /   │
  │       qkv_split_bias_add) ── prereq for any CUDA MHA             │
  │         │                                                        │
  │         ▼                                                        │
  │   [C] CUDA SDPA forward (cuBLAS stridedBatched + cuda_softmax)   │
  │         │                                                        │
  │         ▼                                                        │
  │   [D] CUDA SDPA backward (stridedBatched + softmax grad)         │
  │         │                                                        │
  │         ▼                                                        │
  │   [E] cuDNN conv integration (optional on-demand) ── closes the  │
  │       deep-layer 31-47 % gap vs TF GPU                           │
  │         │                                                        │
  │         ▼                                                        │
  │   [F] Multi-stream pipeline + CUDA memory pool                   │
  └─────────────────────────────────────────────────────────────────┘
                   │
  ┌────────────────┴────────────────────────────────────────────────┐
  │ Infrastructure                                                   │
  │   [G] CI polish: fix test-embedded/arm64 jobs; add CUDA CI row;  │
  │       extend summarize.py for CUDA results; add regression gates │
  └─────────────────────────────────────────────────────────────────┘
```

Dependencies in strict order: A (CPU, independent) → B → C → D. E and F
can start after B is in (they don't need C/D). G can piggyback on any
phase.

---

## Phase A — CPU SDPA backward pack-reuse

**Target**: `mha_train_B1_S512_D1024_H16` TF -30 % → within ±5 %.

**Problem** (from `PERF_REPORT.md` profile): `sdpa_bwd` is 53 % of MHA
backward for this shape. The kernel `attn_bwd_head` packs Q and dO in
both `pack_a` and `pack_b` layouts **per (kj, qi) tile**. For S=512 and
ATTN_BQ=ATTN_BK=128 there are 16 tiles per head — so Q and dO get
packed 16× each per head when they only depend on qi. Cost: ~1.3 MB of
needless pack traffic per head per backward call × 16 heads × many
calls per iteration.

**Blocker**: `ATTN_BQ = 128` is not a multiple of `GEMM_MR = 6` (AVX2)
or 14 (AVX-512), so qi values 128, 256, 384 are not strip-aligned in
the pre-packed layout.

**Steps**:

- **A.1** At tile-size init, pick `ATTN_BQ = (128 / GEMM_MR) * GEMM_MR`
  so every qi value is strip-aligned:
  - AVX2 MR=6 → 126
  - AVX-512 MR=14 → 126
  - NEON MR=8 → 128 (already aligned)
  - scalar MR=4 → 128 (already aligned)
- **A.2** Add TLS buffers: `tl_bwd_q_pa`, `tl_bwd_q_pb`, `tl_bwd_dO_pa`,
  `tl_bwd_dO_pb`. Each sized `S * dk` (plus MR/NR rounding). Grow-on-
  demand via `ax_tls_grow` mirroring the existing Kt/Vt buffers.
- **A.3** Inside `attn_bwd_head`, before the `(kj, qi)` loop, run the
  four full-S packs once: `pack_a(Q, …)`, `pack_b(Q, …)`,
  `pack_a(dO, …)`, `pack_b(dO, …)`.
- **A.4** Replace per-tile `pack_a(Q + (qi+ir)*dk, …)` with
  `Q_pa + (qi+ir) * dk` (strip offset, guaranteed aligned by A.1).
  Same for dO pack_a, Q pack_b, dO pack_b.
- **A.5** `test_grad_verify` must still pass. Re-run bench_mha 5× on
  `mha_train_B1_S512_D1024_H16` and the B8 S128 case; require median
  within ±5 % of TF.

**Exit criterion**: `python3 benchmarks/summarize.py axiom_mha.txt
tf_mha.txt` reports zero regressions on mha_train at threshold 5 %.
All 29 ctest still pass.

**Risk**: highest in A.1 — changing `ATTN_BQ` affects every tile-size
constant that references it (score_strip size, P_tile size, etc.). Must
audit every `ATTN_BQ` / `ATTN_BQ_MAX` usage.

**LOC estimate**: ~200 edit + audit.

---

## Phase B — CUDA layout transforms

**Target**: Unblock CUDA MHA. Prerequisite for C/D.

**Problem**: `attention.c` uses five CPU-only helpers:
`head_interleave`, `head_deinterleave`, `head_deinterleave_slot`,
`head_interleave_qkv_split_bias_add`, `head_deinterleave_qkv_merge`.
All deref raw host pointers. On CUDA they read garbage → SEGV.

**Steps**:

- **B.1** Define dispatched ops in the backend vtable:
  ```
  .head_interleave
  .head_deinterleave
  .head_deinterleave_qkv_merge
  .head_interleave_qkv_split_bias_add
  ```
  CPU impls point at the existing `head_*` statics (move from
  `attention.c` to `cpu_opt.c` / reuse by symbol).
- **B.2** Write CUDA kernels for each — simple coalesced element-wise
  copies with a single thread per (b, h, s, d) element. File:
  `src/compute/backends/cuda/ops_attention.cu`.
- **B.3** Register CUDA impls in `cuda_ops` vtable.
- **B.4** Replace direct `head_interleave(...)` calls in `attention.c`
  with `ax_compute_head_interleave(...)` via a new
  `include/axiom/compute.h` facade.
- **B.5** Write `tests/test_attention_cuda.c` (runs existing reference
  SDPA pathway once on each backend and compares within fp32 tolerance).
  Wire into CMakeLists under `if(AX_CUDA)`.

**Exit criterion**: the CUDA MHA smoke test from `PERF_REPORT.md`
completes without SEGV. `test_attention_cuda` passes. CUDA build ctest
still 29/29.

**LOC**: ~400 (ops_attention.cu + vtable + compute.h facade).

---

## Phase C — CUDA SDPA forward

**Target**: replace the CPU-only `ax_fused_attention_fwd_save` when
tensors live on CUDA. No more H2D round-trip.

**Steps**:

- **C.1** Implement `cuda_sdpa_fwd(Q, K, V, O, L, BH, S, dk, scale,
  causal, P_save)` in `src/compute/backends/cuda/ops_attention.cu`:
  - `S_scores = scale * Q @ K^T` via
    `cublasSgemmStridedBatched` — stride `S*dk` per head, batch `BH`.
  - If `causal`, launch a masking kernel that writes `-inf` for j>i.
  - Run `cuda_softmax_rowwise` over the `[BH*S, S]` score matrix.
  - `O = P @ V` via `cublasSgemmStridedBatched` again.
  - Compute L (log-sum-exp) in a small kernel if requested.
  - Save P if `P_save` non-null (just retain the post-softmax buffer).
- **C.2** Extend `src/core/fused_attention.c` to resolve to
  `cuda_sdpa_fwd` when active backend is CUDA (add backend-aware
  dispatcher).
- **C.3** Remove the host-fallback prototype from `attention.c`
  (superseded by the dispatcher).
- **C.4** `test_attention_cuda` now compares CUDA MHA forward
  against CPU reference element-wise.
- **C.5** Bench CUDA MHA forward vs TF GPU on the bench_mha shapes.

**Exit criterion**: `mha_fwd_B*_S*_D*_H*` on CUDA within ±10 % of TF
GPU, correct to fp32 tolerance vs CPU reference.

**LOC**: ~500 (SDPA forward kernel + causal mask + dispatcher + test).

---

## Phase D — CUDA SDPA backward

**Target**: `mha_train` on CUDA produces correct gradients at
competitive throughput.

**Steps**:

- **D.1** Implement `cuda_sdpa_bwd(Q, K, V, O, dO, L, dQ, dK, dV, BH,
  S, dk, scale, causal, P_saved)`:
  - If P_saved provided, reuse; else recompute P from Q @ K^T using
    stridedBatched.
  - `dV = P^T @ dO` (stridedBatched).
  - `dP = dO @ V^T` (stridedBatched).
  - `dS = P * (dP - Di) * scale` via a small kernel.
  - `dQ = dS @ K` and `dK = dS^T @ Q` (stridedBatched).
- **D.2** Wire through the same dispatcher as C.
- **D.3** Add to `test_grad_verify` a CUDA branch that compares CPU
  vs CUDA gradients at small shapes.
- **D.4** Bench `mha_train_B*_S*_D*_H*` on CUDA vs TF GPU.

**Exit criterion**: `mha_train` on CUDA within ±10 % of TF GPU. No
grad divergence vs CPU reference at small shapes.

**LOC**: ~600.

---

## Phase E — cuDNN conv integration

**Target**: close the 31-47 % conv gap on deep layers
(`c32x{128,256,512}x*_..._k3_s1`).

**Rationale**: TF uses cuDNN which picks from several optimized
algorithms per shape (Winograd, FFT, implicit-precomp-gemm). Our
im2col + cuBLAS path cannot match it without rewriting those algorithms
from scratch. Linking cuDNN is the 80/20 choice — ships value now,
leaves a future "remove cuDNN dependency" track for the embedded niche
value prop.

**Steps**:

- **E.1** Make cuDNN optional behind `AX_CUDNN=ON` cmake flag. When
  linked, add `src/compute/backends/cuda/ops_cudnn.cu` with
  `cuda_conv_cudnn_fwd`.
- **E.2** In `conv2d_forward` CUDA fast path, dispatch to cudnn if
  available, else fall back to the existing `cuda_conv_gemm_batched`.
  Shape-based algo selection via `cudnnGetConvolutionForwardAlgorithm_v7`.
- **E.3** Bench conv vs TF GPU; validate correctness via test_conv
  extended with CUDA branch.
- **E.4** Document the embedded-niche caveat in README: "cuDNN is
  optional and only used when you build with `-DAX_CUDNN=ON`. The
  native im2col+cuBLAS path works without it."

**Exit criterion**: conv on CUDA within ±5 % of TF on every bench
shape (with AX_CUDNN=ON).

**LOC**: ~300 (cuDNN wrapper + cmake + test).

---

## Phase F — Multi-stream pipeline + CUDA memory pool

**Target**: 10-15 % throughput gain on inference loops by overlapping
H2D / compute / D2H.

**Steps**:

- **F.1** Extend `backend.cu` to manage N CUDA streams (default 2 for
  H2D/compute overlap).
- **F.2** Add optional stream argument to key ops (gemm, conv, sdpa).
  Default remains stream 0 for backward compat.
- **F.3** Per-thread storage pool on CUDA (mirror the CPU
  `pool_get/put` at `tensor.c:107-145` but with cudaMalloc).
- **F.4** Wire `ax_session_predict_into(output, input)` from the
  earlier T5.1 plan — now uses streams + persistent buffers.
- **F.5** Bench inference loop vs TF GPU, measure overlap win.

**Exit criterion**: ≥10 % throughput gain on `bench_large infer-gpu`.

**LOC**: ~500.

---

## Phase G — CI polish + regression gates

**Target**: keep green status honest; catch CUDA regressions on every
PR.

**Steps**:

- **G.1** Investigate + fix `test-embedded` and `test-arm64` jobs in
  `.github/workflows/ci.yml` (failing since April 17 per `PERF_REPORT.md`).
- **G.2** Add CUDA rows to `.github/workflows/benchmarks.yml` using a
  self-hosted GPU runner. Runs `bench_cuda_gemm`, CUDA conv, CUDA MHA
  against TF GPU sidecars. Gated by `REGRESSION_PCT=10` just like CPU.
- **G.3** Extend `benchmarks/summarize.py` with `--backend` flag and
  tagged case names (`nn_1024^3_cpu`, `nn_1024^3_cuda`) so one report
  covers both.
- **G.4** Add `calibrate_cuda` job: warm up cuBLAS, dump
  `cuDevicePrimaryCtxGetState` stats, run the GEMM tile auto-tuner if
  it exists in the CUDA backend.
- **G.5** Nightly + on-PR: if CUDA gap exceeds threshold, fail CI
  with a clear "axiom regressed on shape X by Y %" message.

**Exit criterion**: both workflows green on main, CUDA regressions
caught at PR time.

**LOC**: ~200 (YAML + summarize.py).

---

## Execution order (strict)

1. **A** — lowest risk, biggest immediate CPU win, no dependencies.
2. **B** — prerequisite for all CUDA MHA work, and unblocks the
   CUDA MHA SEGV documented in PERF_REPORT.md.
3. **C** — first CUDA MHA value delivery (forward inference).
4. **D** — closes CUDA MHA training story.
5. **E** — closes the CUDA conv deep-layer gap.
6. **F** — incremental polish; don't block on it.
7. **G** — should happen in parallel with any of the above.

**Total LOC estimate**: ~2800 net added (new kernels + plumbing) plus
audit-and-modify passes.

**Time**: this is multi-session work. I will execute in order this
turn starting with Phase A; later phases carry over to subsequent
sessions in the same order.
