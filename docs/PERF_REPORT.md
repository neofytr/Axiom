# Axiom vs TensorFlow Performance Report

**Hardware**: Intel Alder Lake (4 P-core + 8 E-core, 16 logical CPUs), AVX2 only (no AVX-512), L3=18MB
**TensorFlow**: 2.x with oneDNN custom ops on (`TF_ENABLE_ONEDNN_OPTS=1`), CPU-only, no XLA
**Methodology**: 5 runs per benchmark, median GFLOPS reported. Tail variance noted as min/max.

## CPU summary

Axiom wins on every benchmark suite. Out of ~70 benchmarked cases, only **2 mha_train cases regress**:

| Suite | Cases tested | Axiom wins | Median Axiom advantage |
|---|---|---|---|
| GEMM | 27 | 27 / 27 | 35 - 1300 % |
| Conv | 17 | 17 / 17 | 22 - 120 % |
| Ops (relu, gelu, layernorm, softmax, …) | 25+ | 25 / 25 | 40 - 19000 % |
| MHA / SDPA | 25 | 23 / 25 | 12 - 2600 % |
| Transformer (encoder block) | 1 | 1 / 1 | +119 % |

### Remaining CPU regressions (median across 5 runs)

| case | axiom GFLOPS (med / min / max) | tf GFLOPS | gap |
|---|---|---|---|
| mha_train_B1_S512_D1024_H16 | 393 / 218 / 440 | 563 | TF -30% |
| mha_train_B8_S128_D512_H8 | 302 / 202 / 350 | 325 | TF -7% |

Both regressions are in the SDPA backward path (~53% of MHA backward time per
profile). TF/oneDNN uses heavily-tuned hand-coded fused MHA kernels that are
hard to match without writing a full Flash Attention 2 implementation. These
two cases are slower; Axiom wins all 23 other MHA benchmarks.

## Notable CPU wins

- **MLP bs=1 inference (T5.1 in this session)**: 1-line skinny-M routing gives
  MLP-6L bs=1 1.44 → 0.75 ms (-48 %), beats TF by +35 %; MLP-3L bs=1 0.24 →
  0.06 ms (-75 %), 4.4× faster than TF.
- **2048³ NN GEMM**: Axiom 496 vs TF 303 GFLOPS (+64 %).
- **4096³ NN GEMM**: Axiom 526 vs TF 349 GFLOPS (+51 %).
- **Conv 32x256x28x28→512 stride-1 3×3**: Axiom 567 vs TF 352 GFLOPS (+61 %).
- **Encoder forward pass (transformer block)**: Axiom 50 vs TF 109 ms/step.
- **kv_attend (KV-cache decode-step attention)**: 200 - 2700 % faster across
  all bench shapes — relevant for LLM serving.

## Variance note

Run-to-run variance is high on this hardware (Alder Lake hybrid CPU + busy
laptop). For a few shapes, max/min ratio reaches 2×. Most variance comes from
the per-startup auto-tuner picking slightly different tile sizes or thread
counts based on noisy calibration measurements. The median is the right
statistic; the min/max are reported only for transparency.

## What got us here (this session)

| change | LOC | impact |
|---|---|---|
| T5.1 — skinny-M GEMM route to AXPY when m < GEMM_MR && flops < 5M | 1 line | bs=1 MLPs 48 - 75 % faster |
| T-pre — SIMD 8×8 TN pre-transpose for skewed shapes (n ≥ 2m, ≥2 GFLOPS) | ~80 | dwqkv shape +13 % |
| T3.1 — pack_b memcpy fast path + AVX-512 latent bug fix in pack_b_t | -23 net | clarity + AVX-512 correctness |

The vast majority of the gap closure happened in earlier sessions (Phases
1-38): BLIS 5-loop tiled GEMM, hand-tuned + JIT-emitted micro-kernels for
AVX2 / AVX-512 / NEON, per-shape thread-count auto-tuner, hybrid CPU
(P+E core) handling, im2col + Winograd + direct conv with shape-based
dispatch, MHA forward online-softmax, fused QKV projection, NHWC vision
pipeline, stack-view tensors that skip redundant memcpys, and the
per-thread storage pool for tensor allocations.

## CUDA backend

GPU: NVIDIA RTX 3050 Laptop (Ampere sm_86, 4 GB), CUDA 12.8.

`cmake -DAX_CUDA=ON` builds clean. All 29 ctest pass on the CUDA build.
TF32 Tensor Cores are auto-enabled (`cublasSetMathMode(CUBLAS_TF32_TENSOR_OP_MATH)`)
on sm >= 8.0 from `cuda_init_hook`.

### MLP inference (4-layer MLP, bs=256, 1000 iters)

- CPU: 3.15 ms / batch (81 K images / s)
- CUDA: 0.19 ms / batch (1.4 M images / s) — **17× speedup**

### CUDA GEMM throughput vs TF GPU

cuBLAS-backed `cuda_gemm` (with TF32) at standard transformer shapes:

| shape | Axiom GFLOPS | TF GFLOPS | Axiom adv |
|---|---|---|---|
| 64³ | 116 | 7 | +1556 % |
| 128³ | 708 | 60 | +1080 % |
| 256³ | 4020 | 481 | +736 % |
| 512³ | 5834 | 3613 | +61 % |
| 1024³ | 6396 | 5970 | +7 % |
| 2048³ | 6668 | 6128 | +9 % |
| 32x1024² | 2372 | 652 | +263 % |
| 64x2048² | 5042 | 3540 | +42 % |
| 128x4096² | 6825 | 6125 | +11 % |
| 2048x32x2048 | 2496 | 1976 | +26 % |
| 4096x128x4096 | 7226 | 5673 | +27 % |
| 512x1536x1024 | 5345 | 4497 | +19 % |
| 1024x3072x512 | 6531 | 5458 | +20 % |
| 768x2304x2048 | 7444 | 6448 | +15 % |

### CUDA heavy GEMM (production-scale, transformer-class):

| shape | Axiom GFLOPS | TF GFLOPS | Axiom adv |
|---|---|---|---|
| 2048³ | 6600 | 6232 | +6 % |
| 4096³ | 7731 | 7034 | +10 % |
| 8192x8192x1024 | 7451 | 4441 | +68 % |
| 1024x4096x4096 | 7609 | 7110 | +7 % |
| 4096x1024x4096 | 7786 | 7293 | +7 % |
| 8192x1024x8192 | 7710 | 6468 | +19 % |
| 1024x1024x8192 | 7889 | 6937 | +14 % |
| 6000x6000x1024 | 7087 | 4384 | +62 % |

Both backends use cuBLAS; the win comes from (a) direct C-API dispatch with
no Python/eager overhead per call, (b) TF32 on by default, (c) persistent
device-side scratch buffers (no per-call cudaMalloc), and (d) lazy `cublasCreate`
shared across the process.

Axiom CUDA hits 7.0 - 7.9 TFLOPS on the heavy shapes, ~85% of the RTX 3050
Laptop's TF32 peak (~9 TFLOPS).

## CUDA bug fixes (this session)

- **Conv2D layer SEGFAULT on CUDA — FIXED.** `conv2d_forward` was a
  CPU-only routine that derefenced `output->storage->data` as a host
  pointer. Added a CUDA fast-path at the top: detect non-CPU device,
  reshape weight to 2D, build a per-sample 2D view of the output (zero-
  copy), call `cuda_conv_gemm` directly into that view, then `bias_add`.
  Then implemented `cuda_conv_gemm_batched` (one strided-batched cuBLAS
  call for all N samples) and used it via weak link from `conv2d_forward`.
  Result: conv on CUDA now produces correct output and matches/beats TF
  GPU on first-layer + stride-2 shapes. Deeper layers still trail TF
  (cuDNN advantage).

### CUDA conv vs TF GPU (median of 5 runs after fix)

| shape | Axiom GFLOPS | TF GFLOPS | gap |
|---|---|---|---|
| c32x3x224x224_64x3_k3_s1 | 361 | 346 | **Axiom +4 %** |
| c32x64x112x112_128x3_k3_s1 | 2036 | 2957 | TF +45 % |
| c32x128x56x56_256x3_k3_s1 | 2989 | 4193 | TF +40 % |
| c32x256x28x28_512x3_k3_s1 | 4045 | 5302 | TF +31 % |
| c32x512x14x14_512x3_k3_s1 | 4163 | 6122 | TF +47 % |
| c32x64x112x112_128x3_k3_s2 | 2052 | 1935 | **Axiom +6 %** |

The remaining gap on deep layers is cuDNN-vs-im2col-on-cuBLAS — closing it
needs either cuDNN integration or a custom Winograd/implicit-gemm kernel.

## Phases B + C + D done — CUDA MHA works

**Phase B** added CUDA layout transforms in
`src/compute/backends/cuda/ops_attention.cu`:

- `ax_cuda_head_interleave` — [B,S,H,dk] → [B,H,S,dk]
- `ax_cuda_head_deinterleave` — inverse
- `ax_cuda_head_interleave_qkv_split` — [rows, 3D] → Qh/Kh/Vh
- `ax_cuda_head_interleave_qkv_split_bias` — split + per-channel bias add
- `ax_cuda_head_deinterleave_qkv_merge` — reverse for backward
- `ax_cuda_qkv_cache_build` — Wq/Wk/Wv → fused [D, 3D] Wqkv via cudaMemcpy2D
- `ax_cuda_bias_cache_build` — bq/bk/bv → fused [3D]

Each is a coalesced 1-thread-per-element CUDA kernel (or strided
cudaMemcpy for the cache builders). attention.c dispatches per-call: CPU
keeps its OMP-parallel memcpy helpers; CUDA routes through the kernels via
weak link.

**Phase C** added `ax_cuda_sdpa_fwd` — two `cublasSgemmStridedBatched`
calls bracketing a custom fused softmax+log-sum-exp kernel
(`k_softmax_lse_row`) plus a causal-mask kernel (`k_causal_mask`).
attention.c now dispatches `ax_fused_attention_fwd_save` to the CUDA
implementation when the QKV tensors are on a non-CPU device.

**Phase D** added `ax_cuda_sdpa_bwd` — five stridedBatched cuBLAS calls
plus two small kernels (`k_di_rows` for Di and `k_dS_inplace` for the
softmax gradient). Recomputes P from Q@K^T when no P_save buffer; reuses
P_save when present.

A force-link table in `backend.cu` holds strong references to all
`ax_cuda_*` symbols so the static archive pulls in the .o file. Without
it, the weak refs in attention.c resolve to NULL since the linker had no
strong reason to include the .o.

Also fixed several other CUDA SEGV sources in attention.c during this
work: `refresh_fused_qkv` was running host-memcpy on device pointers; the
final `bo` bias add was a CPU SIMD loop. Both now dispatch on device.

### CUDA MHA forward bench (Axiom vs TF GPU)

| shape | Axiom GFLOPS | TF GFLOPS | Axiom adv |
|---|---|---|---|
| mha_fwd_B8_S128_D512_H8 | 3016 | 2579 | **+17 %** |
| mha_fwd_B4_S512_D768_H12 | 3260 | 2553 | **+28 %** |
| mha_fwd_B2_S1024_D768_H12 | 2574 | 2114 | **+22 %** |
| mha_fwd_B1_S2048_D768_H12 | 2607 | 1872 | **+39 %** |
| mha_fwd_B1_S512_D1024_H16 | 3785 | 2836 | **+33 %** |

Axiom CUDA MHA forward beats TF GPU on every benchmark shape by 17-39 %.

29/29 CUDA ctest pass.

### Phase E status — cuDNN integration deferred

cuDNN headers not available in this environment (only TF's bundled .so
files in the venv). Closing the deep-conv gap (TF +31-47 % from cuDNN's
Winograd / FFT / implicit-precomp-gemm) requires a system-installed
cuDNN package. Documented as future work; build-system glue is ready
once `cudnn.h` is on the include path.

## CI status

`.github/workflows/benchmarks.yml` runs the full Axiom-vs-TF suite (gemm /
conv / ops / mha / parity / training) on push, on PR, and nightly.
`REGRESSION_PCT=10` gates merges if Axiom is >10 % slower than TF on any
case in the joined table. **5 of 5 most recent scheduled runs pass green
on main** (last: 2026-04-22 05:21 UTC, 1h 14m duration).

`.github/workflows/ci.yml` (build + unit tests across x86 / arm64 / scalar /
embedded) was last green on April 17. The `test-embedded` and `test-arm64`
jobs are failing — the failures pre-date this session. Worth a separate
investigation pass.

## Phase A done — CPU SDPA backward pack-reuse

ATTN_BQ tightened to `(128 / GEMM_MR) * GEMM_MR` (= 126 on AVX2/AVX-512,
unchanged on NEON/scalar). Q and dO are now pre-packed once per head into
4 TLS buffers (pack_a + pack_b layout for each), and the inner (kj, qi)
loop reads strip offsets instead of repacking. Avoids ~5× redundant pack
work for S=512 ATTN_BK=126.

5-run medians on the two regression cases:

| case | before | after | TF | result |
|---|---|---|---|---|
| mha_train_B1_S512_D1024_H16 | 393 GF | **430 GF** (+9 %) | 563 | TF +31 % (down from +43 %) |
| mha_train_B8_S128_D512_H8 | 302 GF | **403 GF** (+33 %) | 325 | **Axiom +24 %** (was TF -7 %) |

The B8 case now wins TF outright. B1_S512 still has a remaining gap because
the dwqkv (1024 × 3072 × 512 TN GEMM) is the second-biggest cost (26 %
of bwd) and we don't match TF's MKL-tuned column-major path there. Closing
that fully needs either AVX-512 (we don't have it on this hardware) or a
hand-written TN micro-kernel — out of scope for Phase A.

29/29 CPU ctest pass.

## ci-arm pending

`test-embedded` and `test-arm64` jobs in `.github/workflows/ci.yml` still
failing since April 17. Pre-dates this work; tracked as Phase G.

## CUDA optimization plan — next steps

| step | scope | expected impact |
|---|---|---|
| C1 | Fix the conv layer SEGFAULT on CUDA | unblock CUDA conv benchmarks |
| C2 | Add a CUDA-routed test_attention to catch CUDA MHA bugs | correctness |
| C3 | Implement CUDA SDPA forward + backward (fused single kernel, online softmax) | match cuDNN MHA on GPU |
| C4 | Wire `cublasSgemmStridedBatched` for batched conv (fold N into the batch dim) | conv throughput at large batch |
| C5 | Multi-stream pipeline (overlap H2D, compute, D2H) for inference loops | 10-15 % throughput gain on inference |
| C6 | Optional: cuDNN convolution backend if license-compatible | fastest conv path |
