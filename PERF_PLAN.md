# Axiom performance plan — close TF gap, raise floor

Drawn from the full Axiom-vs-TF benchmark on 12th-gen i5-12500H (4P + 8E, 16T,
AVX2, 18 MB L3). Snapshot of where Axiom currently stands and the ordered work
to make it strictly faster than TF on every measured workload.

Optimization scope:
- **Level 0 (algorithm)**: data layout, fused passes, batching, recompute vs
  save tradeoffs.
- **Level 1 (ISA-generic SIMD)**: AVX2, AVX512, NEON micro-kernels via
  `simd_defs.h`, auto-tuning, calibration. NO per-arch tuning (Skylake / Zen /
  Apple-silicon-specific paths).
- (Out of scope) **Level 2 (per-arch)**: not pursued at this stage.

## Current standings (snapshot)

- **Parity**: all 5 ops match TF within float32 tolerance.
- **Ops suite**: 27/27 cases faster, no regressions.
- **GEMM**: 23/27 cases faster.
- **MHA**: 21/25 cases faster (forward + causal SDPA dominant).
- **Stress**: huge GEMMs and long-context attention all wins.
- **Small MLP train**: 5.25× faster than TF.
- **Mini Transformer**: ~2× faster than TF.
- **Wide MLP train**: 167s vs 276s (+66% faster).
- **VGG train**: pending clean rerun.

## Regressions to fix (priority order)

| # | regression | gap vs TF | root cause | phase |
|---|---|---|---|---|
| 1 | `conv_32x3x224x224_64x3` (input layer) | -54% | im2col on C_in=3 wastes 9× input copy; K=27 GEMM under-utilized | 1.1 |
| 2 | `conv_*_512x1_k1_s1` (1×1 convs) | -28% | per-sample GEMM still small even with view-skip | 1.2 |
| 3 | `mha_train_B8_S128_D512_H8` etc | -26% to -62% | bwd recomputes attn weights | 1.3 |
| 4 | `gemm TN/NT 2048×512×512` | -19% to -34% | pack_b cache miss on transposed; scalar pack | 2.2 |
| 5 | `maxpool_32x*_k2` | -11% to -32% | scalar inner loop | 2.1 |
| 6 | `huge_gemm_8192` | -9.5% | pack overhead at huge sizes | 3.1 |
| 7 | `long_ctx_sdpa_S2048` | -12% | suboptimal Q-tile size at S=2048 | 3.2 |

## Phase 1 — algo-level (Level 0): biggest ROI, ~1 day each

### 1.1. Direct first-layer conv kernel
- **Targets**: `conv_32x3x224x224 -54%`
- **Plan**: skip im2col when `C_in ≤ 4`. Loop `(co, oh, ow)` SIMD over `ow`;
  accumulator stays in a vector register; full unroll over `(ci, ky, kx)`.
  Three ow-regions: left-pad (scalar), middle (pure SIMD, no bounds check),
  right-pad (scalar). Bias folded into accumulator init.
- **Files**: new `conv2d_direct_smallcin_sample` in `src/core/conv.c`;
  dispatch from `conv2d_forward` + `conv_bn_relu_forward` when
  `C_in ≤ 4 && (kh, kw) ≤ (7, 7)`.
- **Verify**: `conv_32x3x224x224_64x3` ≥ 56 GFLOPS (TF parity).

### 1.2. Batched 1×1 conv as one giant GEMM
- **Targets**: `conv_*_512x1_k1_s1 -28%`
- **Plan**: in `use_batched` path, detect `kh==kw==1, sh==sw==1, ph==pw==0`
  and bypass `im2col_into_strided` entirely — point the GEMM B-operand at
  the input strided as `[C_in, N·H·W]` (which im2col would have produced
  anyway, identity). Saves N·C_in·H·W floats of copy per call.
- **Files**: `src/core/conv.c` `use_batched_cbr_fwd` and `conv2d_forward`
  batched paths.
- **Verify**: `conv_32x512x14x14_512x1` ≥ 376 GFLOPS.

### 1.3. Save attn weights in MHA forward, reuse in backward
- **Targets**: `mha_train -26% to -62%`
- **Plan**: in MHA forward (when `requires_grad`), save the post-softmax
  attention matrix to `forward_arena`. Backward reads it instead of
  recomputing softmax(QK^T).
- **Memory cost**: BH × S × S floats per layer (B8 S128 H8: 1 MB).
- **Files**: `src/core/mha.c` (sdpa_fwd + sdpa_bwd_head).
- **Verify**: `mha_train_B8_S128_D512_H8` ≥ 500 GFLOPS.

### 1.4. Fuse bias-add into conv GEMM accumulator
- **Targets**: ~5% across all conv layers
- **Plan**: extend `gemm_relu` API to support optional bias and
  optional relu independently. Conv calls with bias only; CBR with both.
- **Files**: `src/compute/backends/cpu_opt.c`, `src/core/conv.c`.

## Phase 2 — micro-kernel + SIMD upgrades (Level 1): ~half day each

### 2.1. SIMD maxpool / avgpool kernel
- **Targets**: `maxpool -11% to -32%`
- **Plan**: vectorize inner `kw` loop with `ax_vf32_max` accumulator.
  Special-case k=2 stride=2 with hand-unrolled 2×2 vmax. ISA-generic via
  `simd_defs.h`.
- **Files**: `src/core/pool.c`.
- **Verify**: `maxpool_32x256x56x56` ≥ 24 GB/s.

### 2.2. Pack-B reuse for transposed GEMM
- **Targets**: `gemm TN/NT 2048×512×512 -19% to -34%`
- **Plan**: (a) include transpose flag in `pack_b_cached` cache key so
  back-to-back identical TN gemms reuse pack; (b) SIMD register-tile
  transpose-during-pack using `_mm256_unpacklo/hi` (AVX2) / `vtrn` (NEON).
- **Files**: `src/compute/backends/cpu_opt.c` `pack_b_cached`, `pack_b`.
- **Verify**: TN/NT 2048×512×512 ≥ 350 GFLOPS.

### 2.3. Per-op parallel-threshold calibration
- **Plan**: at calibration, measure actual serial-vs-parallel crossover
  for representative kernels (sum, add, gemv, softmax). Store 3–4 thresholds,
  not one. Replace global `ax_par_threshold_elems` with per-family lookup.
- **Files**: `src/compute/dispatch.c` `ax_calibrate_thresholds`.

### 2.4. Micro-kernel selection at calibration time
- **Plan**: emit two AVX2 micro-kernel variants (6×16 default + 4×24
  skinny); calibration timed-trials pick best per shape. Same idea for
  AVX512 (16×8 vs 8×16) and NEON (8×8 vs 4×16). All ISA-generic.
- **Files**: `src/compute/backends/cpu_opt.c` `opt_gemm_microkernel`.

## Phase 3 — calibration upgrades (cross-cutting)

### 3.1. Skip pack_a/pack_b when input is already aligned + large
- **Targets**: `huge_gemm_8192 -9.5%`
- **Plan**: when source A is row-major-contiguous, K is multiple of
  micro-kernel KC, and pointer is 64-byte-aligned, point pack_a directly
  at the source. Same for B. Detected at GEMM dispatch.
- **Files**: `src/compute/backends/cpu_opt.c` pack dispatch.

### 3.2. Long-context SDPA tile selection
- **Targets**: `long_ctx_sdpa_S2048 -12%`
- **Plan**: pick Q-tile from `{32, 64, 128}` based on S
  (algorithm-derivable, not measured); fold KV-tile into the same dispatcher.
- **Files**: `src/compute/backends/cpu_opt.c` SDPA section.

### 3.3. Calibrate "all-cores vs fast-cores" crossover
- **Plan**: at calibration, time the same 100M-FLOP GEMM with `fast` vs
  `all` threads; pick the actual crossover point. One short measurement,
  no per-arch logic.
- **Files**: `src/compute/dispatch.c`.

## Phase 4 — fused primitives (more code, biggest model wins)

### 4.1. Fused MHA forward+backward primitive (extends 1.3)
- **Plan**: `ax_mha_step()` does fwd+bwd in one call, owning all
  intermediates. Reuses 1.3 attn-save.
- **Verify**: `mha_train_B8_S128_D512_H8` beats TF.

### 4.2. Conv backward with batched grad-input GEMM
- **Plan**: same trick as forward — batched GEMM across N for both dW
  and dX. Currently per-sample.
- **Verify**: VGG training step drops ~10%.

## Order of execution

| Phase | Item | Why first | Effort |
|---|---|---|---|
| 1.1 | Direct first-layer conv | Single biggest regression (-54%) | 0.5 day |
| 1.2 | Batched 1×1 conv | Common pattern (ResNet/transformer) | 0.5 day |
| 2.1 | SIMD maxpool | Trivial change, multiple cases | 0.5 day |
| 2.2 | Pack-B for TN/NT | Lifts all transposed GEMM | 1 day |
| 1.3 | Save attn for backward | Largest training-perf win | 1 day |
| 1.4 | Fuse conv + bias | ~5% across all conv | 0.5 day |
| 3.1 | Skip-pack heuristic | Closes huge-GEMM gap | 0.5 day |
| 2.3 | Per-op par thresholds | General floor lift | 1 day |
| 2.4 | Micro-kernel calibration | Skinny GEMM lift | 1 day |
| 3.2 | SDPA tile dispatch | Long-context lift | 0.5 day |
| 3.3 | Hybrid crossover calibration | Robustness | 0.5 day |
| 4.1 | Fused MHA fwd+bwd | Final training gap | 2 days |
| 4.2 | Batched conv backward | VGG training | 1 day |

**Total**: ~10 working days. After Phase 1+2 (~4 days) every current regression
should close to ≤10%. After Phase 3 calibration auto-picks per-system without
arch hints. Phase 4 polish to make Axiom faster than TF on every workload
measured.

## Process

After each Phase item:
1. Build clean (`make -C build -j`), zero warnings.
2. Run `ctest -j --output-on-failure` — all 24 tests pass.
3. Run the targeted micro-bench vs TF; verify regression closed.
4. Run a broader suite (gemm/ops/conv) to check no new regression.
5. Commit with one-line subject + 1–3-line body.
