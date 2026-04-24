# F.4: fully-fused MHA train kernel — design

## Goal

Match TensorFlow's `@tf.function(jit_compile=True)` (XLA) baseline on
`mha_train` benchmarks by collapsing the entire forward + backward
into one cache-resident pass. The current per-stage path materialises
intermediate tensors (qkv, Qh/Kh/Vh, Oh, attn_flat, dout, d_attn_flat,
dQh/dKh/dVh, dQKV, dWqkv) that traverse DRAM between stages; XLA fuses
across stage boundaries and keeps tile data in registers/L1.

Estimated remaining gap to TF after the per-stage wins shipped this
iteration:
- B8_S128: ~22 ms vs TF 11 ms (TF -50 %)
- B4_S512: ~85 ms vs TF 65 ms (TF -23 %)
- B1_S2048: ~175 ms vs TF 155 ms (TF -11 %)

The cross-stage tile fusion this kernel enables is what closes this
gap. ~2 k LOC, multi-week incremental delivery.

## Architecture

`ax_mha_train_step_fused(layer, x, dout, y_out)` — same public
signature as `ax_mha_train_step` (commit 623ea45) but the
implementation is a single monolithic kernel.

Per-(qi, kj) tile work (drawn for one head; outer loop over heads):

```
load_or_recompute Qh_tile = X[qi:qi+Bq] @ Wq_h + bq_h    # [Bq, dk]
load_or_recompute Kh_tile = X[kj:kj+Bk] @ Wk_h + bk_h    # [Bk, dk]
load_or_recompute Vh_tile = X[kj:kj+Bk] @ Wv_h + bv_h    # [Bk, dk]

# FA-2 forward step
S_tile  = Qh_tile @ Kh_tile^T * scale          # [Bq, Bk]; causal mask
P_tile  = exp(S_tile - L_running)              # [Bq, Bk]
O_local = P_tile @ Vh_tile                     # [Bq, dk]; head's contribution
                                               # to attn_flat[qi:qi+Bq] for h

# After all kj for this qi tile have accumulated O_local:
# attn_flat[qi:qi+Bq, :] holds the full sum across heads
# (per-head O_local writes into [qi:qi+Bq, h*dk:(h+1)*dk])

# Forward output projection (when full qi tile is ready)
y[qi:qi+Bq] = attn_flat[qi:qi+Bq] @ Wo + bo

# Backward begins (uses dout[qi:qi+Bq])
dout_tile = dout[qi:qi+Bq]                     # [Bq, D]
dWo += attn_flat[qi:qi+Bq]^T @ dout_tile       # accumulate into Wo grad
dbo += sum(dout_tile, axis=0)                  # accumulate into bo grad
dattn_tile = dout_tile @ Wo^T                  # [Bq, D]; dO for SDPA bwd

# SDPA backward (FA-2 style, per (qi, kj) tile)
dV_tile += P_tile^T @ dO_tile                  # [Bk, dk] partial
dP_tile = dO_tile @ Vh_tile^T                  # [Bq, Bk]
dS_tile = P_tile * (dP_tile - Di[qi:qi+Bq])    # softmax bwd
dQ_tile += dS_tile @ Kh_tile                   # [Bq, dk]
dK_tile += dS_tile^T @ Qh_tile                 # [Bk, dk] partial

# Wqkv weight grads (per-tile accumulation)
dWq += X[qi:qi+Bq]^T @ dQ_tile                 # accumulating
dWk += X[kj:kj+Bk]^T @ dK_tile
dWv += X[kj:kj+Bk]^T @ dV_tile
dbq += sum(dQ_tile, axis=0)
dbk += sum(dK_tile, axis=0)
dbv += sum(dV_tile, axis=0)
```

## Phased plan

Each phase is a separate commit landing on `ax_mha_train_step_fused`.
All phases share the same parity test against the autograd path, so
correctness regressions surface immediately.

### F.4.0: scaffold + parity test

- Public API `ax_mha_train_step_fused` — initially a thin wrapper around
  `ax_mha_train_step` so the parity test can run end-to-end.
- `test_mha_train_step_fused_parity` — same shape as the existing
  `test_mha_train_step_parity`, pointing at the new entry.
- `bench_mha` adds an `(F2)` row alongside the existing `(F)` row.

This phase ships the contract; subsequent phases swap the body.

### F.4.1: FA-2 fused as the default sdpa_bwd path

Set `attn_bwd_kj_block_fused` as the default in
`ax_attn_bwd_get_impl`. AX_SDPA_FUSED=1 was opt-in because earlier
measurements were mixed; with the OMP_PROC_BIND=spread + dynamic-head
schedule wins this iteration, re-bench to see if the fused variant
now wins consistently.

**Re-bench result (5-run medians, post OMP_PROC_BIND wins)**:

| shape                            | default | FUSED | delta |
|----------------------------------|---------|-------|-------|
| mha_train_B8_S128_D512_H8        | 23.5    | 21.1  | -10%  |
| mha_train_B4_S512_D768_H12       | 106     | 104   | -2%   |
| mha_train_B2_S1024_D768_H12      | 146     | 144   | -1%   |
| mha_train_B1_S512_D1024_H16      | 48.5    | 54.2  | **+12%** |
| mha_train_B1_S2048_D768_H12      | 215     | 212   | -1%   |

B1_S512_D1024_H16 (BH=16, dk=64, S=512) regresses cleanly. Hypothesis:
this shape has BH == NT exactly so each thread owns one head — the
fused kernel's MR-strip stack-array approach (P/dP/dS in stack)
allocates ~190 KB stack per head call, and on this shape each thread
bears that stack pressure once per call. The materialised path uses
TLS heap buffers shared across calls so the per-call stack pressure
is lower.

Defer this phase — the cross-shape variance means it shouldn't be a
blanket default. Either:
- shape-dispatched (use FUSED only when BH > NT or stack budget OK),
- or revisit after F.4.4 (whose per-tile design subsumes this choice).

### F.4.2: tile-fused output projection

Replace the monolithic `attn_flat → Wo → y` + `dout → Wo^T → dattn`
with a per-`qi`-strip pass that handles both directions. dout_tile
lives in L1 throughout the dWo + dattn computation.

Estimated win: 1-3 % (eliminates one full attn_flat / dattn DRAM
round-trip per layer call).

### F.4.3: tile-fused input projection (Wqkv recompute) — DEFERRED

**Status: deferred.** The per-tile recompute cost exceeds the BW
saving on this CPU. Analysis for the worst-case shape
(B=1, S=2048, D=768, H=12, Bq=Bk=32):

| quantity | value |
|---|---|
| Qh/Kh/Vh full materialisation (one projection) | 2 * rows * D² = 2 * 2048 * 768² ≈ 2.4 Gflops |
| Full Wqkv projection (Q+K+V combined) | 3 * 2.4 = 7.2 Gflops |
| Per-tile Q recompute | 2 * Bq * D * dk = 3.15 Mflops |
| Q recomputes per head for outer-qi inner-kj loop | S/Bq = 64 (once per qi tile) |
| K (or V) recomputes per head | (S/Bq) * (S/Bk) = 64 * 64 = 4096 |
| Extra Q/K/V recompute per head | 64 + 2 * 4096 tile-ops * 3.15 Mflops ≈ 26 Gflops |
| Extra across H=12 heads | 312 Gflops |
| Wall clock of extra recompute @ ~500 GFLOPS aggregate peak | ~620 ms |
| BW saved by skipping Qh/Kh/Vh materialisation (3 * rows * D floats) | ~18 MB |
| Wall clock saved @ 40 GB/s DRAM BW | ~0.45 ms |

Net: **+620 ms flops cost versus ~0.45 ms BW saving** on the big
shape. Even accounting for the recomputed K/V staying cache-hot
across the qi dimension (so only the first qi iteration truly hits
DRAM for K/V), the flops penalty is orders of magnitude over the
BW win on AVX2 hardware without tensor-core acceleration.

FA-2's per-tile recompute on GPU pays off because GPU flops are
essentially free relative to HBM BW (e.g., A100: 312 TFLOPS tensor
fp16 vs 2 TB/s HBM → flops/BW ratio ~150 ops/byte); on CPU AVX2 the
ratio is ~10 ops/byte, flipping the tradeoff.

**Path forward:** skip standalone F.4.3. F.4.4's monolithic per-tile
loop still eliminates the full Qh/Kh/Vh materialisation for the
fwd+bwd shared tile workspace (the point at which the data is
in-register already for SDPA fwd and the same tile gets reused for
SDPA bwd). The BW win there comes from fwd→bwd register/L1 reuse,
not from skipping the projection itself.

### F.4.4: full FA-2 fwd+bwd in one pass

Combine F.4.1+F.4.2+F.4.3 into one monolithic per-tile loop body that
streams every weight grad accumulator. This is the structural match
to what XLA produces.

Expect this to close most of the remaining 15-30 % gap to TF.

**Status (2026-04-24): staged via primitive composition.** The plan's
fully-monolithic per-(qi, kj) kernel body is multi-week scope. This
session's path builds F.4.4 from composable primitives instead:

| primitive | what it eliminates | wired in |
|---|---|---|
| F.3.a `opt_qkv_head_gemm` | qkv [rows, 3D] intermediate (~18 MB) + head_split layout pass | train_step.c, train_step_fused.c |
| F.3.c `opt_dwqkv_split_acc` | dWqkv [D, 3D] intermediate (~6 MB) | train_step.c (D >= 1024) |
| F.3.d `opt_dattn_head_gemm_nt` | d_attn_flat [rows, D] intermediate (~6 MB) | train_step.c, train_step_fused.c |
| F.3.e `ax_fused_attention_fwd_save_to_flat` | Oh [BH, S, dk] intermediate (~6 MB) + head_deinterleave pass | train_step.c, train_step_fused.c (default) |
| F.3.e companion `bwd_use_from_flat` | reads O strided from attn_flat instead of materialised Oh | train_step.c, train_step_fused.c (default) |
| F.4.2 proper `opt_mha_output_proj_fused` | strip-fused y + dWo + dbo + dattn (one pass over dout) | train_step_fused.c (opt-in via AX_F42_PROPER=1; default-on regresses ~50-160 % because the simple AVX2 inner loops don't match opt_gemm's packed micro_kernel — needs ~500 LOC rewrite to use micro_kernel for parity) |

Cumulative DRAM traffic eliminated per train_step call on
`B1_S2048_D768_H12` (the largest shape): qkv 18 MB + dattn 6 MB +
Oh 6 MB + head_deinterleave-pass 12 MB ≈ **42 MB saved**.

**True monolithic kernel (still TODO):** the remaining structural
fusion that's NOT in the primitive composition above is:

1. **Per-(qi, kj) FA-2 fwd+bwd combined** — currently fwd and bwd are
   separate per-head functions (`attn_fwd_head` / `attn_bwd_head`)
   with the output projection (and a barrier) between. fully-fused
   would do `fwd-tile then bwd-tile` per (qi, kj) inside one function,
   eliminating L's full materialisation and re-using Q_tile / K_tile /
   V_tile across fwd+bwd register-resident state.

   Blocked by: the output projection mixes heads (Wo is [D, D] not
   per-head), so `dattn_tile` for head h depends on `attn_flat_tile`
   contributions from ALL heads. A per-head fwd+bwd structure would
   need either (a) per-head Wo decomposition (mathematically equivalent
   only when Wo is reformulated as a sum of per-head outer products),
   or (b) outer qi-block loop with a within-qi-block barrier for the
   output projection step — the second is feasible but is essentially
   the staged-orchestration that the primitive composition above
   already realises at full-S granularity. Going to per-block
   granularity adds ~5 % memory wins on attn_flat (block-resident
   instead of full-rows) but requires a custom per-qi-block SDPA
   fwd+bwd entry (~400 LOC).

2. **Per-tile `dWqkv` accumulation** — currently dQh/dKh/dVh are full
   tensors materialised by SDPA bwd, then merged into `dQKV` and fed
   to `opt_dwqkv_split_acc` (F.3.c) for the dWqkv += X^T @ dQKV update.
   per-tile would update `dWq[h_slice] += X[qi]^T @ dQ_tile` and
   similarly for dWk/dWv during the SDPA bwd inner loop, eliminating
   the dQh/dKh/dVh + dQKV intermediates (~36 MB on B1_S2048).

   Blocked by: dWk and dWv accumulate per-kj (need dK_tile / dV_tile
   summed over qi for given kj). dWq accumulates per-qi (dQ_tile
   summed over kj for given qi). Different loop nestings needed,
   forcing either separate qi-outer + kj-outer passes (defeats
   fusion) or holding dQ_tile / dK_tile per-qi or per-kj in scratch
   (same as current dQh/dKh/dVh — no savings).

   Resolvable but requires either custom per-(qi, kj) double-loop
   that computes dWq inside qi-outer and stashes dK/dV partials per
   kj for a later qi-level dWk/dWv reduction — ~500-700 LOC.

Both items require deep cpu_opt.c surgery and are deferred to a
future session with dedicated 1-2 week budget. The primitive
composition above is the practical realisation for this session;
the building blocks are all in place when the monolithic kernel
work resumes.

### F.4.5: bench + iterate

5-run medians on every shape; iterate F.4.4 until Axiom ≥ TF on every
mha_train benchmark.

## Risks

- **Accumulator races**: Multiple tiles write to the same dWq / dWo
  rows. Need either per-tile accumulators + final reduction (the
  pattern used in `ax_attn_bwd_inner_threads` for dQ) or atomic adds.
- **Per-tile gemm overhead**: Small gemms (Bq×dk) have lower
  flops/sec than full gemms. Need to verify the BW saving exceeds the
  flops/sec hit.
- **Code size**: This kernel will be ~1500 LOC. Keep it in
  `attention/train_step_fused.c` separate from `train_step.c` so the
  per-stage version remains the readable reference.
- **JIT micro-kernel reuse**: The existing 6×16 / 14×32 / 8×12 / 4×4
  micro-kernels work at fixed `MR × NR`. Per-tile gemms with
  `Bq × dk = 126 × 64` need careful tile-edge handling.
