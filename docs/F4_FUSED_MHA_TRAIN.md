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

### F.4.3: tile-fused input projection (Wqkv recompute)

Per-`(qi, kj)` tile, recompute Qh/Kh/Vh slices from `X * Wqkv` instead
of carrying the full Qh/Kh/Vh arena tensors. Trades the Q/K/V
intermediate for extra recompute on each tile visit.

Net win depends on whether the BW saved exceeds the recompute cost.
On large-S shapes (where Q/K/V are big) the BW saving dominates; on
small-S shapes the recompute may outweigh.

### F.4.4: full FA-2 fwd+bwd in one pass

Combine F.4.1+F.4.2+F.4.3 into one monolithic per-tile loop body that
streams every weight grad accumulator. This is the structural match
to what XLA produces.

Expect this to close most of the remaining 15-30 % gap to TF.

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
