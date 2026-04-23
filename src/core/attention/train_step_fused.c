/* attention/train_step_fused.c — F.4: fully-fused MHA train kernel.

   target: match TF's @tf.function(jit_compile=True) baseline by
   collapsing the entire forward + backward into one cache-resident
   pass. same external contract as ax_mha_train_step but the body is
   a per-(qi, kj) tile loop that streams every intermediate through
   L1 — no qkv / Qh / Kh / Vh / Oh / attn_flat / dout / d_attn_flat
   / dQh / dKh / dVh / dQKV / dWqkv arena tensors between stages.

   phased delivery (see docs/F4_FUSED_MHA_TRAIN.md):

     F.4.0 (this file, initial commit):
       thin wrapper around ax_mha_train_step. ships the public API
       contract + parity test so subsequent phases can swap the body
       in place without touching call sites.

     F.4.1: AX_SDPA_FUSED=1 default if it now wins consistently after
       the OMP_PROC_BIND=spread + dynamic-head-schedule wins.

     F.4.2: tile-fused output projection (dWo + dattn share a single
       qi-strip pass over dout).

     F.4.3: tile-fused input projection (per-tile X*Wqkv recompute,
       dropping the Qh/Kh/Vh full materialisation).

     F.4.4: full FA-2 fwd+bwd in one pass with all weight grad
       accumulators streaming per tile.

   each phase keeps the parity test green and is a separate commit. */

#include "axiom/attention.h"
#include "axiom/error.h"

ax_status_t ax_mha_train_step_fused(ax_layer_t *layer,
                                     const ax_tensor_t *x,
                                     const ax_tensor_t *dout,
                                     ax_tensor_t *y_out)
{
    /* F.4.0: contract-only scaffold. delegates to ax_mha_train_step
       so the parity test passes immediately. each subsequent phase
       (F.4.1 onward) swaps progressively more of the body in place. */
    return ax_mha_train_step(layer, x, dout, y_out);
}
