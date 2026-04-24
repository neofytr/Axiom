/* attention/train_step_v4.c — F.4.4 monolithic train kernel scaffold.

   the public entry ax_mha_train_step_v4 is the landing site for the
   multi-phase F.4.4 work documented in docs/F4_FUSED_MHA_TRAIN.md
   ("F.4.4 phased implementation plan"):

     Phase A: per-qi-block SDPA fwd/bwd primitives in cpu_opt.c
     Phase B: per-qi-block driver in this file
     Phase C: per-tile dWqkv accumulation (eliminates dQh/dKh/dVh + dQKV)
     Phase D: per-tile output-projection fusion across qi-block boundary
     Phase E: per-(qi, kj) tile-level fwd+bwd combined kernel
     Phase F: bench + iterate

   each phase is a separate commit landing on this entry point. parity
   test test_mha_train_step_v4_parity stays green on every commit.

   initial scaffold (this commit): delegates to ax_mha_train_step_fused
   so the API + parity test contract are wired. each phase swaps more
   of the body in place — by Phase E this function will be a full
   monolithic per-(qi, kj) tile loop body, eliminating Qh/Kh/Vh,
   attn_flat-as-intermediate (kept as cache-resident block staging),
   dattn, dQh/dKh/dVh, dQKV intermediates entirely.

   Phase B already partially realised by composition: train_step_fused.c
   uses F.3.a (qkv → Qh/Kh/Vh fused), F.3.e (sdpa fwd → attn_flat
   directly, no Oh), F.3.e companion (sdpa bwd from attn_flat strided),
   F.3.d (dattn → dO_head fused, opt-in), F.3.c (dwqkv split fused),
   and F.4.2 reorder (output projection ordering for cache reuse).
   what's missing is the per-qi-block restructuring (Phase B proper)
   and the per-tile dWqkv elimination (Phase C). */

#include "axiom/attention.h"
#include "axiom/error.h"

ax_status_t ax_mha_train_step_v4(ax_layer_t *layer,
                                  const ax_tensor_t *x,
                                  const ax_tensor_t *dout,
                                  ax_tensor_t *y_out)
{
    /* scaffold: delegate to the existing fused entry. each F.4.4
       phase swaps progressively more of the body in place. */
    return ax_mha_train_step_fused(layer, x, dout, y_out);
}
