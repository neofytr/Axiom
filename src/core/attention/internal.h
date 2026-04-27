/* attention/internal.h — file-private contracts shared between the
   pieces of the multi-head attention module (layer.c, layout.c, cache.c).

   not part of any public api — kept under src/core/attention/ rather
   than include/axiom/internal/ because the contract changes whenever
   the mha layer is restructured. user code that needs MHA should go
   through the public ax_mha_create / ax_layer_forward api in
   axiom/attention.h. */

#ifndef AX_CORE_ATTENTION_INTERNAL_H
#define AX_CORE_ATTENTION_INTERNAL_H

#include "axiom/tensor.h"
#include "axiom/attention.h"
#include "axiom/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
   shared per-thread scratch arena for all train_step variants.
   the three variants (train_step, train_step_fused, train_step_v4)
   are mutually exclusive per call — only one runs at a time on a
   given thread. sharing one TLS arena saves 2 × 16 MB per thread
   vs the original three separate arenas.
   defined in train_step.c.
   ================================================================ */

extern __thread ax_arena_t *tl_train_shared_arena;
ax_arena_t *ax_train_get_scratch_arena(void);

/* ================================================================
   layout helpers (defined in attention/layout.c)
   ================================================================ */

/* transpose a [B, S, H, dk] row-major buffer into [B, H, S, dk]
   (= [BH, S, dk]). distinct src/dst required. */
void ax_attn_head_interleave(const float *src, float *dst,
                              int64_t B, int64_t S, int64_t H, int64_t dk);

/* inverse of head_interleave: [B, H, S, dk] -> [B, S, H, dk]. */
void ax_attn_head_deinterleave(const float *src, float *dst,
                                int64_t B, int64_t S, int64_t H, int64_t dk);

/* head-deinterleave into one slot of a wider [rows, dst_cols] buffer.
   used by the backward path to pack dQh/dKh/dVh into a single dQKV
   tensor for fused gradient gemms. */
void ax_attn_head_deinterleave_slot(const float *src, float *dst,
                                     int64_t B, int64_t S, int64_t H, int64_t dk,
                                     int64_t dst_cols, int64_t col_off);

/* fused merge of dQh/dKh/dVh into a single [rows, 3D] buffer. one
   parallel pass; replaces three back-to-back deinterleave_slot calls. */
void ax_attn_head_deinterleave_qkv_merge(const float *srcQ, const float *srcK,
                                          const float *srcV, float *dst,
                                          int64_t B, int64_t S, int64_t H,
                                          int64_t dk, int64_t D);

/* head-interleave from one slot of a wider [rows, src_cols] buffer.
   inverse of head_deinterleave_slot. */
void ax_attn_head_interleave_from_slot(const float *src, float *dst,
                                        int64_t B, int64_t S, int64_t H, int64_t dk,
                                        int64_t src_cols, int64_t col_off);

/* fused split of [rows, 3D] into three [BH, S, dk] tensors. one
   parallel pass; replaces three back-to-back interleave_from_slot calls. */
void ax_attn_head_interleave_qkv_split(const float *src,
                                        float *dstQ, float *dstK, float *dstV,
                                        int64_t B, int64_t S, int64_t H, int64_t dk,
                                        int64_t D);

/* fused split + bias add. bias layout = [bq[D] | bk[D] | bv[D]]. */
void ax_attn_head_interleave_qkv_split_bias(const float *src, const float *bias,
                                             float *dstQ, float *dstK, float *dstV,
                                             int64_t B, int64_t S, int64_t H, int64_t dk,
                                             int64_t D);

/* ================================================================
   cache helpers (defined in attention/cache.c)
   ================================================================ */

/* rebuild the fused [D, 3D] Wqkv panel and [3D] bqkv vector from the
   individual Wq/Wk/Wv/bq/bk/bv layer parameters. checked for staleness
   via the storage generation counter — no-op when nothing has changed. */
void ax_attn_refresh_fused_qkv(ax_mha_t *m);

#ifdef __cplusplus
}
#endif

#endif /* AX_CORE_ATTENTION_INTERNAL_H */
