/* attention/train_step_v4.c — F.4.4 monolithic train kernel.

   Phase B: per-qi-block driver. same external contract as
   ax_mha_train_step_fused, but the SDPA fwd, d_attn gemm_nt, head
   interleave, and SDPA bwd are sequenced inside a single qi-block
   loop so attn_flat[qi-block] / dO_head[qi-block] stay cache-hot
   across the fwd → interleave → bwd handoff (the big win vs
   train_step_fused, where attn_flat is a full [rows, D] intermediate
   that spills to L3 between stages).

   stays outside the loop (done once over full-rows):
     - input-proj qkv + head_split (Phase C eliminates)
     - output-proj fwd y = attn_flat @ Wo + bo (Phase D fuses in-loop)
     - dWo / dbo accumulation (Phase D fuses in-loop)
     - dQKV merge + dWqkv + dbqkv (Phase C eliminates dQKV)

   parity with ax_mha_train_step / ax_mha_train_step_fused is maintained
   via test_mha_train_step_v4_parity (tests/test_attention.c).

   later phases: C — per-tile dWqkv; D — per-tile output proj fusion;
   E — per-(qi, kj) tile-level combined fwd+bwd kernel. */

#include "axiom/attention.h"
#include "axiom/compute.h"
#include "axiom/error.h"
#include "axiom/memory.h"
#include "axiom/tensor.h"
#include "axiom/internal/cuda_extension.h"
#include "internal.h"

#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../compute/backends/simd_defs.h"

/* same contract as train_step.c / train_step_fused.c — tells the gemm
   backends to accumulate into dst rather than overwrite. */
extern void ax_gemm_set_skip_init(bool v);

/* per-thread scratch arena, reset per call. separate TLS slot so v4
   and _fused can coexist without aliasing. */
static __thread ax_arena_t *tl_train_v4_arena = NULL;

static ax_arena_t *get_scratch_arena(void)
{
    if (!tl_train_v4_arena) {
        tl_train_v4_arena = ax_arena_create(16 * 1024 * 1024);
        if (!tl_train_v4_arena) return NULL;
    } else {
        ax_arena_reset(tl_train_v4_arena);
    }
    return tl_train_v4_arena;
}

static inline void make_stack_view(ax_tensor_t *tv, ax_storage_t *st,
                                    const float *data,
                                    int64_t rows, int64_t cols)
{
    st->data         = (void *)(uintptr_t)data;
    st->size_bytes   = (size_t)(rows * cols) * sizeof(float);
    atomic_store(&st->refcount, 0);
    st->device       = AX_DEVICE_CPU;
    st->is_arena_temp = true;
    st->generation   = 1;
    memset(tv, 0, sizeof(*tv));
    tv->storage      = st;
    tv->ndim         = 2;
    tv->dtype        = AX_FLOAT32;
    tv->shape[0]     = rows;
    tv->shape[1]     = cols;
    tv->strides[0]   = cols;
    tv->strides[1]   = 1;
}

static float *ensure_grad_ptr(ax_tensor_t *p)
{
    if (!p || !p->requires_grad) return NULL;
    if (!p->grad) {
        p->grad = ax_tensor_zeros(p->shape, p->ndim, p->dtype);
        if (!p->grad) return NULL;
    }
    return (float *)p->grad->storage->data;
}

static void acc_f32(const float *src, float *dst, int64_t n)
{
    int64_t i = 0;
#if defined(AX_HAS_SIMD)
    int64_t ve = n - (n % AX_VF32_WIDTH);
    for (; i < ve; i += AX_VF32_WIDTH) {
        ax_vf32 s = ax_vf32_loadu(src + i);
        ax_vf32 d = ax_vf32_loadu(dst + i);
        ax_vf32_storeu(dst + i, ax_vf32_add(s, d));
    }
#endif
    for (; i < n; i++) dst[i] += src[i];
}

static void col_sum_acc(const float *M, float *out,
                         int64_t rows, int64_t cols)
{
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t c = 0; c < cols; c++) {
        float s = 0.0f;
        for (int64_t r = 0; r < rows; r++) s += M[r * cols + c];
        out[c] += s;
    }
}

static void fill_f32(float *p, float v, int64_t n)
{
    int64_t i = 0;
#if defined(AX_HAS_SIMD)
    ax_vf32 vv = ax_vf32_set1(v);
    int64_t ve = n - (n % AX_VF32_WIDTH);
    for (; i < ve; i += AX_VF32_WIDTH) ax_vf32_storeu(p + i, vv);
#endif
    for (; i < n; i++) p[i] = v;
}

/* qi-range variant of ax_attn_head_interleave. d_attn is [B, S, D]
   row-major; dO_head is [B, H, S, dk] (= [BH, S, dk]). copies only
   rows [qi_start, qi_end) — rest of dO_head is untouched, assumed
   already populated by prior qi-block iterations. a qi-range-aware
   fused gemm_nt + interleave (Phase D) would skip this copy entirely. */
static void head_interleave_qi_range(const float *d_attn, float *dO_head,
                                      int64_t B, int64_t S, int64_t H, int64_t dk,
                                      int64_t qi_start, int64_t qi_end)
{
    int64_t D = H * dk;
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t s = qi_start; s < qi_end; s++) {
                const float *src = d_attn + b * S * D + s * D + h * dk;
                float *dst = dO_head + (b * H + h) * S * dk + s * dk;
                memcpy(dst, src, (size_t)dk * sizeof(float));
            }
        }
    }
}

/* qi_block size selection. AX_V4_QI_BLOCK env overrides the default;
   clamp to [1, S]. default 126 matches ATTN_BQ_DEFAULT (the SDPA
   kernel's internal tile) so an entire qi-block is one SDPA tile-row
   on AVX2 (GEMM_MR=6; 126 = 21 MR strips). read each call so tests
   can toggle it without a restart. */
static inline int64_t resolve_qi_block(int64_t S)
{
    const char *e = getenv("AX_V4_QI_BLOCK");
    int64_t block = (e && e[0]) ? (int64_t)atoll(e) : 126;
    if (block < 1) block = 1;
    return block > S ? S : block;
}

ax_status_t ax_mha_train_step_v4(ax_layer_t *layer,
                                  const ax_tensor_t *x,
                                  const ax_tensor_t *dout,
                                  ax_tensor_t *y_out)
{
    if (!layer || !x || !y_out) {
        ax_err_set(AX_ERR_NULL_ARG, "ax_mha_train_step_v4: NULL arg");
        return AX_ERR_NULL_ARG;
    }
    ax_mha_t *m = (ax_mha_t *)layer;
    if (x->ndim != 3 || y_out->ndim != 3) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "train_step_v4: x/y must be 3-D");
        return AX_ERR_SHAPE_MISMATCH;
    }
    int64_t B  = x->shape[0];
    int64_t S  = x->shape[1];
    int64_t D  = x->shape[2];
    if (D != m->d_model) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "train_step_v4: x last dim != d_model");
        return AX_ERR_SHAPE_MISMATCH;
    }
    if (y_out->shape[0] != B || y_out->shape[1] != S || y_out->shape[2] != D) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "train_step_v4: y_out shape mismatch");
        return AX_ERR_SHAPE_MISMATCH;
    }
    if (x->dtype != AX_FLOAT32 || y_out->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "train_step_v4: must be fp32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    if (dout) {
        if (dout->ndim != 3 ||
            dout->shape[0] != B || dout->shape[1] != S || dout->shape[2] != D ||
            dout->dtype != AX_FLOAT32) {
            ax_err_set(AX_ERR_SHAPE_MISMATCH, "train_step_v4: dout shape mismatch");
            return AX_ERR_SHAPE_MISMATCH;
        }
    }

    int64_t H    = m->n_heads;
    int64_t dk   = m->d_k;
    int64_t rows = B * S;
    int64_t bh   = B * H;

    ax_arena_t *arena = get_scratch_arena();
    if (!arena) {
        ax_err_set(AX_ERR_ALLOC, "train_step_v4: scratch arena alloc failed");
        return AX_ERR_ALLOC;
    }

    ax_attn_refresh_fused_qkv(m);

    ax_tensor_t  x_flat_v;
    ax_storage_t x_flat_st;
    make_stack_view(&x_flat_v, &x_flat_st,
                     (const float *)x->storage->data, rows, D);

    ax_tensor_t  y_flat_v;
    ax_storage_t y_flat_st;
    make_stack_view(&y_flat_v, &y_flat_st,
                     (const float *)y_out->storage->data, rows, D);

    int64_t qkv_sh[]  = {rows, 3 * D};
    int64_t head_sh[] = {bh, S, dk};
    int64_t L_sh[]    = {bh, S};
    int64_t flat_sh[] = {rows, D};

    /* input projection + head split (full-rows). F.3.a fused path writes
       directly to Qh/Kh/Vh; fallback is unfused gemm + split. Phase C
       will per-tile-recompute X @ Wqkv inside the qi-block loop to
       eliminate Qh/Kh/Vh. */
    ax_tensor_t *Qh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    ax_tensor_t *Kh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    ax_tensor_t *Vh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    if (!Qh || !Kh || !Vh) return AX_ERR_ALLOC;

    if (ax_compute_has_qkv_head_gemm()) {
        const ax_tensor_t *bqkv_arg = m->use_bias ? m->bqkv_cache : NULL;
        if (ax_compute_qkv_head_gemm(&x_flat_v, m->Wqkv_cache, bqkv_arg,
                                       B, S, H, dk, Qh, Kh, Vh) != AX_OK)
            return AX_ERR_BACKEND;
    } else {
        ax_tensor_t *qkv = ax_tensor_arena_create(arena, qkv_sh, 2, AX_FLOAT32);
        if (!qkv) return AX_ERR_ALLOC;
        if (ax_compute_gemm(&x_flat_v, m->Wqkv_cache, qkv) != AX_OK) return AX_ERR_BACKEND;
        if (m->use_bias) {
            ax_attn_head_interleave_qkv_split_bias(
                (const float *)qkv->storage->data,
                (const float *)m->bqkv_cache->storage->data,
                (float *)Qh->storage->data,
                (float *)Kh->storage->data,
                (float *)Vh->storage->data,
                B, S, H, dk, D);
        } else {
            ax_attn_head_interleave_qkv_split(
                (const float *)qkv->storage->data,
                (float *)Qh->storage->data,
                (float *)Kh->storage->data,
                (float *)Vh->storage->data,
                B, S, H, dk, D);
        }
    }

    /* L (log-sum-exp per qi row, needed by SDPA bwd) + optional P_save
       (skip softmax recompute on bwd). same gating as train_step_fused. */
    ax_tensor_t *L_t = ax_tensor_arena_create(arena, L_sh, 2, AX_FLOAT32);
    if (!L_t) return AX_ERR_ALLOC;

    ax_tensor_t *P_save_t = NULL;
    int64_t p_save_bytes = bh * S * S * (int64_t)sizeof(float);
    bool save_p = (p_save_bytes <= (int64_t)8 * 1024 * 1024);
    if (save_p && S <= 128 && dk <= 64) save_p = false;
    const char *env_save_p = getenv("AX_MHA_SAVE_P");
    if (env_save_p) save_p = (env_save_p[0] == '1');
    if (save_p) {
        int64_t P_sh[] = {bh, S, S};
        P_save_t = ax_tensor_arena_create(arena, P_sh, 3, AX_FLOAT32);
    }
    float scale = 1.0f / sqrtf((float)dk);
    float *P_save_ptr = P_save_t ? (float *)P_save_t->storage->data : NULL;

    /* dout view (defaults to ones when NULL — parity with train_step.c
       scalar-loss convention used by the graph test harness). */
    ax_tensor_t  dout_flat_v;
    ax_storage_t dout_flat_st;
    ax_tensor_t *dout_flat;
    ax_tensor_t *dout_owned = NULL;
    if (dout) {
        make_stack_view(&dout_flat_v, &dout_flat_st,
                         (const float *)dout->storage->data, rows, D);
        dout_flat = &dout_flat_v;
    } else {
        dout_owned = ax_tensor_arena_create(arena, flat_sh, 2, AX_FLOAT32);
        if (!dout_owned) return AX_ERR_ALLOC;
        fill_f32((float *)dout_owned->storage->data, 1.0f, rows * D);
        dout_flat = dout_owned;
    }

    /* working tensors that span all of S — the qi-block loop writes
       into disjoint [qi_start, qi_end) row-ranges of each. */
    ax_tensor_t *attn_flat = ax_tensor_arena_create(arena, flat_sh, 2, AX_FLOAT32);
    ax_tensor_t *d_attn    = ax_tensor_arena_create(arena, flat_sh, 2, AX_FLOAT32);
    ax_tensor_t *dO_head   = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    ax_tensor_t *dQh       = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    ax_tensor_t *dKh       = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    ax_tensor_t *dVh       = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    if (!attn_flat || !d_attn || !dO_head || !dQh || !dKh || !dVh) return AX_ERR_ALLOC;

    /* the qi-block bwd primitive overwrites dQh[qi_start..qi_end] per call
       but ACCUMULATES into dK/dV — caller must pre-zero dKh/dVh once
       before the first block call. dQh needs no pre-zero since each
       block owns its row range. */
    memset((float *)dKh->storage->data, 0, (size_t)bh * S * dk * sizeof(float));
    memset((float *)dVh->storage->data, 0, (size_t)bh * S * dk * sizeof(float));

    int64_t qi_block = resolve_qi_block(S);

    /* Phase C: per-qi-block dWq accumulation. dQh[b*H+h, qi..qi_end, :] is
       final after each bwd call (dQ is overwritten per block, not
       accumulated), so we can deinterleave it into a [bq, D] layout per
       batch and immediately gemm_tn into dWq — no full-S dQKV buffer.
       dWk/dWv still need full-S dKh/dVh (cross-qi accumulation inside the
       bwd primitive), so they're handled after the loop.

       the deinterleave buffer is sized for the largest possible qi-block
       (bq_max = qi_block) and reused across iterations. */
    int64_t bq_max = qi_block;
    int64_t db_sh1[] = {3 * D};
    (void)db_sh1;  /* silences unused when bias branch doesn't fire */
    float *dQh_batch_buf = NULL;
    ax_tensor_t *dQh_batch_view = NULL;
    ax_storage_t *dQh_batch_st = NULL;
    bool need_wq_grad = m->Wq->requires_grad;
    bool need_bq_grad = m->use_bias && m->bq && m->bq->requires_grad;
    if (need_wq_grad || need_bq_grad) {
        int64_t buf_sh[] = {bq_max, D};
        ax_tensor_t *dQh_buf_t = ax_tensor_arena_create(arena, buf_sh, 2, AX_FLOAT32);
        if (!dQh_buf_t) return AX_ERR_ALLOC;
        dQh_batch_buf = (float *)dQh_buf_t->storage->data;
        dQh_batch_view = dQh_buf_t;
    }
    /* ensure grad buffers exist for the streaming accumulator path (skip_init
       on gemm_tn requires a non-null dst tensor). */
    if (need_wq_grad) {
        if (!ensure_grad_ptr(m->Wq)) return AX_ERR_ALLOC;
    }
    float *dbq_acc = NULL;
    if (need_bq_grad) {
        dbq_acc = ensure_grad_ptr(m->bq);
        if (!dbq_acc) return AX_ERR_ALLOC;
    }

    for (int64_t qi = 0; qi < S; qi += qi_block) {
        int64_t qi_end = (qi + qi_block <= S) ? (qi + qi_block) : S;
        int64_t bq = qi_end - qi;

        /* (1) SDPA fwd for [qi, qi_end) → attn_flat rows, L rows, P_save rows */
        ax_fused_attention_fwd_save_to_flat_qi_block(
            (const float *)Qh->storage->data,
            (const float *)Kh->storage->data,
            (const float *)Vh->storage->data,
            (float *)attn_flat->storage->data,
            (float *)L_t->storage->data,
            P_save_ptr,
            B, S, H, dk, qi, qi_end, scale, m->causal);

        /* (2) d_attn[b, qi..qi_end, :] = dout[b, qi..qi_end, :] @ Wo^T
              sub-blocks are contiguous [bq, D] within each batch b, so B
              separate gemm_nt calls — no row stride mismatch. */
        for (int64_t b = 0; b < B; b++) {
            const float *dout_ptr = (const float *)dout_flat->storage->data
                                    + b * S * D + qi * D;
            float *dattn_ptr = (float *)d_attn->storage->data
                               + b * S * D + qi * D;
            ax_tensor_t dout_v, dattn_v;
            ax_storage_t dout_st, dattn_st;
            make_stack_view(&dout_v,  &dout_st,  dout_ptr,  bq, D);
            make_stack_view(&dattn_v, &dattn_st, dattn_ptr, bq, D);
            if (ax_compute_gemm_nt(&dout_v, m->Wo, &dattn_v) != AX_OK)
                return AX_ERR_BACKEND;
        }

        /* (3) head_interleave d_attn[:, qi..qi_end, :] → dO_head */
        head_interleave_qi_range(
            (const float *)d_attn->storage->data,
            (float *)dO_head->storage->data,
            B, S, H, dk, qi, qi_end);

        /* (4) SDPA bwd for [qi, qi_end). overwrites dQh rows, accumulates
              dKh/dVh. reads the attn_flat / L / P_save / dO_head rows
              just written above — all cache-hot. */
        ax_fused_attention_bwd_use_from_flat_qi_block(
            (const float *)Qh->storage->data,
            (const float *)Kh->storage->data,
            (const float *)Vh->storage->data,
            (const float *)attn_flat->storage->data,
            (const float *)dO_head->storage->data,
            (const float *)L_t->storage->data,
            P_save_ptr,
            (float *)dQh->storage->data,
            (float *)dKh->storage->data,
            (float *)dVh->storage->data,
            B, S, H, dk, qi, qi_end, scale, m->causal);

        /* (5) Phase C: stream dWq / dbq from dQh[qi..qi_end] while it's
              cache-hot. per-batch: deinterleave the H head-strips into a
              [bq, D] block, gemm_tn X[b, qi..qi_end]^T @ buf → dWq
              (accumulate). buf is reused across iterations. */
        if (dQh_batch_buf) {
            dQh_batch_view->shape[0] = bq;
            dQh_batch_view->storage->size_bytes =
                (size_t)(bq * D) * sizeof(float);
            const float *dQh_data = (const float *)dQh->storage->data;
            const float *x_data   = (const float *)x_flat_v.storage->data;
            size_t dk_bytes = (size_t)dk * sizeof(float);
            for (int64_t b = 0; b < B; b++) {
                /* head_deinterleave dQh[b*H+h, qi+s, :] → buf[s, h*dk:]. */
#ifdef _OPENMP
                #pragma omp parallel for collapse(2) schedule(static)
#endif
                for (int64_t s_rel = 0; s_rel < bq; s_rel++) {
                    for (int64_t h = 0; h < H; h++) {
                        const float *src = dQh_data
                            + (b * H + h) * S * dk + (qi + s_rel) * dk;
                        float *dst = dQh_batch_buf + s_rel * D + h * dk;
                        memcpy(dst, src, dk_bytes);
                    }
                }
                if (need_wq_grad) {
                    ax_tensor_t x_sub;
                    ax_storage_t x_sub_st;
                    const float *x_ptr = x_data + b * S * D + qi * D;
                    make_stack_view(&x_sub, &x_sub_st, x_ptr, bq, D);
                    ax_gemm_set_skip_init(true);
                    ax_status_t s = ax_compute_gemm_tn(&x_sub, dQh_batch_view,
                                                         m->Wq->grad);
                    ax_gemm_set_skip_init(false);
                    if (s != AX_OK) return s;
                }
                if (need_bq_grad) {
                    col_sum_acc(dQh_batch_buf, dbq_acc, bq, D);
                }
            }
            if (need_wq_grad) ax_storage_touch(m->Wq->grad->storage);
            if (need_bq_grad) ax_storage_touch(m->bq->grad->storage);
        }

        (void)bq;
    }

    /* (6) fwd y = attn_flat @ Wo + bo (full-rows — all attn_flat rows are
       filled by now). */
    if (ax_compute_gemm(attn_flat, m->Wo, &y_flat_v) != AX_OK) return AX_ERR_BACKEND;
    if (m->use_bias) {
        const float *bd = (const float *)m->bo->storage->data;
        float *od = (float *)y_flat_v.storage->data;
        int64_t de = D - (D % AX_VF32_WIDTH);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int64_t r = 0; r < rows; r++) {
            float *row = od + r * D;
            int64_t c = 0;
            for (; c < de; c += AX_VF32_WIDTH) {
                ax_vf32 v = ax_vf32_loadu(row + c);
                ax_vf32 b = ax_vf32_loadu(bd + c);
                ax_vf32_storeu(row + c, ax_vf32_add(v, b));
            }
            for (; c < D; c++) row[c] += bd[c];
        }
    }
    ax_storage_touch(y_out->storage);

    /* (6) dWo += attn_flat^T @ dout (full-rows). accumulates into existing
       grad — caller is expected to have zero_grad'd before the train step. */
    if (m->Wo->requires_grad) {
        float *dWo = ensure_grad_ptr(m->Wo);
        if (dWo) {
            ax_gemm_set_skip_init(true);
            ax_status_t s = ax_compute_gemm_tn(attn_flat, dout_flat, m->Wo->grad);
            ax_gemm_set_skip_init(false);
            if (s != AX_OK) return s;
            ax_storage_touch(m->Wo->grad->storage);
        }
    }

    /* (7) dbo += colsum(dout) */
    if (m->use_bias && m->bo->requires_grad) {
        float *dbo = ensure_grad_ptr(m->bo);
        if (dbo) {
            col_sum_acc((const float *)dout_flat->storage->data, dbo, rows, D);
            ax_storage_touch(m->bo->grad->storage);
        }
    }

    /* (9) Phase C: dWk / dWv / dbk / dbv from dKh, dVh (now finalised after
       the qi-block loop — the bwd primitive cross-qi accumulates into
       them, so they only become valid here).

       head_deinterleave dKh → dKV_flat[:, 0..D], dVh → dKV_flat[:, D..2D]
       into a single [rows, 2D] buffer, one gemm_tn: dWkv = x^T @ dKV_flat
       ([D, 2D]), split and accumulate into Wk/Wv grads. memory vs the
       Phase B dQKV[rows, 3D] path: saves the [rows, D] Q partition
       (~B*S*D floats; ~6 MB on B=2, S=1024, D=768). */
    bool need_wkv_grad  = m->Wk->requires_grad || m->Wv->requires_grad;
    bool need_bkv_grad  = m->use_bias && ((m->bk && m->bk->requires_grad) ||
                                           (m->bv && m->bv->requires_grad));
    if (need_wkv_grad || need_bkv_grad) {
        int64_t dKV_sh[] = {rows, 2 * D};
        ax_tensor_t *dKV_flat = ax_tensor_arena_create(arena, dKV_sh, 2, AX_FLOAT32);
        if (!dKV_flat) return AX_ERR_ALLOC;

        float *dKV_ptr = (float *)dKV_flat->storage->data;
        ax_attn_head_deinterleave_slot((const float *)dKh->storage->data,
                                         dKV_ptr, B, S, H, dk, 2 * D, 0);
        ax_attn_head_deinterleave_slot((const float *)dVh->storage->data,
                                         dKV_ptr, B, S, H, dk, 2 * D, D);

        if (need_wkv_grad) {
            int64_t dW_sh[] = {D, 2 * D};
            ax_tensor_t *dWkv = ax_tensor_arena_create(arena, dW_sh, 2, AX_FLOAT32);
            if (!dWkv) return AX_ERR_ALLOC;
            if (ax_compute_gemm_tn(&x_flat_v, dKV_flat, dWkv) != AX_OK)
                return AX_ERR_BACKEND;

            const float *dw = (const float *)dWkv->storage->data;
            float *dp_k = m->Wk->requires_grad ? ensure_grad_ptr(m->Wk) : NULL;
            float *dp_v = m->Wv->requires_grad ? ensure_grad_ptr(m->Wv) : NULL;

#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int64_t i = 0; i < D; i++) {
                if (dp_k) acc_f32(dw + i * 2 * D,     dp_k + i * D, D);
                if (dp_v) acc_f32(dw + i * 2 * D + D, dp_v + i * D, D);
            }
            if (dp_k) ax_storage_touch(m->Wk->grad->storage);
            if (dp_v) ax_storage_touch(m->Wv->grad->storage);
        }

        if (need_bkv_grad) {
            int64_t db_sh[] = {2 * D};
            ax_tensor_t *dbkv = ax_tensor_arena_zeros(arena, db_sh, 1, AX_FLOAT32);
            if (!dbkv) return AX_ERR_ALLOC;
            col_sum_acc(dKV_ptr, (float *)dbkv->storage->data, rows, 2 * D);
            const float *db = (const float *)dbkv->storage->data;
            if (m->bk && m->bk->requires_grad) {
                float *dp = ensure_grad_ptr(m->bk);
                if (dp) { acc_f32(db,     dp, D); ax_storage_touch(m->bk->grad->storage); }
            }
            if (m->bv && m->bv->requires_grad) {
                float *dp = ensure_grad_ptr(m->bv);
                if (dp) { acc_f32(db + D, dp, D); ax_storage_touch(m->bv->grad->storage); }
            }
        }
    }

    return AX_OK;
}
