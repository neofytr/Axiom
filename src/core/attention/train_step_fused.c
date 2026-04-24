/* attention/train_step_fused.c — F.4: fully-fused MHA train kernel.

   target: match TF's @tf.function(jit_compile=True) baseline by
   collapsing the entire forward + backward into one cache-resident
   pass. same external contract as ax_mha_train_step but the body is
   a per-(qi, kj) tile loop that streams every intermediate through
   L1 — no qkv / Qh / Kh / Vh / Oh / attn_flat / dout / d_attn_flat
   / dQh / dKh / dVh / dQKV / dWqkv arena tensors between stages.

   phased delivery (see docs/F4_FUSED_MHA_TRAIN.md):

     F.4.0: thin wrapper around ax_mha_train_step. shipped the public
       API contract + parity test scaffold.

     F.4.1: AX_SDPA_FUSED=1 default — deferred per commit 3b51031.

     F.4.2 (this commit): output-projection reorder. forward y uses
       Wo, backward dattn also uses Wo — fused by putting dattn_nt
       immediately after the backward section starts so Wo stays
       warmer in L3 across the fwd/bwd boundary. the three bwd ops
       (dattn_nt, dWo_tn, dbo_col_sum) are reordered so dout is
       loaded once and reused across all three instead of evicted
       between calls. body is a copy of ax_mha_train_step with the
       reorder applied; aggressive per-strip fusion with custom
       micro-kernels was evaluated and rejected — the custom AVX2
       kernels hit ~50-60 % of opt_gemm peak, losing more on flops
       than cache saves on this CPU. F.4.4 is where real strip
       fusion lands because it collapses SDPA fwd+bwd too.

     F.4.3: tile-fused input projection (per-tile X*Wqkv recompute,
       dropping the Qh/Kh/Vh full materialisation).

     F.4.4: full FA-2 fwd+bwd in one pass with all weight grad
       accumulators streaming per tile.

   each phase keeps the parity test green and is a separate commit. */

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

/* same as train_step.c — accumulate into dst instead of overwriting
   when the next gemm writes there. */
extern void ax_gemm_set_skip_init(bool v);

/* per-thread scratch arena, reset per call. mirror of train_step.c
   but separate TLS slot so the two entries don't alias. */
static __thread ax_arena_t *tl_train_fused_arena = NULL;

static ax_arena_t *get_scratch_arena(void)
{
    if (!tl_train_fused_arena) {
        tl_train_fused_arena = ax_arena_create(16 * 1024 * 1024);
        if (!tl_train_fused_arena) return NULL;
    } else {
        ax_arena_reset(tl_train_fused_arena);
    }
    return tl_train_fused_arena;
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

ax_status_t ax_mha_train_step_fused(ax_layer_t *layer,
                                     const ax_tensor_t *x,
                                     const ax_tensor_t *dout,
                                     ax_tensor_t *y_out)
{
    if (!layer || !x || !y_out) {
        ax_err_set(AX_ERR_NULL_ARG, "ax_mha_train_step_fused: NULL arg");
        return AX_ERR_NULL_ARG;
    }
    ax_mha_t *m = (ax_mha_t *)layer;
    if (x->ndim != 3 || y_out->ndim != 3) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "train_step_fused: x/y must be 3-D");
        return AX_ERR_SHAPE_MISMATCH;
    }
    int64_t B  = x->shape[0];
    int64_t S  = x->shape[1];
    int64_t D  = x->shape[2];
    if (D != m->d_model) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "train_step_fused: x last dim != d_model");
        return AX_ERR_SHAPE_MISMATCH;
    }
    if (y_out->shape[0] != B || y_out->shape[1] != S || y_out->shape[2] != D) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "train_step_fused: y_out shape mismatch");
        return AX_ERR_SHAPE_MISMATCH;
    }
    if (x->dtype != AX_FLOAT32 || y_out->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "train_step_fused: must be fp32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    if (dout) {
        if (dout->ndim != 3 ||
            dout->shape[0] != B || dout->shape[1] != S || dout->shape[2] != D ||
            dout->dtype != AX_FLOAT32) {
            ax_err_set(AX_ERR_SHAPE_MISMATCH, "train_step_fused: dout shape mismatch");
            return AX_ERR_SHAPE_MISMATCH;
        }
    }

    int64_t H    = m->n_heads;
    int64_t dk   = m->d_k;
    int64_t rows = B * S;
    int64_t bh   = B * H;

    ax_arena_t *arena = get_scratch_arena();
    if (!arena) {
        ax_err_set(AX_ERR_ALLOC, "train_step_fused: scratch arena alloc failed");
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

    /* forward: qkv projection + head-split. F.3.a fused path writes
       directly to head-interleaved Qh/Kh/Vh, eliminating the qkv
       intermediate. fall back to gemm + split if backend lacks slot. */
    int64_t qkv_sh[]  = {rows, 3 * D};
    int64_t head_sh[] = {bh, S, dk};
    int64_t L_sh[]    = {bh, S};
    int64_t flat_sh[] = {rows, D};
    (void)qkv_sh; /* only used in fallback */

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

    /* SDPA forward + head-merge to attn_flat. F.3.e fused entry skips
       the Oh tensor entirely, writing per-head output directly into
       attn_flat. fall back to fwd_save + head_deinterleave when AX_NO_F3E=1. */
    ax_tensor_t *L_t = ax_tensor_arena_create(arena, L_sh, 2, AX_FLOAT32);
    if (!L_t) return AX_ERR_ALLOC;

    ax_tensor_t *P_save_t = NULL;
    int64_t p_save_bytes = bh * S * S * (int64_t)sizeof(float);
    bool save_p = (p_save_bytes <= (int64_t)8 * 1024 * 1024);
    if (save_p && S <= 128 && dk <= 64) save_p = false;
    const char *env = getenv("AX_MHA_SAVE_P");
    if (env) save_p = (env[0] == '1');
    if (save_p) {
        int64_t P_sh[] = {bh, S, S};
        P_save_t = ax_tensor_arena_create(arena, P_sh, 3, AX_FLOAT32);
    }
    float scale = 1.0f / sqrtf((float)dk);
    float *P_save_ptr = P_save_t ? (float *)P_save_t->storage->data : NULL;

    static int use_f3e_resolved = 0;
    static int use_f3e = 1;
    if (!use_f3e_resolved) {
        const char *e = getenv("AX_NO_F3E");
        use_f3e = (e && e[0] == '1') ? 0 : 1;
        use_f3e_resolved = 1;
    }

    ax_tensor_t *Oh = NULL;
    ax_tensor_t *attn_flat = ax_tensor_arena_create(arena, flat_sh, 2, AX_FLOAT32);
    if (!attn_flat) return AX_ERR_ALLOC;

    if (use_f3e) {
        ax_fused_attention_fwd_save_to_flat(
            (const float *)Qh->storage->data,
            (const float *)Kh->storage->data,
            (const float *)Vh->storage->data,
            (float *)attn_flat->storage->data,
            (float *)L_t->storage->data,
            P_save_ptr,
            B, S, H, dk, scale, m->causal);
    } else {
        Oh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
        if (!Oh) return AX_ERR_ALLOC;
        ax_fused_attention_fwd_save(
            (const float *)Qh->storage->data,
            (const float *)Kh->storage->data,
            (const float *)Vh->storage->data,
            (float *)Oh->storage->data,
            (float *)L_t->storage->data,
            P_save_ptr,
            bh, S, dk, scale, m->causal);
        ax_attn_head_deinterleave(
            (const float *)Oh->storage->data,
            (float *)attn_flat->storage->data, B, S, H, dk);
    }

    /* forward y = attn_flat @ Wo + bo. reads Wo. */
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

    /* backward starts. dout view — caller-supplied or default ones. */
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

    /* F.4.2 output-projection reorder.
       current train_step.c ordering: dWo_tn, dbo_col_sum, dattn_nt.
       the reorder below is: dattn_nt → dWo_tn → dbo_col_sum.
       why:
         1. dattn_nt reads Wo. the forward y gemm also read Wo and
            y_out's bias add ran in between. Wo may still be in L3
            for dattn_nt if the deinterleave + dout setup didn't
            evict it.
         2. dWo_tn reads dout. dattn_nt also reads dout just before.
            dout stays warm from dattn_nt's read into dWo_tn's.
         3. dbo_col_sum reads dout once more; warm from dWo_tn. */

    /* dattn + head_interleave: F.3.d fused path streams dout @ Wo^T
       directly into dO_head, no d_attn_flat intermediate. */
    ax_tensor_t *dO_head = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    if (!dO_head) return AX_ERR_ALLOC;

    if (ax_compute_has_dattn_head_gemm_nt()) {
        if (ax_compute_dattn_head_gemm_nt(dout_flat, m->Wo, B, S, H, dk, dO_head) != AX_OK)
            return AX_ERR_BACKEND;
    } else {
        ax_tensor_t *d_attn_flat = ax_tensor_arena_create(arena, flat_sh, 2, AX_FLOAT32);
        if (!d_attn_flat) return AX_ERR_ALLOC;
        if (ax_compute_gemm_nt(dout_flat, m->Wo, d_attn_flat) != AX_OK) return AX_ERR_BACKEND;
        ax_attn_head_interleave(
            (const float *)d_attn_flat->storage->data,
            (float *)dO_head->storage->data, B, S, H, dk);
    }

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

    if (m->use_bias && m->bo->requires_grad) {
        float *dbo = ensure_grad_ptr(m->bo);
        if (dbo) {
            col_sum_acc((const float *)dout_flat->storage->data, dbo, rows, D);
            ax_storage_touch(m->bo->grad->storage);
        }
    }

    ax_tensor_t *dQh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    ax_tensor_t *dKh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    ax_tensor_t *dVh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    if (!dQh || !dKh || !dVh) return AX_ERR_ALLOC;

    if (Oh) {
        ax_fused_attention_bwd_use(
            (const float *)Qh->storage->data,
            (const float *)Kh->storage->data,
            (const float *)Vh->storage->data,
            (const float *)Oh->storage->data,
            (const float *)dO_head->storage->data,
            (const float *)L_t->storage->data,
            P_save_ptr,
            (float *)dQh->storage->data,
            (float *)dKh->storage->data,
            (float *)dVh->storage->data,
            bh, S, dk, scale, m->causal);
    } else {
        /* F.3.e: Oh never materialised; read O strided from attn_flat */
        ax_fused_attention_bwd_use_from_flat(
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
            B, S, H, dk, scale, m->causal);
    }

    /* step 3 — head-deinterleave + merge → [rows, 3D] dQKV */
    ax_tensor_t *dQKV = ax_tensor_arena_create(arena, qkv_sh, 2, AX_FLOAT32);
    if (!dQKV) return AX_ERR_ALLOC;
    ax_attn_head_deinterleave_qkv_merge(
        (const float *)dQh->storage->data,
        (const float *)dKh->storage->data,
        (const float *)dVh->storage->data,
        (float *)dQKV->storage->data,
        B, S, H, dk, D);

    /* step 4 — fused weight grad: dWqkv = x_flat^T @ dQKV. the 3 ACC_PARAM
       split-into-Wq/Wk/Wv passes are merged into one parallel region so
       we pay the omp spawn/join cost once instead of three times — saves
       ~20-30 us per train_step call at NT=16. the dw[i*3D:(i+1)*3D] row
       stays in L1 across the three acc_f32 calls inside one iteration,
       eliminating the row's second and third load on most hardware. */
    bool any_wqkv_grad = m->Wq->requires_grad || m->Wk->requires_grad ||
                         m->Wv->requires_grad;
    if (any_wqkv_grad) {
        int64_t dW_sh[] = {D, 3 * D};
        ax_tensor_t *dWqkv = ax_tensor_arena_create(arena, dW_sh, 2, AX_FLOAT32);
        if (!dWqkv) return AX_ERR_ALLOC;
        if (ax_compute_gemm_tn(&x_flat_v, dQKV, dWqkv) != AX_OK) return AX_ERR_BACKEND;

        const float *dw = (const float *)dWqkv->storage->data;
        float *dp_q = m->Wq->requires_grad ? ensure_grad_ptr(m->Wq) : NULL;
        float *dp_k = m->Wk->requires_grad ? ensure_grad_ptr(m->Wk) : NULL;
        float *dp_v = m->Wv->requires_grad ? ensure_grad_ptr(m->Wv) : NULL;

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int64_t i = 0; i < D; i++) {
            if (dp_q) acc_f32(dw + i * 3 * D,         dp_q + i * D, D);
            if (dp_k) acc_f32(dw + i * 3 * D + D,     dp_k + i * D, D);
            if (dp_v) acc_f32(dw + i * 3 * D + 2 * D, dp_v + i * D, D);
        }

        if (dp_q) ax_storage_touch(m->Wq->grad->storage);
        if (dp_k) ax_storage_touch(m->Wk->grad->storage);
        if (dp_v) ax_storage_touch(m->Wv->grad->storage);
    }

    /* step 5 — bias grads */
    if (m->use_bias && (m->bq->requires_grad || m->bk->requires_grad ||
                         m->bv->requires_grad)) {
        int64_t db_sh[] = {3 * D};
        ax_tensor_t *dbqkv = ax_tensor_arena_zeros(arena, db_sh, 1, AX_FLOAT32);
        if (!dbqkv) return AX_ERR_ALLOC;
        col_sum_acc((const float *)dQKV->storage->data,
                     (float *)dbqkv->storage->data, rows, 3 * D);
        const float *db = (const float *)dbqkv->storage->data;
        #define ACC_BIAS(W, col_off) do {                                     \
            if ((W)->requires_grad) {                                         \
                float *dp = ensure_grad_ptr(W);                               \
                if (dp) {                                                     \
                    acc_f32(db + (col_off), dp, D);                           \
                    ax_storage_touch((W)->grad->storage);                     \
                }                                                             \
            }                                                                 \
        } while (0)
        ACC_BIAS(m->bq, 0);
        ACC_BIAS(m->bk, D);
        ACC_BIAS(m->bv, 2 * D);
        #undef ACC_BIAS
    }

    return AX_OK;
}
