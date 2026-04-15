/* attention.c — multi-head attention layer.

   fused QKV projection: one [B*S, D] @ [D, 3D] GEMM rebuilds Q, K, V
   together. the [D, 3D] panel is cached between calls and rebuilt only
   when one of Wq/Wk/Wv mutates (checked via storage generation counter).

   forward flow:
     x:[B,S,D]
       -> QKV = x @ Wqkv + bqkv                           [B*S, 3D]
       -> split + head-interleave to [BH, S, dk] each     Q,K,V
       -> SDPA(Q,K,V)                                      O_attn:[BH,S,dk], L:[BH,S]
       -> merge heads back to [B*S, D]                     attn_flat
       -> out = attn_flat @ Wo + bo                        [B,S,D]

   backward is a single fused grad_fn that reverses each step. parameters
   (Wq/Wk/Wv/Wo/bq/bk/bv/bo) are accessed via the saved layer pointer,
   the input is saved as the only autograd input (so dInput is routed to
   the user tensor). all intermediates (x_flat copy, Q/K/V, O_attn, L)
   live in the forward arena so they're released on graph cleanup. */

#include "axiom/attention.h"
#include "axiom/ops.h"
#include "axiom/autograd.h"
#include "axiom/compute.h"
#include "axiom/init.h"
#include "axiom/error.h"
#include "../compute/backends/simd_defs.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ================================================================
   helpers
   ================================================================ */

/* transpose a [B, S, H, dk] row-major buffer into [B, H, S, dk] (= [BH,S,dk]).
   both buffers must be distinct. */
static void head_interleave(const float *src, float *dst,
                            int64_t B, int64_t S, int64_t H, int64_t dk)
{
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t s = 0; s < S; s++) {
                const float *src_row = src + (((b * S) + s) * H + h) * dk;
                float *dst_row       = dst + (((b * H) + h) * S + s) * dk;
                memcpy(dst_row, src_row, (size_t)dk * sizeof(float));
            }
        }
    }
}

/* inverse of head_interleave: [B, H, S, dk] -> [B, S, H, dk] (= [B*S, D]). */
static void head_deinterleave(const float *src, float *dst,
                              int64_t B, int64_t S, int64_t H, int64_t dk)
{
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int64_t b = 0; b < B; b++) {
        for (int64_t s = 0; s < S; s++) {
            for (int64_t h = 0; h < H; h++) {
                const float *src_row = src + (((b * H) + h) * S + s) * dk;
                float *dst_row       = dst + (((b * S) + s) * H + h) * dk;
                memcpy(dst_row, src_row, (size_t)dk * sizeof(float));
            }
        }
    }
}

/* head-deinterleave into a specific column offset of a wider [rows, dst_cols]
   buffer. used to pack dQh, dKh, dVh into one [rows, 3D] dQKV tensor so the
   weight- and input-gradient GEMMs can be fused into single large calls. */
static void head_deinterleave_slot(const float *src, float *dst,
                                   int64_t B, int64_t S, int64_t H, int64_t dk,
                                   int64_t dst_cols, int64_t col_off)
{
    int64_t D = H * dk;
    (void)D;
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int64_t b = 0; b < B; b++) {
        for (int64_t s = 0; s < S; s++) {
            for (int64_t h = 0; h < H; h++) {
                const float *src_row = src + (((b * H) + h) * S + s) * dk;
                float *dst_row       = dst + ((b * S) + s) * dst_cols + col_off + h * dk;
                memcpy(dst_row, src_row, (size_t)dk * sizeof(float));
            }
        }
    }
}

/* head-interleave FROM a specific column slot of a wider [rows, src_cols]
   buffer. fuses the extract+interleave steps so the QKV fused-projection
   output can feed SDPA in a single pass instead of going through the
   intermediate Qf/Kf/Vf [rows, D] buffers. */
static void head_interleave_from_slot(const float *src, float *dst,
                                      int64_t B, int64_t S, int64_t H, int64_t dk,
                                      int64_t src_cols, int64_t col_off)
{
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t s = 0; s < S; s++) {
                const float *src_row = src + ((b * S) + s) * src_cols + col_off + h * dk;
                float *dst_row       = dst + (((b * H) + h) * S + s) * dk;
                memcpy(dst_row, src_row, (size_t)dk * sizeof(float));
            }
        }
    }
}

/* fused variant: read one [rows, 3D] qkv buffer, split into three
   [BH, S, dk] tensors in one parallel pass. replaces three back-to-back
   head_interleave_from_slot calls, saving 2 omp fork/join barriers and
   giving each thread better cache reuse since the same src row is read
   for Qh, Kh, and Vh. */
static void head_interleave_qkv_split(const float *src,
                                       float *dstQ, float *dstK, float *dstV,
                                       int64_t B, int64_t S, int64_t H, int64_t dk,
                                       int64_t D)
{
    int64_t src_cols = 3 * D;
    size_t dk_bytes = (size_t)dk * sizeof(float);
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t s = 0; s < S; s++) {
                const float *src_row = src + ((b * S) + s) * src_cols + h * dk;
                int64_t dst_off = (((b * H) + h) * S + s) * dk;
                memcpy(dstQ + dst_off, src_row,           dk_bytes);
                memcpy(dstK + dst_off, src_row + D,       dk_bytes);
                memcpy(dstV + dst_off, src_row + 2 * D,   dk_bytes);
            }
        }
    }
}

/* rebuild Wqkv_cache [D, 3D] and bqkv_cache [3D] from Wq/Wk/Wv/bq/bk/bv
   if any of the source generations have changed since last call. */
static void refresh_fused_qkv(ax_mha_t *m)
{
    uint64_t gq = m->Wq->storage->generation;
    uint64_t gk = m->Wk->storage->generation;
    uint64_t gv = m->Wv->storage->generation;

    int64_t D = m->d_model;

    /* bump weight panel if anything changed OR cache doesn't exist */
    bool w_stale = !m->Wqkv_cache || gq != m->Wq_gen || gk != m->Wk_gen || gv != m->Wv_gen;
    if (w_stale) {
        if (!m->Wqkv_cache) {
            int64_t sh[] = {D, 3 * D};
            m->Wqkv_cache = ax_tensor_create(sh, 2, AX_FLOAT32);
        }
        float *dst = (float *)m->Wqkv_cache->storage->data;
        const float *wq = (const float *)m->Wq->storage->data;
        const float *wk = (const float *)m->Wk->storage->data;
        const float *wv = (const float *)m->Wv->storage->data;
        /* [D, 3D] row-major: row i -> [wq[i,:] | wk[i,:] | wv[i,:]] */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int64_t i = 0; i < D; i++) {
            memcpy(dst + i * 3 * D,            wq + i * D, (size_t)D * sizeof(float));
            memcpy(dst + i * 3 * D + D,        wk + i * D, (size_t)D * sizeof(float));
            memcpy(dst + i * 3 * D + 2 * D,    wv + i * D, (size_t)D * sizeof(float));
        }
        ax_storage_touch(m->Wqkv_cache->storage);
        m->Wq_gen = gq; m->Wk_gen = gk; m->Wv_gen = gv;
    }

    /* biases: rebuild unconditionally when used — cheap (3D floats) */
    if (m->use_bias) {
        if (!m->bqkv_cache) {
            int64_t sh[] = {3 * D};
            m->bqkv_cache = ax_tensor_create(sh, 1, AX_FLOAT32);
        }
        float *dst = (float *)m->bqkv_cache->storage->data;
        memcpy(dst,         m->bq->storage->data, (size_t)D * sizeof(float));
        memcpy(dst + D,     m->bk->storage->data, (size_t)D * sizeof(float));
        memcpy(dst + 2 * D, m->bv->storage->data, (size_t)D * sizeof(float));
        ax_storage_touch(m->bqkv_cache->storage);
    }
}

/* ================================================================
   backward context
   ================================================================ */

/* forward declaration — defined below forward() */
static void mha_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out);

typedef struct {
    ax_mha_t *layer;
    int64_t B, S, D, H, dk;

    /* tensors saved for backward — all arena-allocated (forward arena).
       we keep raw float pointers too so the backward path can run without
       re-indexing. */
    ax_tensor_t *x_flat;       /* [B*S, D]     input after reshape    */
    ax_tensor_t *Q_head;       /* [BH, S, dk]  Q after head_interleave */
    ax_tensor_t *K_head;       /* [BH, S, dk]                          */
    ax_tensor_t *V_head;       /* [BH, S, dk]                          */
    ax_tensor_t *O_head;       /* [BH, S, dk]  SDPA output             */
    ax_tensor_t *attn_flat;    /* [B*S, D]     after head_deinterleave */
    ax_tensor_t *L_tensor;     /* [BH, S]      logsumexp per row       */
} mha_ctx_t;

static void mha_ctx_cleanup(void *p) { free(p); }

/* ================================================================
   forward
   ================================================================ */

static ax_tensor_t *mha_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_mha_t *m = (ax_mha_t *)self;
    if (!input || input->ndim != 3) return NULL;

    int64_t B  = input->shape[0];
    int64_t S  = input->shape[1];
    int64_t D  = input->shape[2];
    if (D != m->d_model) return NULL;

    int64_t H  = m->n_heads;
    int64_t dk = m->d_k;
    int64_t rows = B * S;

    /* contiguous input as a flat [B*S, D] view */
    ax_tensor_t *inp = ax_ensure_contiguous(input);
    if (!inp) return NULL;

    /* ensure fused weight+bias panels are up to date */
    refresh_fused_qkv(m);

    bool record = ax_grad_enabled() && self->training && input->requires_grad;
    /* always record if any of our params require grad too */
    if (self->training && ax_grad_enabled()) {
        if (m->Wq->requires_grad || m->Wk->requires_grad || m->Wv->requires_grad ||
            m->Wo->requires_grad) record = true;
        if (m->use_bias && (m->bq->requires_grad || m->bk->requires_grad ||
                            m->bv->requires_grad || m->bo->requires_grad)) record = true;
    }

    ax_arena_t *fa = record ? ax_forward_arena() : NULL;

    /* helper: pick arena for training (so tensor survives forward), else heap */
    #define ALLOC_SHAPE(t, sh, nd) do {                                   \
        if (record) (t) = ax_tensor_arena_create(fa, (sh), (nd), AX_FLOAT32); \
        else        (t) = ax_tensor_create((sh), (nd), AX_FLOAT32);           \
    } while (0)
    #define ALLOC_ZEROS(t, sh, nd) do {                                   \
        if (record) (t) = ax_tensor_arena_zeros(fa, (sh), (nd), AX_FLOAT32);\
        else        (t) = ax_tensor_zeros((sh), (nd), AX_FLOAT32);            \
    } while (0)

    /* ---- build x_flat [rows, D] view of input ---- */
    int64_t flat_sh[] = {rows, D};
    ax_tensor_t *x_flat = NULL;
    ALLOC_SHAPE(x_flat, flat_sh, 2);
    if (!x_flat) goto fail;
    memcpy(x_flat->storage->data, inp->storage->data,
           (size_t)rows * (size_t)D * sizeof(float));

    /* ---- fused QKV: [rows, D] @ [D, 3D] -> [rows, 3D] ---- */
    int64_t qkv_sh[] = {rows, 3 * D};
    ax_tensor_t *qkv = NULL;
    ALLOC_SHAPE(qkv, qkv_sh, 2);
    if (!qkv) goto fail;
    if (ax_compute_gemm(x_flat, m->Wqkv_cache, qkv) != AX_OK) goto fail;
    if (m->use_bias) {
        float *qd = (float *)qkv->storage->data;
        const float *bd = (const float *)m->bqkv_cache->storage->data;
        int64_t cols = 3 * D;
        int64_t ce = cols - (cols % AX_VF32_WIDTH);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if (rows * cols >= ax_par_threshold_elems)
#endif
        for (int64_t r = 0; r < rows; r++) {
            float *row = qd + r * cols;
            int64_t c = 0;
            for (; c < ce; c += AX_VF32_WIDTH) {
                ax_vf32 v = ax_vf32_loadu(row + c);
                ax_vf32 b = ax_vf32_loadu(bd + c);
                ax_vf32_storeu(row + c, ax_vf32_add(v, b));
            }
            for (; c < cols; c++) row[c] += bd[c];
        }
    }

    /* ---- head-interleave directly from the fused QKV buffer into
       Qh/Kh/Vh, skipping intermediate [rows, D] copies. one memory pass
       per tensor (vs previous two). ---- */
    int64_t head_sh[] = {B * H, S, dk};
    ax_tensor_t *Qh = NULL, *Kh = NULL, *Vh = NULL;
    ALLOC_SHAPE(Qh, head_sh, 3);
    ALLOC_SHAPE(Kh, head_sh, 3);
    ALLOC_SHAPE(Vh, head_sh, 3);
    if (!Qh || !Kh || !Vh) goto fail;

    head_interleave_qkv_split((const float *)qkv->storage->data,
                               (float *)Qh->storage->data,
                               (float *)Kh->storage->data,
                               (float *)Vh->storage->data,
                               B, S, H, dk, D);

    /* qkv no longer needed (non-recorded path); arena reclaims it otherwise */
    if (!record) ax_tensor_destroy(qkv);

    /* ---- SDPA ---- */
    ax_tensor_t *Oh = NULL;
    ALLOC_SHAPE(Oh, head_sh, 3);
    if (!Oh) goto fail;

    int64_t L_sh[] = {B * H, S};
    ax_tensor_t *L_t = NULL;
    if (record) {
        ALLOC_SHAPE(L_t, L_sh, 2);
        if (!L_t) goto fail;
    }

    float scale = 1.0f / sqrtf((float)dk);
    if (m->causal) {
        ax_fused_attention_fwd_causal(
            (const float *)Qh->storage->data,
            (const float *)Kh->storage->data,
            (const float *)Vh->storage->data,
            (float *)Oh->storage->data,
            L_t ? (float *)L_t->storage->data : NULL,
            B * H, S, dk, scale);
    } else {
        ax_fused_attention_fwd(
            (const float *)Qh->storage->data,
            (const float *)Kh->storage->data,
            (const float *)Vh->storage->data,
            (float *)Oh->storage->data,
            L_t ? (float *)L_t->storage->data : NULL,
            B * H, S, dk, scale);
    }

    /* ---- merge heads back to [rows, D] ---- */
    int64_t attn_sh[] = {rows, D};
    ax_tensor_t *attn_flat = NULL;
    ALLOC_SHAPE(attn_flat, attn_sh, 2);
    if (!attn_flat) goto fail;
    head_deinterleave((const float *)Oh->storage->data,
                      (float *)attn_flat->storage->data, B, S, H, dk);

    /* ---- output projection: attn_flat @ Wo + bo ---- */
    int64_t out_sh[] = {B, S, D};
    ax_tensor_t *out = ax_tensor_create(out_sh, 3, AX_FLOAT32);
    if (!out) goto fail;

    /* use a [rows, D] view of the output for GEMM */
    int64_t out_flat_sh[] = {rows, D};
    ax_tensor_t *out_flat_view = ax_tensor_reshape(out, out_flat_sh, 2);
    if (!out_flat_view) { ax_tensor_destroy(out); goto fail; }

    if (ax_compute_gemm(attn_flat, m->Wo, out_flat_view) != AX_OK) {
        ax_tensor_destroy(out_flat_view);
        ax_tensor_destroy(out);
        goto fail;
    }
    if (m->use_bias) {
        float *od = (float *)out_flat_view->storage->data;
        const float *bd = (const float *)m->bo->storage->data;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int64_t r = 0; r < rows; r++) {
            float *row = od + r * D;
            for (int64_t c = 0; c < D; c++) row[c] += bd[c];
        }
    }
    ax_storage_touch(out->storage);
    ax_tensor_destroy(out_flat_view); /* view; storage kept alive by `out` */

    /* ---- hook up backward ---- */
    if (record) {
        mha_ctx_t *ctx = (mha_ctx_t *)calloc(1, sizeof(mha_ctx_t));
        if (!ctx) {
            /* gracefully degrade to inference output */
            if (inp != input) ax_tensor_destroy(inp);
            return out;
        }
        ctx->layer = m;
        ctx->B = B; ctx->S = S; ctx->D = D; ctx->H = H; ctx->dk = dk;
        ctx->x_flat    = x_flat;
        ctx->Q_head    = Qh;
        ctx->K_head    = Kh;
        ctx->V_head    = Vh;
        ctx->O_head    = Oh;
        ctx->attn_flat = attn_flat;
        ctx->L_tensor  = L_t;

        ax_grad_fn_t *gf = ax_grad_fn_create(mha_backward);
        gf->inputs[0] = input;
        gf->n_inputs = 1;
        gf->ctx = ctx;
        gf->ctx_cleanup = mha_ctx_cleanup;
        out->requires_grad = true;
        out->grad_fn = gf;
    }

    if (inp != input) ax_tensor_destroy(inp);
    return out;

fail:
    if (inp != input) ax_tensor_destroy(inp);
    return NULL;

    #undef ALLOC_SHAPE
    #undef ALLOC_ZEROS
}

/* ================================================================
   backward
   ================================================================ */

/* allocate a param grad lazily (same pattern as other layers) */
static float *param_grad_ptr(ax_tensor_t *p)
{
    if (!p || !p->requires_grad) return NULL;
    if (!p->grad) {
        p->grad = ax_tensor_zeros(p->shape, p->ndim, p->dtype);
        if (!p->grad) return NULL;
    }
    return (float *)p->grad->storage->data;
}

/* SIMD-width row-vector sum reduction: out[c] += sum_r M[r, c] for c<cols.
   one pass over M, writes directly into the (possibly already populated) bias
   grad. parallelized over column tiles to stay within a single memory pass. */
static void sum_rows_acc(const float *M, float *out,
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

/* accumulate src into dst (SIMD). used where gemm_tn/nt returns into a
   fresh buffer and we want to add that into an existing grad. much faster
   than the previous scalar for-loop. */
static void accumulate_f32(const float *src, float *dst, int64_t n)
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

static void mha_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    mha_ctx_t *ctx = (mha_ctx_t *)self->ctx;
    ax_mha_t *m = ctx->layer;
    ax_tensor_t *input = self->inputs[0];

    int64_t B  = ctx->B;
    int64_t S  = ctx->S;
    int64_t D  = ctx->D;
    int64_t H  = ctx->H;
    int64_t dk = ctx->dk;
    int64_t rows = B * S;

    /* contiguous grad_out flattened to [rows, D] */
    ax_arena_t *arena = ax_backward_arena();
    int64_t flat_sh[] = {rows, D};
    ax_tensor_t *dOut_flat = ax_tensor_arena_create(arena, flat_sh, 2, AX_FLOAT32);
    if (!dOut_flat) return;
    memcpy(dOut_flat->storage->data, grad_out->storage->data,
           (size_t)rows * (size_t)D * sizeof(float));

    /* ================================================================
       step 1 — output projection backward
       ================================================================ */
    /* grad_Wo += attn_flat^T @ dOut_flat   (single [D, rows] @ [rows, D] gemm) */
    if (m->Wo->requires_grad) {
        int64_t wo_sh[] = {D, D};
        ax_tensor_t *scratch = ax_tensor_arena_create(arena, wo_sh, 2, AX_FLOAT32);
        if (!scratch) return;
        float *dWo = param_grad_ptr(m->Wo);
        if (dWo) {
            ax_compute_gemm_tn(ctx->attn_flat, dOut_flat, scratch);
            accumulate_f32((const float *)scratch->storage->data, dWo, D * D);
            ax_storage_touch(m->Wo->grad->storage);
        }
    }
    /* grad_bo += col-sum(dOut_flat) */
    if (m->use_bias && m->bo->requires_grad) {
        float *dbo = param_grad_ptr(m->bo);
        if (dbo) {
            sum_rows_acc((const float *)dOut_flat->storage->data, dbo, rows, D);
            ax_storage_touch(m->bo->grad->storage);
        }
    }
    /* d_attn_flat = dOut_flat @ Wo^T */
    ax_tensor_t *d_attn_flat = ax_tensor_arena_create(arena, flat_sh, 2, AX_FLOAT32);
    if (!d_attn_flat) return;
    if (ax_compute_gemm_nt(dOut_flat, m->Wo, d_attn_flat) != AX_OK) return;

    /* ================================================================
       step 2 — re-interleave heads + SDPA backward
       ================================================================ */
    int64_t head_sh[] = {B * H, S, dk};
    ax_tensor_t *dO_head = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    if (!dO_head) return;
    head_interleave((const float *)d_attn_flat->storage->data,
                    (float *)dO_head->storage->data, B, S, H, dk);

    ax_tensor_t *dQh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    ax_tensor_t *dKh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    ax_tensor_t *dVh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    if (!dQh || !dKh || !dVh) return;

    float scale = 1.0f / sqrtf((float)dk);
    if (m->causal) {
        ax_fused_attention_bwd_causal(
            (const float *)ctx->Q_head->storage->data,
            (const float *)ctx->K_head->storage->data,
            (const float *)ctx->V_head->storage->data,
            (const float *)ctx->O_head->storage->data,
            (const float *)dO_head->storage->data,
            (const float *)ctx->L_tensor->storage->data,
            (float *)dQh->storage->data,
            (float *)dKh->storage->data,
            (float *)dVh->storage->data,
            B * H, S, dk, scale);
    } else {
        ax_fused_attention_bwd(
            (const float *)ctx->Q_head->storage->data,
            (const float *)ctx->K_head->storage->data,
            (const float *)ctx->V_head->storage->data,
            (const float *)ctx->O_head->storage->data,
            (const float *)dO_head->storage->data,
            (const float *)ctx->L_tensor->storage->data,
            (float *)dQh->storage->data,
            (float *)dKh->storage->data,
            (float *)dVh->storage->data,
            B * H, S, dk, scale);
    }

    /* ================================================================
       step 3 — FUSED de-interleave into a single dQKV [rows, 3D] buffer,
                so the following gradient GEMMs can each be one big
                call instead of three separate D-wide ones.
       ================================================================ */
    int64_t qkv_sh[] = {rows, 3 * D};
    ax_tensor_t *dQKV = ax_tensor_arena_create(arena, qkv_sh, 2, AX_FLOAT32);
    if (!dQKV) return;
    head_deinterleave_slot((const float *)dQh->storage->data,
                           (float *)dQKV->storage->data, B, S, H, dk, 3 * D, 0);
    head_deinterleave_slot((const float *)dKh->storage->data,
                           (float *)dQKV->storage->data, B, S, H, dk, 3 * D, D);
    head_deinterleave_slot((const float *)dVh->storage->data,
                           (float *)dQKV->storage->data, B, S, H, dk, 3 * D, 2 * D);

    /* ================================================================
       step 4 — FUSED weight grad: dWqkv = x_flat^T @ dQKV → [D, 3D].
                one big gemm_tn covers all three of Wq/Wk/Wv, then we
                split the columns into their respective grad tensors
                with a SIMD accumulate. 3x fewer gemm calls, better
                cache reuse on x_flat (single pass over it vs three).
       ================================================================ */
    bool any_wqkv_grad = m->Wq->requires_grad || m->Wk->requires_grad || m->Wv->requires_grad;
    if (any_wqkv_grad) {
        int64_t dW_sh[] = {D, 3 * D};
        ax_tensor_t *dWqkv = ax_tensor_arena_create(arena, dW_sh, 2, AX_FLOAT32);
        if (!dWqkv) return;
        if (ax_compute_gemm_tn(ctx->x_flat, dQKV, dWqkv) != AX_OK) return;

        const float *dw_data = (const float *)dWqkv->storage->data;
        /* dWqkv row i: [dWq[i,:] | dWk[i,:] | dWv[i,:]], each D wide.
           we iterate rows and accumulate the three column slices into
           the three param-grad tensors. */
        #define ACC_PARAM(W, col_off) do {                                     \
            if ((W)->requires_grad) {                                          \
                float *dp = param_grad_ptr(W);                                 \
                if (dp) {                                                      \
                    _Pragma("omp parallel for schedule(static)")               \
                    for (int64_t i = 0; i < D; i++)                            \
                        accumulate_f32(dw_data + i * 3 * D + (col_off),        \
                                       dp + i * D, D);                         \
                    ax_storage_touch((W)->grad->storage);                      \
                }                                                              \
            }                                                                  \
        } while (0)
        ACC_PARAM(m->Wq, 0);
        ACC_PARAM(m->Wk, D);
        ACC_PARAM(m->Wv, 2 * D);
        #undef ACC_PARAM
    }

    /* ================================================================
       step 5 — FUSED bias grads: dbqkv = col-sum(dQKV) → [3D], then
                split into bq/bk/bv. one reduction pass vs three.
       ================================================================ */
    if (m->use_bias) {
        bool any_b = m->bq->requires_grad || m->bk->requires_grad || m->bv->requires_grad;
        if (any_b) {
            float *dbqkv = (float *)ax_tensor_arena_zeros(arena, qkv_sh + 0, 1, AX_FLOAT32) ? NULL : NULL;
            /* simpler: reduce into stack-ish arena tensor of shape [3D] */
            int64_t b_sh[] = {3 * D};
            ax_tensor_t *dbqkv_t = ax_tensor_arena_zeros(arena, b_sh, 1, AX_FLOAT32);
            if (!dbqkv_t) return;
            dbqkv = (float *)dbqkv_t->storage->data;
            sum_rows_acc((const float *)dQKV->storage->data, dbqkv, rows, 3 * D);
            #define ACC_BIAS(b, off) do {                                     \
                if ((b)->requires_grad) {                                     \
                    float *dp = param_grad_ptr(b);                            \
                    if (dp) {                                                 \
                        accumulate_f32(dbqkv + (off), dp, D);                 \
                        ax_storage_touch((b)->grad->storage);                 \
                    }                                                         \
                }                                                             \
            } while (0)
            ACC_BIAS(m->bq, 0);
            ACC_BIAS(m->bk, D);
            ACC_BIAS(m->bv, 2 * D);
            #undef ACC_BIAS
        }
    }

    /* ================================================================
       step 6 — FUSED input grad: dX = dQKV @ Wqkv^T → [rows, D], single
                gemm_nt covering the three chained projections in one
                shot. accumulated into input->grad via SIMD axpy.
       ================================================================ */
    if (input->requires_grad) {
        if (!input->grad) {
            input->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
            if (!input->grad) return;
        }
        ax_tensor_t *dX = ax_tensor_arena_create(arena, flat_sh, 2, AX_FLOAT32);
        if (!dX) return;
        /* dQKV is [rows, 3D]; Wqkv_cache is [D, 3D]; dX = dQKV @ Wqkv^T */
        if (ax_compute_gemm_nt(dQKV, m->Wqkv_cache, dX) != AX_OK) return;

        float *gx = (float *)input->grad->storage->data;
        accumulate_f32((const float *)dX->storage->data, gx, rows * D);
        ax_storage_touch(input->grad->storage);
    }

    /* ctx is freed by ctx_cleanup (free). saved arena tensors are reclaimed
       when the forward arena resets in ax_graph_cleanup. */
}

/* ================================================================
   lifecycle
   ================================================================ */

static void mha_destroy(ax_layer_t *self)
{
    ax_mha_t *m = (ax_mha_t *)self;
    if (m->Wq) ax_tensor_destroy(m->Wq);
    if (m->Wk) ax_tensor_destroy(m->Wk);
    if (m->Wv) ax_tensor_destroy(m->Wv);
    if (m->Wo) ax_tensor_destroy(m->Wo);
    if (m->bq) ax_tensor_destroy(m->bq);
    if (m->bk) ax_tensor_destroy(m->bk);
    if (m->bv) ax_tensor_destroy(m->bv);
    if (m->bo) ax_tensor_destroy(m->bo);
    if (m->Wqkv_cache) ax_tensor_destroy(m->Wqkv_cache);
    if (m->bqkv_cache) ax_tensor_destroy(m->bqkv_cache);
    free(m);
}

ax_layer_t *ax_mha_create(int64_t d_model, int n_heads,
                           bool use_bias, bool causal)
{
    if (d_model <= 0 || n_heads <= 0) return NULL;
    if (d_model % n_heads != 0) return NULL;

    ax_mha_t *m = (ax_mha_t *)calloc(1, sizeof(ax_mha_t));
    if (!m) return NULL;

    m->base.ops.forward = mha_forward;
    m->base.ops.destroy = mha_destroy;
    m->base.type = AX_LAYER_MHA;
    m->base.training = true;
    m->base.input_features = d_model;
    m->base.output_features = d_model;

    m->d_model = d_model;
    m->n_heads = n_heads;
    m->d_k = d_model / n_heads;
    m->use_bias = use_bias;
    m->causal = causal;

    int64_t ws[] = {d_model, d_model};
    m->Wq = ax_tensor_create(ws, 2, AX_FLOAT32);
    m->Wk = ax_tensor_create(ws, 2, AX_FLOAT32);
    m->Wv = ax_tensor_create(ws, 2, AX_FLOAT32);
    m->Wo = ax_tensor_create(ws, 2, AX_FLOAT32);
    if (!m->Wq || !m->Wk || !m->Wv || !m->Wo) { mha_destroy((ax_layer_t *)m); return NULL; }

    /* xavier / kaiming_uniform is fine here; transformers typically use
       xavier (fan_in + fan_out) on projection weights. kaiming was tuned
       for relu so use xavier for a plain linear projection. */
    ax_init_xavier_uniform(m->Wq, d_model, d_model);
    ax_init_xavier_uniform(m->Wk, d_model, d_model);
    ax_init_xavier_uniform(m->Wv, d_model, d_model);
    ax_init_xavier_uniform(m->Wo, d_model, d_model);

    m->Wq->requires_grad = true;
    m->Wk->requires_grad = true;
    m->Wv->requires_grad = true;
    m->Wo->requires_grad = true;

    m->base.params[0] = m->Wq;
    m->base.params[1] = m->Wk;
    m->base.params[2] = m->Wv;
    m->base.params[3] = m->Wo;
    m->base.n_params = 4;

    if (use_bias) {
        int64_t bs[] = {d_model};
        m->bq = ax_tensor_zeros(bs, 1, AX_FLOAT32);
        m->bk = ax_tensor_zeros(bs, 1, AX_FLOAT32);
        m->bv = ax_tensor_zeros(bs, 1, AX_FLOAT32);
        m->bo = ax_tensor_zeros(bs, 1, AX_FLOAT32);
        if (!m->bq || !m->bk || !m->bv || !m->bo) { mha_destroy((ax_layer_t *)m); return NULL; }
        m->bq->requires_grad = true;
        m->bk->requires_grad = true;
        m->bv->requires_grad = true;
        m->bo->requires_grad = true;
        m->base.params[4] = m->bq;
        m->base.params[5] = m->bk;
        m->base.params[6] = m->bv;
        m->base.params[7] = m->bo;
        m->base.n_params = 8;
    }

    /* Wqkv_cache / bqkv_cache are allocated lazily in refresh_fused_qkv */
    m->Wqkv_cache = NULL;
    m->bqkv_cache = NULL;
    m->Wq_gen = 0;
    m->Wk_gen = 0;
    m->Wv_gen = 0;

    return (ax_layer_t *)m;
}
