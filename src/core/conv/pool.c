/* conv/pool.c — pooling layers (maxpool, avgpool, global avgpool, flatten).

   k.2 split: extracted from src/core/conv.c. these layers are
   conceptually distinct from convolution — they don't share scratch,
   im2col, or winograd state with conv2d — and lived in conv.c only
   because they were small enough at the time. now that conv.c is at
   4.6 K LOC, separating them improves navigation without any
   functional change.

   pool_ctx_t below is the per-call backward context (input geometry +
   layout) and is local to this tu. all four layers share the
   pool_destroy hook since they hold no extra resources beyond their
   own struct.

   nhwc fast paths use channels-innermost layout for cache-friendly
   simd: for maxpool, channels are reduced independently so the
   per-window vmax does no horizontal work; for avgpool, the running
   sum is also channel-vectorised. */

#include "axiom/conv.h"
#include "axiom/ops.h"
#include "axiom/autograd.h"
#include "axiom/tensor.h"
#include "axiom/error.h"
#include "../../compute/backends/simd_defs.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* compute output spatial dimension. mirrors the helper in conv.c — both
   tus need it independently, kept static here so pool.c links without a
   header dependency on conv.c internals. returns -1 on invalid config. */
static inline int64_t conv_out_dim(int64_t in_dim, int kernel, int stride, int pad)
{
    if (stride <= 0) return -1;
    int64_t out = (in_dim + 2 * pad - kernel) / stride + 1;
    if (out <= 0) return -1;
    return out;
}

/* maxpool2d */

/* context for maxpool backward: stores input shape, pool params, layout. */
typedef struct {
    int64_t N, C, H, W;
    int k, s, p;
    ax_layout_t layout;  /* AX_LAYOUT_NCHW or AX_LAYOUT_NHWC */
} pool_ctx_t;

/* NHWC maxpool forward: input [N, H, W, C] → output [N, OH, OW, C].
   per (n, oy, ox), process channels in SIMD blocks of AX_VF32_WIDTH; the
   max is taken across the kxk window via vmax — no horizontal reduction
   needed (channels are independent). this is fundamentally cheaper per
   output than the NCHW pmax_pack pattern.
   indices buffer (when recording) stores the linear input HW index of the
   argmax per output channel. */
static void maxpool2d_nhwc_forward(
    const float *id, float *od, float *idxd,
    int64_t N, int64_t H, int64_t W, int64_t C,
    int64_t oh, int64_t ow, int k, int s, int p, bool record)
{
    int64_t HWC = H * W * C;
    int64_t OWC = ow * C;
    int64_t TC = AX_VF32_WIDTH;

    /* parallelize over n*oy outer dim — each (n, oy) is independent. */
    int64_t NOH = N * oh;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t noh = 0; noh < NOH; noh++) {
        int64_t n  = noh / oh;
        int64_t oy = noh % oh;
        const float *in_n = id + n * HWC;
        float *out_n_y = od + (n * oh + oy) * OWC;
        float *idx_n_y = record ? idxd + (n * oh + oy) * OWC : NULL;

        for (int64_t ox = 0; ox < ow; ox++) {
            float *out_pos = out_n_y + ox * C;
            float *idx_pos = record ? idx_n_y + ox * C : NULL;

            /* SIMD across channels (when not recording argmax) */
            int64_t c = 0;
            if (!record) {
                int64_t cv_end = C - (C % TC);
                for (; c < cv_end; c += TC) {
                    ax_vf32 acc = ax_vf32_set1(-INFINITY);
                    for (int ky = 0; ky < k; ky++) {
                        int64_t iy = oy * s - p + ky;
                        if (iy < 0 || iy >= H) continue;
                        for (int kx = 0; kx < k; kx++) {
                            int64_t ix = ox * s - p + kx;
                            if (ix < 0 || ix >= W) continue;
                            ax_vf32 v = ax_vf32_loadu(in_n + (iy * W + ix) * C + c);
                            acc = ax_vf32_max(acc, v);
                        }
                    }
                    ax_vf32_storeu(out_pos + c, acc);
                }
            }
            /* scalar path (always for recording, also for tail channels) */
            for (; c < C; c++) {
                float mx = -INFINITY;
                int64_t max_pos = -1;
                for (int ky = 0; ky < k; ky++) {
                    int64_t iy = oy * s - p + ky;
                    if (iy < 0 || iy >= H) continue;
                    for (int kx = 0; kx < k; kx++) {
                        int64_t ix = ox * s - p + kx;
                        if (ix < 0 || ix >= W) continue;
                        float v = in_n[(iy * W + ix) * C + c];
                        if (v > mx) { mx = v; max_pos = iy * W + ix; }
                    }
                }
                out_pos[c] = (max_pos >= 0) ? mx : 0.0f;
                if (record) idx_pos[c] = (float)max_pos;
            }
        }
    }
}

/* NHWC maxpool backward: scatter grad_out into input_grad at the recorded
   argmax positions. each (n, oy, ox, c) contributes one scalar add. */
static void maxpool2d_nhwc_backward(
    float *ig, const float *go, const float *idx,
    int64_t N, int64_t H, int64_t W, int64_t C,
    int64_t oh, int64_t ow)
{
    int64_t HWC = H * W * C;

    /* per-(n, oy) writes are within a single sample's input grad slice; rows
       (oy values) within a sample can have argmax indices that point to the
       SAME input pixel in adjacent oy outputs (overlapping pool windows). but
       maxpool with stride==k has disjoint windows; with stride<k there can
       be collisions. for safety, use per-sample parallelism only. */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t n = 0; n < N; n++) {
        const float *go_n = go + (n * oh) * (ow * C);
        const float *idx_n = idx + (n * oh) * (ow * C);
        float *ig_n = ig + n * HWC;
        for (int64_t oy = 0; oy < oh; oy++) {
            for (int64_t ox = 0; ox < ow; ox++) {
                const float *go_pos = go_n + (oy * ow + ox) * C;
                const float *idx_pos = idx_n + (oy * ow + ox) * C;
                for (int64_t c = 0; c < C; c++) {
                    int64_t pos = (int64_t)idx_pos[c];
                    if (pos >= 0)
                        ig_n[pos * C + c] += go_pos[c];
                }
            }
        }
    }
}

/* NHWC avgpool forward: input [N, H, W, C] → output [N, OH, OW, C].
   per (n, oy, ox), accumulate sum across kxk window for SIMD blocks of C
   then divide by count (handling pad). */
static void avgpool2d_nhwc_forward(
    const float *id, float *od,
    int64_t N, int64_t H, int64_t W, int64_t C,
    int64_t oh, int64_t ow, int k, int s, int p)
{
    int64_t HWC = H * W * C;
    int64_t OWC = ow * C;
    int64_t TC = AX_VF32_WIDTH;
    int64_t cv_end = C - (C % TC);

    int64_t NOH = N * oh;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t noh = 0; noh < NOH; noh++) {
        int64_t n  = noh / oh;
        int64_t oy = noh % oh;
        const float *in_n = id + n * HWC;
        float *out_n_y = od + (n * oh + oy) * OWC;

        for (int64_t ox = 0; ox < ow; ox++) {
            float *out_pos = out_n_y + ox * C;

            int count = 0;
            /* count valid positions in this window (depends on pad/edge). */
            for (int ky = 0; ky < k; ky++) {
                int64_t iy = oy * s - p + ky;
                if (iy < 0 || iy >= H) continue;
                for (int kx = 0; kx < k; kx++) {
                    int64_t ix = ox * s - p + kx;
                    if (ix < 0 || ix >= W) continue;
                    count++;
                }
            }
            if (count == 0) {
                memset(out_pos, 0, (size_t)C * sizeof(float));
                continue;
            }
            float inv_count = 1.0f / (float)count;
            ax_vf32 v_inv = ax_vf32_set1(inv_count);

            int64_t c = 0;
            for (; c < cv_end; c += TC) {
                ax_vf32 acc = ax_vf32_zero();
                for (int ky = 0; ky < k; ky++) {
                    int64_t iy = oy * s - p + ky;
                    if (iy < 0 || iy >= H) continue;
                    for (int kx = 0; kx < k; kx++) {
                        int64_t ix = ox * s - p + kx;
                        if (ix < 0 || ix >= W) continue;
                        acc = ax_vf32_add(acc, ax_vf32_loadu(in_n + (iy * W + ix) * C + c));
                    }
                }
                ax_vf32_storeu(out_pos + c, ax_vf32_mul(acc, v_inv));
            }
            for (; c < C; c++) {
                float sum = 0.0f;
                for (int ky = 0; ky < k; ky++) {
                    int64_t iy = oy * s - p + ky;
                    if (iy < 0 || iy >= H) continue;
                    for (int kx = 0; kx < k; kx++) {
                        int64_t ix = ox * s - p + kx;
                        if (ix < 0 || ix >= W) continue;
                        sum += in_n[(iy * W + ix) * C + c];
                    }
                }
                out_pos[c] = sum * inv_count;
            }
        }
    }
}

/* NHWC avgpool backward: each output's grad is uniformly distributed
   over its kxk valid input positions. SIMD across C per scatter. */
static void avgpool2d_nhwc_backward(
    float *ig, const float *go,
    int64_t N, int64_t H, int64_t W, int64_t C,
    int64_t oh, int64_t ow, int k, int s, int p)
{
    int64_t HWC = H * W * C;
    int64_t OWC = ow * C;
    int64_t TC = AX_VF32_WIDTH;
    int64_t cv_end = C - (C % TC);

    /* per-sample to avoid races between overlapping windows. */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t n = 0; n < N; n++) {
        const float *go_n = go + n * oh * OWC;
        float *ig_n = ig + n * HWC;
        for (int64_t oy = 0; oy < oh; oy++) {
            for (int64_t ox = 0; ox < ow; ox++) {
                int count = 0;
                for (int ky = 0; ky < k; ky++) {
                    int64_t iy = oy * s - p + ky;
                    if (iy < 0 || iy >= H) continue;
                    for (int kx = 0; kx < k; kx++) {
                        int64_t ix = ox * s - p + kx;
                        if (ix < 0 || ix >= W) continue;
                        count++;
                    }
                }
                if (count == 0) continue;
                float inv_count = 1.0f / (float)count;
                ax_vf32 v_inv = ax_vf32_set1(inv_count);
                const float *go_pos = go_n + (oy * ow + ox) * C;

                for (int ky = 0; ky < k; ky++) {
                    int64_t iy = oy * s - p + ky;
                    if (iy < 0 || iy >= H) continue;
                    for (int kx = 0; kx < k; kx++) {
                        int64_t ix = ox * s - p + kx;
                        if (ix < 0 || ix >= W) continue;
                        float *ig_pos = ig_n + (iy * W + ix) * C;
                        int64_t c = 0;
                        for (; c < cv_end; c += TC) {
                            ax_vf32 g = ax_vf32_loadu(go_pos + c);
                            ax_vf32 v = ax_vf32_mul(g, v_inv);
                            ax_vf32_storeu(ig_pos + c,
                                ax_vf32_add(ax_vf32_loadu(ig_pos + c), v));
                        }
                        for (; c < C; c++) ig_pos[c] += go_pos[c] * inv_count;
                    }
                }
            }
        }
    }
}

static void maxpool2d_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *input = self->inputs[0];
    ax_tensor_t *indices = self->saved[0]; /* argmax linear indices into input spatial */

    if (!input->requires_grad) return;

    pool_ctx_t *ctx = (pool_ctx_t *)self->ctx;
    int64_t N = ctx->N, C = ctx->C, H = ctx->H, W = ctx->W;

    if (!input->grad) {
        input->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
        if (input->grad) input->grad->layout = ctx->layout;  /* match input layout */
    }
    if (!input->grad) { free(ctx); self->ctx = NULL; return; }

    /* NHWC dispatch: for NHWC, grad_out shape is [N, OH, OW, C], indices same. */
    if (ctx->layout == AX_LAYOUT_NHWC) {
        int64_t oh = grad_out->shape[1], ow = grad_out->shape[2];
        maxpool2d_nhwc_backward(
            (float *)input->grad->storage->data,
            (const float *)grad_out->storage->data,
            (const float *)indices->storage->data,
            N, H, W, C, oh, ow);
        free(ctx); self->ctx = NULL;
        return;
    }

    int64_t oh = grad_out->shape[2], ow = grad_out->shape[3];
    float *ig = (float *)input->grad->storage->data;
    float *go = (float *)grad_out->storage->data;
    float *idx = (float *)indices->storage->data;
    int64_t NC = N * C;

    /* per-(n,c) writes are disjoint: each (n,c) writes only into its own
       channel slab of the input gradient. safe to parallelize. */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t nc = 0; nc < NC; nc++)
    {
        int64_t n = nc / C;
        int64_t c = nc % C;
        for (int64_t y = 0; y < oh; y++)
            for (int64_t x = 0; x < ow; x++) {
                int64_t out_idx = ((n * C + c) * oh + y) * ow + x;
                int64_t max_pos = (int64_t)idx[out_idx];
                if (max_pos >= 0) {
                    int64_t iy = max_pos / W;
                    int64_t ix = max_pos % W;
                    ig[((n * C + c) * H + iy) * W + ix] += go[out_idx];
                }
            }
    }
    free(ctx); self->ctx = NULL;
}

static ax_tensor_t *maxpool2d_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_maxpool2d_t *pool = (ax_maxpool2d_t *)self;

    if (input->ndim != 4)
    {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "maxpool2d expects 4D input");
        return NULL;
    }

    ax_tensor_t *inp = ax_ensure_contiguous(input);
    if (!inp) return NULL;

    ax_layout_t layout = ax_tensor_get_layout(input);
    int k = pool->kernel_size, s = pool->stride, p = pool->padding;

    /* layout-aware shape extraction. */
    int64_t N, C, H, W;
    if (layout == AX_LAYOUT_NHWC) {
        N = inp->shape[0]; H = inp->shape[1]; W = inp->shape[2]; C = inp->shape[3];
    } else {
        N = inp->shape[0]; C = inp->shape[1]; H = inp->shape[2]; W = inp->shape[3];
    }
    int64_t oh = conv_out_dim(H, k, s, p);
    int64_t ow = conv_out_dim(W, k, s, p);
    if (oh <= 0 || ow <= 0)
    {
        ax_err_set(AX_ERR_INVALID_SHAPE, "maxpool2d: invalid output dimensions");
        if (inp != input) ax_tensor_destroy(inp);
        return NULL;
    }

    /* output shape mirrors input layout. */
    int64_t out_shape_nchw[] = {N, C, oh, ow};
    int64_t out_shape_nhwc[] = {N, oh, ow, C};
    const int64_t *out_shape = (layout == AX_LAYOUT_NHWC) ? out_shape_nhwc : out_shape_nchw;
    ax_tensor_t *output = ax_tensor_zeros(out_shape, 4, AX_FLOAT32);
    if (output) output->layout = layout;

    /* NHWC fast path: SIMD across channels, no horizontal reduction. */
    if (layout == AX_LAYOUT_NHWC && output) {
        bool record_nhwc = ax_grad_enabled() && input->requires_grad;
        ax_tensor_t *indices_nhwc = NULL;
        if (record_nhwc) {
            indices_nhwc = ax_tensor_zeros(out_shape, 4, AX_FLOAT32);
            if (indices_nhwc) indices_nhwc->layout = AX_LAYOUT_NHWC;
            else record_nhwc = false;
        }
        maxpool2d_nhwc_forward(
            (const float *)inp->storage->data,
            (float *)output->storage->data,
            record_nhwc ? (float *)indices_nhwc->storage->data : NULL,
            N, H, W, C, oh, ow, k, s, p, record_nhwc);

        if (inp != input) ax_tensor_destroy(inp);

        if (record_nhwc) {
            pool_ctx_t *ctx = malloc(sizeof(pool_ctx_t));
            if (!ctx) { ax_tensor_destroy(indices_nhwc); return output; }
            ctx->N = N; ctx->C = C; ctx->H = H; ctx->W = W;
            ctx->k = k; ctx->s = s; ctx->p = p;
            ctx->layout = AX_LAYOUT_NHWC;
            ax_grad_fn_t *gf = ax_grad_fn_create(maxpool2d_backward);
            gf->inputs[0] = input;
            gf->n_inputs = 1;
            gf->saved[0] = indices_nhwc;
            gf->saved_owned[0] = true;
            gf->n_saved = 1;
            gf->ctx = ctx;
            gf->ctx_cleanup = (void(*)(void*))free;
            output->requires_grad = true;
            output->grad_fn = gf;
        }
        return output;
    }

    /* NCHW path (existing): */
    if (!output) { if (inp != input) ax_tensor_destroy(inp); return NULL; }

    bool record = ax_grad_enabled() && input->requires_grad;
    ax_tensor_t *indices = NULL;
    float *idxd = NULL;
    if (record) {
        indices = ax_tensor_zeros(out_shape, 4, AX_FLOAT32);
        if (!indices) record = false;
        else idxd = (float *)indices->storage->data;
    }

    float *id = (float *)inp->storage->data;
    float *od = (float *)output->storage->data;

    /* parallelize over (n*C) — each (n,c) pair is independent.
       collapse(2) requires perfectly nested loops with no code between, so we use a fused index. */
    int64_t NC = N * C;

    /* fast path: k=2, s=2, p=0 (most common maxpool config).
       SIMD over the output-width dim using ax_vf32_pmax_pack to do the
       horizontal pair-reduce in one ISA-generic intrinsic (vpmaxq_f32 on
       NEON, _mm512_permutex2var+max on AVX-512, shuffle-permute on AVX2,
       scalar fallback). per inner iter: 4 contiguous SIMD loads (2 from
       each of 2 input rows), 2 elementwise vmax across rows, 2 horizontal
       pair-max → 2 SIMD output vectors (= 2*WIDTH outputs). the index
       record path stays scalar; backward needs the exact picked argmax
       and SIMD branchless tracking is more code than it's worth. */
    if (k == 2 && s == 2 && p == 0)
    {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t nc = 0; nc < NC; nc++)
        {
            int64_t n = nc / C;
            int64_t c = nc % C;
            const float *ic = id + (n * C + c) * H * W;
            float *oc = od + (n * C + c) * oh * ow;
            float *ix = record ? idxd + (n * C + c) * oh * ow : NULL;

            for (int64_t y = 0; y < oh; y++)
            {
                int64_t iy = y * 2;
                const float *row0 = ic + iy * W;
                const float *row1 = ic + (iy + 1) * W;
                float *oc_row = oc + y * ow;

                int64_t x = 0;

                /* SIMD path: only when not recording argmax (backward needs
                   exact picked argmax index per output, scalar branch is
                   simpler than tracking through the SIMD pair-max). */
                if (!record) {
                    /* each iter consumes 2*TW input columns and writes TW
                       output columns. step = TW. */
                    int64_t TW = AX_VF32_WIDTH;
                    int64_t vend = ow - (ow % TW);
                    for (; x < vend; x += TW) {
                        int64_t ix2 = x * 2;
                        ax_vf32 a0 = ax_vf32_loadu(row0 + ix2);
                        ax_vf32 a1 = ax_vf32_loadu(row0 + ix2 + TW);
                        ax_vf32 b0 = ax_vf32_loadu(row1 + ix2);
                        ax_vf32 b1 = ax_vf32_loadu(row1 + ix2 + TW);
                        ax_vf32 m_lo = ax_vf32_max(a0, b0);  /* row-wise max, lo halves */
                        ax_vf32 m_hi = ax_vf32_max(a1, b1);  /* row-wise max, hi halves */
                        /* horizontal pair-max across the 2*TW elems → TW outputs */
                        ax_vf32 out = ax_vf32_pmax_pack(m_lo, m_hi);
                        ax_vf32_storeu(oc_row + x, out);
                    }
                    /* overlapping-last-tile trick: when (ow % TW) != 0, do one
                       more SIMD iter starting at ow-TW. it overlaps the previous
                       iter's writes but recomputes the same values, so the final
                       state is correct. avoids 4-element scalar tail on shapes
                       like ow=28 (= 3 SIMD iters + 4 scalar = 86% SIMD); this
                       way the same row finishes in 4 SIMD iters with overlap. */
                    if (x < ow && ow >= TW) {
                        x = ow - TW;
                        int64_t ix2 = x * 2;
                        ax_vf32 a0 = ax_vf32_loadu(row0 + ix2);
                        ax_vf32 a1 = ax_vf32_loadu(row0 + ix2 + TW);
                        ax_vf32 b0 = ax_vf32_loadu(row1 + ix2);
                        ax_vf32 b1 = ax_vf32_loadu(row1 + ix2 + TW);
                        ax_vf32 m_lo = ax_vf32_max(a0, b0);
                        ax_vf32 m_hi = ax_vf32_max(a1, b1);
                        ax_vf32 out = ax_vf32_pmax_pack(m_lo, m_hi);
                        ax_vf32_storeu(oc_row + x, out);
                        x = ow;  /* skip scalar tail */
                    }
                }

                /* scalar tail (or full row when recording argmax) */
                for (; x < ow; x++)
                {
                    int64_t ix2 = x * 2;
                    float a = row0[ix2], b = row0[ix2 + 1];
                    float c2 = row1[ix2], d = row1[ix2 + 1];

                    float m01 = a > b ? a : b;
                    float m23 = c2 > d ? c2 : d;
                    float mx = m01 > m23 ? m01 : m23;
                    oc_row[x] = mx;

                    if (record) {
                        int64_t mi;
                        if (mx == a)      mi = iy * W + ix2;
                        else if (mx == b) mi = iy * W + ix2 + 1;
                        else if (mx == c2) mi = (iy+1) * W + ix2;
                        else               mi = (iy+1) * W + ix2 + 1;
                        ix[y * ow + x] = (float)mi;
                    }
                }
            }
        }
    }
    else
    {
        /* general path with boundary checks */
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t nc = 0; nc < NC; nc++)
        {
            int64_t n = nc / C;
            int64_t c = nc % C;
            for (int64_t y = 0; y < oh; y++)
            {
                for (int64_t x = 0; x < ow; x++)
                {
                    float mx = -FLT_MAX;
                    int64_t max_iy = -1, max_ix = -1;
                    for (int ky = 0; ky < k; ky++)
                    {
                        for (int kx = 0; kx < k; kx++)
                        {
                            int64_t iy = y * s - p + ky;
                            int64_t ix2 = x * s - p + kx;
                            if (iy >= 0 && iy < H && ix2 >= 0 && ix2 < W)
                            {
                                float v = id[((n * C + c) * H + iy) * W + ix2];
                                if (v > mx) { mx = v; max_iy = iy; max_ix = ix2; }
                            }
                        }
                    }
                    int64_t oi = ((n * C + c) * oh + y) * ow + x;
                    od[oi] = mx;
                    if (record)
                        idxd[oi] = (max_iy >= 0) ? (float)(max_iy * W + max_ix) : -1.0f;
                }
            }
        }
    }

    if (inp != input) ax_tensor_destroy(inp);

    if (record) {
        pool_ctx_t *ctx = malloc(sizeof(pool_ctx_t));
        if (!ctx) { ax_tensor_destroy(indices); return output; }
        ctx->N = N; ctx->C = C; ctx->H = H; ctx->W = W;
        ctx->k = k; ctx->s = s; ctx->p = p;
        ctx->layout = AX_LAYOUT_NCHW;

        ax_grad_fn_t *gf = ax_grad_fn_create(maxpool2d_backward);
        gf->inputs[0] = input;
        gf->n_inputs = 1;
        gf->saved[0] = indices;
        gf->saved_owned[0] = true; /* we created the indices tensor */
        gf->n_saved = 1;
        gf->ctx = ctx;
        gf->ctx_cleanup = (void(*)(void*))free;
        output->requires_grad = true;
        output->grad_fn = gf;
    }

    return output;
}

static void pool_destroy(ax_layer_t *self) { free(self); }

ax_layer_t *ax_maxpool2d_create(int kernel_size, int stride, int padding)
{
    ax_maxpool2d_t *p = calloc(1, sizeof(ax_maxpool2d_t));
    AX_RETURN_NULL_IF_ALLOC_FAIL(p, "ax_maxpool2d_create");
    p->base.ops.forward = maxpool2d_forward;
    p->base.ops.destroy = pool_destroy;
    p->base.type = AX_LAYER_MAXPOOL2D;
    p->base.training = true;
    p->kernel_size = kernel_size;
    p->stride = stride;
    p->padding = padding;
    return (ax_layer_t *)p;
}


/* avgpool2d */

static void avgpool2d_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *input = self->inputs[0];
    if (!input->requires_grad) { free(self->ctx); self->ctx = NULL; return; }

    pool_ctx_t *ctx = (pool_ctx_t *)self->ctx;
    int64_t N = ctx->N, C = ctx->C, H = ctx->H, W = ctx->W;
    int k = ctx->k, s = ctx->s, p = ctx->p;

    if (!input->grad) {
        input->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
        if (input->grad) input->grad->layout = ctx->layout;
    }
    if (!input->grad) { free(ctx); self->ctx = NULL; return; }

    /* NHWC dispatch */
    if (ctx->layout == AX_LAYOUT_NHWC) {
        int64_t oh = grad_out->shape[1], ow = grad_out->shape[2];
        avgpool2d_nhwc_backward(
            (float *)input->grad->storage->data,
            (const float *)grad_out->storage->data,
            N, H, W, C, oh, ow, k, s, p);
        free(ctx); self->ctx = NULL;
        return;
    }
    int64_t oh = grad_out->shape[2], ow = grad_out->shape[3];

    float *ig = (float *)input->grad->storage->data;
    float *go = (float *)grad_out->storage->data;
    int64_t NC = N * C;

    /* per-(n,c) writes are disjoint: each (n,c) writes only into its own
       channel slab. safe to parallelize over (n,c). */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t nc = 0; nc < NC; nc++)
    {
        int64_t n = nc / C;
        int64_t c = nc % C;
        for (int64_t y = 0; y < oh; y++)
            for (int64_t x = 0; x < ow; x++) {
                int count = 0;
                for (int ky = 0; ky < k; ky++)
                    for (int kx = 0; kx < k; kx++) {
                        int64_t iy = y * s - p + ky;
                        int64_t ix = x * s - p + kx;
                        if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                            count++;
                    }
                if (count == 0) continue;
                float g = go[((n * C + c) * oh + y) * ow + x] / (float)count;
                for (int ky = 0; ky < k; ky++)
                    for (int kx = 0; kx < k; kx++) {
                        int64_t iy = y * s - p + ky;
                        int64_t ix = x * s - p + kx;
                        if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                            ig[((n * C + c) * H + iy) * W + ix] += g;
                    }
            }
    }
    free(ctx); self->ctx = NULL;
}

static ax_tensor_t *avgpool2d_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_avgpool2d_t *pool = (ax_avgpool2d_t *)self;

    if (input->ndim != 4) return NULL;

    ax_tensor_t *inp = ax_ensure_contiguous(input);
    if (!inp) return NULL;

    ax_layout_t layout = ax_tensor_get_layout(input);
    int64_t N, C, H, W;
    if (layout == AX_LAYOUT_NHWC) {
        N = inp->shape[0]; H = inp->shape[1]; W = inp->shape[2]; C = inp->shape[3];
    } else {
        N = inp->shape[0]; C = inp->shape[1]; H = inp->shape[2]; W = inp->shape[3];
    }
    int k = pool->kernel_size, s = pool->stride, p = pool->padding;
    int64_t oh = conv_out_dim(H, k, s, p);
    int64_t ow = conv_out_dim(W, k, s, p);
    if (oh <= 0 || ow <= 0)
    {
        ax_err_set(AX_ERR_INVALID_SHAPE, "avgpool2d: invalid output dimensions");
        if (inp != input) ax_tensor_destroy(inp);
        return NULL;
    }

    int64_t out_shape_nchw[] = {N, C, oh, ow};
    int64_t out_shape_nhwc[] = {N, oh, ow, C};
    const int64_t *out_shape = (layout == AX_LAYOUT_NHWC) ? out_shape_nhwc : out_shape_nchw;
    ax_tensor_t *output = ax_tensor_zeros(out_shape, 4, AX_FLOAT32);
    if (!output) { if (inp != input) ax_tensor_destroy(inp); return NULL; }
    output->layout = layout;

    /* NHWC fast path */
    if (layout == AX_LAYOUT_NHWC) {
        avgpool2d_nhwc_forward((const float *)inp->storage->data,
                                (float *)output->storage->data,
                                N, H, W, C, oh, ow, k, s, p);
        if (inp != input) ax_tensor_destroy(inp);
        if (ax_grad_enabled() && input->requires_grad) {
            pool_ctx_t *ctx = malloc(sizeof(pool_ctx_t));
            if (!ctx) return output;
            ctx->N = N; ctx->C = C; ctx->H = H; ctx->W = W;
            ctx->k = k; ctx->s = s; ctx->p = p;
            ctx->layout = AX_LAYOUT_NHWC;
            ax_grad_fn_t *gf = ax_grad_fn_create(avgpool2d_backward);
            gf->inputs[0] = input;
            gf->n_inputs = 1;
            gf->ctx = ctx;
            gf->ctx_cleanup = (void(*)(void*))free;
            output->requires_grad = true;
            output->grad_fn = gf;
        }
        return output;
    }

    float *id = (float *)inp->storage->data;
    float *od = (float *)output->storage->data;
    int64_t NC = N * C;

    /* fast path: k=2 s=2 p=0 with SIMD (mirror of maxpool fast path).
       per inner iter: 4 loads + 2 vadd across rows + 1 horizontal pair-sum
       + 1 multiply by 0.25 → TW outputs. */
    if (k == 2 && s == 2 && p == 0)
    {
        const ax_vf32 v_quarter = ax_vf32_set1(0.25f);
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t nc = 0; nc < NC; nc++)
        {
            int64_t n = nc / C;
            int64_t c = nc % C;
            const float *ic = id + (n * C + c) * H * W;
            float *oc = od + (n * C + c) * oh * ow;
            for (int64_t y = 0; y < oh; y++) {
                int64_t iy = y * 2;
                const float *row0 = ic + iy * W;
                const float *row1 = ic + (iy + 1) * W;
                float *oc_row = oc + y * ow;
                int64_t x = 0;
                int64_t TW = AX_VF32_WIDTH;
                int64_t vend = ow - (ow % TW);
                for (; x < vend; x += TW) {
                    int64_t ix2 = x * 2;
                    ax_vf32 a0 = ax_vf32_loadu(row0 + ix2);
                    ax_vf32 a1 = ax_vf32_loadu(row0 + ix2 + TW);
                    ax_vf32 b0 = ax_vf32_loadu(row1 + ix2);
                    ax_vf32 b1 = ax_vf32_loadu(row1 + ix2 + TW);
                    ax_vf32 s_lo = ax_vf32_add(a0, b0);
                    ax_vf32 s_hi = ax_vf32_add(a1, b1);
                    ax_vf32 sum = ax_vf32_padd_pack(s_lo, s_hi);
                    ax_vf32_storeu(oc_row + x, ax_vf32_mul(sum, v_quarter));
                }
                /* overlapping last tile (see maxpool comment) */
                if (x < ow && ow >= TW) {
                    x = ow - TW;
                    int64_t ix2 = x * 2;
                    ax_vf32 a0 = ax_vf32_loadu(row0 + ix2);
                    ax_vf32 a1 = ax_vf32_loadu(row0 + ix2 + TW);
                    ax_vf32 b0 = ax_vf32_loadu(row1 + ix2);
                    ax_vf32 b1 = ax_vf32_loadu(row1 + ix2 + TW);
                    ax_vf32 s_lo = ax_vf32_add(a0, b0);
                    ax_vf32 s_hi = ax_vf32_add(a1, b1);
                    ax_vf32 sum = ax_vf32_padd_pack(s_lo, s_hi);
                    ax_vf32_storeu(oc_row + x, ax_vf32_mul(sum, v_quarter));
                    x = ow;
                }
                /* scalar tail (only when ow < TW) */
                for (; x < ow; x++) {
                    int64_t ix2 = x * 2;
                    float a = row0[ix2], b = row0[ix2 + 1];
                    float c2 = row1[ix2], d = row1[ix2 + 1];
                    oc_row[x] = (a + b + c2 + d) * 0.25f;
                }
            }
        }
    }
    else
    {
    /* general path: bounds-checked window sum + count. */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t nc = 0; nc < NC; nc++)
    {
        int64_t n = nc / C;
        int64_t c = nc % C;
        for (int64_t y = 0; y < oh; y++)
        {
            for (int64_t x = 0; x < ow; x++)
            {
                float sum = 0;
                int count = 0;
                for (int ky = 0; ky < k; ky++)
                {
                    for (int kx = 0; kx < k; kx++)
                    {
                        int64_t iy = y * s - p + ky;
                        int64_t ix = x * s - p + kx;
                        if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                        {
                            sum += id[((n * C + c) * H + iy) * W + ix];
                            count++;
                        }
                    }
                }
                od[((n * C + c) * oh + y) * ow + x] = count > 0 ? sum / (float)count : 0.0f;
            }
        }
    }
    }

    if (inp != input) ax_tensor_destroy(inp);

    if (ax_grad_enabled() && input->requires_grad) {
        pool_ctx_t *ctx = malloc(sizeof(pool_ctx_t));
        if (ctx) {
            ctx->N = N; ctx->C = C; ctx->H = H; ctx->W = W;
            ctx->k = k; ctx->s = s; ctx->p = p;
            ctx->layout = AX_LAYOUT_NCHW;

            ax_grad_fn_t *gf = ax_grad_fn_create(avgpool2d_backward);
            gf->inputs[0] = input;
            gf->n_inputs = 1;
            gf->ctx = ctx;
            gf->ctx_cleanup = (void(*)(void*))free;
            output->requires_grad = true;
            output->grad_fn = gf;
        }
    }

    return output;
}

ax_layer_t *ax_avgpool2d_create(int kernel_size, int stride, int padding)
{
    ax_avgpool2d_t *p = calloc(1, sizeof(ax_avgpool2d_t));
    AX_RETURN_NULL_IF_ALLOC_FAIL(p, "ax_avgpool2d_create");
    p->base.ops.forward = avgpool2d_forward;
    p->base.ops.destroy = pool_destroy;
    p->base.type = AX_LAYER_AVGPOOL2D;
    p->base.training = true;
    p->kernel_size = kernel_size;
    p->stride = stride;
    p->padding = padding;
    return (ax_layer_t *)p;
}


/* global average pool: [N, C, H, W] -> [N, C] */

/* context for global avgpool backward */
typedef struct {
    int64_t N, C, H, W;
} gap_ctx_t;

static void global_avgpool_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *input = self->inputs[0];
    if (!input->requires_grad) { free(self->ctx); self->ctx = NULL; return; }

    gap_ctx_t *ctx = (gap_ctx_t *)self->ctx;
    int64_t N = ctx->N, C = ctx->C, H = ctx->H, W = ctx->W;
    int64_t spatial = H * W;

    if (!input->grad)
        input->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
    if (!input->grad) { free(ctx); self->ctx = NULL; return; }

    float *ig = (float *)input->grad->storage->data;
    float *go = (float *)grad_out->storage->data;

    for (int64_t n = 0; n < N; n++)
        for (int64_t c = 0; c < C; c++) {
            float g = go[n * C + c] / (float)spatial;
            for (int64_t i = 0; i < spatial; i++)
                ig[(n * C + c) * spatial + i] += g;
        }
    free(ctx); self->ctx = NULL;
}

static ax_tensor_t *global_avgpool_forward(ax_layer_t *self, ax_tensor_t *input)
{
    if (input->ndim != 4) return NULL;

    ax_tensor_t *inp = ax_ensure_contiguous(input);
    if (!inp) return NULL;

    int64_t N = inp->shape[0], C = inp->shape[1];
    int64_t H = inp->shape[2], W = inp->shape[3];
    int64_t spatial = H * W;

    int64_t out_shape[] = {N, C};
    ax_tensor_t *output = ax_tensor_zeros(out_shape, 2, AX_FLOAT32);
    if (!output) { if (inp != input) ax_tensor_destroy(inp); return NULL; }

    float *id = (float *)inp->storage->data;
    float *od = (float *)output->storage->data;

    for (int64_t n = 0; n < N; n++)
    {
        for (int64_t c = 0; c < C; c++)
        {
            float sum = 0;
            for (int64_t i = 0; i < spatial; i++)
                sum += id[(n * C + c) * spatial + i];
            od[n * C + c] = sum / (float)spatial;
        }
    }

    if (inp != input) ax_tensor_destroy(inp);

    if (ax_grad_enabled() && input->requires_grad) {
        gap_ctx_t *ctx = malloc(sizeof(gap_ctx_t));
        if (ctx) {
            ctx->N = N; ctx->C = C; ctx->H = H; ctx->W = W;

            ax_grad_fn_t *gf = ax_grad_fn_create(global_avgpool_backward);
            gf->inputs[0] = input;
            gf->n_inputs = 1;
            gf->ctx = ctx;
            gf->ctx_cleanup = (void(*)(void*))free;
            output->requires_grad = true;
            output->grad_fn = gf;
        }
    }

    return output;
}

ax_layer_t *ax_global_avgpool2d_create(void)
{
    ax_layer_t *l = calloc(1, sizeof(ax_layer_t));
    AX_RETURN_NULL_IF_ALLOC_FAIL(l, "ax_global_avgpool2d_create");
    l->ops.forward = global_avgpool_forward;
    l->ops.destroy = (void (*)(ax_layer_t *))free;
    l->type = AX_LAYER_GLOBAL_AVGPOOL2D;
    l->training = true;
    return l;
}


/* flatten: [N, C, H, W] -> [N, C*H*W] */

/* context for flatten backward: original shape */
typedef struct {
    int64_t orig_shape[AX_MAX_DIMS];
    int orig_ndim;
} flatten_ctx_t;

static void flatten_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *input = self->inputs[0];
    if (!input->requires_grad) { free(self->ctx); self->ctx = NULL; return; }

    flatten_ctx_t *ctx = (flatten_ctx_t *)self->ctx;

    if (!input->grad)
        input->grad = ax_tensor_zeros(ctx->orig_shape, ctx->orig_ndim, input->dtype);
    if (!input->grad) { free(ctx); self->ctx = NULL; return; }

    /* just copy grad_out (which is [N, flat]) into input->grad (which is [N, C, H, W]) */
    int64_t n = ax_tensor_numel(grad_out);
    float *ig = (float *)input->grad->storage->data;
    float *go = (float *)grad_out->storage->data;
    for (int64_t i = 0; i < n; i++)
        ig[i] += go[i];

    free(ctx); self->ctx = NULL;
}

static ax_tensor_t *flatten_forward(ax_layer_t *self, ax_tensor_t *input)
{
    if (input->ndim < 2) return NULL;

    int64_t N = input->shape[0];
    int64_t flat = 1;
    for (int d = 1; d < input->ndim; d++)
        flat *= input->shape[d];

    int64_t out_shape[] = {N, flat};

    bool record = ax_grad_enabled() && input->requires_grad;

    /* We need a real copy for autograd so the output is a distinct tensor */
    ax_tensor_t *output;
    if (!record && ax_tensor_is_contiguous(input))
        return ax_tensor_reshape(input, out_shape, 2);

    /* make a contiguous copy reshaped to [N, flat] */
    output = ax_tensor_zeros(out_shape, 2, input->dtype);
    if (!output) return NULL;
    int64_t n = ax_tensor_numel(input);
    float *od = (float *)output->storage->data;
    float *id = (float *)input->storage->data;
    /* must respect strides for non-contiguous inputs (e.g., after transpose) */
    for (int64_t i = 0; i < n; i++)
    {
        int64_t remaining = i;
        int64_t src_offset = input->offset;
        for (int d = input->ndim - 1; d >= 0; d--)
        {
            int64_t idx = remaining % input->shape[d];
            remaining /= input->shape[d];
            src_offset += idx * input->strides[d];
        }
        od[i] = id[src_offset];
    }

    if (record) {
        flatten_ctx_t *ctx = malloc(sizeof(flatten_ctx_t));
        if (ctx) {
            ctx->orig_ndim = input->ndim;
            for (int d = 0; d < input->ndim; d++)
                ctx->orig_shape[d] = input->shape[d];

            ax_grad_fn_t *gf = ax_grad_fn_create(flatten_backward);
            gf->inputs[0] = input;
            gf->n_inputs = 1;
            gf->ctx = ctx;
        gf->ctx_cleanup = (void(*)(void*))free;
            output->requires_grad = true;
            output->grad_fn = gf;
        }
    }

    return output;
}

ax_layer_t *ax_flatten_create(void)
{
    ax_layer_t *l = calloc(1, sizeof(ax_layer_t));
    AX_RETURN_NULL_IF_ALLOC_FAIL(l, "ax_flatten_create");
    l->ops.forward = flatten_forward;
    l->ops.destroy = (void (*)(ax_layer_t *))free;
    l->type = AX_LAYER_FLATTEN;
    l->training = true;
    return l;
}
