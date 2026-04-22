/* conv/conv_bn_relu.c — fused conv2d + batchnorm + relu composite layer.

   k.2 split: extracted from src/core/conv.c. this layer is logically a
   composite of three other layers (conv, bn, relu) rather than a
   variant of conv2d, so it kept its forward/backward in conv.c only
   for historical reasons (the bn fast path needs access to conv's
   per-thread scratch). pulled out here both because it's the largest
   single chunk in the legacy file (~770 LOC) and because it has a
   clear interface — ax_conv_bn_relu_create_from / ax_sequential_fuse —
   that doesn't bleed into anything else.

   forward: collapses three sequential passes over the conv output buffer
   (conv write, bn read+write, relu read+write) into two: conv write +
   bn_relu read+write. the stats reduction (mean/var) still does one extra
   read per channel so overall we go from ~5 buffer passes to ~4, a
   meaningful bandwidth win on memory-bound cnn workloads.

   backward: uses the saved-tensors approach — inlines relu-backward ->
   bn-backward -> conv-backward to keep autograd wiring simple while
   still reusing the per-layer scratch through ensure_scratch_cbr's shim. */

#include "axiom/conv.h"
#include "axiom/ops.h"
#include "axiom/autograd.h"
#include "axiom/init.h"
#include "axiom/compute.h"
#include "axiom/error.h"
#include "axiom/norm.h" /* for ax_batchnorm_t layout */
#include "internal.h"
#include "../../compute/backends/simd_defs.h"
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#define AX_OMP_MAX_THREADS() omp_get_max_threads()
#define AX_OMP_THREAD_NUM() omp_get_thread_num()
#else
#define AX_OMP_MAX_THREADS() 1
#define AX_OMP_THREAD_NUM() 0
#endif

/* backward ctx: shape snapshot + layer pointer. the saved tensors
   (x_hat, inv_std, contig input) live in gf->saved[] so graph cleanup
   frees them automatically. */
typedef struct {
    ax_conv_bn_relu_t *layer;
    int64_t N, C_out, out_h, out_w;
    int64_t C_in, H, W;
} cbr_bwd_ctx_t;

static void cbr_ctx_cleanup(void *p) { free(p); }

static void conv_bn_relu_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    cbr_bwd_ctx_t *ctx = (cbr_bwd_ctx_t *)self->ctx;
    ax_conv_bn_relu_t *L = ctx->layer;
    ax_tensor_t *input_orig = self->inputs[0];
    ax_tensor_t *input_data = self->saved[0]; /* contiguous input */
    ax_tensor_t *x_hat_t    = self->saved[1];
    ax_tensor_t *inv_std_t  = self->saved[2];

    int64_t N = ctx->N, C_out = ctx->C_out, out_h = ctx->out_h, out_w = ctx->out_w;
    int64_t C_in = ctx->C_in, H = ctx->H, W = ctx->W;
    int kh = L->kernel_h, kw = L->kernel_w;
    int sh = L->stride_h, sw = L->stride_w;
    int ph = L->pad_h, pw = L->pad_w;
    int64_t spatial = out_h * out_w;
    float Nf = (float)(N * spatial);

    /* build the gradient wrt the conv output into an arena-allocated buffer.
       this is: relu_mask * bn_backward_dx(grad_out). the relu mask is
       reconstructed from saved x_hat by computing bn_out = gamma*x_hat+beta
       (cheap, avoids saving yet another tensor). */
    ax_arena_t *arena = ax_backward_arena();
    int64_t gconv_shape[] = {N, C_out, out_h, out_w};
    ax_tensor_t *grad_conv = ax_tensor_arena_zeros(arena, gconv_shape, 4, AX_FLOAT32);
    if (!grad_conv) return;

    float *go = (float *)grad_out->storage->data;
    float *xh = (float *)x_hat_t->storage->data;
    float *istd = (float *)inv_std_t->storage->data;
    float *gd = (float *)L->gamma->storage->data;
    float *bd = (float *)L->beta->storage->data;
    float *gc = (float *)grad_conv->storage->data;

    float *dg = NULL, *db = NULL;
    if (L->gamma->requires_grad) {
        if (!L->gamma->grad)
            L->gamma->grad = ax_tensor_zeros(L->gamma->shape, L->gamma->ndim, L->gamma->dtype);
        if (L->gamma->grad) dg = (float *)L->gamma->grad->storage->data;
    }
    if (L->beta->requires_grad) {
        if (!L->beta->grad)
            L->beta->grad = ax_tensor_zeros(L->beta->shape, L->beta->ndim, L->beta->dtype);
        if (L->beta->grad) db = (float *)L->beta->grad->storage->data;
    }

    /* per-channel parallel: apply relu mask (bn_out > 0 gate), accumulate
       dgamma/dbeta via SIMD, then compute dx formula into grad_conv.
       each channel c owns disjoint slices of go/xh/gc, so no atomics needed
       on dg[c]/db[c] — they're accessed by exactly one thread. */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
    #endif
    for (int64_t c = 0; c < C_out; c++) {
        float g_c = gd[c], b_c = bd[c];
        ax_vf32 v_gc  = ax_vf32_set1(g_c);
        ax_vf32 v_bc  = ax_vf32_set1(b_c);
        ax_vf32 v_zero = ax_vf32_zero();
        ax_vf32 v_sgo = ax_vf32_zero(), v_sgx = ax_vf32_zero();
        float s_sgo = 0.0f, s_sgx = 0.0f;
        int64_t se = spatial - (spatial % AX_VF32_WIDTH);

        for (int64_t n = 0; n < N; n++) {
            int64_t base = n * C_out * spatial + c * spatial;
            const float *go_p = go + base;
            const float *xh_p = xh + base;
            float *gc_p = gc + base;
            int64_t s = 0;
            for (; s < se; s += AX_VF32_WIDTH) {
                ax_vf32 x = ax_vf32_loadu(xh_p + s);
                ax_vf32 bn_out = ax_vf32_fmadd(v_gc, x, v_bc);
                /* relu mask: cmpgt returns 1.0f where true, 0.0f otherwise.
                   mul by mask gates the gradient without a bitwise AND. */
                ax_vf32 mask = ax_vf32_cmpgt(bn_out, v_zero);
                ax_vf32 g = ax_vf32_mul(ax_vf32_loadu(go_p + s), mask);
                ax_vf32_storeu(gc_p + s, g);
                v_sgo = ax_vf32_add(v_sgo, g);
                v_sgx = ax_vf32_fmadd(g, x, v_sgx);
            }
            for (; s < spatial; s++) {
                float bn_out = g_c * xh_p[s] + b_c;
                float m = (bn_out > 0.0f) ? go_p[s] : 0.0f;
                gc_p[s] = m;
                s_sgo += m;
                s_sgx += m * xh_p[s];
            }
        }
        float sum_go    = ax_vf32_hsum(v_sgo) + s_sgo;
        float sum_go_xh = ax_vf32_hsum(v_sgx) + s_sgx;

        /* no atomic: each thread owns exactly one c, no contention */
        if (dg) dg[c] += sum_go_xh;
        if (db) db[c] += sum_go;

        float coeff = g_c * istd[c] / Nf;
        ax_vf32 v_coeff  = ax_vf32_set1(coeff);
        ax_vf32 v_Nf     = ax_vf32_set1(Nf);
        ax_vf32 v_sum_go = ax_vf32_set1(sum_go);
        ax_vf32 v_sum_gx = ax_vf32_set1(sum_go_xh);

        for (int64_t n = 0; n < N; n++) {
            int64_t base = n * C_out * spatial + c * spatial;
            const float *xh_p = xh + base;
            float *gc_p = gc + base;
            int64_t s = 0;
            for (; s < se; s += AX_VF32_WIDTH) {
                ax_vf32 m = ax_vf32_loadu(gc_p + s);
                ax_vf32 x = ax_vf32_loadu(xh_p + s);
                /* dx = coeff * (N*m - sum_go - x*sum_go_xh) */
                ax_vf32 dx = ax_vf32_mul(v_coeff,
                    ax_vf32_sub(ax_vf32_sub(ax_vf32_mul(v_Nf, m), v_sum_go),
                                ax_vf32_mul(x, v_sum_gx)));
                ax_vf32_storeu(gc_p + s, dx);
            }
            for (; s < spatial; s++) {
                float m = gc_p[s];
                gc_p[s] = coeff * (Nf * m - sum_go - xh_p[s] * sum_go_xh);
            }
        }
    }

    /* step 2: conv backward using grad_conv as upstream gradient.
       near-copy of conv2d_backward using the layer's cached scratch. */
    ax_tensor_t *weight = L->weight;

    if (weight->requires_grad && !weight->grad)
        weight->grad = ax_tensor_zeros(weight->shape, weight->ndim, weight->dtype);
    if (input_orig->requires_grad && !input_orig->grad)
        input_orig->grad = ax_tensor_zeros(input_orig->shape, input_orig->ndim, input_orig->dtype);

    if (L->use_bias && L->bias && L->bias->requires_grad) {
        if (!L->bias->grad)
            L->bias->grad = ax_tensor_zeros(L->bias->shape, L->bias->ndim, L->bias->dtype);
        float *bg = (float *)L->bias->grad->storage->data;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t c = 0; c < C_out; c++) {
            float sum = 0.0f;
            for (int64_t n = 0; n < N; n++) {
                int64_t base = n * C_out * spatial + c * spatial;
                for (int64_t s = 0; s < spatial; s++) sum += gc[base + s];
            }
            bg[c] += sum;
        }
    }

    float *wdata = (float *)weight->storage->data;
    int64_t K = C_in * kh * kw;
    int64_t M = out_h * out_w;

    struct ax_conv_scratch *s = L->scratch;
    if (!s || s->N != N || s->H != H || s->W != W || !s->dw_bufs) {
        /* forward should have built this; if not we can't proceed safely */
        return;
    }
    int T = s->T;

    bool have_gemm_nt_b = ax_compute_has_gemm_nt();
    bool have_gemm_tn_b = ax_compute_has_gemm_tn();

    if (input_orig->requires_grad && !have_gemm_tn_b) {
        if (!s->wt_contig) {
            int64_t wt_shape[] = {K, C_out};
            s->wt_contig = ax_tensor_create(wt_shape, 2, AX_FLOAT32);
        }
        if (s->wt_contig) {
            float *wtd = (float *)s->wt_contig->storage->data;
            for (int64_t r = 0; r < C_out; r++)
                for (int64_t col = 0; col < K; col++)
                    wtd[col * C_out + r] = wdata[r * K + col];
        }
    }
    ax_tensor_t *w2d_bn = s->w2d;
    if (input_orig->requires_grad && have_gemm_tn_b && w2d_bn) {
        memcpy(w2d_bn->storage->data, wdata, (size_t)(C_out * K) * sizeof(float));
    }

    if (weight->requires_grad) {
        for (int t = 0; t < T; t++)
            memset(s->dw_bufs[t]->storage->data, 0, (size_t)(C_out * K) * sizeof(float));
    }

    float *ind = (float *)input_data->storage->data;
    ax_tensor_t *wt_contig = s->wt_contig;

    bool use_batched_cbr = N > 1 && M < AX_CONV_BATCH_M_THRESH
                           && s->batch_col_buf && s->batch_aux_buf;

    if (use_batched_cbr) {
        float *bcd = (float *)s->batch_col_buf->storage->data;
        float *gbd = (float *)s->batch_aux_buf->storage->data;
        int64_t NM = N * M;

        #ifdef _OPENMP
        #pragma omp parallel for num_threads(T) schedule(static)
        #endif
        for (int64_t n = 0; n < N; n++) {
            ax_conv_im2col_into_strided(ind + n * C_in * H * W, C_in, H, W,
                                 kh, kw, sh, sw, ph, pw, out_h, out_w,
                                 bcd, NM, n * M);
            for (int64_t co = 0; co < C_out; co++)
                memcpy(gbd + co * NM + n * M, gc + n * C_out * M + co * M,
                       (size_t)M * sizeof(float));
        }

        if (weight->requires_grad) {
            ax_tensor_t *dw_total = s->dw_bufs[0];
            memset(dw_total->storage->data, 0, (size_t)(C_out * K) * sizeof(float));
            ax_compute_gemm_nt(s->batch_aux_buf, s->batch_col_buf, dw_total);
            float *wg  = (float *)weight->grad->storage->data;
            float *dwl = (float *)dw_total->storage->data;
            int64_t wn = C_out * K, wi = 0, wve = wn - (wn % AX_VF32_WIDTH);
            for (; wi < wve; wi += AX_VF32_WIDTH)
                ax_vf32_storeu(wg + wi, ax_vf32_add(ax_vf32_loadu(wg + wi), ax_vf32_loadu(dwl + wi)));
            for (; wi < wn; wi++) wg[wi] += dwl[wi];
        }

        if (input_orig->requires_grad && (wt_contig || have_gemm_tn_b)) {
            memset(bcd, 0, (size_t)(K * NM) * sizeof(float));
            if (have_gemm_tn_b)
                ax_compute_gemm_tn(w2d_bn, s->batch_aux_buf, s->batch_col_buf);
            else
                ax_compute_gemm(wt_contig, s->batch_aux_buf, s->batch_col_buf);

            #ifdef _OPENMP
            #pragma omp parallel for num_threads(T) schedule(static)
            #endif
            for (int64_t n = 0; n < N; n++) {
                int tid = AX_OMP_THREAD_NUM();
                if (tid >= T) tid = 0;
                float *dcol_d = (float *)s->dcol_bufs[tid]->storage->data;
                for (int64_t k = 0; k < K; k++)
                    memcpy(dcol_d + k * M, bcd + k * NM + n * M, (size_t)M * sizeof(float));
                float *dimg_d = (float *)s->dimg_bufs[tid]->storage->data;
                memset(dimg_d, 0, (size_t)(C_in * H * W) * sizeof(float));
                ax_conv_col2im_into(dcol_d, C_in, H, W, kh, kw, sh, sw, ph, pw, out_h, out_w, dimg_d);
                float *ig = (float *)input_orig->grad->storage->data + n * C_in * H * W;
                int64_t total = C_in * H * W, i = 0, ve = total - (total % AX_VF32_WIDTH);
                for (; i < ve; i += AX_VF32_WIDTH)
                    ax_vf32_storeu(ig + i, ax_vf32_add(ax_vf32_loadu(ig + i), ax_vf32_loadu(dimg_d + i)));
                for (; i < total; i++) ig[i] += dimg_d[i];
            }
        }
    } else {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t n = 0; n < N; n++) {
        int tid = AX_OMP_THREAD_NUM();
        if (tid >= T) tid = 0;
        float *cbd = (float *)s->col_bufs[tid]->storage->data;
        ax_tensor_t *col_buf = s->col_bufs[tid];
        ax_tensor_t *go_mat = s->go_bufs[tid];

        ax_conv_im2col_into(ind + n * C_in * H * W, C_in, H, W,
                     kh, kw, sh, sw, ph, pw, out_h, out_w, cbd);

        memcpy(go_mat->storage->data, gc + n * C_out * M,
               (size_t)(C_out * M) * sizeof(float));

        if (weight->requires_grad) {
            ax_tensor_t *dw_local = s->dw_bufs[tid];
            ax_tensor_t *dw_sample = s->dws_bufs[tid];
            float *dws = (float *)dw_sample->storage->data;
            memset(dws, 0, (size_t)(C_out * K) * sizeof(float));

            if (have_gemm_nt_b) {
                ax_compute_gemm_nt(go_mat, col_buf, dw_sample);
            } else {
                ax_tensor_t *colt_buf = s->colt_bufs[tid];
                float *ctd = (float *)colt_buf->storage->data;
                for (int64_t r = 0; r < K; r++)
                    for (int64_t col = 0; col < M; col++)
                        ctd[col * K + r] = cbd[r * M + col];
                ax_compute_gemm(go_mat, colt_buf, dw_sample);
            }

            float *dwl = (float *)dw_local->storage->data;
            int64_t wn = C_out * K;
            int64_t wi = 0, wve = wn - (wn % AX_VF32_WIDTH);
            for (; wi < wve; wi += AX_VF32_WIDTH)
                ax_vf32_storeu(dwl + wi, ax_vf32_add(ax_vf32_loadu(dwl + wi), ax_vf32_loadu(dws + wi)));
            for (; wi < wn; wi++) dwl[wi] += dws[wi];
        }

        if (input_orig->requires_grad && (wt_contig || have_gemm_tn_b)) {
            ax_tensor_t *dcol_buf = s->dcol_bufs[tid];
            ax_tensor_t *dimg_buf = s->dimg_bufs[tid];

            memset(dcol_buf->storage->data, 0, (size_t)(K * M) * sizeof(float));
            if (have_gemm_tn_b) {
                ax_compute_gemm_tn(w2d_bn, go_mat, dcol_buf);
            } else {
                ax_compute_gemm(wt_contig, go_mat, dcol_buf);
            }

            float *dimg_d = (float *)dimg_buf->storage->data;
            memset(dimg_d, 0, (size_t)(C_in * H * W) * sizeof(float));
            ax_conv_col2im_into((float *)dcol_buf->storage->data, C_in, H, W,
                         kh, kw, sh, sw, ph, pw, out_h, out_w, dimg_d);

            float *ig = (float *)input_orig->grad->storage->data + n * C_in * H * W;
            int64_t total = C_in * H * W;
            int64_t i = 0, ve = total - (total % AX_VF32_WIDTH);
            for (; i < ve; i += AX_VF32_WIDTH)
                ax_vf32_storeu(ig + i, ax_vf32_add(ax_vf32_loadu(ig + i), ax_vf32_loadu(dimg_d + i)));
            for (; i < total; i++) ig[i] += dimg_d[i];
        }
    }

    if (weight->requires_grad) {
        float *wg = (float *)weight->grad->storage->data;
        int64_t wn = C_out * K;
        for (int t = 0; t < T; t++) {
            float *dwl = (float *)s->dw_bufs[t]->storage->data;
            int64_t wi = 0, wve = wn - (wn % AX_VF32_WIDTH);
            for (; wi < wve; wi += AX_VF32_WIDTH)
                ax_vf32_storeu(wg + wi, ax_vf32_add(ax_vf32_loadu(wg + wi), ax_vf32_loadu(dwl + wi)));
            for (; wi < wn; wi++) wg[wi] += dwl[wi];
        }
    }
    } /* end use_batched_cbr else */
}

/* ensure_scratch operates on ax_conv2d_t*; use a transient shim so we can
   share the same scratch logic across the plain and fused layers without
   duplicating ~100 lines of allocation code. */
static struct ax_conv_scratch *ensure_scratch_cbr(ax_conv_bn_relu_t *L,
                                                    int64_t N, int64_t H, int64_t W,
                                                    bool need_backward)
{
    ax_conv2d_t shim;
    memset(&shim, 0, sizeof(shim));
    shim.in_channels  = L->in_channels;
    shim.out_channels = L->out_channels;
    shim.kernel_h = L->kernel_h; shim.kernel_w = L->kernel_w;
    shim.stride_h = L->stride_h; shim.stride_w = L->stride_w;
    shim.pad_h = L->pad_h;       shim.pad_w = L->pad_w;
    shim.scratch = L->scratch;
    struct ax_conv_scratch *s = ax_conv_ensure_scratch(&shim, N, H, W, need_backward);
    L->scratch = shim.scratch;
    return s;
}

static ax_tensor_t *conv_bn_relu_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_conv_bn_relu_t *L = (ax_conv_bn_relu_t *)self;
    if (!input || input->ndim != 4) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "conv_bn_relu expects [N,C,H,W] input");
        return NULL;
    }

    ax_tensor_t *inp = ax_ensure_contiguous(input);
    if (!inp) return NULL;

    int64_t N = inp->shape[0];
    int64_t C_in = inp->shape[1];
    int64_t H = inp->shape[2];
    int64_t W = inp->shape[3];
    int64_t C_out = L->out_channels;
    int kh = L->kernel_h, kw = L->kernel_w;
    int sh = L->stride_h, sw = L->stride_w;
    int ph = L->pad_h, pw = L->pad_w;

    int64_t out_h = ax_conv_out_dim(H, kh, sh, ph);
    int64_t out_w = ax_conv_out_dim(W, kw, sw, pw);
    if (out_h <= 0 || out_w <= 0) {
        ax_err_set(AX_ERR_INVALID_SHAPE, "conv_bn_relu: invalid output dims");
        if (inp != input) ax_tensor_destroy(inp);
        return NULL;
    }

    int64_t K = C_in * kh * kw;
    int64_t M = out_h * out_w;
    int64_t spatial = M;
    int64_t out_shape[] = {N, C_out, out_h, out_w};
    /* skip zero pass: pass 1 below writes every output (direct/smallcin
       with bias init, gemm paths via res→od bias-copy). */
    ax_tensor_t *output = ax_tensor_create(out_shape, 4, AX_FLOAT32);
    if (!output) { if (inp != input) ax_tensor_destroy(inp); return NULL; }

    float *od = (float *)output->storage->data;
    float *wd = (float *)L->weight->storage->data;

    bool record = ax_grad_enabled() && self->training &&
                  (input->requires_grad || L->weight->requires_grad ||
                   L->gamma->requires_grad || L->beta->requires_grad);

    struct ax_conv_scratch *s = ensure_scratch_cbr(L, N, H, W, record);
    if (!s) { if (inp != input) ax_tensor_destroy(inp); ax_tensor_destroy(output); return NULL; }

    ax_tensor_t *w2d = s->w2d;
    memcpy(w2d->storage->data, wd, (size_t)(C_out * K) * sizeof(float));

    int T = s->T;
    float *ind = (float *)inp->storage->data;
    const float *bias_data = (L->use_bias && L->bias)
        ? (const float *)L->bias->storage->data : NULL;

    /* shape-aware path selection mirrors conv2d_forward. */
    bool use_direct = ax_conv_can_direct_3x3(kh, kw, sh, sw, ph, pw, C_in);
    bool use_smallcin = !use_direct && ax_conv_can_direct_smallcin(kh, kw, sh, sw, C_in, out_w);
    bool use_implicit = !use_direct && !use_smallcin && ax_compute_has_conv_gemm() && ax_conv_prefer_implicit_gemm(K, M);
    bool use_batched_cbr_fwd = !use_direct && !use_smallcin && !use_implicit && N > 1
                               && M < AX_CONV_BATCH_M_THRESH
                               && s->batch_col_buf && s->batch_aux_buf;

    /* pass 1: materialize conv output (including bias) into the final output buffer.
       pass 2/3 (bn stats + fused bn+relu apply) overwrite it in place. this saves
       one buffer pass compared to the unfused path which would first write bn
       output into a separate tensor and then relu into another. */
    if (use_batched_cbr_fwd) {
        float *bcd = (float *)s->batch_col_buf->storage->data;
        float *brd = (float *)s->batch_aux_buf->storage->data;
        int64_t NM = N * M;

        #ifdef _OPENMP
        #pragma omp parallel for num_threads(T) schedule(static)
        #endif
        for (int64_t n = 0; n < N; n++)
            ax_conv_im2col_into_strided(ind + n * C_in * H * W, C_in, H, W,
                                 kh, kw, sh, sw, ph, pw, out_h, out_w,
                                 bcd, NM, n * M);

        /* opt_gemm zero-fills C internally; skip our memset. */
        ax_compute_gemm(w2d, s->batch_col_buf, s->batch_aux_buf);

        /* scatter brd[C_out, N, M] → od[N, C_out, M] with bias add (co-outer
           = contiguous brd reads; see conv2d_forward batched path). */
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(T) schedule(static)
        #endif
        for (int64_t co = 0; co < C_out; co++) {
            float bias_val = bias_data ? bias_data[co] : 0.0f;
            ax_vf32 v_b = ax_vf32_set1(bias_val);
            const float *src_co = brd + co * NM;
            for (int64_t n = 0; n < N; n++) {
                float *dst = od + (n * C_out + co) * M;
                const float *src = src_co + n * M;
                int64_t m = 0, me = M - (M % AX_VF32_WIDTH);
                for (; m < me; m += AX_VF32_WIDTH)
                    ax_vf32_storeu(dst + m, ax_vf32_add(ax_vf32_loadu(src + m), v_b));
                for (; m < M; m++) dst[m] = src[m] + bias_val;
            }
        }
    } else {
    /* num_threads(T) caps the team to per-thread scratch slot count. */
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(T) schedule(dynamic, 1)
    #endif
    for (int64_t n = 0; n < N; n++) {
        if (use_direct) {
            /* direct conv writes [C_out, H, W] for the sample, bias folded in.
               this gives us the conv result directly in `od` without scratch. */
            ax_conv_direct_3x3_sample(
                ind + n * C_in * H * W, wd, bias_data,
                od + n * C_out * M, C_in, C_out, H, W);
            continue;
        }
        if (use_smallcin) {
            /* small-C_in direct: writes conv result with bias into od[n].
               BN+ReLU pass below overwrites in-place. */
            ax_conv_direct_smallcin_sample(
                ind + n * C_in * H * W, wd, bias_data,
                od + n * C_out * M,
                C_in, C_out, H, W,
                kh, kw, sh, sw, ph, pw, out_h, out_w);
            continue;
        }

        int tid = AX_OMP_THREAD_NUM();
        if (tid >= T) tid = 0;
        ax_tensor_t *col = s->col_bufs[tid];
        ax_tensor_t *res = s->res_bufs[tid];
        float *rd = (float *)res->storage->data;

        if (use_implicit) {
            ax_conv_params_t cp = {
                .input = ind + n * C_in * H * W,
                .C_in = C_in, .H = H, .W = W,
                .kh = kh, .kw = kw,
                .sh = sh, .sw = sw,
                .ph = ph, .pw = pw,
                .out_h = out_h, .out_w = out_w,
            };
            ax_compute_conv_gemm(w2d, &cp, res);
        } else if (kh == 1 && kw == 1 && sh == 1 && sw == 1
                   && ph == 0 && pw == 0 && H == out_h && W == out_w) {
            /* 1x1 conv: im2col is the identity layout — skip the copy.
               stack-allocated view into inp[n] with refcount=0 (parent keeps
               storage alive). matches conv2d_forward path. */
            ax_storage_t in_storage;
            in_storage.data = (void *)(ind + n * C_in * H * W);
            in_storage.size_bytes = (size_t)(C_in * H * W) * sizeof(float);
            atomic_store(&in_storage.refcount, 0);
            in_storage.device = AX_DEVICE_CPU;
            in_storage.is_arena_temp = true;
            in_storage.generation = 1;

            ax_tensor_t in_slice;
            memset(&in_slice, 0, sizeof(in_slice));
            in_slice.storage = &in_storage;
            in_slice.ndim = 2;
            in_slice.dtype = AX_FLOAT32;
            in_slice.offset = 0;
            in_slice.shape[0] = C_in;
            in_slice.shape[1] = M;
            in_slice.strides[0] = M;
            in_slice.strides[1] = 1;

            /* opt_gemm zero-fills C internally; skip our memset. */
            ax_compute_gemm(w2d, &in_slice, res);
        } else {
            float *cd = (float *)col->storage->data;
            ax_conv_im2col_into(ind + n * C_in * H * W, C_in, H, W,
                         kh, kw, sh, sw, ph, pw, out_h, out_w, cd);
            /* opt_gemm zero-fills C internally; skip our memset. */
            ax_compute_gemm(w2d, col, res);
        }

        for (int64_t co = 0; co < C_out; co++) {
            float bias_val = bias_data ? bias_data[co] : 0.0f;
            float *dst = od + ((n * C_out + co) * out_h) * out_w;
            float *src = rd + co * M;
            int64_t m = 0, me = M - (M % AX_VF32_WIDTH);
            ax_vf32 v_b = ax_vf32_set1(bias_val);
            for (; m < me; m += AX_VF32_WIDTH)
                ax_vf32_storeu(dst + m, ax_vf32_add(ax_vf32_loadu(src + m), v_b));
            for (; m < M; m++) dst[m] = src[m] + bias_val;
        }
    }
    } /* end use_batched_cbr_fwd else */

    /* allocate backward saves after pass 1 so a failed alloc still lets us
       return a valid forward result (just without gradients).
       use the forward arena: both tensors are fully overwritten below, so
       ax_tensor_zeros / memset would be wasted bandwidth. the arena frees
       everything in bulk at ax_graph_cleanup, matching the lifetime. */
    ax_tensor_t *x_hat_save = NULL;
    ax_tensor_t *inv_std_save = NULL;
    if (record) {
        ax_arena_t *fa = ax_forward_arena();
        int64_t is_shape[] = {C_out};
        if (fa) {
            x_hat_save  = ax_tensor_arena_create(fa, out_shape, 4, AX_FLOAT32);
            inv_std_save = ax_tensor_arena_create(fa, is_shape, 1, AX_FLOAT32);
        } else {
            x_hat_save  = ax_tensor_create(out_shape, 4, AX_FLOAT32);
            inv_std_save = ax_tensor_create(is_shape, 1, AX_FLOAT32);
        }
        if (!x_hat_save || !inv_std_save) {
            /* non-fatal: forward result is still valid, just no grads */
            x_hat_save = inv_std_save = NULL;
            record = false;
        }
    }

    float *gd = (float *)L->gamma->storage->data;
    float *bnbeta = (float *)L->beta->storage->data;
    float *rm = (float *)L->running_mean->storage->data;
    float *rv = (float *)L->running_var->storage->data;

    if (self->training) {
        float *xh_d = record ? (float *)x_hat_save->storage->data : NULL;
        float *is_d = record ? (float *)inv_std_save->storage->data : NULL;
        float eff_n = (float)(N * spatial);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
        #endif
        for (int64_t c = 0; c < C_out; c++) {
            /* 1-pass fused sum + sum^2: halves memory reads vs 2-pass.
               E[x^2] - E[x]^2 is numerically adequate for float32 at
               typical BN working ranges (activation magnitudes ~0-10). */
            ax_vf32 vs = ax_vf32_zero(), vs2 = ax_vf32_zero();
            float s_tail = 0.0f, s2_tail = 0.0f;
            for (int64_t n = 0; n < N; n++) {
                int64_t base = n * C_out * spatial + c * spatial;
                int64_t m = 0, me = spatial - (spatial % AX_VF32_WIDTH);
                for (; m < me; m += AX_VF32_WIDTH) {
                    ax_vf32 x = ax_vf32_loadu(od + base + m);
                    vs  = ax_vf32_add(vs, x);
                    vs2 = ax_vf32_fmadd(x, x, vs2);
                }
                for (; m < spatial; m++) {
                    float x = od[base + m];
                    s_tail  += x;
                    s2_tail += x * x;
                }
            }
            float sum  = ax_vf32_hsum(vs)  + s_tail;
            float sum2 = ax_vf32_hsum(vs2) + s2_tail;
            float mean = sum / eff_n;
            float var  = sum2 / eff_n - mean * mean;
            if (var < 0.0f) var = 0.0f;
            float var_sum = var * eff_n; /* Σ(x)^2 - N*mean^2, for unbiased estimator */
            float inv_std = 1.0f / sqrtf(var + L->bn_eps);
            if (record) is_d[c] = inv_std;

            /* fused apply: out = max(0, gamma * (conv - mean) * inv_std + beta)
               collapses to: out = max(0, scale*conv + bias_out). */
            float scale = gd[c] * inv_std;
            float bias_out = bnbeta[c] - gd[c] * mean * inv_std;
            ax_vf32 v_sc = ax_vf32_set1(scale);
            ax_vf32 v_bi = ax_vf32_set1(bias_out);
            ax_vf32 v_is = ax_vf32_set1(inv_std);
            ax_vf32 v_mn = ax_vf32_set1(mean);
            ax_vf32 v_zero = ax_vf32_zero();

            for (int64_t n = 0; n < N; n++) {
                int64_t base = n * C_out * spatial + c * spatial;
                if (record) {
                    int64_t m = 0, me = spatial - (spatial % AX_VF32_WIDTH);
                    for (; m < me; m += AX_VF32_WIDTH) {
                        ax_vf32 v_in = ax_vf32_loadu(od + base + m);
                        ax_vf32 xh = ax_vf32_mul(ax_vf32_sub(v_in, v_mn), v_is);
                        ax_vf32 bn = ax_vf32_fmadd(v_sc, v_in, v_bi);
                        ax_vf32_storeu(od + base + m, ax_vf32_max(bn, v_zero));
                        ax_vf32_storeu(xh_d + base + m, xh);
                    }
                    for (; m < spatial; m++) {
                        float conv = od[base + m];
                        float xh = (conv - mean) * inv_std;
                        float bn = scale * conv + bias_out;
                        od[base + m] = bn > 0.0f ? bn : 0.0f;
                        xh_d[base + m] = xh;
                    }
                } else {
                    int64_t m = 0, me = spatial - (spatial % AX_VF32_WIDTH);
                    for (; m < me; m += AX_VF32_WIDTH) {
                        ax_vf32 bn = ax_vf32_fmadd(v_sc, ax_vf32_loadu(od + base + m), v_bi);
                        ax_vf32_storeu(od + base + m, ax_vf32_max(bn, v_zero));
                    }
                    for (; m < spatial; m++) {
                        float bn = scale * od[base + m] + bias_out;
                        od[base + m] = bn > 0.0f ? bn : 0.0f;
                    }
                }
            }

            rm[c] = (1.0f - L->bn_momentum) * rm[c] + L->bn_momentum * mean;
            int64_t eff_count = (int64_t)eff_n;
            float unbiased_var = (eff_count > 1) ? var_sum / (float)(eff_count - 1) : var;
            rv[c] = (1.0f - L->bn_momentum) * rv[c] + L->bn_momentum * unbiased_var;
        }
    } else {
        /* eval path: use running stats, fused scale+shift+relu */
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
        #endif
        for (int64_t c = 0; c < C_out; c++) {
            float inv_std = 1.0f / sqrtf(rv[c] + L->bn_eps);
            float scale = gd[c] * inv_std;
            float bias_out = bnbeta[c] - gd[c] * rm[c] * inv_std;
            ax_vf32 v_sc = ax_vf32_set1(scale);
            ax_vf32 v_bi = ax_vf32_set1(bias_out);
            ax_vf32 v_zero = ax_vf32_zero();
            for (int64_t n = 0; n < N; n++) {
                int64_t base = n * C_out * spatial + c * spatial;
                int64_t m = 0, me = spatial - (spatial % AX_VF32_WIDTH);
                for (; m < me; m += AX_VF32_WIDTH) {
                    ax_vf32 bn = ax_vf32_fmadd(v_sc, ax_vf32_loadu(od + base + m), v_bi);
                    ax_vf32_storeu(od + base + m, ax_vf32_max(bn, v_zero));
                }
                for (; m < spatial; m++) {
                    float bn = scale * od[base + m] + bias_out;
                    od[base + m] = bn > 0.0f ? bn : 0.0f;
                }
            }
        }
    }

    if (record) {
        cbr_bwd_ctx_t *ctx = (cbr_bwd_ctx_t *)malloc(sizeof(cbr_bwd_ctx_t));
        if (!ctx) {
            ax_tensor_destroy(x_hat_save);
            ax_tensor_destroy(inv_std_save);
            if (inp != input) ax_tensor_destroy(inp);
            return output;
        }
        ctx->layer = L;
        ctx->N = N; ctx->C_out = C_out; ctx->out_h = out_h; ctx->out_w = out_w;
        ctx->C_in = C_in; ctx->H = H; ctx->W = W;

        output->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(conv_bn_relu_backward);
        gf->inputs[0] = input;
        gf->n_inputs = 1;
        gf->saved[0] = inp;                      /* contig input for conv bwd */
        gf->saved_owned[0] = (inp != input);
        gf->saved[1] = x_hat_save;
        gf->saved_owned[1] = true;
        gf->saved[2] = inv_std_save;
        gf->saved_owned[2] = true;
        gf->n_saved = 3;
        gf->ctx = ctx;
        gf->ctx_cleanup = cbr_ctx_cleanup;
        output->grad_fn = gf;
    } else {
        if (inp != input) ax_tensor_destroy(inp);
    }
    return output;
}

static void conv_bn_relu_destroy(ax_layer_t *self)
{
    ax_conv_bn_relu_t *L = (ax_conv_bn_relu_t *)self;
    if (L->scratch) ax_conv_scratch_destroy(L->scratch);
    if (L->weight) ax_tensor_destroy(L->weight);
    if (L->bias) ax_tensor_destroy(L->bias);
    if (L->gamma) ax_tensor_destroy(L->gamma);
    if (L->beta) ax_tensor_destroy(L->beta);
    if (L->running_mean) ax_tensor_destroy(L->running_mean);
    if (L->running_var) ax_tensor_destroy(L->running_var);
    free(L);
}

ax_layer_t *ax_conv_bn_relu_create_from(ax_layer_t *conv_base, ax_layer_t *bn_base)
{
    if (!conv_base || !bn_base) return NULL;
    if (conv_base->type != AX_LAYER_CONV2D || bn_base->type != AX_LAYER_BATCHNORM)
        return NULL;

    ax_conv2d_t *conv = (ax_conv2d_t *)conv_base;
    ax_batchnorm_t *bn = (ax_batchnorm_t *)bn_base;

    ax_conv_bn_relu_t *L = (ax_conv_bn_relu_t *)calloc(1, sizeof(ax_conv_bn_relu_t));
    AX_RETURN_NULL_IF_ALLOC_FAIL(L, "ax_conv_bn_relu_create_from");

    L->base.ops.forward = conv_bn_relu_forward;
    L->base.ops.destroy = conv_bn_relu_destroy;
    L->base.type = AX_LAYER_CONV_BN_RELU;
    L->base.training = conv_base->training;
    L->base.input_features = conv_base->input_features;
    L->base.output_features = conv_base->output_features;

    /* steal conv tensors */
    L->weight = conv->weight;      conv->weight = NULL;
    L->bias = conv->bias;          conv->bias = NULL;
    L->scratch = conv->scratch;    conv->scratch = NULL;
    L->in_channels = conv->in_channels;
    L->out_channels = conv->out_channels;
    L->kernel_h = conv->kernel_h;  L->kernel_w = conv->kernel_w;
    L->stride_h = conv->stride_h;  L->stride_w = conv->stride_w;
    L->pad_h = conv->pad_h;        L->pad_w = conv->pad_w;
    L->use_bias = conv->use_bias;

    /* steal batchnorm tensors */
    L->gamma = bn->gamma;                 bn->gamma = NULL;
    L->beta = bn->beta;                   bn->beta = NULL;
    L->running_mean = bn->running_mean;   bn->running_mean = NULL;
    L->running_var = bn->running_var;     bn->running_var = NULL;
    L->bn_eps = bn->eps;
    L->bn_momentum = bn->momentum;

    /* expose params+buffers to ax_layer_get_params/buffers.
       layout: [weight, (bias?), gamma, beta], buffers: [running_mean, running_var] */
    L->base.params[0] = L->weight;
    int np = 1;
    if (L->use_bias && L->bias) L->base.params[np++] = L->bias;
    L->base.params[np++] = L->gamma;
    L->base.params[np++] = L->beta;
    L->base.n_params = np;

    L->base.buffers[0] = L->running_mean;
    L->base.buffers[1] = L->running_var;
    L->base.n_buffers = 2;

    return (ax_layer_t *)L;
}

ax_layer_t *ax_sequential_fuse(ax_layer_t *model)
{
    if (!model || model->type != AX_LAYER_SEQUENTIAL) return model;
    ax_sequential_t *seq = (ax_sequential_t *)model;

    int w = 0;
    int r = 0;
    while (r < seq->n_layers) {
        ax_layer_t *a = seq->layers[r];
        if (r + 2 < seq->n_layers &&
            a->type == AX_LAYER_CONV2D &&
            seq->layers[r+1]->type == AX_LAYER_BATCHNORM &&
            seq->layers[r+2]->type == AX_LAYER_RELU) {

            ax_layer_t *b = seq->layers[r+1];
            ax_layer_t *c = seq->layers[r+2];

            ax_layer_t *fused = ax_conv_bn_relu_create_from(a, b);
            if (!fused) {
                seq->layers[w++] = a;
                r++;
                continue;
            }

            /* free old layer shells. tensors were moved into the fused
               layer so the original destroy paths will see NULL fields. */
            a->ops.destroy(a);
            b->ops.destroy(b);
            c->ops.destroy(c);

            seq->layers[w++] = fused;
            r += 3;
        } else {
            seq->layers[w++] = a;
            r++;
        }
    }
    seq->n_layers = w;
    return model;
}
