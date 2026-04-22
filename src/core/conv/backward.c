/* conv/backward.c — autograd hook for conv2d (NCHW + NHWC paths).

   k.2 split: extracted from src/core/conv.c. backward is a single
   monolithic ax_grad_fn_t callback that handles both layouts and
   the {dW, db, dX} grad set the autograd graph requested. it shares
   most infrastructure with forward — the per-thread scratch struct,
   the im2col+gemm pipeline, the 1x1 zero-copy fast path — but the
   compute pattern (gemm_nt for dW, gemm_tn for dX, im2col for both
   in matrix form) and the per-thread reduction at the end make it
   distinct enough to live in its own tu.

   parallelism strategy:
   * dW: per-thread accumulator buffers + final reduction. each
     thread's work is a disjoint sample chunk; the reduction is a
     parallel-for over the weight gradient buffer.
   * dX: each sample writes a disjoint slice of input_grad, so the
     per-sample pass needs no atomics — straight parallel-for.
   * batched fast path (M < 512, N > 1): merges all samples into
     one wide GEMM per stage and uses skip_init to accumulate dW
     across chunks rather than per-sample gemm. wins on shapes
     where the per-call GEMM overhead dominates compute. */

#include "internal.h"
#include "axiom/compute.h"
#include "axiom/autograd.h"
#include "axiom/tensor.h"
#include "axiom/error.h"
#include "../../compute/backends/simd_defs.h"
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#define AX_OMP_MAX_THREADS() omp_get_max_threads()
#define AX_OMP_THREAD_NUM() omp_get_thread_num()
#else
#define AX_OMP_MAX_THREADS() 1
#define AX_OMP_THREAD_NUM() 0
#endif


void ax_conv_conv2d_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_conv2d_t *conv = (ax_conv2d_t *)self->ctx;
    ax_tensor_t *input_orig = self->inputs[0]; /* original — for grad accumulation */
    ax_tensor_t *input_data = self->saved[0];  /* contiguous — for data access */
    ax_tensor_t *weight = conv->weight;

    ax_layout_t layout = ax_tensor_get_layout(input_data);
    int64_t N, C_in, H, W;
    if (layout == AX_LAYOUT_NHWC) {
        N = input_data->shape[0]; H = input_data->shape[1]; W = input_data->shape[2]; C_in = input_data->shape[3];
    } else {
        N = input_data->shape[0]; C_in = input_data->shape[1]; H = input_data->shape[2]; W = input_data->shape[3];
    }
    int64_t C_out = weight->shape[0];
    int kh = conv->kernel_h, kw = conv->kernel_w;
    int sh = conv->stride_h, sw = conv->stride_w;
    int ph = conv->pad_h, pw = conv->pad_w;
    int64_t out_h = ax_conv_out_dim(H, kh, sh, ph);
    int64_t out_w = ax_conv_out_dim(W, kw, sw, pw);

    /* grad_out matches input layout: NCHW [N,C_out,oh,ow] or NHWC [N,oh,ow,C_out] */

    /* weight gradient and input gradient */
    if (weight->requires_grad)
    {
        if (!weight->grad)
            weight->grad = ax_tensor_zeros(weight->shape, weight->ndim, weight->dtype);
    }
    if (input_orig->requires_grad)
    {
        if (!input_orig->grad) {
            input_orig->grad = ax_tensor_zeros(input_orig->shape, input_orig->ndim, input_orig->dtype);
            if (input_orig->grad) input_orig->grad->layout = layout;
        }
    }

    /* NHWC dispatch: do all three grads (dW, db, dX) in one helper that uses
       the NHWC im2col + GEMM pipeline. */
    if (layout == AX_LAYOUT_NHWC) {
        struct ax_conv_scratch *s = ax_conv_ensure_scratch(conv, N, H, W, true);
        if (!s) return;

        float *dwd = (weight->requires_grad && weight->grad)
                     ? (float *)weight->grad->storage->data : NULL;
        float *dbd = NULL;
        if (conv->use_bias && conv->bias && conv->bias->requires_grad) {
            if (!conv->bias->grad)
                conv->bias->grad = ax_tensor_zeros(conv->bias->shape, conv->bias->ndim, conv->bias->dtype);
            if (conv->bias->grad) dbd = (float *)conv->bias->grad->storage->data;
        }
        float *dxd = (input_orig->requires_grad && input_orig->grad)
                     ? (float *)input_orig->grad->storage->data : NULL;

        conv2d_nhwc_backward_impl(s,
            (const float *)input_data->storage->data,
            (const float *)grad_out->storage->data,
            (const float *)weight->storage->data,
            dwd, dbd, dxd,
            N, C_in, H, W, C_out, out_h, out_w,
            kh, kw, sh, sw, ph, pw,
            weight->storage->generation);

        if (dwd) ax_storage_touch(weight->grad->storage);
        if (dbd) ax_storage_touch(conv->bias->grad->storage);
        if (dxd) ax_storage_touch(input_orig->grad->storage);
        return;
    }

    /* bias gradient: sum grad_out over N, H, W.
       parallelize over channels: each channel's M*N sum is independent. */
    if (conv->use_bias && conv->bias && conv->bias->requires_grad)
    {
        if (!conv->bias->grad)
            conv->bias->grad = ax_tensor_zeros(conv->bias->shape, conv->bias->ndim, conv->bias->dtype);

        float *bg = (float *)conv->bias->grad->storage->data;
        const float *gd = (const float *)grad_out->storage->data;
        int64_t M_bias = out_h * out_w;

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t c = 0; c < C_out; c++) {
            float sum = 0.0f;
            for (int64_t n = 0; n < N; n++) {
                const float *row = gd + (n * C_out + c) * M_bias;
                int64_t m = 0, ve = M_bias - (M_bias % AX_VF32_WIDTH);
                ax_vf32 acc = ax_vf32_zero();
                for (; m < ve; m += AX_VF32_WIDTH)
                    acc = ax_vf32_add(acc, ax_vf32_loadu(row + m));
                /* horizontal sum of acc */
                float tmp[AX_VF32_WIDTH];
                ax_vf32_storeu(tmp, acc);
                for (int k = 0; k < AX_VF32_WIDTH; k++) sum += tmp[k];
                for (; m < M_bias; m++) sum += row[m];
            }
            bg[c] += sum; /* one thread per c, no race */
        }
    }

    float *wdata = (float *)weight->storage->data;
    int64_t K = C_in * kh * kw;
    int64_t M = out_h * out_w;

    /* use cached scratch from layer struct (allocated in forward via ax_conv_ensure_scratch).
       if scratch is missing for some reason, allocate fresh as fallback. */
    struct ax_conv_scratch *s = conv->scratch;
    if (!s || s->N != N || s->H != H || s->W != W || !s->dw_bufs) {
        /* scratch missing or stale — call ax_conv_ensure_scratch to (re)build */
        s = ax_conv_ensure_scratch(conv, N, H, W, true);
        if (!s) return;
    }

    int T = s->T;
    ax_tensor_t *wt_contig = s->wt_contig;
    /* when gemm_tn is available we skip the physical weight transpose —
       gemm_tn walks w2d as if transposed. keep the legacy refresh only
       for the fallback path so backends without gemm_tn still work. */
    bool have_gemm_tn = ax_compute_has_gemm_tn();
    bool have_gemm_nt = ax_compute_has_gemm_nt();
    if (input_orig->requires_grad && wt_contig && !have_gemm_tn) {
        float *wtd = (float *)wt_contig->storage->data;
        for (int64_t r = 0; r < C_out; r++)
            for (int64_t c = 0; c < K; c++)
                wtd[c * C_out + r] = wdata[r * K + c];
    }
    /* w2d for gemm_tn fast path: fresh copy of current weights in
       [C_out, K] shape. s->w2d is refreshed by forward, but forward
       may have been called long enough ago that weights changed; do
       it here too for safety. */
    ax_tensor_t *w2d = s->w2d;
    if (input_orig->requires_grad && have_gemm_tn && w2d) {
        memcpy(w2d->storage->data, wdata, (size_t)(C_out * K) * sizeof(float));
    }

    /* zero per-thread dW buffers (we accumulate into them across samples) */
    if (weight->requires_grad) {
        for (int t = 0; t < T; t++)
            memset(s->dw_bufs[t]->storage->data, 0, (size_t)(C_out * K) * sizeof(float));
    }

    float *ind = (float *)input_data->storage->data;
    float *grd = (float *)grad_out->storage->data;

    /* same 1×1 carve-out as forward: per-sample backward will use the 1×1
       zero-copy fast path (skip im2col, accumulate dW/dX with skip_init). */
    bool is_1x1_fast_bwd = ax_conv_is_1x1_pad0_stride1(kh, kw, sh, sw, ph, pw)
                           && (H == out_h) && (W == out_w);
    bool use_batched_bwd = N > 1 && M < AX_CONV_BATCH_M_THRESH && !is_1x1_fast_bwd
                           && s->batch_col_buf && s->batch_aux_buf;

    if (use_batched_bwd) {
        /* sub-batched backward: loop over n_batch-sample chunks.
           dW accumulates across chunks via skip_init; dX processes per chunk. */
        float *bcd = (float *)s->batch_col_buf->storage->data;
        float *gbd = (float *)s->batch_aux_buf->storage->data;
        int64_t nb = s->n_batch;

        ax_tensor_t *dw_total = s->dw_bufs[0];
        /* dw_total is already zeroed at line 684-688 above; chunks accumulate into it. */

        for (int64_t n_start = 0; n_start < N; n_start += nb) {
            int64_t n_b   = (n_start + nb <= N) ? nb : (N - n_start);
            int64_t NM_b  = n_b * M;
            int threads_b = (int)(n_b < T ? n_b : T);

            /* im2col → bcd [K, NM_b] and scatter go → gbd [C_out, NM_b] */
            #ifdef _OPENMP
            #pragma omp parallel for num_threads(threads_b) schedule(static)
            #endif
            for (int64_t n = 0; n < n_b; n++) {
                ax_conv_im2col_into_strided(ind + (n_start + n) * C_in * H * W, C_in, H, W,
                                     kh, kw, sh, sw, ph, pw, out_h, out_w,
                                     bcd, NM_b, n * M);
                for (int64_t co = 0; co < C_out; co++)
                    memcpy(gbd + co * NM_b + n * M,
                           grd + (n_start + n) * C_out * M + co * M,
                           (size_t)M * sizeof(float));
            }

            /* stack views for this chunk */
            ax_storage_t col_st, aux_st;
            ax_tensor_t  col_tv, aux_tv;
            ax_conv_make_stack_view(&col_tv, &col_st, bcd, K,     NM_b);
            ax_conv_make_stack_view(&aux_tv, &aux_st, gbd, C_out, NM_b);

            /* dW += go_chunk @ col_chunk^T: use skip_init to accumulate across chunks
               without zeroing dw_total between iterations. */
            if (weight->requires_grad) {
                ax_gemm_set_skip_init(true);
                ax_compute_gemm_nt(&aux_tv, &col_tv, dw_total);
                ax_gemm_set_skip_init(false);
            }

            /* dcol_chunk = w^T @ go_chunk: reuse bcd (im2col already consumed).
               opt_gemm_tn zeros bcd internally; explicit memset not needed. */
            if (input_orig->requires_grad && (wt_contig || have_gemm_tn)) {
                if (have_gemm_tn)
                    ax_compute_gemm_tn(w2d, &aux_tv, &col_tv);
                else {
                    /* wt_contig path: need a [K, NM_b] view for output */
                    ax_compute_gemm(wt_contig, &aux_tv, &col_tv);
                }

                /* parallel col2im per sample from dcol_chunk [K, NM_b].
                   threads_b = min(n_b, T) avoids spawning idle workers. */
                #ifdef _OPENMP
                #pragma omp parallel for num_threads(threads_b) schedule(static)
                #endif
                for (int64_t n = 0; n < n_b; n++) {
                    int tid = AX_OMP_THREAD_NUM();
                    if (tid >= T) tid = 0;
                    ax_tensor_t *dcol_buf = s->dcol_bufs[tid];
                    ax_tensor_t *dimg_buf = s->dimg_bufs[tid];
                    float *dcol_d = (float *)dcol_buf->storage->data;
                    /* gather dcol for sample n: dcol[k, m] = bcd[k*NM_b + n*M + m] */
                    for (int64_t k = 0; k < K; k++)
                        memcpy(dcol_d + k * M, bcd + k * NM_b + n * M, (size_t)M * sizeof(float));
                    float *dimg_d = (float *)dimg_buf->storage->data;
                    memset(dimg_d, 0, (size_t)(C_in * H * W) * sizeof(float));
                    ax_conv_col2im_into(dcol_d, C_in, H, W, kh, kw, sh, sw, ph, pw, out_h, out_w, dimg_d);
                    float *ig = (float *)input_orig->grad->storage->data + (n_start + n) * C_in * H * W;
                    int64_t total = C_in * H * W, i = 0, ve = total - (total % AX_VF32_WIDTH);
                    for (; i < ve; i += AX_VF32_WIDTH)
                        ax_vf32_storeu(ig + i, ax_vf32_add(ax_vf32_loadu(ig + i), ax_vf32_loadu(dimg_d + i)));
                    for (; i < total; i++) ig[i] += dimg_d[i];
                }
            }
        }

        /* accumulate dw_total into weight->grad */
        if (weight->requires_grad) {
            float *wg  = (float *)weight->grad->storage->data;
            float *dwl = (float *)dw_total->storage->data;
            int64_t wn = C_out * K, wi = 0, wve = wn - (wn % AX_VF32_WIDTH);
            for (; wi < wve; wi += AX_VF32_WIDTH)
                ax_vf32_storeu(wg + wi, ax_vf32_add(ax_vf32_loadu(wg + wi), ax_vf32_loadu(dwl + wi)));
            for (; wi < wn; wi++) wg[wi] += dwl[wi];
        }
    } else {
    /* num_threads(T) prevents oversubscription of per-thread scratch slots
       (col_bufs, go_bufs, dw_bufs, etc. — all sized to T). */
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(T) schedule(dynamic, 1)
    #endif
    for (int64_t n = 0; n < N; n++)
    {
        int tid = AX_OMP_THREAD_NUM();
        if (tid >= T) tid = 0;

        /* 1×1 stride=1 pad=0 zero-copy backward: skip im2col, go memcpy,
           dw_sample temp + add, and dcol_buf + col2im + add. dW accumulates
           directly into dw_bufs[tid] via skip_init; dX accumulates directly
           into input_grad slice via skip_init. eliminates ~4 memory
           operations per sample (significant for 1×1-heavy networks like
           ResNet bottlenecks). */
        if (is_1x1_fast_bwd) {
            ax_storage_t in_st, go_st;
            ax_tensor_t  in_tv, go_tv;
            ax_conv_make_stack_view(&in_tv, &in_st, ind + n * C_in * H * W, C_in, M);
            ax_conv_make_stack_view(&go_tv, &go_st, grd + n * C_out * M, C_out, M);

            if (weight->requires_grad) {
                /* dW += go @ in^T: gemm_nt with skip_init accumulates into
                   dw_bufs[tid] (per-thread, sequential within a thread). */
                ax_tensor_t *dw_local = s->dw_bufs[tid];
                ax_gemm_set_skip_init(true);
                ax_compute_gemm_nt(&go_tv, &in_tv, dw_local);
                ax_gemm_set_skip_init(false);
            }

            if (input_orig->requires_grad && (wt_contig || have_gemm_tn)) {
                /* dX += w^T @ go: gemm_tn with skip_init accumulates into the
                   input_grad slice for sample n (disjoint across samples). */
                ax_storage_t ig_st;
                ax_tensor_t  ig_tv;
                float *ig_slice = (float *)input_orig->grad->storage->data
                                + n * C_in * H * W;
                ax_conv_make_stack_view(&ig_tv, &ig_st, ig_slice, C_in, M);
                ax_gemm_set_skip_init(true);
                if (have_gemm_tn)
                    ax_compute_gemm_tn(w2d, &go_tv, &ig_tv);
                else
                    ax_compute_gemm(wt_contig, &go_tv, &ig_tv);
                ax_gemm_set_skip_init(false);
            }
            continue;
        }

        float *cbd = (float *)s->col_bufs[tid]->storage->data;
        ax_tensor_t *col_buf = s->col_bufs[tid];
        ax_tensor_t *go_mat = s->go_bufs[tid];

        /* im2col into per-thread buffer */
        ax_conv_im2col_into(ind + n * C_in * H * W, C_in, H, W,
                     kh, kw, sh, sw, ph, pw, out_h, out_w, cbd);

        /* fill per-thread grad_out matrix */
        memcpy(go_mat->storage->data, grd + n * C_out * M,
               (size_t)(C_out * M) * sizeof(float));

        /* weight gradient: per-thread dW accumulates across this thread's samples */
        if (weight->requires_grad)
        {
            ax_tensor_t *dw_local = s->dw_bufs[tid];
            ax_tensor_t *dw_sample = s->dws_bufs[tid];
            float *dws = (float *)dw_sample->storage->data;
            memset(dws, 0, (size_t)(C_out * K) * sizeof(float));

            if (have_gemm_nt) {
                /* dw = go @ col^T via gemm_nt — skips the 60-line block
                   cache-blocked transpose of col that the legacy path did. */
                ax_compute_gemm_nt(go_mat, col_buf, dw_sample);
            } else {
                ax_tensor_t *colt_buf = s->colt_bufs[tid];
                float *ctd = (float *)colt_buf->storage->data;
                const int64_t BT = 32;
                for (int64_t r0 = 0; r0 < K; r0 += BT) {
                    int64_t r_end = (r0 + BT < K) ? r0 + BT : K;
                    for (int64_t c0 = 0; c0 < M; c0 += BT) {
                        int64_t c_end = (c0 + BT < M) ? c0 + BT : M;
                        for (int64_t r = r0; r < r_end; r++) {
                            const float *src_row = cbd + r * M;
                            int64_t c = c0;
                            for (; c + 4 <= c_end; c += 4) {
                                ctd[(c + 0) * K + r] = src_row[c + 0];
                                ctd[(c + 1) * K + r] = src_row[c + 1];
                                ctd[(c + 2) * K + r] = src_row[c + 2];
                                ctd[(c + 3) * K + r] = src_row[c + 3];
                            }
                            for (; c < c_end; c++)
                                ctd[c * K + r] = src_row[c];
                        }
                    }
                }
                ax_compute_gemm(go_mat, colt_buf, dw_sample);
            }

            float *dwl = (float *)dw_local->storage->data;
            int64_t wn = C_out * K;
            int64_t wi = 0, wve = wn - (wn % AX_VF32_WIDTH);
            for (; wi < wve; wi += AX_VF32_WIDTH)
                ax_vf32_storeu(dwl + wi, ax_vf32_add(ax_vf32_loadu(dwl + wi), ax_vf32_loadu(dws + wi)));
            for (; wi < wn; wi++) dwl[wi] += dws[wi];
        }

        /* input gradient — disjoint per-sample writes, no reduction needed */
        if (input_orig->requires_grad && (wt_contig || have_gemm_tn))
        {
            ax_tensor_t *dcol_buf = s->dcol_bufs[tid];
            ax_tensor_t *dimg_buf = s->dimg_bufs[tid];

            memset(dcol_buf->storage->data, 0, (size_t)(K * M) * sizeof(float));
            if (have_gemm_tn) {
                /* dcol = w^T @ go via gemm_tn — skips the wt_contig scalar
                   transpose refresh done above in the legacy path. */
                ax_compute_gemm_tn(w2d, go_mat, dcol_buf);
            } else {
                ax_compute_gemm(wt_contig, go_mat, dcol_buf);
            }

            float *dimg_d = (float *)dimg_buf->storage->data;
            memset(dimg_d, 0, (size_t)(C_in * H * W) * sizeof(float));
            ax_conv_col2im_into((float *)dcol_buf->storage->data, C_in, H, W,
                         kh, kw, sh, sw, ph, pw, out_h, out_w, dimg_d);

            /* SIMD accumulate into input gradient (disjoint per n) */
            float *ig = (float *)input_orig->grad->storage->data + n * C_in * H * W;
            int64_t total = C_in * H * W;
            int64_t i = 0, ve = total - (total % AX_VF32_WIDTH);
            for (; i < ve; i += AX_VF32_WIDTH)
                ax_vf32_storeu(ig + i, ax_vf32_add(ax_vf32_loadu(ig + i), ax_vf32_loadu(dimg_d + i)));
            for (; i < total; i++)
                ig[i] += dimg_d[i];
        }
    }

    /* parallel reduction: each thread owns a slice of [0, wn) and sums all T
       per-thread dW buffers into weight->grad for that slice. wins for large
       convs (e.g. 512×4608 = 9.4 MB per buffer × T=16 = 150 MB data to
       reduce); the original serial loop streamed all of this through one
       core's L1. */
    if (weight->requires_grad) {
        float *wg = (float *)weight->grad->storage->data;
        int64_t wn = C_out * K;
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(T) schedule(static)
        #endif
        for (int64_t wi0 = 0; wi0 < wn; wi0 += AX_VF32_WIDTH) {
            int64_t wend = (wi0 + AX_VF32_WIDTH <= wn) ? wi0 + AX_VF32_WIDTH : wn;
            if (wend - wi0 == AX_VF32_WIDTH) {
                ax_vf32 acc = ax_vf32_loadu(wg + wi0);
                for (int t = 0; t < T; t++) {
                    float *dwl = (float *)s->dw_bufs[t]->storage->data;
                    acc = ax_vf32_add(acc, ax_vf32_loadu(dwl + wi0));
                }
                ax_vf32_storeu(wg + wi0, acc);
            } else {
                for (int64_t wi = wi0; wi < wend; wi++) {
                    float sum = wg[wi];
                    for (int t = 0; t < T; t++)
                        sum += ((float *)s->dw_bufs[t]->storage->data)[wi];
                    wg[wi] = sum;
                }
            }
        }
    }
    } /* end use_batched_bwd else */
    /* note: scratch buffers are kept on the layer for reuse — no free here */
}
/* NHWC conv2d backward: produces dW, db, dX as needed. chunked per-sample
   to keep im2col_nhwc in L3 (matches forward's chunk strategy). */
void conv2d_nhwc_backward_impl(
    struct ax_conv_scratch *s,
    const float *id,         /* input data [N, H, W, Cin] */
    const float *grd,        /* grad_out   [N, OH, OW, Cout] */
    const float *wd,         /* weight     [Cout, Cin, kh, kw] */
    float *dwd,              /* weight->grad [Cout, Cin, kh, kw], accumulating */
    float *dbd,              /* bias->grad   [Cout], accumulating, may be NULL */
    float *dxd,              /* input->grad [N, H, W, Cin], accumulating, may be NULL */
    int64_t N, int64_t Cin, int64_t H, int64_t W,
    int64_t Cout, int64_t out_h, int64_t out_w,
    int kh, int kw, int sh, int sw, int ph, int pw,
    uint64_t weight_gen)
{
    int64_t K = (int64_t)Cin * (int64_t)kh * (int64_t)kw;
    int64_t M_per = out_h * out_w;
    int64_t HWC = H * W * Cin;
    int64_t OHOW_C = out_h * out_w * Cout;

    int64_t bytes_per_sample = M_per * K * (int64_t)sizeof(float);
    int64_t n_chunk = (int64_t)(8 * 1024 * 1024) / (bytes_per_sample > 0 ? bytes_per_sample : 1);
    if (n_chunk < 1) n_chunk = 1;
    if (n_chunk > N) n_chunk = N;
    int64_t chunk_M = n_chunk * M_per;

    bool need_alloc = !s->im2col_nhwc
        || s->im2col_nhwc->shape[0] != chunk_M
        || s->im2col_nhwc->shape[1] != K;
    if (need_alloc) {
        if (s->im2col_nhwc) ax_tensor_destroy(s->im2col_nhwc);
        int64_t sh2[] = {chunk_M, K};
        s->im2col_nhwc = ax_tensor_create(sh2, 2, AX_FLOAT32);
        if (!s->im2col_nhwc) return;
    }
    float *cd = (float *)s->im2col_nhwc->storage->data;

    /* dW accumulator [Cout, K] — single allocation across all chunks. */
    size_t dW_bytes = (size_t)Cout * (size_t)K * sizeof(float);
    float *dW_temp = NULL;
    if (dwd) {
        dW_temp = (float *)calloc(1, dW_bytes);
        if (!dW_temp) return;
    }

    if (dxd) {
        ax_conv_refresh_wt_nhwc(s, wd, Cout, Cin, kh, kw, weight_gen);
        if (!s->wt_nhwc) { free(dW_temp); return; }
    }

    int khkw = kh * kw;
    /* ax_gemm_set_skip_init declared in internal.h */

    for (int64_t n_start = 0; n_start < N; n_start += n_chunk) {
        int64_t n_b = (n_start + n_chunk <= N) ? n_chunk : (N - n_start);
        int64_t M_b = n_b * M_per;

        /* shared per-chunk im2col (used by dW and dX). */
        if (dwd || dxd)
            ax_conv_im2col_nhwc(id + n_start * HWC, n_b, H, W, Cin,
                        kh, kw, sh, sw, ph, pw, out_h, out_w, cd);

        if (dwd) {
            /* dW [Cout, K] += grd_chunk^T @ col_chunk via gemm_tn(grd, col).
               accumulate across chunks via skip_init. */
            ax_storage_t a_st, b_st, c_st;
            ax_tensor_t  a_tv, b_tv, c_tv;
            ax_conv_make_stack_view(&a_tv, &a_st, (float *)(grd + n_start * OHOW_C), M_b, Cout);
            ax_conv_make_stack_view(&b_tv, &b_st, cd,                                  M_b, K);
            ax_conv_make_stack_view(&c_tv, &c_st, dW_temp,                             Cout, K);
            ax_gemm_set_skip_init(true);  /* accumulate into dW_temp across chunks */
            if (ax_compute_gemm_tn(&a_tv, &b_tv, &c_tv) != AX_OK) {
                ax_gemm_set_skip_init(false);
                free(dW_temp); return;
            }
            ax_gemm_set_skip_init(false);
        }

        if (dxd) {
            /* dX_chunk = grd_chunk @ wt_nhwc^T → dcol [M_b, K] (overwrites cd).
               then col2im_nhwc into dxd at sample offset. */
            ax_storage_t a_st, b_st, c_st;
            ax_tensor_t  a_tv, b_tv, c_tv;
            ax_conv_make_stack_view(&a_tv, &a_st, (float *)(grd + n_start * OHOW_C), M_b, Cout);
            ax_conv_make_stack_view(&b_tv, &b_st, (float *)s->wt_nhwc->storage->data, K, Cout);
            ax_conv_make_stack_view(&c_tv, &c_st, cd, M_b, K);
            if (ax_compute_gemm_nt(&a_tv, &b_tv, &c_tv) != AX_OK) {
                free(dW_temp); return;
            }
            ax_conv_col2im_nhwc(cd, n_b, H, W, Cin, kh, kw, sh, sw, ph, pw, out_h, out_w,
                        dxd + n_start * HWC);
        }
    }

    /* dW_temp [Cout, K] in [Cout, kh*kw, Cin] order → dwd [Cout, Cin, kh*kw] */
    if (dwd) {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t co = 0; co < Cout; co++) {
            for (int64_t ci = 0; ci < Cin; ci++) {
                for (int kk = 0; kk < khkw; kk++) {
                    int64_t src = co * K + kk * Cin + ci;
                    int64_t dst = (co * Cin + ci) * khkw + kk;
                    dwd[dst] += dW_temp[src];
                }
            }
        }
        free(dW_temp);
    }

    /* db: column-sum of grd over all rows. */
    if (dbd) {
        int64_t M_total = N * M_per;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t co = 0; co < Cout; co++) {
            float sum = 0.0f;
            for (int64_t m = 0; m < M_total; m++) sum += grd[m * Cout + co];
            dbd[co] += sum;
        }
    }
}
