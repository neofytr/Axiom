/* norm.c — batchnorm, layernorm, dropout, gradient clipping */

#include "axiom/norm.h"
#include "axiom/ops.h"
#include "axiom/autograd.h"
#include "axiom/compute.h"
#include "axiom/init.h"
#include "axiom/error.h"
#include "axiom/rng.h"
#include "../compute/backends/simd_defs.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif


/* context saved during batchnorm forward for the backward pass */
typedef struct {
    ax_tensor_t *x_hat;    /* normalized input [batch, feat] */
    ax_tensor_t *inv_std;  /* 1/sqrt(var+eps) per feature [feat] */
    ax_batchnorm_t *bn;    /* layer pointer for gamma/beta access */
} bn_backward_ctx_t;

static void batchnorm_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    bn_backward_ctx_t *ctx = (bn_backward_ctx_t *)self->ctx;
    ax_tensor_t *input = self->inputs[0];
    ax_tensor_t *x_hat_t = ctx->x_hat;
    ax_tensor_t *inv_std_t = ctx->inv_std;
    ax_batchnorm_t *bn = ctx->bn;

    int ndim = grad_out->ndim;
    int64_t batch = grad_out->shape[0];
    int64_t feat = grad_out->shape[1];
    int64_t H = (ndim == 4) ? grad_out->shape[2] : 1;
    int64_t W = (ndim == 4) ? grad_out->shape[3] : 1;
    int64_t spatial = H * W;
    float N = (float)(batch * spatial); /* effective sample count per channel */

    float *go = (float *)grad_out->storage->data;
    float *xh = (float *)x_hat_t->storage->data;
    float *istd = (float *)inv_std_t->storage->data;
    float *gd = (float *)bn->gamma->storage->data;

    /* fused dgamma + dbeta + dx with SIMD.
       pass 1: compute sum_go, sum_go_xh, dgamma, dbeta per channel
       pass 2: apply dx formula per element */

    float *dg = NULL, *db = NULL, *ig = NULL;
    if (bn->gamma->requires_grad) {
        if (!bn->gamma->grad)
            bn->gamma->grad = ax_tensor_zeros(bn->gamma->shape, bn->gamma->ndim, bn->gamma->dtype);
        if (!bn->gamma->grad) goto cleanup;
        dg = (float *)bn->gamma->grad->storage->data;
    }
    if (bn->beta->requires_grad) {
        if (!bn->beta->grad)
            bn->beta->grad = ax_tensor_zeros(bn->beta->shape, bn->beta->ndim, bn->beta->dtype);
        if (!bn->beta->grad) goto cleanup;
        db = (float *)bn->beta->grad->storage->data;
    }
    if (input->requires_grad) {
        if (!input->grad)
            input->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
        if (!input->grad) goto cleanup;
        ig = (float *)input->grad->storage->data;
    }

    #ifdef _OPENMP
    /* dynamic-1 for hybrid P/E core load balancing: fast cores steal from slow ones */
    #pragma omp parallel for schedule(dynamic, 1)
    #endif
    for (int64_t c = 0; c < feat; c++) {
        /* pass 1: SIMD reduction for sum_go, sum_go_xh (also accumulates dgamma, dbeta) */
        ax_vf32 v_sgo = ax_vf32_zero();
        ax_vf32 v_sgx = ax_vf32_zero();
        float s_sgo = 0, s_sgx = 0;

        for (int64_t n = 0; n < batch; n++) {
            int64_t base = n * feat * spatial + c * spatial;
            int64_t s = 0, se = spatial - (spatial % AX_VF32_WIDTH);
            for (; s < se; s += AX_VF32_WIDTH) {
                ax_vf32 g = ax_vf32_loadu(go + base + s);
                ax_vf32 x = ax_vf32_loadu(xh + base + s);
                v_sgo = ax_vf32_add(v_sgo, g);
                v_sgx = ax_vf32_fmadd(g, x, v_sgx);
            }
            for (; s < spatial; s++) {
                float g = go[base + s];
                s_sgo += g;
                s_sgx += g * xh[base + s];
            }
        }
        float sum_go    = ax_vf32_hsum(v_sgo) + s_sgo;
        float sum_go_xh = ax_vf32_hsum(v_sgx) + s_sgx;

        if (dg) dg[c] += sum_go_xh;
        if (db) db[c] += sum_go;

        /* pass 2: dx = (1/N) * gamma * inv_std * (N*go - sum_go - xh*sum_go_xh) */
        if (ig) {
            float coeff = gd[c] * istd[c] / N;
            ax_vf32 v_coeff  = ax_vf32_set1(coeff);
            ax_vf32 v_N      = ax_vf32_set1(N);
            ax_vf32 v_sum_go = ax_vf32_set1(sum_go);
            ax_vf32 v_sum_gx = ax_vf32_set1(sum_go_xh);

            for (int64_t n = 0; n < batch; n++) {
                int64_t base = n * feat * spatial + c * spatial;
                int64_t s = 0, se = spatial - (spatial % AX_VF32_WIDTH);
                for (; s < se; s += AX_VF32_WIDTH) {
                    ax_vf32 g = ax_vf32_loadu(go + base + s);
                    ax_vf32 x = ax_vf32_loadu(xh + base + s);
                    /* dx = coeff * (N*g - sum_go - x*sum_go_xh) */
                    ax_vf32 dx = ax_vf32_mul(v_coeff,
                        ax_vf32_sub(ax_vf32_sub(ax_vf32_mul(v_N, g), v_sum_go),
                                    ax_vf32_mul(x, v_sum_gx)));
                    ax_vf32_storeu(ig + base + s,
                        ax_vf32_add(ax_vf32_loadu(ig + base + s), dx));
                }
                for (; s < spatial; s++) {
                    float dx = coeff * (N * go[base + s] - sum_go - xh[base + s] * sum_go_xh);
                    ig[base + s] += dx;
                }
            }
        }
    }

cleanup:
    ax_tensor_destroy(ctx->x_hat);
    ax_tensor_destroy(ctx->inv_std);
    free(ctx);
    self->ctx = NULL;
}

static void bn_ctx_cleanup(void *p)
{
    bn_backward_ctx_t *ctx = (bn_backward_ctx_t *)p;
    if (ctx->x_hat) ax_tensor_destroy(ctx->x_hat);
    if (ctx->inv_std) ax_tensor_destroy(ctx->inv_std);
    free(ctx);
}

static ax_tensor_t *batchnorm_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_batchnorm_t *bn = (ax_batchnorm_t *)self;
    if (!input || (input->ndim != 2 && input->ndim != 4)) return NULL;

    /* ensure contiguous so flat indexing below is correct */
    ax_tensor_t *inp = ax_ensure_contiguous(input);
    if (!inp) return NULL;

    int64_t batch = inp->shape[0];
    int64_t feat = inp->shape[1];
    int64_t H = (inp->ndim == 4) ? inp->shape[2] : 1;
    int64_t W = (inp->ndim == 4) ? inp->shape[3] : 1;
    int64_t spatial = H * W;
    /* float eff_batch removed — unused */
    float *id = (float *)inp->storage->data;

    /* uninitialized output — every element is written by the normalize
       loop below, so the memset in ax_tensor_zeros is wasted bandwidth.
       BN on a [256, 4096] tensor is 4 MB of pointless zeroing per call. */
    ax_tensor_t *out = ax_tensor_create(inp->shape, inp->ndim, inp->dtype);
    if (!out) { if (inp != input) ax_tensor_destroy(inp); return NULL; }
    float *od = (float *)out->storage->data;
    float *gd = (float *)bn->gamma->storage->data;
    float *bd = (float *)bn->beta->storage->data;
    float *rm = (float *)bn->running_mean->storage->data;
    float *rv = (float *)bn->running_var->storage->data;

    /* tensors to save for backward (only allocated if needed) */
    ax_tensor_t *x_hat_save = NULL;
    ax_tensor_t *inv_std_save = NULL;
    bool record = ax_grad_enabled() && self->training &&
                  (input->requires_grad || bn->gamma->requires_grad || bn->beta->requires_grad);

    if (record) {
        /* allocate from the forward arena — these live until ax_graph_cleanup
           resets the arena after backward completes. uninitialized: both are
           fully overwritten by the normalize loop below. */
        ax_arena_t *fa = ax_forward_arena();
        x_hat_save = ax_tensor_arena_create(fa, input->shape, input->ndim, input->dtype);
        int64_t is_shape[] = {feat};
        inv_std_save = ax_tensor_arena_create(fa, is_shape, 1, input->dtype);
        if (!x_hat_save || !inv_std_save) {
            if (x_hat_save) ax_tensor_destroy(x_hat_save);
            if (inv_std_save) ax_tensor_destroy(inv_std_save);
            record = false;
        }
    }

    if (inp->ndim == 2)
    {
        /* 2D BN fast path: shape [N, F], spatial=1.
           the generic channel-outer loop below degenerates to strided loads
           (one float per cache line) because spatial=1 kills the SIMD loop.
           rewrite as a row-major sweep: pass 1 accumulates per-channel sum
           and sum-of-squares via SIMD across F (per-thread scratch, reduced
           at end); pass 2 fuses scale+shift into a single FMA across F per
           row. every access is sequential. */
        float *xh_d = record ? (float *)x_hat_save->storage->data : NULL;
        float *is_d = record ? (float *)inv_std_save->storage->data : NULL;

        /* scratch: mean[F], invstd[F], scale[F], bias_out[F] */
        size_t chan_scratch = 4 * (size_t)feat * sizeof(float);
        float *chan_buf = (float *)aligned_alloc(64, (chan_scratch + 63u) & ~(size_t)63u);
        if (!chan_buf) {
            ax_tensor_destroy(out);
            if (inp != input) ax_tensor_destroy(inp);
            return NULL;
        }
        float *mean_arr   = chan_buf;
        float *invstd_arr = chan_buf + feat;
        float *scale_arr  = chan_buf + 2 * feat;
        float *biasout_arr = chan_buf + 3 * feat;

        if (self->training) {
            /* per-thread sum + sum^2 arrays for reduction across rows. */
#ifdef _OPENMP
            int n_threads = omp_get_max_threads();
#else
            int n_threads = 1;
#endif
            size_t per_thread = (size_t)feat;
            size_t total = 2u * (size_t)n_threads * per_thread * sizeof(float);
            float *redbuf = (float *)aligned_alloc(64, (total + 63u) & ~(size_t)63u);
            if (!redbuf) {
                free(chan_buf);
                ax_tensor_destroy(out);
                if (inp != input) ax_tensor_destroy(inp);
                return NULL;
            }
            memset(redbuf, 0, total);
            float *sum_all  = redbuf;
            float *sum2_all = redbuf + (size_t)n_threads * per_thread;

            /* pass 1: per-thread row-major accumulation. */
#ifdef _OPENMP
            #pragma omp parallel
#endif
            {
#ifdef _OPENMP
                int tid = omp_get_thread_num();
#else
                int tid = 0;
#endif
                float *sum_l  = sum_all  + (size_t)tid * per_thread;
                float *sum2_l = sum2_all + (size_t)tid * per_thread;
                int64_t fe = feat - (feat % AX_VF32_WIDTH);
#ifdef _OPENMP
                #pragma omp for schedule(static)
#endif
                for (int64_t n = 0; n < batch; n++) {
                    const float *ip = id + n * feat;
                    int64_t f = 0;
                    for (; f < fe; f += AX_VF32_WIDTH) {
                        ax_vf32 x  = ax_vf32_loadu(ip + f);
                        ax_vf32 s  = ax_vf32_loadu(sum_l + f);
                        ax_vf32 s2 = ax_vf32_loadu(sum2_l + f);
                        ax_vf32_storeu(sum_l  + f, ax_vf32_add(s, x));
                        ax_vf32_storeu(sum2_l + f, ax_vf32_fmadd(x, x, s2));
                    }
                    for (; f < feat; f++) {
                        float x = ip[f];
                        sum_l[f]  += x;
                        sum2_l[f] += x * x;
                    }
                }
            }

            /* reduce per-thread scratch into thread-0 slot. */
            {
                int64_t fe = feat - (feat % AX_VF32_WIDTH);
                for (int t = 1; t < n_threads; t++) {
                    float *sp  = sum_all  + (size_t)t * per_thread;
                    float *s2p = sum2_all + (size_t)t * per_thread;
                    int64_t f = 0;
                    for (; f < fe; f += AX_VF32_WIDTH) {
                        ax_vf32_storeu(sum_all + f,
                            ax_vf32_add(ax_vf32_loadu(sum_all + f),
                                        ax_vf32_loadu(sp + f)));
                        ax_vf32_storeu(sum2_all + f,
                            ax_vf32_add(ax_vf32_loadu(sum2_all + f),
                                        ax_vf32_loadu(s2p + f)));
                    }
                    for (; f < feat; f++) {
                        sum_all[f]  += sp[f];
                        sum2_all[f] += s2p[f];
                    }
                }
            }

            /* per-channel mean, var, inv_std, running stats update. */
            float Nf = (float)batch;
            float inv_N = 1.0f / Nf;
            float denom = (batch > 1) ? 1.0f / (Nf - 1.0f) : 1.0f;
            float mom = bn->momentum;
            float one_minus_mom = 1.0f - mom;
            float eps = bn->eps;
            for (int64_t c = 0; c < feat; c++) {
                float s  = sum_all[c];
                float s2 = sum2_all[c];
                float mean = s * inv_N;
                float var  = s2 * inv_N - mean * mean;
                if (var < 0.0f) var = 0.0f;
                float inv_std = 1.0f / sqrtf(var + eps);
                mean_arr[c]   = mean;
                invstd_arr[c] = inv_std;
                scale_arr[c]  = gd[c] * inv_std;
                biasout_arr[c] = bd[c] - gd[c] * mean * inv_std;
                if (is_d) is_d[c] = inv_std;

                float var_sum = s2 - Nf * mean * mean; /* == Σ(x-mean)^2 */
                float unbiased = (batch > 1) ? var_sum * denom : var;
                rm[c] = one_minus_mom * rm[c] + mom * mean;
                rv[c] = one_minus_mom * rv[c] + mom * unbiased;
            }
            free(redbuf);
        } else {
            /* eval: derive scale/bias from running stats. */
            float eps = bn->eps;
            for (int64_t c = 0; c < feat; c++) {
                float inv_std = 1.0f / sqrtf(rv[c] + eps);
                mean_arr[c]   = rm[c];
                invstd_arr[c] = inv_std;
                scale_arr[c]  = gd[c] * inv_std;
                biasout_arr[c] = bd[c] - gd[c] * rm[c] * inv_std;
            }
        }

        /* pass 2: row-major SIMD normalize. out[n,f] = scale[f] * in[n,f] + bias_out[f]. */
        int64_t fe = feat - (feat % AX_VF32_WIDTH);
#ifdef _OPENMP
        int64_t bn_work = batch * feat;
        #pragma omp parallel for schedule(static) if (batch >= ax_par_threshold_batch && bn_work >= ax_par_threshold_elems)
#endif
        for (int64_t n = 0; n < batch; n++) {
            const float *ip = id + n * feat;
            float *op = od + n * feat;
            float *xhp = xh_d ? xh_d + n * feat : NULL;
            int64_t f = 0;
            for (; f < fe; f += AX_VF32_WIDTH) {
                ax_vf32 x  = ax_vf32_loadu(ip + f);
                ax_vf32 sc = ax_vf32_loadu(scale_arr + f);
                ax_vf32 bo = ax_vf32_loadu(biasout_arr + f);
                ax_vf32_storeu(op + f, ax_vf32_fmadd(sc, x, bo));
                if (xhp) {
                    ax_vf32 m  = ax_vf32_loadu(mean_arr + f);
                    ax_vf32 is = ax_vf32_loadu(invstd_arr + f);
                    ax_vf32_storeu(xhp + f, ax_vf32_mul(ax_vf32_sub(x, m), is));
                }
            }
            for (; f < feat; f++) {
                op[f] = scale_arr[f] * ip[f] + biasout_arr[f];
                if (xhp) xhp[f] = (ip[f] - mean_arr[f]) * invstd_arr[f];
            }
        }

        free(chan_buf);
    }
    else
    {
        /* 4D BN row-major path: shape [N, C, H, W], spatial = H*W.

           the old per-channel-outer loop strode between batch elements
           by C*H*W floats every iteration (400KB+ in VGG shapes), blowing
           past L2 on every step. rewrite as row-major: walk (n, c, spatial)
           in natural layout order so the hw prefetcher can track both the
           within-channel run of HW floats and the channel-to-channel jumps
           of HW floats (both cache-friendly when HW fits in L1).

           training: one fused pass computes per-channel sum and sum^2 via
           per-thread C-sized accumulators reduced at the end. this halves
           the memory bandwidth of the old mean-then-variance two-pass.

           eval: running stats go straight into scale/biasout so the hot
           pass is a single fused FMA per element. */
        float *xh_d = record ? (float *)x_hat_save->storage->data : NULL;
        float *is_d = record ? (float *)inv_std_save->storage->data : NULL;

        size_t chan_scratch = 4 * (size_t)feat * sizeof(float);
        float *chan_buf = (float *)aligned_alloc(64, (chan_scratch + 63u) & ~(size_t)63u);
        if (!chan_buf) {
            ax_tensor_destroy(out);
            if (inp != input) ax_tensor_destroy(inp);
            return NULL;
        }
        float *mean_arr    = chan_buf;
        float *invstd_arr  = chan_buf + feat;
        float *scale_arr   = chan_buf + 2 * feat;
        float *biasout_arr = chan_buf + 3 * feat;

        if (self->training) {
#ifdef _OPENMP
            int n_threads = omp_get_max_threads();
#else
            int n_threads = 1;
#endif
            size_t per_thread = (size_t)feat;
            size_t total = 2u * (size_t)n_threads * per_thread * sizeof(float);
            float *redbuf = (float *)aligned_alloc(64, (total + 63u) & ~(size_t)63u);
            if (!redbuf) {
                free(chan_buf);
                ax_tensor_destroy(out);
                if (inp != input) ax_tensor_destroy(inp);
                return NULL;
            }
            memset(redbuf, 0, total);
            float *sum_all  = redbuf;
            float *sum2_all = redbuf + (size_t)n_threads * per_thread;

            /* pass 1: row-major walk. each thread takes a contiguous slice
               of batch elements and, for every (n, c) slice of `spatial`
               contiguous floats, runs a SIMD sum+sum^2 reduction into its
               local sum_l[c] / sum2_l[c]. spatial is HW which is almost
               always a multiple of AX_VF32_WIDTH (e.g. 196, 784, 3136). */
#ifdef _OPENMP
            #pragma omp parallel
#endif
            {
#ifdef _OPENMP
                int tid = omp_get_thread_num();
#else
                int tid = 0;
#endif
                float *sum_l  = sum_all  + (size_t)tid * per_thread;
                float *sum2_l = sum2_all + (size_t)tid * per_thread;
                int64_t se = spatial - (spatial % AX_VF32_WIDTH);
#ifdef _OPENMP
                #pragma omp for schedule(static)
#endif
                for (int64_t n = 0; n < batch; n++) {
                    for (int64_t c = 0; c < feat; c++) {
                        const float *p = id + (n * feat + c) * spatial;
                        ax_vf32 vs = ax_vf32_zero(), vs2 = ax_vf32_zero();
                        int64_t s = 0;
                        for (; s < se; s += AX_VF32_WIDTH) {
                            ax_vf32 x = ax_vf32_loadu(p + s);
                            vs  = ax_vf32_add(vs, x);
                            vs2 = ax_vf32_fmadd(x, x, vs2);
                        }
                        float s_tail = 0.0f, s2_tail = 0.0f;
                        for (; s < spatial; s++) {
                            float x = p[s];
                            s_tail  += x;
                            s2_tail += x * x;
                        }
                        sum_l[c]  += ax_vf32_hsum(vs)  + s_tail;
                        sum2_l[c] += ax_vf32_hsum(vs2) + s2_tail;
                    }
                }
            }

            /* reduce per-thread scratch into thread-0 slot via SIMD adds. */
            {
                int64_t fe_r = feat - (feat % AX_VF32_WIDTH);
                for (int t = 1; t < n_threads; t++) {
                    float *sp  = sum_all  + (size_t)t * per_thread;
                    float *s2p = sum2_all + (size_t)t * per_thread;
                    int64_t f = 0;
                    for (; f < fe_r; f += AX_VF32_WIDTH) {
                        ax_vf32_storeu(sum_all + f,
                            ax_vf32_add(ax_vf32_loadu(sum_all + f),
                                        ax_vf32_loadu(sp + f)));
                        ax_vf32_storeu(sum2_all + f,
                            ax_vf32_add(ax_vf32_loadu(sum2_all + f),
                                        ax_vf32_loadu(s2p + f)));
                    }
                    for (; f < feat; f++) {
                        sum_all[f]  += sp[f];
                        sum2_all[f] += s2p[f];
                    }
                }
            }

            /* per-channel stats + running stats update. eff_count is the
               number of elements aggregated per channel (batch * spatial). */
            float Nf = (float)(batch * spatial);
            float inv_N = 1.0f / Nf;
            float denom = (Nf > 1.0f) ? 1.0f / (Nf - 1.0f) : 1.0f;
            float mom = bn->momentum;
            float one_minus_mom = 1.0f - mom;
            float eps = bn->eps;
            for (int64_t c = 0; c < feat; c++) {
                float s  = sum_all[c];
                float s2 = sum2_all[c];
                float mean = s * inv_N;
                float var  = s2 * inv_N - mean * mean;
                if (var < 0.0f) var = 0.0f;
                float inv_std = 1.0f / sqrtf(var + eps);
                mean_arr[c]    = mean;
                invstd_arr[c]  = inv_std;
                scale_arr[c]   = gd[c] * inv_std;
                biasout_arr[c] = bd[c] - gd[c] * mean * inv_std;
                if (is_d) is_d[c] = inv_std;

                float var_sum  = s2 - Nf * mean * mean;
                float unbiased = (Nf > 1.0f) ? var_sum * denom : var;
                rm[c] = one_minus_mom * rm[c] + mom * mean;
                rv[c] = one_minus_mom * rv[c] + mom * unbiased;
            }
            free(redbuf);
        } else {
            /* eval path: fold running stats into scale/biasout. */
            float eps = bn->eps;
            for (int64_t c = 0; c < feat; c++) {
                float inv_std = 1.0f / sqrtf(rv[c] + eps);
                mean_arr[c]    = rm[c];
                invstd_arr[c]  = inv_std;
                scale_arr[c]   = gd[c] * inv_std;
                biasout_arr[c] = bd[c] - gd[c] * rm[c] * inv_std;
            }
        }

        /* pass 2: row-major normalize. broadcast scale[c] and biasout[c]
           across the HW slice; if recording for backward also store x_hat. */
        int64_t se2 = spatial - (spatial % AX_VF32_WIDTH);
#ifdef _OPENMP
        int64_t bn4_work = batch * feat * spatial;
        #pragma omp parallel for schedule(static) if (batch >= ax_par_threshold_batch && bn4_work >= ax_par_threshold_elems)
#endif
        for (int64_t n = 0; n < batch; n++) {
            for (int64_t c = 0; c < feat; c++) {
                const float *ip = id + (n * feat + c) * spatial;
                float *op = od + (n * feat + c) * spatial;
                float *xhp = xh_d ? xh_d + (n * feat + c) * spatial : NULL;
                ax_vf32 v_sc = ax_vf32_set1(scale_arr[c]);
                ax_vf32 v_bo = ax_vf32_set1(biasout_arr[c]);
                ax_vf32 v_mn = xhp ? ax_vf32_set1(mean_arr[c])   : v_sc;
                ax_vf32 v_is = xhp ? ax_vf32_set1(invstd_arr[c]) : v_sc;
                int64_t s = 0;
                for (; s < se2; s += AX_VF32_WIDTH) {
                    ax_vf32 x = ax_vf32_loadu(ip + s);
                    ax_vf32_storeu(op + s, ax_vf32_fmadd(v_sc, x, v_bo));
                    if (xhp) ax_vf32_storeu(xhp + s, ax_vf32_mul(ax_vf32_sub(x, v_mn), v_is));
                }
                for (; s < spatial; s++) {
                    op[s] = scale_arr[c] * ip[s] + biasout_arr[c];
                    if (xhp) xhp[s] = (ip[s] - mean_arr[c]) * invstd_arr[c];
                }
            }
        }

        free(chan_buf);
    }

    /* hook up backward */
    if (record) {
        bn_backward_ctx_t *ctx = malloc(sizeof(bn_backward_ctx_t));
        if (!ctx) {
            ax_tensor_destroy(x_hat_save);
            ax_tensor_destroy(inv_std_save);
            if (inp != input) ax_tensor_destroy(inp);
            return out;
        }
        ctx->x_hat = x_hat_save;
        ctx->inv_std = inv_std_save;
        ctx->bn = bn;

        ax_grad_fn_t *gf = ax_grad_fn_create(batchnorm_backward);
        gf->inputs[0] = input; /* route grad to original (possibly non-contiguous) input */
        gf->n_inputs = 1;
        gf->ctx = ctx;
        gf->ctx_cleanup = bn_ctx_cleanup;
        out->requires_grad = true;
        out->grad_fn = gf;
    }

    if (inp != input) ax_tensor_destroy(inp);
    return out;
}

static void batchnorm_destroy(ax_layer_t *self)
{
    ax_batchnorm_t *bn = (ax_batchnorm_t *)self;
    if (bn->gamma) ax_tensor_destroy(bn->gamma);
    if (bn->beta) ax_tensor_destroy(bn->beta);
    if (bn->running_mean) ax_tensor_destroy(bn->running_mean);
    if (bn->running_var) ax_tensor_destroy(bn->running_var);
    free(bn);
}

ax_layer_t *ax_batchnorm_create(int64_t num_features, float eps, float momentum)
{
    ax_batchnorm_t *bn = calloc(1, sizeof(ax_batchnorm_t));
    if (!bn) return NULL;

    bn->base.ops.forward = batchnorm_forward;
    bn->base.ops.destroy = batchnorm_destroy;
    bn->base.type = AX_LAYER_BATCHNORM;
    bn->base.training = true;
    bn->num_features = num_features;
    bn->eps = eps > 0 ? eps : 1e-5f;
    bn->momentum = momentum > 0 ? momentum : 0.1f;

    int64_t shape[] = {num_features};
    bn->gamma = ax_tensor_ones(shape, 1, AX_FLOAT32);
    bn->beta = ax_tensor_zeros(shape, 1, AX_FLOAT32);
    bn->running_mean = ax_tensor_zeros(shape, 1, AX_FLOAT32);
    bn->running_var = ax_tensor_ones(shape, 1, AX_FLOAT32);

    if (!bn->gamma || !bn->beta || !bn->running_mean || !bn->running_var)
    {
        batchnorm_destroy((ax_layer_t *)bn);
        return NULL;
    }

    bn->gamma->requires_grad = true;
    bn->beta->requires_grad = true;
    bn->base.params[0] = bn->gamma;
    bn->base.params[1] = bn->beta;
    bn->base.n_params = 2;
    bn->base.buffers[0] = bn->running_mean;
    bn->base.buffers[1] = bn->running_var;
    bn->base.n_buffers = 2;
    bn->base.input_features = num_features;
    bn->base.output_features = num_features;

    return (ax_layer_t *)bn;
}


/* context saved during layernorm forward for the backward pass */
typedef struct {
    ax_tensor_t *x_hat;    /* normalized input [batch, feat] */
    ax_tensor_t *inv_std;  /* 1/sqrt(var+eps) per sample [batch] */
    ax_layernorm_t *ln;
} ln_backward_ctx_t;

static void layernorm_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ln_backward_ctx_t *ctx = (ln_backward_ctx_t *)self->ctx;
    ax_tensor_t *input = self->inputs[0];
    ax_tensor_t *x_hat_t = ctx->x_hat;
    ax_tensor_t *inv_std_t = ctx->inv_std;
    ax_layernorm_t *ln = ctx->ln;

    /* support 2D [N, D] and 3D [B, S, D] — collapse leading dims. */
    int64_t batch, feat;
    if (grad_out->ndim == 2) {
        batch = grad_out->shape[0];
        feat = grad_out->shape[1];
    } else {
        batch = grad_out->shape[0] * grad_out->shape[1];
        feat = grad_out->shape[2];
    }
    float *go = (float *)grad_out->storage->data;
    float *xh = (float *)x_hat_t->storage->data;
    float *istd = (float *)inv_std_t->storage->data;
    float *gd = (float *)ln->gamma->storage->data;

    int64_t fe = feat - (feat % AX_VF32_WIDTH);

    /* dgamma = sum(grad_out * x_hat, axis=0), dbeta = sum(grad_out, axis=0).
       rewrite with feat as the outer loop so the reduction accumulator
       is 1d (no contention across threads) and the inner batch loop
       can be simd across feat. actually simpler: walk row-major with
       simd fmadd into per-thread [feat] accumulators, then reduce. */
    bool need_dg = ln->gamma->requires_grad;
    bool need_db = ln->beta->requires_grad;
    if (need_dg && !ln->gamma->grad)
        ln->gamma->grad = ax_tensor_zeros(ln->gamma->shape, ln->gamma->ndim, ln->gamma->dtype);
    if (need_db && !ln->beta->grad)
        ln->beta->grad = ax_tensor_zeros(ln->beta->shape, ln->beta->ndim, ln->beta->dtype);
    if ((need_dg && !ln->gamma->grad) || (need_db && !ln->beta->grad)) goto cleanup;

    if (need_dg || need_db) {
        float *dg = need_dg ? (float *)ln->gamma->grad->storage->data : NULL;
        float *db = need_db ? (float *)ln->beta->grad->storage->data : NULL;
#ifdef _OPENMP
        int64_t dgdb_work = batch * feat;
        #pragma omp parallel for schedule(static) if (dgdb_work >= ax_par_threshold_elems)
#endif
        for (int64_t f = 0; f < feat; f++) {
            float sum_g = 0, sum_gx = 0;
            for (int64_t b = 0; b < batch; b++) {
                float g = go[b * feat + f];
                sum_g += g;
                if (dg) sum_gx += g * xh[b * feat + f];
            }
            if (dg) dg[f] += sum_gx;
            if (db) db[f] += sum_g;
        }
    }

    /* dx: per-sample, SIMD over feat. each sample is independent so
       parallelize over batch with a reduction on nothing (each writes
       disjoint rows). */
    if (input->requires_grad) {
        if (!input->grad)
            input->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
        if (!input->grad) goto cleanup;
        float *ig = (float *)input->grad->storage->data;
        float D = (float)feat;
        float inv_D = 1.0f / D;

#ifdef _OPENMP
        int64_t dx_work = batch * feat;
        #pragma omp parallel for schedule(static) if (batch >= ax_par_threshold_batch && dx_work >= ax_par_threshold_elems)
#endif
        for (int64_t b = 0; b < batch; b++) {
            const float *gp = go + b * feat;
            const float *xp = xh + b * feat;
            float is = istd[b];
            /* pass 1: sum_go, sum_go_xh via simd fmadd. */
            ax_vf32 v_sg = ax_vf32_zero();
            ax_vf32 v_sgx = ax_vf32_zero();
            int64_t f = 0;
            for (; f < fe; f += AX_VF32_WIDTH) {
                ax_vf32 g = ax_vf32_mul(ax_vf32_loadu(gp + f), ax_vf32_loadu(gd + f));
                ax_vf32 x = ax_vf32_loadu(xp + f);
                v_sg  = ax_vf32_add(v_sg, g);
                v_sgx = ax_vf32_fmadd(g, x, v_sgx);
            }
            float sum_g  = ax_vf32_hsum(v_sg);
            float sum_gx = ax_vf32_hsum(v_sgx);
            for (; f < feat; f++) {
                float g = gp[f] * gd[f];
                sum_g  += g;
                sum_gx += g * xp[f];
            }
            /* pass 2: dx = inv_D * inv_std * (D*dx_hat - sum_go - x_hat*sum_go_xh) */
            float *ip = ig + b * feat;
            ax_vf32 v_coeff = ax_vf32_set1(inv_D * is);
            ax_vf32 v_D     = ax_vf32_set1(D);
            ax_vf32 v_sg_b  = ax_vf32_set1(sum_g);
            ax_vf32 v_sgx_b = ax_vf32_set1(sum_gx);
            f = 0;
            for (; f < fe; f += AX_VF32_WIDTH) {
                ax_vf32 dx_hat = ax_vf32_mul(ax_vf32_loadu(gp + f), ax_vf32_loadu(gd + f));
                ax_vf32 x      = ax_vf32_loadu(xp + f);
                ax_vf32 inner  = ax_vf32_sub(ax_vf32_sub(ax_vf32_mul(v_D, dx_hat), v_sg_b),
                                             ax_vf32_mul(x, v_sgx_b));
                ax_vf32 dx     = ax_vf32_mul(v_coeff, inner);
                ax_vf32_storeu(ip + f, ax_vf32_add(ax_vf32_loadu(ip + f), dx));
            }
            for (; f < feat; f++) {
                float dx_hat = gp[f] * gd[f];
                float dx = inv_D * is * (D * dx_hat - sum_g - xp[f] * sum_gx);
                ip[f] += dx;
            }
        }
    }

cleanup:
    ax_tensor_destroy(ctx->x_hat);
    ax_tensor_destroy(ctx->inv_std);
    free(ctx);
    self->ctx = NULL;
}

static void ln_ctx_cleanup(void *p)
{
    ln_backward_ctx_t *ctx = (ln_backward_ctx_t *)p;
    if (ctx->x_hat) ax_tensor_destroy(ctx->x_hat);
    if (ctx->inv_std) ax_tensor_destroy(ctx->inv_std);
    free(ctx);
}

static ax_tensor_t *layernorm_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_layernorm_t *ln = (ax_layernorm_t *)self;
    /* accept 2D [N, D] or 3D [B, S, D] — transformer encoders feed 3D.
       higher dims get collapsed into the leading "batch" dim; the last
       dim is always the feature axis being normalized across. */
    if (!input || (input->ndim != 2 && input->ndim != 3)) return NULL;

    /* ensure contiguous so flat indexing below is correct */
    ax_tensor_t *inp = ax_ensure_contiguous(input);
    if (!inp) return NULL;

    int64_t batch, feat;
    if (inp->ndim == 2) {
        batch = inp->shape[0];
        feat = inp->shape[1];
    } else {
        /* 3D: collapse leading dims */
        batch = inp->shape[0] * inp->shape[1];
        feat = inp->shape[2];
    }
    float *id = (float *)inp->storage->data;

    /* uninitialized output — fully overwritten below. */
    ax_tensor_t *out = ax_tensor_create(inp->shape, inp->ndim, inp->dtype);
    if (!out) { if (inp != input) ax_tensor_destroy(inp); return NULL; }
    float *od = (float *)out->storage->data;
    float *gd = (float *)ln->gamma->storage->data;
    float *bd = (float *)ln->beta->storage->data;

    bool record = ax_grad_enabled() &&
                  (input->requires_grad || ln->gamma->requires_grad || ln->beta->requires_grad);

    ax_tensor_t *x_hat_save = NULL;
    ax_tensor_t *inv_std_save = NULL;
    if (record) {
        /* these get fully filled in the normalize loop; skip the memset */
        x_hat_save = ax_tensor_create(input->shape, input->ndim, input->dtype);
        int64_t is_shape[] = {batch};
        inv_std_save = ax_tensor_create(is_shape, 1, input->dtype);
        if (!x_hat_save || !inv_std_save) {
            if (x_hat_save) ax_tensor_destroy(x_hat_save);
            if (inv_std_save) ax_tensor_destroy(inv_std_save);
            record = false;
        }
    }

    float *xh_d = record ? (float *)x_hat_save->storage->data : NULL;
    float *is_d = record ? (float *)inv_std_save->storage->data : NULL;

    /* layernorm normalizes per-sample, so each sample is independent.
       dynamic-1 for hybrid P/E core load balancing.

       SIMD 2-pass: pass 1 sums via ax_vf32, pass 2 sums squared deviations
       via ax_vf32 fmadd, pass 3 fused normalize + affine.
       double-accumulated hsum keeps the variance stable for feat up to ~1M. */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
    #endif
    for (int64_t b = 0; b < batch; b++)
    {
        const float *ip = id + b * feat;
        float *op = od + b * feat;
        float *xhp = record ? xh_d + b * feat : NULL;

        int64_t fe = feat - (feat % AX_VF32_WIDTH);

        /* pass 1: sum via SIMD, double hsum. */
        double sum_d = 0.0;
        {
            ax_vf32 vs = ax_vf32_zero();
            int64_t f = 0;
            for (; f < fe; f += AX_VF32_WIDTH)
                vs = ax_vf32_add(vs, ax_vf32_loadu(ip + f));
            sum_d += (double)ax_vf32_hsum(vs);
            for (; f < feat; f++) sum_d += (double)ip[f];
        }
        float mean = (float)(sum_d / (double)feat);

        /* pass 2: sum of (x - mean)^2 via SIMD fmadd. */
        double var_d = 0.0;
        {
            ax_vf32 v_mean = ax_vf32_set1(mean);
            ax_vf32 vv = ax_vf32_zero();
            int64_t f = 0;
            for (; f < fe; f += AX_VF32_WIDTH) {
                ax_vf32 d = ax_vf32_sub(ax_vf32_loadu(ip + f), v_mean);
                vv = ax_vf32_fmadd(d, d, vv);
            }
            var_d += (double)ax_vf32_hsum(vv);
            for (; f < feat; f++) { float d = ip[f] - mean; var_d += (double)d * (double)d; }
        }
        float var = (feat > 0) ? (float)(var_d / (double)feat) : 0.0f;
        float inv_std = 1.0f / sqrtf(var + ln->eps);

        if (record) is_d[b] = inv_std;

        /* pass 3: fused normalize + affine. out = gamma * ((x - mean) * inv_std) + beta.
           also stashes x_hat for backward when record is set. */
        {
            ax_vf32 v_mean = ax_vf32_set1(mean);
            ax_vf32 v_is   = ax_vf32_set1(inv_std);
            int64_t f = 0;
            for (; f < fe; f += AX_VF32_WIDTH) {
                ax_vf32 x  = ax_vf32_loadu(ip + f);
                ax_vf32 xh = ax_vf32_mul(ax_vf32_sub(x, v_mean), v_is);
                ax_vf32 g  = ax_vf32_loadu(gd + f);
                ax_vf32 bt = ax_vf32_loadu(bd + f);
                ax_vf32_storeu(op + f, ax_vf32_fmadd(g, xh, bt));
                if (xhp) ax_vf32_storeu(xhp + f, xh);
            }
            for (; f < feat; f++) {
                float xh = (ip[f] - mean) * inv_std;
                op[f] = gd[f] * xh + bd[f];
                if (xhp) xhp[f] = xh;
            }
        }
    }

    if (record) {
        ln_backward_ctx_t *ctx = malloc(sizeof(ln_backward_ctx_t));
        if (!ctx) {
            ax_tensor_destroy(x_hat_save);
            ax_tensor_destroy(inv_std_save);
            if (inp != input) ax_tensor_destroy(inp);
            return out;
        }
        ctx->x_hat = x_hat_save;
        ctx->inv_std = inv_std_save;
        ctx->ln = ln;

        ax_grad_fn_t *gf = ax_grad_fn_create(layernorm_backward);
        gf->inputs[0] = input; /* route grad to original (possibly non-contiguous) input */
        gf->n_inputs = 1;
        gf->ctx = ctx;
        gf->ctx_cleanup = ln_ctx_cleanup;
        out->requires_grad = true;
        out->grad_fn = gf;
    }

    if (inp != input) ax_tensor_destroy(inp);
    return out;
}

static void layernorm_destroy(ax_layer_t *self)
{
    ax_layernorm_t *ln = (ax_layernorm_t *)self;
    if (ln->gamma) ax_tensor_destroy(ln->gamma);
    if (ln->beta) ax_tensor_destroy(ln->beta);
    free(ln);
}

ax_layer_t *ax_layernorm_create(int64_t num_features, float eps)
{
    ax_layernorm_t *ln = calloc(1, sizeof(ax_layernorm_t));
    if (!ln) return NULL;

    ln->base.ops.forward = layernorm_forward;
    ln->base.ops.destroy = layernorm_destroy;
    ln->base.type = AX_LAYER_LAYERNORM;
    ln->base.training = true;
    ln->num_features = num_features;
    ln->eps = eps > 0 ? eps : 1e-5f;

    int64_t shape[] = {num_features};
    ln->gamma = ax_tensor_ones(shape, 1, AX_FLOAT32);
    ln->beta = ax_tensor_zeros(shape, 1, AX_FLOAT32);
    if (!ln->gamma || !ln->beta)
    {
        layernorm_destroy((ax_layer_t *)ln);
        return NULL;
    }

    ln->gamma->requires_grad = true;
    ln->beta->requires_grad = true;
    ln->base.params[0] = ln->gamma;
    ln->base.params[1] = ln->beta;
    ln->base.n_params = 2;
    ln->base.input_features = num_features;
    ln->base.output_features = num_features;

    return (ax_layer_t *)ln;
}


/* dropout */

static void dropout_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *input = self->inputs[0];
    ax_tensor_t *mask = self->saved[0];  /* mask * scale stored during forward */

    if (!input->requires_grad) return;

    int64_t n = ax_tensor_numel(grad_out);
    if (!input->grad)
        input->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
    if (!input->grad) return;

    float *ig = (float *)input->grad->storage->data;
    float *go = (float *)grad_out->storage->data;
    float *md = (float *)mask->storage->data;
    int64_t igoff = (int64_t)input->grad->offset;
    int64_t gooff = (int64_t)grad_out->offset;

    /* fast path: all contiguous, offset 0 — SIMD fused multiply-add accumulate */
    if (igoff == 0 && gooff == 0
        && ax_tensor_is_contiguous(input->grad)
        && ax_tensor_is_contiguous(grad_out))
    {
        int64_t i = 0, ve = n - (n % AX_VF32_WIDTH);
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(n > 65536)
        #endif
        for (int64_t j = 0; j < ve; j += AX_VF32_WIDTH) {
            ax_vf32 g = ax_vf32_loadu(go + j);
            ax_vf32 m = ax_vf32_loadu(md + j);
            ax_vf32_storeu(ig + j, ax_vf32_fmadd(g, m, ax_vf32_loadu(ig + j)));
        }
        for (i = ve; i < n; i++)
            ig[i] += go[i] * md[i];
        return;
    }

    /* fallback */
    for (int64_t i = 0; i < n; i++)
        ig[igoff + i] += go[gooff + i] * md[i];
}

/* xorshift64* — fast per-thread prng for dropout mask. small state, no globals,
   each thread advances its own copy so iterations stay independent. */
static inline uint64_t xs64_next(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static ax_tensor_t *dropout_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_dropout_t *dp = (ax_dropout_t *)self;

    if (!self->training || dp->p <= 0.0f)
        return ax_tensor_view(input);

    ax_tensor_t *out = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
    if (!out) return NULL;

    int64_t n = ax_tensor_numel(input);
    float *id = (float *)input->storage->data;
    float *od = (float *)out->storage->data;
    float keep = 1.0f - dp->p;
    float scale = 1.0f / keep;

    /* save mask*scale for backward */
    bool record = ax_grad_enabled() && input->requires_grad;
    ax_tensor_t *mask = NULL;
    float *md = NULL;
    if (record) {
        mask = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
        if (!mask) record = false;
        else md = (float *)mask->storage->data;
    }

    /* small-n path: serial, uses the global rng (matches pre-parallel behavior). */
    if (n <= 65536)
    {
        for (int64_t i = 0; i < n; i++)
        {
            float r = ax_rng_float();
            if (r < keep) {
                od[out->offset + i] = id[input->offset + i] * scale;
                if (record) md[i] = scale;
            }
            /* else: od stays 0, md stays 0 */
        }
    }
    else
    {
        /* large-n path: parallel with per-thread xorshift64*.
           seed = global rng draw mixed with thread id so each thread produces
           a different sequence and the overall mask varies between calls. */
        uint64_t call_seed = ax_rng_uint64();
        int64_t off_in = (int64_t)input->offset;
        int64_t off_out = (int64_t)out->offset;
        /* xorshift requires non-zero state; OR in a constant just in case. */
        #ifdef _OPENMP
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            uint64_t st = (call_seed ^ ((uint64_t)(unsigned)tid * (uint64_t)0x9E3779B97F4A7C15ULL))
                          | 0x1ULL;
            /* burn one round to scramble correlated low bits across threads */
            (void)xs64_next(&st);
            /* xs64 -> float in [0,1) by taking the high 24 bits */
            const float inv_2p24 = 1.0f / (float)(1u << 24);

            #pragma omp for schedule(static)
            for (int64_t i = 0; i < n; i++)
            {
                float r = (float)(xs64_next(&st) >> 40) * inv_2p24;
                if (r < keep) {
                    od[off_out + i] = id[off_in + i] * scale;
                    if (record) md[i] = scale;
                }
            }
        }
        #else
        uint64_t st = (call_seed ^ 0x9E3779B97F4A7C15ULL) | 0x1ULL;
        (void)xs64_next(&st);
        const float inv_2p24 = 1.0f / (float)(1u << 24);
        for (int64_t i = 0; i < n; i++)
        {
            float r = (float)(xs64_next(&st) >> 40) * inv_2p24;
            if (r < keep) {
                od[off_out + i] = id[off_in + i] * scale;
                if (record) md[i] = scale;
            }
        }
        #endif
    }

    if (record) {
        ax_grad_fn_t *gf = ax_grad_fn_create(dropout_backward);
        gf->inputs[0] = input;
        gf->n_inputs = 1;
        gf->saved[0] = mask;
        gf->saved_owned[0] = true; /* we created the mask, we own it */
        gf->n_saved = 1;
        out->requires_grad = true;
        out->grad_fn = gf;
    }

    return out;
}

static void dropout_destroy(ax_layer_t *self) { free(self); }

ax_layer_t *ax_dropout_create(float p)
{
    ax_dropout_t *dp = calloc(1, sizeof(ax_dropout_t));
    if (!dp) return NULL;
    dp->base.ops.forward = dropout_forward;
    dp->base.ops.destroy = dropout_destroy;
    dp->base.type = AX_LAYER_DROPOUT;
    dp->base.training = true;
    dp->p = (p >= 0.0f && p < 1.0f) ? p : 0.5f;
    return (ax_layer_t *)dp;
}


/* gradient clipping */

void ax_clip_grad_value(ax_tensor_t **params, int n_params, float max_val)
{
    if (!params || max_val <= 0) return;
    for (int i = 0; i < n_params; i++)
    {
        if (!params[i] || !params[i]->grad) continue;
        int64_t n = ax_tensor_numel(params[i]->grad);
        float *gd = (float *)params[i]->grad->storage->data;
        for (int64_t j = 0; j < n; j++)
        {
            float g = gd[params[i]->grad->offset + j];
            if (g > max_val) g = max_val;
            if (g < -max_val) g = -max_val;
            gd[params[i]->grad->offset + j] = g;
        }
    }
}

float ax_clip_grad_norm(ax_tensor_t **params, int n_params, float max_norm)
{
    if (!params || max_norm <= 0) return 0.0f;

    /* sum-of-squares — parallelize across params with reduction.
       per-param SIMD reduction with 4 independent accumulators. */
    double total_norm_sq = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:total_norm_sq) schedule(dynamic, 1)
    #endif
    for (int i = 0; i < n_params; i++)
    {
        if (!params[i] || !params[i]->grad) continue;
        int64_t n = ax_tensor_numel(params[i]->grad);
        float *gd = (float *)params[i]->grad->storage->data + params[i]->grad->offset;

        ax_vf32 a0 = ax_vf32_zero(), a1 = ax_vf32_zero();
        ax_vf32 a2 = ax_vf32_zero(), a3 = ax_vf32_zero();
        int64_t j = 0, ve = n - (n % (AX_VF32_WIDTH * 4));
        for (; j < ve; j += AX_VF32_WIDTH * 4) {
            ax_vf32 v0 = ax_vf32_loadu(gd + j);
            ax_vf32 v1 = ax_vf32_loadu(gd + j + AX_VF32_WIDTH);
            ax_vf32 v2 = ax_vf32_loadu(gd + j + AX_VF32_WIDTH * 2);
            ax_vf32 v3 = ax_vf32_loadu(gd + j + AX_VF32_WIDTH * 3);
            a0 = ax_vf32_fmadd(v0, v0, a0);
            a1 = ax_vf32_fmadd(v1, v1, a1);
            a2 = ax_vf32_fmadd(v2, v2, a2);
            a3 = ax_vf32_fmadd(v3, v3, a3);
        }
        a0 = ax_vf32_add(ax_vf32_add(a0, a1), ax_vf32_add(a2, a3));
        double s = (double)ax_vf32_hsum(a0);
        for (; j < n; j++) s += (double)gd[j] * (double)gd[j];
        total_norm_sq += s;
    }

    float total_norm = (float)sqrt(total_norm_sq);

    if (total_norm > max_norm)
    {
        float scale = max_norm / total_norm;
        ax_vf32 v_scale = ax_vf32_set1(scale);
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
        #endif
        for (int i = 0; i < n_params; i++)
        {
            if (!params[i] || !params[i]->grad) continue;
            int64_t n = ax_tensor_numel(params[i]->grad);
            float *gd = (float *)params[i]->grad->storage->data + params[i]->grad->offset;
            int64_t j = 0, ve = n - (n % AX_VF32_WIDTH);
            for (; j < ve; j += AX_VF32_WIDTH)
                ax_vf32_storeu(gd + j, ax_vf32_mul(ax_vf32_loadu(gd + j), v_scale));
            for (; j < n; j++) gd[j] *= scale;
        }
    }
    return total_norm;
}
