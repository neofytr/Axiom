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
    #pragma omp parallel for schedule(static)
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
                ax_vf32 g = ax_vf32_load(go + base + s);
                ax_vf32 x = ax_vf32_load(xh + base + s);
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
                    ax_vf32 g = ax_vf32_load(go + base + s);
                    ax_vf32 x = ax_vf32_load(xh + base + s);
                    /* dx = coeff * (N*g - sum_go - x*sum_go_xh) */
                    ax_vf32 dx = ax_vf32_mul(v_coeff,
                        ax_vf32_sub(ax_vf32_sub(ax_vf32_mul(v_N, g), v_sum_go),
                                    ax_vf32_mul(x, v_sum_gx)));
                    ax_vf32_store(ig + base + s,
                        ax_vf32_add(ax_vf32_load(ig + base + s), dx));
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
    float eff_batch = (float)(batch * spatial); /* elements per channel */
    float *id = (float *)inp->storage->data;

    ax_tensor_t *out = ax_tensor_zeros(inp->shape, inp->ndim, inp->dtype);
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
        x_hat_save = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
        int64_t is_shape[] = {feat};
        inv_std_save = ax_tensor_zeros(is_shape, 1, input->dtype);
        if (!x_hat_save || !inv_std_save) {
            if (x_hat_save) ax_tensor_destroy(x_hat_save);
            if (inv_std_save) ax_tensor_destroy(inv_std_save);
            record = false;
        }
    }

    if (self->training)
    {
        float *xh_d = record ? (float *)x_hat_save->storage->data : NULL;
        float *is_d = record ? (float *)inv_std_save->storage->data : NULL;

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t c = 0; c < feat; c++)
        {
            /* pass 1: compute mean with SIMD reduction */
            float mean;
            {
                double dsum = 0.0;
                for (int64_t n = 0; n < batch; n++) {
                    int64_t base = n * feat * spatial + c * spatial;
                    int64_t s = 0, se = spatial - (spatial % AX_VF32_WIDTH);
                    ax_vf32 vs = ax_vf32_zero();
                    for (; s < se; s += AX_VF32_WIDTH)
                        vs = ax_vf32_add(vs, ax_vf32_load(id + base + s));
                    dsum += (double)ax_vf32_hsum(vs);
                    for (; s < spatial; s++) dsum += (double)id[base + s];
                }
                mean = (float)(dsum / (double)(batch * spatial));
            }

            /* pass 2: compute variance with SIMD */
            ax_vf32 v_mean = ax_vf32_set1(mean);
            double var_sum_d = 0.0;
            for (int64_t n = 0; n < batch; n++)
            {
                int64_t base = n * feat * spatial + c * spatial;
                int64_t s = 0, se = spatial - (spatial % AX_VF32_WIDTH);
                ax_vf32 vv = ax_vf32_zero();
                for (; s < se; s += AX_VF32_WIDTH) {
                    ax_vf32 d = ax_vf32_sub(ax_vf32_load(id + base + s), v_mean);
                    vv = ax_vf32_fmadd(d, d, vv);
                }
                var_sum_d += (double)ax_vf32_hsum(vv);
                for (; s < spatial; s++) {
                    float d = id[base + s] - mean;
                    var_sum_d += (double)d * (double)d;
                }
            }
            float var = (float)(var_sum_d / (double)(batch * spatial));
            float var_sum = (float)var_sum_d;
            float inv_std = 1.0f / sqrtf(var + bn->eps);

            if (record) is_d[c] = inv_std;

            /* pass 2: normalize and apply affine.
               fuse to single FMA: out = scale * input + bias_out */
            float scale = gd[c] * inv_std;
            float bias_out = bd[c] - gd[c] * mean * inv_std;
            ax_vf32 v_sc = ax_vf32_set1(scale);
            ax_vf32 v_bi = ax_vf32_set1(bias_out);
            ax_vf32 v_is = ax_vf32_set1(inv_std);
            ax_vf32 v_mn = ax_vf32_set1(mean);

            for (int64_t n = 0; n < batch; n++)
            {
                int64_t base = n * feat * spatial + c * spatial;
                if (record) {
                    int64_t s = 0, se = spatial - (spatial % AX_VF32_WIDTH);
                    for (; s < se; s += AX_VF32_WIDTH) {
                        ax_vf32 inp = ax_vf32_load(id + base + s);
                        ax_vf32 xh = ax_vf32_mul(ax_vf32_sub(inp, v_mn), v_is);
                        ax_vf32_store(od + base + s, ax_vf32_fmadd(v_sc, ax_vf32_load(id + base + s), v_bi));
                        ax_vf32_store(xh_d + base + s, xh);
                    }
                    for (; s < spatial; s++) {
                        float xh = (id[base + s] - mean) * inv_std;
                        od[base + s] = scale * id[base + s] + bias_out;
                        xh_d[base + s] = xh;
                    }
                } else {
                    int64_t s = 0, se = spatial - (spatial % AX_VF32_WIDTH);
                    for (; s < se; s += AX_VF32_WIDTH)
                        ax_vf32_store(od + base + s,
                            ax_vf32_fmadd(v_sc, ax_vf32_load(id + base + s), v_bi));
                    for (; s < spatial; s++)
                        od[base + s] = scale * id[base + s] + bias_out;
                }
            }

            rm[c] = (1.0f - bn->momentum) * rm[c] + bn->momentum * mean;
            int64_t eff_count = batch * spatial;
            float unbiased_var = (eff_count > 1) ? var_sum / (float)(eff_count - 1) : var;
            rv[c] = (1.0f - bn->momentum) * rv[c] + bn->momentum * unbiased_var;
        }
    }
    else
    {
        /* eval path: fused scale+shift per channel.
           precompute scale = gamma * inv_std, bias_out = beta - gamma * mean * inv_std
           then out[idx] = scale * input[idx] + bias_out (single FMA per element) */
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t c = 0; c < feat; c++)
        {
            float inv_std = 1.0f / sqrtf(rv[c] + bn->eps);
            float scale = gd[c] * inv_std;
            float bias_out = bd[c] - gd[c] * rm[c] * inv_std;
            ax_vf32 v_scale = ax_vf32_set1(scale);
            ax_vf32 v_bias = ax_vf32_set1(bias_out);

            for (int64_t n = 0; n < batch; n++)
            {
                int64_t base = n * feat * spatial + c * spatial;
                int64_t s = 0, se = spatial - (spatial % AX_VF32_WIDTH);
                for (; s < se; s += AX_VF32_WIDTH)
                    ax_vf32_store(od + base + s,
                        ax_vf32_fmadd(v_scale, ax_vf32_load(id + base + s), v_bias));
                for (; s < spatial; s++)
                    od[base + s] = scale * id[base + s] + bias_out;
            }
        }
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

    int64_t batch = grad_out->shape[0];
    int64_t feat = grad_out->shape[1];
    float *go = (float *)grad_out->storage->data;
    float *xh = (float *)x_hat_t->storage->data;
    float *istd = (float *)inv_std_t->storage->data;
    float *gd = (float *)ln->gamma->storage->data;

    /* dgamma = sum(grad_out * x_hat, axis=0) */
    if (ln->gamma->requires_grad) {
        if (!ln->gamma->grad)
            ln->gamma->grad = ax_tensor_zeros(ln->gamma->shape, ln->gamma->ndim, ln->gamma->dtype);
        if (!ln->gamma->grad) goto cleanup;
        float *dg = (float *)ln->gamma->grad->storage->data;
        for (int64_t f = 0; f < feat; f++)
            for (int64_t b = 0; b < batch; b++)
                dg[f] += go[b * feat + f] * xh[b * feat + f];
    }
    /* dbeta = sum(grad_out, axis=0) */
    if (ln->beta->requires_grad) {
        if (!ln->beta->grad)
            ln->beta->grad = ax_tensor_zeros(ln->beta->shape, ln->beta->ndim, ln->beta->dtype);
        if (!ln->beta->grad) goto cleanup;
        float *db = (float *)ln->beta->grad->storage->data;
        for (int64_t f = 0; f < feat; f++)
            for (int64_t b = 0; b < batch; b++)
                db[f] += go[b * feat + f];
    }

    /* dx: same simplified formula as batchnorm but per-sample across features */
    if (input->requires_grad) {
        if (!input->grad)
            input->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
        if (!input->grad) goto cleanup;
        float *ig = (float *)input->grad->storage->data;
        float D = (float)feat;

        for (int64_t b = 0; b < batch; b++) {
            float is = istd[b];
            float sum_go = 0, sum_go_xh = 0;
            for (int64_t f = 0; f < feat; f++) {
                float g = go[b * feat + f] * gd[f]; /* dx_hat component */
                sum_go += g;
                sum_go_xh += g * xh[b * feat + f];
            }
            for (int64_t f = 0; f < feat; f++) {
                float dx_hat = go[b * feat + f] * gd[f];
                float dx = (1.0f / D) * is *
                    (D * dx_hat - sum_go - xh[b * feat + f] * sum_go_xh);
                ig[b * feat + f] += dx;
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
    if (!input || input->ndim != 2) return NULL;

    /* ensure contiguous so flat indexing below is correct */
    ax_tensor_t *inp = ax_ensure_contiguous(input);
    if (!inp) return NULL;

    int64_t batch = inp->shape[0];
    int64_t feat = inp->shape[1];
    float *id = (float *)inp->storage->data;

    ax_tensor_t *out = ax_tensor_zeros(inp->shape, inp->ndim, inp->dtype);
    if (!out) { if (inp != input) ax_tensor_destroy(inp); return NULL; }
    float *od = (float *)out->storage->data;
    float *gd = (float *)ln->gamma->storage->data;
    float *bd = (float *)ln->beta->storage->data;

    bool record = ax_grad_enabled() &&
                  (input->requires_grad || ln->gamma->requires_grad || ln->beta->requires_grad);

    ax_tensor_t *x_hat_save = NULL;
    ax_tensor_t *inv_std_save = NULL;
    if (record) {
        x_hat_save = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
        int64_t is_shape[] = {batch};
        inv_std_save = ax_tensor_zeros(is_shape, 1, input->dtype);
        if (!x_hat_save || !inv_std_save) {
            if (x_hat_save) ax_tensor_destroy(x_hat_save);
            if (inv_std_save) ax_tensor_destroy(inv_std_save);
            record = false;
        }
    }

    float *xh_d = record ? (float *)x_hat_save->storage->data : NULL;
    float *is_d = record ? (float *)inv_std_save->storage->data : NULL;

    /* layernorm normalizes per-sample, so each sample is independent */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t b = 0; b < batch; b++)
    {
        /* pass 1: welford's online mean + variance */
        double w_mean = 0.0, m2 = 0.0;
        for (int64_t f = 0; f < feat; f++)
        {
            double x = (double)id[b * feat + f];
            double delta = x - w_mean;
            w_mean += delta / (double)(f + 1);
            double delta2 = x - w_mean;
            m2 += delta * delta2;
        }
        float mean = (float)w_mean;
        float var = (feat > 0) ? (float)(m2 / (double)feat) : 0.0f;
        float inv_std = 1.0f / sqrtf(var + ln->eps);

        if (record) is_d[b] = inv_std;

        /* pass 2: normalize and apply affine */
        for (int64_t f = 0; f < feat; f++)
        {
            float x_hat = (id[b * feat + f] - mean) * inv_std;
            od[b * feat + f] = gd[f] * x_hat + bd[f];
            if (record) xh_d[b * feat + f] = x_hat;
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

    for (int64_t i = 0; i < n; i++)
        ig[input->grad->offset + i] += go[grad_out->offset + i] * md[i];
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

    for (int64_t i = 0; i < n; i++)
    {
        float r = ax_rng_float();
        if (r < keep) {
            od[out->offset + i] = id[input->offset + i] * scale;
            if (record) md[i] = scale;
        }
        /* else: od stays 0, md stays 0 */
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

    double total_norm_sq = 0.0;
    for (int i = 0; i < n_params; i++)
    {
        if (!params[i] || !params[i]->grad) continue;
        int64_t n = ax_tensor_numel(params[i]->grad);
        float *gd = (float *)params[i]->grad->storage->data;
        for (int64_t j = 0; j < n; j++)
        {
            float g = gd[params[i]->grad->offset + j];
            total_norm_sq += (double)g * (double)g;
        }
    }

    float total_norm = (float)sqrt(total_norm_sq);

    if (total_norm > max_norm)
    {
        float scale = max_norm / total_norm;
        for (int i = 0; i < n_params; i++)
        {
            if (!params[i] || !params[i]->grad) continue;
            int64_t n = ax_tensor_numel(params[i]->grad);
            float *gd = (float *)params[i]->grad->storage->data;
            for (int64_t j = 0; j < n; j++)
                gd[params[i]->grad->offset + j] *= scale;
        }
    }
    return total_norm;
}
