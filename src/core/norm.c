/* norm.c — batchnorm, layernorm, dropout, gradient clipping */

#include "axiom/norm.h"
#include "axiom/ops.h"
#include "axiom/autograd.h"
#include "axiom/compute.h"
#include "axiom/init.h"
#include "axiom/error.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


static ax_tensor_t *batchnorm_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_batchnorm_t *bn = (ax_batchnorm_t *)self;
    if (!input || input->ndim != 2) return NULL;

    int64_t batch = input->shape[0];
    int64_t feat = input->shape[1];
    float *id = (float *)input->storage->data;

    ax_tensor_t *out = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
    if (!out) return NULL;
    float *od = (float *)out->storage->data;
    float *gd = (float *)bn->gamma->storage->data;
    float *bd = (float *)bn->beta->storage->data;
    float *rm = (float *)bn->running_mean->storage->data;
    float *rv = (float *)bn->running_var->storage->data;

    if (self->training)
    {
        for (int64_t f = 0; f < feat; f++)
        {
            float sum = 0;
            for (int64_t b = 0; b < batch; b++)
                sum += id[b * feat + f];
            float mean = sum / (float)batch;

            float var_sum = 0;
            for (int64_t b = 0; b < batch; b++)
            {
                float d = id[b * feat + f] - mean;
                var_sum += d * d;
            }
            float var = var_sum / (float)batch;
            float inv_std = 1.0f / sqrtf(var + bn->eps);

            for (int64_t b = 0; b < batch; b++)
            {
                float x_hat = (id[b * feat + f] - mean) * inv_std;
                od[b * feat + f] = gd[f] * x_hat + bd[f];
            }

            rm[f] = (1.0f - bn->momentum) * rm[f] + bn->momentum * mean;
            rv[f] = (1.0f - bn->momentum) * rv[f] + bn->momentum * var;
        }
    }
    else
    {
        for (int64_t f = 0; f < feat; f++)
        {
            float inv_std = 1.0f / sqrtf(rv[f] + bn->eps);
            for (int64_t b = 0; b < batch; b++)
            {
                float x_hat = (id[b * feat + f] - rm[f]) * inv_std;
                od[b * feat + f] = gd[f] * x_hat + bd[f];
            }
        }
    }
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
    bn->base.input_features = num_features;
    bn->base.output_features = num_features;

    return (ax_layer_t *)bn;
}


static ax_tensor_t *layernorm_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_layernorm_t *ln = (ax_layernorm_t *)self;
    if (!input || input->ndim != 2) return NULL;

    int64_t batch = input->shape[0];
    int64_t feat = input->shape[1];
    float *id = (float *)input->storage->data;

    ax_tensor_t *out = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
    if (!out) return NULL;
    float *od = (float *)out->storage->data;
    float *gd = (float *)ln->gamma->storage->data;
    float *bd = (float *)ln->beta->storage->data;

    for (int64_t b = 0; b < batch; b++)
    {
        float sum = 0;
        for (int64_t f = 0; f < feat; f++)
            sum += id[b * feat + f];
        float mean = sum / (float)feat;

        float var_sum = 0;
        for (int64_t f = 0; f < feat; f++)
        {
            float d = id[b * feat + f] - mean;
            var_sum += d * d;
        }
        float var = var_sum / (float)feat;
        float inv_std = 1.0f / sqrtf(var + ln->eps);

        for (int64_t f = 0; f < feat; f++)
        {
            float x_hat = (id[b * feat + f] - mean) * inv_std;
            od[b * feat + f] = gd[f] * x_hat + bd[f];
        }
    }
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

static bool dropout_seeded = false;

static ax_tensor_t *dropout_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_dropout_t *dp = (ax_dropout_t *)self;

    if (!self->training || dp->p <= 0.0f)
        return ax_tensor_view(input);

    if (!dropout_seeded) { srand((unsigned)time(NULL)); dropout_seeded = true; }

    ax_tensor_t *out = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
    if (!out) return NULL;

    int64_t n = ax_tensor_numel(input);
    float *id = (float *)input->storage->data;
    float *od = (float *)out->storage->data;
    float keep = 1.0f - dp->p;
    float scale = 1.0f / keep;

    for (int64_t i = 0; i < n; i++)
    {
        float r = (float)rand() / (float)RAND_MAX;
        if (r < keep)
            od[out->offset + i] = id[input->offset + i] * scale;
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
