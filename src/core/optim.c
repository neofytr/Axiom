/* optim.c — optimizer implementations.
   each one follows the same pattern:
   1. iterate over parameters
   2. skip if no gradient
   3. apply weight decay (if any)
   4. update using the optimizer-specific rule
   5. increment step count */

#include "axiom/optim.h"
#include "axiom/compute.h"
#include "axiom/error.h"
#include "../compute/backends/simd_defs.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* allocate optimizer base struct + state array */
static ax_optimizer_t *optim_alloc(ax_optim_type_t type, ax_tensor_t **params, int n)
{
    ax_optimizer_t *opt = calloc(1, sizeof(ax_optimizer_t));
    if (!opt) return NULL;

    opt->type = type;
    opt->params = params;
    opt->n_params = n;
    opt->state = calloc(n, sizeof(ax_param_state_t));
    if (!opt->state)
    {
        free(opt);
        return NULL;
    }
    return opt;
}

/* lazily init per-param state tensors (zeros, same shape as param) */
static void ensure_state_m(ax_param_state_t *s, ax_tensor_t *p)
{
    if (!s->m)
        s->m = ax_tensor_zeros(p->shape, p->ndim, p->dtype);
}

static void ensure_state_v(ax_param_state_t *s, ax_tensor_t *p)
{
    if (!s->v)
        s->v = ax_tensor_zeros(p->shape, p->ndim, p->dtype);
}


/* sgd with optional momentum and nesterov.

   without momentum:
     w = w - lr * grad

   with momentum:
     v = momentum * v + grad
     w = w - lr * v

   with nesterov:
     v = momentum * v + grad
     w = w - lr * (grad + momentum * v) */

ax_optimizer_t *ax_sgd_create(ax_tensor_t **params, int n_params,
                              float lr, float momentum, float weight_decay,
                              bool nesterov)
{
    ax_optimizer_t *opt = optim_alloc(AX_OPTIM_SGD, params, n_params);
    if (!opt) return NULL;
    opt->lr = lr;
    opt->momentum = momentum;
    opt->weight_decay = weight_decay;
    opt->nesterov = nesterov;
    return opt;
}

static void sgd_step(ax_optimizer_t *opt)
{
    ax_vf32 v_lr   = ax_vf32_set1(opt->lr);
    ax_vf32 v_wd   = ax_vf32_set1(opt->weight_decay);
    ax_vf32 v_mom  = ax_vf32_set1(opt->momentum);

    /* serial pre-pass: lazily allocate momentum state for params that need it.
       avoids ax_tensor_zeros calls inside the parallel region. */
    if (opt->momentum > 0.0f) {
        for (int i = 0; i < opt->n_params; i++) {
            if (opt->params[i]->grad)
                ensure_state_m(&opt->state[i], opt->params[i]);
        }
    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
    #endif
    for (int i = 0; i < opt->n_params; i++)
    {
        ax_tensor_t *p = opt->params[i];
        if (!p->grad) continue;

        int64_t n = ax_tensor_numel(p);
        int64_t wo = p->offset;
        int64_t go = p->grad->offset;
        float *wd = (float *)p->storage->data;
        float *gd = (float *)p->grad->storage->data;

        /* fuse weight decay into grad if needed */
        if (opt->weight_decay > 0.0f)
        {
            int64_t j = 0, ve = n - (n % AX_VF32_WIDTH);
            for (; j < ve; j += AX_VF32_WIDTH) {
                ax_vf32 g = ax_vf32_load(gd + go + j);
                ax_vf32 w = ax_vf32_load(wd + wo + j);
                ax_vf32_store(gd + go + j, ax_vf32_fmadd(v_wd, w, g));
            }
            for (; j < n; j++)
                gd[go + j] += opt->weight_decay * wd[wo + j];
        }

        if (opt->momentum > 0.0f)
        {
            ensure_state_m(&opt->state[i], p);
            float *vd = (float *)opt->state[i].m->storage->data;

            if (opt->nesterov)
            {
                int64_t j = 0, ve = n - (n % AX_VF32_WIDTH);
                for (; j < ve; j += AX_VF32_WIDTH) {
                    ax_vf32 g = ax_vf32_load(gd + go + j);
                    ax_vf32 v = ax_vf32_load(vd + j);
                    v = ax_vf32_fmadd(v_mom, v, g); /* v = mom*v + g */
                    ax_vf32_store(vd + j, v);
                    ax_vf32 upd = ax_vf32_fmadd(v_mom, v, g); /* g + mom*v */
                    ax_vf32 w = ax_vf32_load(wd + wo + j);
                    ax_vf32_store(wd + wo + j, ax_vf32_sub(w, ax_vf32_mul(v_lr, upd)));
                }
                for (; j < n; j++) {
                    vd[j] = opt->momentum * vd[j] + gd[go + j];
                    wd[wo + j] -= opt->lr * (gd[go + j] + opt->momentum * vd[j]);
                }
            }
            else
            {
                int64_t j = 0, ve = n - (n % AX_VF32_WIDTH);
                for (; j < ve; j += AX_VF32_WIDTH) {
                    ax_vf32 g = ax_vf32_load(gd + go + j);
                    ax_vf32 v = ax_vf32_fmadd(v_mom, ax_vf32_load(vd + j), g);
                    ax_vf32_store(vd + j, v);
                    ax_vf32 w = ax_vf32_load(wd + wo + j);
                    ax_vf32_store(wd + wo + j, ax_vf32_sub(w, ax_vf32_mul(v_lr, v)));
                }
                for (; j < n; j++) {
                    vd[j] = opt->momentum * vd[j] + gd[go + j];
                    wd[wo + j] -= opt->lr * vd[j];
                }
            }
        }
        else
        {
            int64_t j = 0, ve = n - (n % AX_VF32_WIDTH);
            for (; j < ve; j += AX_VF32_WIDTH) {
                ax_vf32 w = ax_vf32_load(wd + wo + j);
                ax_vf32 g = ax_vf32_load(gd + go + j);
                ax_vf32_store(wd + wo + j, ax_vf32_sub(w, ax_vf32_mul(v_lr, g)));
            }
            for (; j < n; j++)
                wd[wo + j] -= opt->lr * gd[go + j];
        }

        opt->state[i].step_count++;
    }
}


/* adam: adaptive moment estimation.

   m = beta1 * m + (1 - beta1) * grad          (first moment, like momentum)
   v = beta2 * v + (1 - beta2) * grad^2        (second moment, per-param learning rate)
   m_hat = m / (1 - beta1^t)                    (bias correction)
   v_hat = v / (1 - beta2^t)                    (bias correction)
   w = w - lr * m_hat / (sqrt(v_hat) + eps)     (update) */

ax_optimizer_t *ax_adam_create(ax_tensor_t **params, int n_params,
                               float lr, float beta1, float beta2,
                               float eps, float weight_decay)
{
    ax_optimizer_t *opt = optim_alloc(AX_OPTIM_ADAM, params, n_params);
    if (!opt) return NULL;
    opt->lr = lr;
    opt->beta1 = beta1 > 0 ? beta1 : 0.9f;
    opt->beta2 = beta2 > 0 ? beta2 : 0.999f;
    opt->eps = eps > 0 ? eps : 1e-8f;
    opt->weight_decay = weight_decay;
    return opt;
}

static void adam_step(ax_optimizer_t *opt, bool decoupled_decay)
{
    /* increment global step once per call — all parameters share the same t.
       per-parameter step counts would give different bias corrections to different
       params (e.g. if some have no grad for a batch), which is incorrect. */
    opt->global_step++;
    int64_t t = opt->global_step;

    float bc1 = 1.0f - powf(opt->beta1, (float)t);
    float bc2 = 1.0f - powf(opt->beta2, (float)t);

    /* serial pre-pass: lazily allocate moments for params that need them */
    for (int i = 0; i < opt->n_params; i++) {
        if (opt->params[i]->grad) {
            ensure_state_m(&opt->state[i], opt->params[i]);
            ensure_state_v(&opt->state[i], opt->params[i]);
        }
    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
    #endif
    for (int i = 0; i < opt->n_params; i++)
    {
        ax_tensor_t *p = opt->params[i];
        if (!p->grad) continue;
        int64_t n = ax_tensor_numel(p);
        float *wd = (float *)p->storage->data;
        float *gd = (float *)p->grad->storage->data;
        float *md = (float *)opt->state[i].m->storage->data;
        float *vd = (float *)opt->state[i].v->storage->data;

        /* fused adam: weight decay + moment update + weight update in one pass.
           parameter tensors are always contiguous (created by layer constructors). */
        float decay_scale = decoupled_decay ? (1.0f - opt->lr * opt->weight_decay) : 1.0f;
        if (decay_scale < 0.0f) decay_scale = 0.0f; /* clamp for extreme hyperparams */
        float wd_grad = (!decoupled_decay && opt->weight_decay > 0.0f) ? opt->weight_decay : 0.0f;
        int64_t wo = p->offset;
        int64_t go = p->grad->offset;

        ax_vf32 v_beta1   = ax_vf32_set1(opt->beta1);
        ax_vf32 v_beta2   = ax_vf32_set1(opt->beta2);
        ax_vf32 v_1mb1    = ax_vf32_set1(1.0f - opt->beta1);
        ax_vf32 v_1mb2    = ax_vf32_set1(1.0f - opt->beta2);
        ax_vf32 v_lr      = ax_vf32_set1(opt->lr);
        ax_vf32 v_eps     = ax_vf32_set1(opt->eps);
        ax_vf32 v_bc1     = ax_vf32_set1(bc1);
        ax_vf32 v_bc2     = ax_vf32_set1(bc2);
        ax_vf32 v_decay   = ax_vf32_set1(decay_scale);
        ax_vf32 v_wd_grad = ax_vf32_set1(wd_grad);

        int64_t j = 0;
        int64_t vec_end = n - (n % AX_VF32_WIDTH);
        for (; j < vec_end; j += AX_VF32_WIDTH)
        {
            ax_vf32 w = ax_vf32_load(wd + wo + j);
            ax_vf32 g = ax_vf32_load(gd + go + j);
            ax_vf32 m = ax_vf32_load(md + j);
            ax_vf32 v = ax_vf32_load(vd + j);

            /* decoupled weight decay */
            w = ax_vf32_mul(w, v_decay);
            /* l2 through gradient */
            g = ax_vf32_fmadd(v_wd_grad, w, g);

            /* moment updates */
            m = ax_vf32_fmadd(v_1mb1, g, ax_vf32_mul(v_beta1, m));
            v = ax_vf32_fmadd(v_1mb2, ax_vf32_mul(g, g), ax_vf32_mul(v_beta2, v));

            /* bias-corrected update */
            ax_vf32 m_hat = ax_vf32_div(m, v_bc1);
            ax_vf32 v_hat = ax_vf32_div(v, v_bc2);
            ax_vf32 denom = ax_vf32_add(ax_vf32_sqrt(v_hat), v_eps);
            w = ax_vf32_sub(w, ax_vf32_mul(v_lr, ax_vf32_div(m_hat, denom)));

            ax_vf32_store(wd + wo + j, w);
            ax_vf32_store(md + j, m);
            ax_vf32_store(vd + j, v);
        }
        /* scalar tail */
        for (; j < n; j++)
        {
            float w = wd[wo + j] * decay_scale;
            float g = gd[go + j] + wd_grad * w;
            md[j] = opt->beta1 * md[j] + (1.0f - opt->beta1) * g;
            vd[j] = opt->beta2 * vd[j] + (1.0f - opt->beta2) * g * g;
            float m_hat = md[j] / bc1;
            float v_hat = vd[j] / bc2;
            wd[wo + j] = w - opt->lr * m_hat / (sqrtf(v_hat) + opt->eps);
        }
    }
}


/* adamw: adam with decoupled weight decay.
   the only difference from adam is that weight decay is applied
   directly to the weights, not through the gradient. */

ax_optimizer_t *ax_adamw_create(ax_tensor_t **params, int n_params,
                                float lr, float beta1, float beta2,
                                float eps, float weight_decay)
{
    ax_optimizer_t *opt = ax_adam_create(params, n_params, lr, beta1, beta2, eps, weight_decay);
    if (!opt) return NULL;
    opt->type = AX_OPTIM_ADAMW;
    return opt;
}


/* rmsprop: running mean of squared gradients.
   v = rho * v + (1 - rho) * grad^2
   w = w - lr * grad / (sqrt(v) + eps) */

ax_optimizer_t *ax_rmsprop_create(ax_tensor_t **params, int n_params,
                                  float lr, float rho, float eps,
                                  float weight_decay)
{
    ax_optimizer_t *opt = optim_alloc(AX_OPTIM_RMSPROP, params, n_params);
    if (!opt) return NULL;
    opt->lr = lr;
    opt->rho = rho > 0 ? rho : 0.99f;
    opt->eps = eps > 0 ? eps : 1e-8f;
    opt->weight_decay = weight_decay;
    return opt;
}

static void rmsprop_step(ax_optimizer_t *opt)
{
    ax_vf32 v_rho   = ax_vf32_set1(opt->rho);
    ax_vf32 v_1mr   = ax_vf32_set1(1.0f - opt->rho);
    ax_vf32 v_lr    = ax_vf32_set1(opt->lr);
    ax_vf32 v_eps   = ax_vf32_set1(opt->eps);
    ax_vf32 v_wd    = ax_vf32_set1(opt->weight_decay);

    for (int i = 0; i < opt->n_params; i++) {
        if (opt->params[i]->grad)
            ensure_state_v(&opt->state[i], opt->params[i]);
    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
    #endif
    for (int i = 0; i < opt->n_params; i++)
    {
        ax_tensor_t *p = opt->params[i];
        if (!p->grad) continue;

        ensure_state_v(&opt->state[i], p);

        int64_t n = ax_tensor_numel(p);
        int64_t wo = p->offset, go = p->grad->offset;
        float *wd = (float *)p->storage->data;
        float *gd = (float *)p->grad->storage->data;
        float *vd = (float *)opt->state[i].v->storage->data;

        int64_t j = 0, ve = n - (n % AX_VF32_WIDTH);
        for (; j < ve; j += AX_VF32_WIDTH)
        {
            ax_vf32 g = ax_vf32_load(gd + go + j);
            ax_vf32 w = ax_vf32_load(wd + wo + j);
            if (opt->weight_decay > 0.0f)
                g = ax_vf32_fmadd(v_wd, w, g);
            ax_vf32 v = ax_vf32_fmadd(v_1mr, ax_vf32_mul(g, g), ax_vf32_mul(v_rho, ax_vf32_load(vd + j)));
            ax_vf32_store(vd + j, v);
            ax_vf32 denom = ax_vf32_add(ax_vf32_sqrt(v), v_eps);
            ax_vf32_store(wd + wo + j, ax_vf32_sub(w, ax_vf32_mul(v_lr, ax_vf32_div(g, denom))));
        }
        for (; j < n; j++)
        {
            float g = gd[go + j];
            if (opt->weight_decay > 0.0f) g += opt->weight_decay * wd[wo + j];
            vd[j] = opt->rho * vd[j] + (1.0f - opt->rho) * g * g;
            wd[wo + j] -= opt->lr * g / (sqrtf(vd[j]) + opt->eps);
        }
        opt->state[i].step_count++;
    }
}


/* adagrad: accumulates squared gradients.
   v = v + grad^2
   w = w - lr * grad / (sqrt(v) + eps)

   the key property: learning rate decays over time for frequently updated params. */

ax_optimizer_t *ax_adagrad_create(ax_tensor_t **params, int n_params,
                                  float lr, float eps, float weight_decay)
{
    ax_optimizer_t *opt = optim_alloc(AX_OPTIM_ADAGRAD, params, n_params);
    if (!opt) return NULL;
    opt->lr = lr;
    opt->eps = eps > 0 ? eps : 1e-10f;
    opt->weight_decay = weight_decay;
    return opt;
}

static void adagrad_step(ax_optimizer_t *opt)
{
    ax_vf32 v_lr  = ax_vf32_set1(opt->lr);
    ax_vf32 v_eps = ax_vf32_set1(opt->eps);
    ax_vf32 v_wd  = ax_vf32_set1(opt->weight_decay);

    for (int i = 0; i < opt->n_params; i++) {
        if (opt->params[i]->grad)
            ensure_state_v(&opt->state[i], opt->params[i]);
    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
    #endif
    for (int i = 0; i < opt->n_params; i++)
    {
        ax_tensor_t *p = opt->params[i];
        if (!p->grad) continue;

        int64_t n = ax_tensor_numel(p);
        int64_t wo = p->offset, go = p->grad->offset;
        float *wd = (float *)p->storage->data;
        float *gd = (float *)p->grad->storage->data;
        float *vd = (float *)opt->state[i].v->storage->data;

        int64_t j = 0, ve = n - (n % AX_VF32_WIDTH);
        for (; j < ve; j += AX_VF32_WIDTH)
        {
            ax_vf32 g = ax_vf32_load(gd + go + j);
            ax_vf32 w = ax_vf32_load(wd + wo + j);
            if (opt->weight_decay > 0.0f)
                g = ax_vf32_fmadd(v_wd, w, g);
            ax_vf32 v = ax_vf32_add(ax_vf32_load(vd + j), ax_vf32_mul(g, g));
            ax_vf32_store(vd + j, v);
            ax_vf32 denom = ax_vf32_add(ax_vf32_sqrt(v), v_eps);
            ax_vf32_store(wd + wo + j, ax_vf32_sub(w, ax_vf32_mul(v_lr, ax_vf32_div(g, denom))));
        }
        for (; j < n; j++)
        {
            float g = gd[go + j];
            if (opt->weight_decay > 0.0f) g += opt->weight_decay * wd[wo + j];
            vd[j] += g * g;
            wd[wo + j] -= opt->lr * g / (sqrtf(vd[j]) + opt->eps);
        }
        opt->state[i].step_count++;
    }
}


/* dispatch to the right step function */
void ax_optimizer_step(ax_optimizer_t *opt)
{
    if (!opt) return;
    switch (opt->type)
    {
        case AX_OPTIM_SGD:     sgd_step(opt); break;
        case AX_OPTIM_ADAM:    adam_step(opt, false); break;
        case AX_OPTIM_ADAMW:   adam_step(opt, true); break;
        case AX_OPTIM_RMSPROP: rmsprop_step(opt); break;
        case AX_OPTIM_ADAGRAD: adagrad_step(opt); break;
    }
}

void ax_optimizer_zero_grad(ax_optimizer_t *opt)
{
    if (!opt) return;
    for (int i = 0; i < opt->n_params; i++)
    {
        if (opt->params[i]->grad)
            ax_compute_fill(opt->params[i]->grad, 0.0);
    }
}

void ax_optimizer_set_lr(ax_optimizer_t *opt, float lr) { if (opt) opt->lr = lr; }
float ax_optimizer_get_lr(ax_optimizer_t *opt) { return opt ? opt->lr : 0.0f; }

void ax_optimizer_destroy(ax_optimizer_t *opt)
{
    if (!opt) return;
    for (int i = 0; i < opt->n_params; i++)
    {
        if (opt->state[i].m) ax_tensor_destroy(opt->state[i].m);
        if (opt->state[i].v) ax_tensor_destroy(opt->state[i].v);
    }
    free(opt->state);
    free(opt);
}
