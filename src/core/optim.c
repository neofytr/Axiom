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
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    for (int i = 0; i < opt->n_params; i++)
    {
        ax_tensor_t *p = opt->params[i];
        if (!p->grad) continue;

        int64_t n = ax_tensor_numel(p);
        float *wd = (float *)p->storage->data;
        float *gd = (float *)p->grad->storage->data;

        /* l2 weight decay: grad += weight_decay * w */
        if (opt->weight_decay > 0.0f)
        {
            for (int64_t j = 0; j < n; j++)
                gd[p->grad->offset + j] += opt->weight_decay * wd[p->offset + j];
        }

        if (opt->momentum > 0.0f)
        {
            ensure_state_m(&opt->state[i], p);
            float *vd = (float *)opt->state[i].m->storage->data;

            for (int64_t j = 0; j < n; j++)
            {
                vd[j] = opt->momentum * vd[j] + gd[p->grad->offset + j];

                if (opt->nesterov)
                    wd[p->offset + j] -= opt->lr * (gd[p->grad->offset + j] + opt->momentum * vd[j]);
                else
                    wd[p->offset + j] -= opt->lr * vd[j];
            }
        }
        else
        {
            for (int64_t j = 0; j < n; j++)
                wd[p->offset + j] -= opt->lr * gd[p->grad->offset + j];
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

    for (int i = 0; i < opt->n_params; i++)
    {
        ax_tensor_t *p = opt->params[i];
        if (!p->grad) continue;

        ensure_state_m(&opt->state[i], p);
        ensure_state_v(&opt->state[i], p);
        int64_t n = ax_tensor_numel(p);
        float *wd = (float *)p->storage->data;
        float *gd = (float *)p->grad->storage->data;
        float *md = (float *)opt->state[i].m->storage->data;
        float *vd = (float *)opt->state[i].v->storage->data;

        /* adamw: decoupled weight decay (applied directly to weights, not grad) */
        if (decoupled_decay && opt->weight_decay > 0.0f)
        {
            for (int64_t j = 0; j < n; j++)
                wd[p->offset + j] *= (1.0f - opt->lr * opt->weight_decay);
        }

        /* regular adam: l2 regularization through gradient */
        if (!decoupled_decay && opt->weight_decay > 0.0f)
        {
            for (int64_t j = 0; j < n; j++)
                gd[p->grad->offset + j] += opt->weight_decay * wd[p->offset + j];
        }

        for (int64_t j = 0; j < n; j++)
        {
            float g = gd[p->grad->offset + j];

            /* update moments */
            md[j] = opt->beta1 * md[j] + (1.0f - opt->beta1) * g;
            vd[j] = opt->beta2 * vd[j] + (1.0f - opt->beta2) * g * g;

            /* bias-corrected moments */
            float m_hat = md[j] / bc1;
            float v_hat = vd[j] / bc2;

            /* update */
            wd[p->offset + j] -= opt->lr * m_hat / (sqrtf(v_hat) + opt->eps);
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
    for (int i = 0; i < opt->n_params; i++)
    {
        ax_tensor_t *p = opt->params[i];
        if (!p->grad) continue;

        ensure_state_v(&opt->state[i], p);

        int64_t n = ax_tensor_numel(p);
        float *wd = (float *)p->storage->data;
        float *gd = (float *)p->grad->storage->data;
        float *vd = (float *)opt->state[i].v->storage->data;

        for (int64_t j = 0; j < n; j++)
        {
            float g = gd[p->grad->offset + j];

            if (opt->weight_decay > 0.0f)
                g += opt->weight_decay * wd[p->offset + j];

            vd[j] = opt->rho * vd[j] + (1.0f - opt->rho) * g * g;
            wd[p->offset + j] -= opt->lr * g / (sqrtf(vd[j]) + opt->eps);
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
    for (int i = 0; i < opt->n_params; i++)
    {
        ax_tensor_t *p = opt->params[i];
        if (!p->grad) continue;

        ensure_state_v(&opt->state[i], p);

        int64_t n = ax_tensor_numel(p);
        float *wd = (float *)p->storage->data;
        float *gd = (float *)p->grad->storage->data;
        float *vd = (float *)opt->state[i].v->storage->data;

        for (int64_t j = 0; j < n; j++)
        {
            float g = gd[p->grad->offset + j];

            if (opt->weight_decay > 0.0f)
                g += opt->weight_decay * wd[p->offset + j];

            vd[j] += g * g;
            wd[p->offset + j] -= opt->lr * g / (sqrtf(vd[j]) + opt->eps);
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
