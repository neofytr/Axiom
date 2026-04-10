/* losses.c — loss function implementations.
   most are composed from existing differentiable ops
   so autograd handles the backward pass automatically. */

#include "axiom/losses.h"
#include "axiom/ops.h"
#include "axiom/autograd.h"
#include "axiom/activations.h"
#include "axiom/error.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>


/* mse: mean((pred - target)^2)
   composed from sub -> square -> mean, so autograd just works. */

ax_tensor_t *ax_mse_loss(ax_tensor_t *pred, ax_tensor_t *target)
{
    ax_tensor_t *diff = ax_sub(pred, target);
    if (!diff) return NULL;

    ax_tensor_t *sq = ax_square(diff);
    if (!sq) return NULL;

    ax_tensor_t *loss = ax_mean(sq, -1);

    /* don't destroy intermediates — backward needs them for the chain rule.
       this is a known tradeoff: compose ops = simple code, but intermediates
       stay alive until the user destroys the loss tensor.
       TODO: proper memory management with refcounting through the graph. */

    return loss;
}


/* mae: mean(|pred - target|)
   abs isn't differentiable at 0 but we handle it manually here. */

static void mae_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *pred = self->saved[0];
    ax_tensor_t *target = self->saved[1];
    int64_t n = ax_tensor_numel(pred);
    float scale = 1.0f / (float)n;

    float g = ((float *)grad_out->storage->data)[grad_out->offset];

    if (pred->requires_grad)
    {
        if (!pred->grad)
            pred->grad = ax_tensor_zeros(pred->shape, pred->ndim, pred->dtype);

        float *pg = (float *)pred->grad->storage->data;
        float *pd = (float *)pred->storage->data;
        float *td = (float *)target->storage->data;

        for (int64_t i = 0; i < n; i++)
        {
            float d = pd[pred->offset + i] - td[target->offset + i];
            float sign = (d > 0.0f) ? 1.0f : (d < 0.0f ? -1.0f : 0.0f);
            pg[pred->grad->offset + i] += g * sign * scale;
        }
    }
}

ax_tensor_t *ax_mae_loss(ax_tensor_t *pred, ax_tensor_t *target)
{
    ax_tensor_t *diff = ax_sub(pred, target);
    if (!diff) return NULL;

    ax_tensor_t *absd = ax_abs(diff);
    ax_tensor_destroy(diff);
    if (!absd) return NULL;

    /* we can't just use ax_mean here because abs doesn't have autograd.
       so we compute the mean manually and attach a custom backward. */
    int64_t n = ax_tensor_numel(absd);
    float total = 0.0f;
    float *ad = (float *)absd->storage->data;
    for (int64_t i = 0; i < n; i++)
        total += ad[absd->offset + i];

    ax_tensor_t *loss = ax_tensor_scalar(total / (float)n);
    ax_tensor_destroy(absd);

    if (ax_grad_enabled() && pred->requires_grad)
    {
        loss->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(mae_backward);
        gf->inputs[0] = pred;
        gf->n_inputs = 1;
        gf->saved[0] = pred;
        gf->saved[1] = target;
        gf->n_saved = 2;
        loss->grad_fn = gf;
    }
    return loss;
}


/* cross-entropy loss with logits.
   does log-softmax internally for numerical stability.

   log_softmax(x)_i = x_i - log(sum(exp(x_j)))
                     = x_i - max(x) - log(sum(exp(x_j - max(x))))

   loss = -sum(target * log_softmax(pred)) / batch_size */

static void ce_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *pred = self->saved[0];
    ax_tensor_t *target = self->saved[1];
    ax_tensor_t *softmax_out = self->saved[2]; /* we precomputed this */

    if (!pred->requires_grad) return;
    if (!pred->grad)
        pred->grad = ax_tensor_zeros(pred->shape, pred->ndim, pred->dtype);

    float g = ((float *)grad_out->storage->data)[grad_out->offset];
    int64_t batch = pred->shape[0];
    int64_t classes = pred->shape[1];
    float scale = g / (float)batch;

    float *pg = (float *)pred->grad->storage->data;
    float *sd = (float *)softmax_out->storage->data;
    float *td = (float *)target->storage->data;

    /* gradient of cross-entropy with softmax is simply: softmax(pred) - target
       this is one of the cleanest results in all of deep learning.
       note: pred->grad and softmax_out have their own (contiguous) strides;
       target may also have non-default strides. use each tensor's own strides. */
    for (int64_t b = 0; b < batch; b++)
    {
        for (int64_t c = 0; c < classes; c++)
        {
            int64_t grad_idx = b * pred->grad->strides[0] + c * pred->grad->strides[1];
            int64_t sm_idx = b * softmax_out->strides[0] + c * softmax_out->strides[1];
            int64_t tgt_idx = b * target->strides[0] + c * target->strides[1];
            pg[pred->grad->offset + grad_idx] +=
                (sd[softmax_out->offset + sm_idx] - td[target->offset + tgt_idx]) * scale;
        }
    }
}

ax_tensor_t *ax_cross_entropy_loss(ax_tensor_t *pred, ax_tensor_t *target)
{
    if (pred->ndim != 2 || target->ndim != 2)
    {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "cross_entropy needs [batch, classes] tensors");
        return NULL;
    }

    int64_t batch = pred->shape[0];
    int64_t classes = pred->shape[1];
    float *pd = (float *)pred->storage->data;
    float *td = (float *)target->storage->data;

    /* compute log-softmax and cross entropy in one pass */
    float total_loss = 0.0f;

    /* also compute softmax for the backward pass */
    ax_tensor_t *sm = ax_tensor_zeros(pred->shape, pred->ndim, pred->dtype);
    if (!sm) return NULL;
    float *sd = (float *)sm->storage->data;

    for (int64_t b = 0; b < batch; b++)
    {
        /* find max for stability */
        float mx = -FLT_MAX;
        for (int64_t c = 0; c < classes; c++)
        {
            float v = pd[pred->offset + b * pred->strides[0] + c * pred->strides[1]];
            if (v > mx) mx = v;
        }

        /* compute exp and sum */
        float sum_exp = 0.0f;
        for (int64_t c = 0; c < classes; c++)
        {
            float e = expf(pd[pred->offset + b * pred->strides[0] + c * pred->strides[1]] - mx);
            sd[sm->offset + b * sm->strides[0] + c * sm->strides[1]] = e;
            sum_exp += e;
        }

        /* normalize to get softmax, and accumulate loss */
        float log_sum = logf(sum_exp);
        for (int64_t c = 0; c < classes; c++)
        {
            int64_t pred_idx = b * pred->strides[0] + c * pred->strides[1];
            int64_t sm_idx = b * sm->strides[0] + c * sm->strides[1];
            int64_t tgt_idx = b * target->strides[0] + c * target->strides[1];
            sd[sm->offset + sm_idx] /= sum_exp;  /* softmax */

            float log_sm = (pd[pred->offset + pred_idx] - mx) - log_sum;
            total_loss -= td[target->offset + tgt_idx] * log_sm;
        }
    }

    ax_tensor_t *loss = ax_tensor_scalar(total_loss / (float)batch);

    if (ax_grad_enabled() && pred->requires_grad)
    {
        loss->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(ce_backward);
        gf->inputs[0] = pred;
        gf->n_inputs = 1;
        gf->saved[0] = pred;
        gf->saved[1] = target;
        gf->saved[2] = sm;
        gf->n_saved = 3;
        loss->grad_fn = gf;
    }
    else
    {
        ax_tensor_destroy(sm);
    }

    return loss;
}


/* binary cross-entropy with logits.
   numerically stable: loss = max(x, 0) - x*t + log(1 + exp(-|x|)) */

static void bce_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *pred = self->saved[0];
    ax_tensor_t *target = self->saved[1];

    if (!pred->requires_grad) return;
    if (!pred->grad)
        pred->grad = ax_tensor_zeros(pred->shape, pred->ndim, pred->dtype);

    float g = ((float *)grad_out->storage->data)[grad_out->offset];
    int64_t n = ax_tensor_numel(pred);
    float scale = g / (float)n;

    float *pg = (float *)pred->grad->storage->data;
    float *pd = (float *)pred->storage->data;
    float *td = (float *)target->storage->data;

    /* d/dpred of bce = sigmoid(pred) - target */
    for (int64_t i = 0; i < n; i++)
    {
        float x = pd[pred->offset + i];
        float sig = 1.0f / (1.0f + expf(-x));
        float t = td[target->offset + i];
        pg[pred->grad->offset + i] += (sig - t) * scale;
    }
}

ax_tensor_t *ax_bce_with_logits_loss(ax_tensor_t *pred, ax_tensor_t *target)
{
    int64_t n = ax_tensor_numel(pred);
    float *pd = (float *)pred->storage->data;
    float *td = (float *)target->storage->data;

    float total = 0.0f;
    for (int64_t i = 0; i < n; i++)
    {
        float x = pd[pred->offset + i];
        float t = td[target->offset + i];
        /* max(x,0) - x*t + log(1 + exp(-|x|)) */
        float mx = x > 0.0f ? x : 0.0f;
        float ax = x > 0.0f ? x : -x;
        total += mx - x * t + logf(1.0f + expf(-ax));
    }

    ax_tensor_t *loss = ax_tensor_scalar(total / (float)n);

    if (ax_grad_enabled() && pred->requires_grad)
    {
        loss->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(bce_backward);
        gf->inputs[0] = pred;
        gf->n_inputs = 1;
        gf->saved[0] = pred;
        gf->saved[1] = target;
        gf->n_saved = 2;
        loss->grad_fn = gf;
    }
    return loss;
}


/* huber loss: smooth transition from quadratic to linear at delta */

static void huber_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *p = self->saved[0];
    ax_tensor_t *t = self->saved[1];
    float dl = (float)self->scalar_ctx;
    int64_t n = ax_tensor_numel(p);
    float g = ((float *)grad_out->storage->data)[grad_out->offset];
    float sc = g / (float)n;

    if (!p->requires_grad) return;
    if (!p->grad)
        p->grad = ax_tensor_zeros(p->shape, p->ndim, p->dtype);

    float *pg = (float *)p->grad->storage->data;
    float *pd = (float *)p->storage->data;
    float *td = (float *)t->storage->data;

    for (int64_t i = 0; i < n; i++)
    {
        float diff = pd[p->offset + i] - td[t->offset + i];
        float adiff = diff > 0.0f ? diff : -diff;
        float deriv;
        if (adiff <= dl)
            deriv = diff;
        else
            deriv = dl * (diff > 0.0f ? 1.0f : -1.0f);
        pg[p->grad->offset + i] += deriv * sc;
    }
}

ax_tensor_t *ax_huber_loss(ax_tensor_t *pred, ax_tensor_t *target, float delta)
{
    int64_t n = ax_tensor_numel(pred);
    float *pd = (float *)pred->storage->data;
    float *td = (float *)target->storage->data;

    float total = 0.0f;
    for (int64_t i = 0; i < n; i++)
    {
        float d = pd[pred->offset + i] - td[target->offset + i];
        float ad = d > 0.0f ? d : -d;
        if (ad <= delta)
            total += 0.5f * d * d;
        else
            total += delta * (ad - 0.5f * delta);
    }

    ax_tensor_t *loss = ax_tensor_scalar(total / (float)n);

    if (ax_grad_enabled() && pred->requires_grad)
    {
        loss->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(huber_backward);
        gf->inputs[0] = pred;
        gf->n_inputs = 1;
        gf->saved[0] = pred;
        gf->saved[1] = target;
        gf->n_saved = 2;
        gf->scalar_ctx = (double)delta;
        loss->grad_fn = gf;
    }
    return loss;
}
