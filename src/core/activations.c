/* activations.c — activation function implementations.
   each one does the forward pass and hooks up a backward function
   so autograd can flow gradients through them. */

#include "axiom/activations.h"
#include "axiom/autograd.h"
#include "axiom/compute.h"
#include "axiom/error.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>

/* selu constants — from the original self-normalizing networks paper */
#define SELU_ALPHA  1.6732632423543772f
#define SELU_LAMBDA 1.0507009873554805f

/* gelu approximation constant */
#define GELU_COEFF 0.044715f
#define SQRT_2_PI  0.7978845608028654f  /* sqrt(2/pi) */

/* helper to check we're working with float32 */
static inline bool check_f32(ax_tensor_t *a)
{
    if (a->dtype != AX_FLOAT32)
    {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "activation only supports float32");
        return false;
    }
    return true;
}

/* helper to allocate output and get data pointers.
   most activations follow the same pattern so this saves repetition. */
static ax_tensor_t *alloc_out(ax_tensor_t *a)
{
    return ax_tensor_zeros(a->shape, a->ndim, a->dtype);
}

static inline bool needs_grad(ax_tensor_t *a)
{
    return ax_grad_enabled() && a->requires_grad;
}


/* leaky relu */

static void leaky_relu_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *input = self->saved[0];
    float alpha = (float)self->scalar_ctx;
    int64_t n = ax_tensor_numel(grad_out);

    ax_tensor_t *grad_a = ax_tensor_zeros(grad_out->shape, grad_out->ndim, grad_out->dtype);
    if (!grad_a) return;
    float *gd = (float *)grad_a->storage->data;
    float *go = (float *)grad_out->storage->data;
    float *ad = (float *)input->storage->data;

    for (int64_t i = 0; i < n; i++)
    {
        float x = ad[input->offset + i];
        gd[grad_a->offset + i] = go[grad_out->offset + i] * (x > 0.0f ? 1.0f : alpha);
    }

    /* accumulate into input grad */
    if (self->inputs[0]->requires_grad)
    {
        if (!self->inputs[0]->grad)
            self->inputs[0]->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
        float *ig = (float *)self->inputs[0]->grad->storage->data;
        for (int64_t i = 0; i < n; i++)
            ig[self->inputs[0]->grad->offset + i] += gd[grad_a->offset + i];
    }
    ax_tensor_destroy(grad_a);
}

ax_tensor_t *ax_leaky_relu(ax_tensor_t *a, float alpha)
{
    if (!check_f32(a)) return NULL;
    ax_tensor_t *out = alloc_out(a);
    if (!out) return NULL;

    int64_t n = ax_tensor_numel(a);
    float *od = (float *)out->storage->data;
    float *ad = (float *)a->storage->data;

    for (int64_t i = 0; i < n; i++)
    {
        float x = ad[a->offset + i];
        od[out->offset + i] = x > 0.0f ? x : alpha * x;
    }

    if (needs_grad(a))
    {
        out->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(leaky_relu_backward);
        gf->inputs[0] = a;
        gf->n_inputs = 1;
        ax_tensor_t *a_safe = ax_ensure_contiguous(a);
        gf->saved[0] = a_safe;
        gf->saved_owned[0] = (a_safe != a);
        gf->n_saved = 1;
        gf->scalar_ctx = (double)alpha;
        out->grad_fn = gf;
    }
    return out;
}


/* elu */

static void elu_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *input = self->saved[0];
    ax_tensor_t *output = self->saved[1];
    float alpha = (float)self->scalar_ctx;
    int64_t n = ax_tensor_numel(grad_out);

    ax_tensor_t *grad_a = ax_tensor_zeros(grad_out->shape, grad_out->ndim, grad_out->dtype);
    if (!grad_a) return;
    float *gd = (float *)grad_a->storage->data;
    float *go = (float *)grad_out->storage->data;
    float *ad = (float *)input->storage->data;
    float *od = (float *)output->storage->data;

    for (int64_t i = 0; i < n; i++)
    {
        float x = ad[input->offset + i];
        /* elu'(x) = 1 if x > 0, else output + alpha (since output = alpha*(exp(x)-1)) */
        gd[grad_a->offset + i] = go[grad_out->offset + i] *
            (x > 0.0f ? 1.0f : od[output->offset + i] + alpha);
    }

    if (self->inputs[0]->requires_grad)
    {
        if (!self->inputs[0]->grad)
            self->inputs[0]->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
        float *ig = (float *)self->inputs[0]->grad->storage->data;
        for (int64_t i = 0; i < n; i++)
            ig[self->inputs[0]->grad->offset + i] += gd[grad_a->offset + i];
    }
    ax_tensor_destroy(grad_a);
}

ax_tensor_t *ax_elu(ax_tensor_t *a, float alpha)
{
    if (!check_f32(a)) return NULL;
    ax_tensor_t *out = alloc_out(a);
    if (!out) return NULL;

    int64_t n = ax_tensor_numel(a);
    float *od = (float *)out->storage->data;
    float *ad = (float *)a->storage->data;

    for (int64_t i = 0; i < n; i++)
    {
        float x = ad[a->offset + i];
        od[out->offset + i] = x > 0.0f ? x : alpha * (expf(x) - 1.0f);
    }

    if (needs_grad(a))
    {
        out->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(elu_backward);
        gf->inputs[0] = a;
        gf->n_inputs = 1;
        ax_tensor_t *a_safe = ax_ensure_contiguous(a);
        gf->saved[0] = a_safe;
        gf->saved_owned[0] = (a_safe != a);
        gf->saved[1] = out;
        gf->n_saved = 2;
        gf->scalar_ctx = (double)alpha;
        out->grad_fn = gf;
    }
    return out;
}


/* selu — just scaled elu with fixed constants */

ax_tensor_t *ax_selu(ax_tensor_t *a)
{
    if (!check_f32(a)) return NULL;

    /* selu(x) = lambda * elu(x, alpha) */
    ax_tensor_t *e = ax_elu(a, SELU_ALPHA);
    if (!e) return NULL;

    ax_tensor_t *out = ax_mul_scalar(e, (double)SELU_LAMBDA);
    ax_tensor_destroy(e);
    return out;
}


/* gelu (approximate) */

static void gelu_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *input = self->saved[0];
    int64_t n = ax_tensor_numel(grad_out);

    ax_tensor_t *grad_a = ax_tensor_zeros(grad_out->shape, grad_out->ndim, grad_out->dtype);
    if (!grad_a) return;
    float *gd = (float *)grad_a->storage->data;
    float *go = (float *)grad_out->storage->data;
    float *ad = (float *)input->storage->data;

    for (int64_t i = 0; i < n; i++)
    {
        float x = ad[input->offset + i];
        /* gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
           derivative is messy but well-known:
           let t = tanh(sqrt(2/pi) * (x + 0.044715*x^3))
           gelu'(x) = 0.5*(1+t) + 0.5*x*(1-t^2)*sqrt(2/pi)*(1 + 3*0.044715*x^2) */
        float inner = SQRT_2_PI * (x + GELU_COEFF * x * x * x);
        float t = tanhf(inner);
        float deriv = 0.5f * (1.0f + t) +
                      0.5f * x * (1.0f - t * t) * SQRT_2_PI * (1.0f + 3.0f * GELU_COEFF * x * x);
        gd[grad_a->offset + i] = go[grad_out->offset + i] * deriv;
    }

    if (self->inputs[0]->requires_grad)
    {
        if (!self->inputs[0]->grad)
            self->inputs[0]->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
        float *ig = (float *)self->inputs[0]->grad->storage->data;
        for (int64_t i = 0; i < n; i++)
            ig[self->inputs[0]->grad->offset + i] += gd[grad_a->offset + i];
    }
    ax_tensor_destroy(grad_a);
}

ax_tensor_t *ax_gelu(ax_tensor_t *a)
{
    if (!check_f32(a)) return NULL;
    ax_tensor_t *out = alloc_out(a);
    if (!out) return NULL;

    int64_t n = ax_tensor_numel(a);
    float *od = (float *)out->storage->data;
    float *ad = (float *)a->storage->data;

    for (int64_t i = 0; i < n; i++)
    {
        float x = ad[a->offset + i];
        float inner = SQRT_2_PI * (x + GELU_COEFF * x * x * x);
        od[out->offset + i] = 0.5f * x * (1.0f + tanhf(inner));
    }

    if (needs_grad(a))
    {
        out->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(gelu_backward);
        gf->inputs[0] = a;
        gf->n_inputs = 1;
        ax_tensor_t *a_safe = ax_ensure_contiguous(a);
        gf->saved[0] = a_safe;
        gf->saved_owned[0] = (a_safe != a);
        gf->n_saved = 1;
        out->grad_fn = gf;
    }
    return out;
}


/* swish / silu: x * sigmoid(x) */

static void swish_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *input = self->saved[0];
    ax_tensor_t *output = self->saved[1];
    int64_t n = ax_tensor_numel(grad_out);

    ax_tensor_t *grad_a = ax_tensor_zeros(grad_out->shape, grad_out->ndim, grad_out->dtype);
    if (!grad_a) return;
    float *gd = (float *)grad_a->storage->data;
    float *go = (float *)grad_out->storage->data;
    float *ad = (float *)input->storage->data;
    float *od = (float *)output->storage->data;

    for (int64_t i = 0; i < n; i++)
    {
        float x = ad[input->offset + i];
        float sig = 1.0f / (1.0f + expf(-x));
        /* swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                      = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                      = swish(x) + sigmoid(x) * (1 - swish(x)) */
        float sw = od[output->offset + i];
        float deriv = sw + sig * (1.0f - sw);
        gd[grad_a->offset + i] = go[grad_out->offset + i] * deriv;
    }

    if (self->inputs[0]->requires_grad)
    {
        if (!self->inputs[0]->grad)
            self->inputs[0]->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
        float *ig = (float *)self->inputs[0]->grad->storage->data;
        for (int64_t i = 0; i < n; i++)
            ig[self->inputs[0]->grad->offset + i] += gd[grad_a->offset + i];
    }
    ax_tensor_destroy(grad_a);
}

ax_tensor_t *ax_swish(ax_tensor_t *a)
{
    if (!check_f32(a)) return NULL;
    ax_tensor_t *out = alloc_out(a);
    if (!out) return NULL;

    int64_t n = ax_tensor_numel(a);
    float *od = (float *)out->storage->data;
    float *ad = (float *)a->storage->data;

    for (int64_t i = 0; i < n; i++)
    {
        float x = ad[a->offset + i];
        od[out->offset + i] = x / (1.0f + expf(-x));
    }

    if (needs_grad(a))
    {
        out->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(swish_backward);
        gf->inputs[0] = a;
        gf->n_inputs = 1;
        ax_tensor_t *a_safe = ax_ensure_contiguous(a);
        gf->saved[0] = a_safe;
        gf->saved_owned[0] = (a_safe != a);
        gf->saved[1] = out;
        gf->n_saved = 2;
        out->grad_fn = gf;
    }
    return out;
}


/* softplus: log(1 + exp(x)) */

static void softplus_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *input = self->saved[0];
    int64_t n = ax_tensor_numel(grad_out);

    ax_tensor_t *grad_a = ax_tensor_zeros(grad_out->shape, grad_out->ndim, grad_out->dtype);
    if (!grad_a) return;
    float *gd = (float *)grad_a->storage->data;
    float *go = (float *)grad_out->storage->data;
    float *ad = (float *)input->storage->data;

    /* softplus'(x) = sigmoid(x) = 1 / (1 + exp(-x)) */
    for (int64_t i = 0; i < n; i++)
    {
        float x = ad[input->offset + i];
        float sig = 1.0f / (1.0f + expf(-x));
        gd[grad_a->offset + i] = go[grad_out->offset + i] * sig;
    }

    if (self->inputs[0]->requires_grad)
    {
        if (!self->inputs[0]->grad)
            self->inputs[0]->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
        float *ig = (float *)self->inputs[0]->grad->storage->data;
        for (int64_t i = 0; i < n; i++)
            ig[self->inputs[0]->grad->offset + i] += gd[grad_a->offset + i];
    }
    ax_tensor_destroy(grad_a);
}

ax_tensor_t *ax_softplus(ax_tensor_t *a)
{
    if (!check_f32(a)) return NULL;
    ax_tensor_t *out = alloc_out(a);
    if (!out) return NULL;

    int64_t n = ax_tensor_numel(a);
    float *od = (float *)out->storage->data;
    float *ad = (float *)a->storage->data;

    for (int64_t i = 0; i < n; i++)
    {
        float x = ad[a->offset + i];
        /* numerically stable: if x is large, softplus(x) ~ x */
        if (x > 20.0f)
            od[out->offset + i] = x;
        else if (x < -20.0f)
            od[out->offset + i] = expf(x);
        else
            od[out->offset + i] = logf(1.0f + expf(x));
    }

    if (needs_grad(a))
    {
        out->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(softplus_backward);
        gf->inputs[0] = a;
        gf->n_inputs = 1;
        ax_tensor_t *a_safe = ax_ensure_contiguous(a);
        gf->saved[0] = a_safe;
        gf->saved_owned[0] = (a_safe != a);
        gf->n_saved = 1;
        out->grad_fn = gf;
    }
    return out;
}


/* mish: x * tanh(softplus(x))
   we compose it from existing ops so autograd works for free */

ax_tensor_t *ax_mish(ax_tensor_t *a)
{
    if (!check_f32(a)) return NULL;

    ax_tensor_t *sp = ax_softplus(a);
    if (!sp) return NULL;
    ax_tensor_t *t = ax_tanh_op(sp);
    if (!t) { ax_tensor_destroy(sp); return NULL; }
    ax_tensor_t *out = ax_mul(a, t);

    ax_tensor_destroy(sp);
    ax_tensor_destroy(t);
    return out;
}


/* softmax backward:
   grad_input = s * (grad_out - sum(grad_out * s, keepdim))
   where s is the softmax output */

static void softmax_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *input = self->inputs[0];
    ax_tensor_t *softmax_out = self->saved[0]; /* saved softmax output */

    if (!input->requires_grad) return;

    if (!input->grad)
        input->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
    if (!input->grad) return;

    float *ig = (float *)input->grad->storage->data;
    float *go = (float *)grad_out->storage->data;
    float *sd = (float *)softmax_out->storage->data;

    if (input->ndim == 1)
    {
        int64_t n = input->shape[0];
        /* dot = sum(grad_out * s) */
        float dot = 0;
        for (int64_t i = 0; i < n; i++)
            dot += go[grad_out->offset + i] * sd[softmax_out->offset + i];
        for (int64_t i = 0; i < n; i++)
            ig[input->grad->offset + i] +=
                sd[softmax_out->offset + i] * (go[grad_out->offset + i] - dot);
    }
    else if (input->ndim == 2)
    {
        int64_t rows = input->shape[0];
        int64_t cols = input->shape[1];
        for (int64_t r = 0; r < rows; r++)
        {
            float dot = 0;
            for (int64_t c = 0; c < cols; c++)
                dot += go[grad_out->offset + r * cols + c] *
                       sd[softmax_out->offset + r * cols + c];
            for (int64_t c = 0; c < cols; c++) {
                int64_t idx = r * cols + c;
                ig[input->grad->offset + idx] +=
                    sd[softmax_out->offset + idx] *
                    (go[grad_out->offset + idx] - dot);
            }
        }
    }
}


/* softmax along an axis. numerically stable version:
   softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_i - max(x))) */

ax_tensor_t *ax_softmax(ax_tensor_t *a, int axis)
{
    if (!check_f32(a)) return NULL;

    /* for now only support last-axis softmax on 1d or 2d */
    if (axis == -1) axis = a->ndim - 1;

    ax_tensor_t *out = alloc_out(a);
    if (!out) return NULL;

    float *ad = (float *)a->storage->data;
    float *od = (float *)out->storage->data;

    if (a->ndim == 1)
    {
        int64_t n = a->shape[0];

        /* find max for stability */
        float mx = -FLT_MAX;
        for (int64_t i = 0; i < n; i++)
        {
            float v = ad[a->offset + i];
            if (v > mx) mx = v;
        }

        /* exp and sum */
        float total = 0.0f;
        for (int64_t i = 0; i < n; i++)
        {
            float e = expf(ad[a->offset + i] - mx);
            od[out->offset + i] = e;
            total += e;
        }

        /* normalize */
        float inv = 1.0f / total;
        for (int64_t i = 0; i < n; i++)
            od[out->offset + i] *= inv;
    }
    else if (a->ndim == 2 && axis == 1)
    {
        /* softmax along columns for each row (most common case: batch of logits) */
        int64_t rows = a->shape[0];
        int64_t cols = a->shape[1];

        for (int64_t r = 0; r < rows; r++)
        {
            float mx = -FLT_MAX;
            for (int64_t c = 0; c < cols; c++)
            {
                float v = ad[a->offset + r * a->strides[0] + c * a->strides[1]];
                if (v > mx) mx = v;
            }

            float total = 0.0f;
            for (int64_t c = 0; c < cols; c++)
            {
                float e = expf(ad[a->offset + r * a->strides[0] + c * a->strides[1]] - mx);
                od[out->offset + r * out->strides[0] + c * out->strides[1]] = e;
                total += e;
            }

            float inv = 1.0f / total;
            for (int64_t c = 0; c < cols; c++)
                od[out->offset + r * out->strides[0] + c * out->strides[1]] *= inv;
        }
    }
    else
    {
        ax_err_set(AX_ERR_NOT_IMPLEMENTED,
                   "softmax only supports 1d or 2d (axis=1) for now");
        ax_tensor_destroy(out);
        return NULL;
    }

    /* hook up backward */
    if (needs_grad(a))
    {
        out->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(softmax_backward);
        gf->inputs[0] = a;
        gf->n_inputs = 1;
        gf->saved[0] = out;
        gf->n_saved = 1;
        gf->int_ctx = axis;
        out->grad_fn = gf;
    }

    return out;
}
