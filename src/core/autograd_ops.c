/* autograd_ops.c — backward functions for each differentiable operation.

   each op gets two things:
   1. a backward function that computes input gradients from output gradient
   2. a "make" function that creates the grad_fn and saves what backward needs

   the gradient math follows standard calculus:
     d/da (a + b)  = 1
     d/da (a * b)  = b
     d/da (a @ b)  = grad_out @ b^T    (matmul)
     d/da sigmoid(a) = sigmoid(a) * (1 - sigmoid(a))
     etc. */

#include "axiom/autograd.h"
#include "axiom/compute.h"
#include "axiom/broadcast.h"
#include "axiom/error.h"
#include <stdlib.h>
#include <string.h>

/* helper: make sure a tensor has a grad tensor allocated and ready to accumulate into */
static void ensure_grad(ax_tensor_t *t)
{
    if (!t->requires_grad) return;
    if (!t->grad)
    {
        t->grad = ax_tensor_zeros(t->shape, t->ndim, t->dtype);
    }
}

/* helper: accumulate grad_to_add into t->grad (element-wise addition).
   handles the case where grad_to_add has been broadcast and needs
   to be reduced back to match t's shape. */
static void accumulate_grad(ax_tensor_t *t, ax_tensor_t *grad_to_add)
{
    if (!t->requires_grad) return;
    ensure_grad(t);

    /* if shapes match exactly, just add */
    if (t->grad->ndim == grad_to_add->ndim)
    {
        bool same = true;
        for (int i = 0; i < t->grad->ndim; i++)
        {
            if (t->grad->shape[i] != grad_to_add->shape[i])
            {
                same = false;
                break;
            }
        }
        if (same)
        {
            /* straightforward accumulation */
            int64_t n = ax_tensor_numel(t->grad);
            float *gd = (float *)t->grad->storage->data;
            float *ad = (float *)grad_to_add->storage->data;
            for (int64_t i = 0; i < n; i++)
            {
                gd[t->grad->offset + i] += ad[grad_to_add->offset + i];
            }
            return;
        }
    }

    /* shapes differ due to broadcasting — need to sum out the broadcast dims.
       gradient flows backwards: if a was broadcast from [3] to [2,3],
       we need to sum the [2,3] gradient along axis 0 to get [3]. */
    int64_t n_grad = ax_tensor_numel(grad_to_add);
    float *gd = (float *)t->grad->storage->data;
    float *ad = (float *)grad_to_add->storage->data;

    /* for each element in the larger gradient, figure out which
       element in the smaller param gradient it maps to */
    for (int64_t i = 0; i < n_grad; i++)
    {
        int64_t remaining = i;
        int64_t param_flat = 0;

        /* walk from right to left, mapping indices */
        for (int d = grad_to_add->ndim - 1; d >= 0; d--)
        {
            int64_t idx = remaining % grad_to_add->shape[d];
            remaining /= grad_to_add->shape[d];

            /* corresponding dim in param (right-aligned) */
            int pd = d - (grad_to_add->ndim - t->grad->ndim);
            if (pd >= 0)
            {
                int64_t pidx = (t->grad->shape[pd] == 1) ? 0 : idx;
                /* compute flat index contribution */
                int64_t ps = 1;
                for (int k = t->grad->ndim - 1; k > pd; k--) ps *= t->grad->shape[k];
                param_flat += pidx * ps;
            }
        }

        gd[t->grad->offset + param_flat] += ad[grad_to_add->offset + i];
    }
}


/* add: d/da(a+b) = 1, d/db(a+b) = 1
   gradient just passes through unchanged */
static void add_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    accumulate_grad(self->inputs[0], grad_out);
    accumulate_grad(self->inputs[1], grad_out);
}

ax_grad_fn_t *ax_make_add_backward(ax_tensor_t *a, ax_tensor_t *b, ax_tensor_t *out)
{
    ax_grad_fn_t *gf = ax_grad_fn_create(add_backward);
    gf->inputs[0] = a;
    gf->inputs[1] = b;
    gf->n_inputs = 2;
    return gf;
}


/* sub: d/da(a-b) = 1, d/db(a-b) = -1 */
static void sub_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    accumulate_grad(self->inputs[0], grad_out);

    /* negate grad for b */
    ax_tensor_t *neg_grad = ax_tensor_zeros(grad_out->shape, grad_out->ndim, grad_out->dtype);
    ax_compute_neg(grad_out, neg_grad);
    accumulate_grad(self->inputs[1], neg_grad);
    ax_tensor_destroy(neg_grad);
}

ax_grad_fn_t *ax_make_sub_backward(ax_tensor_t *a, ax_tensor_t *b, ax_tensor_t *out)
{
    ax_grad_fn_t *gf = ax_grad_fn_create(sub_backward);
    gf->inputs[0] = a;
    gf->inputs[1] = b;
    gf->n_inputs = 2;
    return gf;
}


/* mul: d/da(a*b) = b, d/db(a*b) = a
   saved[0] = a, saved[1] = b */
static void mul_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *a = self->saved[0];
    ax_tensor_t *b = self->saved[1];

    /* grad_a = grad_out * b */
    ax_tensor_t *grad_a = ax_tensor_zeros(grad_out->shape, grad_out->ndim, grad_out->dtype);
    ax_compute_mul(grad_out, b, grad_a);
    accumulate_grad(self->inputs[0], grad_a);
    ax_tensor_destroy(grad_a);

    /* grad_b = grad_out * a */
    ax_tensor_t *grad_b = ax_tensor_zeros(grad_out->shape, grad_out->ndim, grad_out->dtype);
    ax_compute_mul(grad_out, a, grad_b);
    accumulate_grad(self->inputs[1], grad_b);
    ax_tensor_destroy(grad_b);
}

ax_grad_fn_t *ax_make_mul_backward(ax_tensor_t *a, ax_tensor_t *b, ax_tensor_t *out)
{
    ax_grad_fn_t *gf = ax_grad_fn_create(mul_backward);
    gf->inputs[0] = a;
    gf->inputs[1] = b;
    gf->n_inputs = 2;
    gf->saved[0] = a;
    gf->saved[1] = b;
    gf->n_saved = 2;
    return gf;
}


/* div: d/da(a/b) = 1/b, d/db(a/b) = -a/b^2
   saved[0] = a, saved[1] = b */
static void div_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *a = self->saved[0];
    ax_tensor_t *b = self->saved[1];

    /* grad_a = grad_out / b */
    ax_tensor_t *grad_a = ax_tensor_zeros(grad_out->shape, grad_out->ndim, grad_out->dtype);
    ax_compute_div(grad_out, b, grad_a);
    accumulate_grad(self->inputs[0], grad_a);
    ax_tensor_destroy(grad_a);

    /* grad_b = -grad_out * a / b^2 */
    ax_tensor_t *b_sq = ax_tensor_zeros(b->shape, b->ndim, b->dtype);
    ax_compute_square(b, b_sq);

    ax_tensor_t *tmp = ax_tensor_zeros(grad_out->shape, grad_out->ndim, grad_out->dtype);
    ax_compute_mul(grad_out, a, tmp);
    ax_compute_div(tmp, b_sq, tmp);
    ax_compute_neg(tmp, tmp);
    accumulate_grad(self->inputs[1], tmp);

    ax_tensor_destroy(b_sq);
    ax_tensor_destroy(tmp);
}

ax_grad_fn_t *ax_make_div_backward(ax_tensor_t *a, ax_tensor_t *b, ax_tensor_t *out)
{
    ax_grad_fn_t *gf = ax_grad_fn_create(div_backward);
    gf->inputs[0] = a;
    gf->inputs[1] = b;
    gf->n_inputs = 2;
    gf->saved[0] = a;
    gf->saved[1] = b;
    gf->n_saved = 2;
    return gf;
}


/* neg: d/da(-a) = -1 */
static void neg_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *neg_grad = ax_tensor_zeros(grad_out->shape, grad_out->ndim, grad_out->dtype);
    ax_compute_neg(grad_out, neg_grad);
    accumulate_grad(self->inputs[0], neg_grad);
    ax_tensor_destroy(neg_grad);
}

ax_grad_fn_t *ax_make_neg_backward(ax_tensor_t *a, ax_tensor_t *out)
{
    ax_grad_fn_t *gf = ax_grad_fn_create(neg_backward);
    gf->inputs[0] = a;
    gf->n_inputs = 1;
    return gf;
}


/* exp: d/da(exp(a)) = exp(a)
   we save the output since exp(a) was already computed */
static void exp_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *exp_out = self->saved[0]; /* = exp(a) */
    ax_tensor_t *grad_a = ax_tensor_zeros(grad_out->shape, grad_out->ndim, grad_out->dtype);
    ax_compute_mul(grad_out, exp_out, grad_a);
    accumulate_grad(self->inputs[0], grad_a);
    ax_tensor_destroy(grad_a);
}

ax_grad_fn_t *ax_make_exp_backward(ax_tensor_t *a, ax_tensor_t *out)
{
    ax_grad_fn_t *gf = ax_grad_fn_create(exp_backward);
    gf->inputs[0] = a;
    gf->n_inputs = 1;
    gf->saved[0] = out; /* save exp(a) */
    gf->n_saved = 1;
    return gf;
}


/* log: d/da(log(a)) = 1/a
   saved[0] = a (the input) */
static void log_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *a = self->saved[0];
    ax_tensor_t *grad_a = ax_tensor_zeros(grad_out->shape, grad_out->ndim, grad_out->dtype);
    ax_compute_div(grad_out, a, grad_a);
    accumulate_grad(self->inputs[0], grad_a);
    ax_tensor_destroy(grad_a);
}

ax_grad_fn_t *ax_make_log_backward(ax_tensor_t *a, ax_tensor_t *out)
{
    ax_grad_fn_t *gf = ax_grad_fn_create(log_backward);
    gf->inputs[0] = a;
    gf->n_inputs = 1;
    gf->saved[0] = a;
    gf->n_saved = 1;
    return gf;
}


/* sqrt: d/da(sqrt(a)) = 1 / (2 * sqrt(a))
   saved[0] = out (= sqrt(a)) */
static void sqrt_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *sqrt_out = self->saved[0];

    /* grad = grad_out / (2 * sqrt(a)) */
    ax_tensor_t *denom = ax_tensor_zeros(sqrt_out->shape, sqrt_out->ndim, sqrt_out->dtype);
    ax_compute_mul_scalar(sqrt_out, 2.0, denom);

    ax_tensor_t *grad_a = ax_tensor_zeros(grad_out->shape, grad_out->ndim, grad_out->dtype);
    ax_compute_div(grad_out, denom, grad_a);

    accumulate_grad(self->inputs[0], grad_a);
    ax_tensor_destroy(denom);
    ax_tensor_destroy(grad_a);
}

ax_grad_fn_t *ax_make_sqrt_backward(ax_tensor_t *a, ax_tensor_t *out)
{
    ax_grad_fn_t *gf = ax_grad_fn_create(sqrt_backward);
    gf->inputs[0] = a;
    gf->n_inputs = 1;
    gf->saved[0] = out;
    gf->n_saved = 1;
    return gf;
}


/* square: d/da(a^2) = 2a
   saved[0] = a */
static void square_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *a = self->saved[0];

    /* grad = grad_out * 2 * a */
    ax_tensor_t *two_a = ax_tensor_zeros(a->shape, a->ndim, a->dtype);
    ax_compute_mul_scalar(a, 2.0, two_a);

    ax_tensor_t *grad_a = ax_tensor_zeros(grad_out->shape, grad_out->ndim, grad_out->dtype);
    ax_compute_mul(grad_out, two_a, grad_a);

    accumulate_grad(self->inputs[0], grad_a);
    ax_tensor_destroy(two_a);
    ax_tensor_destroy(grad_a);
}

ax_grad_fn_t *ax_make_square_backward(ax_tensor_t *a, ax_tensor_t *out)
{
    ax_grad_fn_t *gf = ax_grad_fn_create(square_backward);
    gf->inputs[0] = a;
    gf->n_inputs = 1;
    gf->saved[0] = a;
    gf->n_saved = 1;
    return gf;
}


/* add_scalar: d/da(a + c) = 1 (same as add, scalar doesn't matter) */
static void add_scalar_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    accumulate_grad(self->inputs[0], grad_out);
}

ax_grad_fn_t *ax_make_add_scalar_backward(ax_tensor_t *a, ax_tensor_t *out)
{
    ax_grad_fn_t *gf = ax_grad_fn_create(add_scalar_backward);
    gf->inputs[0] = a;
    gf->n_inputs = 1;
    return gf;
}


/* mul_scalar: d/da(a * c) = c */
static void mul_scalar_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    double c = self->scalar_ctx;
    ax_tensor_t *grad_a = ax_tensor_zeros(grad_out->shape, grad_out->ndim, grad_out->dtype);
    ax_compute_mul_scalar(grad_out, c, grad_a);
    accumulate_grad(self->inputs[0], grad_a);
    ax_tensor_destroy(grad_a);
}

ax_grad_fn_t *ax_make_mul_scalar_backward(ax_tensor_t *a, double s, ax_tensor_t *out)
{
    ax_grad_fn_t *gf = ax_grad_fn_create(mul_scalar_backward);
    gf->inputs[0] = a;
    gf->n_inputs = 1;
    gf->scalar_ctx = s;
    return gf;
}


/* matmul: out = a @ b
   d/da = grad_out @ b^T
   d/db = a^T @ grad_out
   saved[0] = a, saved[1] = b */
static void matmul_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *a = self->saved[0];
    ax_tensor_t *b = self->saved[1];

    /* d/da = grad_out @ b^T */
    if (self->inputs[0]->requires_grad)
    {
        ax_tensor_t *bt = ax_tensor_transpose(b, 0, 1);
        ax_tensor_t *bt_contig = ax_tensor_contiguous(bt);

        int64_t ga_shape[] = {grad_out->shape[0], bt_contig->shape[1]};
        ax_tensor_t *grad_a = ax_tensor_zeros(ga_shape, 2, grad_out->dtype);
        ax_compute_gemm(grad_out, bt_contig, grad_a);

        accumulate_grad(self->inputs[0], grad_a);
        ax_tensor_destroy(bt);
        ax_tensor_destroy(bt_contig);
        ax_tensor_destroy(grad_a);
    }

    /* d/db = a^T @ grad_out */
    if (self->inputs[1]->requires_grad)
    {
        ax_tensor_t *at = ax_tensor_transpose(a, 0, 1);
        ax_tensor_t *at_contig = ax_tensor_contiguous(at);

        int64_t gb_shape[] = {at_contig->shape[0], grad_out->shape[1]};
        ax_tensor_t *grad_b = ax_tensor_zeros(gb_shape, 2, grad_out->dtype);
        ax_compute_gemm(at_contig, grad_out, grad_b);

        accumulate_grad(self->inputs[1], grad_b);
        ax_tensor_destroy(at);
        ax_tensor_destroy(at_contig);
        ax_tensor_destroy(grad_b);
    }
}

ax_grad_fn_t *ax_make_matmul_backward(ax_tensor_t *a, ax_tensor_t *b, ax_tensor_t *out)
{
    ax_grad_fn_t *gf = ax_grad_fn_create(matmul_backward);
    gf->inputs[0] = a;
    gf->inputs[1] = b;
    gf->n_inputs = 2;
    gf->saved[0] = a;
    gf->saved[1] = b;
    gf->n_saved = 2;
    return gf;
}


/* sum: d/da(sum(a)) = ones_like(a) (gradient broadcasts back)
   for axis reduction: gradient is broadcast along the reduced axis.
   int_ctx = axis */
static void sum_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *a = self->saved[0];
    int axis = self->int_ctx;

    /* the gradient of sum is just 1 everywhere, so we broadcast
       grad_out back to the input shape.
       simplest way: create a ones tensor and multiply by grad_out element */

    if (axis == -1)
    {
        /* full reduction: grad_out is scalar, broadcast to input shape */
        float g = ((float *)grad_out->storage->data)[grad_out->offset];
        ax_tensor_t *grad_a = ax_tensor_full(a->shape, a->ndim, a->dtype, (double)g);
        accumulate_grad(self->inputs[0], grad_a);
        ax_tensor_destroy(grad_a);
    }
    else
    {
        /* axis reduction: expand grad_out along the reduced axis.
           e.g., if a is [2,3] and we summed axis 0, grad_out is [3].
           we need to expand it to [2,3] by repeating along axis 0. */
        ax_tensor_t *grad_a = ax_tensor_zeros(a->shape, a->ndim, a->dtype);
        int64_t n = ax_tensor_numel(grad_a);

        float *gad = (float *)grad_a->storage->data;
        float *god = (float *)grad_out->storage->data;

        for (int64_t i = 0; i < n; i++)
        {
            /* figure out which element of grad_out this maps to
               (skip the reduced axis when computing the output index) */
            int64_t remaining = i;
            int64_t out_flat = 0;

            for (int d = a->ndim - 1; d >= 0; d--)
            {
                int64_t idx = remaining % a->shape[d];
                remaining /= a->shape[d];
                if (d != axis)
                {
                    int od = (d > axis) ? d - 1 : d;
                    int64_t os = 1;
                    for (int k = grad_out->ndim - 1; k > od; k--) os *= grad_out->shape[k];
                    out_flat += idx * os;
                }
            }

            gad[grad_a->offset + i] = god[grad_out->offset + out_flat];
        }

        accumulate_grad(self->inputs[0], grad_a);
        ax_tensor_destroy(grad_a);
    }
}

ax_grad_fn_t *ax_make_sum_backward(ax_tensor_t *a, int axis, ax_tensor_t *out)
{
    ax_grad_fn_t *gf = ax_grad_fn_create(sum_backward);
    gf->inputs[0] = a;
    gf->n_inputs = 1;
    gf->saved[0] = a;
    gf->n_saved = 1;
    gf->int_ctx = axis;
    return gf;
}


/* mean: like sum but divided by count.
   d/da(mean(a)) = 1/N
   d/da(mean(a, axis=k)) = 1/shape[k]

   instead of reusing sum_backward and rescaling (which corrupts
   previously accumulated gradients), we scale grad_out first
   and then do the broadcast/accumulate. */
static void mean_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *a = self->saved[0];
    int axis = self->int_ctx;

    int64_t count;
    if (axis == -1)
        count = ax_tensor_numel(a);
    else
        count = a->shape[axis];

    float inv_count = 1.0f / (float)count;

    if (!self->inputs[0]->requires_grad) return;

    if (!self->inputs[0]->grad)
        self->inputs[0]->grad = ax_tensor_zeros(a->shape, a->ndim, a->dtype);

    if (axis == -1)
    {
        /* full reduction: grad_out is scalar, broadcast scaled version to input shape */
        float g = ((float *)grad_out->storage->data)[grad_out->offset] * inv_count;
        float *gd = (float *)self->inputs[0]->grad->storage->data;
        int64_t n = ax_tensor_numel(a);
        for (int64_t i = 0; i < n; i++)
            gd[self->inputs[0]->grad->offset + i] += g;
    }
    else
    {
        /* axis reduction: expand scaled grad_out along the reduced axis */
        float *gad = (float *)self->inputs[0]->grad->storage->data;
        float *god = (float *)grad_out->storage->data;
        int64_t n = ax_tensor_numel(a);

        for (int64_t i = 0; i < n; i++)
        {
            int64_t remaining = i;
            int64_t out_flat = 0;

            for (int d = a->ndim - 1; d >= 0; d--)
            {
                int64_t idx = remaining % a->shape[d];
                remaining /= a->shape[d];
                if (d != axis)
                {
                    int od = (d > axis) ? d - 1 : d;
                    int64_t os = 1;
                    for (int k = grad_out->ndim - 1; k > od; k--) os *= grad_out->shape[k];
                    out_flat += idx * os;
                }
            }

            gad[self->inputs[0]->grad->offset + i] +=
                god[grad_out->offset + out_flat] * inv_count;
        }
    }
}

ax_grad_fn_t *ax_make_mean_backward(ax_tensor_t *a, int axis, ax_tensor_t *out)
{
    ax_grad_fn_t *gf = ax_grad_fn_create(mean_backward);
    gf->inputs[0] = a;
    gf->n_inputs = 1;
    gf->saved[0] = a;
    gf->n_saved = 1;
    gf->int_ctx = axis;
    return gf;
}


/* relu: d/da(relu(a)) = (a > 0) ? 1 : 0
   saved[0] = a (input) */
static void relu_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *a = self->saved[0];
    int64_t n = ax_tensor_numel(grad_out);

    ax_tensor_t *grad_a = ax_tensor_zeros(grad_out->shape, grad_out->ndim, grad_out->dtype);
    float *gad = (float *)grad_a->storage->data;
    float *god = (float *)grad_out->storage->data;
    float *ad = (float *)a->storage->data;

    for (int64_t i = 0; i < n; i++)
    {
        float aval = ad[a->offset + i];
        gad[grad_a->offset + i] = (aval > 0.0f) ? god[grad_out->offset + i] : 0.0f;
    }

    accumulate_grad(self->inputs[0], grad_a);
    ax_tensor_destroy(grad_a);
}

ax_grad_fn_t *ax_make_relu_backward(ax_tensor_t *a, ax_tensor_t *out)
{
    ax_grad_fn_t *gf = ax_grad_fn_create(relu_backward);
    gf->inputs[0] = a;
    gf->n_inputs = 1;
    gf->saved[0] = a;
    gf->n_saved = 1;
    return gf;
}


/* sigmoid: d/da(sig(a)) = sig(a) * (1 - sig(a))
   saved[0] = out (= sigmoid(a), already computed) */
static void sigmoid_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *sig_out = self->saved[0];
    int64_t n = ax_tensor_numel(grad_out);

    ax_tensor_t *grad_a = ax_tensor_zeros(grad_out->shape, grad_out->ndim, grad_out->dtype);
    float *gad = (float *)grad_a->storage->data;
    float *god = (float *)grad_out->storage->data;
    float *sd = (float *)sig_out->storage->data;

    for (int64_t i = 0; i < n; i++)
    {
        float s = sd[sig_out->offset + i];
        gad[grad_a->offset + i] = god[grad_out->offset + i] * s * (1.0f - s);
    }

    accumulate_grad(self->inputs[0], grad_a);
    ax_tensor_destroy(grad_a);
}

ax_grad_fn_t *ax_make_sigmoid_backward(ax_tensor_t *a, ax_tensor_t *out)
{
    ax_grad_fn_t *gf = ax_grad_fn_create(sigmoid_backward);
    gf->inputs[0] = a;
    gf->n_inputs = 1;
    gf->saved[0] = out; /* save sigmoid output */
    gf->n_saved = 1;
    return gf;
}


/* tanh: d/da(tanh(a)) = 1 - tanh(a)^2
   saved[0] = out (= tanh(a)) */
static void tanh_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *tanh_out = self->saved[0];
    int64_t n = ax_tensor_numel(grad_out);

    ax_tensor_t *grad_a = ax_tensor_zeros(grad_out->shape, grad_out->ndim, grad_out->dtype);
    float *gad = (float *)grad_a->storage->data;
    float *god = (float *)grad_out->storage->data;
    float *td = (float *)tanh_out->storage->data;

    for (int64_t i = 0; i < n; i++)
    {
        float t = td[tanh_out->offset + i];
        gad[grad_a->offset + i] = god[grad_out->offset + i] * (1.0f - t * t);
    }

    accumulate_grad(self->inputs[0], grad_a);
    ax_tensor_destroy(grad_a);
}

ax_grad_fn_t *ax_make_tanh_backward(ax_tensor_t *a, ax_tensor_t *out)
{
    ax_grad_fn_t *gf = ax_grad_fn_create(tanh_backward);
    gf->inputs[0] = a;
    gf->n_inputs = 1;
    gf->saved[0] = out;
    gf->n_saved = 1;
    return gf;
}
