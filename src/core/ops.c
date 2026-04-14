/* ops.c — user-facing tensor operations.
   handles output allocation, broadcast shapes, and autograd tape recording.
   the actual math is delegated to the compute dispatch layer. */

#include "axiom/ops.h"
#include "axiom/compute.h"
#include "axiom/broadcast.h"
#include "axiom/error.h"
#include "axiom/autograd.h"
#include <stdlib.h>
#include <stdatomic.h>
#include <inttypes.h>

/* internal helpers */

/* figure out the output shape for a binary op (broadcasting) and allocate the result */
static ax_tensor_t *alloc_binop_result(ax_tensor_t *a, ax_tensor_t *b)
{
    if (!a || !b) return NULL;

    int64_t out_shape[AX_MAX_DIMS];
    int out_ndim;

    ax_status_t s = ax_broadcast_shape(
        a->shape, a->ndim, b->shape, b->ndim,
        out_shape, &out_ndim);
    if (s != AX_OK) return NULL;

    /* uninit: every binop (add/mul/relu/etc.) fully overwrites the output.
       the old zeros call wasted one full memset pass per allocation. */
    return ax_tensor_create(out_shape, out_ndim, a->dtype);
}

/* allocate an output tensor with same shape as input */
static ax_tensor_t *alloc_like(ax_tensor_t *t)
{
    if (!t) return NULL;
    return ax_tensor_create(t->shape, t->ndim, t->dtype);
}

/* compute the shape that results from reducing along an axis */
static ax_tensor_t *alloc_reduce_result(ax_tensor_t *t, int axis)
{
    if (!t) return NULL;

    if (axis == -1)
    {
        /* full reduction -> scalar (1d with 1 element) */
        int64_t one = 1;
        return ax_tensor_zeros(&one, 1, t->dtype);
    }

    if (axis < 0 || axis >= t->ndim)
    {
        ax_err_set(AX_ERR_INVALID_AXIS, "axis %d out of range for %d dims", axis, t->ndim);
        return NULL;
    }

    /* drop the reduced dimension */
    int64_t out_shape[AX_MAX_DIMS];
    int out_ndim = 0;
    for (int i = 0; i < t->ndim; i++)
    {
        if (i != axis) out_shape[out_ndim++] = t->shape[i];
    }

    /* edge case: if we reduced everything to 0 dims, make it a scalar */
    if (out_ndim == 0)
    {
        int64_t one = 1;
        return ax_tensor_zeros(&one, 1, t->dtype);
    }

    return ax_tensor_zeros(out_shape, out_ndim, t->dtype);
}

/* binary ops */

/* forward declarations for backward fns — defined in autograd_ops.c */
extern ax_grad_fn_t *ax_make_add_backward(ax_tensor_t *a, ax_tensor_t *b, ax_tensor_t *out);
extern ax_grad_fn_t *ax_make_sub_backward(ax_tensor_t *a, ax_tensor_t *b, ax_tensor_t *out);
extern ax_grad_fn_t *ax_make_mul_backward(ax_tensor_t *a, ax_tensor_t *b, ax_tensor_t *out);
extern ax_grad_fn_t *ax_make_div_backward(ax_tensor_t *a, ax_tensor_t *b, ax_tensor_t *out);
extern ax_grad_fn_t *ax_make_matmul_backward(ax_tensor_t *a, ax_tensor_t *b, ax_tensor_t *out);
extern ax_grad_fn_t *ax_make_neg_backward(ax_tensor_t *a, ax_tensor_t *out);
extern ax_grad_fn_t *ax_make_exp_backward(ax_tensor_t *a, ax_tensor_t *out);
extern ax_grad_fn_t *ax_make_log_backward(ax_tensor_t *a, ax_tensor_t *out);
extern ax_grad_fn_t *ax_make_sqrt_backward(ax_tensor_t *a, ax_tensor_t *out);
extern ax_grad_fn_t *ax_make_square_backward(ax_tensor_t *a, ax_tensor_t *out);
extern ax_grad_fn_t *ax_make_add_scalar_backward(ax_tensor_t *a, ax_tensor_t *out);
extern ax_grad_fn_t *ax_make_mul_scalar_backward(ax_tensor_t *a, double s, ax_tensor_t *out);
extern ax_grad_fn_t *ax_make_sum_backward(ax_tensor_t *a, int axis, ax_tensor_t *out);
extern ax_grad_fn_t *ax_make_mean_backward(ax_tensor_t *a, int axis, ax_tensor_t *out);
extern ax_grad_fn_t *ax_make_relu_backward(ax_tensor_t *a, ax_tensor_t *out);
extern ax_grad_fn_t *ax_make_sigmoid_backward(ax_tensor_t *a, ax_tensor_t *out);
extern ax_grad_fn_t *ax_make_tanh_backward(ax_tensor_t *a, ax_tensor_t *out);

/* check if we need to track gradients for this operation */
static inline bool needs_grad(ax_tensor_t *a, ax_tensor_t *b)
{
    if (!ax_grad_enabled()) return false;
    if (a->requires_grad) return true;
    if (b && b->requires_grad) return true;
    return false;
}

ax_tensor_t *ax_add(ax_tensor_t *a, ax_tensor_t *b)
{
    ax_tensor_t *out = alloc_binop_result(a, b);
    if (!out) return NULL;

    ax_compute_add(a, b, out);

    if (needs_grad(a, b))
    {
        out->requires_grad = true;
        out->grad_fn = ax_make_add_backward(a, b, out);
    }
    return out;
}

ax_tensor_t *ax_sub(ax_tensor_t *a, ax_tensor_t *b)
{
    ax_tensor_t *out = alloc_binop_result(a, b);
    if (!out) return NULL;

    ax_compute_sub(a, b, out);

    if (needs_grad(a, b))
    {
        out->requires_grad = true;
        out->grad_fn = ax_make_sub_backward(a, b, out);
    }
    return out;
}

ax_tensor_t *ax_mul(ax_tensor_t *a, ax_tensor_t *b)
{
    ax_tensor_t *out = alloc_binop_result(a, b);
    if (!out) return NULL;

    ax_compute_mul(a, b, out);

    if (needs_grad(a, b))
    {
        out->requires_grad = true;
        out->grad_fn = ax_make_mul_backward(a, b, out);
    }
    return out;
}

ax_tensor_t *ax_div(ax_tensor_t *a, ax_tensor_t *b)
{
    ax_tensor_t *out = alloc_binop_result(a, b);
    if (!out) return NULL;

    ax_compute_div(a, b, out);

    if (needs_grad(a, b))
    {
        out->requires_grad = true;
        out->grad_fn = ax_make_div_backward(a, b, out);
    }
    return out;
}

/* unary ops */

ax_tensor_t *ax_neg(ax_tensor_t *a)
{
    ax_tensor_t *out = alloc_like(a);
    if (!out) return NULL;

    ax_compute_neg(a, out);

    if (needs_grad(a, NULL))
    {
        out->requires_grad = true;
        out->grad_fn = ax_make_neg_backward(a, out);
    }
    return out;
}

ax_tensor_t *ax_abs(ax_tensor_t *a)
{
    /* abs is not differentiable at 0, but we don't track grad for it (for now) */
    ax_tensor_t *out = alloc_like(a);
    if (!out) return NULL;
    ax_compute_abs(a, out);
    return out;
}

ax_tensor_t *ax_exp(ax_tensor_t *a)
{
    ax_tensor_t *out = alloc_like(a);
    if (!out) return NULL;

    ax_compute_exp(a, out);

    if (needs_grad(a, NULL))
    {
        out->requires_grad = true;
        out->grad_fn = ax_make_exp_backward(a, out);
    }
    return out;
}

ax_tensor_t *ax_log(ax_tensor_t *a)
{
    ax_tensor_t *out = alloc_like(a);
    if (!out) return NULL;

    ax_compute_log(a, out);

    if (needs_grad(a, NULL))
    {
        out->requires_grad = true;
        out->grad_fn = ax_make_log_backward(a, out);
    }
    return out;
}

ax_tensor_t *ax_sqrt(ax_tensor_t *a)
{
    ax_tensor_t *out = alloc_like(a);
    if (!out) return NULL;

    ax_compute_sqrt(a, out);

    if (needs_grad(a, NULL))
    {
        out->requires_grad = true;
        out->grad_fn = ax_make_sqrt_backward(a, out);
    }
    return out;
}

ax_tensor_t *ax_square(ax_tensor_t *a)
{
    ax_tensor_t *out = alloc_like(a);
    if (!out) return NULL;

    ax_compute_square(a, out);

    if (needs_grad(a, NULL))
    {
        out->requires_grad = true;
        out->grad_fn = ax_make_square_backward(a, out);
    }
    return out;
}

/* scalar ops */

ax_tensor_t *ax_add_scalar(ax_tensor_t *a, double s)
{
    ax_tensor_t *out = alloc_like(a);
    if (!out) return NULL;

    ax_compute_add_scalar(a, s, out);

    if (needs_grad(a, NULL))
    {
        out->requires_grad = true;
        out->grad_fn = ax_make_add_scalar_backward(a, out);
    }
    return out;
}

ax_tensor_t *ax_mul_scalar(ax_tensor_t *a, double s)
{
    ax_tensor_t *out = alloc_like(a);
    if (!out) return NULL;

    ax_compute_mul_scalar(a, s, out);

    if (needs_grad(a, NULL))
    {
        out->requires_grad = true;
        out->grad_fn = ax_make_mul_scalar_backward(a, s, out);
    }
    return out;
}

/* matmul */

ax_tensor_t *ax_matmul(ax_tensor_t *a, ax_tensor_t *b)
{
    return ax_matmul_bias(a, b, NULL);
}

extern ax_grad_fn_t *ax_make_matmul_bias_backward(ax_tensor_t *a, ax_tensor_t *b,
                                                    ax_tensor_t *bias, ax_tensor_t *out);

ax_tensor_t *ax_matmul_bias(ax_tensor_t *a, ax_tensor_t *b, ax_tensor_t *bias)
{
    if (!a || !b)
    {
        ax_err_set(AX_ERR_NULL_ARG, "matmul: NULL tensor");
        return NULL;
    }
    /* accept 2D [M, K] or 3D [B, S, K] on the left — transformer encoders
       feed [B, S, d_model] through dense layers. when 3D, we auto-reshape
       to [B*S, K], run the fused gemm+bias, then reshape back to [B, S, N].
       weight B must stay 2D (shared across B/S). */
    if ((a->ndim != 2 && a->ndim != 3) || b->ndim != 2)
    {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "matmul needs a:[M,K] or [B,S,K] and b:[K,N]");
        return NULL;
    }
    int64_t inner_a = (a->ndim == 2) ? a->shape[1] : a->shape[2];
    if (inner_a != b->shape[0])
    {
        ax_err_set(AX_ERR_SHAPE_MISMATCH,
                   "matmul: inner dims don't match (a_inner=%" PRId64 " b_outer=%" PRId64 ")",
                   inner_a, b->shape[0]);
        return NULL;
    }

    /* 3D fast-path: temporarily view input as [B*S, K] for the gemm,
       then set the output shape back to [B, S, N]. the gemm only sees
       a flat [rows, K], and matmul_bias_backward was already written
       to handle the flat case; it uses saved pointers rather than
       ndim-specific indexing. */
    int64_t M, N, K;
    int out_ndim;
    int64_t out_shape[3];
    ax_tensor_t *a_flat = a;
    bool a_viewed = false;
    if (a->ndim == 3) {
        int64_t flat_sh[] = {a->shape[0] * a->shape[1], a->shape[2]};
        a_flat = ax_tensor_reshape(a, flat_sh, 2);
        if (!a_flat) return NULL;
        a_viewed = true;
        M = flat_sh[0]; K = flat_sh[1]; N = b->shape[1];
        out_shape[0] = a->shape[0]; out_shape[1] = a->shape[1]; out_shape[2] = N;
        out_ndim = 3;
    } else {
        M = a->shape[0]; K = a->shape[1]; N = b->shape[1];
        out_shape[0] = M; out_shape[1] = N;
        out_ndim = 2;
    }
    /* temporary 2D view that the gemm+backward will see. */
    int64_t flat_out_sh[] = {M, N};
    /* allocate the flat 2D output first, run gemm+bias into it, then
       reshape to 3D for the caller when applicable. */
    ax_tensor_t *out_flat = ax_tensor_create(flat_out_sh, 2, a->dtype);
    if (!out_flat) {
        if (a_viewed) ax_tensor_destroy(a_flat);
        return NULL;
    }

    ax_compute_gemm(a_flat, b, out_flat);

    if (bias) {
        if (ax_compute_has_bias_add()) {
            ax_compute_bias_add(out_flat, bias, 0, out_flat);
        } else {
            float *od = (float *)out_flat->storage->data + out_flat->offset;
            const float *bd = (const float *)bias->storage->data + bias->offset;
            int64_t b_stride = (bias->ndim >= 1) ? bias->strides[bias->ndim - 1] : 1;
            for (int64_t i = 0; i < M; i++)
                for (int64_t j = 0; j < N; j++)
                    od[i * N + j] += bd[j * b_stride];
        }
    }

    /* if caller wants 3D shape, reshape the flat output (zero-copy — shares
       storage, just rewrites shape/strides). */
    ax_tensor_t *out = out_flat;
    if (out_ndim == 3) {
        out = ax_tensor_reshape(out_flat, out_shape, 3);
        if (!out) {
            ax_tensor_destroy(out_flat);
            if (a_viewed) ax_tensor_destroy(a_flat);
            return NULL;
        }
        ax_tensor_destroy(out_flat); /* view retained storage */
    }

    if (ax_grad_enabled() && (needs_grad(a, b) || (bias && bias->requires_grad)))
    {
        out->requires_grad = true;
        /* pass the ORIGINAL a (not the reshape view) to backward so the
           grad is routed back to the user-visible tensor shape. */
        out->grad_fn = ax_make_matmul_bias_backward(a, b, bias, out);
    }
    if (a_viewed) ax_tensor_destroy(a_flat);
    return out;
}

/* fused backward for matmul+bias+relu: same as matmul_bias_backward but
   the dX and dW computations must account for the relu mask. the relu was
   applied to the forward output, so the backward needs to zero the gradient
   where the output was <= 0. we save the relu'd output for this purpose. */
extern ax_grad_fn_t *ax_make_matmul_bias_relu_backward(ax_tensor_t *a, ax_tensor_t *b,
                                                         ax_tensor_t *bias, ax_tensor_t *out);

ax_tensor_t *ax_matmul_bias_relu(ax_tensor_t *a, ax_tensor_t *b, ax_tensor_t *bias)
{
    if (!a || !b) {
        ax_err_set(AX_ERR_NULL_ARG, "matmul_bias_relu: NULL tensor");
        return NULL;
    }
    if (a->ndim != 2 || b->ndim != 2) return NULL;
    if (a->shape[1] != b->shape[0]) return NULL;

    int64_t out_shape[] = {a->shape[0], b->shape[1]};
    ax_tensor_t *out = ax_tensor_create(out_shape, 2, a->dtype);
    if (!out) return NULL;

    if (ax_compute_has_gemm_relu()) {
        /* fused path: gemm + bias + relu in one call (output stays cache-hot) */
        ax_compute_gemm_relu(a, b, bias, out);
    } else {
        /* fallback: gemm → bias → relu as separate passes */
        ax_compute_gemm(a, b, out);
        if (bias) {
            if (ax_compute_has_bias_add())
                ax_compute_bias_add(out, bias, 0, out);
            else {
                float *od = (float *)out->storage->data + out->offset;
                const float *bd = (const float *)bias->storage->data + bias->offset;
                int64_t m = out->shape[0], n = out->shape[1];
                int64_t b_stride = (bias->ndim >= 1) ? bias->strides[bias->ndim - 1] : 1;
                for (int64_t i = 0; i < m; i++)
                    for (int64_t j = 0; j < n; j++)
                        od[i * n + j] += bd[j * b_stride];
            }
        }
        ax_compute_relu(out, out);
    }

    if (ax_grad_enabled() && (needs_grad(a, b) || (bias && bias->requires_grad)))
    {
        out->requires_grad = true;
        out->grad_fn = ax_make_matmul_bias_relu_backward(a, b, bias, out);
    }
    return out;
}

/* reductions */

ax_tensor_t *ax_sum(ax_tensor_t *a, int axis)
{
    ax_tensor_t *out = alloc_reduce_result(a, axis);
    if (!out) return NULL;

    ax_compute_sum(a, axis, out);

    if (needs_grad(a, NULL))
    {
        out->requires_grad = true;
        out->grad_fn = ax_make_sum_backward(a, axis, out);
    }
    return out;
}

ax_tensor_t *ax_mean(ax_tensor_t *a, int axis)
{
    ax_tensor_t *out = alloc_reduce_result(a, axis);
    if (!out) return NULL;

    ax_compute_mean(a, axis, out);

    if (needs_grad(a, NULL))
    {
        out->requires_grad = true;
        out->grad_fn = ax_make_mean_backward(a, axis, out);
    }
    return out;
}

ax_tensor_t *ax_max(ax_tensor_t *a, int axis)
{
    /* max is technically differentiable but we skip it for now —
       the gradient is a mess with ties */
    ax_tensor_t *out = alloc_reduce_result(a, axis);
    if (!out) return NULL;
    ax_compute_max(a, axis, out);
    return out;
}

ax_tensor_t *ax_min(ax_tensor_t *a, int axis)
{
    ax_tensor_t *out = alloc_reduce_result(a, axis);
    if (!out) return NULL;
    ax_compute_min(a, axis, out);
    return out;
}

/* activations */

ax_tensor_t *ax_relu(ax_tensor_t *a)
{
    ax_tensor_t *out = alloc_like(a);
    if (!out) return NULL;

    ax_compute_relu(a, out);

    if (needs_grad(a, NULL))
    {
        out->requires_grad = true;
        out->grad_fn = ax_make_relu_backward(a, out);
    }
    return out;
}

ax_tensor_t *ax_sigmoid(ax_tensor_t *a)
{
    ax_tensor_t *out = alloc_like(a);
    if (!out) return NULL;

    ax_compute_sigmoid(a, out);

    if (needs_grad(a, NULL))
    {
        out->requires_grad = true;
        out->grad_fn = ax_make_sigmoid_backward(a, out);
    }
    return out;
}

ax_tensor_t *ax_tanh_op(ax_tensor_t *a)
{
    ax_tensor_t *out = alloc_like(a);
    if (!out) return NULL;

    ax_compute_tanh(a, out);

    if (needs_grad(a, NULL))
    {
        out->requires_grad = true;
        out->grad_fn = ax_make_tanh_backward(a, out);
    }
    return out;
}

/* in-place activations.
   check storage refcount == 1 so we don't clobber a shared buffer.
   also require that the tensor has no existing grad_fn (i.e., it's a
   graph leaf). otherwise fall back to the allocating variant — replacing
   grad_fn in place would lose the upstream chain of backward fns.
   backend ops read id[i] then write od[i] at the same index, so passing
   out == in is safe for all elementwise activations. */

static inline bool storage_is_unique(const ax_tensor_t *t)
{
    if (!t || !t->storage) return false;
    return atomic_load(&t->storage->refcount) == 1;
}

/* inplace is only safe for contiguous, zero-offset tensors because the
   backward transform walks grad/data linearly via a flat index. */
static inline bool inplace_safe(const ax_tensor_t *t)
{
    if (!t) return false;
    if (!storage_is_unique(t)) return false;
    if (t->grad_fn != NULL) return false;
    if (t->offset != 0) return false;
    if (!ax_tensor_is_contiguous(t)) return false;
    return true;
}

/* backward fn for inplace activations. because the input and output alias
   the same storage, input->grad IS grad_out — we transform it in place.
   n_inputs is set to 0 so the topo walker doesn't try to recurse on self.
   for this reason inplace is only safe when the tensor is a leaf from the
   graph's perspective (no prior grad_fn). */

static void relu_inplace_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *t = self->saved[0]; /* same pointer as the tensor being backprop'd */
    if (!t || !t->requires_grad || !t->grad) return;

    int64_t n = ax_tensor_numel(grad_out);
    float *gd = (float *)t->grad->storage->data + t->grad->offset;
    const float *td = (const float *)t->storage->data + t->offset;
    /* post-relu values: td[i] > 0 iff pre-relu was positive. transform in place. */
    for (int64_t i = 0; i < n; i++)
        gd[i] = (td[i] > 0.0f) ? gd[i] : 0.0f;
}

static void sigmoid_inplace_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *t = self->saved[0];
    if (!t || !t->requires_grad || !t->grad) return;

    int64_t n = ax_tensor_numel(grad_out);
    float *gd = (float *)t->grad->storage->data + t->grad->offset;
    const float *sd = (const float *)t->storage->data + t->offset;
    /* sigmoid derivative = s * (1 - s) where s is the output */
    for (int64_t i = 0; i < n; i++) {
        float s = sd[i];
        gd[i] = gd[i] * s * (1.0f - s);
    }
}

static void tanh_inplace_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *t = self->saved[0];
    if (!t || !t->requires_grad || !t->grad) return;

    int64_t n = ax_tensor_numel(grad_out);
    float *gd = (float *)t->grad->storage->data + t->grad->offset;
    const float *td = (const float *)t->storage->data + t->offset;
    /* tanh derivative = 1 - t^2 where t is the output */
    for (int64_t i = 0; i < n; i++) {
        float tv = td[i];
        gd[i] = gd[i] * (1.0f - tv * tv);
    }
}

static ax_grad_fn_t *make_inplace_grad_fn(ax_tensor_t *t, ax_backward_fn_t fn)
{
    ax_grad_fn_t *gf = ax_grad_fn_create(fn);
    if (!gf) return NULL;
    /* no recorded inputs — inputs[0] would alias self and confuse topo walk */
    gf->n_inputs = 0;
    gf->saved[0] = t;
    gf->saved_owned[0] = false;
    gf->n_saved = 1;
    return gf;
}

ax_tensor_t *ax_relu_inplace(ax_tensor_t *a)
{
    if (!a) return NULL;
    /* fallback conditions: shared storage or already part of a grad chain */
    if (!inplace_safe(a)) return ax_relu(a);

    ax_compute_relu(a, a);

    if (needs_grad(a, NULL))
    {
        a->requires_grad = true;
        a->grad_fn = make_inplace_grad_fn(a, relu_inplace_backward);
    }
    return a;
}

ax_tensor_t *ax_sigmoid_inplace(ax_tensor_t *a)
{
    if (!a) return NULL;
    if (!inplace_safe(a)) return ax_sigmoid(a);

    ax_compute_sigmoid(a, a);

    if (needs_grad(a, NULL))
    {
        a->requires_grad = true;
        a->grad_fn = make_inplace_grad_fn(a, sigmoid_inplace_backward);
    }
    return a;
}

ax_tensor_t *ax_tanh_inplace(ax_tensor_t *a)
{
    if (!a) return NULL;
    if (!inplace_safe(a)) return ax_tanh_op(a);

    ax_compute_tanh(a, a);

    if (needs_grad(a, NULL))
    {
        a->requires_grad = true;
        a->grad_fn = make_inplace_grad_fn(a, tanh_inplace_backward);
    }
    return a;
}

/* comparisons (no grad — these return 0/1) */

ax_tensor_t *ax_equal(ax_tensor_t *a, ax_tensor_t *b)
{
    ax_tensor_t *out = alloc_binop_result(a, b);
    if (!out) return NULL;
    ax_compute_equal(a, b, out);
    return out;
}

ax_tensor_t *ax_greater(ax_tensor_t *a, ax_tensor_t *b)
{
    ax_tensor_t *out = alloc_binop_result(a, b);
    if (!out) return NULL;
    ax_compute_greater(a, b, out);
    return out;
}
