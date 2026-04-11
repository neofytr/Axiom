/* ops.c — user-facing tensor operations.
   handles output allocation, broadcast shapes, and autograd tape recording.
   the actual math is delegated to the compute dispatch layer. */

#include "axiom/ops.h"
#include "axiom/compute.h"
#include "axiom/broadcast.h"
#include "axiom/error.h"
#include "axiom/autograd.h"
#include <stdlib.h>
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

    return ax_tensor_zeros(out_shape, out_ndim, a->dtype);
}

/* allocate an output tensor with same shape as input */
static ax_tensor_t *alloc_like(ax_tensor_t *t)
{
    if (!t) return NULL;
    return ax_tensor_zeros(t->shape, t->ndim, t->dtype);
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
    if (!a || !b)
    {
        ax_err_set(AX_ERR_NULL_ARG, "matmul: NULL tensor");
        return NULL;
    }
    if (a->ndim != 2 || b->ndim != 2)
    {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "matmul needs 2d tensors");
        return NULL;
    }
    if (a->shape[1] != b->shape[0])
    {
        ax_err_set(AX_ERR_SHAPE_MISMATCH,
                   "matmul: [%" PRId64 ",%" PRId64 "] @ [%" PRId64 ",%" PRId64 "] inner dims don't match",
                   a->shape[0], a->shape[1], b->shape[0], b->shape[1]);
        return NULL;
    }

    int64_t out_shape[] = {a->shape[0], b->shape[1]};
    ax_tensor_t *out = ax_tensor_zeros(out_shape, 2, a->dtype);
    if (!out) return NULL;

    ax_compute_gemm(a, b, out);

    if (needs_grad(a, b))
    {
        out->requires_grad = true;
        out->grad_fn = ax_make_matmul_backward(a, b, out);
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
