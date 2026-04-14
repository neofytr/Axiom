/* activations.c — activation function implementations.
   each one does the forward pass and hooks up a backward function
   so autograd can flow gradients through them. */

#include "axiom/activations.h"
#include "axiom/autograd.h"
#include "axiom/compute.h"
#include "axiom/error.h"
#include "../compute/backends/simd_defs.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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
   most activations follow the same pattern so this saves repetition.
   uses the uninitialized allocator — every activation forward below
   writes every element before return, so the memset in ax_tensor_zeros
   is pure wasted bandwidth on large tensors. */
static ax_tensor_t *alloc_out(ax_tensor_t *a)
{
    return ax_tensor_create(a->shape, a->ndim, a->dtype);
}

static inline bool needs_grad(ax_tensor_t *a)
{
    return ax_grad_enabled() && a->requires_grad;
}


/* leaky relu */

static void leaky_relu_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    if (!self->inputs[0]->requires_grad) return;

    ax_tensor_t *input = self->saved[0];
    float alpha = (float)self->scalar_ctx;
    int64_t n = ax_tensor_numel(grad_out);

    if (!self->inputs[0]->grad)
        self->inputs[0]->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
    if (!self->inputs[0]->grad) return;

    float *ig = (float *)self->inputs[0]->grad->storage->data;
    float *go = (float *)grad_out->storage->data;
    float *ad = (float *)input->storage->data;

    for (int64_t i = 0; i < n; i++)
    {
        float x = ad[input->offset + i];
        ig[self->inputs[0]->grad->offset + i] += go[grad_out->offset + i] * (x > 0.0f ? 1.0f : alpha);
    }
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
    if (!self->inputs[0]->requires_grad) return;

    ax_tensor_t *input = self->saved[0];
    ax_tensor_t *output = self->saved[1];
    float alpha = (float)self->scalar_ctx;
    int64_t n = ax_tensor_numel(grad_out);

    if (!self->inputs[0]->grad)
        self->inputs[0]->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
    if (!self->inputs[0]->grad) return;

    float *ig = (float *)self->inputs[0]->grad->storage->data;
    float *go = (float *)grad_out->storage->data;
    float *ad = (float *)input->storage->data;
    float *od = (float *)output->storage->data;

    for (int64_t i = 0; i < n; i++)
    {
        float x = ad[input->offset + i];
        /* elu'(x) = 1 if x > 0, else output + alpha (since output = alpha*(exp(x)-1)) */
        ig[self->inputs[0]->grad->offset + i] += go[grad_out->offset + i] *
            (x > 0.0f ? 1.0f : od[output->offset + i] + alpha);
    }
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


/* selu: lambda * (x if x > 0, alpha * (exp(x) - 1) otherwise)
   single-node implementation with custom backward. */

static void selu_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    if (!self->inputs[0]->requires_grad) return;

    ax_tensor_t *input = self->saved[0];
    int64_t n = ax_tensor_numel(grad_out);

    if (!self->inputs[0]->grad)
        self->inputs[0]->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
    if (!self->inputs[0]->grad) return;

    float *ig = (float *)self->inputs[0]->grad->storage->data;
    float *go = (float *)grad_out->storage->data;
    float *ad = (float *)input->storage->data;

    for (int64_t i = 0; i < n; i++)
    {
        float x = ad[input->offset + i];
        /* selu'(x) = lambda if x > 0, lambda * alpha * exp(x) otherwise */
        float deriv = (x > 0.0f) ? SELU_LAMBDA : SELU_LAMBDA * SELU_ALPHA * expf(x);
        ig[self->inputs[0]->grad->offset + i] += go[grad_out->offset + i] * deriv;
    }
}

ax_tensor_t *ax_selu(ax_tensor_t *a)
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
        od[out->offset + i] = (x > 0.0f)
            ? SELU_LAMBDA * x
            : SELU_LAMBDA * SELU_ALPHA * (expf(x) - 1.0f);
    }

    if (needs_grad(a))
    {
        out->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(selu_backward);
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


/* gelu (approximate) */

static void gelu_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    if (!self->inputs[0]->requires_grad) return;

    ax_tensor_t *input = self->saved[0];
    int64_t n = ax_tensor_numel(grad_out);

    if (!self->inputs[0]->grad)
        self->inputs[0]->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
    if (!self->inputs[0]->grad) return;

    float *ig = (float *)self->inputs[0]->grad->storage->data;
    float *go = (float *)grad_out->storage->data;
    float *ad = (float *)input->storage->data;

    for (int64_t i = 0; i < n; i++)
    {
        float x = ad[input->offset + i];
        /* gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
           let t = tanh(sqrt(2/pi) * (x + 0.044715*x^3))
           gelu'(x) = 0.5*(1+t) + 0.5*x*(1-t^2)*sqrt(2/pi)*(1 + 3*0.044715*x^2) */
        float inner = SQRT_2_PI * (x + GELU_COEFF * x * x * x);
        float t = tanhf(inner);
        float deriv = 0.5f * (1.0f + t) +
                      0.5f * x * (1.0f - t * t) * SQRT_2_PI * (1.0f + 3.0f * GELU_COEFF * x * x);
        ig[self->inputs[0]->grad->offset + i] += go[grad_out->offset + i] * deriv;
    }
}

ax_tensor_t *ax_gelu(ax_tensor_t *a)
{
    if (!check_f32(a)) return NULL;
    ax_tensor_t *out = alloc_out(a);
    if (!out) return NULL;

    int64_t n = ax_tensor_numel(a);
    float *od = (float *)out->storage->data;
    float *ad = (float *)a->storage->data;
    int64_t off_a = a->offset;
    int64_t off_o = out->offset;

    /* SIMD tanh-approximation GELU: matches tf.nn.gelu(..., approximate=True).
       parallel for big tensors to amortize OMP fork-join overhead. serial
       scalar loop for tiny tensors where threading cost dominates. */
#if defined(AX_HAS_SIMD)
    ax_vf32 v_half = ax_vf32_set1(0.5f);
    ax_vf32 v_one  = ax_vf32_set1(1.0f);
    ax_vf32 v_coef = ax_vf32_set1(GELU_COEFF);
    ax_vf32 v_sqrt = ax_vf32_set1(SQRT_2_PI);
    int64_t ve = n - (n % AX_VF32_WIDTH);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if (n > 65536)
    #endif
    for (int64_t i = 0; i < ve; i += AX_VF32_WIDTH) {
        ax_vf32 x = ax_vf32_loadu(ad + off_a + i);
        ax_vf32 xx = ax_vf32_mul(x, x);
        ax_vf32 xxx = ax_vf32_mul(xx, x);
        /* inner = sqrt(2/pi) * (x + coeff * x^3) */
        ax_vf32 inner = ax_vf32_mul(v_sqrt,
                          ax_vf32_fmadd(v_coef, xxx, x));
        ax_vf32 t = ax_vf32_tanh(inner);
        /* 0.5 * x * (1 + t) */
        ax_vf32 y = ax_vf32_mul(ax_vf32_mul(v_half, x),
                                 ax_vf32_add(v_one, t));
        ax_vf32_storeu(od + off_o + i, y);
    }
    for (int64_t i = ve; i < n; i++) {
        float x = ad[off_a + i];
        float inner = SQRT_2_PI * (x + GELU_COEFF * x * x * x);
        od[off_o + i] = 0.5f * x * (1.0f + tanhf(inner));
    }
#else
    for (int64_t i = 0; i < n; i++) {
        float x = ad[off_a + i];
        float inner = SQRT_2_PI * (x + GELU_COEFF * x * x * x);
        od[off_o + i] = 0.5f * x * (1.0f + tanhf(inner));
    }
#endif

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
    if (!self->inputs[0]->requires_grad) return;

    ax_tensor_t *input = self->saved[0];
    ax_tensor_t *output = self->saved[1];
    int64_t n = ax_tensor_numel(grad_out);

    if (!self->inputs[0]->grad)
        self->inputs[0]->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
    if (!self->inputs[0]->grad) return;

    float *ig = (float *)self->inputs[0]->grad->storage->data;
    float *go = (float *)grad_out->storage->data;
    float *ad = (float *)input->storage->data;
    float *od = (float *)output->storage->data;

    for (int64_t i = 0; i < n; i++)
    {
        float x = ad[input->offset + i];
        float sig = 1.0f / (1.0f + expf(-x));
        /* swish'(x) = swish(x) + sigmoid(x) * (1 - swish(x)) */
        float sw = od[output->offset + i];
        ig[self->inputs[0]->grad->offset + i] += go[grad_out->offset + i] * (sw + sig * (1.0f - sw));
    }
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
    if (!self->inputs[0]->requires_grad) return;

    ax_tensor_t *input = self->saved[0];
    int64_t n = ax_tensor_numel(grad_out);

    if (!self->inputs[0]->grad)
        self->inputs[0]->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
    if (!self->inputs[0]->grad) return;

    float *ig = (float *)self->inputs[0]->grad->storage->data;
    float *go = (float *)grad_out->storage->data;
    float *ad = (float *)input->storage->data;

    /* softplus'(x) = sigmoid(x) = 1 / (1 + exp(-x)) */
    for (int64_t i = 0; i < n; i++)
    {
        float x = ad[input->offset + i];
        ig[self->inputs[0]->grad->offset + i] += go[grad_out->offset + i] / (1.0f + expf(-x));
    }
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
   single-node implementation. mish'(x) = tanh(sp) + x * sig(x) * (1 - tanh(sp)^2)
   where sp = softplus(x), sig = sigmoid */

static void mish_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    if (!self->inputs[0]->requires_grad) return;

    ax_tensor_t *input = self->saved[0];
    int64_t n = ax_tensor_numel(grad_out);

    if (!self->inputs[0]->grad)
        self->inputs[0]->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
    if (!self->inputs[0]->grad) return;

    float *ig = (float *)self->inputs[0]->grad->storage->data;
    float *go = (float *)grad_out->storage->data;
    float *ad = (float *)input->storage->data;

    for (int64_t i = 0; i < n; i++)
    {
        float x = ad[input->offset + i];
        float sp = (x > 20.0f) ? x : ((x < -20.0f) ? expf(x) : logf(1.0f + expf(x)));
        float tsp = tanhf(sp);
        float sig = 1.0f / (1.0f + expf(-x));
        ig[self->inputs[0]->grad->offset + i] += go[grad_out->offset + i] *
            (tsp + x * sig * (1.0f - tsp * tsp));
    }
}

ax_tensor_t *ax_mish(ax_tensor_t *a)
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
        float sp = (x > 20.0f) ? x : ((x < -20.0f) ? expf(x) : logf(1.0f + expf(x)));
        od[out->offset + i] = x * tanhf(sp);
    }

    if (needs_grad(a))
    {
        out->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(mish_backward);
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
        int64_t sa = a->strides[0];
        int64_t so = out->strides[0];
        const float *ap = ad + a->offset;
        float *op = od + out->offset;

        /* pass 1: max. SIMD when contiguous, scalar otherwise. */
        float mx;
        if (sa == 1) {
            int64_t i = 0, ie = n - (n % AX_VF32_WIDTH);
            if (ie > 0) {
                ax_vf32 vmax = ax_vf32_set1(-FLT_MAX);
                for (; i < ie; i += AX_VF32_WIDTH)
                    vmax = ax_vf32_max(vmax, ax_vf32_loadu(ap + i));
                mx = ax_vf32_hmax(vmax);
            } else mx = -FLT_MAX;
            for (; i < n; i++) if (ap[i] > mx) mx = ap[i];
        } else {
            mx = -FLT_MAX;
            for (int64_t i = 0; i < n; i++) {
                float v = ap[i * sa]; if (v > mx) mx = v;
            }
        }

        /* pass 2: exp(x - mx), write to out, accumulate sum. */
        float total;
        if (sa == 1 && so == 1) {
            ax_vf32 vmx = ax_vf32_set1(mx), vsum = ax_vf32_zero();
            int64_t i = 0, ie = n - (n % AX_VF32_WIDTH);
            for (; i < ie; i += AX_VF32_WIDTH) {
                ax_vf32 v = ax_vf32_exp(ax_vf32_sub(ax_vf32_loadu(ap + i), vmx));
                ax_vf32_storeu(op + i, v);
                vsum = ax_vf32_add(vsum, v);
            }
            total = ax_vf32_hsum(vsum);
            for (; i < n; i++) { float e = expf(ap[i] - mx); op[i] = e; total += e; }
        } else {
            total = 0.0f;
            for (int64_t i = 0; i < n; i++) {
                float e = expf(ap[i * sa] - mx);
                op[i * so] = e; total += e;
            }
        }

        /* pass 3: normalize. */
        float inv = 1.0f / total;
        if (so == 1) {
            ax_vf32 vinv = ax_vf32_set1(inv);
            int64_t i = 0, ie = n - (n % AX_VF32_WIDTH);
            for (; i < ie; i += AX_VF32_WIDTH)
                ax_vf32_storeu(op + i, ax_vf32_mul(ax_vf32_loadu(op + i), vinv));
            for (; i < n; i++) op[i] *= inv;
        } else {
            for (int64_t i = 0; i < n; i++) op[i * so] *= inv;
        }
    }
    else if (a->ndim == 2 && axis == 1)
    {
        /* softmax along columns for each row (most common case: batch of logits).
           SIMD fast-path when inner stride is contiguous; scalar fallback otherwise.
           rows are independent so OMP parallel over them is safe. */
        int64_t rows = a->shape[0];
        int64_t cols = a->shape[1];
        int64_t sa0 = a->strides[0], sa1 = a->strides[1];
        int64_t so0 = out->strides[0], so1 = out->strides[1];
        bool contig_in  = (sa1 == 1);
        bool contig_out = (so1 == 1);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) if (rows > 1)
        #endif
        for (int64_t r = 0; r < rows; r++)
        {
            const float *ap = ad + a->offset + r * sa0;
            float *op = od + out->offset + r * so0;

            /* pass 1: max */
            float mx;
            if (contig_in) {
                int64_t c = 0, ce = cols - (cols % AX_VF32_WIDTH);
                if (ce > 0) {
                    ax_vf32 vmax = ax_vf32_set1(-FLT_MAX);
                    for (; c < ce; c += AX_VF32_WIDTH)
                        vmax = ax_vf32_max(vmax, ax_vf32_loadu(ap + c));
                    mx = ax_vf32_hmax(vmax);
                } else mx = -FLT_MAX;
                for (; c < cols; c++) if (ap[c] > mx) mx = ap[c];
            } else {
                mx = -FLT_MAX;
                for (int64_t c = 0; c < cols; c++) {
                    float v = ap[c * sa1]; if (v > mx) mx = v;
                }
            }

            /* pass 2: exp(x - mx), store, accumulate sum */
            float total;
            if (contig_in && contig_out) {
                ax_vf32 vmx = ax_vf32_set1(mx), vsum = ax_vf32_zero();
                int64_t c = 0, ce = cols - (cols % AX_VF32_WIDTH);
                for (; c < ce; c += AX_VF32_WIDTH) {
                    ax_vf32 v = ax_vf32_exp(ax_vf32_sub(ax_vf32_loadu(ap + c), vmx));
                    ax_vf32_storeu(op + c, v);
                    vsum = ax_vf32_add(vsum, v);
                }
                total = ax_vf32_hsum(vsum);
                for (; c < cols; c++) { float e = expf(ap[c] - mx); op[c] = e; total += e; }
            } else {
                total = 0.0f;
                for (int64_t c = 0; c < cols; c++) {
                    float e = expf(ap[c * sa1] - mx);
                    op[c * so1] = e; total += e;
                }
            }

            /* pass 3: normalize */
            float inv = 1.0f / total;
            if (contig_out) {
                ax_vf32 vinv = ax_vf32_set1(inv);
                int64_t c = 0, ce = cols - (cols % AX_VF32_WIDTH);
                for (; c < ce; c += AX_VF32_WIDTH)
                    ax_vf32_storeu(op + c, ax_vf32_mul(ax_vf32_loadu(op + c), vinv));
                for (; c < cols; c++) op[c] *= inv;
            } else {
                for (int64_t c = 0; c < cols; c++) op[c * so1] *= inv;
            }
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
