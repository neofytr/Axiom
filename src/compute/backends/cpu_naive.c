/* cpu_naive.c — pure c fallback backend, no simd, no external deps.
   correctness over performance — this is the reference implementation. */

#include "axiom/backend_ops.h"
#include "axiom/tensor.h"
#include "axiom/error.h"
#include <math.h>
#include <string.h>
#include <float.h>
#include <inttypes.h>

/* helpers */

/* get a float32 element from a tensor at a flat index, respecting strides */
static inline float tensor_get_f32(const ax_tensor_t *t, int64_t flat_idx) {
    /* convert flat index to per-dimension indices and apply strides */
    int64_t remaining = flat_idx;
    int64_t offset = t->offset;
    for (int d = t->ndim - 1; d >= 0; d--) {
        int64_t idx = remaining % t->shape[d];
        remaining /= t->shape[d];
        offset += idx * t->strides[d];
    }
    return ((float *)t->storage->data)[offset];
}

/* set a float32 element in a tensor at a flat index */
static inline void tensor_set_f32(ax_tensor_t *t, int64_t flat_idx, float val) {
    int64_t remaining = flat_idx;
    int64_t offset = t->offset;
    for (int d = t->ndim - 1; d >= 0; d--) {
        int64_t idx = remaining % t->shape[d];
        remaining /= t->shape[d];
        offset += idx * t->strides[d];
    }
    ((float *)t->storage->data)[offset] = val;
}

/* total number of elements in a tensor */
static inline int64_t tensor_numel(const ax_tensor_t *t) {
    int64_t n = 1;
    for (int d = 0; d < t->ndim; d++) n *= t->shape[d];
    return n;
}

/* compute broadcast index: maps an output flat index to an input flat index,
   handling broadcasting (dimensions of size 1 get index 0) */
static inline int64_t broadcast_index(const ax_tensor_t *t, const ax_tensor_t *out, int64_t out_flat) {
    int64_t remaining = out_flat;
    int64_t offset = t->offset;
    for (int d = out->ndim - 1; d >= 0; d--) {
        int64_t idx = remaining % out->shape[d];
        remaining /= out->shape[d];
        /* map to input dimension — if input dim is 1, broadcast (index stays 0) */
        int t_dim = d - (out->ndim - t->ndim);
        if (t_dim >= 0 && t->shape[t_dim] > 1) {
            offset += idx * t->strides[t_dim];
        }
    }
    return offset;
}

/* get f32 value from a tensor with broadcasting against an output shape */
static inline float bcast_get_f32(const ax_tensor_t *t, const ax_tensor_t *out, int64_t out_flat) {
    int64_t offset = broadcast_index(t, out, out_flat);
    return ((float *)t->storage->data)[offset];
}

/* element-wise binary ops */

/* generic binary op with broadcasting */
#define DEFINE_BINOP(name, op_expr) \
static ax_status_t cpu_##name(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) { \
    if (a->dtype != AX_FLOAT32 || b->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32) { \
        ax_err_set(AX_ERR_DTYPE_MISMATCH, #name " only supports float32 currently"); \
        return AX_ERR_DTYPE_MISMATCH; \
    } \
    int64_t n = tensor_numel(out); \
    for (int64_t i = 0; i < n; i++) { \
        float va = bcast_get_f32(a, out, i); \
        float vb = bcast_get_f32(b, out, i); \
        float result = (op_expr); \
        tensor_set_f32(out, i, result); \
    } \
    return AX_OK; \
}

DEFINE_BINOP(add, va + vb)
DEFINE_BINOP(sub, va - vb)
DEFINE_BINOP(mul, va * vb)
/* division by zero produces ieee 754 infinity, matching standard math behavior.
   this lets log_backward and similar produce correct (large) gradients near zero
   instead of silently zeroing them. users should use gradient clipping if needed. */
DEFINE_BINOP(div_op, va / vb)

/* element-wise unary ops */

#define DEFINE_UNOP(name, op_expr) \
static ax_status_t cpu_##name(const ax_tensor_t *in, ax_tensor_t *out) { \
    if (in->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32) { \
        ax_err_set(AX_ERR_DTYPE_MISMATCH, #name " only supports float32 currently"); \
        return AX_ERR_DTYPE_MISMATCH; \
    } \
    int64_t n = tensor_numel(out); \
    for (int64_t i = 0; i < n; i++) { \
        float v = tensor_get_f32(in, i); \
        tensor_set_f32(out, i, (op_expr)); \
    } \
    return AX_OK; \
}

DEFINE_UNOP(neg, -v)
DEFINE_UNOP(abs_op, fabsf(v))
DEFINE_UNOP(exp_op, expf(v))
DEFINE_UNOP(log_op, v > 0.0f ? logf(v) : -FLT_MAX)
DEFINE_UNOP(sqrt_op, v >= 0.0f ? sqrtf(v) : 0.0f)
DEFINE_UNOP(square, v * v)

/* scalar ops */

static ax_status_t cpu_add_scalar(const ax_tensor_t *in, double scalar, ax_tensor_t *out) {
    if (in->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "add_scalar only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    float s = (float)scalar;
    int64_t n = tensor_numel(out);
    for (int64_t i = 0; i < n; i++) {
        tensor_set_f32(out, i, tensor_get_f32(in, i) + s);
    }
    return AX_OK;
}

static ax_status_t cpu_mul_scalar(const ax_tensor_t *in, double scalar, ax_tensor_t *out) {
    if (in->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "mul_scalar only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    float s = (float)scalar;
    int64_t n = tensor_numel(out);
    for (int64_t i = 0; i < n; i++) {
        tensor_set_f32(out, i, tensor_get_f32(in, i) * s);
    }
    return AX_OK;
}

/* gemm: general matrix multiply */
/* out[m,n] = a[m,k] @ b[k,n] — 2d only for now */

static ax_status_t cpu_gemm(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    if (a->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "gemm only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    if (a->ndim != 2 || b->ndim != 2 || out->ndim != 2) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "gemm requires 2d tensors");
        return AX_ERR_SHAPE_MISMATCH;
    }

    int64_t m = a->shape[0];
    int64_t k = a->shape[1];
    int64_t n = b->shape[1];

    if (b->shape[0] != k || out->shape[0] != m || out->shape[1] != n) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "gemm shape mismatch: [%" PRId64 ",%" PRId64 "] @ [%" PRId64 ",%" PRId64 "] -> [%" PRId64 ",%" PRId64 "]",
                   m, k, b->shape[0], b->shape[1], out->shape[0], out->shape[1]);
        return AX_ERR_SHAPE_MISMATCH;
    }

    float *ad = (float *)a->storage->data;
    float *bd = (float *)b->storage->data;
    float *od = (float *)out->storage->data;

    /* naive triple loop — correct, not fast */
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int64_t p = 0; p < k; p++) {
                float av = ad[a->offset + i * a->strides[0] + p * a->strides[1]];
                float bv = bd[b->offset + p * b->strides[0] + j * b->strides[1]];
                sum += av * bv;
            }
            od[out->offset + i * out->strides[0] + j * out->strides[1]] = sum;
        }
    }
    return AX_OK;
}

/* fused primitives — naive reference implementations.
   correctness first; cpu_opt supplies SIMD variants. */

static ax_status_t cpu_add_relu(const ax_tensor_t *a, const ax_tensor_t *b,
                                 ax_tensor_t *out)
{
    if (a->dtype != AX_FLOAT32 || b->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32)
        return AX_ERR_DTYPE_MISMATCH;
    int64_t n = 1;
    for (int d = 0; d < out->ndim; d++) n *= out->shape[d];
    /* require same-shape for the fused path; broadcast callers go
       through plain add + relu. */
    int64_t na = 1, nb = 1;
    for (int d = 0; d < a->ndim; d++) na *= a->shape[d];
    for (int d = 0; d < b->ndim; d++) nb *= b->shape[d];
    if (na != n || nb != n) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "add_relu requires matching shapes");
        return AX_ERR_SHAPE_MISMATCH;
    }
    const float *ad = (const float *)a->storage->data + a->offset;
    const float *bd = (const float *)b->storage->data + b->offset;
    float *od = (float *)out->storage->data + out->offset;
    for (int64_t i = 0; i < n; i++) {
        float v = ad[i] + bd[i];
        od[i] = v > 0.0f ? v : 0.0f;
    }
    return AX_OK;
}

static ax_status_t cpu_axpy(const ax_tensor_t *x, float alpha, ax_tensor_t *y)
{
    if (x->dtype != AX_FLOAT32 || y->dtype != AX_FLOAT32)
        return AX_ERR_DTYPE_MISMATCH;
    int64_t nx = 1, ny = 1;
    for (int d = 0; d < x->ndim; d++) nx *= x->shape[d];
    for (int d = 0; d < y->ndim; d++) ny *= y->shape[d];
    if (nx != ny) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "axpy requires matching numel");
        return AX_ERR_SHAPE_MISMATCH;
    }
    const float *xd = (const float *)x->storage->data + x->offset;
    float *yd = (float *)y->storage->data + y->offset;
    for (int64_t i = 0; i < nx; i++) yd[i] += alpha * xd[i];
    return AX_OK;
}

static ax_status_t cpu_softmax_rowwise(const ax_tensor_t *in, ax_tensor_t *out)
{
    if (in->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32)
        return AX_ERR_DTYPE_MISMATCH;
    if (in->ndim != 2 || out->ndim != 2 ||
        in->shape[0] != out->shape[0] || in->shape[1] != out->shape[1]) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "softmax_rowwise requires matching 2d shapes");
        return AX_ERR_SHAPE_MISMATCH;
    }
    int64_t rows = in->shape[0], cols = in->shape[1];
    const float *id = (const float *)in->storage->data + in->offset;
    float *od = (float *)out->storage->data + out->offset;
    int64_t id_rs = in->strides[0], id_cs = in->strides[1];
    int64_t od_rs = out->strides[0], od_cs = out->strides[1];
    for (int64_t r = 0; r < rows; r++) {
        /* max-subtract for numerical stability */
        float mx = id[r * id_rs];
        for (int64_t c = 1; c < cols; c++) {
            float v = id[r * id_rs + c * id_cs];
            if (v > mx) mx = v;
        }
        float sum = 0.0f;
        for (int64_t c = 0; c < cols; c++) {
            float e = expf(id[r * id_rs + c * id_cs] - mx);
            od[r * od_rs + c * od_cs] = e;
            sum += e;
        }
        float inv = 1.0f / sum;
        for (int64_t c = 0; c < cols; c++) {
            od[r * od_rs + c * od_cs] *= inv;
        }
    }
    return AX_OK;
}

/* fused-scaling gemm: out = alpha * (a @ b) + beta * out.
   naive reference implementation — correct for arbitrary alpha/beta. */
static ax_status_t cpu_gemm_ex(const ax_tensor_t *a, const ax_tensor_t *b,
                                float alpha, float beta, ax_tensor_t *out)
{
    if (a->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "gemm_ex only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    if (a->ndim != 2 || b->ndim != 2 || out->ndim != 2) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "gemm_ex requires 2d tensors");
        return AX_ERR_SHAPE_MISMATCH;
    }
    int64_t m = a->shape[0], k = a->shape[1], n = b->shape[1];
    if (b->shape[0] != k || out->shape[0] != m || out->shape[1] != n) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "gemm_ex shape mismatch");
        return AX_ERR_SHAPE_MISMATCH;
    }

    float *ad = (float *)a->storage->data;
    float *bd = (float *)b->storage->data;
    float *od = (float *)out->storage->data;

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int64_t p = 0; p < k; p++) {
                float av = ad[a->offset + i * a->strides[0] + p * a->strides[1]];
                float bv = bd[b->offset + p * b->strides[0] + j * b->strides[1]];
                sum += av * bv;
            }
            int64_t oi = out->offset + i * out->strides[0] + j * out->strides[1];
            float prev = (beta == 0.0f) ? 0.0f : beta * od[oi];
            od[oi] = alpha * sum + prev;
        }
    }
    return AX_OK;
}

/* reduction ops */

static ax_status_t cpu_sum(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    if (in->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "sum only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }

    /* axis == -1 means reduce all dimensions to a scalar */
    if (axis == -1) {
        float total = 0.0f;
        int64_t n = tensor_numel(in);
        for (int64_t i = 0; i < n; i++) {
            total += tensor_get_f32(in, i);
        }
        tensor_set_f32(out, 0, total);
        return AX_OK;
    }

    /* reduce along a specific axis */
    if (axis < 0 || axis >= in->ndim) {
        ax_err_set(AX_ERR_INVALID_AXIS, "axis %d out of range for %d-dim tensor", axis, in->ndim);
        return AX_ERR_INVALID_AXIS;
    }

    int64_t n_out = tensor_numel(out);

    /* zero the output first */
    for (int64_t i = 0; i < n_out; i++) {
        tensor_set_f32(out, i, 0.0f);
    }

    /* iterate over all input elements, map each to its output position */
    int64_t n_in = tensor_numel(in);
    for (int64_t i = 0; i < n_in; i++) {
        float val = tensor_get_f32(in, i);

        /* compute output flat index by skipping the reduced axis */
        int64_t remaining = i;
        int64_t out_flat = 0;

        /* walk dimensions right to left */
        for (int d = in->ndim - 1; d >= 0; d--) {
            int64_t idx = remaining % in->shape[d];
            remaining /= in->shape[d];
            if (d != axis) {
                /* find corresponding output dimension */
                int od = (d > axis) ? d - 1 : d;
                int64_t os = 1;
                for (int k = out->ndim - 1; k > od; k--) os *= out->shape[k];
                out_flat += idx * os;
            }
        }

        float cur = tensor_get_f32(out, out_flat);
        tensor_set_f32(out, out_flat, cur + val);
    }
    return AX_OK;
}

static ax_status_t cpu_mean(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    /* mean = sum / count along the axis */
    ax_status_t s = cpu_sum(in, axis, out);
    if (s != AX_OK) return s;

    int64_t count;
    if (axis == -1) {
        count = tensor_numel(in);
    } else {
        count = in->shape[axis];
    }

    int64_t n = tensor_numel(out);
    float inv = 1.0f / (float)count;
    for (int64_t i = 0; i < n; i++) {
        tensor_set_f32(out, i, tensor_get_f32(out, i) * inv);
    }
    return AX_OK;
}

static ax_status_t cpu_max(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    if (in->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "max only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }

    if (axis == -1) {
        int64_t n = tensor_numel(in);
        float m = -FLT_MAX;
        for (int64_t i = 0; i < n; i++) {
            float v = tensor_get_f32(in, i);
            if (v > m) m = v;
        }
        tensor_set_f32(out, 0, m);
        return AX_OK;
    }

    /* initialize output to -FLT_MAX */
    int64_t n_out = tensor_numel(out);
    for (int64_t i = 0; i < n_out; i++) {
        tensor_set_f32(out, i, -FLT_MAX);
    }

    int64_t n_in = tensor_numel(in);
    for (int64_t i = 0; i < n_in; i++) {
        float val = tensor_get_f32(in, i);
        int64_t remaining = i;
        int64_t out_flat = 0;

        for (int d = in->ndim - 1; d >= 0; d--) {
            int64_t idx = remaining % in->shape[d];
            remaining /= in->shape[d];
            if (d != axis) {
                int od = (d > axis) ? d - 1 : d;
                int64_t os = 1;
                for (int k = out->ndim - 1; k > od; k--) os *= out->shape[k];
                out_flat += idx * os;
            }
        }

        float cur = tensor_get_f32(out, out_flat);
        if (val > cur) tensor_set_f32(out, out_flat, val);
    }
    return AX_OK;
}

static ax_status_t cpu_min(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    if (in->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "min only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }

    if (axis == -1) {
        int64_t n = tensor_numel(in);
        float m = FLT_MAX;
        for (int64_t i = 0; i < n; i++) {
            float v = tensor_get_f32(in, i);
            if (v < m) m = v;
        }
        tensor_set_f32(out, 0, m);
        return AX_OK;
    }

    int64_t n_out = tensor_numel(out);
    for (int64_t i = 0; i < n_out; i++) {
        tensor_set_f32(out, i, FLT_MAX);
    }

    int64_t n_in = tensor_numel(in);
    for (int64_t i = 0; i < n_in; i++) {
        float val = tensor_get_f32(in, i);
        int64_t remaining = i;
        int64_t out_flat = 0;

        for (int d = in->ndim - 1; d >= 0; d--) {
            int64_t idx = remaining % in->shape[d];
            remaining /= in->shape[d];
            if (d != axis) {
                int od = (d > axis) ? d - 1 : d;
                int64_t os = 1;
                for (int k = out->ndim - 1; k > od; k--) os *= out->shape[k];
                out_flat += idx * os;
            }
        }

        float cur = tensor_get_f32(out, out_flat);
        if (val < cur) tensor_set_f32(out, out_flat, val);
    }
    return AX_OK;
}

/* comparison ops */

static ax_status_t cpu_equal(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    if (a->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "equal only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    int64_t n = tensor_numel(out);
    for (int64_t i = 0; i < n; i++) {
        float va = bcast_get_f32(a, out, i);
        float vb = bcast_get_f32(b, out, i);
        tensor_set_f32(out, i, (va == vb) ? 1.0f : 0.0f);
    }
    return AX_OK;
}

static ax_status_t cpu_greater(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    if (a->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "greater only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    int64_t n = tensor_numel(out);
    for (int64_t i = 0; i < n; i++) {
        float va = bcast_get_f32(a, out, i);
        float vb = bcast_get_f32(b, out, i);
        tensor_set_f32(out, i, (va > vb) ? 1.0f : 0.0f);
    }
    return AX_OK;
}

/* data movement */

static ax_status_t cpu_fill(ax_tensor_t *t, double value) {
    if (t->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "fill only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    float v = (float)value;
    int64_t n = tensor_numel(t);
    for (int64_t i = 0; i < n; i++) {
        tensor_set_f32(t, i, v);
    }
    return AX_OK;
}

static ax_status_t cpu_copy(const ax_tensor_t *src, ax_tensor_t *dst) {
    if (src->dtype != AX_FLOAT32 || dst->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "copy only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    int64_t n = tensor_numel(src);
    for (int64_t i = 0; i < n; i++) {
        tensor_set_f32(dst, i, tensor_get_f32(src, i));
    }
    return AX_OK;
}

/* activations */

static ax_status_t cpu_relu(const ax_tensor_t *in, ax_tensor_t *out) {
    if (in->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "relu only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    int64_t n = tensor_numel(out);
    for (int64_t i = 0; i < n; i++) {
        float v = tensor_get_f32(in, i);
        tensor_set_f32(out, i, v > 0.0f ? v : 0.0f);
    }
    return AX_OK;
}

static ax_status_t cpu_sigmoid(const ax_tensor_t *in, ax_tensor_t *out) {
    if (in->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "sigmoid only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    int64_t n = tensor_numel(out);
    for (int64_t i = 0; i < n; i++) {
        float v = tensor_get_f32(in, i);
        tensor_set_f32(out, i, 1.0f / (1.0f + expf(-v)));
    }
    return AX_OK;
}

static ax_status_t cpu_tanh_op(const ax_tensor_t *in, ax_tensor_t *out) {
    if (in->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "tanh only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    int64_t n = tensor_numel(out);
    for (int64_t i = 0; i < n; i++) {
        float v = tensor_get_f32(in, i);
        tensor_set_f32(out, i, tanhf(v));
    }
    return AX_OK;
}

/* backend registration */

const ax_backend_ops_t ax_cpu_naive_ops = {
    .name       = "cpu_naive",
    .add        = cpu_add,
    .sub        = cpu_sub,
    .mul        = cpu_mul,
    .div_op     = cpu_div_op,
    .neg        = cpu_neg,
    .abs_op     = cpu_abs_op,
    .exp_op     = cpu_exp_op,
    .log_op     = cpu_log_op,
    .sqrt_op    = cpu_sqrt_op,
    .square     = cpu_square,
    .add_scalar = cpu_add_scalar,
    .mul_scalar = cpu_mul_scalar,
    .gemm       = cpu_gemm,
    .gemm_ex    = cpu_gemm_ex,
    .add_relu   = cpu_add_relu,
    .axpy       = cpu_axpy,
    .softmax_rowwise = cpu_softmax_rowwise,
    .sum        = cpu_sum,
    .mean       = cpu_mean,
    .max_op     = cpu_max,
    .min_op     = cpu_min,
    .equal      = cpu_equal,
    .greater    = cpu_greater,
    .fill       = cpu_fill,
    .copy       = cpu_copy,
    .relu       = cpu_relu,
    .sigmoid    = cpu_sigmoid,
    .tanh_op    = cpu_tanh_op,
};
