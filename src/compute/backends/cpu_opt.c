/* cpu_opt.c — optimized cpu backend with contiguous fast paths.
   falls back to cpu_naive for strided/broadcast cases.
   every op validates storage bounds before pointer access. */

#include "axiom/backend_ops.h"
#include "axiom/tensor.h"
#include "axiom/error.h"
#include "simd_defs.h"
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdint.h>

/* reference backend for fallback */
extern const ax_backend_ops_t ax_cpu_naive_ops;

/* validation helpers */

static inline int64_t fast_numel(const ax_tensor_t *t) {
    int64_t n = 1;
    for (int d = 0; d < t->ndim; d++) n *= t->shape[d];
    return n;
}

/* check that a tensor is contiguous float32 with valid storage.
   returns numel on success, -1 on failure. */
static inline int64_t validate_contig_f32(const ax_tensor_t *t) {
    if (!t || !t->storage || !t->storage->data) return -1;
    if (t->dtype != AX_FLOAT32) return -1;
    if (t->offset != 0) return -1;
    if (!ax_tensor_is_contiguous(t)) return -1;
    int64_t n = fast_numel(t);
    if (n <= 0) return -1;
    /* bounds check: numel * sizeof(float) must fit in storage */
    if ((size_t)n > t->storage->size_bytes / sizeof(float)) return -1;
    return n;
}

/* get raw float pointer for a validated contiguous tensor */
static inline float *raw_f32(const ax_tensor_t *t) {
    return (float *)t->storage->data;
}

/* check two tensors are both contiguous f32 with matching numel */
static inline int64_t validate_pair(const ax_tensor_t *a, const ax_tensor_t *b) {
    int64_t na = validate_contig_f32(a);
    int64_t nb = validate_contig_f32(b);
    if (na < 0 || nb < 0) return -1;
    /* for binary ops, shapes may differ (broadcast) — caller handles that */
    return na;
}

/* check all three tensors are contiguous f32 with same numel (no broadcast) */
static inline int64_t validate_triple_same(const ax_tensor_t *a, const ax_tensor_t *b, const ax_tensor_t *out) {
    int64_t na = validate_contig_f32(a);
    int64_t nb = validate_contig_f32(b);
    int64_t no = validate_contig_f32(out);
    if (na < 0 || nb < 0 || no < 0) return -1;
    if (na != nb || na != no) return -1;
    return na;
}


/* element-wise binary ops (contiguous, no broadcast) */

#define DEFINE_OPT_BINOP(name, expr, naive_fn) \
static ax_status_t opt_##name(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) { \
    int64_t n = validate_triple_same(a, b, out); \
    if (n < 0) return ax_cpu_naive_ops.naive_fn(a, b, out); \
    const float *ad = raw_f32(a); \
    const float *bd = raw_f32(b); \
    float *od = raw_f32(out); \
    int64_t i = 0; \
    int64_t vec_end = n - (n % AX_VF32_WIDTH); \
    for (; i < vec_end; i += AX_VF32_WIDTH) { \
        ax_vf32 va = ax_vf32_load(ad + i); \
        ax_vf32 vb = ax_vf32_load(bd + i); \
        ax_vf32_store(od + i, expr); \
    } \
    for (; i < n; i++) { \
        float va = ad[i], vb = bd[i]; \
        od[i] = (float)(expr); \
    } \
    return AX_OK; \
}

DEFINE_OPT_BINOP(add, ax_vf32_add(va, vb), add)
DEFINE_OPT_BINOP(sub, ax_vf32_sub(va, vb), sub)
DEFINE_OPT_BINOP(mul, ax_vf32_mul(va, vb), mul)
DEFINE_OPT_BINOP(div_op, ax_vf32_div(va, vb), div_op)


/* element-wise unary ops (contiguous) */

#define DEFINE_OPT_UNOP(name, expr, naive_fn) \
static ax_status_t opt_##name(const ax_tensor_t *in, ax_tensor_t *out) { \
    int64_t ni = validate_contig_f32(in); \
    int64_t no = validate_contig_f32(out); \
    if (ni < 0 || no < 0 || ni != no) return ax_cpu_naive_ops.naive_fn(in, out); \
    const float *id = raw_f32(in); \
    float *od = raw_f32(out); \
    int64_t n = ni; \
    int64_t i = 0; \
    int64_t vec_end = n - (n % AX_VF32_WIDTH); \
    for (; i < vec_end; i += AX_VF32_WIDTH) { \
        ax_vf32 v = ax_vf32_load(id + i); \
        ax_vf32_store(od + i, expr); \
    } \
    for (; i < n; i++) { \
        float v = id[i]; \
        od[i] = (float)(expr); \
    } \
    return AX_OK; \
}

DEFINE_OPT_UNOP(neg, ax_vf32_neg(v), neg)
DEFINE_OPT_UNOP(abs_op, ax_vf32_abs(v), abs_op)
DEFINE_OPT_UNOP(exp_op, ax_vf32_exp(v), exp_op)
DEFINE_OPT_UNOP(log_op, (v > 0.0f ? ax_vf32_log(v) : -FLT_MAX), log_op)
DEFINE_OPT_UNOP(sqrt_op, (v >= 0.0f ? ax_vf32_sqrt(v) : 0.0f), sqrt_op)
DEFINE_OPT_UNOP(square, ax_vf32_mul(v, v), square)

/* activations */
DEFINE_OPT_UNOP(relu, ax_vf32_relu(v), relu)
DEFINE_OPT_UNOP(sigmoid, ax_vf32_sigmoid(v), sigmoid)
DEFINE_OPT_UNOP(tanh_op, ax_vf32_tanh(v), tanh_op)


/* scalar ops */

static ax_status_t opt_add_scalar(const ax_tensor_t *in, double scalar, ax_tensor_t *out) {
    int64_t ni = validate_contig_f32(in);
    int64_t no = validate_contig_f32(out);
    if (ni < 0 || no < 0 || ni != no) return ax_cpu_naive_ops.add_scalar(in, scalar, out);
    const float *id = raw_f32(in);
    float *od = raw_f32(out);
    float s = (float)scalar;
    for (int64_t i = 0; i < ni; i++)
        od[i] = id[i] + s;
    return AX_OK;
}

static ax_status_t opt_mul_scalar(const ax_tensor_t *in, double scalar, ax_tensor_t *out) {
    int64_t ni = validate_contig_f32(in);
    int64_t no = validate_contig_f32(out);
    if (ni < 0 || no < 0 || ni != no) return ax_cpu_naive_ops.mul_scalar(in, scalar, out);
    const float *id = raw_f32(in);
    float *od = raw_f32(out);
    float s = (float)scalar;
    for (int64_t i = 0; i < ni; i++)
        od[i] = id[i] * s;
    return AX_OK;
}


/* gemm: contiguous fast path with stride-free inner loop.
   falls back to naive for strided inputs. tiling added in phase 1. */

static ax_status_t opt_gemm(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    if (a->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "gemm only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    if (a->ndim != 2 || b->ndim != 2 || out->ndim != 2) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "gemm requires 2d tensors");
        return AX_ERR_SHAPE_MISMATCH;
    }

    int64_t m = a->shape[0], k = a->shape[1], n = b->shape[1];
    if (b->shape[0] != k || out->shape[0] != m || out->shape[1] != n) {
        return ax_cpu_naive_ops.gemm(a, b, out);
    }

    /* check all three are contiguous */
    int64_t na = validate_contig_f32(a);
    int64_t nb = validate_contig_f32(b);
    int64_t no = validate_contig_f32(out);
    if (na < 0 || nb < 0 || no < 0) {
        return ax_cpu_naive_ops.gemm(a, b, out);
    }

    const float *ad = raw_f32(a);
    const float *bd = raw_f32(b);
    float *od = raw_f32(out);

    /* zero output first */
    memset(od, 0, (size_t)(m * n) * sizeof(float));

    /* ijk loop with contiguous pointer arithmetic.
       no stride computation — just row-major addressing.
       tiling optimization comes in phase 1. */
    for (int64_t i = 0; i < m; i++) {
        for (int64_t p = 0; p < k; p++) {
            float a_ip = ad[i * k + p];
            for (int64_t j = 0; j < n; j++) {
                od[i * n + j] += a_ip * bd[p * n + j];
            }
        }
    }
    return AX_OK;
}


/* reductions */

static ax_status_t opt_sum(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    /* fast path: full reduction on contiguous tensor */
    if (axis == -1) {
        int64_t n = validate_contig_f32(in);
        int64_t no = validate_contig_f32(out);
        if (n < 0 || no < 0 || no != 1) return ax_cpu_naive_ops.sum(in, axis, out);
        const float *d = raw_f32(in);
        double total = 0.0; /* double accumulator for precision */
        for (int64_t i = 0; i < n; i++) total += (double)d[i];
        raw_f32(out)[0] = (float)total;
        return AX_OK;
    }
    return ax_cpu_naive_ops.sum(in, axis, out);
}

static ax_status_t opt_mean(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    if (axis == -1) {
        int64_t n = validate_contig_f32(in);
        int64_t no = validate_contig_f32(out);
        if (n < 0 || no < 0 || no != 1) return ax_cpu_naive_ops.mean(in, axis, out);
        const float *d = raw_f32(in);
        double total = 0.0;
        for (int64_t i = 0; i < n; i++) total += (double)d[i];
        raw_f32(out)[0] = (float)(total / (double)n);
        return AX_OK;
    }
    return ax_cpu_naive_ops.mean(in, axis, out);
}

static ax_status_t opt_max(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    if (axis == -1) {
        int64_t n = validate_contig_f32(in);
        int64_t no = validate_contig_f32(out);
        if (n < 0 || no < 0 || no != 1) return ax_cpu_naive_ops.max_op(in, axis, out);
        const float *d = raw_f32(in);
        float mx = -FLT_MAX;
        for (int64_t i = 0; i < n; i++) if (d[i] > mx) mx = d[i];
        raw_f32(out)[0] = mx;
        return AX_OK;
    }
    return ax_cpu_naive_ops.max_op(in, axis, out);
}

static ax_status_t opt_min(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    if (axis == -1) {
        int64_t n = validate_contig_f32(in);
        int64_t no = validate_contig_f32(out);
        if (n < 0 || no < 0 || no != 1) return ax_cpu_naive_ops.min_op(in, axis, out);
        const float *d = raw_f32(in);
        float mn = FLT_MAX;
        for (int64_t i = 0; i < n; i++) if (d[i] < mn) mn = d[i];
        raw_f32(out)[0] = mn;
        return AX_OK;
    }
    return ax_cpu_naive_ops.min_op(in, axis, out);
}


/* comparisons */

static ax_status_t opt_equal(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    int64_t n = validate_triple_same(a, b, out);
    if (n < 0) return ax_cpu_naive_ops.equal(a, b, out);
    const float *ad = raw_f32(a);
    const float *bd = raw_f32(b);
    float *od = raw_f32(out);
    for (int64_t i = 0; i < n; i++)
        od[i] = (ad[i] == bd[i]) ? 1.0f : 0.0f;
    return AX_OK;
}

static ax_status_t opt_greater(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    int64_t n = validate_triple_same(a, b, out);
    if (n < 0) return ax_cpu_naive_ops.greater(a, b, out);
    const float *ad = raw_f32(a);
    const float *bd = raw_f32(b);
    float *od = raw_f32(out);
    for (int64_t i = 0; i < n; i++)
        od[i] = (ad[i] > bd[i]) ? 1.0f : 0.0f;
    return AX_OK;
}


/* data movement */

static ax_status_t opt_fill(ax_tensor_t *t, double value) {
    int64_t n = validate_contig_f32(t);
    if (n < 0) return ax_cpu_naive_ops.fill(t, value);
    float v = (float)value;
    float *d = raw_f32(t);
    /* special case: fill with 0 uses memset (fast) */
    if (v == 0.0f) {
        memset(d, 0, (size_t)n * sizeof(float));
    } else {
        for (int64_t i = 0; i < n; i++) d[i] = v;
    }
    return AX_OK;
}

static ax_status_t opt_copy(const ax_tensor_t *src, ax_tensor_t *dst) {
    int64_t ns = validate_contig_f32(src);
    int64_t nd = validate_contig_f32(dst);
    if (ns < 0 || nd < 0 || ns != nd) return ax_cpu_naive_ops.copy(src, dst);
    memcpy(raw_f32(dst), raw_f32(src), (size_t)ns * sizeof(float));
    return AX_OK;
}


/* vtable registration */

const ax_backend_ops_t ax_cpu_opt_ops = {
    .name       = "cpu_opt",
    .add        = opt_add,
    .sub        = opt_sub,
    .mul        = opt_mul,
    .div_op     = opt_div_op,
    .neg        = opt_neg,
    .abs_op     = opt_abs_op,
    .exp_op     = opt_exp_op,
    .log_op     = opt_log_op,
    .sqrt_op    = opt_sqrt_op,
    .square     = opt_square,
    .add_scalar = opt_add_scalar,
    .mul_scalar = opt_mul_scalar,
    .gemm       = opt_gemm,
    .sum        = opt_sum,
    .mean       = opt_mean,
    .max_op     = opt_max,
    .min_op     = opt_min,
    .equal      = opt_equal,
    .greater    = opt_greater,
    .fill       = opt_fill,
    .copy       = opt_copy,
    .relu       = opt_relu,
    .sigmoid    = opt_sigmoid,
    .tanh_op    = opt_tanh_op,
};
