/* dispatch.c — routes compute calls to the active backend */

#include "axiom/compute.h"
#include "axiom/error.h"
#include <stddef.h>

/* declared in cpu_naive.c */
extern const ax_backend_ops_t ax_cpu_naive_ops;

/* registered backends table */
static const ax_backend_ops_t *backends[AX_BACKEND_COUNT] = { NULL };

/* currently active backend */
static ax_backend_id_t active_id = AX_BACKEND_CPU_NAIVE;
static const ax_backend_ops_t *active_ops = NULL;
static int compute_initialized = 0;

/* ensure the compute system is ready; called lazily from dispatch macros */
static void ensure_compute_init(void) {
    if (compute_initialized) return;
    ax_compute_init();
}

ax_status_t ax_compute_init(void) {
    /* register the cpu naive backend — always available */
    backends[AX_BACKEND_CPU_NAIVE] = &ax_cpu_naive_ops;

    /* select the best available backend (just cpu naive for now) */
    active_id = AX_BACKEND_CPU_NAIVE;
    active_ops = backends[AX_BACKEND_CPU_NAIVE];
    compute_initialized = 1;
    return AX_OK;
}

void ax_compute_shutdown(void) {
    active_ops = NULL;
    compute_initialized = 0;
}

ax_status_t ax_compute_set_backend(ax_backend_id_t id) {
    if (id >= AX_BACKEND_COUNT || !backends[id]) {
        ax_err_set(AX_ERR_BACKEND, "backend %d not available", (int)id);
        return AX_ERR_BACKEND;
    }
    active_id = id;
    active_ops = backends[id];
    return AX_OK;
}

ax_backend_id_t ax_compute_get_backend(void) {
    return active_id;
}

const ax_backend_ops_t *ax_compute_get_ops(void) {
    return active_ops;
}

ax_status_t ax_compute_register_backend(ax_backend_id_t id, const ax_backend_ops_t *ops) {
    if (id >= AX_BACKEND_COUNT) {
        ax_err_set(AX_ERR_BACKEND, "invalid backend id %d", (int)id);
        return AX_ERR_BACKEND;
    }
    if (!ops) {
        ax_err_set(AX_ERR_NULL_ARG, "null ops table");
        return AX_ERR_NULL_ARG;
    }
    backends[id] = ops;
    return AX_OK;
}

/* dispatch helpers */
/* macro to reduce boilerplate: check that backend is initialized,
   check that the op is implemented, then call it */
#define DISPATCH_BINOP(op, a, b, out) \
    do { \
        ensure_compute_init(); \
        if (!active_ops) { \
            ax_err_set(AX_ERR_BACKEND, "compute not initialized"); \
            return AX_ERR_BACKEND; \
        } \
        if (!active_ops->op) { \
            ax_err_set(AX_ERR_NOT_IMPLEMENTED, #op " not implemented in %s", active_ops->name); \
            return AX_ERR_NOT_IMPLEMENTED; \
        } \
        return active_ops->op(a, b, out); \
    } while (0)

#define DISPATCH_UNOP(op, in, out) \
    do { \
        ensure_compute_init(); \
        if (!active_ops) { \
            ax_err_set(AX_ERR_BACKEND, "compute not initialized"); \
            return AX_ERR_BACKEND; \
        } \
        if (!active_ops->op) { \
            ax_err_set(AX_ERR_NOT_IMPLEMENTED, #op " not implemented in %s", active_ops->name); \
            return AX_ERR_NOT_IMPLEMENTED; \
        } \
        return active_ops->op(in, out); \
    } while (0)

/* binary ops */
ax_status_t ax_compute_add(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) { DISPATCH_BINOP(add, a, b, out); }
ax_status_t ax_compute_sub(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) { DISPATCH_BINOP(sub, a, b, out); }
ax_status_t ax_compute_mul(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) { DISPATCH_BINOP(mul, a, b, out); }
ax_status_t ax_compute_div(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) { DISPATCH_BINOP(div_op, a, b, out); }

/* unary ops */
ax_status_t ax_compute_neg(const ax_tensor_t *in, ax_tensor_t *out) { DISPATCH_UNOP(neg, in, out); }
ax_status_t ax_compute_abs(const ax_tensor_t *in, ax_tensor_t *out) { DISPATCH_UNOP(abs_op, in, out); }
ax_status_t ax_compute_exp(const ax_tensor_t *in, ax_tensor_t *out) { DISPATCH_UNOP(exp_op, in, out); }
ax_status_t ax_compute_log(const ax_tensor_t *in, ax_tensor_t *out) { DISPATCH_UNOP(log_op, in, out); }
ax_status_t ax_compute_sqrt(const ax_tensor_t *in, ax_tensor_t *out) { DISPATCH_UNOP(sqrt_op, in, out); }
ax_status_t ax_compute_square(const ax_tensor_t *in, ax_tensor_t *out) { DISPATCH_UNOP(square, in, out); }

/* scalar ops */
ax_status_t ax_compute_add_scalar(const ax_tensor_t *in, double scalar, ax_tensor_t *out) {
    ensure_compute_init();
    if (!active_ops) { ax_err_set(AX_ERR_BACKEND, "compute not initialized"); return AX_ERR_BACKEND; }
    if (!active_ops->add_scalar) { ax_err_set(AX_ERR_NOT_IMPLEMENTED, "add_scalar not implemented"); return AX_ERR_NOT_IMPLEMENTED; }
    return active_ops->add_scalar(in, scalar, out);
}

ax_status_t ax_compute_mul_scalar(const ax_tensor_t *in, double scalar, ax_tensor_t *out) {
    ensure_compute_init();
    if (!active_ops) { ax_err_set(AX_ERR_BACKEND, "compute not initialized"); return AX_ERR_BACKEND; }
    if (!active_ops->mul_scalar) { ax_err_set(AX_ERR_NOT_IMPLEMENTED, "mul_scalar not implemented"); return AX_ERR_NOT_IMPLEMENTED; }
    return active_ops->mul_scalar(in, scalar, out);
}

/* matrix ops */
ax_status_t ax_compute_gemm(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) { DISPATCH_BINOP(gemm, a, b, out); }

/* reduction ops */
ax_status_t ax_compute_sum(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    ensure_compute_init();
    if (!active_ops) { ax_err_set(AX_ERR_BACKEND, "compute not initialized"); return AX_ERR_BACKEND; }
    if (!active_ops->sum) { ax_err_set(AX_ERR_NOT_IMPLEMENTED, "sum not implemented"); return AX_ERR_NOT_IMPLEMENTED; }
    return active_ops->sum(in, axis, out);
}

ax_status_t ax_compute_mean(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    ensure_compute_init();
    if (!active_ops) { ax_err_set(AX_ERR_BACKEND, "compute not initialized"); return AX_ERR_BACKEND; }
    if (!active_ops->mean) { ax_err_set(AX_ERR_NOT_IMPLEMENTED, "mean not implemented"); return AX_ERR_NOT_IMPLEMENTED; }
    return active_ops->mean(in, axis, out);
}

ax_status_t ax_compute_max(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    ensure_compute_init();
    if (!active_ops) { ax_err_set(AX_ERR_BACKEND, "compute not initialized"); return AX_ERR_BACKEND; }
    if (!active_ops->max_op) { ax_err_set(AX_ERR_NOT_IMPLEMENTED, "max not implemented"); return AX_ERR_NOT_IMPLEMENTED; }
    return active_ops->max_op(in, axis, out);
}

ax_status_t ax_compute_min(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    ensure_compute_init();
    if (!active_ops) { ax_err_set(AX_ERR_BACKEND, "compute not initialized"); return AX_ERR_BACKEND; }
    if (!active_ops->min_op) { ax_err_set(AX_ERR_NOT_IMPLEMENTED, "min not implemented"); return AX_ERR_NOT_IMPLEMENTED; }
    return active_ops->min_op(in, axis, out);
}

/* comparison ops */
ax_status_t ax_compute_equal(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) { DISPATCH_BINOP(equal, a, b, out); }
ax_status_t ax_compute_greater(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) { DISPATCH_BINOP(greater, a, b, out); }

/* data movement */
ax_status_t ax_compute_fill(ax_tensor_t *t, double value) {
    ensure_compute_init();
    if (!active_ops) { ax_err_set(AX_ERR_BACKEND, "compute not initialized"); return AX_ERR_BACKEND; }
    if (!active_ops->fill) { ax_err_set(AX_ERR_NOT_IMPLEMENTED, "fill not implemented"); return AX_ERR_NOT_IMPLEMENTED; }
    return active_ops->fill(t, value);
}

ax_status_t ax_compute_copy(const ax_tensor_t *src, ax_tensor_t *dst) {
    ensure_compute_init();
    if (!active_ops) { ax_err_set(AX_ERR_BACKEND, "compute not initialized"); return AX_ERR_BACKEND; }
    if (!active_ops->copy) { ax_err_set(AX_ERR_NOT_IMPLEMENTED, "copy not implemented"); return AX_ERR_NOT_IMPLEMENTED; }
    return active_ops->copy(src, dst);
}

/* activations */
ax_status_t ax_compute_relu(const ax_tensor_t *in, ax_tensor_t *out) { DISPATCH_UNOP(relu, in, out); }
ax_status_t ax_compute_sigmoid(const ax_tensor_t *in, ax_tensor_t *out) { DISPATCH_UNOP(sigmoid, in, out); }
ax_status_t ax_compute_tanh(const ax_tensor_t *in, ax_tensor_t *out) { DISPATCH_UNOP(tanh_op, in, out); }
