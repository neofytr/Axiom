/* dispatch.c — routes compute calls to the active backend */

#ifdef __linux__
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif

#include "axiom/compute.h"
#include "axiom/device.h"
#include "axiom/error.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/* thread counts for hybrid cpu awareness. on hybrid cpus (P+E cores),
   large GEMMs use all_threads (P+E) while small ops use fast_threads
   (P-cores only). on non-hybrid, both are equal. set by ax_autotune_threads. */
int ax_gemm_fast_threads = 0;
int ax_gemm_all_threads = 0;

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* declared in cpu_naive.c and cpu_opt.c */
extern const ax_backend_ops_t ax_cpu_naive_ops;

/* cpu_opt vtable(s). under AX_CPU_ISA_DISPATCH the cpu_opt source is
   compiled twice — once with avx2/fma and once without — producing
   two distinct vtable symbols. ax_compute_init picks between them at
   runtime via __builtin_cpu_supports. single-build mode (default)
   exports the plain ax_cpu_opt_ops symbol. */
#ifdef AX_CPU_ISA_DISPATCH
extern const ax_backend_ops_t ax_cpu_opt_ops_avx512;
extern const ax_backend_ops_t ax_cpu_opt_ops_avx2;
extern const ax_backend_ops_t ax_cpu_opt_ops_scalar;
extern void ax_cpu_opt_tune_init_avx512(void);
extern void ax_cpu_opt_tune_init_avx2(void);
extern void ax_cpu_opt_tune_init_scalar(void);
#else
extern const ax_backend_ops_t ax_cpu_opt_ops;
extern void ax_cpu_opt_tune_init(void);
#endif

#ifdef AX_HAVE_CUDA
extern const ax_backend_ops_t ax_cuda_ops;
#endif

/* registered backends table */
static const ax_backend_ops_t *backends[AX_BACKEND_COUNT] = { NULL };

/* device -> owning-backend table. any backend whose vtable declares
   ops->device != AX_DEVICE_COUNT is installed here during init and its
   lifecycle init hook is fired. core memory paths route through this
   table instead of #ifdef'ing on specific device types. */
static const ax_backend_ops_t *device_backends[AX_DEVICE_COUNT] = { NULL };
static int device_backend_inited[AX_DEVICE_COUNT] = { 0 };

/* currently active backend */
static ax_backend_id_t active_id = AX_BACKEND_CPU_NAIVE;
static const ax_backend_ops_t *active_ops = NULL;
static int compute_initialized = 0;

/* claim ownership of ops->device in the device table and fire ops->init
   exactly once per process. safe to call with NULL or with a backend
   that doesn't own a device. */
static void register_device_owner(const ax_backend_ops_t *ops) {
    if (!ops) return;
    if ((int)ops->device < 0 || (int)ops->device >= AX_DEVICE_COUNT) return;
    if (ops->device == AX_DEVICE_CPU) return;  /* cpu is handled inline */
    if (device_backends[ops->device]) return;  /* first one wins */
    device_backends[ops->device] = ops;
    if (!device_backend_inited[ops->device] && ops->init) {
        ops->init();
    }
    device_backend_inited[ops->device] = 1;
}

/* exposed to core (tensor.c etc.) so non-cpu code paths can look up
   memory/transfer hooks. returns NULL for AX_DEVICE_CPU or for devices
   whose owning backend is not compiled in. */
const ax_backend_ops_t *ax_backend_for_device(ax_device_t device) {
    if ((int)device < 0 || (int)device >= AX_DEVICE_COUNT) return NULL;
    return device_backends[device];
}

/* ensure the compute system is ready; called lazily from dispatch macros */
static void ensure_compute_init(void) {
    if (compute_initialized) return;
    ax_compute_init();
}

ax_status_t ax_compute_init(void) {
    /* register cpu backends. under AX_CPU_ISA_DISPATCH we probe for
       avx2+fma and pick the corresponding cpu_opt variant at runtime;
       otherwise the single-build vtable wins unconditionally. */
    backends[AX_BACKEND_CPU_NAIVE] = &ax_cpu_naive_ops;
#ifdef AX_CPU_ISA_DISPATCH
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma")) {
        backends[AX_BACKEND_CPU_SIMD] = &ax_cpu_opt_ops_avx512;
    } else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        backends[AX_BACKEND_CPU_SIMD] = &ax_cpu_opt_ops_avx2;
    } else {
        backends[AX_BACKEND_CPU_SIMD] = &ax_cpu_opt_ops_scalar;
    }
#else
    backends[AX_BACKEND_CPU_SIMD] = &ax_cpu_opt_ops;
#endif
#ifdef AX_HAVE_CUDA
    backends[AX_BACKEND_CUDA]      = &ax_cuda_ops;
#endif

    /* claim device ownership + fire init hooks for any backend that
       manages a non-cpu device. cpu backends skip this. */
    for (int i = 0; i < AX_BACKEND_COUNT; i++) {
        register_device_owner(backends[i]);
    }

    /* select the optimized backend by default (falls back to naive internally) */
    active_id = AX_BACKEND_CPU_SIMD;
    active_ops = backends[AX_BACKEND_CPU_SIMD];
    compute_initialized = 1;

    /* explicit fallback for baremetal: the cpu_opt tile-size init is
       normally driven by an __attribute__((constructor)), but some
       embedded crt0 scripts don't walk .init_array, so we also call it
       here. idempotent — a second call is a no-op on hosted builds.
       under AX_CPU_ISA_DISPATCH we call the variant that got selected. */
#ifdef AX_CPU_ISA_DISPATCH
    if (active_ops == &ax_cpu_opt_ops_avx512) {
        ax_cpu_opt_tune_init_avx512();
    } else if (active_ops == &ax_cpu_opt_ops_avx2) {
        ax_cpu_opt_tune_init_avx2();
    } else {
        ax_cpu_opt_tune_init_scalar();
    }
#else
    ax_cpu_opt_tune_init();
#endif

    /* runtime hybrid-cpu autotune. only emits a stderr line when it
       actually calibrates (i.e. user did not pin threads explicitly).
       compiled out entirely on baremetal (AX_NO_AUTOTUNE). */
    ax_autotune_threads();

    return AX_OK;
}

void ax_compute_shutdown(void) {
    /* fire device-owner shutdown hooks in reverse order */
    for (int d = AX_DEVICE_COUNT - 1; d >= 0; d--) {
        const ax_backend_ops_t *ops = device_backends[d];
        if (ops && ops->shutdown && device_backend_inited[d]) {
            ops->shutdown();
        }
        device_backends[d] = NULL;
        device_backend_inited[d] = 0;
    }
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
    /* also install as the owner of its device, if any */
    register_device_owner(ops);
    return AX_OK;
}

/* ---- generic device-management api (axiom/device.h) ----
   these route through the owning backend's vtable; no device-specific
   code leaks into core. */

int ax_device_count(ax_device_t device) {
    if (device == AX_DEVICE_CPU) return 1;
    ensure_compute_init();
    const ax_backend_ops_t *ops = ax_backend_for_device(device);
    if (!ops || !ops->device_count) return 0;
    return ops->device_count();
}

void ax_device_synchronize(ax_device_t device) {
    if (device == AX_DEVICE_CPU) return;
    ensure_compute_init();
    const ax_backend_ops_t *ops = ax_backend_for_device(device);
    if (ops && ops->synchronize) ops->synchronize();
}

int ax_device_is_available(ax_device_t device) {
    if (device == AX_DEVICE_CPU) return 1;
    ensure_compute_init();
    return ax_backend_for_device(device) != NULL;
}

/* bump out's generation counter on successful op completion. used as
   a return-value adapter by every ax_compute_* wrapper so cpu_opt's
   pack_b cache (keyed on the raw b->storage pointer + generation) can
   detect in-place mutations and invalidate stale entries. one-line
   atomic-free increment; perf impact is in the noise. */
static inline ax_status_t dispatch_touch_on_ok(ax_tensor_t *out, ax_status_t s) {
    if (s == AX_OK && out) ax_storage_touch(out->storage);
    return s;
}

/* dispatch helpers.
   every wrapper bumps out->storage->generation on successful completion
   via dispatch_touch_on_ok so read-only caches (pack_b in cpu_opt, ...)
   keyed on the raw storage pointer can detect in-place mutations. */
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
        return dispatch_touch_on_ok((out), active_ops->op(a, b, out)); \
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
        return dispatch_touch_on_ok((out), active_ops->op(in, out)); \
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
    return dispatch_touch_on_ok(out, active_ops->add_scalar(in, scalar, out));
}

ax_status_t ax_compute_mul_scalar(const ax_tensor_t *in, double scalar, ax_tensor_t *out) {
    ensure_compute_init();
    if (!active_ops) { ax_err_set(AX_ERR_BACKEND, "compute not initialized"); return AX_ERR_BACKEND; }
    if (!active_ops->mul_scalar) { ax_err_set(AX_ERR_NOT_IMPLEMENTED, "mul_scalar not implemented"); return AX_ERR_NOT_IMPLEMENTED; }
    return dispatch_touch_on_ok(out, active_ops->mul_scalar(in, scalar, out));
}

/* matrix ops */
ax_status_t ax_compute_gemm(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) { DISPATCH_BINOP(gemm, a, b, out); }

/* optional transposed-b gemm: out = a @ b^T. returns
   AX_ERR_NOT_IMPLEMENTED when the active backend lacks the slot — callers
   should check and fall back to physical transpose + plain gemm. */
ax_status_t ax_compute_gemm_nt(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out)
{
    ensure_compute_init();
    if (!active_ops) { ax_err_set(AX_ERR_BACKEND, "compute not initialized"); return AX_ERR_BACKEND; }
    if (!active_ops->gemm_nt) {
        ax_err_set(AX_ERR_NOT_IMPLEMENTED, "gemm_nt not implemented in %s", active_ops->name);
        return AX_ERR_NOT_IMPLEMENTED;
    }
    return dispatch_touch_on_ok(out, active_ops->gemm_nt(a, b, out));
}

/* optional transposed-a gemm: out = a^T @ b. */
ax_status_t ax_compute_gemm_tn(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out)
{
    ensure_compute_init();
    if (!active_ops) { ax_err_set(AX_ERR_BACKEND, "compute not initialized"); return AX_ERR_BACKEND; }
    if (!active_ops->gemm_tn) {
        ax_err_set(AX_ERR_NOT_IMPLEMENTED, "gemm_tn not implemented in %s", active_ops->name);
        return AX_ERR_NOT_IMPLEMENTED;
    }
    return dispatch_touch_on_ok(out, active_ops->gemm_tn(a, b, out));
}

/* fused matmul+relu */
ax_status_t ax_compute_gemm_relu(const ax_tensor_t *a, const ax_tensor_t *b,
                                  const ax_tensor_t *bias, ax_tensor_t *out)
{
    ensure_compute_init();
    if (!active_ops || !active_ops->gemm_relu) {
        ax_err_set(AX_ERR_NOT_IMPLEMENTED, "gemm_relu not implemented in %s",
                   active_ops ? active_ops->name : "none");
        return AX_ERR_NOT_IMPLEMENTED;
    }
    return dispatch_touch_on_ok(out, active_ops->gemm_relu(a, b, bias, out));
}
int ax_compute_has_gemm_relu(void) { ensure_compute_init(); return (active_ops && active_ops->gemm_relu) ? 1 : 0; }

int ax_compute_has_gemm_nt(void) { ensure_compute_init(); return (active_ops && active_ops->gemm_nt) ? 1 : 0; }
int ax_compute_has_gemm_tn(void) { ensure_compute_init(); return (active_ops && active_ops->gemm_tn) ? 1 : 0; }

/* fused-scaling gemm: out = alpha * (a @ b) + beta * out. */
ax_status_t ax_compute_gemm_ex(const ax_tensor_t *a, const ax_tensor_t *b,
                                float alpha, float beta, ax_tensor_t *out)
{
    ensure_compute_init();
    if (!active_ops) { ax_err_set(AX_ERR_BACKEND, "compute not initialized"); return AX_ERR_BACKEND; }
    if (!active_ops->gemm_ex) {
        ax_err_set(AX_ERR_NOT_IMPLEMENTED, "gemm_ex not implemented in %s", active_ops->name);
        return AX_ERR_NOT_IMPLEMENTED;
    }
    return dispatch_touch_on_ok(out, active_ops->gemm_ex(a, b, alpha, beta, out));
}

int ax_compute_has_gemm_ex(void) { ensure_compute_init(); return (active_ops && active_ops->gemm_ex) ? 1 : 0; }

/* fused relu(a + b) */
ax_status_t ax_compute_add_relu(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out)
{
    ensure_compute_init();
    if (!active_ops) { ax_err_set(AX_ERR_BACKEND, "compute not initialized"); return AX_ERR_BACKEND; }
    if (!active_ops->add_relu) {
        ax_err_set(AX_ERR_NOT_IMPLEMENTED, "add_relu not implemented in %s", active_ops->name);
        return AX_ERR_NOT_IMPLEMENTED;
    }
    return dispatch_touch_on_ok(out, active_ops->add_relu(a, b, out));
}
int ax_compute_has_add_relu(void) { ensure_compute_init(); return (active_ops && active_ops->add_relu) ? 1 : 0; }

/* y += alpha * x, in-place on y */
ax_status_t ax_compute_axpy(const ax_tensor_t *x, float alpha, ax_tensor_t *y)
{
    ensure_compute_init();
    if (!active_ops) { ax_err_set(AX_ERR_BACKEND, "compute not initialized"); return AX_ERR_BACKEND; }
    if (!active_ops->axpy) {
        ax_err_set(AX_ERR_NOT_IMPLEMENTED, "axpy not implemented in %s", active_ops->name);
        return AX_ERR_NOT_IMPLEMENTED;
    }
    return dispatch_touch_on_ok(y, active_ops->axpy(x, alpha, y));
}
int ax_compute_has_axpy(void) { ensure_compute_init(); return (active_ops && active_ops->axpy) ? 1 : 0; }

/* row-wise stable softmax */
ax_status_t ax_compute_softmax_rowwise(const ax_tensor_t *in, ax_tensor_t *out)
{
    ensure_compute_init();
    if (!active_ops) { ax_err_set(AX_ERR_BACKEND, "compute not initialized"); return AX_ERR_BACKEND; }
    if (!active_ops->softmax_rowwise) {
        ax_err_set(AX_ERR_NOT_IMPLEMENTED, "softmax_rowwise not implemented in %s", active_ops->name);
        return AX_ERR_NOT_IMPLEMENTED;
    }
    return dispatch_touch_on_ok(out, active_ops->softmax_rowwise(in, out));
}
int ax_compute_has_softmax_rowwise(void) { ensure_compute_init(); return (active_ops && active_ops->softmax_rowwise) ? 1 : 0; }

/* fused bias add: out[..., axis, ...] = in[..., axis, ...] + bias. */
ax_status_t ax_compute_bias_add(const ax_tensor_t *in, const ax_tensor_t *bias,
                                 int axis, ax_tensor_t *out)
{
    ensure_compute_init();
    if (!active_ops) { ax_err_set(AX_ERR_BACKEND, "compute not initialized"); return AX_ERR_BACKEND; }
    if (!active_ops->bias_add) {
        ax_err_set(AX_ERR_NOT_IMPLEMENTED, "bias_add not implemented in %s", active_ops->name);
        return AX_ERR_NOT_IMPLEMENTED;
    }
    return dispatch_touch_on_ok(out, active_ops->bias_add(in, bias, axis, out));
}

int ax_compute_has_bias_add(void) { ensure_compute_init(); return (active_ops && active_ops->bias_add) ? 1 : 0; }

/* argmax along an axis; output is int64 with reduced dim removed. */
ax_status_t ax_compute_argmax(const ax_tensor_t *in, int axis, ax_tensor_t *out)
{
    ensure_compute_init();
    if (!active_ops) { ax_err_set(AX_ERR_BACKEND, "compute not initialized"); return AX_ERR_BACKEND; }
    if (!active_ops->argmax) {
        ax_err_set(AX_ERR_NOT_IMPLEMENTED, "argmax not implemented in %s", active_ops->name);
        return AX_ERR_NOT_IMPLEMENTED;
    }
    return dispatch_touch_on_ok(out, active_ops->argmax(in, axis, out));
}

/* implicit im2col conv gemm — optional backend op */
ax_status_t ax_compute_conv_gemm(const ax_tensor_t *weight,
                                  const ax_conv_params_t *params,
                                  ax_tensor_t *out)
{
    ensure_compute_init();
    if (!active_ops) { ax_err_set(AX_ERR_BACKEND, "compute not initialized"); return AX_ERR_BACKEND; }
    if (!active_ops->conv_gemm) {
        ax_err_set(AX_ERR_NOT_IMPLEMENTED, "conv_gemm not implemented in %s", active_ops->name);
        return AX_ERR_NOT_IMPLEMENTED;
    }
    return dispatch_touch_on_ok(out, active_ops->conv_gemm(weight, params, out));
}

int ax_compute_has_conv_gemm(void)
{
    ensure_compute_init();
    return (active_ops && active_ops->conv_gemm) ? 1 : 0;
}

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

/* fused optimizer updates */
ax_status_t ax_compute_adam_update(ax_tensor_t *weight, ax_tensor_t *grad,
                                    ax_tensor_t *m, ax_tensor_t *v,
                                    float lr, float beta1, float beta2, float eps,
                                    float weight_decay, float bc1, float bc2,
                                    bool decoupled)
{
    ensure_compute_init();
    if (!active_ops || !active_ops->adam_update)
        return AX_ERR_NOT_IMPLEMENTED;
    return active_ops->adam_update(weight, grad, m, v, lr, beta1, beta2, eps,
                                   weight_decay, bc1, bc2, decoupled);
}

ax_status_t ax_compute_sgd_update(ax_tensor_t *weight, ax_tensor_t *grad,
                                    ax_tensor_t *momentum_buf,
                                    float lr, float momentum, float weight_decay,
                                    bool nesterov)
{
    ensure_compute_init();
    if (!active_ops || !active_ops->sgd_update)
        return AX_ERR_NOT_IMPLEMENTED;
    return active_ops->sgd_update(weight, grad, momentum_buf, lr, momentum,
                                   weight_decay, nesterov);
}

int ax_compute_has_adam_update(void) { ensure_compute_init(); return (active_ops && active_ops->adam_update) ? 1 : 0; }
int ax_compute_has_sgd_update(void) { ensure_compute_init(); return (active_ops && active_ops->sgd_update) ? 1 : 0; }

/* thread control */
void ax_set_num_threads(int n)
{
#ifdef _OPENMP
    if (n <= 0) {
        /* reset to default: hardware concurrency */
        omp_set_num_threads(omp_get_num_procs());
    } else {
        omp_set_num_threads(n);
    }
#else
    (void)n;
#endif
}

int ax_get_num_threads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* hybrid-cpu autotune.

   strategy: pin a tiny serial fma-style kernel to each available logical
   cpu via sched_setaffinity, time it, and treat cores within 25% of the
   fastest as "fast". sets omp default to the fast-core count so that
   p-cores aren't blocked at omp barriers waiting on slower e-cores.

   bounded under 200ms total (kernel sized at ~1m ops, ~1ms per core on
   modern hardware, ~16 cores typical -> ~16ms; 200ms gives huge headroom). */

static double ax_autotune_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

#ifdef __linux__
/* read a single long from a sysfs file. returns -1 on failure. */
static long read_sysfs_long(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    long v = -1;
    if (fscanf(f, "%ld", &v) != 1) v = -1;
    fclose(f);
    return v;
}

/* classify cores by sysfs frequency: read base_frequency (or
   cpuinfo_max_freq) for each online cpu, find the fastest tier, and
   mark cpus within 85% of max as "fast". this replaces the 200ms
   per-core probe loop on systems with frequency info (all modern
   linux kernels with cpufreq). returns the number of fast cpus
   written to out_fast[], or -1 if frequency data is unavailable
   (caller should fall back to the probe). */
static int classify_cores_by_freq(const int *cpu_ids, int n_cpus,
                                   int *out_fast, int max_out)
{
    long freqs[CPU_SETSIZE];
    long max_freq = 0;
    int have_freq = 0;

    for (int i = 0; i < n_cpus; i++) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/base_frequency",
                 cpu_ids[i]);
        freqs[i] = read_sysfs_long(path);
        if (freqs[i] <= 0) {
            /* try cpuinfo_max_freq as fallback (always present if cpufreq is) */
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq",
                     cpu_ids[i]);
            freqs[i] = read_sysfs_long(path);
        }
        if (freqs[i] > 0) {
            have_freq = 1;
            if (freqs[i] > max_freq) max_freq = freqs[i];
        }
    }

    if (!have_freq || max_freq <= 0) return -1;

    /* cpus within 85% of the fastest are "fast". on alder lake P-cores
       run at 2.5 GHz base vs E-cores at 1.8 GHz → ratio 0.72 → E-cores
       fail the 85% test. on non-hybrid systems all cores have the same
       freq → all pass. */
    long threshold = (long)((double)max_freq * 0.85);
    int fc = 0;
    for (int i = 0; i < n_cpus && fc < max_out; i++) {
        if (freqs[i] >= threshold) {
            out_fast[fc++] = cpu_ids[i];
        }
    }
    return fc > 0 ? fc : -1;
}

/* tiny serial workload. volatile sink prevents the optimizer from
   collapsing the loop. ~1m fma-ish iterations is plenty for relative
   speed measurement on modern x86. only used as a FALLBACK when
   sysfs frequency data is unavailable. */
static double ax_autotune_run_kernel(void)
{
    const int iters = 1000000;
    volatile float sink = 0.0f;
    float a = 1.0f, b = 1.0001f, c = 0.9999f, d = 0.5f;
    double t0 = ax_autotune_now_ms();
    for (int i = 0; i < iters; i++) {
        a = a * b + c * d;
        a -= 0.0001f;
        if (a > 1e6f || a < -1e6f) a = 1.0f;
    }
    sink = a;
    (void)sink;
    return ax_autotune_now_ms() - t0;
}
#endif

/* picks fast cores on hybrid cpus and pins the omp worker pool to them
   so the os scheduler can't migrate workers back onto slow e-cores.
   compiled out entirely when AX_NO_AUTOTUNE is defined at build time —
   baremetal profiles set this because sched_setaffinity/sysconf aren't
   available and there's no threading runtime to tune anyway. */
int ax_autotune_threads(void)
{
#if defined(AX_NO_AUTOTUNE)
    return 1;
#elif !defined(_OPENMP)
    /* nothing to tune without openmp */
    return 1;
#else
    /* user opt-out */
    const char *no_auto = getenv("AX_NO_AUTOTUNE");
    if (no_auto && no_auto[0] == '1') {
        return omp_get_max_threads();
    }

    /* user-explicit thread count wins */
    const char *omp_env = getenv("OMP_NUM_THREADS");
    if (omp_env && omp_env[0] != '\0') {
        int n = atoi(omp_env);
        if (n > 0) return n;
        return omp_get_max_threads();
    }

#ifndef __linux__
    /* no affinity api on this platform; leave defaults alone */
    return omp_get_max_threads();
#else
    /* enumerate the cpus we're allowed to run on */
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return omp_get_max_threads();
    }

    int max_cpus = CPU_SETSIZE;
    int cpu_ids[CPU_SETSIZE];
    int n_cpus = 0;
    for (int i = 0; i < max_cpus && n_cpus < CPU_SETSIZE; i++) {
        if (CPU_ISSET((size_t)i, &allowed)) {
            cpu_ids[n_cpus++] = i;
        }
    }
    if (n_cpus <= 1) {
        return omp_get_max_threads();
    }

    double t_start = ax_autotune_now_ms();

    /* fast path for small uniform systems: if we have ≤ 4 logical cores,
       there's no meaningful classification to do — use all of them.
       this avoids running the 200ms fma microbench probe on containers
       and VMs where sysfs frequency data is unavailable. the probe leaves
       OMP thread affinity in a suboptimal state that hurts subsequent
       GEMM throughput by 10-25%. on large systems (many cores, possible
       hybrid topology) the probe still runs. */
    if (n_cpus <= 4) {
        omp_set_num_threads(n_cpus);
        ax_gemm_fast_threads = n_cpus;
        ax_gemm_all_threads = n_cpus;
#ifndef AX_NO_STDIO
        fprintf(stderr, "axiom: small uniform system (%d cores), no calibration needed\n", n_cpus);
#endif
        return n_cpus;
    }

    /* primary path: classify cores by sysfs frequency. reads
       base_frequency for each cpu — instant (<1ms), no probing.
       on hybrid cpus (alder lake etc.) this cleanly separates P-cores
       from E-cores by their base clock without any magic threshold
       on measured runtime. on non-hybrid cpus all cores have the same
       freq and all pass — the ht dedup below then picks physical
       cores. */
    int fast_cpus_all[CPU_SETSIZE];
    int fast_all = classify_cores_by_freq(cpu_ids, n_cpus,
                                           fast_cpus_all, CPU_SETSIZE);

    /* fallback: if sysfs frequency data is unavailable (containers,
       old kernels, non-linux-like environments), probe per-core speed
       with a scalar fma microbench. this is the old path — ~1ms per
       core, 200ms budget. */
    if (fast_all < 0) {
        cpu_set_t saved = allowed;
        double times[CPU_SETSIZE];
        double budget_ms = 200.0;

        for (int i = 0; i < n_cpus; i++) {
            if (ax_autotune_now_ms() - t_start > budget_ms - 5.0) {
                sched_setaffinity(0, sizeof(saved), &saved);
                return omp_get_max_threads();
            }
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET((size_t)cpu_ids[i], &one);
            if (sched_setaffinity(0, sizeof(one), &one) != 0) {
                times[i] = 1e9;
                continue;
            }
            sched_yield();
            times[i] = ax_autotune_run_kernel();
        }
        sched_setaffinity(0, sizeof(saved), &saved);

        double best = times[0];
        for (int i = 1; i < n_cpus; i++)
            if (times[i] < best) best = times[i];
        if (best <= 0.0) return omp_get_max_threads();

        double thr = best * 1.25;
        fast_all = 0;
        for (int i = 0; i < n_cpus; i++)
            if (times[i] <= thr)
                fast_cpus_all[fast_all++] = cpu_ids[i];
    }
    if (fast_all <= 0) { fast_cpus_all[fast_all++] = cpu_ids[0]; }

    /* smt/ht deduplication: on hyperthreaded cpus, two logical cores
       share the same physical core and its fma units. running one
       thread per PHYSICAL core avoids fma contention and gives each
       thread full throughput. two ht siblings typically score within
       5% of each other in the calibration (they are equally "fast"),
       so both make it past the threshold — but using both costs ~40%
       gemm throughput on alder lake due to fma port sharing.

       detect siblings via /sys thread_siblings_list: if cpu A lists
       cpu B as a sibling and B is already in the deduped set, skip A.

       exception: when the deduped count would be ≤ 3 physical cores,
       keep ALL fast logical cores. on small machines (2 physical cores,
       4 smt threads) the smt parallelism adds ~20% total throughput
       that outweighs the per-thread fma contention. the breakeven
       is around 4 physical cores — above that, dedup wins. */
    int fast_cpus[CPU_SETSIZE];
    int fc = 0;
    int used_physical[CPU_SETSIZE];
    memset(used_physical, 0, sizeof(used_physical));

    /* first pass: count unique physical cores */
    int n_physical = 0;
    for (int i = 0; i < fast_all; i++) {
        int cpu = fast_cpus_all[i];
        int phys_id = cpu;
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        FILE *f = fopen(path, "r");
        if (f) {
            int lo = cpu;
            if (fscanf(f, "%d", &lo) == 1 && lo < cpu) phys_id = lo;
            fclose(f);
        }
        if (phys_id < CPU_SETSIZE && !used_physical[phys_id]) {
            used_physical[phys_id] = 1;
            n_physical++;
        }
    }

    /* decide: dedup or keep all */
    if (n_physical >= 4) {
        /* enough physical cores — dedup to avoid smt fma contention */
        memset(used_physical, 0, sizeof(used_physical));
        for (int i = 0; i < fast_all; i++) {
            int cpu = fast_cpus_all[i];
            int phys_id = cpu;
            char path[128];
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
            FILE *f = fopen(path, "r");
            if (f) {
                int lo = cpu;
                if (fscanf(f, "%d", &lo) == 1 && lo < cpu) phys_id = lo;
                fclose(f);
            }
            if (phys_id < CPU_SETSIZE && used_physical[phys_id]) continue;
            if (phys_id < CPU_SETSIZE) used_physical[phys_id] = 1;
            fast_cpus[fc++] = cpu;
        }
    } else {
        /* small machine — keep all fast logical cores (smt helps) */
        for (int i = 0; i < fast_all; i++)
            fast_cpus[fc++] = fast_cpus_all[i];
    }
    if (fc == 0) { fast_cpus[fc++] = fast_cpus_all[0]; }

    int fast_count = fc;
    bool is_hybrid = (fast_all < n_cpus);

    double total_ms = ax_autotune_now_ms() - t_start;

    if (is_hybrid) {
        /* hybrid cpu (e.g., alder lake): P-cores are "fast", E-cores are slower.
           for GEMM, E-cores contribute meaningful throughput through JC-parallel
           column strips. strategy: set default thread count to all cores (no pinning),
           and export both counts so the GEMM path can pick per-call.
           no sched_setaffinity — let the OS scheduler handle hybrid placement. */
        omp_set_num_threads(n_cpus);
        ax_gemm_fast_threads = fast_count;
        ax_gemm_all_threads = n_cpus;
#ifndef AX_NO_STDIO
        fprintf(stderr, "axiom: hybrid cpu detected: %d fast + %d slow = %d total (%.1fms)\n",
                fast_count, n_cpus - fast_count, n_cpus, total_ms);
        fprintf(stderr, "axiom: GEMM will use %d threads (large) or %d threads (small)\n",
                n_cpus, fast_count);
#endif
        return n_cpus;
    }

    /* non-hybrid: set thread count to fast cores. only pin when we're
       NOT using every available cpu — pinning when fast_count == n_cpus
       provides no benefit (no slow cores to avoid) and can hurt in
       virtualized / containerized environments where vCPU→pCPU mapping
       is managed by the hypervisor. pinning to specific vCPUs can force
       suboptimal physical core placement. */
    omp_set_num_threads(fast_count);
    ax_gemm_fast_threads = fast_count;
    ax_gemm_all_threads = fast_count;

    int pin_failures = 0;
    bool should_pin = (fast_count < n_cpus);
    if (should_pin) {
        #pragma omp parallel num_threads(fast_count) reduction(+:pin_failures)
        {
            int w = omp_get_thread_num();
            int cpu = fast_cpus[w % fc];
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET((size_t)cpu, &one);
            if (sched_setaffinity(0, sizeof(one), &one) != 0) {
                pin_failures += 1;
            }
        }
    }

    char cpu_list[128];
    size_t off = 0;
    for (int i = 0; i < fc && off + 8 < sizeof(cpu_list); i++) {
        int w_ = snprintf(cpu_list + off, sizeof(cpu_list) - off,
                          (i == 0) ? "%d" : ",%d", fast_cpus[i]);
        if (w_ < 0) break;
        off += (size_t)w_;
    }

#ifndef AX_NO_STDIO
    fprintf(stderr, "axiom: autotune chose %d fast cores (%.1fms calibration)\n",
            fast_count, total_ms);
    if (should_pin) {
        fprintf(stderr, "axiom: pinned %d omp workers to cores [%s]%s\n",
                fast_count, cpu_list,
                pin_failures ? " (some pins failed)" : "");
    } else {
        fprintf(stderr, "axiom: using all %d cores, no pinning needed\n", fast_count);
    }
#else
    (void)total_ms; (void)cpu_list; (void)pin_failures;
#endif

    return fast_count;
#endif /* __linux__ */
#endif /* _OPENMP */
}
