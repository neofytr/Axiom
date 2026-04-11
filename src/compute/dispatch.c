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

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* declared in cpu_naive.c and cpu_opt.c */
extern const ax_backend_ops_t ax_cpu_naive_ops;
extern const ax_backend_ops_t ax_cpu_opt_ops;

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
    /* register all built-in backends */
    backends[AX_BACKEND_CPU_NAIVE] = &ax_cpu_naive_ops;
    backends[AX_BACKEND_CPU_SIMD]  = &ax_cpu_opt_ops;
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
    active_ops = &ax_cpu_opt_ops;
    compute_initialized = 1;

    /* runtime hybrid-cpu autotune. only emits a stderr line when it
       actually calibrates (i.e. user did not pin threads explicitly). */
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
    return active_ops->conv_gemm(weight, params, out);
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
/* tiny serial workload. volatile sink prevents the optimizer from
   collapsing the loop. ~1m fma-ish iterations is plenty for relative
   speed measurement on modern x86. */
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

int ax_autotune_threads(void)
{
#ifndef _OPENMP
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
        if (CPU_ISSET(i, &allowed)) {
            cpu_ids[n_cpus++] = i;
        }
    }
    if (n_cpus <= 1) {
        return omp_get_max_threads();
    }

    /* save current affinity so we can restore it after probing */
    cpu_set_t saved = allowed;

    double times[CPU_SETSIZE];
    double t_start = ax_autotune_now_ms();
    double budget_ms = 200.0;

    for (int i = 0; i < n_cpus; i++) {
        /* abort calibration cleanly if we're about to blow the budget */
        if (ax_autotune_now_ms() - t_start > budget_ms - 5.0) {
            /* not enough samples to cluster; bail out */
            sched_setaffinity(0, sizeof(saved), &saved);
            return omp_get_max_threads();
        }

        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(cpu_ids[i], &one);
        if (sched_setaffinity(0, sizeof(one), &one) != 0) {
            /* couldn't pin; treat as worst-case */
            times[i] = 1e9;
            continue;
        }
        /* short yield so the kernel actually migrates us */
        sched_yield();
        times[i] = ax_autotune_run_kernel();
    }

    /* restore affinity */
    sched_setaffinity(0, sizeof(saved), &saved);

    /* find the fastest (smallest time) */
    double best = times[0];
    for (int i = 1; i < n_cpus; i++) {
        if (times[i] < best) best = times[i];
    }
    if (best <= 0.0) {
        return omp_get_max_threads();
    }

    /* fast cores: within 25% of best (i.e. time <= 1.25 * best) */
    double threshold = best * 1.25;
    int fast_count = 0;
    for (int i = 0; i < n_cpus; i++) {
        if (times[i] <= threshold) fast_count++;
    }
    if (fast_count <= 0) fast_count = 1;
    if (fast_count > n_cpus) fast_count = n_cpus;

    double total_ms = ax_autotune_now_ms() - t_start;

    omp_set_num_threads(fast_count);

    fprintf(stderr, "axiom: autotune chose %d fast cores (%.1fms calibration)\n",
            fast_count, total_ms);

    return fast_count;
#endif /* __linux__ */
#endif /* _OPENMP */
}
