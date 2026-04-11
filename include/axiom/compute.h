/* axiom/compute.h — compute dispatch api.
   all math goes through these functions; they route to the active backend. */

#ifndef AX_COMPUTE_H
#define AX_COMPUTE_H

#include "types.h"
#include "backend_ops.h"

/* backend identifiers */
typedef enum {
    AX_BACKEND_CPU_NAIVE = 0,  /* pure c fallback — always available */
    AX_BACKEND_CPU_SIMD,       /* avx2/neon optimized — future */
    AX_BACKEND_CPU_BLAS,       /* openblas/mkl — future */
    AX_BACKEND_CUDA,           /* gpu via cuda — future */
    AX_BACKEND_COUNT
} ax_backend_id_t;

/* initialize the compute system — probes available backends, selects best.
   call once at startup before any tensor operations. */
ax_status_t ax_compute_init(void);

/* shut down the compute system */
void ax_compute_shutdown(void);

/* manually select a backend; returns error if backend not available */
ax_status_t ax_compute_set_backend(ax_backend_id_t id);

/* get the currently active backend id */
ax_backend_id_t ax_compute_get_backend(void);

/* get the ops vtable for the active backend (used internally by tensor ops) */
const ax_backend_ops_t *ax_compute_get_ops(void);

/* register a custom backend at runtime (for plugins/extensions) */
ax_status_t ax_compute_register_backend(ax_backend_id_t id, const ax_backend_ops_t *ops);

/* set the maximum number of threads used by parallel ops.
   wraps omp_set_num_threads() when openmp is enabled.
   no-op when openmp is disabled (always 1 thread).
   pass 0 to reset to default (omp_get_max_threads). */
void ax_set_num_threads(int n);

/* get the current maximum number of threads.
   returns 1 if openmp is disabled. */
int ax_get_num_threads(void);

/* runtime auto-tune for hybrid cpus (intel 12th+ p-core/e-core, big.LITTLE).
   benchmarks each logical cpu with a tiny serial workload pinned via
   sched_setaffinity, clusters cores within 25% of the fastest as "fast",
   and sets the default thread count to the fast-core count. this avoids
   omp barrier stalls where slow e-cores block fast p-cores.

   skipped if env AX_NO_AUTOTUNE=1 or OMP_NUM_THREADS is set (user-explicit).
   only runs on linux; other platforms return omp_get_max_threads() unchanged.
   total calibration time is bounded under 200ms.
   returns the chosen thread count. */
int ax_autotune_threads(void);

/* dispatch functions */
/* these call through to the active backend's function pointers.
   tensor.c calls these; user code normally calls the tensor-level api instead. */

ax_status_t ax_compute_add(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
ax_status_t ax_compute_sub(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
ax_status_t ax_compute_mul(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
ax_status_t ax_compute_div(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);

ax_status_t ax_compute_neg(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t ax_compute_abs(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t ax_compute_exp(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t ax_compute_log(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t ax_compute_sqrt(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t ax_compute_square(const ax_tensor_t *in, ax_tensor_t *out);

ax_status_t ax_compute_add_scalar(const ax_tensor_t *in, double scalar, ax_tensor_t *out);
ax_status_t ax_compute_mul_scalar(const ax_tensor_t *in, double scalar, ax_tensor_t *out);

ax_status_t ax_compute_gemm(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);

ax_status_t ax_compute_sum(const ax_tensor_t *in, int axis, ax_tensor_t *out);
ax_status_t ax_compute_mean(const ax_tensor_t *in, int axis, ax_tensor_t *out);
ax_status_t ax_compute_max(const ax_tensor_t *in, int axis, ax_tensor_t *out);
ax_status_t ax_compute_min(const ax_tensor_t *in, int axis, ax_tensor_t *out);

ax_status_t ax_compute_equal(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
ax_status_t ax_compute_greater(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);

ax_status_t ax_compute_fill(ax_tensor_t *t, double value);
ax_status_t ax_compute_copy(const ax_tensor_t *src, ax_tensor_t *dst);

ax_status_t ax_compute_relu(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t ax_compute_sigmoid(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t ax_compute_tanh(const ax_tensor_t *in, ax_tensor_t *out);

#endif /* AX_COMPUTE_H */
