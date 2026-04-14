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

/* look up the backend that owns memory/lifecycle for a given device.
   returns NULL for AX_DEVICE_CPU or for devices whose backend module is
   not compiled in. core uses this to route storage allocation and
   host<->device transfers without knowing specific device types. */
const ax_backend_ops_t *ax_backend_for_device(ax_device_t device);

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

/* calibrated parallelism thresholds. set by ax_calibrate_thresholds()
   based on measured omp fork/join overhead at init time. kernels with
   per-iteration work below the threshold skip omp and run serial
   (fork-join barrier would dominate). */
extern int64_t ax_par_threshold_elems;   /* reduction / elementwise */
extern int64_t ax_par_threshold_batch;   /* per-sample loops (norm, loss) */
extern int64_t ax_par_threshold_flops;   /* generic flop-count threshold */
extern double  ax_omp_overhead_ms;       /* measured fork/join cost */

/* measure omp fork/join overhead and derive the thresholds above.
   cheap (<50ms). called lazily from ax_compute_init(). safe to call
   multiple times (idempotent). no-op under AX_NO_AUTOTUNE=1 or
   when compiled without openmp. */
void ax_calibrate_thresholds(void);

/* measure gemm tile (mc/nc/kc) performance across a small grid of
   candidate configurations and pick the best. sweeps ~5 configs on a
   representative shape; budget ~500ms. opt-in via AX_GEMM_CALIBRATE=1
   env var because it adds startup cost. */
void ax_calibrate_gemm_tiles(void);

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

/* out = a @ b^T (b stored normally, walked transposed). returns
   AX_ERR_NOT_IMPLEMENTED if the active backend lacks gemm_nt; callers
   should fall back to physical transpose + plain gemm. */
ax_status_t ax_compute_gemm_nt(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);

/* out = a^T @ b. same fallback contract as gemm_nt. */
ax_status_t ax_compute_gemm_tn(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);

/* fused matmul+relu: out = relu(a @ b + bias). bias may be NULL.
   returns AX_ERR_NOT_IMPLEMENTED if the backend lacks the slot. */
ax_status_t ax_compute_gemm_relu(const ax_tensor_t *a, const ax_tensor_t *b,
                                  const ax_tensor_t *bias, ax_tensor_t *out);
int ax_compute_has_gemm_relu(void);

int ax_compute_has_gemm_nt(void);
int ax_compute_has_gemm_tn(void);

/* fused-scaling gemm: out = alpha * (a @ b) + beta * out.
   classic blas sgemm signature. returns AX_ERR_NOT_IMPLEMENTED if
   the active backend lacks the slot; callers should fall back to
   plain gemm plus a scalar pass. */
ax_status_t ax_compute_gemm_ex(const ax_tensor_t *a, const ax_tensor_t *b,
                                float alpha, float beta, ax_tensor_t *out);
int ax_compute_has_gemm_ex(void);

/* fused relu(a + b). returns AX_ERR_NOT_IMPLEMENTED when absent. */
ax_status_t ax_compute_add_relu(const ax_tensor_t *a, const ax_tensor_t *b,
                                 ax_tensor_t *out);
int ax_compute_has_add_relu(void);

/* y += alpha * x, in-place. same shape required, no broadcast. */
ax_status_t ax_compute_axpy(const ax_tensor_t *x, float alpha, ax_tensor_t *y);
int ax_compute_has_axpy(void);

/* row-wise stable softmax on a 2d [rows, cols] input. */
ax_status_t ax_compute_softmax_rowwise(const ax_tensor_t *in, ax_tensor_t *out);
int ax_compute_has_softmax_rowwise(void);

/* fused bias add. returns AX_ERR_NOT_IMPLEMENTED when the backend
   doesn't provide a fused path; callers can either fall back to a
   broadcast add or accept the error. */
ax_status_t ax_compute_bias_add(const ax_tensor_t *in, const ax_tensor_t *bias,
                                 int axis, ax_tensor_t *out);
int ax_compute_has_bias_add(void);

/* implicit im2col conv forward gemm, per-sample.
   returns AX_ERR_NOT_IMPLEMENTED if the active backend lacks conv_gemm;
   callers should fall back to the standard im2col + gemm path in that case. */
ax_status_t ax_compute_conv_gemm(const ax_tensor_t *weight,
                                  const ax_conv_params_t *params,
                                  ax_tensor_t *out);

/* returns 1 if the active backend implements conv_gemm, 0 otherwise.
   lets conv.c choose between implicit-gemm and im2col+gemm paths once
   per forward call instead of paying the dispatch cost per sample. */
int ax_compute_has_conv_gemm(void);

ax_status_t ax_compute_sum(const ax_tensor_t *in, int axis, ax_tensor_t *out);
ax_status_t ax_compute_mean(const ax_tensor_t *in, int axis, ax_tensor_t *out);
ax_status_t ax_compute_max(const ax_tensor_t *in, int axis, ax_tensor_t *out);
ax_status_t ax_compute_min(const ax_tensor_t *in, int axis, ax_tensor_t *out);

/* argmax along an axis. out must be int64 rank (in->ndim - 1) with
   reduced-dim removed. axis=-1 reduces all dims to a single scalar. */
ax_status_t ax_compute_argmax(const ax_tensor_t *in, int axis, ax_tensor_t *out);

ax_status_t ax_compute_equal(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
ax_status_t ax_compute_greater(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);

ax_status_t ax_compute_fill(ax_tensor_t *t, double value);
ax_status_t ax_compute_copy(const ax_tensor_t *src, ax_tensor_t *dst);

ax_status_t ax_compute_relu(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t ax_compute_sigmoid(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t ax_compute_tanh(const ax_tensor_t *in, ax_tensor_t *out);

/* fused optimizer updates. return AX_ERR_NOT_IMPLEMENTED if the active
   backend lacks the slot — optim.c falls back to the cpu simd path. */
ax_status_t ax_compute_adam_update(ax_tensor_t *weight, ax_tensor_t *grad,
                                    ax_tensor_t *m, ax_tensor_t *v,
                                    float lr, float beta1, float beta2, float eps,
                                    float weight_decay, float bc1, float bc2,
                                    bool decoupled);
ax_status_t ax_compute_sgd_update(ax_tensor_t *weight, ax_tensor_t *grad,
                                    ax_tensor_t *momentum_buf,
                                    float lr, float momentum, float weight_decay,
                                    bool nesterov);
int ax_compute_has_adam_update(void);
int ax_compute_has_sgd_update(void);

#endif /* AX_COMPUTE_H */
