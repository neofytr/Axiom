/* axiom/compute.h — public compute dispatch api.

   high-level compute primitives that route to the active backend. all
   tensor-level math goes through here. user code rarely calls these
   directly — the tensor and ops apis (axiom/ops.h, axiom/tensor.h) are
   the recommended entry points.

   internal-only declarations (vtable struct, register hook, parallelism
   thresholds, calibration entry points) live in
   `axiom/internal/compute_internal.h` and are not part of the abi. */

#ifndef AX_COMPUTE_H
#define AX_COMPUTE_H

#include "types.h"
#include "device.h"

#ifdef __cplusplus
extern "C" {
#endif

/* opaque forward declaration. the full struct definition lives in the
   internal header `axiom/internal/backend_ops.h` so that user code can
   pass and store pointers without depending on the layout. */
typedef struct ax_backend_ops ax_backend_ops_t;

/* tensor forward declaration — the canonical definition lives in
   axiom/tensor.h. forward-declared here so this header doesn't need to
   pull tensor.h in just to name the type. */
typedef struct ax_tensor ax_tensor_t;

/* per-sample conv parameters for the implicit-im2col gemm. describes
   one [C_in, H, W] sample (pointer into the batched input buffer) plus
   the kernel/stride/pad geometry. the input pointer is borrowed — no
   ownership transfer. used by ax_compute_conv_gemm() and the conv layer. */
typedef struct {
    const float *input;  /* [C_in, H, W] contiguous, row-major */
    int64_t C_in, H, W;
    int kh, kw;
    int sh, sw;
    int ph, pw;
    int64_t out_h, out_w;
} ax_conv_params_t;

/* backend identifiers */
typedef enum {
    AX_BACKEND_CPU_NAIVE = 0,  /* pure c fallback — always available */
    AX_BACKEND_CPU_SIMD,       /* avx2/neon optimized */
    AX_BACKEND_CPU_BLAS,       /* openblas/mkl — future */
    AX_BACKEND_CUDA,           /* gpu via cuda */
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

/* return the human-readable name of the active backend (e.g. "cpu_opt",
   "cpu_naive", "cuda"). pointer is owned by the framework and lives
   until ax_compute_shutdown(). NULL if no backend is active. */
const char *ax_compute_backend_name(void);

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

/* out = a @ b^T (b stored normally, walked transposed). returns
   AX_ERR_NOT_IMPLEMENTED if the active backend lacks gemm_nt; callers
   should fall back to physical transpose + plain gemm. */
ax_status_t ax_compute_gemm_nt(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);

/* out = a^T @ b. same fallback contract as gemm_nt. */
ax_status_t ax_compute_gemm_tn(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);

/* fused dwqkv weight-grad: dWq += x_flat^T @ dQKV[:, 0:D],
                            dWk += x_flat^T @ dQKV[:, D:2D],
                            dWv += x_flat^T @ dQKV[:, 2D:3D].
   single-pass equivalent of `gemm_tn(x_flat, dQKV) → [D, 3D]` followed by
   three SIMD column-slice accumulators into dWq/dWk/dWv. saves the
   intermediate's read+write traffic at MHA backward. all five tensors
   contiguous fp32; x_flat=[K, D], dQKV=[K, 3D], dW*=[D, D]. returns
   AX_ERR_NOT_IMPLEMENTED if the backend lacks the slot. */
ax_status_t ax_compute_dwqkv_split_acc(const ax_tensor_t *x_flat,
                                        const ax_tensor_t *dQKV,
                                        ax_tensor_t *dWq,
                                        ax_tensor_t *dWk,
                                        ax_tensor_t *dWv);
int ax_compute_has_dwqkv_split_acc(void);

/* F.3.a fused forward QKV projection + head_interleave_split.
   computes:
     qkv = X @ Wqkv (+ bqkv)
     Qh, Kh, Vh = head_interleave_split(qkv)
   in one pass, eliminating the [B*S, 3D] qkv intermediate (~18 MB on
   B1_S2048_D768) and the head_interleave_split layout pass.

   shapes: X=[B*S, D], Wqkv=[D, 3D], bqkv=[3D] or NULL,
           Qh, Kh, Vh each [B*H, S, dk]   where D = H*dk.
   Qh/Kh/Vh are OVERWRITTEN (initialised to bias if bqkv != NULL else
   zero, then gemm accumulates). callers must not pre-populate them.
   returns AX_ERR_NOT_IMPLEMENTED if the backend lacks the slot —
   callers fall back to gemm + ax_attn_head_interleave_qkv_split. */
ax_status_t ax_compute_qkv_head_gemm(const ax_tensor_t *X,
                                      const ax_tensor_t *Wqkv,
                                      const ax_tensor_t *bqkv,
                                      int64_t B, int64_t S, int64_t H, int64_t dk,
                                      ax_tensor_t *Qh,
                                      ax_tensor_t *Kh,
                                      ax_tensor_t *Vh);
int ax_compute_has_qkv_head_gemm(void);

/* F.3.d fused d_attn backward gemm_nt + head_interleave.
   computes:
     dattn = dout @ Wo^T
     dO_head = head_interleave(dattn)
   in one pass, eliminating the [B*S, D] dattn intermediate (~6 MB on
   B1_S2048_D768).
   shapes: dout=[B*S, D], Wo=[D, D], dO_head=[B*H, S, dk]   where D = H*dk.
   dO_head is OVERWRITTEN.
   returns AX_ERR_NOT_IMPLEMENTED if the backend lacks the slot —
   callers fall back to gemm_nt + ax_attn_head_interleave. */
ax_status_t ax_compute_dattn_head_gemm_nt(const ax_tensor_t *dout,
                                            const ax_tensor_t *Wo,
                                            int64_t B, int64_t S, int64_t H, int64_t dk,
                                            ax_tensor_t *dO_head);
int ax_compute_has_dattn_head_gemm_nt(void);

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

#ifdef __cplusplus
}
#endif

#endif /* AX_COMPUTE_H */
