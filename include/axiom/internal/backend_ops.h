/* axiom/internal/backend_ops.h — backend vtable, internal contract.

   defines the full ax_backend_ops_t struct that every compute backend
   fills in. NOT part of the public abi — library users must not include
   this. the public compute.h forward-declares the type as opaque. backend
   authors writing new modules in src/compute/backends/ include this to
   construct their vtable instance. */

#ifndef AX_INTERNAL_BACKEND_OPS_H
#define AX_INTERNAL_BACKEND_OPS_H

#include "axiom/types.h"
#include "axiom/device.h"
#include "axiom/compute.h"  /* for ax_conv_params_t and forward decls */

#ifdef __cplusplus
extern "C" {
#endif

/* ax_tensor_t and ax_conv_params_t already forward-declared by compute.h
   above. ax_backend_ops_t the typedef name comes from compute.h too;
   the struct body here completes the type. */

/* backend operation vtable.
   each backend (cpu naive, cpu simd, cuda, etc.) provides one of these
   with function pointers filled in for every supported operation.
   the public compute.h forward-declares this struct as opaque
   (typedef struct ax_backend_ops ax_backend_ops_t;) so library users
   never see the layout. */
struct ax_backend_ops {
    const char *name;   /* human-readable backend name, e.g. "cpu_naive" */

    /* element-wise binary ops */
    /* all binary ops support broadcasting */
    ax_status_t (*add)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
    ax_status_t (*sub)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
    ax_status_t (*mul)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
    ax_status_t (*div_op)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);

    /* element-wise unary ops */
    ax_status_t (*neg)(const ax_tensor_t *in, ax_tensor_t *out);
    ax_status_t (*abs_op)(const ax_tensor_t *in, ax_tensor_t *out);
    ax_status_t (*exp_op)(const ax_tensor_t *in, ax_tensor_t *out);
    ax_status_t (*log_op)(const ax_tensor_t *in, ax_tensor_t *out);
    ax_status_t (*sqrt_op)(const ax_tensor_t *in, ax_tensor_t *out);
    ax_status_t (*square)(const ax_tensor_t *in, ax_tensor_t *out);

    /* scalar ops */
    ax_status_t (*add_scalar)(const ax_tensor_t *in, double scalar, ax_tensor_t *out);
    ax_status_t (*mul_scalar)(const ax_tensor_t *in, double scalar, ax_tensor_t *out);

    /* matrix ops */
    /* general matrix multiply: out = a @ b */
    ax_status_t (*gemm)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);

    /* optional: out = a @ b^T. b is [N, K] stored normally; backend walks
       it as if transposed without materializing a physical copy. saves a
       full pass in the common backward pattern dX = dY @ W^T.
       NULL when unimplemented — dispatch falls back to a physical transpose. */
    ax_status_t (*gemm_nt)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);

    /* optional: out = a^T @ b. a is [K, M] stored normally; backend walks
       it as if transposed. saves a physical transpose in dW = X^T @ dY.
       NULL when unimplemented. */
    ax_status_t (*gemm_tn)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);

    /* optional: fused matmul + relu. out = relu(a @ b + bias).
       bias may be NULL for matmul+relu without bias.
       applies max(0, x) during the gemm writeback step, saving a full
       read+write pass over the output tensor vs separate gemm then relu.
       NULL when unimplemented — callers fall back to gemm + compute_relu. */
    ax_status_t (*gemm_relu)(const ax_tensor_t *a, const ax_tensor_t *b,
                              const ax_tensor_t *bias, ax_tensor_t *out);

    /* optional: fused-scaling gemm. out = alpha * (a @ b) + beta * out.
       matches the standard blas sgemm signature so callers can fuse
       an accumulate (beta=1), a scale (alpha!=1), or a bias-add
       pre-initialised out (out=bias then beta=1 alpha=1) into a
       single dispatch. NULL when unimplemented — callers should fall
       back to a plain gemm + scalar post-pass. */
    ax_status_t (*gemm_ex)(const ax_tensor_t *a, const ax_tensor_t *b,
                            float alpha, float beta, ax_tensor_t *out);

    /* optional fused primitives for bandwidth-bound workloads.
       same shape rules as their elementwise analogues (same ndim,
       matching shape; no broadcast). NULL when unimplemented. */

    /* fused relu(a + b). single pass, half the memory traffic of
       the separate add then relu sequence. */
    ax_status_t (*add_relu)(const ax_tensor_t *a, const ax_tensor_t *b,
                             ax_tensor_t *out);

    /* y += alpha * x, in-place on y. the classic blas axpy. used in
       optimiser updates and gradient accumulation. y and x must have
       the same numel and be contiguous. */
    ax_status_t (*axpy)(const ax_tensor_t *x, float alpha, ax_tensor_t *y);

    /* row-wise softmax on a 2d input [rows, cols]. numerically stable
       via the max-subtract trick: for each row, out = exp(in - max)
       then divide by row sum. out has the same shape as in. */
    ax_status_t (*softmax_rowwise)(const ax_tensor_t *in, ax_tensor_t *out);

    /* optional: fused bias add. out = in + broadcast(bias) along `axis`.
       bias is rank-1 with numel matching in->shape[axis]. saves a separate
       pass over out after linear/conv. NULL when unimplemented. */
    ax_status_t (*bias_add)(const ax_tensor_t *in, const ax_tensor_t *bias,
                             int axis, ax_tensor_t *out);

    /* implicit im2col conv forward gemm, per-sample.
       weight is [C_out, C_in*kh*kw] (row-major), out is [C_out, out_h*out_w].
       the backend gathers input patches directly into packed b buffers during
       the gemm, skipping the materialized im2col matrix. optional — backends
       that don't implement this leave the pointer NULL and conv.c falls back
       to the standard im2col + gemm path. */
    ax_status_t (*conv_gemm)(const ax_tensor_t *weight,
                              const ax_conv_params_t *params,
                              ax_tensor_t *out);

    /* reduction ops */
    /* axis=-1 means reduce all dimensions */
    ax_status_t (*sum)(const ax_tensor_t *in, int axis, ax_tensor_t *out);
    ax_status_t (*mean)(const ax_tensor_t *in, int axis, ax_tensor_t *out);
    ax_status_t (*max_op)(const ax_tensor_t *in, int axis, ax_tensor_t *out);
    ax_status_t (*min_op)(const ax_tensor_t *in, int axis, ax_tensor_t *out);

    /* optional: argmax along an axis. output is rank in->ndim-1 (reduced
       dim removed) with dtype int64. axis=-1 reduces all dims to a single
       scalar index. NULL when unimplemented. */
    ax_status_t (*argmax)(const ax_tensor_t *in, int axis, ax_tensor_t *out);

    /* comparison ops */
    ax_status_t (*equal)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
    ax_status_t (*greater)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);

    /* data movement */
    /* fill all elements with a scalar value */
    ax_status_t (*fill)(ax_tensor_t *t, double value);
    /* copy data from src to dst (same shape required) */
    ax_status_t (*copy)(const ax_tensor_t *src, ax_tensor_t *dst);

    /* activation functions (fused for performance) */
    ax_status_t (*relu)(const ax_tensor_t *in, ax_tensor_t *out);
    ax_status_t (*sigmoid)(const ax_tensor_t *in, ax_tensor_t *out);
    ax_status_t (*tanh_op)(const ax_tensor_t *in, ax_tensor_t *out);

    /* extended activations. leaky_relu and elu take a float alpha
       parameter. gelu uses the tanh approximation. swish is x*sigmoid(x).
       all optional — NULL when unimplemented. */
    ax_status_t (*leaky_relu)(const ax_tensor_t *in, float alpha, ax_tensor_t *out);
    ax_status_t (*elu_op)(const ax_tensor_t *in, float alpha, ax_tensor_t *out);
    ax_status_t (*gelu_op)(const ax_tensor_t *in, ax_tensor_t *out);
    ax_status_t (*swish_op)(const ax_tensor_t *in, ax_tensor_t *out);

    /* optional: fused optimizer update kernels. the optimizer iterates
       per-parameter and calls these with each param's weight, grad, and
       state tensors. avoids reading all 4 tensors back to the host for
       a scalar update loop. NULL when unimplemented — optim.c falls
       back to the existing cpu simd path. */
    ax_status_t (*adam_update)(ax_tensor_t *weight, ax_tensor_t *grad,
                                ax_tensor_t *m, ax_tensor_t *v,
                                float lr, float beta1, float beta2, float eps,
                                float weight_decay, float bc1, float bc2,
                                bool decoupled);
    ax_status_t (*sgd_update)(ax_tensor_t *weight, ax_tensor_t *grad,
                               ax_tensor_t *momentum_buf,
                               float lr, float momentum, float weight_decay,
                               bool nesterov);

    /* ---- device-owner hooks (phase 0 backend module extension) ----
       a backend that owns a non-cpu device (e.g. cuda) fills these in so
       core code can manage memory without knowing the device specifics.
       cpu backends leave them NULL — the cpu path in tensor.c uses the
       struct/data pool directly. */

    /* which device this backend's memory hooks operate on. a backend that
       only provides compute (no memory management) sets this to
       AX_DEVICE_COUNT. backends that own a device (cuda) set it to that
       device; they will be registered in the device->backend table at
       init time. */
    ax_device_t device;

    /* one-shot lifecycle. init is called on first use of the backend;
       shutdown is called from ax_compute_shutdown. both may be NULL. */
    ax_status_t (*init)(void);
    void        (*shutdown)(void);

    /* device-wide barrier. blocks until all pending work on the device
       completes. called from core via ax_device_synchronize. */
    void        (*synchronize)(void);

    /* number of physical devices this backend can address (>=0). */
    int         (*device_count)(void);

    /* raw memory allocation on the owned device. return NULL on failure.
       the pointer is opaque to core and is only passed back to the
       backend's storage_free / memcpy_* hooks. */
    void       *(*storage_alloc)(size_t size_bytes);
    void        (*storage_free)(void *ptr);

    /* host <-> device memcpy primitives. offsets are in bytes. */
    ax_status_t (*memcpy_h2d)(void *dst, const void *src, size_t bytes);
    ax_status_t (*memcpy_d2h)(void *dst, const void *src, size_t bytes);
    ax_status_t (*memcpy_d2d)(void *dst, const void *src, size_t bytes);

};
/* the typedef itself is declared opaquely in compute.h (public). */

#ifdef __cplusplus
}
#endif

#endif /* AX_INTERNAL_BACKEND_OPS_H */
