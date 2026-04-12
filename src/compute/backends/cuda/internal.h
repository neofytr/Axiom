/* internal.h — shared prologue for all .cu files in the cuda backend.
   must be the first include in every cuda backend translation unit.

   contents:
     1. std::atomic compat shims so axiom/tensor.h (which uses c11
        <stdatomic.h> via atomic_int) can be included from c++ code
        compiled by nvcc.
     2. axiom public headers.
     3. cuda runtime + cublas + c stdlib includes.
     4. module-private constants (block size etc).
     5. forward declarations of every op function that the vtable in
        backend.cu references across translation units. all cross-tu
        functions use extern "C" linkage to avoid c++ name mangling. */

#ifndef AX_CUDA_INTERNAL_H
#define AX_CUDA_INTERNAL_H

/* ---- 1. stdatomic compat ---- */
#include <atomic>
typedef std::atomic<int> atomic_int;
static inline void atomic_init(atomic_int *obj, int val) {
    obj->store(val, std::memory_order_relaxed);
}
static inline int atomic_fetch_add(atomic_int *obj, int val) {
    return obj->fetch_add(val, std::memory_order_relaxed);
}
static inline int atomic_fetch_sub(atomic_int *obj, int val) {
    return obj->fetch_sub(val, std::memory_order_relaxed);
}
/* prevent c11 <stdatomic.h> from redefining these if something
   pulls it in transitively. */
#define __CLANG_STDATOMIC_H
#define _STDATOMIC_H

/* ---- 2. axiom headers ---- */
#include "axiom/backend_ops.h"
#include "axiom/tensor.h"
#include "axiom/error.h"

/* ---- 3. cuda + c stdlib ---- */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <string.h>

/* ---- 4. module-private constants ---- */
#define AX_CUDA_BLOCK 256

/* ---- 5. cross-tu function declarations ---- */

/* cublas handle: created lazily in backend.cu, shared across all tus
   that need gemm. */
cublasHandle_t ax_cuda_cublas_handle(void);

/* device-side scratch arena.

   ops that need to upload a small metadata struct to the device (e.g.
   BinopMeta for broadcasting) previously did a cudaMalloc + cudaMemcpy
   + cudaFree on every single call. on real workloads this adds up to
   thousands of mallocs per training step and dominates the op cost.

   the scratch arena is one persistent ~4 KB device buffer, created by
   backend.cu's init hook, that kernel-metadata uploads bump into. it
   is reset at the start of every op that uses it (ops hold no state
   across calls) and sized statically because real metadata structs
   are tens of bytes each.

   not thread-safe w.r.t. concurrent host-side calls into the backend:
   caller is expected to serialise. the cuda driver already serialises
   kernel launches on the default stream, and every backend op in
   axiom is a single kernel launch today, so this is fine for now.
   when we add streams in phase 5 we will revisit. */

/* 64 KB scratch: binop metadata is ~200 B, reductions need
   `num_blocks * sizeof(float)` which is (n / 256) * 4 B — at 64 KB
   the arena covers inputs up to ~4 million elements before falling
   back to cudaMalloc, which is more than enough for everything in
   the current test suite and typical training batch sizes. */
#define AX_CUDA_SCRATCH_BYTES (64 * 1024)

/* return a pointer into the scratch arena with at least `bytes` free
   and 16-byte aligned. the returned pointer is invalidated on the
   next reset. returns NULL if `bytes` exceeds the arena capacity. */
void *ax_cuda_scratch_alloc(size_t bytes);

/* rewind the arena so a subsequent ax_cuda_scratch_alloc returns the
   first slot again. typically called at the start of an op. */
void  ax_cuda_scratch_reset(void);

/* kernel-launch error check.

   cuda kernel launches are asynchronous and return the driver error
   for pre-launch problems (e.g. invalid launch config). the actual
   kernel-body error (e.g. illegal address, out-of-bounds write) only
   surfaces on the next synchronising call, which in our case can be
   much later — by then the error is attributed to whatever op happens
   to synchronise first, making debugging miserable.

   this macro grabs the post-launch error immediately via
   cudaGetLastError. if set, it populates ax_err_set with the cuda
   error string and returns AX_ERR_BACKEND from the enclosing
   function. call it directly after every kernel launch. */

#define AX_CUDA_CHECK_LAUNCH(tag)                                           \
    do {                                                                    \
        cudaError_t _ax_cerr = cudaGetLastError();                          \
        if (_ax_cerr != cudaSuccess) {                                      \
            ax_err_set(AX_ERR_BACKEND, "cuda %s launch failed: %s",         \
                       (tag), cudaGetErrorString(_ax_cerr));                \
            return AX_ERR_BACKEND;                                          \
        }                                                                   \
    } while (0)

extern "C" {

/* elementwise ops (ops_elementwise.cu) */
ax_status_t cuda_add(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
ax_status_t cuda_sub(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
ax_status_t cuda_mul(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
ax_status_t cuda_div(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
ax_status_t cuda_equal(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
ax_status_t cuda_greater(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);

ax_status_t cuda_neg(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t cuda_abs(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t cuda_exp(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t cuda_log(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t cuda_sqrt(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t cuda_square(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t cuda_relu(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t cuda_sigmoid(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t cuda_tanh_op(const ax_tensor_t *in, ax_tensor_t *out);

ax_status_t cuda_add_scalar(const ax_tensor_t *in, double scalar, ax_tensor_t *out);
ax_status_t cuda_mul_scalar(const ax_tensor_t *in, double scalar, ax_tensor_t *out);

ax_status_t cuda_fill(ax_tensor_t *t, double value);
ax_status_t cuda_copy(const ax_tensor_t *src, ax_tensor_t *dst);

/* reductions (ops_reduce.cu) */
ax_status_t cuda_sum(const ax_tensor_t *in, int axis, ax_tensor_t *out);
ax_status_t cuda_mean(const ax_tensor_t *in, int axis, ax_tensor_t *out);
ax_status_t cuda_max(const ax_tensor_t *in, int axis, ax_tensor_t *out);
ax_status_t cuda_min(const ax_tensor_t *in, int axis, ax_tensor_t *out);

/* matrix multiply (ops_gemm.cu) */
ax_status_t cuda_gemm(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);

/* implicit im2col conv forward (ops_conv.cu). per-sample. */
ax_status_t cuda_conv_gemm(const ax_tensor_t *weight,
                            const ax_conv_params_t *params,
                            ax_tensor_t *out);

/* memory + transfer hooks (memory.cu) */
void       *cuda_storage_alloc_hook(size_t size_bytes);
void        cuda_storage_free_hook(void *ptr);
ax_status_t cuda_memcpy_h2d_hook(void *dst, const void *src, size_t bytes);
ax_status_t cuda_memcpy_d2h_hook(void *dst, const void *src, size_t bytes);
ax_status_t cuda_memcpy_d2d_hook(void *dst, const void *src, size_t bytes);

} /* extern "C" */

#endif /* AX_CUDA_INTERNAL_H */
