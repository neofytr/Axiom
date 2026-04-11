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

/* memory + transfer hooks (memory.cu) */
void       *cuda_storage_alloc_hook(size_t size_bytes);
void        cuda_storage_free_hook(void *ptr);
ax_status_t cuda_memcpy_h2d_hook(void *dst, const void *src, size_t bytes);
ax_status_t cuda_memcpy_d2h_hook(void *dst, const void *src, size_t bytes);
ax_status_t cuda_memcpy_d2d_hook(void *dst, const void *src, size_t bytes);

} /* extern "C" */

#endif /* AX_CUDA_INTERNAL_H */
