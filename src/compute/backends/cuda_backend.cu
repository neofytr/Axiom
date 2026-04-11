/* cuda_backend.cu — NVIDIA GPU compute backend using CUDA + cuBLAS.
   All 24 ops in ax_backend_ops_t are implemented here.

   Design notes:
   - All tensors reaching this backend must have storage->device == AX_DEVICE_CUDA.
   - Only AX_FLOAT32 is supported (matches the CPU backends).
   - Broadcasting follows the same semantic as cpu_naive: broadcast_index maps
     each output flat index to an input offset via stride/shape arithmetic.
   - cuBLAS GEMM: row-major A[M,K] @ B[K,N] -> C[M,N] is expressed as
       cublasSgemm(CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 1, B, N, A, K, 0, C, N)
     because cuBLAS expects column-major (C^T = B^T @ A^T trick).
   - Reductions use a two-level warp-shuffle approach for full-reduce (axis=-1),
     and a simple per-output-element loop for named-axis reductions.
   - All kernels use 256-thread blocks.                                           */

/* nvcc compiles .cu files as C++, but axiom headers use C11 <stdatomic.h>.
   Provide compatibility types/macros before any axiom include. */
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
/* prevent <stdatomic.h> from redefining these */
#define __CLANG_STDATOMIC_H
#define _STDATOMIC_H

#include "axiom/backend_ops.h"
#include "axiom/tensor.h"
#include "axiom/error.h"
#include "axiom/cuda.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <string.h>

/* ── cuBLAS handle (one per process) ─────────────────────────────────────── */

static cublasHandle_t g_cublas = NULL;

static cublasHandle_t cublas_handle(void) {
    if (!g_cublas) cublasCreate(&g_cublas);
    return g_cublas;
}

/* ── lifecycle + memory hooks (phase 0 module-ification) ──────────────────
   these are called by core via the backend vtable. they fully own cuda
   memory and device sync; core never touches cuda_runtime.h. */

extern "C" {

static ax_status_t cuda_init_hook(void) {
    /* eager-create the cublas handle so the first kernel launch doesn't
       race with allocation. harmless if no cuda device is present: cublas
       will lazy-fail later via status codes. */
    cublas_handle();
    return AX_OK;
}

static void cuda_shutdown_hook(void) {
    if (g_cublas) {
        cublasDestroy(g_cublas);
        g_cublas = NULL;
    }
}

static void cuda_synchronize_hook(void) {
    cudaDeviceSynchronize();
}

static int cuda_device_count_hook(void) {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
}

/* phase 0: keep unified memory semantics so existing tests (which mix cpu
   and cuda storage via page-faulting) still pass. phase 2 switches this
   to cudaMalloc + explicit transfers. */
static void *cuda_storage_alloc_hook(size_t size_bytes) {
    void *p = NULL;
    if (cudaMallocManaged(&p, size_bytes, cudaMemAttachGlobal) != cudaSuccess)
        return NULL;
    return p;
}

static void cuda_storage_free_hook(void *ptr) {
    if (ptr) cudaFree(ptr);
}

static ax_status_t cuda_memcpy_h2d_hook(void *dst, const void *src, size_t bytes) {
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        ax_err_set(AX_ERR_BACKEND, "cudaMemcpy H2D failed");
        return AX_ERR_BACKEND;
    }
    return AX_OK;
}

static ax_status_t cuda_memcpy_d2h_hook(void *dst, const void *src, size_t bytes) {
    /* unified memory is coherent after a device sync; cudaMemcpy handles
       the sync for us implicitly but an explicit barrier here is cheap
       insurance while we still use managed memory. */
    cudaDeviceSynchronize();
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        ax_err_set(AX_ERR_BACKEND, "cudaMemcpy D2H failed");
        return AX_ERR_BACKEND;
    }
    return AX_OK;
}

static ax_status_t cuda_memcpy_d2d_hook(void *dst, const void *src, size_t bytes) {
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice) != cudaSuccess) {
        ax_err_set(AX_ERR_BACKEND, "cudaMemcpy D2D failed");
        return AX_ERR_BACKEND;
    }
    return AX_OK;
}

/* direct allocation of a zero tensor on gpu (public convenience api).
   kept here so cuda.h can forward to it without tensor.c knowing cuda
   exists. uses the thread-local default-device mechanism. */
extern ax_tensor_t *ax_tensor_zeros(const int64_t *, int, ax_dtype_t);
extern void        ax_set_default_device(ax_device_t);
extern ax_device_t ax_get_default_device(void);

ax_tensor_t *ax_tensor_cuda_zeros(const int64_t *shape, int ndim, ax_dtype_t dtype) {
    ax_device_t prev = ax_get_default_device();
    ax_set_default_device(AX_DEVICE_CUDA);
    ax_tensor_t *t = ax_tensor_zeros(shape, ndim, dtype);
    ax_set_default_device(prev);
    return t;
}

} /* extern "C" */

/* ── kernel constants ─────────────────────────────────────────────────────── */

#define BLOCK 256

/* ── broadcasting helper ──────────────────────────────────────────────────── */

/* map a flat output index to a flat input pointer offset, respecting broadcast */
__device__ static inline int64_t bcast_offset(
        const int64_t *t_shape,   const int64_t *t_strides,   int t_ndim,
        const int64_t *out_shape, int out_ndim,
        int64_t t_base_offset,    int64_t out_flat)
{
    int64_t remaining = out_flat;
    int64_t off = t_base_offset;
    for (int d = out_ndim - 1; d >= 0; d--) {
        int64_t idx = remaining % out_shape[d];
        remaining  /= out_shape[d];
        int td = d - (out_ndim - t_ndim);
        if (td >= 0 && t_shape[td] > 1)
            off += idx * t_strides[td];
    }
    return off;
}

/* ── metadata struct uploaded once per binary op call ────────────────────── */

struct BinopMeta {
    int64_t as[8], ast[8]; int an;
    int64_t bs[8], bst[8]; int bn;
    int64_t os[8], ost[8]; int on;
};

/* ── binary op kernels ────────────────────────────────────────────────────── */

#define BINOP_KERNEL(kname, expr)                                                   \
__global__ static void kname(                                                        \
        const float *a, const int64_t *as, const int64_t *ast, int an,              \
        const float *b, const int64_t *bs, const int64_t *bst, int bn,              \
        float *o,       const int64_t *os, const int64_t *ost, int on,              \
        int64_t oa_off, int64_t ob_off, int64_t oo_off, int64_t n)                  \
{                                                                                    \
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (i >= n) return;                                                              \
    float va = a[bcast_offset(as, ast, an, os, on, oa_off, i)];                     \
    float vb = b[bcast_offset(bs, bst, bn, os, on, ob_off, i)];                     \
    o[oo_off + i] = (expr);                                                          \
}

BINOP_KERNEL(k_add,     va + vb)
BINOP_KERNEL(k_sub,     va - vb)
BINOP_KERNEL(k_mul,     va * vb)
BINOP_KERNEL(k_div,     va / vb)
BINOP_KERNEL(k_equal,   (va == vb) ? 1.0f : 0.0f)
BINOP_KERNEL(k_greater, (va >  vb) ? 1.0f : 0.0f)

static ax_status_t run_binop(
    void(*kernel)(const float *, const int64_t *, const int64_t *, int,
                  const float *, const int64_t *, const int64_t *, int,
                  float *,       const int64_t *, const int64_t *, int,
                  int64_t, int64_t, int64_t, int64_t),
    const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out)
{
    if (a->dtype != AX_FLOAT32 || b->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cuda binop only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    int64_t n = 1;
    for (int d = 0; d < out->ndim; d++) n *= out->shape[d];

    BinopMeta *dm;
    cudaMalloc(&dm, sizeof(BinopMeta));
    BinopMeta hm;
    memcpy(hm.as,  a->shape,     sizeof(int64_t) * (size_t)a->ndim);
    memcpy(hm.ast, a->strides,   sizeof(int64_t) * (size_t)a->ndim);
    hm.an = a->ndim;
    memcpy(hm.bs,  b->shape,     sizeof(int64_t) * (size_t)b->ndim);
    memcpy(hm.bst, b->strides,   sizeof(int64_t) * (size_t)b->ndim);
    hm.bn = b->ndim;
    memcpy(hm.os,  out->shape,   sizeof(int64_t) * (size_t)out->ndim);
    memcpy(hm.ost, out->strides, sizeof(int64_t) * (size_t)out->ndim);
    hm.on = out->ndim;
    cudaMemcpy(dm, &hm, sizeof(BinopMeta), cudaMemcpyHostToDevice);

    int blocks = (int)((n + BLOCK - 1) / BLOCK);
    // pass scalar fields from host copy (hm.*) — dereferencing device pointer
    // dm->* on the host would segfault. array members decay to offset
    // computations, so dm->as / dm->bs / dm->os are safe device pointers.
    kernel<<<blocks, BLOCK>>>(
        (const float *)a->storage->data, dm->as, dm->ast, hm.an,
        (const float *)b->storage->data, dm->bs, dm->bst, hm.bn,
        (float *)      out->storage->data, dm->os, dm->ost, hm.on,
        (int64_t)a->offset, (int64_t)b->offset, (int64_t)out->offset, n);

    cudaFree(dm);
    return AX_OK;
}

/* ── unary op kernels ─────────────────────────────────────────────────────── */

#define UNOP_KERNEL(kname, expr)                                                     \
__global__ static void kname(const float *in, float *out,                            \
                              int64_t in_off, int64_t out_off, int64_t n) {          \
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (i >= n) return;                                                              \
    float v = in[in_off + i];                                                        \
    out[out_off + i] = (expr);                                                       \
}

UNOP_KERNEL(k_neg,     -v)
UNOP_KERNEL(k_abs,     fabsf(v))
UNOP_KERNEL(k_exp,     expf(v))
UNOP_KERNEL(k_log,     v > 0.0f ? logf(v) : -FLT_MAX)
UNOP_KERNEL(k_sqrt,    v >= 0.0f ? sqrtf(v) : 0.0f)
UNOP_KERNEL(k_square,  v * v)
UNOP_KERNEL(k_relu,    v > 0.0f ? v : 0.0f)
UNOP_KERNEL(k_sigmoid, 1.0f / (1.0f + expf(-v)))
UNOP_KERNEL(k_tanh,    tanhf(v))

static ax_status_t run_unop(
    void(*kernel)(const float *, float *, int64_t, int64_t, int64_t),
    const ax_tensor_t *in, ax_tensor_t *out)
{
    if (in->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cuda unop only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    int64_t n = 1;
    for (int d = 0; d < out->ndim; d++) n *= out->shape[d];
    int blocks = (int)((n + BLOCK - 1) / BLOCK);
    kernel<<<blocks, BLOCK>>>(
        (const float *)in->storage->data, (float *)out->storage->data,
        (int64_t)in->offset, (int64_t)out->offset, n);
    return AX_OK;
}

/* ── scalar op kernels ────────────────────────────────────────────────────── */

__global__ static void k_add_scalar(const float *in, float s, float *out,
                                     int64_t in_off, int64_t out_off, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[out_off + i] = in[in_off + i] + s;
}

__global__ static void k_mul_scalar(const float *in, float s, float *out,
                                     int64_t in_off, int64_t out_off, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[out_off + i] = in[in_off + i] * s;
}

/* ── fill / copy kernels ──────────────────────────────────────────────────── */

__global__ static void k_fill(float *d, float v, int64_t off, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d[off + i] = v;
}

/* ── reduction helpers ────────────────────────────────────────────────────── */

__device__ static inline float warp_reduce_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}
__device__ static inline float warp_reduce_max(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    return v;
}
__device__ static inline float warp_reduce_min(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v = fminf(v, __shfl_down_sync(0xffffffff, v, off));
    return v;
}

/* full-reduce: one value per block */
__global__ static void k_reduce_sum_full(const float *in, float *out, int64_t off, int64_t n) {
    __shared__ float smem[BLOCK / 32];
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? in[off + i] : 0.0f;
    v = warp_reduce_sum(v);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x / 32] = v;
    __syncthreads();
    if (threadIdx.x < (BLOCK / 32)) {
        v = smem[threadIdx.x];
        v = warp_reduce_sum(v);
    }
    if (threadIdx.x == 0) out[blockIdx.x] = v;
}

__global__ static void k_reduce_max_full(const float *in, float *out, int64_t off, int64_t n) {
    __shared__ float smem[BLOCK / 32];
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? in[off + i] : -FLT_MAX;
    v = warp_reduce_max(v);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x / 32] = v;
    __syncthreads();
    if (threadIdx.x < (BLOCK / 32)) {
        v = smem[threadIdx.x];
        v = warp_reduce_max(v);
    }
    if (threadIdx.x == 0) out[blockIdx.x] = v;
}

__global__ static void k_reduce_min_full(const float *in, float *out, int64_t off, int64_t n) {
    __shared__ float smem[BLOCK / 32];
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? in[off + i] : FLT_MAX;
    v = warp_reduce_min(v);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x / 32] = v;
    __syncthreads();
    if (threadIdx.x < (BLOCK / 32)) {
        v = smem[threadIdx.x];
        v = warp_reduce_min(v);
    }
    if (threadIdx.x == 0) out[blockIdx.x] = v;
}

/* axis-reduce kernels: each thread owns one output element */
__global__ static void k_reduce_sum_axis(const float *in, float *out,
        int64_t outer, int64_t axis_len, int64_t inner) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer * inner) return;
    int64_t o = idx / inner, j = idx % inner;
    float s = 0.0f;
    for (int64_t a = 0; a < axis_len; a++)
        s += in[o * axis_len * inner + a * inner + j];
    out[idx] = s;
}

__global__ static void k_reduce_max_axis(const float *in, float *out,
        int64_t outer, int64_t axis_len, int64_t inner) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer * inner) return;
    int64_t o = idx / inner, j = idx % inner;
    float m = -FLT_MAX;
    for (int64_t a = 0; a < axis_len; a++) {
        float v = in[o * axis_len * inner + a * inner + j];
        if (v > m) m = v;
    }
    out[idx] = m;
}

__global__ static void k_reduce_min_axis(const float *in, float *out,
        int64_t outer, int64_t axis_len, int64_t inner) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer * inner) return;
    int64_t o = idx / inner, j = idx % inner;
    float m = FLT_MAX;
    for (int64_t a = 0; a < axis_len; a++) {
        float v = in[o * axis_len * inner + a * inner + j];
        if (v < m) m = v;
    }
    out[idx] = m;
}

/* two-level full-reduce helper returning a host float */
static float full_reduce_sum(const float *d_in, int64_t off, int64_t n) {
    int blocks = (int)((n + BLOCK - 1) / BLOCK);
    float *d_tmp; cudaMalloc(&d_tmp, (size_t)blocks * sizeof(float));
    k_reduce_sum_full<<<blocks, BLOCK>>>(d_in, d_tmp, off, n);
    if (blocks > 1) {
        float *d2; cudaMalloc(&d2, sizeof(float));
        k_reduce_sum_full<<<1, BLOCK>>>(d_tmp, d2, 0, (int64_t)blocks);
        float r; cudaMemcpy(&r, d2, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_tmp); cudaFree(d2); return r;
    }
    float r; cudaMemcpy(&r, d_tmp, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_tmp); return r;
}

static float full_reduce_max(const float *d_in, int64_t off, int64_t n) {
    int blocks = (int)((n + BLOCK - 1) / BLOCK);
    float *d_tmp; cudaMalloc(&d_tmp, (size_t)blocks * sizeof(float));
    k_reduce_max_full<<<blocks, BLOCK>>>(d_in, d_tmp, off, n);
    if (blocks > 1) {
        float *d2; cudaMalloc(&d2, sizeof(float));
        k_reduce_max_full<<<1, BLOCK>>>(d_tmp, d2, 0, (int64_t)blocks);
        float r; cudaMemcpy(&r, d2, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_tmp); cudaFree(d2); return r;
    }
    float r; cudaMemcpy(&r, d_tmp, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_tmp); return r;
}

static float full_reduce_min(const float *d_in, int64_t off, int64_t n) {
    int blocks = (int)((n + BLOCK - 1) / BLOCK);
    float *d_tmp; cudaMalloc(&d_tmp, (size_t)blocks * sizeof(float));
    k_reduce_min_full<<<blocks, BLOCK>>>(d_in, d_tmp, off, n);
    if (blocks > 1) {
        float *d2; cudaMalloc(&d2, sizeof(float));
        k_reduce_min_full<<<1, BLOCK>>>(d_tmp, d2, 0, (int64_t)blocks);
        float r; cudaMemcpy(&r, d2, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_tmp); cudaFree(d2); return r;
    }
    float r; cudaMemcpy(&r, d_tmp, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_tmp); return r;
}

static void reduction_dims(const ax_tensor_t *in, int axis,
                            int64_t *outer, int64_t *axis_len, int64_t *inner) {
    *outer = 1; *inner = 1; *axis_len = in->shape[axis];
    for (int d = 0;       d < axis;      d++) *outer *= in->shape[d];
    for (int d = axis + 1; d < in->ndim; d++) *inner *= in->shape[d];
}

static void dev_write_scalar(ax_tensor_t *out, float v) {
    cudaMemcpy((float *)out->storage->data + out->offset, &v,
               sizeof(float), cudaMemcpyHostToDevice);
}

/* ── backend op implementations ──────────────────────────────────────────── */

extern "C" {

static ax_status_t cuda_add(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    return run_binop(k_add, a, b, out); }
static ax_status_t cuda_sub(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    return run_binop(k_sub, a, b, out); }
static ax_status_t cuda_mul(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    return run_binop(k_mul, a, b, out); }
static ax_status_t cuda_div(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    return run_binop(k_div, a, b, out); }
static ax_status_t cuda_equal(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    return run_binop(k_equal, a, b, out); }
static ax_status_t cuda_greater(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    return run_binop(k_greater, a, b, out); }

static ax_status_t cuda_neg(const ax_tensor_t *in, ax_tensor_t *out)     { return run_unop(k_neg,     in, out); }
static ax_status_t cuda_abs(const ax_tensor_t *in, ax_tensor_t *out)     { return run_unop(k_abs,     in, out); }
static ax_status_t cuda_exp(const ax_tensor_t *in, ax_tensor_t *out)     { return run_unop(k_exp,     in, out); }
static ax_status_t cuda_log(const ax_tensor_t *in, ax_tensor_t *out)     { return run_unop(k_log,     in, out); }
static ax_status_t cuda_sqrt(const ax_tensor_t *in, ax_tensor_t *out)    { return run_unop(k_sqrt,    in, out); }
static ax_status_t cuda_square(const ax_tensor_t *in, ax_tensor_t *out)  { return run_unop(k_square,  in, out); }
static ax_status_t cuda_relu(const ax_tensor_t *in, ax_tensor_t *out)    { return run_unop(k_relu,    in, out); }
static ax_status_t cuda_sigmoid(const ax_tensor_t *in, ax_tensor_t *out) { return run_unop(k_sigmoid, in, out); }
static ax_status_t cuda_tanh_op(const ax_tensor_t *in, ax_tensor_t *out) { return run_unop(k_tanh,    in, out); }

static ax_status_t cuda_add_scalar(const ax_tensor_t *in, double scalar, ax_tensor_t *out) {
    if (in->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cuda add_scalar only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    int64_t n = 1;
    for (int d = 0; d < out->ndim; d++) n *= out->shape[d];
    int blocks = (int)((n + BLOCK - 1) / BLOCK);
    k_add_scalar<<<blocks, BLOCK>>>(
        (const float *)in->storage->data, (float)scalar,
        (float *)out->storage->data,
        (int64_t)in->offset, (int64_t)out->offset, n);
    return AX_OK;
}

static ax_status_t cuda_mul_scalar(const ax_tensor_t *in, double scalar, ax_tensor_t *out) {
    if (in->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cuda mul_scalar only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    int64_t n = 1;
    for (int d = 0; d < out->ndim; d++) n *= out->shape[d];
    int blocks = (int)((n + BLOCK - 1) / BLOCK);
    k_mul_scalar<<<blocks, BLOCK>>>(
        (const float *)in->storage->data, (float)scalar,
        (float *)out->storage->data,
        (int64_t)in->offset, (int64_t)out->offset, n);
    return AX_OK;
}

/* GEMM: A[M,K] @ B[K,N] = C[M,N]
   Row-major layout: cuBLAS C^T = B^T @ A^T trick
   => cublasSgemm(N, N, N, M, K, &alpha, B, N, A, K, &beta, C, N) */
static ax_status_t cuda_gemm(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    if (a->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cuda gemm only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    if (a->ndim != 2 || b->ndim != 2 || out->ndim != 2) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "cuda gemm requires 2d tensors");
        return AX_ERR_SHAPE_MISMATCH;
    }
    int M = (int)a->shape[0], K = (int)a->shape[1], N = (int)b->shape[1];
    const float *ad = (const float *)a->storage->data   + a->offset;
    const float *bd = (const float *)b->storage->data   + b->offset;
    float       *cd = (float *)      out->storage->data + out->offset;
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasSgemm(
        cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha, bd, N, ad, K, &beta, cd, N);
    if (st != CUBLAS_STATUS_SUCCESS) {
        ax_err_set(AX_ERR_BACKEND, "cublasSgemm failed (%d)", (int)st);
        return AX_ERR_BACKEND;
    }
    return AX_OK;
}

static ax_status_t cuda_sum(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    if (in->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cuda sum only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    const float *d = (const float *)in->storage->data;
    if (axis == -1) {
        int64_t n = 1;
        for (int i = 0; i < in->ndim; i++) n *= in->shape[i];
        dev_write_scalar(out, full_reduce_sum(d, (int64_t)in->offset, n));
        return AX_OK;
    }
    if (axis < 0 || axis >= in->ndim) {
        ax_err_set(AX_ERR_INVALID_AXIS, "axis %d out of range", axis);
        return AX_ERR_INVALID_AXIS;
    }
    int64_t outer, axis_len, inner;
    reduction_dims(in, axis, &outer, &axis_len, &inner);
    int64_t n_out = outer * inner;
    k_reduce_sum_axis<<<(int)((n_out+BLOCK-1)/BLOCK), BLOCK>>>(
        d + in->offset, (float *)out->storage->data + out->offset,
        outer, axis_len, inner);
    return AX_OK;
}

static ax_status_t cuda_mean(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    ax_status_t s = cuda_sum(in, axis, out);
    if (s != AX_OK) return s;
    int64_t count = 1;
    if (axis == -1) { for (int i = 0; i < in->ndim; i++) count *= in->shape[i]; }
    else            { count = in->shape[axis]; }
    int64_t n = 1;
    for (int d = 0; d < out->ndim; d++) n *= out->shape[d];
    int blocks = (int)((n + BLOCK - 1) / BLOCK);
    k_mul_scalar<<<blocks, BLOCK>>>(
        (const float *)out->storage->data, 1.0f / (float)count,
        (float *)out->storage->data,
        (int64_t)out->offset, (int64_t)out->offset, n);
    return AX_OK;
}

static ax_status_t cuda_max(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    if (in->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cuda max only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    const float *d = (const float *)in->storage->data;
    if (axis == -1) {
        int64_t n = 1;
        for (int i = 0; i < in->ndim; i++) n *= in->shape[i];
        dev_write_scalar(out, full_reduce_max(d, (int64_t)in->offset, n));
        return AX_OK;
    }
    if (axis < 0 || axis >= in->ndim) {
        ax_err_set(AX_ERR_INVALID_AXIS, "axis %d out of range", axis);
        return AX_ERR_INVALID_AXIS;
    }
    int64_t outer, axis_len, inner;
    reduction_dims(in, axis, &outer, &axis_len, &inner);
    int64_t n_out = outer * inner;
    k_reduce_max_axis<<<(int)((n_out+BLOCK-1)/BLOCK), BLOCK>>>(
        d + in->offset, (float *)out->storage->data + out->offset,
        outer, axis_len, inner);
    return AX_OK;
}

static ax_status_t cuda_min(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    if (in->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cuda min only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    const float *d = (const float *)in->storage->data;
    if (axis == -1) {
        int64_t n = 1;
        for (int i = 0; i < in->ndim; i++) n *= in->shape[i];
        dev_write_scalar(out, full_reduce_min(d, (int64_t)in->offset, n));
        return AX_OK;
    }
    if (axis < 0 || axis >= in->ndim) {
        ax_err_set(AX_ERR_INVALID_AXIS, "axis %d out of range", axis);
        return AX_ERR_INVALID_AXIS;
    }
    int64_t outer, axis_len, inner;
    reduction_dims(in, axis, &outer, &axis_len, &inner);
    int64_t n_out = outer * inner;
    k_reduce_min_axis<<<(int)((n_out+BLOCK-1)/BLOCK), BLOCK>>>(
        d + in->offset, (float *)out->storage->data + out->offset,
        outer, axis_len, inner);
    return AX_OK;
}

static ax_status_t cuda_fill(ax_tensor_t *t, double value) {
    if (t->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cuda fill only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    int64_t n = 1;
    for (int d = 0; d < t->ndim; d++) n *= t->shape[d];
    int blocks = (int)((n + BLOCK - 1) / BLOCK);
    k_fill<<<blocks, BLOCK>>>((float *)t->storage->data, (float)value, (int64_t)t->offset, n);
    return AX_OK;
}

static ax_status_t cuda_copy(const ax_tensor_t *src, ax_tensor_t *dst) {
    if (src->dtype != AX_FLOAT32 || dst->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cuda copy only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    int64_t n = 1;
    for (int d = 0; d < src->ndim; d++) n *= src->shape[d];
    size_t bytes = (size_t)n * sizeof(float);
    bool src_gpu = (src->storage->device == AX_DEVICE_CUDA);
    bool dst_gpu = (dst->storage->device == AX_DEVICE_CUDA);
    cudaMemcpyKind kind =
        ( src_gpu &&  dst_gpu) ? cudaMemcpyDeviceToDevice :
        ( src_gpu && !dst_gpu) ? cudaMemcpyDeviceToHost   :
        (!src_gpu &&  dst_gpu) ? cudaMemcpyHostToDevice   :
                                  cudaMemcpyHostToHost;
    cudaMemcpy((float *)dst->storage->data + dst->offset,
               (const float *)src->storage->data + src->offset,
               bytes, kind);
    return AX_OK;
}

/* ── vtable ───────────────────────────────────────────────────────────────── */

/* extern overrides C++ default internal linkage for const globals */
extern const ax_backend_ops_t ax_cuda_ops = {
    .name       = "cuda",
    .add        = cuda_add,
    .sub        = cuda_sub,
    .mul        = cuda_mul,
    .div_op     = cuda_div,
    .neg        = cuda_neg,
    .abs_op     = cuda_abs,
    .exp_op     = cuda_exp,
    .log_op     = cuda_log,
    .sqrt_op    = cuda_sqrt,
    .square     = cuda_square,
    .add_scalar = cuda_add_scalar,
    .mul_scalar = cuda_mul_scalar,
    .gemm       = cuda_gemm,
    .conv_gemm  = NULL,
    .sum        = cuda_sum,
    .mean       = cuda_mean,
    .max_op     = cuda_max,
    .min_op     = cuda_min,
    .equal      = cuda_equal,
    .greater    = cuda_greater,
    .fill       = cuda_fill,
    .copy       = cuda_copy,
    .relu       = cuda_relu,
    .sigmoid    = cuda_sigmoid,
    .tanh_op    = cuda_tanh_op,

    /* device-owner hooks */
    .device        = AX_DEVICE_CUDA,
    .init          = cuda_init_hook,
    .shutdown      = cuda_shutdown_hook,
    .synchronize   = cuda_synchronize_hook,
    .device_count  = cuda_device_count_hook,
    .storage_alloc = cuda_storage_alloc_hook,
    .storage_free  = cuda_storage_free_hook,
    .memcpy_h2d    = cuda_memcpy_h2d_hook,
    .memcpy_d2h    = cuda_memcpy_d2h_hook,
    .memcpy_d2d    = cuda_memcpy_d2d_hook,
};

} /* extern "C" */
