/* ops_elementwise.cu — all per-element cuda ops.
   binary (add/sub/mul/div/equal/greater), unary (neg/abs/exp/log/sqrt/
   square/relu/sigmoid/tanh), scalar (add_scalar/mul_scalar), and the
   data-movement wrappers fill + copy.

   design notes:
     * broadcasting follows the same semantics as cpu_naive: each output
       flat index is mapped back to an input offset via stride/shape
       arithmetic in bcast_offset.
     * binop uses a single device-side BinopMeta struct uploaded once
       per call to hold the shape/stride arrays. phase 1 still uses
       cudaMalloc/cudaFree per op; the persistent scratch arena lands
       in a follow-up. */

#include "internal.h"

/* ── broadcasting helper ──────────────────────────────────────────── */

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

/* ── binop metadata (device-side) ─────────────────────────────────── */

struct BinopMeta {
    int64_t as[8], ast[8]; int an;
    int64_t bs[8], bst[8]; int bn;
    int64_t os[8], ost[8]; int on;
};

/* ── binary op kernels ────────────────────────────────────────────── */

#define BINOP_KERNEL(kname, expr)                                               \
__global__ static void kname(                                                    \
        const float *a, const int64_t *as, const int64_t *ast, int an,          \
        const float *b, const int64_t *bs, const int64_t *bst, int bn,          \
        float *o,       const int64_t *os, const int64_t *ost, int on,          \
        int64_t oa_off, int64_t ob_off, int64_t oo_off, int64_t n)              \
{                                                                                \
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;                 \
    if (i >= n) return;                                                          \
    float va = a[bcast_offset(as, ast, an, os, on, oa_off, i)];                 \
    float vb = b[bcast_offset(bs, bst, bn, os, on, ob_off, i)];                 \
    o[oo_off + i] = (expr);                                                      \
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

    /* build meta on host, upload once to the persistent scratch arena.
       replaces the old cudaMalloc/cudaFree-per-call pattern. */
    ax_cuda_scratch_reset();
    BinopMeta *dm = (BinopMeta *)ax_cuda_scratch_alloc(sizeof(BinopMeta));
    if (!dm) {
        ax_err_set(AX_ERR_BACKEND, "cuda scratch arena too small for binop meta");
        return AX_ERR_BACKEND;
    }
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

    int blocks = (int)((n + AX_CUDA_BLOCK - 1) / AX_CUDA_BLOCK);
    /* scalar fields (an/bn/on) come from the host copy hm.* —
       dereferencing the device pointer dm->* on the host would
       segfault. array members decay to offset computations so
       dm->as / dm->bs / dm->os are still valid device pointers. */
    kernel<<<blocks, AX_CUDA_BLOCK>>>(
        (const float *)a->storage->data, dm->as, dm->ast, hm.an,
        (const float *)b->storage->data, dm->bs, dm->bst, hm.bn,
        (float *)      out->storage->data, dm->os, dm->ost, hm.on,
        (int64_t)a->offset, (int64_t)b->offset, (int64_t)out->offset, n);
    AX_CUDA_CHECK_LAUNCH("binop");
    return AX_OK;
}

/* ── unary op kernels ─────────────────────────────────────────────── */

#define UNOP_KERNEL(kname, expr)                                                \
__global__ static void kname(const float *in, float *out,                       \
                              int64_t in_off, int64_t out_off, int64_t n) {     \
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;                 \
    if (i >= n) return;                                                          \
    float v = in[in_off + i];                                                    \
    out[out_off + i] = (expr);                                                   \
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

/* ── float4 vectorized unary kernels ─────────────────────────────
   process 4 elements per thread via float4 load/store. dispatched
   when n is divisible by 4 and both offsets are 0 (guaranteeing
   16-byte alignment on the base pointer). */

#define UNOP_KERNEL_VEC4(kname, expr)                                           \
__global__ static void kname(const float *in, float *out, int64_t n4) {         \
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;                 \
    if (i >= n4) return;                                                         \
    float4 ld = ((const float4 *)in)[i];                                         \
    float4 st;                                                                   \
    { float v = ld.x; st.x = (expr); }                                          \
    { float v = ld.y; st.y = (expr); }                                          \
    { float v = ld.z; st.z = (expr); }                                          \
    { float v = ld.w; st.w = (expr); }                                          \
    ((float4 *)out)[i] = st;                                                     \
}

UNOP_KERNEL_VEC4(k_neg_v4,     -v)
UNOP_KERNEL_VEC4(k_abs_v4,     fabsf(v))
UNOP_KERNEL_VEC4(k_exp_v4,     expf(v))
UNOP_KERNEL_VEC4(k_log_v4,     v > 0.0f ? logf(v) : -FLT_MAX)
UNOP_KERNEL_VEC4(k_sqrt_v4,    v >= 0.0f ? sqrtf(v) : 0.0f)
UNOP_KERNEL_VEC4(k_square_v4,  v * v)
UNOP_KERNEL_VEC4(k_relu_v4,    v > 0.0f ? v : 0.0f)
UNOP_KERNEL_VEC4(k_sigmoid_v4, 1.0f / (1.0f + expf(-v)))
UNOP_KERNEL_VEC4(k_tanh_v4,    tanhf(v))

static ax_status_t run_unop(
    void(*kernel)(const float *, float *, int64_t, int64_t, int64_t),
    void(*kernel_v4)(const float *, float *, int64_t),
    const ax_tensor_t *in, ax_tensor_t *out)
{
    if (in->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cuda unop only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    int64_t n = 1;
    for (int d = 0; d < out->ndim; d++) n *= out->shape[d];

    /* dispatch to float4 path when aligned and evenly divisible */
    if (kernel_v4 && in->offset == 0 && out->offset == 0 && (n % 4) == 0) {
        int64_t n4 = n / 4;
        int blocks = (int)((n4 + AX_CUDA_BLOCK - 1) / AX_CUDA_BLOCK);
        kernel_v4<<<blocks, AX_CUDA_BLOCK>>>(
            (const float *)in->storage->data,
            (float *)out->storage->data, n4);
        AX_CUDA_CHECK_LAUNCH("unop_v4");
        return AX_OK;
    }

    int blocks = (int)((n + AX_CUDA_BLOCK - 1) / AX_CUDA_BLOCK);
    kernel<<<blocks, AX_CUDA_BLOCK>>>(
        (const float *)in->storage->data, (float *)out->storage->data,
        (int64_t)in->offset, (int64_t)out->offset, n);
    AX_CUDA_CHECK_LAUNCH("unop");
    return AX_OK;
}

/* ── scalar op kernels ────────────────────────────────────────────── */

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

/* ── fill kernel ──────────────────────────────────────────────────── */

__global__ static void k_fill(float *d, float v, int64_t off, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d[off + i] = v;
}

/* ── backend op wrappers (extern "C" to match internal.h decls) ──── */

extern "C" {

ax_status_t cuda_add(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    return run_binop(k_add, a, b, out); }
ax_status_t cuda_sub(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    return run_binop(k_sub, a, b, out); }
ax_status_t cuda_mul(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    return run_binop(k_mul, a, b, out); }
ax_status_t cuda_div(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    return run_binop(k_div, a, b, out); }
ax_status_t cuda_equal(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    return run_binop(k_equal, a, b, out); }
ax_status_t cuda_greater(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    return run_binop(k_greater, a, b, out); }

ax_status_t cuda_neg(const ax_tensor_t *in, ax_tensor_t *out)     { return run_unop(k_neg,     k_neg_v4,     in, out); }
ax_status_t cuda_abs(const ax_tensor_t *in, ax_tensor_t *out)     { return run_unop(k_abs,     k_abs_v4,     in, out); }
ax_status_t cuda_exp(const ax_tensor_t *in, ax_tensor_t *out)     { return run_unop(k_exp,     k_exp_v4,     in, out); }
ax_status_t cuda_log(const ax_tensor_t *in, ax_tensor_t *out)     { return run_unop(k_log,     k_log_v4,     in, out); }
ax_status_t cuda_sqrt(const ax_tensor_t *in, ax_tensor_t *out)    { return run_unop(k_sqrt,    k_sqrt_v4,    in, out); }
ax_status_t cuda_square(const ax_tensor_t *in, ax_tensor_t *out)  { return run_unop(k_square,  k_square_v4,  in, out); }
ax_status_t cuda_relu(const ax_tensor_t *in, ax_tensor_t *out)    { return run_unop(k_relu,    k_relu_v4,    in, out); }
ax_status_t cuda_sigmoid(const ax_tensor_t *in, ax_tensor_t *out) { return run_unop(k_sigmoid, k_sigmoid_v4, in, out); }
ax_status_t cuda_tanh_op(const ax_tensor_t *in, ax_tensor_t *out) { return run_unop(k_tanh,    k_tanh_v4,    in, out); }

ax_status_t cuda_add_scalar(const ax_tensor_t *in, double scalar, ax_tensor_t *out) {
    if (in->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cuda add_scalar only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    int64_t n = 1;
    for (int d = 0; d < out->ndim; d++) n *= out->shape[d];
    int blocks = (int)((n + AX_CUDA_BLOCK - 1) / AX_CUDA_BLOCK);
    k_add_scalar<<<blocks, AX_CUDA_BLOCK>>>(
        (const float *)in->storage->data, (float)scalar,
        (float *)out->storage->data,
        (int64_t)in->offset, (int64_t)out->offset, n);
    AX_CUDA_CHECK_LAUNCH("add_scalar");
    return AX_OK;
}

ax_status_t cuda_mul_scalar(const ax_tensor_t *in, double scalar, ax_tensor_t *out) {
    if (in->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cuda mul_scalar only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    int64_t n = 1;
    for (int d = 0; d < out->ndim; d++) n *= out->shape[d];
    int blocks = (int)((n + AX_CUDA_BLOCK - 1) / AX_CUDA_BLOCK);
    k_mul_scalar<<<blocks, AX_CUDA_BLOCK>>>(
        (const float *)in->storage->data, (float)scalar,
        (float *)out->storage->data,
        (int64_t)in->offset, (int64_t)out->offset, n);
    AX_CUDA_CHECK_LAUNCH("mul_scalar");
    return AX_OK;
}

ax_status_t cuda_fill(ax_tensor_t *t, double value) {
    if (t->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cuda fill only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    int64_t n = 1;
    for (int d = 0; d < t->ndim; d++) n *= t->shape[d];
    int blocks = (int)((n + AX_CUDA_BLOCK - 1) / AX_CUDA_BLOCK);
    k_fill<<<blocks, AX_CUDA_BLOCK>>>(
        (float *)t->storage->data, (float)value, (int64_t)t->offset, n);
    AX_CUDA_CHECK_LAUNCH("fill");
    return AX_OK;
}

ax_status_t cuda_copy(const ax_tensor_t *src, ax_tensor_t *dst) {
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
    if (cudaMemcpy((float *)dst->storage->data + dst->offset,
                   (const float *)src->storage->data + src->offset,
                   bytes, kind) != cudaSuccess) {
        ax_err_set(AX_ERR_BACKEND, "cuda copy: cudaMemcpy failed (%s)",
                   cudaGetErrorString(cudaGetLastError()));
        return AX_ERR_BACKEND;
    }
    return AX_OK;
}

} /* extern "C" */
