/* ops_fused.cu — fused primitives for bandwidth-bound patterns.

   add_relu, axpy, softmax_rowwise. each is one or two kernel launches
   vs. the 2-3 separate dispatches the chain-of-ops variant would
   take. on gpu the memory-bandwidth saving matters more than the
   kernel-launch reduction — fewer round trips through hbm. */

#include "internal.h"

/* ── add_relu ──────────────────────────────────────────────────────
   single-thread-per-element kernel. same-shape only (broadcast
   version would need a BinopMeta like the rest of ops_elementwise;
   callers with broadcast should go through add then relu instead). */

__global__ static void k_add_relu(
        const float *a, const float *b, float *o,
        int64_t a_off, int64_t b_off, int64_t o_off, int64_t n)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = a[a_off + i] + b[b_off + i];
    o[o_off + i] = v > 0.0f ? v : 0.0f;
}

/* ── axpy ──────────────────────────────────────────────────────────
   y += alpha * x. y is read-modify-write; x is read-only. */

__global__ static void k_axpy(
        const float *x, float alpha, float *y,
        int64_t x_off, int64_t y_off, int64_t n)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[y_off + i] += alpha * x[x_off + i];
}

/* ── softmax_rowwise ──────────────────────────────────────────────
   numerically stable row-wise softmax. one block per row; block
   cooperates to find the row max, compute exp(x - max), sum, and
   normalise. row must fit in shared memory for the current impl —
   128-column rows at block size 256 is comfortable for mnist-class
   classifier heads. for wider rows we'd need a multi-pass reduction;
   deferred until a caller actually needs it. */

#define AX_SOFTMAX_ROW_MAX_COLS 1024

__global__ static void k_softmax_row(
        const float *in, float *out, int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *irow = in  + (int64_t)row * cols;
    float       *orow = out + (int64_t)row * cols;

    __shared__ float s_reduce[AX_CUDA_BLOCK];

    /* pass 1: row max via block reduction */
    float local_max = -FLT_MAX;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = irow[c];
        if (v > local_max) local_max = v;
    }
    s_reduce[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float r = s_reduce[threadIdx.x + stride];
            if (r > s_reduce[threadIdx.x]) s_reduce[threadIdx.x] = r;
        }
        __syncthreads();
    }
    float row_max = s_reduce[0];

    /* pass 2: exp(x - max) written to out, row sum via block reduction */
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float e = expf(irow[c] - row_max);
        orow[c] = e;
        local_sum += e;
    }
    s_reduce[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
        __syncthreads();
    }
    float row_sum = s_reduce[0];

    /* pass 3: normalise */
    float inv = 1.0f / row_sum;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        orow[c] *= inv;
    }
}

/* ── op wrappers ──────────────────────────────────────────────── */

extern "C" {

ax_status_t cuda_add_relu(const ax_tensor_t *a, const ax_tensor_t *b,
                           ax_tensor_t *out)
{
    if (a->dtype != AX_FLOAT32 || b->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32)
        return AX_ERR_DTYPE_MISMATCH;
    int64_t n = 1;
    for (int d = 0; d < out->ndim; d++) n *= out->shape[d];
    int64_t na = 1, nb = 1;
    for (int d = 0; d < a->ndim; d++) na *= a->shape[d];
    for (int d = 0; d < b->ndim; d++) nb *= b->shape[d];
    if (na != n || nb != n) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "cuda add_relu requires matching shapes");
        return AX_ERR_SHAPE_MISMATCH;
    }
    int blocks = (int)((n + AX_CUDA_BLOCK - 1) / AX_CUDA_BLOCK);
    k_add_relu<<<blocks, AX_CUDA_BLOCK>>>(
        (const float *)a->storage->data,
        (const float *)b->storage->data,
        (float *)      out->storage->data,
        (int64_t)a->offset, (int64_t)b->offset, (int64_t)out->offset, n);
    AX_CUDA_CHECK_LAUNCH("add_relu");
    return AX_OK;
}

ax_status_t cuda_axpy(const ax_tensor_t *x, float alpha, ax_tensor_t *y)
{
    if (x->dtype != AX_FLOAT32 || y->dtype != AX_FLOAT32)
        return AX_ERR_DTYPE_MISMATCH;
    int64_t nx = 1, ny = 1;
    for (int d = 0; d < x->ndim; d++) nx *= x->shape[d];
    for (int d = 0; d < y->ndim; d++) ny *= y->shape[d];
    if (nx != ny) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "cuda axpy requires matching numel");
        return AX_ERR_SHAPE_MISMATCH;
    }
    int blocks = (int)((nx + AX_CUDA_BLOCK - 1) / AX_CUDA_BLOCK);
    k_axpy<<<blocks, AX_CUDA_BLOCK>>>(
        (const float *)x->storage->data, alpha,
        (float *)      y->storage->data,
        (int64_t)x->offset, (int64_t)y->offset, nx);
    AX_CUDA_CHECK_LAUNCH("axpy");
    return AX_OK;
}

ax_status_t cuda_softmax_rowwise(const ax_tensor_t *in, ax_tensor_t *out)
{
    if (in->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32)
        return AX_ERR_DTYPE_MISMATCH;
    if (in->ndim != 2 || out->ndim != 2 ||
        in->shape[0] != out->shape[0] || in->shape[1] != out->shape[1]) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "cuda softmax_rowwise shape mismatch");
        return AX_ERR_SHAPE_MISMATCH;
    }
    int rows = (int)in->shape[0];
    int cols = (int)in->shape[1];
    if (cols > AX_SOFTMAX_ROW_MAX_COLS) {
        ax_err_set(AX_ERR_NOT_IMPLEMENTED,
                   "cuda softmax_rowwise: cols=%d exceeds single-block limit %d",
                   cols, AX_SOFTMAX_ROW_MAX_COLS);
        return AX_ERR_NOT_IMPLEMENTED;
    }
    k_softmax_row<<<rows, AX_CUDA_BLOCK>>>(
        (const float *)in->storage->data  + in->offset,
        (float *)      out->storage->data + out->offset,
        rows, cols);
    AX_CUDA_CHECK_LAUNCH("softmax_rowwise");
    return AX_OK;
}

/* bias_add: out[..., axis, ...] = in[..., axis, ...] + bias[axis].
   broadcast along the given axis. generic: computes (outer, axis_len,
   inner) strides and runs one thread per output element. */
__global__ static void k_bias_add(
    const float *in, const float *bias, float *out,
    int64_t in_off, int64_t b_off, int64_t out_off,
    int64_t outer, int64_t axis_len, int64_t inner, int64_t total)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int64_t a = (i / inner) % axis_len;
    out[out_off + i] = in[in_off + i] + bias[b_off + a];
}

ax_status_t cuda_bias_add(const ax_tensor_t *in, const ax_tensor_t *bias,
                           int axis, ax_tensor_t *out)
{
    if (!in || !bias || !out) return AX_ERR_NULL_ARG;
    if (in->dtype != AX_FLOAT32 || bias->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32)
        return AX_ERR_DTYPE_MISMATCH;
    if (bias->ndim != 1 || axis < 0 || axis >= in->ndim)
        return AX_ERR_SHAPE_MISMATCH;
    if (bias->shape[0] != in->shape[axis])
        return AX_ERR_SHAPE_MISMATCH;

    int64_t outer = 1, inner = 1;
    for (int d = 0; d < axis; d++) outer *= in->shape[d];
    int64_t axis_len = in->shape[axis];
    for (int d = axis + 1; d < in->ndim; d++) inner *= in->shape[d];
    int64_t total = outer * axis_len * inner;

    int blocks = (int)((total + AX_CUDA_BLOCK - 1) / AX_CUDA_BLOCK);
    k_bias_add<<<blocks, AX_CUDA_BLOCK>>>(
        (const float *)in->storage->data,
        (const float *)bias->storage->data,
        (float *)out->storage->data,
        (int64_t)in->offset, (int64_t)bias->offset, (int64_t)out->offset,
        outer, axis_len, inner, total);
    AX_CUDA_CHECK_LAUNCH("bias_add");
    return AX_OK;
}

} /* extern "C" */
