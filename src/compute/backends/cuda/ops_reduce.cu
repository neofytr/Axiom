/* ops_reduce.cu — sum / mean / max / min.
   full reductions use a two-level warp-shuffle pattern (one block-wide
   reduce, then a second pass over block results). named-axis reductions
   use a simple one-thread-per-output-element loop — fine for the sizes
   we care about now; revisit if profiling shows it matters.

   cuda_mean is implemented as cuda_sum followed by a scalar-multiply;
   rather than duplicate the k_mul_scalar kernel here, we call the
   elementwise op directly through its extern "C" declaration in
   internal.h. cross-tu call but not cross-kernel. */

#include "internal.h"

/* ── warp-level helpers ───────────────────────────────────────────── */

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

/* ── full-reduce kernels: one value per block ─────────────────────── */

__global__ static void k_reduce_sum_full(const float *in, float *out, int64_t off, int64_t n) {
    __shared__ float smem[AX_CUDA_BLOCK / 32];
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? in[off + i] : 0.0f;
    v = warp_reduce_sum(v);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x / 32] = v;
    __syncthreads();
    if (threadIdx.x < (AX_CUDA_BLOCK / 32)) {
        v = smem[threadIdx.x];
        v = warp_reduce_sum(v);
    }
    if (threadIdx.x == 0) out[blockIdx.x] = v;
}

__global__ static void k_reduce_max_full(const float *in, float *out, int64_t off, int64_t n) {
    __shared__ float smem[AX_CUDA_BLOCK / 32];
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? in[off + i] : -FLT_MAX;
    v = warp_reduce_max(v);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x / 32] = v;
    __syncthreads();
    if (threadIdx.x < (AX_CUDA_BLOCK / 32)) {
        v = smem[threadIdx.x];
        v = warp_reduce_max(v);
    }
    if (threadIdx.x == 0) out[blockIdx.x] = v;
}

__global__ static void k_reduce_min_full(const float *in, float *out, int64_t off, int64_t n) {
    __shared__ float smem[AX_CUDA_BLOCK / 32];
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? in[off + i] : FLT_MAX;
    v = warp_reduce_min(v);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x / 32] = v;
    __syncthreads();
    if (threadIdx.x < (AX_CUDA_BLOCK / 32)) {
        v = smem[threadIdx.x];
        v = warp_reduce_min(v);
    }
    if (threadIdx.x == 0) out[blockIdx.x] = v;
}

/* ── axis-reduce kernels: one thread per output element ───────────── */

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

/* ── two-level full-reduce helpers returning a host float ─────────── */

static float full_reduce_sum(const float *d_in, int64_t off, int64_t n) {
    int blocks = (int)((n + AX_CUDA_BLOCK - 1) / AX_CUDA_BLOCK);
    float *d_tmp; cudaMalloc(&d_tmp, (size_t)blocks * sizeof(float));
    k_reduce_sum_full<<<blocks, AX_CUDA_BLOCK>>>(d_in, d_tmp, off, n);
    if (blocks > 1) {
        float *d2; cudaMalloc(&d2, sizeof(float));
        k_reduce_sum_full<<<1, AX_CUDA_BLOCK>>>(d_tmp, d2, 0, (int64_t)blocks);
        float r; cudaMemcpy(&r, d2, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_tmp); cudaFree(d2); return r;
    }
    float r; cudaMemcpy(&r, d_tmp, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_tmp); return r;
}

static float full_reduce_max(const float *d_in, int64_t off, int64_t n) {
    int blocks = (int)((n + AX_CUDA_BLOCK - 1) / AX_CUDA_BLOCK);
    float *d_tmp; cudaMalloc(&d_tmp, (size_t)blocks * sizeof(float));
    k_reduce_max_full<<<blocks, AX_CUDA_BLOCK>>>(d_in, d_tmp, off, n);
    if (blocks > 1) {
        float *d2; cudaMalloc(&d2, sizeof(float));
        k_reduce_max_full<<<1, AX_CUDA_BLOCK>>>(d_tmp, d2, 0, (int64_t)blocks);
        float r; cudaMemcpy(&r, d2, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_tmp); cudaFree(d2); return r;
    }
    float r; cudaMemcpy(&r, d_tmp, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_tmp); return r;
}

static float full_reduce_min(const float *d_in, int64_t off, int64_t n) {
    int blocks = (int)((n + AX_CUDA_BLOCK - 1) / AX_CUDA_BLOCK);
    float *d_tmp; cudaMalloc(&d_tmp, (size_t)blocks * sizeof(float));
    k_reduce_min_full<<<blocks, AX_CUDA_BLOCK>>>(d_in, d_tmp, off, n);
    if (blocks > 1) {
        float *d2; cudaMalloc(&d2, sizeof(float));
        k_reduce_min_full<<<1, AX_CUDA_BLOCK>>>(d_tmp, d2, 0, (int64_t)blocks);
        float r; cudaMemcpy(&r, d2, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_tmp); cudaFree(d2); return r;
    }
    float r; cudaMemcpy(&r, d_tmp, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_tmp); return r;
}

/* ── shape helpers ────────────────────────────────────────────────── */

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

/* ── backend op wrappers ──────────────────────────────────────────── */

extern "C" {

ax_status_t cuda_sum(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
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
    k_reduce_sum_axis<<<(int)((n_out+AX_CUDA_BLOCK-1)/AX_CUDA_BLOCK), AX_CUDA_BLOCK>>>(
        d + in->offset, (float *)out->storage->data + out->offset,
        outer, axis_len, inner);
    return AX_OK;
}

ax_status_t cuda_mean(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    ax_status_t s = cuda_sum(in, axis, out);
    if (s != AX_OK) return s;
    int64_t count = 1;
    if (axis == -1) { for (int i = 0; i < in->ndim; i++) count *= in->shape[i]; }
    else            { count = in->shape[axis]; }
    /* in-place scalar divide on out: call the elementwise op directly
       (cross-tu extern "C" declared in internal.h). */
    return cuda_mul_scalar(out, 1.0 / (double)count, out);
}

ax_status_t cuda_max(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
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
    k_reduce_max_axis<<<(int)((n_out+AX_CUDA_BLOCK-1)/AX_CUDA_BLOCK), AX_CUDA_BLOCK>>>(
        d + in->offset, (float *)out->storage->data + out->offset,
        outer, axis_len, inner);
    return AX_OK;
}

ax_status_t cuda_min(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
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
    k_reduce_min_axis<<<(int)((n_out+AX_CUDA_BLOCK-1)/AX_CUDA_BLOCK), AX_CUDA_BLOCK>>>(
        d + in->offset, (float *)out->storage->data + out->offset,
        outer, axis_len, inner);
    return AX_OK;
}

} /* extern "C" */
