/* ops_activations.cu — extended activation kernels.
   leaky_relu, elu, gelu (approximate), swish. leaky_relu and elu
   take a float alpha parameter so they use custom kernels rather
   than the UNOP_KERNEL macro. gelu uses the tanh approximation:
   0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))). swish is x*sigmoid(x). */

#include "internal.h"

/* ── gelu / swish via UNOP_KERNEL-style macros ───────────────────── */

#define AX_GELU_COEFF 0.7978845608f  /* sqrt(2/pi) */

#define UNOP_KERNEL(kname, expr)                                                \
__global__ static void kname(const float *in, float *out,                       \
                              int64_t in_off, int64_t out_off, int64_t n) {     \
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;                 \
    if (i >= n) return;                                                          \
    float v = in[in_off + i];                                                    \
    out[out_off + i] = (expr);                                                   \
}

UNOP_KERNEL(k_gelu,  0.5f * v * (1.0f + tanhf(AX_GELU_COEFF * (v + 0.044715f * v * v * v))))
UNOP_KERNEL(k_swish, v / (1.0f + expf(-v)))

/* ── leaky_relu: custom kernel with alpha param ──────────────────── */

__global__ static void k_leaky_relu(const float *in, float *out,
                                     int64_t in_off, int64_t out_off,
                                     int64_t n, float alpha)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[in_off + i];
    out[out_off + i] = v > 0.0f ? v : alpha * v;
}

/* ── elu: custom kernel with alpha param ─────────────────────────── */

__global__ static void k_elu(const float *in, float *out,
                              int64_t in_off, int64_t out_off,
                              int64_t n, float alpha)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[in_off + i];
    out[out_off + i] = v > 0.0f ? v : alpha * (expf(v) - 1.0f);
}

/* ── run helpers ─────────────────────────────────────────────────── */

static ax_status_t run_unop(
    void(*kernel)(const float *, float *, int64_t, int64_t, int64_t),
    const ax_tensor_t *in, ax_tensor_t *out, const char *tag)
{
    if (in->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cuda %s only supports float32", tag);
        return AX_ERR_DTYPE_MISMATCH;
    }
    int64_t n = 1;
    for (int d = 0; d < out->ndim; d++) n *= out->shape[d];
    int blocks = (int)((n + AX_CUDA_BLOCK - 1) / AX_CUDA_BLOCK);
    kernel<<<blocks, AX_CUDA_BLOCK>>>(
        (const float *)in->storage->data, (float *)out->storage->data,
        (int64_t)in->offset, (int64_t)out->offset, n);
    AX_CUDA_CHECK_LAUNCH(tag);
    return AX_OK;
}

/* ── extern "C" wrappers ─────────────────────────────────────────── */

extern "C" {

ax_status_t cuda_leaky_relu(const ax_tensor_t *in, float alpha, ax_tensor_t *out) {
    if (in->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cuda leaky_relu only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    int64_t n = 1;
    for (int d = 0; d < out->ndim; d++) n *= out->shape[d];
    int blocks = (int)((n + AX_CUDA_BLOCK - 1) / AX_CUDA_BLOCK);
    k_leaky_relu<<<blocks, AX_CUDA_BLOCK>>>(
        (const float *)in->storage->data, (float *)out->storage->data,
        (int64_t)in->offset, (int64_t)out->offset, n, alpha);
    AX_CUDA_CHECK_LAUNCH("leaky_relu");
    return AX_OK;
}

ax_status_t cuda_elu_op(const ax_tensor_t *in, float alpha, ax_tensor_t *out) {
    if (in->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cuda elu only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    int64_t n = 1;
    for (int d = 0; d < out->ndim; d++) n *= out->shape[d];
    int blocks = (int)((n + AX_CUDA_BLOCK - 1) / AX_CUDA_BLOCK);
    k_elu<<<blocks, AX_CUDA_BLOCK>>>(
        (const float *)in->storage->data, (float *)out->storage->data,
        (int64_t)in->offset, (int64_t)out->offset, n, alpha);
    AX_CUDA_CHECK_LAUNCH("elu");
    return AX_OK;
}

ax_status_t cuda_gelu_op(const ax_tensor_t *in, ax_tensor_t *out) {
    return run_unop(k_gelu, in, out, "gelu");
}

ax_status_t cuda_swish_op(const ax_tensor_t *in, ax_tensor_t *out) {
    return run_unop(k_swish, in, out, "swish");
}

} /* extern "C" */
