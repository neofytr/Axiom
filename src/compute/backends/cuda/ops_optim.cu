/* ops_optim.cu — fused optimizer update kernels on gpu.
   each kernel does the full weight_decay + moment_update + weight_update
   in a single pass over the parameter tensor. one thread per element. */

#include "internal.h"

/* adam: fused moment update + bias correction + weight update.
   matches the cpu adam_step in optim.c exactly:
     w *= decay_scale (decoupled decay)
     g += wd_grad * w (l2 through gradient)
     m = beta1*m + (1-beta1)*g
     v = beta2*v + (1-beta2)*g^2
     m_hat = m / (1 - beta1^t)
     v_hat = v / (1 - beta2^t)
     w -= lr * m_hat / (sqrt(v_hat) + eps) */
__global__ static void k_adam_update(
    float *w, const float *g, float *m, float *v,
    int64_t w_off, int64_t g_off, int64_t n,
    float lr, float beta1, float beta2, float eps,
    float decay_scale, float wd_grad, float bc1, float bc2)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float wi = w[w_off + i] * decay_scale;
    float gi = g[g_off + i] + wd_grad * wi;
    float mi = beta1 * m[i] + (1.0f - beta1) * gi;
    float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
    float mh = mi / bc1;
    float vh = vi / bc2;
    w[w_off + i] = wi - lr * mh / (sqrtf(vh) + eps);
    m[i] = mi;
    v[i] = vi;
}

/* sgd: fused weight_decay + momentum + optional nesterov.
   matches the cpu sgd_step exactly. if mom_buf is NULL, does
   simple w -= lr * g with optional l2 weight decay. */
__global__ static void k_sgd_update(
    float *w, const float *g, float *mom_buf,
    int64_t w_off, int64_t g_off, int64_t n,
    float lr, float momentum, float weight_decay, int nesterov)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float gi = g[g_off + i];
    if (weight_decay > 0.0f) gi += weight_decay * w[w_off + i];

    if (mom_buf && momentum > 0.0f) {
        float vi = momentum * mom_buf[i] + gi;
        mom_buf[i] = vi;
        if (nesterov)
            w[w_off + i] -= lr * (gi + momentum * vi);
        else
            w[w_off + i] -= lr * vi;
    } else {
        w[w_off + i] -= lr * gi;
    }
}


extern "C" {

ax_status_t cuda_adam_update(
    ax_tensor_t *weight, ax_tensor_t *grad,
    ax_tensor_t *m, ax_tensor_t *v,
    float lr, float beta1, float beta2, float eps,
    float weight_decay, float bc1, float bc2, bool decoupled)
{
    if (!weight || !grad || !m || !v) return AX_ERR_NULL_ARG;
    int64_t n = 1;
    for (int d = 0; d < weight->ndim; d++) n *= weight->shape[d];

    float decay_scale = decoupled ? (1.0f - lr * weight_decay) : 1.0f;
    if (decay_scale < 0.0f) decay_scale = 0.0f;
    float wd_grad = (!decoupled && weight_decay > 0.0f) ? weight_decay : 0.0f;

    int blocks = (int)((n + AX_CUDA_BLOCK - 1) / AX_CUDA_BLOCK);
    k_adam_update<<<blocks, AX_CUDA_BLOCK>>>(
        (float *)weight->storage->data,
        (const float *)grad->storage->data,
        (float *)m->storage->data,
        (float *)v->storage->data,
        (int64_t)weight->offset, (int64_t)grad->offset, n,
        lr, beta1, beta2, eps, decay_scale, wd_grad, bc1, bc2);
    AX_CUDA_CHECK_LAUNCH("adam_update");
    return AX_OK;
}

ax_status_t cuda_sgd_update(
    ax_tensor_t *weight, ax_tensor_t *grad,
    ax_tensor_t *momentum_buf,
    float lr, float momentum, float weight_decay, bool nesterov)
{
    if (!weight || !grad) return AX_ERR_NULL_ARG;
    int64_t n = 1;
    for (int d = 0; d < weight->ndim; d++) n *= weight->shape[d];

    float *mb = momentum_buf ? (float *)momentum_buf->storage->data : NULL;
    int blocks = (int)((n + AX_CUDA_BLOCK - 1) / AX_CUDA_BLOCK);
    k_sgd_update<<<blocks, AX_CUDA_BLOCK>>>(
        (float *)weight->storage->data,
        (const float *)grad->storage->data,
        mb,
        (int64_t)weight->offset, (int64_t)grad->offset, n,
        lr, momentum, weight_decay, nesterov ? 1 : 0);
    AX_CUDA_CHECK_LAUNCH("sgd_update");
    return AX_OK;
}

} /* extern "C" */
