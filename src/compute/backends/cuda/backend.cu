/* backend.cu — cuda backend vtable, lifecycle hooks, cublas handle.
   this is the only translation unit that defines the vtable symbol
   ax_cuda_ops; every other .cu in this directory provides op
   implementations referenced here through internal.h. */

#include "internal.h"

/* ── cublas handle (one per process, shared across gemm + any other
      cublas-using op). created eagerly in cuda_init_hook so first
      kernel dispatch doesn't race. ────────────────────────────────── */

static cublasHandle_t g_cublas = NULL;

cublasHandle_t ax_cuda_cublas_handle(void) {
    if (!g_cublas) cublasCreate(&g_cublas);
    return g_cublas;
}

/* ── lifecycle hooks ──────────────────────────────────────────────── */

extern "C" {

static ax_status_t cuda_init_hook(void) {
    /* force-create the cublas handle now so the first kernel launch
       doesn't race with lazy allocation. if no cuda device is present
       this silently leaves g_cublas NULL and later ops will fail with
       a cublas status error. */
    ax_cuda_cublas_handle();
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

/* ── public convenience helper (axiom/cuda.h) ─────────────────────
   direct zero-tensor allocation on gpu without routing through cpu.
   kept here so cuda.h can forward to it without tensor.c knowing
   cuda exists. uses the thread-local default-device mechanism. */
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

/* ── vtable ────────────────────────────────────────────────────────
   extern overrides c++ default internal linkage for const globals,
   so dispatch.c sees the symbol. */

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
