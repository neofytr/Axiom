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

/* ── device-side scratch arena ─────────────────────────────────────
   single persistent buffer used by ops that need to upload small
   metadata structs without paying cudaMalloc/cudaFree per call. see
   internal.h for the contract. */

static void *g_cuda_scratch    = NULL;
static size_t g_cuda_scratch_used = 0;

/* round up to 16-byte alignment so structs with int64 arrays land on
   natural boundaries. */
static inline size_t align_up_16(size_t x) {
    return (x + 15u) & ~((size_t)15u);
}

void *ax_cuda_scratch_alloc(size_t bytes) {
    if (!g_cuda_scratch) return NULL;
    size_t aligned = align_up_16(bytes);
    if (g_cuda_scratch_used + aligned > AX_CUDA_SCRATCH_BYTES) return NULL;
    void *p = (char *)g_cuda_scratch + g_cuda_scratch_used;
    g_cuda_scratch_used += aligned;
    return p;
}

void ax_cuda_scratch_reset(void) {
    g_cuda_scratch_used = 0;
}

/* ── lifecycle hooks ──────────────────────────────────────────────── */

extern "C" {

static ax_status_t cuda_init_hook(void) {
    /* force-create the cublas handle now so the first kernel launch
       doesn't race with lazy allocation. if no cuda device is present
       this silently leaves g_cublas NULL and later ops will fail with
       a cublas status error. */
    cublasHandle_t h = ax_cuda_cublas_handle();

    /* enable tf32 tensor-core math on ampere+ gpus (sm_80+). this gives
       ~2x gemm throughput at 19-bit mantissa precision — indistinguishable
       from fp32 for training. on pre-ampere gpus the call succeeds but
       has no effect. the mode only applies to cublasSgemm; custom kernels
       still use full fp32. */
    if (h) {
        int dev = 0;
        cudaGetDevice(&dev);
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        if (major >= 8) {
            cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
        }
    }
    /* allocate the persistent device-side scratch arena. on failure
       we silently leave it NULL; ops that need scratch will then fall
       back to their per-call cudaMalloc path (or just fail). */
    if (!g_cuda_scratch) {
        if (cudaMalloc(&g_cuda_scratch, AX_CUDA_SCRATCH_BYTES) != cudaSuccess) {
            g_cuda_scratch = NULL;
        }
        g_cuda_scratch_used = 0;
    }
    /* k.4: register the cuda extension table so cpu paths can dispatch
       cuda-only ops (sdpa, layout transforms, batched conv gemm)
       through ax_compute_get_cuda_extension() instead of weak symbols. */
    ax_compute_register_cuda_extension(&ax_cuda_ext_table);
    return AX_OK;
}

static void cuda_shutdown_hook(void) {
    /* clear the extension registration first so any in-flight cpu code
       falls back to its non-cuda path on the way out. */
    ax_compute_register_cuda_extension(NULL);
    if (g_cuda_scratch) {
        cudaFree(g_cuda_scratch);
        g_cuda_scratch = NULL;
        g_cuda_scratch_used = 0;
    }
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
    .gemm_nt    = cuda_gemm_nt,
    .gemm_tn    = cuda_gemm_tn,
    .gemm_ex    = cuda_gemm_ex,
    .add_relu   = cuda_add_relu,
    .axpy       = cuda_axpy,
    .softmax_rowwise = cuda_softmax_rowwise,
    .bias_add   = cuda_bias_add,
    .conv_gemm  = cuda_conv_gemm,
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
    .leaky_relu = cuda_leaky_relu,
    .elu_op     = cuda_elu_op,
    .gelu_op    = cuda_gelu_op,
    .swish_op   = cuda_swish_op,
    .adam_update = cuda_adam_update,
    .sgd_update  = cuda_sgd_update,

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

/* ── cuda extension registry (k.4) ──
   ops_attention.cu and ops_conv.cu provide cuda implementations of
   layout transforms, sdpa, weight/bias cache builds, and batched conv
   gemm. previously cpu code reached them via __attribute__((weak))
   externs and a force-link table here kept the .o files in the static
   archive. that approach scattered the contract across many call sites.

   k.4 collapses the contract into one ax_cuda_extension_t struct
   defined in axiom/internal/cuda_extension.h. cuda_init_hook below
   constructs the table and registers it via
   ax_compute_register_cuda_extension(). cpu code calls
   ax_compute_get_cuda_extension() and dispatches through the struct
   instead of through weak symbols. removes 10 weak externs, the
   force-link table, and the implicit symbol-keep-alive contract. */
#include "axiom/internal/cuda_extension.h"

extern "C" {
extern void ax_cuda_head_interleave(const float *, float *,
    int64_t, int64_t, int64_t, int64_t);
extern void ax_cuda_head_deinterleave(const float *, float *,
    int64_t, int64_t, int64_t, int64_t);
extern void ax_cuda_head_interleave_qkv_split(const float *,
    float *, float *, float *,
    int64_t, int64_t, int64_t, int64_t, int64_t);
extern void ax_cuda_head_interleave_qkv_split_bias(const float *, const float *,
    float *, float *, float *,
    int64_t, int64_t, int64_t, int64_t, int64_t);
extern void ax_cuda_head_deinterleave_qkv_merge(const float *, const float *, const float *,
    float *, int64_t, int64_t, int64_t, int64_t, int64_t);
extern ax_status_t ax_cuda_qkv_cache_build(float *,
    const float *, const float *, const float *, int64_t);
extern ax_status_t ax_cuda_bias_cache_build(float *,
    const float *, const float *, const float *, int64_t);
extern ax_status_t ax_cuda_sdpa_fwd(const float *, const float *, const float *,
    float *, float *, int64_t, int64_t, int64_t, float, bool, float *);
extern ax_status_t ax_cuda_sdpa_bwd(const float *, const float *, const float *,
    const float *, const float *, const float *,
    float *, float *, float *,
    int64_t, int64_t, int64_t, float, bool, const float *);
extern ax_status_t cuda_conv_gemm_batched(const ax_tensor_t *,
    const float *, int64_t, const ax_conv_params_t *, ax_tensor_t *);
}

/* the extension table — populated at module load. shared across all
   threads (read-only after init); registered with the dispatch
   subsystem during cuda_init_hook. */
static const ax_cuda_extension_t ax_cuda_ext_table = {
    .head_interleave_qkv_split        = ax_cuda_head_interleave_qkv_split,
    .head_interleave_qkv_split_bias   = ax_cuda_head_interleave_qkv_split_bias,
    .head_interleave                  = ax_cuda_head_interleave,
    .head_deinterleave                = ax_cuda_head_deinterleave,
    .head_deinterleave_qkv_merge      = ax_cuda_head_deinterleave_qkv_merge,
    .qkv_cache_build                  = ax_cuda_qkv_cache_build,
    .bias_cache_build                 = ax_cuda_bias_cache_build,
    .sdpa_fwd                         = ax_cuda_sdpa_fwd,
    .sdpa_bwd                         = ax_cuda_sdpa_bwd,
    .conv_gemm_batched                = cuda_conv_gemm_batched,
};
