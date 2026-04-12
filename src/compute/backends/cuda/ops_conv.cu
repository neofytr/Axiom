/* ops_conv.cu — per-sample im2col + cuBLAS gemm for conv forward.

   this implements the conv_gemm vtable slot for the cuda backend.
   conv.c's forward pass chooses between direct-3x3, implicit-gemm
   (this), and explicit im2col+gemm based on shape heuristics and
   the backend's has_conv_gemm capability.

   algorithm (per sample):
     1. im2col: gather input patches [C_in, H, W] into a dense
        column matrix col[C_in*kh*kw, out_h*out_w]. one gpu thread
        per output element; bounds-checked with zero-padding.
     2. gemm: row-major weight[C_out, C_in*kh*kw] @ col =
        out[C_out, out_h*out_w] via cublasSgemm using the standard
        column-major-in-disguise identity.

   scratch: the col matrix can be large (e.g. 32*9*784*4 = 904 KB
   for a common mnist-style conv), too big for the persistent scratch
   arena. we cudaMalloc/cudaFree per call for now. a follow-up could
   cache it per-layer the same way cpu conv does in conv.c. */

#include "internal.h"

/* ── im2col kernel ────────────────────────────────────────────────
   each thread writes one element of the col matrix. the col matrix
   layout is row-major [C_in*kh*kw, out_h*out_w], so element (r, c)
   is at col[r * out_hw + c] where r indexes (channel, ky, kx) and c
   indexes (out_y, out_x). */

__global__ static void k_im2col_sample(
        const float *input, int64_t C_in, int64_t H, int64_t W,
        int kh, int kw, int sh, int sw, int ph, int pw,
        int64_t out_h, int64_t out_w,
        float *col)
{
    int64_t r = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
    int64_t c = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t rows = C_in * kh * kw;
    int64_t cols = out_h * out_w;
    if (r >= rows || c >= cols) return;

    /* unpack r -> (channel, ky, kx) */
    int64_t kx = r % kw;
    int64_t ky = (r / kw) % kh;
    int64_t ch = r / (kh * kw);

    /* unpack c -> (out_y, out_x) */
    int64_t out_y = c / out_w;
    int64_t out_x = c % out_w;

    int64_t in_y = out_y * sh - ph + ky;
    int64_t in_x = out_x * sw - pw + kx;

    float v = 0.0f;
    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
        v = input[ch * H * W + in_y * W + in_x];
    }
    col[r * cols + c] = v;
}

extern "C" {

ax_status_t cuda_conv_gemm(const ax_tensor_t *weight,
                            const ax_conv_params_t *params,
                            ax_tensor_t *out)
{
    if (weight->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH,
                   "cuda conv_gemm only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    if (weight->ndim != 2 || out->ndim != 2) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH,
                   "cuda conv_gemm requires 2d weight and output tensors");
        return AX_ERR_SHAPE_MISMATCH;
    }

    int64_t C_out = weight->shape[0];
    int64_t K     = weight->shape[1];            /* C_in * kh * kw */
    int64_t M     = params->out_h * params->out_w;
    int64_t expect_K = params->C_in * params->kh * params->kw;
    if (K != expect_K) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH,
                   "cuda conv_gemm: weight K=%ld does not match C_in*kh*kw=%ld",
                   (long)K, (long)expect_K);
        return AX_ERR_SHAPE_MISMATCH;
    }
    if (out->shape[0] != C_out || out->shape[1] != M) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH,
                   "cuda conv_gemm: out shape [%ld,%ld] doesn't match [%ld,%ld]",
                   (long)out->shape[0], (long)out->shape[1], (long)C_out, (long)M);
        return AX_ERR_SHAPE_MISMATCH;
    }

    /* allocate the col scratch on device. too big for the persistent
       arena in typical conv shapes. */
    size_t col_bytes = (size_t)K * (size_t)M * sizeof(float);
    float *d_col = NULL;
    if (cudaMalloc(&d_col, col_bytes) != cudaSuccess) {
        ax_err_set(AX_ERR_ALLOC, "cuda conv_gemm: cudaMalloc col failed");
        return AX_ERR_ALLOC;
    }

    /* im2col: 2d grid, x along out columns (M), y along in rows (K).
       one 16x16 block shape balances coalescing along x with warp
       utilisation along y for typical K values. */
    dim3 block(16, 16);
    dim3 grid((unsigned)((M + block.x - 1) / block.x),
              (unsigned)((K + block.y - 1) / block.y));
    k_im2col_sample<<<grid, block>>>(
        params->input, params->C_in, params->H, params->W,
        params->kh, params->kw,
        params->sh, params->sw,
        params->ph, params->pw,
        params->out_h, params->out_w,
        d_col);
    cudaError_t cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        cudaFree(d_col);
        ax_err_set(AX_ERR_BACKEND, "cuda conv_gemm im2col launch failed: %s",
                   cudaGetErrorString(cerr));
        return AX_ERR_BACKEND;
    }

    /* gemm: out[C_out, M] = weight[C_out, K] @ d_col[K, M].
       row-major via the cublas column-major trick: compute
       out^T = d_col^T @ weight^T in cublas, which means passing
       (M, C_out, K) as (m, n, k) with leading dims (M, K, M). */
    const float *wd = (const float *)weight->storage->data + weight->offset;
    float       *od = (float *)      out->storage->data    + out->offset;
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasSgemm(
        ax_cuda_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        (int)M, (int)C_out, (int)K,
        &alpha,
        d_col, (int)M,
        wd,    (int)K,
        &beta,
        od,    (int)M);
    cudaFree(d_col);
    if (st != CUBLAS_STATUS_SUCCESS) {
        ax_err_set(AX_ERR_BACKEND, "cuda conv_gemm cublasSgemm failed (%d)", (int)st);
        return AX_ERR_BACKEND;
    }
    return AX_OK;
}

} /* extern "C" */
