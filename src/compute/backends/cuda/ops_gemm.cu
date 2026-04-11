/* ops_gemm.cu — row-major f32 matmul via cuBLAS.

   A[M,K] @ B[K,N] = C[M,N] in row-major layout. cuBLAS is column-major,
   so we use the identity (AB)^T = B^T A^T and ask cublas to compute
   C^T = B^T @ A^T using the same raw buffers interpreted column-major.
   the resulting C in column-major = C^T of the row-major desired C,
   which means the bytes on disk end up in the correct row-major order
   without any transpose work. this is the standard "trick" for using
   column-major blas on row-major data. */

#include "internal.h"

extern "C" {

ax_status_t cuda_gemm(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
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
        ax_cuda_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha, bd, N, ad, K, &beta, cd, N);
    if (st != CUBLAS_STATUS_SUCCESS) {
        ax_err_set(AX_ERR_BACKEND, "cublasSgemm failed (%d)", (int)st);
        return AX_ERR_BACKEND;
    }
    return AX_OK;
}

} /* extern "C" */
