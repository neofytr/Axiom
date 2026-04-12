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

/* fused-scaling gemm: out = alpha * (a @ b) + beta * out.
   cublas natively supports alpha/beta — this is a one-line change
   from cuda_gemm, just threading the user's alpha/beta into the
   sgemm call instead of hardcoding 1.0/0.0. */
ax_status_t cuda_gemm_ex(const ax_tensor_t *a, const ax_tensor_t *b,
                          float alpha, float beta, ax_tensor_t *out) {
    if (a->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cuda gemm_ex only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    if (a->ndim != 2 || b->ndim != 2 || out->ndim != 2) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "cuda gemm_ex requires 2d tensors");
        return AX_ERR_SHAPE_MISMATCH;
    }
    int M = (int)a->shape[0], K = (int)a->shape[1], N = (int)b->shape[1];
    const float *ad = (const float *)a->storage->data   + a->offset;
    const float *bd = (const float *)b->storage->data   + b->offset;
    float       *cd = (float *)      out->storage->data + out->offset;
    cublasStatus_t st = cublasSgemm(
        ax_cuda_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha, bd, N, ad, K, &beta, cd, N);
    if (st != CUBLAS_STATUS_SUCCESS) {
        ax_err_set(AX_ERR_BACKEND, "cublasSgemm (ex) failed (%d)", (int)st);
        return AX_ERR_BACKEND;
    }
    return AX_OK;
}

/* out = a @ b^T. a is [M,K] row-major, b is [N,K] row-major.
   the row-major transpose trick maps to cuBLAS as:
   C_col[N,M] = B_col^T[N,K] @ A_col[K,M]
   where B_col is the [K,N] col-major view of b (ldb=K) and A_col
   is the [K,M] col-major view of a (lda=K). cuBLAS transposes
   B_col via CUBLAS_OP_T and reads A_col directly. */
ax_status_t cuda_gemm_nt(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    if (a->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cuda gemm_nt only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    if (a->ndim != 2 || b->ndim != 2 || out->ndim != 2)
        return AX_ERR_SHAPE_MISMATCH;
    int M = (int)a->shape[0], K = (int)a->shape[1];
    int N = (int)b->shape[0]; /* b is [N, K] */
    if ((int)b->shape[1] != K || (int)out->shape[0] != M || (int)out->shape[1] != N)
        return AX_ERR_SHAPE_MISMATCH;
    const float *ad = (const float *)a->storage->data   + a->offset;
    const float *bd = (const float *)b->storage->data   + b->offset;
    float       *cd = (float *)      out->storage->data + out->offset;
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasSgemm(
        ax_cuda_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K, &alpha, bd, K, ad, K, &beta, cd, N);
    if (st != CUBLAS_STATUS_SUCCESS) {
        ax_err_set(AX_ERR_BACKEND, "cublasSgemm (nt) failed (%d)", (int)st);
        return AX_ERR_BACKEND;
    }
    return AX_OK;
}

/* out = a^T @ b. a is [K,M] row-major, b is [K,N] row-major.
   C_col[N,M] = B_col[N,K] @ A_col^T[K,M]
   B_col is [N,K] col-major view (ldb=N, CUBLAS_OP_N gives [N,K]).
   A_col is [M,K] col-major view (lda=M). cuBLAS transposes it via
   CUBLAS_OP_T to get [K,M]. */
ax_status_t cuda_gemm_tn(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    if (a->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cuda gemm_tn only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    if (a->ndim != 2 || b->ndim != 2 || out->ndim != 2)
        return AX_ERR_SHAPE_MISMATCH;
    int K = (int)a->shape[0], M = (int)a->shape[1];
    int N = (int)b->shape[1];
    if ((int)b->shape[0] != K || (int)out->shape[0] != M || (int)out->shape[1] != N)
        return AX_ERR_SHAPE_MISMATCH;
    const float *ad = (const float *)a->storage->data   + a->offset;
    const float *bd = (const float *)b->storage->data   + b->offset;
    float       *cd = (float *)      out->storage->data + out->offset;
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasSgemm(
        ax_cuda_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
        N, M, K, &alpha, bd, N, ad, M, &beta, cd, N);
    if (st != CUBLAS_STATUS_SUCCESS) {
        ax_err_set(AX_ERR_BACKEND, "cublasSgemm (tn) failed (%d)", (int)st);
        return AX_ERR_BACKEND;
    }
    return AX_OK;
}

} /* extern "C" */
