/* ops_attention.cu — CUDA implementations of MHA layout transforms
   (head_interleave, head_deinterleave, qkv_split, qkv_merge).

   These mirror the static helpers in src/core/attention.c. they each do
   a permutation between the [B, S, H, dk] (= [B*S, D]) and [B, H, S, dk]
   (= [BH, S, dk]) layouts. one CUDA thread per element, fully coalesced.

   exposed via C linkage so attention.c can weakly-link them at the call
   site without an explicit cuda dependency in the core lib. */

#include "internal.h"

extern "C" {

/* ── head_interleave: [B, S, H, dk] → [B, H, S, dk] ────────────────
   one thread per output element. read/write both fully coalesced
   along the dk axis. */
__global__ static void k_head_interleave(
    const float *src, float *dst,
    int64_t B, int64_t S, int64_t H, int64_t dk)
{
    int64_t total = B * H * S * dk;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    /* unpack idx as (b, h, s, d) in [B, H, S, dk] order. */
    int64_t d = idx % dk;
    int64_t s = (idx / dk) % S;
    int64_t h = (idx / (dk * S)) % H;
    int64_t b = idx / (dk * S * H);
    /* source index in [B, S, H, dk]. */
    int64_t src_idx = (((b * S) + s) * H + h) * dk + d;
    dst[idx] = src[src_idx];
}

void ax_cuda_head_interleave(const float *src, float *dst,
                              int64_t B, int64_t S, int64_t H, int64_t dk)
{
    int64_t total = B * H * S * dk;
    int block = 256;
    int64_t grid = (total + block - 1) / block;
    k_head_interleave<<<(unsigned)grid, block>>>(src, dst, B, S, H, dk);
}

/* ── head_deinterleave: [B, H, S, dk] → [B, S, H, dk] ──────────── */
__global__ static void k_head_deinterleave(
    const float *src, float *dst,
    int64_t B, int64_t S, int64_t H, int64_t dk)
{
    int64_t total = B * S * H * dk;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    /* unpack idx as (b, s, h, d) in [B, S, H, dk] order. */
    int64_t d = idx % dk;
    int64_t h = (idx / dk) % H;
    int64_t s = (idx / (dk * H)) % S;
    int64_t b = idx / (dk * H * S);
    int64_t src_idx = (((b * H) + h) * S + s) * dk + d;
    dst[idx] = src[src_idx];
}

void ax_cuda_head_deinterleave(const float *src, float *dst,
                                int64_t B, int64_t S, int64_t H, int64_t dk)
{
    int64_t total = B * S * H * dk;
    int block = 256;
    int64_t grid = (total + block - 1) / block;
    k_head_deinterleave<<<(unsigned)grid, block>>>(src, dst, B, S, H, dk);
}

/* ── head_interleave_qkv_split: [rows, 3D] → Qh, Kh, Vh on [BH, S, dk] ──
   src layout: [b, s, slot, h, d] where slot is q/k/v (0/1/2), each col
   span = D = H*dk. one thread per OUTPUT element across all three heads
   produces Qh[bh, s, d], Kh[bh, s, d], Vh[bh, s, d] in parallel. */
__global__ static void k_head_interleave_qkv_split(
    const float *src,
    float *dstQ, float *dstK, float *dstV,
    int64_t B, int64_t S, int64_t H, int64_t dk, int64_t D)
{
    int64_t total = B * H * S * dk;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    /* unpack as (b, h, s, d) in [B, H, S, dk]. */
    int64_t d = idx % dk;
    int64_t s = (idx / dk) % S;
    int64_t h = (idx / (dk * S)) % H;
    int64_t b = idx / (dk * S * H);
    /* src row offset: ((b*S)+s)*3D, col h*dk + d in each slot of width D. */
    int64_t row_off = ((b * S) + s) * (3 * D);
    int64_t col_off = h * dk + d;
    dstQ[idx] = src[row_off + 0 * D + col_off];
    dstK[idx] = src[row_off + 1 * D + col_off];
    dstV[idx] = src[row_off + 2 * D + col_off];
}

void ax_cuda_head_interleave_qkv_split(const float *src,
                                         float *dstQ, float *dstK, float *dstV,
                                         int64_t B, int64_t S, int64_t H, int64_t dk,
                                         int64_t D)
{
    int64_t total = B * H * S * dk;
    int block = 256;
    int64_t grid = (total + block - 1) / block;
    k_head_interleave_qkv_split<<<(unsigned)grid, block>>>(
        src, dstQ, dstK, dstV, B, S, H, dk, D);
}

/* ── head_interleave_qkv_split_bias: split + add per-channel bias ──── */
__global__ static void k_head_interleave_qkv_split_bias(
    const float *src, const float *bias,
    float *dstQ, float *dstK, float *dstV,
    int64_t B, int64_t S, int64_t H, int64_t dk, int64_t D)
{
    int64_t total = B * H * S * dk;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int64_t d = idx % dk;
    int64_t s = (idx / dk) % S;
    int64_t h = (idx / (dk * S)) % H;
    int64_t b = idx / (dk * S * H);
    int64_t row_off = ((b * S) + s) * (3 * D);
    int64_t col_off = h * dk + d;
    dstQ[idx] = src[row_off + 0 * D + col_off] + bias[0 * D + col_off];
    dstK[idx] = src[row_off + 1 * D + col_off] + bias[1 * D + col_off];
    dstV[idx] = src[row_off + 2 * D + col_off] + bias[2 * D + col_off];
}

void ax_cuda_head_interleave_qkv_split_bias(const float *src, const float *bias,
                                              float *dstQ, float *dstK, float *dstV,
                                              int64_t B, int64_t S, int64_t H, int64_t dk,
                                              int64_t D)
{
    int64_t total = B * H * S * dk;
    int block = 256;
    int64_t grid = (total + block - 1) / block;
    k_head_interleave_qkv_split_bias<<<(unsigned)grid, block>>>(
        src, bias, dstQ, dstK, dstV, B, S, H, dk, D);
}

/* ── head_deinterleave_qkv_merge: srcQ, srcK, srcV [BH,S,dk] → dst[rows, 3D] ─ */
__global__ static void k_head_deinterleave_qkv_merge(
    const float *srcQ, const float *srcK, const float *srcV,
    float *dst,
    int64_t B, int64_t S, int64_t H, int64_t dk, int64_t D)
{
    int64_t total = B * H * S * dk;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int64_t d = idx % dk;
    int64_t s = (idx / dk) % S;
    int64_t h = (idx / (dk * S)) % H;
    int64_t b = idx / (dk * S * H);
    int64_t row_off = ((b * S) + s) * (3 * D);
    int64_t col_off = h * dk + d;
    dst[row_off + 0 * D + col_off] = srcQ[idx];
    dst[row_off + 1 * D + col_off] = srcK[idx];
    dst[row_off + 2 * D + col_off] = srcV[idx];
}

void ax_cuda_head_deinterleave_qkv_merge(const float *srcQ, const float *srcK, const float *srcV,
                                          float *dst,
                                          int64_t B, int64_t S, int64_t H, int64_t dk,
                                          int64_t D)
{
    int64_t total = B * H * S * dk;
    int block = 256;
    int64_t grid = (total + block - 1) / block;
    k_head_deinterleave_qkv_merge<<<(unsigned)grid, block>>>(
        srcQ, srcK, srcV, dst, B, S, H, dk, D);
}

/* ============================================================
   QKV weight cache build (Phase B helper)
   ============================================================
   builds the fused [D, 3D] Wqkv weight cache from three separate
   [D, D] Wq/Wk/Wv weights on device. used by attention.c when MHA
   tensors live on CUDA — host memcpy would dereference device pointers.
   uses cudaMemcpy2D with a per-row stride so it's a single launch each. */
ax_status_t ax_cuda_qkv_cache_build(float *dst,
                                     const float *wq, const float *wk, const float *wv,
                                     int64_t D)
{
    size_t row_bytes = (size_t)D * sizeof(float);
    size_t dst_pitch = (size_t)(3 * D) * sizeof(float);
    cudaError_t e;
    e = cudaMemcpy2D(dst,             dst_pitch, wq, row_bytes,
                     row_bytes, (size_t)D, cudaMemcpyDeviceToDevice);
    if (e != cudaSuccess) return AX_ERR_BACKEND;
    e = cudaMemcpy2D(dst + D,         dst_pitch, wk, row_bytes,
                     row_bytes, (size_t)D, cudaMemcpyDeviceToDevice);
    if (e != cudaSuccess) return AX_ERR_BACKEND;
    e = cudaMemcpy2D(dst + 2 * D,     dst_pitch, wv, row_bytes,
                     row_bytes, (size_t)D, cudaMemcpyDeviceToDevice);
    if (e != cudaSuccess) return AX_ERR_BACKEND;
    return AX_OK;
}

/* same idea for biases: dst[3D] = bq[D] | bk[D] | bv[D]. */
ax_status_t ax_cuda_bias_cache_build(float *dst,
                                      const float *bq, const float *bk, const float *bv,
                                      int64_t D)
{
    size_t b = (size_t)D * sizeof(float);
    cudaError_t e;
    e = cudaMemcpy(dst,          bq, b, cudaMemcpyDeviceToDevice);
    if (e != cudaSuccess) return AX_ERR_BACKEND;
    e = cudaMemcpy(dst + D,      bk, b, cudaMemcpyDeviceToDevice);
    if (e != cudaSuccess) return AX_ERR_BACKEND;
    e = cudaMemcpy(dst + 2 * D,  bv, b, cudaMemcpyDeviceToDevice);
    if (e != cudaSuccess) return AX_ERR_BACKEND;
    return AX_OK;
}

/* ============================================================
   SDPA forward — Phase C
   ============================================================
   computes O = softmax(scale * Q @ K^T) @ V per head, in batched form.
   Q/K/V/O all [BH, S, dk] device pointers.
   uses cublasSgemmStridedBatched for the two matmuls and a custom
   softmax+L kernel that fuses max, exp_sum, normalize, log-sum-exp
   in one pass per row. */

/* causal mask: write -inf where j > i (per head). one block per (bh, i),
   threads cover j in [0, S). */
__global__ static void k_causal_mask(float *score, int64_t S)
{
    int64_t bh_i = blockIdx.x;
    int64_t i = bh_i % S;
    int64_t bh_off = (bh_i / S) * S * S + i * S;
    for (int64_t j = threadIdx.x; j < S; j += blockDim.x) {
        if (j > i) score[bh_off + j] = -INFINITY;
    }
}

/* fused softmax + log-sum-exp per row.
   - input:  score[BH*S, S] (modified in place to P)
   - output: P (overwritten into score), L[BH*S] (=max + log(sum exp(score-max)))
   one block per row. threads use shuffle reductions for max + sum. */
__global__ static void k_softmax_lse_row(float *score, float *L, int64_t S)
{
    int64_t row = blockIdx.x;
    float *r = score + row * S;
    /* pass 1: max */
    float my_max = -INFINITY;
    for (int64_t j = threadIdx.x; j < S; j += blockDim.x) {
        float v = r[j];
        if (v > my_max) my_max = v;
    }
    __shared__ float s_max[32];
    /* warp-level reduce */
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, my_max, o);
        if (ov > my_max) my_max = ov;
    }
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) s_max[wid] = my_max;
    __syncthreads();
    if (wid == 0) {
        my_max = (threadIdx.x < (blockDim.x + 31) / 32) ? s_max[lane] : -INFINITY;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, my_max, o);
            if (ov > my_max) my_max = ov;
        }
        if (lane == 0) s_max[0] = my_max;
    }
    __syncthreads();
    float row_max = s_max[0];
    /* pass 2: exp(x - max) and sum */
    float my_sum = 0.0f;
    for (int64_t j = threadIdx.x; j < S; j += blockDim.x) {
        float e = __expf(r[j] - row_max);
        r[j] = e;
        my_sum += e;
    }
    __shared__ float s_sum[32];
    for (int o = 16; o > 0; o >>= 1)
        my_sum += __shfl_xor_sync(0xffffffff, my_sum, o);
    if (lane == 0) s_sum[wid] = my_sum;
    __syncthreads();
    if (wid == 0) {
        my_sum = (threadIdx.x < (blockDim.x + 31) / 32) ? s_sum[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1)
            my_sum += __shfl_xor_sync(0xffffffff, my_sum, o);
        if (lane == 0) s_sum[0] = my_sum;
    }
    __syncthreads();
    float row_sum = s_sum[0];
    float inv = 1.0f / row_sum;
    /* pass 3: normalize */
    for (int64_t j = threadIdx.x; j < S; j += blockDim.x) {
        r[j] *= inv;
    }
    /* L = max + log(sum_exp) */
    if (threadIdx.x == 0 && L) {
        L[row] = row_max + __logf(row_sum);
    }
}

ax_status_t ax_cuda_sdpa_fwd(const float *Q, const float *K, const float *V,
                              float *O, float *L,
                              int64_t BH, int64_t S, int64_t dk, float scale,
                              bool causal, float *P_save)
{
    cublasHandle_t h = ax_cuda_cublas_handle();
    /* persistent score buffer, reused across calls. sized BH*S*S floats. */
    static float *s_score = NULL;
    static size_t s_score_cap = 0;
    size_t want = (size_t)BH * (size_t)S * (size_t)S * sizeof(float);
    if (want > s_score_cap) {
        if (s_score) cudaFree(s_score);
        s_score = NULL;
        s_score_cap = 0;
        if (cudaMalloc(&s_score, want) != cudaSuccess) {
            ax_err_set(AX_ERR_ALLOC, "cuda_sdpa_fwd: cudaMalloc score failed");
            return AX_ERR_ALLOC;
        }
        s_score_cap = want;
    }
    float *score = s_score;

    /* score = scale * Q @ K^T per head. row-major [S, dk] @ [S, dk]^T = [S, S].
       cuBLAS column-major equivalent (using the standard trick): treat as
       cublasSgemm(handle, OP_T, OP_N, S, S, dk, alpha, K, dk, Q, dk, 0, score, S)
       see cuda_gemm_nt for the same pattern. batched with stride per-head. */
    float alpha = scale, beta = 0.0f;
    cublasStatus_t st = cublasSgemmStridedBatched(
        h, CUBLAS_OP_T, CUBLAS_OP_N,
        (int)S, (int)S, (int)dk,
        &alpha,
        K, (int)dk, S * dk,
        Q, (int)dk, S * dk,
        &beta,
        score, (int)S, S * S,
        (int)BH);
    if (st != CUBLAS_STATUS_SUCCESS) {
        ax_err_set(AX_ERR_BACKEND, "cuda_sdpa_fwd score gemm failed (%d)", (int)st);
        return AX_ERR_BACKEND;
    }

    if (causal) {
        k_causal_mask<<<(unsigned)(BH * S), 256>>>(score, S);
        cudaError_t cerr = cudaGetLastError();
        if (cerr != cudaSuccess) {
            ax_err_set(AX_ERR_BACKEND, "cuda_sdpa_fwd causal mask launch failed: %s",
                       cudaGetErrorString(cerr));
            return AX_ERR_BACKEND;
        }
    }

    /* fused softmax + L computation. one block per row [BH*S]. use
       AX_CUDA_BLOCK threads if S is large; cap at 1024. */
    int sm_block = (S < 1024) ? (int)S : 1024;
    if (sm_block < 32) sm_block = 32;
    /* pad to multiple of 32 for warp reductions. */
    sm_block = ((sm_block + 31) / 32) * 32;
    if (sm_block > 1024) sm_block = 1024;
    k_softmax_lse_row<<<(unsigned)(BH * S), sm_block>>>(score, L, S);
    cudaError_t cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        ax_err_set(AX_ERR_BACKEND, "cuda_sdpa_fwd softmax launch failed: %s",
                   cudaGetErrorString(cerr));
        return AX_ERR_BACKEND;
    }

    /* if caller wants P saved (for the backward fast path), copy now. */
    if (P_save) {
        if (cudaMemcpyAsync(P_save, score, want, cudaMemcpyDeviceToDevice, 0)
            != cudaSuccess) {
            ax_err_set(AX_ERR_BACKEND, "cuda_sdpa_fwd P_save copy failed");
            return AX_ERR_BACKEND;
        }
    }

    /* O = P @ V per head. row-major [S, S] @ [S, dk] = [S, dk].
       cuBLAS column-major: cublasSgemm(handle, OP_N, OP_N, dk, S, S,
                                          alpha, V, dk, P, S, 0, O, dk). */
    alpha = 1.0f; beta = 0.0f;
    st = cublasSgemmStridedBatched(
        h, CUBLAS_OP_N, CUBLAS_OP_N,
        (int)dk, (int)S, (int)S,
        &alpha,
        V,     (int)dk, S * dk,
        score, (int)S,  S * S,
        &beta,
        O,     (int)dk, S * dk,
        (int)BH);
    if (st != CUBLAS_STATUS_SUCCESS) {
        ax_err_set(AX_ERR_BACKEND, "cuda_sdpa_fwd PV gemm failed (%d)", (int)st);
        return AX_ERR_BACKEND;
    }
    return AX_OK;
}

/* ============================================================
   SDPA backward — Phase D
   ============================================================
   computes dQ, dK, dV from Q, K, V, O, dO, L. uses cuBLAS
   stridedBatched for the four matmuls and small kernels for
   the per-row Di and the elementwise dS = P * (dP - Di) * scale. */

/* Di[i] = dot(dO[i], O[i]) for each row i in [BH*S]. */
__global__ static void k_di_rows(const float *dO, const float *O, float *Di,
                                  int64_t S, int64_t dk)
{
    int64_t row = blockIdx.x;
    const float *do_r = dO + row * dk;
    const float *o_r  = O  + row * dk;
    float my_sum = 0.0f;
    for (int64_t d = threadIdx.x; d < dk; d += blockDim.x)
        my_sum += do_r[d] * o_r[d];
    /* warp reduce */
    for (int o = 16; o > 0; o >>= 1)
        my_sum += __shfl_xor_sync(0xffffffff, my_sum, o);
    __shared__ float s[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) s[wid] = my_sum;
    __syncthreads();
    if (wid == 0) {
        my_sum = (threadIdx.x < (blockDim.x + 31) / 32) ? s[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1)
            my_sum += __shfl_xor_sync(0xffffffff, my_sum, o);
        if (lane == 0) Di[row] = my_sum;
    }
}

/* dS = P * (dP - Di) * scale, written in place over dP.
   one block per row [BH*S], threads cover S columns. */
__global__ static void k_dS_inplace(float *dP, const float *P, const float *Di,
                                     float scale, int64_t S)
{
    int64_t row = blockIdx.x;
    float di = Di[row];
    float *dp = dP + row * S;
    const float *p = P + row * S;
    for (int64_t j = threadIdx.x; j < S; j += blockDim.x) {
        dp[j] = p[j] * (dp[j] - di) * scale;
    }
}

/* recompute P = softmax(scale * Q @ K^T - mask). same flow as
   ax_cuda_sdpa_fwd but P_save into the supplied scratch; L not needed. */
ax_status_t ax_cuda_sdpa_bwd(const float *Q, const float *K, const float *V,
                              const float *O, const float *dO, const float *L,
                              float *dQ, float *dK, float *dV,
                              int64_t BH, int64_t S, int64_t dk, float scale,
                              bool causal, const float *P_saved)
{
    cublasHandle_t h = ax_cuda_cublas_handle();
    /* persistent scratch buffers (P, dP, Di) — sized BH*S*S + BH*S*S + BH*S */
    static float *s_P = NULL,  *s_dP = NULL,  *s_Di = NULL;
    static size_t s_P_cap = 0, s_dP_cap = 0, s_Di_cap = 0;
    size_t pdims = (size_t)BH * (size_t)S * (size_t)S * sizeof(float);
    size_t didims = (size_t)BH * (size_t)S * sizeof(float);

    auto grow = [](float **p, size_t *cap, size_t want) -> ax_status_t {
        if (want > *cap) {
            if (*p) cudaFree(*p);
            *p = NULL; *cap = 0;
            if (cudaMalloc(p, want) != cudaSuccess) return AX_ERR_ALLOC;
            *cap = want;
        }
        return AX_OK;
    };
    if (grow(&s_P,  &s_P_cap,  pdims)  != AX_OK) return AX_ERR_ALLOC;
    if (grow(&s_dP, &s_dP_cap, pdims)  != AX_OK) return AX_ERR_ALLOC;
    if (grow(&s_Di, &s_Di_cap, didims) != AX_OK) return AX_ERR_ALLOC;
    float *Pbuf  = s_P;
    float *dPbuf = s_dP;
    float *Di    = s_Di;

    cublasStatus_t st;
    float alpha, beta;
    /* zero dQ/dK/dV (caller may pass accumulators in MHA-bwd path; matches the
       CPU-side memset in ax_cpu_sdpa_bwd). */
    cudaMemsetAsync(dQ, 0, (size_t)BH * S * dk * sizeof(float), 0);
    cudaMemsetAsync(dK, 0, (size_t)BH * S * dk * sizeof(float), 0);
    cudaMemsetAsync(dV, 0, (size_t)BH * S * dk * sizeof(float), 0);

    /* Step A: P = softmax. either reuse P_saved or recompute. */
    if (P_saved) {
        cudaMemcpyAsync(Pbuf, P_saved, pdims, cudaMemcpyDeviceToDevice, 0);
    } else {
        /* score = scale * Q @ K^T (per batch). */
        alpha = scale; beta = 0.0f;
        st = cublasSgemmStridedBatched(
            h, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)S, (int)S, (int)dk,
            &alpha,
            K, (int)dk, S * dk,
            Q, (int)dk, S * dk,
            &beta,
            Pbuf, (int)S, S * S,
            (int)BH);
        if (st != CUBLAS_STATUS_SUCCESS) {
            ax_err_set(AX_ERR_BACKEND, "sdpa_bwd score gemm failed (%d)", (int)st);
            return AX_ERR_BACKEND;
        }
        if (causal) {
            k_causal_mask<<<(unsigned)(BH * S), 256>>>(Pbuf, S);
        }
        /* softmax in-place (no L needed for backward). */
        int sm_block = (S < 1024) ? (int)S : 1024;
        if (sm_block < 32) sm_block = 32;
        sm_block = ((sm_block + 31) / 32) * 32;
        if (sm_block > 1024) sm_block = 1024;
        k_softmax_lse_row<<<(unsigned)(BH * S), sm_block>>>(Pbuf, NULL, S);
    }
    (void)L; /* L unused in this implementation (we recompute softmax). */

    /* Step B: Di[row] = dot(dO[row], O[row]). */
    int dk_block = (dk < 1024) ? (int)dk : 1024;
    dk_block = ((dk_block + 31) / 32) * 32;
    if (dk_block < 32) dk_block = 32;
    if (dk_block > 1024) dk_block = 1024;
    k_di_rows<<<(unsigned)(BH * S), dk_block>>>(dO, O, Di, S, dk);

    /* Step C: dV = P^T @ dO per batch. row-major [S, S]^T @ [S, dk] = [S, dk].
       cublas col-major: cublasSgemm(N, T, dk, S, S, alpha, dO, dk, P, S, 0, dV, dk) */
    alpha = 1.0f; beta = 0.0f;
    st = cublasSgemmStridedBatched(
        h, CUBLAS_OP_N, CUBLAS_OP_T,
        (int)dk, (int)S, (int)S,
        &alpha,
        dO,   (int)dk, S * dk,
        Pbuf, (int)S,  S * S,
        &beta,
        dV,   (int)dk, S * dk,
        (int)BH);
    if (st != CUBLAS_STATUS_SUCCESS) {
        ax_err_set(AX_ERR_BACKEND, "sdpa_bwd dV gemm failed (%d)", (int)st);
        return AX_ERR_BACKEND;
    }

    /* Step D: dP = dO @ V^T per batch. row-major [S, dk] @ [S, dk]^T = [S, S].
       same shape as score gemm. cublas: cublasSgemm(T, N, S, S, dk, alpha, V, dk, dO, dk, 0, dP, S) */
    st = cublasSgemmStridedBatched(
        h, CUBLAS_OP_T, CUBLAS_OP_N,
        (int)S, (int)S, (int)dk,
        &alpha,
        V,     (int)dk, S * dk,
        dO,    (int)dk, S * dk,
        &beta,
        dPbuf, (int)S,  S * S,
        (int)BH);
    if (st != CUBLAS_STATUS_SUCCESS) {
        ax_err_set(AX_ERR_BACKEND, "sdpa_bwd dP gemm failed (%d)", (int)st);
        return AX_ERR_BACKEND;
    }

    /* Step E: dS = P * (dP - Di) * scale, in place on dPbuf. */
    int ds_block = (S < 1024) ? (int)S : 1024;
    ds_block = ((ds_block + 31) / 32) * 32;
    if (ds_block < 32) ds_block = 32;
    if (ds_block > 1024) ds_block = 1024;
    k_dS_inplace<<<(unsigned)(BH * S), ds_block>>>(dPbuf, Pbuf, Di, scale, S);

    /* Step F: dQ = dS @ K per batch. row-major [S, S] @ [S, dk] = [S, dk].
       cublas: cublasSgemm(N, N, dk, S, S, alpha, K, dk, dS, S, 0, dQ, dk) */
    st = cublasSgemmStridedBatched(
        h, CUBLAS_OP_N, CUBLAS_OP_N,
        (int)dk, (int)S, (int)S,
        &alpha,
        K,     (int)dk, S * dk,
        dPbuf, (int)S,  S * S,
        &beta,
        dQ,    (int)dk, S * dk,
        (int)BH);
    if (st != CUBLAS_STATUS_SUCCESS) {
        ax_err_set(AX_ERR_BACKEND, "sdpa_bwd dQ gemm failed (%d)", (int)st);
        return AX_ERR_BACKEND;
    }

    /* Step G: dK = dS^T @ Q per batch. row-major [S, S]^T @ [S, dk] = [S, dk].
       cublas: cublasSgemm(N, T, dk, S, S, alpha, Q, dk, dS, S, 0, dK, dk) */
    st = cublasSgemmStridedBatched(
        h, CUBLAS_OP_N, CUBLAS_OP_T,
        (int)dk, (int)S, (int)S,
        &alpha,
        Q,     (int)dk, S * dk,
        dPbuf, (int)S,  S * S,
        &beta,
        dK,    (int)dk, S * dk,
        (int)BH);
    if (st != CUBLAS_STATUS_SUCCESS) {
        ax_err_set(AX_ERR_BACKEND, "sdpa_bwd dK gemm failed (%d)", (int)st);
        return AX_ERR_BACKEND;
    }
    return AX_OK;
}

} /* extern "C" */
