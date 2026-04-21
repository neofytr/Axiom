/* quantize.c — int8 weight quantization + W8A32 GEMM. */

#include "axiom/quantize.h"
#include "axiom/memory.h"
#include "../compute/backends/simd_defs.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

/* ===== quantization ================================================== */

ax_qweight_t *ax_qweight_create_from_fp32(const float *fp32_w, int64_t N, int64_t K) {
    if (!fp32_w || N <= 0 || K <= 0) return NULL;
    ax_qweight_t *qw = (ax_qweight_t *)calloc(1, sizeof(*qw));
    if (!qw) return NULL;
    qw->data   = (int8_t *)ax_aligned_alloc((size_t)N * (size_t)K, 64);
    qw->scales = (float  *)ax_aligned_alloc((size_t)N * sizeof(float), 64);
    if (!qw->data || !qw->scales) {
        ax_aligned_free(qw->data);
        ax_aligned_free(qw->scales);
        free(qw);
        return NULL;
    }
    qw->N = N;
    qw->K = K;

    /* per-row symmetric quantization */
    for (int64_t n = 0; n < N; n++) {
        const float *row = fp32_w + n * K;
        float absmax = 0.0f;
        for (int64_t k = 0; k < K; k++) {
            float a = fabsf(row[k]);
            if (a > absmax) absmax = a;
        }
        /* scale: maps [-127, 127] int8 range to [-absmax, +absmax] fp32 */
        float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
        float inv_scale = 1.0f / scale;
        qw->scales[n] = scale;
        for (int64_t k = 0; k < K; k++) {
            float q = row[k] * inv_scale;
            int v = (int)lrintf(q);
            if (v >  127) v =  127;
            if (v < -127) v = -127;
            qw->data[n * K + k] = (int8_t)v;
        }
    }
    return qw;
}

void ax_qweight_destroy(ax_qweight_t *qw) {
    if (!qw) return;
    ax_aligned_free(qw->data);
    ax_aligned_free(qw->scales);
    free(qw);
}

void ax_qweight_dequantize(const ax_qweight_t *qw, float *out_fp32) {
    if (!qw || !out_fp32) return;
    for (int64_t n = 0; n < qw->N; n++) {
        float s = qw->scales[n];
        const int8_t *row = qw->data + n * qw->K;
        float *or_ = out_fp32 + n * qw->K;
        for (int64_t k = 0; k < qw->K; k++) {
            or_[k] = (float)row[k] * s;
        }
    }
}

/* ===== W8A32 GEMM ====================================================
   inner product: int32 acc = sum_k (int8 w[n, k]) * (fp32 a[m, k])
   careful: mixing int8 and fp32 in the same FMA isn't direct on x86. we
   convert the int8 chunk to fp32 inline and FMA in fp32. that's still a
   bandwidth win (load 1 byte vs 4 bytes per weight) but the FMA throughput
   matches plain fp32.

   scalar reference impl + AVX2 SIMD path (8 columns at a time).
   parallelism: omp parallel over n (output channels). */

#if defined(AX_SIMD_AVX2) && AX_VF32_WIDTH == 8
/* AVX2: load 8 int8 weights, sign-extend to int32, convert to fp32. */
static inline __m256 ax_load8_int8_to_fp32(const int8_t *p) {
    /* load 8 bytes into low 64 bits of an xmm */
    __m128i v8 = _mm_loadl_epi64((const __m128i *)p);
    /* sign-extend 8 int8 → 8 int32 */
    __m256i v32 = _mm256_cvtepi8_epi32(v8);
    /* int32 → fp32 */
    return _mm256_cvtepi32_ps(v32);
}
#endif

ax_status_t ax_qgemm_w8a32(
    const float *a,
    const ax_qweight_t *qw,
    float *out,
    int64_t M)
{
    if (!a || !qw || !out) {
        ax_err_set(AX_ERR_NULL_ARG, "qgemm_w8a32: NULL");
        return AX_ERR_NULL_ARG;
    }
    int64_t N = qw->N, K = qw->K;

    /* parallelize over output channels (n). each (m, n) is independent;
       the standard parallelism for narrow-batch inference is over n. */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t n = 0; n < N; n++) {
        const int8_t *w_n = qw->data + n * K;
        float scale_n = qw->scales[n];

        for (int64_t m = 0; m < M; m++) {
            const float *a_m = a + m * K;
            float acc_scalar = 0.0f;

#if defined(AX_SIMD_AVX2) && AX_VF32_WIDTH == 8
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            int64_t k = 0;
            int64_t k_end_32 = K - (K % 32);
            /* process 32 floats per loop iter (4 × 8) for ILP */
            for (; k < k_end_32; k += 32) {
                __m256 a0 = _mm256_loadu_ps(a_m + k +  0);
                __m256 a1 = _mm256_loadu_ps(a_m + k +  8);
                __m256 a2 = _mm256_loadu_ps(a_m + k + 16);
                __m256 a3 = _mm256_loadu_ps(a_m + k + 24);
                __m256 w0 = ax_load8_int8_to_fp32(w_n + k +  0);
                __m256 w1 = ax_load8_int8_to_fp32(w_n + k +  8);
                __m256 w2 = ax_load8_int8_to_fp32(w_n + k + 16);
                __m256 w3 = ax_load8_int8_to_fp32(w_n + k + 24);
                acc0 = _mm256_fmadd_ps(a0, w0, acc0);
                acc1 = _mm256_fmadd_ps(a1, w1, acc1);
                acc2 = _mm256_fmadd_ps(a2, w2, acc2);
                acc3 = _mm256_fmadd_ps(a3, w3, acc3);
            }
            /* tail: 8 at a time */
            int64_t k_end_8 = K - (K % 8);
            for (; k < k_end_8; k += 8) {
                __m256 a0 = _mm256_loadu_ps(a_m + k);
                __m256 w0 = ax_load8_int8_to_fp32(w_n + k);
                acc0 = _mm256_fmadd_ps(a0, w0, acc0);
            }
            __m256 acc01 = _mm256_add_ps(acc0, acc1);
            __m256 acc23 = _mm256_add_ps(acc2, acc3);
            __m256 sum_v = _mm256_add_ps(acc01, acc23);
            /* horizontal sum */
            __m128 lo = _mm256_castps256_ps128(sum_v);
            __m128 hi = _mm256_extractf128_ps(sum_v, 1);
            lo = _mm_add_ps(lo, hi);
            lo = _mm_hadd_ps(lo, lo);
            lo = _mm_hadd_ps(lo, lo);
            acc_scalar = _mm_cvtss_f32(lo);
            /* scalar tail */
            for (; k < K; k++) {
                acc_scalar += a_m[k] * (float)w_n[k];
            }
#else
            for (int64_t k = 0; k < K; k++) {
                acc_scalar += a_m[k] * (float)w_n[k];
            }
#endif
            out[m * N + n] = acc_scalar * scale_n;
        }
    }
    return AX_OK;
}
