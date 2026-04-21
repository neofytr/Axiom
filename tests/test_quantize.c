/* test_quantize.c — verify int8 quantization correctness against fp32 GEMM. */

#include "axiom/axiom.h"
#include "axiom/quantize.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int rc = 0;

static void check_close(const char *name, float got, float want, float rel_tol) {
    float denom = fabsf(want) + 1e-6f;
    float err = fabsf(got - want) / denom;
    if (err > rel_tol) {
        fprintf(stderr, "FAIL %s: got %.6f want %.6f (rel err %.4f > tol %.4f)\n",
                name, (double)got, (double)want, (double)err, (double)rel_tol);
        rc = 1;
    }
}

/* fill buffer with deterministic random fp32 in [-1, 1] */
static void fill_rand(float *buf, int64_t n, uint32_t seed) {
    uint32_t s = seed;
    for (int64_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        buf[i] = ((float)(s & 0xFFFF) / 32768.0f) - 1.0f;
    }
}

static void test_quantize_dequantize(int64_t N, int64_t K) {
    float *fp32 = aligned_alloc(64, (size_t)N * K * sizeof(float));
    fill_rand(fp32, N * K, 42);

    ax_qweight_t *qw = ax_qweight_create_from_fp32(fp32, N, K);
    if (!qw) { fprintf(stderr, "FAIL quantize alloc N=%ld K=%ld\n", (long)N, (long)K); rc = 1; free(fp32); return; }

    float *deq = aligned_alloc(64, (size_t)N * K * sizeof(float));
    ax_qweight_dequantize(qw, deq);

    /* check that dequantized values are within scale/127 of the originals
       (the inherent quantization error for symmetric int8). */
    int bad = 0;
    for (int64_t n = 0; n < N; n++) {
        float scale = qw->scales[n];
        for (int64_t k = 0; k < K; k++) {
            float orig = fp32[n * K + k];
            float got = deq[n * K + k];
            float tol = scale * 1.5f;  /* up to 1.5 quant steps */
            if (fabsf(orig - got) > tol) bad++;
        }
    }
    if (bad > 0) {
        fprintf(stderr, "FAIL quantize-dequantize N=%ld K=%ld: %d outliers\n", (long)N, (long)K, bad);
        rc = 1;
    } else {
        printf("  quantize-dequantize N=%ld K=%ld: OK\n", (long)N, (long)K);
    }

    ax_qweight_destroy(qw);
    free(fp32);
    free(deq);
}

static void test_qgemm_w8a32(int64_t M, int64_t N, int64_t K) {
    /* compare W8A32 GEMM against (a @ dequant(qw)^T) using the regular fp32 GEMM */
    float *a    = aligned_alloc(64, (size_t)M * K * sizeof(float));
    float *fp32 = aligned_alloc(64, (size_t)N * K * sizeof(float));
    fill_rand(a,    M * K, 7);
    fill_rand(fp32, N * K, 13);

    ax_qweight_t *qw = ax_qweight_create_from_fp32(fp32, N, K);
    if (!qw) { fprintf(stderr, "FAIL alloc\n"); rc = 1; free(a); free(fp32); return; }

    /* dequantize for ground truth */
    float *deq = aligned_alloc(64, (size_t)N * K * sizeof(float));
    ax_qweight_dequantize(qw, deq);

    /* compute qgemm result */
    float *out_q = aligned_alloc(64, (size_t)M * N * sizeof(float));
    memset(out_q, 0, (size_t)M * N * sizeof(float));
    if (ax_qgemm_w8a32(a, qw, out_q, M) != AX_OK) {
        fprintf(stderr, "FAIL qgemm call\n"); rc = 1;
    }

    /* reference: out_ref[m, n] = sum_k a[m, k] * deq[n, k] */
    float *out_ref = aligned_alloc(64, (size_t)M * N * sizeof(float));
    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            float s = 0;
            for (int64_t k = 0; k < K; k++) s += a[m * K + k] * deq[n * K + k];
            out_ref[m * N + n] = s;
        }
    }

    int bad = 0;
    float max_err = 0;
    /* mixed abs+rel tolerance: pass if either |err| < abs_tol OR rel < rel_tol.
       fp32 GEMM with K=1024 has up to ~1e-4 inherent rounding noise; SIMD
       reordering pushes it to ~1e-3 in cancellation cases. */
    for (int64_t i = 0; i < M * N; i++) {
        float diff = fabsf(out_q[i] - out_ref[i]);
        float rel  = diff / (fabsf(out_ref[i]) + 1e-5f);
        if (rel > max_err) max_err = rel;
        if (diff > 1e-2f && rel > 1e-3f) bad++;
    }
    if (bad > 0) {
        fprintf(stderr, "FAIL qgemm M=%ld N=%ld K=%ld: %d/%ld off (max rel err %.6f)\n",
                (long)M, (long)N, (long)K, bad, (long)(M*N), (double)max_err);
        rc = 1;
    } else {
        printf("  qgemm M=%ld N=%ld K=%ld: OK (max rel err %.2e)\n", (long)M, (long)N, (long)K, (double)max_err);
    }

    ax_qweight_destroy(qw);
    free(a); free(fp32); free(deq); free(out_q); free(out_ref);
    (void)check_close;
}

int main(void) {
    ax_init();
    printf("test_quantize\n");
    test_quantize_dequantize(8, 8);
    test_quantize_dequantize(128, 128);
    test_quantize_dequantize(64, 1024);

    test_qgemm_w8a32(1, 8, 8);
    test_qgemm_w8a32(4, 16, 32);
    test_qgemm_w8a32(8, 64, 128);
    test_qgemm_w8a32(32, 128, 512);
    test_qgemm_w8a32(16, 256, 1024);

    if (rc == 0) printf("all tests passed\n");
    else fprintf(stderr, "FAIL\n");
    ax_shutdown();
    return rc;
}
