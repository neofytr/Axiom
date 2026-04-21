/* bench_qgemm.c — compare W8A32 quantized GEMM vs fp32 GEMM throughput. */

#include "axiom/axiom.h"
#include "axiom/quantize.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void bench_one(int64_t M, int64_t N, int64_t K) {
    /* fp32 inputs */
    float *a    = aligned_alloc(64, (size_t)M * K * sizeof(float));
    float *fp32 = aligned_alloc(64, (size_t)N * K * sizeof(float));
    /* deterministic fill */
    uint32_t s = 1234;
    for (int64_t i = 0; i < M*K; i++) { s ^= s<<13; s ^= s>>17; s ^= s<<5; a[i]    = (float)(s & 0xFFFF) / 32768.0f - 1.0f; }
    for (int64_t i = 0; i < N*K; i++) { s ^= s<<13; s ^= s>>17; s ^= s<<5; fp32[i] = (float)(s & 0xFFFF) / 32768.0f - 1.0f; }

    /* fp32 GEMM via ax_compute_gemm: A[M,K] @ B[K,N] = C[M,N].
       fp32 GEMM expects B in [K,N] layout, but our quantized weights are
       in [N,K]. for a fair comparison, transpose fp32 to [K,N] for the fp32 path. */
    float *fp32_T = aligned_alloc(64, (size_t)K * N * sizeof(float));
    for (int64_t n = 0; n < N; n++)
        for (int64_t k = 0; k < K; k++)
            fp32_T[k * N + n] = fp32[n * K + k];

    float *out_fp = aligned_alloc(64, (size_t)M * N * sizeof(float));
    float *out_q  = aligned_alloc(64, (size_t)M * N * sizeof(float));
    memset(out_fp, 0, (size_t)M * N * sizeof(float));
    memset(out_q,  0, (size_t)M * N * sizeof(float));

    int64_t a_sh[2] = {M, K}, b_sh[2] = {K, N}, c_sh[2] = {M, N};
    ax_tensor_t *t_a = ax_tensor_from_array(a, a_sh, 2, AX_FLOAT32);
    ax_tensor_t *t_b = ax_tensor_from_array(fp32_T, b_sh, 2, AX_FLOAT32);
    ax_tensor_t *t_c = ax_tensor_from_array(out_fp, c_sh, 2, AX_FLOAT32);

    ax_qweight_t *qw = ax_qweight_create_from_fp32(fp32, N, K);

    /* warm */
    for (int i = 0; i < 3; i++) {
        ax_compute_gemm(t_a, t_b, t_c);
        ax_qgemm_w8a32(a, qw, out_q, M);
    }
    int reps = 10;
    double t0 = now_ms();
    for (int i = 0; i < reps; i++) ax_compute_gemm(t_a, t_b, t_c);
    double t_fp = (now_ms() - t0) / reps;

    t0 = now_ms();
    for (int i = 0; i < reps; i++) ax_qgemm_w8a32(a, qw, out_q, M);
    double t_q = (now_ms() - t0) / reps;

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double g_fp = flops / t_fp / 1e6;
    double g_q  = flops / t_q  / 1e6;
    printf("M=%4ld N=%4ld K=%4ld   fp32: %7.2f ms  %6.1f GFLOPS    q8a32: %7.2f ms  %6.1f GFLOPS    speedup: %.2fx\n",
           (long)M, (long)N, (long)K, t_fp, g_fp, t_q, g_q, t_fp / t_q);

    ax_tensor_destroy(t_a);
    ax_tensor_destroy(t_b);
    ax_tensor_destroy(t_c);
    ax_qweight_destroy(qw);
    free(a); free(fp32); free(fp32_T); free(out_fp); free(out_q);
}

int main(void) {
    ax_init();
    printf("bench_qgemm: W8A32 vs fp32 GEMM\n");
    /* typical Linear shapes from inference */
    bench_one(1,    768,  3072);   /* GPT MLP up */
    bench_one(1,    3072, 768);    /* GPT MLP down */
    bench_one(8,    768,  768);    /* QKV proj at small batch */
    bench_one(32,   768,  768);    /* QKV proj at medium batch */
    bench_one(128,  4096, 1024);   /* MNIST fc */
    bench_one(64,   1024, 4096);   /* projection */
    bench_one(1,    4096, 4096);   /* token-by-token decode */
    ax_shutdown();
    return 0;
}
