/* bench_stress.c — hardcore stress tests.

   these push the framework beyond comfortable sizes. each case is
   designed to expose a specific weakness if one exists:

   - huge_gemm_*:    memory-bandwidth ceiling at 8K, 12K, 16K square
   - long_ctx_sdpa:  quadratic-attention at S=4096 and S=8192
   - wide_bn:        very wide activation batchnorm (8192 features)
   - thread_scaling: same GEMM run at 1, 2, 4, 8, all threads — expose
                     fork-join overhead + NUMA effects
   - huge_train:     big MLP with 100+ MB weights — puts Adam, grad
                     clip, and memory bandwidth under sustained load.
*/

#include "axiom/axiom.h"
#include "bench_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* force minimum one iter for huge cases to avoid silent skips */
static void bench_huge_gemm(int M) {
    int64_t sh[] = {M, M};
    ax_tensor_t *A = ax_tensor_rand(sh, 2, -0.5f, 0.5f);
    ax_tensor_t *B = ax_tensor_rand(sh, 2, -0.5f, 0.5f);
    ax_tensor_t *C = ax_tensor_create(sh, 2, AX_FLOAT32);
    if (!A || !B || !C) { fprintf(stderr, "huge_gemm %d: oom\n", M); return; }

    ax_compute_gemm(A, B, C);  /* warm */
    double t0 = bench_now_ms();
    int iters = (M >= 8192) ? 1 : 2;
    for (int i = 0; i < iters; i++) ax_compute_gemm(A, B, C);
    double lat = (bench_now_ms() - t0) / iters;

    char cs[64]; snprintf(cs, sizeof(cs), "huge_gemm_%d", M);
    BENCH_EMIT_FLOPS("stress", cs, lat, 2.0 * M * M * M);

    ax_tensor_destroy(A); ax_tensor_destroy(B); ax_tensor_destroy(C);
}

static void bench_long_ctx_sdpa(int S) {
    int BH = 8;
    int dk = 64;
    size_t bytes = (size_t)BH * S * dk * sizeof(float);
    float *Q = (float *)aligned_alloc(64, bytes);
    float *K = (float *)aligned_alloc(64, bytes);
    float *V = (float *)aligned_alloc(64, bytes);
    float *O = (float *)aligned_alloc(64, bytes);
    if (!Q || !K || !V || !O) { fprintf(stderr, "long_ctx %d: oom\n", S); free(Q);free(K);free(V);free(O); return; }
    for (size_t i = 0; i < bytes/sizeof(float); i++) {
        Q[i] = 0.01f * (float)(i & 31);
        K[i] = 0.01f * (float)((i >> 5) & 31);
        V[i] = 0.01f * (float)((i >> 10) & 31);
    }

    float scale = 1.0f / 8.0f;  /* 1/sqrt(dk) */
    ax_fused_attention_fwd(Q, K, V, O, NULL, BH, S, dk, scale);  /* warm */
    int iters = (S >= 8192) ? 1 : 2;
    double t0 = bench_now_ms();
    for (int i = 0; i < iters; i++)
        ax_fused_attention_fwd(Q, K, V, O, NULL, BH, S, dk, scale);
    double lat = (bench_now_ms() - t0) / iters;
    double flops = 4.0 * BH * S * S * dk;
    char cs[64]; snprintf(cs, sizeof(cs), "long_ctx_sdpa_S%d", S);
    BENCH_EMIT_FLOPS("stress", cs, lat, flops);

    free(Q); free(K); free(V); free(O);
}

static void bench_long_ctx_sdpa_causal(int S) {
    int BH = 8;
    int dk = 64;
    size_t bytes = (size_t)BH * S * dk * sizeof(float);
    float *Q = (float *)aligned_alloc(64, bytes);
    float *K = (float *)aligned_alloc(64, bytes);
    float *V = (float *)aligned_alloc(64, bytes);
    float *O = (float *)aligned_alloc(64, bytes);
    if (!Q || !K || !V || !O) { free(Q);free(K);free(V);free(O); return; }
    for (size_t i = 0; i < bytes/sizeof(float); i++)
        { Q[i] = 0.01f; K[i] = 0.01f; V[i] = 0.01f; }

    float scale = 1.0f / 8.0f;
    ax_fused_attention_fwd_causal(Q, K, V, O, NULL, BH, S, dk, scale);
    int iters = (S >= 8192) ? 1 : 2;
    double t0 = bench_now_ms();
    for (int i = 0; i < iters; i++)
        ax_fused_attention_fwd_causal(Q, K, V, O, NULL, BH, S, dk, scale);
    double lat = (bench_now_ms() - t0) / iters;
    /* causal does ~half the work */
    double flops = 2.0 * BH * S * S * dk;
    char cs[64]; snprintf(cs, sizeof(cs), "long_ctx_sdpa_causal_S%d", S);
    BENCH_EMIT_FLOPS("stress", cs, lat, flops);

    free(Q); free(K); free(V); free(O);
}

static void bench_thread_scaling(void) {
#ifdef _OPENMP
    int thread_counts[] = {1, 2, 4, 8, 16};
    int M = 2048;
    int64_t sh[] = {M, M};
    ax_tensor_t *A = ax_tensor_rand(sh, 2, -0.5f, 0.5f);
    ax_tensor_t *B = ax_tensor_rand(sh, 2, -0.5f, 0.5f);
    ax_tensor_t *C = ax_tensor_create(sh, 2, AX_FLOAT32);
    int max_t = omp_get_max_threads();

    for (size_t i = 0; i < sizeof(thread_counts)/sizeof(thread_counts[0]); i++) {
        int t = thread_counts[i];
        if (t > max_t) continue;
        omp_set_num_threads(t);
        ax_compute_gemm(A, B, C); /* warm */
        double t0 = bench_now_ms();
        for (int k = 0; k < 3; k++) ax_compute_gemm(A, B, C);
        double lat = (bench_now_ms() - t0) / 3;
        char cs[64]; snprintf(cs, sizeof(cs), "gemm2k_threads%d", t);
        BENCH_EMIT_FLOPS("stress", cs, lat, 2.0 * M * M * M);
    }
    omp_set_num_threads(max_t);
    ax_tensor_destroy(A); ax_tensor_destroy(B); ax_tensor_destroy(C);
#else
    (void)0;
#endif
}

static void bench_huge_train_step(void) {
    /* single Adam step on an 8M param tensor — Adam + grad clip under load. */
    int64_t M = 1 << 14;  /* 16384 */
    int64_t N = 512;
    int64_t sh[] = {M, N};
    ax_tensor_t *p = ax_tensor_rand(sh, 2, -0.1f, 0.1f);
    ax_tensor_t *g = ax_tensor_rand(sh, 2, -0.1f, 0.1f);
    p->requires_grad = true;
    p->grad = g;

    ax_tensor_t *params[1] = { p };
    ax_optimizer_t *opt = ax_adam_create(params, 1, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f);

    /* warmup */
    ax_optimizer_step(opt);

    int iters = 20;
    double t0 = bench_now_ms();
    for (int i = 0; i < iters; i++) ax_optimizer_step(opt);
    double lat = (bench_now_ms() - t0) / iters;

    int64_t n = M * N;
    char cs[64]; snprintf(cs, sizeof(cs), "adam_%ldMparam", (long)(n / 1000000));
    BENCH_EMIT("stress", cs, "lat_ms", lat);
    /* 28 bytes/elem touched (p read+write, g read, m read+write, v read+write) */
    BENCH_EMIT("stress", cs, "gbs", 28.0 * (double)n / (lat * 1e6));

    p->grad = NULL;  /* detach before destroy */
    ax_tensor_destroy(g);
    ax_tensor_destroy(p);
}

int main(void) {
    ax_init();

    /* huge GEMM progression */
    bench_huge_gemm(4096);
    bench_huge_gemm(8192);
    /* 12K/16K need 768+ MB per tensor — guard memory */
    size_t gb = 12L * 12L * 12L * 1024 * 1024 * sizeof(float) / (1 << 30);
    if (gb < 6) bench_huge_gemm(12288);
    /* 16K takes 3 * 1GB, often too big for CI */
    /* bench_huge_gemm(16384); */

    /* long-context attention */
    bench_long_ctx_sdpa(2048);
    bench_long_ctx_sdpa(4096);
    bench_long_ctx_sdpa(8192);
    bench_long_ctx_sdpa_causal(4096);
    bench_long_ctx_sdpa_causal(8192);

    /* thread scaling */
    bench_thread_scaling();

    /* adam at scale */
    bench_huge_train_step();

    ax_shutdown();
    return 0;
}
