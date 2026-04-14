/* bench_gemm_suite.c — exhaustive GEMM benchmark across shape regimes.

   covers the shapes that dominate real workloads:
     - square (baseline)
     - skinny-M (batched projection, inference pass)
     - skinny-N (classifier head)
     - tall-K (long-context QK attention)
     - MLP sizes (256 x {784, 4096, 2048, 1024})
   plus the two transposed variants: gemm_nt (a @ b^T) and gemm_tn (a^T @ b).

   emits RESULT lines for summarize.py. */

#include "axiom/axiom.h"
#include "bench_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int M, N, K; } gemm_shape_t;

/* choose iters so each case runs ~100ms minimum */
static int iters_for(int M, int N, int K) {
    double flops = 2.0 * M * N * K;
    /* aim at ~0.3s per case; assume ~100 GFLOPS as floor */
    int it = (int)(3e10 / flops);
    if (it < 3) it = 3;
    if (it > 500) it = 500;
    return it;
}

enum { GEMM_NN = 0, GEMM_NT = 1, GEMM_TN = 2 };

static void run_gemm_case(const char *suite, int mode, int M, int N, int K) {
    int64_t a_sh[2], b_sh[2], c_sh[2] = {M, N};
    if (mode == GEMM_NN)      { a_sh[0] = M; a_sh[1] = K; b_sh[0] = K; b_sh[1] = N; }
    else if (mode == GEMM_NT) { a_sh[0] = M; a_sh[1] = K; b_sh[0] = N; b_sh[1] = K; }
    else                       { a_sh[0] = K; a_sh[1] = M; b_sh[0] = K; b_sh[1] = N; }

    ax_tensor_t *A = ax_tensor_rand(a_sh, 2, -0.5f, 0.5f);
    ax_tensor_t *B = ax_tensor_rand(b_sh, 2, -0.5f, 0.5f);
    ax_tensor_t *C = ax_tensor_create(c_sh, 2, AX_FLOAT32);
    if (!A || !B || !C) { fprintf(stderr, "alloc fail\n"); return; }

    int it = iters_for(M, N, K);
    int warm = 2;

    double lat_ms;
    if (mode == GEMM_NN) {
        for (int w = 0; w < warm; w++) ax_compute_gemm(A, B, C);
        double t0 = bench_now_ms();
        for (int i = 0; i < it; i++) ax_compute_gemm(A, B, C);
        lat_ms = (bench_now_ms() - t0) / it;
    } else if (mode == GEMM_NT) {
        for (int w = 0; w < warm; w++) ax_compute_gemm_nt(A, B, C);
        double t0 = bench_now_ms();
        for (int i = 0; i < it; i++) ax_compute_gemm_nt(A, B, C);
        lat_ms = (bench_now_ms() - t0) / it;
    } else {
        for (int w = 0; w < warm; w++) ax_compute_gemm_tn(A, B, C);
        double t0 = bench_now_ms();
        for (int i = 0; i < it; i++) ax_compute_gemm_tn(A, B, C);
        lat_ms = (bench_now_ms() - t0) / it;
    }

    char cs[64];
    const char *tag = (mode == GEMM_NN) ? "nn" : (mode == GEMM_NT ? "nt" : "tn");
    snprintf(cs, sizeof(cs), "%s_%dx%dx%d", tag, M, N, K);
    BENCH_EMIT_FLOPS(suite, cs, lat_ms, 2.0 * M * N * K);

    ax_tensor_destroy(A); ax_tensor_destroy(B); ax_tensor_destroy(C);
}

int main(void) {
    ax_init();
    const char *suite = "gemm";

    /* square sweep — fits different cache regimes */
    gemm_shape_t squares[] = {
        {  64,   64,   64}, /* L1 */
        { 128,  128,  128}, /* L1 */
        { 256,  256,  256}, /* L1 / L2 boundary */
        { 512,  512,  512}, /* L2 */
        {1024, 1024, 1024}, /* L2 / L3 */
        {2048, 2048, 2048}, /* L3 */
        {4096, 4096, 4096}, /* L3 saturated */
    };
    for (size_t i = 0; i < sizeof(squares)/sizeof(squares[0]); i++) {
        run_gemm_case(suite, GEMM_NN, squares[i].M, squares[i].N, squares[i].K);
    }

    /* skinny-M (batched projection, M small) */
    int skinny_m[][3] = {
        {  32, 1024, 1024}, /* e.g. batch=32 encoder FFN */
        {  64, 2048, 2048},
        { 128, 4096, 4096},
        { 256, 4096, 1024},
    };
    for (size_t i = 0; i < sizeof(skinny_m)/sizeof(skinny_m[0]); i++) {
        run_gemm_case(suite, GEMM_NN, skinny_m[i][0], skinny_m[i][1], skinny_m[i][2]);
    }

    /* skinny-N (classifier head, N tiny) */
    int skinny_n[][3] = {
        {1024, 10,   1024}, /* classifier */
        {2048, 32,   2048},
        {4096, 128,  4096},
    };
    for (size_t i = 0; i < sizeof(skinny_n)/sizeof(skinny_n[0]); i++) {
        run_gemm_case(suite, GEMM_NN, skinny_n[i][0], skinny_n[i][1], skinny_n[i][2]);
    }

    /* tall-K (attention QK, K=seq_len large) */
    int tall_k[][3] = {
        {128,  128, 2048}, /* attention score */
        {256,  256, 4096},
    };
    for (size_t i = 0; i < sizeof(tall_k)/sizeof(tall_k[0]); i++) {
        run_gemm_case(suite, GEMM_NN, tall_k[i][0], tall_k[i][1], tall_k[i][2]);
    }

    /* MLP layer shapes (256 batch) — mirrors wide MLP benchmark */
    int mlp[][3] = {
        {256,  4096,  784},
        {256,  2048, 4096},
        {256,  1024, 2048},
        {256,   512, 1024},
        {256,   256,  512},
    };
    for (size_t i = 0; i < sizeof(mlp)/sizeof(mlp[0]); i++) {
        run_gemm_case(suite, GEMM_NN, mlp[i][0], mlp[i][1], mlp[i][2]);
    }

    /* transposed variants at a few representative sizes */
    int trans[][3] = {
        {1024, 1024, 1024},
        {512,  2048, 512},   /* dW = x^T @ dY for dense layer */
        {2048, 512,  512},
    };
    for (size_t i = 0; i < sizeof(trans)/sizeof(trans[0]); i++) {
        run_gemm_case(suite, GEMM_NT, trans[i][0], trans[i][1], trans[i][2]);
        run_gemm_case(suite, GEMM_TN, trans[i][0], trans[i][1], trans[i][2]);
    }

    ax_shutdown();
    return 0;
}
