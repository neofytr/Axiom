/* bench_hotpaths.c — targeted benchmarks for RISK 1-10 optimization paths.
   reports min/median/mean/max per benchmark for stable comparison. */

#include "axiom/axiom.h"
#include "bench_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static int dcmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

typedef struct {
    double min_us, median_us, mean_us, max_us;
} stats_t;

static stats_t compute_stats(double *samples, int n) {
    qsort(samples, (size_t)n, sizeof(double), dcmp);
    double sum = 0;
    for (int i = 0; i < n; i++) sum += samples[i];
    stats_t s;
    s.min_us = samples[0];
    s.max_us = samples[n - 1];
    s.mean_us = sum / n;
    s.median_us = (n & 1) ? samples[n / 2] : (samples[n / 2 - 1] + samples[n / 2]) * 0.5;
    return s;
}

static void emit_stats(const char *name, const char *shape, stats_t s, double flops) {
    double gflops = (flops > 0 && s.median_us > 0) ? flops / (s.median_us * 1e3) : 0;
    printf("%s\t%s\t%.1f\t%.1f\t%.1f\t%.1f",
           name, shape, s.min_us, s.median_us, s.mean_us, s.max_us);
    if (gflops > 0) printf("\t%.1f GFLOPS", gflops);
    printf("\n");
    fflush(stdout);
}

static void emit_stats_bw(const char *name, const char *shape, stats_t s, double bytes) {
    double gbs = (bytes > 0 && s.median_us > 0) ? bytes / (s.median_us * 1e3) : 0;
    printf("%s\t%s\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f GB/s\n",
           name, shape, s.min_us, s.median_us, s.mean_us, s.max_us, gbs);
    fflush(stdout);
}

/* bench 1: MHA train_step variants (RISK 2,3,5,7,8,10) */
static void bench_mha_train(void) {
    int64_t B = 1, S = 512, D = 768, H = 12;
    int warm = 5, timed = 50;
    double samples[50];

    int64_t sh[3] = {B, S, D};
    ax_layer_t *mha = ax_mha_create(D, (int)H, 0, 0);
    ax_layer_train(mha);
    ax_tensor_t *x    = ax_tensor_rand(sh, 3, -0.5f, 0.5f);
    ax_tensor_t *dout = ax_tensor_rand(sh, 3, -0.5f, 0.5f);
    ax_tensor_t *y    = ax_tensor_create(sh, 3, AX_FLOAT32);

    double flops = (double)B * S * D * D * 12.0;

    for (int w = 0; w < warm; w++) ax_mha_train_step(mha, x, dout, y);
    for (int i = 0; i < timed; i++) {
        double t0 = bench_now_ms();
        ax_mha_train_step(mha, x, dout, y);
        samples[i] = (bench_now_ms() - t0) * 1000.0;
    }
    emit_stats("mha_train", "B1_S512_D768_H12", compute_stats(samples, timed), flops);

    for (int w = 0; w < warm; w++) ax_mha_train_step_fused(mha, x, dout, y);
    for (int i = 0; i < timed; i++) {
        double t0 = bench_now_ms();
        ax_mha_train_step_fused(mha, x, dout, y);
        samples[i] = (bench_now_ms() - t0) * 1000.0;
    }
    emit_stats("mha_train_fused", "B1_S512_D768_H12", compute_stats(samples, timed), flops);

    for (int w = 0; w < warm; w++) ax_mha_train_step_v4(mha, x, dout, y);
    for (int i = 0; i < timed; i++) {
        double t0 = bench_now_ms();
        ax_mha_train_step_v4(mha, x, dout, y);
        samples[i] = (bench_now_ms() - t0) * 1000.0;
    }
    emit_stats("mha_train_v4", "B1_S512_D768_H12", compute_stats(samples, timed), flops);

    ax_tensor_destroy(y); ax_tensor_destroy(dout); ax_tensor_destroy(x);
    ax_layer_destroy(mha);

    B = 4; S = 512; D = 768; H = 12;
    sh[0] = B;
    mha = ax_mha_create(D, (int)H, 0, 0);
    ax_layer_train(mha);
    x    = ax_tensor_rand(sh, 3, -0.5f, 0.5f);
    dout = ax_tensor_rand(sh, 3, -0.5f, 0.5f);
    y    = ax_tensor_create(sh, 3, AX_FLOAT32);
    flops = (double)B * S * D * D * 12.0;

    for (int w = 0; w < warm; w++) ax_mha_train_step_v4(mha, x, dout, y);
    for (int i = 0; i < timed; i++) {
        double t0 = bench_now_ms();
        ax_mha_train_step_v4(mha, x, dout, y);
        samples[i] = (bench_now_ms() - t0) * 1000.0;
    }
    emit_stats("mha_train_v4", "B4_S512_D768_H12", compute_stats(samples, timed), flops);

    ax_tensor_destroy(y); ax_tensor_destroy(dout); ax_tensor_destroy(x);
    ax_layer_destroy(mha);
}

/* bench 2: conv forward+backward (RISK 8 — JIT strided-A in conv weight grad) */
static void bench_conv_fwd_bwd(void) {
    int N = 1, Cin = 64, Cout = 128, H = 28, W = 28, K = 3;
    int warm = 5, timed = 50;
    double samples[50];

    ax_layer_t *conv = ax_conv2d_create_ex(Cin, Cout, K, K, 1, 1, 1, 1, 1);
    ax_layer_train(conv);
    int64_t in_sh[] = {N, Cin, H, W};
    ax_tensor_t *x = ax_tensor_rand(in_sh, 4, -0.5f, 0.5f);

    for (int w = 0; w < warm; w++) {
        ax_tensor_t *y = ax_layer_forward(conv, x);
        ax_backward(y);
        ax_graph_cleanup(y);
        ax_tensor_destroy(y);
    }

    int Hout = H, Wout = W;
    double flops = 2.0 * 2.0 * N * Cout * Hout * Wout * Cin * K * K;

    for (int i = 0; i < timed; i++) {
        double t0 = bench_now_ms();
        ax_tensor_t *y = ax_layer_forward(conv, x);
        ax_backward(y);
        ax_graph_cleanup(y);
        ax_tensor_destroy(y);
        samples[i] = (bench_now_ms() - t0) * 1000.0;
    }
    char shape[64];
    snprintf(shape, sizeof(shape), "N%d_C%d-%d_%dx%d_K%d", N, Cin, Cout, H, W, K);
    emit_stats("conv_fwd_bwd", shape, compute_stats(samples, timed), flops);

    ax_tensor_destroy(x); ax_layer_destroy(conv);
}

/* bench 3: batchnorm forward+backward (RISK 1 — TLS scratch) */
static void bench_bn_fwd_bwd(void) {
    int warm = 5, timed = 50;
    double samples[50];

    int64_t N = 32, C = 256, H = 14, W = 14;
    int64_t feat = C * H * W;
    int64_t sh[] = {N, C, H, W};
    ax_tensor_t *x = ax_tensor_rand(sh, 4, -1.0f, 1.0f);
    ax_layer_t *bn = ax_batchnorm_create(C, 1e-5f, 0.1f);
    ax_layer_train(bn);

    for (int w = 0; w < warm; w++) {
        ax_tensor_t *y = ax_layer_forward(bn, x);
        ax_backward(y);
        ax_graph_cleanup(y);
        ax_tensor_destroy(y);
    }

    double bytes = 2.0 * N * feat * sizeof(float);

    for (int i = 0; i < timed; i++) {
        double t0 = bench_now_ms();
        ax_tensor_t *y = ax_layer_forward(bn, x);
        ax_backward(y);
        ax_graph_cleanup(y);
        ax_tensor_destroy(y);
        samples[i] = (bench_now_ms() - t0) * 1000.0;
    }
    char shape[64];
    snprintf(shape, sizeof(shape), "N%ld_C%ld_%ldx%ld", (long)N, (long)C, (long)H, (long)W);
    emit_stats_bw("bn_fwd_bwd", shape, compute_stats(samples, timed), bytes);

    ax_tensor_destroy(x); ax_layer_destroy(bn);
}

/* bench 4: raw GEMM dispatch (RISK 8 — JIT paths) */
static void bench_gemm_raw(void) {
    typedef struct { int M, N, K; const char *tag; } shape_t;
    shape_t shapes[] = {
        {2048, 768,  768,  "dwqkv"},
        { 768, 2304, 2048, "mha_wgrad"},
        { 768, 768,  2048, "mha_tn"},
    };
    int warm = 5, timed = 50;
    double samples[50];

    for (int s = 0; s < 3; s++) {
        int M = shapes[s].M, N = shapes[s].N, K = shapes[s].K;
        double flops = 2.0 * M * N * K;
        char shape[64];

        int64_t a_sh[] = {M, K}, b_sh[] = {K, N}, c_sh[] = {M, N};
        ax_tensor_t *A = ax_tensor_rand(a_sh, 2, -0.5f, 0.5f);
        ax_tensor_t *B = ax_tensor_rand(b_sh, 2, -0.5f, 0.5f);
        ax_tensor_t *C = ax_tensor_create(c_sh, 2, AX_FLOAT32);

        for (int w = 0; w < warm; w++) ax_compute_gemm(A, B, C);
        for (int i = 0; i < timed; i++) {
            double t0 = bench_now_ms();
            ax_compute_gemm(A, B, C);
            samples[i] = (bench_now_ms() - t0) * 1000.0;
        }
        snprintf(shape, sizeof(shape), "nn_%dx%dx%d_%s", M, N, K, shapes[s].tag);
        emit_stats("gemm", shape, compute_stats(samples, timed), flops);

        ax_tensor_destroy(C);

        int64_t at_sh[] = {K, M}, c2_sh[] = {M, N};
        ax_tensor_t *At = ax_tensor_rand(at_sh, 2, -0.5f, 0.5f);
        ax_tensor_t *C2 = ax_tensor_create(c2_sh, 2, AX_FLOAT32);

        for (int w = 0; w < warm; w++) ax_compute_gemm_tn(At, B, C2);
        for (int i = 0; i < timed; i++) {
            double t0 = bench_now_ms();
            ax_compute_gemm_tn(At, B, C2);
            samples[i] = (bench_now_ms() - t0) * 1000.0;
        }
        snprintf(shape, sizeof(shape), "tn_%dx%dx%d_%s", M, N, K, shapes[s].tag);
        emit_stats("gemm", shape, compute_stats(samples, timed), flops);

        ax_tensor_destroy(At); ax_tensor_destroy(B); ax_tensor_destroy(A);
        ax_tensor_destroy(C2);
    }
}

/* bench 5: full autograd backward (3-layer MLP) */
static void bench_mlp_backward(void) {
    int warm = 5, timed = 50;
    double samples[50];

    ax_layer_t *fc1 = ax_dense_create(768, 512, 1);
    ax_layer_t *fc2 = ax_dense_create(512, 256, 1);
    ax_layer_t *fc3 = ax_dense_create(256, 10, 1);
    ax_layer_train(fc1); ax_layer_train(fc2); ax_layer_train(fc3);

    int64_t x_sh[] = {32, 768};
    ax_tensor_t *x = ax_tensor_rand(x_sh, 2, -0.5f, 0.5f);
    int64_t labels_sh[] = {32};
    ax_tensor_t *labels = ax_tensor_create(labels_sh, 1, AX_INT64);
    int64_t *ld = (int64_t *)labels->storage->data;
    for (int i = 0; i < 32; i++) ld[i] = i % 10;

    for (int w = 0; w < warm; w++) {
        ax_tensor_t *h1 = ax_layer_forward(fc1, x);
        ax_tensor_t *a1 = ax_relu(h1);
        ax_tensor_t *h2 = ax_layer_forward(fc2, a1);
        ax_tensor_t *a2 = ax_relu(h2);
        ax_tensor_t *h3 = ax_layer_forward(fc3, a2);
        ax_tensor_t *loss = ax_cross_entropy_loss(h3, labels);
        ax_backward(loss);
        ax_graph_cleanup(loss);
        ax_tensor_destroy(loss); ax_tensor_destroy(h3);
        ax_tensor_destroy(a2); ax_tensor_destroy(h2);
        ax_tensor_destroy(a1); ax_tensor_destroy(h1);
    }

    double flops = 2.0 * 32 * (768*512 + 512*256 + 256*10) * 2.0;

    for (int i = 0; i < timed; i++) {
        double t0 = bench_now_ms();
        ax_tensor_t *h1 = ax_layer_forward(fc1, x);
        ax_tensor_t *a1 = ax_relu(h1);
        ax_tensor_t *h2 = ax_layer_forward(fc2, a1);
        ax_tensor_t *a2 = ax_relu(h2);
        ax_tensor_t *h3 = ax_layer_forward(fc3, a2);
        ax_tensor_t *loss = ax_cross_entropy_loss(h3, labels);
        ax_backward(loss);
        ax_graph_cleanup(loss);
        ax_tensor_destroy(loss); ax_tensor_destroy(h3);
        ax_tensor_destroy(a2); ax_tensor_destroy(h2);
        ax_tensor_destroy(a1); ax_tensor_destroy(h1);
        samples[i] = (bench_now_ms() - t0) * 1000.0;
    }
    emit_stats("mlp_fwd_bwd", "768-512-256-10_B32", compute_stats(samples, timed), flops);

    ax_tensor_destroy(labels); ax_tensor_destroy(x);
    ax_layer_destroy(fc3); ax_layer_destroy(fc2); ax_layer_destroy(fc1);
}

int main(void) {
    ax_init();

#ifdef _OPENMP
    printf("OMP_NUM_THREADS=%d\n", omp_get_max_threads());
#else
    printf("OMP_NUM_THREADS=1 (no openmp)\n");
#endif
    printf("\nbench_name\tshape\tmin_us\tmedian_us\tmean_us\tmax_us\tthroughput\n");
    printf("------------------------------------------------------------------------\n");

    bench_mha_train();
    bench_conv_fwd_bwd();
    bench_bn_fwd_bwd();
    bench_gemm_raw();
    bench_mlp_backward();

    return 0;
}
