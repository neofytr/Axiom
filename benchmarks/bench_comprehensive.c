/* bench_comprehensive.c — exhaustive axiom benchmark for tf comparison.
   matches bench_comprehensive_tf.py shapes exactly. */

#include "axiom/axiom.h"
#include "bench_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MAX_SAMPLES 100

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

static void emit(const char *name, const char *shape, stats_t s,
                 double flops, double bw_bytes) {
    printf("%-20s %-30s %10.1f %10.1f %10.1f %10.1f",
           name, shape, s.min_us, s.median_us, s.mean_us, s.max_us);
    if (flops > 0 && s.median_us > 0)
        printf("  %8.1f GFLOPS", flops / (s.median_us * 1e3));
    else if (bw_bytes > 0 && s.median_us > 0)
        printf("  %8.1f GB/s", bw_bytes / (s.median_us * 1e3));
    printf("\n");
    fflush(stdout);
}

/* gemm shapes */

typedef struct { int M, N, K; const char *tag; int warm, timed; } gemm_shape_t;

static const gemm_shape_t GEMM_SHAPES[] = {
    {128,  128,  128,  "sq_128",     5, 60},
    {256,  256,  256,  "sq_256",     5, 60},
    {512,  512,  512,  "sq_512",     5, 50},
    {1024, 1024, 1024, "sq_1024",    5, 40},
    {2048, 2048, 2048, "sq_2048",    3, 20},
    {4096, 4096, 4096, "sq_4096",    2, 10},
    {2048, 768,  768,  "xfmr_qkv",  5, 40},
    {768,  2304, 2048, "xfmr_wgrad",5, 40},
    {768,  768,  2048, "xfmr_proj", 5, 40},
    {4096, 64,   256,  "tall_4k",   5, 50},
    {8192, 128,  64,   "tall_8k",   5, 40},
    {128,  4096, 256,  "wide",      5, 40},
    {384,  384,  384,  "non2_384",  5, 50},
    {577,  733,  512,  "non2_odd",  5, 40},
    {1,    128,  64,   "tiny_1x128",10, 80},
    {1,    256,  128,  "tiny_1x256",10, 80},
    {4,    64,   32,   "tiny_4x64", 10, 80},
    {16,   10,   64,   "tiny_cls",  10, 80},
};
#define N_GEMM (int)(sizeof(GEMM_SHAPES)/sizeof(GEMM_SHAPES[0]))

static void bench_gemm_all(void) {
    double samples[MAX_SAMPLES];

    printf("\n=== GEMM NN ===\n");
    for (int s = 0; s < N_GEMM; s++) {
        int M = GEMM_SHAPES[s].M, N = GEMM_SHAPES[s].N, K = GEMM_SHAPES[s].K;
        int warm = GEMM_SHAPES[s].warm, timed = GEMM_SHAPES[s].timed;
        double flops = 2.0 * M * N * K;

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
        char tag[80];
        snprintf(tag, sizeof(tag), "%dx%dx%d_%s", M, N, K, GEMM_SHAPES[s].tag);
        emit("gemm_nn", tag, compute_stats(samples, timed), flops, 0);

        ax_tensor_destroy(C); ax_tensor_destroy(B); ax_tensor_destroy(A);
    }

    printf("\n=== GEMM TN ===\n");
    for (int s = 0; s < N_GEMM; s++) {
        int M = GEMM_SHAPES[s].M, N = GEMM_SHAPES[s].N, K = GEMM_SHAPES[s].K;
        int warm = GEMM_SHAPES[s].warm, timed = GEMM_SHAPES[s].timed;
        double flops = 2.0 * M * N * K;

        int64_t at_sh[] = {K, M}, b_sh[] = {K, N}, c_sh[] = {M, N};
        ax_tensor_t *At = ax_tensor_rand(at_sh, 2, -0.5f, 0.5f);
        ax_tensor_t *B  = ax_tensor_rand(b_sh, 2, -0.5f, 0.5f);
        ax_tensor_t *C  = ax_tensor_create(c_sh, 2, AX_FLOAT32);

        for (int w = 0; w < warm; w++) ax_compute_gemm_tn(At, B, C);
        for (int i = 0; i < timed; i++) {
            double t0 = bench_now_ms();
            ax_compute_gemm_tn(At, B, C);
            samples[i] = (bench_now_ms() - t0) * 1000.0;
        }
        char tag[80];
        snprintf(tag, sizeof(tag), "%dx%dx%d_%s", M, N, K, GEMM_SHAPES[s].tag);
        emit("gemm_tn", tag, compute_stats(samples, timed), flops, 0);

        ax_tensor_destroy(C); ax_tensor_destroy(B); ax_tensor_destroy(At);
    }
}

/* conv shapes */

typedef struct {
    int N, Cin, Cout, H, W, K, S, P;
    const char *tag; int warm, timed;
} conv_shape_t;

static const conv_shape_t CONV_SHAPES[] = {
    {1,  64,  64,  56, 56, 3, 1, 1, "res_s1",      5, 30},
    {1,  64,  128, 56, 56, 3, 2, 1, "res_down",     5, 30},
    {1,  128, 256, 28, 28, 3, 1, 1, "res_s3",       5, 30},
    {1,  256, 512, 14, 14, 3, 2, 1, "res_s4",       5, 30},
    {1,  256, 64,  14, 14, 1, 1, 0, "bottleneck",  10, 50},
    {1,  64,  128, 28, 28, 5, 1, 2, "k5",           5, 30},
    {1,  3,   64, 224,224, 7, 2, 3, "stem_7x7",     3, 15},
    {8,  64,  128, 28, 28, 3, 1, 1, "batch8",       3, 20},
    {32, 64,  128, 28, 28, 3, 1, 1, "batch32",      2, 10},
    {1,  64,  128, 28, 28, 3, 1, 1, "baseline",     5, 30},
    {1,  3,   16,  32, 32, 3, 1, 1, "tiny_3x16",   10, 50},
    {1,  16,  32,  16, 16, 3, 1, 1, "tiny_16x32",  10, 50},
};
#define N_CONV (int)(sizeof(CONV_SHAPES)/sizeof(CONV_SHAPES[0]))

static void bench_conv_all(void) {
    double samples[MAX_SAMPLES];

    printf("\n=== CONV FWD+BWD ===\n");
    for (int s = 0; s < N_CONV; s++) {
        const conv_shape_t *sh = &CONV_SHAPES[s];
        int Hout = (sh->H + 2*sh->P - sh->K) / sh->S + 1;
        int Wout = (sh->W + 2*sh->P - sh->K) / sh->S + 1;
        double flops = 2.0 * 2.0 * sh->N * sh->Cout * Hout * Wout * sh->Cin * sh->K * sh->K;

        ax_layer_t *conv = ax_conv2d_create_ex(sh->Cin, sh->Cout,
                                                sh->K, sh->K,
                                                sh->S, sh->S,
                                                sh->P, sh->P, 1);
        ax_layer_train(conv);
        int64_t in_sh[] = {sh->N, sh->Cin, sh->H, sh->W};
        ax_tensor_t *x = ax_tensor_rand(in_sh, 4, -0.5f, 0.5f);
        x->requires_grad = true;

        for (int w = 0; w < sh->warm; w++) {
            ax_tensor_t *y = ax_layer_forward(conv, x);
            ax_tensor_t *loss = ax_sum(y, -1);
            ax_backward(loss);
            ax_graph_cleanup(loss);
            ax_tensor_destroy(loss);
            ax_tensor_destroy(y);
        }

        for (int i = 0; i < sh->timed; i++) {
            double t0 = bench_now_ms();
            ax_tensor_t *y = ax_layer_forward(conv, x);
            ax_tensor_t *loss = ax_sum(y, -1);
            ax_backward(loss);
            ax_graph_cleanup(loss);
            ax_tensor_destroy(loss);
            ax_tensor_destroy(y);
            samples[i] = (bench_now_ms() - t0) * 1000.0;
        }
        char tag[80];
        snprintf(tag, sizeof(tag), "N%d_C%d-%d_%dx%d_K%d_S%d_%s",
                 sh->N, sh->Cin, sh->Cout, sh->H, sh->W, sh->K, sh->S, sh->tag);
        emit("conv_fwd_bwd", tag, compute_stats(samples, sh->timed), flops, 0);

        ax_tensor_destroy(x);
        ax_layer_destroy(conv);
    }
}

/* batchnorm shapes */

typedef struct { int N, C, H, W; const char *tag; int warm, timed; } bn_shape_t;

static const bn_shape_t BN_SHAPES[] = {
    {1,  64,  56, 56, "b1_c64",   10, 60},
    {8,  128, 28, 28, "b8_c128",   5, 50},
    {16, 256, 14, 14, "b16_c256",  5, 50},
    {32, 256, 14, 14, "b32_c256",  5, 40},
    {64, 512,  7,  7, "b64_c512",  5, 40},
    {1,  32,  16, 16, "tiny",     10, 60},
};
#define N_BN (int)(sizeof(BN_SHAPES)/sizeof(BN_SHAPES[0]))

static void bench_bn_all(void) {
    double samples[MAX_SAMPLES];

    printf("\n=== BATCHNORM FWD+BWD ===\n");
    for (int s = 0; s < N_BN; s++) {
        const bn_shape_t *sh = &BN_SHAPES[s];
        double bytes = 2.0 * sh->N * sh->C * sh->H * sh->W * sizeof(float);

        int64_t shape[] = {sh->N, sh->C, sh->H, sh->W};
        ax_tensor_t *x = ax_tensor_rand(shape, 4, -1.0f, 1.0f);
        x->requires_grad = true;
        ax_layer_t *bn = ax_batchnorm_create(sh->C, 1e-5f, 0.1f);
        ax_layer_train(bn);

        for (int w = 0; w < sh->warm; w++) {
            ax_tensor_t *y = ax_layer_forward(bn, x);
            ax_tensor_t *loss = ax_sum(y, -1);
            ax_backward(loss);
            ax_graph_cleanup(loss);
            ax_tensor_destroy(loss);
            ax_tensor_destroy(y);
        }

        for (int i = 0; i < sh->timed; i++) {
            double t0 = bench_now_ms();
            ax_tensor_t *y = ax_layer_forward(bn, x);
            ax_tensor_t *loss = ax_sum(y, -1);
            ax_backward(loss);
            ax_graph_cleanup(loss);
            ax_tensor_destroy(loss);
            ax_tensor_destroy(y);
            samples[i] = (bench_now_ms() - t0) * 1000.0;
        }
        char tag[80];
        snprintf(tag, sizeof(tag), "N%d_C%d_%dx%d_%s",
                 sh->N, sh->C, sh->H, sh->W, sh->tag);
        emit("bn_fwd_bwd", tag, compute_stats(samples, sh->timed), 0, bytes);

        ax_tensor_destroy(x);
        ax_layer_destroy(bn);
    }
}

/* mlp benchmarks */

static void bench_mlp_case(int B, int d1, int d2, int d3, int d4,
                           const char *tag, int warm, int timed) {
    double samples[MAX_SAMPLES];

    ax_layer_t *fc1 = ax_dense_create(d1, d2, 1);
    ax_layer_t *fc2 = ax_dense_create(d2, d3, 1);
    ax_layer_t *fc3 = ax_dense_create(d3, d4, 1);
    ax_layer_train(fc1); ax_layer_train(fc2); ax_layer_train(fc3);

    int64_t x_sh[] = {B, d1};
    ax_tensor_t *x = ax_tensor_rand(x_sh, 2, -0.5f, 0.5f);
    int64_t lbl_sh[] = {B};
    ax_tensor_t *labels = ax_tensor_create(lbl_sh, 1, AX_INT64);
    int64_t *ld = (int64_t *)labels->storage->data;
    for (int i = 0; i < B; i++) ld[i] = i % d4;

    double flops = 2.0 * B * ((double)d1*d2 + (double)d2*d3 + (double)d3*d4) * 2.0;

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
    emit("mlp_fwd_bwd", tag, compute_stats(samples, timed), flops, 0);

    ax_tensor_destroy(labels); ax_tensor_destroy(x);
    ax_layer_destroy(fc3); ax_layer_destroy(fc2); ax_layer_destroy(fc1);
}

static void bench_mlp_all(void) {
    printf("\n=== MLP FWD+BWD ===\n");
    bench_mlp_case(32,  768,  512,  256,  10,   "768-512-256-10_B32",   5, 40);
    bench_mlp_case(1,   784,  256,  128,  10,   "784-256-128-10_B1",   10, 60);
    bench_mlp_case(64,  784,  512,  256,  10,   "784-512-256-10_B64",   3, 30);
    bench_mlp_case(16,  1024, 1024, 1024, 1024, "1024^4_B16",           3, 20);
    bench_mlp_case(1,   64,   32,   16,   10,   "64-32-16-10_B1",      10, 80);
}

/* mha benchmarks */

static void bench_mha_case(int B, int S, int D, int H,
                           const char *tag, int warm, int timed) {
    double samples[MAX_SAMPLES];

    int64_t sh[] = {B, S, D};
    ax_layer_t *mha = ax_mha_create(D, H, 0, 0);
    ax_layer_train(mha);
    ax_tensor_t *x    = ax_tensor_rand(sh, 3, -0.5f, 0.5f);
    ax_tensor_t *dout = ax_tensor_rand(sh, 3, -0.5f, 0.5f);
    ax_tensor_t *y    = ax_tensor_create(sh, 3, AX_FLOAT32);
    double flops = (double)B * S * D * D * 12.0;

    for (int w = 0; w < warm; w++)
        ax_mha_train_step_v4(mha, x, dout, y);

    for (int i = 0; i < timed; i++) {
        double t0 = bench_now_ms();
        ax_mha_train_step_v4(mha, x, dout, y);
        samples[i] = (bench_now_ms() - t0) * 1000.0;
    }
    emit("mha_train_v4", tag, compute_stats(samples, timed), flops, 0);

    ax_tensor_destroy(y); ax_tensor_destroy(dout); ax_tensor_destroy(x);
    ax_layer_destroy(mha);
}

static void bench_mha_all(void) {
    printf("\n=== MHA TRAIN V4 ===\n");
    bench_mha_case(1,  64,  128,  4,  "B1_S64_D128_H4",    5, 40);
    bench_mha_case(1,  128, 256,  4,  "B1_S128_D256_H4",   5, 30);
    bench_mha_case(1,  256, 512,  8,  "B1_S256_D512_H8",   5, 30);
    bench_mha_case(1,  384, 768,  12, "B1_S384_D768_H12",  3, 20);
    bench_mha_case(4,  384, 768,  12, "B4_S384_D768_H12",  2, 10);
    bench_mha_case(8,  256, 512,  8,  "B8_S256_D512_H8",   2, 10);
    bench_mha_case(1,  256, 768,  12, "B1_S256_D768_H12",  3, 20);
    bench_mha_case(16, 128, 512,  8,  "B16_S128_D512_H8",  2, 8);
}

/* softmax benchmarks */

static void bench_softmax_all(void) {
    typedef struct { int rows, cols; const char *tag; } sm_t;
    sm_t shapes[] = {
        {32,   10,    "cls_32x10"},
        {32,   1024,  "32x1024"},
        {512,  512,   "attn_512"},
        {2048, 768,   "xfmr_2048x768"},
        {1,    65536, "long_1x64k"},
        {4096, 128,   "tall_4kx128"},
    };
    int n = sizeof(shapes) / sizeof(shapes[0]);
    double samples[MAX_SAMPLES];

    printf("\n=== SOFTMAX ===\n");
    for (int s = 0; s < n; s++) {
        int64_t sh[] = {shapes[s].rows, shapes[s].cols};
        ax_tensor_t *in  = ax_tensor_rand(sh, 2, -1.0f, 1.0f);
        ax_tensor_t *out = ax_tensor_create(sh, 2, AX_FLOAT32);
        double bw = 2.0 * shapes[s].rows * shapes[s].cols * sizeof(float);

        for (int w = 0; w < 10; w++) ax_compute_softmax_rowwise(in, out);
        for (int i = 0; i < 60; i++) {
            double t0 = bench_now_ms();
            ax_compute_softmax_rowwise(in, out);
            samples[i] = (bench_now_ms() - t0) * 1000.0;
        }
        emit("softmax", shapes[s].tag, compute_stats(samples, 60), 0, bw);

        ax_tensor_destroy(out); ax_tensor_destroy(in);
    }
}

/* relu benchmarks */

static void bench_relu_all(void) {
    typedef struct { int n; const char *tag; } relu_t;
    relu_t shapes[] = {
        {1024,     "1k"},
        {65536,    "64k"},
        {1048576,  "1M"},
        {16777216, "16M"},
    };
    int n = sizeof(shapes) / sizeof(shapes[0]);
    double samples[MAX_SAMPLES];

    printf("\n=== RELU ===\n");
    for (int s = 0; s < n; s++) {
        int64_t sh[] = {shapes[s].n};
        ax_tensor_t *in  = ax_tensor_rand(sh, 1, -1.0f, 1.0f);
        ax_tensor_t *out = ax_tensor_create(sh, 1, AX_FLOAT32);
        double bw = 2.0 * shapes[s].n * sizeof(float);

        for (int w = 0; w < 10; w++) ax_compute_relu(in, out);
        for (int i = 0; i < 80; i++) {
            double t0 = bench_now_ms();
            ax_compute_relu(in, out);
            samples[i] = (bench_now_ms() - t0) * 1000.0;
        }
        emit("relu", shapes[s].tag, compute_stats(samples, 80), 0, bw);

        ax_tensor_destroy(out); ax_tensor_destroy(in);
    }
}

int main(void) {
    ax_init();

#ifdef _OPENMP
    printf("Axiom Comprehensive Benchmark — OMP_NUM_THREADS=%d\n",
           omp_get_max_threads());
#else
    printf("Axiom Comprehensive Benchmark — OMP_NUM_THREADS=1 (no openmp)\n");
#endif
    printf("\n%-20s %-30s %10s %10s %10s %10s  %s\n",
           "bench", "shape", "min_us", "med_us", "mean_us", "max_us", "throughput");

    bench_gemm_all();
    bench_conv_all();
    bench_bn_all();
    bench_mlp_all();
    bench_mha_all();
    bench_softmax_all();
    bench_relu_all();

    printf("\nDone.\n");
    return 0;
}
