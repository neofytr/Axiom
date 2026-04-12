/* bench_backend.c — compares naive vs optimized backend.
   tests two things:
     1. correctness: same outputs from both backends (within fp tolerance)
     2. speed: wall-clock time for gemm and a real training loop

   run this after any backend change to confirm nothing regressed. */

#include "axiom/axiom.h"
#include "axiom/compute.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* timing                                                               */
/* ------------------------------------------------------------------ */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ */
/* gemm correctness + throughput                                        */
/* ------------------------------------------------------------------ */

static double gemm_gflops(int64_t m, int64_t k, int64_t n, int reps, double secs)
{
    /* each gemm: 2*m*k*n flops (multiply-add pairs) */
    double flops = 2.0 * (double)m * (double)k * (double)n * (double)reps;
    return flops / secs / 1e9;
}

static void bench_gemm(void)
{
    printf("=== GEMM benchmark ===\n");

    /* sizes representative of dense layer forward pass in a typical network */
    static const struct { int64_t m, k, n; const char *label; } sizes[] = {
        { 32,  128, 128, "batch=32  128x128" },
        { 32,  512, 512, "batch=32  512x512" },
        { 128, 512, 512, "batch=128 512x512" },
        { 900, 64,  64,  "batch=900  64x64 (spiral)" },
    };
    int n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));

    for (int s = 0; s < n_sizes; s++)
    {
        int64_t m = sizes[s].m, k = sizes[s].k, n = sizes[s].n;
        int64_t sha[] = {m, k}, shb[] = {k, n}, shc[] = {m, n};

        ax_tensor_t *a   = ax_tensor_create(sha, 2, AX_FLOAT32);
        ax_tensor_t *b   = ax_tensor_create(shb, 2, AX_FLOAT32);
        ax_tensor_t *out_naive = ax_tensor_create(shc, 2, AX_FLOAT32);
        ax_tensor_t *out_opt   = ax_tensor_create(shc, 2, AX_FLOAT32);
        if (!a || !b || !out_naive || !out_opt) { fprintf(stderr, "alloc failed\n"); return; }

        /* fill with deterministic values */
        float *ad = (float *)a->storage->data;
        float *bd = (float *)b->storage->data;
        for (int64_t i = 0; i < m * k; i++) ad[i] = (float)(i % 17) * 0.01f - 0.08f;
        for (int64_t i = 0; i < k * n; i++) bd[i] = (float)(i % 13) * 0.01f - 0.06f;

        /* warmup */
        ax_compute_set_backend(AX_BACKEND_CPU_NAIVE);
        ax_compute_gemm(a, b, out_naive);
        ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
        ax_compute_gemm(a, b, out_opt);

        /* correctness check */
        float *nd = (float *)out_naive->storage->data;
        float *od = (float *)out_opt->storage->data;
        float max_diff = 0.0f;
        for (int64_t i = 0; i < m * n; i++)
        {
            float d = fabsf(nd[i] - od[i]);
            if (d > max_diff) max_diff = d;
        }

        /* timed runs */
        const int reps = (m * k * n > 1000000) ? 20 : 200;

        ax_compute_set_backend(AX_BACKEND_CPU_NAIVE);
        double t0 = now_sec();
        for (int r = 0; r < reps; r++) ax_compute_gemm(a, b, out_naive);
        double naive_sec = now_sec() - t0;

        ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
        t0 = now_sec();
        for (int r = 0; r < reps; r++) ax_compute_gemm(a, b, out_opt);
        double opt_sec = now_sec() - t0;

        double naive_gf = gemm_gflops(m, k, n, reps, naive_sec);
        double opt_gf   = gemm_gflops(m, k, n, reps, opt_sec);

        printf("  %-22s  naive=%5.1f GFLOPS  opt=%5.1f GFLOPS  speedup=%.2fx  max_diff=%.2e\n",
               sizes[s].label, naive_gf, opt_gf, opt_gf / naive_gf, (double)max_diff);

        if (max_diff > 1e-3f)
            printf("  *** WARNING: large output difference — possible correctness bug ***\n");

        ax_tensor_destroy(a);
        ax_tensor_destroy(b);
        ax_tensor_destroy(out_naive);
        ax_tensor_destroy(out_opt);
    }
    printf("\n");
}

/* ------------------------------------------------------------------ */
/* spiral training benchmark                                            */
/* ------------------------------------------------------------------ */

#define N_CLASSES      3
#define N_TRAIN_PER    300
#define N_TRAIN        (N_CLASSES * N_TRAIN_PER)
#define N_FEATURES     2
#define NOISE          0.05f

static float  train_xy[N_TRAIN * N_FEATURES];
static uint8_t train_lbl[N_TRAIN];

static void generate_spiral(void)
{
    int idx = 0;
    /* generate without noise for a pure benchmark (no ax_rng dependency) */
    for (int k = 0; k < N_CLASSES; k++)
    {
        for (int i = 0; i < N_TRAIN_PER; i++)
        {
            float t     = (float)i / (float)(N_TRAIN_PER - 1);
            float angle = t * 4.0f * (float)M_PI
                        + (float)k * (2.0f * (float)M_PI / (float)N_CLASSES);
            float r     = 0.05f + t * 0.95f;
            train_xy[idx * 2 + 0] = r * cosf(angle);
            train_xy[idx * 2 + 1] = r * sinf(angle);
            train_lbl[idx] = (uint8_t)k;
            idx++;
        }
    }
}

static ax_tensor_t *make_train_x(void)
{
    int64_t shape[] = {N_TRAIN, N_FEATURES};
    ax_tensor_t *t = ax_tensor_create(shape, 2, AX_FLOAT32);
    if (!t) return NULL;
    memcpy(t->storage->data, train_xy, sizeof(train_xy));
    return t;
}

static ax_tensor_t *make_train_y(void)
{
    int64_t shape[] = {N_TRAIN, N_CLASSES};
    ax_tensor_t *t = ax_tensor_zeros(shape, 2, AX_FLOAT32);
    if (!t) return NULL;
    float *d = (float *)t->storage->data;
    for (int i = 0; i < N_TRAIN; i++)
        d[i * N_CLASSES + train_lbl[i]] = 1.0f;
    return t;
}

typedef struct { double elapsed_sec; float final_loss; } train_result_t;

static train_result_t run_training(ax_backend_id_t backend, int epochs)
{
    train_result_t res = {0};

    ax_compute_set_backend(backend);

    /* fixed seed so both backends start from identical weights */
    ax_set_seed(42);

    ax_tensor_t *train_x = make_train_x();
    ax_tensor_t *train_y = make_train_y();
    if (!train_x || !train_y)
    {
        fprintf(stderr, "data alloc failed\n");
        ax_tensor_destroy(train_x);
        ax_tensor_destroy(train_y);
        return res;
    }

    ax_layer_t *model = ax_sequential_create();
    ax_sequential_add(model, ax_dense_create(N_FEATURES, 64, true));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_dense_create(64, 64, true));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_dense_create(64, N_CLASSES, true));

    ax_tensor_t *params[32];
    int n_params = ax_layer_get_params(model, params, 32);
    ax_optimizer_t *opt = ax_adamw_create(params, n_params,
                                           1e-2f, 0.9f, 0.999f, 1e-8f, 1e-3f);

    double t0 = now_sec();
    float last_loss = 0.0f;

    for (int ep = 0; ep < epochs; ep++)
    {
        ax_layer_train(model);
        ax_enable_grad();

        ax_tensor_t *logits = ax_layer_forward(model, train_x);
        ax_tensor_t *loss   = ax_cross_entropy_loss(logits, train_y);
        last_loss = ((float *)loss->storage->data)[0];

        ax_optimizer_zero_grad(opt);
        ax_backward(loss);
        ax_optimizer_step(opt);
        ax_graph_cleanup(loss);
        ax_tensor_destroy(loss);
    }

    res.elapsed_sec = now_sec() - t0;
    res.final_loss  = last_loss;

    ax_optimizer_destroy(opt);
    ax_layer_destroy(model);
    ax_tensor_destroy(train_x);
    ax_tensor_destroy(train_y);
    return res;
}

static void bench_training(void)
{
    printf("=== Training benchmark: spiral, 2->64(relu)->64(relu)->3, 200 epochs ===\n");

    generate_spiral();
    const int epochs = 200;

    train_result_t naive = run_training(AX_BACKEND_CPU_NAIVE, epochs);
    train_result_t opt   = run_training(AX_BACKEND_CPU_SIMD,  epochs);

    printf("  naive:   %.3f sec  final_loss=%.4f\n", naive.elapsed_sec, naive.final_loss);
    printf("  opt:     %.3f sec  final_loss=%.4f\n", opt.elapsed_sec,   opt.final_loss);
    printf("  speedup: %.2fx\n", naive.elapsed_sec / opt.elapsed_sec);

    float loss_diff = fabsf(naive.final_loss - opt.final_loss);
    if (loss_diff > 0.05f)
        printf("  *** WARNING: loss difference %.4f exceeds tolerance — check correctness ***\n", loss_diff);
    else
        printf("  correctness: loss difference %.4f (within tolerance)\n", loss_diff);

    printf("\n");

    /* restore opt backend as default */
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
}

/* ------------------------------------------------------------------ */

int main(void)
{
    ax_init();

    printf("Axiom backend benchmark\n");
    printf("backend: naive=%s  opt=%s\n\n",
           "cpu_naive", "cpu_opt (AVX2+FMA)");

    bench_gemm();
    bench_training();

    ax_shutdown();
    return 0;
}
