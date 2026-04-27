/* bench_scalability.c — thread scaling for each hot path.
   sweeps 1..max_threads, reports TSV with efficiency metric. */

#include "axiom/axiom.h"
#include "bench_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define WARMUP   3
#define TIMED    20
#define MAX_SWEEP 6

static const int thread_counts[MAX_SWEEP] = {1, 2, 4, 8, 12, 16};

static int dcmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *s, int n) {
    qsort(s, (size_t)n, sizeof(double), dcmp);
    return (n & 1) ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) * 0.5;
}

typedef struct {
    const char *name;
    double median_us;
    double gflops;
} result_t;

static result_t results[64];
static int n_results = 0;

static void set_threads(int n) {
    ax_set_num_threads(n);
}

static void record(const char *bench, int threads, double med_us, double flops) {
    double gflops = (flops > 0 && med_us > 0) ? flops / (med_us * 1e3) : 0;

    results[n_results].name = bench;
    results[n_results].median_us = med_us;
    results[n_results].gflops = gflops;
    n_results++;

    /* efficiency = gflops_N / (N * gflops_1). threads=1 baseline is always 1.0 */
    double eff = 1.0;
    if (threads > 1) {
        for (int i = 0; i < n_results; i++) {
            if (strcmp(results[i].name, bench) == 0) {
                if (results[i].gflops > 0)
                    eff = gflops / ((double)threads * results[i].gflops);
                break;
            }
        }
    }

    printf("%s\t%d\t%.1f\t%.1f\t%.3f\n", bench, threads, med_us, gflops, eff);
    fflush(stdout);
}

/* bandwidth variant for BN (gflops column holds GB/s instead) */
static void record_bw(const char *bench, int threads, double med_us, double bytes) {
    double gbs = (bytes > 0 && med_us > 0) ? bytes / (med_us * 1e3) : 0;

    results[n_results].name = bench;
    results[n_results].median_us = med_us;
    results[n_results].gflops = gbs;
    n_results++;

    double eff = 1.0;
    if (threads > 1) {
        for (int i = 0; i < n_results; i++) {
            if (strcmp(results[i].name, bench) == 0) {
                if (results[i].gflops > 0)
                    eff = gbs / ((double)threads * results[i].gflops);
                break;
            }
        }
    }

    printf("%s\t%d\t%.1f\t%.1f\t%.3f\n", bench, threads, med_us, gbs, eff);
    fflush(stdout);
}

static int get_sweep_count(void) {
    int hw_max = 1;
#ifdef _OPENMP
    hw_max = omp_get_max_threads();
#endif
    int count = 0;
    for (int i = 0; i < MAX_SWEEP; i++) {
        if (thread_counts[i] <= hw_max)
            count++;
    }
    return count > 0 ? count : 1;
}

/* 1. GEMM NN 2048x768x768 (weight gradient shape) */
static void bench_gemm_nn_wgrad(void) {
    const char *name = "gemm_nn_2048x768x768";
    int M = 2048, N = 768, K = 768;
    double flops = 2.0 * M * N * K;

    int64_t a_sh[] = {M, K}, b_sh[] = {K, N}, c_sh[] = {M, N};
    ax_tensor_t *A = ax_tensor_rand(a_sh, 2, -0.5f, 0.5f);
    ax_tensor_t *B = ax_tensor_rand(b_sh, 2, -0.5f, 0.5f);
    ax_tensor_t *C = ax_tensor_create(c_sh, 2, AX_FLOAT32);

    int sweep = get_sweep_count();
    double samples[TIMED];

    for (int si = 0; si < sweep; si++) {
        set_threads(thread_counts[si]);
        for (int w = 0; w < WARMUP; w++) ax_compute_gemm(A, B, C);
        for (int i = 0; i < TIMED; i++) {
            double t0 = bench_now_ms();
            ax_compute_gemm(A, B, C);
            samples[i] = (bench_now_ms() - t0) * 1000.0;
        }
        record(name, thread_counts[si], median(samples, TIMED), flops);
    }

    ax_tensor_destroy(C); ax_tensor_destroy(B); ax_tensor_destroy(A);
}

/* 2. GEMM TN 2048x768x768 (transposed-A) */
static void bench_gemm_tn_wgrad(void) {
    const char *name = "gemm_tn_2048x768x768";
    int M = 2048, N = 768, K = 768;
    double flops = 2.0 * M * N * K;

    /* tn: A is [K, M], result is [M, N] */
    int64_t at_sh[] = {K, M}, b_sh[] = {K, N}, c_sh[] = {M, N};
    ax_tensor_t *At = ax_tensor_rand(at_sh, 2, -0.5f, 0.5f);
    ax_tensor_t *B  = ax_tensor_rand(b_sh, 2, -0.5f, 0.5f);
    ax_tensor_t *C  = ax_tensor_create(c_sh, 2, AX_FLOAT32);

    int sweep = get_sweep_count();
    double samples[TIMED];

    for (int si = 0; si < sweep; si++) {
        set_threads(thread_counts[si]);
        for (int w = 0; w < WARMUP; w++) ax_compute_gemm_tn(At, B, C);
        for (int i = 0; i < TIMED; i++) {
            double t0 = bench_now_ms();
            ax_compute_gemm_tn(At, B, C);
            samples[i] = (bench_now_ms() - t0) * 1000.0;
        }
        record(name, thread_counts[si], median(samples, TIMED), flops);
    }

    ax_tensor_destroy(C); ax_tensor_destroy(B); ax_tensor_destroy(At);
}

/* 3. GEMM NN 768x768x2048 (projection shape) */
static void bench_gemm_nn_proj(void) {
    const char *name = "gemm_nn_768x768x2048";
    int M = 768, N = 768, K = 2048;
    double flops = 2.0 * M * N * K;

    int64_t a_sh[] = {M, K}, b_sh[] = {K, N}, c_sh[] = {M, N};
    ax_tensor_t *A = ax_tensor_rand(a_sh, 2, -0.5f, 0.5f);
    ax_tensor_t *B = ax_tensor_rand(b_sh, 2, -0.5f, 0.5f);
    ax_tensor_t *C = ax_tensor_create(c_sh, 2, AX_FLOAT32);

    int sweep = get_sweep_count();
    double samples[TIMED];

    for (int si = 0; si < sweep; si++) {
        set_threads(thread_counts[si]);
        for (int w = 0; w < WARMUP; w++) ax_compute_gemm(A, B, C);
        for (int i = 0; i < TIMED; i++) {
            double t0 = bench_now_ms();
            ax_compute_gemm(A, B, C);
            samples[i] = (bench_now_ms() - t0) * 1000.0;
        }
        record(name, thread_counts[si], median(samples, TIMED), flops);
    }

    ax_tensor_destroy(C); ax_tensor_destroy(B); ax_tensor_destroy(A);
}

/* 4. MHA train_step_v4: B=2, S=256, D=256, H=4 */
static void bench_mha_v4(void) {
    const char *name = "mha_v4_B2_S256_D256_H4";
    int64_t B = 2, S = 256, D = 256, H = 4;
    /* rough flop estimate: projections + attention + backward */
    double flops = (double)B * (double)S * (double)D * (double)D * 12.0;

    int64_t sh[3] = {B, S, D};
    ax_layer_t *mha = ax_mha_create(D, (int)H, 0, 0);
    ax_layer_train(mha);
    ax_tensor_t *x    = ax_tensor_rand(sh, 3, -0.5f, 0.5f);
    ax_tensor_t *dout = ax_tensor_rand(sh, 3, -0.5f, 0.5f);
    ax_tensor_t *y    = ax_tensor_create(sh, 3, AX_FLOAT32);

    int sweep = get_sweep_count();
    double samples[TIMED];

    for (int si = 0; si < sweep; si++) {
        set_threads(thread_counts[si]);
        for (int w = 0; w < WARMUP; w++) ax_mha_train_step_v4(mha, x, dout, y);
        for (int i = 0; i < TIMED; i++) {
            double t0 = bench_now_ms();
            ax_mha_train_step_v4(mha, x, dout, y);
            samples[i] = (bench_now_ms() - t0) * 1000.0;
        }
        record(name, thread_counts[si], median(samples, TIMED), flops);
    }

    ax_tensor_destroy(y); ax_tensor_destroy(dout); ax_tensor_destroy(x);
    ax_layer_destroy(mha);
}

/* 5. BN fwd+bwd: N=32, C=256, H=14, W=14 */
static void bench_bn(void) {
    const char *name = "bn_fwd_bwd_N32_C256_14x14";
    int64_t N = 32, C = 256, H = 14, W = 14;
    double bytes = 2.0 * (double)N * (double)C * (double)H * (double)W * (double)sizeof(float);

    int64_t sh[] = {N, C, H, W};
    ax_tensor_t *x = ax_tensor_rand(sh, 4, -1.0f, 1.0f);
    ax_layer_t *bn = ax_batchnorm_create(C, 1e-5f, 0.1f);
    ax_layer_train(bn);

    int sweep = get_sweep_count();
    double samples[TIMED];

    for (int si = 0; si < sweep; si++) {
        set_threads(thread_counts[si]);

        for (int w = 0; w < WARMUP; w++) {
            ax_tensor_t *out = ax_layer_forward(bn, x);
            ax_backward(out);
            ax_graph_cleanup(out);
            ax_tensor_destroy(out);
        }
        for (int i = 0; i < TIMED; i++) {
            double t0 = bench_now_ms();
            ax_tensor_t *out = ax_layer_forward(bn, x);
            ax_backward(out);
            ax_graph_cleanup(out);
            ax_tensor_destroy(out);
            samples[i] = (bench_now_ms() - t0) * 1000.0;
        }
        record_bw(name, thread_counts[si], median(samples, TIMED), bytes);
    }

    ax_tensor_destroy(x); ax_layer_destroy(bn);
}

/* 6. Conv fwd+bwd: N=1, Cin=64, Cout=128, H=W=28, K=3, S=1, P=1 */
static void bench_conv(void) {
    const char *name = "conv_fwd_bwd_N1_64-128_28x28_K3";
    int N = 1, Cin = 64, Cout = 128, HH = 28, WW = 28, K = 3;
    int Hout = HH, Wout = WW;
    /* fwd + bwd flops */
    double flops = 2.0 * 2.0 * N * Cout * Hout * Wout * Cin * K * K;

    int64_t in_sh[] = {N, Cin, HH, WW};
    ax_tensor_t *x = ax_tensor_rand(in_sh, 4, -0.5f, 0.5f);
    ax_layer_t *conv = ax_conv2d_create_ex(Cin, Cout, K, K, 1, 1, 1, 1, 1);
    ax_layer_train(conv);

    int sweep = get_sweep_count();
    double samples[TIMED];

    for (int si = 0; si < sweep; si++) {
        set_threads(thread_counts[si]);

        for (int w = 0; w < WARMUP; w++) {
            ax_tensor_t *out = ax_layer_forward(conv, x);
            ax_backward(out);
            ax_graph_cleanup(out);
            ax_tensor_destroy(out);
        }
        for (int i = 0; i < TIMED; i++) {
            double t0 = bench_now_ms();
            ax_tensor_t *out = ax_layer_forward(conv, x);
            ax_backward(out);
            ax_graph_cleanup(out);
            ax_tensor_destroy(out);
            samples[i] = (bench_now_ms() - t0) * 1000.0;
        }
        record(name, thread_counts[si], median(samples, TIMED), flops);
    }

    ax_tensor_destroy(x); ax_layer_destroy(conv);
}

static void print_summary(void) {
    printf("\n== optimal thread count per benchmark ==\n");

    const char *seen[64];
    int n_seen = 0;

    for (int i = 0; i < n_results; i++) {
        int already = 0;
        for (int s = 0; s < n_seen; s++) {
            if (strcmp(seen[s], results[i].name) == 0) { already = 1; break; }
        }
        if (already) continue;

        const char *bench = results[i].name;
        seen[n_seen++] = bench;

        double best_gflops = 0;
        int best_threads = 1;
        for (int j = i; j < n_results; j++) {
            if (strcmp(results[j].name, bench) != 0) continue;
            if (results[j].gflops > best_gflops) {
                best_gflops = results[j].gflops;
                /* recover thread count from position in sweep */
                int idx = j - i;
                if (idx >= 0 && idx < MAX_SWEEP)
                    best_threads = thread_counts[idx];
            }
        }
        printf("  %-40s  best=%d threads  (%.1f GFLOPS)\n",
               bench, best_threads, best_gflops);
    }
    fflush(stdout);
}

int main(void) {
    ax_init();

    int hw_max = 1;
#ifdef _OPENMP
    hw_max = omp_get_max_threads();
#endif
    printf("# thread scalability benchmark (max hardware threads: %d)\n", hw_max);
    printf("bench\tthreads\tmedian_us\tgflops\tefficiency\n");

    bench_gemm_nn_wgrad();
    bench_gemm_tn_wgrad();
    bench_gemm_nn_proj();
    bench_mha_v4();
    bench_bn();
    bench_conv();

    print_summary();

    /* restore default thread count */
    ax_set_num_threads(0);
    return 0;
}
