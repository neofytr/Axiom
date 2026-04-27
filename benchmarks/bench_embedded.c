/* bench_embedded.c — performance on shapes relevant to edge/embedded deployment.
   small batches, small dims, inference-focused, single-threaded. */

#include "axiom/axiom.h"
#include "axiom/quantize.h"
#include "bench_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define WARMUP  10
#define TIMED   100

static int dcmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

typedef struct {
    double min_us, median_us, mean_us;
} stats_t;

static stats_t compute_stats(double *samples, int n) {
    qsort(samples, (size_t)n, sizeof(double), dcmp);
    double sum = 0;
    for (int i = 0; i < n; i++) sum += samples[i];
    stats_t s;
    s.min_us    = samples[0];
    s.mean_us   = sum / n;
    s.median_us = (n & 1) ? samples[n / 2]
                          : (samples[n / 2 - 1] + samples[n / 2]) * 0.5;
    return s;
}

static void emit(const char *bench, const char *shape, stats_t s, double metric) {
    if (metric > 0)
        printf("%s\t%s\t%.1f\t%.1f\t%.1f\t%.2f\n",
               bench, shape, s.min_us, s.median_us, s.mean_us, metric);
    else
        printf("%s\t%s\t%.1f\t%.1f\t%.1f\t-\n",
               bench, shape, s.min_us, s.median_us, s.mean_us);
    fflush(stdout);
}

static void emit_amortization(const char *bench, const char *shape,
                              double first_us, double steady_us) {
    double ratio = (steady_us > 0) ? first_us / steady_us : 0;
    printf("%s\t%s\tfirst=%.1f\tsteady=%.1f\tamort=%.1fx\n",
           bench, shape, first_us, steady_us, ratio);
    fflush(stdout);
}

/* ================================================================
   1. tiny GEMM (B=1 inference)
   ================================================================ */

static void bench_tiny_gemm(void) {
    typedef struct { int M, K, N; const char *tag; } shape_t;
    shape_t shapes[] = {
        {1, 32,  64,  "1x32x64_classifier"},
        {1, 128, 64,  "1x128x64_small_mlp"},
        {1, 256, 128, "1x256x128_med_mlp"},
        {1, 64,  10,  "1x64x10_final_proj"},
    };
    int ns = (int)(sizeof(shapes) / sizeof(shapes[0]));
    double samples[TIMED];

    for (int s = 0; s < ns; s++) {
        int M = shapes[s].M, K = shapes[s].K, N = shapes[s].N;
        double flops = 2.0 * M * K * N;

        int64_t a_sh[] = {M, K}, b_sh[] = {K, N}, c_sh[] = {M, N};
        ax_tensor_t *A = ax_tensor_rand(a_sh, 2, -0.5f, 0.5f);
        ax_tensor_t *B = ax_tensor_rand(b_sh, 2, -0.5f, 0.5f);
        ax_tensor_t *C = ax_tensor_create(c_sh, 2, AX_FLOAT32);

        /* first-call latency */
        double t0 = bench_now_ms();
        ax_compute_gemm(A, B, C);
        double first_us = (bench_now_ms() - t0) * 1000.0;

        for (int w = 0; w < WARMUP; w++) ax_compute_gemm(A, B, C);

        for (int i = 0; i < TIMED; i++) {
            t0 = bench_now_ms();
            ax_compute_gemm(A, B, C);
            samples[i] = (bench_now_ms() - t0) * 1000.0;
        }
        stats_t st = compute_stats(samples, TIMED);
        double gflops = (st.median_us > 0) ? flops / (st.median_us * 1e3) : 0;
        emit("tiny_gemm", shapes[s].tag, st, gflops);
        emit_amortization("tiny_gemm", shapes[s].tag, first_us, st.median_us);

        ax_tensor_destroy(C);
        ax_tensor_destroy(B);
        ax_tensor_destroy(A);
    }
}

/* ================================================================
   2. tiny MLP inference
   ================================================================ */

static void bench_tiny_mlp(void) {
    int batches[] = {1, 4, 16};
    int nb = 3;
    double samples[TIMED];

    for (int bi = 0; bi < nb; bi++) {
        int B = batches[bi];

        ax_layer_t *model = ax_sequential_create();
        ax_sequential_add(model, ax_dense_create(64, 128, true));
        ax_sequential_add(model, ax_relu_layer_create());
        ax_sequential_add(model, ax_dense_create(128, 64, true));
        ax_sequential_add(model, ax_relu_layer_create());
        ax_sequential_add(model, ax_dense_create(64, 10, false));
        ax_layer_eval(model);

        int64_t x_sh[] = {B, 64};
        ax_tensor_t *x = ax_tensor_rand(x_sh, 2, -1.0f, 1.0f);

        ax_no_grad();

        /* first-call latency */
        double t0 = bench_now_ms();
        ax_tensor_t *y0 = ax_layer_forward(model, x);
        double first_us = (bench_now_ms() - t0) * 1000.0;
        ax_tensor_destroy(y0);

        for (int w = 0; w < WARMUP; w++) {
            ax_tensor_t *y = ax_layer_forward(model, x);
            ax_tensor_destroy(y);
        }

        for (int i = 0; i < TIMED; i++) {
            t0 = bench_now_ms();
            ax_tensor_t *y = ax_layer_forward(model, x);
            samples[i] = (bench_now_ms() - t0) * 1000.0;
            ax_tensor_destroy(y);
        }
        ax_enable_grad();

        stats_t st = compute_stats(samples, TIMED);
        double per_sample = st.median_us / B;

        char shape[64];
        snprintf(shape, sizeof(shape), "B%d_64-128-64-10", B);
        emit("tiny_mlp", shape, st, per_sample);
        emit_amortization("tiny_mlp", shape, first_us, st.median_us);

        ax_tensor_destroy(x);
        ax_layer_destroy(model);
    }
}

/* ================================================================
   3. small conv inference
   ================================================================ */

static void bench_small_conv(void) {
    typedef struct {
        int Cin, Cout, H, W, K, S, P;
        const char *tag;
    } conv_shape_t;
    conv_shape_t shapes[] = {
        {3,  16, 32, 32, 3, 1, 1, "N1_3-16_32x32_K3"},
        {16, 32, 16, 16, 3, 1, 1, "N1_16-32_16x16_K3"},
        {32, 64, 8,  8,  3, 1, 1, "N1_32-64_8x8_K3"},
    };
    int ns = (int)(sizeof(shapes) / sizeof(shapes[0]));
    double samples[TIMED];

    for (int s = 0; s < ns; s++) {
        conv_shape_t *sh = &shapes[s];
        int Hout = (sh->H + 2 * sh->P - sh->K) / sh->S + 1;
        int Wout = (sh->W + 2 * sh->P - sh->K) / sh->S + 1;
        double flops = 2.0 * 1 * sh->Cout * Hout * Wout * sh->Cin * sh->K * sh->K;

        ax_layer_t *conv = ax_conv2d_create(sh->Cin, sh->Cout, sh->K, sh->S, sh->P, true);
        ax_layer_eval(conv);
        int64_t x_sh[] = {1, sh->Cin, sh->H, sh->W};
        ax_tensor_t *x = ax_tensor_rand(x_sh, 4, -0.5f, 0.5f);

        ax_no_grad();

        double t0 = bench_now_ms();
        ax_tensor_t *y0 = ax_layer_forward(conv, x);
        double first_us = (bench_now_ms() - t0) * 1000.0;
        ax_tensor_destroy(y0);

        for (int w = 0; w < WARMUP; w++) {
            ax_tensor_t *y = ax_layer_forward(conv, x);
            ax_tensor_destroy(y);
        }

        for (int i = 0; i < TIMED; i++) {
            t0 = bench_now_ms();
            ax_tensor_t *y = ax_layer_forward(conv, x);
            samples[i] = (bench_now_ms() - t0) * 1000.0;
            ax_tensor_destroy(y);
        }
        ax_enable_grad();

        stats_t st = compute_stats(samples, TIMED);
        double gflops = (st.median_us > 0) ? flops / (st.median_us * 1e3) : 0;
        emit("small_conv", sh->tag, st, gflops);
        emit_amortization("small_conv", sh->tag, first_us, st.median_us);

        ax_tensor_destroy(x);
        ax_layer_destroy(conv);
    }
}

/* ================================================================
   4. small attention (edge NLP, inference forward only)
   ================================================================ */

static void bench_small_attn(void) {
    typedef struct { int B, S, D, H; const char *tag; } attn_shape_t;
    attn_shape_t shapes[] = {
        {1, 64,  64,  2, "B1_S64_D64_H2"},
        {1, 128, 128, 4, "B1_S128_D128_H4"},
    };
    int ns = (int)(sizeof(shapes) / sizeof(shapes[0]));
    double samples[TIMED];

    for (int s = 0; s < ns; s++) {
        attn_shape_t *sh = &shapes[s];
        ax_layer_t *mha = ax_mha_create(sh->D, sh->H, false, false);
        ax_layer_eval(mha);

        int64_t x_sh[] = {sh->B, sh->S, sh->D};
        ax_tensor_t *x = ax_tensor_rand(x_sh, 3, -0.5f, 0.5f);

        ax_no_grad();

        double t0 = bench_now_ms();
        ax_tensor_t *y0 = ax_layer_forward(mha, x);
        double first_us = (bench_now_ms() - t0) * 1000.0;
        ax_tensor_destroy(y0);

        for (int w = 0; w < WARMUP; w++) {
            ax_tensor_t *y = ax_layer_forward(mha, x);
            ax_tensor_destroy(y);
        }

        for (int i = 0; i < TIMED; i++) {
            t0 = bench_now_ms();
            ax_tensor_t *y = ax_layer_forward(mha, x);
            samples[i] = (bench_now_ms() - t0) * 1000.0;
            ax_tensor_destroy(y);
        }
        ax_enable_grad();

        stats_t st = compute_stats(samples, TIMED);
        emit("small_attn", sh->tag, st, 0);
        emit_amortization("small_attn", sh->tag, first_us, st.median_us);

        ax_tensor_destroy(x);
        ax_layer_destroy(mha);
    }
}

/* ================================================================
   5. BN inference (eval mode: scale+shift only)
   ================================================================ */

static void bench_bn_infer(void) {
    typedef struct { int N, C, H, W; const char *tag; } bn_shape_t;
    bn_shape_t shapes[] = {
        {1, 32, 16, 16, "N1_C32_16x16"},
        {1, 64, 8,  8,  "N1_C64_8x8"},
    };
    int ns = (int)(sizeof(shapes) / sizeof(shapes[0]));
    double samples[TIMED];

    for (int s = 0; s < ns; s++) {
        bn_shape_t *sh = &shapes[s];
        /* read + write of the full tensor + gamma/beta loads */
        double bytes = 2.0 * sh->N * sh->C * sh->H * sh->W * sizeof(float)
                       + 2.0 * sh->C * sizeof(float);

        ax_layer_t *bn = ax_batchnorm_create(sh->C, 1e-5f, 0.1f);
        ax_layer_eval(bn);

        int64_t x_sh[] = {sh->N, sh->C, sh->H, sh->W};
        ax_tensor_t *x = ax_tensor_rand(x_sh, 4, -1.0f, 1.0f);

        ax_no_grad();

        double t0 = bench_now_ms();
        ax_tensor_t *y0 = ax_layer_forward(bn, x);
        double first_us = (bench_now_ms() - t0) * 1000.0;
        ax_tensor_destroy(y0);

        for (int w = 0; w < WARMUP; w++) {
            ax_tensor_t *y = ax_layer_forward(bn, x);
            ax_tensor_destroy(y);
        }

        for (int i = 0; i < TIMED; i++) {
            t0 = bench_now_ms();
            ax_tensor_t *y = ax_layer_forward(bn, x);
            samples[i] = (bench_now_ms() - t0) * 1000.0;
            ax_tensor_destroy(y);
        }
        ax_enable_grad();

        stats_t st = compute_stats(samples, TIMED);
        double gbs = (st.median_us > 0) ? bytes / (st.median_us * 1e3) : 0;
        emit("bn_infer", sh->tag, st, gbs);
        emit_amortization("bn_infer", sh->tag, first_us, st.median_us);

        ax_tensor_destroy(x);
        ax_layer_destroy(bn);
    }
}

/* ================================================================
   6. quantized GEMM: W8A32 vs fp32 on embedded shape
   ================================================================ */

static void bench_qgemm_embedded(void) {
    int64_t M = 1, N = 64, K = 128;
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double samples_fp[TIMED], samples_q[TIMED];

    /* fp32 path: A[M,K] @ B[K,N] */
    float *a_raw = aligned_alloc(64, (size_t)(M * K) * sizeof(float));
    float *w_raw = aligned_alloc(64, (size_t)(N * K) * sizeof(float));
    uint32_t seed = 42;
    for (int64_t i = 0; i < M * K; i++) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        a_raw[i] = (float)(seed & 0xFFFF) / 32768.0f - 1.0f;
    }
    for (int64_t i = 0; i < N * K; i++) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        w_raw[i] = (float)(seed & 0xFFFF) / 32768.0f - 1.0f;
    }

    /* transpose w from [N,K] to [K,N] for fp32 gemm */
    float *w_T = aligned_alloc(64, (size_t)(K * N) * sizeof(float));
    for (int64_t n = 0; n < N; n++)
        for (int64_t k = 0; k < K; k++)
            w_T[k * N + n] = w_raw[n * K + k];

    float *out_fp = aligned_alloc(64, (size_t)(M * N) * sizeof(float));
    float *out_q  = aligned_alloc(64, (size_t)(M * N) * sizeof(float));

    int64_t a_sh[] = {M, K}, b_sh[] = {K, N}, c_sh[] = {M, N};
    ax_tensor_t *tA = ax_tensor_from_array(a_raw, a_sh, 2, AX_FLOAT32);
    ax_tensor_t *tB = ax_tensor_from_array(w_T, b_sh, 2, AX_FLOAT32);
    ax_tensor_t *tC = ax_tensor_from_array(out_fp, c_sh, 2, AX_FLOAT32);

    ax_qweight_t *qw = ax_qweight_create_from_fp32(w_raw, N, K);

    for (int w = 0; w < WARMUP; w++) {
        ax_compute_gemm(tA, tB, tC);
        ax_qgemm_w8a32(a_raw, qw, out_q, M);
    }

    for (int i = 0; i < TIMED; i++) {
        double t0 = bench_now_ms();
        ax_compute_gemm(tA, tB, tC);
        samples_fp[i] = (bench_now_ms() - t0) * 1000.0;
    }

    for (int i = 0; i < TIMED; i++) {
        double t0 = bench_now_ms();
        ax_qgemm_w8a32(a_raw, qw, out_q, M);
        samples_q[i] = (bench_now_ms() - t0) * 1000.0;
    }

    stats_t st_fp = compute_stats(samples_fp, TIMED);
    stats_t st_q  = compute_stats(samples_q, TIMED);
    double gflops_fp = (st_fp.median_us > 0) ? flops / (st_fp.median_us * 1e3) : 0;
    double gflops_q  = (st_q.median_us > 0)  ? flops / (st_q.median_us * 1e3)  : 0;

    emit("qgemm_fp32", "1x128x64", st_fp, gflops_fp);
    emit("qgemm_w8a32", "1x128x64", st_q, gflops_q);

    double speedup = (st_q.median_us > 0) ? st_fp.median_us / st_q.median_us : 0;
    printf("qgemm_speedup\t1x128x64\t%.2fx\n", speedup);
    fflush(stdout);

    ax_tensor_destroy(tA);
    ax_tensor_destroy(tB);
    ax_tensor_destroy(tC);
    ax_qweight_destroy(qw);
    free(a_raw); free(w_raw); free(w_T); free(out_fp); free(out_q);
}

/* ================================================================
   7. full edge inference pipeline
   Conv(3->16,3x3) -> BN -> ReLU -> Conv(16->32,3x3) -> BN -> ReLU
   -> GlobalAvgPool -> Dense(32->10)
   ================================================================ */

static void bench_edge_pipeline(void) {
    double samples[TIMED];

    ax_layer_t *model = ax_sequential_create();
    ax_sequential_add(model, ax_conv2d_create(3, 16, 3, 1, 1, true));
    ax_sequential_add(model, ax_batchnorm_create(16, 1e-5f, 0.1f));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_conv2d_create(16, 32, 3, 1, 1, true));
    ax_sequential_add(model, ax_batchnorm_create(32, 1e-5f, 0.1f));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_global_avgpool2d_create());
    ax_sequential_add(model, ax_flatten_create());
    ax_sequential_add(model, ax_dense_create(32, 10, false));
    ax_layer_eval(model);

    int64_t x_sh[] = {1, 3, 32, 32};
    ax_tensor_t *x = ax_tensor_rand(x_sh, 4, -0.5f, 0.5f);

    ax_no_grad();

    /* first-call latency */
    double t0 = bench_now_ms();
    ax_tensor_t *y0 = ax_layer_forward(model, x);
    double first_us = (bench_now_ms() - t0) * 1000.0;
    ax_tensor_destroy(y0);

    for (int w = 0; w < WARMUP; w++) {
        ax_tensor_t *y = ax_layer_forward(model, x);
        ax_tensor_destroy(y);
    }

    for (int i = 0; i < TIMED; i++) {
        t0 = bench_now_ms();
        ax_tensor_t *y = ax_layer_forward(model, x);
        samples[i] = (bench_now_ms() - t0) * 1000.0;
        ax_tensor_destroy(y);
    }
    ax_enable_grad();

    stats_t st = compute_stats(samples, TIMED);
    emit("edge_pipeline", "1x3x32x32_conv-bn-relu-gap-fc", st, 0);
    emit_amortization("edge_pipeline", "1x3x32x32", first_us, st.median_us);

    ax_tensor_destroy(x);
    ax_layer_destroy(model);
}

/* ================================================================
   KV cache attention (autoregressive single-token decode)
   ================================================================ */

static void bench_kvcache_attn(void) {
    typedef struct { int B, H, S, dk; const char *tag; } kv_shape_t;
    kv_shape_t shapes[] = {
        {1, 2, 64,  32, "B1_H2_S64_dk32"},
        {1, 4, 128, 32, "B1_H4_S128_dk32"},
    };
    int ns = (int)(sizeof(shapes) / sizeof(shapes[0]));
    double samples[TIMED];

    for (int s = 0; s < ns; s++) {
        kv_shape_t *sh = &shapes[s];
        int64_t BH = sh->B * sh->H;
        float scale = 1.0f / sqrtf((float)sh->dk);

        ax_kv_cache_t *cache = ax_kv_cache_create(BH, sh->S, sh->dk);

        /* fill the cache to capacity */
        float *new_K = aligned_alloc(64, (size_t)(BH * sh->dk) * sizeof(float));
        float *new_V = aligned_alloc(64, (size_t)(BH * sh->dk) * sizeof(float));
        uint32_t seed = 7;
        for (int64_t i = 0; i < BH * sh->dk; i++) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            new_K[i] = (float)(seed & 0xFFFF) / 32768.0f - 1.0f;
            new_V[i] = (float)((seed >> 16) & 0xFFFF) / 32768.0f - 1.0f;
        }
        for (int t = 0; t < sh->S; t++)
            ax_kv_cache_append(cache, new_K, new_V);

        float *Q_new = aligned_alloc(64, (size_t)(BH * sh->dk) * sizeof(float));
        float *out   = aligned_alloc(64, (size_t)(BH * sh->dk) * sizeof(float));
        for (int64_t i = 0; i < BH * sh->dk; i++) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            Q_new[i] = (float)(seed & 0xFFFF) / 32768.0f - 1.0f;
        }

        double t0 = bench_now_ms();
        ax_kv_cache_attend(cache, Q_new, out, scale);
        double first_us = (bench_now_ms() - t0) * 1000.0;

        for (int w = 0; w < WARMUP; w++)
            ax_kv_cache_attend(cache, Q_new, out, scale);

        for (int i = 0; i < TIMED; i++) {
            t0 = bench_now_ms();
            ax_kv_cache_attend(cache, Q_new, out, scale);
            samples[i] = (bench_now_ms() - t0) * 1000.0;
        }

        stats_t st = compute_stats(samples, TIMED);
        emit("kv_attend", sh->tag, st, 0);
        emit_amortization("kv_attend", sh->tag, first_us, st.median_us);

        free(new_K); free(new_V); free(Q_new); free(out);
        ax_kv_cache_destroy(cache);
    }
}

/* ================================================================
   alloc audit: measure heap allocations in cold and steady state
   ================================================================ */

#ifdef AX_COUNT_ALLOCS
static void bench_alloc_audit(void) {
    /* model creation: weights, scratch buffers, layer structs */
    ax_reset_alloc_count();
    ax_layer_t *model = ax_sequential_create();
    ax_sequential_add(model, ax_conv2d_create(3, 16, 3, 1, 1, true));
    ax_sequential_add(model, ax_batchnorm_create(16, 1e-5f, 0.1f));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_conv2d_create(16, 32, 3, 1, 1, true));
    ax_sequential_add(model, ax_batchnorm_create(32, 1e-5f, 0.1f));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_global_avgpool2d_create());
    ax_sequential_add(model, ax_flatten_create());
    ax_sequential_add(model, ax_dense_create(32, 10, false));
    ax_layer_eval(model);
    uint64_t model_allocs = ax_get_alloc_count();

    int64_t x_sh[] = {1, 3, 32, 32};
    ax_tensor_t *x = ax_tensor_rand(x_sh, 4, -0.5f, 0.5f);

    ax_no_grad();

    /* first forward: cold pool, every intermediate is a fresh alloc */
    ax_reset_alloc_count();
    ax_tensor_t *y0 = ax_layer_forward(model, x);
    uint64_t first_fwd = ax_get_alloc_count();
    ax_tensor_destroy(y0);

    /* warmup: fill pool with recycled buffers */
    for (int w = 0; w < WARMUP; w++) {
        ax_tensor_t *y = ax_layer_forward(model, x);
        ax_tensor_destroy(y);
    }

    /* steady state: should be zero if pool serves all requests */
    ax_reset_alloc_count();
    for (int i = 0; i < TIMED; i++) {
        ax_tensor_t *y = ax_layer_forward(model, x);
        ax_tensor_destroy(y);
    }
    uint64_t steady_allocs = ax_get_alloc_count();

    ax_enable_grad();

    printf("alloc_audit\tedge_pipeline\tmodel=%llu\tfirst_fwd=%llu\tsteady=%llu\tper_iter=%.2f\n",
           (unsigned long long)model_allocs,
           (unsigned long long)first_fwd,
           (unsigned long long)steady_allocs,
           (double)steady_allocs / TIMED);
    fflush(stdout);

    ax_tensor_destroy(x);
    ax_layer_destroy(model);
}
#endif

int main(void) {
    ax_init();
    ax_set_num_threads(1);

    printf("bench_embedded: edge/embedded inference shapes (1 thread)\n");
    printf("backend: %s\n\n", ax_compute_backend_name());
    printf("bench\tshape\tmin_us\tmedian_us\tmean_us\tgflops_or_gbs\n");

#ifdef AX_COUNT_ALLOCS
    /* run audit first so pool is cold */
    bench_alloc_audit();
    printf("\n");
#endif

    bench_tiny_gemm();
    printf("\n");
    bench_tiny_mlp();
    printf("\n");
    bench_small_conv();
    printf("\n");
    bench_small_attn();
    printf("\n");
    bench_bn_infer();
    printf("\n");
    bench_qgemm_embedded();
    printf("\n");
    bench_edge_pipeline();
    printf("\n");
    bench_kvcache_attn();

    ax_shutdown();
    return 0;
}
