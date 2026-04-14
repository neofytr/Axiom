/* bench_ops_suite.c — everything that's NOT a GEMM.

   covers elementwise, reductions, softmax, norm layers. each case
   reports latency; compute-bound cases add GFLOPS, memory-bound
   cases add GB/s. summarize.py compares to tf_ops_suite.py. */

#include "axiom/axiom.h"
#include "bench_common.h"
#include <stdio.h>
#include <stdlib.h>

static void bench_relu(int64_t N) {
    int64_t sh[] = {N};
    ax_tensor_t *x = ax_tensor_rand(sh, 1, -1.0f, 1.0f);
    int iters = (int)(1e9 / N); if (iters < 10) iters = 10; if (iters > 10000) iters = 10000;
    /* warmup */
    for (int w = 0; w < 3; w++) { ax_tensor_t *y = ax_relu(x); ax_tensor_destroy(y); }
    double t0 = bench_now_ms();
    for (int i = 0; i < iters; i++) {
        ax_tensor_t *y = ax_relu(x);
        ax_tensor_destroy(y);
    }
    double lat = (bench_now_ms() - t0) / iters;
    char cs[64]; snprintf(cs, sizeof(cs), "relu_%ld", (long)N);
    /* memory traffic: read N + write N = 8N bytes */
    BENCH_EMIT_BW("ops", cs, lat, 8.0 * (double)N);
    ax_tensor_destroy(x);
}

static void bench_gelu(int64_t N) {
    int64_t sh[] = {N};
    ax_tensor_t *x = ax_tensor_rand(sh, 1, -1.0f, 1.0f);
    int iters = (int)(5e8 / N); if (iters < 10) iters = 10; if (iters > 10000) iters = 10000;
    for (int w = 0; w < 3; w++) { ax_tensor_t *y = ax_gelu(x); ax_tensor_destroy(y); }
    double t0 = bench_now_ms();
    for (int i = 0; i < iters; i++) {
        ax_tensor_t *y = ax_gelu(x);
        ax_tensor_destroy(y);
    }
    double lat = (bench_now_ms() - t0) / iters;
    char cs[64]; snprintf(cs, sizeof(cs), "gelu_%ld", (long)N);
    BENCH_EMIT_BW("ops", cs, lat, 8.0 * (double)N);
    ax_tensor_destroy(x);
}

static void bench_add(int64_t N) {
    int64_t sh[] = {N};
    ax_tensor_t *a = ax_tensor_rand(sh, 1, -1.0f, 1.0f);
    ax_tensor_t *b = ax_tensor_rand(sh, 1, -1.0f, 1.0f);
    int iters = (int)(1e9 / N); if (iters < 10) iters = 10; if (iters > 10000) iters = 10000;
    for (int w = 0; w < 3; w++) { ax_tensor_t *y = ax_add(a, b); ax_tensor_destroy(y); }
    double t0 = bench_now_ms();
    for (int i = 0; i < iters; i++) {
        ax_tensor_t *y = ax_add(a, b);
        ax_tensor_destroy(y);
    }
    double lat = (bench_now_ms() - t0) / iters;
    char cs[64]; snprintf(cs, sizeof(cs), "add_%ld", (long)N);
    /* 2 reads + 1 write = 12N bytes */
    BENCH_EMIT_BW("ops", cs, lat, 12.0 * (double)N);
    ax_tensor_destroy(a); ax_tensor_destroy(b);
}

static void bench_sum(int64_t N) {
    int64_t sh[] = {N};
    ax_tensor_t *x = ax_tensor_rand(sh, 1, -1.0f, 1.0f);
    int iters = (int)(1e9 / N); if (iters < 10) iters = 10; if (iters > 10000) iters = 10000;
    for (int w = 0; w < 3; w++) { ax_tensor_t *y = ax_sum(x, -1); ax_tensor_destroy(y); }
    double t0 = bench_now_ms();
    for (int i = 0; i < iters; i++) {
        ax_tensor_t *y = ax_sum(x, -1);
        ax_tensor_destroy(y);
    }
    double lat = (bench_now_ms() - t0) / iters;
    char cs[64]; snprintf(cs, sizeof(cs), "sum_%ld", (long)N);
    BENCH_EMIT_BW("ops", cs, lat, 4.0 * (double)N); /* read only */
    ax_tensor_destroy(x);
}

static void bench_softmax(int64_t rows, int64_t cols) {
    int64_t sh[] = {rows, cols};
    ax_tensor_t *x = ax_tensor_rand(sh, 2, -1.0f, 1.0f);
    int iters = (int)(5e8 / (rows * cols));
    if (iters < 10) iters = 10; if (iters > 2000) iters = 2000;
    for (int w = 0; w < 3; w++) { ax_tensor_t *y = ax_softmax(x, 1); ax_tensor_destroy(y); }
    double t0 = bench_now_ms();
    for (int i = 0; i < iters; i++) {
        ax_tensor_t *y = ax_softmax(x, 1);
        ax_tensor_destroy(y);
    }
    double lat = (bench_now_ms() - t0) / iters;
    char cs[64]; snprintf(cs, sizeof(cs), "softmax_%ldx%ld", (long)rows, (long)cols);
    BENCH_EMIT_BW("ops", cs, lat, 8.0 * (double)(rows * cols));
    ax_tensor_destroy(x);
}

static void bench_batchnorm(int64_t batch, int64_t feat) {
    int64_t sh[] = {batch, feat};
    ax_tensor_t *x = ax_tensor_rand(sh, 2, -1.0f, 1.0f);
    ax_layer_t *bn = ax_batchnorm_create(feat, 1e-5f, 0.1f);
    ax_layer_eval(bn); /* eval path — no running stat update */
    ax_no_grad();
    for (int w = 0; w < 3; w++) { ax_tensor_t *y = ax_layer_forward(bn, x); ax_tensor_destroy(y); }
    int iters = (int)(3e8 / (batch * feat)); if (iters < 10) iters = 10; if (iters > 1000) iters = 1000;
    double t0 = bench_now_ms();
    for (int i = 0; i < iters; i++) {
        ax_tensor_t *y = ax_layer_forward(bn, x);
        ax_tensor_destroy(y);
    }
    double lat = (bench_now_ms() - t0) / iters;
    ax_enable_grad();
    char cs[64]; snprintf(cs, sizeof(cs), "batchnorm_%ldx%ld", (long)batch, (long)feat);
    BENCH_EMIT_BW("ops", cs, lat, 8.0 * (double)(batch * feat));
    ax_tensor_destroy(x); ax_layer_destroy(bn);
}

static void bench_layernorm(int64_t batch, int64_t feat) {
    int64_t sh[] = {batch, feat};
    ax_tensor_t *x = ax_tensor_rand(sh, 2, -1.0f, 1.0f);
    ax_layer_t *ln = ax_layernorm_create(feat, 1e-5f);
    ax_no_grad();
    for (int w = 0; w < 3; w++) { ax_tensor_t *y = ax_layer_forward(ln, x); ax_tensor_destroy(y); }
    int iters = (int)(3e8 / (batch * feat)); if (iters < 10) iters = 10; if (iters > 1000) iters = 1000;
    double t0 = bench_now_ms();
    for (int i = 0; i < iters; i++) {
        ax_tensor_t *y = ax_layer_forward(ln, x);
        ax_tensor_destroy(y);
    }
    double lat = (bench_now_ms() - t0) / iters;
    ax_enable_grad();
    char cs[64]; snprintf(cs, sizeof(cs), "layernorm_%ldx%ld", (long)batch, (long)feat);
    BENCH_EMIT_BW("ops", cs, lat, 8.0 * (double)(batch * feat));
    ax_tensor_destroy(x); ax_layer_destroy(ln);
}

int main(void) {
    ax_init();

    /* elementwise across sizes from L1 to beyond L3 */
    int64_t sizes[] = {1 << 14, 1 << 18, 1 << 22, 1 << 24}; /* 64KB, 1MB, 16MB, 64MB */
    for (size_t i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
        bench_relu(sizes[i]);
        bench_gelu(sizes[i]);
        bench_add(sizes[i]);
        bench_sum(sizes[i]);
    }

    /* softmax across realistic transformer shapes (batch*head, seq_len) */
    int64_t sm_shapes[][2] = {
        {32,    512},     /* small */
        {128,   1024},
        {256,   2048},
    };
    for (size_t i = 0; i < sizeof(sm_shapes)/sizeof(sm_shapes[0]); i++)
        bench_softmax(sm_shapes[i][0], sm_shapes[i][1]);

    /* norm layers at realistic MLP/transformer shapes */
    int64_t norm_shapes[][2] = {
        {256,  512},
        {256,  1024},
        {256,  4096},
        {64,   8192},
    };
    for (size_t i = 0; i < sizeof(norm_shapes)/sizeof(norm_shapes[0]); i++) {
        bench_batchnorm(norm_shapes[i][0], norm_shapes[i][1]);
        bench_layernorm(norm_shapes[i][0], norm_shapes[i][1]);
    }

    ax_shutdown();
    return 0;
}
