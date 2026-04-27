/* bench_memory.c — memory footprint benchmarks.
   tracks RSS, arena usage, and allocation patterns at lifecycle points
   during model creation, training, and inference. */

#include "axiom/axiom.h"
#include "bench_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static long get_rss_kb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long rss = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, " %ld", &rss);
            break;
        }
    }
    fclose(f);
    return rss;
}

/* force pages resident so RSS reflects actual allocation */
static void touch_tensor(ax_tensor_t *t) {
    if (!t || !t->storage || !t->storage->data) return;
    memset(t->storage->data, 0, t->storage->size_bytes);
}

static void print_rss_row(const char *stage, long rss_kb, long baseline_kb) {
    printf("%-28s %8ld    %+8ld\n", stage, rss_kb, rss_kb - baseline_kb);
}

/* 1: MLP lifecycle RSS */
static void bench_mlp_lifecycle(void) {
    printf("\n=== MLP Lifecycle RSS ===\n");
    printf("%-28s %8s    %8s\n", "stage", "rss_kb", "delta_kb");

    long baseline = get_rss_kb();
    print_rss_row("baseline", baseline, baseline);

    ax_layer_t *fc1 = ax_dense_create(768, 512, 1);
    ax_layer_t *fc2 = ax_dense_create(512, 256, 1);
    ax_layer_t *fc3 = ax_dense_create(256, 10, 1);
    ax_layer_train(fc1); ax_layer_train(fc2); ax_layer_train(fc3);

    long rss = get_rss_kb();
    print_rss_row("layers_created", rss, baseline);

    int64_t x_sh[] = {32, 768};
    ax_tensor_t *x = ax_tensor_rand(x_sh, 2, -0.5f, 0.5f);
    touch_tensor(x);

    int64_t labels_sh[] = {32};
    ax_tensor_t *labels = ax_tensor_create(labels_sh, 1, AX_INT64);
    int64_t *ld = (int64_t *)labels->storage->data;
    for (int i = 0; i < 32; i++) ld[i] = i % 10;

    ax_tensor_t *h1 = ax_layer_forward(fc1, x);
    ax_tensor_t *a1 = ax_relu(h1);
    ax_tensor_t *h2 = ax_layer_forward(fc2, a1);
    ax_tensor_t *a2 = ax_relu(h2);
    ax_tensor_t *h3 = ax_layer_forward(fc3, a2);
    ax_tensor_t *loss = ax_cross_entropy_loss(h3, labels);

    rss = get_rss_kb();
    print_rss_row("first_forward", rss, baseline);

    ax_backward(loss);

    rss = get_rss_kb();
    print_rss_row("first_backward", rss, baseline);

    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss);
    ax_tensor_destroy(h3); ax_tensor_destroy(a2); ax_tensor_destroy(h2);
    ax_tensor_destroy(a1); ax_tensor_destroy(h1);

    rss = get_rss_kb();
    print_rss_row("after_cleanup", rss, baseline);

    for (int step = 0; step < 10; step++) {
        h1 = ax_layer_forward(fc1, x);
        a1 = ax_relu(h1);
        h2 = ax_layer_forward(fc2, a1);
        a2 = ax_relu(h2);
        h3 = ax_layer_forward(fc3, a2);
        loss = ax_cross_entropy_loss(h3, labels);
        ax_backward(loss);
        ax_graph_cleanup(loss);
        ax_tensor_destroy(loss); ax_tensor_destroy(h3);
        ax_tensor_destroy(a2); ax_tensor_destroy(h2);
        ax_tensor_destroy(a1); ax_tensor_destroy(h1);
    }

    rss = get_rss_kb();
    print_rss_row("after_10_steps", rss, baseline);

    ax_tensor_destroy(labels); ax_tensor_destroy(x);
    ax_layer_destroy(fc3); ax_layer_destroy(fc2); ax_layer_destroy(fc1);

    rss = get_rss_kb();
    print_rss_row("after_destroy", rss, baseline);

    fflush(stdout);
}

/* 2: CNN lifecycle RSS */
static void bench_cnn_lifecycle(void) {
    printf("\n=== CNN Lifecycle RSS ===\n");
    printf("%-28s %8s    %8s\n", "stage", "rss_kb", "delta_kb");

    long baseline = get_rss_kb();
    print_rss_row("baseline", baseline, baseline);

    ax_layer_t *conv1 = ax_conv2d_create(3, 32, 3, 1, 1, 1);
    ax_layer_t *bn1   = ax_batchnorm_create(32, 1e-5f, 0.1f);
    ax_layer_t *conv2 = ax_conv2d_create(32, 64, 3, 1, 1, 1);
    ax_layer_t *bn2   = ax_batchnorm_create(64, 1e-5f, 0.1f);
    ax_layer_t *gap   = ax_global_avgpool2d_create();
    ax_layer_t *flat  = ax_flatten_create();
    ax_layer_t *fc    = ax_dense_create(64, 10, 1);

    ax_layer_train(conv1); ax_layer_train(bn1);
    ax_layer_train(conv2); ax_layer_train(bn2);
    ax_layer_train(gap); ax_layer_train(flat); ax_layer_train(fc);

    long rss = get_rss_kb();
    print_rss_row("layers_created", rss, baseline);

    int64_t x_sh[] = {4, 3, 32, 32};
    ax_tensor_t *x = ax_tensor_rand(x_sh, 4, -0.5f, 0.5f);
    touch_tensor(x);

    int64_t labels_sh[] = {4};
    ax_tensor_t *labels = ax_tensor_create(labels_sh, 1, AX_INT64);
    int64_t *ld = (int64_t *)labels->storage->data;
    for (int i = 0; i < 4; i++) ld[i] = i % 10;

    ax_tensor_t *c1 = ax_layer_forward(conv1, x);
    ax_tensor_t *b1 = ax_layer_forward(bn1, c1);
    ax_tensor_t *r1 = ax_relu(b1);
    ax_tensor_t *c2 = ax_layer_forward(conv2, r1);
    ax_tensor_t *b2 = ax_layer_forward(bn2, c2);
    ax_tensor_t *r2 = ax_relu(b2);
    ax_tensor_t *gp = ax_layer_forward(gap, r2);
    ax_tensor_t *fl = ax_layer_forward(flat, gp);
    ax_tensor_t *out = ax_layer_forward(fc, fl);
    ax_tensor_t *loss = ax_cross_entropy_loss(out, labels);

    rss = get_rss_kb();
    print_rss_row("first_forward", rss, baseline);

    ax_backward(loss);

    rss = get_rss_kb();
    print_rss_row("first_backward", rss, baseline);

    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss); ax_tensor_destroy(out);
    ax_tensor_destroy(fl); ax_tensor_destroy(gp);
    ax_tensor_destroy(r2); ax_tensor_destroy(b2); ax_tensor_destroy(c2);
    ax_tensor_destroy(r1); ax_tensor_destroy(b1); ax_tensor_destroy(c1);

    rss = get_rss_kb();
    print_rss_row("after_cleanup", rss, baseline);

    for (int step = 0; step < 10; step++) {
        c1 = ax_layer_forward(conv1, x);
        b1 = ax_layer_forward(bn1, c1);
        r1 = ax_relu(b1);
        c2 = ax_layer_forward(conv2, r1);
        b2 = ax_layer_forward(bn2, c2);
        r2 = ax_relu(b2);
        gp = ax_layer_forward(gap, r2);
        fl = ax_layer_forward(flat, gp);
        out = ax_layer_forward(fc, fl);
        loss = ax_cross_entropy_loss(out, labels);
        ax_backward(loss);
        ax_graph_cleanup(loss);
        ax_tensor_destroy(loss); ax_tensor_destroy(out);
        ax_tensor_destroy(fl); ax_tensor_destroy(gp);
        ax_tensor_destroy(r2); ax_tensor_destroy(b2); ax_tensor_destroy(c2);
        ax_tensor_destroy(r1); ax_tensor_destroy(b1); ax_tensor_destroy(c1);
    }

    rss = get_rss_kb();
    print_rss_row("after_10_steps", rss, baseline);

    ax_tensor_destroy(labels); ax_tensor_destroy(x);
    ax_layer_destroy(fc); ax_layer_destroy(flat); ax_layer_destroy(gap);
    ax_layer_destroy(bn2); ax_layer_destroy(conv2);
    ax_layer_destroy(bn1); ax_layer_destroy(conv1);

    rss = get_rss_kb();
    print_rss_row("after_destroy", rss, baseline);

    fflush(stdout);
}

/* 3: MHA lifecycle RSS */
static void bench_mha_lifecycle(void) {
    printf("\n=== MHA Lifecycle RSS ===\n");
    printf("%-28s %8s    %8s\n", "stage", "rss_kb", "delta_kb");

    long baseline = get_rss_kb();
    print_rss_row("baseline", baseline, baseline);

    ax_layer_t *mha = ax_mha_create(256, 4, 0, 0);
    ax_layer_train(mha);

    long rss = get_rss_kb();
    print_rss_row("mha_created", rss, baseline);

    int64_t sh[] = {1, 128, 256};
    ax_tensor_t *x    = ax_tensor_rand(sh, 3, -0.5f, 0.5f);
    ax_tensor_t *dout = ax_tensor_rand(sh, 3, -0.5f, 0.5f);
    ax_tensor_t *y    = ax_tensor_create(sh, 3, AX_FLOAT32);
    touch_tensor(x); touch_tensor(dout); touch_tensor(y);

    ax_mha_train_step_v4(mha, x, dout, y);

    rss = get_rss_kb();
    print_rss_row("first_train_step", rss, baseline);

    for (int step = 0; step < 10; step++)
        ax_mha_train_step_v4(mha, x, dout, y);

    rss = get_rss_kb();
    print_rss_row("after_10_steps", rss, baseline);

    ax_tensor_destroy(y); ax_tensor_destroy(dout); ax_tensor_destroy(x);
    ax_layer_destroy(mha);

    rss = get_rss_kb();
    print_rss_row("after_destroy", rss, baseline);

    fflush(stdout);
}

/* 4: peak RSS under batch size sweep (MLP) */
static void bench_batch_sweep(void) {
    printf("\n=== Peak RSS vs Batch Size ===\n");
    printf("%-12s %12s\n", "batch_size", "peak_rss_kb");

    int batch_sizes[] = {1, 4, 16, 32, 64, 128};
    int n_sizes = (int)(sizeof(batch_sizes) / sizeof(batch_sizes[0]));

    for (int bi = 0; bi < n_sizes; bi++) {
        int B = batch_sizes[bi];

        ax_layer_t *fc1 = ax_dense_create(768, 512, 1);
        ax_layer_t *fc2 = ax_dense_create(512, 256, 1);
        ax_layer_t *fc3 = ax_dense_create(256, 10, 1);
        ax_layer_train(fc1); ax_layer_train(fc2); ax_layer_train(fc3);

        int64_t x_sh[] = {B, 768};
        ax_tensor_t *x = ax_tensor_rand(x_sh, 2, -0.5f, 0.5f);
        touch_tensor(x);

        int64_t labels_sh[] = {B};
        ax_tensor_t *labels = ax_tensor_create(labels_sh, 1, AX_INT64);
        int64_t *ld = (int64_t *)labels->storage->data;
        for (int i = 0; i < B; i++) ld[i] = i % 10;

        long peak = 0;

        for (int step = 0; step < 5; step++) {
            ax_tensor_t *h1 = ax_layer_forward(fc1, x);
            ax_tensor_t *a1 = ax_relu(h1);
            ax_tensor_t *h2 = ax_layer_forward(fc2, a1);
            ax_tensor_t *a2 = ax_relu(h2);
            ax_tensor_t *h3 = ax_layer_forward(fc3, a2);
            ax_tensor_t *loss = ax_cross_entropy_loss(h3, labels);

            long rss = get_rss_kb();
            if (rss > peak) peak = rss;

            ax_backward(loss);

            rss = get_rss_kb();
            if (rss > peak) peak = rss;

            ax_graph_cleanup(loss);
            ax_tensor_destroy(loss); ax_tensor_destroy(h3);
            ax_tensor_destroy(a2); ax_tensor_destroy(h2);
            ax_tensor_destroy(a1); ax_tensor_destroy(h1);
        }

        printf("%-12d %12ld\n", B, peak);
        fflush(stdout);

        ax_tensor_destroy(labels); ax_tensor_destroy(x);
        ax_layer_destroy(fc3); ax_layer_destroy(fc2); ax_layer_destroy(fc1);
    }

    fflush(stdout);
}

/* 5: thread count vs RSS (TLS arena overhead) */
static void bench_thread_rss(void) {
    printf("\n=== Thread Count vs RSS ===\n");
    printf("%-10s %12s %16s\n", "threads", "rss_kb", "rss_per_thread_kb");

#ifdef _OPENMP
    int thread_counts[] = {1, 2, 4, 8, 16};
    int n_counts = (int)(sizeof(thread_counts) / sizeof(thread_counts[0]));

    int64_t a_sh[] = {2048, 768};
    int64_t b_sh[] = {768, 768};
    int64_t c_sh[] = {2048, 768};
    ax_tensor_t *A = ax_tensor_rand(a_sh, 2, -0.5f, 0.5f);
    ax_tensor_t *B = ax_tensor_rand(b_sh, 2, -0.5f, 0.5f);
    ax_tensor_t *C = ax_tensor_create(c_sh, 2, AX_FLOAT32);
    touch_tensor(A); touch_tensor(B); touch_tensor(C);

    for (int ti = 0; ti < n_counts; ti++) {
        int nt = thread_counts[ti];
        omp_set_num_threads(nt);

        /* warmup to allocate TLS arenas */
        for (int w = 0; w < 3; w++)
            ax_compute_gemm(A, B, C);

        long rss = get_rss_kb();
        printf("%-10d %12ld %16ld\n", nt, rss, rss / nt);
        fflush(stdout);
    }

    ax_tensor_destroy(C); ax_tensor_destroy(B); ax_tensor_destroy(A);
#else
    printf("(skipped: no openmp)\n");
#endif

    fflush(stdout);
}

/* 6: allocation density test */
static void bench_alloc_density(void) {
    printf("\n=== Allocation Density ===\n");
    printf("%-20s %8s %12s %10s\n", "phase", "count", "tensor_size", "rss_kb");

    long baseline = get_rss_kb();

    int N = 1000;
    ax_tensor_t **tensors = malloc((size_t)(3 * N) * sizeof(ax_tensor_t *));

    /* small tensors [1, 16] */
    int64_t small_sh[] = {1, 16};
    for (int i = 0; i < N; i++) {
        tensors[i] = ax_tensor_zeros(small_sh, 2, AX_FLOAT32);
        touch_tensor(tensors[i]);
    }
    long rss = get_rss_kb();
    printf("%-20s %8d %12s %10ld\n", "small_alloc", N, "[1,16]", rss);

    /* medium tensors [32, 64] */
    int64_t med_sh[] = {32, 64};
    for (int i = 0; i < N; i++) {
        tensors[N + i] = ax_tensor_zeros(med_sh, 2, AX_FLOAT32);
        touch_tensor(tensors[N + i]);
    }
    rss = get_rss_kb();
    printf("%-20s %8d %12s %10ld\n", "medium_alloc", N, "[32,64]", rss);

    /* large tensors [256, 256] */
    int64_t large_sh[] = {256, 256};
    for (int i = 0; i < N; i++) {
        tensors[2 * N + i] = ax_tensor_zeros(large_sh, 2, AX_FLOAT32);
        touch_tensor(tensors[2 * N + i]);
    }
    rss = get_rss_kb();
    printf("%-20s %8d %12s %10ld\n", "large_alloc", N, "[256,256]", rss);

    /* destroy all */
    for (int i = 0; i < 3 * N; i++)
        ax_tensor_destroy(tensors[i]);
    free(tensors);

    rss = get_rss_kb();
    printf("%-20s %8s %12s %10ld\n", "after_destroy", "-", "-", rss);

    printf("%-20s %8s %12s %10ld\n", "reclaimed", "-", "-", baseline - rss);

    fflush(stdout);
}

int main(void) {
    ax_init();

#ifdef _OPENMP
    printf("OMP_NUM_THREADS=%d\n", omp_get_max_threads());
#else
    printf("OMP_NUM_THREADS=1 (no openmp)\n");
#endif

    bench_mlp_lifecycle();
    bench_cnn_lifecycle();
    bench_mha_lifecycle();
    bench_batch_sweep();
    bench_thread_rss();
    bench_alloc_density();

    return 0;
}
