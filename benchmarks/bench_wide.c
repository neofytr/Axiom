/* bench_wide.c -- wide mlp benchmark on full mnist.
   mirrors benchmarks/tf_wide.py exactly for fair comparison.

   architecture (~14.4M params):
     784 -> 4096 (bn, relu, dropout 0.3)
         -> 2048 (bn, relu, dropout 0.3)
         -> 1024 (bn, relu, dropout 0.2)
         ->  512 (relu)
         ->  256 (relu)
         ->   10

   optim: adam lr=1e-3 b1=0.9 b2=0.999 eps=1e-8
   batch: 256
   epochs: 5
   dataset: full 60000 train / 10000 test
   seed: 42

   modes (argv[1]):
     train    full training + eval
     infer    forward-only throughput (1000 batches of 256) */

#include "axiom/axiom.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define TRAIN_IMAGES "examples/data/train-images-idx3-ubyte"
#define TRAIN_LABELS "examples/data/train-labels-idx1-ubyte"
#define TEST_IMAGES  "examples/data/t10k-images-idx3-ubyte"
#define TEST_LABELS  "examples/data/t10k-labels-idx1-ubyte"

#define N_TRAIN   60000
#define N_TEST    10000
#define N_PIXELS  784
#define N_CLASSES 10
#define BATCH     256
#define EPOCHS    5

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static uint32_t read_be32(FILE *f) {
    uint8_t b[4]; if (fread(b, 1, 4, f) != 4) return 0;
    return ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|(uint32_t)b[3];
}

static ax_tensor_t *load_images(const char *path, int64_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    read_be32(f); read_be32(f); read_be32(f); read_be32(f);
    uint8_t *raw = malloc((size_t)(n * N_PIXELS));
    if (!raw || fread(raw, 1, (size_t)(n * N_PIXELS), f) != (size_t)(n * N_PIXELS))
        { free(raw); fclose(f); return NULL; }
    fclose(f);
    int64_t shape[] = {n, N_PIXELS};
    ax_tensor_t *t = ax_tensor_create(shape, 2, AX_FLOAT32);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n * N_PIXELS; i++) d[i] = (float)raw[i] / 255.0f;
    free(raw);
    return t;
}

static ax_tensor_t *load_labels_onehot(const char *path, int64_t n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    read_be32(f); read_be32(f);
    uint8_t *labels = malloc((size_t)n);
    if (!labels || fread(labels, 1, (size_t)n, f) != (size_t)n)
        { free(labels); fclose(f); return NULL; }
    fclose(f);
    int64_t shape[] = {n, N_CLASSES};
    ax_tensor_t *t = ax_tensor_zeros(shape, 2, AX_FLOAT32);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++) d[i * N_CLASSES + labels[i]] = 1.0f;
    free(labels);
    return t;
}

static uint8_t *load_labels_raw(const char *path, int64_t n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    read_be32(f); read_be32(f);
    uint8_t *labels = malloc((size_t)n);
    if (!labels || fread(labels, 1, (size_t)n, f) != (size_t)n)
        { free(labels); fclose(f); return NULL; }
    fclose(f);
    return labels;
}

static float eval_accuracy(ax_layer_t *model, ax_tensor_t *images,
                            const uint8_t *labels, int64_t n) {
    ax_layer_eval(model);
    ax_no_grad();
    int64_t correct = 0;
    const int64_t bs = 512;
    for (int64_t start = 0; start < n; start += bs) {
        int64_t b = (start + bs <= n) ? bs : (n - start);
        int64_t xshape[] = {b, N_PIXELS};
        ax_tensor_t *x = ax_tensor_create(xshape, 2, AX_FLOAT32);
        memcpy(x->storage->data,
               (float *)images->storage->data + start * N_PIXELS,
               (size_t)(b * N_PIXELS) * sizeof(float));
        ax_tensor_t *logits = ax_layer_forward(model, x);
        ax_tensor_destroy(x);
        if (!logits) break;
        float *ld = (float *)logits->storage->data;
        for (int64_t i = 0; i < b; i++) {
            int pred = 0; float best = ld[i * N_CLASSES];
            for (int c = 1; c < N_CLASSES; c++)
                if (ld[i * N_CLASSES + c] > best) { best = ld[i * N_CLASSES + c]; pred = c; }
            if (pred == (int)labels[start + i]) correct++;
        }
        ax_tensor_destroy(logits);
    }
    ax_layer_train(model);
    ax_enable_grad();
    return 100.0f * (float)correct / (float)n;
}

static ax_layer_t *build_wide(void) {
    ax_layer_t *m = ax_sequential_create();

    ax_sequential_add(m, ax_dense_create(N_PIXELS, 4096, true));
    ax_sequential_add(m, ax_batchnorm_create(4096, 1e-5f, 0.1f));
    ax_sequential_add(m, ax_relu_layer_create());
    ax_sequential_add(m, ax_dropout_create(0.3f));

    ax_sequential_add(m, ax_dense_create(4096, 2048, true));
    ax_sequential_add(m, ax_batchnorm_create(2048, 1e-5f, 0.1f));
    ax_sequential_add(m, ax_relu_layer_create());
    ax_sequential_add(m, ax_dropout_create(0.3f));

    ax_sequential_add(m, ax_dense_create(2048, 1024, true));
    ax_sequential_add(m, ax_batchnorm_create(1024, 1e-5f, 0.1f));
    ax_sequential_add(m, ax_relu_layer_create());
    ax_sequential_add(m, ax_dropout_create(0.2f));

    ax_sequential_add(m, ax_dense_create(1024, 512, true));
    ax_sequential_add(m, ax_relu_layer_create());

    ax_sequential_add(m, ax_dense_create(512, 256, true));
    ax_sequential_add(m, ax_relu_layer_create());

    ax_sequential_add(m, ax_dense_create(256, N_CLASSES, true));

    return m;
}

static void run_train(void) {
    ax_set_seed(42);

    printf("=== axiom wide-mlp train (cpu) ===\n");
    printf("threads: %d\n", ax_get_num_threads());

    ax_tensor_t *train_x = load_images(TRAIN_IMAGES, N_TRAIN);
    ax_tensor_t *train_y = load_labels_onehot(TRAIN_LABELS, N_TRAIN);
    ax_tensor_t *test_x  = load_images(TEST_IMAGES, N_TEST);
    uint8_t *test_labels = load_labels_raw(TEST_LABELS, N_TEST);
    if (!train_x || !train_y || !test_x || !test_labels)
        { fprintf(stderr, "data load failed\n"); return; }

    ax_layer_t *model = build_wide();
    int64_t param_count = ax_layer_param_count(model);
    printf("model: wide-mlp  params: %ld\n", param_count);
    printf("arch: 784 -> 4096 -> 2048 -> 1024 -> 512 -> 256 -> 10\n");
    printf("batch: %d  epochs: %d  dataset: %d train / %d test\n\n",
           BATCH, EPOCHS, N_TRAIN, N_TEST);
    fflush(stdout);

    ax_tensor_t *params[128];
    int np = ax_layer_get_params(model, params, 128);
    ax_optimizer_t *opt = ax_adam_create(params, np, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f);

    ax_dataset_t *ds = ax_tensor_dataset_create(train_x, train_y);
    ax_dataloader_t *dl = ax_dataloader_create(ds, BATCH, true);

    /* warmup batch */
    {
        ax_dataloader_reset(dl);
        ax_batch_t b;
        if (ax_dataloader_next(dl, &b)) {
            ax_enable_grad();
            ax_tensor_t *logits = ax_layer_forward(model, b.input);
            ax_tensor_t *loss = ax_cross_entropy_loss(logits, b.target);
            if (logits && loss) {
                ax_optimizer_zero_grad(opt);
                ax_backward(loss);
                ax_optimizer_step(opt);
                ax_graph_cleanup(loss);
                ax_tensor_destroy(loss);
            }
            ax_tensor_destroy(b.input);
            ax_tensor_destroy(b.target);
        }
    }

    double t_start = now_s();
    double per_epoch[EPOCHS];

    for (int ep = 0; ep < EPOCHS; ep++) {
        double t_ep = now_s();
        ax_dataloader_reset(dl);
        ax_layer_train(model);
        double total_loss = 0.0;
        int64_t n_seen = 0;
        ax_batch_t b;

        /* per-phase timings accumulated across the epoch */
        double t_fwd = 0, t_bwd = 0, t_opt = 0, t_other = 0;

        while (ax_dataloader_next(dl, &b)) {
            ax_enable_grad();
            double ta = now_s();
            ax_tensor_t *logits = ax_layer_forward(model, b.input);
            ax_tensor_t *loss = ax_cross_entropy_loss(logits, b.target);
            double tb = now_s();
            if (!logits || !loss) {
                if (logits) ax_tensor_destroy(logits);
                ax_tensor_destroy(b.input);
                ax_tensor_destroy(b.target);
                continue;
            }
            total_loss += (double)((float *)loss->storage->data)[0] * (double)b.input->shape[0];
            n_seen += b.input->shape[0];

            ax_optimizer_zero_grad(opt);
            ax_backward(loss);
            double tc = now_s();
            ax_clip_grad_norm(params, np, 1.0f);
            ax_optimizer_step(opt);
            double td = now_s();

            ax_graph_cleanup(loss);
            ax_tensor_destroy(loss);
            ax_tensor_destroy(b.input);
            ax_tensor_destroy(b.target);
            double te = now_s();

            t_fwd   += tb - ta;
            t_bwd   += tc - tb;
            t_opt   += td - tc;
            t_other += te - td;
        }

        float acc = eval_accuracy(model, test_x, test_labels, N_TEST);
        double dt = now_s() - t_ep;
        per_epoch[ep] = dt;
        printf("epoch %d/%d  loss=%.4f  test_acc=%.2f%%  time=%.3fs\n",
               ep + 1, EPOCHS,
               n_seen > 0 ? (float)(total_loss / (double)n_seen) : 0.0f,
               acc, dt);
        printf("  breakdown: fwd=%.3fs bwd=%.3fs opt=%.3fs cleanup=%.3fs\n",
               t_fwd, t_bwd, t_opt, t_other);
        fflush(stdout);
    }

    double t_total = now_s() - t_start;
    double sorted[EPOCHS];
    memcpy(sorted, per_epoch, sizeof(per_epoch));
    for (int i = 0; i < EPOCHS; i++)
        for (int j = i + 1; j < EPOCHS; j++)
            if (sorted[j] < sorted[i]) { double t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }
    double mean = 0.0;
    for (int i = 0; i < EPOCHS; i++) mean += per_epoch[i];
    mean /= EPOCHS;

    float final_acc = eval_accuracy(model, test_x, test_labels, N_TEST);
    printf("\n--- axiom wide-mlp cpu summary ---\n");
    printf("params: %ld\n", param_count);
    printf("total training time: %.3fs\n", t_total);
    printf("mean per-epoch: %.3fs\n", mean);
    printf("median per-epoch: %.3fs\n", sorted[EPOCHS / 2]);
    printf("final test accuracy: %.2f%%\n", final_acc);

    ax_optimizer_destroy(opt);
    ax_dataloader_destroy(dl);
    ax_dataset_destroy(ds);
    ax_layer_destroy(model);
    ax_tensor_destroy(train_x); ax_tensor_destroy(train_y);
    ax_tensor_destroy(test_x); free(test_labels);
}

static void run_infer(void) {
    ax_set_seed(42);
    printf("=== axiom wide-mlp infer (cpu) ===\n");
    printf("threads: %d\n", ax_get_num_threads());

    ax_layer_t *model = build_wide();
    ax_layer_eval(model);
    int64_t param_count = ax_layer_param_count(model);
    printf("model: wide-mlp  params: %ld\n", param_count);

    const int iters = 1000;
    printf("batch: %d  forward iters: %d\n\n", BATCH, iters);
    fflush(stdout);

    int64_t xshape[] = {BATCH, N_PIXELS};
    ax_tensor_t *x = ax_tensor_rand(xshape, 2, 0.0f, 1.0f);

    /* warmup */
    ax_no_grad();
    for (int i = 0; i < 10; i++) {
        ax_tensor_t *out = ax_layer_forward(model, x);
        ax_tensor_destroy(out);
    }

    double t0 = now_s();
    for (int i = 0; i < iters; i++) {
        ax_tensor_t *out = ax_layer_forward(model, x);
        ax_tensor_destroy(out);
    }
    double t1 = now_s();
    ax_enable_grad();

    double total = t1 - t0;
    double per_batch_ms = total / iters * 1000.0;
    double imgs_per_sec = (double)iters * BATCH / total;

    printf("--- axiom wide-mlp cpu forward summary ---\n");
    printf("total forward time: %.3fs\n", total);
    printf("per-batch: %.3f ms\n", per_batch_ms);
    printf("throughput: %.0f images/s\n", imgs_per_sec);

    ax_tensor_destroy(x);
    ax_layer_destroy(model);
}

int main(int argc, char **argv) {
    ax_init();
    const char *mode = argc > 1 ? argv[1] : "train";
    if (strcmp(mode, "train") == 0) run_train();
    else if (strcmp(mode, "infer") == 0) run_infer();
    else { fprintf(stderr, "usage: %s {train|infer}\n", argv[0]); return 1; }
    ax_shutdown();
    return 0;
}
