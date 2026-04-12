/* ax_mnist.c — mnist mlp benchmark mirroring benchmarks/tf_mnist.py exactly.

   model: 784 -> 128 relu -> 64 relu -> 10
   optim: adam lr=1e-3 b1=0.9 b2=0.999 eps=1e-8
   batch: 256
   epochs: 8
   subset: 10000 train / 10000 test
   seed: 42

   timer starts after data load + one warmup batch and stops after the
   last training iteration. exactly the wall-clock slice tf_mnist.py
   reports. per-epoch breakdown + final test accuracy printed. */

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

#define N_TRAIN   10000
#define N_TEST    10000
#define N_PIXELS  784
#define N_CLASSES 10

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static uint32_t read_be32(FILE *f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] << 8)  |  (uint32_t)b[3];
}

static ax_tensor_t *load_images(const char *path, int64_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    read_be32(f); read_be32(f); read_be32(f); read_be32(f);
    uint8_t *raw = malloc((size_t)(n * N_PIXELS));
    if (!raw || fread(raw, 1, (size_t)(n * N_PIXELS), f) != (size_t)(n * N_PIXELS)) {
        free(raw); fclose(f); return NULL;
    }
    fclose(f);
    int64_t shape[] = {n, N_PIXELS};
    ax_tensor_t *t = ax_tensor_create(shape, 2, AX_FLOAT32);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n * N_PIXELS; i++) d[i] = (float)raw[i] / 255.0f;
    free(raw);
    return t;
}

static ax_tensor_t *load_labels_onehot(const char *path, int64_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    read_be32(f); read_be32(f);
    uint8_t *labels = malloc((size_t)n);
    if (!labels || fread(labels, 1, (size_t)n, f) != (size_t)n) {
        free(labels); fclose(f); return NULL;
    }
    fclose(f);
    int64_t shape[] = {n, N_CLASSES};
    ax_tensor_t *t = ax_tensor_zeros(shape, 2, AX_FLOAT32);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++) d[i * N_CLASSES + labels[i]] = 1.0f;
    free(labels);
    return t;
}

static uint8_t *load_labels_raw(const char *path, int64_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    read_be32(f); read_be32(f);
    uint8_t *labels = malloc((size_t)n);
    if (!labels || fread(labels, 1, (size_t)n, f) != (size_t)n) {
        free(labels); fclose(f); return NULL;
    }
    fclose(f);
    return labels;
}

static float eval_accuracy(ax_layer_t *model, ax_tensor_t *images, const uint8_t *labels, int64_t n) {
    ax_layer_eval(model);
    ax_no_grad();
    const int64_t bs = 512;
    int64_t correct = 0;
    for (int64_t start = 0; start < n; start += bs) {
        int64_t b = (start + bs <= n) ? bs : (n - start);
        int64_t xshape[] = {b, N_PIXELS};
        ax_tensor_t *x = ax_tensor_create(xshape, 2, AX_FLOAT32);
        float *xd = (float *)x->storage->data;
        float *id = (float *)images->storage->data;
        memcpy(xd, id + start * N_PIXELS, (size_t)(b * N_PIXELS) * sizeof(float));
        ax_tensor_t *logits = ax_layer_forward(model, x);
        ax_tensor_destroy(x);
        float *ld = (float *)logits->storage->data;
        for (int64_t i = 0; i < b; i++) {
            int pred = 0; float best = ld[i * N_CLASSES];
            for (int c = 1; c < N_CLASSES; c++) if (ld[i * N_CLASSES + c] > best) { best = ld[i * N_CLASSES + c]; pred = c; }
            if (pred == (int)labels[start + i]) correct++;
        }
        ax_tensor_destroy(logits);
    }
    ax_layer_train(model);
    ax_enable_grad();
    return 100.0f * (float)correct / (float)n;
}

int main(void) {
    ax_init();
    ax_set_seed(42);

    printf("threads: %d\n", ax_get_num_threads());
    printf("loading mnist...\n"); fflush(stdout);

    ax_tensor_t *train_x = load_images(TRAIN_IMAGES, N_TRAIN);
    ax_tensor_t *train_y = load_labels_onehot(TRAIN_LABELS, N_TRAIN);
    ax_tensor_t *test_x  = load_images(TEST_IMAGES,  N_TEST);
    uint8_t *test_labels = load_labels_raw(TEST_LABELS, N_TEST);
    if (!train_x || !train_y || !test_x || !test_labels) {
        fprintf(stderr, "mnist data load failed\n"); return 1;
    }
    printf("  train: %d  test: %d  pixels: %d\n", N_TRAIN, N_TEST, N_PIXELS);

    ax_layer_t *model = ax_sequential_create();
    ax_sequential_add(model, ax_dense_create(N_PIXELS,   128, true));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_dense_create(128,         64, true));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_dense_create(64, N_CLASSES, true));

    ax_tensor_t *params[32];
    int n_params = ax_layer_get_params(model, params, 32);
    ax_optimizer_t *opt = ax_adam_create(params, n_params, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f);

    ax_dataset_t *ds = ax_tensor_dataset_create(train_x, train_y);
    ax_dataloader_t *dl = ax_dataloader_create(ds, 256, true);

    printf("model: %ld parameters\n", ax_layer_param_count(model));
    printf("training: %ld batches/epoch\n\n", ax_dataloader_num_batches(dl));
    fflush(stdout);

    /* warmup: one training step to initialize adam state + prime caches */
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

    const int epochs = 8;
    double t_start = now_s();
    double per_epoch[32];

    for (int ep = 0; ep < epochs; ep++) {
        double t_ep = now_s();
        ax_dataloader_reset(dl);
        ax_layer_train(model);
        double total_loss = 0.0;
        int64_t n_seen = 0;
        ax_batch_t b;
        while (ax_dataloader_next(dl, &b)) {
            ax_enable_grad();
            ax_tensor_t *logits = ax_layer_forward(model, b.input);
            ax_tensor_t *loss = ax_cross_entropy_loss(logits, b.target);
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
            ax_optimizer_step(opt);
            ax_graph_cleanup(loss);
            ax_tensor_destroy(loss);
            ax_tensor_destroy(b.input);
            ax_tensor_destroy(b.target);
        }
        float acc = eval_accuracy(model, test_x, test_labels, N_TEST);
        double dt = now_s() - t_ep;
        per_epoch[ep] = dt;
        printf("epoch %d/%d  loss=%.4f  test_acc=%.2f%%  time=%.3fs\n",
               ep + 1, epochs, total_loss / (double)n_seen, acc, dt);
    }
    double t_total = now_s() - t_start;

    /* sort per_epoch copy for median */
    double sorted[32];
    memcpy(sorted, per_epoch, (size_t)epochs * sizeof(double));
    for (int i = 1; i < epochs; i++) {
        double x = sorted[i]; int j = i;
        while (j > 0 && sorted[j-1] > x) { sorted[j] = sorted[j-1]; j--; }
        sorted[j] = x;
    }
    double mean = 0.0;
    for (int i = 0; i < epochs; i++) mean += per_epoch[i];
    mean /= epochs;

    float final_acc = eval_accuracy(model, test_x, test_labels, N_TEST);
    printf("\n--- axiom summary ---\n");
    printf("total training time: %.3fs\n", t_total);
    printf("mean per-epoch: %.3fs\n", mean);
    printf("median per-epoch: %.3fs\n", sorted[epochs / 2]);
    printf("final test accuracy: %.2f%%\n", final_acc);

    return 0;
}
