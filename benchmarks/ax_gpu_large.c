/* large mlp stress test on gpu.
   784 -> 512 -> 256 -> 128 -> 64 -> 10
   adam lr=1e-3, bs=256, 10 epochs, full 60000 training samples.
   measures throughput, accuracy, and memory stability. */

#include "axiom/axiom.h"
#include "axiom/compute.h"
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
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    read_be32(f); read_be32(f); read_be32(f); read_be32(f);
    uint8_t *raw = malloc((size_t)(n * N_PIXELS));
    if (!raw || fread(raw, 1, (size_t)(n * N_PIXELS), f) != (size_t)(n * N_PIXELS))
        { free(raw); fclose(f); return NULL; }
    fclose(f);
    int64_t shape[] = {n, N_PIXELS};
    ax_tensor_t *t = ax_tensor_create(shape, 2, AX_FLOAT32);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n * N_PIXELS; i++) d[i] = (float)raw[i] / 255.0f;
    free(raw); return t;
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
    free(labels); return t;
}

static uint8_t *load_labels_raw(const char *path, int64_t n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    read_be32(f); read_be32(f);
    uint8_t *labels = malloc((size_t)n);
    if (!labels || fread(labels, 1, (size_t)n, f) != (size_t)n)
        { free(labels); fclose(f); return NULL; }
    fclose(f); return labels;
}

static float eval_accuracy(ax_layer_t *model, ax_tensor_t *images_cpu,
                            const uint8_t *labels, int64_t n) {
    ax_layer_eval(model); ax_no_grad();
    const int64_t bs = 1024;
    int64_t correct = 0;
    for (int64_t start = 0; start < n; start += bs) {
        int64_t b = (start + bs <= n) ? bs : (n - start);
        ax_set_default_device(AX_DEVICE_CPU);
        int64_t xs[] = {b, N_PIXELS};
        ax_tensor_t *x = ax_tensor_create(xs, 2, AX_FLOAT32);
        memcpy(x->storage->data,
               (float *)images_cpu->storage->data + start * N_PIXELS,
               (size_t)(b * N_PIXELS) * sizeof(float));
        ax_tensor_t *gx = ax_tensor_to_cuda(x);
        ax_tensor_destroy(x);
        ax_set_default_device(AX_DEVICE_CUDA);
        ax_tensor_t *logits = ax_layer_forward(model, gx);
        ax_tensor_destroy(gx);
        if (!logits) continue;
        ax_tensor_t *lc = ax_tensor_to_cpu(logits);
        float *ld = (float *)lc->storage->data;
        for (int64_t i = 0; i < b; i++) {
            int pred = 0; float best = ld[i * N_CLASSES];
            for (int c = 1; c < N_CLASSES; c++)
                if (ld[i * N_CLASSES + c] > best) { best = ld[i * N_CLASSES + c]; pred = c; }
            if (pred == (int)labels[start + i]) correct++;
        }
        ax_tensor_destroy(lc);
        ax_tensor_destroy(logits);
    }
    ax_layer_train(model); ax_enable_grad();
    return 100.0f * (float)correct / (float)n;
}

int main(void) {
    ax_init(); ax_set_seed(42);

    printf("loading full mnist (60k train, 10k test)...\n"); fflush(stdout);
    ax_tensor_t *train_x = load_images(TRAIN_IMAGES, N_TRAIN);
    ax_tensor_t *train_y = load_labels_onehot(TRAIN_LABELS, N_TRAIN);
    ax_tensor_t *test_x  = load_images(TEST_IMAGES, N_TEST);
    uint8_t *test_labels = load_labels_raw(TEST_LABELS, N_TEST);
    if (!train_x || !train_y || !test_x || !test_labels)
        { fprintf(stderr, "data load failed\n"); return 1; }

    ax_compute_set_backend(AX_BACKEND_CUDA);
    ax_set_default_device(AX_DEVICE_CUDA);
    printf("backend: CUDA\n");

    /* deep mlp: 784 -> 512 -> 256 -> 128 -> 64 -> 10 */
    ax_layer_t *model = ax_sequential_create();
    ax_sequential_add(model, ax_dense_create(N_PIXELS, 512, true));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_dense_create(512, 256, true));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_dense_create(256, 128, true));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_dense_create(128, 64, true));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_dense_create(64, N_CLASSES, true));

    ax_tensor_t *params[64];
    int np = ax_layer_get_params(model, params, 64);
    ax_optimizer_t *opt = ax_adam_create(params, np, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f);

    ax_set_default_device(AX_DEVICE_CPU);
    ax_dataset_t *ds = ax_tensor_dataset_create(train_x, train_y);
    ax_dataloader_t *dl = ax_dataloader_create(ds, 256, true);

    int64_t param_count = ax_layer_param_count(model);
    int64_t n_batches = ax_dataloader_num_batches(dl);
    printf("model: %ld parameters (%ld layers)\n", param_count, (long)5);
    printf("arch: 784 -> 512 -> 256 -> 128 -> 64 -> 10\n");
    printf("training: %ld batches/epoch, %d samples\n\n", n_batches, N_TRAIN);
    fflush(stdout);

    /* warmup */
    {
        ax_dataloader_reset(dl); ax_batch_t b;
        if (ax_dataloader_next(dl, &b)) {
            ax_tensor_t *gi = ax_tensor_to_cuda(b.input);
            ax_tensor_t *gt = ax_tensor_to_cuda(b.target);
            ax_tensor_destroy(b.input); ax_tensor_destroy(b.target);
            ax_set_default_device(AX_DEVICE_CUDA);
            ax_enable_grad();
            ax_tensor_t *logits = ax_layer_forward(model, gi);
            ax_tensor_t *loss = ax_cross_entropy_loss(logits, gt);
            if (logits && loss) {
                ax_optimizer_zero_grad(opt);
                ax_backward(loss);
                ax_optimizer_step(opt);
                ax_graph_cleanup(loss);
                ax_tensor_destroy(loss);
            }
            ax_tensor_destroy(gi); ax_tensor_destroy(gt);
            ax_set_default_device(AX_DEVICE_CPU);
        }
    }

    const int epochs = 10;
    double t_start = now_s();
    double per_epoch[32];
    int64_t total_samples = 0;

    for (int ep = 0; ep < epochs; ep++) {
        double t_ep = now_s();
        ax_set_default_device(AX_DEVICE_CPU);
        ax_dataloader_reset(dl); ax_layer_train(model);
        double total_loss = 0.0; int64_t n_seen = 0;
        ax_batch_t b;

        while (ax_dataloader_next(dl, &b)) {
            ax_tensor_t *gi = ax_tensor_to_cuda(b.input);
            ax_tensor_t *gt = ax_tensor_to_cuda(b.target);
            ax_tensor_destroy(b.input); ax_tensor_destroy(b.target);
            ax_set_default_device(AX_DEVICE_CUDA);
            ax_enable_grad();

            ax_tensor_t *logits = ax_layer_forward(model, gi);
            ax_tensor_t *loss = ax_cross_entropy_loss(logits, gt);
            if (!logits || !loss) {
                if (logits) ax_tensor_destroy(logits);
                ax_tensor_destroy(gi); ax_tensor_destroy(gt);
                continue;
            }

            int64_t i0[] = {0};
            total_loss += (double)ax_tensor_get_f32(loss, i0) * (double)gi->shape[0];
            n_seen += gi->shape[0];

            ax_optimizer_zero_grad(opt);
            ax_backward(loss);
            ax_optimizer_step(opt);
            ax_graph_cleanup(loss);
            ax_tensor_destroy(loss);
            ax_tensor_destroy(gi);
            ax_tensor_destroy(gt);
            ax_set_default_device(AX_DEVICE_CPU);
        }

        total_samples += n_seen;
        float acc = eval_accuracy(model, test_x, test_labels, N_TEST);
        double dt = now_s() - t_ep;
        per_epoch[ep] = dt;
        double samples_per_sec = (double)n_seen / dt;
        printf("epoch %2d/%d  loss=%.4f  test_acc=%.2f%%  time=%.3fs  throughput=%.0f samples/s\n",
               ep + 1, epochs,
               n_seen > 0 ? total_loss / (double)n_seen : 0.0,
               acc, dt, samples_per_sec);
        fflush(stdout);
    }

    double t_total = now_s() - t_start;
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
    double avg_throughput = (double)total_samples / t_total;

    printf("\n--- large model gpu summary ---\n");
    printf("architecture: 784 -> 512 -> 256 -> 128 -> 64 -> 10\n");
    printf("parameters: %ld\n", param_count);
    printf("training samples: %d x %d epochs = %ld total\n", N_TRAIN, epochs, total_samples);
    printf("total training time: %.3fs\n", t_total);
    printf("mean per-epoch: %.3fs\n", mean);
    printf("median per-epoch: %.3fs\n", sorted[epochs / 2]);
    printf("avg throughput: %.0f samples/s\n", avg_throughput);
    printf("final test accuracy: %.2f%%\n", final_acc);
    return 0;
}
