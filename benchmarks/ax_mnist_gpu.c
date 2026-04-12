/* ax_mnist_gpu.c — mnist mlp training on gpu via cuda backend.

   same model and hyperparameters as ax_mnist.c (cpu benchmark):
   784 -> 128 relu -> 64 relu -> 10, adam lr=1e-3, bs=256, 8 epochs.

   gpu training flow:
   1. data loaded on cpu (disk i/o is always cpu-side)
   2. model created on gpu (default device = cuda during layer creation)
   3. per batch: cpu batch tensors -> ax_tensor_to_cuda -> forward ->
      loss -> backward -> optimizer step (all on gpu)
   4. eval: forward on gpu, read logits back to cpu for accuracy count */

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
    uint8_t b[4]; if (fread(b, 1, 4, f) != 4) return 0;
    return ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|(uint32_t)b[3];
}

static ax_tensor_t *load_images(const char *path, int64_t n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    read_be32(f); read_be32(f); read_be32(f); read_be32(f);
    uint8_t *raw = malloc((size_t)(n*N_PIXELS));
    if (!raw || fread(raw, 1, (size_t)(n*N_PIXELS), f) != (size_t)(n*N_PIXELS)) { free(raw); fclose(f); return NULL; }
    fclose(f);
    int64_t shape[] = {n, N_PIXELS};
    ax_tensor_t *t = ax_tensor_create(shape, 2, AX_FLOAT32);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n*N_PIXELS; i++) d[i] = (float)raw[i] / 255.0f;
    free(raw); return t;
}

static ax_tensor_t *load_labels_onehot(const char *path, int64_t n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    read_be32(f); read_be32(f);
    uint8_t *labels = malloc((size_t)n);
    if (!labels || fread(labels, 1, (size_t)n, f) != (size_t)n) { free(labels); fclose(f); return NULL; }
    fclose(f);
    int64_t shape[] = {n, N_CLASSES};
    ax_tensor_t *t = ax_tensor_zeros(shape, 2, AX_FLOAT32);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++) d[i*N_CLASSES + labels[i]] = 1.0f;
    free(labels); return t;
}

static uint8_t *load_labels_raw(const char *path, int64_t n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    read_be32(f); read_be32(f);
    uint8_t *labels = malloc((size_t)n);
    if (!labels || fread(labels, 1, (size_t)n, f) != (size_t)n) { free(labels); fclose(f); return NULL; }
    fclose(f); return labels;
}

/* evaluate accuracy: forward on gpu, read logits back to cpu */
static float eval_accuracy(ax_layer_t *model, ax_tensor_t *images_cpu,
                            const uint8_t *labels, int64_t n) {
    ax_layer_eval(model); ax_no_grad();
    const int64_t bs = 512;
    int64_t correct = 0;
    for (int64_t start = 0; start < n; start += bs) {
        int64_t b = (start + bs <= n) ? bs : (n - start);
        /* create batch on cpu, copy data, move to gpu */
        ax_device_t prev = ax_get_default_device();
        ax_set_default_device(AX_DEVICE_CPU);
        int64_t xs[] = {b, N_PIXELS};
        ax_tensor_t *x = ax_tensor_create(xs, 2, AX_FLOAT32);
        memcpy(x->storage->data, (float*)images_cpu->storage->data + start*N_PIXELS,
               (size_t)(b*N_PIXELS)*sizeof(float));
        ax_set_default_device(prev);

#ifdef AX_HAVE_CUDA
        ax_tensor_t *gx = ax_tensor_to_cuda(x);
        ax_tensor_destroy(x);
#else
        ax_tensor_t *gx = x;
#endif
        ax_tensor_t *logits = ax_layer_forward(model, gx);
        ax_tensor_destroy(gx);
        if (!logits) continue;

        /* read logits back to cpu for argmax */
#ifdef AX_HAVE_CUDA
        ax_tensor_t *logits_cpu = ax_tensor_to_cpu(logits);
#else
        ax_tensor_t *logits_cpu = logits;
#endif
        float *ld = (float *)logits_cpu->storage->data;
        for (int64_t i = 0; i < b; i++) {
            int pred = 0; float best = ld[i*N_CLASSES];
            for (int c = 1; c < N_CLASSES; c++) if (ld[i*N_CLASSES+c] > best) { best = ld[i*N_CLASSES+c]; pred = c; }
            if (pred == (int)labels[start+i]) correct++;
        }
#ifdef AX_HAVE_CUDA
        ax_tensor_destroy(logits_cpu);
#endif
        ax_tensor_destroy(logits);
    }
    ax_layer_train(model); ax_enable_grad();
    return 100.0f * (float)correct / (float)n;
}

int main(void) {
    ax_init();
    ax_set_seed(42);

    /* step 1: load data on cpu */
    printf("loading mnist (cpu)...\n"); fflush(stdout);
    ax_tensor_t *train_x = load_images(TRAIN_IMAGES, N_TRAIN);
    ax_tensor_t *train_y = load_labels_onehot(TRAIN_LABELS, N_TRAIN);
    ax_tensor_t *test_x  = load_images(TEST_IMAGES, N_TEST);
    uint8_t *test_labels = load_labels_raw(TEST_LABELS, N_TEST);
    if (!train_x || !train_y || !test_x || !test_labels) { fprintf(stderr, "load failed\n"); return 1; }
    printf("  train: %d  test: %d  pixels: %d\n", N_TRAIN, N_TEST, N_PIXELS);

#ifdef AX_HAVE_CUDA
    /* step 2: switch to cuda for model creation */
    ax_compute_set_backend(AX_BACKEND_CUDA);
    ax_set_default_device(AX_DEVICE_CUDA);
    printf("backend: CUDA (gpu)\n");
#else
    printf("backend: CPU (compiled without CUDA)\n");
#endif

    ax_layer_t *model = ax_sequential_create();
    ax_sequential_add(model, ax_dense_create(N_PIXELS, 128, true));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_dense_create(128, 64, true));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_dense_create(64, N_CLASSES, true));

    ax_tensor_t *params[32];
    int n_params = ax_layer_get_params(model, params, 32);
    ax_optimizer_t *opt = ax_adam_create(params, n_params, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f);

    /* step 3: dataloader works on cpu source data */
    ax_set_default_device(AX_DEVICE_CPU);
    ax_dataset_t *ds = ax_tensor_dataset_create(train_x, train_y);
    ax_dataloader_t *dl = ax_dataloader_create(ds, 256, true);

    printf("model: %ld parameters\n", ax_layer_param_count(model));
    printf("training: %ld batches/epoch\n\n", ax_dataloader_num_batches(dl)); fflush(stdout);

    /* warmup: one training step */
    {
        ax_dataloader_reset(dl); ax_batch_t b;
        if (ax_dataloader_next(dl, &b)) {
#ifdef AX_HAVE_CUDA
            ax_tensor_t *gi = ax_tensor_to_cuda(b.input);
            ax_tensor_t *gt = ax_tensor_to_cuda(b.target);
            ax_tensor_destroy(b.input); ax_tensor_destroy(b.target);
            ax_set_default_device(AX_DEVICE_CUDA);
#else
            ax_tensor_t *gi = b.input, *gt = b.target;
#endif
            ax_enable_grad();
            ax_tensor_t *logits = ax_layer_forward(model, gi);
            ax_tensor_t *loss = ax_cross_entropy_loss(logits, gt);
            if (logits && loss) {
                ax_optimizer_zero_grad(opt); ax_backward(loss);
                ax_optimizer_step(opt); ax_graph_cleanup(loss); ax_tensor_destroy(loss);
            }
            ax_tensor_destroy(gi); ax_tensor_destroy(gt);
            ax_set_default_device(AX_DEVICE_CPU);
        }
    }

    const int epochs = 8;
    double t_start = now_s();
    double per_epoch[32];

    for (int ep = 0; ep < epochs; ep++) {
        double t_ep = now_s();
        ax_set_default_device(AX_DEVICE_CPU);
        ax_dataloader_reset(dl); ax_layer_train(model);
        double total_loss = 0.0; int64_t n_seen = 0;
        ax_batch_t b;
        while (ax_dataloader_next(dl, &b)) {
#ifdef AX_HAVE_CUDA
            ax_tensor_t *gi = ax_tensor_to_cuda(b.input);
            ax_tensor_t *gt = ax_tensor_to_cuda(b.target);
            ax_tensor_destroy(b.input); ax_tensor_destroy(b.target);
            ax_set_default_device(AX_DEVICE_CUDA);
#else
            ax_tensor_t *gi = b.input, *gt = b.target;
#endif
            ax_enable_grad();
            ax_tensor_t *logits = ax_layer_forward(model, gi);
            ax_tensor_t *loss = ax_cross_entropy_loss(logits, gt);
            if (!logits || !loss) {
                if (logits) ax_tensor_destroy(logits);
                ax_tensor_destroy(gi); ax_tensor_destroy(gt); continue;
            }
            /* read scalar loss via d2h round-trip */
            int64_t i0[] = {0};
            total_loss += (double)ax_tensor_get_f32(loss, i0) * (double)gi->shape[0];
            n_seen += gi->shape[0];
            ax_optimizer_zero_grad(opt); ax_backward(loss);
            ax_optimizer_step(opt); ax_graph_cleanup(loss); ax_tensor_destroy(loss);
            ax_tensor_destroy(gi); ax_tensor_destroy(gt);
        }
        float acc = eval_accuracy(model, test_x, test_labels, N_TEST);
        double dt = now_s() - t_ep;
        per_epoch[ep] = dt;
        printf("epoch %d/%d  loss=%.4f  test_acc=%.2f%%  time=%.3fs\n",
               ep+1, epochs, total_loss/(double)n_seen, acc, dt);
    }
    double t_total = now_s() - t_start;

    double sorted[32]; memcpy(sorted, per_epoch, (size_t)epochs*sizeof(double));
    for (int i = 1; i < epochs; i++) { double x = sorted[i]; int j = i; while (j > 0 && sorted[j-1] > x) { sorted[j] = sorted[j-1]; j--; } sorted[j] = x; }
    double mean = 0.0; for (int i = 0; i < epochs; i++) mean += per_epoch[i]; mean /= epochs;
    float final_acc = eval_accuracy(model, test_x, test_labels, N_TEST);

    printf("\n--- axiom-gpu summary ---\n");
    printf("total training time: %.3fs\n", t_total);
    printf("mean per-epoch: %.3fs\n", mean);
    printf("median per-epoch: %.3fs\n", sorted[epochs/2]);
    printf("final test accuracy: %.2f%%\n", final_acc);
    return 0;
}
