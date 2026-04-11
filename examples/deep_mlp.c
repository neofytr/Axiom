/* deep_mlp.c — deep MLP on full MNIST (60k) demonstrating a production-scale
   training pipeline: BatchNorm, Dropout, gradient clipping, warmup+cosine LR.

   architecture  (~4.4M parameters):
     784 → 2048 (bn, relu, dropout 0.3)
         → 1024 (bn, relu, dropout 0.3)
         →  512 (bn, relu, dropout 0.2)
         →  256 (relu)
         →   10 (softmax)

   training:
     optimizer : adam  lr=1e-3, weight_decay=1e-4
     schedule  : linear warmup (3 epochs) + cosine decay to 1e-5 (20 epochs total)
     grad clip : global norm ≤ 1.0
     batch size: 128

   expected results (~20 epochs on 60k MNIST):
     test accuracy 98.5–99.0%

   run from project root:
     ./build/deep_mlp */

#include "axiom/axiom.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* ── paths ─────────────────────────────────────────────────────────────── */

#define TRAIN_IMAGES "examples/data/train-images-idx3-ubyte"
#define TRAIN_LABELS "examples/data/train-labels-idx1-ubyte"
#define TEST_IMAGES  "examples/data/t10k-images-idx3-ubyte"
#define TEST_LABELS  "examples/data/t10k-labels-idx1-ubyte"

#define N_TRAIN   60000
#define N_TEST    10000
#define N_PIXELS  784
#define N_CLASSES 10

/* ── data loading ───────────────────────────────────────────────────────── */

static uint32_t read_be32(FILE *f)
{
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] <<  8) |  (uint32_t)b[3];
}

static ax_tensor_t *load_images(const char *path, int64_t n)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    if (read_be32(f) != 0x00000803) { fclose(f); return NULL; }
    read_be32(f);
    int64_t rows = (int64_t)read_be32(f);
    int64_t cols = (int64_t)read_be32(f);
    if (rows * cols != N_PIXELS) { fclose(f); return NULL; }

    uint8_t *raw = malloc((size_t)(n * N_PIXELS));
    if (!raw) { fclose(f); return NULL; }
    if (fread(raw, 1, (size_t)(n * N_PIXELS), f) != (size_t)(n * N_PIXELS))
    { free(raw); fclose(f); return NULL; }
    fclose(f);

    int64_t shape[] = {n, N_PIXELS};
    ax_tensor_t *t = ax_tensor_create(shape, 2, AX_FLOAT32);
    if (!t) { free(raw); return NULL; }
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n * N_PIXELS; i++)
        d[i] = (float)raw[i] / 255.0f;
    free(raw);
    return t;
}

static ax_tensor_t *load_labels_onehot(const char *path, int64_t n)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    if (read_be32(f) != 0x00000801) { fclose(f); return NULL; }
    read_be32(f);

    uint8_t *raw = malloc((size_t)n);
    if (!raw) { fclose(f); return NULL; }
    if (fread(raw, 1, (size_t)n, f) != (size_t)n) { free(raw); fclose(f); return NULL; }
    fclose(f);

    int64_t shape[] = {n, N_CLASSES};
    ax_tensor_t *t = ax_tensor_zeros(shape, 2, AX_FLOAT32);
    if (!t) { free(raw); return NULL; }
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++)
        d[i * N_CLASSES + raw[i]] = 1.0f;
    free(raw);
    return t;
}

static uint8_t *load_labels_raw(const char *path, int64_t n)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    read_be32(f); read_be32(f);
    uint8_t *labels = malloc((size_t)n);
    if (!labels) { fclose(f); return NULL; }
    if (fread(labels, 1, (size_t)n, f) != (size_t)n)
    { free(labels); fclose(f); return NULL; }
    fclose(f);
    return labels;
}

/* ── evaluation ─────────────────────────────────────────────────────────── */

static float eval_accuracy(ax_layer_t *net, ax_tensor_t *images,
                            const uint8_t *labels, int64_t n)
{
    ax_layer_eval(net);
    ax_no_grad();

    const int64_t bs = 512;
    int64_t correct = 0;
    float *imgd = (float *)images->storage->data;

    for (int64_t start = 0; start < n; start += bs)
    {
        int64_t b = (start + bs <= n) ? bs : (n - start);
        int64_t xshape[] = {b, N_PIXELS};
        ax_tensor_t *x = ax_tensor_create(xshape, 2, AX_FLOAT32);
        if (!x) break;
        memcpy((float *)x->storage->data, imgd + start * N_PIXELS,
               (size_t)(b * N_PIXELS) * sizeof(float));

        ax_tensor_t *logits = ax_layer_forward(net, x);
        ax_tensor_destroy(x);
        if (!logits) break;

        float *ld = (float *)logits->storage->data;
        for (int64_t i = 0; i < b; i++)
        {
            int pred = 0;
            float best = ld[i * N_CLASSES];
            for (int c = 1; c < N_CLASSES; c++)
                if (ld[i * N_CLASSES + c] > best)
                    { best = ld[i * N_CLASSES + c]; pred = c; }
            if (pred == (int)labels[start + i]) correct++;
        }
        ax_tensor_destroy(logits);
    }

    ax_layer_train(net);
    ax_enable_grad();
    return 100.0f * (float)correct / (float)n;
}

/* ── main ───────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    ax_init();
    ax_set_seed(42);

    /* load data */
    printf("loading MNIST (60k)...\n"); fflush(stdout);
    ax_tensor_t *train_x = load_images(TRAIN_IMAGES, N_TRAIN);
    ax_tensor_t *train_y = load_labels_onehot(TRAIN_LABELS, N_TRAIN);
    ax_tensor_t *test_x  = load_images(TEST_IMAGES,  N_TEST);
    uint8_t *train_labels = load_labels_raw(TRAIN_LABELS, N_TRAIN);
    uint8_t *test_labels  = load_labels_raw(TEST_LABELS,  N_TEST);
    if (!train_x || !train_y || !test_x || !train_labels || !test_labels)
    {
        fprintf(stderr, "failed to load data — run from project root\n");
        return 1;
    }

    /* architecture:
         784 → 2048 (bn, relu, dropout 0.3)
             → 1024 (bn, relu, dropout 0.3)
             →  512 (bn, relu, dropout 0.2)
             →  256 (relu)
             →   10                          */
    ax_layer_t *net = ax_sequential_create();

    ax_sequential_add(net, ax_dense_create(N_PIXELS, 2048, true));
    ax_sequential_add(net, ax_batchnorm_create(2048, 1e-5f, 0.1f));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dropout_create(0.3f));

    ax_sequential_add(net, ax_dense_create(2048, 1024, true));
    ax_sequential_add(net, ax_batchnorm_create(1024, 1e-5f, 0.1f));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dropout_create(0.3f));

    ax_sequential_add(net, ax_dense_create(1024, 512, true));
    ax_sequential_add(net, ax_batchnorm_create(512, 1e-5f, 0.1f));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dropout_create(0.2f));

    ax_sequential_add(net, ax_dense_create(512, 256, true));
    ax_sequential_add(net, ax_relu_layer_create());

    ax_sequential_add(net, ax_dense_create(256, N_CLASSES, true));

    /* optimizer and lr schedule */
    ax_tensor_t *params[64];
    int n_params = ax_layer_get_params(net, params, 64);

    ax_optimizer_t *opt = ax_adam_create(params, n_params,
                                          1e-3f, 0.9f, 0.999f, 1e-8f, 1e-4f);

    const int epochs       = (argc > 1) ? atoi(argv[1]) : 20;
    const int warmup_steps = 3;      /* warmup for 3 epochs */
    int64_t batches_per_epoch = (N_TRAIN + 127) / 128;
    int total_steps = epochs * (int)batches_per_epoch;
    int warm_steps  = warmup_steps * (int)batches_per_epoch;
    ax_lr_scheduler_t *sched = ax_sched_warmup_cosine(opt, warm_steps,
                                                       total_steps, 1e-5f);

    /* data pipeline */
    ax_dataset_t  *ds = ax_tensor_dataset_create(train_x, train_y);
    ax_dataloader_t *dl = ax_dataloader_create(ds, 128, true);

    printf("parameters : %ld\n", ax_layer_param_count(net));
    printf("batches/ep : %ld\n", ax_dataloader_num_batches(dl));
    printf("epochs     : %d\n\n", epochs);
    fflush(stdout);

    /* training loop */
    for (int ep = 0; ep < epochs; ep++)
    {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        ax_dataloader_reset(dl);
        ax_layer_train(net);

        double total_loss = 0.0;
        int64_t n_batches = 0;
        ax_batch_t batch;

        while (ax_dataloader_next(dl, &batch))
        {
            ax_enable_grad();
            ax_tensor_t *logits = ax_layer_forward(net, batch.input);
            ax_tensor_t *loss   = ax_cross_entropy_loss(logits, batch.target);

            if (!logits || !loss)
            {
                if (logits) ax_tensor_destroy(logits);
                ax_tensor_destroy(batch.input);
                ax_tensor_destroy(batch.target);
                continue;
            }

            total_loss += (double)((float *)loss->storage->data)[0];
            n_batches++;

            ax_optimizer_zero_grad(opt);
            ax_backward(loss);
            ax_clip_grad_norm(params, n_params, 1.0f);
            ax_optimizer_step(opt);
            ax_sched_step(sched);

            ax_graph_cleanup(loss);
            ax_tensor_destroy(loss);
            ax_tensor_destroy(batch.input);
            ax_tensor_destroy(batch.target);
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (double)(t1.tv_sec - t0.tv_sec) +
                         (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;

        float train_acc = eval_accuracy(net, train_x, train_labels, N_TRAIN);
        float test_acc  = eval_accuracy(net, test_x,  test_labels,  N_TEST);

        printf("epoch %2d/%d  loss=%.4f  train=%.2f%%  test=%.2f%%  "
               "lr=%.2e  time=%.1fs\n",
               ep + 1, epochs,
               (float)(total_loss / (double)n_batches),
               train_acc, test_acc,
               (double)ax_sched_get_lr(sched),
               elapsed);
        fflush(stdout);
    }

    /* save */
    ax_model_t *wrapper = ax_model_create(net);
    if (ax_model_save(wrapper, "deep_mlp.axm") == AX_OK)
        printf("\nmodel saved to deep_mlp.axm\n");
    ax_model_destroy(wrapper);

    /* cleanup */
    ax_sched_destroy(sched);
    ax_dataloader_destroy(dl);
    ax_dataset_destroy(ds);
    ax_optimizer_destroy(opt);
    ax_tensor_destroy(train_x);
    ax_tensor_destroy(train_y);
    ax_tensor_destroy(test_x);
    free(train_labels);
    free(test_labels);
    ax_shutdown();
    return 0;
}
