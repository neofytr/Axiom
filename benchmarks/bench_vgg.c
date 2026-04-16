/* bench_vgg.c -- vgg-style deep cnn benchmark on full mnist.
   mirrors benchmarks/tf_vgg.py exactly for fair comparison.

   architecture (~8M params):
     block 1: conv(1,64,3,pad=1) bn relu conv(64,64,3,pad=1) bn relu maxpool(2)
     block 2: conv(64,128,3,pad=1) bn relu conv(128,128,3,pad=1) bn relu maxpool(2)
     block 3: conv(128,256,3,pad=1) bn relu conv(256,256,3,pad=1) bn relu maxpool(2)
     classifier: flatten dense(2304,2048) relu dropout(0.5)
                 dense(2048,1024) relu dropout(0.5) dense(1024,10)

   optim: adam lr=1e-3 b1=0.9 b2=0.999 eps=1e-8
   batch: 64
   epochs: 5
   dataset: full 60000 train / 10000 test
   seed: 42

   modes (argv[1]):
     train    full training + eval
     infer    forward-only throughput (500 batches of 64) */

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
#define IMG_H     28
#define IMG_W     28
#define N_PIXELS  (IMG_H * IMG_W)
#define N_CLASSES 10
#define BATCH     64
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

static ax_tensor_t *load_images_4d(const char *path, int64_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    read_be32(f); read_be32(f); read_be32(f); read_be32(f);
    uint8_t *raw = malloc((size_t)(n * N_PIXELS));
    if (!raw || fread(raw, 1, (size_t)(n * N_PIXELS), f) != (size_t)(n * N_PIXELS))
        { free(raw); fclose(f); return NULL; }
    fclose(f);
    int64_t shape[] = {n, 1, IMG_H, IMG_W};
    ax_tensor_t *t = ax_tensor_create(shape, 4, AX_FLOAT32);
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
    const int64_t bs = 256;
    for (int64_t start = 0; start < n; start += bs) {
        int64_t b = (start + bs <= n) ? bs : (n - start);
        int64_t xshape[] = {b, 1, IMG_H, IMG_W};
        ax_tensor_t *x = ax_tensor_create(xshape, 4, AX_FLOAT32);
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

static ax_layer_t *build_vgg(void) {
    ax_layer_t *m = ax_sequential_create();

    /* block 1: -> [N,64,14,14] */
    ax_sequential_add(m, ax_conv2d_create(1, 64, 3, 1, 1, true));
    ax_sequential_add(m, ax_batchnorm_create(64, 1e-5f, 0.1f));
    ax_sequential_add(m, ax_relu_layer_create());
    ax_sequential_add(m, ax_conv2d_create(64, 64, 3, 1, 1, true));
    ax_sequential_add(m, ax_batchnorm_create(64, 1e-5f, 0.1f));
    ax_sequential_add(m, ax_relu_layer_create());
    ax_sequential_add(m, ax_maxpool2d_create(2, 2, 0));

    /* block 2: -> [N,128,7,7] */
    ax_sequential_add(m, ax_conv2d_create(64, 128, 3, 1, 1, true));
    ax_sequential_add(m, ax_batchnorm_create(128, 1e-5f, 0.1f));
    ax_sequential_add(m, ax_relu_layer_create());
    ax_sequential_add(m, ax_conv2d_create(128, 128, 3, 1, 1, true));
    ax_sequential_add(m, ax_batchnorm_create(128, 1e-5f, 0.1f));
    ax_sequential_add(m, ax_relu_layer_create());
    ax_sequential_add(m, ax_maxpool2d_create(2, 2, 0));

    /* block 3: -> [N,256,3,3] */
    ax_sequential_add(m, ax_conv2d_create(128, 256, 3, 1, 1, true));
    ax_sequential_add(m, ax_batchnorm_create(256, 1e-5f, 0.1f));
    ax_sequential_add(m, ax_relu_layer_create());
    ax_sequential_add(m, ax_conv2d_create(256, 256, 3, 1, 1, true));
    ax_sequential_add(m, ax_batchnorm_create(256, 1e-5f, 0.1f));
    ax_sequential_add(m, ax_relu_layer_create());
    ax_sequential_add(m, ax_maxpool2d_create(2, 2, 0));

    /* classifier: flatten -> dense(2304,2048) relu dropout -> dense(2048,1024) relu dropout -> dense(1024,10) */
    ax_sequential_add(m, ax_flatten_create());
    ax_sequential_add(m, ax_dense_create(2304, 2048, true));
    ax_sequential_add(m, ax_relu_layer_create());
    ax_sequential_add(m, ax_dropout_create(0.5f));
    ax_sequential_add(m, ax_dense_create(2048, 1024, true));
    ax_sequential_add(m, ax_relu_layer_create());
    ax_sequential_add(m, ax_dropout_create(0.5f));
    ax_sequential_add(m, ax_dense_create(1024, N_CLASSES, true));

    return m;
}

static void run_train(void) {
    ax_set_seed(42);

    printf("=== axiom vgg train (cpu) ===\n");
    printf("threads: %d\n", ax_get_num_threads());

    ax_tensor_t *train_x = load_images_4d(TRAIN_IMAGES, N_TRAIN);
    ax_tensor_t *train_y = load_labels_onehot(TRAIN_LABELS, N_TRAIN);
    ax_tensor_t *test_x  = load_images_4d(TEST_IMAGES, N_TEST);
    uint8_t *test_labels = load_labels_raw(TEST_LABELS, N_TEST);
    if (!train_x || !train_y || !test_x || !test_labels)
        { fprintf(stderr, "data load failed\n"); return; }

    ax_layer_t *model = build_vgg();
    int64_t param_count = ax_layer_param_count(model);
    printf("model: vgg-mnist  params: %ld\n", param_count);
    printf("batch: %d  epochs: %d  dataset: %d train / %d test\n\n",
           BATCH, EPOCHS, N_TRAIN, N_TEST);
    fflush(stdout);

    ax_tensor_t *params[128];
    int np = ax_layer_get_params(model, params, 128);
    ax_optimizer_t *opt = ax_adam_create(params, np, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f);

    /* shuffle indices */
    int64_t *idx = malloc((size_t)N_TRAIN * sizeof(int64_t));
    for (int64_t i = 0; i < N_TRAIN; i++) idx[i] = i;

    float *train_xd = (float *)train_x->storage->data;
    float *train_yd = (float *)train_y->storage->data;

    /* warmup batch */
    {
        int64_t xshape[] = {BATCH, 1, IMG_H, IMG_W};
        int64_t yshape[] = {BATCH, N_CLASSES};
        ax_tensor_t *wx = ax_tensor_zeros(xshape, 4, AX_FLOAT32);
        ax_tensor_t *wy = ax_tensor_zeros(yshape, 2, AX_FLOAT32);
        ax_enable_grad();
        ax_tensor_t *logits = ax_layer_forward(model, wx);
        ax_tensor_t *loss = ax_cross_entropy_loss(logits, wy);
        if (logits && loss) {
            ax_optimizer_zero_grad(opt);
            ax_backward(loss);
            ax_optimizer_step(opt);
            ax_graph_cleanup(loss);
            ax_tensor_destroy(loss);
        }
        ax_tensor_destroy(wx);
        ax_tensor_destroy(wy);
    }

    double t_start = now_s();
    double per_epoch[EPOCHS];

    for (int ep = 0; ep < EPOCHS; ep++) {
        double t_ep = now_s();

        /* shuffle */
        for (int64_t i = N_TRAIN - 1; i > 0; i--) {
            int64_t j = (int64_t)ax_rng_bounded((uint64_t)(i + 1));
            int64_t tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }

        ax_layer_train(model);
        double total_loss = 0.0;
        int64_t batches = 0;

        for (int64_t bstart = 0; bstart + BATCH <= N_TRAIN; bstart += BATCH) {
            int64_t xshape[] = {BATCH, 1, IMG_H, IMG_W};
            int64_t yshape[] = {BATCH, N_CLASSES};
            ax_tensor_t *bx = ax_tensor_create(xshape, 4, AX_FLOAT32);
            ax_tensor_t *by = ax_tensor_zeros(yshape, 2, AX_FLOAT32);
            float *bxd = (float *)bx->storage->data;
            float *byd = (float *)by->storage->data;
            for (int64_t i = 0; i < BATCH; i++) {
                int64_t si = idx[bstart + i];
                memcpy(bxd + i * N_PIXELS, train_xd + si * N_PIXELS,
                       N_PIXELS * sizeof(float));
                memcpy(byd + i * N_CLASSES, train_yd + si * N_CLASSES,
                       N_CLASSES * sizeof(float));
            }

            ax_enable_grad();
            ax_tensor_t *logits = ax_layer_forward(model, bx);
            ax_tensor_t *loss = ax_cross_entropy_loss(logits, by);
            if (!logits || !loss) {
                if (logits) ax_tensor_destroy(logits);
                ax_tensor_destroy(bx); ax_tensor_destroy(by);
                continue;
            }

            total_loss += ((float *)loss->storage->data)[0];
            batches++;

            ax_optimizer_zero_grad(opt);
            ax_backward(loss);
            ax_clip_grad_norm(params, np, 1.0f);
            ax_optimizer_step(opt);

            ax_graph_cleanup(loss);
            ax_tensor_destroy(loss);
            ax_tensor_destroy(bx);
            ax_tensor_destroy(by);
        }

        float acc = eval_accuracy(model, test_x, test_labels, N_TEST);
        double dt = now_s() - t_ep;
        per_epoch[ep] = dt;
        printf("epoch %d/%d  loss=%.4f  test_acc=%.2f%%  time=%.3fs\n",
               ep + 1, EPOCHS,
               batches > 0 ? (float)(total_loss / (double)batches) : 0.0f,
               acc, dt);
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
    printf("\n--- axiom vgg cpu summary ---\n");
    printf("params: %ld\n", param_count);
    printf("total training time: %.3fs\n", t_total);
    printf("mean per-epoch: %.3fs\n", mean);
    printf("median per-epoch: %.3fs\n", sorted[EPOCHS / 2]);
    printf("final test accuracy: %.2f%%\n", final_acc);

    ax_optimizer_destroy(opt);
    ax_layer_destroy(model);
    ax_tensor_destroy(train_x); ax_tensor_destroy(train_y);
    ax_tensor_destroy(test_x); free(test_labels); free(idx);
}

static void run_infer(void) {
    ax_set_seed(42);
    printf("=== axiom vgg infer (cpu) ===\n");
    printf("threads: %d\n", ax_get_num_threads());

    ax_layer_t *model = build_vgg();
    ax_layer_eval(model);
    int64_t param_count = ax_layer_param_count(model);
    printf("model: vgg-mnist  params: %ld\n", param_count);

    const int iters = 500;
    printf("batch: %d  forward iters: %d\n\n", BATCH, iters);
    fflush(stdout);

    int64_t xshape[] = {BATCH, 1, IMG_H, IMG_W};
    ax_tensor_t *x = ax_tensor_rand(xshape, 4, 0.0f, 1.0f);

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

    printf("--- axiom vgg cpu forward summary ---\n");
    printf("total forward time: %.3fs\n", total);
    printf("per-batch: %.3f ms\n", per_batch_ms);
    printf("throughput: %.0f images/s\n", imgs_per_sec);

    ax_tensor_destroy(x);
    ax_layer_destroy(model);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    ax_init();
    const char *mode = argc > 1 ? argv[1] : "train";
    if (strcmp(mode, "train") == 0) run_train();
    else if (strcmp(mode, "infer") == 0) run_infer();
    else { fprintf(stderr, "usage: %s {train|infer}\n", argv[0]); return 1; }
    ax_shutdown();
    return 0;
}
