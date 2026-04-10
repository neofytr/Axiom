/* mnist_cnn.c — convolutional network trained on full MNIST.
   exercises every major layer type in the framework:
   conv2d, batchnorm, maxpool, relu, flatten, dense, dropout.

   architecture (lenet-5 inspired):
     conv2d(1, 32, 3, pad=1) -> batchnorm -> relu -> maxpool(2)
     conv2d(32, 64, 3, pad=1) -> batchnorm -> relu -> maxpool(2)
     flatten -> dense(3136, 256) -> relu -> dropout(0.3)
     dense(256, 10)

   training:
     full 60k MNIST, batch size 64, adam with cosine lr schedule.
     expected test accuracy: 98-99% after 10 epochs.

   run from the project root:
     ./build/mnist_cnn */

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

/* read big-endian uint32 */
static uint32_t read_be32(FILE *f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] << 8)  |  (uint32_t)b[3];
}

/* load MNIST images as [N, 1, 28, 28] float32 in [0,1] (NCHW for conv2d) */
static ax_tensor_t *load_images_4d(const char *path, int64_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    uint32_t magic = read_be32(f);
    if (magic != 0x00000803) { fclose(f); return NULL; }
    read_be32(f); read_be32(f); read_be32(f); /* skip counts */

    uint8_t *raw = malloc((size_t)(n * N_PIXELS));
    if (!raw) { fclose(f); return NULL; }
    if (fread(raw, 1, (size_t)(n * N_PIXELS), f) != (size_t)(n * N_PIXELS)) {
        free(raw); fclose(f); return NULL;
    }
    fclose(f);

    int64_t shape[] = {n, 1, IMG_H, IMG_W};
    ax_tensor_t *t = ax_tensor_create(shape, 4, AX_FLOAT32);
    if (!t) { free(raw); return NULL; }
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n * N_PIXELS; i++)
        d[i] = (float)raw[i] / 255.0f;
    free(raw);
    return t;
}

/* load one-hot labels [N, 10] */
static ax_tensor_t *load_labels_onehot(const char *path, int64_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    read_be32(f); read_be32(f);
    uint8_t *raw = malloc((size_t)n);
    if (!raw) { fclose(f); return NULL; }
    if (fread(raw, 1, (size_t)n, f) != (size_t)n) { free(raw); fclose(f); return NULL; }
    fclose(f);
    int64_t shape[] = {n, N_CLASSES};
    ax_tensor_t *t = ax_tensor_zeros(shape, 2, AX_FLOAT32);
    if (!t) { free(raw); return NULL; }
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++) d[i * N_CLASSES + raw[i]] = 1.0f;
    free(raw);
    return t;
}

/* load raw uint8 labels for accuracy calc */
static uint8_t *load_labels_raw(const char *path, int64_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    read_be32(f); read_be32(f);
    uint8_t *labels = malloc((size_t)n);
    if (!labels) { fclose(f); return NULL; }
    if (fread(labels, 1, (size_t)n, f) != (size_t)n) { free(labels); fclose(f); return NULL; }
    fclose(f);
    return labels;
}

/* wall clock in seconds */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* evaluate accuracy on a dataset using batched inference.
   images are 4D [N,1,28,28] for the CNN. */
static float eval_accuracy(ax_layer_t *model, ax_tensor_t *images,
                            const uint8_t *labels, int64_t n)
{
    ax_layer_eval(model);
    ax_no_grad();

    int64_t correct = 0;
    const int64_t bs = 256;

    for (int64_t start = 0; start < n; start += bs) {
        int64_t b = (start + bs <= n) ? bs : (n - start);

        int64_t xshape[] = {b, 1, IMG_H, IMG_W};
        ax_tensor_t *x = ax_tensor_create(xshape, 4, AX_FLOAT32);
        if (!x) break;

        float *xd = (float *)x->storage->data;
        float *id = (float *)images->storage->data;
        memcpy(xd, id + start * N_PIXELS, (size_t)(b * N_PIXELS) * sizeof(float));

        ax_tensor_t *logits = ax_layer_forward(model, x);
        ax_tensor_destroy(x);
        if (!logits) break;

        float *ld = (float *)logits->storage->data;
        for (int64_t i = 0; i < b; i++) {
            int pred = 0;
            float best = ld[i * N_CLASSES];
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


int main(void)
{
    ax_init();
    ax_rng_seed(42);

    printf("mnist_cnn: convolutional network on full MNIST\n");
    printf("backend: %s\n\n", ax_compute_get_ops()->name);

    /* load data */
    printf("loading data...\n"); fflush(stdout);
    ax_tensor_t *train_x = load_images_4d(TRAIN_IMAGES, N_TRAIN);
    ax_tensor_t *test_x  = load_images_4d(TEST_IMAGES,  N_TEST);

    /* one-hot labels are 2D [N, 10] for cross-entropy.
       the dataloader will batch these into [bs, 10]. */
    ax_tensor_t *train_y = load_labels_onehot(TRAIN_LABELS, N_TRAIN);
    uint8_t *train_labels = load_labels_raw(TRAIN_LABELS, N_TRAIN);
    uint8_t *test_labels  = load_labels_raw(TEST_LABELS,  N_TEST);

    if (!train_x || !test_x || !train_y || !train_labels || !test_labels) {
        fprintf(stderr, "failed to load MNIST from examples/data/\n");
        return 1;
    }
    printf("  train: %d images [1,28,28], test: %d images\n\n", N_TRAIN, N_TEST);

    /* build CNN
       conv2d(1,32,3,pad=1) -> batchnorm(32) -> relu -> maxpool(2)
       conv2d(32,64,3,pad=1) -> batchnorm(64) -> relu -> maxpool(2)
       flatten -> dense(3136,256) -> relu -> dropout(0.3) -> dense(256,10)

       after conv1+pool: [N, 32, 14, 14]
       after conv2+pool: [N, 64, 7, 7]
       after flatten:    [N, 3136]  (64*7*7 = 3136) */

    ax_layer_t *model = ax_sequential_create();

    /* block 1 */
    ax_sequential_add(model, ax_conv2d_create(1, 32, 3, 1, 1, true));
    ax_sequential_add(model, ax_batchnorm_create(32, 1e-5f, 0.1f));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_maxpool2d_create(2, 2, 0));

    /* block 2 */
    ax_sequential_add(model, ax_conv2d_create(32, 64, 3, 1, 1, true));
    ax_sequential_add(model, ax_batchnorm_create(64, 1e-5f, 0.1f));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_maxpool2d_create(2, 2, 0));

    /* classifier */
    ax_sequential_add(model, ax_flatten_create());
    ax_sequential_add(model, ax_dense_create(3136, 256, true));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_dropout_create(0.3f));
    ax_sequential_add(model, ax_dense_create(256, N_CLASSES, true));

    int64_t total_params = ax_layer_param_count(model);
    printf("model architecture:\n");
    ax_layer_summary(model);
    printf("\n");

    /* optimizer + lr schedule */
    ax_tensor_t *params[64];
    int n_params = ax_layer_get_params(model, params, 64);
    ax_optimizer_t *opt = ax_adam_create(params, n_params,
                                         1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f);

    const int epochs = 10;
    ax_lr_scheduler_t *sched = ax_sched_cosine(opt, epochs, 1e-5f);

    /* dataloader: we need 4D images but 2D labels.
       since ax_tensor_dataset requires matching dim-0, and our images are [N,1,28,28]
       while labels are [N,10], we'll do manual batching from the 4D images. */

    printf("training: %d epochs, batch_size=64, adam + cosine lr\n\n", epochs);
    fflush(stdout);

    const int64_t batch_size = 64;
    int64_t n_batches = (N_TRAIN + batch_size - 1) / batch_size;

    /* shuffle index array */
    int64_t *indices = malloc((size_t)N_TRAIN * sizeof(int64_t));
    for (int64_t i = 0; i < N_TRAIN; i++) indices[i] = i;

    float *train_xd = (float *)train_x->storage->data;
    float *train_yd = (float *)train_y->storage->data;

    double total_time = 0.0;

    for (int ep = 0; ep < epochs; ep++) {
        double ep_start = now_sec();

        /* shuffle */
        for (int64_t i = N_TRAIN - 1; i > 0; i--) {
            int64_t j = (int64_t)ax_rng_bounded((uint64_t)(i + 1));
            int64_t tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }

        ax_layer_train(model);
        double total_loss = 0.0;
        int64_t batches_done = 0;

        for (int64_t batch_start = 0; batch_start < N_TRAIN; batch_start += batch_size) {
            int64_t bs = (batch_start + batch_size <= N_TRAIN) ? batch_size : (N_TRAIN - batch_start);

            /* build batch tensors: x is [bs, 1, 28, 28], y is [bs, 10] */
            int64_t xshape[] = {bs, 1, IMG_H, IMG_W};
            int64_t yshape[] = {bs, N_CLASSES};
            ax_tensor_t *bx = ax_tensor_create(xshape, 4, AX_FLOAT32);
            ax_tensor_t *by = ax_tensor_zeros(yshape, 2, AX_FLOAT32);
            if (!bx || !by) { ax_tensor_destroy(bx); ax_tensor_destroy(by); continue; }

            float *bxd = (float *)bx->storage->data;
            float *byd = (float *)by->storage->data;
            for (int64_t i = 0; i < bs; i++) {
                int64_t idx = indices[batch_start + i];
                memcpy(bxd + i * N_PIXELS, train_xd + idx * N_PIXELS, N_PIXELS * sizeof(float));
                memcpy(byd + i * N_CLASSES, train_yd + idx * N_CLASSES, N_CLASSES * sizeof(float));
            }

            /* forward */
            ax_enable_grad();
            ax_tensor_t *logits = ax_layer_forward(model, bx);
            if (!logits) { ax_tensor_destroy(bx); ax_tensor_destroy(by); continue; }

            /* loss: cross-entropy expects [bs, 10] logits and [bs, 10] one-hot */
            ax_tensor_t *loss = ax_cross_entropy_loss(logits, by);
            if (!loss) { ax_tensor_destroy(bx); ax_tensor_destroy(by); continue; }

            total_loss += ((float *)loss->storage->data)[0];
            batches_done++;

            /* backward + step */
            ax_optimizer_zero_grad(opt);
            ax_backward(loss);
            ax_clip_grad_norm(params, n_params, 1.0f);
            ax_optimizer_step(opt);

            ax_graph_cleanup(loss);
            ax_tensor_destroy(loss);
            ax_tensor_destroy(bx);
            ax_tensor_destroy(by);

            /* progress dot every 100 batches */
            if (batches_done % 100 == 0) { printf("."); fflush(stdout); }
        }

        ax_sched_step(sched);
        double ep_time = now_sec() - ep_start;
        total_time += ep_time;

        /* evaluate */
        float train_acc = eval_accuracy(model, train_x, train_labels, N_TRAIN);
        float test_acc  = eval_accuracy(model, test_x,  test_labels,  N_TEST);

        printf("\repoch %2d/%d  loss=%.4f  train=%.2f%%  test=%.2f%%  lr=%.6f  (%.1fs)\n",
               ep + 1, epochs,
               (float)(total_loss / (double)batches_done),
               train_acc, test_acc,
               ax_sched_get_lr(sched), ep_time);
        fflush(stdout);
    }

    printf("\ntotal training time: %.1f sec (%.1f sec/epoch)\n",
           total_time, total_time / epochs);

    /* final evaluation */
    float final_test = eval_accuracy(model, test_x, test_labels, N_TEST);
    printf("final test accuracy: %.2f%%\n\n", final_test);

    /* save model */
    ax_model_t *wrapper = ax_model_create(model);
    const char *save_path = "mnist_cnn.axm";
    if (ax_model_save(wrapper, save_path) == AX_OK)
        printf("model saved to %s (%ld params)\n", save_path, total_params);

    /* cleanup */
    ax_model_destroy(wrapper);
    ax_sched_destroy(sched);
    ax_optimizer_destroy(opt);
    ax_tensor_destroy(train_x);
    ax_tensor_destroy(train_y);
    ax_tensor_destroy(test_x);
    free(train_labels);
    free(test_labels);
    free(indices);
    ax_shutdown();

    return 0;
}
