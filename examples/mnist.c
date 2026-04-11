/* mnist.c — MLP trained on MNIST handwritten digit recognition.
   demonstrates the full training pipeline:
   data loading -> preprocessing -> training -> evaluation.

   expects raw IDX binary files in examples/data/:
     train-images-idx3-ubyte  (60000 images)
     train-labels-idx1-ubyte  (60000 labels)
     t10k-images-idx3-ubyte   (10000 images)
     t10k-labels-idx1-ubyte   (10000 labels)

   architecture: 784 -> 256 (relu) -> 128 (relu) -> 10
   optimizer: adam, lr=1e-3
   expected test accuracy: ~97-98% after 5 epochs */

#include "axiom/axiom.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TRAIN_IMAGES "examples/data/train-images-idx3-ubyte"
#define TRAIN_LABELS "examples/data/train-labels-idx1-ubyte"
#define TEST_IMAGES  "examples/data/t10k-images-idx3-ubyte"
#define TEST_LABELS  "examples/data/t10k-labels-idx1-ubyte"

#define N_TRAIN_FULL  60000
#define N_TRAIN       10000   /* subset for reasonable runtime on naive cpu backend */
#define N_TEST        10000
#define N_PIXELS  784
#define N_CLASSES 10

/* read a big-endian uint32 from a file */
static uint32_t read_be32(FILE *f)
{
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] << 8)  |  (uint32_t)b[3];
}

/* load MNIST images into a [n, 784] float32 tensor, pixels in [0,1] */
static ax_tensor_t *load_images(const char *path, int64_t n)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "error: cannot open %s\n", path); return NULL; }

    uint32_t magic = read_be32(f);
    if (magic != 0x00000803)
    {
        fprintf(stderr, "error: bad magic 0x%08x in %s\n", magic, path);
        fclose(f);
        return NULL;
    }
    read_be32(f); /* n_images — we use our own n */
    int64_t rows = (int64_t)read_be32(f);
    int64_t cols = (int64_t)read_be32(f);
    if (rows * cols != N_PIXELS)
    {
        fprintf(stderr, "error: unexpected image size %ldx%ld\n", rows, cols);
        fclose(f);
        return NULL;
    }

    uint8_t *raw = malloc((size_t)(n * N_PIXELS));
    if (!raw) { fclose(f); return NULL; }
    if (fread(raw, 1, (size_t)(n * N_PIXELS), f) != (size_t)(n * N_PIXELS))
    {
        free(raw); fclose(f); return NULL;
    }
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

/* load MNIST labels into a [n, 10] one-hot float32 tensor */
static ax_tensor_t *load_labels_onehot(const char *path, int64_t n)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "error: cannot open %s\n", path); return NULL; }

    uint32_t magic = read_be32(f);
    if (magic != 0x00000801)
    {
        fprintf(stderr, "error: bad magic 0x%08x in %s\n", magic, path);
        fclose(f);
        return NULL;
    }
    read_be32(f); /* n_labels */

    uint8_t *raw = malloc((size_t)n);
    if (!raw) { fclose(f); return NULL; }
    if (fread(raw, 1, (size_t)n, f) != (size_t)n)
    {
        free(raw); fclose(f); return NULL;
    }
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

/* load raw labels (not one-hot) for accuracy calculation */
static uint8_t *load_labels_raw(const char *path, int64_t n)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "error: cannot open %s\n", path); return NULL; }
    read_be32(f); read_be32(f); /* magic + count */
    uint8_t *labels = malloc((size_t)n);
    if (!labels) { fclose(f); return NULL; }
    if (fread(labels, 1, (size_t)n, f) != (size_t)n)
    {
        free(labels); fclose(f); return NULL;
    }
    fclose(f);
    return labels;
}

/* evaluate accuracy: run model in eval mode with no_grad, compute argmax */
static float eval_accuracy(ax_layer_t *model, ax_tensor_t *images,
                            const uint8_t *labels, int64_t n)
{
    ax_layer_eval(model);
    ax_no_grad();

    const int64_t eval_bs = 512;
    int64_t correct = 0;

    for (int64_t start = 0; start < n; start += eval_bs)
    {
        int64_t bs = (start + eval_bs <= n) ? eval_bs : (n - start);

        int64_t xshape[] = {bs, N_PIXELS};
        ax_tensor_t *x = ax_tensor_create(xshape, 2, AX_FLOAT32);
        if (!x) break;

        float *xd = (float *)x->storage->data;
        float *id = (float *)images->storage->data;
        memcpy(xd, id + start * N_PIXELS, (size_t)(bs * N_PIXELS) * sizeof(float));

        ax_tensor_t *logits = ax_layer_forward(model, x);
        ax_tensor_destroy(x);
        if (!logits) break;

        float *ld = (float *)logits->storage->data;
        for (int64_t i = 0; i < bs; i++)
        {
            int pred = 0;
            float best = ld[i * N_CLASSES];
            for (int c = 1; c < N_CLASSES; c++)
            {
                if (ld[i * N_CLASSES + c] > best)
                {
                    best = ld[i * N_CLASSES + c];
                    pred = c;
                }
            }
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
    ax_set_seed(42);

    /* load data */
    printf("loading MNIST...\n"); fflush(stdout);
    ax_tensor_t *train_x = load_images(TRAIN_IMAGES, N_TRAIN);
    ax_tensor_t *train_y = load_labels_onehot(TRAIN_LABELS, N_TRAIN);
    ax_tensor_t *test_x  = load_images(TEST_IMAGES,  N_TEST);
    uint8_t *train_labels = load_labels_raw(TRAIN_LABELS, N_TRAIN);
    uint8_t *test_labels  = load_labels_raw(TEST_LABELS,  N_TEST);

    if (!train_x || !train_y || !test_x || !train_labels || !test_labels)
    {
        fprintf(stderr, "failed to load MNIST data from examples/data/\n");
        return 1;
    }
    printf("  train: %d samples (of %d), test: %d samples, %d pixels each\n",
           N_TRAIN, N_TRAIN_FULL, N_TEST, N_PIXELS);
    fflush(stdout);

    /* model: 784 -> 128 (relu) -> 64 (relu) -> 10
       smaller hidden layers = faster on the naive cpu backend */
    ax_layer_t *model = ax_sequential_create();
    ax_sequential_add(model, ax_dense_create(N_PIXELS,   128, true));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_dense_create(128,         64, true));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_dense_create(64, N_CLASSES, true));

    /* optimizer */
    ax_tensor_t *params[32];
    int n_params = ax_layer_get_params(model, params, 32);
    ax_optimizer_t *opt = ax_adam_create(params, n_params,
                                          1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f);

    /* data pipeline */
    ax_dataset_t *ds = ax_tensor_dataset_create(train_x, train_y);
    ax_dataloader_t *dl = ax_dataloader_create(ds, 256, true);

    printf("model: %ld parameters\n", ax_layer_param_count(model));
    printf("training: %ld batches/epoch\n\n", ax_dataloader_num_batches(dl));
    fflush(stdout);

    /* training loop */
    const int epochs = 8;
    for (int ep = 0; ep < epochs; ep++)
    {
        ax_dataloader_reset(dl);
        ax_layer_train(model);

        double total_loss = 0.0;
        int64_t n_batches = 0;
        ax_batch_t b;

        while (ax_dataloader_next(dl, &b))
        {
            ax_enable_grad();
            ax_tensor_t *logits = ax_layer_forward(model, b.input);
            ax_tensor_t *loss   = ax_cross_entropy_loss(logits, b.target);

            if (!logits || !loss)
            {
                if (logits) ax_tensor_destroy(logits);
                ax_tensor_destroy(b.input);
                ax_tensor_destroy(b.target);
                continue;
            }

            total_loss += ((float *)loss->storage->data)[0];
            n_batches++;

            ax_optimizer_zero_grad(opt);
            ax_backward(loss);
            ax_optimizer_step(opt);

            /* ax_graph_cleanup destroys all graph intermediates including logits */
            ax_graph_cleanup(loss);
            ax_tensor_destroy(loss);

            /* batch tensors are leaves — not in the graph, destroy manually */
            ax_tensor_destroy(b.input);
            ax_tensor_destroy(b.target);
        }

        float train_acc = eval_accuracy(model, train_x, train_labels, N_TRAIN);
        float test_acc  = eval_accuracy(model, test_x,  test_labels,  N_TEST);
        (void)N_TRAIN_FULL;

        printf("epoch %d/%d  loss=%.4f  train_acc=%.2f%%  test_acc=%.2f%%\n",
               ep + 1, epochs,
               (float)(total_loss / (double)n_batches),
               train_acc, test_acc);
        fflush(stdout);
    }

    /* save the trained model */
    const char *model_path = "mnist_model.axm";
    ax_model_t *wrapper = ax_model_create(model);
    if (ax_model_save(wrapper, model_path) == AX_OK)
        printf("\nmodel saved to %s\n", model_path);
    ax_model_destroy(wrapper);

    /* cleanup */
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
