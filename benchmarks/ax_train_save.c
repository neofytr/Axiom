/* ax_train_save.c — train an mnist mlp in the FULL axiom build and save
   the resulting model to disk. the companion ax_infer_load.c then loads
   this exact file in the INFERENCE-ONLY build and verifies it produces
   byte-identical test-set accuracy.

   this proves the inference-only library is a strict subset of the full
   library for forward-path computation — no silent numerical drift from
   #ifdef'd code paths, no missed ops, no surprises at deployment. */

#include "axiom/axiom.h"
#include "axiom/serialize.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TRAIN_IMAGES "examples/data/train-images-idx3-ubyte"
#define TRAIN_LABELS "examples/data/train-labels-idx1-ubyte"
#define TEST_IMAGES  "examples/data/t10k-images-idx3-ubyte"
#define TEST_LABELS  "examples/data/t10k-labels-idx1-ubyte"
#define MODEL_PATH   "benchmarks/mnist_parity.axm"
#define LOGITS_PATH  "benchmarks/mnist_parity_logits.bin"

#define N_TRAIN   10000
#define N_TEST    10000
#define N_PIXELS  784
#define N_CLASSES 10

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

/* run forward on the whole test set, return the full [n, 10] logit tensor
   (contig float32 — we'll save this raw so the inference program can
   compare against it). */
static ax_tensor_t *forward_all(ax_layer_t *model, ax_tensor_t *images, int64_t n) {
    int64_t out_shape[] = {n, N_CLASSES};
    ax_tensor_t *all = ax_tensor_zeros(out_shape, 2, AX_FLOAT32);
    const int64_t bs = 512;
    float *all_d = (float *)all->storage->data;
    float *im_d = (float *)images->storage->data;

    ax_layer_eval(model);
    ax_no_grad();

    for (int64_t start = 0; start < n; start += bs) {
        int64_t b = (start + bs <= n) ? bs : (n - start);
        int64_t xs[] = {b, N_PIXELS};
        ax_tensor_t *x = ax_tensor_create(xs, 2, AX_FLOAT32);
        memcpy(x->storage->data, im_d + start * N_PIXELS,
               (size_t)(b * N_PIXELS) * sizeof(float));
        ax_tensor_t *y = ax_layer_forward(model, x);
        ax_tensor_destroy(x);
        memcpy(all_d + start * N_CLASSES,
               y->storage->data,
               (size_t)(b * N_CLASSES) * sizeof(float));
        ax_tensor_destroy(y);
    }

    ax_layer_train(model);
    ax_enable_grad();
    return all;
}

static int argmax10(const float *row) {
    int best = 0; float bestv = row[0];
    for (int c = 1; c < N_CLASSES; c++) if (row[c] > bestv) { bestv = row[c]; best = c; }
    return best;
}

static float accuracy_from_logits(const ax_tensor_t *logits, const uint8_t *labels, int64_t n) {
    const float *d = (const float *)logits->storage->data;
    int64_t correct = 0;
    for (int64_t i = 0; i < n; i++)
        if (argmax10(d + i * N_CLASSES) == (int)labels[i]) correct++;
    return 100.0f * (float)correct / (float)n;
}

int main(void) {
    ax_init();
    ax_set_seed(42);

    printf("[train] loading mnist...\n");
    ax_tensor_t *train_x = load_images(TRAIN_IMAGES, N_TRAIN);
    ax_tensor_t *train_y = load_labels_onehot(TRAIN_LABELS, N_TRAIN);
    ax_tensor_t *test_x  = load_images(TEST_IMAGES,  N_TEST);
    uint8_t *test_labels = load_labels_raw(TEST_LABELS, N_TEST);
    if (!train_x || !train_y || !test_x || !test_labels) {
        fprintf(stderr, "[train] mnist load failed\n"); return 1;
    }

    /* architecture matches ax_mnist benchmark so we can reuse the
       numerical baseline: 784 -> 128 relu -> 64 relu -> 10 */
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(N_PIXELS,   128, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(128,         64, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(64, N_CLASSES, true));

    ax_tensor_t *params[32];
    int n_params = ax_layer_get_params(net, params, 32);
    ax_optimizer_t *opt = ax_adam_create(params, n_params, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f);

    ax_dataset_t *ds = ax_tensor_dataset_create(train_x, train_y);
    ax_dataloader_t *dl = ax_dataloader_create(ds, 256, true);

    const int epochs = 8;
    printf("[train] %ld params, %d epochs\n", (long)ax_layer_param_count(net), epochs);

    for (int ep = 0; ep < epochs; ep++) {
        ax_dataloader_reset(dl);
        ax_layer_train(net);
        double tl = 0.0;
        int64_t nseen = 0;
        ax_batch_t b;
        while (ax_dataloader_next(dl, &b)) {
            ax_enable_grad();
            ax_tensor_t *logits = ax_layer_forward(net, b.input);
            ax_tensor_t *loss   = ax_cross_entropy_loss(logits, b.target);
            if (!logits || !loss) {
                if (logits) ax_tensor_destroy(logits);
                ax_tensor_destroy(b.input); ax_tensor_destroy(b.target);
                continue;
            }
            tl += (double)((float *)loss->storage->data)[0] * (double)b.input->shape[0];
            nseen += b.input->shape[0];
            ax_optimizer_zero_grad(opt);
            ax_backward(loss);
            ax_optimizer_step(opt);
            ax_graph_cleanup(loss);
            ax_tensor_destroy(loss);
            ax_tensor_destroy(b.input); ax_tensor_destroy(b.target);
        }
        printf("[train] epoch %d/%d  loss=%.4f\n", ep + 1, epochs, tl / (double)nseen);
    }

    /* run forward on the test set once — capture every logit */
    ax_tensor_t *logits = forward_all(net, test_x, N_TEST);
    float acc = accuracy_from_logits(logits, test_labels, N_TEST);
    printf("[train] final test accuracy: %.4f%%\n", acc);

    /* save the model + the exact logit tensor so the inference-only
       program can compare bit-for-bit. */
    ax_model_t *model = ax_model_create(net);
    if (ax_model_save(model, MODEL_PATH) != AX_OK) {
        fprintf(stderr, "[train] model save failed\n"); return 1;
    }
    if (ax_tensor_save(logits, LOGITS_PATH) != AX_OK) {
        fprintf(stderr, "[train] logits save failed\n"); return 1;
    }
    printf("[train] saved model -> %s\n", MODEL_PATH);
    printf("[train] saved logits -> %s\n", LOGITS_PATH);
    printf("[train] accuracy to match: %.4f%%\n", acc);
    return 0;
}
