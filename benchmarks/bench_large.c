/* bench_large.c — "relatively large" mnist mlp benchmark across
   axiom's cpu_opt and cuda backends. matches benchmarks/bench_large_tf.py
   on model architecture, optimizer, batch size, and timing scope.

   architecture: 784 -> 1024 relu -> 512 relu -> 256 relu -> 10
   params:       ~1.46 million
   dataset:      full mnist (60000 train / 10000 test)
   optim:        adam  lr=1e-3  b1=0.9  b2=0.999  eps=1e-8
   batch size:   256
   epochs:       5
   seed:         42
   measurement:  total training wall time + per-epoch + final test acc.

   modes (argv[1]):
     train-cpu   train on cpu_opt, report training perf vs tf
     infer-cpu   forward-only throughput on cpu_opt (1000 batches)
     infer-gpu   forward-only throughput on cuda (1000 batches)

   training on gpu is not supported yet — core backward/optim passes
   do host-side raw-pointer access on tensor data, which only works
   with the old unified-memory cuda path. phase 3 (full training-
   surface dispatch) is the follow-up that unlocks it. */

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

#define H1 1024
#define H2 512
#define H3 256

#define BATCH     256
#define EPOCHS    5
#define LR        1e-3f

#define FORWARD_ITERS 1000

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
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    read_be32(f); read_be32(f); read_be32(f); read_be32(f);
    uint8_t *raw = malloc((size_t)(n * N_PIXELS));
    if (!raw || fread(raw, 1, (size_t)(n * N_PIXELS), f) != (size_t)(n * N_PIXELS)) {
        free(raw); fclose(f); return NULL;
    }
    fclose(f);
    float *host = malloc((size_t)(n * N_PIXELS) * sizeof(float));
    for (int64_t i = 0; i < n * N_PIXELS; i++) host[i] = (float)raw[i] / 255.0f;
    free(raw);
    int64_t shape[] = {n, N_PIXELS};
    ax_tensor_t *t = ax_tensor_from_array(host, shape, 2, AX_FLOAT32);
    free(host);
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
    float *host = calloc((size_t)(n * N_CLASSES), sizeof(float));
    for (int64_t i = 0; i < n; i++) host[i * N_CLASSES + labels[i]] = 1.0f;
    free(labels);
    int64_t shape[] = {n, N_CLASSES};
    ax_tensor_t *t = ax_tensor_from_array(host, shape, 2, AX_FLOAT32);
    free(host);
    return t;
}

static uint8_t *load_label_bytes(const char *path, int64_t n) {
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

/* model — four layers so the forward pass is 4 gemms + 3 relus + softmax */
typedef struct {
    ax_layer_t *fc1, *fc2, *fc3, *fc4;
} model_t;

static void model_init(model_t *m) {
    m->fc1 = ax_dense_create(N_PIXELS, H1, true);
    m->fc2 = ax_dense_create(H1, H2, true);
    m->fc3 = ax_dense_create(H2, H3, true);
    m->fc4 = ax_dense_create(H3, N_CLASSES, true);
}

static ax_tensor_t *model_forward(model_t *m, ax_tensor_t *x) {
    ax_tensor_t *h = ax_layer_forward(m->fc1, x);
    h = ax_relu(h);
    h = ax_layer_forward(m->fc2, h);
    h = ax_relu(h);
    h = ax_layer_forward(m->fc3, h);
    h = ax_relu(h);
    h = ax_layer_forward(m->fc4, h);
    return h;
}

static int argmax_cpu_row(const float *row, int cols) {
    int best = 0;
    float bv = row[0];
    for (int i = 1; i < cols; i++) if (row[i] > bv) { bv = row[i]; best = i; }
    return best;
}

static float evaluate(model_t *m, ax_tensor_t *test_x, const uint8_t *test_labels_raw) {
    int correct = 0;
    const float *tx = (const float *)test_x->storage->data;
    for (int64_t i = 0; i < N_TEST; i += 512) {
        int64_t bsz = (i + 512 <= N_TEST) ? 512 : (N_TEST - i);
        int64_t bshape[] = {bsz, N_PIXELS};
        /* direct slice via memcpy for speed */
        ax_tensor_t *batch = ax_tensor_from_array(
            tx + i * N_PIXELS, bshape, 2, AX_FLOAT32);

        ax_tensor_t *logits = model_forward(m, batch);
        ax_tensor_t *logits_cpu = (logits->storage->device == AX_DEVICE_CPU)
                                   ? logits : ax_tensor_to_cpu(logits);
        const float *ld = (const float *)logits_cpu->storage->data;
        for (int64_t r = 0; r < bsz; r++) {
            if (argmax_cpu_row(ld + r * N_CLASSES, N_CLASSES) == (int)test_labels_raw[i + r])
                correct++;
        }
        if (logits_cpu != logits) ax_tensor_destroy(logits_cpu);
        ax_graph_cleanup(NULL);
    }
    return (float)correct / (float)N_TEST * 100.0f;
}

/* ---- training on cpu_opt ---- */

static void run_train_cpu(void) {
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
    ax_set_default_device(AX_DEVICE_CPU);
    ax_set_seed(42);

    printf("=== axiom train-cpu (cpu_opt backend) ===\n");
    printf("model: 784 -> %d -> %d -> %d -> 10\n", H1, H2, H3);
    int64_t n_params = (int64_t)N_PIXELS * H1 + H1
                     + (int64_t)H1 * H2 + H2
                     + (int64_t)H2 * H3 + H3
                     + (int64_t)H3 * N_CLASSES + N_CLASSES;
    printf("params: %ld\n", (long)n_params);
    printf("threads: %d\n", ax_get_num_threads());

    double t_load = now_s();
    ax_tensor_t *train_x = load_images(TRAIN_IMAGES, N_TRAIN);
    ax_tensor_t *train_y = load_labels_onehot(TRAIN_LABELS, N_TRAIN);
    ax_tensor_t *test_x  = load_images(TEST_IMAGES, N_TEST);
    uint8_t *test_labels = load_label_bytes(TEST_LABELS, N_TEST);
    if (!train_x || !train_y || !test_x || !test_labels) {
        fprintf(stderr, "failed to load mnist data\n");
        return;
    }
    printf("loaded mnist in %.2fs\n", now_s() - t_load);

    model_t m;
    model_init(&m);
    ax_tensor_t *params[16];
    int n_params_total = 0;
    n_params_total += ax_layer_get_params(m.fc1, params + n_params_total, 16 - n_params_total);
    n_params_total += ax_layer_get_params(m.fc2, params + n_params_total, 16 - n_params_total);
    n_params_total += ax_layer_get_params(m.fc3, params + n_params_total, 16 - n_params_total);
    n_params_total += ax_layer_get_params(m.fc4, params + n_params_total, 16 - n_params_total);
    ax_optimizer_t *opt = ax_adam_create(params, n_params_total, LR, 0.9f, 0.999f, 1e-8f, 0.0f);

    /* warm up one batch so first-op dispatch init doesn't skew epoch 1 */
    {
        int64_t shape[] = {BATCH, N_PIXELS};
        ax_tensor_t *wx = ax_tensor_zeros(shape, 2, AX_FLOAT32);
        int64_t yshape[] = {BATCH, N_CLASSES};
        ax_tensor_t *wy = ax_tensor_zeros(yshape, 2, AX_FLOAT32);
        ax_tensor_t *out = model_forward(&m, wx);
        ax_tensor_t *loss = ax_cross_entropy_loss(out, wy);
        ax_backward(loss);
        ax_optimizer_zero_grad(opt);
        ax_graph_cleanup(NULL);
    }

    printf("training: %d epochs, batch %d\n\n", EPOCHS, BATCH);
    double t_start = now_s();
    double per_epoch[EPOCHS];

    for (int ep = 0; ep < EPOCHS; ep++) {
        double t_ep = now_s();
        float loss_sum = 0.0f;
        int n_batches = 0;
        for (int64_t i = 0; i + BATCH <= N_TRAIN; i += BATCH) {
            int64_t bshape[] = {BATCH, N_PIXELS};
            int64_t yshape[] = {BATCH, N_CLASSES};
            ax_tensor_t *xb = ax_tensor_zeros(bshape, 2, AX_FLOAT32);
            ax_tensor_t *yb = ax_tensor_zeros(yshape, 2, AX_FLOAT32);
            /* cheap slice via direct memcpy on contiguous cpu storage */
            float *xd = (float *)xb->storage->data;
            float *yd = (float *)yb->storage->data;
            const float *xs = (const float *)train_x->storage->data;
            const float *ys = (const float *)train_y->storage->data;
            memcpy(xd, xs + i * N_PIXELS, (size_t)(BATCH * N_PIXELS) * sizeof(float));
            memcpy(yd, ys + i * N_CLASSES, (size_t)(BATCH * N_CLASSES) * sizeof(float));

            ax_tensor_t *out = model_forward(&m, xb);
            ax_tensor_t *loss = ax_cross_entropy_loss(out, yb);
            loss_sum += ax_tensor_get_f32(loss, (int64_t[]){0});
            n_batches++;

            ax_optimizer_zero_grad(opt);
            ax_backward(loss);
            ax_optimizer_step(opt);
            ax_graph_cleanup(NULL);
        }
        double dt = now_s() - t_ep;
        per_epoch[ep] = dt;
        float test_acc = evaluate(&m, test_x, test_labels);
        printf("epoch %d/%d  loss=%.4f  test_acc=%.2f%%  time=%.3fs\n",
               ep + 1, EPOCHS, loss_sum / n_batches, test_acc, dt);
    }

    double t_total = now_s() - t_start;

    /* sort per-epoch for median */
    double sorted[EPOCHS];
    for (int i = 0; i < EPOCHS; i++) sorted[i] = per_epoch[i];
    for (int i = 0; i < EPOCHS; i++)
        for (int j = i + 1; j < EPOCHS; j++)
            if (sorted[j] < sorted[i]) { double t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }
    double median = sorted[EPOCHS / 2];
    double mean = 0.0;
    for (int i = 0; i < EPOCHS; i++) mean += per_epoch[i];
    mean /= EPOCHS;

    printf("\n--- axiom cpu_opt training summary ---\n");
    printf("total training time: %.3fs\n", t_total);
    printf("mean per-epoch: %.3fs\n", mean);
    printf("median per-epoch: %.3fs\n", median);

    ax_optimizer_destroy(opt);
    ax_layer_destroy(m.fc1); ax_layer_destroy(m.fc2);
    ax_layer_destroy(m.fc3); ax_layer_destroy(m.fc4);
    ax_tensor_destroy(train_x); ax_tensor_destroy(train_y);
    ax_tensor_destroy(test_x);  free(test_labels);
}

/* ---- forward-only on cpu_opt OR cuda ---- */

static void run_infer(int use_cuda) {
    const char *label = use_cuda ? "cuda" : "cpu_opt";
    if (use_cuda) {
        ax_compute_set_backend(AX_BACKEND_CUDA);
        ax_set_default_device(AX_DEVICE_CUDA);
    } else {
        ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
        ax_set_default_device(AX_DEVICE_CPU);
    }
    ax_set_seed(42);

    printf("=== axiom infer-%s (%s backend) ===\n", label, label);
    printf("model: 784 -> %d -> %d -> %d -> 10\n", H1, H2, H3);
    printf("batch: %d  forward iters: %d\n\n", BATCH, FORWARD_ITERS);

    /* weights on the target device via the default-device mechanism */
    ax_tensor_t *w1 = ax_tensor_rand((int64_t[]){N_PIXELS, H1}, 2, -0.05f, 0.05f);
    ax_tensor_t *w2 = ax_tensor_rand((int64_t[]){H1, H2},       2, -0.05f, 0.05f);
    ax_tensor_t *w3 = ax_tensor_rand((int64_t[]){H2, H3},       2, -0.05f, 0.05f);
    ax_tensor_t *w4 = ax_tensor_rand((int64_t[]){H3, N_CLASSES},2, -0.05f, 0.05f);
    ax_tensor_t *x  = ax_tensor_rand((int64_t[]){BATCH, N_PIXELS}, 2, 0.0f, 1.0f);

    /* allocate output tensors once and reuse — mimics real inference
       loops where the activations are scratch space. */
    ax_tensor_t *h1 = ax_tensor_zeros((int64_t[]){BATCH, H1},       2, AX_FLOAT32);
    ax_tensor_t *h2 = ax_tensor_zeros((int64_t[]){BATCH, H2},       2, AX_FLOAT32);
    ax_tensor_t *h3 = ax_tensor_zeros((int64_t[]){BATCH, H3},       2, AX_FLOAT32);
    ax_tensor_t *logits = ax_tensor_zeros((int64_t[]){BATCH, N_CLASSES}, 2, AX_FLOAT32);
    ax_tensor_t *probs  = ax_tensor_zeros((int64_t[]){BATCH, N_CLASSES}, 2, AX_FLOAT32);

    /* warm up */
    for (int i = 0; i < 10; i++) {
        ax_compute_gemm(x, w1, h1);
        ax_compute_relu(h1, h1);
        ax_compute_gemm(h1, w2, h2);
        ax_compute_relu(h2, h2);
        ax_compute_gemm(h2, w3, h3);
        ax_compute_relu(h3, h3);
        ax_compute_gemm(h3, w4, logits);
        if (ax_compute_has_softmax_rowwise())
            ax_compute_softmax_rowwise(logits, probs);
    }
    if (use_cuda) ax_device_synchronize(AX_DEVICE_CUDA);

    double t0 = now_s();
    for (int i = 0; i < FORWARD_ITERS; i++) {
        ax_compute_gemm(x, w1, h1);
        ax_compute_relu(h1, h1);
        ax_compute_gemm(h1, w2, h2);
        ax_compute_relu(h2, h2);
        ax_compute_gemm(h2, w3, h3);
        ax_compute_relu(h3, h3);
        ax_compute_gemm(h3, w4, logits);
        if (ax_compute_has_softmax_rowwise())
            ax_compute_softmax_rowwise(logits, probs);
    }
    if (use_cuda) ax_device_synchronize(AX_DEVICE_CUDA);
    double t1 = now_s();

    double total = t1 - t0;
    double per_batch_ms = total / FORWARD_ITERS * 1000.0;
    double imgs_per_sec = (double)FORWARD_ITERS * BATCH / total;

    printf("--- axiom %s forward summary ---\n", label);
    printf("total forward time: %.3fs\n", total);
    printf("per-batch:   %.3f ms\n", per_batch_ms);
    printf("throughput:  %.0f images/s\n", imgs_per_sec);

    ax_tensor_destroy(w1); ax_tensor_destroy(w2);
    ax_tensor_destroy(w3); ax_tensor_destroy(w4);
    ax_tensor_destroy(x);
    ax_tensor_destroy(h1); ax_tensor_destroy(h2); ax_tensor_destroy(h3);
    ax_tensor_destroy(logits); ax_tensor_destroy(probs);
}

int main(int argc, char **argv) {
    ax_init();
    const char *mode = argc > 1 ? argv[1] : "train-cpu";
    if      (strcmp(mode, "train-cpu") == 0) run_train_cpu();
    else if (strcmp(mode, "infer-cpu") == 0) run_infer(0);
    else if (strcmp(mode, "infer-gpu") == 0) run_infer(1);
    else {
        fprintf(stderr, "usage: %s {train-cpu|infer-cpu|infer-gpu}\n", argv[0]);
        return 1;
    }
    ax_shutdown();
    return 0;
}
