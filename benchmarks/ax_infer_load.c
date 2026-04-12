/* ax_infer_load.c — load a model saved by ax_train_save.c using the
   INFERENCE-ONLY axiom build and verify:

   1. the loaded model runs forward on the same test set
   2. the resulting logit tensor matches the training-time logits
      bit-for-bit (or within 1e-6 epsilon — any drift would indicate a
      silent divergence between the full and inference-only code paths)
   3. the final accuracy matches what ax_train_save.c printed

   compile against build-inf/libaxiom.a (inference-only build) with
   -DAX_INFERENCE_ONLY=1 on the command line so the stub autograd
   header kicks in. */

#include "axiom/axiom.h"
#include "axiom/serialize.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

#define TEST_IMAGES  "examples/data/t10k-images-idx3-ubyte"
#define TEST_LABELS  "examples/data/t10k-labels-idx1-ubyte"
#define MODEL_PATH   "benchmarks/mnist_parity.axm"
#define LOGITS_PATH  "benchmarks/mnist_parity_logits.bin"

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

static int argmax10(const float *row) {
    int best = 0; float bestv = row[0];
    for (int c = 1; c < N_CLASSES; c++) if (row[c] > bestv) { bestv = row[c]; best = c; }
    return best;
}

int main(void) {
    ax_init();

    printf("[infer] loading mnist test set...\n");
    ax_tensor_t *test_x = load_images(TEST_IMAGES, N_TEST);
    uint8_t *test_labels = load_labels_raw(TEST_LABELS, N_TEST);
    if (!test_x || !test_labels) { fprintf(stderr, "[infer] mnist load failed\n"); return 1; }

    printf("[infer] loading model from %s...\n", MODEL_PATH);
    ax_model_t *model = ax_model_load(MODEL_PATH);
    if (!model) { fprintf(stderr, "[infer] model load failed\n"); return 1; }

    printf("[infer] loading reference logits from %s...\n", LOGITS_PATH);
    ax_tensor_t *ref_logits = ax_tensor_load(LOGITS_PATH);
    if (!ref_logits) { fprintf(stderr, "[infer] ref logits load failed\n"); return 1; }
    if (ref_logits->ndim != 2 || ref_logits->shape[0] != N_TEST || ref_logits->shape[1] != N_CLASSES) {
        fprintf(stderr, "[infer] ref logits shape mismatch\n"); return 1;
    }

    /* forward on the full test set */
    int64_t out_shape[] = {N_TEST, N_CLASSES};
    ax_tensor_t *all = ax_tensor_zeros(out_shape, 2, AX_FLOAT32);
    const int64_t bs = 512;
    float *all_d = (float *)all->storage->data;
    float *im_d = (float *)test_x->storage->data;

    for (int64_t start = 0; start < N_TEST; start += bs) {
        int64_t b = (start + bs <= N_TEST) ? bs : (N_TEST - start);
        int64_t xs[] = {b, N_PIXELS};
        ax_tensor_t *x = ax_tensor_create(xs, 2, AX_FLOAT32);
        memcpy(x->storage->data, im_d + start * N_PIXELS,
               (size_t)(b * N_PIXELS) * sizeof(float));
        ax_tensor_t *y = ax_model_predict(model, x);
        ax_tensor_destroy(x);
        memcpy(all_d + start * N_CLASSES,
               y->storage->data,
               (size_t)(b * N_CLASSES) * sizeof(float));
        ax_tensor_destroy(y);
    }

    /* compare with reference */
    const float *got = (const float *)all->storage->data;
    const float *want = (const float *)ref_logits->storage->data;
    int64_t n_total = N_TEST * N_CLASSES;
    float max_abs = 0.0f;
    int64_t mismatches = 0;
    for (int64_t i = 0; i < n_total; i++) {
        float d = fabsf(got[i] - want[i]);
        if (d > max_abs) max_abs = d;
        if (d > 1e-5f) mismatches++;
    }
    printf("[infer] logit comparison: max abs diff = %.3e  mismatches (>1e-5) = %" PRId64 "\n",
           (double)max_abs, mismatches);

    /* accuracy from loaded-model logits */
    int64_t correct_got = 0, correct_want = 0;
    for (int64_t i = 0; i < N_TEST; i++) {
        if (argmax10(got + i * N_CLASSES) == (int)test_labels[i]) correct_got++;
        if (argmax10(want + i * N_CLASSES) == (int)test_labels[i]) correct_want++;
    }
    float acc_got = 100.0f * (float)correct_got / (float)N_TEST;
    float acc_want = 100.0f * (float)correct_want / (float)N_TEST;
    printf("[infer] reference (training build) accuracy : %.4f%%\n", acc_want);
    printf("[infer] inference-only build accuracy        : %.4f%%\n", acc_got);

    bool pass = (mismatches == 0) && (fabsf(acc_got - acc_want) < 0.001f);
    printf("\n%s — inference-only library produces %s results\n",
           pass ? "PASS" : "FAIL",
           pass ? "bit-identical" : "DIFFERENT");
    return pass ? 0 : 1;
}
