/* spiral.c — 3-class spiral classification on synthetic 2D data.
   tests the framework on a problem completely different from MNIST:
     - 2D input (not 784D images)
     - non-linearly separable (decision boundary is a spiral)
     - purely synthetic (no data files needed)
     - smaller/faster than MNIST

   architecture: 2 -> 128 (gelu) -> 128 (gelu) -> 128 (gelu) -> 3
   optimizer: adam, lr=1e-3
   expected train accuracy: ~90%+ after 2000 epochs */

#include "axiom/axiom.h"
#include "axiom/rng.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define N_CLASSES      3
#define N_TRAIN_PER    300     /* training samples per class */
#define N_TEST_PER     100     /* test samples per class */
#define N_TRAIN        (N_CLASSES * N_TRAIN_PER)
#define N_TEST         (N_CLASSES * N_TEST_PER)
#define N_FEATURES     2
#define NOISE          0.15f   /* gaussian noise added to spiral arms */


/* box-muller: two independent standard normal samples from two uniform [0,1] */
static float randn(void)
{
    return ax_rng_normal();
}

/* generate N_CLASSES-arm spiral data.
   xy:     [n, 2] float  (output)
   labels: [n]    uint8  (output) */
static void generate_spiral(float *xy, uint8_t *labels, int n_per, int split_offset)
{
    int idx = 0;
    for (int k = 0; k < N_CLASSES; k++)
    {
        for (int i = 0; i < n_per; i++)
        {
            float t     = (float)i / (float)(n_per - 1);   /* [0, 1] */
            float angle = t * 4.0f * (float)M_PI
                        + (float)k * (2.0f * (float)M_PI / (float)N_CLASSES);
            float r     = 0.05f + t * 0.95f;               /* radius 0.05 -> 1.0 */

            int base = (split_offset + idx) * N_FEATURES;
            xy[base + 0] = r * cosf(angle) + NOISE * randn();
            xy[base + 1] = r * sinf(angle) + NOISE * randn();
            labels[split_offset + idx] = (uint8_t)k;
            idx++;
        }
    }
}

/* fisher-yates shuffle of [n] pairs (xy row, label) */
static void shuffle(float *xy, uint8_t *labels, int n)
{
    for (int i = n - 1; i > 0; i--)
    {
        int j = (int)ax_rng_bounded((uint64_t)(i + 1));
        /* swap labels */
        uint8_t tmp_l = labels[i]; labels[i] = labels[j]; labels[j] = tmp_l;
        /* swap xy rows */
        float tmp_x = xy[i * N_FEATURES + 0]; xy[i * N_FEATURES + 0] = xy[j * N_FEATURES + 0]; xy[j * N_FEATURES + 0] = tmp_x;
        float tmp_y = xy[i * N_FEATURES + 1]; xy[i * N_FEATURES + 1] = xy[j * N_FEATURES + 1]; xy[j * N_FEATURES + 1] = tmp_y;
    }
}

/* build one-hot [n, N_CLASSES] tensor from raw labels */
static ax_tensor_t *onehot(const uint8_t *labels, int n)
{
    int64_t shape[] = {(int64_t)n, (int64_t)N_CLASSES};
    ax_tensor_t *t = ax_tensor_zeros(shape, 2, AX_FLOAT32);
    if (!t) return NULL;
    float *d = (float *)t->storage->data;
    for (int i = 0; i < n; i++)
        d[i * N_CLASSES + labels[i]] = 1.0f;
    return t;
}

/* wrap a raw float array as an [n, N_FEATURES] tensor (copy) */
static ax_tensor_t *wrap_xy(const float *xy, int n)
{
    int64_t shape[] = {(int64_t)n, (int64_t)N_FEATURES};
    ax_tensor_t *t = ax_tensor_create(shape, 2, AX_FLOAT32);
    if (!t) return NULL;
    memcpy(t->storage->data, xy, (size_t)(n * N_FEATURES) * sizeof(float));
    return t;
}

/* evaluate accuracy (no_grad, eval mode) */
static float eval_accuracy(ax_layer_t *model,
                            ax_tensor_t *x, const uint8_t *labels, int n)
{
    ax_layer_eval(model);
    ax_no_grad();

    const int bs = 128;
    int correct = 0;
    float *xd = (float *)x->storage->data;

    for (int start = 0; start < n; start += bs)
    {
        int b = (start + bs <= n) ? bs : (n - start);
        int64_t shape[] = {(int64_t)b, (int64_t)N_FEATURES};
        ax_tensor_t *xb = ax_tensor_create(shape, 2, AX_FLOAT32);
        if (!xb) break;
        memcpy(xb->storage->data, xd + start * N_FEATURES,
               (size_t)(b * N_FEATURES) * sizeof(float));

        ax_tensor_t *logits = ax_layer_forward(model, xb);
        ax_tensor_destroy(xb);
        if (!logits) break;

        float *ld = (float *)logits->storage->data;
        for (int i = 0; i < b; i++)
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

    ax_layer_train(model);
    ax_enable_grad();
    return 100.0f * (float)correct / (float)n;
}


int main(void)
{
    ax_init();
    ax_set_seed(42);

    /* ---- generate data ---- */
    float   *train_xy  = malloc((size_t)(N_TRAIN * N_FEATURES) * sizeof(float));
    uint8_t *train_lbl = malloc((size_t)N_TRAIN);
    float   *test_xy   = malloc((size_t)(N_TEST  * N_FEATURES) * sizeof(float));
    uint8_t *test_lbl  = malloc((size_t)N_TEST);
    if (!train_xy || !train_lbl || !test_xy || !test_lbl)
    {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    generate_spiral(train_xy, train_lbl, N_TRAIN_PER, 0);
    generate_spiral(test_xy,  test_lbl,  N_TEST_PER,  0);
    shuffle(train_xy, train_lbl, N_TRAIN);

    printf("spiral dataset: %d train, %d test, %d features, %d classes\n",
           N_TRAIN, N_TEST, N_FEATURES, N_CLASSES);

    ax_tensor_t *train_x = wrap_xy(train_xy, N_TRAIN);
    ax_tensor_t *train_y = onehot(train_lbl, N_TRAIN);
    ax_tensor_t *test_x  = wrap_xy(test_xy,  N_TEST);
    if (!train_x || !train_y || !test_x)
    {
        fprintf(stderr, "failed to create tensors\n");
        return 1;
    }

    /* ---- overfit sanity check: memorize 9 noise-free samples ---- */
    {
        float tiny_xy[9 * 2];
        uint8_t tiny_lbl[9];
        /* 3 clean points per class (no noise, no shuffle) */
        int idx = 0;
        for (int k = 0; k < 3; k++) {
            for (int i = 0; i < 3; i++) {
                float t = 0.1f + i * 0.4f;  /* t in {0.1, 0.5, 0.9} */
                float angle = t * 4.0f * (float)M_PI + k * (2.0f * (float)M_PI / 3.0f);
                float r = 0.05f + t * 0.95f;
                tiny_xy[idx * 2 + 0] = r * cosf(angle);
                tiny_xy[idx * 2 + 1] = r * sinf(angle);
                tiny_lbl[idx] = (uint8_t)k;
                idx++;
            }
        }
        int64_t tx_shape[] = {9, 2};
        int64_t ty_shape[] = {9, 3};
        ax_tensor_t *tx = ax_tensor_create(tx_shape, 2, AX_FLOAT32);
        ax_tensor_t *ty = ax_tensor_zeros(ty_shape, 2, AX_FLOAT32);
        memcpy(tx->storage->data, tiny_xy, 9 * 2 * sizeof(float));
        float *tyd = (float *)ty->storage->data;
        for (int i = 0; i < 9; i++) tyd[i * 3 + tiny_lbl[i]] = 1.0f;

        ax_layer_t *tiny_model = ax_sequential_create();
        ax_sequential_add(tiny_model, ax_dense_create(2, 64, true));
        ax_sequential_add(tiny_model, ax_relu_layer_create());
        ax_sequential_add(tiny_model, ax_dense_create(64, 64, true));
        ax_sequential_add(tiny_model, ax_relu_layer_create());
        ax_sequential_add(tiny_model, ax_dense_create(64, 3, true));

        ax_tensor_t *tparams[32];
        int tnp = ax_layer_get_params(tiny_model, tparams, 32);
        ax_optimizer_t *topt = ax_adam_create(tparams, tnp, 1e-2f, 0.9f, 0.999f, 1e-8f, 0.0f);

        printf("overfit test: 9 samples, 2000 epochs, single batch\n");
        float prev_loss = 1e9f;
        for (int ep = 0; ep < 2000; ep++) {
            ax_enable_grad();
            /* must copy tx each step since ax_graph_cleanup destroys leaf tensors too?
               No — ax_graph_cleanup only destroys non-leaf intermediates.
               But the input to the model (tx) is passed through graph, saved in grad_fn.
               Actually, tx is a leaf (no grad_fn, requires_grad=false) so it's safe. */
            ax_tensor_t *logits = ax_layer_forward(tiny_model, tx);
            ax_tensor_t *loss = ax_cross_entropy_loss(logits, ty);
            float lv = ((float *)loss->storage->data)[0];

            ax_optimizer_zero_grad(topt);
            ax_backward(loss);
            ax_optimizer_step(topt);
            ax_graph_cleanup(loss);
            ax_tensor_destroy(loss);

            if (ep == 0 || (ep + 1) % 500 == 0)
                printf("  epoch %4d  loss=%.6f\n", ep + 1, lv);
            prev_loss = lv;
        }
        printf("  final loss=%.6f (target: <0.01 to confirm gradients work)\n\n", prev_loss);
        fflush(stdout);

        ax_optimizer_destroy(topt);
        ax_layer_destroy(tiny_model);
        ax_tensor_destroy(tx);
        ax_tensor_destroy(ty);
    }

    /* ---- model: 2 -> 128 (gelu) -> 128 (gelu) -> 128 (gelu) -> 3 ---- */
    ax_layer_t *model = ax_sequential_create();
    ax_sequential_add(model, ax_dense_create(N_FEATURES, 128, true));
    ax_sequential_add(model, ax_gelu_layer_create());
    ax_sequential_add(model, ax_dense_create(128, 128, true));
    ax_sequential_add(model, ax_gelu_layer_create());
    ax_sequential_add(model, ax_dense_create(128, 128, true));
    ax_sequential_add(model, ax_gelu_layer_create());
    ax_sequential_add(model, ax_dense_create(128, N_CLASSES, true));

    ax_tensor_t *params[32];
    int n_params = ax_layer_get_params(model, params, 32);
    ax_optimizer_t *opt = ax_adam_create(params, n_params,
                                          1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f);

    ax_dataset_t    *ds = ax_tensor_dataset_create(train_x, train_y);
    ax_dataloader_t *dl = ax_dataloader_create(ds, 32, true);

    printf("model: %ld parameters, %ld batches/epoch\n\n",
           ax_layer_param_count(model), ax_dataloader_num_batches(dl));
    fflush(stdout);

    /* ---- training loop ---- */
    const int epochs = 2000;
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

            ax_graph_cleanup(loss);
            ax_tensor_destroy(loss);
            ax_tensor_destroy(b.input);
            ax_tensor_destroy(b.target);
        }

        /* print gradient norms for epoch 1 */
        if (ep == 0)
        {
            printf("  grad norms: ");
            const char *pnames[] = {"W1","b1","W2","b2","W3","b3"};
            for (int pi = 0; pi < n_params; pi++) {
                if (!params[pi]->grad) { printf("%s=NULL ", pnames[pi]); continue; }
                float s = 0.0f;
                float *gd = (float *)params[pi]->grad->storage->data;
                int64_t np2 = ax_tensor_numel(params[pi]->grad);
                for (int64_t j = 0; j < np2; j++) s += gd[j]*gd[j];
                printf("%s=%.4f ", pnames[pi], sqrtf(s));
            }
            printf("\n"); fflush(stdout);
        }
        /* print every 50 epochs */
        if ((ep + 1) % 500 == 0 || ep == 0)
        {
            float train_acc = eval_accuracy(model, train_x, train_lbl, N_TRAIN);
            float test_acc  = eval_accuracy(model, test_x,  test_lbl,  N_TEST);
            printf("epoch %3d/%d  loss=%.4f  train_acc=%.2f%%  test_acc=%.2f%%\n",
                   ep + 1, epochs,
                   (float)(total_loss / (double)n_batches),
                   train_acc, test_acc);
            fflush(stdout);
        }
    }

    /* ---- cleanup ---- */
    ax_dataloader_destroy(dl);
    ax_dataset_destroy(ds);
    ax_optimizer_destroy(opt);
    ax_layer_destroy(model);
    ax_tensor_destroy(train_x);
    ax_tensor_destroy(train_y);
    ax_tensor_destroy(test_x);
    free(train_xy);
    free(train_lbl);
    free(test_xy);
    free(test_lbl);

    ax_shutdown();
    return 0;
}
