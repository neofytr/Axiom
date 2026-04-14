/* bench_transformer.c — mini transformer encoder training loop.

   architecture:
     input:  [B, S, d_model]
     repeat N_LAYERS of:
        residual(MHA(LN(x)))
        residual(MLP(LN(x)))
     final classifier head → [B, S, vocab]

   reports steps/sec on synthetic data.  this is the end-to-end sanity
   check for the MHA layer: forward + backward + optimizer, running
   against a realistic transformer shape. */

#include "axiom/axiom.h"
#include "bench_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define B          8
#define S          128
#define D_MODEL    256
#define N_HEADS    4
#define D_FF       1024
#define N_LAYERS   6
#define VOCAB      1024
#define STEPS      20

/* build one encoder block as a sequential: LN -> MHA -> residual (approx
   via a Dense) -> LN -> MLP. true residuals need a graph; this mini
   benchmark approximates by placing LN / MHA / projections in sequence.
   the flop mix is representative of a real transformer encoder. */
static ax_layer_t *build_block(void)
{
    ax_layer_t *m = ax_sequential_create();
    ax_sequential_add(m, ax_layernorm_create(D_MODEL, 1e-5f));
    ax_sequential_add(m, ax_mha_create(D_MODEL, N_HEADS, true, false));
    ax_sequential_add(m, ax_layernorm_create(D_MODEL, 1e-5f));
    /* FFN: two dense layers with GELU between */
    ax_sequential_add(m, ax_dense_create(D_MODEL, D_FF, true));
    ax_sequential_add(m, ax_gelu_layer_create());
    ax_sequential_add(m, ax_dense_create(D_FF, D_MODEL, true));
    return m;
}

static ax_layer_t *build_model(void)
{
    ax_layer_t *m = ax_sequential_create();
    for (int i = 0; i < N_LAYERS; i++)
        ax_sequential_add(m, build_block());
    ax_sequential_add(m, ax_layernorm_create(D_MODEL, 1e-5f));
    ax_sequential_add(m, ax_dense_create(D_MODEL, VOCAB, true));
    return m;
}

int main(void)
{
    ax_init();
    ax_set_seed(42);

    ax_layer_t *model = build_model();

    int64_t x_sh[] = {B, S, D_MODEL};
    ax_tensor_t *x = ax_tensor_rand(x_sh, 3, -0.1f, 0.1f);
    x->requires_grad = true;

    /* target: cross-entropy over the last-position logits. use random
       one-hot-ish labels for timing stability. */
    int64_t y_sh[] = {B * S, VOCAB};
    ax_tensor_t *y = ax_tensor_zeros(y_sh, 2, AX_FLOAT32);
    float *yd = (float *)y->storage->data;
    for (int64_t i = 0; i < B * S; i++)
        yd[i * VOCAB + (i % VOCAB)] = 1.0f;

    ax_tensor_t *params[1024];
    int np = ax_layer_get_params(model, params, 1024);
    printf("# transformer: B=%d S=%d d=%d H=%d ff=%d layers=%d params=%d\n",
           B, S, D_MODEL, N_HEADS, D_FF, N_LAYERS, (int)ax_layer_param_count(model));
    ax_optimizer_t *opt = ax_adam_create(params, np, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f);

    /* warmup */
    {
        ax_enable_grad();
        ax_tensor_t *logits = ax_layer_forward(model, x);
        if (logits) {
            /* reshape [B, S, VOCAB] -> [B*S, VOCAB] for cross-entropy */
            int64_t flat_sh[] = {B * S, VOCAB};
            ax_tensor_t *flat = ax_tensor_reshape(logits, flat_sh, 2);
            ax_tensor_t *loss = ax_cross_entropy_loss(flat, y);
            if (loss) {
                ax_optimizer_zero_grad(opt);
                ax_backward(loss);
                ax_optimizer_step(opt);
                ax_graph_cleanup(loss);
                ax_tensor_destroy(loss);
            }
            ax_tensor_destroy(flat);
        }
    }

    double t0 = bench_now_s();
    for (int step = 0; step < STEPS; step++) {
        ax_enable_grad();
        ax_tensor_t *logits = ax_layer_forward(model, x);
        if (!logits) { fprintf(stderr, "fwd fail at step %d\n", step); break; }

        int64_t flat_sh[] = {B * S, VOCAB};
        ax_tensor_t *flat = ax_tensor_reshape(logits, flat_sh, 2);
        ax_tensor_t *loss = ax_cross_entropy_loss(flat, y);
        if (!loss) { ax_tensor_destroy(flat); break; }

        ax_optimizer_zero_grad(opt);
        ax_backward(loss);
        ax_optimizer_step(opt);

        ax_graph_cleanup(loss);
        ax_tensor_destroy(loss);
        ax_tensor_destroy(flat);
    }
    double dt = bench_now_s() - t0;

    /* tokens/sec and approx GFLOPS estimate (6x forward flops for train) */
    double tokens = (double)STEPS * B * S;
    double tok_per_sec = tokens / dt;
    char cs[96];
    snprintf(cs, sizeof(cs), "encoder_B%d_S%d_d%d_L%d", B, S, D_MODEL, N_LAYERS);
    BENCH_EMIT("transformer", cs, "sec", dt);
    BENCH_EMIT("transformer", cs, "tokens_per_sec", tok_per_sec);
    BENCH_EMIT("transformer", cs, "ms_per_step", dt * 1000.0 / STEPS);

    ax_tensor_destroy(x); ax_tensor_destroy(y);
    ax_layer_destroy(model);
    ax_shutdown();
    return 0;
}
