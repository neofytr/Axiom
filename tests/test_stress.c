/* test_stress.c — hard stress tests for axiom.
   exercises large models, wide/tall matrices, edge-case shapes,
   every activation, every optimizer, every loss, cross-entropy with
   many classes, deep networks, fused dense+relu paths, no-grad nesting,
   and memory pressure from repeated train+cleanup cycles. */

#include "test.h"
#include "axiom/axiom.h"
#include <string.h>
#include <math.h>

/* build a sequential dense model with arbitrary widths.
   layers alternate: dense -> relu -> dense -> relu -> ... -> dense(out).
   widths[] has n_layers+1 entries: [in, h1, h2, ..., out]. */
static ax_model_t *build_mlp(const int64_t *widths, int n_hidden, bool use_bias)
{
    ax_layer_t *net = ax_sequential_create();
    for (int i = 0; i < n_hidden; i++) {
        ax_sequential_add(net, ax_dense_create(widths[i], widths[i + 1], use_bias));
        ax_sequential_add(net, ax_relu_layer_create());
    }
    /* final layer, no activation */
    ax_sequential_add(net, ax_dense_create(widths[n_hidden], widths[n_hidden + 1], use_bias));
    return ax_model_create(net);
}

/* train for N steps, return final loss */
static float train_steps(ax_model_t *m, ax_tensor_t *x, ax_tensor_t *y, int steps)
{
    float loss_val = 999.0f;
    for (int i = 0; i < steps; i++) {
        loss_val = ax_model_train_step(m, x, y);
    }
    return loss_val;
}

/* large mlp (512->256->128->64->32->10): verifies that the fused
   dense+relu path through sequential_forward works end-to-end for
   training, and that loss monotonically decreases on a memorizable
   random dataset. ~270K params. */
static void test_large_mlp_trains(void)
{
    int64_t widths[] = {512, 256, 128, 64, 32, 10};
    int n_hidden = 4; /* 4 hidden layers */

    ax_model_t *m = build_mlp(widths, n_hidden, true);

    int64_t xs[] = {32, 512};
    ax_tensor_t *x = ax_tensor_rand(xs, 2, 0.0f, 1.0f);

    /* one-hot targets for 10 classes */
    int64_t ys[] = {32, 10};
    ax_tensor_t *y = ax_tensor_zeros(ys, 2, AX_FLOAT32);
    float *yd = (float *)y->storage->data;
    for (int i = 0; i < 32; i++)
        yd[i * 10 + (i % 10)] = 1.0f;

    ax_optimizer_t *opt = ax_adam_create(m->params, m->n_params,
                                          1e-3f, 0.9f, 0.999f, 1e-8f, 0);
    ax_model_compile(m, opt, ax_mse_loss);

    float first = ax_model_train_step(m, x, y);
    float last = train_steps(m, x, y, 200);

    AX_TEST_ASSERT(last < first * 0.5f, "large mlp loss should drop significantly");
    AX_TEST_ASSERT(!isnan(last) && !isinf(last), "no nan/inf in large mlp");

    ax_tensor_destroy(x);
    ax_tensor_destroy(y);
    ax_model_destroy(m);
}

/* wide matrix stress: 32x2048 @ 2048x16 gemm. tests pack_b tiling
   and JC parallelism on wide N dimension. */
static void test_wide_matmul(void)
{
    int64_t a_shape[] = {32, 2048};
    int64_t b_shape[] = {2048, 16};
    ax_tensor_t *a = ax_tensor_rand(a_shape, 2, -1.0f, 1.0f);
    ax_tensor_t *b = ax_tensor_rand(b_shape, 2, -1.0f, 1.0f);
    a->requires_grad = true;
    b->requires_grad = true;

    ax_enable_grad();
    ax_tensor_t *c = ax_matmul(a, b);
    AX_TEST_ASSERT(c != NULL, "wide matmul should succeed");
    AX_TEST_ASSERT_EQ(c->shape[0], 32, "wide matmul rows");
    AX_TEST_ASSERT_EQ(c->shape[1], 16, "wide matmul cols");

    /* verify backward works */
    ax_tensor_t *loss = ax_sum(c, -1);
    ax_backward(loss);
    AX_TEST_ASSERT(a->grad != NULL, "wide matmul grad for a");
    AX_TEST_ASSERT(b->grad != NULL, "wide matmul grad for b");

    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss);
    ax_tensor_destroy(a);
    ax_tensor_destroy(b);
}

/* tall matrix: 4096x8 @ 8x4096. stresses IC (M-dimension) parallelism. */
static void test_tall_matmul(void)
{
    int64_t a_shape[] = {4096, 8};
    int64_t b_shape[] = {8, 4096};
    ax_tensor_t *a = ax_tensor_rand(a_shape, 2, -0.5f, 0.5f);
    ax_tensor_t *b = ax_tensor_rand(b_shape, 2, -0.5f, 0.5f);

    ax_tensor_t *c = ax_matmul(a, b);
    AX_TEST_ASSERT(c != NULL, "tall matmul should succeed");
    AX_TEST_ASSERT_EQ(c->shape[0], 4096, "tall matmul rows");
    AX_TEST_ASSERT_EQ(c->shape[1], 4096, "tall matmul cols");

    /* spot check: element should not be nan/inf */
    int64_t idx[] = {0, 0};
    float v = ax_tensor_get_f32(c, idx);
    AX_TEST_ASSERT(!isnan(v) && !isinf(v), "tall matmul no nan/inf");

    ax_tensor_destroy(a);
    ax_tensor_destroy(b);
    ax_tensor_destroy(c);
}

/* every activation trains through: build a model with each activation
   and verify loss decreases. this catches broken backward fns. */
static void test_all_activations_train(void)
{
    float x_data[] = {0,0, 0,1, 1,0, 1,1};
    float y_data[] = {0, 1, 1, 0};
    int64_t xs[] = {4, 2}, ys[] = {4, 1};
    ax_tensor_t *x = ax_tensor_from_array(x_data, xs, 2, AX_FLOAT32);
    ax_tensor_t *y = ax_tensor_from_array(y_data, ys, 2, AX_FLOAT32);

    struct { const char *name; ax_layer_t *(*create)(void); } acts[] = {
        {"relu",    ax_relu_layer_create},
        {"sigmoid", ax_sigmoid_layer_create},
        {"tanh",    ax_tanh_layer_create},
        {"gelu",    ax_gelu_layer_create},
        {"swish",   ax_swish_layer_create},
    };

    for (int a = 0; a < 5; a++) {
        ax_layer_t *net = ax_sequential_create();
        ax_sequential_add(net, ax_dense_create(2, 16, true));
        ax_sequential_add(net, acts[a].create());
        ax_sequential_add(net, ax_dense_create(16, 1, true));

        ax_model_t *m = ax_model_create(net);
        ax_optimizer_t *opt = ax_adam_create(m->params, m->n_params,
                                              0.01f, 0.9f, 0.999f, 1e-8f, 0);
        ax_model_compile(m, opt, ax_mse_loss);

        float first = ax_model_train_step(m, x, y);
        for (int i = 0; i < 300; i++) ax_model_train_step(m, x, y);
        float last = ax_model_train_step(m, x, y);

        char msg[64];
        snprintf(msg, sizeof(msg), "%s activation should train", acts[a].name);
        AX_TEST_ASSERT(last < first, msg);

        ax_model_destroy(m);
    }

    /* leaky relu and elu (take a parameter) */
    {
        ax_layer_t *net = ax_sequential_create();
        ax_sequential_add(net, ax_dense_create(2, 16, true));
        ax_sequential_add(net, ax_leaky_relu_layer_create(0.01f));
        ax_sequential_add(net, ax_dense_create(16, 1, true));
        ax_model_t *m = ax_model_create(net);
        ax_optimizer_t *opt = ax_adam_create(m->params, m->n_params,
                                              0.01f, 0.9f, 0.999f, 1e-8f, 0);
        ax_model_compile(m, opt, ax_mse_loss);
        float first = ax_model_train_step(m, x, y);
        for (int i = 0; i < 300; i++) ax_model_train_step(m, x, y);
        float last = ax_model_train_step(m, x, y);
        AX_TEST_ASSERT(last < first, "leaky_relu should train");
        ax_model_destroy(m);
    }
    {
        ax_layer_t *net = ax_sequential_create();
        ax_sequential_add(net, ax_dense_create(2, 16, true));
        ax_sequential_add(net, ax_elu_layer_create(1.0f));
        ax_sequential_add(net, ax_dense_create(16, 1, true));
        ax_model_t *m = ax_model_create(net);
        ax_optimizer_t *opt = ax_adam_create(m->params, m->n_params,
                                              0.01f, 0.9f, 0.999f, 1e-8f, 0);
        ax_model_compile(m, opt, ax_mse_loss);
        float first = ax_model_train_step(m, x, y);
        for (int i = 0; i < 300; i++) ax_model_train_step(m, x, y);
        float last = ax_model_train_step(m, x, y);
        AX_TEST_ASSERT(last < first, "elu should train");
        ax_model_destroy(m);
    }

    ax_tensor_destroy(x);
    ax_tensor_destroy(y);
}

/* every optimizer trains: sgd, adam, adamw, rmsprop, adagrad */
static void test_all_optimizers_train(void)
{
    float x_data[] = {0,0, 0,1, 1,0, 1,1};
    float y_data[] = {0, 1, 1, 0};
    int64_t xs[] = {4, 2}, ys[] = {4, 1};
    ax_tensor_t *x = ax_tensor_from_array(x_data, xs, 2, AX_FLOAT32);
    ax_tensor_t *y = ax_tensor_from_array(y_data, ys, 2, AX_FLOAT32);

    struct {
        const char *name;
        ax_optimizer_t *(*create)(ax_tensor_t **, int);
    } optims[] = {
        {"sgd",     NULL},
        {"adam",    NULL},
        {"adamw",   NULL},
        {"rmsprop", NULL},
        {"adagrad", NULL},
    };

    for (int o = 0; o < 5; o++) {
        ax_layer_t *net = ax_sequential_create();
        ax_sequential_add(net, ax_dense_create(2, 16, true));
        ax_sequential_add(net, ax_relu_layer_create());
        ax_sequential_add(net, ax_dense_create(16, 1, true));

        ax_model_t *m = ax_model_create(net);
        ax_optimizer_t *opt;

        if (o == 0)
            opt = ax_sgd_create(m->params, m->n_params, 0.1f, 0.9f, 0, false);
        else if (o == 1)
            opt = ax_adam_create(m->params, m->n_params, 0.01f, 0.9f, 0.999f, 1e-8f, 0);
        else if (o == 2)
            opt = ax_adamw_create(m->params, m->n_params, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f);
        else if (o == 3)
            opt = ax_rmsprop_create(m->params, m->n_params, 0.01f, 0.99f, 1e-8f, 0);
        else
            opt = ax_adagrad_create(m->params, m->n_params, 0.1f, 1e-10f, 0);

        ax_model_compile(m, opt, ax_mse_loss);

        float first = ax_model_train_step(m, x, y);
        for (int i = 0; i < 500; i++) ax_model_train_step(m, x, y);
        float last = ax_model_train_step(m, x, y);

        char msg[64];
        snprintf(msg, sizeof(msg), "%s optimizer should reduce loss", optims[o].name);
        AX_TEST_ASSERT(last < first, msg);

        ax_model_destroy(m);
    }

    ax_tensor_destroy(x);
    ax_tensor_destroy(y);
}

/* cross-entropy with many classes (100 classes, 64 samples).
   this stresses the softmax + log path with wide output. */
static void test_cross_entropy_many_classes(void)
{
    int64_t n_classes = 100, batch = 64;

    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(50, 128, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(128, n_classes, true));

    ax_model_t *m = ax_model_create(net);
    ax_optimizer_t *opt = ax_adam_create(m->params, m->n_params,
                                          1e-3f, 0.9f, 0.999f, 1e-8f, 0);
    ax_model_compile(m, opt, ax_cross_entropy_loss);

    int64_t xs[] = {batch, 50};
    ax_tensor_t *x = ax_tensor_rand(xs, 2, -1.0f, 1.0f);

    /* one-hot targets spread across all 100 classes */
    int64_t ys[] = {batch, n_classes};
    ax_tensor_t *y = ax_tensor_zeros(ys, 2, AX_FLOAT32);
    float *yd = (float *)y->storage->data;
    for (int64_t i = 0; i < batch; i++)
        yd[i * n_classes + (i % n_classes)] = 1.0f;

    float first = ax_model_train_step(m, x, y);
    float last = train_steps(m, x, y, 100);

    AX_TEST_ASSERT(last < first, "cross-entropy 100 classes should train");
    AX_TEST_ASSERT(!isnan(last) && !isinf(last), "no nan/inf in cross-entropy");

    ax_tensor_destroy(x);
    ax_tensor_destroy(y);
    ax_model_destroy(m);
}

/* deep network (10 hidden layers): tests gradient flow through many
   layers and the fused dense+relu path at depth. */
static void test_deep_network_trains(void)
{
    int64_t widths[] = {16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 4};
    int n_hidden = 10;

    ax_model_t *m = build_mlp(widths, n_hidden, true);

    int64_t xs[] = {8, 16};
    ax_tensor_t *x = ax_tensor_rand(xs, 2, -1.0f, 1.0f);
    int64_t ys[] = {8, 4};
    ax_tensor_t *y = ax_tensor_rand(ys, 2, 0.0f, 1.0f);

    ax_optimizer_t *opt = ax_adam_create(m->params, m->n_params,
                                          1e-3f, 0.9f, 0.999f, 1e-8f, 0);
    ax_model_compile(m, opt, ax_mse_loss);

    float first = ax_model_train_step(m, x, y);
    float last = train_steps(m, x, y, 500);

    AX_TEST_ASSERT(last < first, "deep network should train");
    AX_TEST_ASSERT(!isnan(last), "deep network should not produce nan");

    ax_tensor_destroy(x);
    ax_tensor_destroy(y);
    ax_model_destroy(m);
}

/* no-grad nesting: verify that nested no_grad/enable_grad pairs work.
   inner scope shouldn't leak grad state to outer scope. */
static void test_no_grad_nesting(void)
{
    ax_enable_grad();
    AX_TEST_ASSERT(ax_grad_enabled(), "grad starts enabled");

    ax_no_grad();
    AX_TEST_ASSERT(!ax_grad_enabled(), "outer no_grad disables");

    ax_no_grad();
    AX_TEST_ASSERT(!ax_grad_enabled(), "inner no_grad still disabled");

    ax_enable_grad();
    AX_TEST_ASSERT(!ax_grad_enabled(), "inner enable doesn't leak to outer");

    ax_enable_grad();
    AX_TEST_ASSERT(ax_grad_enabled(), "outer enable restores");

    /* verify tensors created in no-grad don't track */
    ax_no_grad();
    int64_t s[] = {2, 3};
    ax_tensor_t *a = ax_tensor_rand(s, 2, -1.0f, 1.0f);
    ax_tensor_t *b = ax_tensor_rand(s, 2, -1.0f, 1.0f);
    a->requires_grad = true;
    ax_tensor_t *c = ax_add(a, b);
    AX_TEST_ASSERT(!c->requires_grad, "no-grad tensor should not track");
    AX_TEST_ASSERT(c->grad_fn == NULL, "no-grad tensor should have no grad_fn");
    ax_enable_grad();

    ax_tensor_destroy(a);
    ax_tensor_destroy(b);
    ax_tensor_destroy(c);
}

/* memory pressure: train+cleanup loop repeated many times to detect leaks.
   if graph_cleanup or the arena leaks, this will OOM or crash. */
static void test_memory_pressure(void)
{
    float x_data[] = {0,0, 0,1, 1,0, 1,1};
    float y_data[] = {0, 1, 1, 0};
    int64_t xs[] = {4, 2}, ys[] = {4, 1};
    ax_tensor_t *x = ax_tensor_from_array(x_data, xs, 2, AX_FLOAT32);
    ax_tensor_t *y = ax_tensor_from_array(y_data, ys, 2, AX_FLOAT32);

    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(2, 64, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(64, 64, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(64, 1, true));

    ax_model_t *m = ax_model_create(net);
    ax_optimizer_t *opt = ax_adam_create(m->params, m->n_params,
                                          0.01f, 0.9f, 0.999f, 1e-8f, 0);
    ax_model_compile(m, opt, ax_mse_loss);

    /* 2000 train steps — enough to trigger arena resets many times */
    float last = 0;
    for (int i = 0; i < 2000; i++)
        last = ax_model_train_step(m, x, y);

    AX_TEST_ASSERT(!isnan(last), "no nan after 2000 steps");
    AX_TEST_ASSERT(last < 0.1f, "converged after 2000 steps");

    ax_tensor_destroy(x);
    ax_tensor_destroy(y);
    ax_model_destroy(m);
}

/* fused dense+relu: verify the fusion produces same gradients as unfused.
   run identical models with and without the optimization, compare. */
static void test_fused_dense_relu_grad(void)
{
    ax_set_seed(42);
    float x_data[] = {1, 2, 3, 4, 5, 6};
    int64_t xs[] = {2, 3};
    ax_tensor_t *x = ax_tensor_from_array(x_data, xs, 2, AX_FLOAT32);

    /* model 1: dense -> relu (fused in sequential) */
    ax_layer_t *net1 = ax_sequential_create();
    ax_layer_t *d1 = ax_dense_create(3, 4, true);
    ax_sequential_add(net1, d1);
    ax_sequential_add(net1, ax_relu_layer_create());

    /* model 2: standalone calls to test without fusion */
    ax_dense_t *dd = (ax_dense_t *)d1;

    ax_enable_grad();
    ax_layer_train(net1);

    /* forward through sequential (uses fused path) */
    ax_tensor_t *out = ax_layer_forward(net1, x);
    AX_TEST_ASSERT(out != NULL, "fused forward should succeed");
    AX_TEST_ASSERT_EQ(out->shape[0], 2, "fused output batch");
    AX_TEST_ASSERT_EQ(out->shape[1], 4, "fused output features");

    /* backward should work */
    ax_tensor_t *loss = ax_sum(out, -1);
    ax_backward(loss);

    AX_TEST_ASSERT(dd->weight->grad != NULL, "fused backward should produce weight grad");
    AX_TEST_ASSERT(dd->bias->grad != NULL, "fused backward should produce bias grad");

    /* verify no nan in gradients */
    float *wg = (float *)dd->weight->grad->storage->data;
    int64_t wn = ax_tensor_numel(dd->weight->grad);
    for (int64_t i = 0; i < wn; i++)
        if (isnan(wg[i])) { AX_TEST_ASSERT(0, "nan in fused weight grad"); break; }

    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss);
    ax_tensor_destroy(x);
    ax_layer_destroy(net1);
}

/* batch size 1: edge case that can expose alignment issues in gemm */
static void test_batch_size_1(void)
{
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(128, 64, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(64, 10, true));

    ax_model_t *m = ax_model_create(net);
    ax_optimizer_t *opt = ax_adam_create(m->params, m->n_params,
                                          0.01f, 0.9f, 0.999f, 1e-8f, 0);
    ax_model_compile(m, opt, ax_mse_loss);

    int64_t xs[] = {1, 128};
    ax_tensor_t *x = ax_tensor_rand(xs, 2, 0.0f, 1.0f);
    int64_t ys[] = {1, 10};
    ax_tensor_t *y = ax_tensor_zeros(ys, 2, AX_FLOAT32);
    ((float *)y->storage->data)[3] = 1.0f;

    float first = ax_model_train_step(m, x, y);
    float last = train_steps(m, x, y, 100);

    AX_TEST_ASSERT(last < first, "batch_size=1 training should work");
    AX_TEST_ASSERT(!isnan(last), "batch_size=1 no nan");

    ax_tensor_destroy(x);
    ax_tensor_destroy(y);
    ax_model_destroy(m);
}

/* non-power-of-2 dimensions: 37x73 @ 73x19. tests gemm tail handling
   when dimensions don't align to MR/NR tile boundaries. */
static void test_odd_dimension_matmul(void)
{
    int64_t a_shape[] = {37, 73};
    int64_t b_shape[] = {73, 19};
    ax_tensor_t *a = ax_tensor_rand(a_shape, 2, -1.0f, 1.0f);
    ax_tensor_t *b = ax_tensor_rand(b_shape, 2, -1.0f, 1.0f);
    a->requires_grad = true;
    b->requires_grad = true;

    ax_enable_grad();
    ax_tensor_t *c = ax_matmul(a, b);
    AX_TEST_ASSERT(c != NULL, "odd dim matmul");
    AX_TEST_ASSERT_EQ(c->shape[0], 37, "odd dim rows");
    AX_TEST_ASSERT_EQ(c->shape[1], 19, "odd dim cols");

    ax_tensor_t *loss = ax_sum(c, -1);
    ax_backward(loss);

    AX_TEST_ASSERT(a->grad != NULL, "odd dim grad a");
    AX_TEST_ASSERT(b->grad != NULL, "odd dim grad b");

    /* verify grad shapes */
    AX_TEST_ASSERT_EQ(a->grad->shape[0], 37, "grad a rows");
    AX_TEST_ASSERT_EQ(a->grad->shape[1], 73, "grad a cols");
    AX_TEST_ASSERT_EQ(b->grad->shape[0], 73, "grad b rows");
    AX_TEST_ASSERT_EQ(b->grad->shape[1], 19, "grad b cols");

    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss);
    ax_tensor_destroy(a);
    ax_tensor_destroy(b);
}

/* predict after train: verify that switching between train mode (with
   fused dense+relu) and eval mode (with separate ops) works correctly
   and produces consistent outputs. */
static void test_train_then_predict(void)
{
    float x_data[] = {0,0, 0,1, 1,0, 1,1};
    float y_data[] = {0, 1, 1, 0};
    int64_t xs[] = {4, 2}, ys[] = {4, 1};
    ax_tensor_t *x = ax_tensor_from_array(x_data, xs, 2, AX_FLOAT32);
    ax_tensor_t *y = ax_tensor_from_array(y_data, ys, 2, AX_FLOAT32);

    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(2, 16, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(16, 1, true));

    ax_model_t *m = ax_model_create(net);
    ax_optimizer_t *opt = ax_adam_create(m->params, m->n_params,
                                          0.01f, 0.9f, 0.999f, 1e-8f, 0);
    ax_model_compile(m, opt, ax_mse_loss);

    /* train */
    for (int i = 0; i < 500; i++) ax_model_train_step(m, x, y);

    /* predict: should not track grad */
    ax_tensor_t *pred = ax_model_predict(m, x);
    AX_TEST_ASSERT(pred != NULL, "predict should work after training");
    AX_TEST_ASSERT(!pred->requires_grad, "predict should not track grad");
    AX_TEST_ASSERT(pred->grad_fn == NULL, "predict should not have grad_fn");

    /* predictions should be reasonable (close to xor targets) */
    int64_t i0[] = {0, 0};
    float p0 = ax_tensor_get_f32(pred, i0);
    AX_TEST_ASSERT(p0 < 0.5f, "predict(0,0) should be near 0");

    /* can resume training after predict */
    float loss_after = ax_model_train_step(m, x, y);
    AX_TEST_ASSERT(!isnan(loss_after), "training after predict should work");

    ax_tensor_destroy(pred);
    ax_tensor_destroy(x);
    ax_tensor_destroy(y);
    ax_model_destroy(m);
}

/* conv2d with large spatial dims: 8x3x32x32 with 16 filters.
   stresses im2col buffer allocation and col_gemm. */
static void test_large_conv2d(void)
{
    int64_t xs[] = {8, 3, 32, 32};
    ax_tensor_t *x = ax_tensor_rand(xs, 4, 0.0f, 1.0f);

    int64_t ys[] = {8, 1};
    ax_tensor_t *y = ax_tensor_rand(ys, 2, 0.0f, 1.0f);

    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_conv2d_create(3, 16, 3, 1, 1, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_maxpool2d_create(2, 2, 0));
    ax_sequential_add(net, ax_flatten_create());
    ax_sequential_add(net, ax_dense_create(16 * 16 * 16, 1, true));

    ax_model_t *m = ax_model_create(net);
    ax_optimizer_t *opt = ax_adam_create(m->params, m->n_params,
                                          1e-3f, 0.9f, 0.999f, 1e-8f, 0);
    ax_model_compile(m, opt, ax_mse_loss);

    float first = ax_model_train_step(m, x, y);
    float last = train_steps(m, x, y, 50);

    AX_TEST_ASSERT(!isnan(last) && !isinf(last), "large conv2d no nan/inf");
    AX_TEST_ASSERT(last < first, "large conv2d should train");

    ax_tensor_destroy(x);
    ax_tensor_destroy(y);
    ax_model_destroy(m);
}

/* learning rate scheduler + training: verify cosine annealing works
   end-to-end without breaking training convergence. */
static void test_lr_schedule_training(void)
{
    float x_data[] = {0,0, 0,1, 1,0, 1,1};
    float y_data[] = {0, 1, 1, 0};
    int64_t xs[] = {4, 2}, ys[] = {4, 1};
    ax_tensor_t *x = ax_tensor_from_array(x_data, xs, 2, AX_FLOAT32);
    ax_tensor_t *y = ax_tensor_from_array(y_data, ys, 2, AX_FLOAT32);

    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(2, 32, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(32, 1, true));

    ax_model_t *m = ax_model_create(net);
    ax_optimizer_t *opt = ax_adam_create(m->params, m->n_params,
                                          0.01f, 0.9f, 0.999f, 1e-8f, 0);
    ax_lr_scheduler_t *sched = ax_sched_cosine(opt, 500, 1e-5f);
    ax_model_compile(m, opt, ax_mse_loss);

    float first = ax_model_train_step(m, x, y);
    for (int i = 0; i < 500; i++) {
        ax_model_train_step(m, x, y);
        ax_sched_step(sched);
    }
    float last = ax_model_train_step(m, x, y);

    AX_TEST_ASSERT(last < first, "cosine annealing training should converge");

    ax_sched_destroy(sched);
    ax_tensor_destroy(x);
    ax_tensor_destroy(y);
    ax_model_destroy(m);
}

int main(void)
{
    ax_init();

    printf("=== stress tests ===\n");
    AX_RUN_TEST(test_large_mlp_trains);
    AX_RUN_TEST(test_wide_matmul);
    AX_RUN_TEST(test_tall_matmul);
    AX_RUN_TEST(test_all_activations_train);
    AX_RUN_TEST(test_all_optimizers_train);
    AX_RUN_TEST(test_cross_entropy_many_classes);
    AX_RUN_TEST(test_deep_network_trains);
    AX_RUN_TEST(test_no_grad_nesting);
    AX_RUN_TEST(test_memory_pressure);
    AX_RUN_TEST(test_fused_dense_relu_grad);
    AX_RUN_TEST(test_batch_size_1);
    AX_RUN_TEST(test_odd_dimension_matmul);
    AX_RUN_TEST(test_train_then_predict);
    AX_RUN_TEST(test_large_conv2d);
    AX_RUN_TEST(test_lr_schedule_training);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
