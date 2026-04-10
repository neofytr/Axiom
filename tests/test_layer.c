/* test_layer.c — tests for layers, sequential model, and the model api.
   this is the first "integration" test: it actually trains a tiny network. */

#include "test.h"
#include "axiom/axiom.h"

static ax_tensor_t *make_2d(float *data, int64_t rows, int64_t cols)
{
    int64_t shape[] = {rows, cols};
    return ax_tensor_from_array(data, shape, 2, AX_FLOAT32);
}

static void test_dense_create(void)
{
    ax_layer_t *d = ax_dense_create(4, 3, true);
    AX_TEST_ASSERT(d != NULL, "dense should be created");
    AX_TEST_ASSERT_EQ(d->type, AX_LAYER_DENSE, "type should be dense");
    AX_TEST_ASSERT_EQ(d->n_params, 2, "should have weight + bias");

    /* weight should be [4, 3] */
    ax_dense_t *dd = (ax_dense_t *)d;
    AX_TEST_ASSERT_EQ(dd->weight->shape[0], 4, "weight rows");
    AX_TEST_ASSERT_EQ(dd->weight->shape[1], 3, "weight cols");
    AX_TEST_ASSERT(dd->weight->requires_grad, "weight should track grad");

    /* bias should be [3] */
    AX_TEST_ASSERT_EQ(dd->bias->shape[0], 3, "bias size");

    /* param count: 4*3 + 3 = 15 */
    AX_TEST_ASSERT_EQ(ax_layer_param_count(d), 15, "param count");

    ax_layer_destroy(d);
}

static void test_dense_forward(void)
{
    ax_layer_t *d = ax_dense_create(2, 3, false);

    /* manual weight for predictable output */
    ax_dense_t *dd = (ax_dense_t *)d;
    float w[] = {1, 0, 0, 0, 1, 0};
    int64_t ws[] = {2, 3};
    ax_tensor_destroy(dd->weight);
    dd->weight = ax_tensor_from_array(w, ws, 2, AX_FLOAT32);
    dd->weight->requires_grad = true;
    dd->base.params[0] = dd->weight;

    /* input: [1, 2] -> output should be [1, 2, 0] */
    ax_tensor_t *input = make_2d((float[]){1, 2}, 1, 2);
    ax_tensor_t *out = ax_layer_forward(d, input);

    AX_TEST_ASSERT(out != NULL, "forward should work");
    AX_TEST_ASSERT_EQ(out->shape[0], 1, "batch dim");
    AX_TEST_ASSERT_EQ(out->shape[1], 3, "output features");

    int64_t i00[] = {0, 0}, i01[] = {0, 1}, i02[] = {0, 2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i00), 1.0f, 1e-5, "out[0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i01), 2.0f, 1e-5, "out[1]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i02), 0.0f, 1e-5, "out[2]");

    ax_tensor_destroy(input);
    ax_layer_destroy(d);
}

static void test_sequential(void)
{
    ax_layer_t *model = ax_sequential_create();
    ax_sequential_add(model, ax_dense_create(2, 4, true));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_dense_create(4, 1, true));

    ax_sequential_t *seq = (ax_sequential_t *)model;
    AX_TEST_ASSERT_EQ(seq->n_layers, 3, "should have 3 layers");

    /* param count: (2*4 + 4) + 0 + (4*1 + 1) = 12 + 5 = 17 */
    AX_TEST_ASSERT_EQ(ax_layer_param_count(model), 17, "total params");

    /* forward should produce [batch, 1] output */
    ax_tensor_t *input = make_2d((float[]){1, 2}, 1, 2);
    ax_tensor_t *out = ax_layer_forward(model, input);
    AX_TEST_ASSERT(out != NULL, "sequential forward should work");
    AX_TEST_ASSERT_EQ(out->shape[1], 1, "output dim should be 1");

    ax_tensor_destroy(input);
    ax_layer_destroy(model);
}

static void test_get_params(void)
{
    ax_layer_t *model = ax_sequential_create();
    ax_sequential_add(model, ax_dense_create(2, 4, true));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_dense_create(4, 1, true));

    ax_tensor_t *params[32];
    int n = ax_layer_get_params(model, params, 32);

    /* 2 dense layers with bias = 4 param tensors */
    AX_TEST_ASSERT_EQ(n, 4, "should collect 4 param tensors");

    /* all should have requires_grad */
    for (int i = 0; i < n; i++)
        AX_TEST_ASSERT(params[i]->requires_grad, "params should require grad");

    ax_layer_destroy(model);
}

static void test_train_eval_mode(void)
{
    ax_layer_t *model = ax_sequential_create();
    ax_sequential_add(model, ax_dense_create(2, 2, true));
    ax_sequential_add(model, ax_relu_layer_create());

    ax_layer_eval(model);
    ax_sequential_t *seq = (ax_sequential_t *)model;
    AX_TEST_ASSERT(!seq->layers[0]->training, "should be eval mode");
    AX_TEST_ASSERT(!seq->layers[1]->training, "should be eval mode");

    ax_layer_train(model);
    AX_TEST_ASSERT(seq->layers[0]->training, "should be train mode");

    ax_layer_destroy(model);
}

/* the real test: can we actually train a tiny xor network? */
static void test_xor_training(void)
{
    /* xor dataset */
    ax_tensor_t *x = make_2d((float[]){0,0, 0,1, 1,0, 1,1}, 4, 2);
    ax_tensor_t *y = make_2d((float[]){0, 1, 1, 0}, 4, 1);

    /* model: 2 -> 16 -> relu -> 16 -> relu -> 1
       wider hidden layers to avoid dead relu sensitivity */
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(2, 16, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(16, 16, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(16, 1, true));

    ax_model_t *model = ax_model_create(net);
    ax_optimizer_t *opt = ax_adam_create(model->params, model->n_params,
                                         0.01f, 0.9f, 0.999f, 1e-8f, 0.0f);
    ax_model_compile(model, opt, ax_mse_loss);

    /* train for enough steps to reliably converge */
    float first_loss = ax_model_train_step(model, x, y);
    for (int i = 0; i < 1000; i++)
        ax_model_train_step(model, x, y);
    float last_loss = ax_model_train_step(model, x, y);

    /* loss should have decreased */
    AX_TEST_ASSERT(last_loss < first_loss, "training should reduce loss");
    AX_TEST_ASSERT(last_loss < 0.1f, "xor should converge to low loss");

    ax_tensor_destroy(x); ax_tensor_destroy(y);
    ax_model_destroy(model);
}

static void test_model_predict(void)
{
    ax_tensor_t *x = make_2d((float[]){1, 2, 3, 4}, 2, 2);

    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(2, 3, true));
    ax_sequential_add(net, ax_relu_layer_create());

    ax_model_t *model = ax_model_create(net);

    /* predict should work without compiling (no optimizer needed) */
    ax_tensor_t *out = ax_model_predict(model, x);
    AX_TEST_ASSERT(out != NULL, "predict should work");
    AX_TEST_ASSERT_EQ(out->shape[0], 2, "batch preserved");
    AX_TEST_ASSERT_EQ(out->shape[1], 3, "output features");

    /* predict should not create gradients */
    AX_TEST_ASSERT(!out->requires_grad, "predict should not track grad");

    ax_tensor_destroy(x);
    ax_model_destroy(model);
}

int main(void)
{
    ax_init();

    printf("=== layer tests ===\n");
    AX_RUN_TEST(test_dense_create);
    AX_RUN_TEST(test_dense_forward);
    AX_RUN_TEST(test_sequential);
    AX_RUN_TEST(test_get_params);
    AX_RUN_TEST(test_train_eval_mode);
    AX_RUN_TEST(test_xor_training);
    AX_RUN_TEST(test_model_predict);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
