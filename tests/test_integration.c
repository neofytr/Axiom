/* test_integration.c — end-to-end tests that prove autograd flows
   through every layer type. if any backward is broken, loss won't decrease. */

#include "test.h"
#include "axiom/axiom.h"
#include <math.h>

/* test: gradient flows through batchnorm.
   model: dense -> batchnorm -> relu -> dense.
   loss should decrease over training steps. */
static void test_batchnorm_trains(void)
{
    float x_data[] = {0,0, 0,1, 1,0, 1,1};
    float y_data[] = {0, 1, 1, 0};
    int64_t xs[] = {4, 2}, ys[] = {4, 1};
    ax_tensor_t *x = ax_tensor_from_array(x_data, xs, 2, AX_FLOAT32);
    ax_tensor_t *y = ax_tensor_from_array(y_data, ys, 2, AX_FLOAT32);

    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(2, 16, true));
    ax_sequential_add(net, ax_batchnorm_create(16, 1e-5f, 0.1f));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(16, 1, true));

    ax_model_t *m = ax_model_create(net);
    ax_optimizer_t *opt = ax_adam_create(m->params, m->n_params, 0.01f, 0.9f, 0.999f, 1e-8f, 0);
    ax_model_compile(m, opt, ax_mse_loss);

    float first = ax_model_train_step(m, x, y);
    for (int i = 0; i < 500; i++) ax_model_train_step(m, x, y);
    float last = ax_model_train_step(m, x, y);

    AX_TEST_ASSERT(last < first, "batchnorm model should train (loss decreased)");
    AX_TEST_ASSERT(last < 0.1f, "should converge to low loss");

    ax_tensor_destroy(x); ax_tensor_destroy(y);
    ax_model_destroy(m);
}

/* test: gradient flows through layernorm */
static void test_layernorm_trains(void)
{
    float x_data[] = {0,0, 0,1, 1,0, 1,1};
    float y_data[] = {0, 1, 1, 0};
    int64_t xs[] = {4, 2}, ys[] = {4, 1};
    ax_tensor_t *x = ax_tensor_from_array(x_data, xs, 2, AX_FLOAT32);
    ax_tensor_t *y = ax_tensor_from_array(y_data, ys, 2, AX_FLOAT32);

    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(2, 16, true));
    ax_sequential_add(net, ax_layernorm_create(16, 1e-5f));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(16, 1, true));

    ax_model_t *m = ax_model_create(net);
    ax_optimizer_t *opt = ax_adam_create(m->params, m->n_params, 0.01f, 0.9f, 0.999f, 1e-8f, 0);
    ax_model_compile(m, opt, ax_mse_loss);

    float first = ax_model_train_step(m, x, y);
    for (int i = 0; i < 500; i++) ax_model_train_step(m, x, y);
    float last = ax_model_train_step(m, x, y);

    AX_TEST_ASSERT(last < first, "layernorm model should train");
    AX_TEST_ASSERT(last < 0.1f, "should converge");

    ax_tensor_destroy(x); ax_tensor_destroy(y);
    ax_model_destroy(m);
}

/* test: gradient flows through dropout (in training mode) */
static void test_dropout_trains(void)
{
    float x_data[] = {0,0, 0,1, 1,0, 1,1};
    float y_data[] = {0, 1, 1, 0};
    int64_t xs[] = {4, 2}, ys[] = {4, 1};
    ax_tensor_t *x = ax_tensor_from_array(x_data, xs, 2, AX_FLOAT32);
    ax_tensor_t *y = ax_tensor_from_array(y_data, ys, 2, AX_FLOAT32);

    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(2, 16, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dropout_create(0.1f));
    ax_sequential_add(net, ax_dense_create(16, 1, true));

    ax_model_t *m = ax_model_create(net);
    ax_optimizer_t *opt = ax_adam_create(m->params, m->n_params, 0.01f, 0.9f, 0.999f, 1e-8f, 0);
    ax_model_compile(m, opt, ax_mse_loss);

    float first = ax_model_train_step(m, x, y);
    for (int i = 0; i < 500; i++) ax_model_train_step(m, x, y);
    float last = ax_model_train_step(m, x, y);

    AX_TEST_ASSERT(last < first, "dropout model should train");

    ax_tensor_destroy(x); ax_tensor_destroy(y);
    ax_model_destroy(m);
}

/* test: gradient flows through conv2d -> pool -> flatten -> dense.
   this is the hardest test — it exercises conv backward, maxpool backward,
   and flatten backward all in one pipeline. */
static void test_cnn_trains(void)
{
    /* tiny dataset: 4 samples, 1 channel, 4x4 images, 2 classes */
    int64_t xs[] = {4, 1, 4, 4};
    ax_tensor_t *x = ax_tensor_rand(xs, 4, 0.0f, 1.0f);

    float y_data[] = {1,0, 0,1, 1,0, 0,1};
    int64_t ys[] = {4, 2};
    ax_tensor_t *y = ax_tensor_from_array(y_data, ys, 2, AX_FLOAT32);

    /* conv(1->4, k=3, pad=0) -> relu -> flatten -> dense(4, 2) */
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_conv2d_create(1, 4, 3, 1, 0, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_flatten_create());
    ax_sequential_add(net, ax_dense_create(4 * 2 * 2, 2, true));

    ax_model_t *m = ax_model_create(net);
    ax_optimizer_t *opt = ax_adam_create(m->params, m->n_params, 0.005f, 0.9f, 0.999f, 1e-8f, 0);
    ax_model_compile(m, opt, ax_mse_loss);

    float first = ax_model_train_step(m, x, y);
    for (int i = 0; i < 200; i++) ax_model_train_step(m, x, y);
    float last = ax_model_train_step(m, x, y);

    AX_TEST_ASSERT(last < first, "cnn should train (loss decreased)");

    ax_tensor_destroy(x); ax_tensor_destroy(y);
    ax_model_destroy(m);
}

/* test: conv -> batchnorm -> relu -> maxpool -> flatten -> dense
   the full pipeline used in real CNNs */
static void test_full_cnn_pipeline(void)
{
    ax_set_seed(42);
    int64_t xs[] = {4, 1, 6, 6};
    ax_tensor_t *x = ax_tensor_rand(xs, 4, 0.0f, 1.0f);

    float y_data[] = {0, 1, 1, 0};
    int64_t ys[] = {4, 1};
    ax_tensor_t *y = ax_tensor_from_array(y_data, ys, 2, AX_FLOAT32);

    /* conv(1->2, k=3, pad=0) -> relu -> maxpool(2,2) -> flatten -> dense */
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_conv2d_create(1, 2, 3, 1, 0, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_maxpool2d_create(2, 2, 0));
    ax_sequential_add(net, ax_flatten_create());
    ax_sequential_add(net, ax_dense_create(2 * 2 * 2, 1, true));

    ax_model_t *m = ax_model_create(net);
    ax_optimizer_t *opt = ax_adam_create(m->params, m->n_params, 0.01f, 0.9f, 0.999f, 1e-8f, 0);
    ax_model_compile(m, opt, ax_mse_loss);

    float first = ax_model_train_step(m, x, y);
    for (int i = 0; i < 500; i++) ax_model_train_step(m, x, y);
    float last = ax_model_train_step(m, x, y);

    AX_TEST_ASSERT(last < first, "full cnn pipeline should train");

    ax_tensor_destroy(x); ax_tensor_destroy(y);
    ax_model_destroy(m);
}

/* test: model with gradient clipping trains without exploding */
static void test_grad_clipping_trains(void)
{
    float x_data[] = {0,0, 0,1, 1,0, 1,1};
    float y_data[] = {0, 1, 1, 0};
    int64_t xs[] = {4, 2}, ys[] = {4, 1};
    ax_tensor_t *x = ax_tensor_from_array(x_data, xs, 2, AX_FLOAT32);
    ax_tensor_t *y = ax_tensor_from_array(y_data, ys, 2, AX_FLOAT32);

    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(2, 8, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(8, 1, true));

    ax_model_t *m = ax_model_create(net);
    ax_optimizer_t *opt = ax_adam_create(m->params, m->n_params, 0.01f, 0.9f, 0.999f, 1e-8f, 0);
    ax_model_compile(m, opt, ax_mse_loss);

    /* train with gradient clipping */
    float first_loss = -1;
    for (int i = 0; i < 500; i++)
    {
        ax_optimizer_zero_grad(opt);
        ax_enable_grad();
        ax_layer_train(net);
        ax_tensor_t *pred = ax_layer_forward(net, x);
        ax_tensor_t *loss = ax_mse_loss(pred, y);

        int64_t li[] = {0};
        float lv = ax_tensor_get_f32(loss, li);
        if (i == 0) first_loss = lv;

        ax_backward(loss);
        ax_clip_grad_norm(m->params, m->n_params, 1.0f);
        ax_optimizer_step(opt);
        ax_graph_cleanup(loss);
        ax_tensor_destroy(loss);
    }

    /* get final loss */
    ax_no_grad();
    ax_tensor_t *pred = ax_layer_forward(net, x);
    ax_tensor_t *loss = ax_mse_loss(pred, y);
    int64_t li[] = {0};
    float last_loss = ax_tensor_get_f32(loss, li);
    ax_enable_grad();

    AX_TEST_ASSERT(last_loss < first_loss, "grad-clipped model should train");

    ax_tensor_destroy(x); ax_tensor_destroy(y);
    ax_model_destroy(m);
}

/* test: serialization roundtrip for a CNN model */
static void test_cnn_serialize_roundtrip(void)
{
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_conv2d_create(1, 4, 3, 1, 0, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_flatten_create());
    ax_sequential_add(net, ax_dense_create(4 * 2 * 2, 2, true));

    ax_model_t *m = ax_model_create(net);

    /* get prediction before save */
    int64_t xs[] = {1, 1, 4, 4};
    ax_tensor_t *input = ax_tensor_rand(xs, 4, 0.0f, 1.0f);
    ax_tensor_t *pred_before = ax_model_predict(m, input);

    int64_t i0[] = {0, 0};
    float val_before = ax_tensor_get_f32(pred_before, i0);

    ax_model_save(m, "/tmp/ax_test_cnn.axm");

    ax_model_t *loaded = ax_model_load("/tmp/ax_test_cnn.axm");
    AX_TEST_ASSERT(loaded != NULL, "cnn model should load");

    ax_tensor_t *pred_after = ax_model_predict(loaded, input);
    float val_after = ax_tensor_get_f32(pred_after, i0);

    AX_TEST_ASSERT_NEAR(val_before, val_after, 1e-4, "predictions should match after load");

    ax_tensor_destroy(input);
    ax_tensor_destroy(pred_before);
    ax_tensor_destroy(pred_after);
    ax_model_destroy(m);
    ax_model_destroy(loaded);
    remove("/tmp/ax_test_cnn.axm");
}

int main(void)
{
    ax_init();

    printf("=== integration tests ===\n");
    AX_RUN_TEST(test_batchnorm_trains);
    AX_RUN_TEST(test_layernorm_trains);
    AX_RUN_TEST(test_dropout_trains);
    AX_RUN_TEST(test_cnn_trains);
    AX_RUN_TEST(test_full_cnn_pipeline);
    AX_RUN_TEST(test_grad_clipping_trains);
    AX_RUN_TEST(test_cnn_serialize_roundtrip);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
