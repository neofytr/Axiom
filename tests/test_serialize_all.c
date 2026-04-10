/* test_serialize_all.c — roundtrip save/load for every layer type.
   for each layer: build a model containing it, predict, save, load,
   predict again, verify outputs match. */

#include "test.h"
#include "axiom/axiom.h"
#include <unistd.h>
#include <math.h>

#define PATH "/tmp/ax_ser_test.axm"

/* helper: predict with both models, compare output at [0,0] */
static bool predictions_match(ax_model_t *m1, ax_model_t *m2,
                               ax_tensor_t *input, float tol)
{
    ax_tensor_t *p1 = ax_model_predict(m1, input);
    ax_tensor_t *p2 = ax_model_predict(m2, input);
    if (!p1 || !p2)
    {
        if (p1) ax_tensor_destroy(p1);
        if (p2) ax_tensor_destroy(p2);
        return false;
    }

    int64_t n = ax_tensor_numel(p1);
    float *d1 = (float *)p1->storage->data;
    float *d2 = (float *)p2->storage->data;

    bool match = true;
    for (int64_t i = 0; i < n && i < 10; i++)
    {
        if (fabsf(d1[p1->offset + i] - d2[p2->offset + i]) > tol)
        {
            match = false;
            break;
        }
    }

    ax_tensor_destroy(p1);
    ax_tensor_destroy(p2);
    return match;
}

/* macro to reduce boilerplate for each test */
#define ROUNDTRIP_TEST(name, build_net, make_input) \
static void name(void) \
{ \
    ax_layer_t *net = build_net; \
    ax_model_t *m = ax_model_create(net); \
    ax_tensor_t *input = make_input; \
    \
    ax_status_t s = ax_model_save(m, PATH); \
    AX_TEST_ASSERT_EQ(s, AX_OK, "save should succeed"); \
    \
    ax_model_t *loaded = ax_model_load(PATH); \
    AX_TEST_ASSERT(loaded != NULL, "load should succeed"); \
    \
    if (loaded) { \
        AX_TEST_ASSERT(predictions_match(m, loaded, input, 1e-4f), \
                       "predictions should match after roundtrip"); \
        ax_model_destroy(loaded); \
    } \
    \
    ax_tensor_destroy(input); \
    ax_model_destroy(m); \
    unlink(PATH); \
}

/* test builders */

static ax_layer_t *build_dense_model(void)
{
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(4, 8, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(8, 2, true));
    return net;
}

static ax_layer_t *build_sigmoid_model(void)
{
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(4, 8, true));
    ax_sequential_add(net, ax_sigmoid_layer_create());
    ax_sequential_add(net, ax_dense_create(8, 2, true));
    return net;
}

static ax_layer_t *build_tanh_model(void)
{
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(4, 8, true));
    ax_sequential_add(net, ax_tanh_layer_create());
    ax_sequential_add(net, ax_dense_create(8, 2, true));
    return net;
}

static ax_layer_t *build_leaky_relu_model(void)
{
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(4, 8, true));
    ax_sequential_add(net, ax_leaky_relu_layer_create(0.01f));
    ax_sequential_add(net, ax_dense_create(8, 2, true));
    return net;
}

static ax_layer_t *build_elu_model(void)
{
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(4, 8, true));
    ax_sequential_add(net, ax_elu_layer_create(1.0f));
    ax_sequential_add(net, ax_dense_create(8, 2, true));
    return net;
}

static ax_layer_t *build_gelu_model(void)
{
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(4, 8, true));
    ax_sequential_add(net, ax_gelu_layer_create());
    ax_sequential_add(net, ax_dense_create(8, 2, true));
    return net;
}

static ax_layer_t *build_swish_model(void)
{
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(4, 8, true));
    ax_sequential_add(net, ax_swish_layer_create());
    ax_sequential_add(net, ax_dense_create(8, 2, true));
    return net;
}

static ax_layer_t *build_batchnorm_model(void)
{
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(4, 8, true));
    ax_sequential_add(net, ax_batchnorm_create(8, 1e-5f, 0.1f));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(8, 2, true));
    return net;
}

static ax_layer_t *build_layernorm_model(void)
{
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(4, 8, true));
    ax_sequential_add(net, ax_layernorm_create(8, 1e-5f));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(8, 2, true));
    return net;
}

static ax_layer_t *build_dropout_model(void)
{
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(4, 8, true));
    ax_sequential_add(net, ax_dropout_create(0.5f));
    ax_sequential_add(net, ax_dense_create(8, 2, true));
    return net;
}

static ax_layer_t *build_conv_model(void)
{
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_conv2d_create(1, 4, 3, 1, 0, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_flatten_create());
    ax_sequential_add(net, ax_dense_create(4 * 2 * 2, 2, true));
    return net;
}

static ax_layer_t *build_pool_model(void)
{
    /* input [2,1,6,6] -> conv(1->2,k=3,pad=0) -> [2,2,4,4]
       -> relu -> maxpool(2,2) -> [2,2,2,2] -> flatten -> [2,8] -> dense(8,2) */
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_conv2d_create(1, 2, 3, 1, 0, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_maxpool2d_create(2, 2, 0));
    ax_sequential_add(net, ax_flatten_create());
    ax_sequential_add(net, ax_dense_create(2 * 2 * 2, 2, true));
    return net;
}

static ax_layer_t *build_avgpool_model(void)
{
    /* same shapes as maxpool */
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_conv2d_create(1, 2, 3, 1, 0, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_avgpool2d_create(2, 2, 0));
    ax_sequential_add(net, ax_flatten_create());
    ax_sequential_add(net, ax_dense_create(2 * 2 * 2, 2, true));
    return net;
}

static ax_layer_t *build_global_avgpool_model(void)
{
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_conv2d_create(1, 4, 3, 1, 0, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_global_avgpool2d_create());
    ax_sequential_add(net, ax_dense_create(4, 2, true));
    return net;
}

/* 2d input for dense models */
static ax_tensor_t *make_dense_input(void)
{
    return ax_tensor_rand((int64_t[]){3, 4}, 2, 0.0f, 1.0f);
}

/* 4d input for conv models */
static ax_tensor_t *make_conv_input(void)
{
    return ax_tensor_rand((int64_t[]){2, 1, 4, 4}, 4, 0.0f, 1.0f);
}

static ax_tensor_t *make_conv_input_6(void)
{
    return ax_tensor_rand((int64_t[]){2, 1, 6, 6}, 4, 0.0f, 1.0f);
}

/* define all roundtrip tests */
ROUNDTRIP_TEST(test_ser_dense, build_dense_model(), make_dense_input())
ROUNDTRIP_TEST(test_ser_sigmoid, build_sigmoid_model(), make_dense_input())
ROUNDTRIP_TEST(test_ser_tanh, build_tanh_model(), make_dense_input())
ROUNDTRIP_TEST(test_ser_leaky_relu, build_leaky_relu_model(), make_dense_input())
ROUNDTRIP_TEST(test_ser_elu, build_elu_model(), make_dense_input())
ROUNDTRIP_TEST(test_ser_gelu, build_gelu_model(), make_dense_input())
ROUNDTRIP_TEST(test_ser_swish, build_swish_model(), make_dense_input())
ROUNDTRIP_TEST(test_ser_batchnorm, build_batchnorm_model(), make_dense_input())
ROUNDTRIP_TEST(test_ser_layernorm, build_layernorm_model(), make_dense_input())
ROUNDTRIP_TEST(test_ser_dropout, build_dropout_model(), make_dense_input())
ROUNDTRIP_TEST(test_ser_conv, build_conv_model(), make_conv_input())
ROUNDTRIP_TEST(test_ser_maxpool, build_pool_model(), make_conv_input_6())
ROUNDTRIP_TEST(test_ser_avgpool, build_avgpool_model(), make_conv_input_6())
ROUNDTRIP_TEST(test_ser_global_avgpool, build_global_avgpool_model(), make_conv_input())

int main(void)
{
    ax_init();

    printf("=== serialization roundtrip for all layer types ===\n");
    AX_RUN_TEST(test_ser_dense);
    AX_RUN_TEST(test_ser_sigmoid);
    AX_RUN_TEST(test_ser_tanh);
    AX_RUN_TEST(test_ser_leaky_relu);
    AX_RUN_TEST(test_ser_elu);
    AX_RUN_TEST(test_ser_gelu);
    AX_RUN_TEST(test_ser_swish);
    AX_RUN_TEST(test_ser_batchnorm);
    AX_RUN_TEST(test_ser_layernorm);
    AX_RUN_TEST(test_ser_dropout);
    AX_RUN_TEST(test_ser_conv);
    AX_RUN_TEST(test_ser_maxpool);
    AX_RUN_TEST(test_ser_avgpool);
    AX_RUN_TEST(test_ser_global_avgpool);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
