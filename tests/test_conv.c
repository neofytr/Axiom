/* test_conv.c — tests for conv2d, pooling, flatten, im2col */

#include "test.h"
#include "axiom/axiom.h"

/* helper: make a 4d NCHW tensor */
static ax_tensor_t *make_4d(float *data, int64_t n, int64_t c, int64_t h, int64_t w)
{
    int64_t shape[] = {n, c, h, w};
    return ax_tensor_from_array(data, shape, 4, AX_FLOAT32);
}

static void test_im2col_basic(void)
{
    /* single channel 3x3 image, 2x2 kernel, stride 1, no padding
       should produce a [4, 4] column matrix (4 = 1*2*2, 4 = 2*2 output positions) */
    float img_data[] = {1, 2, 3,
                        4, 5, 6,
                        7, 8, 9};
    int64_t img_shape[] = {1, 3, 3};
    ax_tensor_t *img = ax_tensor_from_array(img_data, img_shape, 3, AX_FLOAT32);

    ax_tensor_t *cols = ax_im2col(img, 2, 2, 1, 1, 0, 0);
    AX_TEST_ASSERT(cols != NULL, "im2col should work");
    AX_TEST_ASSERT_EQ(cols->shape[0], 4, "rows = C*kh*kw = 1*2*2");
    AX_TEST_ASSERT_EQ(cols->shape[1], 4, "cols = oh*ow = 2*2");

    /* first column should be the top-left 2x2 patch: [1, 2, 4, 5] */
    int64_t i00[] = {0, 0}, i10[] = {1, 0}, i20[] = {2, 0}, i30[] = {3, 0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(cols, i00), 1.0f, 1e-5, "patch[0,0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(cols, i10), 2.0f, 1e-5, "patch[1,0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(cols, i20), 4.0f, 1e-5, "patch[2,0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(cols, i30), 5.0f, 1e-5, "patch[3,0]");

    ax_tensor_destroy(img); ax_tensor_destroy(cols);
}

static void test_conv2d_create(void)
{
    ax_layer_t *c = ax_conv2d_create(3, 16, 3, 1, 1, true);
    AX_TEST_ASSERT(c != NULL, "conv2d should be created");

    /* weight: [16, 3, 3, 3], bias: [16] */
    ax_conv2d_t *cc = (ax_conv2d_t *)c;
    AX_TEST_ASSERT_EQ(cc->weight->shape[0], 16, "out channels");
    AX_TEST_ASSERT_EQ(cc->weight->shape[1], 3, "in channels");
    AX_TEST_ASSERT_EQ(cc->weight->shape[2], 3, "kernel h");
    AX_TEST_ASSERT_EQ(cc->weight->shape[3], 3, "kernel w");
    AX_TEST_ASSERT(cc->weight->requires_grad, "weight needs grad");

    AX_TEST_ASSERT_EQ(cc->bias->shape[0], 16, "bias size");

    /* params: weight + bias = 16*3*3*3 + 16 = 432 + 16 = 448 */
    AX_TEST_ASSERT_EQ(ax_layer_param_count(c), 448, "param count");

    ax_layer_destroy(c);
}

static void test_conv2d_forward(void)
{
    /* 1 image, 1 channel, 4x4, kernel 3x3, stride 1, pad 0
       output should be [1, out_ch, 2, 2] */
    ax_layer_t *c = ax_conv2d_create(1, 1, 3, 1, 0, false);

    /* set kernel to all ones for easy verification */
    ax_conv2d_t *cc = (ax_conv2d_t *)c;
    ax_compute_fill(cc->weight, 1.0);

    float in_data[16];
    for (int i = 0; i < 16; i++) in_data[i] = 1.0f;
    ax_tensor_t *input = make_4d(in_data, 1, 1, 4, 4);

    ax_tensor_t *out = ax_layer_forward(c, input);
    AX_TEST_ASSERT(out != NULL, "conv2d forward should work");
    AX_TEST_ASSERT_EQ(out->ndim, 4, "output is 4d");
    AX_TEST_ASSERT_EQ(out->shape[0], 1, "batch");
    AX_TEST_ASSERT_EQ(out->shape[1], 1, "channels");
    AX_TEST_ASSERT_EQ(out->shape[2], 2, "out height");
    AX_TEST_ASSERT_EQ(out->shape[3], 2, "out width");

    /* all-ones input with all-ones 3x3 kernel: each output = 9 */
    float *od = (float *)out->storage->data;
    AX_TEST_ASSERT_NEAR(od[0], 9.0f, 1e-4, "conv output[0,0]");
    AX_TEST_ASSERT_NEAR(od[1], 9.0f, 1e-4, "conv output[0,1]");
    AX_TEST_ASSERT_NEAR(od[2], 9.0f, 1e-4, "conv output[1,0]");
    AX_TEST_ASSERT_NEAR(od[3], 9.0f, 1e-4, "conv output[1,1]");

    ax_tensor_destroy(input);
    ax_layer_destroy(c);
}

static void test_conv2d_with_padding(void)
{
    /* pad=1 with 3x3 kernel should preserve spatial dims */
    ax_layer_t *c = ax_conv2d_create(1, 1, 3, 1, 1, false);
    ax_conv2d_t *cc = (ax_conv2d_t *)c;
    ax_compute_fill(cc->weight, 1.0);

    float in_data[9] = {1,1,1, 1,1,1, 1,1,1};
    ax_tensor_t *input = make_4d(in_data, 1, 1, 3, 3);

    ax_tensor_t *out = ax_layer_forward(c, input);
    AX_TEST_ASSERT(out != NULL, "padded conv should work");
    AX_TEST_ASSERT_EQ(out->shape[2], 3, "height preserved with pad=1");
    AX_TEST_ASSERT_EQ(out->shape[3], 3, "width preserved with pad=1");

    /* center pixel sees full 3x3 of ones: output = 9
       corner pixel sees 2x2 of ones: output = 4 */
    float *od = (float *)out->storage->data;
    AX_TEST_ASSERT_NEAR(od[4], 9.0f, 1e-4, "center = 9");
    AX_TEST_ASSERT_NEAR(od[0], 4.0f, 1e-4, "corner = 4");

    ax_tensor_destroy(input);
    ax_layer_destroy(c);
}

static void test_maxpool2d(void)
{
    /* [1, 1, 4, 4] with 2x2 maxpool, stride 2 -> [1, 1, 2, 2] */
    float in_data[] = {1, 3, 2, 4,
                       5, 6, 7, 8,
                       3, 2, 1, 0,
                       1, 2, 3, 4};
    ax_tensor_t *input = make_4d(in_data, 1, 1, 4, 4);

    ax_layer_t *pool = ax_maxpool2d_create(2, 2, 0);
    ax_tensor_t *out = ax_layer_forward(pool, input);

    AX_TEST_ASSERT(out != NULL, "maxpool should work");
    AX_TEST_ASSERT_EQ(out->shape[2], 2, "pooled height");
    AX_TEST_ASSERT_EQ(out->shape[3], 2, "pooled width");

    float *od = (float *)out->storage->data;
    AX_TEST_ASSERT_NEAR(od[0], 6.0f, 1e-5, "max of top-left 2x2");
    AX_TEST_ASSERT_NEAR(od[1], 8.0f, 1e-5, "max of top-right 2x2");
    AX_TEST_ASSERT_NEAR(od[2], 3.0f, 1e-5, "max of bottom-left 2x2");
    AX_TEST_ASSERT_NEAR(od[3], 4.0f, 1e-5, "max of bottom-right 2x2");

    ax_tensor_destroy(input);
    ax_layer_destroy(pool);
}

static void test_avgpool2d(void)
{
    float in_data[] = {1, 2, 3, 4,
                       5, 6, 7, 8,
                       9, 10, 11, 12,
                       13, 14, 15, 16};
    ax_tensor_t *input = make_4d(in_data, 1, 1, 4, 4);

    ax_layer_t *pool = ax_avgpool2d_create(2, 2, 0);
    ax_tensor_t *out = ax_layer_forward(pool, input);

    AX_TEST_ASSERT(out != NULL, "avgpool should work");
    float *od = (float *)out->storage->data;
    /* top-left 2x2: avg(1,2,5,6) = 3.5 */
    AX_TEST_ASSERT_NEAR(od[0], 3.5f, 1e-5, "avg of top-left 2x2");
    /* bottom-right 2x2: avg(11,12,15,16) = 13.5 */
    AX_TEST_ASSERT_NEAR(od[3], 13.5f, 1e-5, "avg of bottom-right 2x2");

    ax_tensor_destroy(input);
    ax_layer_destroy(pool);
}

static void test_global_avgpool(void)
{
    float in_data[] = {1, 2, 3, 4};
    ax_tensor_t *input = make_4d(in_data, 1, 1, 2, 2);

    ax_layer_t *pool = ax_global_avgpool2d_create();
    ax_tensor_t *out = ax_layer_forward(pool, input);

    AX_TEST_ASSERT(out != NULL, "global avgpool should work");
    AX_TEST_ASSERT_EQ(out->ndim, 2, "output is [N, C]");
    AX_TEST_ASSERT_EQ(out->shape[0], 1, "batch");
    AX_TEST_ASSERT_EQ(out->shape[1], 1, "channels");

    int64_t i0[] = {0, 0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 2.5f, 1e-5, "avg(1,2,3,4) = 2.5");

    ax_tensor_destroy(input);
    ax_layer_destroy(pool);
}

static void test_flatten(void)
{
    float in_data[24];
    for (int i = 0; i < 24; i++) in_data[i] = (float)i;
    ax_tensor_t *input = make_4d(in_data, 2, 3, 2, 2);

    ax_layer_t *flat = ax_flatten_create();
    ax_tensor_t *out = ax_layer_forward(flat, input);

    AX_TEST_ASSERT(out != NULL, "flatten should work");
    AX_TEST_ASSERT_EQ(out->ndim, 2, "output is 2d");
    AX_TEST_ASSERT_EQ(out->shape[0], 2, "batch preserved");
    AX_TEST_ASSERT_EQ(out->shape[1], 12, "3*2*2 = 12");

    ax_tensor_destroy(input);
    ax_layer_destroy(flat);
}

static void test_conv_pipeline(void)
{
    /* mini cnn pipeline: conv -> relu -> pool -> flatten
       input: [1, 1, 6, 6] */
    float in_data[36];
    for (int i = 0; i < 36; i++) in_data[i] = (float)(i % 5);
    ax_tensor_t *input = make_4d(in_data, 1, 1, 6, 6);

    ax_layer_t *model = ax_sequential_create();
    ax_sequential_add(model, ax_conv2d_create(1, 4, 3, 1, 0, true));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_maxpool2d_create(2, 2, 0));
    ax_sequential_add(model, ax_flatten_create());

    ax_tensor_t *out = ax_layer_forward(model, input);
    AX_TEST_ASSERT(out != NULL, "conv pipeline should work");

    /* conv: [1,1,6,6] -> [1,4,4,4], pool: -> [1,4,2,2], flatten: -> [1,16] */
    AX_TEST_ASSERT_EQ(out->ndim, 2, "final output is 2d");
    AX_TEST_ASSERT_EQ(out->shape[0], 1, "batch");
    AX_TEST_ASSERT_EQ(out->shape[1], 16, "4*2*2 = 16 features");

    ax_tensor_destroy(input);
    ax_layer_destroy(model);
}

int main(void)
{
    ax_init();

    printf("=== conv tests ===\n");
    AX_RUN_TEST(test_im2col_basic);
    AX_RUN_TEST(test_conv2d_create);
    AX_RUN_TEST(test_conv2d_forward);
    AX_RUN_TEST(test_conv2d_with_padding);
    AX_RUN_TEST(test_maxpool2d);
    AX_RUN_TEST(test_avgpool2d);
    AX_RUN_TEST(test_global_avgpool);
    AX_RUN_TEST(test_flatten);
    AX_RUN_TEST(test_conv_pipeline);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
