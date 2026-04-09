/* test_norm.c -- batchnorm, layernorm, dropout, gradient clipping */

#include "test.h"
#include "axiom/axiom.h"

static ax_tensor_t *make_2d(float *data, int64_t rows, int64_t cols)
{
    int64_t shape[] = {rows, cols};
    return ax_tensor_from_array(data, shape, 2, AX_FLOAT32);
}

static void test_batchnorm_forward(void)
{
    ax_layer_t *bn = ax_batchnorm_create(2, 1e-5f, 0.1f);
    AX_TEST_ASSERT(bn != NULL, "batchnorm create");
    AX_TEST_ASSERT_EQ(bn->n_params, 2, "gamma + beta");

    ax_tensor_t *input = make_2d((float[]){1,10, 2,10, 3,10}, 3, 2);
    ax_layer_train(bn);
    ax_tensor_t *out = ax_layer_forward(bn, input);
    AX_TEST_ASSERT(out != NULL, "forward works");

    /* normalized feature 0 should have ~zero mean */
    int64_t i00[] = {0, 0}, i10[] = {1, 0}, i20[] = {2, 0};
    float sum = ax_tensor_get_f32(out, i00) +
                ax_tensor_get_f32(out, i10) +
                ax_tensor_get_f32(out, i20);
    AX_TEST_ASSERT_NEAR(sum, 0.0f, 0.01f, "zero mean");

    ax_tensor_destroy(input); ax_tensor_destroy(out);
    ax_layer_destroy(bn);
}

static void test_batchnorm_eval(void)
{
    ax_layer_t *bn = ax_batchnorm_create(2, 1e-5f, 1.0f);

    ax_tensor_t *train_in = make_2d((float[]){0,0, 2,4, 4,8}, 3, 2);
    ax_layer_train(bn);
    ax_tensor_t *train_out = ax_layer_forward(bn, train_in);

    ax_layer_eval(bn);
    ax_tensor_t *test_in = make_2d((float[]){2,4}, 1, 2);
    ax_tensor_t *eval_out = ax_layer_forward(bn, test_in);
    AX_TEST_ASSERT(eval_out != NULL, "eval forward");

    /* input at mean of training data -> output ~0 */
    int64_t i00[] = {0, 0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(eval_out, i00), 0.0f, 0.2f, "eval ~0 at mean");

    ax_tensor_destroy(train_in); ax_tensor_destroy(train_out);
    ax_tensor_destroy(test_in); ax_tensor_destroy(eval_out);
    ax_layer_destroy(bn);
}

static void test_layernorm(void)
{
    ax_layer_t *ln = ax_layernorm_create(3, 1e-5f);
    AX_TEST_ASSERT(ln != NULL, "layernorm create");

    ax_tensor_t *input = make_2d((float[]){1,2,3, 10,10,10}, 2, 3);
    ax_tensor_t *out = ax_layer_forward(ln, input);
    AX_TEST_ASSERT(out != NULL, "forward");

    /* sample 0 normalized across features: zero mean */
    int64_t i00[] = {0, 0}, i01[] = {0, 1}, i02[] = {0, 2};
    float s = ax_tensor_get_f32(out, i00) +
              ax_tensor_get_f32(out, i01) +
              ax_tensor_get_f32(out, i02);
    AX_TEST_ASSERT_NEAR(s, 0.0f, 0.01f, "zero mean across features");

    ax_tensor_destroy(input); ax_tensor_destroy(out);
    ax_layer_destroy(ln);
}

static void test_dropout_train(void)
{
    ax_layer_t *dp = ax_dropout_create(0.5f);
    ax_layer_train(dp);

    int64_t shape[] = {1, 1000};
    ax_tensor_t *input = ax_tensor_ones(shape, 2, AX_FLOAT32);
    ax_tensor_t *out = ax_layer_forward(dp, input);
    AX_TEST_ASSERT(out != NULL, "dropout forward");

    int64_t n = ax_tensor_numel(out);
    int zeros = 0;
    float *od = (float *)out->storage->data;
    for (int64_t i = 0; i < n; i++)
        if (od[out->offset + i] == 0.0f) zeros++;

    AX_TEST_ASSERT(zeros > 200 && zeros < 800, "roughly 50% dropped");

    ax_tensor_destroy(input); ax_tensor_destroy(out);
    ax_layer_destroy(dp);
}

static void test_dropout_eval(void)
{
    ax_layer_t *dp = ax_dropout_create(0.5f);
    ax_layer_eval(dp);

    int64_t shape[] = {1, 100};
    ax_tensor_t *input = ax_tensor_ones(shape, 2, AX_FLOAT32);
    ax_tensor_t *out = ax_layer_forward(dp, input);
    AX_TEST_ASSERT(out != NULL, "eval forward");

    int64_t i0[] = {0, 0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 1.0f, 1e-6, "no dropout in eval");

    ax_tensor_destroy(input); ax_tensor_destroy(out);
    ax_layer_destroy(dp);
}

static void test_clip_grad_value(void)
{
    ax_tensor_t *w = ax_tensor_from_array(
        (float[]){1, 2, 3, 4}, &(int64_t){4}, 1, AX_FLOAT32);
    w->requires_grad = true;
    w->grad = ax_tensor_from_array(
        (float[]){-10, 5, 0.5f, 100}, &(int64_t){4}, 1, AX_FLOAT32);

    ax_tensor_t *params[] = {w};
    ax_clip_grad_value(params, 1, 3.0f);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2}, i3[] = {3};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(w->grad, i0), -3.0f, 1e-6, "clipped");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(w->grad, i1), 3.0f, 1e-6, "clipped");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(w->grad, i2), 0.5f, 1e-6, "not clipped");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(w->grad, i3), 3.0f, 1e-6, "clipped");

    ax_tensor_destroy(w);
}

static void test_clip_grad_norm(void)
{
    ax_tensor_t *w = ax_tensor_from_array(
        (float[]){3, 4}, &(int64_t){2}, 1, AX_FLOAT32);
    w->requires_grad = true;
    w->grad = ax_tensor_from_array(
        (float[]){3, 4}, &(int64_t){2}, 1, AX_FLOAT32);

    ax_tensor_t *params[] = {w};
    float norm = ax_clip_grad_norm(params, 1, 1.0f);
    AX_TEST_ASSERT_NEAR(norm, 5.0f, 1e-4, "original norm=5");

    int64_t i0[] = {0}, i1[] = {1};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(w->grad, i0), 0.6f, 1e-4, "3/5");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(w->grad, i1), 0.8f, 1e-4, "4/5");

    ax_tensor_destroy(w);
}

int main(void)
{
    ax_init();
    printf("=== norm tests ===\n");
    AX_RUN_TEST(test_batchnorm_forward);
    AX_RUN_TEST(test_batchnorm_eval);
    AX_RUN_TEST(test_layernorm);
    AX_RUN_TEST(test_dropout_train);
    AX_RUN_TEST(test_dropout_eval);
    AX_RUN_TEST(test_clip_grad_value);
    AX_RUN_TEST(test_clip_grad_norm);
    ax_shutdown();
    AX_TEST_SUMMARY();
}
