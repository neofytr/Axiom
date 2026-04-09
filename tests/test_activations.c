/* test_activations.c — verify activation functions and their gradients */

#include "test.h"
#include "axiom/axiom.h"

static ax_tensor_t *make_1d(float *data, int64_t n)
{
    return ax_tensor_from_array(data, &n, 1, AX_FLOAT32);
}

static void test_leaky_relu(void)
{
    ax_tensor_t *a = make_1d((float[]){-2, -1, 0, 1, 2}, 5);
    ax_tensor_t *out = ax_leaky_relu(a, 0.01f);

    int64_t i0[] = {0}, i1[] = {1}, i3[] = {3}, i4[] = {4};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), -0.02f, 1e-5, "leaky_relu(-2)");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i1), -0.01f, 1e-5, "leaky_relu(-1)");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i3), 1.0f, 1e-5, "leaky_relu(1)");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i4), 2.0f, 1e-5, "leaky_relu(2)");

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

static void test_leaky_relu_grad(void)
{
    ax_tensor_t *a = make_1d((float[]){-1, 2}, 2);
    a->requires_grad = true;

    ax_tensor_t *out = ax_leaky_relu(a, 0.1f);
    ax_tensor_t *loss = ax_sum(out, -1);
    ax_backward(loss);

    int64_t i0[] = {0}, i1[] = {1};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i0), 0.1f, 1e-5, "leaky_relu'(-1) = alpha");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i1), 1.0f, 1e-5, "leaky_relu'(2) = 1");

    ax_tensor_destroy(loss); ax_tensor_destroy(out); ax_tensor_destroy(a);
}

static void test_elu(void)
{
    ax_tensor_t *a = make_1d((float[]){-1, 0, 1}, 3);
    ax_tensor_t *out = ax_elu(a, 1.0f);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    /* elu(-1) = 1.0 * (exp(-1) - 1) ~ -0.6321 */
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), -0.6321f, 1e-3, "elu(-1)");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i1), 0.0f, 1e-5, "elu(0)");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i2), 1.0f, 1e-5, "elu(1)");

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

static void test_gelu(void)
{
    ax_tensor_t *a = make_1d((float[]){0, 1, -1}, 3);
    ax_tensor_t *out = ax_gelu(a);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    /* gelu(0) = 0 */
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 0.0f, 1e-4, "gelu(0)");
    /* gelu(1) ~ 0.8412 */
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i1), 0.8412f, 1e-3, "gelu(1)");
    /* gelu(-1) ~ -0.1588 */
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i2), -0.1588f, 1e-3, "gelu(-1)");

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

static void test_swish(void)
{
    ax_tensor_t *a = make_1d((float[]){0, 2, -2}, 3);
    ax_tensor_t *out = ax_swish(a);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    /* swish(0) = 0 * sigmoid(0) = 0 */
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 0.0f, 1e-5, "swish(0)");
    /* swish(2) = 2 * sigmoid(2) ~ 2 * 0.8808 = 1.7616 */
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i1), 1.7616f, 1e-3, "swish(2)");
    /* swish(-2) = -2 * sigmoid(-2) ~ -2 * 0.1192 = -0.2384 */
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i2), -0.2384f, 1e-3, "swish(-2)");

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

static void test_softplus(void)
{
    ax_tensor_t *a = make_1d((float[]){0, 10, -10}, 3);
    ax_tensor_t *out = ax_softplus(a);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    /* softplus(0) = log(2) ~ 0.6931 */
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 0.6931f, 1e-3, "softplus(0)");
    /* softplus(10) ~ 10 (large x) */
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i1), 10.0f, 1e-3, "softplus(10)");
    /* softplus(-10) ~ exp(-10) ~ 0 */
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i2), 0.0f, 1e-3, "softplus(-10)");

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

static void test_softmax(void)
{
    ax_tensor_t *a = make_1d((float[]){1, 2, 3}, 3);
    ax_tensor_t *out = ax_softmax(a, -1);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    float s0 = ax_tensor_get_f32(out, i0);
    float s1 = ax_tensor_get_f32(out, i1);
    float s2 = ax_tensor_get_f32(out, i2);

    /* softmax should sum to 1 */
    AX_TEST_ASSERT_NEAR(s0 + s1 + s2, 1.0f, 1e-5, "softmax sums to 1");
    /* values should be ordered: s0 < s1 < s2 */
    AX_TEST_ASSERT(s0 < s1, "softmax preserves order");
    AX_TEST_ASSERT(s1 < s2, "softmax preserves order");
    /* known values */
    AX_TEST_ASSERT_NEAR(s2, 0.6652f, 1e-3, "softmax(3)");

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

static void test_softmax_2d(void)
{
    float data[] = {1, 2, 3, 4, 5, 6};
    int64_t shape[] = {2, 3};
    ax_tensor_t *a = ax_tensor_from_array(data, shape, 2, AX_FLOAT32);
    ax_tensor_t *out = ax_softmax(a, 1);

    /* each row should sum to 1 */
    int64_t i00[] = {0, 0}, i01[] = {0, 1}, i02[] = {0, 2};
    float row0 = ax_tensor_get_f32(out, i00) +
                 ax_tensor_get_f32(out, i01) +
                 ax_tensor_get_f32(out, i02);
    AX_TEST_ASSERT_NEAR(row0, 1.0f, 1e-5, "row 0 sums to 1");

    int64_t i10[] = {1, 0}, i11[] = {1, 1}, i12[] = {1, 2};
    float row1 = ax_tensor_get_f32(out, i10) +
                 ax_tensor_get_f32(out, i11) +
                 ax_tensor_get_f32(out, i12);
    AX_TEST_ASSERT_NEAR(row1, 1.0f, 1e-5, "row 1 sums to 1");

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

int main(void)
{
    ax_init();

    printf("=== activation tests ===\n");
    AX_RUN_TEST(test_leaky_relu);
    AX_RUN_TEST(test_leaky_relu_grad);
    AX_RUN_TEST(test_elu);
    AX_RUN_TEST(test_gelu);
    AX_RUN_TEST(test_swish);
    AX_RUN_TEST(test_softplus);
    AX_RUN_TEST(test_softmax);
    AX_RUN_TEST(test_softmax_2d);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
