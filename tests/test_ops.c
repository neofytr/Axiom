/* test_ops.c — tests for the high-level tensor ops api (broadcasting, auto allocation) */

#include "test.h"
#include "axiom/axiom.h"

static ax_tensor_t *make_1d(float *data, int64_t n)
{
    return ax_tensor_from_array(data, &n, 1, AX_FLOAT32);
}

static ax_tensor_t *make_2d(float *data, int64_t rows, int64_t cols)
{
    int64_t shape[] = {rows, cols};
    return ax_tensor_from_array(data, shape, 2, AX_FLOAT32);
}

static void test_ops_add(void)
{
    ax_tensor_t *a = make_1d((float[]){1, 2, 3}, 3);
    ax_tensor_t *b = make_1d((float[]){10, 20, 30}, 3);
    ax_tensor_t *c = ax_add(a, b);

    AX_TEST_ASSERT(c != NULL, "add should return a tensor");
    int64_t i0[] = {0}, i2[] = {2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(c, i0), 11.0f, 1e-6, "1+10");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(c, i2), 33.0f, 1e-6, "3+30");

    ax_tensor_destroy(a); ax_tensor_destroy(b); ax_tensor_destroy(c);
}

static void test_ops_broadcast_add(void)
{
    /* [2,3] + [3] should broadcast to [2,3] */
    ax_tensor_t *a = make_2d((float[]){1, 2, 3, 4, 5, 6}, 2, 3);
    ax_tensor_t *b = make_1d((float[]){10, 20, 30}, 3);
    ax_tensor_t *c = ax_add(a, b);

    AX_TEST_ASSERT(c != NULL, "broadcast add should work");
    AX_TEST_ASSERT_EQ(c->ndim, 2, "result should be 2d");
    AX_TEST_ASSERT_EQ(c->shape[0], 2, "shape[0]");
    AX_TEST_ASSERT_EQ(c->shape[1], 3, "shape[1]");

    int64_t i00[] = {0, 0}, i01[] = {0, 1}, i10[] = {1, 0}, i12[] = {1, 2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(c, i00), 11.0f, 1e-6, "[0,0]: 1+10");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(c, i01), 22.0f, 1e-6, "[0,1]: 2+20");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(c, i10), 14.0f, 1e-6, "[1,0]: 4+10");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(c, i12), 36.0f, 1e-6, "[1,2]: 6+30");

    ax_tensor_destroy(a); ax_tensor_destroy(b); ax_tensor_destroy(c);
}

static void test_ops_matmul(void)
{
    ax_tensor_t *a = make_2d((float[]){1, 2, 3, 4, 5, 6}, 2, 3);
    ax_tensor_t *b = make_2d((float[]){7, 8, 9, 10, 11, 12}, 3, 2);
    ax_tensor_t *c = ax_matmul(a, b);

    AX_TEST_ASSERT(c != NULL, "matmul should work");
    AX_TEST_ASSERT_EQ(c->shape[0], 2, "rows");
    AX_TEST_ASSERT_EQ(c->shape[1], 2, "cols");

    int64_t i00[] = {0, 0}, i01[] = {0, 1}, i10[] = {1, 0}, i11[] = {1, 1};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(c, i00), 58.0f, 1e-4, "[0,0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(c, i01), 64.0f, 1e-4, "[0,1]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(c, i10), 139.0f, 1e-4, "[1,0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(c, i11), 154.0f, 1e-4, "[1,1]");

    ax_tensor_destroy(a); ax_tensor_destroy(b); ax_tensor_destroy(c);
}

static void test_ops_chain(void)
{
    /* test chaining: relu(a * b + c) */
    ax_tensor_t *a = make_1d((float[]){-1, 2, -3, 4}, 4);
    ax_tensor_t *b = make_1d((float[]){2, 3, 2, 1}, 4);
    ax_tensor_t *c = make_1d((float[]){1, 1, 1, 1}, 4);

    ax_tensor_t *ab = ax_mul(a, b);
    ax_tensor_t *abc = ax_add(ab, c);
    ax_tensor_t *out = ax_relu(abc);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2}, i3[] = {3};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 0.0f, 1e-6, "relu(-2+1)=0");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i1), 7.0f, 1e-6, "relu(6+1)=7");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i2), 0.0f, 1e-6, "relu(-6+1)=0");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i3), 5.0f, 1e-6, "relu(4+1)=5");

    ax_tensor_destroy(a); ax_tensor_destroy(b); ax_tensor_destroy(c);
    ax_tensor_destroy(ab); ax_tensor_destroy(abc); ax_tensor_destroy(out);
}

static void test_ops_sum_mean(void)
{
    ax_tensor_t *a = make_1d((float[]){1, 2, 3, 4}, 4);

    ax_tensor_t *s = ax_sum(a, -1);
    int64_t i0[] = {0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(s, i0), 10.0f, 1e-5, "sum");

    ax_tensor_t *m = ax_mean(a, -1);
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(m, i0), 2.5f, 1e-5, "mean");

    ax_tensor_destroy(a); ax_tensor_destroy(s); ax_tensor_destroy(m);
}

static void test_ops_scalar(void)
{
    ax_tensor_t *a = make_1d((float[]){2, 4, 6}, 3);
    ax_tensor_t *b = ax_mul_scalar(a, 0.5);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(b, i0), 1.0f, 1e-6, "2*0.5");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(b, i1), 2.0f, 1e-6, "4*0.5");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(b, i2), 3.0f, 1e-6, "6*0.5");

    ax_tensor_destroy(a); ax_tensor_destroy(b);
}

int main(void)
{
    ax_init();

    printf("=== ops tests ===\n");
    AX_RUN_TEST(test_ops_add);
    AX_RUN_TEST(test_ops_broadcast_add);
    AX_RUN_TEST(test_ops_matmul);
    AX_RUN_TEST(test_ops_chain);
    AX_RUN_TEST(test_ops_sum_mean);
    AX_RUN_TEST(test_ops_scalar);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
