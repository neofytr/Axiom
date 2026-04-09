/* test_autograd.c — tests for automatic differentiation.
   verifies that backward() computes correct gradients for every op. */

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

/* test: loss = sum(a + b), check both grads are 1 */
static void test_grad_add(void)
{
    ax_tensor_t *a = make_1d((float[]){1, 2, 3}, 3);
    ax_tensor_t *b = make_1d((float[]){4, 5, 6}, 3);
    a->requires_grad = true;
    b->requires_grad = true;

    ax_tensor_t *c = ax_add(a, b);
    ax_tensor_t *loss = ax_sum(c, -1);

    ax_backward(loss);

    /* d(sum(a+b))/da = [1, 1, 1], same for b */
    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    AX_TEST_ASSERT(a->grad != NULL, "a should have grad");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i0), 1.0f, 1e-5, "da[0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i1), 1.0f, 1e-5, "da[1]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(b->grad, i2), 1.0f, 1e-5, "db[2]");

    ax_tensor_destroy(loss); ax_tensor_destroy(c);
    ax_tensor_destroy(a); ax_tensor_destroy(b);
}

/* test: loss = sum(a * b), grad_a should be b, grad_b should be a */
static void test_grad_mul(void)
{
    ax_tensor_t *a = make_1d((float[]){2, 3, 4}, 3);
    ax_tensor_t *b = make_1d((float[]){5, 6, 7}, 3);
    a->requires_grad = true;
    b->requires_grad = true;

    ax_tensor_t *c = ax_mul(a, b);
    ax_tensor_t *loss = ax_sum(c, -1);
    ax_backward(loss);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    /* d(sum(a*b))/da = b */
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i0), 5.0f, 1e-5, "da[0]=b[0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i1), 6.0f, 1e-5, "da[1]=b[1]");
    /* d(sum(a*b))/db = a */
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(b->grad, i0), 2.0f, 1e-5, "db[0]=a[0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(b->grad, i2), 4.0f, 1e-5, "db[2]=a[2]");

    ax_tensor_destroy(loss); ax_tensor_destroy(c);
    ax_tensor_destroy(a); ax_tensor_destroy(b);
}

/* test: loss = sum(a^2), grad should be 2*a */
static void test_grad_square(void)
{
    ax_tensor_t *a = make_1d((float[]){1, 2, 3}, 3);
    a->requires_grad = true;

    ax_tensor_t *sq = ax_square(a);
    ax_tensor_t *loss = ax_sum(sq, -1);
    ax_backward(loss);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i0), 2.0f, 1e-5, "d(1^2)/da = 2");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i1), 4.0f, 1e-5, "d(2^2)/da = 4");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i2), 6.0f, 1e-5, "d(3^2)/da = 6");

    ax_tensor_destroy(loss); ax_tensor_destroy(sq); ax_tensor_destroy(a);
}

/* test: loss = sum(sigmoid(a)), check against known derivative */
static void test_grad_sigmoid(void)
{
    ax_tensor_t *a = make_1d((float[]){0.0f, 1.0f, -1.0f}, 3);
    a->requires_grad = true;

    ax_tensor_t *s = ax_sigmoid(a);
    ax_tensor_t *loss = ax_sum(s, -1);
    ax_backward(loss);

    /* sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
       sigmoid(0) = 0.5, so sigmoid'(0) = 0.25 */
    int64_t i0[] = {0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i0), 0.25f, 1e-4, "sigmoid'(0) = 0.25");

    /* sigmoid(1) ~ 0.7311, sigmoid'(1) ~ 0.1966 */
    int64_t i1[] = {1};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i1), 0.1966f, 1e-3, "sigmoid'(1)");

    ax_tensor_destroy(loss); ax_tensor_destroy(s); ax_tensor_destroy(a);
}

/* test: loss = sum(relu(a)) */
static void test_grad_relu(void)
{
    ax_tensor_t *a = make_1d((float[]){-2, 0, 3, -1, 5}, 5);
    a->requires_grad = true;

    ax_tensor_t *r = ax_relu(a);
    ax_tensor_t *loss = ax_sum(r, -1);
    ax_backward(loss);

    /* relu'(x) = 1 if x > 0, 0 otherwise */
    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2}, i3[] = {3}, i4[] = {4};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i0), 0.0f, 1e-6, "relu'(-2) = 0");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i1), 0.0f, 1e-6, "relu'(0) = 0");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i2), 1.0f, 1e-6, "relu'(3) = 1");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i3), 0.0f, 1e-6, "relu'(-1) = 0");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i4), 1.0f, 1e-6, "relu'(5) = 1");

    ax_tensor_destroy(loss); ax_tensor_destroy(r); ax_tensor_destroy(a);
}

/* test: matmul gradient. loss = sum(a @ b) */
static void test_grad_matmul(void)
{
    ax_tensor_t *a = make_2d((float[]){1, 2, 3, 4}, 2, 2);
    ax_tensor_t *b = make_2d((float[]){5, 6, 7, 8}, 2, 2);
    a->requires_grad = true;
    b->requires_grad = true;

    ax_tensor_t *c = ax_matmul(a, b);
    ax_tensor_t *loss = ax_sum(c, -1);
    ax_backward(loss);

    /* d(sum(a@b))/da = ones @ b^T = [[5+6, 7+8], [5+6, 7+8]] = ...
       actually: grad_out = [[1,1],[1,1]]
       grad_a = grad_out @ b^T
       b^T = [[5,7],[6,8]]
       grad_a = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]] */
    int64_t i00[] = {0, 0}, i01[] = {0, 1}, i10[] = {1, 0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i00), 11.0f, 1e-4, "grad_a[0,0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i01), 15.0f, 1e-4, "grad_a[0,1]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i10), 11.0f, 1e-4, "grad_a[1,0]");

    /* grad_b = a^T @ grad_out
       a^T = [[1,3],[2,4]]
       grad_b = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]] */
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(b->grad, i00), 4.0f, 1e-4, "grad_b[0,0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(b->grad, i01), 4.0f, 1e-4, "grad_b[0,1]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(b->grad, i10), 6.0f, 1e-4, "grad_b[1,0]");

    ax_tensor_destroy(loss); ax_tensor_destroy(c);
    ax_tensor_destroy(a); ax_tensor_destroy(b);
}

/* test: chain rule. loss = sum((a*b + c)^2) */
static void test_grad_chain(void)
{
    ax_tensor_t *a = make_1d((float[]){1, 2}, 2);
    ax_tensor_t *b = make_1d((float[]){3, 4}, 2);
    ax_tensor_t *c = make_1d((float[]){-1, 1}, 2);
    a->requires_grad = true;

    ax_tensor_t *ab = ax_mul(a, b);     /* [3, 8] */
    ax_tensor_t *abc = ax_add(ab, c);   /* [2, 9] */
    ax_tensor_t *sq = ax_square(abc);   /* [4, 81] */
    ax_tensor_t *loss = ax_sum(sq, -1); /* 85 */

    ax_backward(loss);

    /* d(loss)/da = d(sum((a*b+c)^2))/da
       = 2*(a*b+c) * b
       for a[0]: 2*(1*3 + (-1)) * 3 = 2*2*3 = 12
       for a[1]: 2*(2*4 + 1) * 4 = 2*9*4 = 72 */
    int64_t i0[] = {0}, i1[] = {1};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i0), 12.0f, 1e-4, "chain grad a[0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i1), 72.0f, 1e-4, "chain grad a[1]");

    ax_tensor_destroy(loss); ax_tensor_destroy(sq);
    ax_tensor_destroy(abc); ax_tensor_destroy(ab);
    ax_tensor_destroy(a); ax_tensor_destroy(b); ax_tensor_destroy(c);
}

/* test: mul_scalar gradient. loss = sum(a * 3.0) -> grad = 3 */
static void test_grad_mul_scalar(void)
{
    ax_tensor_t *a = make_1d((float[]){1, 2, 3}, 3);
    a->requires_grad = true;

    ax_tensor_t *scaled = ax_mul_scalar(a, 3.0);
    ax_tensor_t *loss = ax_sum(scaled, -1);
    ax_backward(loss);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i0), 3.0f, 1e-5, "grad = scalar");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i1), 3.0f, 1e-5, "grad = scalar");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i2), 3.0f, 1e-5, "grad = scalar");

    ax_tensor_destroy(loss); ax_tensor_destroy(scaled); ax_tensor_destroy(a);
}

/* test: neg gradient. loss = sum(-a) -> grad = -1 */
static void test_grad_neg(void)
{
    ax_tensor_t *a = make_1d((float[]){5, 10, 15}, 3);
    a->requires_grad = true;

    ax_tensor_t *neg = ax_neg(a);
    ax_tensor_t *loss = ax_sum(neg, -1);
    ax_backward(loss);

    int64_t i0[] = {0}, i1[] = {1};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i0), -1.0f, 1e-5, "neg grad = -1");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i1), -1.0f, 1e-5, "neg grad = -1");

    ax_tensor_destroy(loss); ax_tensor_destroy(neg); ax_tensor_destroy(a);
}

/* test: mean gradient. loss = mean(a) -> grad = 1/n */
static void test_grad_mean(void)
{
    ax_tensor_t *a = make_1d((float[]){2, 4, 6, 8}, 4);
    a->requires_grad = true;

    ax_tensor_t *m = ax_mean(a, -1);
    ax_backward(m);

    int64_t i0[] = {0}, i3[] = {3};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i0), 0.25f, 1e-5, "mean grad = 1/4");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i3), 0.25f, 1e-5, "mean grad = 1/4");

    ax_tensor_destroy(m); ax_tensor_destroy(a);
}

/* test: no_grad disables tracking */
static void test_no_grad(void)
{
    ax_tensor_t *a = make_1d((float[]){1, 2, 3}, 3);
    a->requires_grad = true;

    ax_no_grad();
    ax_tensor_t *b = ax_mul_scalar(a, 2.0);
    ax_enable_grad();

    /* b should not track grad because we were in no_grad mode */
    AX_TEST_ASSERT(!b->requires_grad, "no_grad should prevent grad tracking");
    AX_TEST_ASSERT(b->grad_fn == NULL, "no_grad should prevent grad_fn creation");

    ax_tensor_destroy(a); ax_tensor_destroy(b);
}

/* test: exp gradient. loss = sum(exp(a)) -> grad = exp(a) */
static void test_grad_exp(void)
{
    ax_tensor_t *a = make_1d((float[]){0.0f, 1.0f}, 2);
    a->requires_grad = true;

    ax_tensor_t *e = ax_exp(a);
    ax_tensor_t *loss = ax_sum(e, -1);
    ax_backward(loss);

    int64_t i0[] = {0}, i1[] = {1};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i0), 1.0f, 1e-4, "exp'(0) = 1");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i1), 2.71828f, 1e-3, "exp'(1) = e");

    ax_tensor_destroy(loss); ax_tensor_destroy(e); ax_tensor_destroy(a);
}

/* test: tanh gradient. loss = sum(tanh(a)) -> grad = 1 - tanh(a)^2 */
static void test_grad_tanh(void)
{
    ax_tensor_t *a = make_1d((float[]){0.0f}, 1);
    a->requires_grad = true;

    ax_tensor_t *t = ax_tanh_op(a);
    ax_tensor_t *loss = ax_sum(t, -1);
    ax_backward(loss);

    /* tanh'(0) = 1 - 0^2 = 1 */
    int64_t i0[] = {0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i0), 1.0f, 1e-4, "tanh'(0) = 1");

    ax_tensor_destroy(loss); ax_tensor_destroy(t); ax_tensor_destroy(a);
}

int main(void)
{
    ax_init();

    printf("=== autograd tests ===\n");
    AX_RUN_TEST(test_grad_add);
    AX_RUN_TEST(test_grad_mul);
    AX_RUN_TEST(test_grad_square);
    AX_RUN_TEST(test_grad_sigmoid);
    AX_RUN_TEST(test_grad_relu);
    AX_RUN_TEST(test_grad_matmul);
    AX_RUN_TEST(test_grad_chain);
    AX_RUN_TEST(test_grad_mul_scalar);
    AX_RUN_TEST(test_grad_neg);
    AX_RUN_TEST(test_grad_mean);
    AX_RUN_TEST(test_no_grad);
    AX_RUN_TEST(test_grad_exp);
    AX_RUN_TEST(test_grad_tanh);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
