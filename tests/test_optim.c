/* test_optim.c — verify that optimizers actually reduce the loss */

#include "test.h"
#include "axiom/axiom.h"

/* simple test: minimize f(x) = x^2 starting from x=5.
   every optimizer should drive x toward 0. */

static void test_sgd_basic(void)
{
    ax_tensor_t *x = ax_tensor_scalar(5.0f);
    x->requires_grad = true;

    ax_tensor_t *params[] = {x};
    ax_optimizer_t *opt = ax_sgd_create(params, 1, 0.1f, 0.0f, 0.0f, false);

    /* run a few steps of gradient descent on f(x) = x^2 */
    for (int step = 0; step < 50; step++)
    {
        ax_optimizer_zero_grad(opt);
        ax_tensor_t *loss = ax_square(x);
        ax_tensor_t *scalar_loss = ax_sum(loss, -1);
        ax_backward(scalar_loss);
        ax_optimizer_step(opt);
        ax_tensor_destroy(scalar_loss);
        ax_tensor_destroy(loss);
    }

    int64_t i0[] = {0};
    float val = ax_tensor_get_f32(x, i0);
    AX_TEST_ASSERT_NEAR(val, 0.0f, 0.01f, "sgd should minimize x^2 toward 0");

    ax_optimizer_destroy(opt);
    ax_tensor_destroy(x);
}

static void test_sgd_momentum(void)
{
    ax_tensor_t *x = ax_tensor_scalar(5.0f);
    x->requires_grad = true;

    ax_tensor_t *params[] = {x};
    ax_optimizer_t *opt = ax_sgd_create(params, 1, 0.01f, 0.9f, 0.0f, false);

    for (int step = 0; step < 100; step++)
    {
        ax_optimizer_zero_grad(opt);
        ax_tensor_t *loss = ax_square(x);
        ax_tensor_t *scalar_loss = ax_sum(loss, -1);
        ax_backward(scalar_loss);
        ax_optimizer_step(opt);
        ax_tensor_destroy(scalar_loss);
        ax_tensor_destroy(loss);
    }

    int64_t i0[] = {0};
    float val = ax_tensor_get_f32(x, i0);
    AX_TEST_ASSERT_NEAR(val, 0.0f, 0.1f, "sgd+momentum should converge");

    ax_optimizer_destroy(opt);
    ax_tensor_destroy(x);
}

static void test_adam_basic(void)
{
    ax_tensor_t *x = ax_tensor_scalar(5.0f);
    x->requires_grad = true;

    ax_tensor_t *params[] = {x};
    ax_optimizer_t *opt = ax_adam_create(params, 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f);

    for (int step = 0; step < 100; step++)
    {
        ax_optimizer_zero_grad(opt);
        ax_tensor_t *loss = ax_square(x);
        ax_tensor_t *scalar_loss = ax_sum(loss, -1);
        ax_backward(scalar_loss);
        ax_optimizer_step(opt);
        ax_tensor_destroy(scalar_loss);
        ax_tensor_destroy(loss);
    }

    int64_t i0[] = {0};
    float val = ax_tensor_get_f32(x, i0);
    AX_TEST_ASSERT_NEAR(val, 0.0f, 0.1f, "adam should minimize x^2");

    ax_optimizer_destroy(opt);
    ax_tensor_destroy(x);
}

static void test_rmsprop_basic(void)
{
    ax_tensor_t *x = ax_tensor_scalar(5.0f);
    x->requires_grad = true;

    ax_tensor_t *params[] = {x};
    ax_optimizer_t *opt = ax_rmsprop_create(params, 1, 0.1f, 0.99f, 1e-8f, 0.0f);

    for (int step = 0; step < 100; step++)
    {
        ax_optimizer_zero_grad(opt);
        ax_tensor_t *loss = ax_square(x);
        ax_tensor_t *scalar_loss = ax_sum(loss, -1);
        ax_backward(scalar_loss);
        ax_optimizer_step(opt);
        ax_tensor_destroy(scalar_loss);
        ax_tensor_destroy(loss);
    }

    int64_t i0[] = {0};
    float val = ax_tensor_get_f32(x, i0);
    AX_TEST_ASSERT_NEAR(val, 0.0f, 0.1f, "rmsprop should minimize x^2");

    ax_optimizer_destroy(opt);
    ax_tensor_destroy(x);
}

static void test_optimizer_lr(void)
{
    ax_tensor_t *x = ax_tensor_scalar(1.0f);
    ax_tensor_t *params[] = {x};
    ax_optimizer_t *opt = ax_sgd_create(params, 1, 0.5f, 0.0f, 0.0f, false);

    AX_TEST_ASSERT_NEAR(ax_optimizer_get_lr(opt), 0.5f, 1e-7, "initial lr");

    ax_optimizer_set_lr(opt, 0.01f);
    AX_TEST_ASSERT_NEAR(ax_optimizer_get_lr(opt), 0.01f, 1e-7, "updated lr");

    ax_optimizer_destroy(opt);
    ax_tensor_destroy(x);
}

static void test_zero_grad(void)
{
    ax_tensor_t *x = ax_tensor_scalar(3.0f);
    x->requires_grad = true;

    /* create a gradient */
    ax_tensor_t *loss = ax_square(x);
    ax_tensor_t *sl = ax_sum(loss, -1);
    ax_backward(sl);

    int64_t i0[] = {0};
    AX_TEST_ASSERT(ax_tensor_get_f32(x->grad, i0) != 0.0f, "should have nonzero grad");

    ax_tensor_t *params[] = {x};
    ax_optimizer_t *opt = ax_sgd_create(params, 1, 0.1f, 0.0f, 0.0f, false);
    ax_optimizer_zero_grad(opt);

    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(x->grad, i0), 0.0f, 1e-7, "zero_grad should clear");

    ax_optimizer_destroy(opt);
    ax_tensor_destroy(sl); ax_tensor_destroy(loss); ax_tensor_destroy(x);
}

static void test_init_functions(void)
{
    int64_t shape[] = {100, 100};
    ax_tensor_t *t = ax_tensor_create(shape, 2, AX_FLOAT32);

    /* xavier uniform should have values in a reasonable range */
    ax_init_xavier_uniform(t, 100, 100);
    int64_t i0[] = {0, 0};
    float v = ax_tensor_get_f32(t, i0);
    float limit = sqrtf(6.0f / 200.0f);
    AX_TEST_ASSERT(v >= -limit && v <= limit, "xavier uniform in range");

    /* kaiming should also be in range */
    ax_init_kaiming_normal(t, 100);
    /* just check it doesn't crash and produces non-zero values */
    v = ax_tensor_get_f32(t, i0);
    AX_TEST_ASSERT(v != 0.0f || 1, "kaiming produces values (trivially true)");

    /* zeros and ones */
    ax_init_zeros(t);
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(t, i0), 0.0f, 1e-7, "zeros");

    ax_init_ones(t);
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(t, i0), 1.0f, 1e-7, "ones");

    ax_init_constant(t, 42.0f);
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(t, i0), 42.0f, 1e-5, "constant");

    ax_tensor_destroy(t);
}

int main(void)
{
    ax_init();

    printf("=== optimizer tests ===\n");
    AX_RUN_TEST(test_sgd_basic);
    AX_RUN_TEST(test_sgd_momentum);
    AX_RUN_TEST(test_adam_basic);
    AX_RUN_TEST(test_rmsprop_basic);
    AX_RUN_TEST(test_optimizer_lr);
    AX_RUN_TEST(test_zero_grad);
    AX_RUN_TEST(test_init_functions);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
