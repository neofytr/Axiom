/* test_losses.c — verify loss functions and their gradients */

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

static void test_mse_loss(void)
{
    ax_tensor_t *pred = make_1d((float[]){1, 2, 3}, 3);
    ax_tensor_t *target = make_1d((float[]){1, 2, 3}, 3);
    pred->requires_grad = true;

    ax_tensor_t *loss = ax_mse_loss(pred, target);

    int64_t i0[] = {0};
    /* perfect prediction: loss should be 0 */
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(loss, i0), 0.0f, 1e-6, "mse of perfect pred = 0");

    ax_tensor_destroy(loss);
    ax_tensor_destroy(pred); ax_tensor_destroy(target);

    /* now with actual error */
    pred = make_1d((float[]){1, 2, 3}, 3);
    target = make_1d((float[]){2, 2, 4}, 3);
    pred->requires_grad = true;

    loss = ax_mse_loss(pred, target);
    /* mse = mean((1-2)^2 + (2-2)^2 + (3-4)^2) = mean(1 + 0 + 1) = 2/3 */
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(loss, i0), 2.0f / 3.0f, 1e-4, "mse value");

    ax_backward(loss);

    /* grad of mse w.r.t. pred: 2*(pred-target)/n */
    int64_t ii0[] = {0}, ii1[] = {1}, ii2[] = {2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(pred->grad, ii0), -2.0f/3.0f, 1e-4, "mse grad[0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(pred->grad, ii1), 0.0f, 1e-4, "mse grad[1]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(pred->grad, ii2), -2.0f/3.0f, 1e-4, "mse grad[2]");

    ax_tensor_destroy(loss);
    ax_tensor_destroy(pred); ax_tensor_destroy(target);
}

static void test_cross_entropy(void)
{
    /* 2 samples, 3 classes */
    ax_tensor_t *logits = make_2d((float[]){
        2.0f, 1.0f, 0.1f,   /* sample 0: class 0 should win */
        0.5f, 2.5f, 0.3f    /* sample 1: class 1 should win */
    }, 2, 3);

    ax_tensor_t *target = make_2d((float[]){
        1, 0, 0,    /* sample 0 is class 0 */
        0, 1, 0     /* sample 1 is class 1 */
    }, 2, 3);

    logits->requires_grad = true;

    ax_tensor_t *loss = ax_cross_entropy_loss(logits, target);
    AX_TEST_ASSERT(loss != NULL, "cross entropy should work");

    int64_t i0[] = {0};
    float lv = ax_tensor_get_f32(loss, i0);
    /* loss should be positive and reasonably small since predictions are decent */
    AX_TEST_ASSERT(lv > 0.0f, "ce loss should be positive");
    AX_TEST_ASSERT(lv < 2.0f, "ce loss should be reasonable");

    ax_backward(loss);

    /* gradient should exist */
    AX_TEST_ASSERT(logits->grad != NULL, "logits should have gradient");

    /* for the correct class, gradient should be (softmax - 1)/batch_size (negative)
       for wrong classes, gradient should be softmax/batch_size (positive) */
    int64_t g00[] = {0, 0};
    float g = ax_tensor_get_f32(logits->grad, g00);
    AX_TEST_ASSERT(g < 0.0f, "grad for correct class should be negative");

    ax_tensor_destroy(loss);
    ax_tensor_destroy(logits); ax_tensor_destroy(target);
}

static void test_bce_loss(void)
{
    ax_tensor_t *logits = make_1d((float[]){10, -10, 0}, 3);
    ax_tensor_t *target = make_1d((float[]){1, 0, 0.5f}, 3);
    logits->requires_grad = true;

    ax_tensor_t *loss = ax_bce_with_logits_loss(logits, target);
    AX_TEST_ASSERT(loss != NULL, "bce should work");

    int64_t i0[] = {0};
    float lv = ax_tensor_get_f32(loss, i0);
    /* high confidence correct predictions: loss should be small */
    AX_TEST_ASSERT(lv >= 0.0f, "bce loss should be non-negative");

    ax_backward(loss);
    AX_TEST_ASSERT(logits->grad != NULL, "logits should have gradient");

    /* for logit=10, target=1: grad = sigmoid(10) - 1 ~ 0 (almost perfect) */
    int64_t g0[] = {0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(logits->grad, g0), 0.0f, 0.01f, "near-perfect pred grad ~0");

    ax_tensor_destroy(loss);
    ax_tensor_destroy(logits); ax_tensor_destroy(target);
}

static void test_huber_loss(void)
{
    ax_tensor_t *pred = make_1d((float[]){0, 3, 10}, 3);
    ax_tensor_t *target = make_1d((float[]){0, 0, 0}, 3);
    pred->requires_grad = true;

    ax_tensor_t *loss = ax_huber_loss(pred, target, 1.0f);
    AX_TEST_ASSERT(loss != NULL, "huber should work");

    int64_t i0[] = {0};
    float lv = ax_tensor_get_f32(loss, i0);

    /* pred=0, target=0: contributes 0
       pred=3, target=0: |d|=3 > delta=1, contributes 1*(3-0.5) = 2.5
       pred=10, target=0: |d|=10 > delta=1, contributes 1*(10-0.5) = 9.5
       mean = (0 + 2.5 + 9.5) / 3 = 4.0 */
    AX_TEST_ASSERT_NEAR(lv, 4.0f, 1e-4, "huber loss value");

    ax_backward(loss);
    AX_TEST_ASSERT(pred->grad != NULL, "pred should have gradient");

    ax_tensor_destroy(loss);
    ax_tensor_destroy(pred); ax_tensor_destroy(target);
}

static void test_mae_loss(void)
{
    ax_tensor_t *pred = make_1d((float[]){1, 3, 5}, 3);
    ax_tensor_t *target = make_1d((float[]){2, 3, 3}, 3);
    pred->requires_grad = true;

    ax_tensor_t *loss = ax_mae_loss(pred, target);

    int64_t i0[] = {0};
    /* mae = mean(|1-2| + |3-3| + |5-3|) = mean(1 + 0 + 2) = 1.0 */
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(loss, i0), 1.0f, 1e-4, "mae value");

    ax_backward(loss);

    /* grad: sign(pred - target) / n */
    int64_t g0[] = {0}, g1[] = {1}, g2[] = {2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(pred->grad, g0), -1.0f/3.0f, 1e-4, "mae grad[0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(pred->grad, g1), 0.0f, 1e-4, "mae grad[1]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(pred->grad, g2), 1.0f/3.0f, 1e-4, "mae grad[2]");

    ax_tensor_destroy(loss);
    ax_tensor_destroy(pred); ax_tensor_destroy(target);
}

int main(void)
{
    ax_init();

    printf("=== loss function tests ===\n");
    AX_RUN_TEST(test_mse_loss);
    AX_RUN_TEST(test_cross_entropy);
    AX_RUN_TEST(test_bce_loss);
    AX_RUN_TEST(test_huber_loss);
    AX_RUN_TEST(test_mae_loss);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
