/* test_lr_sched.c — verify lr scheduler behavior */

#include "test.h"
#include "axiom/axiom.h"
#include "axiom/lr_scheduler.h"
#include <math.h>

static ax_optimizer_t *make_dummy_opt(float lr)
{
    /* we just need an optimizer struct to hold the lr.
       no actual params needed for scheduler testing. */
    return ax_sgd_create(NULL, 0, lr, 0, 0, false);
}

static void test_step_decay(void)
{
    ax_optimizer_t *opt = make_dummy_opt(0.1f);
    ax_lr_scheduler_t *s = ax_sched_step_decay(opt, 3, 0.5f);
    AX_TEST_ASSERT(s != NULL, "should create");

    /* initial lr */
    AX_TEST_ASSERT_NEAR(ax_sched_get_lr(s), 0.1f, 1e-6, "initial lr");

    /* steps 1, 2: no decay yet (step/step_size = 0) */
    ax_sched_step(s);
    AX_TEST_ASSERT_NEAR(ax_sched_get_lr(s), 0.1f, 1e-6, "step 1");
    ax_sched_step(s);
    AX_TEST_ASSERT_NEAR(ax_sched_get_lr(s), 0.1f, 1e-6, "step 2");

    /* step 3: first decay (3/3 = 1, 0.1 * 0.5^1 = 0.05) */
    ax_sched_step(s);
    AX_TEST_ASSERT_NEAR(ax_sched_get_lr(s), 0.05f, 1e-6, "step 3: first decay");

    /* step 6: second decay (6/3 = 2, 0.1 * 0.5^2 = 0.025) */
    ax_sched_step(s); ax_sched_step(s); ax_sched_step(s);
    AX_TEST_ASSERT_NEAR(ax_sched_get_lr(s), 0.025f, 1e-5, "step 6: second decay");

    ax_sched_destroy(s);
    ax_optimizer_destroy(opt);
}

static void test_exponential(void)
{
    ax_optimizer_t *opt = make_dummy_opt(1.0f);
    ax_lr_scheduler_t *s = ax_sched_exponential(opt, 0.9f);

    ax_sched_step(s);
    AX_TEST_ASSERT_NEAR(ax_sched_get_lr(s), 0.9f, 1e-4, "step 1: 1.0 * 0.9");

    ax_sched_step(s);
    AX_TEST_ASSERT_NEAR(ax_sched_get_lr(s), 0.81f, 1e-3, "step 2: 1.0 * 0.9^2");

    ax_sched_step(s);
    AX_TEST_ASSERT_NEAR(ax_sched_get_lr(s), 0.729f, 1e-3, "step 3: 1.0 * 0.9^3");

    ax_sched_destroy(s);
    ax_optimizer_destroy(opt);
}

static void test_cosine(void)
{
    ax_optimizer_t *opt = make_dummy_opt(0.1f);
    ax_lr_scheduler_t *s = ax_sched_cosine(opt, 100, 0.001f);

    /* at step 0 (initial): lr = 0.1 */
    AX_TEST_ASSERT_NEAR(ax_sched_get_lr(s), 0.1f, 1e-6, "initial");

    /* at step 50 (halfway): lr should be approximately (0.1 + 0.001) / 2 */
    for (int i = 0; i < 50; i++) ax_sched_step(s);
    float mid = ax_sched_get_lr(s);
    AX_TEST_ASSERT(mid < 0.1f, "mid should be less than initial");
    AX_TEST_ASSERT(mid > 0.001f, "mid should be more than min");
    /* cos(pi * 0.5) = 0, so lr = 0.001 + 0.5 * 0.099 * (1 + 0) = 0.0505 */
    AX_TEST_ASSERT_NEAR(mid, 0.0505f, 0.005f, "midpoint ~ 0.05");

    /* at step 100 (end): lr should be close to min_lr */
    for (int i = 0; i < 50; i++) ax_sched_step(s);
    AX_TEST_ASSERT_NEAR(ax_sched_get_lr(s), 0.001f, 0.001f, "end ~ min_lr");

    ax_sched_destroy(s);
    ax_optimizer_destroy(opt);
}

static void test_warmup_cosine(void)
{
    ax_optimizer_t *opt = make_dummy_opt(0.1f);
    ax_lr_scheduler_t *s = ax_sched_warmup_cosine(opt, 10, 100, 0.0f);

    /* warmup phase: lr ramps linearly from 0 to 0.1 over 10 steps */
    ax_sched_step(s); /* step 1 */
    AX_TEST_ASSERT_NEAR(ax_sched_get_lr(s), 0.01f, 0.002f, "warmup step 1: 0.1 * 1/10");

    for (int i = 0; i < 4; i++) ax_sched_step(s); /* step 5 */
    AX_TEST_ASSERT_NEAR(ax_sched_get_lr(s), 0.05f, 0.002f, "warmup step 5: 0.1 * 5/10");

    for (int i = 0; i < 5; i++) ax_sched_step(s); /* step 10 */
    AX_TEST_ASSERT_NEAR(ax_sched_get_lr(s), 0.1f, 0.002f, "warmup done: full lr");

    /* decay phase: cosine from 0.1 to 0 over 90 remaining steps */
    for (int i = 0; i < 45; i++) ax_sched_step(s); /* step 55, midpoint of decay */
    float mid = ax_sched_get_lr(s);
    AX_TEST_ASSERT(mid < 0.1f, "decay started");
    AX_TEST_ASSERT(mid > 0.0f, "not at zero yet");

    /* run to end */
    for (int i = 0; i < 45; i++) ax_sched_step(s);
    AX_TEST_ASSERT_NEAR(ax_sched_get_lr(s), 0.0f, 0.005f, "end ~ 0");

    ax_sched_destroy(s);
    ax_optimizer_destroy(opt);
}

static void test_scheduler_with_training(void)
{
    /* verify that a scheduler actually helps training */
    float x_data[] = {0,0, 0,1, 1,0, 1,1};
    float y_data[] = {0, 1, 1, 0};
    int64_t xs[] = {4, 2}, ys[] = {4, 1};
    ax_tensor_t *x = ax_tensor_from_array(x_data, xs, 2, AX_FLOAT32);
    ax_tensor_t *y = ax_tensor_from_array(y_data, ys, 2, AX_FLOAT32);

    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(2, 16, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(16, 1, true));

    ax_model_t *m = ax_model_create(net);
    ax_optimizer_t *opt = ax_adam_create(m->params, m->n_params,
                                         0.01f, 0.9f, 0.999f, 1e-8f, 0);
    ax_model_compile(m, opt, ax_mse_loss);

    ax_lr_scheduler_t *sched = ax_sched_cosine(opt, 500, 0.0001f);

    float first = ax_model_train_step(m, x, y);
    for (int i = 0; i < 499; i++)
    {
        ax_model_train_step(m, x, y);
        ax_sched_step(sched);
    }
    float last = ax_model_train_step(m, x, y);

    AX_TEST_ASSERT(last < first, "training with scheduler should reduce loss");
    AX_TEST_ASSERT(last < 0.05f, "should converge");

    /* lr should have decayed */
    AX_TEST_ASSERT(ax_sched_get_lr(sched) < 0.01f, "lr should have decayed");

    ax_tensor_destroy(x); ax_tensor_destroy(y);
    ax_sched_destroy(sched);
    ax_model_destroy(m);
}

int main(void)
{
    ax_init();
    printf("=== lr scheduler tests ===\n");
    AX_RUN_TEST(test_step_decay);
    AX_RUN_TEST(test_exponential);
    AX_RUN_TEST(test_cosine);
    AX_RUN_TEST(test_warmup_cosine);
    AX_RUN_TEST(test_scheduler_with_training);
    ax_shutdown();
    AX_TEST_SUMMARY();
}
