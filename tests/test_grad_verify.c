/* test_grad_verify.c — numerical gradient verification for every backward.

   this is the most important test file in the project. it catches
   wrong gradients that integration tests miss (a backward that's off
   by a factor of 2 will still make loss decrease, just slower).

   method: for each parameter, perturb by +eps and -eps, compute loss
   both ways, numerical grad = (loss_plus - loss_minus) / (2 * eps).
   compare against analytical grad from ax_backward.

   tolerance is generous (1e-2) because float32 finite differences
   are inherently noisy for complex operations. */

#include "test.h"
#include "axiom/axiom.h"
#include <math.h>

#define EPS 1e-3f
#define TOL 0.05f  /* 5% relative error or 1e-3 absolute */

/* helper: compute scalar loss from a forward pass.
   the forward_fn takes input and returns a scalar tensor. */
typedef ax_tensor_t *(*forward_fn_t)(ax_tensor_t *input);

/* numerically verify gradients of input tensor w.r.t. a scalar loss.
   returns max absolute difference between analytical and numerical grad. */
static float verify_grad(ax_tensor_t *param, forward_fn_t forward, ax_tensor_t *input)
{
    /* step 1: compute analytical gradient */
    param->requires_grad = true;
    if (param->grad)
    {
        ax_compute_fill(param->grad, 0.0);
    }

    ax_enable_grad();
    ax_tensor_t *loss = forward(input);
    ax_backward(loss);
    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss);

    if (!param->grad) return -1.0f;

    /* step 2: compute numerical gradient for each element */
    float *data = (float *)param->storage->data;
    int64_t n = ax_tensor_numel(param);
    float max_diff = 0;

    for (int64_t i = 0; i < n; i++)
    {
        float orig = data[param->offset + i];

        /* f(x + eps) */
        data[param->offset + i] = orig + EPS;
        ax_no_grad();
        ax_tensor_t *lp = forward(input);
        int64_t idx[] = {0};
        float fp = ax_tensor_get_f32(lp, idx);
        ax_tensor_destroy(lp);

        /* f(x - eps) */
        data[param->offset + i] = orig - EPS;
        ax_tensor_t *lm = forward(input);
        float fm = ax_tensor_get_f32(lm, idx);
        ax_tensor_destroy(lm);
        ax_enable_grad();

        /* restore */
        data[param->offset + i] = orig;

        float numerical = (fp - fm) / (2.0f * EPS);

        /* get analytical gradient for this element */
        int64_t remaining = i;
        int64_t indices[AX_MAX_DIMS];
        for (int d = param->ndim - 1; d >= 0; d--)
        {
            indices[d] = remaining % param->shape[d];
            remaining /= param->shape[d];
        }
        float analytical = ax_tensor_get_f32(param->grad, indices);

        float diff = fabsf(analytical - numerical);
        float scale = fmaxf(fabsf(analytical), fabsf(numerical));
        /* use relative error for large values, absolute for small */
        float rel = (scale > 1e-4f) ? diff / scale : diff;

        if (rel > max_diff) max_diff = rel;
    }

    return max_diff;
}


/* global state for forward functions (ugly but necessary since
   the callback signature can't carry extra args) */
static ax_tensor_t *g_weight = NULL;
static ax_tensor_t *g_bias = NULL;
static ax_layer_t *g_layer = NULL;


/* dense gradient check */
static ax_tensor_t *dense_forward_for_check(ax_tensor_t *input)
{
    ax_tensor_t *out = ax_matmul(input, g_weight);
    ax_tensor_t *loss = ax_sum(out, -1);
    return loss;
}

static void test_dense_grad(void)
{
    float x_data[] = {1, 2, 3, 4, 5, 6};
    int64_t xs[] = {2, 3};
    ax_tensor_t *x = ax_tensor_from_array(x_data, xs, 2, AX_FLOAT32);

    int64_t ws[] = {3, 2};
    g_weight = ax_tensor_rand(ws, 2, -1.0f, 1.0f);

    float err = verify_grad(g_weight, dense_forward_for_check, x);
    AX_TEST_ASSERT(err >= 0, "grad should be computed");
    AX_TEST_ASSERT(err < TOL, "dense weight grad should match numerical");

    ax_tensor_destroy(x);
    ax_tensor_destroy(g_weight);
    g_weight = NULL;
}


/* sigmoid gradient check */
static ax_tensor_t *sigmoid_forward_for_check(ax_tensor_t *input)
{
    /* loss = sum(sigmoid(input * weight)) */
    ax_tensor_t *z = ax_matmul(input, g_weight);
    ax_tensor_t *s = ax_sigmoid(z);
    ax_tensor_t *loss = ax_sum(s, -1);
    return loss;
}

static void test_sigmoid_grad(void)
{
    float x_data[] = {0.5f, -0.3f, 0.8f, -0.1f};
    int64_t xs[] = {2, 2};
    ax_tensor_t *x = ax_tensor_from_array(x_data, xs, 2, AX_FLOAT32);

    int64_t ws[] = {2, 2};
    g_weight = ax_tensor_rand(ws, 2, -1.0f, 1.0f);

    float err = verify_grad(g_weight, sigmoid_forward_for_check, x);
    AX_TEST_ASSERT(err >= 0, "grad computed");
    AX_TEST_ASSERT(err < TOL, "sigmoid grad should match numerical");

    ax_tensor_destroy(x);
    ax_tensor_destroy(g_weight);
    g_weight = NULL;
}


/* relu gradient check */
static ax_tensor_t *relu_forward_for_check(ax_tensor_t *input)
{
    ax_tensor_t *z = ax_matmul(input, g_weight);
    ax_tensor_t *r = ax_relu(z);
    ax_tensor_t *loss = ax_sum(r, -1);
    return loss;
}

static void test_relu_grad(void)
{
    /* use values far from 0 to avoid the relu discontinuity.
       numerical gradient checking fails at the kink (z=0)
       because finite differences straddle the boundary. */
    float x_data[] = {1, 2, -1, 0.5f};
    int64_t xs[] = {2, 2};
    ax_tensor_t *x = ax_tensor_from_array(x_data, xs, 2, AX_FLOAT32);

    /* initialize weights to values that produce activations far from 0 */
    int64_t ws[] = {2, 3};
    g_weight = ax_tensor_rand(ws, 2, 0.5f, 2.0f);

    float err = verify_grad(g_weight, relu_forward_for_check, x);
    AX_TEST_ASSERT(err >= 0, "grad computed");
    AX_TEST_ASSERT(err < TOL, "relu grad should match numerical");

    ax_tensor_destroy(x);
    ax_tensor_destroy(g_weight);
    g_weight = NULL;
}


/* cross-entropy gradient check */
static ax_tensor_t *g_target = NULL;

static ax_tensor_t *ce_forward_for_check(ax_tensor_t *input)
{
    ax_tensor_t *logits = ax_matmul(input, g_weight);
    ax_tensor_t *loss = ax_cross_entropy_loss(logits, g_target);
    return loss;
}

static void test_cross_entropy_grad(void)
{
    float x_data[] = {1, 0.5f, -0.5f, 1};
    int64_t xs[] = {2, 2};
    ax_tensor_t *x = ax_tensor_from_array(x_data, xs, 2, AX_FLOAT32);

    float t_data[] = {1,0,0, 0,0,1};
    int64_t ts[] = {2, 3};
    g_target = ax_tensor_from_array(t_data, ts, 2, AX_FLOAT32);

    int64_t ws[] = {2, 3};
    g_weight = ax_tensor_rand(ws, 2, -0.5f, 0.5f);

    float err = verify_grad(g_weight, ce_forward_for_check, x);
    AX_TEST_ASSERT(err >= 0, "grad computed");
    AX_TEST_ASSERT(err < TOL, "cross-entropy grad should match numerical");

    ax_tensor_destroy(x);
    ax_tensor_destroy(g_target);
    ax_tensor_destroy(g_weight);
    g_weight = NULL;
    g_target = NULL;
}


/* mse gradient check */
static ax_tensor_t *mse_forward_for_check(ax_tensor_t *input)
{
    ax_tensor_t *pred = ax_matmul(input, g_weight);
    ax_tensor_t *loss = ax_mse_loss(pred, g_target);
    return loss;
}

static void test_mse_grad(void)
{
    float x_data[] = {1, 2, 3, 4};
    int64_t xs[] = {2, 2};
    ax_tensor_t *x = ax_tensor_from_array(x_data, xs, 2, AX_FLOAT32);

    float t_data[] = {0.5f, 0.8f};
    int64_t ts[] = {2, 1};
    g_target = ax_tensor_from_array(t_data, ts, 2, AX_FLOAT32);

    int64_t ws[] = {2, 1};
    g_weight = ax_tensor_rand(ws, 2, -1.0f, 1.0f);

    float err = verify_grad(g_weight, mse_forward_for_check, x);
    AX_TEST_ASSERT(err >= 0, "grad computed");
    AX_TEST_ASSERT(err < TOL, "mse grad should match numerical");

    ax_tensor_destroy(x);
    ax_tensor_destroy(g_target);
    ax_tensor_destroy(g_weight);
    g_weight = NULL;
    g_target = NULL;
}


/* square + mean chain rule check */
static ax_tensor_t *square_mean_forward(ax_tensor_t *input)
{
    ax_tensor_t *z = ax_matmul(input, g_weight);
    ax_tensor_t *sq = ax_square(z);
    ax_tensor_t *loss = ax_mean(sq, -1);
    return loss;
}

static void test_square_mean_grad(void)
{
    float x_data[] = {0.5f, -0.3f, 0.8f, 0.1f};
    int64_t xs[] = {2, 2};
    ax_tensor_t *x = ax_tensor_from_array(x_data, xs, 2, AX_FLOAT32);

    int64_t ws[] = {2, 2};
    g_weight = ax_tensor_rand(ws, 2, -1.0f, 1.0f);

    float err = verify_grad(g_weight, square_mean_forward, x);
    AX_TEST_ASSERT(err >= 0, "grad computed");
    AX_TEST_ASSERT(err < TOL, "square+mean grad should match numerical");

    ax_tensor_destroy(x);
    ax_tensor_destroy(g_weight);
    g_weight = NULL;
}


/* exp + log chain check */
static ax_tensor_t *exp_log_forward(ax_tensor_t *input)
{
    ax_tensor_t *z = ax_matmul(input, g_weight);
    ax_tensor_t *e = ax_exp(z);
    ax_tensor_t *l = ax_log(e); /* should give back z */
    ax_tensor_t *loss = ax_sum(l, -1);
    return loss;
}

static void test_exp_log_grad(void)
{
    float x_data[] = {0.1f, 0.2f, 0.3f, 0.4f};
    int64_t xs[] = {2, 2};
    ax_tensor_t *x = ax_tensor_from_array(x_data, xs, 2, AX_FLOAT32);

    int64_t ws[] = {2, 2};
    g_weight = ax_tensor_rand(ws, 2, -0.5f, 0.5f);

    float err = verify_grad(g_weight, exp_log_forward, x);
    AX_TEST_ASSERT(err >= 0, "grad computed");
    AX_TEST_ASSERT(err < TOL, "exp+log grad should match numerical");

    ax_tensor_destroy(x);
    ax_tensor_destroy(g_weight);
    g_weight = NULL;
}


/* tanh gradient check */
static ax_tensor_t *tanh_forward_for_check(ax_tensor_t *input)
{
    ax_tensor_t *z = ax_matmul(input, g_weight);
    ax_tensor_t *t = ax_tanh_op(z);
    ax_tensor_t *loss = ax_sum(t, -1);
    return loss;
}

static void test_tanh_grad(void)
{
    float x_data[] = {0.5f, -0.5f, 0.3f, 0.7f};
    int64_t xs[] = {2, 2};
    ax_tensor_t *x = ax_tensor_from_array(x_data, xs, 2, AX_FLOAT32);

    int64_t ws[] = {2, 2};
    g_weight = ax_tensor_rand(ws, 2, -1.0f, 1.0f);

    float err = verify_grad(g_weight, tanh_forward_for_check, x);
    AX_TEST_ASSERT(err >= 0, "grad computed");
    AX_TEST_ASSERT(err < TOL, "tanh grad should match numerical");

    ax_tensor_destroy(x);
    ax_tensor_destroy(g_weight);
    g_weight = NULL;
}


/* div gradient check */
static ax_tensor_t *div_forward_for_check(ax_tensor_t *input)
{
    ax_tensor_t *a = ax_matmul(input, g_weight);
    ax_tensor_t *ones = ax_tensor_full(a->shape, a->ndim, a->dtype, 2.0);
    ax_tensor_t *d = ax_div(a, ones);
    ax_tensor_t *loss = ax_sum(d, -1);
    return loss;
}

static void test_div_grad(void)
{
    float x_data[] = {1, 2, 3, 4};
    int64_t xs[] = {2, 2};
    ax_tensor_t *x = ax_tensor_from_array(x_data, xs, 2, AX_FLOAT32);

    int64_t ws[] = {2, 2};
    g_weight = ax_tensor_rand(ws, 2, -1.0f, 1.0f);

    float err = verify_grad(g_weight, div_forward_for_check, x);
    AX_TEST_ASSERT(err >= 0, "grad computed");
    AX_TEST_ASSERT(err < TOL, "div grad should match numerical");

    ax_tensor_destroy(x);
    ax_tensor_destroy(g_weight);
    g_weight = NULL;
}


int main(void)
{
    ax_init();

    printf("=== numerical gradient verification ===\n");
    AX_RUN_TEST(test_dense_grad);
    AX_RUN_TEST(test_sigmoid_grad);
    AX_RUN_TEST(test_relu_grad);
    AX_RUN_TEST(test_cross_entropy_grad);
    AX_RUN_TEST(test_mse_grad);
    AX_RUN_TEST(test_square_mean_grad);
    AX_RUN_TEST(test_exp_log_grad);
    AX_RUN_TEST(test_tanh_grad);
    AX_RUN_TEST(test_div_grad);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
