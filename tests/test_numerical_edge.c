/* test_numerical_edge.c -- NaN, Inf, and numerical stability edge cases */

#include "test.h"
#include "axiom/axiom.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static ax_tensor_t *make_2d(float *data, int64_t rows, int64_t cols)
{
    int64_t shape[] = {rows, cols};
    return ax_tensor_from_array(data, shape, 2, AX_FLOAT32);
}

/* 1. softmax with logits spanning -100..100 should not produce NaN/Inf */
static void test_softmax_large_logits(void)
{
    float data[10];
    for (int i = 0; i < 10; i++)
        data[i] = -100.0f + 200.0f * (float)i / 9.0f;

    int64_t shape[] = {1, 10};
    ax_tensor_t *a = ax_tensor_from_array(data, shape, 2, AX_FLOAT32);
    ax_tensor_t *out = ax_softmax(a, -1);
    AX_TEST_ASSERT(out != NULL, "softmax should not fail");

    float sum = 0;
    for (int i = 0; i < 10; i++) {
        int64_t idx[] = {0, i};
        float v = ax_tensor_get_f32(out, idx);
        AX_TEST_ASSERT(!isnan(v), "no NaN in softmax output");
        AX_TEST_ASSERT(!isinf(v), "no Inf in softmax output");
        AX_TEST_ASSERT(v >= 0.0f && v <= 1.0f, "softmax in [0,1]");
        sum += v;
    }
    AX_TEST_ASSERT_NEAR(sum, 1.0f, 1e-5f, "softmax sums to 1");

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

/* 2. identical logits should give uniform distribution */
static void test_softmax_identical_logits(void)
{
    float data[10];
    for (int i = 0; i < 10; i++) data[i] = 1000.0f;

    int64_t shape[] = {1, 10};
    ax_tensor_t *a = ax_tensor_from_array(data, shape, 2, AX_FLOAT32);
    ax_tensor_t *out = ax_softmax(a, -1);
    AX_TEST_ASSERT(out != NULL, "softmax identical logits");

    for (int i = 0; i < 10; i++) {
        int64_t idx[] = {0, i};
        float v = ax_tensor_get_f32(out, idx);
        AX_TEST_ASSERT(!isnan(v), "no NaN");
        AX_TEST_ASSERT(!isinf(v), "no Inf");
        AX_TEST_ASSERT_NEAR(v, 0.1f, 1e-5f, "uniform 1/10");
    }

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

/* 3. single element softmax must be 1.0 */
static void test_softmax_single_element(void)
{
    int64_t shape[] = {1, 1};
    ax_tensor_t *a = ax_tensor_full(shape, 2, AX_FLOAT32, 42.0);
    ax_tensor_t *out = ax_softmax(a, -1);
    AX_TEST_ASSERT(out != NULL, "single element softmax");

    int64_t idx[] = {0, 0};
    float v = ax_tensor_get_f32(out, idx);
    AX_TEST_ASSERT(!isnan(v), "no NaN");
    AX_TEST_ASSERT_NEAR(v, 1.0f, 1e-6f, "single element = 1.0");

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

/* 4. exp of large values near float max exp boundary */
static void test_exp_overflow(void)
{
    int64_t shape[] = {1, 2};
    float data[] = {88.0f, 100.0f};
    ax_tensor_t *a = ax_tensor_from_array(data, shape, 2, AX_FLOAT32);
    ax_tensor_t *out = ax_exp(a);
    AX_TEST_ASSERT(out != NULL, "exp should not crash");

    int64_t i0[] = {0, 0}, i1[] = {0, 1};
    float v88 = ax_tensor_get_f32(out, i0);
    float v100 = ax_tensor_get_f32(out, i1);

    /* exp(88) ~ 1.65e38, within float range */
    AX_TEST_ASSERT(!isnan(v88), "exp(88) not NaN");
    AX_TEST_ASSERT(v88 > 1e38f, "exp(88) is large");

    /* exp(100) overflows float; Inf is acceptable, NaN is not */
    AX_TEST_ASSERT(!isnan(v100), "exp(100) not NaN");

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

/* 5. log(0) should produce -Inf, not crash */
static void test_log_zero(void)
{
    int64_t shape[] = {1, 2};
    float data[] = {0.0f, 1.0f};
    ax_tensor_t *a = ax_tensor_from_array(data, shape, 2, AX_FLOAT32);
    ax_tensor_t *out = ax_log(a);
    AX_TEST_ASSERT(out != NULL, "log should not crash");

    int64_t i0[] = {0, 0}, i1[] = {0, 1};
    float v0 = ax_tensor_get_f32(out, i0);
    float v1 = ax_tensor_get_f32(out, i1);

    AX_TEST_ASSERT(!isnan(v0), "log(0) not NaN");
    AX_TEST_ASSERT(isinf(v0) || v0 < -1e30f, "log(0) = -Inf");
    AX_TEST_ASSERT_NEAR(v1, 0.0f, 1e-6f, "log(1) = 0");

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

/* 6. cross-entropy with high confidence one-hot predictions */
static void test_cross_entropy_one_hot(void)
{
    float logits_data[] = {10.0f, -10.0f, -10.0f, -10.0f, -10.0f};
    float target_data[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    ax_tensor_t *logits = make_2d(logits_data, 1, 5);
    ax_tensor_t *target = make_2d(target_data, 1, 5);
    logits->requires_grad = true;

    ax_tensor_t *loss = ax_cross_entropy_loss(logits, target);
    AX_TEST_ASSERT(loss != NULL, "cross entropy one-hot");

    int64_t i0[] = {0};
    float lv = ax_tensor_get_f32(loss, i0);
    AX_TEST_ASSERT(!isnan(lv), "no NaN in loss");
    AX_TEST_ASSERT(!isinf(lv), "no Inf in loss");
    AX_TEST_ASSERT(lv >= 0.0f, "positive loss");
    AX_TEST_ASSERT(lv < 5.0f, "reasonable loss for confident correct pred");

    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss);
    ax_tensor_destroy(logits); ax_tensor_destroy(target);
}

/* 7. cross-entropy with uniform logits should give log(C) */
static void test_cross_entropy_uniform(void)
{
    float logits_data[10] = {0};
    float target_data[10] = {0};
    target_data[0] = 1.0f;

    ax_tensor_t *logits = make_2d(logits_data, 1, 10);
    ax_tensor_t *target = make_2d(target_data, 1, 10);
    logits->requires_grad = true;

    ax_tensor_t *loss = ax_cross_entropy_loss(logits, target);
    AX_TEST_ASSERT(loss != NULL, "cross entropy uniform");

    int64_t i0[] = {0};
    float lv = ax_tensor_get_f32(loss, i0);
    float expected = logf(10.0f);
    AX_TEST_ASSERT(!isnan(lv), "no NaN");
    AX_TEST_ASSERT_NEAR(lv, expected, 0.01f, "loss = log(10)");

    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss);
    ax_tensor_destroy(logits); ax_tensor_destroy(target);
}

/* 8. cross-entropy with very wrong prediction: large but finite loss */
static void test_cross_entropy_wrong_class(void)
{
    float logits_data[] = {-100.0f, 10.0f, 10.0f, 10.0f, 10.0f};
    float target_data[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    ax_tensor_t *logits = make_2d(logits_data, 1, 5);
    ax_tensor_t *target = make_2d(target_data, 1, 5);
    logits->requires_grad = true;

    ax_tensor_t *loss = ax_cross_entropy_loss(logits, target);
    AX_TEST_ASSERT(loss != NULL, "cross entropy wrong class");

    int64_t i0[] = {0};
    float lv = ax_tensor_get_f32(loss, i0);
    AX_TEST_ASSERT(!isnan(lv), "no NaN");
    AX_TEST_ASSERT(!isinf(lv), "finite loss");
    AX_TEST_ASSERT(lv > 10.0f, "large loss for wrong prediction");

    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss);
    ax_tensor_destroy(logits); ax_tensor_destroy(target);
}

/* 9. batchnorm with zero variance channel */
static void test_bn_zero_variance(void)
{
    ax_layer_t *bn = ax_batchnorm_create(2, 1e-5f, 0.1f);
    AX_TEST_ASSERT(bn != NULL, "bn create");
    ax_layer_train(bn);

    /* channel 1 has identical values across batch => zero variance */
    float data[] = {1.0f, 5.0f,
                    2.0f, 5.0f,
                    3.0f, 5.0f,
                    4.0f, 5.0f};
    ax_tensor_t *input = make_2d(data, 4, 2);
    ax_tensor_t *out = ax_layer_forward(bn, input);
    AX_TEST_ASSERT(out != NULL, "bn fwd with zero var channel");

    /* zero-var channel normalizes to (x - mean) / sqrt(0 + eps) ~ 0 */
    for (int i = 0; i < 4; i++) {
        int64_t idx[] = {i, 1};
        float v = ax_tensor_get_f32(out, idx);
        AX_TEST_ASSERT(!isnan(v), "no NaN in zero-var channel");
        AX_TEST_ASSERT(!isinf(v), "no Inf in zero-var channel");
        AX_TEST_ASSERT_NEAR(v, 0.0f, 1e-2f, "zero-var channel output ~0");
    }

    ax_tensor_destroy(input); ax_tensor_destroy(out);
    ax_layer_destroy(bn);
}

/* 10. batchnorm with N=1: degenerate statistics, must not crash */
static void test_bn_single_sample(void)
{
    ax_layer_t *bn = ax_batchnorm_create(3, 1e-5f, 0.1f);
    AX_TEST_ASSERT(bn != NULL, "bn create");
    ax_layer_train(bn);

    float data[] = {1.0f, 2.0f, 3.0f};
    ax_tensor_t *input = make_2d(data, 1, 3);
    ax_tensor_t *out = ax_layer_forward(bn, input);
    AX_TEST_ASSERT(out != NULL, "bn N=1 should not crash");

    for (int i = 0; i < 3; i++) {
        int64_t idx[] = {0, i};
        float v = ax_tensor_get_f32(out, idx);
        AX_TEST_ASSERT(!isnan(v), "no NaN with N=1");
        AX_TEST_ASSERT(!isinf(v), "no Inf with N=1");
    }

    /* running stats should exist and be finite */
    ax_batchnorm_t *bnl = (ax_batchnorm_t *)bn;
    float *rm = (float *)bnl->running_mean->storage->data;
    float *rv = (float *)bnl->running_var->storage->data;
    for (int i = 0; i < 3; i++) {
        AX_TEST_ASSERT(!isnan(rm[i]), "running mean finite");
        AX_TEST_ASSERT(!isnan(rv[i]), "running var finite");
    }

    ax_tensor_destroy(input); ax_tensor_destroy(out);
    ax_layer_destroy(bn);
}

/* 11. layernorm with constant features => output all zeros */
static void test_layernorm_constant_input(void)
{
    ax_layer_t *ln = ax_layernorm_create(4, 1e-5f);
    AX_TEST_ASSERT(ln != NULL, "ln create");

    float data[] = {7.0f, 7.0f, 7.0f, 7.0f,
                    3.0f, 3.0f, 3.0f, 3.0f};
    ax_tensor_t *input = make_2d(data, 2, 4);
    ax_tensor_t *out = ax_layer_forward(ln, input);
    AX_TEST_ASSERT(out != NULL, "ln constant input");

    for (int b = 0; b < 2; b++) {
        for (int f = 0; f < 4; f++) {
            int64_t idx[] = {b, f};
            float v = ax_tensor_get_f32(out, idx);
            AX_TEST_ASSERT(!isnan(v), "no NaN in constant layernorm");
            AX_TEST_ASSERT(!isinf(v), "no Inf in constant layernorm");
            AX_TEST_ASSERT_NEAR(v, 0.0f, 1e-3f, "constant input => ~0 output");
        }
    }

    ax_tensor_destroy(input); ax_tensor_destroy(out);
    ax_layer_destroy(ln);
}

/* 12. relu at zero: forward should be 0, backward grad at 0 should be 0 */
static void test_relu_at_zero(void)
{
    float data[] = {0.0f, -1.0f, 1.0f};
    int64_t n = 3;
    ax_tensor_t *a = ax_tensor_from_array(data, &n, 1, AX_FLOAT32);
    a->requires_grad = true;

    ax_tensor_t *out = ax_relu(a);
    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 0.0f, 1e-7f, "relu(0) = 0");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i1), 0.0f, 1e-7f, "relu(-1) = 0");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i2), 1.0f, 1e-7f, "relu(1) = 1");

    ax_tensor_t *loss = ax_sum(out, -1);
    ax_backward(loss);

    AX_TEST_ASSERT(a->grad != NULL, "relu grad exists");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i0), 0.0f, 1e-7f, "relu'(0) = 0");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i1), 0.0f, 1e-7f, "relu'(-1) = 0");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a->grad, i2), 1.0f, 1e-7f, "relu'(1) = 1");

    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss); ax_tensor_destroy(out); ax_tensor_destroy(a);
}

/* 13. gemm with zero matrices */
static void test_gemm_zero_matrices(void)
{
    int64_t sa[] = {8, 16}, sb[] = {16, 4};
    ax_tensor_t *a = ax_tensor_zeros(sa, 2, AX_FLOAT32);
    ax_tensor_t *b = ax_tensor_zeros(sb, 2, AX_FLOAT32);
    ax_tensor_t *c = ax_matmul(a, b);
    AX_TEST_ASSERT(c != NULL, "gemm zeros");

    int64_t total = ax_tensor_numel(c);
    float *cd = (float *)c->storage->data;
    for (int64_t i = 0; i < total; i++) {
        AX_TEST_ASSERT(!isnan(cd[i]), "no NaN in zero gemm");
        AX_TEST_ASSERT_NEAR(cd[i], 0.0f, 1e-7f, "zero * zero = zero");
    }

    ax_tensor_destroy(a); ax_tensor_destroy(b); ax_tensor_destroy(c);
}

/* 14. gemm with large values to test accumulator overflow */
static void test_gemm_large_values(void)
{
    int64_t K = 1024;
    int64_t sa[] = {4, K}, sb[] = {K, 4};
    ax_tensor_t *a = ax_tensor_full(sa, 2, AX_FLOAT32, 1e4);
    ax_tensor_t *b = ax_tensor_full(sb, 2, AX_FLOAT32, 1e4);
    ax_tensor_t *c = ax_matmul(a, b);
    AX_TEST_ASSERT(c != NULL, "gemm large values");

    /* expected: 1e4 * 1e4 * 1024 = 1.024e11, within float range */
    float expected = 1e4f * 1e4f * (float)K;
    int64_t total = ax_tensor_numel(c);
    float *cd = (float *)c->storage->data;
    for (int64_t i = 0; i < total; i++) {
        AX_TEST_ASSERT(!isnan(cd[i]), "no NaN in large gemm");
        AX_TEST_ASSERT(!isinf(cd[i]), "no Inf in large gemm");
        AX_TEST_ASSERT_NEAR(cd[i], expected, expected * 1e-4f, "large gemm value");
    }

    ax_tensor_destroy(a); ax_tensor_destroy(b); ax_tensor_destroy(c);
}

/* 15. gemm with K=1: outer product */
static void test_gemm_k_equals_one(void)
{
    float ad[] = {1, 2, 3, 4};
    float bd[] = {5, 6};
    int64_t sa[] = {4, 1}, sb[] = {1, 2};
    ax_tensor_t *a = ax_tensor_from_array(ad, sa, 2, AX_FLOAT32);
    ax_tensor_t *b = ax_tensor_from_array(bd, sb, 2, AX_FLOAT32);
    ax_tensor_t *c = ax_matmul(a, b);
    AX_TEST_ASSERT(c != NULL, "gemm K=1");

    /* outer product: c[i][j] = a[i] * b[j] */
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            int64_t idx[] = {i, j};
            float expected = ad[i] * bd[j];
            AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(c, idx), expected, 1e-5f, "outer product");
        }
    }

    ax_tensor_destroy(a); ax_tensor_destroy(b); ax_tensor_destroy(c);
}

/* 16. MHA with long sequence S=1024 */
static void test_attention_long_sequence(void)
{
    int64_t S = 1024, D = 64, H = 2, B = 1;
    int64_t BH = B * H;
    int64_t dk = D / H;
    int64_t elems = BH * S * dk;

    size_t n = (size_t)elems;
    float *Q = (float *)malloc(n * sizeof(float));
    float *K = (float *)malloc(n * sizeof(float));
    float *V = (float *)malloc(n * sizeof(float));
    float *O = (float *)calloc(n, sizeof(float));
    float *L = (float *)calloc((size_t)(BH * S), sizeof(float));

    ax_set_seed(42);
    for (int64_t i = 0; i < elems; i++) {
        Q[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
        K[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
        V[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
    }

    float scale = 1.0f / sqrtf((float)dk);
    ax_fused_attention_fwd(Q, K, V, O, L, BH, S, dk, scale);

    int bad = 0;
    for (int64_t i = 0; i < elems; i++) {
        if (isnan(O[i]) || isinf(O[i])) bad++;
    }
    AX_TEST_ASSERT(bad == 0, "no NaN/Inf in long sequence attention output");

    for (int64_t i = 0; i < BH * S; i++) {
        AX_TEST_ASSERT(!isnan(L[i]), "logsumexp finite");
    }

    free(Q); free(K); free(V); free(O); free(L);
}

/* 17. causal attention: masked positions must have zero contribution */
static void test_attention_causal_mask(void)
{
    int64_t S = 32, dk = 32, H = 2, B = 1;
    int64_t BH = B * H;
    int64_t elems = BH * S * dk;

    size_t n = (size_t)elems;
    float *Q = (float *)malloc(n * sizeof(float));
    float *K = (float *)malloc(n * sizeof(float));
    float *V = (float *)calloc(n, sizeof(float));
    float *O = (float *)calloc(n, sizeof(float));

    ax_set_seed(123);
    for (int64_t i = 0; i < elems; i++) {
        Q[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.5f;
        K[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.5f;
    }

    /* set V so that each position j has a unique signature:
       V[h, j, :] = j * ones. then O[h, i, :] = weighted sum of j-values.
       with causal masking, only j <= i contribute. */
    for (int64_t h = 0; h < BH; h++)
        for (int64_t j = 0; j < S; j++)
            for (int64_t d = 0; d < dk; d++)
                V[h * S * dk + j * dk + d] = (float)j;

    float scale = 1.0f / sqrtf((float)dk);
    ax_fused_attention_fwd_causal(Q, K, V, O, NULL, BH, S, dk, scale);

    /* for query position 0 with causal mask, only key position 0
       is visible. so O[h, 0, :] = softmax(s00) * V[h, 0, :] = V[h, 0, :] = 0 */
    int bad = 0;
    for (int64_t h = 0; h < BH; h++) {
        for (int64_t d = 0; d < dk; d++) {
            float v = O[h * S * dk + 0 * dk + d];
            if (fabsf(v) > 1e-5f) bad++;
        }
    }
    AX_TEST_ASSERT(bad == 0, "causal: q=0 only sees k=0 (V=0)");

    /* general check: with this V encoding, O[h,i,:] must be <= i
       because the weighted average of {0,1,...,i} is at most i */
    int violated = 0;
    for (int64_t h = 0; h < BH; h++) {
        for (int64_t i = 1; i < S; i++) {
            float v = O[h * S * dk + i * dk + 0];
            if (v > (float)i + 1e-3f) violated++;
        }
    }
    AX_TEST_ASSERT(violated == 0, "causal: attention avg bounded by position");

    /* no NaN/Inf */
    for (int64_t i = 0; i < elems; i++) {
        AX_TEST_ASSERT(!isnan(O[i]) && !isinf(O[i]), "causal output finite");
    }

    free(Q); free(K); free(V); free(O);
}

/* 18. division by zero: should produce Inf, not crash */
static void test_division_by_zero(void)
{
    int64_t shape[] = {1, 3};
    float ad[] = {1.0f, 0.0f, -1.0f};
    float bd[] = {0.0f, 0.0f, 0.0f};
    ax_tensor_t *a = ax_tensor_from_array(ad, shape, 2, AX_FLOAT32);
    ax_tensor_t *b = ax_tensor_from_array(bd, shape, 2, AX_FLOAT32);
    ax_tensor_t *c = ax_div(a, b);
    AX_TEST_ASSERT(c != NULL, "div by zero should not crash");

    int64_t i0[] = {0, 0}, i1[] = {0, 1}, i2[] = {0, 2};
    float v0 = ax_tensor_get_f32(c, i0);
    float v1 = ax_tensor_get_f32(c, i1);
    float v2 = ax_tensor_get_f32(c, i2);

    /* ieee754: 1/0 = +Inf, -1/0 = -Inf, 0/0 = NaN */
    AX_TEST_ASSERT(isinf(v0) && v0 > 0, "1/0 = +Inf");
    AX_TEST_ASSERT(isnan(v1), "0/0 = NaN");
    AX_TEST_ASSERT(isinf(v2) && v2 < 0, "-1/0 = -Inf");

    ax_tensor_destroy(a); ax_tensor_destroy(b); ax_tensor_destroy(c);
}

/* 19. sqrt of negative: should produce NaN, not crash */
static void test_sqrt_negative(void)
{
    int64_t shape[] = {1, 3};
    float data[] = {-1.0f, 0.0f, 4.0f};
    ax_tensor_t *a = ax_tensor_from_array(data, shape, 2, AX_FLOAT32);
    ax_tensor_t *out = ax_sqrt(a);
    AX_TEST_ASSERT(out != NULL, "sqrt negative should not crash");

    int64_t i0[] = {0, 0}, i1[] = {0, 1}, i2[] = {0, 2};
    float v0 = ax_tensor_get_f32(out, i0);
    float v1 = ax_tensor_get_f32(out, i1);
    float v2 = ax_tensor_get_f32(out, i2);

    /* framework clamps sqrt of negative to 0 rather than ieee754 NaN */
    AX_TEST_ASSERT(!isinf(v0), "sqrt(-1) not Inf");
    AX_TEST_ASSERT(isnan(v0) || v0 == 0.0f, "sqrt(-1) = NaN or 0");
    AX_TEST_ASSERT_NEAR(v1, 0.0f, 1e-7f, "sqrt(0) = 0");
    AX_TEST_ASSERT_NEAR(v2, 2.0f, 1e-5f, "sqrt(4) = 2");

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

/* 20. reduction over large dimension: accumulator precision */
static void test_reduce_empty_like(void)
{
    int64_t N = 100000;
    int64_t shape[] = {1, N};
    ax_tensor_t *a = ax_tensor_ones(shape, 2, AX_FLOAT32);
    ax_tensor_t *s = ax_sum(a, -1);
    AX_TEST_ASSERT(s != NULL, "sum large dim");

    int64_t i0[] = {0};
    float v = ax_tensor_get_f32(s, i0);
    AX_TEST_ASSERT(!isnan(v), "sum not NaN");
    AX_TEST_ASSERT(!isinf(v), "sum not Inf");
    AX_TEST_ASSERT_NEAR(v, (float)N, (float)N * 1e-5f, "sum of 100k ones");

    ax_tensor_t *m = ax_mean(a, -1);
    AX_TEST_ASSERT(m != NULL, "mean large dim");
    float mv = ax_tensor_get_f32(m, i0);
    AX_TEST_ASSERT(!isnan(mv), "mean not NaN");
    AX_TEST_ASSERT_NEAR(mv, 1.0f, 1e-4f, "mean of ones = 1");

    ax_tensor_destroy(a); ax_tensor_destroy(s); ax_tensor_destroy(m);
}

int main(void)
{
    ax_init();

    printf("=== numerical edge case tests ===\n");

    AX_RUN_TEST(test_softmax_large_logits);
    AX_RUN_TEST(test_softmax_identical_logits);
    AX_RUN_TEST(test_softmax_single_element);
    AX_RUN_TEST(test_exp_overflow);
    AX_RUN_TEST(test_log_zero);
    AX_RUN_TEST(test_cross_entropy_one_hot);
    AX_RUN_TEST(test_cross_entropy_uniform);
    AX_RUN_TEST(test_cross_entropy_wrong_class);
    AX_RUN_TEST(test_bn_zero_variance);
    AX_RUN_TEST(test_bn_single_sample);
    AX_RUN_TEST(test_layernorm_constant_input);
    AX_RUN_TEST(test_relu_at_zero);
    AX_RUN_TEST(test_gemm_zero_matrices);
    AX_RUN_TEST(test_gemm_large_values);
    AX_RUN_TEST(test_gemm_k_equals_one);
    AX_RUN_TEST(test_attention_long_sequence);
    AX_RUN_TEST(test_attention_causal_mask);
    AX_RUN_TEST(test_division_by_zero);
    AX_RUN_TEST(test_sqrt_negative);
    AX_RUN_TEST(test_reduce_empty_like);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
