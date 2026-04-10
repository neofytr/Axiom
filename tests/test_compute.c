/* test_compute.c — tests for compute dispatch and cpu naive backend */

#include "test.h"
#include "axiom/axiom.h"

/* helpers */

/* create a 1d f32 tensor from a c array */
static ax_tensor_t *make_1d(float *data, int64_t n) {
    return ax_tensor_from_array(data, &n, 1, AX_FLOAT32);
}

/* create a 2d f32 tensor from a c array */
static ax_tensor_t *make_2d(float *data, int64_t rows, int64_t cols) {
    int64_t shape[] = {rows, cols};
    return ax_tensor_from_array(data, shape, 2, AX_FLOAT32);
}

/* element-wise binary ops */

static void test_add(void) {
    ax_tensor_t *a = make_1d((float[]){1, 2, 3}, 3);
    ax_tensor_t *b = make_1d((float[]){4, 5, 6}, 3);
    ax_tensor_t *out = ax_tensor_zeros(&(int64_t){3}, 1, AX_FLOAT32);

    ax_status_t s = ax_compute_add(a, b, out);
    AX_TEST_ASSERT_EQ(s, AX_OK, "add should succeed");

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 5.0f, 1e-7, "1+4=5");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i1), 7.0f, 1e-7, "2+5=7");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i2), 9.0f, 1e-7, "3+6=9");

    ax_tensor_destroy(a); ax_tensor_destroy(b); ax_tensor_destroy(out);
}

static void test_sub(void) {
    ax_tensor_t *a = make_1d((float[]){10, 20, 30}, 3);
    ax_tensor_t *b = make_1d((float[]){1, 2, 3}, 3);
    ax_tensor_t *out = ax_tensor_zeros(&(int64_t){3}, 1, AX_FLOAT32);

    ax_status_t s = ax_compute_sub(a, b, out);
    AX_TEST_ASSERT_EQ(s, AX_OK, "sub should succeed");

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 9.0f, 1e-7, "10-1=9");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i1), 18.0f, 1e-7, "20-2=18");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i2), 27.0f, 1e-7, "30-3=27");

    ax_tensor_destroy(a); ax_tensor_destroy(b); ax_tensor_destroy(out);
}

static void test_mul(void) {
    ax_tensor_t *a = make_1d((float[]){2, 3, 4}, 3);
    ax_tensor_t *b = make_1d((float[]){5, 6, 7}, 3);
    ax_tensor_t *out = ax_tensor_zeros(&(int64_t){3}, 1, AX_FLOAT32);

    ax_compute_mul(a, b, out);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 10.0f, 1e-7, "2*5=10");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i1), 18.0f, 1e-7, "3*6=18");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i2), 28.0f, 1e-7, "4*7=28");

    ax_tensor_destroy(a); ax_tensor_destroy(b); ax_tensor_destroy(out);
}

static void test_div(void) {
    ax_tensor_t *a = make_1d((float[]){10, 20, 30}, 3);
    ax_tensor_t *b = make_1d((float[]){2, 4, 5}, 3);
    ax_tensor_t *out = ax_tensor_zeros(&(int64_t){3}, 1, AX_FLOAT32);

    ax_compute_div(a, b, out);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 5.0f, 1e-7, "10/2=5");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i1), 5.0f, 1e-7, "20/4=5");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i2), 6.0f, 1e-7, "30/5=6");

    ax_tensor_destroy(a); ax_tensor_destroy(b); ax_tensor_destroy(out);
}

/* unary ops */

static void test_neg(void) {
    ax_tensor_t *a = make_1d((float[]){1, -2, 3}, 3);
    ax_tensor_t *out = ax_tensor_zeros(&(int64_t){3}, 1, AX_FLOAT32);

    ax_compute_neg(a, out);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), -1.0f, 1e-7, "neg(1)=-1");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i1), 2.0f, 1e-7, "neg(-2)=2");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i2), -3.0f, 1e-7, "neg(3)=-3");

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

static void test_exp_log(void) {
    ax_tensor_t *a = make_1d((float[]){0, 1, 2}, 3);
    ax_tensor_t *out = ax_tensor_zeros(&(int64_t){3}, 1, AX_FLOAT32);

    ax_compute_exp(a, out);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 1.0f, 1e-5, "exp(0)=1");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i1), 2.71828f, 1e-3, "exp(1)=e");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i2), 7.38906f, 1e-3, "exp(2)");

    /* log of the exp should give back original */
    ax_tensor_t *back = ax_tensor_zeros(&(int64_t){3}, 1, AX_FLOAT32);
    ax_compute_log(out, back);

    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(back, i0), 0.0f, 1e-5, "log(exp(0))=0");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(back, i1), 1.0f, 1e-5, "log(exp(1))=1");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(back, i2), 2.0f, 1e-5, "log(exp(2))=2");

    ax_tensor_destroy(a); ax_tensor_destroy(out); ax_tensor_destroy(back);
}

static void test_sqrt_square(void) {
    ax_tensor_t *a = make_1d((float[]){4, 9, 16}, 3);
    ax_tensor_t *out = ax_tensor_zeros(&(int64_t){3}, 1, AX_FLOAT32);

    ax_compute_sqrt(a, out);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 2.0f, 1e-5, "sqrt(4)=2");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i1), 3.0f, 1e-5, "sqrt(9)=3");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i2), 4.0f, 1e-5, "sqrt(16)=4");

    /* square the sqrt should give back original */
    ax_tensor_t *sq = ax_tensor_zeros(&(int64_t){3}, 1, AX_FLOAT32);
    ax_compute_square(out, sq);

    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(sq, i0), 4.0f, 1e-5, "square(sqrt(4))=4");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(sq, i1), 9.0f, 1e-5, "square(sqrt(9))=9");

    ax_tensor_destroy(a); ax_tensor_destroy(out); ax_tensor_destroy(sq);
}

/* scalar ops */

static void test_scalar_ops(void) {
    ax_tensor_t *a = make_1d((float[]){1, 2, 3}, 3);
    ax_tensor_t *out = ax_tensor_zeros(&(int64_t){3}, 1, AX_FLOAT32);

    ax_compute_add_scalar(a, 10.0, out);
    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 11.0f, 1e-7, "1+10=11");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i2), 13.0f, 1e-7, "3+10=13");

    ax_compute_mul_scalar(a, 3.0, out);
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 3.0f, 1e-7, "1*3=3");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i1), 6.0f, 1e-7, "2*3=6");

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

/* gemm */

static void test_gemm(void) {
    /* [2,3] @ [3,2] -> [2,2] */
    ax_tensor_t *a = make_2d((float[]){1, 2, 3, 4, 5, 6}, 2, 3);
    ax_tensor_t *b = make_2d((float[]){7, 8, 9, 10, 11, 12}, 3, 2);
    int64_t out_shape[] = {2, 2};
    ax_tensor_t *out = ax_tensor_zeros(out_shape, 2, AX_FLOAT32);

    ax_status_t s = ax_compute_gemm(a, b, out);
    AX_TEST_ASSERT_EQ(s, AX_OK, "gemm should succeed");

    /* row 0: [1,2,3] @ [[7,8],[9,10],[11,12]] = [58, 64] */
    /* row 1: [4,5,6] @ [[7,8],[9,10],[11,12]] = [139, 154] */
    int64_t i00[] = {0, 0}, i01[] = {0, 1}, i10[] = {1, 0}, i11[] = {1, 1};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i00), 58.0f, 1e-4, "[0,0]=58");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i01), 64.0f, 1e-4, "[0,1]=64");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i10), 139.0f, 1e-4, "[1,0]=139");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i11), 154.0f, 1e-4, "[1,1]=154");

    ax_tensor_destroy(a); ax_tensor_destroy(b); ax_tensor_destroy(out);
}

static void test_gemm_identity(void) {
    /* multiply by identity matrix */
    ax_tensor_t *a = make_2d((float[]){1, 2, 3, 4}, 2, 2);
    ax_tensor_t *eye = make_2d((float[]){1, 0, 0, 1}, 2, 2);
    int64_t shape[] = {2, 2};
    ax_tensor_t *out = ax_tensor_zeros(shape, 2, AX_FLOAT32);

    ax_compute_gemm(a, eye, out);

    int64_t i00[] = {0, 0}, i01[] = {0, 1}, i10[] = {1, 0}, i11[] = {1, 1};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i00), 1.0f, 1e-7, "a @ I = a [0,0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i01), 2.0f, 1e-7, "a @ I = a [0,1]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i10), 3.0f, 1e-7, "a @ I = a [1,0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i11), 4.0f, 1e-7, "a @ I = a [1,1]");

    ax_tensor_destroy(a); ax_tensor_destroy(eye); ax_tensor_destroy(out);
}

/* reduction ops */

static void test_sum_all(void) {
    ax_tensor_t *a = make_1d((float[]){1, 2, 3, 4}, 4);
    ax_tensor_t *out = ax_tensor_zeros(&(int64_t){1}, 1, AX_FLOAT32);

    ax_compute_sum(a, -1, out);

    int64_t i0[] = {0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 10.0f, 1e-5, "sum([1,2,3,4])=10");

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

static void test_sum_axis(void) {
    /* [[1,2,3],[4,5,6]] sum along axis 0 -> [5,7,9] */
    ax_tensor_t *a = make_2d((float[]){1, 2, 3, 4, 5, 6}, 2, 3);
    ax_tensor_t *out = ax_tensor_zeros(&(int64_t){3}, 1, AX_FLOAT32);

    ax_compute_sum(a, 0, out);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 5.0f, 1e-5, "sum axis 0 [0]=5");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i1), 7.0f, 1e-5, "sum axis 0 [1]=7");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i2), 9.0f, 1e-5, "sum axis 0 [2]=9");

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

static void test_mean(void) {
    ax_tensor_t *a = make_1d((float[]){2, 4, 6, 8}, 4);
    ax_tensor_t *out = ax_tensor_zeros(&(int64_t){1}, 1, AX_FLOAT32);

    ax_compute_mean(a, -1, out);

    int64_t i0[] = {0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 5.0f, 1e-5, "mean([2,4,6,8])=5");

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

static void test_max_min(void) {
    ax_tensor_t *a = make_1d((float[]){3, 1, 4, 1, 5, 9, 2, 6}, 8);
    ax_tensor_t *mx = ax_tensor_zeros(&(int64_t){1}, 1, AX_FLOAT32);
    ax_tensor_t *mn = ax_tensor_zeros(&(int64_t){1}, 1, AX_FLOAT32);

    ax_compute_max(a, -1, mx);
    ax_compute_min(a, -1, mn);

    int64_t i0[] = {0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(mx, i0), 9.0f, 1e-5, "max should be 9");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(mn, i0), 1.0f, 1e-5, "min should be 1");

    ax_tensor_destroy(a); ax_tensor_destroy(mx); ax_tensor_destroy(mn);
}

/* comparison ops */

static void test_comparisons(void) {
    ax_tensor_t *a = make_1d((float[]){1, 2, 3}, 3);
    ax_tensor_t *b = make_1d((float[]){1, 3, 2}, 3);
    ax_tensor_t *eq = ax_tensor_zeros(&(int64_t){3}, 1, AX_FLOAT32);
    ax_tensor_t *gt = ax_tensor_zeros(&(int64_t){3}, 1, AX_FLOAT32);

    ax_compute_equal(a, b, eq);
    ax_compute_greater(a, b, gt);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(eq, i0), 1.0f, 1e-7, "1==1 -> 1");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(eq, i1), 0.0f, 1e-7, "2==3 -> 0");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(gt, i1), 0.0f, 1e-7, "2>3 -> 0");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(gt, i2), 1.0f, 1e-7, "3>2 -> 1");

    ax_tensor_destroy(a); ax_tensor_destroy(b);
    ax_tensor_destroy(eq); ax_tensor_destroy(gt);
}

/* data movement */

static void test_fill_copy(void) {
    int64_t shape[] = {2, 3};
    ax_tensor_t *a = ax_tensor_create(shape, 2, AX_FLOAT32);
    ax_compute_fill(a, 7.0);

    int64_t i00[] = {0, 0}, i11[] = {1, 1};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a, i00), 7.0f, 1e-7, "fill should set all to 7");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(a, i11), 7.0f, 1e-7, "fill should set all to 7");

    ax_tensor_t *b = ax_tensor_zeros(shape, 2, AX_FLOAT32);
    ax_compute_copy(a, b);
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(b, i00), 7.0f, 1e-7, "copy should transfer values");

    ax_tensor_destroy(a); ax_tensor_destroy(b);
}

/* activations */

static void test_relu(void) {
    ax_tensor_t *a = make_1d((float[]){-2, -1, 0, 1, 2}, 5);
    ax_tensor_t *out = ax_tensor_zeros(&(int64_t){5}, 1, AX_FLOAT32);

    ax_compute_relu(a, out);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2}, i3[] = {3}, i4[] = {4};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 0.0f, 1e-7, "relu(-2)=0");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i1), 0.0f, 1e-7, "relu(-1)=0");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i2), 0.0f, 1e-7, "relu(0)=0");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i3), 1.0f, 1e-7, "relu(1)=1");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i4), 2.0f, 1e-7, "relu(2)=2");

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

static void test_sigmoid(void) {
    ax_tensor_t *a = make_1d((float[]){0, 100, -100}, 3);
    ax_tensor_t *out = ax_tensor_zeros(&(int64_t){3}, 1, AX_FLOAT32);

    ax_compute_sigmoid(a, out);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 0.5f, 1e-5, "sigmoid(0)=0.5");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i1), 1.0f, 1e-5, "sigmoid(100)~=1");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i2), 0.0f, 1e-5, "sigmoid(-100)~=0");

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

static void test_tanh(void) {
    ax_tensor_t *a = make_1d((float[]){0, 100, -100}, 3);
    ax_tensor_t *out = ax_tensor_zeros(&(int64_t){3}, 1, AX_FLOAT32);

    ax_compute_tanh(a, out);

    int64_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 0.0f, 1e-5, "tanh(0)=0");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i1), 1.0f, 1e-5, "tanh(100)~=1");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i2), -1.0f, 1e-5, "tanh(-100)~=-1");

    ax_tensor_destroy(a); ax_tensor_destroy(out);
}

/* backend dispatch tests */

static void test_backend_selection(void) {
    /* default is cpu_opt (optimized with naive fallback) */
    AX_TEST_ASSERT_EQ(ax_compute_get_backend(), AX_BACKEND_CPU_SIMD,
                       "default backend should be cpu_opt");

    const ax_backend_ops_t *ops = ax_compute_get_ops();
    AX_TEST_ASSERT(ops != NULL, "ops should not be null");
    AX_TEST_ASSERT(strcmp(ops->name, "cpu_opt") == 0, "backend name should be cpu_opt");

    /* can switch to naive */
    ax_status_t s1 = ax_compute_set_backend(AX_BACKEND_CPU_NAIVE);
    AX_TEST_ASSERT_EQ(s1, AX_OK, "switching to naive should succeed");
    AX_TEST_ASSERT_EQ(ax_compute_get_backend(), AX_BACKEND_CPU_NAIVE,
                       "backend should be cpu_naive after switch");

    /* trying to set unavailable backend should fail */
    ax_status_t s2 = ax_compute_set_backend(AX_BACKEND_CUDA);
    AX_TEST_ASSERT(s2 != AX_OK, "setting cuda backend should fail (not available)");

    /* should still be naive after failed switch */
    AX_TEST_ASSERT_EQ(ax_compute_get_backend(), AX_BACKEND_CPU_NAIVE,
                       "backend should still be cpu_naive after failed switch");

    /* switch back to opt */
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
}

int main(void) {
    ax_init();

    printf("=== compute tests ===\n");
    AX_RUN_TEST(test_add);
    AX_RUN_TEST(test_sub);
    AX_RUN_TEST(test_mul);
    AX_RUN_TEST(test_div);
    AX_RUN_TEST(test_neg);
    AX_RUN_TEST(test_exp_log);
    AX_RUN_TEST(test_sqrt_square);
    AX_RUN_TEST(test_scalar_ops);
    AX_RUN_TEST(test_gemm);
    AX_RUN_TEST(test_gemm_identity);
    AX_RUN_TEST(test_sum_all);
    AX_RUN_TEST(test_sum_axis);
    AX_RUN_TEST(test_mean);
    AX_RUN_TEST(test_max_min);
    AX_RUN_TEST(test_comparisons);
    AX_RUN_TEST(test_fill_copy);
    AX_RUN_TEST(test_relu);
    AX_RUN_TEST(test_sigmoid);
    AX_RUN_TEST(test_tanh);
    AX_RUN_TEST(test_backend_selection);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
