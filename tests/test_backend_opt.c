/* test_backend_opt.c — cross-check every optimized backend path against naive.
   every test runs the same op on the same data through both backends and compares
   outputs element-by-element.  this catches regressions in simd paths, hmax/hmin,
   the 4-accumulator unrolled reductions, and the axis-1 2D fast path. */

#include "test.h"
#include "axiom/axiom.h"
#include "axiom/compute.h"
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* helpers                                                              */
/* ------------------------------------------------------------------ */

/* fill a tensor with deterministic pseudo-random floats */
static void fill_deterministic(ax_tensor_t *t, float bias)
{
    float *d = (float *)t->storage->data;
    int64_t n = (int64_t)(t->storage->size_bytes / sizeof(float));
    for (int64_t i = 0; i < n; i++)
        d[i] = bias + (float)(i % 97) * 0.01f - 0.48f;
}

/* check two float tensors match within tol; report on first mismatch */
static void check_close(ax_tensor_t *naive_out, ax_tensor_t *opt_out,
                         float tol, const char *label)
{
    float *nd = (float *)naive_out->storage->data;
    float *od = (float *)opt_out->storage->data;
    int64_t n  = (int64_t)(naive_out->storage->size_bytes / sizeof(float));
    int64_t on = (int64_t)(opt_out->storage->size_bytes / sizeof(float));
    AX_TEST_ASSERT(n == on, label);

    for (int64_t i = 0; i < n; i++) {
        float diff = nd[i] - od[i];
        if (diff < 0) diff = -diff;
        if (diff > tol) {
            printf("  MISMATCH at [%lld]: naive=%.8f  opt=%.8f  diff=%.2e\n",
                   (long long)i, (double)nd[i], (double)od[i], (double)diff);
            AX_TEST_ASSERT(0, label);
            return;
        }
    }
    AX_TEST_ASSERT(1, label);
}

/* allocate a pair of equal-shaped output tensors (naive, opt) */
static void alloc_pair(int64_t *shape, int ndim,
                        ax_tensor_t **naive_out, ax_tensor_t **opt_out)
{
    *naive_out = ax_tensor_zeros(shape, ndim, AX_FLOAT32);
    *opt_out   = ax_tensor_zeros(shape, ndim, AX_FLOAT32);
}

/* ------------------------------------------------------------------ */
/* element-wise binary ops — large arrays to exercise SIMD main loop  */
/* ------------------------------------------------------------------ */

/* sizes: one that exercises the 4×VF32_WIDTH unroll + cleanup tail */
#define LARGE_N  ((int64_t)4096)   /* 16KB, well inside L1 */
#define HUGE_N   ((int64_t)65536)  /* 256KB, stresses L2 */

#define DEFINE_BINOP_TEST(name, compute_fn) \
static void test_opt_##name(void) \
{ \
    int64_t shape[] = {LARGE_N}; \
    ax_tensor_t *a = ax_tensor_zeros(shape, 1, AX_FLOAT32); \
    ax_tensor_t *b = ax_tensor_zeros(shape, 1, AX_FLOAT32); \
    ax_tensor_t *naive_out, *opt_out; \
    alloc_pair(shape, 1, &naive_out, &opt_out); \
    fill_deterministic(a, 0.1f); \
    fill_deterministic(b, 0.3f); \
    ax_compute_set_backend(AX_BACKEND_CPU_NAIVE); \
    compute_fn(a, b, naive_out); \
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD); \
    compute_fn(a, b, opt_out); \
    check_close(naive_out, opt_out, 1e-6f, #name " large array"); \
    ax_tensor_destroy(a); ax_tensor_destroy(b); \
    ax_tensor_destroy(naive_out); ax_tensor_destroy(opt_out); \
}

DEFINE_BINOP_TEST(add,     ax_compute_add)
DEFINE_BINOP_TEST(sub,     ax_compute_sub)
DEFINE_BINOP_TEST(mul,     ax_compute_mul)
/* div uses positive-only values to avoid div-by-zero */

static void test_opt_div(void)
{
    int64_t shape[] = {LARGE_N};
    ax_tensor_t *a = ax_tensor_zeros(shape, 1, AX_FLOAT32);
    ax_tensor_t *b = ax_tensor_zeros(shape, 1, AX_FLOAT32);
    ax_tensor_t *naive_out, *opt_out;
    alloc_pair(shape, 1, &naive_out, &opt_out);

    float *ad = (float *)a->storage->data;
    float *bd = (float *)b->storage->data;
    for (int64_t i = 0; i < LARGE_N; i++) {
        ad[i] = 1.0f + (float)(i % 50) * 0.01f;
        bd[i] = 0.5f + (float)(i % 47) * 0.01f;
    }

    ax_compute_set_backend(AX_BACKEND_CPU_NAIVE);
    ax_compute_div(a, b, naive_out);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
    ax_compute_div(a, b, opt_out);
    check_close(naive_out, opt_out, 1e-5f, "div large array");

    ax_tensor_destroy(a); ax_tensor_destroy(b);
    ax_tensor_destroy(naive_out); ax_tensor_destroy(opt_out);
}

/* ------------------------------------------------------------------ */
/* unary ops                                                            */
/* ------------------------------------------------------------------ */

#define DEFINE_UNOP_TEST(name, compute_fn) \
static void test_opt_##name(void) \
{ \
    int64_t shape[] = {LARGE_N}; \
    ax_tensor_t *in = ax_tensor_zeros(shape, 1, AX_FLOAT32); \
    ax_tensor_t *naive_out, *opt_out; \
    alloc_pair(shape, 1, &naive_out, &opt_out); \
    fill_deterministic(in, 0.0f); \
    ax_compute_set_backend(AX_BACKEND_CPU_NAIVE); \
    compute_fn(in, naive_out); \
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD); \
    compute_fn(in, opt_out); \
    check_close(naive_out, opt_out, 1e-4f, #name " large array"); \
    ax_tensor_destroy(in); \
    ax_tensor_destroy(naive_out); ax_tensor_destroy(opt_out); \
}

DEFINE_UNOP_TEST(neg,     ax_compute_neg)
DEFINE_UNOP_TEST(abs_op,  ax_compute_abs)
DEFINE_UNOP_TEST(relu,    ax_compute_relu)
DEFINE_UNOP_TEST(sigmoid, ax_compute_sigmoid)
DEFINE_UNOP_TEST(square,  ax_compute_square)

/* exp/log/sqrt need positive inputs */
static void test_opt_exp(void)
{
    int64_t shape[] = {LARGE_N};
    ax_tensor_t *in = ax_tensor_zeros(shape, 1, AX_FLOAT32);
    ax_tensor_t *naive_out, *opt_out;
    alloc_pair(shape, 1, &naive_out, &opt_out);
    float *d = (float *)in->storage->data;
    for (int64_t i = 0; i < LARGE_N; i++)
        d[i] = (float)(i % 50) * 0.04f - 1.0f; /* range [-1, 1] */

    ax_compute_set_backend(AX_BACKEND_CPU_NAIVE);  ax_compute_exp(in, naive_out);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);   ax_compute_exp(in, opt_out);
    /* fast exp has ~1e-4 relative error */
    check_close(naive_out, opt_out, 1e-3f, "exp large array");
    ax_tensor_destroy(in); ax_tensor_destroy(naive_out); ax_tensor_destroy(opt_out);
}

static void test_opt_log_sqrt(void)
{
    int64_t shape[] = {LARGE_N};
    ax_tensor_t *in = ax_tensor_zeros(shape, 1, AX_FLOAT32);
    ax_tensor_t *naive_log, *opt_log, *naive_sqrt, *opt_sqrt;
    alloc_pair(shape, 1, &naive_log, &opt_log);
    alloc_pair(shape, 1, &naive_sqrt, &opt_sqrt);
    float *d = (float *)in->storage->data;
    for (int64_t i = 0; i < LARGE_N; i++)
        d[i] = 0.01f + (float)(i % 100) * 0.01f; /* (0, 1] */

    ax_compute_set_backend(AX_BACKEND_CPU_NAIVE);
    ax_compute_log(in, naive_log); ax_compute_sqrt(in, naive_sqrt);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
    ax_compute_log(in, opt_log);   ax_compute_sqrt(in, opt_sqrt);
    check_close(naive_log,  opt_log,  1e-5f, "log large array");
    check_close(naive_sqrt, opt_sqrt, 1e-5f, "sqrt large array");
    ax_tensor_destroy(in);
    ax_tensor_destroy(naive_log); ax_tensor_destroy(opt_log);
    ax_tensor_destroy(naive_sqrt); ax_tensor_destroy(opt_sqrt);
}

/* ------------------------------------------------------------------ */
/* scalar ops — previously unvectorized, now SIMD                      */
/* ------------------------------------------------------------------ */

static void test_opt_add_scalar(void)
{
    int64_t shape[] = {LARGE_N};
    ax_tensor_t *in = ax_tensor_zeros(shape, 1, AX_FLOAT32);
    ax_tensor_t *naive_out, *opt_out;
    alloc_pair(shape, 1, &naive_out, &opt_out);
    fill_deterministic(in, 0.0f);

    ax_compute_set_backend(AX_BACKEND_CPU_NAIVE); ax_compute_add_scalar(in, 3.14, naive_out);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);  ax_compute_add_scalar(in, 3.14, opt_out);
    check_close(naive_out, opt_out, 1e-6f, "add_scalar large array");
    ax_tensor_destroy(in); ax_tensor_destroy(naive_out); ax_tensor_destroy(opt_out);
}

static void test_opt_mul_scalar(void)
{
    int64_t shape[] = {LARGE_N};
    ax_tensor_t *in = ax_tensor_zeros(shape, 1, AX_FLOAT32);
    ax_tensor_t *naive_out, *opt_out;
    alloc_pair(shape, 1, &naive_out, &opt_out);
    fill_deterministic(in, 0.0f);

    ax_compute_set_backend(AX_BACKEND_CPU_NAIVE); ax_compute_mul_scalar(in, 2.71828, naive_out);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);  ax_compute_mul_scalar(in, 2.71828, opt_out);
    check_close(naive_out, opt_out, 1e-6f, "mul_scalar large array");
    ax_tensor_destroy(in); ax_tensor_destroy(naive_out); ax_tensor_destroy(opt_out);
}

/* ------------------------------------------------------------------ */
/* fill — non-zero path is now SIMD                                     */
/* ------------------------------------------------------------------ */

static void test_opt_fill(void)
{
    int64_t shape[] = {LARGE_N};
    ax_tensor_t *naive_out, *opt_out;
    alloc_pair(shape, 1, &naive_out, &opt_out);

    /* zero fill */
    ax_compute_set_backend(AX_BACKEND_CPU_NAIVE); ax_compute_fill(naive_out, 0.0);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);  ax_compute_fill(opt_out,   0.0);
    check_close(naive_out, opt_out, 0.0f, "fill(0) large array");

    /* non-zero fill — the newly vectorized path */
    ax_compute_set_backend(AX_BACKEND_CPU_NAIVE); ax_compute_fill(naive_out, -1.5);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);  ax_compute_fill(opt_out,   -1.5);
    check_close(naive_out, opt_out, 0.0f, "fill(-1.5) large array");

    ax_tensor_destroy(naive_out); ax_tensor_destroy(opt_out);
}

/* ------------------------------------------------------------------ */
/* reductions — 4-accumulator unroll + hsum/hmax/hmin                  */
/* ------------------------------------------------------------------ */

static void test_opt_sum_full(void)
{
    /* HUGE_N: exercises multiple rounds of the 4-accumulator unroll */
    int64_t shape[] = {HUGE_N};
    ax_tensor_t *in = ax_tensor_zeros(shape, 1, AX_FLOAT32);
    ax_tensor_t *naive_out, *opt_out;
    alloc_pair(&(int64_t){1}, 1, &naive_out, &opt_out);
    fill_deterministic(in, 0.0f);

    ax_compute_set_backend(AX_BACKEND_CPU_NAIVE); ax_compute_sum(in, -1, naive_out);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);  ax_compute_sum(in, -1, opt_out);
    /* larger tolerance because double vs float accumulation may differ slightly */
    check_close(naive_out, opt_out, 0.5f, "sum full large array");
    ax_tensor_destroy(in); ax_tensor_destroy(naive_out); ax_tensor_destroy(opt_out);
}

static void test_opt_mean_full(void)
{
    int64_t shape[] = {LARGE_N};
    ax_tensor_t *in = ax_tensor_zeros(shape, 1, AX_FLOAT32);
    ax_tensor_t *naive_out, *opt_out;
    alloc_pair(&(int64_t){1}, 1, &naive_out, &opt_out);
    fill_deterministic(in, 0.0f);

    ax_compute_set_backend(AX_BACKEND_CPU_NAIVE); ax_compute_mean(in, -1, naive_out);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);  ax_compute_mean(in, -1, opt_out);
    check_close(naive_out, opt_out, 1e-4f, "mean full large array");
    ax_tensor_destroy(in); ax_tensor_destroy(naive_out); ax_tensor_destroy(opt_out);
}

static void test_opt_max_min(void)
{
    /* HUGE_N: exercises hmax/hmin properly */
    int64_t shape[] = {HUGE_N};
    ax_tensor_t *in = ax_tensor_zeros(shape, 1, AX_FLOAT32);
    ax_tensor_t *naive_max, *opt_max, *naive_min, *opt_min;
    alloc_pair(&(int64_t){1}, 1, &naive_max, &opt_max);
    alloc_pair(&(int64_t){1}, 1, &naive_min, &opt_min);
    fill_deterministic(in, 0.0f);
    /* insert a known max and min at interior positions */
    ((float *)in->storage->data)[1000]  =  999.0f;
    ((float *)in->storage->data)[50000] = -999.0f;

    ax_compute_set_backend(AX_BACKEND_CPU_NAIVE);
    ax_compute_max(in, -1, naive_max); ax_compute_min(in, -1, naive_min);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
    ax_compute_max(in, -1, opt_max);   ax_compute_min(in, -1, opt_min);

    check_close(naive_max, opt_max, 1e-6f, "max full large array");
    check_close(naive_min, opt_min, 1e-6f, "min full large array");

    /* also verify the actual values */
    int64_t i0[] = {0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(opt_max, i0),  999.0f, 1e-5f, "max = 999");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(opt_min, i0), -999.0f, 1e-5f, "min = -999");

    ax_tensor_destroy(in);
    ax_tensor_destroy(naive_max); ax_tensor_destroy(opt_max);
    ax_tensor_destroy(naive_min); ax_tensor_destroy(opt_min);
}

/* ------------------------------------------------------------------ */
/* axis-1 sum/mean on 2D tensors — the new fast path                   */
/* ------------------------------------------------------------------ */

static void test_opt_sum_axis1_2d(void)
{
    /* [rows, cols]: cols exercises SIMD within each row */
    const int64_t rows = 64, cols = 512;
    int64_t in_shape[]  = {rows, cols};
    int64_t out_shape[] = {rows};

    ax_tensor_t *in = ax_tensor_zeros(in_shape, 2, AX_FLOAT32);
    ax_tensor_t *naive_out, *opt_out;
    alloc_pair(out_shape, 1, &naive_out, &opt_out);
    fill_deterministic(in, 0.0f);

    ax_compute_set_backend(AX_BACKEND_CPU_NAIVE); ax_compute_sum(in, 1, naive_out);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);  ax_compute_sum(in, 1, opt_out);
    check_close(naive_out, opt_out, 0.1f, "sum axis=1 2D large");

    ax_tensor_destroy(in); ax_tensor_destroy(naive_out); ax_tensor_destroy(opt_out);
}

static void test_opt_mean_axis1_2d(void)
{
    const int64_t rows = 64, cols = 512;
    int64_t in_shape[]  = {rows, cols};
    int64_t out_shape[] = {rows};

    ax_tensor_t *in = ax_tensor_zeros(in_shape, 2, AX_FLOAT32);
    ax_tensor_t *naive_out, *opt_out;
    alloc_pair(out_shape, 1, &naive_out, &opt_out);
    fill_deterministic(in, 0.0f);

    ax_compute_set_backend(AX_BACKEND_CPU_NAIVE); ax_compute_mean(in, 1, naive_out);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);  ax_compute_mean(in, 1, opt_out);
    check_close(naive_out, opt_out, 1e-4f, "mean axis=1 2D large");

    ax_tensor_destroy(in); ax_tensor_destroy(naive_out); ax_tensor_destroy(opt_out);
}

/* edge case: axis-1 on a single-row tensor */
static void test_opt_sum_axis1_single_row(void)
{
    const int64_t rows = 1, cols = 33; /* 33 = 4*VF32_WIDTH + 1 for scalar tail */
    int64_t in_shape[]  = {rows, cols};
    int64_t out_shape[] = {rows};

    ax_tensor_t *in = ax_tensor_zeros(in_shape, 2, AX_FLOAT32);
    float *d = (float *)in->storage->data;
    for (int64_t i = 0; i < cols; i++) d[i] = (float)(i + 1);
    /* expected sum: 1+2+...+33 = 33*34/2 = 561 */

    ax_tensor_t *naive_out, *opt_out;
    alloc_pair(out_shape, 1, &naive_out, &opt_out);

    ax_compute_set_backend(AX_BACKEND_CPU_NAIVE); ax_compute_sum(in, 1, naive_out);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);  ax_compute_sum(in, 1, opt_out);
    check_close(naive_out, opt_out, 1e-4f, "sum axis=1 single-row");

    int64_t i0[] = {0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(opt_out, i0), 561.0f, 0.5f, "sum=561");

    ax_tensor_destroy(in); ax_tensor_destroy(naive_out); ax_tensor_destroy(opt_out);
}

/* edge case: axis-1 with cols < VF32_WIDTH (all scalar tail) */
static void test_opt_sum_axis1_small_cols(void)
{
    const int64_t rows = 8, cols = 3;
    int64_t in_shape[]  = {rows, cols};
    int64_t out_shape[] = {rows};

    ax_tensor_t *in = ax_tensor_zeros(in_shape, 2, AX_FLOAT32);
    fill_deterministic(in, 0.0f);

    ax_tensor_t *naive_out, *opt_out;
    alloc_pair(out_shape, 1, &naive_out, &opt_out);

    ax_compute_set_backend(AX_BACKEND_CPU_NAIVE); ax_compute_sum(in, 1, naive_out);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);  ax_compute_sum(in, 1, opt_out);
    check_close(naive_out, opt_out, 1e-5f, "sum axis=1 small-cols");

    ax_tensor_destroy(in); ax_tensor_destroy(naive_out); ax_tensor_destroy(opt_out);
}

/* ------------------------------------------------------------------ */
/* comparisons — SIMD cmpeq/cmpgt paths                                */
/* ------------------------------------------------------------------ */

static void test_opt_comparisons_large(void)
{
    int64_t shape[] = {LARGE_N};
    ax_tensor_t *a = ax_tensor_zeros(shape, 1, AX_FLOAT32);
    ax_tensor_t *b = ax_tensor_zeros(shape, 1, AX_FLOAT32);
    ax_tensor_t *naive_eq, *opt_eq, *naive_gt, *opt_gt;
    alloc_pair(shape, 1, &naive_eq, &opt_eq);
    alloc_pair(shape, 1, &naive_gt, &opt_gt);

    float *ad = (float *)a->storage->data;
    float *bd = (float *)b->storage->data;
    for (int64_t i = 0; i < LARGE_N; i++) {
        ad[i] = (float)(i % 10);
        bd[i] = (float)((i + 3) % 10);
    }

    ax_compute_set_backend(AX_BACKEND_CPU_NAIVE);
    ax_compute_equal(a, b, naive_eq); ax_compute_greater(a, b, naive_gt);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
    ax_compute_equal(a, b, opt_eq);   ax_compute_greater(a, b, opt_gt);

    check_close(naive_eq, opt_eq, 0.0f, "equal large array");
    check_close(naive_gt, opt_gt, 0.0f, "greater large array");

    ax_tensor_destroy(a); ax_tensor_destroy(b);
    ax_tensor_destroy(naive_eq); ax_tensor_destroy(opt_eq);
    ax_tensor_destroy(naive_gt); ax_tensor_destroy(opt_gt);
}

/* ------------------------------------------------------------------ */
/* end-to-end: train a small network, check loss converges on opt      */
/* ------------------------------------------------------------------ */

static void test_opt_training_converges(void)
{
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
    ax_set_seed(42);

    /* xor dataset */
    float xy[][2] = {{0,0},{0,1},{1,0},{1,1}};
    float lbl[]   = {0,1,1,0};
    int64_t xs[] = {4, 2};
    int64_t ys[] = {4, 1};
    ax_tensor_t *x = ax_tensor_create(xs, 2, AX_FLOAT32);
    ax_tensor_t *y = ax_tensor_zeros(ys, 2, AX_FLOAT32);
    memcpy(x->storage->data, xy, sizeof(xy));
    memcpy(y->storage->data, lbl, sizeof(lbl));

    ax_layer_t *model = ax_sequential_create();
    ax_sequential_add(model, ax_dense_create(2, 8, true));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_dense_create(8, 1, true));

    ax_tensor_t *params[16];
    int np = ax_layer_get_params(model, params, 16);
    ax_optimizer_t *opt = ax_adamw_create(params, np, 1e-2f, 0.9f, 0.999f, 1e-8f, 0.0f);

    float last_loss = 1.0f;
    for (int ep = 0; ep < 500; ep++) {
        ax_layer_train(model);
        ax_enable_grad();
        ax_tensor_t *pred = ax_layer_forward(model, x);
        ax_tensor_t *loss = ax_mse_loss(pred, y);
        last_loss = ((float *)loss->storage->data)[0];
        ax_optimizer_zero_grad(opt);
        ax_backward(loss);
        ax_optimizer_step(opt);
        ax_graph_cleanup(loss);
        ax_tensor_destroy(loss);
    }

    /* xor is solvable — loss should be well under 0.1 in 500 epochs */
    AX_TEST_ASSERT(last_loss < 0.1f, "opt backend training converges on xor");

    ax_optimizer_destroy(opt);
    ax_layer_destroy(model);
    ax_tensor_destroy(x);
    ax_tensor_destroy(y);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
}

/* ------------------------------------------------------------------ */

int main(void)
{
    ax_init();

    printf("=== optimized backend cross-check tests ===\n");

    /* element-wise binary */
    AX_RUN_TEST(test_opt_add);
    AX_RUN_TEST(test_opt_sub);
    AX_RUN_TEST(test_opt_mul);
    AX_RUN_TEST(test_opt_div);

    /* unary */
    AX_RUN_TEST(test_opt_neg);
    AX_RUN_TEST(test_opt_abs_op);
    AX_RUN_TEST(test_opt_relu);
    AX_RUN_TEST(test_opt_sigmoid);
    AX_RUN_TEST(test_opt_square);
    AX_RUN_TEST(test_opt_exp);
    AX_RUN_TEST(test_opt_log_sqrt);

    /* scalar ops — newly vectorized */
    AX_RUN_TEST(test_opt_add_scalar);
    AX_RUN_TEST(test_opt_mul_scalar);

    /* fill — non-zero path newly vectorized */
    AX_RUN_TEST(test_opt_fill);

    /* reductions — 4-accumulator unroll + hsum */
    AX_RUN_TEST(test_opt_sum_full);
    AX_RUN_TEST(test_opt_mean_full);

    /* max/min — hmax/hmin */
    AX_RUN_TEST(test_opt_max_min);

    /* axis-1 2D fast path */
    AX_RUN_TEST(test_opt_sum_axis1_2d);
    AX_RUN_TEST(test_opt_mean_axis1_2d);
    AX_RUN_TEST(test_opt_sum_axis1_single_row);
    AX_RUN_TEST(test_opt_sum_axis1_small_cols);

    /* comparisons — SIMD paths */
    AX_RUN_TEST(test_opt_comparisons_large);

    /* end-to-end sanity */
    AX_RUN_TEST(test_opt_training_converges);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
