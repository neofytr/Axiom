/* test_property.c — property-based testing for tensor + compute APIs.
   M.4 of the production plan. complements test_backend_opt.c (which
   uses fixed shapes) with random-shape + random-data fuzzing of the
   same opt-vs-naive comparison harness. catches regressions in:
     - simd tail-loop handling at non-power-of-2 sizes
     - unrolled accumulator paths at small sizes
     - axis-aware reductions on mixed strides
     - gemm tile-edge handling at non-multiple-of-MR/NR sizes

   determinism: every case is keyed on (op_id, case_idx) → seed, so a
   failing case logs the seed and the user can re-run only that one.
   unlike random fuzzing with wall-time seeds, this is fully
   reproducible across runs and machines.

   the cpu_naive backend is the reference (correct by construction —
   straightforward triple-loop GEMM, scalar reductions, etc.). the
   cpu_opt backend is what we're testing. when they disagree past the
   per-op tolerance, opt is buggy. */

#include "test.h"
#include "axiom/axiom.h"
#include "axiom/compute.h"
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

/* xorshift64 — deterministic across platforms */
static uint64_t prng_state = 0;
static void     prng_seed(uint64_t s)         { prng_state = s ? s : 0xdeadbeefULL; }
static uint64_t prng_next(void) {
    uint64_t x = prng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    prng_state = x;
    return x;
}
static int      prng_int(int lo, int hi)      { return lo + (int)(prng_next() % (uint64_t)(hi - lo + 1)); }
static float    prng_float(float lo, float hi) {
    uint32_t r = (uint32_t)(prng_next() & 0xFFFFFF);
    float u = (float)r / 16777216.0f;
    return lo + u * (hi - lo);
}

typedef struct { int64_t shape[4]; int ndim; } prop_shape_t;

static prop_shape_t random_shape(int min_dim, int max_dim, int min_size, int max_size) {
    prop_shape_t s;
    s.ndim = prng_int(min_dim, max_dim);
    for (int i = 0; i < s.ndim; i++) s.shape[i] = prng_int(min_size, max_size);
    return s;
}

/* fill with seeded random floats in [-2, 2]. range chosen so
   exp/sigmoid/tanh stay well-behaved (no overflow, distinguishable). */
static void prop_fill(ax_tensor_t *t) {
    float *d = (float *)t->storage->data;
    int64_t n = (int64_t)(t->storage->size_bytes / sizeof(float));
    for (int64_t i = 0; i < n; i++) d[i] = prng_float(-2.0f, 2.0f);
}

/* fill with non-negative values (for log/sqrt domain) */
static void prop_fill_pos(ax_tensor_t *t) {
    float *d = (float *)t->storage->data;
    int64_t n = (int64_t)(t->storage->size_bytes / sizeof(float));
    for (int64_t i = 0; i < n; i++) d[i] = prng_float(0.01f, 4.0f);
}

/* compute logical element count from shape (n_logical), independent of
   storage padding/alignment. tensors with the same shape but different
   underlying storage allocation strategies still compare correctly. */
static int64_t prop_n_elems(const ax_tensor_t *t) {
    int64_t n = 1;
    for (int d = 0; d < t->ndim; d++) n *= t->shape[d];
    return n;
}

/* element-wise compare, tolerant; prints first mismatch with index */
static int prop_close(const ax_tensor_t *naive, const ax_tensor_t *opt,
                       float atol, float rtol, uint64_t seed,
                       const char *op_name)
{
    int64_t n_n = prop_n_elems(naive);
    int64_t n_o = prop_n_elems(opt);
    if (n_n != n_o) {
        printf("  PROP[%s seed=%llu]: shape elem count mismatch naive=%lld opt=%lld\n",
               op_name, (unsigned long long)seed,
               (long long)n_n, (long long)n_o);
        return 0;
    }
    const float *a = (const float *)naive->storage->data + naive->offset;
    const float *b = (const float *)opt->storage->data + opt->offset;
    int64_t n = n_n;
    for (int64_t i = 0; i < n; i++) {
        float d = a[i] - b[i];
        if (d < 0) d = -d;
        float ref = fabsf(a[i]);
        float bound = atol + rtol * ref;
        if (d > bound) {
            printf("  PROP[%s seed=%llu]: mismatch at [%lld]: "
                   "naive=%.6f opt=%.6f diff=%.2e tol=%.2e\n",
                   op_name, (unsigned long long)seed,
                   (long long)i, (double)a[i], (double)b[i],
                   (double)d, (double)bound);
            return 0;
        }
    }
    return 1;
}

typedef ax_status_t (*binop_fn_t)(const ax_tensor_t *, const ax_tensor_t *, ax_tensor_t *);
typedef ax_status_t (*unop_fn_t) (const ax_tensor_t *, ax_tensor_t *);

static int run_binop_property(const char *name, binop_fn_t fn,
                                int n_cases, float atol, float rtol)
{
    int passed = 0;
    for (int i = 0; i < n_cases; i++) {
        uint64_t seed = ((uint64_t)0xB1A0DA11ULL << 32) ^ ((uint64_t)i + 1);
        prng_seed(seed);
        prop_shape_t s = random_shape(1, 3, 1, 64);
        ax_tensor_t *a   = ax_tensor_zeros(s.shape, s.ndim, AX_FLOAT32);
        ax_tensor_t *b   = ax_tensor_zeros(s.shape, s.ndim, AX_FLOAT32);
        ax_tensor_t *out_n = ax_tensor_zeros(s.shape, s.ndim, AX_FLOAT32);
        ax_tensor_t *out_o = ax_tensor_zeros(s.shape, s.ndim, AX_FLOAT32);
        prop_fill(a); prop_fill(b);

        ax_compute_set_backend(AX_BACKEND_CPU_NAIVE);
        if (fn(a, b, out_n) != AX_OK) goto fail;
        ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
        if (fn(a, b, out_o) != AX_OK) goto fail;
        if (prop_close(out_n, out_o, atol, rtol, seed, name)) passed++;
        goto next;
    fail:
        printf("  PROP[%s seed=%llu]: backend op returned error\n",
               name, (unsigned long long)seed);
    next:
        ax_tensor_destroy(a); ax_tensor_destroy(b);
        ax_tensor_destroy(out_n); ax_tensor_destroy(out_o);
    }
    return passed;
}

static int run_unop_property(const char *name, unop_fn_t fn,
                               int n_cases, int positive_only,
                               float atol, float rtol)
{
    int passed = 0;
    for (int i = 0; i < n_cases; i++) {
        uint64_t seed = ((uint64_t)0xB1A0DA11ULL << 32) ^ ((uint64_t)i + 1)
                        ^ ((uint64_t)name[0] * 7919ULL);
        prng_seed(seed);
        prop_shape_t s = random_shape(1, 3, 1, 64);
        ax_tensor_t *a   = ax_tensor_zeros(s.shape, s.ndim, AX_FLOAT32);
        ax_tensor_t *out_n = ax_tensor_zeros(s.shape, s.ndim, AX_FLOAT32);
        ax_tensor_t *out_o = ax_tensor_zeros(s.shape, s.ndim, AX_FLOAT32);
        if (positive_only) prop_fill_pos(a); else prop_fill(a);

        ax_compute_set_backend(AX_BACKEND_CPU_NAIVE);
        if (fn(a, out_n) != AX_OK) goto fail;
        ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
        if (fn(a, out_o) != AX_OK) goto fail;
        if (prop_close(out_n, out_o, atol, rtol, seed, name)) passed++;
        goto next;
    fail:
        printf("  PROP[%s seed=%llu]: backend op returned error\n",
               name, (unsigned long long)seed);
    next:
        ax_tensor_destroy(a);
        ax_tensor_destroy(out_n); ax_tensor_destroy(out_o);
    }
    return passed;
}

/* gemm property: random M, N, K, contiguous fp32. exercises tile-edge
   cases when M/N aren't multiples of MR/NR. */
static int run_gemm_property(int n_cases) {
    int passed = 0;
    for (int i = 0; i < n_cases; i++) {
        uint64_t seed = ((uint64_t)0x6E6D6D6DULL << 32) ^ ((uint64_t)i + 1);
        prng_seed(seed);
        int M = prng_int(1, 96);
        int N = prng_int(1, 96);
        int K = prng_int(1, 96);
        int64_t a_shape[2] = {M, K};
        int64_t b_shape[2] = {K, N};
        int64_t c_shape[2] = {M, N};
        ax_tensor_t *a = ax_tensor_zeros(a_shape, 2, AX_FLOAT32);
        ax_tensor_t *b = ax_tensor_zeros(b_shape, 2, AX_FLOAT32);
        ax_tensor_t *c_n = ax_tensor_zeros(c_shape, 2, AX_FLOAT32);
        ax_tensor_t *c_o = ax_tensor_zeros(c_shape, 2, AX_FLOAT32);
        prop_fill(a); prop_fill(b);

        ax_compute_set_backend(AX_BACKEND_CPU_NAIVE);
        if (ax_compute_gemm(a, b, c_n) != AX_OK) goto fail;
        ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
        if (ax_compute_gemm(a, b, c_o) != AX_OK) goto fail;
        /* gemm summation order differs across backends; needs looser tol */
        if (prop_close(c_n, c_o, 1e-3f, 1e-4f, seed, "gemm")) passed++;
        goto next;
    fail:
        printf("  PROP[gemm seed=%llu M=%d N=%d K=%d]: backend op returned error\n",
               (unsigned long long)seed, M, N, K);
    next:
        ax_tensor_destroy(a); ax_tensor_destroy(b);
        ax_tensor_destroy(c_n); ax_tensor_destroy(c_o);
    }
    return passed;
}

/* reduce_sum property: random shape + axis (or -1 for full reduction). */
static int run_reduce_sum_property(int n_cases) {
    int passed = 0;
    for (int i = 0; i < n_cases; i++) {
        uint64_t seed = ((uint64_t)0x52454442ULL << 32) ^ ((uint64_t)i + 1);
        prng_seed(seed);
        prop_shape_t s = random_shape(1, 3, 2, 32);
        int axis = prng_int(-1, s.ndim - 1);
        int64_t o_shape[3]; int o_ndim = 0;
        if (axis < 0) {
            o_shape[0] = 1; o_ndim = 1;
        } else {
            for (int d = 0; d < s.ndim; d++) {
                if (d == axis) continue;
                o_shape[o_ndim++] = s.shape[d];
            }
            if (o_ndim == 0) { o_shape[0] = 1; o_ndim = 1; }
        }
        ax_tensor_t *a = ax_tensor_zeros(s.shape, s.ndim, AX_FLOAT32);
        ax_tensor_t *out_n = ax_tensor_zeros(o_shape, o_ndim, AX_FLOAT32);
        ax_tensor_t *out_o = ax_tensor_zeros(o_shape, o_ndim, AX_FLOAT32);
        prop_fill(a);

        ax_compute_set_backend(AX_BACKEND_CPU_NAIVE);
        if (ax_compute_sum(a, axis, out_n) != AX_OK) goto fail;
        ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
        if (ax_compute_sum(a, axis, out_o) != AX_OK) goto fail;
        if (prop_close(out_n, out_o, 1e-4f, 1e-4f, seed, "sum")) passed++;
        goto next;
    fail:
        printf("  PROP[sum seed=%llu axis=%d]: backend op returned error\n",
               (unsigned long long)seed, axis);
    next:
        ax_tensor_destroy(a);
        ax_tensor_destroy(out_n); ax_tensor_destroy(out_o);
    }
    return passed;
}

#define N_CASES_DEFAULT 60

static void test_property_binops(void) {
    int p_add = run_binop_property("add", ax_compute_add, N_CASES_DEFAULT, 1e-6f, 1e-6f);
    int p_sub = run_binop_property("sub", ax_compute_sub, N_CASES_DEFAULT, 1e-6f, 1e-6f);
    int p_mul = run_binop_property("mul", ax_compute_mul, N_CASES_DEFAULT, 1e-6f, 1e-6f);
    printf("  binops: add %d/%d, sub %d/%d, mul %d/%d\n",
           p_add, N_CASES_DEFAULT, p_sub, N_CASES_DEFAULT, p_mul, N_CASES_DEFAULT);
    AX_TEST_ASSERT(p_add == N_CASES_DEFAULT
                   && p_sub == N_CASES_DEFAULT
                   && p_mul == N_CASES_DEFAULT,
                   "all binops match naive on random shapes");
}

static void test_property_unops(void) {
    int p_neg  = run_unop_property("neg",  ax_compute_neg,  N_CASES_DEFAULT, 0, 1e-6f, 1e-6f);
    int p_abs  = run_unop_property("abs",  ax_compute_abs,  N_CASES_DEFAULT, 0, 1e-6f, 1e-6f);
    int p_relu = run_unop_property("relu", ax_compute_relu, N_CASES_DEFAULT, 0, 1e-6f, 1e-6f);
    int p_sig  = run_unop_property("sig",  ax_compute_sigmoid, N_CASES_DEFAULT, 0, 1e-5f, 1e-5f);
    int p_tan  = run_unop_property("tanh", ax_compute_tanh, N_CASES_DEFAULT, 0, 1e-5f, 1e-5f);
    int p_exp  = run_unop_property("exp",  ax_compute_exp,  N_CASES_DEFAULT, 0, 1e-5f, 1e-4f);
    int p_log  = run_unop_property("log",  ax_compute_log,  N_CASES_DEFAULT, 1, 1e-5f, 1e-5f);
    int p_sqr  = run_unop_property("square", ax_compute_square, N_CASES_DEFAULT, 0, 1e-5f, 1e-5f);
    printf("  unops: neg %d sig %d tanh %d exp %d log %d sqr %d abs %d relu %d (each /%d)\n",
           p_neg, p_sig, p_tan, p_exp, p_log, p_sqr, p_abs, p_relu, N_CASES_DEFAULT);
    AX_TEST_ASSERT(p_neg == N_CASES_DEFAULT && p_abs == N_CASES_DEFAULT
                   && p_relu == N_CASES_DEFAULT && p_sig == N_CASES_DEFAULT
                   && p_tan == N_CASES_DEFAULT && p_exp == N_CASES_DEFAULT
                   && p_log == N_CASES_DEFAULT && p_sqr == N_CASES_DEFAULT,
                   "all unops match naive on random shapes");
}

static void test_property_gemm(void) {
    int p = run_gemm_property(N_CASES_DEFAULT);
    printf("  gemm: %d/%d random (M,N,K) shapes\n", p, N_CASES_DEFAULT);
    AX_TEST_ASSERT(p == N_CASES_DEFAULT, "gemm matches naive on random shapes");
}

static void test_property_reduce_sum(void) {
    int p = run_reduce_sum_property(N_CASES_DEFAULT);
    printf("  sum: %d/%d random shape+axis combos\n", p, N_CASES_DEFAULT);
    AX_TEST_ASSERT(p == N_CASES_DEFAULT, "sum matches naive on random shapes");
}

int main(void) {
    ax_init();
    AX_RUN_TEST(test_property_binops);
    AX_RUN_TEST(test_property_unops);
    AX_RUN_TEST(test_property_gemm);
    AX_RUN_TEST(test_property_reduce_sum);
    ax_shutdown();
    AX_TEST_SUMMARY();
}
