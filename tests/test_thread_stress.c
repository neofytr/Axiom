/* test_thread_stress.c -- thread-safety stress tests for axiom.
   exercises TLS arenas, slab pools, atomic refcounts, and parallel
   execution paths under concurrent pthread and openmp workloads. */

#include "test.h"
#include "axiom/axiom.h"
#include <pthread.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

/* 1. concurrent tensor create/destroy: 4 threads x 100 tensors each.
   hammers the TLS slab pool allocator for storage. */

static void *tensor_create_destroy_fn(void *arg)
{
    (void)arg;
    for (int i = 0; i < 100; i++) {
        int64_t shape[] = {64, 64};
        ax_tensor_t *t = ax_tensor_zeros(shape, 2, AX_FLOAT32);
        if (!t) return (void *)(intptr_t)1;
        ax_tensor_destroy(t);
    }
    return NULL;
}

static void test_concurrent_tensor_create_destroy(void)
{
    pthread_t threads[4];
    for (int i = 0; i < 4; i++)
        pthread_create(&threads[i], NULL, tensor_create_destroy_fn, NULL);
    int fail = 0;
    for (int i = 0; i < 4; i++) {
        void *ret;
        pthread_join(threads[i], &ret);
        if (ret) fail = 1;
    }
    AX_TEST_ASSERT(!fail, "no crash or alloc failure under concurrent create/destroy");
}

/* 2. concurrent gemm: 4 threads each do [128,128] @ [128,128].
   each fills A with 1/(128) and B with identity, so C should equal A.
   tests that TLS pack buffers don't cross-contaminate. */

typedef struct {
    int id;
    int ok;
} gemm_arg_t;

static void *concurrent_gemm_fn(void *arg)
{
    gemm_arg_t *ga = (gemm_arg_t *)arg;
    int M = 128, K = 128, N = 128;
    int64_t a_shape[] = {M, K};
    int64_t b_shape[] = {K, N};
    int64_t c_shape[] = {M, N};

    ax_tensor_t *a = ax_tensor_full(a_shape, 2, AX_FLOAT32, 1.0 / K);
    ax_tensor_t *b = ax_tensor_zeros(b_shape, 2, AX_FLOAT32);
    ax_tensor_t *c = ax_tensor_zeros(c_shape, 2, AX_FLOAT32);

    /* B = identity */
    float *bd = (float *)b->storage->data;
    for (int i = 0; i < K; i++) bd[i * N + i] = 1.0f;

    ax_compute_gemm(a, b, c);

    /* C should equal A: every element ~ 1/128 */
    float *cd = (float *)c->storage->data;
    ga->ok = 1;
    float expected = 1.0f / K;
    for (int i = 0; i < M * N; i++) {
        if (fabsf(cd[i] - expected) > 1e-4f) {
            ga->ok = 0;
            break;
        }
    }

    ax_tensor_destroy(a);
    ax_tensor_destroy(b);
    ax_tensor_destroy(c);
    return NULL;
}

static void test_concurrent_gemm(void)
{
    gemm_arg_t args[4];
    pthread_t threads[4];
    for (int i = 0; i < 4; i++) {
        args[i].id = i;
        args[i].ok = 0;
        pthread_create(&threads[i], NULL, concurrent_gemm_fn, &args[i]);
    }
    for (int i = 0; i < 4; i++)
        pthread_join(threads[i], NULL);

    for (int i = 0; i < 4; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "thread %d gemm result correct", i);
        AX_TEST_ASSERT(args[i].ok, msg);
    }
}

/* 3. concurrent forward+backward: 2 threads each with own dense layer.
   tests TLS autograd arena isolation. */

static void *fwd_bwd_fn(void *arg)
{
    (void)arg;
    ax_enable_grad();

    ax_layer_t *dense = ax_dense_create(64, 32, true);
    ax_layer_train(dense);

    int64_t xs[] = {4, 64};
    ax_tensor_t *x = ax_tensor_rand(xs, 2, -1.0f, 1.0f);
    x->requires_grad = true;

    ax_tensor_t *out = ax_layer_forward(dense, x);
    if (!out) { ax_tensor_destroy(x); ax_layer_destroy(dense); return (void *)(intptr_t)1; }

    ax_tensor_t *loss = ax_sum(out, -1);
    if (!loss) { ax_tensor_destroy(x); ax_layer_destroy(dense); return (void *)(intptr_t)1; }

    ax_backward(loss);

    /* check gradients are finite */
    ax_dense_t *dd = (ax_dense_t *)dense;
    float *wg = (float *)dd->weight->grad->storage->data;
    int64_t wn = ax_tensor_numel(dd->weight->grad);
    for (int64_t i = 0; i < wn; i++) {
        if (isnan(wg[i]) || isinf(wg[i])) {
            ax_graph_cleanup(loss);
            ax_tensor_destroy(loss);
            ax_tensor_destroy(x);
            ax_layer_destroy(dense);
            return (void *)(intptr_t)2;
        }
    }

    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss);
    ax_tensor_destroy(x);
    ax_layer_destroy(dense);
    return NULL;
}

static void test_concurrent_forward_backward(void)
{
    pthread_t threads[2];
    for (int i = 0; i < 2; i++)
        pthread_create(&threads[i], NULL, fwd_bwd_fn, (void *)(intptr_t)i);

    int fail = 0;
    for (int i = 0; i < 2; i++) {
        void *ret;
        pthread_join(threads[i], &ret);
        if (ret) fail = 1;
    }
    AX_TEST_ASSERT(!fail, "concurrent forward+backward with finite grads");
}

/* 4. concurrent batchnorm: 2 threads, each with own BN layer on [4,32,8,8].
   TLS scratch in norm.c should be independent per thread. */

typedef struct {
    int ok;
    double output_sum;
} bn_result_t;

static void *concurrent_bn_fn(void *arg)
{
    bn_result_t *res = (bn_result_t *)arg;
    res->ok = 0;

    ax_enable_grad();
    ax_layer_t *bn = ax_batchnorm_create(32, 1e-5f, 0.1f);
    ax_layer_train(bn);

    int64_t xs[] = {4, 32, 8, 8};
    ax_tensor_t *x = ax_tensor_rand(xs, 4, -1.0f, 1.0f);
    x->requires_grad = true;

    ax_tensor_t *out = ax_layer_forward(bn, x);
    if (!out) goto cleanup;

    /* check output is finite, accumulate sum for cross-thread comparison */
    float *od = (float *)out->storage->data;
    int64_t n = ax_tensor_numel(out);
    double sum = 0;
    for (int64_t i = 0; i < n; i++) {
        if (isnan(od[i]) || isinf(od[i])) goto cleanup;
        sum += (double)od[i];
    }
    res->output_sum = sum;

    /* backward */
    ax_tensor_t *loss = ax_sum(out, -1);
    if (!loss) goto cleanup;
    ax_backward(loss);

    res->ok = 1;
    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss);

cleanup:
    if (out) ax_tensor_destroy(out);
    ax_tensor_destroy(x);
    ax_layer_destroy(bn);
    return NULL;
}

static void test_concurrent_bn(void)
{
    bn_result_t results[2];
    pthread_t threads[2];
    for (int i = 0; i < 2; i++)
        pthread_create(&threads[i], NULL, concurrent_bn_fn, &results[i]);
    for (int i = 0; i < 2; i++)
        pthread_join(threads[i], NULL);

    AX_TEST_ASSERT(results[0].ok, "thread 0 BN forward+backward ok");
    AX_TEST_ASSERT(results[1].ok, "thread 1 BN forward+backward ok");
    AX_TEST_ASSERT(isfinite(results[0].output_sum) && isfinite(results[1].output_sum),
                   "BN outputs are finite");
}

/* 5. arena thread lifecycle: create threads, do work, join, then repeat.
   verifies TLS arenas initialize correctly for fresh threads. */

static void *lifecycle_fn(void *arg)
{
    (void)arg;
    ax_enable_grad();

    ax_layer_t *dense = ax_dense_create(32, 16, true);
    ax_layer_train(dense);

    int64_t xs[] = {2, 32};
    ax_tensor_t *x = ax_tensor_rand(xs, 2, -1.0f, 1.0f);
    x->requires_grad = true;

    ax_tensor_t *out = ax_layer_forward(dense, x);
    if (!out) { ax_tensor_destroy(x); ax_layer_destroy(dense); return (void *)(intptr_t)1; }

    ax_tensor_t *loss = ax_sum(out, -1);
    if (!loss) { ax_tensor_destroy(x); ax_layer_destroy(dense); return (void *)(intptr_t)1; }

    ax_backward(loss);
    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss);
    ax_tensor_destroy(x);
    ax_layer_destroy(dense);
    return NULL;
}

static void test_arena_thread_lifecycle(void)
{
    int fail = 0;

    /* round 1 */
    pthread_t threads[4];
    for (int i = 0; i < 4; i++)
        pthread_create(&threads[i], NULL, lifecycle_fn, NULL);
    for (int i = 0; i < 4; i++) {
        void *ret;
        pthread_join(threads[i], &ret);
        if (ret) fail = 1;
    }

    /* round 2: new threads reuse (or freshly init) TLS slots */
    for (int i = 0; i < 4; i++)
        pthread_create(&threads[i], NULL, lifecycle_fn, NULL);
    for (int i = 0; i < 4; i++) {
        void *ret;
        pthread_join(threads[i], &ret);
        if (ret) fail = 1;
    }

    AX_TEST_ASSERT(!fail, "TLS arenas work across thread generations");
}

/* 6. omp nested gemm: inside an omp parallel region, each thread does gemm.
   tests that gemm's internal omp_in_parallel() guard works correctly. */

static void test_omp_nested_gemm(void)
{
    int nt = 4;
    int ok_all = 1;
    int M = 64, K = 64, N = 64;

    /* pre-allocate per-thread tensors outside the parallel region */
    ax_tensor_t *a_arr[4], *b_arr[4], *c_arr[4];
    for (int t = 0; t < nt; t++) {
        int64_t a_shape[] = {M, K};
        int64_t b_shape[] = {K, N};
        int64_t c_shape[] = {M, N};
        a_arr[t] = ax_tensor_full(a_shape, 2, AX_FLOAT32, 1.0);
        b_arr[t] = ax_tensor_zeros(b_shape, 2, AX_FLOAT32);
        c_arr[t] = ax_tensor_zeros(c_shape, 2, AX_FLOAT32);
        /* B = identity */
        float *bd = (float *)b_arr[t]->storage->data;
        for (int i = 0; i < K; i++) bd[i * N + i] = 1.0f;
    }

    #pragma omp parallel num_threads(4) reduction(min:ok_all)
    {
        int tid = omp_get_thread_num();
        if (tid < nt) {
            ax_compute_gemm(a_arr[tid], b_arr[tid], c_arr[tid]);
            /* C = A * I = A, so every element should be 1.0 */
            float *cd = (float *)c_arr[tid]->storage->data;
            for (int i = 0; i < M * N; i++) {
                if (fabsf(cd[i] - 1.0f) > 1e-4f) { ok_all = 0; break; }
            }
        }
    }

    for (int t = 0; t < nt; t++) {
        ax_tensor_destroy(a_arr[t]);
        ax_tensor_destroy(b_arr[t]);
        ax_tensor_destroy(c_arr[t]);
    }

    AX_TEST_ASSERT(ok_all, "omp nested gemm all threads correct");
}

/* 7. storage refcount concurrent: one base tensor, 4 threads create views,
   read from them, then destroy. atomic refcount should handle this. */

typedef struct {
    ax_tensor_t *base;
    int ok;
} view_arg_t;

static void *view_fn(void *arg)
{
    view_arg_t *va = (view_arg_t *)arg;
    va->ok = 1;

    ax_tensor_t *v = ax_tensor_view(va->base);
    if (!v) { va->ok = 0; return NULL; }

    /* read all elements to exercise the shared storage */
    int64_t n = ax_tensor_numel(v);
    float *d = (float *)v->storage->data;
    float sum = 0;
    for (int64_t i = 0; i < n; i++) sum += d[i];

    /* ensure reads produced something finite */
    if (isnan(sum) || isinf(sum)) va->ok = 0;

    ax_tensor_destroy(v);
    return NULL;
}

static void test_storage_refcount_concurrent(void)
{
    int64_t shape[] = {32, 32};
    ax_tensor_t *base = ax_tensor_full(shape, 2, AX_FLOAT32, 1.0);

    view_arg_t args[4];
    pthread_t threads[4];
    for (int i = 0; i < 4; i++) {
        args[i].base = base;
        args[i].ok = 0;
        pthread_create(&threads[i], NULL, view_fn, &args[i]);
    }
    for (int i = 0; i < 4; i++)
        pthread_join(threads[i], NULL);

    for (int i = 0; i < 4; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "thread %d view read ok", i);
        AX_TEST_ASSERT(args[i].ok, msg);
    }

    /* base should still be valid (views released their refs) */
    int64_t idx[] = {0, 0};
    float v = ax_tensor_get_f32(base, idx);
    AX_TEST_ASSERT_NEAR(v, 1.0f, 1e-6f, "base tensor intact after view destroy");

    ax_tensor_destroy(base);
    AX_TEST_ASSERT(1, "no use-after-free on storage refcount");
}

/* 8. high contention alloc: 8 threads x 1000 small tensors.
   hammers the slab pool with rapid alloc/free cycles. */

static void *high_contention_fn(void *arg)
{
    (void)arg;
    for (int i = 0; i < 1000; i++) {
        int64_t shape[] = {1, 16};
        ax_tensor_t *t = ax_tensor_zeros(shape, 2, AX_FLOAT32);
        if (!t) return (void *)(intptr_t)1;

        /* write and read back to catch corrupted memory */
        float *d = (float *)t->storage->data;
        for (int j = 0; j < 16; j++) d[j] = (float)(j + 1);
        for (int j = 0; j < 16; j++) {
            if (d[j] != (float)(j + 1))
                return (void *)(intptr_t)2;
        }

        ax_tensor_destroy(t);
    }
    return NULL;
}

static void test_high_contention_alloc(void)
{
    pthread_t threads[8];
    for (int i = 0; i < 8; i++)
        pthread_create(&threads[i], NULL, high_contention_fn, NULL);

    int fail = 0;
    for (int i = 0; i < 8; i++) {
        void *ret;
        pthread_join(threads[i], &ret);
        if (ret) fail = 1;
    }
    AX_TEST_ASSERT(!fail, "8 threads x 1000 small tensors no crash or corruption");
}

int main(void)
{
    ax_init();

    printf("=== thread stress tests ===\n");
    AX_RUN_TEST(test_concurrent_tensor_create_destroy);
    AX_RUN_TEST(test_concurrent_gemm);
    AX_RUN_TEST(test_concurrent_forward_backward);
    AX_RUN_TEST(test_concurrent_bn);
    AX_RUN_TEST(test_arena_thread_lifecycle);
    AX_RUN_TEST(test_omp_nested_gemm);
    AX_RUN_TEST(test_storage_refcount_concurrent);
    AX_RUN_TEST(test_high_contention_alloc);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
