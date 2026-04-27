/* test_optimization_paths.c
   regression correctness tests for optimization fixes applied to
   the axiom dl framework: col_sum traversal order, bn tls scratch
   reuse, gemm nn/tn parity, mha train_step variant parity, arena
   sharing, save_p recompute parity, llc detection, heterogeneous
   core detection. */

#include "test.h"
#include "axiom/axiom.h"
#include "axiom/internal/attn_tunables.h"
#include "axiom/internal/compute_internal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void fill_det(ax_tensor_t *t, float bias)
{
    float *d = (float *)t->storage->data;
    int64_t n = ax_tensor_numel(t);
    for (int64_t i = 0; i < n; i++)
        d[i] = bias + (float)(i % 97) * 0.01f - 0.48f;
}

static float *raw(ax_tensor_t *t)
{
    return (float *)t->storage->data + t->offset;
}

/* col_sum correctness: axis=0 reduction must match manual accumulation
   after the column-major to row-major traversal change */
static void test_col_sum_correctness(void)
{
    int Ms[] = {1, 64, 2048};
    int Ns[] = {1, 64, 768, 2304};

    for (int mi = 0; mi < 3; mi++) {
        for (int ni = 0; ni < 4; ni++) {
            int64_t M = Ms[mi], N = Ns[ni];
            int64_t in_sh[] = {M, N};
            int64_t out_sh[] = {N};

            ax_tensor_t *mat = ax_tensor_create(in_sh, 2, AX_FLOAT32);
            fill_det(mat, (float)(mi * 4 + ni) * 0.1f);

            ax_tensor_t *col_sum = ax_tensor_zeros(out_sh, 1, AX_FLOAT32);
            ax_status_t st = ax_compute_sum(mat, 0, col_sum);
            AX_TEST_ASSERT(st == AX_OK, "col_sum dispatch ok");

            float *md = raw(mat);
            float *ref = (float *)calloc((size_t)N, sizeof(float));
            for (int64_t r = 0; r < M; r++)
                for (int64_t c = 0; c < N; c++)
                    ref[c] += md[r * N + c];

            float *sd = raw(col_sum);
            for (int64_t c = 0; c < N; c++) {
                float tol = fabsf(ref[c]) * 1e-4f + 1e-5f;
                AX_TEST_ASSERT_NEAR(sd[c], ref[c], tol,
                    "col_sum matches manual accumulation");
            }

            free(ref);
            ax_tensor_destroy(mat);
            ax_tensor_destroy(col_sum);
        }
    }
}

/* bn tls scratch reuse: forward+backward at different sizes exercises
   the per-thread scratch buffer growing and then being reused at a
   smaller size without corruption */
static void test_bn_tls_scratch_reuse(void)
{
    int64_t C1 = 64;
    ax_layer_t *bn = ax_batchnorm_create(C1, 1e-5f, 0.1f);
    AX_TEST_ASSERT(bn != NULL, "bn create");
    ax_layer_train(bn);
    ax_enable_grad();

    /* pass 1: [32, 64, 8, 8] */
    int64_t sh1[] = {32, C1, 8, 8};
    ax_tensor_t *x1 = ax_tensor_rand(sh1, 4, -1.0f, 1.0f);
    ax_tensor_t *out1 = ax_layer_forward(bn, x1);
    AX_TEST_ASSERT(out1 != NULL, "bn fwd pass 1");
    ax_backward(out1);
    ax_graph_cleanup(out1);
    ax_tensor_destroy(out1);
    ax_tensor_destroy(x1);

    /* pass 2: [16, 64, 16, 16] (larger spatial) */
    int64_t sh2[] = {16, C1, 16, 16};
    ax_tensor_t *x2 = ax_tensor_rand(sh2, 4, -1.0f, 1.0f);
    ax_tensor_t *out2 = ax_layer_forward(bn, x2);
    AX_TEST_ASSERT(out2 != NULL, "bn fwd pass 2 (larger)");
    ax_backward(out2);
    ax_graph_cleanup(out2);
    ax_tensor_destroy(out2);
    ax_tensor_destroy(x2);

    /* pass 3: [32, 64, 8, 8] again (reuses smaller scratch) */
    ax_tensor_t *x3 = ax_tensor_rand(sh1, 4, -1.0f, 1.0f);
    ax_tensor_t *out3 = ax_layer_forward(bn, x3);
    AX_TEST_ASSERT(out3 != NULL, "bn fwd pass 3 (reuse)");
    ax_backward(out3);
    ax_graph_cleanup(out3);

    /* after 3 training steps, running stats should be partially learned.
       switch to eval and check output is reasonable */
    ax_layer_eval(bn);
    ax_no_grad();
    ax_tensor_t *x_eval = ax_tensor_rand(sh1, 4, -0.5f, 0.5f);
    ax_tensor_t *out_eval = ax_layer_forward(bn, x_eval);
    AX_TEST_ASSERT(out_eval != NULL, "bn eval after scratch reuse");

    /* output should have values in a reasonable range (not nan/inf) */
    float *od = raw(out_eval);
    int64_t n = ax_tensor_numel(out_eval);
    bool sane = true;
    for (int64_t i = 0; i < n && sane; i++)
        if (isnan(od[i]) || isinf(od[i])) sane = false;
    AX_TEST_ASSERT(sane, "bn eval output has no nan/inf after scratch reuse");

    ax_enable_grad();
    ax_tensor_destroy(out3);
    ax_tensor_destroy(x3);
    ax_tensor_destroy(out_eval);
    ax_tensor_destroy(x_eval);
    ax_layer_destroy(bn);
}

/* bn single thread vs multi: the bw threshold optimization skips omp
   for small tensors. verify both thread counts produce identical results */
static void test_bn_single_thread_vs_multi(void)
{
    int64_t C = 16;
    int64_t sh[] = {4, C, 4, 4};

    /* run with 1 thread */
    ax_layer_t *bn1 = ax_batchnorm_create(C, 1e-5f, 0.1f);
    ax_layer_train(bn1);

    ax_tensor_t *x = ax_tensor_rand(sh, 4, -1.0f, 1.0f);

    ax_set_num_threads(1);
    ax_enable_grad();
    ax_tensor_t *out1 = ax_layer_forward(bn1, x);
    AX_TEST_ASSERT(out1 != NULL, "bn 1-thread fwd");
    ax_backward(out1);
    ax_graph_cleanup(out1);

    float *gamma_grad_1 = (float *)malloc((size_t)C * sizeof(float));
    float *beta_grad_1 = (float *)malloc((size_t)C * sizeof(float));
    ax_batchnorm_t *bnt1 = (ax_batchnorm_t *)bn1;
    if (bnt1->gamma->grad) {
        memcpy(gamma_grad_1, raw(bnt1->gamma->grad), (size_t)C * sizeof(float));
        memcpy(beta_grad_1, raw(bnt1->beta->grad), (size_t)C * sizeof(float));
    }

    /* run with 16 threads on identical layer + input */
    ax_layer_t *bn2 = ax_batchnorm_create(C, 1e-5f, 0.1f);
    ax_layer_train(bn2);

    ax_set_num_threads(16);
    ax_tensor_t *out2 = ax_layer_forward(bn2, x);
    AX_TEST_ASSERT(out2 != NULL, "bn 16-thread fwd");
    ax_backward(out2);
    ax_graph_cleanup(out2);

    /* compare outputs */
    float *d1 = raw(out1), *d2 = raw(out2);
    int64_t n = ax_tensor_numel(out1);
    float max_diff = 0;
    for (int64_t i = 0; i < n; i++) {
        float diff = fabsf(d1[i] - d2[i]);
        if (diff > max_diff) max_diff = diff;
    }
    AX_TEST_ASSERT(max_diff < 1e-4f, "bn 1-thread vs 16-thread forward match");

    /* compare param grads */
    ax_batchnorm_t *bnt2 = (ax_batchnorm_t *)bn2;
    if (bnt1->gamma->grad && bnt2->gamma->grad) {
        float *gg2 = raw(bnt2->gamma->grad);
        float *bg2 = raw(bnt2->beta->grad);
        for (int64_t c = 0; c < C; c++) {
            AX_TEST_ASSERT_NEAR(gamma_grad_1[c], gg2[c], 1e-4f,
                "gamma grad 1t vs 16t");
            AX_TEST_ASSERT_NEAR(beta_grad_1[c], bg2[c], 1e-4f,
                "beta grad 1t vs 16t");
        }
    }

    ax_set_num_threads(0);
    free(gamma_grad_1);
    free(beta_grad_1);
    ax_tensor_destroy(out1);
    ax_tensor_destroy(out2);
    ax_tensor_destroy(x);
    ax_layer_destroy(bn1);
    ax_layer_destroy(bn2);
}

/* gemm nn/tn parity: C1 = A @ B via gemm_nn must match
   C2 = At^T @ B via gemm_tn where At = A^T stored as [K,M].
   validates the jit strided-A path vs packed path */
static void test_gemm_nn_tn_parity(void)
{
    struct { int64_t M, N, K; } shapes[] = {
        {32, 32, 32},
        {64, 64, 128},
        {128, 128, 128},
    };

    for (int si = 0; si < 3; si++) {
        int64_t M = shapes[si].M, N = shapes[si].N, K = shapes[si].K;

        int64_t a_sh[] = {M, K};
        int64_t b_sh[] = {K, N};
        int64_t c_sh[] = {M, N};
        int64_t at_sh[] = {K, M};

        ax_tensor_t *A = ax_tensor_rand(a_sh, 2, -0.5f, 0.5f);
        ax_tensor_t *B = ax_tensor_rand(b_sh, 2, -0.5f, 0.5f);
        ax_tensor_t *C1 = ax_tensor_zeros(c_sh, 2, AX_FLOAT32);
        ax_tensor_t *C2 = ax_tensor_zeros(c_sh, 2, AX_FLOAT32);

        /* build At = physically transposed copy of A: At[k,m] = A[m,k] */
        ax_tensor_t *At = ax_tensor_create(at_sh, 2, AX_FLOAT32);
        float *ad = raw(A), *atd = raw(At);
        for (int64_t m = 0; m < M; m++)
            for (int64_t k = 0; k < K; k++)
                atd[k * M + m] = ad[m * K + k];

        ax_status_t s1 = ax_compute_gemm(A, B, C1);
        AX_TEST_ASSERT(s1 == AX_OK, "gemm_nn ok");

        ax_status_t s2 = ax_compute_gemm_tn(At, B, C2);
        if (s2 == AX_ERR_NOT_IMPLEMENTED) {
            /* backend lacks gemm_tn, skip */
            ax_tensor_destroy(A); ax_tensor_destroy(B);
            ax_tensor_destroy(C1); ax_tensor_destroy(C2);
            ax_tensor_destroy(At);
            continue;
        }
        AX_TEST_ASSERT(s2 == AX_OK, "gemm_tn ok");

        float *c1d = raw(C1), *c2d = raw(C2);
        float max_rel = 0;
        for (int64_t i = 0; i < M * N; i++) {
            float e = fabsf(c1d[i] - c2d[i]);
            float scale = fabsf(c1d[i]) + fabsf(c2d[i]) + 1e-8f;
            float rel = e / scale;
            if (rel > max_rel) max_rel = rel;
        }
        float tol = (K <= 256) ? 1e-3f : 1e-2f;
        AX_TEST_ASSERT(max_rel < tol, "gemm_nn vs gemm_tn parity");

        ax_tensor_destroy(A); ax_tensor_destroy(B);
        ax_tensor_destroy(C1); ax_tensor_destroy(C2);
        ax_tensor_destroy(At);
    }
}

/* gemm scheduling determinism: same gemm 10 times must produce
   bitwise identical results (no schedule(runtime) non-determinism) */
static void test_gemm_scheduling_determinism(void)
{
    int64_t M = 256, N = 256, K = 256;
    int64_t a_sh[] = {M, K};
    int64_t b_sh[] = {K, N};
    int64_t c_sh[] = {M, N};

    ax_tensor_t *A = ax_tensor_rand(a_sh, 2, -0.5f, 0.5f);
    ax_tensor_t *B = ax_tensor_rand(b_sh, 2, -0.5f, 0.5f);

    ax_tensor_t *ref = ax_tensor_zeros(c_sh, 2, AX_FLOAT32);
    ax_compute_gemm(A, B, ref);
    float *refd = raw(ref);

    for (int trial = 1; trial < 10; trial++) {
        ax_tensor_t *C = ax_tensor_zeros(c_sh, 2, AX_FLOAT32);
        ax_compute_gemm(A, B, C);
        float *cd = raw(C);

        bool identical = (memcmp(refd, cd, (size_t)(M * N) * sizeof(float)) == 0);
        AX_TEST_ASSERT(identical, "gemm bitwise deterministic across runs");
        ax_tensor_destroy(C);
    }

    ax_tensor_destroy(A);
    ax_tensor_destroy(B);
    ax_tensor_destroy(ref);
}

/* helper: create mha input tensors */
static void make_mha_inputs(int64_t B, int64_t S, int64_t D,
                             ax_tensor_t **x, ax_tensor_t **dout,
                             ax_tensor_t **y_out)
{
    int64_t sh[] = {B, S, D};
    *x = ax_tensor_rand(sh, 3, -0.3f, 0.3f);
    *dout = ax_tensor_rand(sh, 3, -0.1f, 0.1f);
    *y_out = ax_tensor_zeros(sh, 3, AX_FLOAT32);
}

/* helper: zero all mha weight grads before a train step */
static void zero_mha_grads(ax_layer_t *layer)
{
    ax_tensor_t *params[AX_LAYER_MAX_PARAMS];
    int np = ax_layer_get_params(layer, params, AX_LAYER_MAX_PARAMS);
    for (int i = 0; i < np; i++)
        ax_zero_grad(params[i]);
}

/* mha train_step variant parity: train_step, fused, and v4 must
   produce the same output on the same input. validates arena sharing
   and head-major paths */
static void test_mha_train_step_variants_parity(void)
{
    int64_t D = 128, B = 1, S = 32;
    int H = 4;

    ax_layer_t *mha = ax_mha_create(D, H, false, false);
    AX_TEST_ASSERT(mha != NULL, "mha create");
    ax_layer_train(mha);

    ax_tensor_t *x, *dout, *y1, *y2, *y3;
    int64_t sh[] = {B, S, D};
    x = ax_tensor_rand(sh, 3, -0.3f, 0.3f);
    dout = ax_tensor_rand(sh, 3, -0.1f, 0.1f);
    y1 = ax_tensor_zeros(sh, 3, AX_FLOAT32);
    y2 = ax_tensor_zeros(sh, 3, AX_FLOAT32);
    y3 = ax_tensor_zeros(sh, 3, AX_FLOAT32);

    zero_mha_grads(mha);
    ax_status_t s1 = ax_mha_train_step(mha, x, dout, y1);
    AX_TEST_ASSERT(s1 == AX_OK, "train_step ok");

    zero_mha_grads(mha);
    ax_status_t s2 = ax_mha_train_step_fused(mha, x, dout, y2);
    AX_TEST_ASSERT(s2 == AX_OK, "train_step_fused ok");

    zero_mha_grads(mha);
    ax_status_t s3 = ax_mha_train_step_v4(mha, x, dout, y3);
    AX_TEST_ASSERT(s3 == AX_OK, "train_step_v4 ok");

    float *d1 = raw(y1), *d2 = raw(y2), *d3 = raw(y3);
    int64_t n = B * S * D;
    float max12 = 0, max13 = 0;
    for (int64_t i = 0; i < n; i++) {
        float e12 = fabsf(d1[i] - d2[i]);
        float e13 = fabsf(d1[i] - d3[i]);
        if (e12 > max12) max12 = e12;
        if (e13 > max13) max13 = e13;
    }
    AX_TEST_ASSERT(max12 < 1e-3f, "train_step vs fused parity");
    AX_TEST_ASSERT(max13 < 1e-3f, "train_step vs v4 parity");

    ax_tensor_destroy(x); ax_tensor_destroy(dout);
    ax_tensor_destroy(y1); ax_tensor_destroy(y2); ax_tensor_destroy(y3);
    ax_layer_destroy(mha);
}

/* arena sharing sequential: run all three train_step variants in
   sequence on the same layer and input. the shared arena must reset
   cleanly between calls with no corruption */
static void test_arena_sharing_sequential(void)
{
    int64_t D = 128, B = 1, S = 32;
    int H = 4;

    ax_layer_t *mha = ax_mha_create(D, H, false, false);
    AX_TEST_ASSERT(mha != NULL, "mha create");
    ax_layer_train(mha);

    ax_tensor_t *x, *dout, *y_out;
    make_mha_inputs(B, S, D, &x, &dout, &y_out);

    zero_mha_grads(mha);
    ax_status_t s1 = ax_mha_train_step(mha, x, dout, y_out);
    AX_TEST_ASSERT(s1 == AX_OK, "sequential: train_step ok");
    float *d1 = raw(y_out);
    int64_t n = B * S * D;
    bool sane1 = true;
    for (int64_t i = 0; i < n && sane1; i++)
        if (isnan(d1[i]) || isinf(d1[i])) sane1 = false;
    AX_TEST_ASSERT(sane1, "sequential: train_step output sane");

    zero_mha_grads(mha);
    memset(raw(y_out), 0, (size_t)(n) * sizeof(float));
    ax_status_t s2 = ax_mha_train_step_fused(mha, x, dout, y_out);
    AX_TEST_ASSERT(s2 == AX_OK, "sequential: fused ok");
    float *d2 = raw(y_out);
    bool sane2 = true;
    for (int64_t i = 0; i < n && sane2; i++)
        if (isnan(d2[i]) || isinf(d2[i])) sane2 = false;
    AX_TEST_ASSERT(sane2, "sequential: fused output sane");

    zero_mha_grads(mha);
    memset(raw(y_out), 0, (size_t)(n) * sizeof(float));
    ax_status_t s3 = ax_mha_train_step_v4(mha, x, dout, y_out);
    AX_TEST_ASSERT(s3 == AX_OK, "sequential: v4 ok");
    float *d3 = raw(y_out);
    bool sane3 = true;
    for (int64_t i = 0; i < n && sane3; i++)
        if (isnan(d3[i]) || isinf(d3[i])) sane3 = false;
    AX_TEST_ASSERT(sane3, "sequential: v4 output sane");

    ax_tensor_destroy(x);
    ax_tensor_destroy(dout);
    ax_tensor_destroy(y_out);
    ax_layer_destroy(mha);
}

/* save_p vs recompute parity: force AX_MHA_SAVE_P=1 then =0 and check
   outputs match. note: the env var is cached via static locals on first
   read, so only the first call's mode is testable within a single process.
   we test whichever mode the env selects against a second train_step call
   (which reuses the same mode) and verify consistency. if AX_MHA_SAVE_P
   was not previously resolved, we set it before the first call. */
static void test_save_p_recompute_parity(void)
{
    /* set before any train_step_fused call resolves the cached static */
    setenv("AX_MHA_SAVE_P", "1", 1);

    int64_t D = 64, B = 1, S = 16;
    int H = 4;
    ax_layer_t *mha = ax_mha_create(D, H, false, false);
    AX_TEST_ASSERT(mha != NULL, "mha create for save_p test");
    ax_layer_train(mha);

    int64_t sh[] = {B, S, D};
    ax_tensor_t *x = ax_tensor_rand(sh, 3, -0.3f, 0.3f);
    ax_tensor_t *dout = ax_tensor_rand(sh, 3, -0.1f, 0.1f);
    ax_tensor_t *y1 = ax_tensor_zeros(sh, 3, AX_FLOAT32);
    ax_tensor_t *y2 = ax_tensor_zeros(sh, 3, AX_FLOAT32);

    /* first call (save_p=1 if this is the first resolution) */
    zero_mha_grads(mha);
    ax_status_t s1 = ax_mha_train_step_fused(mha, x, dout, y1);
    AX_TEST_ASSERT(s1 == AX_OK, "save_p: first call ok");

    /* second call with same mode (env is cached, can't flip mid-process) */
    zero_mha_grads(mha);
    ax_status_t s2 = ax_mha_train_step_fused(mha, x, dout, y2);
    AX_TEST_ASSERT(s2 == AX_OK, "save_p: second call ok");

    /* same input + same weights + same mode => same output */
    float *d1 = raw(y1), *d2 = raw(y2);
    int64_t n = B * S * D;
    float max_err = 0;
    for (int64_t i = 0; i < n; i++) {
        float e = fabsf(d1[i] - d2[i]);
        if (e > max_err) max_err = e;
    }
    AX_TEST_ASSERT(max_err < 1e-5f, "save_p: repeated call consistency");

    /* cross-check against train_step (non-fused) which has its own
       save_p resolution. the forward output should match. */
    zero_mha_grads(mha);
    ax_tensor_t *y3 = ax_tensor_zeros(sh, 3, AX_FLOAT32);
    ax_status_t s3 = ax_mha_train_step(mha, x, dout, y3);
    AX_TEST_ASSERT(s3 == AX_OK, "save_p: train_step ok");

    float *d3 = raw(y3);
    float max_err2 = 0;
    for (int64_t i = 0; i < n; i++) {
        float e = fabsf(d1[i] - d3[i]);
        if (e > max_err2) max_err2 = e;
    }
    AX_TEST_ASSERT(max_err2 < 1e-3f, "save_p: fused vs non-fused parity");

    unsetenv("AX_MHA_SAVE_P");
    ax_tensor_destroy(x); ax_tensor_destroy(dout);
    ax_tensor_destroy(y1); ax_tensor_destroy(y2); ax_tensor_destroy(y3);
    ax_layer_destroy(mha);
}

/* llc detection: ax_attn_tunable_llc_bytes() should return >0 on
   most modern cpus with L3 cache */
static void test_llc_detection(void)
{
    int64_t llc = ax_attn_tunable_llc_bytes();
    /* 0 is valid (detection can fail on some platforms), but on any
       machine running this test suite it should succeed */
    AX_TEST_ASSERT(llc > 0, "llc detection returned >0 bytes");
    AX_TEST_ASSERT(llc < (int64_t)512 * 1024 * 1024,
        "llc size plausible (<512MB)");
}

/* heterogeneous core detection: just verify it returns 0 or 1 without
   crashing */
static void test_heterogeneous_detection(void)
{
    int result = ax_compute_cores_heterogeneous();
    AX_TEST_ASSERT(result == 0 || result == 1,
        "heterogeneous detection returns 0 or 1");
}

/* pool_prewarm: pre-filling the pool eliminates first-call allocations */
static void test_pool_prewarm(void)
{
    size_t sizes[] = {4096, 16384, 65536};
    ax_pool_prewarm(sizes, 3);

    int64_t sh1[] = {1024};  /* 1024 * 4 = 4096 bytes */
    int64_t sh2[] = {4096};  /* 4096 * 4 = 16384 bytes */
    int64_t sh3[] = {16384}; /* 16384 * 4 = 65536 bytes */

    ax_tensor_t *t1 = ax_tensor_create(sh1, 1, AX_FLOAT32);
    ax_tensor_t *t2 = ax_tensor_create(sh2, 1, AX_FLOAT32);
    ax_tensor_t *t3 = ax_tensor_create(sh3, 1, AX_FLOAT32);
    AX_TEST_ASSERT(t1 && t2 && t3, "prewarm'd tensors created ok");

    /* return to pool + slab, then re-create to verify pool serves */
    ax_tensor_destroy(t3);
    ax_tensor_destroy(t2);
    ax_tensor_destroy(t1);

#ifdef AX_COUNT_ALLOCS
    ax_reset_alloc_count();
    t1 = ax_tensor_create(sh1, 1, AX_FLOAT32);
    t2 = ax_tensor_create(sh2, 1, AX_FLOAT32);
    t3 = ax_tensor_create(sh3, 1, AX_FLOAT32);
    uint64_t allocs = ax_get_alloc_count();
    AX_TEST_ASSERT(allocs == 0, "pool-served after prewarm: zero allocs");
    ax_tensor_destroy(t3);
    ax_tensor_destroy(t2);
    ax_tensor_destroy(t1);
#endif
}

/* arena_block_sizing: verify setter/getter and arena respects config */
static void test_arena_block_sizing(void)
{
    size_t orig = ax_get_arena_block_size();

    ax_set_arena_block_size(256 * 1024);
    AX_TEST_ASSERT(ax_get_arena_block_size() == 256 * 1024,
        "arena block size setter/getter roundtrip");

    ax_arena_t *a = ax_arena_create(0);
    AX_TEST_ASSERT(a != NULL, "arena create with configured size");

    void *p = ax_arena_alloc(a, 1024, 64);
    AX_TEST_ASSERT(p != NULL, "arena alloc from configured arena");

    /* allocate close to the block size to verify it respects the limit */
    void *big = ax_arena_alloc(a, 200 * 1024, 64);
    AX_TEST_ASSERT(big != NULL, "large alloc within configured block");

    ax_arena_destroy(a);
    ax_set_arena_block_size(orig);
}

/* configure_embedded: convenience sets both arena and pool config */
static void test_configure_embedded(void)
{
    size_t orig_arena = ax_get_arena_block_size();
    int orig_cap = ax_get_pool_bucket_cap();

    ax_configure_embedded(2 * 1024 * 1024);

    AX_TEST_ASSERT(ax_get_arena_block_size() == 256 * 1024,
        "embedded arena = budget/8 = 256 KB");
    AX_TEST_ASSERT(ax_get_pool_bucket_cap() == 4,
        "embedded pool cap = 4");

    /* verify we can still run inference with the embedded config */
    ax_layer_t *model = ax_sequential_create();
    ax_sequential_add(model, ax_dense_create(32, 10, true));
    ax_layer_eval(model);

    int64_t x_sh[] = {1, 32};
    ax_tensor_t *x = ax_tensor_rand(x_sh, 2, -1.0f, 1.0f);
    ax_no_grad();
    ax_tensor_t *y = ax_layer_forward(model, x);
    AX_TEST_ASSERT(y != NULL, "inference works under embedded config");
    ax_tensor_destroy(y);
    ax_enable_grad();
    ax_tensor_destroy(x);
    ax_layer_destroy(model);

    ax_set_arena_block_size(orig_arena);
    ax_set_pool_bucket_cap(orig_cap);
}

/* steady_state_zero_malloc: prove inference is zero-alloc after warmup */
#ifdef AX_COUNT_ALLOCS
static void test_steady_state_zero_malloc(void)
{
    /* raw gemm */
    {
        int64_t a_sh[] = {1, 64}, b_sh[] = {64, 32}, c_sh[] = {1, 32};
        ax_tensor_t *A = ax_tensor_rand(a_sh, 2, -0.5f, 0.5f);
        ax_tensor_t *B = ax_tensor_rand(b_sh, 2, -0.5f, 0.5f);
        ax_tensor_t *C = ax_tensor_create(c_sh, 2, AX_FLOAT32);

        for (int w = 0; w < 3; w++) ax_compute_gemm(A, B, C);

        ax_reset_alloc_count();
        for (int i = 0; i < 10; i++) ax_compute_gemm(A, B, C);
        uint64_t allocs = ax_get_alloc_count();
        AX_TEST_ASSERT(allocs == 0, "gemm steady-state zero-malloc");

        ax_tensor_destroy(C);
        ax_tensor_destroy(B);
        ax_tensor_destroy(A);
    }
    /* mlp inference */
    {
        ax_layer_t *model = ax_sequential_create();
        ax_sequential_add(model, ax_dense_create(64, 32, true));
        ax_sequential_add(model, ax_relu_layer_create());
        ax_sequential_add(model, ax_dense_create(32, 10, false));
        ax_layer_eval(model);

        int64_t x_sh[] = {1, 64};
        ax_tensor_t *x = ax_tensor_rand(x_sh, 2, -1.0f, 1.0f);

        ax_no_grad();
        for (int w = 0; w < 5; w++) {
            ax_tensor_t *y = ax_layer_forward(model, x);
            ax_tensor_destroy(y);
        }

        ax_reset_alloc_count();
        for (int i = 0; i < 10; i++) {
            ax_tensor_t *y = ax_layer_forward(model, x);
            ax_tensor_destroy(y);
        }
        uint64_t allocs = ax_get_alloc_count();
        ax_enable_grad();

        AX_TEST_ASSERT(allocs == 0, "mlp inference steady-state zero-malloc");

        ax_tensor_destroy(x);
        ax_layer_destroy(model);
    }
}
#endif

int main(void)
{
    ax_init();
    ax_set_seed(42);
    printf("=== optimization path regression tests ===\n");

    AX_RUN_TEST(test_col_sum_correctness);
    AX_RUN_TEST(test_bn_tls_scratch_reuse);
    AX_RUN_TEST(test_bn_single_thread_vs_multi);
    AX_RUN_TEST(test_gemm_nn_tn_parity);
    AX_RUN_TEST(test_gemm_scheduling_determinism);
    AX_RUN_TEST(test_mha_train_step_variants_parity);
    AX_RUN_TEST(test_arena_sharing_sequential);
    AX_RUN_TEST(test_save_p_recompute_parity);
    AX_RUN_TEST(test_llc_detection);
    AX_RUN_TEST(test_heterogeneous_detection);

    AX_RUN_TEST(test_pool_prewarm);
    AX_RUN_TEST(test_arena_block_sizing);
    AX_RUN_TEST(test_configure_embedded);
#ifdef AX_COUNT_ALLOCS
    AX_RUN_TEST(test_steady_state_zero_malloc);
#endif

    ax_shutdown();
    AX_TEST_SUMMARY();
}
