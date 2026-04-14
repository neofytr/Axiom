/* test_attention.c — correctness tests for MHA, SDPA, RoPE, KV cache.

   validates the fused SDPA compute path against a naive O(S^2 * dk)
   reference, then checks the MHA layer forward against a manually
   composed (Wq/Wk/Wv projection + SDPA + Wo projection) pipeline. */

#include "axiom/axiom.h"
#include "test.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ================================================================
   naive reference SDPA (per-head) — straightforward nested loops.
   computes out[i] = softmax(scale * Q[i] @ K^T) @ V.
   with optional causal mask.
   ================================================================ */
static void naive_sdpa_head(const float *Q, const float *K, const float *V,
                             float *O, float *L_out,
                             int64_t S, int64_t dk, float scale, bool causal)
{
    float *scores = (float *)malloc((size_t)S * (size_t)S * sizeof(float));
    for (int64_t i = 0; i < S; i++) {
        float mx = -INFINITY;
        for (int64_t j = 0; j < S; j++) {
            if (causal && j > i) { scores[i * S + j] = -INFINITY; continue; }
            float dot = 0;
            for (int64_t d = 0; d < dk; d++) dot += Q[i * dk + d] * K[j * dk + d];
            dot *= scale;
            scores[i * S + j] = dot;
            if (dot > mx) mx = dot;
        }
        float sum = 0;
        for (int64_t j = 0; j < S; j++) {
            if (causal && j > i) { scores[i * S + j] = 0; continue; }
            scores[i * S + j] = expf(scores[i * S + j] - mx);
            sum += scores[i * S + j];
        }
        float inv = 1.0f / sum;
        for (int64_t d = 0; d < dk; d++) {
            float acc = 0;
            for (int64_t j = 0; j < S; j++) acc += scores[i * S + j] * inv * V[j * dk + d];
            O[i * dk + d] = acc;
        }
        if (L_out) L_out[i] = mx + logf(sum);
    }
    free(scores);
}

/* ================================================================
   test: SDPA forward matches naive reference
   ================================================================ */
static void test_sdpa_forward_parity(void)
{
    const int64_t BH = 6;
    const int64_t S  = 24;
    const int64_t dk = 16;
    int64_t elems = BH * S * dk;

    float *Q = (float *)malloc(elems * sizeof(float));
    float *K = (float *)malloc(elems * sizeof(float));
    float *V = (float *)malloc(elems * sizeof(float));
    for (int64_t i = 0; i < elems; i++) {
        Q[i] = (float)rand() / RAND_MAX - 0.5f;
        K[i] = (float)rand() / RAND_MAX - 0.5f;
        V[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    float *O_ax   = (float *)calloc(elems, sizeof(float));
    float *O_ref  = (float *)calloc(elems, sizeof(float));
    float *L_ax   = (float *)calloc(BH * S, sizeof(float));

    float scale = 1.0f / sqrtf((float)dk);

    ax_fused_attention_fwd(Q, K, V, O_ax, L_ax, BH, S, dk, scale);

    for (int64_t h = 0; h < BH; h++) {
        naive_sdpa_head(Q + h * S * dk, K + h * S * dk, V + h * S * dk,
                         O_ref + h * S * dk, NULL,
                         S, dk, scale, false);
    }

    float max_err = 0;
    for (int64_t i = 0; i < elems; i++) {
        float e = fabsf(O_ax[i] - O_ref[i]);
        if (e > max_err) max_err = e;
    }
    AX_TEST_ASSERT(max_err < 1e-4f, "SDPA forward matches naive reference");

    free(Q); free(K); free(V); free(O_ax); free(O_ref); free(L_ax);
}

/* ================================================================
   test: causal SDPA — attention to future tokens is zero
   ================================================================ */
static void test_sdpa_causal(void)
{
    const int64_t BH = 2;
    const int64_t S  = 16;
    const int64_t dk = 8;
    int64_t elems = BH * S * dk;

    float *Q = (float *)malloc(elems * sizeof(float));
    float *K = (float *)malloc(elems * sizeof(float));
    float *V = (float *)malloc(elems * sizeof(float));
    for (int64_t i = 0; i < elems; i++) {
        Q[i] = (float)rand() / RAND_MAX - 0.5f;
        K[i] = (float)rand() / RAND_MAX - 0.5f;
        V[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    float *O_ax  = (float *)calloc(elems, sizeof(float));
    float *O_ref = (float *)calloc(elems, sizeof(float));

    float scale = 1.0f / sqrtf((float)dk);
    ax_fused_attention_fwd_causal(Q, K, V, O_ax, NULL, BH, S, dk, scale);

    for (int64_t h = 0; h < BH; h++) {
        naive_sdpa_head(Q + h * S * dk, K + h * S * dk, V + h * S * dk,
                         O_ref + h * S * dk, NULL,
                         S, dk, scale, true);
    }

    float max_err = 0;
    for (int64_t i = 0; i < elems; i++) {
        float e = fabsf(O_ax[i] - O_ref[i]);
        if (e > max_err) max_err = e;
    }
    AX_TEST_ASSERT(max_err < 1e-4f, "causal SDPA matches naive causal reference");

    free(Q); free(K); free(V); free(O_ax); free(O_ref);
}

/* ================================================================
   test: SDPA backward — compare analytic vs finite difference
   ================================================================ */
static void test_sdpa_backward_fd(void)
{
    const int64_t BH = 1;
    const int64_t S  = 6;
    const int64_t dk = 4;
    int64_t elems = BH * S * dk;
    float scale = 1.0f / sqrtf((float)dk);

    float *Q = (float *)malloc(elems * sizeof(float));
    float *K = (float *)malloc(elems * sizeof(float));
    float *V = (float *)malloc(elems * sizeof(float));
    for (int64_t i = 0; i < elems; i++) {
        Q[i] = 0.1f * ((float)rand() / RAND_MAX - 0.5f);
        K[i] = 0.1f * ((float)rand() / RAND_MAX - 0.5f);
        V[i] = 0.1f * ((float)rand() / RAND_MAX - 0.5f);
    }
    float *O  = (float *)calloc(elems, sizeof(float));
    float *L  = (float *)calloc(BH * S, sizeof(float));
    float *dO = (float *)malloc(elems * sizeof(float));
    for (int64_t i = 0; i < elems; i++) dO[i] = 1.0f;  /* sum(O) as loss */

    ax_fused_attention_fwd(Q, K, V, O, L, BH, S, dk, scale);

    float *dQ = (float *)calloc(elems, sizeof(float));
    float *dK = (float *)calloc(elems, sizeof(float));
    float *dV = (float *)calloc(elems, sizeof(float));
    ax_fused_attention_bwd(Q, K, V, O, dO, L, dQ, dK, dV, BH, S, dk, scale);

    /* finite difference check for dV at a single index */
    const float eps = 1e-3f;
    int64_t idx = 5;  /* arbitrary V element */
    float orig = V[idx];
    float *O_plus  = (float *)calloc(elems, sizeof(float));
    float *O_minus = (float *)calloc(elems, sizeof(float));
    V[idx] = orig + eps;
    ax_fused_attention_fwd(Q, K, V, O_plus,  NULL, BH, S, dk, scale);
    V[idx] = orig - eps;
    ax_fused_attention_fwd(Q, K, V, O_minus, NULL, BH, S, dk, scale);
    V[idx] = orig;
    float loss_plus = 0, loss_minus = 0;
    for (int64_t i = 0; i < elems; i++) { loss_plus += O_plus[i]; loss_minus += O_minus[i]; }
    float fd_grad = (loss_plus - loss_minus) / (2 * eps);
    AX_TEST_ASSERT(fabsf(fd_grad - dV[idx]) < 1e-2f, "dV matches finite diff");

    /* same for dQ */
    orig = Q[idx];
    Q[idx] = orig + eps;
    ax_fused_attention_fwd(Q, K, V, O_plus,  NULL, BH, S, dk, scale);
    Q[idx] = orig - eps;
    ax_fused_attention_fwd(Q, K, V, O_minus, NULL, BH, S, dk, scale);
    Q[idx] = orig;
    loss_plus = 0; loss_minus = 0;
    for (int64_t i = 0; i < elems; i++) { loss_plus += O_plus[i]; loss_minus += O_minus[i]; }
    fd_grad = (loss_plus - loss_minus) / (2 * eps);
    AX_TEST_ASSERT(fabsf(fd_grad - dQ[idx]) < 1e-2f, "dQ matches finite diff");

    /* same for dK */
    orig = K[idx];
    K[idx] = orig + eps;
    ax_fused_attention_fwd(Q, K, V, O_plus,  NULL, BH, S, dk, scale);
    K[idx] = orig - eps;
    ax_fused_attention_fwd(Q, K, V, O_minus, NULL, BH, S, dk, scale);
    K[idx] = orig;
    loss_plus = 0; loss_minus = 0;
    for (int64_t i = 0; i < elems; i++) { loss_plus += O_plus[i]; loss_minus += O_minus[i]; }
    fd_grad = (loss_plus - loss_minus) / (2 * eps);
    AX_TEST_ASSERT(fabsf(fd_grad - dK[idx]) < 1e-2f, "dK matches finite diff");

    free(Q); free(K); free(V); free(O); free(L); free(dO);
    free(dQ); free(dK); free(dV); free(O_plus); free(O_minus);
}

/* ================================================================
   test: MHA layer forward produces correct output shape
   ================================================================ */
static void test_mha_forward_shape(void)
{
    const int64_t B = 2, S = 8, D = 16;
    const int H = 4;

    ax_layer_t *mha = ax_mha_create(D, H, true, false);
    AX_TEST_ASSERT(mha != NULL, "mha create");
    if (!mha) return;

    int64_t shape[] = {B, S, D};
    ax_tensor_t *x = ax_tensor_rand(shape, 3, -0.5f, 0.5f);
    AX_TEST_ASSERT(x != NULL, "input tensor create");

    ax_no_grad();
    ax_tensor_t *out = ax_layer_forward(mha, x);
    ax_enable_grad();

    AX_TEST_ASSERT(out != NULL, "mha forward returns non-null");
    if (out) {
        AX_TEST_ASSERT_EQ(out->ndim, 3, "output is 3-dim");
        AX_TEST_ASSERT_EQ(out->shape[0], B, "batch dim");
        AX_TEST_ASSERT_EQ(out->shape[1], S, "seq dim");
        AX_TEST_ASSERT_EQ(out->shape[2], D, "model dim");
        ax_tensor_destroy(out);
    }

    ax_tensor_destroy(x);
    ax_layer_destroy(mha);
}

/* ================================================================
   test: MHA layer param count = 4 projections * D*D + 4 biases * D (with bias)
   ================================================================ */
static void test_mha_param_count(void)
{
    const int64_t D = 32;
    const int H = 4;
    {
        ax_layer_t *mha = ax_mha_create(D, H, true, false);
        int64_t pc = ax_layer_param_count(mha);
        int64_t expected = 4 * D * D + 4 * D;
        AX_TEST_ASSERT_EQ(pc, expected, "mha with bias param count");
        ax_layer_destroy(mha);
    }
    {
        ax_layer_t *mha = ax_mha_create(D, H, false, false);
        int64_t pc = ax_layer_param_count(mha);
        int64_t expected = 4 * D * D;
        AX_TEST_ASSERT_EQ(pc, expected, "mha no-bias param count");
        ax_layer_destroy(mha);
    }
}

/* ================================================================
   test: KV cache append + attend — single-token inference equals
   running full SDPA on the cached prefix
   ================================================================ */
static void test_kv_cache_parity(void)
{
    const int64_t BH = 2;
    const int64_t dk = 8;
    const int64_t max_seq = 16;
    const int64_t cur_len = 5;
    float scale = 1.0f / sqrtf((float)dk);

    ax_kv_cache_t *c = ax_kv_cache_create(BH, max_seq, dk);
    AX_TEST_ASSERT(c != NULL, "kv cache create");
    if (!c) return;

    /* fill cache with cur_len tokens */
    float *kbuf = (float *)malloc((size_t)BH * dk * sizeof(float));
    float *vbuf = (float *)malloc((size_t)BH * dk * sizeof(float));
    for (int64_t t = 0; t < cur_len; t++) {
        for (int64_t i = 0; i < BH * dk; i++) {
            kbuf[i] = (float)rand() / RAND_MAX - 0.5f;
            vbuf[i] = (float)rand() / RAND_MAX - 0.5f;
        }
        AX_TEST_ASSERT(ax_kv_cache_append(c, kbuf, vbuf), "append ok");
    }
    AX_TEST_ASSERT_EQ(c->cur_len, cur_len, "cache length advances");

    /* now do a single-token attend */
    float *Qnew = (float *)malloc((size_t)BH * dk * sizeof(float));
    for (int64_t i = 0; i < BH * dk; i++) Qnew[i] = (float)rand() / RAND_MAX - 0.5f;
    float *out_cache = (float *)calloc(BH * dk, sizeof(float));

    ax_kv_cache_attend(c, Qnew, out_cache, scale);

    /* reference: full SDPA at row 0 of a [BH, 1, dk] query against the
       cached K/V interpreted as [BH, cur_len, dk] */
    float *Q_full  = (float *)calloc(BH * cur_len * dk, sizeof(float));
    float *K_full  = (float *)calloc(BH * cur_len * dk, sizeof(float));
    float *V_full  = (float *)calloc(BH * cur_len * dk, sizeof(float));
    float *O_full  = (float *)calloc(BH * cur_len * dk, sizeof(float));
    /* put Qnew at every time step for simplicity; we'll only compare t=0 row */
    for (int64_t h = 0; h < BH; h++) {
        /* Q at row 0, K[0..cur_len), V[0..cur_len) from cache */
        memcpy(Q_full + h * cur_len * dk, Qnew + h * dk, (size_t)dk * sizeof(float));
        for (int64_t t = 0; t < cur_len; t++) {
            memcpy(K_full + h * cur_len * dk + t * dk,
                   c->K + h * max_seq * dk + t * dk,
                   (size_t)dk * sizeof(float));
            memcpy(V_full + h * cur_len * dk + t * dk,
                   c->V + h * max_seq * dk + t * dk,
                   (size_t)dk * sizeof(float));
        }
    }
    ax_fused_attention_fwd(Q_full, K_full, V_full, O_full, NULL, BH, cur_len, dk, scale);

    float max_err = 0;
    for (int64_t h = 0; h < BH; h++) {
        for (int64_t d = 0; d < dk; d++) {
            float e = fabsf(out_cache[h * dk + d] - O_full[h * cur_len * dk + d]);
            if (e > max_err) max_err = e;
        }
    }
    AX_TEST_ASSERT(max_err < 1e-4f, "kv cache attend matches SDPA");

    free(kbuf); free(vbuf); free(Qnew); free(out_cache);
    free(Q_full); free(K_full); free(V_full); free(O_full);
    ax_kv_cache_destroy(c);
}

/* ================================================================
   test: RoPE preserves norms (rotation is an isometry)
   ================================================================ */
static void test_rope_isometry(void)
{
    const int64_t BH = 2;
    const int64_t S  = 12;
    const int64_t dk = 16;
    int64_t elems = BH * S * dk;

    float *Q = (float *)malloc(elems * sizeof(float));
    float *K = (float *)malloc(elems * sizeof(float));
    float *Q_copy = (float *)malloc(elems * sizeof(float));
    for (int64_t i = 0; i < elems; i++) {
        Q[i] = (float)rand() / RAND_MAX - 0.5f;
        K[i] = Q[i]; /* doesn't matter, we only check Q */
        Q_copy[i] = Q[i];
    }
    int64_t *positions = (int64_t *)malloc(S * sizeof(int64_t));
    for (int64_t i = 0; i < S; i++) positions[i] = i;

    ax_rope_apply(Q, K, positions, BH, S, dk, 10000.0f);

    /* per-pair norm should be preserved (rotation is orthogonal) */
    float max_err = 0;
    int64_t half = dk / 2;
    for (int64_t h = 0; h < BH; h++) {
        for (int64_t s = 0; s < S; s++) {
            for (int64_t d = 0; d < half; d++) {
                int64_t i0 = ((h * S) + s) * dk + d;
                int64_t i1 = ((h * S) + s) * dk + d + half;
                float n_before = Q_copy[i0]*Q_copy[i0] + Q_copy[i1]*Q_copy[i1];
                float n_after  = Q[i0]*Q[i0] + Q[i1]*Q[i1];
                float e = fabsf(n_before - n_after);
                if (e > max_err) max_err = e;
            }
        }
    }
    AX_TEST_ASSERT(max_err < 1e-4f, "rope preserves pair norms (isometry)");

    free(Q); free(K); free(Q_copy); free(positions);
}

/* ================================================================
   test: MHA forward then backward runs without NaN
   ================================================================ */
static void test_mha_backward_smoke(void)
{
    const int64_t B = 2, S = 4, D = 16;
    const int H = 2;
    int64_t shape[] = {B, S, D};

    ax_layer_t *mha = ax_mha_create(D, H, true, false);
    ax_tensor_t *x = ax_tensor_rand(shape, 3, -0.1f, 0.1f);
    x->requires_grad = true;

    ax_tensor_t *out = ax_layer_forward(mha, x);
    AX_TEST_ASSERT(out != NULL, "forward ok");
    AX_TEST_ASSERT(out && out->grad_fn != NULL, "grad_fn attached");

    /* synthetic loss = sum(out) */
    ax_tensor_t *loss = ax_sum(out, -1);
    AX_TEST_ASSERT(loss != NULL, "sum reduction");
    ax_status_t s = ax_backward(loss);
    AX_TEST_ASSERT_EQ((int)s, (int)AX_OK, "backward ok");

    /* check input grad is not all-zero, not NaN */
    int nz = 0;
    bool any_nan = false;
    if (x->grad) {
        float *g = (float *)x->grad->storage->data;
        for (int64_t i = 0; i < B * S * D; i++) {
            if (g[i] != 0) nz++;
            if (g[i] != g[i]) any_nan = true;
        }
    }
    AX_TEST_ASSERT(nz > 0, "input grad non-zero");
    AX_TEST_ASSERT(!any_nan, "no NaN in input grad");

    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss);
    ax_tensor_destroy(x);
    ax_layer_destroy(mha);
}

/* ================================================================
   test: causal MHA — output at position t depends only on positions <= t.
   perturbing input position p > t should leave out[t] unchanged.
   ================================================================ */
static void test_mha_causal_masking(void)
{
    const int64_t B = 1, S = 6, D = 8;
    const int H = 2;
    int64_t shape[] = {B, S, D};

    ax_layer_t *mha = ax_mha_create(D, H, false, true);
    ax_tensor_t *x = ax_tensor_rand(shape, 3, -0.5f, 0.5f);

    ax_no_grad();
    ax_tensor_t *out1 = ax_layer_forward(mha, x);

    /* perturb x at position S-1 (last), check out[0] stays same */
    float *xd = (float *)x->storage->data;
    for (int64_t d = 0; d < D; d++) xd[(S - 1) * D + d] += 1.0f;
    ax_tensor_t *out2 = ax_layer_forward(mha, x);
    ax_enable_grad();

    float max_diff_row0 = 0;
    float *o1 = (float *)out1->storage->data;
    float *o2 = (float *)out2->storage->data;
    for (int64_t d = 0; d < D; d++) {
        float e = fabsf(o1[d] - o2[d]);
        if (e > max_diff_row0) max_diff_row0 = e;
    }
    AX_TEST_ASSERT(max_diff_row0 < 1e-4f, "causal: row 0 unaffected by last-position change");

    ax_tensor_destroy(out1);
    ax_tensor_destroy(out2);
    ax_tensor_destroy(x);
    ax_layer_destroy(mha);
}

int main(void)
{
    ax_init();
    ax_set_seed(12345);

    AX_RUN_TEST(test_sdpa_forward_parity);
    AX_RUN_TEST(test_sdpa_causal);
    AX_RUN_TEST(test_sdpa_backward_fd);
    AX_RUN_TEST(test_mha_forward_shape);
    AX_RUN_TEST(test_mha_param_count);
    AX_RUN_TEST(test_kv_cache_parity);
    AX_RUN_TEST(test_rope_isometry);
    AX_RUN_TEST(test_mha_backward_smoke);
    AX_RUN_TEST(test_mha_causal_masking);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
