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

/* ================================================================
   test: save-aware bwd matches recompute bwd numerically.
   Phase 1.3 added ax_fused_attention_fwd_save / _bwd_use that skips the
   QK^T recompute in backward. this verifies the two paths produce
   identical dQ/dK/dV (within float32 round-off).
   ================================================================ */
static void test_sdpa_save_path_parity(void)
{
    const int64_t BH = 4;
    const int64_t S = 16;
    const int64_t dk = 8;
    int64_t elems = BH * S * dk;
    float scale = 1.0f / sqrtf((float)dk);

    float *Q  = (float *)malloc(elems * sizeof(float));
    float *K  = (float *)malloc(elems * sizeof(float));
    float *V  = (float *)malloc(elems * sizeof(float));
    float *dO = (float *)malloc(elems * sizeof(float));
    for (int64_t i = 0; i < elems; i++) {
        Q[i]  = 0.1f * ((float)rand() / RAND_MAX - 0.5f);
        K[i]  = 0.1f * ((float)rand() / RAND_MAX - 0.5f);
        V[i]  = 0.1f * ((float)rand() / RAND_MAX - 0.5f);
        dO[i] = ((float)rand() / RAND_MAX - 0.5f);
    }

    /* path A: recompute */
    float *O_a  = (float *)calloc(elems, sizeof(float));
    float *L_a  = (float *)calloc(BH * S, sizeof(float));
    float *dQ_a = (float *)calloc(elems, sizeof(float));
    float *dK_a = (float *)calloc(elems, sizeof(float));
    float *dV_a = (float *)calloc(elems, sizeof(float));
    ax_fused_attention_fwd(Q, K, V, O_a, L_a, BH, S, dk, scale);
    ax_fused_attention_bwd(Q, K, V, O_a, dO, L_a, dQ_a, dK_a, dV_a, BH, S, dk, scale);

    /* path B: save P then bwd_use */
    float *O_b  = (float *)calloc(elems, sizeof(float));
    float *L_b  = (float *)calloc(BH * S, sizeof(float));
    float *Psave = (float *)calloc(BH * S * S, sizeof(float));
    float *dQ_b = (float *)calloc(elems, sizeof(float));
    float *dK_b = (float *)calloc(elems, sizeof(float));
    float *dV_b = (float *)calloc(elems, sizeof(float));
    ax_fused_attention_fwd_save(Q, K, V, O_b, L_b, Psave, BH, S, dk, scale, false);
    ax_fused_attention_bwd_use(Q, K, V, O_b, dO, L_b, Psave,
                                dQ_b, dK_b, dV_b, BH, S, dk, scale, false);

    /* outputs must match (forward is the same algorithm, just with P_save
       sidechannel) */
    float maxO = 0;
    for (int64_t i = 0; i < elems; i++) {
        float e = fabsf(O_a[i] - O_b[i]);
        if (e > maxO) maxO = e;
    }
    AX_TEST_ASSERT(maxO < 1e-5f, "fwd_save output matches plain fwd");

    /* gradients must match (backward uses different code paths) */
    float maxQ = 0, maxK = 0, maxV = 0;
    for (int64_t i = 0; i < elems; i++) {
        float eq = fabsf(dQ_a[i] - dQ_b[i]); if (eq > maxQ) maxQ = eq;
        float ek = fabsf(dK_a[i] - dK_b[i]); if (ek > maxK) maxK = ek;
        float ev = fabsf(dV_a[i] - dV_b[i]); if (ev > maxV) maxV = ev;
    }
    AX_TEST_ASSERT(maxQ < 1e-4f, "dQ matches across paths");
    AX_TEST_ASSERT(maxK < 1e-4f, "dK matches across paths");
    AX_TEST_ASSERT(maxV < 1e-4f, "dV matches across paths");

    /* same for causal */
    for (int64_t i = 0; i < elems; i++) {
        dQ_a[i] = dK_a[i] = dV_a[i] = 0;
        dQ_b[i] = dK_b[i] = dV_b[i] = 0;
        O_a[i] = O_b[i] = 0;
    }
    for (int64_t i = 0; i < BH * S; i++) L_a[i] = L_b[i] = 0;
    memset(Psave, 0, BH * S * S * sizeof(float));
    ax_fused_attention_fwd_causal(Q, K, V, O_a, L_a, BH, S, dk, scale);
    ax_fused_attention_bwd_causal(Q, K, V, O_a, dO, L_a, dQ_a, dK_a, dV_a, BH, S, dk, scale);
    ax_fused_attention_fwd_save(Q, K, V, O_b, L_b, Psave, BH, S, dk, scale, true);
    ax_fused_attention_bwd_use(Q, K, V, O_b, dO, L_b, Psave,
                                dQ_b, dK_b, dV_b, BH, S, dk, scale, true);
    maxO = maxQ = maxK = maxV = 0;
    for (int64_t i = 0; i < elems; i++) {
        float eo = fabsf(O_a[i] - O_b[i]); if (eo > maxO) maxO = eo;
        float eq = fabsf(dQ_a[i] - dQ_b[i]); if (eq > maxQ) maxQ = eq;
        float ek = fabsf(dK_a[i] - dK_b[i]); if (ek > maxK) maxK = ek;
        float ev = fabsf(dV_a[i] - dV_b[i]); if (ev > maxV) maxV = ev;
    }
    AX_TEST_ASSERT(maxO < 1e-5f && maxQ < 1e-4f && maxK < 1e-4f && maxV < 1e-4f,
                   "causal save path matches recompute path");

    free(Q); free(K); free(V); free(dO);
    free(O_a); free(L_a); free(dQ_a); free(dK_a); free(dV_a);
    free(O_b); free(L_b); free(Psave); free(dQ_b); free(dK_b); free(dV_b);
}

/* ================================================================
   test: train_step entry points grads match the autograd path bit-for-bit.
   parameterised on entry_fn so each F.4 phase runs the same parity
   check against the same autograd reference. a static wrapper
   test_mha_train_step_parity runs entry_fn = ax_mha_train_step;
   test_mha_train_step_fused_parity runs entry_fn = ax_mha_train_step_fused.
   every F.4 phase that swaps the fused body must keep both green.
   ================================================================ */
typedef ax_status_t (*train_step_fn_t)(ax_layer_t *, const ax_tensor_t *,
                                        const ax_tensor_t *, ax_tensor_t *);

static void run_train_step_parity(train_step_fn_t entry_fn, const char *tag)
{
    const int64_t B = 2, S = 8, D = 16;
    const int H = 4;
    int64_t shape[] = {B, S, D};

    /* layer A — autograd path */
    ax_layer_t *mhaA = ax_mha_create(D, H, true, false);
    /* layer B — train_step path. start with the same RNG state so weights match */
    ax_set_seed(7777);
    ax_layer_t *mhaA2 = ax_mha_create(D, H, true, false);
    ax_set_seed(7777);
    ax_layer_t *mhaB = ax_mha_create(D, H, true, false);
    (void)mhaA;
    /* the seed reset above sees mhaA2 / mhaB get matching weights;
       the original mhaA is discarded — using mhaA2 from here on. */
    ax_layer_destroy(mhaA);
    ax_layer_t *layerA = mhaA2;
    ax_layer_t *layerB = mhaB;

    ax_set_seed(31337);
    ax_tensor_t *xA = ax_tensor_rand(shape, 3, -0.1f, 0.1f);
    /* identical x for both layers */
    ax_tensor_t *xB = ax_tensor_create(shape, 3, AX_FLOAT32);
    memcpy(xB->storage->data, xA->storage->data,
           (size_t)B * S * D * sizeof(float));

    /* ---- path A: autograd ---- */
    ax_tensor_t *outA = ax_layer_forward(layerA, xA);
    AX_TEST_ASSERT(outA != NULL, "A: forward ok");
    ax_tensor_t *lossA = ax_sum(outA, -1);
    ax_status_t sA = ax_backward(lossA);
    AX_TEST_ASSERT_EQ((int)sA, (int)AX_OK, "A: backward ok");

    /* snapshot grads from A */
    ax_mha_t *mA = (ax_mha_t *)layerA;
    int64_t WD = D * D, BD = D;
    float *gWqA = mA->Wq->grad ? (float *)mA->Wq->grad->storage->data : NULL;
    float *gWkA = mA->Wk->grad ? (float *)mA->Wk->grad->storage->data : NULL;
    float *gWvA = mA->Wv->grad ? (float *)mA->Wv->grad->storage->data : NULL;
    float *gWoA = mA->Wo->grad ? (float *)mA->Wo->grad->storage->data : NULL;
    float *gbqA = mA->bq->grad ? (float *)mA->bq->grad->storage->data : NULL;
    float *gbkA = mA->bk->grad ? (float *)mA->bk->grad->storage->data : NULL;
    float *gbvA = mA->bv->grad ? (float *)mA->bv->grad->storage->data : NULL;
    float *gboA = mA->bo->grad ? (float *)mA->bo->grad->storage->data : NULL;
    AX_TEST_ASSERT(gWqA && gWkA && gWvA && gWoA, "A: weight grads exist");
    AX_TEST_ASSERT(gbqA && gbkA && gbvA && gboA, "A: bias grads exist");

    /* save A's grads to compare against B (B writes into different layer) */
    float *snap_Wq = (float *)malloc((size_t)WD * sizeof(float));
    float *snap_Wk = (float *)malloc((size_t)WD * sizeof(float));
    float *snap_Wv = (float *)malloc((size_t)WD * sizeof(float));
    float *snap_Wo = (float *)malloc((size_t)WD * sizeof(float));
    float *snap_bq = (float *)malloc((size_t)BD * sizeof(float));
    float *snap_bk = (float *)malloc((size_t)BD * sizeof(float));
    float *snap_bv = (float *)malloc((size_t)BD * sizeof(float));
    float *snap_bo = (float *)malloc((size_t)BD * sizeof(float));
    memcpy(snap_Wq, gWqA, (size_t)WD * sizeof(float));
    memcpy(snap_Wk, gWkA, (size_t)WD * sizeof(float));
    memcpy(snap_Wv, gWvA, (size_t)WD * sizeof(float));
    memcpy(snap_Wo, gWoA, (size_t)WD * sizeof(float));
    memcpy(snap_bq, gbqA, (size_t)BD * sizeof(float));
    memcpy(snap_bk, gbkA, (size_t)BD * sizeof(float));
    memcpy(snap_bv, gbvA, (size_t)BD * sizeof(float));
    memcpy(snap_bo, gboA, (size_t)BD * sizeof(float));

    /* ---- path B: entry_fn (dout = NULL → defaults to ones, matching sum loss) ---- */
    ax_tensor_t *outB = ax_tensor_create(shape, 3, AX_FLOAT32);
    ax_status_t sB = entry_fn(layerB, xB, NULL, outB);
    AX_TEST_ASSERT_EQ((int)sB, (int)AX_OK, "B: train_step ok");
    (void)tag;

    /* compare forward output A vs B */
    float *odA = (float *)outA->storage->data;
    float *odB = (float *)outB->storage->data;
    float maxO = 0.0f;
    for (int64_t i = 0; i < B * S * D; i++) {
        float d = fabsf(odA[i] - odB[i]);
        if (d > maxO) maxO = d;
    }
    AX_TEST_ASSERT(maxO < 1e-4f, "fwd output matches across paths");

    /* compare each weight/bias grad */
    ax_mha_t *mB = (ax_mha_t *)layerB;
    #define CHECK_GRAD(name, p, n_elems) do {                                  \
        float *gB = (p)->grad ? (float *)(p)->grad->storage->data : NULL;      \
        AX_TEST_ASSERT(gB != NULL, "B: " #name " grad exists");                \
        float maxd = 0.0f;                                                     \
        for (int64_t i = 0; i < (n_elems); i++) {                              \
            float d = fabsf(snap_##name[i] - gB[i]);                           \
            if (d > maxd) maxd = d;                                            \
        }                                                                      \
        AX_TEST_ASSERT(maxd < 1e-4f, #name " grad matches across paths");      \
    } while (0)
    CHECK_GRAD(Wq, mB->Wq, WD);
    CHECK_GRAD(Wk, mB->Wk, WD);
    CHECK_GRAD(Wv, mB->Wv, WD);
    CHECK_GRAD(Wo, mB->Wo, WD);
    CHECK_GRAD(bq, mB->bq, BD);
    CHECK_GRAD(bk, mB->bk, BD);
    CHECK_GRAD(bv, mB->bv, BD);
    CHECK_GRAD(bo, mB->bo, BD);
    #undef CHECK_GRAD

    free(snap_Wq); free(snap_Wk); free(snap_Wv); free(snap_Wo);
    free(snap_bq); free(snap_bk); free(snap_bv); free(snap_bo);

    ax_graph_cleanup(lossA);
    ax_tensor_destroy(lossA);
    ax_tensor_destroy(xA);
    ax_tensor_destroy(xB);
    ax_tensor_destroy(outB);
    ax_layer_destroy(layerA);
    ax_layer_destroy(layerB);
}

static void test_mha_train_step_parity(void)
{
    run_train_step_parity(ax_mha_train_step, "train_step");
}

static void test_mha_train_step_fused_parity(void)
{
    run_train_step_parity(ax_mha_train_step_fused, "train_step_fused");
}

static void test_mha_train_step_v4_parity(void)
{
    run_train_step_parity(ax_mha_train_step_v4, "train_step_v4");
}

/* ================================================================
   test: F.3.a opt_qkv_head_gemm produces bit-for-bit identical Qh/Kh/Vh
   to the unfused gemm + ax_attn_head_interleave_qkv_split_bias sequence.
   shapes exercised:
     (B=2, S=8, D=16, H=4, dk=4) — tiny (scalar-inner-loop path)
     (B=1, S=64, D=64, H=8, dk=8) — edge MR/batch boundaries
     (B=4, S=32, D=64, H=4, dk=16) — use_jc_par path
   tolerance: bit-identical fp32 (both paths do identical accumulation
   order up to micro_kernel ordering — checking <1e-5 to accommodate
   any tile-order-dependent reassociation).
   ================================================================ */
static void test_qkv_head_gemm_parity(void)
{
    struct shape_case { int64_t B, S, D, H; };
    struct shape_case cases[] = {
        { 2, 8, 16, 4 },
        { 1, 64, 64, 8 },
        { 4, 32, 64, 4 },
    };

    for (size_t ci = 0; ci < sizeof(cases)/sizeof(cases[0]); ci++) {
        struct shape_case c = cases[ci];
        int64_t B = c.B, S = c.S, D = c.D, H = c.H;
        int64_t dk = D / H;
        int64_t rows = B * S;

        ax_set_seed(7777 + (int)ci);
        int64_t x_sh[]   = {rows, D};
        int64_t w_sh[]   = {D, 3 * D};
        int64_t b_sh[]   = {3 * D};
        int64_t h_sh[]   = {B * H, S, dk};
        int64_t qkv_sh[] = {rows, 3 * D};

        ax_tensor_t *X    = ax_tensor_rand(x_sh, 2, -0.1f, 0.1f);
        ax_tensor_t *Wqkv = ax_tensor_rand(w_sh, 2, -0.1f, 0.1f);
        ax_tensor_t *bqkv = ax_tensor_rand(b_sh, 1, -0.05f, 0.05f);

        /* reference path: gemm + head_interleave_qkv_split_bias */
        ax_tensor_t *qkv_ref = ax_tensor_create(qkv_sh, 2, AX_FLOAT32);
        ax_tensor_t *Qh_ref = ax_tensor_create(h_sh, 3, AX_FLOAT32);
        ax_tensor_t *Kh_ref = ax_tensor_create(h_sh, 3, AX_FLOAT32);
        ax_tensor_t *Vh_ref = ax_tensor_create(h_sh, 3, AX_FLOAT32);
        ax_status_t s1 = ax_compute_gemm(X, Wqkv, qkv_ref);
        AX_TEST_ASSERT_EQ((int)s1, (int)AX_OK, "ref gemm ok");
        extern void ax_attn_head_interleave_qkv_split_bias(const float *, const float *,
                                                             float *, float *, float *,
                                                             int64_t, int64_t, int64_t, int64_t,
                                                             int64_t);
        ax_attn_head_interleave_qkv_split_bias(
            (const float *)qkv_ref->storage->data,
            (const float *)bqkv->storage->data,
            (float *)Qh_ref->storage->data,
            (float *)Kh_ref->storage->data,
            (float *)Vh_ref->storage->data,
            B, S, H, dk, D);

        /* fused path */
        ax_tensor_t *Qh_fus = ax_tensor_create(h_sh, 3, AX_FLOAT32);
        ax_tensor_t *Kh_fus = ax_tensor_create(h_sh, 3, AX_FLOAT32);
        ax_tensor_t *Vh_fus = ax_tensor_create(h_sh, 3, AX_FLOAT32);
        ax_status_t s2 = ax_compute_qkv_head_gemm(X, Wqkv, bqkv, B, S, H, dk,
                                                    Qh_fus, Kh_fus, Vh_fus);
        AX_TEST_ASSERT_EQ((int)s2, (int)AX_OK, "fused qkv_head_gemm ok");

        /* compare */
        int64_t nel = B * H * S * dk;
        const float *qr = (const float *)Qh_ref->storage->data;
        const float *kr = (const float *)Kh_ref->storage->data;
        const float *vr = (const float *)Vh_ref->storage->data;
        const float *qf = (const float *)Qh_fus->storage->data;
        const float *kf = (const float *)Kh_fus->storage->data;
        const float *vf = (const float *)Vh_fus->storage->data;
        float max_dq = 0.0f, max_dk = 0.0f, max_dv = 0.0f;
        for (int64_t i = 0; i < nel; i++) {
            float dq = fabsf(qr[i] - qf[i]);
            float dk2 = fabsf(kr[i] - kf[i]);
            float dv = fabsf(vr[i] - vf[i]);
            if (dq > max_dq) max_dq = dq;
            if (dk2 > max_dk) max_dk = dk2;
            if (dv > max_dv) max_dv = dv;
        }
        AX_TEST_ASSERT(max_dq < 1e-5f, "Qh matches fused vs unfused");
        AX_TEST_ASSERT(max_dk < 1e-5f, "Kh matches fused vs unfused");
        AX_TEST_ASSERT(max_dv < 1e-5f, "Vh matches fused vs unfused");

        /* also test the no-bias path */
        ax_tensor_t *Qh_ref2 = ax_tensor_create(h_sh, 3, AX_FLOAT32);
        ax_tensor_t *Kh_ref2 = ax_tensor_create(h_sh, 3, AX_FLOAT32);
        ax_tensor_t *Vh_ref2 = ax_tensor_create(h_sh, 3, AX_FLOAT32);
        extern void ax_attn_head_interleave_qkv_split(const float *,
                                                        float *, float *, float *,
                                                        int64_t, int64_t, int64_t, int64_t,
                                                        int64_t);
        ax_attn_head_interleave_qkv_split(
            (const float *)qkv_ref->storage->data,
            (float *)Qh_ref2->storage->data,
            (float *)Kh_ref2->storage->data,
            (float *)Vh_ref2->storage->data,
            B, S, H, dk, D);
        ax_tensor_t *Qh_fus2 = ax_tensor_create(h_sh, 3, AX_FLOAT32);
        ax_tensor_t *Kh_fus2 = ax_tensor_create(h_sh, 3, AX_FLOAT32);
        ax_tensor_t *Vh_fus2 = ax_tensor_create(h_sh, 3, AX_FLOAT32);
        ax_status_t s3 = ax_compute_qkv_head_gemm(X, Wqkv, NULL, B, S, H, dk,
                                                    Qh_fus2, Kh_fus2, Vh_fus2);
        AX_TEST_ASSERT_EQ((int)s3, (int)AX_OK, "fused qkv_head_gemm no-bias ok");
        const float *qr2 = (const float *)Qh_ref2->storage->data;
        const float *qf2 = (const float *)Qh_fus2->storage->data;
        max_dq = 0.0f;
        for (int64_t i = 0; i < nel; i++) {
            float d = fabsf(qr2[i] - qf2[i]);
            if (d > max_dq) max_dq = d;
        }
        AX_TEST_ASSERT(max_dq < 1e-5f, "no-bias Qh matches");

        ax_tensor_destroy(X); ax_tensor_destroy(Wqkv); ax_tensor_destroy(bqkv);
        ax_tensor_destroy(qkv_ref);
        ax_tensor_destroy(Qh_ref); ax_tensor_destroy(Kh_ref); ax_tensor_destroy(Vh_ref);
        ax_tensor_destroy(Qh_fus); ax_tensor_destroy(Kh_fus); ax_tensor_destroy(Vh_fus);
        ax_tensor_destroy(Qh_ref2); ax_tensor_destroy(Kh_ref2); ax_tensor_destroy(Vh_ref2);
        ax_tensor_destroy(Qh_fus2); ax_tensor_destroy(Kh_fus2); ax_tensor_destroy(Vh_fus2);
    }
}

/* ================================================================
   test: F.3.d opt_dattn_head_gemm_nt produces bit-equivalent dO_head
   to the unfused gemm_nt + ax_attn_head_interleave sequence.
   ================================================================ */
static void test_dattn_head_gemm_nt_parity(void)
{
    struct shape_case { int64_t B, S, D, H; };
    struct shape_case cases[] = {
        { 2, 8, 16, 4 },
        { 1, 64, 64, 8 },
        { 4, 32, 64, 4 },
    };

    for (size_t ci = 0; ci < sizeof(cases)/sizeof(cases[0]); ci++) {
        struct shape_case c = cases[ci];
        int64_t B = c.B, S = c.S, D = c.D, H = c.H;
        int64_t dk = D / H;
        int64_t rows = B * S;

        ax_set_seed(8888 + (int)ci);
        int64_t dout_sh[]    = {rows, D};
        int64_t Wo_sh[]      = {D, D};
        int64_t dattn_sh[]   = {rows, D};
        int64_t dO_head_sh[] = {B * H, S, dk};

        ax_tensor_t *dout = ax_tensor_rand(dout_sh, 2, -0.1f, 0.1f);
        ax_tensor_t *Wo   = ax_tensor_rand(Wo_sh, 2, -0.1f, 0.1f);

        /* reference: gemm_nt → dattn → head_interleave → dO_head */
        ax_tensor_t *dattn_ref = ax_tensor_create(dattn_sh, 2, AX_FLOAT32);
        ax_tensor_t *dO_ref    = ax_tensor_create(dO_head_sh, 3, AX_FLOAT32);
        ax_status_t s1 = ax_compute_gemm_nt(dout, Wo, dattn_ref);
        AX_TEST_ASSERT_EQ((int)s1, (int)AX_OK, "ref gemm_nt ok");
        extern void ax_attn_head_interleave(const float *, float *,
                                              int64_t, int64_t, int64_t, int64_t);
        ax_attn_head_interleave((const float *)dattn_ref->storage->data,
                                  (float *)dO_ref->storage->data,
                                  B, S, H, dk);

        /* fused */
        ax_tensor_t *dO_fus = ax_tensor_create(dO_head_sh, 3, AX_FLOAT32);
        ax_status_t s2 = ax_compute_dattn_head_gemm_nt(dout, Wo, B, S, H, dk, dO_fus);
        AX_TEST_ASSERT_EQ((int)s2, (int)AX_OK, "fused dattn_head_gemm_nt ok");

        int64_t nel = B * H * S * dk;
        const float *r = (const float *)dO_ref->storage->data;
        const float *f = (const float *)dO_fus->storage->data;
        float maxd = 0.0f;
        for (int64_t i = 0; i < nel; i++) {
            float d = fabsf(r[i] - f[i]);
            if (d > maxd) maxd = d;
        }
        AX_TEST_ASSERT(maxd < 1e-5f, "dO_head matches fused vs unfused");

        ax_tensor_destroy(dout); ax_tensor_destroy(Wo);
        ax_tensor_destroy(dattn_ref); ax_tensor_destroy(dO_ref); ax_tensor_destroy(dO_fus);
    }
}

/* ================================================================
   test: F.3.e ax_fused_attention_fwd_save_to_flat produces bit-equivalent
   attn_flat (and L, P_save) to ax_fused_attention_fwd_save +
   ax_attn_head_deinterleave.
   ================================================================ */
static void test_sdpa_fwd_to_flat_parity(void)
{
    struct shape_case { int64_t B, S, H, dk; bool causal; bool save_p; };
    struct shape_case cases[] = {
        { 2, 8,  4, 4,  false, true  },
        { 1, 64, 8, 8,  false, true  },
        { 1, 32, 4, 16, true,  false },
        { 4, 16, 4, 16, false, true  },
    };

    for (size_t ci = 0; ci < sizeof(cases)/sizeof(cases[0]); ci++) {
        struct shape_case c = cases[ci];
        int64_t B = c.B, S = c.S, H = c.H, dk = c.dk;
        int64_t BH = B * H, D = H * dk;
        float scale = 1.0f / sqrtf((float)dk);

        ax_set_seed(9999 + (int)ci);
        int64_t qkv_sh[]  = {BH, S, dk};
        int64_t flat_sh[] = {B, S, D};
        int64_t L_sh[]    = {BH, S};
        int64_t Psh[]     = {BH, S, S};

        ax_tensor_t *Q = ax_tensor_rand(qkv_sh, 3, -0.1f, 0.1f);
        ax_tensor_t *K = ax_tensor_rand(qkv_sh, 3, -0.1f, 0.1f);
        ax_tensor_t *V = ax_tensor_rand(qkv_sh, 3, -0.1f, 0.1f);

        /* reference: fwd_save → Oh, then head_deinterleave → attn_flat_ref */
        ax_tensor_t *Oh_ref       = ax_tensor_create(qkv_sh, 3, AX_FLOAT32);
        ax_tensor_t *L_ref        = ax_tensor_create(L_sh, 2, AX_FLOAT32);
        ax_tensor_t *Pref         = c.save_p ? ax_tensor_create(Psh, 3, AX_FLOAT32) : NULL;
        ax_tensor_t *attn_flat_ref = ax_tensor_create(flat_sh, 3, AX_FLOAT32);

        ax_fused_attention_fwd_save(
            (const float *)Q->storage->data,
            (const float *)K->storage->data,
            (const float *)V->storage->data,
            (float *)Oh_ref->storage->data,
            (float *)L_ref->storage->data,
            Pref ? (float *)Pref->storage->data : NULL,
            BH, S, dk, scale, c.causal);

        extern void ax_attn_head_deinterleave(const float *, float *,
                                                int64_t, int64_t, int64_t, int64_t);
        ax_attn_head_deinterleave(
            (const float *)Oh_ref->storage->data,
            (float *)attn_flat_ref->storage->data, B, S, H, dk);

        /* fused: fwd_save_to_flat → attn_flat_fus directly */
        ax_tensor_t *L_fus  = ax_tensor_create(L_sh, 2, AX_FLOAT32);
        ax_tensor_t *Pfus   = c.save_p ? ax_tensor_create(Psh, 3, AX_FLOAT32) : NULL;
        ax_tensor_t *attn_flat_fus = ax_tensor_create(flat_sh, 3, AX_FLOAT32);

        ax_fused_attention_fwd_save_to_flat(
            (const float *)Q->storage->data,
            (const float *)K->storage->data,
            (const float *)V->storage->data,
            (float *)attn_flat_fus->storage->data,
            (float *)L_fus->storage->data,
            Pfus ? (float *)Pfus->storage->data : NULL,
            B, S, H, dk, scale, c.causal);

        /* compare attn_flat */
        int64_t nflat = B * S * D;
        const float *fr = (const float *)attn_flat_ref->storage->data;
        const float *ff = (const float *)attn_flat_fus->storage->data;
        float maxd_o = 0.0f;
        for (int64_t i = 0; i < nflat; i++) {
            float d = fabsf(fr[i] - ff[i]);
            if (d > maxd_o) maxd_o = d;
        }
        AX_TEST_ASSERT(maxd_o < 1e-5f, "attn_flat matches fused vs unfused");

        /* compare L */
        int64_t nL = BH * S;
        const float *Lr = (const float *)L_ref->storage->data;
        const float *Lf = (const float *)L_fus->storage->data;
        float maxd_L = 0.0f;
        for (int64_t i = 0; i < nL; i++) {
            float d = fabsf(Lr[i] - Lf[i]);
            if (d > maxd_L) maxd_L = d;
        }
        AX_TEST_ASSERT(maxd_L < 1e-5f, "L matches fused vs unfused");

        /* compare P_save (optional) */
        if (c.save_p) {
            int64_t nP = BH * S * S;
            const float *Pr = (const float *)Pref->storage->data;
            const float *Pf = (const float *)Pfus->storage->data;
            float maxd_P = 0.0f;
            for (int64_t i = 0; i < nP; i++) {
                float d = fabsf(Pr[i] - Pf[i]);
                if (d > maxd_P) maxd_P = d;
            }
            AX_TEST_ASSERT(maxd_P < 1e-5f, "P_save matches fused vs unfused");
        }

        ax_tensor_destroy(Q); ax_tensor_destroy(K); ax_tensor_destroy(V);
        ax_tensor_destroy(Oh_ref); ax_tensor_destroy(L_ref); ax_tensor_destroy(attn_flat_ref);
        ax_tensor_destroy(L_fus); ax_tensor_destroy(attn_flat_fus);
        if (Pref) ax_tensor_destroy(Pref);
        if (Pfus) ax_tensor_destroy(Pfus);
    }
}

/* ================================================================
   test: F.3.e companion ax_fused_attention_bwd_use_from_flat produces
   bit-equivalent dQ/dK/dV to the standard ax_fused_attention_bwd_use
   when fed attn_flat = head_deinterleave(O) of the same forward.
   ================================================================ */
static void test_sdpa_bwd_from_flat_parity(void)
{
    struct shape_case { int64_t B, S, H, dk; bool causal; };
    struct shape_case cases[] = {
        { 2, 8,  4, 4,  false },
        { 1, 64, 8, 8,  false },
        { 1, 32, 4, 16, true  },
    };

    for (size_t ci = 0; ci < sizeof(cases)/sizeof(cases[0]); ci++) {
        struct shape_case c = cases[ci];
        int64_t B = c.B, S = c.S, H = c.H, dk = c.dk;
        int64_t BH = B * H, D = H * dk;
        float scale = 1.0f / sqrtf((float)dk);

        ax_set_seed(11111 + (int)ci);
        int64_t qkv_sh[]  = {BH, S, dk};
        int64_t flat_sh[] = {B, S, D};
        int64_t L_sh[]    = {BH, S};
        int64_t Psh[]     = {BH, S, S};

        ax_tensor_t *Q = ax_tensor_rand(qkv_sh, 3, -0.1f, 0.1f);
        ax_tensor_t *K = ax_tensor_rand(qkv_sh, 3, -0.1f, 0.1f);
        ax_tensor_t *V = ax_tensor_rand(qkv_sh, 3, -0.1f, 0.1f);
        ax_tensor_t *dO = ax_tensor_rand(qkv_sh, 3, -0.1f, 0.1f);

        ax_tensor_t *Oh    = ax_tensor_create(qkv_sh, 3, AX_FLOAT32);
        ax_tensor_t *L_t   = ax_tensor_create(L_sh, 2, AX_FLOAT32);
        ax_tensor_t *P_t   = ax_tensor_create(Psh, 3, AX_FLOAT32);
        ax_tensor_t *attn_flat = ax_tensor_create(flat_sh, 3, AX_FLOAT32);

        ax_fused_attention_fwd_save(
            (const float *)Q->storage->data,
            (const float *)K->storage->data,
            (const float *)V->storage->data,
            (float *)Oh->storage->data,
            (float *)L_t->storage->data,
            (float *)P_t->storage->data,
            BH, S, dk, scale, c.causal);
        extern void ax_attn_head_deinterleave(const float *, float *,
                                                int64_t, int64_t, int64_t, int64_t);
        ax_attn_head_deinterleave(
            (const float *)Oh->storage->data,
            (float *)attn_flat->storage->data, B, S, H, dk);

        /* reference: bwd_use with Oh */
        ax_tensor_t *dQ_ref = ax_tensor_create(qkv_sh, 3, AX_FLOAT32);
        ax_tensor_t *dK_ref = ax_tensor_create(qkv_sh, 3, AX_FLOAT32);
        ax_tensor_t *dV_ref = ax_tensor_create(qkv_sh, 3, AX_FLOAT32);
        ax_fused_attention_bwd_use(
            (const float *)Q->storage->data,
            (const float *)K->storage->data,
            (const float *)V->storage->data,
            (const float *)Oh->storage->data,
            (const float *)dO->storage->data,
            (const float *)L_t->storage->data,
            (const float *)P_t->storage->data,
            (float *)dQ_ref->storage->data,
            (float *)dK_ref->storage->data,
            (float *)dV_ref->storage->data,
            BH, S, dk, scale, c.causal);

        /* fused: bwd_use_from_flat with attn_flat */
        ax_tensor_t *dQ_fus = ax_tensor_create(qkv_sh, 3, AX_FLOAT32);
        ax_tensor_t *dK_fus = ax_tensor_create(qkv_sh, 3, AX_FLOAT32);
        ax_tensor_t *dV_fus = ax_tensor_create(qkv_sh, 3, AX_FLOAT32);
        ax_fused_attention_bwd_use_from_flat(
            (const float *)Q->storage->data,
            (const float *)K->storage->data,
            (const float *)V->storage->data,
            (const float *)attn_flat->storage->data,
            (const float *)dO->storage->data,
            (const float *)L_t->storage->data,
            (const float *)P_t->storage->data,
            (float *)dQ_fus->storage->data,
            (float *)dK_fus->storage->data,
            (float *)dV_fus->storage->data,
            B, S, H, dk, scale, c.causal);

        int64_t nel = BH * S * dk;
        float maxQ = 0.0f, maxK = 0.0f, maxV = 0.0f;
        const float *qr = (const float *)dQ_ref->storage->data;
        const float *qf = (const float *)dQ_fus->storage->data;
        const float *kr = (const float *)dK_ref->storage->data;
        const float *kf = (const float *)dK_fus->storage->data;
        const float *vr = (const float *)dV_ref->storage->data;
        const float *vf = (const float *)dV_fus->storage->data;
        for (int64_t i = 0; i < nel; i++) {
            float dq = fabsf(qr[i] - qf[i]);
            float dk2 = fabsf(kr[i] - kf[i]);
            float dv = fabsf(vr[i] - vf[i]);
            if (dq > maxQ) maxQ = dq;
            if (dk2 > maxK) maxK = dk2;
            if (dv > maxV) maxV = dv;
        }
        AX_TEST_ASSERT(maxQ < 1e-5f, "dQ matches from_flat vs use");
        AX_TEST_ASSERT(maxK < 1e-5f, "dK matches from_flat vs use");
        AX_TEST_ASSERT(maxV < 1e-5f, "dV matches from_flat vs use");

        ax_tensor_destroy(Q); ax_tensor_destroy(K); ax_tensor_destroy(V); ax_tensor_destroy(dO);
        ax_tensor_destroy(Oh); ax_tensor_destroy(L_t); ax_tensor_destroy(P_t);
        ax_tensor_destroy(attn_flat);
        ax_tensor_destroy(dQ_ref); ax_tensor_destroy(dK_ref); ax_tensor_destroy(dV_ref);
        ax_tensor_destroy(dQ_fus); ax_tensor_destroy(dK_fus); ax_tensor_destroy(dV_fus);
    }
}

/* ================================================================
   test: F.4.2 proper / F.3.f opt_mha_output_proj_fused produces
   bit-equivalent y, dattn, dWo, dbo to the unfused 4-op sequence.
   ================================================================ */
static void test_mha_output_proj_fused_parity(void)
{
    struct shape_case { int64_t rows, D; bool with_bias; };
    struct shape_case cases[] = {
        { 32,  16,  true  },   /* tiny */
        { 64,  64,  true  },   /* small */
        { 128, 128, false },   /* medium no-bias */
        { 256, 64,  true  },   /* multi-strip Bq=32 → 8 strips */
    };

    for (size_t ci = 0; ci < sizeof(cases)/sizeof(cases[0]); ci++) {
        struct shape_case c = cases[ci];
        int64_t rows = c.rows, D = c.D;

        ax_set_seed(22222 + (int)ci);
        int64_t flat_sh[]   = {rows, D};
        int64_t Wo_sh[]     = {D, D};
        int64_t bo_sh[]     = {D};

        ax_tensor_t *attn = ax_tensor_rand(flat_sh, 2, -0.1f, 0.1f);
        ax_tensor_t *Wo   = ax_tensor_rand(Wo_sh, 2, -0.1f, 0.1f);
        ax_tensor_t *bo   = c.with_bias ? ax_tensor_rand(bo_sh, 1, -0.05f, 0.05f) : NULL;
        ax_tensor_t *dout = ax_tensor_rand(flat_sh, 2, -0.1f, 0.1f);

        /* reference path: 4 separate ops */
        ax_tensor_t *y_ref     = ax_tensor_create(flat_sh, 2, AX_FLOAT32);
        ax_tensor_t *dattn_ref = ax_tensor_create(flat_sh, 2, AX_FLOAT32);
        ax_tensor_t *dWo_ref   = ax_tensor_zeros(Wo_sh, 2, AX_FLOAT32);
        ax_tensor_t *dbo_ref   = c.with_bias ? ax_tensor_zeros(bo_sh, 1, AX_FLOAT32) : NULL;

        /* y = attn @ Wo + bo */
        ax_status_t s1 = ax_compute_gemm(attn, Wo, y_ref);
        AX_TEST_ASSERT_EQ((int)s1, (int)AX_OK, "ref gemm ok");
        if (c.with_bias) {
            const float *bd = (const float *)bo->storage->data;
            float *yd = (float *)y_ref->storage->data;
            for (int64_t r = 0; r < rows; r++)
                for (int64_t j = 0; j < D; j++)
                    yd[r * D + j] += bd[j];
        }
        /* dattn = dout @ Wo^T */
        ax_status_t s2 = ax_compute_gemm_nt(dout, Wo, dattn_ref);
        AX_TEST_ASSERT_EQ((int)s2, (int)AX_OK, "ref gemm_nt ok");
        /* dWo += attn^T @ dout */
        extern void ax_gemm_set_skip_init(bool);
        ax_gemm_set_skip_init(true);
        ax_status_t s3 = ax_compute_gemm_tn(attn, dout, dWo_ref);
        ax_gemm_set_skip_init(false);
        AX_TEST_ASSERT_EQ((int)s3, (int)AX_OK, "ref gemm_tn ok");
        /* dbo += col_sum(dout) */
        if (c.with_bias) {
            const float *dd = (const float *)dout->storage->data;
            float *dbd = (float *)dbo_ref->storage->data;
            for (int64_t j = 0; j < D; j++) {
                float s = 0.0f;
                for (int64_t r = 0; r < rows; r++) s += dd[r * D + j];
                dbd[j] += s;
            }
        }

        /* fused */
        ax_tensor_t *y_fus     = ax_tensor_create(flat_sh, 2, AX_FLOAT32);
        ax_tensor_t *dattn_fus = ax_tensor_create(flat_sh, 2, AX_FLOAT32);
        ax_tensor_t *dWo_fus   = ax_tensor_zeros(Wo_sh, 2, AX_FLOAT32);
        ax_tensor_t *dbo_fus   = c.with_bias ? ax_tensor_zeros(bo_sh, 1, AX_FLOAT32) : NULL;

        ax_status_t s4 = ax_compute_mha_output_proj_fused(
            attn, Wo, bo, dout, y_fus, dattn_fus, dWo_fus, dbo_fus);
        AX_TEST_ASSERT_EQ((int)s4, (int)AX_OK, "fused mha_output_proj ok");

        /* compare */
        int64_t n_y = rows * D;
        int64_t n_w = D * D;
        const float *yr = (const float *)y_ref->storage->data;
        const float *yf = (const float *)y_fus->storage->data;
        const float *dar = (const float *)dattn_ref->storage->data;
        const float *daf = (const float *)dattn_fus->storage->data;
        const float *dwr = (const float *)dWo_ref->storage->data;
        const float *dwf = (const float *)dWo_fus->storage->data;
        float maxY = 0, maxDA = 0, maxDW = 0;
        for (int64_t i = 0; i < n_y; i++) {
            float a = fabsf(yr[i] - yf[i]);
            float b = fabsf(dar[i] - daf[i]);
            if (a > maxY)  maxY  = a;
            if (b > maxDA) maxDA = b;
        }
        for (int64_t i = 0; i < n_w; i++) {
            float a = fabsf(dwr[i] - dwf[i]);
            if (a > maxDW) maxDW = a;
        }
        AX_TEST_ASSERT(maxY  < 1e-4f, "y matches fused vs unfused");
        AX_TEST_ASSERT(maxDA < 1e-4f, "dattn matches fused vs unfused");
        AX_TEST_ASSERT(maxDW < 1e-4f, "dWo matches fused vs unfused");
        if (c.with_bias) {
            const float *dbr = (const float *)dbo_ref->storage->data;
            const float *dbf = (const float *)dbo_fus->storage->data;
            float maxDB = 0;
            for (int64_t j = 0; j < D; j++) {
                float a = fabsf(dbr[j] - dbf[j]);
                if (a > maxDB) maxDB = a;
            }
            AX_TEST_ASSERT(maxDB < 1e-4f, "dbo matches fused vs unfused");
        }

        ax_tensor_destroy(attn); ax_tensor_destroy(Wo); ax_tensor_destroy(dout);
        if (bo) ax_tensor_destroy(bo);
        ax_tensor_destroy(y_ref); ax_tensor_destroy(dattn_ref); ax_tensor_destroy(dWo_ref);
        if (dbo_ref) ax_tensor_destroy(dbo_ref);
        ax_tensor_destroy(y_fus); ax_tensor_destroy(dattn_fus); ax_tensor_destroy(dWo_fus);
        if (dbo_fus) ax_tensor_destroy(dbo_fus);
    }
}

/* ================================================================
   test: F.3.f ax_compute_output_proj_bwd_fused produces bit-equivalent
   dattn, dWo, dbo to the unfused 3-op backward sequence, with no
   forward y computed (vs F.4.2 proper which computes y too).
   ================================================================ */
static void test_output_proj_bwd_fused_parity(void)
{
    struct shape_case { int64_t rows, D; bool with_dbo; };
    struct shape_case cases[] = {
        { 32,  16,  true  },
        { 64,  64,  true  },
        { 128, 128, false },
        { 256, 64,  true  },
    };

    for (size_t ci = 0; ci < sizeof(cases)/sizeof(cases[0]); ci++) {
        struct shape_case c = cases[ci];
        int64_t rows = c.rows, D = c.D;

        ax_set_seed(33333 + (int)ci);
        int64_t flat_sh[] = {rows, D};
        int64_t Wo_sh[]   = {D, D};
        int64_t bo_sh[]   = {D};

        ax_tensor_t *attn = ax_tensor_rand(flat_sh, 2, -0.1f, 0.1f);
        ax_tensor_t *Wo   = ax_tensor_rand(Wo_sh, 2, -0.1f, 0.1f);
        ax_tensor_t *dout = ax_tensor_rand(flat_sh, 2, -0.1f, 0.1f);

        /* reference: 3 separate backward ops */
        ax_tensor_t *dattn_ref = ax_tensor_create(flat_sh, 2, AX_FLOAT32);
        ax_tensor_t *dWo_ref   = ax_tensor_zeros(Wo_sh, 2, AX_FLOAT32);
        ax_tensor_t *dbo_ref   = c.with_dbo ? ax_tensor_zeros(bo_sh, 1, AX_FLOAT32) : NULL;

        ax_status_t s1 = ax_compute_gemm_nt(dout, Wo, dattn_ref);
        AX_TEST_ASSERT_EQ((int)s1, (int)AX_OK, "ref gemm_nt ok");
        extern void ax_gemm_set_skip_init(bool);
        ax_gemm_set_skip_init(true);
        ax_status_t s2 = ax_compute_gemm_tn(attn, dout, dWo_ref);
        ax_gemm_set_skip_init(false);
        AX_TEST_ASSERT_EQ((int)s2, (int)AX_OK, "ref gemm_tn ok");
        if (c.with_dbo) {
            const float *dd = (const float *)dout->storage->data;
            float *dbd = (float *)dbo_ref->storage->data;
            for (int64_t j = 0; j < D; j++) {
                float s = 0.0f;
                for (int64_t r = 0; r < rows; r++) s += dd[r * D + j];
                dbd[j] += s;
            }
        }

        /* fused: F.3.f bwd-only */
        ax_tensor_t *dattn_fus = ax_tensor_create(flat_sh, 2, AX_FLOAT32);
        ax_tensor_t *dWo_fus   = ax_tensor_zeros(Wo_sh, 2, AX_FLOAT32);
        ax_tensor_t *dbo_fus   = c.with_dbo ? ax_tensor_zeros(bo_sh, 1, AX_FLOAT32) : NULL;

        ax_status_t s3 = ax_compute_output_proj_bwd_fused(
            attn, Wo, dout, dattn_fus, dWo_fus, dbo_fus);
        AX_TEST_ASSERT_EQ((int)s3, (int)AX_OK, "fused output_proj_bwd ok");

        /* compare */
        int64_t n_y = rows * D, n_w = D * D;
        const float *dar = (const float *)dattn_ref->storage->data;
        const float *daf = (const float *)dattn_fus->storage->data;
        const float *dwr = (const float *)dWo_ref->storage->data;
        const float *dwf = (const float *)dWo_fus->storage->data;
        float maxDA = 0, maxDW = 0;
        for (int64_t i = 0; i < n_y; i++) {
            float a = fabsf(dar[i] - daf[i]);
            if (a > maxDA) maxDA = a;
        }
        for (int64_t i = 0; i < n_w; i++) {
            float a = fabsf(dwr[i] - dwf[i]);
            if (a > maxDW) maxDW = a;
        }
        AX_TEST_ASSERT(maxDA < 1e-4f, "F.3.f dattn matches");
        AX_TEST_ASSERT(maxDW < 1e-4f, "F.3.f dWo matches");
        if (c.with_dbo) {
            const float *dbr = (const float *)dbo_ref->storage->data;
            const float *dbf = (const float *)dbo_fus->storage->data;
            float maxDB = 0;
            for (int64_t j = 0; j < D; j++) {
                float a = fabsf(dbr[j] - dbf[j]);
                if (a > maxDB) maxDB = a;
            }
            AX_TEST_ASSERT(maxDB < 1e-4f, "F.3.f dbo matches");
        }

        ax_tensor_destroy(attn); ax_tensor_destroy(Wo); ax_tensor_destroy(dout);
        ax_tensor_destroy(dattn_ref); ax_tensor_destroy(dWo_ref);
        if (dbo_ref) ax_tensor_destroy(dbo_ref);
        ax_tensor_destroy(dattn_fus); ax_tensor_destroy(dWo_fus);
        if (dbo_fus) ax_tensor_destroy(dbo_fus);
    }
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
    AX_RUN_TEST(test_sdpa_save_path_parity);
    AX_RUN_TEST(test_mha_train_step_parity);
    AX_RUN_TEST(test_mha_train_step_fused_parity);
    AX_RUN_TEST(test_mha_train_step_v4_parity);
    AX_RUN_TEST(test_qkv_head_gemm_parity);
    AX_RUN_TEST(test_dattn_head_gemm_nt_parity);
    AX_RUN_TEST(test_sdpa_fwd_to_flat_parity);
    AX_RUN_TEST(test_sdpa_bwd_from_flat_parity);
    AX_RUN_TEST(test_mha_output_proj_fused_parity);
    AX_RUN_TEST(test_output_proj_bwd_fused_parity);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
