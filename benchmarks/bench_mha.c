/* bench_mha.c — benchmark MHA and SDPA across representative shapes.

   reports GFLOPS and throughput for:
     - raw SDPA forward (no projections)
     - full MHA layer forward (with fused QKV + Wo projections)
     - MHA forward + backward
     - KV cache attend (single-token inference)

   uses the analytical FLOP count for attention:
     - Q @ K^T           : 2 * S^2 * dk flops per head
     - softmax           : ~3 * S^2 flops per head (masked as negligible)
     - P @ V             : 2 * S^2 * dk flops per head
   total SDPA = 4 * BH * S^2 * dk

   projections: 4 * B * S * D^2 flops (Wq, Wk, Wv, Wo each = 2 * B*S * D * D flops).
   full MHA forward = 4 B S D^2 + 4 BH S^2 dk = 4 B S D^2 + 4 B S^2 D. */

#include "axiom/axiom.h"
#include "bench_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define now_s bench_now_s

typedef struct { int64_t B, S, D, H; } shape_t;

static void bench_sdpa(const shape_t *s, int iters, bool causal)
{
    int64_t B = s->B, S = s->S, D = s->D, H = s->H;
    int64_t dk = D / H;
    int64_t BH = B * H;
    int64_t elems = BH * S * dk;

    float *Q = (float *)aligned_alloc(64, (size_t)elems * sizeof(float));
    float *K = (float *)aligned_alloc(64, (size_t)elems * sizeof(float));
    float *V = (float *)aligned_alloc(64, (size_t)elems * sizeof(float));
    float *O = (float *)aligned_alloc(64, (size_t)elems * sizeof(float));
    for (int64_t i = 0; i < elems; i++) {
        Q[i] = (float)rand() / RAND_MAX - 0.5f;
        K[i] = (float)rand() / RAND_MAX - 0.5f;
        V[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    float scale = 1.0f / sqrtf((float)dk);

    /* warm up */
    if (causal) ax_fused_attention_fwd_causal(Q, K, V, O, NULL, BH, S, dk, scale);
    else        ax_fused_attention_fwd       (Q, K, V, O, NULL, BH, S, dk, scale);

    double t0 = now_s();
    for (int i = 0; i < iters; i++) {
        if (causal) ax_fused_attention_fwd_causal(Q, K, V, O, NULL, BH, S, dk, scale);
        else        ax_fused_attention_fwd       (Q, K, V, O, NULL, BH, S, dk, scale);
    }
    double t1 = now_s();
    double elapsed = (t1 - t0) / iters;
    double flops = 4.0 * BH * S * S * dk;
    double gflops = flops / elapsed / 1e9;

    printf("  SDPA%s  BH=%4ld S=%4ld dk=%3ld :  %7.3f ms  %8.2f GFLOPS\n",
           causal ? "-causal" : "       ", BH, S, dk, elapsed * 1e3, gflops);

    char cs[96];
    snprintf(cs, sizeof(cs), "sdpa%s_BH%ld_S%ld_dk%ld",
             causal ? "_causal" : "", (long)BH, (long)S, (long)dk);
    BENCH_EMIT("mha", cs, "lat_ms", elapsed * 1e3);
    BENCH_EMIT("mha", cs, "gflops", gflops);

    free(Q); free(K); free(V); free(O);
}

static void bench_mha_forward(const shape_t *s, int iters)
{
    int64_t B = s->B, S = s->S, D = s->D;
    int H = (int)s->H;

    ax_layer_t *mha = ax_mha_create(D, H, true, false);
    int64_t sh[] = {B, S, D};
    ax_tensor_t *x = ax_tensor_rand(sh, 3, -0.5f, 0.5f);

    ax_no_grad();
    ax_layer_eval(mha);

    ax_tensor_t *out = ax_layer_forward(mha, x);
    ax_tensor_destroy(out);

    double t0 = now_s();
    for (int i = 0; i < iters; i++) {
        out = ax_layer_forward(mha, x);
        ax_tensor_destroy(out);
    }
    double t1 = now_s();
    double elapsed = (t1 - t0) / iters;

    /* total flops: 4 projections + 2 attention-tile gemms.
       projections: Wq/Wk/Wv/Wo each = 2 * B*S * D * D = 2 B S D^2 flops.
       attention: 4 * B * S^2 * D (B H * 4 S^2 dk = 4 B S^2 D). */
    double flops = 4.0 * 2.0 * B * S * D * D + 4.0 * B * S * S * D;
    double gflops = flops / elapsed / 1e9;

    printf("  MHA-fwd      B=%2ld S=%4ld D=%4ld H=%2d :  %7.3f ms  %8.2f GFLOPS\n",
           B, S, D, H, elapsed * 1e3, gflops);

    char cs[96];
    snprintf(cs, sizeof(cs), "mha_fwd_B%ld_S%ld_D%ld_H%d", (long)B, (long)S, (long)D, H);
    BENCH_EMIT("mha", cs, "lat_ms", elapsed * 1e3);
    BENCH_EMIT("mha", cs, "gflops", gflops);

    ax_tensor_destroy(x);
    ax_layer_destroy(mha);
    ax_enable_grad();
}

static void bench_mha_train(const shape_t *s, int iters)
{
    int64_t B = s->B, S = s->S, D = s->D;
    int H = (int)s->H;

    ax_layer_t *mha = ax_mha_create(D, H, true, false);
    int64_t sh[] = {B, S, D};
    ax_tensor_t *x = ax_tensor_rand(sh, 3, -0.1f, 0.1f);
    x->requires_grad = true;

    /* warm up */
    ax_tensor_t *out = ax_layer_forward(mha, x);
    ax_tensor_t *loss = ax_sum(out, -1);
    ax_backward(loss);
    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss);

    /* measure forward + backward separately to attribute time */
    double t_fwd = 0, t_bwd = 0, t_cleanup = 0;
    for (int i = 0; i < iters; i++) {
        double a = now_s();
        out  = ax_layer_forward(mha, x);
        loss = ax_sum(out, -1);
        double b = now_s();
        ax_backward(loss);
        double c = now_s();
        ax_graph_cleanup(loss);
        ax_tensor_destroy(loss);
        double d = now_s();
        t_fwd += (b - a);
        t_bwd += (c - b);
        t_cleanup += (d - c);
    }
    t_fwd /= iters; t_bwd /= iters; t_cleanup /= iters;
    double elapsed = t_fwd + t_bwd + t_cleanup;

    double fwd_flops = 4.0 * 2.0 * B * S * D * D + 4.0 * B * S * S * D;
    double gflops = 3.0 * fwd_flops / elapsed / 1e9;

    printf("  MHA-train    B=%2ld S=%4ld D=%4ld H=%2d :  %7.3f ms  %8.2f GFLOPS (est)\n",
           B, S, D, H, elapsed * 1e3, gflops);
    printf("    breakdown: fwd=%.3fms  bwd=%.3fms  cleanup=%.3fms  (bwd/fwd=%.2fx)\n",
           t_fwd * 1e3, t_bwd * 1e3, t_cleanup * 1e3, t_bwd / t_fwd);

    char cs[96];
    snprintf(cs, sizeof(cs), "mha_train_B%ld_S%ld_D%ld_H%d", (long)B, (long)S, (long)D, H);
    BENCH_EMIT("mha", cs, "lat_ms", elapsed * 1e3);
    BENCH_EMIT("mha", cs, "gflops", gflops);

    ax_tensor_destroy(x);
    ax_layer_destroy(mha);
}

static void bench_kv_cache(const shape_t *s, int iters)
{
    int64_t B = s->B, S = s->S, D = s->D;
    int H = (int)s->H;
    int64_t BH = B * H;
    int64_t dk = D / H;

    ax_kv_cache_t *c = ax_kv_cache_create(BH, S, dk);
    float *Ktok = (float *)malloc((size_t)BH * dk * sizeof(float));
    float *Vtok = (float *)malloc((size_t)BH * dk * sizeof(float));
    for (int64_t i = 0; i < BH * dk; i++) {
        Ktok[i] = (float)rand() / RAND_MAX - 0.5f;
        Vtok[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    /* fill cache */
    for (int64_t t = 0; t < S; t++) ax_kv_cache_append(c, Ktok, Vtok);

    float *Qnew = (float *)malloc((size_t)BH * dk * sizeof(float));
    float *out  = (float *)malloc((size_t)BH * dk * sizeof(float));
    for (int64_t i = 0; i < BH * dk; i++) Qnew[i] = (float)rand() / RAND_MAX - 0.5f;
    float scale = 1.0f / sqrtf((float)dk);

    ax_kv_cache_attend(c, Qnew, out, scale);  /* warm */

    /* decoder-step kv_attend is microseconds-per-call; iters=3 is too
       noisy on shared CI runners. scale iters until either the caller's
       count is met OR at least ~50ms of wall time has accumulated. */
    double t0 = now_s();
    int i = 0;
    int min_iters = (iters > 20) ? iters : 20;
    double wall_floor = 0.050;
    double t1;
    for (;;) {
        ax_kv_cache_attend(c, Qnew, out, scale);
        i++;
        t1 = now_s();
        if (i >= min_iters && (t1 - t0) >= wall_floor) break;
        if (i >= 10000) break;
    }
    double elapsed = (t1 - t0) / (double)i;
    double flops = 4.0 * BH * S * dk;  /* Q@K + weighted sum of V, each 2*S*dk */
    double gflops = flops / elapsed / 1e9;

    printf("  KV-attend    BH=%4ld S=%4ld dk=%3ld :  %7.3f ms  %8.2f GFLOPS\n",
           BH, S, dk, elapsed * 1e3, gflops);

    char cs[96];
    snprintf(cs, sizeof(cs), "kv_attend_BH%ld_S%ld_dk%ld", (long)BH, (long)S, (long)dk);
    BENCH_EMIT("mha", cs, "lat_ms", elapsed * 1e3);
    BENCH_EMIT("mha", cs, "gflops", gflops);

    free(Ktok); free(Vtok); free(Qnew); free(out);
    ax_kv_cache_destroy(c);
}

int main(int argc, char **argv)
{
    ax_init();

    shape_t shapes[] = {
        /* small transformer — BERT-style */
        { .B=8,  .S=128,  .D=512,  .H=8  },
        /* medium — GPT-2 small ish */
        { .B=4,  .S=512,  .D=768,  .H=12 },
        /* long context */
        { .B=2,  .S=1024, .D=768,  .H=12 },
        /* wide model — LLaMA-7B-ish at d=4096 */
        { .B=1,  .S=512,  .D=1024, .H=16 },
        /* very long context (stress L2 / L3) */
        { .B=1,  .S=2048, .D=768,  .H=12 },
    };
    int n = (int)(sizeof(shapes) / sizeof(shapes[0]));

    int iters = (argc > 1) ? atoi(argv[1]) : 5;
    if (iters <= 0) iters = 1;

    printf("== MHA / SDPA benchmarks (iters=%d) ==\n\n", iters);
    for (int i = 0; i < n; i++) {
        const shape_t *s = &shapes[i];
        printf("-- shape[%d]  B=%ld S=%ld D=%ld H=%ld --\n", i, s->B, s->S, s->D, s->H);
        bench_sdpa(s, iters, false);
        bench_sdpa(s, iters, true);
        bench_mha_forward(s, iters);
        bench_mha_train(s, iters);
        bench_kv_cache(s, iters);
        printf("\n");
    }

    ax_shutdown();
    return 0;
}
