/* fused_attention.c — thin dispatch layer for the SDPA compute core,
   RoPE, and KV cache.

   the hot code lives in cpu_opt.c so it can share static packing and
   micro-kernel helpers with the matmul path. this file just picks the
   right ISA variant once at startup and forwards calls. */

#include "axiom/attention.h"
#include "axiom/memory.h"
#include "axiom/error.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ================================================================
   cpu_opt entry points — ISA-suffixed under AX_CPU_ISA_DISPATCH.
   this mirrors the pattern in src/compute/dispatch.c; the cpu_opt
   translation unit is compiled once per ISA (avx512 / avx2 / scalar)
   and each variant's public symbols carry the suffix.
   ================================================================ */

#ifdef AX_CPU_ISA_DISPATCH

extern void ax_cpu_sdpa_fwd_avx512(const float *, const float *, const float *,
                                    float *, float *,
                                    int64_t, int64_t, int64_t, float,
                                    bool, const int8_t *, float *);
extern void ax_cpu_sdpa_fwd_avx2  (const float *, const float *, const float *,
                                    float *, float *,
                                    int64_t, int64_t, int64_t, float,
                                    bool, const int8_t *, float *);
extern void ax_cpu_sdpa_fwd_scalar(const float *, const float *, const float *,
                                    float *, float *,
                                    int64_t, int64_t, int64_t, float,
                                    bool, const int8_t *, float *);

extern void ax_cpu_sdpa_fwd_to_flat_avx512(const float *, const float *, const float *,
                                             float *, float *,
                                             int64_t, int64_t, int64_t, int64_t,
                                             float, bool, const int8_t *, float *);
extern void ax_cpu_sdpa_fwd_to_flat_avx2  (const float *, const float *, const float *,
                                             float *, float *,
                                             int64_t, int64_t, int64_t, int64_t,
                                             float, bool, const int8_t *, float *);
extern void ax_cpu_sdpa_fwd_to_flat_scalar(const float *, const float *, const float *,
                                             float *, float *,
                                             int64_t, int64_t, int64_t, int64_t,
                                             float, bool, const int8_t *, float *);

/* F.4.4 Phase A per-qi-block fwd: same args as fwd_to_flat plus qi_start/qi_end. */
extern void ax_cpu_sdpa_fwd_to_flat_qi_block_avx512(const float *, const float *, const float *,
                                                      float *, float *,
                                                      int64_t, int64_t, int64_t, int64_t,
                                                      int64_t, int64_t,
                                                      float, bool, const int8_t *, float *);
extern void ax_cpu_sdpa_fwd_to_flat_qi_block_avx2  (const float *, const float *, const float *,
                                                      float *, float *,
                                                      int64_t, int64_t, int64_t, int64_t,
                                                      int64_t, int64_t,
                                                      float, bool, const int8_t *, float *);
extern void ax_cpu_sdpa_fwd_to_flat_qi_block_scalar(const float *, const float *, const float *,
                                                      float *, float *,
                                                      int64_t, int64_t, int64_t, int64_t,
                                                      int64_t, int64_t,
                                                      float, bool, const int8_t *, float *);

extern void ax_cpu_sdpa_bwd_avx512(const float *, const float *, const float *,
                                    const float *, const float *, const float *,
                                    float *, float *, float *,
                                    int64_t, int64_t, int64_t, float, bool,
                                    const float *);
extern void ax_cpu_sdpa_bwd_avx2  (const float *, const float *, const float *,
                                    const float *, const float *, const float *,
                                    float *, float *, float *,
                                    int64_t, int64_t, int64_t, float, bool,
                                    const float *);
extern void ax_cpu_sdpa_bwd_scalar(const float *, const float *, const float *,
                                    const float *, const float *, const float *,
                                    float *, float *, float *,
                                    int64_t, int64_t, int64_t, float, bool,
                                    const float *);

extern void ax_cpu_sdpa_bwd_from_flat_avx512(const float *, const float *, const float *,
                                               const float *, const float *, const float *,
                                               float *, float *, float *,
                                               int64_t, int64_t, int64_t, int64_t,
                                               float, bool, const float *);
extern void ax_cpu_sdpa_bwd_from_flat_avx2  (const float *, const float *, const float *,
                                               const float *, const float *, const float *,
                                               float *, float *, float *,
                                               int64_t, int64_t, int64_t, int64_t,
                                               float, bool, const float *);
extern void ax_cpu_sdpa_bwd_from_flat_scalar(const float *, const float *, const float *,
                                               const float *, const float *, const float *,
                                               float *, float *, float *,
                                               int64_t, int64_t, int64_t, int64_t,
                                               float, bool, const float *);

/* F.4.4 Phase A per-qi-block bwd. */
extern void ax_cpu_sdpa_bwd_from_flat_qi_block_avx512(const float *, const float *, const float *,
                                                        const float *, const float *, const float *,
                                                        float *, float *, float *,
                                                        int64_t, int64_t, int64_t, int64_t,
                                                        int64_t, int64_t,
                                                        float, bool, const float *);
extern void ax_cpu_sdpa_bwd_from_flat_qi_block_avx2  (const float *, const float *, const float *,
                                                        const float *, const float *, const float *,
                                                        float *, float *, float *,
                                                        int64_t, int64_t, int64_t, int64_t,
                                                        int64_t, int64_t,
                                                        float, bool, const float *);
extern void ax_cpu_sdpa_bwd_from_flat_qi_block_scalar(const float *, const float *, const float *,
                                                        const float *, const float *, const float *,
                                                        float *, float *, float *,
                                                        int64_t, int64_t, int64_t, int64_t,
                                                        int64_t, int64_t,
                                                        float, bool, const float *);

extern void ax_cpu_rope_apply_avx512(float *, float *, const int64_t *,
                                      int64_t, int64_t, int64_t, float);
extern void ax_cpu_rope_apply_avx2  (float *, float *, const int64_t *,
                                      int64_t, int64_t, int64_t, float);
extern void ax_cpu_rope_apply_scalar(float *, float *, const int64_t *,
                                      int64_t, int64_t, int64_t, float);

extern void ax_cpu_kv_cache_attend_avx512(const float *, const float *,
                                            const float *, float *,
                                            int64_t, int64_t, int64_t, int64_t, float);
extern void ax_cpu_kv_cache_attend_avx2  (const float *, const float *,
                                            const float *, float *,
                                            int64_t, int64_t, int64_t, int64_t, float);
extern void ax_cpu_kv_cache_attend_scalar(const float *, const float *,
                                            const float *, float *,
                                            int64_t, int64_t, int64_t, int64_t, float);

#else /* single-ISA build */

extern void ax_cpu_sdpa_fwd(const float *, const float *, const float *,
                             float *, float *,
                             int64_t, int64_t, int64_t, float,
                             bool, const int8_t *, float *);
extern void ax_cpu_sdpa_fwd_to_flat(const float *, const float *, const float *,
                                      float *, float *,
                                      int64_t, int64_t, int64_t, int64_t,
                                      float, bool, const int8_t *, float *);
extern void ax_cpu_sdpa_fwd_to_flat_qi_block(const float *, const float *, const float *,
                                               float *, float *,
                                               int64_t, int64_t, int64_t, int64_t,
                                               int64_t, int64_t,
                                               float, bool, const int8_t *, float *);
extern void ax_cpu_sdpa_bwd(const float *, const float *, const float *,
                             const float *, const float *, const float *,
                             float *, float *, float *,
                             int64_t, int64_t, int64_t, float, bool,
                             const float *);
extern void ax_cpu_sdpa_bwd_from_flat(const float *, const float *, const float *,
                                        const float *, const float *, const float *,
                                        float *, float *, float *,
                                        int64_t, int64_t, int64_t, int64_t,
                                        float, bool, const float *);
extern void ax_cpu_sdpa_bwd_from_flat_qi_block(const float *, const float *, const float *,
                                                 const float *, const float *, const float *,
                                                 float *, float *, float *,
                                                 int64_t, int64_t, int64_t, int64_t,
                                                 int64_t, int64_t,
                                                 float, bool, const float *);
extern void ax_cpu_rope_apply(float *, float *, const int64_t *,
                               int64_t, int64_t, int64_t, float);
extern void ax_cpu_kv_cache_attend(const float *, const float *,
                                     const float *, float *,
                                     int64_t, int64_t, int64_t, int64_t, float);

#endif

/* ================================================================
   resolved function pointers — set once on first call via
   __builtin_cpu_supports probing. the probe cost is one check per
   process lifetime; subsequent calls go straight through.
   ================================================================ */

typedef void (*sdpa_fwd_fn_t)(const float *, const float *, const float *,
                               float *, float *,
                               int64_t, int64_t, int64_t, float,
                               bool, const int8_t *, float *);
typedef void (*sdpa_fwd_to_flat_fn_t)(const float *, const float *, const float *,
                                        float *, float *,
                                        int64_t, int64_t, int64_t, int64_t,
                                        float, bool, const int8_t *, float *);
typedef void (*sdpa_fwd_to_flat_qi_block_fn_t)(
    const float *, const float *, const float *,
    float *, float *,
    int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    float, bool, const int8_t *, float *);
typedef void (*sdpa_bwd_fn_t)(const float *, const float *, const float *,
                               const float *, const float *, const float *,
                               float *, float *, float *,
                               int64_t, int64_t, int64_t, float, bool,
                               const float *);
typedef void (*sdpa_bwd_from_flat_fn_t)(const float *, const float *, const float *,
                                          const float *, const float *, const float *,
                                          float *, float *, float *,
                                          int64_t, int64_t, int64_t, int64_t,
                                          float, bool, const float *);
typedef void (*sdpa_bwd_from_flat_qi_block_fn_t)(
    const float *, const float *, const float *,
    const float *, const float *, const float *,
    float *, float *, float *,
    int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    float, bool, const float *);
typedef void (*rope_fn_t)(float *, float *, const int64_t *,
                           int64_t, int64_t, int64_t, float);
typedef void (*kv_attend_fn_t)(const float *, const float *,
                                const float *, float *,
                                int64_t, int64_t, int64_t, int64_t, float);

static sdpa_fwd_fn_t                       g_sdpa_fwd                       = NULL;
static sdpa_fwd_to_flat_fn_t               g_sdpa_fwd_to_flat               = NULL;
static sdpa_fwd_to_flat_qi_block_fn_t      g_sdpa_fwd_to_flat_qi_block      = NULL;
static sdpa_bwd_fn_t                       g_sdpa_bwd                       = NULL;
static sdpa_bwd_from_flat_fn_t             g_sdpa_bwd_from_flat             = NULL;
static sdpa_bwd_from_flat_qi_block_fn_t    g_sdpa_bwd_from_flat_qi_block    = NULL;
static rope_fn_t                           g_rope                           = NULL;
static kv_attend_fn_t                      g_kv_attend                      = NULL;

static void resolve_once(void)
{
    if (g_sdpa_fwd) return;

#ifdef AX_CPU_ISA_DISPATCH
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma")) {
        g_sdpa_fwd                    = ax_cpu_sdpa_fwd_avx512;
        g_sdpa_fwd_to_flat            = ax_cpu_sdpa_fwd_to_flat_avx512;
        g_sdpa_fwd_to_flat_qi_block   = ax_cpu_sdpa_fwd_to_flat_qi_block_avx512;
        g_sdpa_bwd                    = ax_cpu_sdpa_bwd_avx512;
        g_sdpa_bwd_from_flat          = ax_cpu_sdpa_bwd_from_flat_avx512;
        g_sdpa_bwd_from_flat_qi_block = ax_cpu_sdpa_bwd_from_flat_qi_block_avx512;
        g_rope                        = ax_cpu_rope_apply_avx512;
        g_kv_attend                   = ax_cpu_kv_cache_attend_avx512;
    } else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        g_sdpa_fwd                    = ax_cpu_sdpa_fwd_avx2;
        g_sdpa_fwd_to_flat            = ax_cpu_sdpa_fwd_to_flat_avx2;
        g_sdpa_fwd_to_flat_qi_block   = ax_cpu_sdpa_fwd_to_flat_qi_block_avx2;
        g_sdpa_bwd                    = ax_cpu_sdpa_bwd_avx2;
        g_sdpa_bwd_from_flat          = ax_cpu_sdpa_bwd_from_flat_avx2;
        g_sdpa_bwd_from_flat_qi_block = ax_cpu_sdpa_bwd_from_flat_qi_block_avx2;
        g_rope                        = ax_cpu_rope_apply_avx2;
        g_kv_attend                   = ax_cpu_kv_cache_attend_avx2;
    } else {
        g_sdpa_fwd                    = ax_cpu_sdpa_fwd_scalar;
        g_sdpa_fwd_to_flat            = ax_cpu_sdpa_fwd_to_flat_scalar;
        g_sdpa_fwd_to_flat_qi_block   = ax_cpu_sdpa_fwd_to_flat_qi_block_scalar;
        g_sdpa_bwd                    = ax_cpu_sdpa_bwd_scalar;
        g_sdpa_bwd_from_flat          = ax_cpu_sdpa_bwd_from_flat_scalar;
        g_sdpa_bwd_from_flat_qi_block = ax_cpu_sdpa_bwd_from_flat_qi_block_scalar;
        g_rope                        = ax_cpu_rope_apply_scalar;
        g_kv_attend                   = ax_cpu_kv_cache_attend_scalar;
    }
#else
    g_sdpa_fwd                    = ax_cpu_sdpa_fwd;
    g_sdpa_fwd_to_flat            = ax_cpu_sdpa_fwd_to_flat;
    g_sdpa_fwd_to_flat_qi_block   = ax_cpu_sdpa_fwd_to_flat_qi_block;
    g_sdpa_bwd                    = ax_cpu_sdpa_bwd;
    g_sdpa_bwd_from_flat          = ax_cpu_sdpa_bwd_from_flat;
    g_sdpa_bwd_from_flat_qi_block = ax_cpu_sdpa_bwd_from_flat_qi_block;
    g_rope                        = ax_cpu_rope_apply;
    g_kv_attend                   = ax_cpu_kv_cache_attend;
#endif
}

/* ================================================================
   public SDPA primitives
   ================================================================ */

void ax_fused_attention_fwd(const float *Q, const float *K, const float *V,
                             float *out, float *L,
                             int64_t BH, int64_t S, int64_t dk, float scale)
{
    resolve_once();
    g_sdpa_fwd(Q, K, V, out, L, BH, S, dk, scale, false, NULL, NULL);
}

void ax_fused_attention_fwd_causal(const float *Q, const float *K, const float *V,
                                    float *out, float *L,
                                    int64_t BH, int64_t S, int64_t dk, float scale)
{
    resolve_once();
    g_sdpa_fwd(Q, K, V, out, L, BH, S, dk, scale, true, NULL, NULL);
}

void ax_fused_attention_bwd(const float *Q, const float *K, const float *V,
                             const float *O, const float *dO, const float *L,
                             float *dQ, float *dK, float *dV,
                             int64_t BH, int64_t S, int64_t dk, float scale)
{
    resolve_once();
    g_sdpa_bwd(Q, K, V, O, dO, L, dQ, dK, dV, BH, S, dk, scale, false, NULL);
}

void ax_fused_attention_bwd_causal(const float *Q, const float *K, const float *V,
                                    const float *O, const float *dO, const float *L,
                                    float *dQ, float *dK, float *dV,
                                    int64_t BH, int64_t S, int64_t dk, float scale)
{
    resolve_once();
    g_sdpa_bwd(Q, K, V, O, dO, L, dQ, dK, dV, BH, S, dk, scale, true, NULL);
}

/* save-aware variants — used by the MHA training path. when P_save / P_saved
   is non-NULL, layout is [BH, S, S] row-major. fwd writes post-mask
   pre-softmax scores; bwd reads them and skips the QK^T recompute. */
void ax_fused_attention_fwd_save(const float *Q, const float *K, const float *V,
                                  float *out, float *L, float *P_save,
                                  int64_t BH, int64_t S, int64_t dk, float scale,
                                  bool causal)
{
    resolve_once();
    g_sdpa_fwd(Q, K, V, out, L, BH, S, dk, scale, causal, NULL, P_save);
}

/* F.3.e fused SDPA forward writing into [B, S, D] = [B, S, H*dk] attn_flat.
   eliminates the per-head [BH, S, dk] Oh intermediate + the head_deinterleave
   pass that follows it in the MHA training path. each of B*H heads writes
   its dk-wide slot at attn_flat[b*S*D + h*dk] with row stride D — disjoint
   slots so no cross-head contention. */
void ax_fused_attention_fwd_save_to_flat(const float *Q, const float *K, const float *V,
                                           float *attn_flat, float *L, float *P_save,
                                           int64_t B, int64_t S, int64_t H, int64_t dk,
                                           float scale, bool causal)
{
    resolve_once();
    g_sdpa_fwd_to_flat(Q, K, V, attn_flat, L, B, S, H, dk, scale, causal, NULL, P_save);
}

/* F.3.e companion: SDPA backward reading O from [B, S, D] attn_flat.
   together with ax_fused_attention_fwd_save_to_flat, eliminates the Oh
   tensor entirely from the MHA training path — only the Di = dot(O, dO)
   loop differs from the standard bwd (per-row O read uses stride D
   into attn_flat instead of stride dk into Oh). */
void ax_fused_attention_bwd_use_from_flat(const float *Q, const float *K, const float *V,
                                            const float *attn_flat,
                                            const float *dO, const float *L,
                                            const float *P_saved,
                                            float *dQ, float *dK, float *dV,
                                            int64_t B, int64_t S, int64_t H, int64_t dk,
                                            float scale, bool causal)
{
    resolve_once();
    g_sdpa_bwd_from_flat(Q, K, V, attn_flat, dO, L, dQ, dK, dV,
                         B, S, H, dk, scale, causal, P_saved);
}

/* F.4.4 Phase A: per-qi-block SDPA forward writing into attn_flat.
   processes only [qi_start, qi_end) Q rows. caller (the F.4.4 driver)
   loops over qi-blocks; rows outside the range are unread/unwritten
   so disjoint qi-block calls are independent. */
void ax_fused_attention_fwd_save_to_flat_qi_block(
    const float *Q, const float *K, const float *V,
    float *attn_flat, float *L, float *P_save,
    int64_t B, int64_t S, int64_t H, int64_t dk,
    int64_t qi_start, int64_t qi_end,
    float scale, bool causal)
{
    resolve_once();
    g_sdpa_fwd_to_flat_qi_block(Q, K, V, attn_flat, L, B, S, H, dk,
                                  qi_start, qi_end,
                                  scale, causal, NULL, P_save);
}

/* F.4.4 Phase A: per-qi-block SDPA backward reading O strided from
   attn_flat. dQ rows in [qi_start, qi_end) are OVERWRITTEN per call;
   dK / dV are ACCUMULATED across qi-block calls (caller MUST
   memset dK = dV = 0 once before the first qi-block call). */
void ax_fused_attention_bwd_use_from_flat_qi_block(
    const float *Q, const float *K, const float *V,
    const float *attn_flat, const float *dO, const float *L,
    const float *P_saved,
    float *dQ, float *dK, float *dV,
    int64_t B, int64_t S, int64_t H, int64_t dk,
    int64_t qi_start, int64_t qi_end,
    float scale, bool causal)
{
    resolve_once();
    g_sdpa_bwd_from_flat_qi_block(Q, K, V, attn_flat, dO, L, dQ, dK, dV,
                                    B, S, H, dk,
                                    qi_start, qi_end,
                                    scale, causal, P_saved);
}

void ax_fused_attention_bwd_use(const float *Q, const float *K, const float *V,
                                 const float *O, const float *dO, const float *L,
                                 const float *P_saved,
                                 float *dQ, float *dK, float *dV,
                                 int64_t BH, int64_t S, int64_t dk, float scale,
                                 bool causal)
{
    resolve_once();
    g_sdpa_bwd(Q, K, V, O, dO, L, dQ, dK, dV, BH, S, dk, scale, causal, P_saved);
}

/* ================================================================
   RoPE
   ================================================================ */

void ax_rope_apply(float *Q, float *K, const int64_t *positions,
                    int64_t BH, int64_t S, int64_t dk, float theta_base)
{
    resolve_once();
    g_rope(Q, K, positions, BH, S, dk, theta_base);
}

/* ================================================================
   KV cache
   ================================================================ */

ax_kv_cache_t *ax_kv_cache_create(int64_t BH, int64_t max_seq, int64_t dk)
{
    if (BH <= 0 || max_seq <= 0 || dk <= 0) return NULL;

    ax_kv_cache_t *c = (ax_kv_cache_t *)calloc(1, sizeof(ax_kv_cache_t));
    if (!c) return NULL;

    size_t bytes = (size_t)BH * (size_t)max_seq * (size_t)dk * sizeof(float);
    c->K = (float *)ax_aligned_alloc(bytes, 64);
    c->V = (float *)ax_aligned_alloc(bytes, 64);
    if (!c->K || !c->V) {
        ax_aligned_free(c->K);
        ax_aligned_free(c->V);
        free(c);
        return NULL;
    }
    c->BH = BH;
    c->max_seq = max_seq;
    c->dk = dk;
    c->cur_len = 0;
    return c;
}

void ax_kv_cache_destroy(ax_kv_cache_t *c)
{
    if (!c) return;
    ax_aligned_free(c->K);
    ax_aligned_free(c->V);
    free(c);
}

void ax_kv_cache_reset(ax_kv_cache_t *c)
{
    if (c) c->cur_len = 0;
}

bool ax_kv_cache_append(ax_kv_cache_t *c, const float *new_K, const float *new_V)
{
    if (!c || !new_K || !new_V) return false;
    if (c->cur_len >= c->max_seq) return false;

    int64_t slot = c->cur_len;
    int64_t dk = c->dk;
    int64_t ms = c->max_seq;
    /* layout: [BH, max_seq, dk]; each head contributes one row per token */
    for (int64_t h = 0; h < c->BH; h++) {
        memcpy(c->K + h * ms * dk + slot * dk, new_K + h * dk, (size_t)dk * sizeof(float));
        memcpy(c->V + h * ms * dk + slot * dk, new_V + h * dk, (size_t)dk * sizeof(float));
    }
    c->cur_len++;
    return true;
}

void ax_kv_cache_attend(const ax_kv_cache_t *c,
                         const float *Q_new, float *out,
                         float scale)
{
    if (!c || c->cur_len == 0) {
        if (c && out) memset(out, 0, (size_t)c->BH * (size_t)c->dk * sizeof(float));
        return;
    }
    resolve_once();
    g_kv_attend(c->K, c->V, Q_new, out, c->BH, c->cur_len, c->max_seq, c->dk, scale);
}
