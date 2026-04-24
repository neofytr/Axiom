/* attention.c — multi-head attention layer (mha forward + backward +
   layer constructor / destructor).

   k.3 split: this file used to contain the layout helpers (head
   interleave / deinterleave / qkv split-merge variants) and the
   Wqkv/bqkv cache-rebuild routine inline. those are now in two
   sibling tu's:
     attention/layout.c  — pure-data-shuffling helpers (no compute).
     attention/cache.c   — fused weight + bias panel cache management.
   their declarations live in src/core/attention/internal.h. this file
   keeps the layer-level orchestration (forward / backward grad_fn /
   create / destroy) and ties the pieces together.

   fused QKV projection: one [B*S, D] @ [D, 3D] GEMM rebuilds Q, K, V
   together. the [D, 3D] panel is cached between calls and rebuilt only
   when one of Wq/Wk/Wv mutates (checked via storage generation counter
   in cache.c).

   forward flow:
     x:[B,S,D]
       -> QKV = x @ Wqkv + bqkv                           [B*S, 3D]
       -> split + head-interleave to [BH, S, dk] each     Q,K,V
       -> SDPA(Q,K,V)                                      O_attn:[BH,S,dk], L:[BH,S]
       -> merge heads back to [B*S, D]                     attn_flat
       -> out = attn_flat @ Wo + bo                        [B,S,D]

   backward is a single fused grad_fn that reverses each step. parameters
   (Wq/Wk/Wv/Wo/bq/bk/bv/bo) are accessed via the saved layer pointer,
   the input is saved as the only autograd input (so dInput is routed to
   the user tensor). all intermediates (x_flat copy, Q/K/V, O_attn, L)
   live in the forward arena so they're released on graph cleanup. */

#include "axiom/attention.h"
#include "axiom/ops.h"
#include "axiom/autograd.h"
#include "axiom/compute.h"
#include "axiom/internal/compute_internal.h"
#include "axiom/internal/cuda_extension.h"
#include "axiom/init.h"
#include "axiom/error.h"
#include "attention/internal.h"
#include "../compute/backends/simd_defs.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdatomic.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* helpers used here are now in attention/layout.c (head_*) and
   attention/cache.c (refresh_fused_qkv). their declarations come in
   via attention/internal.h above. call sites use the ax_attn_* names. */

/* ================================================================
   backward context
   ================================================================ */

/* forward declaration — defined below forward() */
static void mha_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out);

/* T1.3/T1.4: stack-allocated 2-D tensor view over an existing buffer.
   used to skip the x_flat / dOut_flat memcpy when input is already
   contiguous and we just need to reinterpret its 3D shape as 2D for
   the GEMM call. zero heap allocation; storage pointer aliases the
   caller's buffer (caller must keep it alive). */
static inline void mha_make_stack_view(ax_tensor_t *tv, ax_storage_t *st,
                                        const float *data,
                                        int64_t rows, int64_t cols)
{
    st->data         = (void *)(uintptr_t)data;  /* drop const for storage_t */
    st->size_bytes   = (size_t)(rows * cols) * sizeof(float);
    atomic_store(&st->refcount, 0);
    st->device       = AX_DEVICE_CPU;
    st->is_arena_temp = true;
    st->generation   = 1;
    memset(tv, 0, sizeof(*tv));
    tv->storage      = st;
    tv->ndim         = 2;
    tv->dtype        = AX_FLOAT32;
    tv->shape[0]     = rows;
    tv->shape[1]     = cols;
    tv->strides[0]   = cols;
    tv->strides[1]   = 1;
}

/* forward decl — same semantics as train_step_fused.c: tell the gemm
   backends to accumulate into dst rather than overwrite. defined in
   src/compute/dispatch.c. */
extern void ax_gemm_set_skip_init(bool v);

typedef struct {
    ax_mha_t *layer;
    int64_t B, S, D, H, dk;

    /* tensors saved for backward — all arena-allocated (forward arena).
       we keep raw float pointers too so the backward path can run without
       re-indexing. */
    ax_tensor_t *x_flat;       /* [B*S, D]     input after reshape    */
    ax_tensor_t *Q_head;       /* [BH, S, dk]  Q after head_interleave */
    ax_tensor_t *K_head;       /* [BH, S, dk]                          */
    ax_tensor_t *V_head;       /* [BH, S, dk]                          */
    ax_tensor_t *O_head;       /* [BH, S, dk]  SDPA output             */
    ax_tensor_t *attn_flat;    /* [B*S, D]     after head_deinterleave */
    ax_tensor_t *L_tensor;     /* [BH, S]      logsumexp per row       */
    /* optional: post-mask pre-softmax scores [BH, S, S] from forward.
       NULL when S is large (would thrash L3) or training wasn't on. */
    ax_tensor_t *P_save_tensor;
} mha_ctx_t;

static void mha_ctx_cleanup(void *p) { free(p); }

/* ================================================================
   forward
   ================================================================ */

static ax_tensor_t *mha_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_mha_t *m = (ax_mha_t *)self;
    if (!input || input->ndim != 3) return NULL;

    int64_t B  = input->shape[0];
    int64_t S  = input->shape[1];
    int64_t D  = input->shape[2];
    if (D != m->d_model) return NULL;

    int64_t H  = m->n_heads;
    int64_t dk = m->d_k;
    int64_t rows = B * S;

    /* contiguous input as a flat [B*S, D] view */
    ax_tensor_t *inp = ax_ensure_contiguous(input);
    if (!inp) return NULL;

    /* ensure fused weight+bias panels are up to date */
    ax_attn_refresh_fused_qkv(m);

    bool record = ax_grad_enabled() && self->training && input->requires_grad;
    /* always record if any of our params require grad too */
    if (self->training && ax_grad_enabled()) {
        if (m->Wq->requires_grad || m->Wk->requires_grad || m->Wv->requires_grad ||
            m->Wo->requires_grad) record = true;
        if (m->use_bias && (m->bq->requires_grad || m->bk->requires_grad ||
                            m->bv->requires_grad || m->bo->requires_grad)) record = true;
    }

    ax_arena_t *fa = record ? ax_forward_arena() : NULL;

    /* helper: pick arena for training (so tensor survives forward), else heap */
    #define ALLOC_SHAPE(t, sh, nd) do {                                   \
        if (record) (t) = ax_tensor_arena_create(fa, (sh), (nd), AX_FLOAT32); \
        else        (t) = ax_tensor_create((sh), (nd), AX_FLOAT32);           \
    } while (0)
    #define ALLOC_ZEROS(t, sh, nd) do {                                   \
        if (record) (t) = ax_tensor_arena_zeros(fa, (sh), (nd), AX_FLOAT32);\
        else        (t) = ax_tensor_zeros((sh), (nd), AX_FLOAT32);            \
    } while (0)

    /* ---- build x_flat [rows, D] view of input ----
       T1.3: skip the memcpy by using a stack-view aliasing inp's data.
       saves 2-6 MB BW per forward call (B*S*D*4 bytes). on the record
       path the input tensor is kept alive by the autograd graph
       (gf->inputs[0]), so backward can reconstruct an equivalent view
       — no need to materialize a copy here either. */
    int64_t flat_sh[] = {rows, D};
    (void)flat_sh;  /* may be unused if view path taken */
    ax_tensor_t *x_flat = NULL;
    ax_tensor_t  x_flat_view;
    ax_storage_t x_flat_view_st;
    bool inp_contig = (inp->storage && inp->storage->data &&
                       inp->dtype == AX_FLOAT32);
    if (inp_contig) {
        mha_make_stack_view(&x_flat_view, &x_flat_view_st,
                             (const float *)inp->storage->data, rows, D);
        x_flat = &x_flat_view;
    } else {
        ALLOC_SHAPE(x_flat, flat_sh, 2);
        if (!x_flat) goto fail;
        memcpy(x_flat->storage->data, inp->storage->data,
               (size_t)rows * (size_t)D * sizeof(float));
    }

    /* ---- fused QKV + head-interleave ----
       F.3.a (ax_compute_qkv_head_gemm): fuses gemm + head-interleave +
       bias_add into one kernel that writes Qh/Kh/Vh directly, skipping
       the qkv [rows, 3D] intermediate. for B1_S2048 this saves ~18 MB
       of L3↔DRAM traffic and one full-rows memory pass.

       shape-gated: F.3.a shows a measurable win on large-S shapes where
       the qkv intermediate dominates traffic but hurts small-S shapes
       (B8_S128 was measured +20 % on an earlier run, likely because
       the unfused path's bias broadcast pre-init bakes the per-head
       bias into the output inside the layer cache in a way that fuses
       well with the subsequent SDPA fwd's first read — on small S the
       qkv intermediate fits in L2 anyway). threshold: enable when
       rows * 3 * D * sizeof(float) > 8 MB (≈ L2 spill point on the
       majority of modern CPUs with L2 ≥ 1 MB per core). env override
       AX_NO_F3A=1 disables the heuristic. */
    int64_t head_sh[] = {B * H, S, dk};
    ax_tensor_t *Qh = NULL, *Kh = NULL, *Vh = NULL;
    ALLOC_SHAPE(Qh, head_sh, 3);
    ALLOC_SHAPE(Kh, head_sh, 3);
    ALLOC_SHAPE(Vh, head_sh, 3);
    if (!Qh || !Kh || !Vh) goto fail;

    const ax_cuda_extension_t *cu = ax_compute_get_cuda_extension();
    bool on_cuda = (x_flat->storage->device != AX_DEVICE_CPU);
    int64_t qkv_bytes = rows * 3 * D * (int64_t)sizeof(float);
    bool use_f3a = (!on_cuda) && ax_compute_has_qkv_head_gemm()
                   && (qkv_bytes > 8 * 1024 * 1024);
    {
        const char *no_f3a = getenv("AX_NO_F3A");
        if (no_f3a && no_f3a[0] == '1') use_f3a = false;
    }
    ax_tensor_t *qkv = NULL;  /* only alloc'd on fallback */

    if (use_f3a) {
        const ax_tensor_t *bqkv_arg = m->use_bias ? m->bqkv_cache : NULL;
        if (ax_compute_qkv_head_gemm(x_flat, m->Wqkv_cache, bqkv_arg,
                                       B, S, H, dk, Qh, Kh, Vh) != AX_OK)
            goto fail;
    } else {
        /* fallback: unfused gemm → split. pre-existing path. */
        int64_t qkv_sh[] = {rows, 3 * D};
        ALLOC_SHAPE(qkv, qkv_sh, 2);
        if (!qkv) goto fail;
        if (ax_compute_gemm(x_flat, m->Wqkv_cache, qkv) != AX_OK) goto fail;

        if (m->use_bias) {
            if (on_cuda && cu && cu->head_interleave_qkv_split_bias) {
                cu->head_interleave_qkv_split_bias(
                    (const float *)qkv->storage->data,
                    (const float *)m->bqkv_cache->storage->data,
                    (float *)Qh->storage->data,
                    (float *)Kh->storage->data,
                    (float *)Vh->storage->data,
                    B, S, H, dk, D);
            } else {
                ax_attn_head_interleave_qkv_split_bias((const float *)qkv->storage->data,
                                                (const float *)m->bqkv_cache->storage->data,
                                                (float *)Qh->storage->data,
                                                (float *)Kh->storage->data,
                                                (float *)Vh->storage->data,
                                                B, S, H, dk, D);
            }
        } else {
            if (on_cuda && cu && cu->head_interleave_qkv_split) {
                cu->head_interleave_qkv_split(
                    (const float *)qkv->storage->data,
                    (float *)Qh->storage->data,
                    (float *)Kh->storage->data,
                    (float *)Vh->storage->data,
                    B, S, H, dk, D);
            } else {
                ax_attn_head_interleave_qkv_split((const float *)qkv->storage->data,
                                           (float *)Qh->storage->data,
                                           (float *)Kh->storage->data,
                                           (float *)Vh->storage->data,
                                           B, S, H, dk, D);
            }
        }
    }

    /* qkv no longer needed (non-recorded, non-F.3.a path); arena reclaims
       it otherwise. qkv is NULL on the F.3.a fast path. */
    if (!record && qkv) ax_tensor_destroy(qkv);

    /* ---- SDPA ----
       F.3.e: when ax_fused_attention_fwd_save_to_flat is available AND
       we're on CPU, write SDPA output DIRECTLY into attn_flat [rows, D],
       skipping both the Oh [BH, S, dk] intermediate and the
       head_deinterleave pass that follows. saves ~6 MB of L3↔DRAM
       traffic per forward call on B1_S2048. the backward companion
       ax_fused_attention_bwd_use_from_flat reads O strided from the
       same attn_flat, so ctx->O_head stays NULL on this path. */
    bool use_f3e_fwd = !on_cuda;
    {
        const char *no_f3e = getenv("AX_NO_F3E");
        if (no_f3e && no_f3e[0] == '1') use_f3e_fwd = false;
    }

    ax_tensor_t *Oh = NULL;
    if (!use_f3e_fwd) {
        ALLOC_SHAPE(Oh, head_sh, 3);
        if (!Oh) goto fail;
    }

    int64_t L_sh[] = {B * H, S};
    ax_tensor_t *L_t = NULL;
    if (record) {
        ALLOC_SHAPE(L_t, L_sh, 2);
        if (!L_t) goto fail;
    }

    /* P_save (post-mask pre-softmax scores) speeds up backward by skipping
       the QK^T recompute. cost is BH*S*S floats (~4 MB on B8 H8 S128).
       only allocate when:
       1. P fits comfortably in cache (≤ ~half L3, threshold 8 MB),
       2. recompute cost dominates per-head fixed overhead — i.e. dk is
          large enough that the saved Q@Kt GEMM is non-trivial relative to
          the softmax-from-saved scan. for very small workloads (S ≤ 128
          AND dk ≤ 64), the fixed bwd overhead (5 GEMMs per head + the
          memory write of 4 MB in fwd) outweighs the saved GEMM, so skip.
       AX_MHA_SAVE_P=1 forces save on (debug); AX_MHA_SAVE_P=0 forces off. */
    ax_tensor_t *P_save_t = NULL;
    int64_t bh = B * H;
    int64_t p_save_bytes = bh * S * S * (int64_t)sizeof(float);
    bool save_enabled = (record && p_save_bytes <= (int64_t)8 * 1024 * 1024);
    /* heuristic: skip when both S and dk are small — with the BLIS path
       the recompute GEMM is cheap enough that the save-write overhead
       in forward eats the gains. */
    if (save_enabled && S <= 128 && dk <= 64) save_enabled = false;
    const char *env = getenv("AX_MHA_SAVE_P");
    if (env) save_enabled = (env[0] == '1');
    if (save_enabled) {
        int64_t P_sh[] = {bh, S, S};
        ALLOC_SHAPE(P_save_t, P_sh, 3);
        /* not fatal if alloc fails — backward will recompute */
    }

    float scale = 1.0f / sqrtf((float)dk);
    float *P_save_ptr = P_save_t ? (float *)P_save_t->storage->data : NULL;

    /* ---- attn_flat [rows, D] allocated up front so the F.3.e fused
       fwd-to-flat can write directly into its rows, skipping the
       separate head_deinterleave pass. ---- */
    int64_t attn_sh[] = {rows, D};
    ax_tensor_t *attn_flat = NULL;
    ALLOC_SHAPE(attn_flat, attn_sh, 2);
    if (!attn_flat) goto fail;

    /* Phase C: dispatch SDPA forward on device. CPU path uses the existing
       CPU FA-style kernel; CUDA path uses cuBLAS stridedBatched + custom
       softmax+L kernel. Phase D will add backward. */
    if (on_cuda && cu && cu->sdpa_fwd) {
        cu->sdpa_fwd(
            (const float *)Qh->storage->data,
            (const float *)Kh->storage->data,
            (const float *)Vh->storage->data,
            (float *)Oh->storage->data,
            L_t ? (float *)L_t->storage->data : NULL,
            bh, S, dk, scale, m->causal, P_save_ptr);
    } else if (use_f3e_fwd) {
        /* F.3.e fused: write SDPA output directly into attn_flat rows,
           skip Oh + head_deinterleave. ctx->O_head stays NULL. */
        ax_fused_attention_fwd_save_to_flat(
            (const float *)Qh->storage->data,
            (const float *)Kh->storage->data,
            (const float *)Vh->storage->data,
            (float *)attn_flat->storage->data,
            L_t ? (float *)L_t->storage->data : NULL,
            P_save_ptr,
            B, S, H, dk, scale, m->causal);
    } else {
        ax_fused_attention_fwd_save(
            (const float *)Qh->storage->data,
            (const float *)Kh->storage->data,
            (const float *)Vh->storage->data,
            (float *)Oh->storage->data,
            L_t ? (float *)L_t->storage->data : NULL,
            P_save_ptr,
            bh, S, dk, scale, m->causal);
    }

    /* ---- merge heads back to [rows, D] (only when Oh was materialized) ---- */
    if (!use_f3e_fwd) {
        /* k.4: dispatch on device through the cuda extension registry. */
        if (on_cuda && cu && cu->head_deinterleave) {
            cu->head_deinterleave((const float *)Oh->storage->data,
                                  (float *)attn_flat->storage->data,
                                  B, S, H, dk);
        } else {
            ax_attn_head_deinterleave((const float *)Oh->storage->data,
                                       (float *)attn_flat->storage->data, B, S, H, dk);
        }
    }

    /* ---- output projection: attn_flat @ Wo + bo ---- */
    int64_t out_sh[] = {B, S, D};
    ax_tensor_t *out = ax_tensor_create(out_sh, 3, AX_FLOAT32);
    if (!out) goto fail;

    /* use a [rows, D] view of the output for GEMM */
    int64_t out_flat_sh[] = {rows, D};
    ax_tensor_t *out_flat_view = ax_tensor_reshape(out, out_flat_sh, 2);
    if (!out_flat_view) { ax_tensor_destroy(out); goto fail; }

    if (ax_compute_gemm(attn_flat, m->Wo, out_flat_view) != AX_OK) {
        ax_tensor_destroy(out_flat_view);
        ax_tensor_destroy(out);
        goto fail;
    }
    if (m->use_bias) {
        if (on_cuda) {
            /* dispatched bias_add (CUDA-aware) */
            ax_compute_bias_add(out_flat_view, m->bo, 1, out_flat_view);
        } else {
            float *od = (float *)out_flat_view->storage->data;
            const float *bd = (const float *)m->bo->storage->data;
            int64_t de = D - (D % AX_VF32_WIDTH);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if (rows * D >= ax_par_threshold_elems)
#endif
            for (int64_t r = 0; r < rows; r++) {
                float *row = od + r * D;
                int64_t c = 0;
                for (; c < de; c += AX_VF32_WIDTH) {
                    ax_vf32 v = ax_vf32_loadu(row + c);
                    ax_vf32 b = ax_vf32_loadu(bd + c);
                    ax_vf32_storeu(row + c, ax_vf32_add(v, b));
                }
                for (; c < D; c++) row[c] += bd[c];
            }
        }
    }
    ax_storage_touch(out->storage);
    ax_tensor_destroy(out_flat_view); /* view; storage kept alive by `out` */

    /* ---- hook up backward ---- */
    if (record) {
        mha_ctx_t *ctx = (mha_ctx_t *)calloc(1, sizeof(mha_ctx_t));
        if (!ctx) {
            /* gracefully degrade to inference output */
            if (inp != input) ax_tensor_destroy(inp);
            return out;
        }
        ctx->layer = m;
        ctx->B = B; ctx->S = S; ctx->D = D; ctx->H = H; ctx->dk = dk;
        /* T1.3: x_flat is now a stack-view aliasing input data (when input
           is contiguous). don't store it in ctx — the stack frame is gone
           by the time backward runs. backward reconstructs an equivalent
           view from self->inputs[0] (which the autograd graph keeps alive). */
        ctx->x_flat    = NULL;
        ctx->Q_head    = Qh;
        ctx->K_head    = Kh;
        ctx->V_head    = Vh;
        ctx->O_head    = Oh;
        ctx->attn_flat = attn_flat;
        ctx->L_tensor  = L_t;
        ctx->P_save_tensor = P_save_t;  /* may be NULL if S too large */

        ax_grad_fn_t *gf = ax_grad_fn_create(mha_backward);
        gf->inputs[0] = input;
        gf->n_inputs = 1;
        gf->ctx = ctx;
        gf->ctx_cleanup = mha_ctx_cleanup;
        out->requires_grad = true;
        out->grad_fn = gf;
    }

    if (inp != input) ax_tensor_destroy(inp);
    return out;

fail:
    if (inp != input) ax_tensor_destroy(inp);
    return NULL;

    #undef ALLOC_SHAPE
    #undef ALLOC_ZEROS
}

/* ================================================================
   backward
   ================================================================ */

/* allocate a param grad lazily (same pattern as other layers) */
static float *param_grad_ptr(ax_tensor_t *p)
{
    if (!p || !p->requires_grad) return NULL;
    if (!p->grad) {
        p->grad = ax_tensor_zeros(p->shape, p->ndim, p->dtype);
        if (!p->grad) return NULL;
    }
    return (float *)p->grad->storage->data;
}

/* SIMD-width row-vector sum reduction: out[c] += sum_r M[r, c] for c<cols.
   one pass over M, writes directly into the (possibly already populated) bias
   grad. parallelized over column tiles to stay within a single memory pass. */
static void sum_rows_acc(const float *M, float *out,
                         int64_t rows, int64_t cols)
{
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t c = 0; c < cols; c++) {
        float s = 0.0f;
        for (int64_t r = 0; r < rows; r++) s += M[r * cols + c];
        out[c] += s;
    }
}

/* accumulate src into dst (SIMD). used where gemm_tn/nt returns into a
   fresh buffer and we want to add that into an existing grad. much faster
   than the previous scalar for-loop. */
static void accumulate_f32(const float *src, float *dst, int64_t n)
{
    int64_t i = 0;
#if defined(AX_HAS_SIMD)
    int64_t ve = n - (n % AX_VF32_WIDTH);
    for (; i < ve; i += AX_VF32_WIDTH) {
        ax_vf32 s = ax_vf32_loadu(src + i);
        ax_vf32 d = ax_vf32_loadu(dst + i);
        ax_vf32_storeu(dst + i, ax_vf32_add(s, d));
    }
#endif
    for (; i < n; i++) dst[i] += src[i];
}

static void mha_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    mha_ctx_t *ctx = (mha_ctx_t *)self->ctx;
    ax_mha_t *m = ctx->layer;
    ax_tensor_t *input = self->inputs[0];

    int64_t B  = ctx->B;
    int64_t S  = ctx->S;
    int64_t D  = ctx->D;
    int64_t H  = ctx->H;
    int64_t dk = ctx->dk;
    int64_t rows = B * S;

    /* T1.1 profile: per-stage cycle counts gated by AX_PROFILE_MHA=1.
       summed across many calls to amortize printing cost. */
    static int prof_enabled = -1;
    if (prof_enabled < 0) prof_enabled = (getenv("AX_PROFILE_MHA") && getenv("AX_PROFILE_MHA")[0] == '1') ? 1 : 0;
    static __thread uint64_t pf_dout_copy=0, pf_wo_grad=0, pf_dattn=0,
                              pf_head_int=0, pf_sdpa_bwd=0, pf_deint=0,
                              pf_dwqkv=0, pf_dx=0;
    static __thread int pf_calls = 0;
    /* high-frequency cycle counter for profile attribution (gated by
       AX_PROFILE_MHA=1). x86 reads rdtsc, aarch64 reads cntvct_el0;
       anything else returns 0 (profile unsupported on that arch). only
       deltas matter so the unit difference between rdtsc cycles and
       cntvct ticks doesn't affect attribution. */
#if defined(__x86_64__) || defined(_M_X64)
    #define PF_TICK() ((uint64_t)__builtin_ia32_rdtsc())
#elif defined(__aarch64__)
    #define PF_TICK() ({ uint64_t _v; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(_v)); _v; })
#else
    #define PF_TICK() ((uint64_t)0)
#endif
    uint64_t pf_t0;

    /* contiguous grad_out flattened to [rows, D].
       T1.4: when grad_out is already contiguous fp32, use a stack-view
       aliasing its data instead of memcpy'ing into a fresh arena tensor.
       saves 2-6 MB BW per backward call. fall back to materialize when
       grad_out is non-contig or wrong dtype. */
    ax_arena_t *arena = ax_backward_arena();
    int64_t flat_sh[] = {rows, D};
    ax_tensor_t *dOut_flat = NULL;
    ax_tensor_t  dOut_view;
    ax_storage_t dOut_view_st;
    bool gout_contig = (grad_out && grad_out->storage && grad_out->storage->data
                        && grad_out->dtype == AX_FLOAT32);
    pf_t0 = prof_enabled ? PF_TICK() : 0;
    if (gout_contig) {
        mha_make_stack_view(&dOut_view, &dOut_view_st,
                             (const float *)grad_out->storage->data, rows, D);
        dOut_flat = &dOut_view;
    } else {
        dOut_flat = ax_tensor_arena_create(arena, flat_sh, 2, AX_FLOAT32);
        if (!dOut_flat) return;
        memcpy(dOut_flat->storage->data, grad_out->storage->data,
               (size_t)rows * (size_t)D * sizeof(float));
    }
    if (prof_enabled) pf_dout_copy += PF_TICK() - pf_t0;

    /* ================================================================
       step 1 — output projection backward
       ================================================================ */
    /* grad_Wo += attn_flat^T @ dOut_flat   (single [D, rows] @ [rows, D] gemm).
       uses ax_gemm_set_skip_init(true) so the gemm accumulates directly
       into m->Wo->grad, eliminating the scratch [D, D] intermediate and
       the separate accumulate_f32 pass (~1 MB traffic on D=1024, ~4 MB
       on D=1024). same pattern train_step_fused / train_step_v4 use. */
    if (m->Wo->requires_grad) {
        float *dWo = param_grad_ptr(m->Wo);
        if (dWo) {
            pf_t0 = prof_enabled ? PF_TICK() : 0;
            ax_gemm_set_skip_init(true);
            ax_status_t s = ax_compute_gemm_tn(ctx->attn_flat, dOut_flat, m->Wo->grad);
            ax_gemm_set_skip_init(false);
            if (prof_enabled) pf_wo_grad += PF_TICK() - pf_t0;
            if (s != AX_OK) return;
            ax_storage_touch(m->Wo->grad->storage);
        }
    }
    /* grad_bo += col-sum(dOut_flat) */
    if (m->use_bias && m->bo->requires_grad) {
        float *dbo = param_grad_ptr(m->bo);
        if (dbo) {
            sum_rows_acc((const float *)dOut_flat->storage->data, dbo, rows, D);
            ax_storage_touch(m->bo->grad->storage);
        }
    }
    /* d_attn_flat = dOut_flat @ Wo^T, then head-interleave → dO_head.
       F.3.d path: when ax_compute_dattn_head_gemm_nt is available and
       we're on CPU, fuse the two passes into one kernel that streams
       dout through Wo^T and emits directly into the [B, H, S, dk]
       head-interleaved layout — saves the d_attn_flat [rows, D]
       intermediate (~6 MB on B1_S2048) and one full-rows memory pass. */
    const ax_cuda_extension_t *cu = ax_compute_get_cuda_extension();
    int64_t head_sh[] = {B * H, S, dk};
    ax_tensor_t *dO_head = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    if (!dO_head) return;
    bool bwd_on_cuda = (dOut_flat->storage->device != AX_DEVICE_CPU);
    bool use_f3d = (!bwd_on_cuda) && ax_compute_has_dattn_head_gemm_nt();
    {
        const char *no_f3d = getenv("AX_NO_F3D");
        if (no_f3d && no_f3d[0] == '1') use_f3d = false;
    }

    if (use_f3d) {
        pf_t0 = prof_enabled ? PF_TICK() : 0;
        if (ax_compute_dattn_head_gemm_nt(dOut_flat, m->Wo, B, S, H, dk, dO_head) != AX_OK)
            return;
        if (prof_enabled) pf_dattn += PF_TICK() - pf_t0;
        /* head_int profile bucket absorbed into dattn above (fused) */
    } else {
        ax_tensor_t *d_attn_flat = ax_tensor_arena_create(arena, flat_sh, 2, AX_FLOAT32);
        if (!d_attn_flat) return;
        pf_t0 = prof_enabled ? PF_TICK() : 0;
        if (ax_compute_gemm_nt(dOut_flat, m->Wo, d_attn_flat) != AX_OK) return;
        if (prof_enabled) pf_dattn += PF_TICK() - pf_t0;

        pf_t0 = prof_enabled ? PF_TICK() : 0;
        if (bwd_on_cuda && cu && cu->head_interleave) {
            cu->head_interleave((const float *)d_attn_flat->storage->data,
                                (float *)dO_head->storage->data,
                                B, S, H, dk);
        } else {
            ax_attn_head_interleave((const float *)d_attn_flat->storage->data,
                                     (float *)dO_head->storage->data, B, S, H, dk);
        }
        if (prof_enabled) pf_head_int += PF_TICK() - pf_t0;
    }

    ax_tensor_t *dQh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    ax_tensor_t *dKh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    ax_tensor_t *dVh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    if (!dQh || !dKh || !dVh) return;

    float scale = 1.0f / sqrtf((float)dk);
    const float *P_saved = ctx->P_save_tensor
        ? (const float *)ctx->P_save_tensor->storage->data : NULL;
    pf_t0 = prof_enabled ? PF_TICK() : 0;
    /* k.4: dispatch sdpa backward on device through the cuda registry. */
    if (bwd_on_cuda && cu && cu->sdpa_bwd) {
        cu->sdpa_bwd(
            (const float *)ctx->Q_head->storage->data,
            (const float *)ctx->K_head->storage->data,
            (const float *)ctx->V_head->storage->data,
            (const float *)ctx->O_head->storage->data,
            (const float *)dO_head->storage->data,
            (const float *)ctx->L_tensor->storage->data,
            (float *)dQh->storage->data,
            (float *)dKh->storage->data,
            (float *)dVh->storage->data,
            B * H, S, dk, scale, m->causal, P_saved);
    } else if (ctx->O_head == NULL) {
        /* F.3.e companion: forward skipped Oh materialization; bwd reads
           O strided from attn_flat. saves the ~6 MB Oh intermediate
           per train step. companion to the F.3.e branch in mha_forward. */
        ax_fused_attention_bwd_use_from_flat(
            (const float *)ctx->Q_head->storage->data,
            (const float *)ctx->K_head->storage->data,
            (const float *)ctx->V_head->storage->data,
            (const float *)ctx->attn_flat->storage->data,
            (const float *)dO_head->storage->data,
            (const float *)ctx->L_tensor->storage->data,
            P_saved,
            (float *)dQh->storage->data,
            (float *)dKh->storage->data,
            (float *)dVh->storage->data,
            B, S, H, dk, scale, m->causal);
    } else {
        ax_fused_attention_bwd_use(
            (const float *)ctx->Q_head->storage->data,
            (const float *)ctx->K_head->storage->data,
            (const float *)ctx->V_head->storage->data,
            (const float *)ctx->O_head->storage->data,
            (const float *)dO_head->storage->data,
            (const float *)ctx->L_tensor->storage->data,
            P_saved,
            (float *)dQh->storage->data,
            (float *)dKh->storage->data,
            (float *)dVh->storage->data,
            B * H, S, dk, scale, m->causal);
    }
    if (prof_enabled) pf_sdpa_bwd += PF_TICK() - pf_t0;

    /* ================================================================
       step 3 — FUSED de-interleave into a single dQKV [rows, 3D] buffer,
                so the following gradient GEMMs can each be one big
                call instead of three separate D-wide ones.
       ================================================================ */
    int64_t qkv_sh[] = {rows, 3 * D};
    ax_tensor_t *dQKV = ax_tensor_arena_create(arena, qkv_sh, 2, AX_FLOAT32);
    if (!dQKV) return;
    pf_t0 = prof_enabled ? PF_TICK() : 0;
    /* k.4: dispatch on device through the cuda extension registry. */
    if (bwd_on_cuda && cu && cu->head_deinterleave_qkv_merge) {
        cu->head_deinterleave_qkv_merge(
            (const float *)dQh->storage->data,
            (const float *)dKh->storage->data,
            (const float *)dVh->storage->data,
            (float *)dQKV->storage->data,
            B, S, H, dk, D);
    } else {
        ax_attn_head_deinterleave_qkv_merge((const float *)dQh->storage->data,
                                     (const float *)dKh->storage->data,
                                     (const float *)dVh->storage->data,
                                     (float *)dQKV->storage->data,
                                     B, S, H, dk, D);
    }
    if (prof_enabled) pf_deint += PF_TICK() - pf_t0;

    /* ================================================================
       step 4 — FUSED weight grad: dWqkv = x_flat^T @ dQKV → [D, 3D].
                one big gemm_tn covers all three of Wq/Wk/Wv, then we
                split the columns into their respective grad tensors
                with a SIMD accumulate. 3x fewer gemm calls, better
                cache reuse on x_flat (single pass over it vs three).
       ================================================================ */
    bool any_wqkv_grad = m->Wq->requires_grad || m->Wk->requires_grad || m->Wv->requires_grad;
    bool all_wqkv_grad = m->Wq->requires_grad && m->Wk->requires_grad && m->Wv->requires_grad;
    if (any_wqkv_grad) {
        /* T1.3: x_flat = stack-view of input data (forward never copied).
           input is alive because the autograd graph holds a ref via
           self->inputs[0]. fall back to ctx->x_flat if forward chose
           the materialized path (non-contig input). */
        ax_tensor_t  x_flat_view;
        ax_storage_t x_flat_view_st;
        ax_tensor_t *x_flat_for_bwd = ctx->x_flat;
        if (!x_flat_for_bwd && input && input->storage && input->storage->data) {
            mha_make_stack_view(&x_flat_view, &x_flat_view_st,
                                 (const float *)input->storage->data, rows, D);
            x_flat_for_bwd = &x_flat_view;
        }
        if (!x_flat_for_bwd) return;

        /* fast path: shape-dispatched fused multi-output dwqkv kernel.
           ax_compute_dwqkv_split_acc (commit 408600e) writes the gemm
           output directly to dWq/dWk/dWv with accumulate, eliminating
           the 12 MB [D, 3D] intermediate. measured -9% on B1_S512
           (D=1024) where the intermediate doesn't fit L3 well; +5%
           regression on B8_S128 / B1_S2048 (D≤768) where the
           intermediate fits L3 cheaply and per-jc dest dispatch adds
           overhead. so gate on D >= 1024.

           additional gates: requires all three of Wq/Wk/Wv to need
           grads (the kernel doesn't support partial — fall back if
           only some need grads), and requires the backend to expose
           the entry. */
        if (all_wqkv_grad && D >= 1024 && ax_compute_has_dwqkv_split_acc()) {
            float *dq_init = param_grad_ptr(m->Wq);
            float *dk_init = param_grad_ptr(m->Wk);
            float *dv_init = param_grad_ptr(m->Wv);
            (void)dq_init; (void)dk_init; (void)dv_init;
            if (m->Wq->grad && m->Wk->grad && m->Wv->grad) {
                pf_t0 = prof_enabled ? PF_TICK() : 0;
                if (ax_compute_dwqkv_split_acc(x_flat_for_bwd, dQKV,
                                                m->Wq->grad, m->Wk->grad,
                                                m->Wv->grad) != AX_OK) return;
                if (prof_enabled) pf_dwqkv += PF_TICK() - pf_t0;
                goto dwqkv_done;
            }
        }

        /* fallback: materialize then split (one gemm_TN into [D, 3D]
           intermediate, then a single-pass SIMD scatter into the 3
           grad tensors). default for D < 1024, partial-grad cases,
           and backends without the fused entry. */
        {
            int64_t dW_sh[] = {D, 3 * D};
            ax_tensor_t *dWqkv = ax_tensor_arena_create(arena, dW_sh, 2, AX_FLOAT32);
            if (!dWqkv) return;

            pf_t0 = prof_enabled ? PF_TICK() : 0;
            if (ax_compute_gemm_tn(x_flat_for_bwd, dQKV, dWqkv) != AX_OK) return;
            if (prof_enabled) pf_dwqkv += PF_TICK() - pf_t0;

            const float *dw_data = (const float *)dWqkv->storage->data;
            float *dWq_g = m->Wq->requires_grad ? param_grad_ptr(m->Wq) : NULL;
            float *dWk_g = m->Wk->requires_grad ? param_grad_ptr(m->Wk) : NULL;
            float *dWv_g = m->Wv->requires_grad ? param_grad_ptr(m->Wv) : NULL;
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < D; i++) {
                const float *src = dw_data + i * 3 * D;
                if (dWq_g) accumulate_f32(src,           dWq_g + i * D, D);
                if (dWk_g) accumulate_f32(src + D,       dWk_g + i * D, D);
                if (dWv_g) accumulate_f32(src + 2 * D,   dWv_g + i * D, D);
            }
            if (dWq_g) ax_storage_touch(m->Wq->grad->storage);
            if (dWk_g) ax_storage_touch(m->Wk->grad->storage);
            if (dWv_g) ax_storage_touch(m->Wv->grad->storage);
        }
        dwqkv_done: ;
    }

    /* ================================================================
       step 5 — FUSED bias grads: dbqkv = col-sum(dQKV) → [3D], then
                split into bq/bk/bv. one reduction pass vs three.
       ================================================================ */
    if (m->use_bias) {
        bool any_b = m->bq->requires_grad || m->bk->requires_grad || m->bv->requires_grad;
        if (any_b) {
            float *dbqkv = (float *)ax_tensor_arena_zeros(arena, qkv_sh + 0, 1, AX_FLOAT32) ? NULL : NULL;
            /* simpler: reduce into stack-ish arena tensor of shape [3D] */
            int64_t b_sh[] = {3 * D};
            ax_tensor_t *dbqkv_t = ax_tensor_arena_zeros(arena, b_sh, 1, AX_FLOAT32);
            if (!dbqkv_t) return;
            dbqkv = (float *)dbqkv_t->storage->data;
            sum_rows_acc((const float *)dQKV->storage->data, dbqkv, rows, 3 * D);
            #define ACC_BIAS(b, off) do {                                     \
                if ((b)->requires_grad) {                                     \
                    float *dp = param_grad_ptr(b);                            \
                    if (dp) {                                                 \
                        accumulate_f32(dbqkv + (off), dp, D);                 \
                        ax_storage_touch((b)->grad->storage);                 \
                    }                                                         \
                }                                                             \
            } while (0)
            ACC_BIAS(m->bq, 0);
            ACC_BIAS(m->bk, D);
            ACC_BIAS(m->bv, 2 * D);
            #undef ACC_BIAS
        }
    }

    /* ================================================================
       step 6 — FUSED input grad: dX = dQKV @ Wqkv^T → [rows, D], single
                gemm_nt covering the three chained projections in one
                shot. accumulated into input->grad via SIMD axpy.
       ================================================================ */
    if (input->requires_grad) {
        if (!input->grad) {
            input->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
            if (!input->grad) return;
        }
        ax_tensor_t *dX = ax_tensor_arena_create(arena, flat_sh, 2, AX_FLOAT32);
        if (!dX) return;
        /* dQKV is [rows, 3D]; Wqkv_cache is [D, 3D]; dX = dQKV @ Wqkv^T */
        pf_t0 = prof_enabled ? PF_TICK() : 0;
        if (ax_compute_gemm_nt(dQKV, m->Wqkv_cache, dX) != AX_OK) return;
        if (prof_enabled) pf_dx += PF_TICK() - pf_t0;

        float *gx = (float *)input->grad->storage->data;
        accumulate_f32((const float *)dX->storage->data, gx, rows * D);
        ax_storage_touch(input->grad->storage);
    }

    if (prof_enabled) {
        pf_calls++;
        if ((pf_calls % 5) == 0) {
            uint64_t total = pf_dout_copy + pf_wo_grad + pf_dattn + pf_head_int
                           + pf_sdpa_bwd + pf_deint + pf_dwqkv + pf_dx;
            if (total > 0) {
                fprintf(stderr,
                    "axiom: mha_backward [B=%ld S=%ld D=%ld H=%ld, %d calls] "
                    "dout_cp=%.0f%% wo_grad=%.0f%% dattn=%.0f%% "
                    "head_int=%.0f%% sdpa_bwd=%.0f%% deint=%.0f%% "
                    "dwqkv=%.0f%% dx=%.0f%%\n",
                    (long)B, (long)S, (long)D, (long)H, pf_calls,
                    100.0 * (double)pf_dout_copy / (double)total,
                    100.0 * (double)pf_wo_grad   / (double)total,
                    100.0 * (double)pf_dattn     / (double)total,
                    100.0 * (double)pf_head_int  / (double)total,
                    100.0 * (double)pf_sdpa_bwd  / (double)total,
                    100.0 * (double)pf_deint     / (double)total,
                    100.0 * (double)pf_dwqkv     / (double)total,
                    100.0 * (double)pf_dx        / (double)total);
            }
        }
    }
    #undef PF_TICK

    /* ctx is freed by ctx_cleanup (free). saved arena tensors are reclaimed
       when the forward arena resets in ax_graph_cleanup. */
}

/* ================================================================
   lifecycle
   ================================================================ */

static void mha_destroy(ax_layer_t *self)
{
    ax_mha_t *m = (ax_mha_t *)self;
    if (m->Wq) ax_tensor_destroy(m->Wq);
    if (m->Wk) ax_tensor_destroy(m->Wk);
    if (m->Wv) ax_tensor_destroy(m->Wv);
    if (m->Wo) ax_tensor_destroy(m->Wo);
    if (m->bq) ax_tensor_destroy(m->bq);
    if (m->bk) ax_tensor_destroy(m->bk);
    if (m->bv) ax_tensor_destroy(m->bv);
    if (m->bo) ax_tensor_destroy(m->bo);
    if (m->Wqkv_cache) ax_tensor_destroy(m->Wqkv_cache);
    if (m->bqkv_cache) ax_tensor_destroy(m->bqkv_cache);
    free(m);
}

ax_layer_t *ax_mha_create(int64_t d_model, int n_heads,
                           bool use_bias, bool causal)
{
    if (d_model <= 0 || n_heads <= 0) return NULL;
    if (d_model % n_heads != 0) {
        ax_err_set(AX_ERR_INVALID_SHAPE,
                   "ax_mha_create: d_model=%lld must be divisible by n_heads=%d",
                   (long long)d_model, n_heads);
        return NULL;
    }

    ax_mha_t *m = (ax_mha_t *)calloc(1, sizeof(ax_mha_t));
    AX_RETURN_NULL_IF_ALLOC_FAIL(m, "ax_mha_create");

    m->base.ops.forward = mha_forward;
    m->base.ops.destroy = mha_destroy;
    m->base.type = AX_LAYER_MHA;
    m->base.training = true;
    m->base.input_features = d_model;
    m->base.output_features = d_model;

    m->d_model = d_model;
    m->n_heads = n_heads;
    m->d_k = d_model / n_heads;
    m->use_bias = use_bias;
    m->causal = causal;

    int64_t ws[] = {d_model, d_model};
    m->Wq = ax_tensor_create(ws, 2, AX_FLOAT32);
    m->Wk = ax_tensor_create(ws, 2, AX_FLOAT32);
    m->Wv = ax_tensor_create(ws, 2, AX_FLOAT32);
    m->Wo = ax_tensor_create(ws, 2, AX_FLOAT32);
    if (!m->Wq || !m->Wk || !m->Wv || !m->Wo) { mha_destroy((ax_layer_t *)m); return NULL; }

    /* xavier / kaiming_uniform is fine here; transformers typically use
       xavier (fan_in + fan_out) on projection weights. kaiming was tuned
       for relu so use xavier for a plain linear projection. */
    ax_init_xavier_uniform(m->Wq, d_model, d_model);
    ax_init_xavier_uniform(m->Wk, d_model, d_model);
    ax_init_xavier_uniform(m->Wv, d_model, d_model);
    ax_init_xavier_uniform(m->Wo, d_model, d_model);

    m->Wq->requires_grad = true;
    m->Wk->requires_grad = true;
    m->Wv->requires_grad = true;
    m->Wo->requires_grad = true;

    m->base.params[0] = m->Wq;
    m->base.params[1] = m->Wk;
    m->base.params[2] = m->Wv;
    m->base.params[3] = m->Wo;
    m->base.n_params = 4;

    if (use_bias) {
        int64_t bs[] = {d_model};
        m->bq = ax_tensor_zeros(bs, 1, AX_FLOAT32);
        m->bk = ax_tensor_zeros(bs, 1, AX_FLOAT32);
        m->bv = ax_tensor_zeros(bs, 1, AX_FLOAT32);
        m->bo = ax_tensor_zeros(bs, 1, AX_FLOAT32);
        if (!m->bq || !m->bk || !m->bv || !m->bo) { mha_destroy((ax_layer_t *)m); return NULL; }
        m->bq->requires_grad = true;
        m->bk->requires_grad = true;
        m->bv->requires_grad = true;
        m->bo->requires_grad = true;
        m->base.params[4] = m->bq;
        m->base.params[5] = m->bk;
        m->base.params[6] = m->bv;
        m->base.params[7] = m->bo;
        m->base.n_params = 8;
    }

    /* Wqkv_cache / bqkv_cache are allocated lazily in refresh_fused_qkv */
    m->Wqkv_cache = NULL;
    m->bqkv_cache = NULL;
    m->Wq_gen = 0;
    m->Wk_gen = 0;
    m->Wv_gen = 0;

    return (ax_layer_t *)m;
}
