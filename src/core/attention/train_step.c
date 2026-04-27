/* attention/train_step.c — fused MHA forward+backward in one call.

   bypasses the autograd tape used by mha_forward + ax_backward. that
   path serves the general case (mixed grad-on/off subgraphs, dx
   propagation, layered chain rule) but adds per-stage overhead:
     - each gemm/softmax allocates a full ax_tensor_t into the forward
       arena, threaded into a grad_fn struct
     - the backward walk re-traverses the tape, calls the per-op grad
       function, which decodes the ctx then calls the same low-level
       compute primitives
     - intermediate arena tensors materialise even when their data is
       only consumed once and could live in registers / TLS scratch

   the train-step path runs the SAME compute primitives (refresh_qkv,
   gemm_tn / nt, sdpa_fwd / bwd, head_interleave / deinterleave) but:
     - holds intermediates in a per-thread scratch arena reset each call
     - writes weight grads directly via ax_gemm_set_skip_init(true)
       (no scratch + accumulate two-step)
     - skips the autograd record/playback bookkeeping entirely

   on this hardware (AVX2 i5-12500H) per-op overhead is small relative
   to the gemm work, so the ceiling for a pure inline+arena pass is
   ~5-15 % over the autograd path. real parity with TF's
   @tf.function(jit_compile=True) needs cross-stage tile fusion (next
   phase) — this file is the scaffold those fusions land on. */

#include "axiom/attention.h"
#include "axiom/compute.h"
#include "axiom/error.h"
#include "axiom/memory.h"
#include "axiom/tensor.h"
#include "axiom/internal/cuda_extension.h"
#include "axiom/internal/attn_tunables.h"
#include "internal.h"

#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../compute/backends/simd_defs.h"

/* extern declared in src/core/conv/internal.h, defined in
   src/compute/dispatch.c. flips the next gemm to accumulate (skip the
   pre-zero) so callers can write directly into a pre-populated dst. */
extern void ax_gemm_set_skip_init(bool v);

/* ================================================================
   per-thread scratch arena
   ================================================================ */

__thread ax_arena_t *tl_train_shared_arena = NULL;

ax_arena_t *ax_train_get_scratch_arena(void)
{
    if (!tl_train_shared_arena) {
        tl_train_shared_arena = ax_arena_create(0);
        if (!tl_train_shared_arena) return NULL;
    } else {
        ax_arena_reset(tl_train_shared_arena);
    }
    return tl_train_shared_arena;
}

static ax_arena_t *get_scratch_arena(void)
{
    return ax_train_get_scratch_arena();
}

/* ================================================================
   helpers shared with attention.c (kept private to this TU)
   ================================================================ */

static inline void make_stack_view(ax_tensor_t *tv, ax_storage_t *st,
                                    const float *data,
                                    int64_t rows, int64_t cols)
{
    st->data         = (void *)(uintptr_t)data;
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

/* lazy grad-tensor allocation, mirrors param_grad_ptr in attention.c */
static float *ensure_grad_ptr(ax_tensor_t *p)
{
    if (!p || !p->requires_grad) return NULL;
    if (!p->grad) {
        p->grad = ax_tensor_zeros(p->shape, p->ndim, p->dtype);
        if (!p->grad) return NULL;
    }
    return (float *)p->grad->storage->data;
}

/* fp32 += accumulator. inlined to keep this file self-contained. */
static void acc_f32(const float *src, float *dst, int64_t n)
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

/* tls scratch for col-sum when cols exceeds stack limit */
static __thread float *tl_csa_buf;
static __thread int64_t tl_csa_cap;

/* row-major col-sum body: private accumulator per thread, omp for over
   rows, critical reduction. sequential access + simd inner loop. */
static void col_sum_inner_(const float *M, float *out,
                            int64_t rows, int64_t cols)
{
    float stk[2048];
    float *acc = stk;
    if (cols > 2048) {
        if (tl_csa_cap < cols) {
            free(tl_csa_buf);
            tl_csa_buf = (float *)malloc((size_t)cols * sizeof(float));
            tl_csa_cap = cols;
        }
        acc = tl_csa_buf;
    }
    memset(acc, 0, (size_t)cols * sizeof(float));
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int64_t r = 0; r < rows; r++) {
        const float *row = M + r * cols;
        int64_t c = 0;
#if defined(AX_HAS_SIMD)
        for (int64_t ve = cols - cols % AX_VF32_WIDTH; c < ve; c += AX_VF32_WIDTH)
            ax_vf32_storeu(acc + c, ax_vf32_add(ax_vf32_loadu(acc + c),
                                                 ax_vf32_loadu(row + c)));
#endif
        for (; c < cols; c++) acc[c] += row[c];
    }
#ifdef _OPENMP
    #pragma omp critical
#endif
    {
        int64_t c = 0;
#if defined(AX_HAS_SIMD)
        for (int64_t ve = cols - cols % AX_VF32_WIDTH; c < ve; c += AX_VF32_WIDTH)
            ax_vf32_storeu(out + c, ax_vf32_add(ax_vf32_loadu(acc + c),
                                                 ax_vf32_loadu(out + c)));
#endif
        for (; c < cols; c++) out[c] += acc[c];
    }
}

/* col-sum reduction: out[c] += sum_r M[r, c]. row-major pass. */
static void col_sum_acc(const float *M, float *out,
                         int64_t rows, int64_t cols)
{
#ifdef _OPENMP
    if (omp_in_parallel()) { col_sum_inner_(M, out, rows, cols); return; }
    #pragma omp parallel
    col_sum_inner_(M, out, rows, cols);
#else
    col_sum_inner_(M, out, rows, cols);
#endif
}

static void col_sum_hm_one(const float *src, float *out,
                             int64_t B, int64_t S, int64_t H, int64_t dk)
{
    int64_t BH = B * H;
    for (int64_t bh = 0; bh < BH; bh++) {
        int64_t h = bh % H;
        float *dst = out + h * dk;
        for (int64_t s = 0; s < S; s++) {
            const float *row = src + (bh * S + s) * dk;
            int64_t d = 0;
#if defined(AX_HAS_SIMD)
            for (int64_t ve = dk - dk % AX_VF32_WIDTH; d < ve; d += AX_VF32_WIDTH)
                ax_vf32_storeu(dst + d, ax_vf32_add(ax_vf32_loadu(dst + d),
                                                     ax_vf32_loadu(row + d)));
#endif
            for (; d < dk; d++) dst[d] += row[d];
        }
    }
}

/* fill n elements with a scalar value (used for default dout=ones). */
static void fill_f32(float *p, float v, int64_t n)
{
    int64_t i = 0;
#if defined(AX_HAS_SIMD)
    ax_vf32 vv = ax_vf32_set1(v);
    int64_t ve = n - (n % AX_VF32_WIDTH);
    for (; i < ve; i += AX_VF32_WIDTH) ax_vf32_storeu(p + i, vv);
#endif
    for (; i < n; i++) p[i] = v;
}

/* ================================================================
   public API
   ================================================================ */

ax_status_t ax_mha_train_step(ax_layer_t *layer,
                               const ax_tensor_t *x,
                               const ax_tensor_t *dout,
                               ax_tensor_t *y_out)
{
    if (!layer || !x || !y_out) {
        ax_err_set(AX_ERR_NULL_ARG, "ax_mha_train_step: NULL arg");
        return AX_ERR_NULL_ARG;
    }
    ax_mha_t *m = (ax_mha_t *)layer;
    if (x->ndim != 3 || y_out->ndim != 3) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "train_step: x/y must be 3-D");
        return AX_ERR_SHAPE_MISMATCH;
    }
    int64_t B  = x->shape[0];
    int64_t S  = x->shape[1];
    int64_t D  = x->shape[2];
    if (D != m->d_model) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "train_step: x last dim != d_model");
        return AX_ERR_SHAPE_MISMATCH;
    }
    if (y_out->shape[0] != B || y_out->shape[1] != S || y_out->shape[2] != D) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "train_step: y_out shape mismatch");
        return AX_ERR_SHAPE_MISMATCH;
    }
    if (x->dtype != AX_FLOAT32 || y_out->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "train_step: must be fp32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    if (dout) {
        if (dout->ndim != 3 ||
            dout->shape[0] != B || dout->shape[1] != S || dout->shape[2] != D ||
            dout->dtype != AX_FLOAT32) {
            ax_err_set(AX_ERR_SHAPE_MISMATCH, "train_step: dout shape mismatch");
            return AX_ERR_SHAPE_MISMATCH;
        }
    }

    int64_t H    = m->n_heads;
    int64_t dk   = m->d_k;
    int64_t rows = B * S;
    int64_t bh   = B * H;

    /* per-stage profile, gated on AX_PROFILE_MHA=1. summed across many
       calls so the printf cost is amortised. mirrors the mha_backward
       profile in attention.c so the two paths can be A/B'd directly. */
    static int prof_enabled = -1;
    if (prof_enabled < 0) {
        const char *pe = getenv("AX_PROFILE_MHA");
        prof_enabled = (pe && pe[0] == '1') ? 1 : 0;
    }
    static __thread uint64_t pf_qkv=0, pf_split=0, pf_sdpa=0, pf_deint=0,
                              pf_wo=0, pf_dout=0, pf_wog=0, pf_dattn=0,
                              pf_hint=0, pf_sdpab=0, pf_dqkv_merge=0,
                              pf_dwqkv=0, pf_dbqkv=0;
    static __thread int pf_calls = 0;
#if defined(__x86_64__) || defined(_M_X64)
    #define PF_TICK() ((uint64_t)__builtin_ia32_rdtsc())
#elif defined(__aarch64__)
    #define PF_TICK() ({ uint64_t _v; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(_v)); _v; })
#else
    #define PF_TICK() ((uint64_t)0)
#endif
    uint64_t pf_t0;

    ax_arena_t *arena = get_scratch_arena();
    if (!arena) {
        ax_err_set(AX_ERR_ALLOC, "train_step: scratch arena alloc failed");
        return AX_ERR_ALLOC;
    }

    /* keep fused [D, 3D] Wqkv panel + bias up to date */
    ax_attn_refresh_fused_qkv(m);

    /* views over input / output flat layouts */
    ax_tensor_t  x_flat_v;
    ax_storage_t x_flat_st;
    make_stack_view(&x_flat_v, &x_flat_st,
                     (const float *)x->storage->data, rows, D);

    ax_tensor_t  y_flat_v;
    ax_storage_t y_flat_st;
    make_stack_view(&y_flat_v, &y_flat_st,
                     (const float *)y_out->storage->data, rows, D);

    /* ============================================================
       FORWARD
       ============================================================ */

    /* qkv projection + head-split: prefer F.3.a fused path which writes
       directly to head-interleaved Qh/Kh/Vh and skips the [rows, 3D] qkv
       intermediate (~18 MB on B1_S2048_D768). falls back to gemm + split
       when the backend lacks the fused slot. */
    int64_t qkv_sh[]  = {rows, 3 * D};
    int64_t head_sh[] = {bh, S, dk};
    int64_t L_sh[]    = {bh, S};
    int64_t flat_sh[] = {rows, D};

    ax_tensor_t *Qh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    ax_tensor_t *Kh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    ax_tensor_t *Vh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    if (!Qh || !Kh || !Vh) return AX_ERR_ALLOC;

    pf_t0 = prof_enabled ? PF_TICK() : 0;
    if (ax_compute_has_qkv_head_gemm()) {
        const ax_tensor_t *bqkv_arg = m->use_bias ? m->bqkv_cache : NULL;
        if (ax_compute_qkv_head_gemm(&x_flat_v, m->Wqkv_cache, bqkv_arg,
                                       B, S, H, dk, Qh, Kh, Vh) != AX_OK)
            return AX_ERR_BACKEND;
        if (prof_enabled) pf_qkv += PF_TICK() - pf_t0;
        /* split is folded into the fused gemm; pf_split stays zero */
    } else {
        ax_tensor_t *qkv = ax_tensor_arena_create(arena, qkv_sh, 2, AX_FLOAT32);
        if (!qkv) return AX_ERR_ALLOC;
        if (ax_compute_gemm(&x_flat_v, m->Wqkv_cache, qkv) != AX_OK) return AX_ERR_BACKEND;
        if (prof_enabled) pf_qkv += PF_TICK() - pf_t0;

        pf_t0 = prof_enabled ? PF_TICK() : 0;
        if (m->use_bias) {
            ax_attn_head_interleave_qkv_split_bias(
                (const float *)qkv->storage->data,
                (const float *)m->bqkv_cache->storage->data,
                (float *)Qh->storage->data,
                (float *)Kh->storage->data,
                (float *)Vh->storage->data,
                B, S, H, dk, D);
        } else {
            ax_attn_head_interleave_qkv_split(
                (const float *)qkv->storage->data,
                (float *)Qh->storage->data,
                (float *)Kh->storage->data,
                (float *)Vh->storage->data,
                B, S, H, dk, D);
        }
        if (prof_enabled) pf_split += PF_TICK() - pf_t0;
    }

    /* SDPA forward — saves L (always) and optionally P. F.3.e fused entry
       writes the per-head Oh contributions directly into attn_flat layout
       (no Oh tensor materialised) when AX_NO_F3E is unset; the Oh tensor
       is only allocated for the fallback path. */
    ax_tensor_t *L_t = ax_tensor_arena_create(arena, L_sh, 2, AX_FLOAT32);
    if (!L_t) return AX_ERR_ALLOC;

    ax_tensor_t *P_save_t = NULL;
    int64_t p_save_bytes = bh * S * S * (int64_t)sizeof(float);
    bool save_p = (p_save_bytes <= ax_attn_tunable_save_p_max_bytes());
    if (save_p && S <= ax_attn_tunable_save_p_small_exclusion_s() && (S * dk) <= ax_attn_tunable_save_p_small_exclusion_sk()) save_p = false;
    static int env_save_p_resolved = 0;
    static int env_save_p_val = -1;
    if (!env_save_p_resolved) {
        const char *e = getenv("AX_MHA_SAVE_P");
        env_save_p_val = e ? (e[0] == '1' ? 1 : 0) : -1;
        env_save_p_resolved = 1;
    }
    if (env_save_p_val >= 0) save_p = (bool)env_save_p_val;
    if (save_p) {
        int64_t P_sh[] = {bh, S, S};
        P_save_t = ax_tensor_arena_create(arena, P_sh, 3, AX_FLOAT32);
    }
    float scale = 1.0f / sqrtf((float)dk);
    float *P_save_ptr = P_save_t ? (float *)P_save_t->storage->data : NULL;

    static int use_f3e_resolved = 0;
    static int use_f3e = 1;
    if (!use_f3e_resolved) {
        const char *e = getenv("AX_NO_F3E");
        use_f3e = (e && e[0] == '1') ? 0 : 1;
        use_f3e_resolved = 1;
    }

    ax_tensor_t *Oh = NULL;
    ax_tensor_t *attn_flat = ax_tensor_arena_create(arena, flat_sh, 2, AX_FLOAT32);
    if (!attn_flat) return AX_ERR_ALLOC;

    pf_t0 = prof_enabled ? PF_TICK() : 0;
    if (use_f3e) {
        /* F.3.e: write attn_flat directly, skip Oh + head_deinterleave */
        ax_fused_attention_fwd_save_to_flat(
            (const float *)Qh->storage->data,
            (const float *)Kh->storage->data,
            (const float *)Vh->storage->data,
            (float *)attn_flat->storage->data,
            (float *)L_t->storage->data,
            P_save_ptr,
            B, S, H, dk, scale, m->causal);
        if (prof_enabled) pf_sdpa += PF_TICK() - pf_t0;
        /* pf_deint stays zero — fused into pf_sdpa */
    } else {
        Oh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
        if (!Oh) return AX_ERR_ALLOC;
        ax_fused_attention_fwd_save(
            (const float *)Qh->storage->data,
            (const float *)Kh->storage->data,
            (const float *)Vh->storage->data,
            (float *)Oh->storage->data,
            (float *)L_t->storage->data,
            P_save_ptr,
            bh, S, dk, scale, m->causal);
        if (prof_enabled) pf_sdpa += PF_TICK() - pf_t0;

        pf_t0 = prof_enabled ? PF_TICK() : 0;
        ax_attn_head_deinterleave(
            (const float *)Oh->storage->data,
            (float *)attn_flat->storage->data, B, S, H, dk);
        if (prof_enabled) pf_deint += PF_TICK() - pf_t0;
    }

    /* output projection: y_flat = attn_flat @ Wo + bo.
       T1.5 bias-fuse: when use_bias, pre-fill y rows with bo broadcast,
       then run gemm with skip_init=true so it accumulates A@B into the
       bias-pre-filled buffer. produces same y as gemm-then-bias-add but
       merges the two memory passes — the gemm's last-K-iter writeback
       is the only write that touches y, instead of writeback + a
       separate bias-add full-rows pass. */
    pf_t0 = prof_enabled ? PF_TICK() : 0;
    if (m->use_bias) {
        const float *bd = (const float *)m->bo->storage->data;
        float *od = (float *)y_flat_v.storage->data;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int64_t r = 0; r < rows; r++) {
            memcpy(od + r * D, bd, (size_t)D * sizeof(float));
        }
        ax_gemm_set_skip_init(true);
        ax_status_t s = ax_compute_gemm(attn_flat, m->Wo, &y_flat_v);
        ax_gemm_set_skip_init(false);
        if (s != AX_OK) return AX_ERR_BACKEND;
    } else {
        if (ax_compute_gemm(attn_flat, m->Wo, &y_flat_v) != AX_OK) return AX_ERR_BACKEND;
    }
    ax_storage_touch(y_out->storage);
    if (prof_enabled) pf_wo += PF_TICK() - pf_t0;

    /* ============================================================
       BACKWARD
       ============================================================ */

    /* dout view: caller-supplied or "loss = sum(out)" → ones */
    ax_tensor_t  dout_flat_v;
    ax_storage_t dout_flat_st;
    ax_tensor_t *dout_flat;
    ax_tensor_t *dout_owned = NULL;
    pf_t0 = prof_enabled ? PF_TICK() : 0;
    if (dout) {
        make_stack_view(&dout_flat_v, &dout_flat_st,
                         (const float *)dout->storage->data, rows, D);
        dout_flat = &dout_flat_v;
    } else {
        dout_owned = ax_tensor_arena_create(arena, flat_sh, 2, AX_FLOAT32);
        if (!dout_owned) return AX_ERR_ALLOC;
        fill_f32((float *)dout_owned->storage->data, 1.0f, rows * D);
        dout_flat = dout_owned;
    }
    if (prof_enabled) pf_dout += PF_TICK() - pf_t0;

    /* step 1 — output projection backward */

    /* dWo += attn_flat^T @ dout_flat. direct accumulate via skip_init,
       skipping the scratch alloc + add pass that mha_backward uses. */
    pf_t0 = prof_enabled ? PF_TICK() : 0;
    if (m->Wo->requires_grad) {
        float *dWo = ensure_grad_ptr(m->Wo);
        if (dWo) {
            ax_gemm_set_skip_init(true);
            ax_status_t s = ax_compute_gemm_tn(attn_flat, dout_flat, m->Wo->grad);
            ax_gemm_set_skip_init(false);
            if (s != AX_OK) return s;
            ax_storage_touch(m->Wo->grad->storage);
        }
    }

    /* dbo += col-sum(dout_flat) */
    if (m->use_bias && m->bo->requires_grad) {
        float *dbo = ensure_grad_ptr(m->bo);
        if (dbo) {
            col_sum_acc((const float *)dout_flat->storage->data, dbo, rows, D);
            ax_storage_touch(m->bo->grad->storage);
        }
    }

    if (prof_enabled) pf_wog += PF_TICK() - pf_t0;

    /* dattn + head_interleave: F.3.d fused path streams dout @ Wo^T
       directly into head-interleaved dO_head, eliminating the [rows, D]
       d_attn_flat intermediate. fall back to gemm_nt + head_interleave
       when the backend lacks the slot. */
    ax_tensor_t *dO_head = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    if (!dO_head) return AX_ERR_ALLOC;

    pf_t0 = prof_enabled ? PF_TICK() : 0;
    if (ax_compute_has_dattn_head_gemm_nt()) {
        if (ax_compute_dattn_head_gemm_nt(dout_flat, m->Wo, B, S, H, dk, dO_head) != AX_OK)
            return AX_ERR_BACKEND;
        if (prof_enabled) pf_dattn += PF_TICK() - pf_t0;
        /* head_interleave is folded into the fused gemm_nt; pf_hint stays zero */
    } else {
        ax_tensor_t *d_attn_flat = ax_tensor_arena_create(arena, flat_sh, 2, AX_FLOAT32);
        if (!d_attn_flat) return AX_ERR_ALLOC;
        if (ax_compute_gemm_nt(dout_flat, m->Wo, d_attn_flat) != AX_OK) return AX_ERR_BACKEND;
        if (prof_enabled) pf_dattn += PF_TICK() - pf_t0;

        pf_t0 = prof_enabled ? PF_TICK() : 0;
        ax_attn_head_interleave(
            (const float *)d_attn_flat->storage->data,
            (float *)dO_head->storage->data, B, S, H, dk);
        if (prof_enabled) pf_hint += PF_TICK() - pf_t0;
    }

    ax_tensor_t *dQh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    ax_tensor_t *dKh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    ax_tensor_t *dVh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    if (!dQh || !dKh || !dVh) return AX_ERR_ALLOC;

    pf_t0 = prof_enabled ? PF_TICK() : 0;
    if (Oh) {
        /* fallback path: Oh was materialised, use the standard bwd_use */
        ax_fused_attention_bwd_use(
            (const float *)Qh->storage->data,
            (const float *)Kh->storage->data,
            (const float *)Vh->storage->data,
            (const float *)Oh->storage->data,
            (const float *)dO_head->storage->data,
            (const float *)L_t->storage->data,
            P_save_ptr,
            (float *)dQh->storage->data,
            (float *)dKh->storage->data,
            (float *)dVh->storage->data,
            bh, S, dk, scale, m->causal);
    } else {
        /* F.3.e companion: Oh was never materialised, read O strided
           from attn_flat for the Di = dot(O, dO) computation. */
        ax_fused_attention_bwd_use_from_flat(
            (const float *)Qh->storage->data,
            (const float *)Kh->storage->data,
            (const float *)Vh->storage->data,
            (const float *)attn_flat->storage->data,
            (const float *)dO_head->storage->data,
            (const float *)L_t->storage->data,
            P_save_ptr,
            (float *)dQh->storage->data,
            (float *)dKh->storage->data,
            (float *)dVh->storage->data,
            B, S, H, dk, scale, m->causal);
    }
    if (prof_enabled) pf_sdpab += PF_TICK() - pf_t0;

    /* steps 3-5: weight grads dWq/dWk/dWv and bias grads dbq/dbk/dbv.
       head-major path reads dQh/dKh/dVh directly, skips [rows, 3D] merge. */
    bool any_wqkv_grad = m->Wq->requires_grad || m->Wk->requires_grad ||
                         m->Wv->requires_grad;
    bool all_wqkv_grad = m->Wq->requires_grad && m->Wk->requires_grad &&
                         m->Wv->requires_grad;
    bool need_bias_grad = m->use_bias && (m->bq->requires_grad ||
                          m->bk->requires_grad || m->bv->requires_grad);

    bool hm_wgrad_done = false;
    if (all_wqkv_grad && D >= ax_attn_tunable_f3c_d_threshold()
        && ax_compute_has_dwqkv_split_acc_hm()) {
        (void)ensure_grad_ptr(m->Wq);
        (void)ensure_grad_ptr(m->Wk);
        (void)ensure_grad_ptr(m->Wv);
        if (m->Wq->grad && m->Wk->grad && m->Wv->grad) {
            pf_t0 = prof_enabled ? PF_TICK() : 0;
            if (ax_compute_dwqkv_split_acc_hm(&x_flat_v, dQh, dKh, dVh,
                                               B, S, H, dk,
                                               m->Wq->grad, m->Wk->grad,
                                               m->Wv->grad) != AX_OK)
                return AX_ERR_BACKEND;
            if (prof_enabled) pf_dwqkv += PF_TICK() - pf_t0;
            ax_storage_touch(m->Wq->grad->storage);
            ax_storage_touch(m->Wk->grad->storage);
            ax_storage_touch(m->Wv->grad->storage);
            hm_wgrad_done = true;
        }
    }

    bool hm_bgrad_done = false;
    if (hm_wgrad_done && need_bias_grad) {
        #define ACC_BIAS_HM(src_tensor, W) do {                               \
            if ((W)->requires_grad) {                                         \
                float *dp = ensure_grad_ptr(W);                               \
                if (dp) {                                                     \
                    col_sum_hm_one((const float *)(src_tensor)->storage->data, \
                                   dp, B, S, H, dk);                          \
                    ax_storage_touch((W)->grad->storage);                     \
                }                                                             \
            }                                                                 \
        } while (0)
        ACC_BIAS_HM(dQh, m->bq);
        ACC_BIAS_HM(dKh, m->bk);
        ACC_BIAS_HM(dVh, m->bv);
        #undef ACC_BIAS_HM
        hm_bgrad_done = true;
    }

    if (!(hm_wgrad_done && (!need_bias_grad || hm_bgrad_done))) {
        /* flat path: merge dQh/dKh/dVh → dQKV [rows, 3D] */
        ax_tensor_t *dQKV = ax_tensor_arena_create(arena, qkv_sh, 2, AX_FLOAT32);
        if (!dQKV) return AX_ERR_ALLOC;
        pf_t0 = prof_enabled ? PF_TICK() : 0;
        ax_attn_head_deinterleave_qkv_merge(
            (const float *)dQh->storage->data,
            (const float *)dKh->storage->data,
            (const float *)dVh->storage->data,
            (float *)dQKV->storage->data,
            B, S, H, dk, D);
        if (prof_enabled) pf_dqkv_merge += PF_TICK() - pf_t0;

        if (any_wqkv_grad && !hm_wgrad_done) {
            if (all_wqkv_grad && D >= ax_attn_tunable_f3c_d_threshold()
                && ax_compute_has_dwqkv_split_acc()) {
                (void)ensure_grad_ptr(m->Wq);
                (void)ensure_grad_ptr(m->Wk);
                (void)ensure_grad_ptr(m->Wv);
                if (m->Wq->grad && m->Wk->grad && m->Wv->grad) {
                    pf_t0 = prof_enabled ? PF_TICK() : 0;
                    if (ax_compute_dwqkv_split_acc(&x_flat_v, dQKV,
                                                    m->Wq->grad, m->Wk->grad,
                                                    m->Wv->grad) != AX_OK)
                        return AX_ERR_BACKEND;
                    if (prof_enabled) pf_dwqkv += PF_TICK() - pf_t0;
                    ax_storage_touch(m->Wq->grad->storage);
                    ax_storage_touch(m->Wk->grad->storage);
                    ax_storage_touch(m->Wv->grad->storage);
                    goto dwqkv_done_ts;
                }
            }

            int64_t dW_sh[] = {D, 3 * D};
            ax_tensor_t *dWqkv = ax_tensor_arena_create(arena, dW_sh, 2, AX_FLOAT32);
            if (!dWqkv) return AX_ERR_ALLOC;
            pf_t0 = prof_enabled ? PF_TICK() : 0;
            if (ax_compute_gemm_tn(&x_flat_v, dQKV, dWqkv) != AX_OK) return AX_ERR_BACKEND;
            if (prof_enabled) pf_dwqkv += PF_TICK() - pf_t0;

            const float *dw = (const float *)dWqkv->storage->data;
            #define ACC_PARAM(W, col_off) do {                                    \
                if ((W)->requires_grad) {                                         \
                    float *dp = ensure_grad_ptr(W);                               \
                    if (dp) {                                                     \
                        _Pragma("omp parallel for schedule(static)")              \
                        for (int64_t i = 0; i < D; i++)                           \
                            acc_f32(dw + i * 3 * D + (col_off), dp + i * D, D);   \
                        ax_storage_touch((W)->grad->storage);                     \
                    }                                                             \
                }                                                                 \
            } while (0)
            ACC_PARAM(m->Wq, 0);
            ACC_PARAM(m->Wk, D);
            ACC_PARAM(m->Wv, 2 * D);
            #undef ACC_PARAM
            dwqkv_done_ts: ;
        }

        if (need_bias_grad && !hm_bgrad_done) {
            int64_t db_sh[] = {3 * D};
            ax_tensor_t *dbqkv = ax_tensor_arena_zeros(arena, db_sh, 1, AX_FLOAT32);
            if (!dbqkv) return AX_ERR_ALLOC;
            col_sum_acc((const float *)dQKV->storage->data,
                         (float *)dbqkv->storage->data, rows, 3 * D);
            const float *db = (const float *)dbqkv->storage->data;
            #define ACC_BIAS(W, col_off) do {                                     \
                if ((W)->requires_grad) {                                         \
                    float *dp = ensure_grad_ptr(W);                               \
                    if (dp) {                                                     \
                        acc_f32(db + (col_off), dp, D);                           \
                        ax_storage_touch((W)->grad->storage);                     \
                    }                                                             \
                }                                                                 \
            } while (0)
            ACC_BIAS(m->bq, 0);
            ACC_BIAS(m->bk, D);
            ACC_BIAS(m->bv, 2 * D);
            #undef ACC_BIAS
        }
    }

    /* dx through QKV is intentionally not computed — train_step is for
       the case where x is treated as a constant (matches TF's
       @tf.function bench: tape.gradient(loss, mha.trainable_variables)
       prunes dX at compile time). callers needing dx should use the
       autograd path. */

    if (prof_enabled) {
        pf_calls++;
        if ((pf_calls % 5) == 0) {
            uint64_t total = pf_qkv + pf_split + pf_sdpa + pf_deint + pf_wo
                           + pf_dout + pf_wog + pf_dattn + pf_hint + pf_sdpab
                           + pf_dqkv_merge + pf_dwqkv + pf_dbqkv;
            if (total > 0) {
                fprintf(stderr,
                    "axiom: mha_train_step [B=%ld S=%ld D=%ld H=%ld, %d calls] "
                    "FWD: qkv=%.0f%% split=%.0f%% sdpa=%.0f%% deint=%.0f%% wo=%.0f%% "
                    "BWD: dout=%.0f%% wog=%.0f%% dattn=%.0f%% hint=%.0f%% sdpab=%.0f%% "
                    "merge=%.0f%% dwqkv=%.0f%% dbqkv=%.0f%%\n",
                    (long)B, (long)S, (long)D, (long)H, pf_calls,
                    100.0 * (double)pf_qkv         / (double)total,
                    100.0 * (double)pf_split       / (double)total,
                    100.0 * (double)pf_sdpa        / (double)total,
                    100.0 * (double)pf_deint       / (double)total,
                    100.0 * (double)pf_wo          / (double)total,
                    100.0 * (double)pf_dout        / (double)total,
                    100.0 * (double)pf_wog         / (double)total,
                    100.0 * (double)pf_dattn       / (double)total,
                    100.0 * (double)pf_hint        / (double)total,
                    100.0 * (double)pf_sdpab       / (double)total,
                    100.0 * (double)pf_dqkv_merge  / (double)total,
                    100.0 * (double)pf_dwqkv       / (double)total,
                    100.0 * (double)pf_dbqkv       / (double)total);
            }
        }
    }
    #undef PF_TICK

    return AX_OK;
}
