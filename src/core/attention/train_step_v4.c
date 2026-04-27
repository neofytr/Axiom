/* attention/train_step_v4.c — F.4.4 monolithic train kernel.

   Phase B: per-qi-block driver. same external contract as
   ax_mha_train_step_fused, but the SDPA fwd, d_attn gemm_nt, head
   interleave, and SDPA bwd are sequenced inside a single qi-block
   loop so attn_flat[qi-block] / dO_head[qi-block] stay cache-hot
   across the fwd → interleave → bwd handoff (the big win vs
   train_step_fused, where attn_flat is a full [rows, D] intermediate
   that spills to L3 between stages).

   stays outside the loop (done once over full-rows):
     - input-proj qkv + head_split (Phase C eliminates)
     - output-proj fwd y = attn_flat @ Wo + bo (Phase D fuses in-loop)
     - dWo / dbo accumulation (Phase D fuses in-loop)
     - dQKV merge + dWqkv + dbqkv (Phase C eliminates dQKV)

   parity with ax_mha_train_step / ax_mha_train_step_fused is maintained
   via test_mha_train_step_v4_parity (tests/test_attention.c).

   later phases: C — per-tile dWqkv; D — per-tile output proj fusion;
   E — per-(qi, kj) tile-level combined fwd+bwd kernel. */

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

/* same contract as train_step.c / train_step_fused.c — tells the gemm
   backends to accumulate into dst rather than overwrite. */
extern void ax_gemm_set_skip_init(bool v);
extern void ax_gemm_set_inner_team(bool v);

static ax_arena_t *get_scratch_arena(void)
{
    return ax_train_get_scratch_arena();
}

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

static float *ensure_grad_ptr(ax_tensor_t *p)
{
    if (!p || !p->requires_grad) return NULL;
    if (!p->grad) {
        p->grad = ax_tensor_zeros(p->shape, p->ndim, p->dtype);
        if (!p->grad) return NULL;
    }
    return (float *)p->grad->storage->data;
}

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

/* col_sum directly from head-major [B*H, S, dk] layout into [D] bias grad.
   avoids materializing the [rows, D] flat merge just for bias accumulation.
   each bh-row contributes to out[h*dk .. h*dk+dk-1] where h = bh % H. */
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

/* qi-range variant of ax_attn_head_interleave. d_attn is [B, S, D]
   row-major; dO_head is [B, H, S, dk] (= [BH, S, dk]). copies only
   rows [qi_start, qi_end) — rest of dO_head is untouched, assumed
   already populated by prior qi-block iterations. a qi-range-aware
   fused gemm_nt + interleave (Phase D) would skip this copy entirely. */
static void head_interleave_qi_range(const float *d_attn, float *dO_head,
                                      int64_t B, int64_t S, int64_t H, int64_t dk,
                                      int64_t qi_start, int64_t qi_end)
{
    int64_t D = H * dk;
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t s = qi_start; s < qi_end; s++) {
                const float *src = d_attn + b * S * D + s * D + h * dk;
                float *dst = dO_head + (b * H + h) * S * dk + s * dk;
                memcpy(dst, src, (size_t)dk * sizeof(float));
            }
        }
    }
}

/* qi_block size selection.

   default: qi_block = S (single-iter — the qi-block primitive devolves to
   a full-S call, byte-identical to ax_fused_attention_fwd_save_to_flat /
   ax_fused_attention_bwd_use_from_flat). no multi-block overhead.

   why single-iter default: multi-block splits pay a non-trivial fixed
   cost inside the SDPA primitives — attn_fwd_head / attn_bwd_head re-pack
   Q, dO full-S into their TLS Q_pa / Q_pb / dO_pa / dO_pb buffers on EACH
   call (packing is per-head-invocation, not cached across calls). with
   qi_block=126 on S=2048 this is 17 full-S repacks per bwd (~1.7 ms of
   wasted bandwidth) — wipes out the cache gain from attn_flat[qi-block]
   L2-reuse. multi-block becomes a win once the primitive learns to
   hoist its packs across qi-block calls (future work, probably Phase E).

   AX_V4_QI_BLOCK overrides — set <= S to force multi-block (benchmarking
   / parity testing). read per-call so tests can flip it without restart. */
static inline int64_t resolve_qi_block(int64_t S)
{
    const char *e = getenv("AX_V4_QI_BLOCK");
    if (!e || !e[0]) return S;
    int64_t block = (int64_t)atoll(e);
    if (block < 1) block = 1;
    return block > S ? S : block;
}

/* AX_V4_HOIST=1 — Phase E driver. wrap the qi-block loop in ONE
   #pragma omp parallel region, call _inner fwd/bwd variants inside.
   removes 2*N team spawn/join per train step (N = number of qi-blocks)
   and preserves thread-locality across fwd→bwd handoff inside each
   qi-block iteration. default off pending bench validation. read per
   call so parity tests can flip the flag without process restart;
   getenv is cheap (pointer comparison after the first lookup). */
static inline bool resolve_hoist(void)
{
    const char *e = getenv("AX_V4_HOIST");
    return (e && e[0] == '1');
}

/* AX_V4_FUSED_BH=1 — Phase E combined kernel. implies AX_V4_HOIST.
   inside the single parallel region, each qi-block runs ONE
   #pragma omp for that does fwd→bwd back-to-back per (b, h) on the
   same thread, halving per-qi-block barriers vs the separate
   fwd_inner + bwd_inner pair. TLS scratch + attn_flat[qi-block] rows
   stay L1-hot across the handoff. off by default pending bench. */
static inline bool resolve_fused_bh(void)
{
    const char *e = getenv("AX_V4_FUSED_BH");
    return (e && e[0] == '1');
}

/* shape-dispatched auto-enable of hoist+fused_bh.
   empirically measured (4-run medians, bg-noise):
     B8_S128_D512_H8   (BH=64, S=128)  : -16 % vs default — BIG WIN
     B4_S512_D768_H12  (BH=48, S=512)  : neutral
     B2_S1024_D768_H12 (BH=24, S=1024) : +2 % regression
     B1_S512_D1024_H16 (BH=16, S=512)  : +3 % regression
     B1_S2048_D768_H12 (BH=12, S=2048) : neutral
   pattern: fused_bh wins on high-BH + small-S. WHY: each thread reuses
   attn_flat[bh_slice] = S*dk floats IN-CACHE across the fwd→bwd handoff.
   when S*dk*4 fits L1 comfortably (≤ 32 KB typical L1d), the whole
   per-head working set lives in L1 throughout and the fwd-writes-bwd-
   reads memcpy via cache is free. when it doesn't fit (S=512 with dk=64
   = 128 KB per head), attn_flat[bh_slice] spills to L2/L3 and the cache-
   reuse benefit disappears — at that point the combined kernel's
   reduced scheduling flexibility (each thread must do fwd+bwd for same
   bh together) becomes a net loss.
   threshold — S * dk * sizeof(float) ≤ 32 KB corresponds to the
   per-head attn_flat slice fitting within ONE L1d line on the vast
   majority of modern CPUs (Intel / AMD x86-64 L1d sizes are 32-48 KB;
   Apple M-series L1d are 128-192 KB so the threshold is conservative).
   when the slice doesn't fit L1, the fwd→bwd cache-reuse benefit
   evaporates and the combined kernel's reduced scheduling flexibility
   becomes a net loss (measured +4% regression on B1_S512 which has
   per_head_bytes = 128 KB).
   BH ≥ max_threads ensures the combined kernel has enough heads to
   saturate the team; if BH < NT the spare-thread n_inner path below
   is preferable for parallelism.
   env-override AX_V4_AUTO=0 disables this heuristic (keeps default path);
   AX_V4_AUTO=1 explicit opt-in (same as default). */
static inline bool auto_enable_fused_bh(int64_t S, int64_t dk, int64_t BH)
{
    const char *e = getenv("AX_V4_AUTO");
    if (e && e[0] == '0') return false;
#ifdef _OPENMP
    int nt = omp_get_max_threads();
#else
    int nt = 1;
#endif
    int64_t per_head_bytes = S * dk * (int64_t)sizeof(float);
    bool fits_l1 = per_head_bytes <= ax_attn_tunable_fused_bh_per_head_bytes();
    bool enough_heads = BH >= (int64_t)nt;
    return fits_l1 && enough_heads;
}

ax_status_t ax_mha_train_step_v4(ax_layer_t *layer,
                                  const ax_tensor_t *x,
                                  const ax_tensor_t *dout,
                                  ax_tensor_t *y_out)
{
    if (!layer || !x || !y_out) {
        ax_err_set(AX_ERR_NULL_ARG, "ax_mha_train_step_v4: NULL arg");
        return AX_ERR_NULL_ARG;
    }
    ax_mha_t *m = (ax_mha_t *)layer;
    if (x->ndim != 3 || y_out->ndim != 3) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "train_step_v4: x/y must be 3-D");
        return AX_ERR_SHAPE_MISMATCH;
    }
    int64_t B  = x->shape[0];
    int64_t S  = x->shape[1];
    int64_t D  = x->shape[2];
    if (D != m->d_model) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "train_step_v4: x last dim != d_model");
        return AX_ERR_SHAPE_MISMATCH;
    }
    if (y_out->shape[0] != B || y_out->shape[1] != S || y_out->shape[2] != D) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "train_step_v4: y_out shape mismatch");
        return AX_ERR_SHAPE_MISMATCH;
    }
    if (x->dtype != AX_FLOAT32 || y_out->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "train_step_v4: must be fp32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    if (dout) {
        if (dout->ndim != 3 ||
            dout->shape[0] != B || dout->shape[1] != S || dout->shape[2] != D ||
            dout->dtype != AX_FLOAT32) {
            ax_err_set(AX_ERR_SHAPE_MISMATCH, "train_step_v4: dout shape mismatch");
            return AX_ERR_SHAPE_MISMATCH;
        }
    }

    int64_t H    = m->n_heads;
    int64_t dk   = m->d_k;
    int64_t rows = B * S;
    int64_t bh   = B * H;

    ax_arena_t *arena = get_scratch_arena();
    if (!arena) {
        ax_err_set(AX_ERR_ALLOC, "train_step_v4: scratch arena alloc failed");
        return AX_ERR_ALLOC;
    }

    ax_attn_refresh_fused_qkv(m);

    ax_tensor_t  x_flat_v;
    ax_storage_t x_flat_st;
    make_stack_view(&x_flat_v, &x_flat_st,
                     (const float *)x->storage->data, rows, D);

    ax_tensor_t  y_flat_v;
    ax_storage_t y_flat_st;
    make_stack_view(&y_flat_v, &y_flat_st,
                     (const float *)y_out->storage->data, rows, D);

    int64_t qkv_sh[]  = {rows, 3 * D};
    int64_t head_sh[] = {bh, S, dk};
    int64_t L_sh[]    = {bh, S};
    int64_t flat_sh[] = {rows, D};

    /* input projection + head split (full-rows). F.3.a fused path writes
       directly to Qh/Kh/Vh; fallback is unfused gemm + split. Phase C
       will per-tile-recompute X @ Wqkv inside the qi-block loop to
       eliminate Qh/Kh/Vh. */
    ax_tensor_t *Qh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    ax_tensor_t *Kh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    ax_tensor_t *Vh = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    if (!Qh || !Kh || !Vh) return AX_ERR_ALLOC;

    if (ax_compute_has_qkv_head_gemm()) {
        const ax_tensor_t *bqkv_arg = m->use_bias ? m->bqkv_cache : NULL;
        if (ax_compute_qkv_head_gemm(&x_flat_v, m->Wqkv_cache, bqkv_arg,
                                       B, S, H, dk, Qh, Kh, Vh) != AX_OK)
            return AX_ERR_BACKEND;
    } else {
        ax_tensor_t *qkv = ax_tensor_arena_create(arena, qkv_sh, 2, AX_FLOAT32);
        if (!qkv) return AX_ERR_ALLOC;
        if (ax_compute_gemm(&x_flat_v, m->Wqkv_cache, qkv) != AX_OK) return AX_ERR_BACKEND;
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
    }

    /* L (log-sum-exp per qi row, needed by SDPA bwd) + optional P_save
       (skip softmax recompute on bwd). same gating as train_step_fused. */
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

    /* dout view (defaults to ones when NULL — parity with train_step.c
       scalar-loss convention used by the graph test harness). */
    ax_tensor_t  dout_flat_v;
    ax_storage_t dout_flat_st;
    ax_tensor_t *dout_flat;
    ax_tensor_t *dout_owned = NULL;
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

    /* working tensors that span all of S — the qi-block loop writes
       into disjoint [qi_start, qi_end) row-ranges of each. */
    ax_tensor_t *attn_flat = ax_tensor_arena_create(arena, flat_sh, 2, AX_FLOAT32);
    ax_tensor_t *d_attn    = ax_tensor_arena_create(arena, flat_sh, 2, AX_FLOAT32);
    ax_tensor_t *dO_head   = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    ax_tensor_t *dQh       = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    ax_tensor_t *dKh       = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    ax_tensor_t *dVh       = ax_tensor_arena_create(arena, head_sh, 3, AX_FLOAT32);
    if (!attn_flat || !d_attn || !dO_head || !dQh || !dKh || !dVh) return AX_ERR_ALLOC;

    /* the qi-block bwd primitive overwrites dQh[qi_start..qi_end] per call
       but ACCUMULATES into dK/dV — caller must pre-zero dKh/dVh once
       before the first block call. dQh needs no pre-zero since each
       block owns its row range. */
    memset((float *)dKh->storage->data, 0, (size_t)bh * S * dk * sizeof(float));
    memset((float *)dVh->storage->data, 0, (size_t)bh * S * dk * sizeof(float));

    int64_t qi_block = resolve_qi_block(S);

    /* d_attn + head_interleave kept OUTSIDE the qi-block loop to preserve
       the F.3.d fused (dout @ Wo^T + interleave) full-rows call. splitting
       this into per-qi-block per-batch gemm_nt + qi-range interleave
       regressed every bench shape (+9-31%) when measured with the naive
       split — too much OMP spawn overhead per small gemm_nt. the only
       thing still per-qi-block is the SDPA fwd→bwd handoff, which is where
       attn_flat[qi..qi_end] gets the cache-hot reuse. future work: a
       qi-range F.3.d variant so we can fuse d_attn into the loop without
       losing the gemm_nt+interleave fusion. */
    if (ax_compute_has_dattn_head_gemm_nt()) {
        if (ax_compute_dattn_head_gemm_nt(dout_flat, m->Wo, B, S, H, dk, dO_head) != AX_OK)
            return AX_ERR_BACKEND;
    } else {
        if (ax_compute_gemm_nt(dout_flat, m->Wo, d_attn) != AX_OK) return AX_ERR_BACKEND;
        ax_attn_head_interleave(
            (const float *)d_attn->storage->data,
            (float *)dO_head->storage->data, B, S, H, dk);
    }

    bool hoist = resolve_hoist();
    bool fused_bh = resolve_fused_bh();
    /* shape-dispatched auto-enable: when the heuristic indicates the
       combined kernel will win, flip both flags. explicit env overrides
       (AX_V4_HOIST / AX_V4_FUSED_BH set) are NEVER downgraded — if the
       user asks for hoist=1 we honour it regardless of shape. */
    if (!hoist && !fused_bh && auto_enable_fused_bh(S, dk, B * H)) {
        hoist = true;
        fused_bh = true;
    }
    if (fused_bh) hoist = true;  /* fused_bh implies hoist */

    if (hoist) {
        /* Phase E: single parallel region wraps the whole qi-block loop.
           inside each iteration, _inner fwd/bwd use the enclosing team's
           #pragma omp for worksharing. the implicit barrier at the end
           of each inner omp for synchronises threads between fwd and
           bwd (bwd reads the L / attn_flat that fwd just wrote). between
           iterations the same barrier ensures iter k+1's fwd doesn't
           start until iter k's bwd is done — which matters because bwd
           ACCUMULATES into dKh/dVh across iters.

           when fused_bh is set, ONE omp-for per iteration does fwd→bwd
           back-to-back per (b, h) on the same thread (halves barriers). */
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            for (int64_t qi = 0; qi < S; qi += qi_block) {
                int64_t qi_end = (qi + qi_block <= S) ? (qi + qi_block) : S;

                if (fused_bh) {
                    ax_fused_attention_fwd_bwd_qi_block_inner(
                        (const float *)Qh->storage->data,
                        (const float *)Kh->storage->data,
                        (const float *)Vh->storage->data,
                        (float *)attn_flat->storage->data,
                        (float *)L_t->storage->data,
                        P_save_ptr,
                        (const float *)dO_head->storage->data,
                        (float *)dQh->storage->data,
                        (float *)dKh->storage->data,
                        (float *)dVh->storage->data,
                        B, S, H, dk, qi, qi_end, scale, m->causal);
                } else {
                    ax_fused_attention_fwd_save_to_flat_qi_block_inner(
                        (const float *)Qh->storage->data,
                        (const float *)Kh->storage->data,
                        (const float *)Vh->storage->data,
                        (float *)attn_flat->storage->data,
                        (float *)L_t->storage->data,
                        P_save_ptr,
                        B, S, H, dk, qi, qi_end, scale, m->causal);

                    ax_fused_attention_bwd_use_from_flat_qi_block_inner(
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
                        B, S, H, dk, qi, qi_end, scale, m->causal);
                }
            }
        }
    } else {
        for (int64_t qi = 0; qi < S; qi += qi_block) {
            int64_t qi_end = (qi + qi_block <= S) ? (qi + qi_block) : S;

            /* (1) SDPA fwd for [qi, qi_end) → attn_flat[qi..qi_end] + L / P_save */
            ax_fused_attention_fwd_save_to_flat_qi_block(
                (const float *)Qh->storage->data,
                (const float *)Kh->storage->data,
                (const float *)Vh->storage->data,
                (float *)attn_flat->storage->data,
                (float *)L_t->storage->data,
                P_save_ptr,
                B, S, H, dk, qi, qi_end, scale, m->causal);

            /* (2) SDPA bwd for [qi, qi_end). overwrites dQh rows, accumulates
                  dKh/dVh. reads attn_flat[qi..qi_end] / L / P_save / dO_head —
                  attn_flat[qi..qi_end] (just written above) is L2-hot for this
                  call even when the full attn_flat exceeds L2. dO_head is the
                  full pre-computed tensor (stream-through read). */
            ax_fused_attention_bwd_use_from_flat_qi_block(
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
                B, S, H, dk, qi, qi_end, scale, m->causal);
        }
    }
    (void)d_attn;  /* only used in the non-F.3.d fallback above */

    /* (5)-(7) output-projection fwd + bwd wrapped in a single parallel
       region. eliminates two fork-joins (Wo gemm, dWo gemm_tn) and keeps
       per-thread pack buffers warm across both gemms. the gemm backends
       detect inner_mode and workshare via omp-for instead of spawning. */
    {
        const bool do_dWo = m->Wo->requires_grad;
        float *dWo_ptr = do_dWo ? ensure_grad_ptr(m->Wo) : NULL;
        const bool do_dbo = m->use_bias && m->bo->requires_grad;
        float *dbo_ptr = do_dbo ? ensure_grad_ptr(m->bo) : NULL;
        ax_status_t wrap_st = AX_OK;

#ifdef _OPENMP
        #pragma omp parallel
        {
            ax_gemm_set_inner_team(true);

            /* (5) fwd y = attn_flat @ Wo + bo */
            if (m->use_bias) {
                const float *bd = (const float *)m->bo->storage->data;
                float *od = (float *)y_flat_v.storage->data;
                #pragma omp for schedule(static)
                for (int64_t r = 0; r < rows; r++)
                    memcpy(od + r * D, bd, (size_t)D * sizeof(float));
                ax_gemm_set_skip_init(true);
            }
            {
                ax_status_t s = ax_compute_gemm(attn_flat, m->Wo, &y_flat_v);
                if (m->use_bias) ax_gemm_set_skip_init(false);
                if (s != AX_OK) wrap_st = s;
            }

            /* (6) dWo += attn_flat^T @ dout */
            if (do_dWo && dWo_ptr) {
                ax_gemm_set_skip_init(true);
                ax_status_t s = ax_compute_gemm_tn(attn_flat, dout_flat, m->Wo->grad);
                ax_gemm_set_skip_init(false);
                if (s != AX_OK) wrap_st = s;
            }

            /* (7) dbo += colsum(dout) */
            if (do_dbo && dbo_ptr) {
                col_sum_acc((const float *)dout_flat->storage->data, dbo_ptr, rows, D);
            }

            ax_gemm_set_inner_team(false);
        }
#else
        /* single-threaded: same logic, no omp. */
        if (m->use_bias) {
            const float *bd = (const float *)m->bo->storage->data;
            float *od = (float *)y_flat_v.storage->data;
            for (int64_t r = 0; r < rows; r++)
                memcpy(od + r * D, bd, (size_t)D * sizeof(float));
            ax_gemm_set_skip_init(true);
            wrap_st = ax_compute_gemm(attn_flat, m->Wo, &y_flat_v);
            ax_gemm_set_skip_init(false);
        } else {
            wrap_st = ax_compute_gemm(attn_flat, m->Wo, &y_flat_v);
        }
        if (wrap_st == AX_OK && do_dWo && dWo_ptr) {
            ax_gemm_set_skip_init(true);
            wrap_st = ax_compute_gemm_tn(attn_flat, dout_flat, m->Wo->grad);
            ax_gemm_set_skip_init(false);
        }
        if (wrap_st == AX_OK && do_dbo && dbo_ptr)
            col_sum_acc((const float *)dout_flat->storage->data, dbo_ptr, rows, D);
#endif
        if (wrap_st != AX_OK) return wrap_st;
        ax_storage_touch(y_out->storage);
        if (do_dWo && m->Wo->grad) ax_storage_touch(m->Wo->grad->storage);
        if (do_dbo && m->bo->grad) ax_storage_touch(m->bo->grad->storage);
    }

    /* (8-10) weight grads dWq/dWk/dWv and bias grads dbq/dbk/dbv.
       head-major path: reads dQh/dKh/dVh directly, skips the [rows, 3D]
       dQKV merge + its allocation (~18 MB at B1_S2048_D768).
       flat fallback: merges dQKV then uses the existing split-acc or
       gemm_tn paths. */
    bool any_wqkv_grad = m->Wq->requires_grad || m->Wk->requires_grad ||
                         m->Wv->requires_grad;
    bool all_wqkv_grad = m->Wq->requires_grad && m->Wk->requires_grad &&
                         m->Wv->requires_grad;
    bool need_bias_grad = m->use_bias && (m->bq->requires_grad ||
                          m->bk->requires_grad || m->bv->requires_grad);

    /* try head-major weight grad path — eliminates dQKV merge entirely */
    bool hm_wgrad_done = false;
    if (all_wqkv_grad && D >= ax_attn_tunable_f3c_d_threshold()
        && ax_compute_has_dwqkv_split_acc_hm()) {
        float *dq_init = ensure_grad_ptr(m->Wq);
        float *dk_init = ensure_grad_ptr(m->Wk);
        float *dv_init = ensure_grad_ptr(m->Wv);
        (void)dq_init; (void)dk_init; (void)dv_init;
        if (m->Wq->grad && m->Wk->grad && m->Wv->grad) {
            if (ax_compute_dwqkv_split_acc_hm(&x_flat_v, dQh, dKh, dVh,
                                               B, S, H, dk,
                                               m->Wq->grad, m->Wk->grad,
                                               m->Wv->grad) != AX_OK)
                return AX_ERR_BACKEND;
            ax_storage_touch(m->Wq->grad->storage);
            ax_storage_touch(m->Wk->grad->storage);
            ax_storage_touch(m->Wv->grad->storage);
            hm_wgrad_done = true;
        }
    }

    /* head-major bias grad path — colsum from dQh/dKh/dVh directly */
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

    if (hm_wgrad_done && (!need_bias_grad || hm_bgrad_done))
        return AX_OK;

    /* flat path: merge dQh/dKh/dVh → dQKV [rows, 3D] */
    ax_tensor_t *dQKV = ax_tensor_arena_create(arena, qkv_sh, 2, AX_FLOAT32);
    if (!dQKV) return AX_ERR_ALLOC;
    ax_attn_head_deinterleave_qkv_merge(
        (const float *)dQh->storage->data,
        (const float *)dKh->storage->data,
        (const float *)dVh->storage->data,
        (float *)dQKV->storage->data,
        B, S, H, dk, D);

    if (any_wqkv_grad && !hm_wgrad_done) {
        if (all_wqkv_grad && D >= ax_attn_tunable_f3c_d_threshold()
            && ax_compute_has_dwqkv_split_acc()) {
            float *dq_init = ensure_grad_ptr(m->Wq);
            float *dk_init = ensure_grad_ptr(m->Wk);
            float *dv_init = ensure_grad_ptr(m->Wv);
            (void)dq_init; (void)dk_init; (void)dv_init;
            if (m->Wq->grad && m->Wk->grad && m->Wv->grad) {
                if (ax_compute_dwqkv_split_acc(&x_flat_v, dQKV,
                                                m->Wq->grad, m->Wk->grad,
                                                m->Wv->grad) != AX_OK)
                    return AX_ERR_BACKEND;
                ax_storage_touch(m->Wq->grad->storage);
                ax_storage_touch(m->Wk->grad->storage);
                ax_storage_touch(m->Wv->grad->storage);
                goto dwqkv_done_v4;
            }
        }

        int64_t dW_sh[] = {D, 3 * D};
        ax_tensor_t *dWqkv = ax_tensor_arena_create(arena, dW_sh, 2, AX_FLOAT32);
        if (!dWqkv) return AX_ERR_ALLOC;
        if (ax_compute_gemm_tn(&x_flat_v, dQKV, dWqkv) != AX_OK) return AX_ERR_BACKEND;

        const float *dw = (const float *)dWqkv->storage->data;
        float *dp_q = m->Wq->requires_grad ? ensure_grad_ptr(m->Wq) : NULL;
        float *dp_k = m->Wk->requires_grad ? ensure_grad_ptr(m->Wk) : NULL;
        float *dp_v = m->Wv->requires_grad ? ensure_grad_ptr(m->Wv) : NULL;

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int64_t i = 0; i < D; i++) {
            if (dp_q) acc_f32(dw + i * 3 * D,         dp_q + i * D, D);
            if (dp_k) acc_f32(dw + i * 3 * D + D,     dp_k + i * D, D);
            if (dp_v) acc_f32(dw + i * 3 * D + 2 * D, dp_v + i * D, D);
        }

        if (dp_q) ax_storage_touch(m->Wq->grad->storage);
        if (dp_k) ax_storage_touch(m->Wk->grad->storage);
        if (dp_v) ax_storage_touch(m->Wv->grad->storage);
        dwqkv_done_v4: ;
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

    return AX_OK;
}
