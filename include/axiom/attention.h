/* axiom/attention.h — multi-head attention and scaled dot-product attention.

   three public APIs, three levels of abstraction:

   1. ax_mha_t layer — trainable MHA for use in ax_sequential / transformers.
      owns Wq/Wk/Wv/Wo weights, integrated with autograd.

   2. ax_fused_attention_fwd / _bwd — the SDPA compute primitive.
      takes raw Q/K/V buffers, computes attention forward or backward.
      use this when you have your own projection logic or for inference
      pipelines that manage Q/K/V directly.

   3. ax_kv_cache_t — preallocated K/V buffer for autoregressive inference.
      new tokens append to the cache; attention is rank-1 (GEMV) against
      all cached positions. orders-of-magnitude faster than recomputing
      attention from scratch per token.

   performance notes (CPU path):
   - forward uses FlashAttention-style MR-strip tiling. score tile
     (MR × BK) stays L1-resident through GEMM → softmax → V-multiply,
     eliminating 4 L2 round-trips per tile vs the naive 3-pass approach.
   - scale factor is fused into packed K^T once per kj block rather than
     applied as a separate pass over the full score matrix.
   - QKV projection is fused: single matmul x @ [D, 3D] instead of
     three x @ [D, D]. saves 2/3 of the weight-matrix traffic.
   - backward uses recompute (not materialized P) — on cache-limited
     cpus, materializing per-head S*S scores thrashes L3. recompute
     is cheaper when the score matrix exceeds ~half of L3.

   ownership: ax_mha_create returns a heap layer; release with
   ax_layer_destroy. ax_kv_cache_create returns a heap cache; release
   with ax_kv_cache_destroy. the SDPA primitive functions
   (ax_fused_attention_*) take all buffers as caller-owned pointers —
   no allocation, no destroy.

   error handling: ax_mha_create / ax_kv_cache_create return NULL on
   alloc failure or invalid args. ax_kv_cache_append returns false when
   the cache is full. SDPA primitives are void: any internal failure is
   logged via ax_err_set; callers should check ax_err_last_status only
   if they suspect a problem.

   thread-safety: per layer.h contract for ax_mha_t (single-thread per
   layer instance). SDPA primitives are reentrant and use openmp
   internally — DO NOT call them from inside another openmp parallel
   region (omp will not nest by default). ax_kv_cache_t mutators
   (append, reset) are not concurrent-safe; ax_kv_cache_attend is
   read-only and safe to call concurrently with disjoint Q_new buffers
   on the same cache. */

#ifndef AX_ATTENTION_H
#define AX_ATTENTION_H

#include "layer.h"

/* ================================================================
   LAYER API — trainable multi-head attention
   ================================================================ */

typedef struct {
    ax_layer_t base;
    int64_t d_model;   /* input/output dimension */
    int n_heads;       /* number of attention heads. d_model must be divisible by n_heads. */
    int64_t d_k;       /* per-head dimension = d_model / n_heads */
    bool use_bias;     /* add learnable bias to each projection */
    bool causal;       /* causal mask: position i can only attend to positions ≤ i */

    /* projection weights: each [d_model, d_model].
       stored individually as parameter tensors so autograd can
       track them. the fused [d_model, 3*d_model] Wqkv is rebuilt
       before each forward from these (cheap, ~3% of forward time). */
    ax_tensor_t *Wq;
    ax_tensor_t *Wk;
    ax_tensor_t *Wv;
    ax_tensor_t *Wo;

    /* optional biases: each [d_model] */
    ax_tensor_t *bq;
    ax_tensor_t *bk;
    ax_tensor_t *bv;
    ax_tensor_t *bo;

    /* fused [d_model, 3*d_model] weight panel, refreshed from
       Wq/Wk/Wv before each forward. cached so the concat only
       happens when at least one of the three has been updated
       (checked via ax_storage generation counter). */
    ax_tensor_t *Wqkv_cache;
    ax_tensor_t *bqkv_cache;
    uint64_t Wq_gen, Wk_gen, Wv_gen; /* last-seen generations */
} ax_mha_t;

/* create a multi-head attention layer.
     d_model:  model dimension (input and output size)
     n_heads:  number of attention heads. must divide d_model evenly.
     use_bias: if true, each of the 4 projections has a learnable bias.
     causal:   if true, apply causal mask (for decoder self-attention).
   returns NULL on allocation failure or invalid args. */
ax_layer_t *ax_mha_create(int64_t d_model, int n_heads,
                           bool use_bias, bool causal);

/* ================================================================
   FUSED TRAINING STEP — XLA-style single-call forward+backward
   ================================================================

   the standard `out = ax_layer_forward(mha, x); ax_backward(loss);`
   path goes through the autograd tape: each gemm/softmax/etc. records
   a grad_fn, allocates an arena tensor for its intermediate, and
   the backward pass walks the tape calling the per-op gradient
   functions one at a time. that flexibility costs ~10-20% on dense
   training steps because:
     - each stage's intermediate is a full ax_tensor_t allocation
     - nothing is fused across stage boundaries (cache lines bounce
       through L3 between e.g. dwqkv -> ACC and Wo gemm -> bias)
     - tape walk + ctx unmarshalling has fixed overhead per backward

   `ax_mha_train_step` skips the tape entirely. it runs the full
   forward + backward in one self-contained call:
     - intermediates live in TLS scratch reused across calls
     - cross-stage transitions (e.g. dwqkv into bias_grad) are fused
       where the layout permits
     - weight grads are accumulated directly (no scratch+accumulate)

   trade-offs vs the autograd path:
     - no input grad (use the autograd path if you need dx)
     - no chaining: the caller must compute dout = dL/dY itself
     - bypasses ax_grad_enabled / ax_no_grad scopes
     - param grads still respect requires_grad — opted-out params
       don't allocate a grad and don't get accumulated into.

   typical caller (training loop):

     for each (x, y) in batch:
         out = ax_tensor_create(...);           // y_out buffer (reusable)
         dout = compute_loss_grad(out, y);      // [B, S, D]
         ax_mha_train_step(mha, x, dout, out);  // fwd+bwd in one call
         optimizer.step(mha->params);

   args:
     layer:  trained MHA layer (in training mode).
     x:      [B, S, D] input. fp32, contiguous, CPU-resident.
     dout:   [B, S, D] dL/dY. NULL → assume loss = sum(out), so
             dout = ones_like(out) (matches bench harness).
     y_out:  [B, S, D] forward output buffer. must be pre-allocated
             with the correct shape. NULL if the caller doesn't need
             y (rare — dout usually depends on y).

   returns AX_OK on success, error status otherwise. on error, weight
   grads may have been partially accumulated; caller should treat the
   layer as dirty.

   thread-safety: reentrant per-layer (different layers in different
   threads is fine). the TLS scratch is per calling thread, so
   parallel calls on the same layer instance are NOT safe. */
ax_status_t ax_mha_train_step(ax_layer_t *layer,
                               const ax_tensor_t *x,
                               const ax_tensor_t *dout,
                               ax_tensor_t *y_out);

/* F.4: fully-fused mha-train kernel — XLA-class cross-stage tile fusion.
   same signature and contract as ax_mha_train_step, but the body is a
   monolithic per-(qi, kj) tile loop that streams every intermediate
   through L1 instead of materialising arena tensors between stages.
   target: closes the 15-30 % residual gap to TF that the per-stage
   fusion in F.3.* can't reach. see docs/F4_FUSED_MHA_TRAIN.md.

   thread-safety, error contract, bias / dx semantics: same as
   ax_mha_train_step. callers can swap one for the other transparently
   (parity-tested in test_mha_train_step_fused_parity). */
ax_status_t ax_mha_train_step_fused(ax_layer_t *layer,
                                     const ax_tensor_t *x,
                                     const ax_tensor_t *dout,
                                     ax_tensor_t *y_out);

/* F.4.4 monolithic train kernel — multi-phase implementation landing
   site (see docs/F4_FUSED_MHA_TRAIN.md "F.4.4 phased implementation
   plan"). same signature/contract as ax_mha_train_step. each F.4.4
   phase swaps progressively more of the body in place; initial scaffold
   delegates to ax_mha_train_step_fused. */
ax_status_t ax_mha_train_step_v4(ax_layer_t *layer,
                                  const ax_tensor_t *x,
                                  const ax_tensor_t *dout,
                                  ax_tensor_t *y_out);

/* F.4.4 Phase A: per-qi-block SDPA fwd/bwd primitives.

   the F.4.4 driver loops over qi-blocks externally; for each block it
   calls these primitives with qi_start/qi_end specifying the row
   range. different qi-blocks own disjoint rows so concurrent block
   calls are independent for fwd. for bwd, dK / dV are accumulated
   across blocks (caller-zeroed before the first block call); dQ
   rows in [qi_start, qi_end) are overwritten per call.

   shapes match ax_fused_attention_fwd_save_to_flat /
   ax_fused_attention_bwd_use_from_flat respectively, with
   qi_start, qi_end inserted between the (B, S, H, dk) shape group
   and the (scale, causal) tail. */
void ax_fused_attention_fwd_save_to_flat_qi_block(
    const float *Q, const float *K, const float *V,
    float *attn_flat, float *L, float *P_save,
    int64_t B, int64_t S, int64_t H, int64_t dk,
    int64_t qi_start, int64_t qi_end,
    float scale, bool causal);

void ax_fused_attention_bwd_use_from_flat_qi_block(
    const float *Q, const float *K, const float *V,
    const float *attn_flat, const float *dO, const float *L,
    const float *P_saved,
    float *dQ, float *dK, float *dV,
    int64_t B, int64_t S, int64_t H, int64_t dk,
    int64_t qi_start, int64_t qi_end,
    float scale, bool causal);

/* ================================================================
   SDPA PRIMITIVE — raw compute, use when you manage Q/K/V yourself
   ================================================================ */

/* scaled dot-product attention forward.
   Q, K, V: [BH, S, dk] row-major (heads flattened into batch dim).
   out:     [BH, S, dk] — written, need not be pre-zeroed.
   L:       [BH, S]     — log-sum-exp per row, needed for backward.
                          caller must allocate. pass NULL to skip
                          (inference-only path, saves ~2% of forward). */
void ax_fused_attention_fwd(const float *Q, const float *K, const float *V,
                             float *out, float *L,
                             int64_t BH, int64_t S, int64_t dk, float scale);

/* same as fwd, with causal mask (position qi attends only to kj ≤ qi). */
void ax_fused_attention_fwd_causal(const float *Q, const float *K, const float *V,
                                    float *out, float *L,
                                    int64_t BH, int64_t S, int64_t dk, float scale);

/* scaled dot-product attention backward.
   requires O (forward output), dO (upstream grad), L (logsumexp).
   produces dQ, dK, dV (accumulated — not overwritten; caller zeros). */
void ax_fused_attention_bwd(const float *Q, const float *K, const float *V,
                             const float *O, const float *dO, const float *L,
                             float *dQ, float *dK, float *dV,
                             int64_t BH, int64_t S, int64_t dk, float scale);

void ax_fused_attention_bwd_causal(const float *Q, const float *K, const float *V,
                                    const float *O, const float *dO, const float *L,
                                    float *dQ, float *dK, float *dV,
                                    int64_t BH, int64_t S, int64_t dk, float scale);

/* save-aware variants — opt-in materialized P[BH, S, S] for backward reuse.
   forward writes the post-mask pre-softmax scores into P_save; backward
   reads them via P_saved instead of recomputing Q @ K^T. cuts the bwd
   per-head work by ~25-30% (skipped GEMM). costs BH*S*S floats memory.
   only worthwhile when the buffer fits in cache (small S). */
void ax_fused_attention_fwd_save(const float *Q, const float *K, const float *V,
                                  float *out, float *L, float *P_save,
                                  int64_t BH, int64_t S, int64_t dk, float scale,
                                  bool causal);

void ax_fused_attention_bwd_use(const float *Q, const float *K, const float *V,
                                 const float *O, const float *dO, const float *L,
                                 const float *P_saved,
                                 float *dQ, float *dK, float *dV,
                                 int64_t BH, int64_t S, int64_t dk, float scale,
                                 bool causal);

/* F.3.e companion: SDPA backward reading O from [B, S, D] attn_flat layout.
   pair with ax_fused_attention_fwd_save_to_flat to eliminate the per-head
   [BH, S, dk] Oh intermediate from the MHA training path entirely.
   only the Di = dot(O, dO) computation differs from the standard
   ax_fused_attention_bwd_use — per-row O read uses stride D into
   attn_flat instead of stride dk into Oh. dO/dQ/dK/dV remain
   head-major [BH, S, dk]. */
void ax_fused_attention_bwd_use_from_flat(const float *Q, const float *K, const float *V,
                                            const float *attn_flat,
                                            const float *dO, const float *L,
                                            const float *P_saved,
                                            float *dQ, float *dK, float *dV,
                                            int64_t B, int64_t S, int64_t H, int64_t dk,
                                            float scale, bool causal);

/* F.3.e fused SDPA forward writing directly into [B, S, D] attn_flat layout.
   eliminates the per-head [BH, S, dk] Oh intermediate AND the
   head_deinterleave pass that converts it to attn_flat in the MHA training
   path. each of B*H heads' per-row output is strided by D = H*dk; the
   dk-wide slot for head h lives at attn_flat[b*S*D + h*dk] and adjacent
   heads' slots fill the rest of each row.

   shapes: Q/K/V are [BH = B*H, S, dk] head-major (same as the standard
   fwd primitives). attn_flat is [B, S, D] = [B*S, H*dk]. L is
   [BH, S] log-sum-exp; P_save is the optional [BH, S, S] post-mask
   pre-softmax score buffer for the backward pass.

   the head_deinterleave pass that follows the standard fwd_save call
   becomes a no-op when this entry is used — saves ~6 MB of redundant
   read+write traffic on B1_S2048_D768 and similar shapes. */
void ax_fused_attention_fwd_save_to_flat(const float *Q, const float *K, const float *V,
                                           float *attn_flat, float *L, float *P_save,
                                           int64_t B, int64_t S, int64_t H, int64_t dk,
                                           float scale, bool causal);

/* ================================================================
   KV CACHE — for autoregressive inference
   ================================================================ */

typedef struct {
    float *K;           /* [BH, max_seq, dk] preallocated, contiguous */
    float *V;           /* [BH, max_seq, dk] preallocated, contiguous */
    int64_t BH;
    int64_t max_seq;    /* capacity */
    int64_t dk;
    int64_t cur_len;    /* current valid length (number of cached tokens) */
} ax_kv_cache_t;

/* allocate a KV cache. K and V buffers are 64-byte-aligned via
   ax_aligned_alloc. returns NULL on failure. */
ax_kv_cache_t *ax_kv_cache_create(int64_t BH, int64_t max_seq, int64_t dk);
void ax_kv_cache_destroy(ax_kv_cache_t *c);

/* reset cur_len to zero. does not zero the buffers (overwritten on next append). */
void ax_kv_cache_reset(ax_kv_cache_t *c);

/* append one new token's K and V. advances cur_len by 1.
   new_K, new_V: [BH, dk] each — a single row for each head.
   returns false if cache is full (cur_len == max_seq). */
bool ax_kv_cache_append(ax_kv_cache_t *c, const float *new_K, const float *new_V);

/* autoregressive attention: compute attention for a SINGLE new query
   token against all cached K/V positions.
   Q_new: [BH, dk] — new token's query vector.
   out:   [BH, dk] — attention output for the new token.
   uses a rank-1 GEMV path (much faster than full SDPA for single-token
   inference). scale = 1/sqrt(dk). */
void ax_kv_cache_attend(const ax_kv_cache_t *c,
                         const float *Q_new, float *out,
                         float scale);

/* ================================================================
   POSITIONAL ENCODING — rotary position embeddings (RoPE)
   ================================================================ */

/* apply RoPE in-place to Q and K. dk must be even.
   positions[i] is the sequence position of token i (0-indexed).
   theta_base is the rotation base frequency (10000.0 is standard).
   Q, K: [BH, S, dk] row-major.
   positions: [S] — token position indices.

   RoPE applies a rotation by angle θ_i,k to dimension pair (2k, 2k+1)
   at position i, where θ_i,k = i / theta_base^(2k/dk). this encodes
   relative position without learnable parameters and generalizes to
   sequences longer than trained on. */
void ax_rope_apply(float *Q, float *K, const int64_t *positions,
                    int64_t BH, int64_t S, int64_t dk, float theta_base);

#endif /* AX_ATTENTION_H */
