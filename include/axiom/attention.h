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
     is cheaper when the score matrix exceeds ~half of L3. */

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
