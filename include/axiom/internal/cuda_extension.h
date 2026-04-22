/* axiom/internal/cuda_extension.h — cuda backend extension vtable.

   the cuda backend exposes a number of operations that the cpu code
   path needs to call when running on cuda tensors (mha layout transforms,
   sdpa fwd/bwd, conv batched gemm, weight/bias cache builds). these are
   genuinely cuda-only — they have no cpu equivalent in the generic
   backend_ops_t vtable, and live in the cuda backend module's tu.

   prior approach: cpu code declared each of these as
   __attribute__((weak)) extern, and the cuda backend.cu had a
   __attribute__((used)) force-link table that prevented the linker from
   stripping the symbols out of the static archive when no strong
   reference existed. the resulting code was scattered, hard to follow,
   and required keeping the force-link table in sync with the call sites.

   k.4 replaces that with a single registry: the cuda backend constructs
   one ax_cuda_extension_t struct, populates the function pointers, and
   calls ax_compute_register_cuda_extension(...) at init. cpu code calls
   ax_compute_get_cuda_extension() and dispatches through the struct.
   removes 10 weak externs, the force-link table, and the implicit
   contract about which symbols backend.cu must keep alive.

   not part of the public abi — lives in axiom/internal/. user code that
   wants to extend axiom with a custom cuda implementation populates this
   struct and registers it; the api is unstable across minor releases. */

#ifndef AX_INTERNAL_CUDA_EXTENSION_H
#define AX_INTERNAL_CUDA_EXTENSION_H

#include "axiom/types.h"
#include "axiom/compute.h"  /* ax_conv_params_t, ax_tensor_t */

#ifdef __cplusplus
extern "C" {
#endif

/* every field is a function pointer to a cuda-side implementation.
   may be NULL if the cuda backend was built without that op (e.g. an
   older module). callers MUST check for NULL before dispatching;
   ax_compute_has_cuda_<op>() helpers are added for the hot paths. */
typedef struct {
    /* layout transforms ----------------------------------------------- */
    /* head-interleave from packed [BH, S, dk] to [B, S, H, dk] split
       across q/k/v output buffers. */
    void (*head_interleave_qkv_split)(const float *src,
                                       float *q, float *k, float *v,
                                       int64_t B, int64_t S, int64_t H,
                                       int64_t dk, int64_t D);

    /* same as head_interleave_qkv_split but adds the concatenated [3*D]
       bias panel (the cached bqkv buffer = [bq | bk | bv]) into the
       corresponding output row. */
    void (*head_interleave_qkv_split_bias)(const float *src,
                                            const float *bqkv,
                                            float *q, float *k, float *v,
                                            int64_t B, int64_t S, int64_t H,
                                            int64_t dk, int64_t D);

    /* head-interleave the gradient block. used in mha backward. */
    void (*head_interleave)(const float *src, float *dst,
                             int64_t B, int64_t S, int64_t H, int64_t dk);

    /* inverse — head-deinterleave. */
    void (*head_deinterleave)(const float *src, float *dst,
                               int64_t B, int64_t S, int64_t H, int64_t dk);

    /* fused head-deinterleave + qkv merge to assemble dQKV from dQ/dK/dV
       blocks. used in mha backward to reverse the qkv split. */
    void (*head_deinterleave_qkv_merge)(const float *dQ, const float *dK, const float *dV,
                                         float *dst,
                                         int64_t B, int64_t S, int64_t H,
                                         int64_t dk, int64_t D);

    /* qkv weight/bias cache rebuilds ---------------------------------- */
    /* assemble [D, 3D] Wqkv panel from three [D, D] inputs. */
    ax_status_t (*qkv_cache_build)(float *dst,
                                    const float *wq, const float *wk, const float *wv,
                                    int64_t D);

    /* assemble [3D] bqkv from three [D] inputs. */
    ax_status_t (*bias_cache_build)(float *dst,
                                     const float *bq, const float *bk, const float *bv,
                                     int64_t D);

    /* sdpa fwd + bwd -------------------------------------------------- */
    /* scaled dot-product attention forward on cuda. fills out + L
       (log-sum-exp). optional P_save buffer materialises post-softmax
       scores for the save-aware backward. */
    ax_status_t (*sdpa_fwd)(const float *Q, const float *K, const float *V,
                             float *out, float *L,
                             int64_t BH, int64_t S, int64_t dk, float scale,
                             bool causal, float *P_save);

    /* scaled dot-product attention backward on cuda. produces dQ/dK/dV
       (additive). when P_saved is non-NULL, skips the QK^T recompute. */
    ax_status_t (*sdpa_bwd)(const float *Q, const float *K, const float *V,
                             const float *O, const float *dO, const float *L,
                             float *dQ, float *dK, float *dV,
                             int64_t BH, int64_t S, int64_t dk, float scale,
                             bool causal, const float *P_saved);

    /* batched conv gemm ---------------------------------------------- */
    /* one cublas stridedBatched gemm for all N samples of a conv.
       in_dev points to all N samples concatenated; out is also N-batched. */
    ax_status_t (*conv_gemm_batched)(const ax_tensor_t *weight,
                                      const float *in_dev, int64_t N,
                                      const ax_conv_params_t *single,
                                      ax_tensor_t *out);
} ax_cuda_extension_t;

/* register the cuda backend's extension table. called once from the
   cuda backend init hook. passing NULL clears the registration (used
   by ax_compute_shutdown for clean teardown). */
ax_status_t ax_compute_register_cuda_extension(const ax_cuda_extension_t *ext);

/* return the currently-registered cuda extension, or NULL if no cuda
   backend is loaded. callers branch on the pointer to decide whether
   to dispatch through cuda or fall back to the cpu path. */
const ax_cuda_extension_t *ax_compute_get_cuda_extension(void);

#ifdef __cplusplus
}
#endif

#endif /* AX_INTERNAL_CUDA_EXTENSION_H */
