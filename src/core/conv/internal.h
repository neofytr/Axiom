/* conv/internal.h — file-private contracts shared between the pieces of
   the conv module (forward.c, backward.c, im2col.c, direct.c, winograd.c,
   path_selection.c, conv_bn_relu.c).

   not part of any public api — kept under src/core/conv/ rather than
   include/axiom/internal/ because the contract changes whenever the conv
   layer is restructured. user code that needs conv2d should go through
   the public ax_conv2d_create / ax_layer_forward api in axiom/conv.h.

   k.2 split: extracted from the 3.6 KLOC src/core/conv.c. the original
   file held all six conv paths (winograd, direct 3x3 s1, direct 3x3 s2,
   direct small-Cin, im2col+gemm, NHWC) plus the fused conv+bn+relu layer,
   and was navigationally unmaintainable. the per-thread scratch struct
   below is shared across forward, backward, NHWC impls, and the cbr
   fused layer — it lives here so each tu can see the full layout. */

#ifndef AX_CORE_CONV_INTERNAL_H
#define AX_CORE_CONV_INTERNAL_H

#include "axiom/tensor.h"
#include "axiom/conv.h"
#include "axiom/autograd.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* conv path identifiers — used by the autotuner to cache the fastest
   path per shape and by forward.c to dispatch. */
typedef enum {
    AX_CONV_PATH_AUTO = -1,
    AX_CONV_PATH_WINOGRAD = 0,
    AX_CONV_PATH_DIRECT_3X3,
    AX_CONV_PATH_SMALLCIN,
    AX_CONV_PATH_IMPLICIT,
    AX_CONV_PATH_IM2COL,
    AX_CONV_PATH_BATCHED,
    AX_CONV_PATH_1X1_FAST,
    AX_CONV_PATH_WINOGRAD_F43,
    AX_CONV_PATH_COUNT
} ax_conv_path_t;

/* from dispatch.c — thread-local GEMM control flags */
extern void ax_gemm_set_skip_init(bool v);
extern void ax_gemm_set_inner_team(bool v);
extern bool ax_gemm_get_inner_team(void);

/* batched conv heuristic thresholds. file-scope so forward, backward, and
   the cbr fused layer share definitions without depending on translation-
   unit ordering. AX_CONV_BATCH_COL_BYTES default is 256 MB so chunking
   effectively never triggers for normal shapes — measurement showed
   chunking adds OMP setup overhead and splits one big GEMM into many
   smaller ones with poor jc-parallel utilization (e.g. nb=2 on N=32 gives
   16 chunks of 2 jc tiles each, wasting threads). embedded targets that
   need bounded scratch can override via -DAX_CONV_BATCH_COL_BYTES at
   compile time. */
#define AX_CONV_BATCH_M_THRESH 512
#ifndef AX_CONV_BATCH_COL_BYTES
#define AX_CONV_BATCH_COL_BYTES ((int64_t)(256 * 1024 * 1024))
#endif

/* per-thread scratch buffers cached on ax_conv2d_t.
   sized for the largest seen (N, H, W) signature; reallocated only when
   the signature changes (e.g., batch size differs from training). */
struct ax_conv_scratch {
    /* signature: shapes are determined by these */
    int64_t N, H, W;
    int T;  /* number of per-thread slots */

    /* per-thread buffers (T slots each) */
    ax_tensor_t **col_bufs;   /* [K, M] for forward and backward im2col */
    ax_tensor_t **res_bufs;   /* [C_out, M] for forward GEMM result */
    ax_tensor_t **go_bufs;    /* [C_out, M] for backward grad_out matrix */
    ax_tensor_t **dw_bufs;    /* [C_out, K] per-thread weight grad accumulator */
    ax_tensor_t **dws_bufs;   /* [C_out, K] per-thread per-sample weight grad scratch */
    ax_tensor_t **dcol_bufs;  /* [K, M] for input gradient via gemm */
    ax_tensor_t **colt_bufs;  /* [M, K] transposed col for weight grad gemm */
    ax_tensor_t **dimg_bufs;  /* [C_in, H, W] for col2im result */

    /* shared (read-only across threads) */
    ax_tensor_t *w2d;       /* [C_out, K] reshaped forward weight */
    ax_tensor_t *wt_contig; /* [K, C_out] transposed weight for backward dx */

    /* batched-GEMM buffers (allocated when M < AX_CONV_BATCH_M_THRESH && N > 1) */
    ax_tensor_t *batch_col_buf; /* [K, n_batch*M]: im2col in fwd; dcol_batch in bwd */
    ax_tensor_t *batch_aux_buf; /* [C_out, n_batch*M]: gemm result in fwd; go_batch in bwd */
    int64_t n_batch;            /* samples per sub-batch; 0 or 1 = batched path disabled */

    /* winograd F(2,3) buffers (allocated when prefer_winograd_f23 returns true).
       U: [16, C_out, C_in] transformed kernel (re-filled each forward; weights change).
       V: [16, C_in, num_tiles] transformed input (per forward).
       Mout: [16, C_out, num_tiles] gemm result, then output-transform to od. */
    ax_tensor_t *wino_U;
    ax_tensor_t *wino_V;
    ax_tensor_t *wino_M;
    int64_t wino_num_tiles;     /* N * Ty * Tx; 0 if winograd not allocated */

    /* winograd F(4,3) buffers (same structure as F(2,3), 36 ij planes, 4x4 output tiles) */
    ax_tensor_t *wino43_U;
    ax_tensor_t *wino43_V;
    ax_tensor_t *wino43_M;
    int64_t wino43_num_tiles;

    /* autotuner result for this shape (AX_CONV_PATH_AUTO = not yet calibrated) */
    ax_conv_path_t calibrated_path;

    /* NHWC path scratch (allocated lazily on first NHWC forward call):
       wt_nhwc: weight transposed to [K=Cin*kh*kw, Cout] for cache-friendly
                GEMM with channels-last input (re-filled when weight->generation
                changes between forward calls — training updates the weight).
       im2col_nhwc: [N*OH*OW, K] gathered patches from [N, H, W, Cin] input. */
    ax_tensor_t *wt_nhwc;
    uint64_t wt_nhwc_gen;
    ax_tensor_t *im2col_nhwc;
};

/* ================================================================
   shared helpers — defined in forward.c (scratch + tensor view)
   ================================================================ */

/* overflow-checked multiply. returns -1 on overflow. */
static inline int64_t ax_conv_safe_mul(int64_t a, int64_t b)
{
    if (a <= 0 || b <= 0) return (a == 0 || b == 0) ? 0 : -1;
    if (a > INT64_MAX / b) return -1;
    return a * b;
}

/* compute output spatial dimension. mirrors the helper in pool.c — both
   tus need it independently. returns -1 on invalid configuration. */
static inline int64_t ax_conv_out_dim(int64_t in_dim, int kernel, int stride, int pad)
{
    if (stride <= 0) return -1;
    int64_t out = (in_dim + 2 * pad - kernel) / stride + 1;
    if (out <= 0) return -1;
    return out;
}

/* 1×1 stride=1 pad=0: im2col is the identity rearrangement.
   the per-sample fast path uses a stack-tensor view to skip the copy
   entirely (zero memcpy). batched path's wider GEMM doesn't help here
   because the im2col-as-transpose cost (~12 MB per forward at vgg shapes)
   exceeds any NC-utilization gain. forward AND backward should route to
   per-sample. */
static inline bool ax_conv_is_1x1_pad0_stride1(int kh, int kw, int sh, int sw, int ph, int pw)
{
    return kh == 1 && kw == 1 && sh == 1 && sw == 1 && ph == 0 && pw == 0;
}

/* construct a zero-overhead 2-D stack tensor view over an existing float
   buffer. storage and tensor must be caller-allocated locals; no heap
   touched. defined in forward.c. */
void ax_conv_make_stack_view(ax_tensor_t *tv, ax_storage_t *st,
                              float *data, int64_t rows, int64_t cols);

/* destroy a scratch struct and all per-thread tensors. defined in forward.c. */
void ax_conv_scratch_destroy(struct ax_conv_scratch *s);

/* lazily allocate or reallocate scratch buffers if signature differs.
   defined in forward.c. */
struct ax_conv_scratch *ax_conv_ensure_scratch(ax_conv2d_t *conv,
                                                int64_t N, int64_t H, int64_t W,
                                                bool need_backward);

/* ================================================================
   im2col / col2im — defined in im2col.c
   ================================================================ */

/* internal: write im2col into a pre-allocated float buffer.
   input must be contiguous [C, H, W]. cd must be at least K*M floats. */
void ax_conv_im2col_into(const float *id, int64_t C, int64_t H, int64_t W,
                          int kh, int kw, int stride_h, int stride_w,
                          int pad_h, int pad_w, int64_t out_h, int64_t out_w,
                          float *cd);

/* like im2col_into but writes into a column-strided batch buffer.
   col_stride is the distance between consecutive columns in `cd` (normally M;
   for batched buffers: N*M). sample_offset is the column start for this
   sample within each row (0 for single; n*M for batched sample n). */
void ax_conv_im2col_into_strided(const float *id, int64_t C, int64_t H, int64_t W,
                                  int kh, int kw, int stride_h, int stride_w,
                                  int pad_h, int pad_w, int64_t out_h, int64_t out_w,
                                  float *cd, int64_t col_stride, int64_t sample_offset);

/* internal: scatter col2im into a pre-zeroed float buffer.
   id must be zeroed before calling (accumulates into it). */
void ax_conv_col2im_into(const float *cd, int64_t channels,
                          int64_t height, int64_t width,
                          int kh, int kw, int stride_h, int stride_w,
                          int pad_h, int pad_w, int64_t out_h, int64_t out_w,
                          float *id);

/* NHWC im2col: input [N, H, W, Cin] → cd [N*OH*OW, Cin*kh*kw]. */
void ax_conv_im2col_nhwc(const float *id, int64_t N, int64_t H, int64_t W,
                          int64_t C_in, int kh, int kw, int sh, int sw,
                          int ph, int pw, int64_t out_h, int64_t out_w, float *cd);

/* NHWC col2im: cd [N*OH*OW, Cin*kh*kw] → id [N, H, W, Cin] (accumulating). */
void ax_conv_col2im_nhwc(const float *cd, int64_t N, int64_t H, int64_t W,
                          int64_t C_in, int kh, int kw, int sh, int sw,
                          int ph, int pw, int64_t out_h, int64_t out_w, float *id);

/* ================================================================
   direct convolutions — defined in direct.c
   ================================================================ */

/* direct 3x3 stride=1 pad=1 conv for a single sample.
   skips the gemm/im2col detour when the kernel is small enough to stay
   hot in L1. output shape [C_out, H, W] (pad=1 keeps spatial size). */
void ax_conv_direct_3x3_sample(const float *in_n, const float *wd, const float *bias,
                                float *out_n, int64_t C_in, int64_t C_out,
                                int64_t H, int64_t W);

/* direct 3x3 stride=2 pad=1 conv for a single sample. */
void ax_conv_direct_3x3_s2_sample(const float *in_n, const float *wd, const float *bias,
                                   float *out_n, int64_t C_in, int64_t C_out,
                                   int64_t H, int64_t W,
                                   int64_t out_h, int64_t out_w);

/* direct conv for very-small input channels (typical first layer C_in=3 RGB). */
void ax_conv_direct_smallcin_sample(const float *in_n, const float *wd, const float *bias,
                                     float *out_n,
                                     int64_t C_in, int64_t C_out, int64_t H, int64_t W,
                                     int kh, int kw, int sh, int sw, int ph, int pw,
                                     int64_t H_out, int64_t W_out);

/* eligibility predicates for direct paths. */
bool ax_conv_can_direct_3x3(int kh, int kw, int sh, int sw, int ph, int pw, int64_t C_in);
bool ax_conv_can_direct_3x3_s2(int kh, int kw, int sh, int sw,
                                int ph, int pw, int64_t C_in,
                                int64_t H, int64_t W);
bool ax_conv_can_direct_smallcin(int kh, int kw, int sh, int sw,
                                  int64_t C_in, int64_t W_out);

/* ================================================================
   winograd F(2,3) — defined in winograd.c
   ================================================================ */

/* per-tile input transform: V = B^T * d * B. d and v are 4x4 row-major. */
void ax_conv_wino_input_transform_tile(const float *d, float *v);

/* per-filter kernel transform: U = G * g * G^T. g is 3x3, u is 4x4. */
void ax_conv_wino_kernel_transform_filter(const float *g, float *u);

/* per-tile output transform: y = A^T * m * A. m is 4x4, y is 2x2. */
void ax_conv_wino_output_transform_tile(const float *m, float *y);

/* eligibility predicate for winograd F(2,3). */
bool ax_conv_prefer_winograd_f23(int kh, int kw, int sh, int sw, int ph, int pw,
                                  int64_t N, int64_t C_in, int64_t C_out,
                                  int64_t out_h, int64_t out_w);

/* winograd F(2,3) forward kernel for the whole batch. */
int ax_conv_winograd_f23_forward(struct ax_conv_scratch *s,
                                  const float *id, const float *wd, const float *bias,
                                  float *od,
                                  int64_t N, int64_t C_in, int64_t H, int64_t W,
                                  int64_t C_out, int64_t out_h, int64_t out_w, int T);

/* winograd F(2,3) backward dX: dX = conv(grad_out, W_rot180, s=1, p=1).
   reuses forward's wino_U/V/M buffers. accumulates into dx. */
int ax_conv_winograd_f23_backward_dx(struct ax_conv_scratch *s,
                                      const float *grad_out, const float *wd,
                                      float *dx,
                                      int64_t N, int64_t C_in, int64_t H, int64_t W,
                                      int64_t C_out, int64_t out_h, int64_t out_w,
                                      int T);

/* ================================================================
   winograd F(4,3) — defined in winograd.c
   ================================================================ */

void ax_conv_wino43_input_transform_tile(const float *d, float *v);
void ax_conv_wino43_kernel_transform_filter(const float *g, float *u);
void ax_conv_wino43_output_transform_tile(const float *m, float *y);

bool ax_conv_prefer_winograd_f43(int kh, int kw, int sh, int sw, int ph, int pw,
                                  int64_t N, int64_t C_in, int64_t C_out,
                                  int64_t out_h, int64_t out_w);

int ax_conv_winograd_f43_forward(struct ax_conv_scratch *s,
                                  const float *id, const float *wd, const float *bias,
                                  float *od,
                                  int64_t N, int64_t C_in, int64_t H, int64_t W,
                                  int64_t C_out, int64_t out_h, int64_t out_w, int T);

int ax_conv_winograd_f43_backward_dx(struct ax_conv_scratch *s,
                                      const float *grad_out, const float *wd,
                                      float *dx,
                                      int64_t N, int64_t C_in, int64_t H, int64_t W,
                                      int64_t C_out, int64_t out_h, int64_t out_w,
                                      int T);

/* ================================================================
   path selection — defined in path_selection.c
   ================================================================ */

/* implicit GEMM path predicate (gather-on-the-fly into pack_b). */
bool ax_conv_prefer_implicit_gemm(int64_t K, int64_t M);

/* ================================================================
   autotuner — defined in autotune.c
   ================================================================ */

/* trial eligible paths for a conv shape and return the fastest.
   writes the winning trial's output into od. called from forward.c
   on the first forward for a new shape. no-op under AX_NO_AUTOTUNE=1
   (returns the static cascade's pick). */
ax_conv_path_t ax_conv_autotune_select(
    struct ax_conv_scratch *s,
    const float *id, const float *wd, const float *bias, float *od,
    int64_t N, int64_t C_in, int64_t H, int64_t W,
    int64_t C_out, int64_t out_h, int64_t out_w,
    int kh, int kw, int sh, int sw, int ph, int pw, int T,
    ax_conv_path_t static_pick);

/* ================================================================
   NHWC kernels — defined in forward.c (forward) and backward.c (backward)
   ================================================================ */

int ax_conv_nhwc_forward_impl(struct ax_conv_scratch *s,
                               const float *id, const float *wd, const float *bias,
                               float *od,
                               int64_t N, int64_t Cin, int64_t H, int64_t W,
                               int64_t Cout, int64_t out_h, int64_t out_w,
                               int kh, int kw, int sh, int sw, int ph, int pw,
                               uint64_t weight_gen);

void ax_conv_nhwc_backward_impl(struct ax_conv_scratch *s,
                                 const float *id, const float *grd, const float *wd,
                                 float *dwd, float *dbd, float *dxd,
                                 int64_t N, int64_t Cin, int64_t H, int64_t W,
                                 int64_t Cout, int64_t out_h, int64_t out_w,
                                 int kh, int kw, int sh, int sw, int ph, int pw,
                                 uint64_t weight_gen);

/* NHWC weight cache refresh helper (used by both forward and backward NHWC). */
void ax_conv_refresh_wt_nhwc(struct ax_conv_scratch *s, const float *wd_orig,
                              int64_t Cout, int64_t Cin, int kh, int kw,
                              uint64_t src_gen);

/* ================================================================
   conv2d_backward (autograd hook) — defined in backward.c
   ================================================================ */

void ax_conv_conv2d_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out);

#ifdef __cplusplus
}
#endif

#endif /* AX_CORE_CONV_INTERNAL_H */
