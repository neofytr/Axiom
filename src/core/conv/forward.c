/* conv/forward.c — conv2d forward + ax_conv2d_create / _create_ex,
   per-layer scratch lifecycle, and the NHWC forward impl.

   k.2 split: this is what's left of the legacy 3.6 KLOC src/core/conv.c
   after the other six tus (im2col, direct, winograd, path_selection,
   backward, conv_bn_relu) were peeled off. forward picks one of six
   conv paths per call (winograd, direct 3x3 s1, direct 3x3 s2, direct
   small-Cin, implicit gemm, im2col+gemm) using the predicates from
   internal.h, parallelises the outer (sample) loop with openmp, and
   wires backward into the autograd graph when grad-tracking is on.

   the per-thread scratch struct (ax_conv_scratch) is allocated lazily
   on the first forward call and re-used across calls — its allocation
   logic (ensure_scratch) is here because it lives on ax_conv2d_t and
   needs to know the full set of per-path buffer requirements.

   the layer constructors (_create / _create_ex) and the destroy hook
   live here because conv2d_forward references them as the layer's
   forward fn-pointer. the bigger fused conv_bn_relu layer is its own
   tu (conv/conv_bn_relu.c) since it has its own forward + backward. */

#include "axiom/conv.h"
#include "axiom/ops.h"
#include "axiom/autograd.h"
#include "axiom/init.h"
#include "axiom/compute.h"
#include "axiom/internal/cuda_extension.h"
#include "axiom/error.h"
#include "internal.h"
#include "../../compute/backends/simd_defs.h"
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#define AX_OMP_MAX_THREADS() omp_get_max_threads()
#define AX_OMP_THREAD_NUM() omp_get_thread_num()
#else
#define AX_OMP_MAX_THREADS() 1
#define AX_OMP_THREAD_NUM() 0
#endif
#include <inttypes.h>
#include <float.h>

/* short-name aliases for the helpers and predicates extracted to other
   tus. these keep the original call sites in this file readable
   (ax_conv_im2col_into is a mouthful in already-dense code) without
   having to reflow every line. the function definitions for scratch
   lifecycle (scratch_destroy, ensure_scratch, make_stack_view) below
   take the same aliases — the macro substitution turns the short name
   into the ax_conv_ symbol both in the definition and at call sites. */
#define safe_mul                      ax_conv_safe_mul
#define conv_out_dim                  ax_conv_out_dim
#define is_1x1_pad0_stride1           ax_conv_is_1x1_pad0_stride1
#define make_stack_view               ax_conv_make_stack_view
#define scratch_destroy               ax_conv_scratch_destroy
#define ensure_scratch                ax_conv_ensure_scratch
#define im2col_into                   ax_conv_im2col_into
#define im2col_into_strided           ax_conv_im2col_into_strided
#define col2im_into                   ax_conv_col2im_into
/* (im2col_nhwc / col2im_nhwc / refresh_wt_nhwc are NOT aliased — their
   short names collide with struct field names. callers in this tu use
   the ax_conv_ prefix explicitly when they need to call them.) */
#define conv2d_direct_3x3_sample      ax_conv_direct_3x3_sample
#define conv2d_direct_3x3_s2_sample   ax_conv_direct_3x3_s2_sample
#define conv2d_direct_smallcin_sample ax_conv_direct_smallcin_sample
#define conv2d_nhwc_forward_impl      ax_conv_nhwc_forward_impl
#define conv2d_nhwc_backward_impl     ax_conv_nhwc_backward_impl
#define conv2d_winograd_f23_forward   ax_conv_winograd_f23_forward
#define wino_input_transform_tile     ax_conv_wino_input_transform_tile
#define wino_kernel_transform_filter  ax_conv_wino_kernel_transform_filter
#define wino_output_transform_tile    ax_conv_wino_output_transform_tile
#define can_direct_3x3                ax_conv_can_direct_3x3
#define can_direct_3x3_s2             ax_conv_can_direct_3x3_s2
#define can_direct_smallcin           ax_conv_can_direct_smallcin
#define prefer_winograd_f23           ax_conv_prefer_winograd_f23
#define prefer_implicit_gemm          ax_conv_prefer_implicit_gemm

void scratch_destroy(struct ax_conv_scratch *s)
{
    if (!s) return;
    int T = s->T;
    #define FREE_BUF_ARR(arr) \
        if (arr) { \
            for (int t = 0; t < T; t++) if (arr[t]) ax_tensor_destroy(arr[t]); \
            free(arr); \
        }
    FREE_BUF_ARR(s->col_bufs);
    FREE_BUF_ARR(s->res_bufs);
    FREE_BUF_ARR(s->go_bufs);
    FREE_BUF_ARR(s->dw_bufs);
    FREE_BUF_ARR(s->dws_bufs);
    FREE_BUF_ARR(s->dcol_bufs);
    FREE_BUF_ARR(s->colt_bufs);
    FREE_BUF_ARR(s->dimg_bufs);
    #undef FREE_BUF_ARR
    if (s->w2d) ax_tensor_destroy(s->w2d);
    if (s->wt_contig) ax_tensor_destroy(s->wt_contig);
    if (s->batch_col_buf) ax_tensor_destroy(s->batch_col_buf);
    if (s->batch_aux_buf) ax_tensor_destroy(s->batch_aux_buf);
    if (s->wino_U) ax_tensor_destroy(s->wino_U);
    if (s->wino_V) ax_tensor_destroy(s->wino_V);
    if (s->wino_M) ax_tensor_destroy(s->wino_M);
    if (s->wt_nhwc) ax_tensor_destroy(s->wt_nhwc);
    if (s->im2col_nhwc) ax_tensor_destroy(s->im2col_nhwc);
    free(s);
}

/* helper: allocate an array of T tensors with the given shape */
static ax_tensor_t **alloc_buf_array(int T, const int64_t *shape, int ndim)
{
    ax_tensor_t **arr = (ax_tensor_t **)calloc((size_t)T, sizeof(ax_tensor_t *));
    if (!arr) return NULL;
    for (int t = 0; t < T; t++) {
        arr[t] = ax_tensor_create(shape, ndim, AX_FLOAT32);
        if (!arr[t]) {
            for (int u = 0; u < t; u++) ax_tensor_destroy(arr[u]);
            free(arr);
            return NULL;
        }
    }
    return arr;
}

/* construct a zero-overhead 2-D stack tensor view over an existing float buffer.
   storage and tensor must be caller-allocated locals; no heap touched. */
void make_stack_view(ax_tensor_t *tv, ax_storage_t *st,
                                   float *data, int64_t rows, int64_t cols)
{
    st->data         = data;
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

/* lazily allocate or reallocate scratch buffers if signature differs.
   call this from conv2d_forward/backward before any parallel region
   (single-threaded init guarantees no race on the scratch pointer). */
struct ax_conv_scratch *ensure_scratch(ax_conv2d_t *conv,
                                                int64_t N, int64_t H, int64_t W,
                                                bool need_backward)
{
    int64_t C_in = conv->in_channels;
    int64_t C_out = conv->out_channels;
    int kh = conv->kernel_h, kw = conv->kernel_w;
    int sh = conv->stride_h, sw = conv->stride_w;
    int ph = conv->pad_h, pw = conv->pad_w;
    int64_t out_h = conv_out_dim(H, kh, sh, ph);
    int64_t out_w = conv_out_dim(W, kw, sw, pw);
    if (out_h <= 0 || out_w <= 0) return NULL;
    int64_t K = C_in * kh * kw;
    int64_t M = out_h * out_w;

    /* compute desired thread count: capped at batch size */
    int T = AX_OMP_MAX_THREADS();
    if (T > N) T = (int)N;
    if (T < 1) T = 1;

    struct ax_conv_scratch *s = conv->scratch;
    if (s && s->N == N && s->H == H && s->W == W && s->T == T) {
        /* signature matches; reuse existing buffers. for backward we need
           the dw/dcol/colt/dimg arrays — reallocate them lazily if missing. */
        if (need_backward && !s->dw_bufs) {
            int64_t dw_shape[] = {C_out, K};
            int64_t dcol_shape[] = {K, M};
            int64_t colt_shape[] = {M, K};
            int64_t dimg_shape[] = {C_in, H, W};
            int64_t go_shape[] = {C_out, M};
            s->go_bufs   = alloc_buf_array(T, go_shape, 2);
            s->dw_bufs   = alloc_buf_array(T, dw_shape, 2);
            s->dws_bufs  = alloc_buf_array(T, dw_shape, 2);
            s->dcol_bufs = alloc_buf_array(T, dcol_shape, 2);
            s->colt_bufs = alloc_buf_array(T, colt_shape, 2);
            s->dimg_bufs = alloc_buf_array(T, dimg_shape, 3);
            if (!s->go_bufs || !s->dw_bufs || !s->dws_bufs || !s->dcol_bufs || !s->colt_bufs || !s->dimg_bufs) {
                /* partial alloc failure: tear down and rebuild from scratch below */
                scratch_destroy(s);
                conv->scratch = s = NULL;
            } else {
                /* wt_contig is allocated lazily by conv2d_forward when need_bwd */
                return s;
            }
        }
        if (s) return s;
    }

    /* signature changed (or first call): destroy old, build new */
    if (s) { scratch_destroy(s); conv->scratch = NULL; }

    s = (struct ax_conv_scratch *)calloc(1, sizeof(struct ax_conv_scratch));
    if (!s) return NULL;
    s->N = N; s->H = H; s->W = W; s->T = T;

    int64_t col_shape[] = {K, M};
    int64_t res_shape[] = {C_out, M};

    s->col_bufs = alloc_buf_array(T, col_shape, 2);
    s->res_bufs = alloc_buf_array(T, res_shape, 2);

    /* w2d is built once from current weights; if weights change between
       forward calls (which happens during training!), we need to refresh
       it. that's done in conv2d_forward itself, not here. */
    int64_t w2d_shape[] = {C_out, K};
    s->w2d = ax_tensor_create(w2d_shape, 2, AX_FLOAT32);

    if (need_backward) {
        int64_t go_shape[] = {C_out, M};
        int64_t dw_shape[] = {C_out, K};
        int64_t dcol_shape[] = {K, M};
        int64_t colt_shape[] = {M, K};
        int64_t dimg_shape[] = {C_in, H, W};
        s->go_bufs   = alloc_buf_array(T, go_shape, 2);
        s->dw_bufs   = alloc_buf_array(T, dw_shape, 2);
        s->dws_bufs  = alloc_buf_array(T, dw_shape, 2);
        s->dcol_bufs = alloc_buf_array(T, dcol_shape, 2);
        s->colt_bufs = alloc_buf_array(T, colt_shape, 2);
        s->dimg_bufs = alloc_buf_array(T, dimg_shape, 3);
    }

    if (!s->col_bufs || !s->res_bufs || !s->w2d ||
        (need_backward && (!s->go_bufs || !s->dw_bufs || !s->dws_bufs || !s->dcol_bufs ||
                            !s->colt_bufs || !s->dimg_bufs))) {
        scratch_destroy(s);
        return NULL;
    }

    /* batch N samples into wide GEMMs when M is narrow. each per-sample GEMM
       [C_out,K]@[K,M] wastes register tiles when M < NC=256. merging n_batch
       samples → [C_out,K]@[K,n_batch*M] fills NC properly. each chunk capped
       at AX_CONV_BATCH_COL_BYTES (file-scope) so it stays in L3.
       1×1 stride=1 pad=0 is excluded — its im2col is just a transpose memcpy
       which costs more than the NC-utilization gain. */
    bool is_direct = can_direct_3x3(kh, kw, sh, sw, ph, pw, C_in);
    bool is_implicit = !is_direct && ax_compute_has_conv_gemm() && prefer_implicit_gemm(K, M);
    bool is_1x1 = is_1x1_pad0_stride1(kh, kw, sh, sw, ph, pw);
    if (!is_direct && !is_implicit && !is_1x1 && N > 1 && M < AX_CONV_BATCH_M_THRESH) {
        int64_t full_bytes = K * M * N * (int64_t)sizeof(float);
        int64_t nb = N;
        if (full_bytes > AX_CONV_BATCH_COL_BYTES) {
            nb = AX_CONV_BATCH_COL_BYTES / (K * M * (int64_t)sizeof(float));
            if (nb > N) nb = N;
        }
        s->n_batch = nb;
        /* nb < 2: batched path degrades to sequential single-sample GEMMs,
           which is worse than the OMP-parallel per-sample path. skip alloc
           so forward/backward fall back naturally. */
        if (nb >= 2) {
            int64_t batch_col_shape[] = {K, nb * M};
            int64_t batch_aux_shape[] = {C_out, nb * M};
            s->batch_col_buf = ax_tensor_create(batch_col_shape, 2, AX_FLOAT32);
            s->batch_aux_buf = ax_tensor_create(batch_aux_shape, 2, AX_FLOAT32);
            /* failure is non-fatal: forward/backward fall back to per-sample path */
        }
    }

    /* winograd F(2,3) scratch: U [16, C_out, C_in], V [16, C_in, num_tiles],
       M [16, C_out, num_tiles]. allocated when the layer matches the
       winograd predicate (3x3 stride=1 pad=1, Cin/Cout >= 32, out >= 4x4).
       failure is non-fatal: forward falls back to im2col+gemm. */
    if (prefer_winograd_f23(kh, kw, sh, sw, ph, pw, N, C_in, C_out, out_h, out_w)) {
        const int64_t Ty = (out_h + 1) / 2;
        const int64_t Tx = (out_w + 1) / 2;
        const int64_t num_tiles = N * Ty * Tx;
        s->wino_num_tiles = num_tiles;
        int64_t U_shape[] = {16, C_out, C_in};
        int64_t V_shape[] = {16, C_in,  num_tiles};
        int64_t M_shape[] = {16, C_out, num_tiles};
        s->wino_U = ax_tensor_create(U_shape, 3, AX_FLOAT32);
        s->wino_V = ax_tensor_create(V_shape, 3, AX_FLOAT32);
        s->wino_M = ax_tensor_create(M_shape, 3, AX_FLOAT32);
    }

    conv->scratch = s;
    return s;
}







/* ────────────────────── NHWC conv2d kernels ──────────────────────
   NHWC conv uses the same im2col + GEMM pipeline as NCHW but with two
   structural advantages:
   1. im2col_nhwc gathers via plain memcpy (Cin contiguous per patch element),
      so stride>1 cases are bandwidth-bound rather than going through the
      NCHW scalar fallback.
   2. weight is cached in [K=Cin*kh*kw, Cout] layout so the GEMM is just
      ax_compute_gemm(im2col [M, K], weight [K, Cout], out [M, Cout])
      where M = N*OH*OW. natural flat-batch GEMM, no per-sample loop.
   ────────────────────────────────────────────────────────────────── */


/* refresh wt_nhwc cache from the source weight (which is [Cout, Cin, kh, kw]).
   transposed layout [K=Cin*kh*kw, Cout] is what NHWC GEMM expects as B.
   layout: wt_nhwc[(ky*kw + kx)*Cin + ci, co] = weight[co, ci, ky, kx]. */
void ax_conv_refresh_wt_nhwc(struct ax_conv_scratch *s, const float *wd_orig,
                             int64_t Cout, int64_t Cin, int kh, int kw,
                             uint64_t src_gen)
{
    if (s->wt_nhwc && s->wt_nhwc_gen == src_gen) return;  /* cache hit */
    int64_t K = (int64_t)Cin * (int64_t)kh * (int64_t)kw;
    if (!s->wt_nhwc) {
        int64_t shape[] = {K, Cout};
        s->wt_nhwc = ax_tensor_create(shape, 2, AX_FLOAT32);
        if (!s->wt_nhwc) return;
    }
    float *dst = (float *)s->wt_nhwc->storage->data;
    /* dst[(ky*kw + kx)*Cin + ci, co] = wd[co, ci, ky, kx]
       == wd[(co * Cin + ci) * kh * kw + ky * kw + kx] */
    int khkw = kh * kw;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int kk = 0; kk < khkw; kk++) {
        for (int64_t ci = 0; ci < Cin; ci++) {
            for (int64_t co = 0; co < Cout; co++) {
                int64_t row = kk * Cin + ci;
                dst[row * Cout + co] = wd_orig[((co * Cin + ci) * khkw) + kk];
            }
        }
    }
    s->wt_nhwc_gen = src_gen;
}

/* NHWC conv2d forward: input [N, H, W, Cin], weight [Cout, Cin, kh, kw],
   output [N, OH, OW, Cout]. fills bias if non-NULL.
   processes samples in chunks sized to keep im2col_nhwc in L3 (8 MB target).
   each chunk: im2col → GEMM → bias-add. avoids the cache-thrashing 230 MB
   monolithic im2col for typical vision shapes. */
int conv2d_nhwc_forward_impl(
    struct ax_conv_scratch *s,
    const float *id, const float *wd, const float *bias,
    float *od, int64_t N, int64_t Cin, int64_t H, int64_t W,
    int64_t Cout, int64_t out_h, int64_t out_w,
    int kh, int kw, int sh, int sw, int ph, int pw,
    uint64_t weight_gen)
{
    int64_t K = (int64_t)Cin * (int64_t)kh * (int64_t)kw;
    int64_t M_per = out_h * out_w;
    int64_t HWC = H * W * Cin;
    int64_t OHOW_C = out_h * out_w * Cout;

    /* per-sample im2col bytes; pick chunk size so chunk_M * K * 4 ≤ ~8 MB. */
    int64_t bytes_per_sample = M_per * K * (int64_t)sizeof(float);
    int64_t n_chunk = (int64_t)(8 * 1024 * 1024) / (bytes_per_sample > 0 ? bytes_per_sample : 1);
    if (n_chunk < 1) n_chunk = 1;
    if (n_chunk > N) n_chunk = N;
    int64_t chunk_M = n_chunk * M_per;

    /* allocate / reuse im2col buffer of shape [chunk_M, K] */
    bool need_alloc = !s->im2col_nhwc
        || s->im2col_nhwc->shape[0] != chunk_M
        || s->im2col_nhwc->shape[1] != K;
    if (need_alloc) {
        if (s->im2col_nhwc) ax_tensor_destroy(s->im2col_nhwc);
        int64_t sh2[] = {chunk_M, K};
        s->im2col_nhwc = ax_tensor_create(sh2, 2, AX_FLOAT32);
        if (!s->im2col_nhwc) return -1;
    }
    float *cd = (float *)s->im2col_nhwc->storage->data;

    /* refresh transposed weight cache (once per call). */
    ax_conv_refresh_wt_nhwc(s, wd, Cout, Cin, kh, kw, weight_gen);
    if (!s->wt_nhwc) return -1;

    /* iterate chunks. */
    for (int64_t n_start = 0; n_start < N; n_start += n_chunk) {
        int64_t n_b = (n_start + n_chunk <= N) ? n_chunk : (N - n_start);
        int64_t M_b = n_b * M_per;

        ax_conv_im2col_nhwc(id + n_start * HWC, n_b, H, W, Cin,
                    kh, kw, sh, sw, ph, pw, out_h, out_w, cd);

        ax_storage_t a_st, b_st, c_st;
        ax_tensor_t  a_tv, b_tv, c_tv;
        make_stack_view(&a_tv, &a_st, cd,                                     M_b,  K);
        make_stack_view(&b_tv, &b_st, (float *)s->wt_nhwc->storage->data,     K,    Cout);
        make_stack_view(&c_tv, &c_st, od + n_start * OHOW_C,                  M_b,  Cout);
        if (ax_compute_gemm(&a_tv, &b_tv, &c_tv) != AX_OK) return -1;

        if (bias) {
            int64_t TC = AX_VF32_WIDTH;
            int64_t cv_end = Cout - (Cout % TC);
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int64_t r = 0; r < M_b; r++) {
                float *row = od + (n_start * M_per + r) * Cout;
                int64_t co = 0;
                for (; co < cv_end; co += TC)
                    ax_vf32_storeu(row + co, ax_vf32_add(ax_vf32_loadu(row + co), ax_vf32_loadu(bias + co)));
                for (; co < Cout; co++) row[co] += bias[co];
            }
        }
    }
    return 0;
}


static ax_tensor_t *conv2d_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_conv2d_t *conv = (ax_conv2d_t *)self;

    if (input->ndim != 4)
    {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "conv2d expects 4D input");
        return NULL;
    }

    /* ensure contiguous so memcpy-based im2col below works correctly */
    ax_tensor_t *inp = ax_ensure_contiguous(input);
    if (!inp) return NULL;

    ax_layout_t layout = ax_tensor_get_layout(input);
    int64_t N, C_in, H, W;
    if (layout == AX_LAYOUT_NHWC) {
        N = inp->shape[0]; H = inp->shape[1]; W = inp->shape[2]; C_in = inp->shape[3];
    } else {
        N = inp->shape[0]; C_in = inp->shape[1]; H = inp->shape[2]; W = inp->shape[3];
    }
    int64_t C_out = conv->out_channels;
    int kh = conv->kernel_h, kw = conv->kernel_w;
    int sh = conv->stride_h, sw = conv->stride_w;
    int ph = conv->pad_h, pw = conv->pad_w;

    int64_t out_h = conv_out_dim(H, kh, sh, ph);
    int64_t out_w = conv_out_dim(W, kw, sw, pw);
    if (out_h <= 0 || out_w <= 0)
    {
        ax_err_set(AX_ERR_INVALID_SHAPE, "conv2d: invalid output dimensions %" PRId64 " x %" PRId64, out_h, out_w);
        return NULL;
    }

    /* reshape weight to [C_out, C_in*kh*kw] for gemm */
    int64_t K = C_in * kh * kw;
    int64_t M = out_h * out_w;
    /* layout-aware output shape. */
    int64_t out_shape_nchw[] = {N, C_out, out_h, out_w};
    int64_t out_shape_nhwc[] = {N, out_h, out_w, C_out};
    const int64_t *out_shape = (layout == AX_LAYOUT_NHWC) ? out_shape_nhwc : out_shape_nchw;
    /* every conv path below writes the full output (direct/smallcin fold bias
       into init; gemm paths copy res→od overwriting all elements). skip the
       zero pass — saves a full output-buffer write (~410 MB on
       conv_32x3x224x224 → 32x64x224x224). */
    ax_tensor_t *output = ax_tensor_create(out_shape, 4, AX_FLOAT32);
    if (!output) { if (inp != input) ax_tensor_destroy(inp); return NULL; }
    output->layout = layout;

    /* CUDA fast path: when tensors live on a non-CPU device the rest of
       this function (path selection + per-tile pack + CPU SIMD bias add)
       would dereference device pointers as host pointers and SEGV. dispatch
       per-sample to the backend's conv_gemm + bias_add, both of which the
       CUDA backend implements. requires NCHW layout; NHWC on CUDA needs a
       separate pass (not yet wired). */
    if (output->storage->device != AX_DEVICE_CPU) {
        if (layout != AX_LAYOUT_NCHW) {
            ax_err_set(AX_ERR_BACKEND, "conv2d on non-cpu device requires NCHW layout");
            ax_tensor_destroy(output);
            if (inp != input) ax_tensor_destroy(inp);
            return NULL;
        }
        if (!ax_compute_has_conv_gemm()) {
            ax_err_set(AX_ERR_BACKEND, "conv2d on non-cpu device requires conv_gemm op");
            ax_tensor_destroy(output);
            if (inp != input) ax_tensor_destroy(inp);
            return NULL;
        }
        /* weight reshape for conv_gemm: [C_out, C_in*kh*kw] (zero-copy view).
           since the output of cuda_conv_gemm writes directly into a per-sample
           [C_out, M] slot, we build a stack-view tensor over the right offset
           in od per batch index. */
        int64_t w_sh[] = {C_out, K};
        ax_tensor_t *w2d = ax_tensor_reshape(conv->weight, w_sh, 2);
        if (!w2d) {
            ax_tensor_destroy(output);
            if (inp != input) ax_tensor_destroy(inp);
            return NULL;
        }
        const float *ind_dev = (const float *)inp->storage->data + inp->offset;
        ax_conv_params_t single = {
            .input = ind_dev,
            .C_in = C_in, .H = H, .W = W,
            .kh = kh, .kw = kw, .sh = sh, .sw = sw,
            .ph = ph, .pw = pw, .out_h = out_h, .out_w = out_w,
        };
        /* k.4: prefer batched cuda_conv_gemm via the cuda extension
           registry. ONE strided-batched cuBLAS GEMM for all N samples
           instead of N separate calls. cu is NULL when the cuda backend
           isn't loaded; falls back to per-sample below. */
        const ax_cuda_extension_t *cu = ax_compute_get_cuda_extension();
        bool used_batched = false;
        if (cu && cu->conv_gemm_batched && N >= 1) {
            ax_tensor_t out_b;
            memset(&out_b, 0, sizeof(out_b));
            out_b.storage = output->storage;
            out_b.dtype = AX_FLOAT32;
            out_b.ndim = 3;
            out_b.shape[0] = N; out_b.shape[1] = C_out; out_b.shape[2] = M;
            out_b.strides[0] = C_out * M; out_b.strides[1] = M; out_b.strides[2] = 1;
            out_b.offset = output->offset;
            /* I.2: opt-in Winograd F(2,3) for 3x3 stride-1 convs. gated
               on AX_CUDA_WINOGRAD=1 to avoid regressing on GPUs where
               my unfused implementation is bandwidth-bound (RTX 3050
               class — neutral to slightly negative on most shapes,
               positive only on a few large ones). datacenter GPUs with
               higher compute/BW ratio likely win more. */
            ax_status_t (*conv_fn)(const ax_tensor_t *, const float *,
                                   int64_t, const ax_conv_params_t *,
                                   ax_tensor_t *) = cu->conv_gemm_batched;
            if (kh == 3 && kw == 3 && sh == 1 && sw == 1
                && cu->conv_winograd_f23 && cu->winograd_f23_prefer
                && cu->winograd_f23_prefer(C_in, C_out, H, W)) {
                static int wg_env_resolved = 0;
                static int wg_enabled = 0;
                if (!wg_env_resolved) {
                    const char *e = getenv("AX_CUDA_WINOGRAD");
                    wg_enabled = (e && e[0] == '1') ? 1 : 0;
                    wg_env_resolved = 1;
                }
                if (wg_enabled) conv_fn = cu->conv_winograd_f23;
            }
            if (conv_fn(w2d, ind_dev, N, &single, &out_b) == AX_OK) {
                used_batched = true;
                if (conv->use_bias && conv->bias) {
                    ax_compute_bias_add(&out_b, conv->bias, 1, &out_b);
                }
            }
        }
        if (!used_batched) {
            for (int64_t n = 0; n < N; n++) {
                ax_tensor_t out_n;
                memset(&out_n, 0, sizeof(out_n));
                out_n.storage = output->storage;
                out_n.dtype = AX_FLOAT32;
                out_n.ndim = 2;
                out_n.shape[0] = C_out; out_n.shape[1] = M;
                out_n.strides[0] = M; out_n.strides[1] = 1;
                out_n.offset = output->offset + n * C_out * M;
                ax_conv_params_t cp = single;
                cp.input = ind_dev + n * C_in * H * W;
                if (ax_compute_conv_gemm(w2d, &cp, &out_n) != AX_OK) {
                    ax_tensor_destroy(w2d);
                    ax_tensor_destroy(output);
                    if (inp != input) ax_tensor_destroy(inp);
                    return NULL;
                }
                if (conv->use_bias && conv->bias) {
                    ax_compute_bias_add(&out_n, conv->bias, 0, &out_n);
                }
            }
        }
        ax_tensor_destroy(w2d);
        if (inp != input) ax_tensor_destroy(inp);
        return output;
    }

    float *od = (float *)output->storage->data;
    float *wd = (float *)conv->weight->storage->data;

    /* lazily allocate (or reuse) per-layer scratch — eliminates per-call malloc */
    bool need_bwd = ax_grad_enabled() && (input->requires_grad || conv->weight->requires_grad);
    struct ax_conv_scratch *s = ensure_scratch(conv, N, H, W, need_bwd);
    if (!s) { if (inp != input) ax_tensor_destroy(inp); ax_tensor_destroy(output); return NULL; }

    /* NHWC fast path: dedicated im2col_nhwc + GEMM with cached transposed weight. */
    if (layout == AX_LAYOUT_NHWC) {
        uint64_t wgen = conv->weight->storage->generation;
        int rc = conv2d_nhwc_forward_impl(s,
            (const float *)inp->storage->data,
            wd, conv->use_bias && conv->bias ? (const float *)conv->bias->storage->data : NULL,
            od, N, C_in, H, W, C_out, out_h, out_w,
            kh, kw, sh, sw, ph, pw, wgen);
        if (rc != 0) { if (inp != input) ax_tensor_destroy(inp); ax_tensor_destroy(output); return NULL; }

        /* hook up backward if needed */
        if (ax_grad_enabled() && (input->requires_grad || conv->weight->requires_grad
                                   || (conv->use_bias && conv->bias && conv->bias->requires_grad))) {
            output->requires_grad = true;
            ax_grad_fn_t *gf = ax_grad_fn_create(ax_conv_conv2d_backward);
            gf->inputs[0] = input;
            gf->n_inputs = 1;
            gf->saved[0] = inp;
            gf->saved_owned[0] = (inp != input);
            gf->n_saved = 1;
            gf->ctx = conv;
            output->grad_fn = gf;
        } else {
            if (inp != input) ax_tensor_destroy(inp);
        }
        return output;
    }

    /* refresh w2d from current weights (they change every training step) */
    ax_tensor_t *w2d = s->w2d;
    memcpy(w2d->storage->data, wd, (size_t)(C_out * K) * sizeof(float));

    /* lazily allocate wt_contig on first backward-enabled call.
       weights change every step but the wt_contig refresh is done in BACKWARD
       (which knows it's about to use it), not in forward — saves wasted work
       on calls that won't need it. */
    if (need_bwd && !s->wt_contig) {
        int64_t wt_shape[] = {K, C_out};
        s->wt_contig = ax_tensor_create(wt_shape, 2, AX_FLOAT32);
    }

    int T = s->T;
    float *ind = (float *)inp->storage->data;
    const float *bias_data = (conv->use_bias && conv->bias)
        ? (const float *)conv->bias->storage->data : NULL;

    /* shape-aware path selection. winograd F(2,3) wins on 3x3 stride=1 pad=1
       with mid-large Cin/Cout (16 muls per 2x2 output tile vs 36 for direct).
       direct 3x3 wins for small kernels (mnist: both convs hit this).
       small-C_in direct beats im2col+gemm on the first layer of image nets
       (C_in=3, K=27 too small for GEMM). implicit gemm wins for large K +
       large M. otherwise fall back to explicit im2col + gemm. */
    bool use_winograd = prefer_winograd_f23(kh, kw, sh, sw, ph, pw, N, C_in, C_out, out_h, out_w)
                        && s->wino_U && s->wino_V && s->wino_M;
    bool use_direct = !use_winograd && can_direct_3x3(kh, kw, sh, sw, ph, pw, C_in);
    /* batch D ATTEMPT (disabled): direct 3x3 stride=2 pad=1 kernel exists
       below as conv2d_direct_3x3_s2_sample but the (ci, ky, kx) outer loop
       order touches the full output buffer Cin*9 times, causing cache
       thrashing on shapes with Cin >= ~32. measured 55 GFLOPS vs 310 for
       implicit-gemm on conv_64x112_128_s2. needs (oh, ow_blk) outer +
       register-accumulator rewrite to be competitive — deferred. */
    bool use_direct_s2 = false;
    (void)can_direct_3x3_s2;
    bool use_smallcin = !use_winograd && !use_direct && !use_direct_s2 && can_direct_smallcin(kh, kw, sh, sw, C_in, out_w);
    bool use_implicit = !use_winograd && !use_direct && !use_direct_s2 && !use_smallcin && ax_compute_has_conv_gemm() && prefer_implicit_gemm(K, M);
    /* 1×1 stride=1 pad=0 routes to per-sample (zero-copy stack tensor),
       not batched (which pays an im2col-as-transpose cost for no GEMM gain). */
    bool is_1x1_fast = is_1x1_pad0_stride1(kh, kw, sh, sw, ph, pw) && (H == out_h) && (W == out_w);
    bool use_batched = !use_winograd && !use_direct && !use_direct_s2 && !use_smallcin && !use_implicit && !is_1x1_fast
                       && N > 1 && M < AX_CONV_BATCH_M_THRESH
                       && s->batch_col_buf && s->batch_aux_buf;

    if (use_winograd) {
        int rc = conv2d_winograd_f23_forward(s, ind, wd, bias_data, od,
                                              N, C_in, H, W, C_out, out_h, out_w, T);
        if (rc != 0) {
            /* shape mismatch (e.g. resized between forward calls) — fall back */
            use_winograd = false;
            use_direct = can_direct_3x3(kh, kw, sh, sw, ph, pw, C_in);
            use_smallcin = !use_direct && can_direct_smallcin(kh, kw, sh, sw, C_in, out_w);
            use_implicit = !use_direct && !use_smallcin && ax_compute_has_conv_gemm() && prefer_implicit_gemm(K, M);
            use_batched = !use_direct && !use_smallcin && !use_implicit && !is_1x1_fast
                          && N > 1 && M < AX_CONV_BATCH_M_THRESH
                          && s->batch_col_buf && s->batch_aux_buf;
        }
    }

    if (use_winograd) {
        /* output already filled by conv2d_winograd_f23_forward; skip the
           im2col+gemm + scatter paths below. */
    } else if (use_batched) {
        /* sub-batched: loop over n_batch-sample chunks, each fitting ≤8 MB.
           im2col → wide GEMM → scatter-with-bias per chunk.
           for the common case (N ≤ n_batch, no splitting), this is a
           single iteration identical to the original whole-N path. */
        float *bcd = (float *)s->batch_col_buf->storage->data;
        float *brd = (float *)s->batch_aux_buf->storage->data;
        int64_t nb = s->n_batch;

        for (int64_t n_start = 0; n_start < N; n_start += nb) {
            int64_t n_b   = (n_start + nb <= N) ? nb : (N - n_start);
            int64_t NM_b  = n_b * M;
            int threads_b = (int)(n_b < T ? n_b : T);

            #ifdef _OPENMP
            #pragma omp parallel for num_threads(threads_b) schedule(static)
            #endif
            for (int64_t n = 0; n < n_b; n++)
                im2col_into_strided(ind + (n_start + n) * C_in * H * W, C_in, H, W,
                                     kh, kw, sh, sw, ph, pw, out_h, out_w,
                                     bcd, NM_b, n * M);

            /* stack tensor views sized to this chunk; opt_gemm zero-fills brd. */
            ax_storage_t col_st, aux_st;
            ax_tensor_t  col_tv, aux_tv;
            make_stack_view(&col_tv, &col_st, bcd,  K,     NM_b);
            make_stack_view(&aux_tv, &aux_st, brd,  C_out, NM_b);
            ax_compute_gemm(w2d, &col_tv, &aux_tv);

            /* scatter brd[C_out, n_b, M] → od[n_start+n, C_out, M] with bias.
               co-outer keeps brd access contiguous per channel. */
            #ifdef _OPENMP
            #pragma omp parallel for num_threads(T) schedule(static)
            #endif
            for (int64_t co = 0; co < C_out; co++) {
                float bias_val = bias_data ? bias_data[co] : 0.0f;
                ax_vf32 vb = ax_vf32_set1(bias_val);
                const float *src_co = brd + co * NM_b;
                for (int64_t n = 0; n < n_b; n++) {
                    float *dst = od + ((n_start + n) * C_out + co) * M;
                    const float *src = src_co + n * M;
                    int64_t m = 0, ve = M - (M % AX_VF32_WIDTH);
                    for (; m < ve; m += AX_VF32_WIDTH)
                        ax_vf32_storeu(dst + m, ax_vf32_add(ax_vf32_loadu(src + m), vb));
                    for (; m < M; m++) dst[m] = src[m] + bias_val;
                }
            }
        }
    } else {
    /* num_threads(T) caps the team to the number of per-thread scratch slots
       so workers never share col_bufs/res_bufs[0] with the tid>=T fallback. */
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(T) schedule(dynamic, 1)
    #endif
    for (int64_t n = 0; n < N; n++)
    {
        if (use_direct) {
            /* writes directly into output[n], bias folded in. no scratch needed. */
            conv2d_direct_3x3_sample(
                ind + n * C_in * H * W, wd, bias_data,
                od + n * C_out * M, C_in, C_out, H, W);
            continue;
        }
        if (use_direct_s2) {
            /* batch D: 3x3 stride=2 pad=1 direct kernel — bypasses im2col */
            conv2d_direct_3x3_s2_sample(
                ind + n * C_in * H * W, wd, bias_data,
                od + n * C_out * M, C_in, C_out, H, W, out_h, out_w);
            continue;
        }
        if (use_smallcin) {
            /* first-layer-style conv (C_in ≤ 4): direct loop with
               register accumulator beats im2col+gemm because K_gemm = C_in*kh*kw
               is too small to amortize pack overhead. bias folded in. */
            conv2d_direct_smallcin_sample(
                ind + n * C_in * H * W, wd, bias_data,
                od + n * C_out * M,
                C_in, C_out, H, W,
                kh, kw, sh, sw, ph, pw, out_h, out_w);
            continue;
        }

        int tid = AX_OMP_THREAD_NUM();
        if (tid >= T) tid = 0; /* defensive */
        ax_tensor_t *col = s->col_bufs[tid];
        ax_tensor_t *res = s->res_bufs[tid];
        float *rd = (float *)res->storage->data;

        if (use_implicit) {
            /* gather im2col patches on-the-fly inside packed b buffers.
               wins on large convs (C_in >= 64 with 3x3+) where the explicit
               im2col copy dominates over the gather overhead. */
            ax_conv_params_t cp = {
                .input = ind + n * C_in * H * W,
                .C_in = C_in, .H = H, .W = W,
                .kh = kh, .kw = kw,
                .sh = sh, .sw = sw,
                .ph = ph, .pw = pw,
                .out_h = out_h, .out_w = out_w,
            };
            ax_compute_conv_gemm(w2d, &cp, res);
        } else if (is_1x1_fast) {
            /* 1×1 stride=1 pad=0: im2col is the identity layout. zero-copy
               stack view over the input slice [C_in, M] saves a C_in*H*W
               memcpy per sample (400 kb+ on vgg shapes). opt_gemm zero-fills
               C internally; no explicit memset needed. */
            ax_storage_t in_st;
            ax_tensor_t  in_tv;
            make_stack_view(&in_tv, &in_st, ind + n * C_in * H * W, C_in, M);
            ax_compute_gemm(w2d, &in_tv, res);
        } else {
            float *cd = (float *)col->storage->data;
            /* explicit im2col + dispatch gemm. fastest for medium K. */
            im2col_into(ind + n * C_in * H * W, C_in, H, W,
                         kh, kw, sh, sw, ph, pw, out_h, out_w, cd);
            /* opt_gemm zero-fills C internally; skip our memset. */
            ax_compute_gemm(w2d, col, res);
        }

        /* copy to output with bias — disjoint output region per n.
           SIMD broadcast-add of bias along the M=out_h*out_w axis. */
        for (int64_t co = 0; co < C_out; co++)
        {
            float bias_val = bias_data ? bias_data[co] : 0.0f;
            float *dst = od + (n * C_out + co) * M;
            const float *src = rd + co * M;
            ax_vf32 vb = ax_vf32_set1(bias_val);
            int64_t m = 0, ve = M - (M % AX_VF32_WIDTH);
            for (; m < ve; m += AX_VF32_WIDTH)
                ax_vf32_storeu(dst + m, ax_vf32_add(ax_vf32_loadu(src + m), vb));
            for (; m < M; m++)
                dst[m] = src[m] + bias_val;
        }
    }
    } /* end use_batched else */

    /* hook up backward */
    if (ax_grad_enabled() && (input->requires_grad || conv->weight->requires_grad))
    {
        output->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(ax_conv_conv2d_backward);
        gf->inputs[0] = input;           /* original — for grad routing */
        gf->n_inputs = 1;
        gf->saved[0] = inp;              /* contiguous — for data access in backward */
        gf->saved_owned[0] = (inp != input);
        gf->n_saved = 1;
        gf->ctx = conv;
        output->grad_fn = gf;
    }
    else
    {
        if (inp != input) ax_tensor_destroy(inp);
    }

    return output;
}

static void conv2d_destroy(ax_layer_t *self)
{
    ax_conv2d_t *c = (ax_conv2d_t *)self;
    if (c->scratch) scratch_destroy(c->scratch);
    if (c->weight) ax_tensor_destroy(c->weight);
    if (c->bias) ax_tensor_destroy(c->bias);
    free(c);
}

ax_layer_t *ax_conv2d_create(int in_ch, int out_ch, int kernel_size,
                              int stride, int padding, bool use_bias)
{
    return ax_conv2d_create_ex(in_ch, out_ch, kernel_size, kernel_size,
                                stride, stride, padding, padding, use_bias);
}

ax_layer_t *ax_conv2d_create_ex(int in_ch, int out_ch,
                                 int kh, int kw, int sh, int sw,
                                 int ph, int pw, bool use_bias)
{
    /* validate parameters before allocating anything */
    if (in_ch <= 0 || out_ch <= 0) {
        ax_err_set(AX_ERR_INVALID_SHAPE, "conv2d: channels must be positive (got in=%d out=%d)", in_ch, out_ch);
        return NULL;
    }
    if (kh <= 0 || kw <= 0) {
        ax_err_set(AX_ERR_INVALID_SHAPE, "conv2d: kernel size must be positive (got %dx%d)", kh, kw);
        return NULL;
    }
    if (sh <= 0 || sw <= 0) {
        ax_err_set(AX_ERR_INVALID_SHAPE, "conv2d: stride must be positive (got %dx%d)", sh, sw);
        return NULL;
    }
    if (ph < 0 || pw < 0) {
        ax_err_set(AX_ERR_INVALID_SHAPE, "conv2d: padding must be non-negative (got %dx%d)", ph, pw);
        return NULL;
    }

    ax_conv2d_t *c = calloc(1, sizeof(ax_conv2d_t));
    AX_RETURN_NULL_IF_ALLOC_FAIL(c, "ax_conv2d_create");

    c->base.ops.forward = conv2d_forward;
    c->base.ops.destroy = conv2d_destroy;
    c->base.type = AX_LAYER_CONV2D;
    c->base.training = true;
    c->base.input_features = in_ch;
    c->base.output_features = out_ch;
    c->in_channels = in_ch;
    c->out_channels = out_ch;
    c->kernel_h = kh; c->kernel_w = kw;
    c->stride_h = sh; c->stride_w = sw;
    c->pad_h = ph; c->pad_w = pw;
    c->use_bias = use_bias;

    /* weight: [out_ch, in_ch, kh, kw] */
    int64_t w_shape[] = {out_ch, in_ch, kh, kw};
    c->weight = ax_tensor_create(w_shape, 4, AX_FLOAT32);
    if (!c->weight) { free(c); return NULL; }
    ax_init_kaiming_uniform(c->weight, in_ch * kh * kw);
    c->weight->requires_grad = true;

    c->base.params[0] = c->weight;
    c->base.n_params = 1;

    if (use_bias)
    {
        int64_t b_shape[] = {out_ch};
        c->bias = ax_tensor_zeros(b_shape, 1, AX_FLOAT32);
        c->bias->requires_grad = true;
        c->base.params[1] = c->bias;
        c->base.n_params = 2;
    }

    return (ax_layer_t *)c;
}



