/* conv.c — convolution, pooling, and flatten implementations.

   the big idea behind im2col:
   a convolution is mathematically a bunch of dot products between
   the kernel and overlapping patches of the input. im2col extracts
   all those patches and lays them out as columns of a matrix.
   then the whole convolution is one matrix multiply:

     output = weight_matrix @ im2col(input)

   this trades memory (the columns matrix is big) for speed
   (gemm is highly optimized everywhere). it's how caffe,
   pytorch cpu, and most other frameworks do it. */

#include "axiom/conv.h"
#include "axiom/ops.h"
#include "axiom/autograd.h"
#include "axiom/init.h"
#include "axiom/compute.h"
#include "axiom/error.h"
#include "../compute/backends/simd_defs.h"
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

/* overflow-checked multiply. returns -1 on overflow. */
static inline int64_t safe_mul(int64_t a, int64_t b)
{
    if (a <= 0 || b <= 0) return (a == 0 || b == 0) ? 0 : -1;
    if (a > INT64_MAX / b) return -1;
    return a * b;
}

/* compute output spatial dimension.
   returns -1 if the result would be non-positive (invalid configuration). */
static inline int64_t conv_out_dim(int64_t in_dim, int kernel, int stride, int pad)
{
    if (stride <= 0) return -1;
    int64_t out = (in_dim + 2 * pad - kernel) / stride + 1;
    if (out <= 0) return -1;
    return out;
}


/* forward declarations for predicates used by ensure_scratch */
static inline bool can_direct_3x3(int kh, int kw, int sh, int sw, int ph, int pw, int64_t C_in);
static inline bool prefer_implicit_gemm(int64_t K, int64_t M);

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
    ax_tensor_t *batch_col_buf; /* [K, N*M]: im2col in fwd; dcol_batch in bwd */
    ax_tensor_t *batch_aux_buf; /* [C_out, N*M]: gemm result in fwd; go_batch in bwd */
};

static void scratch_destroy(struct ax_conv_scratch *s)
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

/* lazily allocate or reallocate scratch buffers if signature differs.
   call this from conv2d_forward/backward before any parallel region
   (single-threaded init guarantees no race on the scratch pointer). */
static struct ax_conv_scratch *ensure_scratch(ax_conv2d_t *conv,
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

/* batch entire N samples into a single GEMM when M is narrow.
   each per-sample GEMM [C_out,K]@[K,M] uses only M/NC columns,
   wasting register tiles. merging → [C_out,K]@[K,N*M] fills NC. */
#define AX_CONV_BATCH_M_THRESH 512
    bool is_direct = can_direct_3x3(kh, kw, sh, sw, ph, pw, C_in);
    bool is_implicit = !is_direct && ax_compute_has_conv_gemm() && prefer_implicit_gemm(K, M);
    if (!is_direct && !is_implicit && N > 1 && M < AX_CONV_BATCH_M_THRESH) {
        int64_t NM = N * M;
        int64_t batch_col_shape[] = {K, NM};
        int64_t batch_aux_shape[] = {C_out, NM};
        s->batch_col_buf = ax_tensor_create(batch_col_shape, 2, AX_FLOAT32);
        s->batch_aux_buf = ax_tensor_create(batch_aux_shape, 2, AX_FLOAT32);
        /* failure is non-fatal: forward/backward fall back to per-sample path */
    }

    conv->scratch = s;
    return s;
}


/* im2col: for a single image [C, H, W], produce a matrix
   [C*kh*kw, out_h*out_w] where each column is a flattened patch */

/* internal: write im2col into a pre-allocated float buffer.
   input must be contiguous [C, H, W]. cd must be at least K*M floats. */
static void im2col_into(const float *id, int64_t C, int64_t H, int64_t W,
                         int kh, int kw, int stride_h, int stride_w,
                         int pad_h, int pad_w, int64_t out_h, int64_t out_w,
                         float *cd)
{
    const int64_t M = out_h * out_w;
    int64_t col_idx = 0;

    /* 1x1 stride-1 pad-0 conv: im2col is the identity layout. col_buf
       [C, M] matches the input [C, H, W] flattened exactly. one big
       memcpy instead of C*H separate row-memcpys. */
    if (kh == 1 && kw == 1 && stride_h == 1 && stride_w == 1
        && pad_h == 0 && pad_w == 0
        && H == out_h && W == out_w)
    {
        memcpy(cd, id, (size_t)(C * H * W) * sizeof(float));
        return;
    }

    /* fast path: stride_w == 1 lets each oh row be a contiguous memcpy
       of the input row, with memset zeroing the pad regions. */
    if (stride_w == 1)
    {
        for (int64_t c = 0; c < C; c++)
        {
            for (int ky = 0; ky < kh; ky++)
            {
                for (int kx = 0; kx < kw; kx++)
                {
                    /* iw as a function of ow: iw = ow + iw_start */
                    const int64_t iw_start = (int64_t)kx - (int64_t)pad_w;

                    /* valid ow range where iw in [0, W) */
                    int64_t ow_lo = 0;
                    if (iw_start < 0) ow_lo = -iw_start;
                    int64_t ow_hi = out_w;
                    const int64_t iw_last = iw_start + out_w - 1;
                    if (iw_last >= W) ow_hi = W - iw_start;
                    if (ow_lo > out_w) ow_lo = out_w;
                    if (ow_hi < 0) ow_hi = 0;
                    if (ow_lo > ow_hi) ow_lo = ow_hi;

                    const int64_t lead_bytes = ow_lo * (int64_t)sizeof(float);
                    const int64_t mid_len = ow_hi - ow_lo;
                    const int64_t mid_bytes = mid_len * (int64_t)sizeof(float);
                    const int64_t tail_n = out_w - ow_hi;
                    const int64_t tail_bytes = tail_n * (int64_t)sizeof(float);

                    for (int64_t oh = 0; oh < out_h; oh++)
                    {
                        const int64_t ih = oh * stride_h - pad_h + ky;
                        float *dst_row = cd + col_idx * M + oh * out_w;

                        if (ih < 0 || ih >= H)
                        {
                            /* whole row is pad */
                            memset(dst_row, 0, (size_t)(out_w * (int64_t)sizeof(float)));
                            continue;
                        }

                        const int64_t src_row_base = c * H * W + ih * W;

                        /* leading pad zone */
                        if (lead_bytes > 0)
                            memset(dst_row, 0, (size_t)lead_bytes);

                        /* valid middle zone */
                        if (mid_len > 0)
                        {
                            memcpy(dst_row + ow_lo,
                                   id + src_row_base + (iw_start + ow_lo),
                                   (size_t)mid_bytes);
                        }

                        /* trailing pad zone */
                        if (tail_bytes > 0)
                            memset(dst_row + ow_hi, 0, (size_t)tail_bytes);
                    }
                    col_idx++;
                }
            }
        }
        return;
    }

    /* fallback: fully general scalar gather for strided width. */
    for (int64_t c = 0; c < C; c++)
    {
        for (int ky = 0; ky < kh; ky++)
        {
            for (int kx = 0; kx < kw; kx++)
            {
                for (int64_t oh = 0; oh < out_h; oh++)
                {
                    for (int64_t ow = 0; ow < out_w; ow++)
                    {
                        int64_t ih = oh * stride_h - pad_h + ky;
                        int64_t iw = ow * stride_w - pad_w + kx;

                        float val = 0.0f;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            val = id[c * H * W + ih * W + iw];

                        cd[col_idx * M + oh * out_w + ow] = val;
                    }
                }
                col_idx++;
            }
        }
    }
}

/* like im2col_into but writes into a column-strided batch buffer.
   col_stride is the distance between consecutive columns in `cd` (normally M;
   for batched buffers: N*M). sample_offset is the column start for this sample
   within each row (0 for single; n*M for batched sample n). */
static void im2col_into_strided(const float *id, int64_t C, int64_t H, int64_t W,
                                 int kh, int kw, int stride_h, int stride_w,
                                 int pad_h, int pad_w, int64_t out_h, int64_t out_w,
                                 float *cd, int64_t col_stride, int64_t sample_offset)
{
    const int64_t M = out_h * out_w;
    int64_t col_idx = 0;

    /* 1x1 stride-1 pad-0: each kernel row is one channel's pixels */
    if (kh == 1 && kw == 1 && stride_h == 1 && stride_w == 1
        && pad_h == 0 && pad_w == 0 && H == out_h && W == out_w)
    {
        for (int64_t c = 0; c < C; c++)
            memcpy(cd + c * col_stride + sample_offset, id + c * M, (size_t)M * sizeof(float));
        return;
    }

    if (stride_w == 1)
    {
        for (int64_t c = 0; c < C; c++)
        {
            for (int ky = 0; ky < kh; ky++)
            {
                for (int kx = 0; kx < kw; kx++)
                {
                    const int64_t iw_start = (int64_t)kx - (int64_t)pad_w;
                    int64_t ow_lo = 0;
                    if (iw_start < 0) ow_lo = -iw_start;
                    int64_t ow_hi = out_w;
                    const int64_t iw_last = iw_start + out_w - 1;
                    if (iw_last >= W) ow_hi = W - iw_start;
                    if (ow_lo > out_w) ow_lo = out_w;
                    if (ow_hi < 0) ow_hi = 0;
                    if (ow_lo > ow_hi) ow_lo = ow_hi;
                    const int64_t lead_bytes = ow_lo * (int64_t)sizeof(float);
                    const int64_t mid_len = ow_hi - ow_lo;
                    const int64_t mid_bytes = mid_len * (int64_t)sizeof(float);
                    const int64_t tail_n = out_w - ow_hi;
                    const int64_t tail_bytes = tail_n * (int64_t)sizeof(float);

                    for (int64_t oh = 0; oh < out_h; oh++)
                    {
                        const int64_t ih = oh * stride_h - pad_h + ky;
                        float *dst_row = cd + col_idx * col_stride + sample_offset + oh * out_w;
                        if (ih < 0 || ih >= H) {
                            memset(dst_row, 0, (size_t)out_w * sizeof(float));
                            continue;
                        }
                        const int64_t src_row_base = c * H * W + ih * W;
                        if (lead_bytes > 0)  memset(dst_row, 0, (size_t)lead_bytes);
                        if (mid_len > 0)
                            memcpy(dst_row + ow_lo, id + src_row_base + (iw_start + ow_lo),
                                   (size_t)mid_bytes);
                        if (tail_bytes > 0) memset(dst_row + ow_hi, 0, (size_t)tail_bytes);
                    }
                    col_idx++;
                }
            }
        }
        return;
    }

    for (int64_t c = 0; c < C; c++)
        for (int ky = 0; ky < kh; ky++)
            for (int kx = 0; kx < kw; kx++) {
                for (int64_t oh = 0; oh < out_h; oh++)
                    for (int64_t ow = 0; ow < out_w; ow++) {
                        int64_t ih = oh * stride_h - pad_h + ky;
                        int64_t iw = ow * stride_w - pad_w + kx;
                        float val = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                    ? id[c * H * W + ih * W + iw] : 0.0f;
                        cd[col_idx * col_stride + sample_offset + oh * out_w + ow] = val;
                    }
                col_idx++;
            }
}

ax_tensor_t *ax_im2col(ax_tensor_t *input, int kh, int kw,
                        int stride_h, int stride_w,
                        int pad_h, int pad_w)
{
    if (input->ndim != 3)
    {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "im2col expects [C, H, W] input");
        return NULL;
    }

    int64_t C = input->shape[0];
    int64_t H = input->shape[1];
    int64_t W = input->shape[2];
    int64_t out_h = conv_out_dim(H, kh, stride_h, pad_h);
    int64_t out_w = conv_out_dim(W, kw, stride_w, pad_w);
    if (out_h <= 0 || out_w <= 0)
    {
        ax_err_set(AX_ERR_INVALID_SHAPE, "im2col: invalid output dimensions %" PRId64 " x %" PRId64, out_h, out_w);
        return NULL;
    }

    int64_t col_shape[] = {C * kh * kw, out_h * out_w};
    ax_tensor_t *cols = ax_tensor_create(col_shape, 2, AX_FLOAT32);
    if (!cols) return NULL;

    float *id = (float *)input->storage->data + input->offset;
    float *cd = (float *)cols->storage->data;

    im2col_into(id, C, H, W, kh, kw, stride_h, stride_w, pad_h, pad_w, out_h, out_w, cd);
    return cols;
}

/* internal: scatter col2im into a pre-zeroed float buffer.
   id must be zeroed before calling (accumulates into it). */
static void col2im_into(const float *cd, int64_t channels,
                         int64_t height, int64_t width,
                         int kh, int kw, int stride_h, int stride_w,
                         int pad_h, int pad_w, int64_t out_h, int64_t out_w,
                         float *id)
{
    int64_t col_idx = 0;
    const int64_t M = out_h * out_w;
    for (int64_t c = 0; c < channels; c++)
    {
        for (int ky = 0; ky < kh; ky++)
        {
            for (int kx = 0; kx < kw; kx++)
            {
                /* precompute valid oh range: ih = oh*sh - ph + ky in [0,height) */
                int64_t oh_lo = 0, oh_hi = out_h;
                {
                    int64_t num_lo = pad_h - ky;  /* oh*sh >= num_lo */
                    int64_t num_hi = height + pad_h - ky;  /* oh*sh < num_hi */
                    if (num_lo > 0) {
                        oh_lo = (num_lo + stride_h - 1) / stride_h;
                    }
                    if (num_hi <= 0) { oh_hi = 0; }
                    else {
                        int64_t hi_cand = (num_hi + stride_h - 1) / stride_h;
                        if (hi_cand < oh_hi) oh_hi = hi_cand;
                    }
                    if (oh_lo > oh_hi) oh_lo = oh_hi;
                }

                /* precompute valid ow range similarly */
                int64_t ow_lo = 0, ow_hi = out_w;
                {
                    int64_t num_lo = pad_w - kx;
                    int64_t num_hi = width + pad_w - kx;
                    if (num_lo > 0) {
                        ow_lo = (num_lo + stride_w - 1) / stride_w;
                    }
                    if (num_hi <= 0) { ow_hi = 0; }
                    else {
                        int64_t hi_cand = (num_hi + stride_w - 1) / stride_w;
                        if (hi_cand < ow_hi) ow_hi = hi_cand;
                    }
                    if (ow_lo > ow_hi) ow_lo = ow_hi;
                }

                const float *col_plane = cd + col_idx * M;
                float *img_plane = id + c * height * width;

                if (stride_w == 1) {
                    /* unit stride in ow means unit stride in both col read and img write.
                       simd-add overlapping segments of the row. */
                    for (int64_t oh = oh_lo; oh < oh_hi; oh++) {
                        int64_t ih = oh * stride_h - pad_h + ky;
                        const float *col_row = col_plane + oh * out_w;
                        float *img_row = img_plane + ih * width + (ow_lo - pad_w + kx);
                        int64_t len = ow_hi - ow_lo;
                        int64_t i = 0;
                        int64_t ve = len - (len % AX_VF32_WIDTH);
                        for (; i < ve; i += AX_VF32_WIDTH) {
                            ax_vf32 a = ax_vf32_loadu(img_row + i);
                            ax_vf32 b = ax_vf32_loadu(col_row + ow_lo + i);
                            ax_vf32_storeu(img_row + i, ax_vf32_add(a, b));
                        }
                        for (; i < len; i++)
                            img_row[i] += col_row[ow_lo + i];
                    }
                } else {
                    /* general strided fallback — skip the bounds check now that ranges are pruned */
                    for (int64_t oh = oh_lo; oh < oh_hi; oh++) {
                        int64_t ih = oh * stride_h - pad_h + ky;
                        const float *col_row = col_plane + oh * out_w;
                        float *img_row = img_plane + ih * width;
                        for (int64_t ow = ow_lo; ow < ow_hi; ow++) {
                            int64_t iw = ow * stride_w - pad_w + kx;
                            img_row[iw] += col_row[ow];
                        }
                    }
                }
                col_idx++;
            }
        }
    }
}

/* col2im: inverse of im2col. allocating version for public API. */
ax_tensor_t *ax_col2im(ax_tensor_t *cols, int64_t channels,
                        int64_t height, int64_t width,
                        int kh, int kw,
                        int stride_h, int stride_w,
                        int pad_h, int pad_w)
{
    int64_t out_h = conv_out_dim(height, kh, stride_h, pad_h);
    int64_t out_w = conv_out_dim(width, kw, stride_w, pad_w);
    if (out_h <= 0 || out_w <= 0)
    {
        ax_err_set(AX_ERR_INVALID_SHAPE, "col2im: invalid output dimensions %" PRId64 " x %" PRId64, out_h, out_w);
        return NULL;
    }

    int64_t img_shape[] = {channels, height, width};
    ax_tensor_t *img = ax_tensor_zeros(img_shape, 3, AX_FLOAT32);
    if (!img) return NULL;

    col2im_into((float *)cols->storage->data, channels, height, width,
                kh, kw, stride_h, stride_w, pad_h, pad_w, out_h, out_w,
                (float *)img->storage->data);
    return img;
}


/* conv2d forward and backward */

static void conv2d_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_conv2d_t *conv = (ax_conv2d_t *)self->ctx;
    ax_tensor_t *input_orig = self->inputs[0]; /* original — for grad accumulation */
    ax_tensor_t *input_data = self->saved[0];  /* contiguous — for data access */
    ax_tensor_t *weight = conv->weight;

    int64_t N = input_data->shape[0];
    int64_t C_in = input_data->shape[1];
    int64_t H = input_data->shape[2];
    int64_t W = input_data->shape[3];
    int64_t C_out = weight->shape[0];
    int kh = conv->kernel_h, kw = conv->kernel_w;
    int sh = conv->stride_h, sw = conv->stride_w;
    int ph = conv->pad_h, pw = conv->pad_w;
    int64_t out_h = conv_out_dim(H, kh, sh, ph);
    int64_t out_w = conv_out_dim(W, kw, sw, pw);

    /* grad_out: [N, C_out, out_h, out_w] */

    /* weight gradient and input gradient */
    if (weight->requires_grad)
    {
        if (!weight->grad)
            weight->grad = ax_tensor_zeros(weight->shape, weight->ndim, weight->dtype);
    }
    if (input_orig->requires_grad)
    {
        if (!input_orig->grad)
            input_orig->grad = ax_tensor_zeros(input_orig->shape, input_orig->ndim, input_orig->dtype);
    }

    /* bias gradient: sum grad_out over N, H, W.
       parallelize over channels: each channel's M*N sum is independent. */
    if (conv->use_bias && conv->bias && conv->bias->requires_grad)
    {
        if (!conv->bias->grad)
            conv->bias->grad = ax_tensor_zeros(conv->bias->shape, conv->bias->ndim, conv->bias->dtype);

        float *bg = (float *)conv->bias->grad->storage->data;
        const float *gd = (const float *)grad_out->storage->data;
        int64_t M_bias = out_h * out_w;

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t c = 0; c < C_out; c++) {
            float sum = 0.0f;
            for (int64_t n = 0; n < N; n++) {
                const float *row = gd + (n * C_out + c) * M_bias;
                int64_t m = 0, ve = M_bias - (M_bias % AX_VF32_WIDTH);
                ax_vf32 acc = ax_vf32_zero();
                for (; m < ve; m += AX_VF32_WIDTH)
                    acc = ax_vf32_add(acc, ax_vf32_loadu(row + m));
                /* horizontal sum of acc */
                float tmp[AX_VF32_WIDTH];
                ax_vf32_storeu(tmp, acc);
                for (int k = 0; k < AX_VF32_WIDTH; k++) sum += tmp[k];
                for (; m < M_bias; m++) sum += row[m];
            }
            bg[c] += sum; /* one thread per c, no race */
        }
    }

    float *wdata = (float *)weight->storage->data;
    int64_t K = C_in * kh * kw;
    int64_t M = out_h * out_w;

    /* use cached scratch from layer struct (allocated in forward via ensure_scratch).
       if scratch is missing for some reason, allocate fresh as fallback. */
    struct ax_conv_scratch *s = conv->scratch;
    if (!s || s->N != N || s->H != H || s->W != W || !s->dw_bufs) {
        /* scratch missing or stale — call ensure_scratch to (re)build */
        s = ensure_scratch(conv, N, H, W, true);
        if (!s) return;
    }

    int T = s->T;
    ax_tensor_t *wt_contig = s->wt_contig;
    /* when gemm_tn is available we skip the physical weight transpose —
       gemm_tn walks w2d as if transposed. keep the legacy refresh only
       for the fallback path so backends without gemm_tn still work. */
    bool have_gemm_tn = ax_compute_has_gemm_tn();
    bool have_gemm_nt = ax_compute_has_gemm_nt();
    if (input_orig->requires_grad && wt_contig && !have_gemm_tn) {
        float *wtd = (float *)wt_contig->storage->data;
        for (int64_t r = 0; r < C_out; r++)
            for (int64_t c = 0; c < K; c++)
                wtd[c * C_out + r] = wdata[r * K + c];
    }
    /* w2d for gemm_tn fast path: fresh copy of current weights in
       [C_out, K] shape. s->w2d is refreshed by forward, but forward
       may have been called long enough ago that weights changed; do
       it here too for safety. */
    ax_tensor_t *w2d = s->w2d;
    if (input_orig->requires_grad && have_gemm_tn && w2d) {
        memcpy(w2d->storage->data, wdata, (size_t)(C_out * K) * sizeof(float));
    }

    /* zero per-thread dW buffers (we accumulate into them across samples) */
    if (weight->requires_grad) {
        for (int t = 0; t < T; t++)
            memset(s->dw_bufs[t]->storage->data, 0, (size_t)(C_out * K) * sizeof(float));
    }

    float *ind = (float *)input_data->storage->data;
    float *grd = (float *)grad_out->storage->data;

    bool use_batched_bwd = N > 1 && M < AX_CONV_BATCH_M_THRESH
                           && s->batch_col_buf && s->batch_aux_buf;

    if (use_batched_bwd) {
        /* batched backward: one large gemm_nt for dW, one large gemm_tn for dX.
           eliminates N separate launches and fills NC tiles properly. */
        float *bcd = (float *)s->batch_col_buf->storage->data;
        float *gbd = (float *)s->batch_aux_buf->storage->data;
        int64_t NM = N * M;

        /* parallel im2col → batch_col_buf [K, N*M]
           parallel go copy → batch_aux_buf [C_out, N*M] */
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(T) schedule(static)
        #endif
        for (int64_t n = 0; n < N; n++) {
            im2col_into_strided(ind + n * C_in * H * W, C_in, H, W,
                                 kh, kw, sh, sw, ph, pw, out_h, out_w,
                                 bcd, NM, n * M);
            /* scatter go[n, co, *] → gbd[co, n*M : (n+1)*M] */
            for (int64_t co = 0; co < C_out; co++)
                memcpy(gbd + co * NM + n * M, grd + n * C_out * M + co * M,
                       (size_t)M * sizeof(float));
        }

        /* dW = go_batch @ col_batch^T via gemm_nt: [C_out, N*M] @ [K, N*M]^T → [C_out, K] */
        if (weight->requires_grad) {
            ax_tensor_t *dw_total = s->dw_bufs[0];
            memset(dw_total->storage->data, 0, (size_t)(C_out * K) * sizeof(float));
            ax_compute_gemm_nt(s->batch_aux_buf, s->batch_col_buf, dw_total);
            float *wg  = (float *)weight->grad->storage->data;
            float *dwl = (float *)dw_total->storage->data;
            int64_t wn = C_out * K, wi = 0, wve = wn - (wn % AX_VF32_WIDTH);
            for (; wi < wve; wi += AX_VF32_WIDTH)
                ax_vf32_storeu(wg + wi, ax_vf32_add(ax_vf32_loadu(wg + wi), ax_vf32_loadu(dwl + wi)));
            for (; wi < wn; wi++) wg[wi] += dwl[wi];
        }

        /* dcol_batch = w^T @ go_batch via gemm_tn: [K, C_out] @ [C_out, N*M] → [K, N*M]
           reuse batch_col_buf as output (im2col data already consumed above). */
        if (input_orig->requires_grad && (wt_contig || have_gemm_tn)) {
            memset(bcd, 0, (size_t)(K * NM) * sizeof(float));
            if (have_gemm_tn)
                ax_compute_gemm_tn(w2d, s->batch_aux_buf, s->batch_col_buf);
            else
                ax_compute_gemm(wt_contig, s->batch_aux_buf, s->batch_col_buf);

            /* parallel col2im per sample from dcol_batch [K, N*M] */
            #ifdef _OPENMP
            #pragma omp parallel for num_threads(T) schedule(static)
            #endif
            for (int64_t n = 0; n < N; n++) {
                int tid = AX_OMP_THREAD_NUM();
                if (tid >= T) tid = 0;
                ax_tensor_t *dcol_buf = s->dcol_bufs[tid];
                ax_tensor_t *dimg_buf = s->dimg_bufs[tid];
                float *dcol_d = (float *)dcol_buf->storage->data;
                /* gather dcol for sample n: dcol[k, m] = bcd[k*NM + n*M + m] */
                for (int64_t k = 0; k < K; k++)
                    memcpy(dcol_d + k * M, bcd + k * NM + n * M, (size_t)M * sizeof(float));
                float *dimg_d = (float *)dimg_buf->storage->data;
                memset(dimg_d, 0, (size_t)(C_in * H * W) * sizeof(float));
                col2im_into(dcol_d, C_in, H, W, kh, kw, sh, sw, ph, pw, out_h, out_w, dimg_d);
                float *ig = (float *)input_orig->grad->storage->data + n * C_in * H * W;
                int64_t total = C_in * H * W, i = 0, ve = total - (total % AX_VF32_WIDTH);
                for (; i < ve; i += AX_VF32_WIDTH)
                    ax_vf32_storeu(ig + i, ax_vf32_add(ax_vf32_loadu(ig + i), ax_vf32_loadu(dimg_d + i)));
                for (; i < total; i++) ig[i] += dimg_d[i];
            }
        }
    } else {
    /* num_threads(T) prevents oversubscription of per-thread scratch slots
       (col_bufs, go_bufs, dw_bufs, etc. — all sized to T). */
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(T) schedule(dynamic, 1)
    #endif
    for (int64_t n = 0; n < N; n++)
    {
        int tid = AX_OMP_THREAD_NUM();
        if (tid >= T) tid = 0;
        float *cbd = (float *)s->col_bufs[tid]->storage->data;
        ax_tensor_t *col_buf = s->col_bufs[tid];
        ax_tensor_t *go_mat = s->go_bufs[tid];

        /* im2col into per-thread buffer */
        im2col_into(ind + n * C_in * H * W, C_in, H, W,
                     kh, kw, sh, sw, ph, pw, out_h, out_w, cbd);

        /* fill per-thread grad_out matrix */
        memcpy(go_mat->storage->data, grd + n * C_out * M,
               (size_t)(C_out * M) * sizeof(float));

        /* weight gradient: per-thread dW accumulates across this thread's samples */
        if (weight->requires_grad)
        {
            ax_tensor_t *dw_local = s->dw_bufs[tid];
            ax_tensor_t *dw_sample = s->dws_bufs[tid];
            float *dws = (float *)dw_sample->storage->data;
            memset(dws, 0, (size_t)(C_out * K) * sizeof(float));

            if (have_gemm_nt) {
                /* dw = go @ col^T via gemm_nt — skips the 60-line block
                   cache-blocked transpose of col that the legacy path did. */
                ax_compute_gemm_nt(go_mat, col_buf, dw_sample);
            } else {
                ax_tensor_t *colt_buf = s->colt_bufs[tid];
                float *ctd = (float *)colt_buf->storage->data;
                const int64_t BT = 32;
                for (int64_t r0 = 0; r0 < K; r0 += BT) {
                    int64_t r_end = (r0 + BT < K) ? r0 + BT : K;
                    for (int64_t c0 = 0; c0 < M; c0 += BT) {
                        int64_t c_end = (c0 + BT < M) ? c0 + BT : M;
                        for (int64_t r = r0; r < r_end; r++) {
                            const float *src_row = cbd + r * M;
                            int64_t c = c0;
                            for (; c + 4 <= c_end; c += 4) {
                                ctd[(c + 0) * K + r] = src_row[c + 0];
                                ctd[(c + 1) * K + r] = src_row[c + 1];
                                ctd[(c + 2) * K + r] = src_row[c + 2];
                                ctd[(c + 3) * K + r] = src_row[c + 3];
                            }
                            for (; c < c_end; c++)
                                ctd[c * K + r] = src_row[c];
                        }
                    }
                }
                ax_compute_gemm(go_mat, colt_buf, dw_sample);
            }

            float *dwl = (float *)dw_local->storage->data;
            int64_t wn = C_out * K;
            int64_t wi = 0, wve = wn - (wn % AX_VF32_WIDTH);
            for (; wi < wve; wi += AX_VF32_WIDTH)
                ax_vf32_storeu(dwl + wi, ax_vf32_add(ax_vf32_loadu(dwl + wi), ax_vf32_loadu(dws + wi)));
            for (; wi < wn; wi++) dwl[wi] += dws[wi];
        }

        /* input gradient — disjoint per-sample writes, no reduction needed */
        if (input_orig->requires_grad && (wt_contig || have_gemm_tn))
        {
            ax_tensor_t *dcol_buf = s->dcol_bufs[tid];
            ax_tensor_t *dimg_buf = s->dimg_bufs[tid];

            memset(dcol_buf->storage->data, 0, (size_t)(K * M) * sizeof(float));
            if (have_gemm_tn) {
                /* dcol = w^T @ go via gemm_tn — skips the wt_contig scalar
                   transpose refresh done above in the legacy path. */
                ax_compute_gemm_tn(w2d, go_mat, dcol_buf);
            } else {
                ax_compute_gemm(wt_contig, go_mat, dcol_buf);
            }

            float *dimg_d = (float *)dimg_buf->storage->data;
            memset(dimg_d, 0, (size_t)(C_in * H * W) * sizeof(float));
            col2im_into((float *)dcol_buf->storage->data, C_in, H, W,
                         kh, kw, sh, sw, ph, pw, out_h, out_w, dimg_d);

            /* SIMD accumulate into input gradient (disjoint per n) */
            float *ig = (float *)input_orig->grad->storage->data + n * C_in * H * W;
            int64_t total = C_in * H * W;
            int64_t i = 0, ve = total - (total % AX_VF32_WIDTH);
            for (; i < ve; i += AX_VF32_WIDTH)
                ax_vf32_storeu(ig + i, ax_vf32_add(ax_vf32_loadu(ig + i), ax_vf32_loadu(dimg_d + i)));
            for (; i < total; i++)
                ig[i] += dimg_d[i];
        }
    }

    /* serial reduction: sum all per-thread dW buffers into weight->grad */
    if (weight->requires_grad) {
        float *wg = (float *)weight->grad->storage->data;
        int64_t wn = C_out * K;
        for (int t = 0; t < T; t++) {
            float *dwl = (float *)s->dw_bufs[t]->storage->data;
            int64_t wi = 0, wve = wn - (wn % AX_VF32_WIDTH);
            for (; wi < wve; wi += AX_VF32_WIDTH)
                ax_vf32_storeu(wg + wi, ax_vf32_add(ax_vf32_loadu(wg + wi), ax_vf32_loadu(dwl + wi)));
            for (; wi < wn; wi++) wg[wi] += dwl[wi];
        }
    }
    } /* end use_batched_bwd else */
    /* note: scratch buffers are kept on the layer for reuse — no free here */
}

/* direct 3x3 stride=1 pad=1 conv for a single sample.
   skips the gemm/im2col detour when the kernel is small enough to stay
   hot in L1. output shape [C_out, H, W] (pad=1 keeps spatial size).
   bias is folded into the row init. */
static void conv2d_direct_3x3_sample(
    const float *in_n,
    const float *wd,
    const float *bias,
    float *out_n,
    int64_t C_in, int64_t C_out,
    int64_t H, int64_t W)
{
    const int64_t HW = H * W;
    const int64_t K9 = C_in * 9;

    for (int64_t co = 0; co < C_out; co++) {
        float bias_val = bias ? bias[co] : 0.0f;
        ax_vf32 vb = ax_vf32_set1(bias_val);
        const float *wco = wd + co * K9;
        float *out_co = out_n + co * HW;

        for (int64_t y = 0; y < H; y++) {
            float *out_row = out_co + y * W;
            int64_t xi = 0, vend_init = W - (W % AX_VF32_WIDTH);
            for (; xi < vend_init; xi += AX_VF32_WIDTH)
                ax_vf32_storeu(out_row + xi, vb);
            for (; xi < W; xi++) out_row[xi] = bias_val;

            for (int64_t ci = 0; ci < C_in; ci++) {
                const float *win = wco + ci * 9;
                const float *in_ci = in_n + ci * HW;

                for (int ky = 0; ky < 3; ky++) {
                    int64_t in_y = y + ky - 1;
                    if (in_y < 0 || in_y >= H) continue;
                    const float *in_row = in_ci + in_y * W;

                    for (int kx = 0; kx < 3; kx++) {
                        float wv = win[ky * 3 + kx];
                        ax_vf32 vw = ax_vf32_set1(wv);
                        int64_t shift = (int64_t)kx - 1;

                        int64_t x_lo = (kx == 0) ? 1 : 0;
                        int64_t x_hi = (kx == 2) ? (W - 1) : W;
                        int64_t x = x_lo;
                        int64_t span = x_hi - x_lo;
                        int64_t xvec_end = x_lo + (span - (span % AX_VF32_WIDTH));
                        for (; x < xvec_end; x += AX_VF32_WIDTH) {
                            ax_vf32 vi = ax_vf32_loadu(in_row + x + shift);
                            ax_vf32 vo = ax_vf32_loadu(out_row + x);
                            ax_vf32_storeu(out_row + x, ax_vf32_fmadd(vi, vw, vo));
                        }
                        for (; x < x_hi; x++)
                            out_row[x] += in_row[x + shift] * wv;
                    }
                }
            }
        }
    }
}

/* shape-aware path selection. measurement note: in practice the BLIS-style
   tiled gemm with explicit im2col beats this hand-rolled direct conv on
   typical mnist conv shapes (74s vs 137s at T=1) because the micro-kernel
   hits ~96 GFLOPS while the direct loop is bandwidth-bound on the bias init
   and the per-tap fmadd. direct 3x3 is kept here as available infrastructure
   for workloads where im2col cost truly dominates (very small spatial dims
   with many channels), but the default predicate disables it. flip
   `AX_USE_DIRECT_3X3=1` at compile time to opt back in. */
#ifndef AX_USE_DIRECT_3X3
#define AX_USE_DIRECT_3X3 0
#endif
static inline bool can_direct_3x3(int kh, int kw, int sh, int sw, int ph, int pw, int64_t C_in)
{
#if AX_USE_DIRECT_3X3
    return kh == 3 && kw == 3 && sh == 1 && sw == 1 && ph == 1 && pw == 1 && (C_in * 9) < 512;
#else
    (void)kh; (void)kw; (void)sh; (void)sw; (void)ph; (void)pw; (void)C_in;
    return false;
#endif
}

/* direct conv for very-small input channels (typical first layer: C_in=3 RGB).
   im2col would copy 9× the input for K=3, then run a GEMM with K_gemm=27 too
   small to amortize pack overhead. tf has a specialized first-layer kernel
   for the same reason. this kernel keeps the per-output-element accumulator
   in a vector register (one bias-init, K*K*C_in fmadds, one store per
   AX_VF32_WIDTH outputs). three regions split out: left-pad scalar, middle
   pure SIMD with no bounds check (the fast path that covers >97% of pixels
   on typical 224×224 inputs with pad=1), right-pad scalar. */
static void conv2d_direct_smallcin_sample(
    const float *in_n, const float *wd, const float *bias, float *out_n,
    int64_t C_in, int64_t C_out, int64_t H, int64_t W,
    int kh, int kw, int sh, int sw, int ph, int pw,
    int64_t H_out, int64_t W_out)
{
    const int64_t K_per_co = C_in * kh * kw;
    /* middle region of ow where every accessed iw is in [0, W).
       iw = ow*sw + kx - pw, ranges over kx in [0, kw).
       safe iff (ow*sw - pw) >= 0  AND  (ow*sw + kw-1 - pw) < W
            iff  ow >= ceil(pw/sw)  AND  ow <= floor((W-1 - (kw-1) + pw) / sw).
       the SIMD load reads AX_VF32_WIDTH consecutive columns when sw==1, so
       we additionally need (ow + AX_VF32_WIDTH - 1)*sw + kw-1 - pw < W. */
    int64_t ow_safe_lo = (pw + sw - 1) / sw;        /* ceil(pw/sw) */
    int64_t last_in = W - 1 - (kw - 1) + pw;        /* iw_base for last safe ow */
    int64_t ow_safe_hi = last_in >= 0 ? (last_in / sw) + 1 : 0;
    if (ow_safe_hi > W_out) ow_safe_hi = W_out;
    if (ow_safe_lo > ow_safe_hi) ow_safe_lo = ow_safe_hi;
    /* SIMD safety needs the last vector lane to also be in range when sw==1 */
    int64_t ow_simd_hi = (sw == 1) ? (ow_safe_hi - (AX_VF32_WIDTH - 1)) : ow_safe_lo;
    if (ow_simd_hi < ow_safe_lo) ow_simd_hi = ow_safe_lo;

    /* 4-co tiling path: amortizes input loads 4× across output channels.
       Each inner (ci, ky, kx) iter does 2 input loads (shared across 4 co)
       + 4 weight broadcasts + 8 fmadds. Fits cleanly in 14 AVX2 ymm regs:
       8 acc + 2 vi + 4 vw. C_out remainder (mod 4) falls through to
       single-co path below. */
    int64_t co = 0;
    if (sw == 1) {
        const int64_t TW = AX_VF32_WIDTH;
        const int64_t TILE_OW = 2 * TW;
        for (; co + 4 <= C_out; co += 4) {
            const float bv0 = bias ? bias[co + 0] : 0.0f;
            const float bv1 = bias ? bias[co + 1] : 0.0f;
            const float bv2 = bias ? bias[co + 2] : 0.0f;
            const float bv3 = bias ? bias[co + 3] : 0.0f;
            const ax_vf32 vb0 = ax_vf32_set1(bv0);
            const ax_vf32 vb1 = ax_vf32_set1(bv1);
            const ax_vf32 vb2 = ax_vf32_set1(bv2);
            const ax_vf32 vb3 = ax_vf32_set1(bv3);
            const float *wco0 = wd + (co + 0) * K_per_co;
            const float *wco1 = wd + (co + 1) * K_per_co;
            const float *wco2 = wd + (co + 2) * K_per_co;
            const float *wco3 = wd + (co + 3) * K_per_co;
            float *oco0 = out_n + (co + 0) * H_out * W_out;
            float *oco1 = out_n + (co + 1) * H_out * W_out;
            float *oco2 = out_n + (co + 2) * H_out * W_out;
            float *oco3 = out_n + (co + 3) * H_out * W_out;

            for (int64_t oh = 0; oh < H_out; oh++) {
                const int64_t ih_base = oh * sh - ph;
                float *out_row0 = oco0 + oh * W_out;
                float *out_row1 = oco1 + oh * W_out;
                float *out_row2 = oco2 + oh * W_out;
                float *out_row3 = oco3 + oh * W_out;
                int64_t ow = 0;

                /* left pad: scalar over the 4-co block */
                for (; ow < ow_safe_lo && ow < W_out; ow++) {
                    float a0 = bv0, a1 = bv1, a2 = bv2, a3 = bv3;
                    int64_t iw_base = ow - pw;
                    for (int64_t ci = 0; ci < C_in; ci++) {
                        const float *in_ci = in_n + ci * H * W;
                        for (int ky = 0; ky < kh; ky++) {
                            int64_t ih = ih_base + ky;
                            if (ih < 0 || ih >= H) continue;
                            const float *in_row = in_ci + ih * W;
                            int64_t wbase = (ci * kh + ky) * kw;
                            for (int kx = 0; kx < kw; kx++) {
                                int64_t iw = iw_base + kx;
                                if (iw < 0 || iw >= W) continue;
                                float v = in_row[iw];
                                a0 += wco0[wbase + kx] * v;
                                a1 += wco1[wbase + kx] * v;
                                a2 += wco2[wbase + kx] * v;
                                a3 += wco3[wbase + kx] * v;
                            }
                        }
                    }
                    out_row0[ow] = a0;
                    out_row1[ow] = a1;
                    out_row2[ow] = a2;
                    out_row3[ow] = a3;
                }

                /* SIMD core: 2 ow tiles × 4 co channels = 8 accumulators */
                int64_t v2end = ow_simd_hi - ((ow_simd_hi - ow) % TILE_OW);
                for (; ow + TILE_OW <= v2end; ow += TILE_OW) {
                    ax_vf32 a00 = vb0, a01 = vb0;
                    ax_vf32 a10 = vb1, a11 = vb1;
                    ax_vf32 a20 = vb2, a21 = vb2;
                    ax_vf32 a30 = vb3, a31 = vb3;
                    int64_t iw_base = ow - pw;
                    for (int64_t ci = 0; ci < C_in; ci++) {
                        const float *in_ci = in_n + ci * H * W;
                        for (int ky = 0; ky < kh; ky++) {
                            int64_t ih = ih_base + ky;
                            if (ih < 0 || ih >= H) continue;
                            const float *in_row = in_ci + ih * W;
                            int64_t wbase = (ci * kh + ky) * kw;
                            for (int kx = 0; kx < kw; kx++) {
                                ax_vf32 v0 = ax_vf32_loadu(in_row + iw_base + kx);
                                ax_vf32 v1 = ax_vf32_loadu(in_row + iw_base + TW + kx);
                                ax_vf32 vw0 = ax_vf32_set1(wco0[wbase + kx]);
                                ax_vf32 vw1 = ax_vf32_set1(wco1[wbase + kx]);
                                ax_vf32 vw2 = ax_vf32_set1(wco2[wbase + kx]);
                                ax_vf32 vw3 = ax_vf32_set1(wco3[wbase + kx]);
                                a00 = ax_vf32_fmadd(vw0, v0, a00);
                                a01 = ax_vf32_fmadd(vw0, v1, a01);
                                a10 = ax_vf32_fmadd(vw1, v0, a10);
                                a11 = ax_vf32_fmadd(vw1, v1, a11);
                                a20 = ax_vf32_fmadd(vw2, v0, a20);
                                a21 = ax_vf32_fmadd(vw2, v1, a21);
                                a30 = ax_vf32_fmadd(vw3, v0, a30);
                                a31 = ax_vf32_fmadd(vw3, v1, a31);
                            }
                        }
                    }
                    ax_vf32_storeu(out_row0 + ow,      a00);
                    ax_vf32_storeu(out_row0 + ow + TW, a01);
                    ax_vf32_storeu(out_row1 + ow,      a10);
                    ax_vf32_storeu(out_row1 + ow + TW, a11);
                    ax_vf32_storeu(out_row2 + ow,      a20);
                    ax_vf32_storeu(out_row2 + ow + TW, a21);
                    ax_vf32_storeu(out_row3 + ow,      a30);
                    ax_vf32_storeu(out_row3 + ow + TW, a31);
                }

                /* 1-tile SIMD tail (still 4-co) */
                int64_t vend = ow_simd_hi - ((ow_simd_hi - ow) % TW);
                for (; ow < vend; ow += TW) {
                    ax_vf32 a0 = vb0, a1 = vb1, a2 = vb2, a3 = vb3;
                    int64_t iw_base = ow - pw;
                    for (int64_t ci = 0; ci < C_in; ci++) {
                        const float *in_ci = in_n + ci * H * W;
                        for (int ky = 0; ky < kh; ky++) {
                            int64_t ih = ih_base + ky;
                            if (ih < 0 || ih >= H) continue;
                            const float *in_row = in_ci + ih * W;
                            int64_t wbase = (ci * kh + ky) * kw;
                            for (int kx = 0; kx < kw; kx++) {
                                ax_vf32 vi = ax_vf32_loadu(in_row + iw_base + kx);
                                a0 = ax_vf32_fmadd(ax_vf32_set1(wco0[wbase + kx]), vi, a0);
                                a1 = ax_vf32_fmadd(ax_vf32_set1(wco1[wbase + kx]), vi, a1);
                                a2 = ax_vf32_fmadd(ax_vf32_set1(wco2[wbase + kx]), vi, a2);
                                a3 = ax_vf32_fmadd(ax_vf32_set1(wco3[wbase + kx]), vi, a3);
                            }
                        }
                    }
                    ax_vf32_storeu(out_row0 + ow, a0);
                    ax_vf32_storeu(out_row1 + ow, a1);
                    ax_vf32_storeu(out_row2 + ow, a2);
                    ax_vf32_storeu(out_row3 + ow, a3);
                }

                /* right-pad / scalar tail (4-co) */
                for (; ow < W_out; ow++) {
                    float a0 = bv0, a1 = bv1, a2 = bv2, a3 = bv3;
                    int64_t iw_base = ow - pw;
                    for (int64_t ci = 0; ci < C_in; ci++) {
                        const float *in_ci = in_n + ci * H * W;
                        for (int ky = 0; ky < kh; ky++) {
                            int64_t ih = ih_base + ky;
                            if (ih < 0 || ih >= H) continue;
                            const float *in_row = in_ci + ih * W;
                            int64_t wbase = (ci * kh + ky) * kw;
                            for (int kx = 0; kx < kw; kx++) {
                                int64_t iw = iw_base + kx;
                                if (iw < 0 || iw >= W) continue;
                                float v = in_row[iw];
                                a0 += wco0[wbase + kx] * v;
                                a1 += wco1[wbase + kx] * v;
                                a2 += wco2[wbase + kx] * v;
                                a3 += wco3[wbase + kx] * v;
                            }
                        }
                    }
                    out_row0[ow] = a0;
                    out_row1[ow] = a1;
                    out_row2[ow] = a2;
                    out_row3[ow] = a3;
                }
            }
        }
    }

    /* remainder co (1..3 channels): single-co fallback path */
    for (; co < C_out; co++) {
        const float bv = bias ? bias[co] : 0.0f;
        const ax_vf32 vb = ax_vf32_set1(bv);
        const float *wco = wd + co * K_per_co;
        float *oco = out_n + co * H_out * W_out;

        for (int64_t oh = 0; oh < H_out; oh++) {
            const int64_t ih_base = oh * sh - ph;
            float *out_row = oco + oh * W_out;
            int64_t ow = 0;

            /* left pad: scalar, bounded loop */
            for (; ow < ow_safe_lo && ow < W_out; ow++) {
                float acc = bv;
                int64_t iw_base = ow * sw - pw;
                for (int64_t ci = 0; ci < C_in; ci++) {
                    const float *in_ci = in_n + ci * H * W;
                    for (int ky = 0; ky < kh; ky++) {
                        int64_t ih = ih_base + ky;
                        if (ih < 0 || ih >= H) continue;
                        const float *in_row = in_ci + ih * W;
                        const float *wky = wco + (ci * kh + ky) * kw;
                        for (int kx = 0; kx < kw; kx++) {
                            int64_t iw = iw_base + kx;
                            if (iw >= 0 && iw < W) acc += wky[kx] * in_row[iw];
                        }
                    }
                }
                out_row[ow] = acc;
            }

            /* middle region: pure SIMD, no per-pixel bounds check.
               4-way ow register tiling to hide fma latency. with C_in=3 and
               K=3 the inner (ci, ky, kx) loop is 27 sequential fmadds; a
               single accumulator chains all 27 dependencies through one
               register and stalls the pipeline (4-cycle fma latency × 27 =
               108 cycles, vs 14 cycles at throughput). 4 parallel
               accumulators turn that into 4 independent chains, fully
               saturating the FMA units. only oh-edge ky values still need
               the ih bounds check. */
            if (sw == 1) {
                const int64_t TW = AX_VF32_WIDTH;
                const int64_t TILE4 = 4 * TW;
                /* 4-tile path: process 4 vector blocks per outer iter */
                int64_t v4end = ow_simd_hi - ((ow_simd_hi - ow) % TILE4);
                for (; ow + TILE4 <= v4end; ow += TILE4) {
                    ax_vf32 a0 = vb, a1 = vb, a2 = vb, a3 = vb;
                    int64_t iw_base = ow - pw;
                    for (int64_t ci = 0; ci < C_in; ci++) {
                        const float *in_ci = in_n + ci * H * W;
                        for (int ky = 0; ky < kh; ky++) {
                            int64_t ih = ih_base + ky;
                            if (ih < 0 || ih >= H) continue;
                            const float *in_row = in_ci + ih * W;
                            const float *wky = wco + (ci * kh + ky) * kw;
                            for (int kx = 0; kx < kw; kx++) {
                                ax_vf32 vw = ax_vf32_set1(wky[kx]);
                                ax_vf32 v0 = ax_vf32_loadu(in_row + iw_base + kx);
                                ax_vf32 v1 = ax_vf32_loadu(in_row + iw_base + TW + kx);
                                ax_vf32 v2 = ax_vf32_loadu(in_row + iw_base + 2*TW + kx);
                                ax_vf32 v3 = ax_vf32_loadu(in_row + iw_base + 3*TW + kx);
                                a0 = ax_vf32_fmadd(vw, v0, a0);
                                a1 = ax_vf32_fmadd(vw, v1, a1);
                                a2 = ax_vf32_fmadd(vw, v2, a2);
                                a3 = ax_vf32_fmadd(vw, v3, a3);
                            }
                        }
                    }
                    ax_vf32_storeu(out_row + ow,          a0);
                    ax_vf32_storeu(out_row + ow + TW,     a1);
                    ax_vf32_storeu(out_row + ow + 2 * TW, a2);
                    ax_vf32_storeu(out_row + ow + 3 * TW, a3);
                }
                /* 1-tile tail */
                int64_t vend = ow_simd_hi - ((ow_simd_hi - ow) % TW);
                for (; ow < vend; ow += TW) {
                    ax_vf32 acc = vb;
                    int64_t iw_base = ow - pw;
                    for (int64_t ci = 0; ci < C_in; ci++) {
                        const float *in_ci = in_n + ci * H * W;
                        for (int ky = 0; ky < kh; ky++) {
                            int64_t ih = ih_base + ky;
                            if (ih < 0 || ih >= H) continue;
                            const float *in_row = in_ci + ih * W;
                            const float *wky = wco + (ci * kh + ky) * kw;
                            for (int kx = 0; kx < kw; kx++) {
                                ax_vf32 vw = ax_vf32_set1(wky[kx]);
                                ax_vf32 vi = ax_vf32_loadu(in_row + iw_base + kx);
                                acc = ax_vf32_fmadd(vw, vi, acc);
                            }
                        }
                    }
                    ax_vf32_storeu(out_row + ow, acc);
                }
            }

            /* tail / strided: scalar */
            for (; ow < W_out; ow++) {
                float acc = bv;
                int64_t iw_base = ow * sw - pw;
                for (int64_t ci = 0; ci < C_in; ci++) {
                    const float *in_ci = in_n + ci * H * W;
                    for (int ky = 0; ky < kh; ky++) {
                        int64_t ih = ih_base + ky;
                        if (ih < 0 || ih >= H) continue;
                        const float *in_row = in_ci + ih * W;
                        const float *wky = wco + (ci * kh + ky) * kw;
                        for (int kx = 0; kx < kw; kx++) {
                            int64_t iw = iw_base + kx;
                            if (iw >= 0 && iw < W) acc += wky[kx] * in_row[iw];
                        }
                    }
                }
                out_row[ow] = acc;
            }
        }
    }
}

/* eligibility for the small-C_in direct path. covers RGB image first-layer
   convs (the regression case): C_in ∈ {1,2,3,4}, kernel up to 7, any stride
   and padding. SIMD path requires the spatial output to be wide enough that
   middle-region SIMD covers more than the (left-pad + tail) overhead, so
   require W_out >= AX_VF32_WIDTH * 2 to be conservative. */
static inline bool can_direct_smallcin(int kh, int kw, int sh, int sw,
                                        int64_t C_in, int64_t W_out)
{
    (void)sh; (void)sw;
    return C_in <= 4 && kh <= 7 && kw <= 7
           && W_out >= (int64_t)AX_VF32_WIDTH * 2;
}

/* implicit gemm: useful when K is large (typically C_in >= 64 with 3x3+ kernels)
   AND M is large enough that the gemm dominates over the gather overhead. */
static inline bool prefer_implicit_gemm(int64_t K, int64_t M)
{
    return K >= 1024 && M >= 256;
}

static ax_tensor_t *conv2d_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_conv2d_t *conv = (ax_conv2d_t *)self;

    if (input->ndim != 4)
    {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "conv2d expects [N, C, H, W] input");
        return NULL;
    }

    /* ensure contiguous so memcpy-based im2col below works correctly */
    ax_tensor_t *inp = ax_ensure_contiguous(input);
    if (!inp) return NULL;

    int64_t N = inp->shape[0];
    int64_t C_in = inp->shape[1];
    int64_t H = inp->shape[2];
    int64_t W = inp->shape[3];
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
    int64_t out_shape[] = {N, C_out, out_h, out_w};
    /* every conv path below writes the full output (direct/smallcin fold bias
       into init; gemm paths copy res→od overwriting all elements). skip the
       zero pass — saves a full output-buffer write (~410 MB on
       conv_32x3x224x224 → 32x64x224x224). */
    ax_tensor_t *output = ax_tensor_create(out_shape, 4, AX_FLOAT32);
    if (!output) { if (inp != input) ax_tensor_destroy(inp); return NULL; }

    float *od = (float *)output->storage->data;
    float *wd = (float *)conv->weight->storage->data;

    /* lazily allocate (or reuse) per-layer scratch — eliminates per-call malloc */
    bool need_bwd = ax_grad_enabled() && (input->requires_grad || conv->weight->requires_grad);
    struct ax_conv_scratch *s = ensure_scratch(conv, N, H, W, need_bwd);
    if (!s) { if (inp != input) ax_tensor_destroy(inp); ax_tensor_destroy(output); return NULL; }

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

    /* shape-aware path selection. direct 3x3 wins for small kernels (mnist:
       both convs hit this). small-C_in direct beats im2col+gemm on the
       first layer of image nets (C_in=3, K=27 too small for GEMM). implicit
       gemm wins for large K + large M. otherwise fall back to explicit
       im2col + gemm. */
    bool use_direct = can_direct_3x3(kh, kw, sh, sw, ph, pw, C_in);
    bool use_smallcin = !use_direct && can_direct_smallcin(kh, kw, sh, sw, C_in, out_w);
    bool use_implicit = !use_direct && !use_smallcin && ax_compute_has_conv_gemm() && prefer_implicit_gemm(K, M);
    bool use_batched = !use_direct && !use_smallcin && !use_implicit && N > 1 && M < AX_CONV_BATCH_M_THRESH
                       && s->batch_col_buf && s->batch_aux_buf;

    if (use_batched) {
        /* batch all N im2cols into [K, N*M], do one GEMM, scatter with bias.
           eliminates N separate GEMM launches and gives full NC utilization
           when M (per sample) is too small to fill register tiles alone. */
        float *bcd = (float *)s->batch_col_buf->storage->data;
        float *brd = (float *)s->batch_aux_buf->storage->data;
        int64_t NM = N * M;

        #ifdef _OPENMP
        #pragma omp parallel for num_threads(T) schedule(static)
        #endif
        for (int64_t n = 0; n < N; n++)
            im2col_into_strided(ind + n * C_in * H * W, C_in, H, W,
                                 kh, kw, sh, sw, ph, pw, out_h, out_w,
                                 bcd, NM, n * M);

        /* opt_gemm zero-fills C internally before accumulating; no need
           to memset brd here. saves ~12.8 MB on the conv_512x1 batched case. */
        ax_compute_gemm(w2d, s->batch_col_buf, s->batch_aux_buf);

        /* scatter brd[C_out, N, M] → od[N, C_out, M] with bias add.
           co-outer reads brd contiguously per channel; the original
           n-outer/co-inner pattern jumped NM*4 = 25 KB per inner step
           (1×1 conv at C_out=512 thrashed L1). */
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(T) schedule(static)
        #endif
        for (int64_t co = 0; co < C_out; co++) {
            float bias_val = bias_data ? bias_data[co] : 0.0f;
            ax_vf32 vb = ax_vf32_set1(bias_val);
            const float *src_co = brd + co * NM;
            for (int64_t n = 0; n < N; n++) {
                float *dst = od + (n * C_out + co) * M;
                const float *src = src_co + n * M;
                int64_t m = 0, ve = M - (M % AX_VF32_WIDTH);
                for (; m < ve; m += AX_VF32_WIDTH)
                    ax_vf32_storeu(dst + m, ax_vf32_add(ax_vf32_loadu(src + m), vb));
                for (; m < M; m++) dst[m] = src[m] + bias_val;
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
        } else if (kh == 1 && kw == 1 && sh == 1 && sw == 1
                   && ph == 0 && pw == 0 && H == out_h && W == out_w) {
            /* 1x1 stride-1 pad-0 conv: im2col is the identity layout.
               skip the copy entirely by constructing a stack tensor view
               over the input slice and handing it straight to gemm. this
               saves a C_in*H*W memcpy per sample (400kb+ on vgg shapes).
               storage/tensor are stack-allocated so there's no refcount
               bookkeeping; the parent input keeps the real storage alive
               for the duration of this forward pass. */
            ax_storage_t in_storage;
            in_storage.data = (void *)(ind + n * C_in * H * W);
            in_storage.size_bytes = (size_t)(C_in * H * W) * sizeof(float);
            atomic_store(&in_storage.refcount, 0);
            in_storage.device = AX_DEVICE_CPU;
            in_storage.is_arena_temp = true;
            in_storage.generation = 1;

            ax_tensor_t in_slice;
            memset(&in_slice, 0, sizeof(in_slice));
            in_slice.storage = &in_storage;
            in_slice.ndim = 2;
            in_slice.dtype = AX_FLOAT32;
            in_slice.offset = 0;
            in_slice.shape[0] = C_in;
            in_slice.shape[1] = M;
            in_slice.strides[0] = M;
            in_slice.strides[1] = 1;

            /* opt_gemm zero-fills C internally; skip our memset. */
            ax_compute_gemm(w2d, &in_slice, res);
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
        ax_grad_fn_t *gf = ax_grad_fn_create(conv2d_backward);
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
    if (!c) return NULL;

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


/* fused conv2d + batchnorm + relu composite layer.
   collapses three sequential passes over the conv output buffer (conv write,
   bn read+write, relu read+write) into two: conv write + bn_relu read+write.
   the stats reduction (mean/var) still does one extra read per channel so
   overall we go from ~5 buffer passes to ~4, a meaningful bandwidth win on
   memory-bound cnn workloads. backward uses the saved-tensors approach,
   inlining relu-backward -> bn-backward -> conv-backward to keep autograd
   wiring simple while still reusing the per-layer scratch. */

#include "axiom/norm.h" /* for ax_batchnorm_t layout */

/* backward ctx: shape snapshot + layer pointer. the saved tensors
   (x_hat, inv_std, contig input) live in gf->saved[] so graph cleanup
   frees them automatically. */
typedef struct {
    ax_conv_bn_relu_t *layer;
    int64_t N, C_out, out_h, out_w;
    int64_t C_in, H, W;
} cbr_bwd_ctx_t;

static void cbr_ctx_cleanup(void *p) { free(p); }

static void conv_bn_relu_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    cbr_bwd_ctx_t *ctx = (cbr_bwd_ctx_t *)self->ctx;
    ax_conv_bn_relu_t *L = ctx->layer;
    ax_tensor_t *input_orig = self->inputs[0];
    ax_tensor_t *input_data = self->saved[0]; /* contiguous input */
    ax_tensor_t *x_hat_t    = self->saved[1];
    ax_tensor_t *inv_std_t  = self->saved[2];

    int64_t N = ctx->N, C_out = ctx->C_out, out_h = ctx->out_h, out_w = ctx->out_w;
    int64_t C_in = ctx->C_in, H = ctx->H, W = ctx->W;
    int kh = L->kernel_h, kw = L->kernel_w;
    int sh = L->stride_h, sw = L->stride_w;
    int ph = L->pad_h, pw = L->pad_w;
    int64_t spatial = out_h * out_w;
    float Nf = (float)(N * spatial);

    /* build the gradient wrt the conv output into an arena-allocated buffer.
       this is: relu_mask * bn_backward_dx(grad_out). the relu mask is
       reconstructed from saved x_hat by computing bn_out = gamma*x_hat+beta
       (cheap, avoids saving yet another tensor). */
    ax_arena_t *arena = ax_backward_arena();
    int64_t gconv_shape[] = {N, C_out, out_h, out_w};
    ax_tensor_t *grad_conv = ax_tensor_arena_zeros(arena, gconv_shape, 4, AX_FLOAT32);
    if (!grad_conv) return;

    float *go = (float *)grad_out->storage->data;
    float *xh = (float *)x_hat_t->storage->data;
    float *istd = (float *)inv_std_t->storage->data;
    float *gd = (float *)L->gamma->storage->data;
    float *bd = (float *)L->beta->storage->data;
    float *gc = (float *)grad_conv->storage->data;

    float *dg = NULL, *db = NULL;
    if (L->gamma->requires_grad) {
        if (!L->gamma->grad)
            L->gamma->grad = ax_tensor_zeros(L->gamma->shape, L->gamma->ndim, L->gamma->dtype);
        if (L->gamma->grad) dg = (float *)L->gamma->grad->storage->data;
    }
    if (L->beta->requires_grad) {
        if (!L->beta->grad)
            L->beta->grad = ax_tensor_zeros(L->beta->shape, L->beta->ndim, L->beta->dtype);
        if (L->beta->grad) db = (float *)L->beta->grad->storage->data;
    }

    /* per-channel parallel: apply relu mask (bn_out > 0 gate), accumulate
       dgamma/dbeta via SIMD, then compute dx formula into grad_conv.
       each channel c owns disjoint slices of go/xh/gc, so no atomics needed
       on dg[c]/db[c] — they're accessed by exactly one thread. */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
    #endif
    for (int64_t c = 0; c < C_out; c++) {
        float g_c = gd[c], b_c = bd[c];
        ax_vf32 v_gc  = ax_vf32_set1(g_c);
        ax_vf32 v_bc  = ax_vf32_set1(b_c);
        ax_vf32 v_zero = ax_vf32_zero();
        ax_vf32 v_sgo = ax_vf32_zero(), v_sgx = ax_vf32_zero();
        float s_sgo = 0.0f, s_sgx = 0.0f;
        int64_t se = spatial - (spatial % AX_VF32_WIDTH);

        for (int64_t n = 0; n < N; n++) {
            int64_t base = n * C_out * spatial + c * spatial;
            const float *go_p = go + base;
            const float *xh_p = xh + base;
            float *gc_p = gc + base;
            int64_t s = 0;
            for (; s < se; s += AX_VF32_WIDTH) {
                ax_vf32 x = ax_vf32_loadu(xh_p + s);
                ax_vf32 bn_out = ax_vf32_fmadd(v_gc, x, v_bc);
                /* relu mask: cmpgt returns 1.0f where true, 0.0f otherwise.
                   mul by mask gates the gradient without a bitwise AND. */
                ax_vf32 mask = ax_vf32_cmpgt(bn_out, v_zero);
                ax_vf32 g = ax_vf32_mul(ax_vf32_loadu(go_p + s), mask);
                ax_vf32_storeu(gc_p + s, g);
                v_sgo = ax_vf32_add(v_sgo, g);
                v_sgx = ax_vf32_fmadd(g, x, v_sgx);
            }
            for (; s < spatial; s++) {
                float bn_out = g_c * xh_p[s] + b_c;
                float m = (bn_out > 0.0f) ? go_p[s] : 0.0f;
                gc_p[s] = m;
                s_sgo += m;
                s_sgx += m * xh_p[s];
            }
        }
        float sum_go    = ax_vf32_hsum(v_sgo) + s_sgo;
        float sum_go_xh = ax_vf32_hsum(v_sgx) + s_sgx;

        /* no atomic: each thread owns exactly one c, no contention */
        if (dg) dg[c] += sum_go_xh;
        if (db) db[c] += sum_go;

        float coeff = g_c * istd[c] / Nf;
        ax_vf32 v_coeff  = ax_vf32_set1(coeff);
        ax_vf32 v_Nf     = ax_vf32_set1(Nf);
        ax_vf32 v_sum_go = ax_vf32_set1(sum_go);
        ax_vf32 v_sum_gx = ax_vf32_set1(sum_go_xh);

        for (int64_t n = 0; n < N; n++) {
            int64_t base = n * C_out * spatial + c * spatial;
            const float *xh_p = xh + base;
            float *gc_p = gc + base;
            int64_t s = 0;
            for (; s < se; s += AX_VF32_WIDTH) {
                ax_vf32 m = ax_vf32_loadu(gc_p + s);
                ax_vf32 x = ax_vf32_loadu(xh_p + s);
                /* dx = coeff * (N*m - sum_go - x*sum_go_xh) */
                ax_vf32 dx = ax_vf32_mul(v_coeff,
                    ax_vf32_sub(ax_vf32_sub(ax_vf32_mul(v_Nf, m), v_sum_go),
                                ax_vf32_mul(x, v_sum_gx)));
                ax_vf32_storeu(gc_p + s, dx);
            }
            for (; s < spatial; s++) {
                float m = gc_p[s];
                gc_p[s] = coeff * (Nf * m - sum_go - xh_p[s] * sum_go_xh);
            }
        }
    }

    /* step 2: conv backward using grad_conv as upstream gradient.
       near-copy of conv2d_backward using the layer's cached scratch. */
    ax_tensor_t *weight = L->weight;

    if (weight->requires_grad && !weight->grad)
        weight->grad = ax_tensor_zeros(weight->shape, weight->ndim, weight->dtype);
    if (input_orig->requires_grad && !input_orig->grad)
        input_orig->grad = ax_tensor_zeros(input_orig->shape, input_orig->ndim, input_orig->dtype);

    if (L->use_bias && L->bias && L->bias->requires_grad) {
        if (!L->bias->grad)
            L->bias->grad = ax_tensor_zeros(L->bias->shape, L->bias->ndim, L->bias->dtype);
        float *bg = (float *)L->bias->grad->storage->data;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t c = 0; c < C_out; c++) {
            float sum = 0.0f;
            for (int64_t n = 0; n < N; n++) {
                int64_t base = n * C_out * spatial + c * spatial;
                for (int64_t s = 0; s < spatial; s++) sum += gc[base + s];
            }
            bg[c] += sum;
        }
    }

    float *wdata = (float *)weight->storage->data;
    int64_t K = C_in * kh * kw;
    int64_t M = out_h * out_w;

    struct ax_conv_scratch *s = L->scratch;
    if (!s || s->N != N || s->H != H || s->W != W || !s->dw_bufs) {
        /* forward should have built this; if not we can't proceed safely */
        return;
    }
    int T = s->T;

    bool have_gemm_nt_b = ax_compute_has_gemm_nt();
    bool have_gemm_tn_b = ax_compute_has_gemm_tn();

    if (input_orig->requires_grad && !have_gemm_tn_b) {
        if (!s->wt_contig) {
            int64_t wt_shape[] = {K, C_out};
            s->wt_contig = ax_tensor_create(wt_shape, 2, AX_FLOAT32);
        }
        if (s->wt_contig) {
            float *wtd = (float *)s->wt_contig->storage->data;
            for (int64_t r = 0; r < C_out; r++)
                for (int64_t col = 0; col < K; col++)
                    wtd[col * C_out + r] = wdata[r * K + col];
        }
    }
    ax_tensor_t *w2d_bn = s->w2d;
    if (input_orig->requires_grad && have_gemm_tn_b && w2d_bn) {
        memcpy(w2d_bn->storage->data, wdata, (size_t)(C_out * K) * sizeof(float));
    }

    if (weight->requires_grad) {
        for (int t = 0; t < T; t++)
            memset(s->dw_bufs[t]->storage->data, 0, (size_t)(C_out * K) * sizeof(float));
    }

    float *ind = (float *)input_data->storage->data;
    ax_tensor_t *wt_contig = s->wt_contig;

    bool use_batched_cbr = N > 1 && M < AX_CONV_BATCH_M_THRESH
                           && s->batch_col_buf && s->batch_aux_buf;

    if (use_batched_cbr) {
        float *bcd = (float *)s->batch_col_buf->storage->data;
        float *gbd = (float *)s->batch_aux_buf->storage->data;
        int64_t NM = N * M;

        #ifdef _OPENMP
        #pragma omp parallel for num_threads(T) schedule(static)
        #endif
        for (int64_t n = 0; n < N; n++) {
            im2col_into_strided(ind + n * C_in * H * W, C_in, H, W,
                                 kh, kw, sh, sw, ph, pw, out_h, out_w,
                                 bcd, NM, n * M);
            for (int64_t co = 0; co < C_out; co++)
                memcpy(gbd + co * NM + n * M, gc + n * C_out * M + co * M,
                       (size_t)M * sizeof(float));
        }

        if (weight->requires_grad) {
            ax_tensor_t *dw_total = s->dw_bufs[0];
            memset(dw_total->storage->data, 0, (size_t)(C_out * K) * sizeof(float));
            ax_compute_gemm_nt(s->batch_aux_buf, s->batch_col_buf, dw_total);
            float *wg  = (float *)weight->grad->storage->data;
            float *dwl = (float *)dw_total->storage->data;
            int64_t wn = C_out * K, wi = 0, wve = wn - (wn % AX_VF32_WIDTH);
            for (; wi < wve; wi += AX_VF32_WIDTH)
                ax_vf32_storeu(wg + wi, ax_vf32_add(ax_vf32_loadu(wg + wi), ax_vf32_loadu(dwl + wi)));
            for (; wi < wn; wi++) wg[wi] += dwl[wi];
        }

        if (input_orig->requires_grad && (wt_contig || have_gemm_tn_b)) {
            memset(bcd, 0, (size_t)(K * NM) * sizeof(float));
            if (have_gemm_tn_b)
                ax_compute_gemm_tn(w2d_bn, s->batch_aux_buf, s->batch_col_buf);
            else
                ax_compute_gemm(wt_contig, s->batch_aux_buf, s->batch_col_buf);

            #ifdef _OPENMP
            #pragma omp parallel for num_threads(T) schedule(static)
            #endif
            for (int64_t n = 0; n < N; n++) {
                int tid = AX_OMP_THREAD_NUM();
                if (tid >= T) tid = 0;
                float *dcol_d = (float *)s->dcol_bufs[tid]->storage->data;
                for (int64_t k = 0; k < K; k++)
                    memcpy(dcol_d + k * M, bcd + k * NM + n * M, (size_t)M * sizeof(float));
                float *dimg_d = (float *)s->dimg_bufs[tid]->storage->data;
                memset(dimg_d, 0, (size_t)(C_in * H * W) * sizeof(float));
                col2im_into(dcol_d, C_in, H, W, kh, kw, sh, sw, ph, pw, out_h, out_w, dimg_d);
                float *ig = (float *)input_orig->grad->storage->data + n * C_in * H * W;
                int64_t total = C_in * H * W, i = 0, ve = total - (total % AX_VF32_WIDTH);
                for (; i < ve; i += AX_VF32_WIDTH)
                    ax_vf32_storeu(ig + i, ax_vf32_add(ax_vf32_loadu(ig + i), ax_vf32_loadu(dimg_d + i)));
                for (; i < total; i++) ig[i] += dimg_d[i];
            }
        }
    } else {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t n = 0; n < N; n++) {
        int tid = AX_OMP_THREAD_NUM();
        if (tid >= T) tid = 0;
        float *cbd = (float *)s->col_bufs[tid]->storage->data;
        ax_tensor_t *col_buf = s->col_bufs[tid];
        ax_tensor_t *go_mat = s->go_bufs[tid];

        im2col_into(ind + n * C_in * H * W, C_in, H, W,
                     kh, kw, sh, sw, ph, pw, out_h, out_w, cbd);

        memcpy(go_mat->storage->data, gc + n * C_out * M,
               (size_t)(C_out * M) * sizeof(float));

        if (weight->requires_grad) {
            ax_tensor_t *dw_local = s->dw_bufs[tid];
            ax_tensor_t *dw_sample = s->dws_bufs[tid];
            float *dws = (float *)dw_sample->storage->data;
            memset(dws, 0, (size_t)(C_out * K) * sizeof(float));

            if (have_gemm_nt_b) {
                ax_compute_gemm_nt(go_mat, col_buf, dw_sample);
            } else {
                ax_tensor_t *colt_buf = s->colt_bufs[tid];
                float *ctd = (float *)colt_buf->storage->data;
                for (int64_t r = 0; r < K; r++)
                    for (int64_t col = 0; col < M; col++)
                        ctd[col * K + r] = cbd[r * M + col];
                ax_compute_gemm(go_mat, colt_buf, dw_sample);
            }

            float *dwl = (float *)dw_local->storage->data;
            int64_t wn = C_out * K;
            int64_t wi = 0, wve = wn - (wn % AX_VF32_WIDTH);
            for (; wi < wve; wi += AX_VF32_WIDTH)
                ax_vf32_storeu(dwl + wi, ax_vf32_add(ax_vf32_loadu(dwl + wi), ax_vf32_loadu(dws + wi)));
            for (; wi < wn; wi++) dwl[wi] += dws[wi];
        }

        if (input_orig->requires_grad && (wt_contig || have_gemm_tn_b)) {
            ax_tensor_t *dcol_buf = s->dcol_bufs[tid];
            ax_tensor_t *dimg_buf = s->dimg_bufs[tid];

            memset(dcol_buf->storage->data, 0, (size_t)(K * M) * sizeof(float));
            if (have_gemm_tn_b) {
                ax_compute_gemm_tn(w2d_bn, go_mat, dcol_buf);
            } else {
                ax_compute_gemm(wt_contig, go_mat, dcol_buf);
            }

            float *dimg_d = (float *)dimg_buf->storage->data;
            memset(dimg_d, 0, (size_t)(C_in * H * W) * sizeof(float));
            col2im_into((float *)dcol_buf->storage->data, C_in, H, W,
                         kh, kw, sh, sw, ph, pw, out_h, out_w, dimg_d);

            float *ig = (float *)input_orig->grad->storage->data + n * C_in * H * W;
            int64_t total = C_in * H * W;
            int64_t i = 0, ve = total - (total % AX_VF32_WIDTH);
            for (; i < ve; i += AX_VF32_WIDTH)
                ax_vf32_storeu(ig + i, ax_vf32_add(ax_vf32_loadu(ig + i), ax_vf32_loadu(dimg_d + i)));
            for (; i < total; i++) ig[i] += dimg_d[i];
        }
    }

    if (weight->requires_grad) {
        float *wg = (float *)weight->grad->storage->data;
        int64_t wn = C_out * K;
        for (int t = 0; t < T; t++) {
            float *dwl = (float *)s->dw_bufs[t]->storage->data;
            int64_t wi = 0, wve = wn - (wn % AX_VF32_WIDTH);
            for (; wi < wve; wi += AX_VF32_WIDTH)
                ax_vf32_storeu(wg + wi, ax_vf32_add(ax_vf32_loadu(wg + wi), ax_vf32_loadu(dwl + wi)));
            for (; wi < wn; wi++) wg[wi] += dwl[wi];
        }
    }
    } /* end use_batched_cbr else */
}

/* ensure_scratch operates on ax_conv2d_t*; use a transient shim so we can
   share the same scratch logic across the plain and fused layers without
   duplicating ~100 lines of allocation code. */
static struct ax_conv_scratch *ensure_scratch_cbr(ax_conv_bn_relu_t *L,
                                                    int64_t N, int64_t H, int64_t W,
                                                    bool need_backward)
{
    ax_conv2d_t shim;
    memset(&shim, 0, sizeof(shim));
    shim.in_channels  = L->in_channels;
    shim.out_channels = L->out_channels;
    shim.kernel_h = L->kernel_h; shim.kernel_w = L->kernel_w;
    shim.stride_h = L->stride_h; shim.stride_w = L->stride_w;
    shim.pad_h = L->pad_h;       shim.pad_w = L->pad_w;
    shim.scratch = L->scratch;
    struct ax_conv_scratch *s = ensure_scratch(&shim, N, H, W, need_backward);
    L->scratch = shim.scratch;
    return s;
}

static ax_tensor_t *conv_bn_relu_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_conv_bn_relu_t *L = (ax_conv_bn_relu_t *)self;
    if (!input || input->ndim != 4) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "conv_bn_relu expects [N,C,H,W] input");
        return NULL;
    }

    ax_tensor_t *inp = ax_ensure_contiguous(input);
    if (!inp) return NULL;

    int64_t N = inp->shape[0];
    int64_t C_in = inp->shape[1];
    int64_t H = inp->shape[2];
    int64_t W = inp->shape[3];
    int64_t C_out = L->out_channels;
    int kh = L->kernel_h, kw = L->kernel_w;
    int sh = L->stride_h, sw = L->stride_w;
    int ph = L->pad_h, pw = L->pad_w;

    int64_t out_h = conv_out_dim(H, kh, sh, ph);
    int64_t out_w = conv_out_dim(W, kw, sw, pw);
    if (out_h <= 0 || out_w <= 0) {
        ax_err_set(AX_ERR_INVALID_SHAPE, "conv_bn_relu: invalid output dims");
        if (inp != input) ax_tensor_destroy(inp);
        return NULL;
    }

    int64_t K = C_in * kh * kw;
    int64_t M = out_h * out_w;
    int64_t spatial = M;
    int64_t out_shape[] = {N, C_out, out_h, out_w};
    /* skip zero pass: pass 1 below writes every output (direct/smallcin
       with bias init, gemm paths via res→od bias-copy). */
    ax_tensor_t *output = ax_tensor_create(out_shape, 4, AX_FLOAT32);
    if (!output) { if (inp != input) ax_tensor_destroy(inp); return NULL; }

    float *od = (float *)output->storage->data;
    float *wd = (float *)L->weight->storage->data;

    bool record = ax_grad_enabled() && self->training &&
                  (input->requires_grad || L->weight->requires_grad ||
                   L->gamma->requires_grad || L->beta->requires_grad);

    struct ax_conv_scratch *s = ensure_scratch_cbr(L, N, H, W, record);
    if (!s) { if (inp != input) ax_tensor_destroy(inp); ax_tensor_destroy(output); return NULL; }

    ax_tensor_t *w2d = s->w2d;
    memcpy(w2d->storage->data, wd, (size_t)(C_out * K) * sizeof(float));

    int T = s->T;
    float *ind = (float *)inp->storage->data;
    const float *bias_data = (L->use_bias && L->bias)
        ? (const float *)L->bias->storage->data : NULL;

    /* shape-aware path selection mirrors conv2d_forward. */
    bool use_direct = can_direct_3x3(kh, kw, sh, sw, ph, pw, C_in);
    bool use_smallcin = !use_direct && can_direct_smallcin(kh, kw, sh, sw, C_in, out_w);
    bool use_implicit = !use_direct && !use_smallcin && ax_compute_has_conv_gemm() && prefer_implicit_gemm(K, M);
    bool use_batched_cbr_fwd = !use_direct && !use_smallcin && !use_implicit && N > 1
                               && M < AX_CONV_BATCH_M_THRESH
                               && s->batch_col_buf && s->batch_aux_buf;

    /* pass 1: materialize conv output (including bias) into the final output buffer.
       pass 2/3 (bn stats + fused bn+relu apply) overwrite it in place. this saves
       one buffer pass compared to the unfused path which would first write bn
       output into a separate tensor and then relu into another. */
    if (use_batched_cbr_fwd) {
        float *bcd = (float *)s->batch_col_buf->storage->data;
        float *brd = (float *)s->batch_aux_buf->storage->data;
        int64_t NM = N * M;

        #ifdef _OPENMP
        #pragma omp parallel for num_threads(T) schedule(static)
        #endif
        for (int64_t n = 0; n < N; n++)
            im2col_into_strided(ind + n * C_in * H * W, C_in, H, W,
                                 kh, kw, sh, sw, ph, pw, out_h, out_w,
                                 bcd, NM, n * M);

        /* opt_gemm zero-fills C internally; skip our memset. */
        ax_compute_gemm(w2d, s->batch_col_buf, s->batch_aux_buf);

        /* scatter brd[C_out, N, M] → od[N, C_out, M] with bias add (co-outer
           = contiguous brd reads; see conv2d_forward batched path). */
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(T) schedule(static)
        #endif
        for (int64_t co = 0; co < C_out; co++) {
            float bias_val = bias_data ? bias_data[co] : 0.0f;
            ax_vf32 v_b = ax_vf32_set1(bias_val);
            const float *src_co = brd + co * NM;
            for (int64_t n = 0; n < N; n++) {
                float *dst = od + (n * C_out + co) * M;
                const float *src = src_co + n * M;
                int64_t m = 0, me = M - (M % AX_VF32_WIDTH);
                for (; m < me; m += AX_VF32_WIDTH)
                    ax_vf32_storeu(dst + m, ax_vf32_add(ax_vf32_loadu(src + m), v_b));
                for (; m < M; m++) dst[m] = src[m] + bias_val;
            }
        }
    } else {
    /* num_threads(T) caps the team to per-thread scratch slot count. */
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(T) schedule(dynamic, 1)
    #endif
    for (int64_t n = 0; n < N; n++) {
        if (use_direct) {
            /* direct conv writes [C_out, H, W] for the sample, bias folded in.
               this gives us the conv result directly in `od` without scratch. */
            conv2d_direct_3x3_sample(
                ind + n * C_in * H * W, wd, bias_data,
                od + n * C_out * M, C_in, C_out, H, W);
            continue;
        }
        if (use_smallcin) {
            /* small-C_in direct: writes conv result with bias into od[n].
               BN+ReLU pass below overwrites in-place. */
            conv2d_direct_smallcin_sample(
                ind + n * C_in * H * W, wd, bias_data,
                od + n * C_out * M,
                C_in, C_out, H, W,
                kh, kw, sh, sw, ph, pw, out_h, out_w);
            continue;
        }

        int tid = AX_OMP_THREAD_NUM();
        if (tid >= T) tid = 0;
        ax_tensor_t *col = s->col_bufs[tid];
        ax_tensor_t *res = s->res_bufs[tid];
        float *rd = (float *)res->storage->data;

        if (use_implicit) {
            ax_conv_params_t cp = {
                .input = ind + n * C_in * H * W,
                .C_in = C_in, .H = H, .W = W,
                .kh = kh, .kw = kw,
                .sh = sh, .sw = sw,
                .ph = ph, .pw = pw,
                .out_h = out_h, .out_w = out_w,
            };
            ax_compute_conv_gemm(w2d, &cp, res);
        } else if (kh == 1 && kw == 1 && sh == 1 && sw == 1
                   && ph == 0 && pw == 0 && H == out_h && W == out_w) {
            /* 1x1 conv: im2col is the identity layout — skip the copy.
               stack-allocated view into inp[n] with refcount=0 (parent keeps
               storage alive). matches conv2d_forward path. */
            ax_storage_t in_storage;
            in_storage.data = (void *)(ind + n * C_in * H * W);
            in_storage.size_bytes = (size_t)(C_in * H * W) * sizeof(float);
            atomic_store(&in_storage.refcount, 0);
            in_storage.device = AX_DEVICE_CPU;
            in_storage.is_arena_temp = true;
            in_storage.generation = 1;

            ax_tensor_t in_slice;
            memset(&in_slice, 0, sizeof(in_slice));
            in_slice.storage = &in_storage;
            in_slice.ndim = 2;
            in_slice.dtype = AX_FLOAT32;
            in_slice.offset = 0;
            in_slice.shape[0] = C_in;
            in_slice.shape[1] = M;
            in_slice.strides[0] = M;
            in_slice.strides[1] = 1;

            /* opt_gemm zero-fills C internally; skip our memset. */
            ax_compute_gemm(w2d, &in_slice, res);
        } else {
            float *cd = (float *)col->storage->data;
            im2col_into(ind + n * C_in * H * W, C_in, H, W,
                         kh, kw, sh, sw, ph, pw, out_h, out_w, cd);
            /* opt_gemm zero-fills C internally; skip our memset. */
            ax_compute_gemm(w2d, col, res);
        }

        for (int64_t co = 0; co < C_out; co++) {
            float bias_val = bias_data ? bias_data[co] : 0.0f;
            float *dst = od + ((n * C_out + co) * out_h) * out_w;
            float *src = rd + co * M;
            int64_t m = 0, me = M - (M % AX_VF32_WIDTH);
            ax_vf32 v_b = ax_vf32_set1(bias_val);
            for (; m < me; m += AX_VF32_WIDTH)
                ax_vf32_storeu(dst + m, ax_vf32_add(ax_vf32_loadu(src + m), v_b));
            for (; m < M; m++) dst[m] = src[m] + bias_val;
        }
    }
    } /* end use_batched_cbr_fwd else */

    /* allocate backward saves after pass 1 so a failed alloc still lets us
       return a valid forward result (just without gradients).
       use the forward arena: both tensors are fully overwritten below, so
       ax_tensor_zeros / memset would be wasted bandwidth. the arena frees
       everything in bulk at ax_graph_cleanup, matching the lifetime. */
    ax_tensor_t *x_hat_save = NULL;
    ax_tensor_t *inv_std_save = NULL;
    if (record) {
        ax_arena_t *fa = ax_forward_arena();
        int64_t is_shape[] = {C_out};
        if (fa) {
            x_hat_save  = ax_tensor_arena_create(fa, out_shape, 4, AX_FLOAT32);
            inv_std_save = ax_tensor_arena_create(fa, is_shape, 1, AX_FLOAT32);
        } else {
            x_hat_save  = ax_tensor_create(out_shape, 4, AX_FLOAT32);
            inv_std_save = ax_tensor_create(is_shape, 1, AX_FLOAT32);
        }
        if (!x_hat_save || !inv_std_save) {
            /* non-fatal: forward result is still valid, just no grads */
            x_hat_save = inv_std_save = NULL;
            record = false;
        }
    }

    float *gd = (float *)L->gamma->storage->data;
    float *bnbeta = (float *)L->beta->storage->data;
    float *rm = (float *)L->running_mean->storage->data;
    float *rv = (float *)L->running_var->storage->data;

    if (self->training) {
        float *xh_d = record ? (float *)x_hat_save->storage->data : NULL;
        float *is_d = record ? (float *)inv_std_save->storage->data : NULL;
        float eff_n = (float)(N * spatial);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
        #endif
        for (int64_t c = 0; c < C_out; c++) {
            /* 1-pass fused sum + sum^2: halves memory reads vs 2-pass.
               E[x^2] - E[x]^2 is numerically adequate for float32 at
               typical BN working ranges (activation magnitudes ~0-10). */
            ax_vf32 vs = ax_vf32_zero(), vs2 = ax_vf32_zero();
            float s_tail = 0.0f, s2_tail = 0.0f;
            for (int64_t n = 0; n < N; n++) {
                int64_t base = n * C_out * spatial + c * spatial;
                int64_t m = 0, me = spatial - (spatial % AX_VF32_WIDTH);
                for (; m < me; m += AX_VF32_WIDTH) {
                    ax_vf32 x = ax_vf32_loadu(od + base + m);
                    vs  = ax_vf32_add(vs, x);
                    vs2 = ax_vf32_fmadd(x, x, vs2);
                }
                for (; m < spatial; m++) {
                    float x = od[base + m];
                    s_tail  += x;
                    s2_tail += x * x;
                }
            }
            float sum  = ax_vf32_hsum(vs)  + s_tail;
            float sum2 = ax_vf32_hsum(vs2) + s2_tail;
            float mean = sum / eff_n;
            float var  = sum2 / eff_n - mean * mean;
            if (var < 0.0f) var = 0.0f;
            float var_sum = var * eff_n; /* Σ(x)^2 - N*mean^2, for unbiased estimator */
            float inv_std = 1.0f / sqrtf(var + L->bn_eps);
            if (record) is_d[c] = inv_std;

            /* fused apply: out = max(0, gamma * (conv - mean) * inv_std + beta)
               collapses to: out = max(0, scale*conv + bias_out). */
            float scale = gd[c] * inv_std;
            float bias_out = bnbeta[c] - gd[c] * mean * inv_std;
            ax_vf32 v_sc = ax_vf32_set1(scale);
            ax_vf32 v_bi = ax_vf32_set1(bias_out);
            ax_vf32 v_is = ax_vf32_set1(inv_std);
            ax_vf32 v_mn = ax_vf32_set1(mean);
            ax_vf32 v_zero = ax_vf32_zero();

            for (int64_t n = 0; n < N; n++) {
                int64_t base = n * C_out * spatial + c * spatial;
                if (record) {
                    int64_t m = 0, me = spatial - (spatial % AX_VF32_WIDTH);
                    for (; m < me; m += AX_VF32_WIDTH) {
                        ax_vf32 v_in = ax_vf32_loadu(od + base + m);
                        ax_vf32 xh = ax_vf32_mul(ax_vf32_sub(v_in, v_mn), v_is);
                        ax_vf32 bn = ax_vf32_fmadd(v_sc, v_in, v_bi);
                        ax_vf32_storeu(od + base + m, ax_vf32_max(bn, v_zero));
                        ax_vf32_storeu(xh_d + base + m, xh);
                    }
                    for (; m < spatial; m++) {
                        float conv = od[base + m];
                        float xh = (conv - mean) * inv_std;
                        float bn = scale * conv + bias_out;
                        od[base + m] = bn > 0.0f ? bn : 0.0f;
                        xh_d[base + m] = xh;
                    }
                } else {
                    int64_t m = 0, me = spatial - (spatial % AX_VF32_WIDTH);
                    for (; m < me; m += AX_VF32_WIDTH) {
                        ax_vf32 bn = ax_vf32_fmadd(v_sc, ax_vf32_loadu(od + base + m), v_bi);
                        ax_vf32_storeu(od + base + m, ax_vf32_max(bn, v_zero));
                    }
                    for (; m < spatial; m++) {
                        float bn = scale * od[base + m] + bias_out;
                        od[base + m] = bn > 0.0f ? bn : 0.0f;
                    }
                }
            }

            rm[c] = (1.0f - L->bn_momentum) * rm[c] + L->bn_momentum * mean;
            int64_t eff_count = (int64_t)eff_n;
            float unbiased_var = (eff_count > 1) ? var_sum / (float)(eff_count - 1) : var;
            rv[c] = (1.0f - L->bn_momentum) * rv[c] + L->bn_momentum * unbiased_var;
        }
    } else {
        /* eval path: use running stats, fused scale+shift+relu */
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
        #endif
        for (int64_t c = 0; c < C_out; c++) {
            float inv_std = 1.0f / sqrtf(rv[c] + L->bn_eps);
            float scale = gd[c] * inv_std;
            float bias_out = bnbeta[c] - gd[c] * rm[c] * inv_std;
            ax_vf32 v_sc = ax_vf32_set1(scale);
            ax_vf32 v_bi = ax_vf32_set1(bias_out);
            ax_vf32 v_zero = ax_vf32_zero();
            for (int64_t n = 0; n < N; n++) {
                int64_t base = n * C_out * spatial + c * spatial;
                int64_t m = 0, me = spatial - (spatial % AX_VF32_WIDTH);
                for (; m < me; m += AX_VF32_WIDTH) {
                    ax_vf32 bn = ax_vf32_fmadd(v_sc, ax_vf32_loadu(od + base + m), v_bi);
                    ax_vf32_storeu(od + base + m, ax_vf32_max(bn, v_zero));
                }
                for (; m < spatial; m++) {
                    float bn = scale * od[base + m] + bias_out;
                    od[base + m] = bn > 0.0f ? bn : 0.0f;
                }
            }
        }
    }

    if (record) {
        cbr_bwd_ctx_t *ctx = (cbr_bwd_ctx_t *)malloc(sizeof(cbr_bwd_ctx_t));
        if (!ctx) {
            ax_tensor_destroy(x_hat_save);
            ax_tensor_destroy(inv_std_save);
            if (inp != input) ax_tensor_destroy(inp);
            return output;
        }
        ctx->layer = L;
        ctx->N = N; ctx->C_out = C_out; ctx->out_h = out_h; ctx->out_w = out_w;
        ctx->C_in = C_in; ctx->H = H; ctx->W = W;

        output->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(conv_bn_relu_backward);
        gf->inputs[0] = input;
        gf->n_inputs = 1;
        gf->saved[0] = inp;                      /* contig input for conv bwd */
        gf->saved_owned[0] = (inp != input);
        gf->saved[1] = x_hat_save;
        gf->saved_owned[1] = true;
        gf->saved[2] = inv_std_save;
        gf->saved_owned[2] = true;
        gf->n_saved = 3;
        gf->ctx = ctx;
        gf->ctx_cleanup = cbr_ctx_cleanup;
        output->grad_fn = gf;
    } else {
        if (inp != input) ax_tensor_destroy(inp);
    }
    return output;
}

static void conv_bn_relu_destroy(ax_layer_t *self)
{
    ax_conv_bn_relu_t *L = (ax_conv_bn_relu_t *)self;
    if (L->scratch) scratch_destroy(L->scratch);
    if (L->weight) ax_tensor_destroy(L->weight);
    if (L->bias) ax_tensor_destroy(L->bias);
    if (L->gamma) ax_tensor_destroy(L->gamma);
    if (L->beta) ax_tensor_destroy(L->beta);
    if (L->running_mean) ax_tensor_destroy(L->running_mean);
    if (L->running_var) ax_tensor_destroy(L->running_var);
    free(L);
}

ax_layer_t *ax_conv_bn_relu_create_from(ax_layer_t *conv_base, ax_layer_t *bn_base)
{
    if (!conv_base || !bn_base) return NULL;
    if (conv_base->type != AX_LAYER_CONV2D || bn_base->type != AX_LAYER_BATCHNORM)
        return NULL;

    ax_conv2d_t *conv = (ax_conv2d_t *)conv_base;
    ax_batchnorm_t *bn = (ax_batchnorm_t *)bn_base;

    ax_conv_bn_relu_t *L = (ax_conv_bn_relu_t *)calloc(1, sizeof(ax_conv_bn_relu_t));
    if (!L) return NULL;

    L->base.ops.forward = conv_bn_relu_forward;
    L->base.ops.destroy = conv_bn_relu_destroy;
    L->base.type = AX_LAYER_CONV_BN_RELU;
    L->base.training = conv_base->training;
    L->base.input_features = conv_base->input_features;
    L->base.output_features = conv_base->output_features;

    /* steal conv tensors */
    L->weight = conv->weight;      conv->weight = NULL;
    L->bias = conv->bias;          conv->bias = NULL;
    L->scratch = conv->scratch;    conv->scratch = NULL;
    L->in_channels = conv->in_channels;
    L->out_channels = conv->out_channels;
    L->kernel_h = conv->kernel_h;  L->kernel_w = conv->kernel_w;
    L->stride_h = conv->stride_h;  L->stride_w = conv->stride_w;
    L->pad_h = conv->pad_h;        L->pad_w = conv->pad_w;
    L->use_bias = conv->use_bias;

    /* steal batchnorm tensors */
    L->gamma = bn->gamma;                 bn->gamma = NULL;
    L->beta = bn->beta;                   bn->beta = NULL;
    L->running_mean = bn->running_mean;   bn->running_mean = NULL;
    L->running_var = bn->running_var;     bn->running_var = NULL;
    L->bn_eps = bn->eps;
    L->bn_momentum = bn->momentum;

    /* expose params+buffers to ax_layer_get_params/buffers.
       layout: [weight, (bias?), gamma, beta], buffers: [running_mean, running_var] */
    L->base.params[0] = L->weight;
    int np = 1;
    if (L->use_bias && L->bias) L->base.params[np++] = L->bias;
    L->base.params[np++] = L->gamma;
    L->base.params[np++] = L->beta;
    L->base.n_params = np;

    L->base.buffers[0] = L->running_mean;
    L->base.buffers[1] = L->running_var;
    L->base.n_buffers = 2;

    return (ax_layer_t *)L;
}

ax_layer_t *ax_sequential_fuse(ax_layer_t *model)
{
    if (!model || model->type != AX_LAYER_SEQUENTIAL) return model;
    ax_sequential_t *seq = (ax_sequential_t *)model;

    int w = 0;
    int r = 0;
    while (r < seq->n_layers) {
        ax_layer_t *a = seq->layers[r];
        if (r + 2 < seq->n_layers &&
            a->type == AX_LAYER_CONV2D &&
            seq->layers[r+1]->type == AX_LAYER_BATCHNORM &&
            seq->layers[r+2]->type == AX_LAYER_RELU) {

            ax_layer_t *b = seq->layers[r+1];
            ax_layer_t *c = seq->layers[r+2];

            ax_layer_t *fused = ax_conv_bn_relu_create_from(a, b);
            if (!fused) {
                seq->layers[w++] = a;
                r++;
                continue;
            }

            /* free old layer shells. tensors were moved into the fused
               layer so the original destroy paths will see NULL fields. */
            a->ops.destroy(a);
            b->ops.destroy(b);
            c->ops.destroy(c);

            seq->layers[w++] = fused;
            r += 3;
        } else {
            seq->layers[w++] = a;
            r++;
        }
    }
    seq->n_layers = w;
    return model;
}


/* maxpool2d */

/* context for maxpool backward: stores input shape and pool params */
typedef struct {
    int64_t N, C, H, W;
    int k, s, p;
} pool_ctx_t;

static void maxpool2d_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *input = self->inputs[0];
    ax_tensor_t *indices = self->saved[0]; /* argmax linear indices into input spatial */

    if (!input->requires_grad) return;

    pool_ctx_t *ctx = (pool_ctx_t *)self->ctx;
    int64_t N = ctx->N, C = ctx->C, H = ctx->H, W = ctx->W;

    if (!input->grad)
        input->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
    if (!input->grad) { free(ctx); self->ctx = NULL; return; }

    int64_t oh = grad_out->shape[2], ow = grad_out->shape[3];
    float *ig = (float *)input->grad->storage->data;
    float *go = (float *)grad_out->storage->data;
    float *idx = (float *)indices->storage->data;
    int64_t NC = N * C;

    /* per-(n,c) writes are disjoint: each (n,c) writes only into its own
       channel slab of the input gradient. safe to parallelize. */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t nc = 0; nc < NC; nc++)
    {
        int64_t n = nc / C;
        int64_t c = nc % C;
        for (int64_t y = 0; y < oh; y++)
            for (int64_t x = 0; x < ow; x++) {
                int64_t out_idx = ((n * C + c) * oh + y) * ow + x;
                int64_t max_pos = (int64_t)idx[out_idx];
                if (max_pos >= 0) {
                    int64_t iy = max_pos / W;
                    int64_t ix = max_pos % W;
                    ig[((n * C + c) * H + iy) * W + ix] += go[out_idx];
                }
            }
    }
    free(ctx); self->ctx = NULL;
}

static ax_tensor_t *maxpool2d_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_maxpool2d_t *pool = (ax_maxpool2d_t *)self;

    if (input->ndim != 4)
    {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "maxpool2d expects [N, C, H, W]");
        return NULL;
    }

    ax_tensor_t *inp = ax_ensure_contiguous(input);
    if (!inp) return NULL;

    int64_t N = inp->shape[0];
    int64_t C = inp->shape[1];
    int64_t H = inp->shape[2];
    int64_t W = inp->shape[3];
    int k = pool->kernel_size, s = pool->stride, p = pool->padding;
    int64_t oh = conv_out_dim(H, k, s, p);
    int64_t ow = conv_out_dim(W, k, s, p);
    if (oh <= 0 || ow <= 0)
    {
        ax_err_set(AX_ERR_INVALID_SHAPE, "maxpool2d: invalid output dimensions");
        if (inp != input) ax_tensor_destroy(inp);
        return NULL;
    }

    int64_t out_shape[] = {N, C, oh, ow};
    ax_tensor_t *output = ax_tensor_zeros(out_shape, 4, AX_FLOAT32);
    if (!output) { if (inp != input) ax_tensor_destroy(inp); return NULL; }

    bool record = ax_grad_enabled() && input->requires_grad;
    ax_tensor_t *indices = NULL;
    float *idxd = NULL;
    if (record) {
        indices = ax_tensor_zeros(out_shape, 4, AX_FLOAT32);
        if (!indices) record = false;
        else idxd = (float *)indices->storage->data;
    }

    float *id = (float *)inp->storage->data;
    float *od = (float *)output->storage->data;

    /* parallelize over (n*C) — each (n,c) pair is independent.
       collapse(2) requires perfectly nested loops with no code between, so we use a fused index. */
    int64_t NC = N * C;

    /* fast path: k=2, s=2, p=0 (most common maxpool config).
       no boundary checks needed, unrolled 2x2 window comparison. */
    if (k == 2 && s == 2 && p == 0)
    {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t nc = 0; nc < NC; nc++)
        {
            int64_t n = nc / C;
            int64_t c = nc % C;
            const float *ic = id + (n * C + c) * H * W;
            float *oc = od + (n * C + c) * oh * ow;
            float *ix = record ? idxd + (n * C + c) * oh * ow : NULL;

            for (int64_t y = 0; y < oh; y++)
            {
                int64_t iy = y * 2;
                const float *row0 = ic + iy * W;
                const float *row1 = ic + (iy + 1) * W;

                for (int64_t x = 0; x < ow; x++)
                {
                    int64_t ix2 = x * 2;
                    float a = row0[ix2], b = row0[ix2 + 1];
                    float c2 = row1[ix2], d = row1[ix2 + 1];

                    float m01 = a > b ? a : b;
                    float m23 = c2 > d ? c2 : d;
                    float mx = m01 > m23 ? m01 : m23;
                    oc[y * ow + x] = mx;

                    if (record) {
                        int64_t mi;
                        if (mx == a)      mi = iy * W + ix2;
                        else if (mx == b) mi = iy * W + ix2 + 1;
                        else if (mx == c2) mi = (iy+1) * W + ix2;
                        else               mi = (iy+1) * W + ix2 + 1;
                        ix[y * ow + x] = (float)mi;
                    }
                }
            }
        }
    }
    else
    {
        /* general path with boundary checks */
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t nc = 0; nc < NC; nc++)
        {
            int64_t n = nc / C;
            int64_t c = nc % C;
            for (int64_t y = 0; y < oh; y++)
            {
                for (int64_t x = 0; x < ow; x++)
                {
                    float mx = -FLT_MAX;
                    int64_t max_iy = -1, max_ix = -1;
                    for (int ky = 0; ky < k; ky++)
                    {
                        for (int kx = 0; kx < k; kx++)
                        {
                            int64_t iy = y * s - p + ky;
                            int64_t ix2 = x * s - p + kx;
                            if (iy >= 0 && iy < H && ix2 >= 0 && ix2 < W)
                            {
                                float v = id[((n * C + c) * H + iy) * W + ix2];
                                if (v > mx) { mx = v; max_iy = iy; max_ix = ix2; }
                            }
                        }
                    }
                    int64_t oi = ((n * C + c) * oh + y) * ow + x;
                    od[oi] = mx;
                    if (record)
                        idxd[oi] = (max_iy >= 0) ? (float)(max_iy * W + max_ix) : -1.0f;
                }
            }
        }
    }

    if (inp != input) ax_tensor_destroy(inp);

    if (record) {
        pool_ctx_t *ctx = malloc(sizeof(pool_ctx_t));
        if (!ctx) { ax_tensor_destroy(indices); return output; }
        ctx->N = N; ctx->C = C; ctx->H = H; ctx->W = W;
        ctx->k = k; ctx->s = s; ctx->p = p;

        ax_grad_fn_t *gf = ax_grad_fn_create(maxpool2d_backward);
        gf->inputs[0] = input;
        gf->n_inputs = 1;
        gf->saved[0] = indices;
        gf->saved_owned[0] = true; /* we created the indices tensor */
        gf->n_saved = 1;
        gf->ctx = ctx;
        gf->ctx_cleanup = (void(*)(void*))free;
        output->requires_grad = true;
        output->grad_fn = gf;
    }

    return output;
}

static void pool_destroy(ax_layer_t *self) { free(self); }

ax_layer_t *ax_maxpool2d_create(int kernel_size, int stride, int padding)
{
    ax_maxpool2d_t *p = calloc(1, sizeof(ax_maxpool2d_t));
    if (!p) return NULL;
    p->base.ops.forward = maxpool2d_forward;
    p->base.ops.destroy = pool_destroy;
    p->base.type = AX_LAYER_MAXPOOL2D;
    p->base.training = true;
    p->kernel_size = kernel_size;
    p->stride = stride;
    p->padding = padding;
    return (ax_layer_t *)p;
}


/* avgpool2d */

static void avgpool2d_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *input = self->inputs[0];
    if (!input->requires_grad) { free(self->ctx); self->ctx = NULL; return; }

    pool_ctx_t *ctx = (pool_ctx_t *)self->ctx;
    int64_t N = ctx->N, C = ctx->C, H = ctx->H, W = ctx->W;
    int k = ctx->k, s = ctx->s, p = ctx->p;
    int64_t oh = grad_out->shape[2], ow = grad_out->shape[3];

    if (!input->grad)
        input->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
    if (!input->grad) { free(ctx); self->ctx = NULL; return; }

    float *ig = (float *)input->grad->storage->data;
    float *go = (float *)grad_out->storage->data;
    int64_t NC = N * C;

    /* per-(n,c) writes are disjoint: each (n,c) writes only into its own
       channel slab. safe to parallelize over (n,c). */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t nc = 0; nc < NC; nc++)
    {
        int64_t n = nc / C;
        int64_t c = nc % C;
        for (int64_t y = 0; y < oh; y++)
            for (int64_t x = 0; x < ow; x++) {
                int count = 0;
                for (int ky = 0; ky < k; ky++)
                    for (int kx = 0; kx < k; kx++) {
                        int64_t iy = y * s - p + ky;
                        int64_t ix = x * s - p + kx;
                        if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                            count++;
                    }
                if (count == 0) continue;
                float g = go[((n * C + c) * oh + y) * ow + x] / (float)count;
                for (int ky = 0; ky < k; ky++)
                    for (int kx = 0; kx < k; kx++) {
                        int64_t iy = y * s - p + ky;
                        int64_t ix = x * s - p + kx;
                        if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                            ig[((n * C + c) * H + iy) * W + ix] += g;
                    }
            }
    }
    free(ctx); self->ctx = NULL;
}

static ax_tensor_t *avgpool2d_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_avgpool2d_t *pool = (ax_avgpool2d_t *)self;

    if (input->ndim != 4) return NULL;

    ax_tensor_t *inp = ax_ensure_contiguous(input);
    if (!inp) return NULL;

    int64_t N = inp->shape[0], C = inp->shape[1];
    int64_t H = inp->shape[2], W = inp->shape[3];
    int k = pool->kernel_size, s = pool->stride, p = pool->padding;
    int64_t oh = conv_out_dim(H, k, s, p);
    int64_t ow = conv_out_dim(W, k, s, p);
    if (oh <= 0 || ow <= 0)
    {
        ax_err_set(AX_ERR_INVALID_SHAPE, "avgpool2d: invalid output dimensions");
        if (inp != input) ax_tensor_destroy(inp);
        return NULL;
    }

    int64_t out_shape[] = {N, C, oh, ow};
    ax_tensor_t *output = ax_tensor_zeros(out_shape, 4, AX_FLOAT32);
    if (!output) { if (inp != input) ax_tensor_destroy(inp); return NULL; }

    float *id = (float *)inp->storage->data;
    float *od = (float *)output->storage->data;
    int64_t NC = N * C;

    /* parallelize over (n,c) — disjoint output regions */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t nc = 0; nc < NC; nc++)
    {
        int64_t n = nc / C;
        int64_t c = nc % C;
        for (int64_t y = 0; y < oh; y++)
        {
            for (int64_t x = 0; x < ow; x++)
            {
                float sum = 0;
                int count = 0;
                for (int ky = 0; ky < k; ky++)
                {
                    for (int kx = 0; kx < k; kx++)
                    {
                        int64_t iy = y * s - p + ky;
                        int64_t ix = x * s - p + kx;
                        if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                        {
                            sum += id[((n * C + c) * H + iy) * W + ix];
                            count++;
                        }
                    }
                }
                od[((n * C + c) * oh + y) * ow + x] = count > 0 ? sum / (float)count : 0.0f;
            }
        }
    }

    if (inp != input) ax_tensor_destroy(inp);

    if (ax_grad_enabled() && input->requires_grad) {
        pool_ctx_t *ctx = malloc(sizeof(pool_ctx_t));
        if (ctx) {
            ctx->N = N; ctx->C = C; ctx->H = H; ctx->W = W;
            ctx->k = k; ctx->s = s; ctx->p = p;

            ax_grad_fn_t *gf = ax_grad_fn_create(avgpool2d_backward);
            gf->inputs[0] = input;
            gf->n_inputs = 1;
            gf->ctx = ctx;
            gf->ctx_cleanup = (void(*)(void*))free;
            output->requires_grad = true;
            output->grad_fn = gf;
        }
    }

    return output;
}

ax_layer_t *ax_avgpool2d_create(int kernel_size, int stride, int padding)
{
    ax_avgpool2d_t *p = calloc(1, sizeof(ax_avgpool2d_t));
    if (!p) return NULL;
    p->base.ops.forward = avgpool2d_forward;
    p->base.ops.destroy = pool_destroy;
    p->base.type = AX_LAYER_AVGPOOL2D;
    p->base.training = true;
    p->kernel_size = kernel_size;
    p->stride = stride;
    p->padding = padding;
    return (ax_layer_t *)p;
}


/* global average pool: [N, C, H, W] -> [N, C] */

/* context for global avgpool backward */
typedef struct {
    int64_t N, C, H, W;
} gap_ctx_t;

static void global_avgpool_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *input = self->inputs[0];
    if (!input->requires_grad) { free(self->ctx); self->ctx = NULL; return; }

    gap_ctx_t *ctx = (gap_ctx_t *)self->ctx;
    int64_t N = ctx->N, C = ctx->C, H = ctx->H, W = ctx->W;
    int64_t spatial = H * W;

    if (!input->grad)
        input->grad = ax_tensor_zeros(input->shape, input->ndim, input->dtype);
    if (!input->grad) { free(ctx); self->ctx = NULL; return; }

    float *ig = (float *)input->grad->storage->data;
    float *go = (float *)grad_out->storage->data;

    for (int64_t n = 0; n < N; n++)
        for (int64_t c = 0; c < C; c++) {
            float g = go[n * C + c] / (float)spatial;
            for (int64_t i = 0; i < spatial; i++)
                ig[(n * C + c) * spatial + i] += g;
        }
    free(ctx); self->ctx = NULL;
}

static ax_tensor_t *global_avgpool_forward(ax_layer_t *self, ax_tensor_t *input)
{
    if (input->ndim != 4) return NULL;

    ax_tensor_t *inp = ax_ensure_contiguous(input);
    if (!inp) return NULL;

    int64_t N = inp->shape[0], C = inp->shape[1];
    int64_t H = inp->shape[2], W = inp->shape[3];
    int64_t spatial = H * W;

    int64_t out_shape[] = {N, C};
    ax_tensor_t *output = ax_tensor_zeros(out_shape, 2, AX_FLOAT32);
    if (!output) { if (inp != input) ax_tensor_destroy(inp); return NULL; }

    float *id = (float *)inp->storage->data;
    float *od = (float *)output->storage->data;

    for (int64_t n = 0; n < N; n++)
    {
        for (int64_t c = 0; c < C; c++)
        {
            float sum = 0;
            for (int64_t i = 0; i < spatial; i++)
                sum += id[(n * C + c) * spatial + i];
            od[n * C + c] = sum / (float)spatial;
        }
    }

    if (inp != input) ax_tensor_destroy(inp);

    if (ax_grad_enabled() && input->requires_grad) {
        gap_ctx_t *ctx = malloc(sizeof(gap_ctx_t));
        if (ctx) {
            ctx->N = N; ctx->C = C; ctx->H = H; ctx->W = W;

            ax_grad_fn_t *gf = ax_grad_fn_create(global_avgpool_backward);
            gf->inputs[0] = input;
            gf->n_inputs = 1;
            gf->ctx = ctx;
            gf->ctx_cleanup = (void(*)(void*))free;
            output->requires_grad = true;
            output->grad_fn = gf;
        }
    }

    return output;
}

ax_layer_t *ax_global_avgpool2d_create(void)
{
    ax_layer_t *l = calloc(1, sizeof(ax_layer_t));
    if (!l) return NULL;
    l->ops.forward = global_avgpool_forward;
    l->ops.destroy = (void (*)(ax_layer_t *))free;
    l->type = AX_LAYER_GLOBAL_AVGPOOL2D;
    l->training = true;
    return l;
}


/* flatten: [N, C, H, W] -> [N, C*H*W] */

/* context for flatten backward: original shape */
typedef struct {
    int64_t orig_shape[AX_MAX_DIMS];
    int orig_ndim;
} flatten_ctx_t;

static void flatten_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *input = self->inputs[0];
    if (!input->requires_grad) { free(self->ctx); self->ctx = NULL; return; }

    flatten_ctx_t *ctx = (flatten_ctx_t *)self->ctx;

    if (!input->grad)
        input->grad = ax_tensor_zeros(ctx->orig_shape, ctx->orig_ndim, input->dtype);
    if (!input->grad) { free(ctx); self->ctx = NULL; return; }

    /* just copy grad_out (which is [N, flat]) into input->grad (which is [N, C, H, W]) */
    int64_t n = ax_tensor_numel(grad_out);
    float *ig = (float *)input->grad->storage->data;
    float *go = (float *)grad_out->storage->data;
    for (int64_t i = 0; i < n; i++)
        ig[i] += go[i];

    free(ctx); self->ctx = NULL;
}

static ax_tensor_t *flatten_forward(ax_layer_t *self, ax_tensor_t *input)
{
    if (input->ndim < 2) return NULL;

    int64_t N = input->shape[0];
    int64_t flat = 1;
    for (int d = 1; d < input->ndim; d++)
        flat *= input->shape[d];

    int64_t out_shape[] = {N, flat};

    bool record = ax_grad_enabled() && input->requires_grad;

    /* We need a real copy for autograd so the output is a distinct tensor */
    ax_tensor_t *output;
    if (!record && ax_tensor_is_contiguous(input))
        return ax_tensor_reshape(input, out_shape, 2);

    /* make a contiguous copy reshaped to [N, flat] */
    output = ax_tensor_zeros(out_shape, 2, input->dtype);
    if (!output) return NULL;
    int64_t n = ax_tensor_numel(input);
    float *od = (float *)output->storage->data;
    float *id = (float *)input->storage->data;
    /* must respect strides for non-contiguous inputs (e.g., after transpose) */
    for (int64_t i = 0; i < n; i++)
    {
        int64_t remaining = i;
        int64_t src_offset = input->offset;
        for (int d = input->ndim - 1; d >= 0; d--)
        {
            int64_t idx = remaining % input->shape[d];
            remaining /= input->shape[d];
            src_offset += idx * input->strides[d];
        }
        od[i] = id[src_offset];
    }

    if (record) {
        flatten_ctx_t *ctx = malloc(sizeof(flatten_ctx_t));
        if (ctx) {
            ctx->orig_ndim = input->ndim;
            for (int d = 0; d < input->ndim; d++)
                ctx->orig_shape[d] = input->shape[d];

            ax_grad_fn_t *gf = ax_grad_fn_create(flatten_backward);
            gf->inputs[0] = input;
            gf->n_inputs = 1;
            gf->ctx = ctx;
        gf->ctx_cleanup = (void(*)(void*))free;
            output->requires_grad = true;
            output->grad_fn = gf;
        }
    }

    return output;
}

ax_layer_t *ax_flatten_create(void)
{
    ax_layer_t *l = calloc(1, sizeof(ax_layer_t));
    if (!l) return NULL;
    l->ops.forward = flatten_forward;
    l->ops.destroy = (void (*)(ax_layer_t *))free;
    l->type = AX_LAYER_FLATTEN;
    l->training = true;
    return l;
}
