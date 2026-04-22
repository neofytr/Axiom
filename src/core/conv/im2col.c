/* conv/im2col.c — im2col / col2im transforms (NCHW + NHWC variants).

   k.2 split: extracted from src/core/conv.c. im2col is the "lower
   convolution to GEMM" trick — for each output position, the patch
   of the input that the kernel sees is laid out as a column of an
   intermediate matrix; the whole conv collapses to one
   weight_matrix @ im2col(input) call, with the resulting GEMM going
   through the same highly-tuned dispatch path as a plain matmul.

   the cost is the staging memory (the columns matrix is K*M floats per
   sample) plus the gather pass to write it. when both K and M are
   large the gather is amortised over the GEMM's compute; when M is
   small or stride > 1, the various fast paths in im2col_into +
   im2col_into_strided cover the common shapes (1x1 stride-1 pad-0 is
   a single memcpy; stride-1 width is row-memcpy with pad memsets;
   AVX2 stride-2 uses a shuffle+permute to extract every-other
   column). col2im is the transpose of the gather — used in backward
   to scatter dcol back into dX with overlapping-window accumulation.

   NHWC variants are simpler — each "column" is just kh*kw blocks of
   Cin contiguous floats, so the gather is plain memcpy with no
   per-channel stride work. */

#include "internal.h"
#include "axiom/error.h"
#include "../../compute/backends/simd_defs.h"
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* im2col: for a single image [C, H, W], produce a matrix
   [C*kh*kw, out_h*out_w] where each column is a flattened patch */

/* internal: write im2col into a pre-allocated float buffer.
   input must be contiguous [C, H, W]. cd must be at least K*M floats. */
void ax_conv_im2col_into(const float *id, int64_t C, int64_t H, int64_t W,
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

    /* stride_w == 2 SIMD fast path: extract every-other input column via
       shuffle. for each block of 8 outputs, load 16 contiguous input floats
       and pack the even-indexed ones into one SIMD register. closes the
       gap on stride-2 conv shapes (e.g. conv_64x112_128_s2) where the
       scalar fallback was a measured bottleneck. */
#if defined(AX_SIMD_AVX2) && AX_VF32_WIDTH == 8
    if (stride_w == 2)
    {
        const __m256i extract_evens = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
        for (int64_t c = 0; c < C; c++) {
            for (int ky = 0; ky < kh; ky++) {
                for (int kx = 0; kx < kw; kx++) {
                    const int64_t iw_start = (int64_t)kx - (int64_t)pad_w;
                    for (int64_t oh = 0; oh < out_h; oh++) {
                        const int64_t ih = oh * stride_h - pad_h + ky;
                        float *dst_row = cd + col_idx * M + oh * out_w;
                        if (ih < 0 || ih >= H) {
                            memset(dst_row, 0, (size_t)out_w * sizeof(float));
                            continue;
                        }
                        const float *src_row = id + c * H * W + ih * W;
                        int64_t ow = 0;
                        for (; ow + 8 <= out_w; ow += 8) {
                            int64_t iw_first = iw_start + 2 * ow;
                            if (iw_first >= 0 && iw_first + 16 <= W) {
                                __m256 a = _mm256_loadu_ps(src_row + iw_first);
                                __m256 b = _mm256_loadu_ps(src_row + iw_first + 8);
                                __m256 t = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
                                __m256 r = _mm256_permutevar8x32_ps(t, extract_evens);
                                _mm256_storeu_ps(dst_row + ow, r);
                            } else {
                                for (int64_t b8 = 0; b8 < 8; b8++) {
                                    int64_t iw = iw_start + 2 * (ow + b8);
                                    dst_row[ow + b8] = (iw >= 0 && iw < W) ? src_row[iw] : 0.0f;
                                }
                            }
                        }
                        for (; ow < out_w; ow++) {
                            int64_t iw = iw_start + 2 * ow;
                            dst_row[ow] = (iw >= 0 && iw < W) ? src_row[iw] : 0.0f;
                        }
                    }
                    col_idx++;
                }
            }
        }
        return;
    }
#endif

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
void ax_conv_im2col_into_strided(const float *id, int64_t C, int64_t H, int64_t W,
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
    int64_t out_h = ax_conv_out_dim(H, kh, stride_h, pad_h);
    int64_t out_w = ax_conv_out_dim(W, kw, stride_w, pad_w);
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

    ax_conv_im2col_into(id, C, H, W, kh, kw, stride_h, stride_w, pad_h, pad_w, out_h, out_w, cd);
    return cols;
}

/* internal: scatter col2im into a pre-zeroed float buffer.
   id must be zeroed before calling (accumulates into it). */
void ax_conv_col2im_into(const float *cd, int64_t channels,
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
    int64_t out_h = ax_conv_out_dim(height, kh, stride_h, pad_h);
    int64_t out_w = ax_conv_out_dim(width, kw, stride_w, pad_w);
    if (out_h <= 0 || out_w <= 0)
    {
        ax_err_set(AX_ERR_INVALID_SHAPE, "col2im: invalid output dimensions %" PRId64 " x %" PRId64, out_h, out_w);
        return NULL;
    }

    int64_t img_shape[] = {channels, height, width};
    ax_tensor_t *img = ax_tensor_zeros(img_shape, 3, AX_FLOAT32);
    if (!img) return NULL;

    ax_conv_col2im_into((float *)cols->storage->data, channels, height, width,
                kh, kw, stride_h, stride_w, pad_h, pad_w, out_h, out_w,
                (float *)img->storage->data);
    return img;
}

/* im2col for NHWC: input [N, H, W, Cin] → cd [N*OH*OW, Cin*kh*kw].
   per output row, write kh*kw blocks of Cin contiguous floats.
   memcpy when in-bounds, memset(0) for the pad regions. */
void ax_conv_im2col_nhwc(const float *id, int64_t N, int64_t H, int64_t W, int64_t C_in,
                          int kh, int kw, int sh, int sw, int ph, int pw,
                          int64_t out_h, int64_t out_w, float *cd)
{
    int64_t HWC  = H * W * C_in;
    int64_t K    = (int64_t)kh * (int64_t)kw * C_in;
    int64_t OH_OW = out_h * out_w;
    size_t  Cin_bytes = (size_t)C_in * sizeof(float);

    int64_t total_rows = N * OH_OW;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t row = 0; row < total_rows; row++) {
        int64_t n  = row / OH_OW;
        int64_t r  = row - n * OH_OW;
        int64_t oy = r / out_w;
        int64_t ox = r - oy * out_w;

        const float *in_n = id + n * HWC;
        float *cd_row = cd + row * K;
        int64_t patch_off = 0;

        for (int ky = 0; ky < kh; ky++) {
            int64_t iy = oy * sh - ph + ky;
            int iy_valid = (iy >= 0 && iy < H);
            for (int kx = 0; kx < kw; kx++) {
                int64_t ix = ox * sw - pw + kx;
                if (iy_valid && ix >= 0 && ix < W) {
                    memcpy(cd_row + patch_off,
                           in_n + (iy * W + ix) * C_in,
                           Cin_bytes);
                } else {
                    memset(cd_row + patch_off, 0, Cin_bytes);
                }
                patch_off += C_in;
            }
        }
    }
}

/* col2im for NHWC: cd [N*OH*OW, Cin*kh*kw] → id [N, H, W, Cin] (accumulating).
   id MUST be pre-zeroed. each row of cd contains the gradients for the
   patches that contributed to one output position. */
void ax_conv_col2im_nhwc(const float *cd, int64_t N, int64_t H, int64_t W, int64_t C_in,
                          int kh, int kw, int sh, int sw, int ph, int pw,
                          int64_t out_h, int64_t out_w, float *id)
{
    int64_t HWC = H * W * C_in;
    int64_t K   = (int64_t)kh * (int64_t)kw * C_in;
    int64_t OH_OW = out_h * out_w;

    /* per-sample accumulation can race across (oy, ox) for overlapping
       windows. parallelize over n only (samples are disjoint). */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t n = 0; n < N; n++) {
        const float *cd_n = cd + n * OH_OW * K;
        float *id_n = id + n * HWC;
        for (int64_t oy = 0; oy < out_h; oy++) {
            for (int64_t ox = 0; ox < out_w; ox++) {
                const float *cd_row = cd_n + (oy * out_w + ox) * K;
                int64_t patch_off = 0;
                for (int ky = 0; ky < kh; ky++) {
                    int64_t iy = oy * sh - ph + ky;
                    int iy_valid = (iy >= 0 && iy < H);
                    for (int kx = 0; kx < kw; kx++) {
                        int64_t ix = ox * sw - pw + kx;
                        if (iy_valid && ix >= 0 && ix < W) {
                            float *id_pos = id_n + (iy * W + ix) * C_in;
                            const float *cd_pos = cd_row + patch_off;
                            int64_t ci = 0;
                            int64_t ve = C_in - (C_in % AX_VF32_WIDTH);
                            for (; ci < ve; ci += AX_VF32_WIDTH)
                                ax_vf32_storeu(id_pos + ci,
                                               ax_vf32_add(ax_vf32_loadu(id_pos + ci),
                                                            ax_vf32_loadu(cd_pos + ci)));
                            for (; ci < C_in; ci++) id_pos[ci] += cd_pos[ci];
                        }
                        patch_off += C_in;
                    }
                }
            }
        }
    }
}
