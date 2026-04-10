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
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include <float.h>

/* compute output spatial dimension.
   returns -1 if the result would be non-positive (invalid configuration). */
static inline int64_t conv_out_dim(int64_t in_dim, int kernel, int stride, int pad)
{
    if (stride <= 0) return -1;
    int64_t out = (in_dim + 2 * pad - kernel) / stride + 1;
    if (out <= 0) return -1;
    return out;
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
    int64_t col_idx = 0;
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

                        cd[col_idx * out_h * out_w + oh * out_w + ow] = val;
                    }
                }
                col_idx++;
            }
        }
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
    for (int64_t c = 0; c < channels; c++)
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
                        if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                            id[c * height * width + ih * width + iw] +=
                                cd[col_idx * out_h * out_w + oh * out_w + ow];
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

    /* bias gradient: sum grad_out over N, H, W */
    if (conv->use_bias && conv->bias && conv->bias->requires_grad)
    {
        if (!conv->bias->grad)
            conv->bias->grad = ax_tensor_zeros(conv->bias->shape, conv->bias->ndim, conv->bias->dtype);

        float *bg = (float *)conv->bias->grad->storage->data;
        float *gd = (float *)grad_out->storage->data;

        for (int64_t n = 0; n < N; n++)
            for (int64_t c = 0; c < C_out; c++)
                for (int64_t h = 0; h < out_h; h++)
                    for (int64_t w = 0; w < out_w; w++)
                        bg[c] += gd[((n * C_out + c) * out_h + h) * out_w + w];
    }

    float *wdata = (float *)weight->storage->data;
    int64_t K = C_in * kh * kw;
    int64_t M = out_h * out_w;

    /* pre-compute W^T once outside the loop (weights don't change during backward) */
    ax_tensor_t *wt_contig = NULL;
    if (input_orig->requires_grad)
    {
        int64_t w2d_shape[] = {C_out, K};
        ax_tensor_t *w2d = ax_tensor_create(w2d_shape, 2, AX_FLOAT32);
        if (w2d) {
            memcpy(w2d->storage->data, wdata, (size_t)(C_out * K) * sizeof(float));
            ax_tensor_t *w2d_t = ax_tensor_transpose(w2d, 0, 1);
            wt_contig = ax_tensor_contiguous(w2d_t);
            ax_tensor_destroy(w2d_t);
            ax_tensor_destroy(w2d);
        }
    }

    /* pre-allocate all reusable scratch tensors outside the loop */
    int64_t go_shape[] = {C_out, M};
    int64_t col_shape[] = {K, M};
    int64_t dw_shape[] = {C_out, K};
    int64_t dcol_shape[] = {K, M};
    int64_t colt_shape[] = {M, K};  /* transposed col for weight grad */

    int64_t img_shape[] = {C_in, H, W};
    ax_tensor_t *col_buf  = ax_tensor_create(col_shape, 2, AX_FLOAT32);
    ax_tensor_t *go_mat   = ax_tensor_create(go_shape, 2, AX_FLOAT32);
    ax_tensor_t *dw_buf   = weight->requires_grad ? ax_tensor_create(dw_shape, 2, AX_FLOAT32) : NULL;
    ax_tensor_t *dcol_buf = input_orig->requires_grad ? ax_tensor_create(dcol_shape, 2, AX_FLOAT32) : NULL;
    ax_tensor_t *colt_buf = weight->requires_grad ? ax_tensor_create(colt_shape, 2, AX_FLOAT32) : NULL;
    ax_tensor_t *dimg_buf = input_orig->requires_grad ? ax_tensor_create(img_shape, 3, AX_FLOAT32) : NULL;

    float *ind = (float *)input_data->storage->data;
    float *grd = (float *)grad_out->storage->data;

    for (int64_t n = 0; n < N; n++)
    {
        /* im2col directly into pre-allocated buffer */
        float *cbd = (float *)col_buf->storage->data;
        im2col_into(ind + n * C_in * H * W, C_in, H, W,
                     kh, kw, sh, sw, ph, pw, out_h, out_w, cbd);

        /* fill pre-allocated grad_out matrix */
        memcpy(go_mat->storage->data, grd + n * C_out * M,
               (size_t)(C_out * M) * sizeof(float));

        /* weight gradient: dW += go_mat @ col^T */
        if (weight->requires_grad && dw_buf && colt_buf)
        {
            /* explicit transpose into pre-allocated buffer (no alloc) */
            float *ctd = (float *)colt_buf->storage->data;
            for (int64_t r = 0; r < K; r++)
                for (int64_t c = 0; c < M; c++)
                    ctd[c * K + r] = cbd[r * M + c];

            memset(dw_buf->storage->data, 0, (size_t)(C_out * K) * sizeof(float));
            ax_compute_gemm(go_mat, colt_buf, dw_buf);

            float *wg = (float *)weight->grad->storage->data;
            float *dwd = (float *)dw_buf->storage->data;
            int64_t wn = C_out * K;
            {
                int64_t wi = 0, wve = wn - (wn % AX_VF32_WIDTH);
                for (; wi < wve; wi += AX_VF32_WIDTH)
                    ax_vf32_store(wg + wi, ax_vf32_add(ax_vf32_load(wg + wi), ax_vf32_load(dwd + wi)));
                for (; wi < wn; wi++) wg[wi] += dwd[wi];
            }
        }

        /* input gradient: dcol = W^T @ go_mat, then col2im into pre-allocated buffer */
        if (input_orig->requires_grad && wt_contig && dcol_buf && dimg_buf)
        {
            memset(dcol_buf->storage->data, 0, (size_t)(K * M) * sizeof(float));
            ax_compute_gemm(wt_contig, go_mat, dcol_buf);

            /* col2im into pre-allocated buffer (no per-sample malloc) */
            float *dimg_d = (float *)dimg_buf->storage->data;
            memset(dimg_d, 0, (size_t)(C_in * H * W) * sizeof(float));
            col2im_into((float *)dcol_buf->storage->data, C_in, H, W,
                         kh, kw, sh, sw, ph, pw, out_h, out_w, dimg_d);

            /* SIMD accumulate into input gradient */
            float *ig = (float *)input_orig->grad->storage->data + n * C_in * H * W;
            int64_t total = C_in * H * W;
            int64_t i = 0, ve = total - (total % AX_VF32_WIDTH);
            for (; i < ve; i += AX_VF32_WIDTH)
                ax_vf32_store(ig + i, ax_vf32_add(ax_vf32_load(ig + i), ax_vf32_load(dimg_d + i)));
            for (; i < total; i++)
                ig[i] += dimg_d[i];
        }
    }

    ax_tensor_destroy(col_buf);
    ax_tensor_destroy(go_mat);
    if (dw_buf) ax_tensor_destroy(dw_buf);
    if (dcol_buf) ax_tensor_destroy(dcol_buf);
    if (colt_buf) ax_tensor_destroy(colt_buf);
    if (dimg_buf) ax_tensor_destroy(dimg_buf);
    if (wt_contig) ax_tensor_destroy(wt_contig);
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
    ax_tensor_t *output = ax_tensor_zeros(out_shape, 4, AX_FLOAT32);
    if (!output) { if (inp != input) ax_tensor_destroy(inp); return NULL; }

    float *od = (float *)output->storage->data;
    float *wd = (float *)conv->weight->storage->data;

    /* pre-allocate weight as 2D tensor once (weights don't change during forward) */
    int64_t w2d_shape[] = {C_out, K};
    ax_tensor_t *w2d = ax_tensor_create(w2d_shape, 2, AX_FLOAT32);
    if (!w2d) { if (inp != input) ax_tensor_destroy(inp); ax_tensor_destroy(output); return NULL; }
    memcpy(w2d->storage->data, wd, (size_t)(C_out * K) * sizeof(float));

    /* pre-allocate reusable scratch: col buffer, result buffer */
    int64_t col_shape[] = {K, M};
    int64_t res_shape[] = {C_out, M};
    ax_tensor_t *col = ax_tensor_create(col_shape, 2, AX_FLOAT32);
    ax_tensor_t *res = ax_tensor_create(res_shape, 2, AX_FLOAT32);
    if (!col || !res) {
        ax_tensor_destroy(w2d);
        if (col) ax_tensor_destroy(col);
        if (res) ax_tensor_destroy(res);
        if (inp != input) ax_tensor_destroy(inp);
        ax_tensor_destroy(output); return NULL;
    }
    float *ind = (float *)inp->storage->data;
    float *cd = (float *)col->storage->data;

    for (int64_t n = 0; n < N; n++)
    {
        /* im2col directly into pre-allocated buffer (no per-sample malloc) */
        im2col_into(ind + n * C_in * H * W, C_in, H, W,
                     kh, kw, sh, sw, ph, pw, out_h, out_w, cd);

        /* result = w2d @ col via dispatch GEMM */
        memset(res->storage->data, 0, (size_t)(C_out * M) * sizeof(float));
        ax_compute_gemm(w2d, col, res);

        /* copy to output with bias */
        float *rd = (float *)res->storage->data;
        for (int64_t co = 0; co < C_out; co++)
        {
            float bias_val = (conv->use_bias && conv->bias)
                ? ((float *)conv->bias->storage->data)[co] : 0.0f;
            for (int64_t m = 0; m < M; m++)
            {
                od[((n * C_out + co) * out_h + m / out_w) * out_w + m % out_w]
                    = rd[co * M + m] + bias_val;
            }
        }
    }
    ax_tensor_destroy(w2d);
    ax_tensor_destroy(col);
    ax_tensor_destroy(res);

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

    for (int64_t n = 0; n < N; n++)
        for (int64_t c = 0; c < C; c++)
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

    /* fast path: k=2, s=2, p=0 (most common maxpool config).
       no boundary checks needed, unrolled 2x2 window comparison. */
    if (k == 2 && s == 2 && p == 0)
    {
        for (int64_t n = 0; n < N; n++)
        {
            for (int64_t c = 0; c < C; c++)
            {
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

                        /* branchless 4-way max */
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
    }
    else
    {
        /* general path with boundary checks */
        for (int64_t n = 0; n < N; n++)
        {
            for (int64_t c = 0; c < C; c++)
            {
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

    for (int64_t n = 0; n < N; n++)
        for (int64_t c = 0; c < C; c++)
            for (int64_t y = 0; y < oh; y++)
                for (int64_t x = 0; x < ow; x++) {
                    /* count valid positions in this window */
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

    for (int64_t n = 0; n < N; n++)
    {
        for (int64_t c = 0; c < C; c++)
        {
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
                    od[((n * C + c) * oh + y) * ow + x] = count > 0 ? sum / count : 0;
                }
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
