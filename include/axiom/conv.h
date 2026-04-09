/* axiom/conv.h — convolutional and pooling layers.

   convolutions slide a small filter (kernel) across an image,
   computing dot products at each position. this exploits spatial
   locality: nearby pixels are related, so a small local detector
   is more efficient than a full dense connection.

   data layout is NCHW throughout:
     N = batch size
     C = channels (e.g. 3 for rgb)
     H = height
     W = width

   NCHW is standard for most frameworks and more cache-friendly
   for convolution. on embedded you'd typically have a single
   image (N=1) so the batch overhead is negligible. */

#ifndef AX_CONV_H
#define AX_CONV_H

#include "layer.h"

/* conv2d layer.
   input:  [N, C_in, H, W]
   output: [N, C_out, H_out, W_out]

   H_out = (H + 2*padding - kernel_size) / stride + 1
   W_out = (W + 2*padding - kernel_size) / stride + 1

   weight shape: [C_out, C_in, kernel_h, kernel_w]
   bias shape:   [C_out] */

typedef struct
{
    ax_layer_t base;
    ax_tensor_t *weight;    /* [out_channels, in_channels, kh, kw] */
    ax_tensor_t *bias;      /* [out_channels] or NULL */
    int in_channels;
    int out_channels;
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    int pad_h, pad_w;
    bool use_bias;
} ax_conv2d_t;

ax_layer_t *ax_conv2d_create(int in_channels, int out_channels,
                              int kernel_size, int stride, int padding,
                              bool use_bias);

/* also support non-square kernels */
ax_layer_t *ax_conv2d_create_ex(int in_channels, int out_channels,
                                 int kernel_h, int kernel_w,
                                 int stride_h, int stride_w,
                                 int pad_h, int pad_w,
                                 bool use_bias);


/* maxpool2d: takes the max value in each kernel-sized window.
   reduces spatial dimensions by stride factor.
   input:  [N, C, H, W]
   output: [N, C, H_out, W_out] */

typedef struct
{
    ax_layer_t base;
    int kernel_size;
    int stride;
    int padding;
    ax_tensor_t *indices;  /* saved argmax positions for backward */
} ax_maxpool2d_t;

ax_layer_t *ax_maxpool2d_create(int kernel_size, int stride, int padding);


/* avgpool2d: average of values in each window */

typedef struct
{
    ax_layer_t base;
    int kernel_size;
    int stride;
    int padding;
} ax_avgpool2d_t;

ax_layer_t *ax_avgpool2d_create(int kernel_size, int stride, int padding);


/* global average pooling: average each channel down to a single value.
   input:  [N, C, H, W]
   output: [N, C] */

ax_layer_t *ax_global_avgpool2d_create(void);


/* flatten: collapses spatial dims into a single dimension.
   input:  [N, C, H, W]
   output: [N, C*H*W]
   needed to connect conv layers to dense layers. */

ax_layer_t *ax_flatten_create(void);


/* im2col: internal function exposed for testing.
   rearranges image patches into columns for matrix multiplication.
   this is the key trick that makes convolutions fast on cpu:
   conv becomes a single gemm call. */

ax_tensor_t *ax_im2col(ax_tensor_t *input, int kh, int kw,
                        int stride_h, int stride_w,
                        int pad_h, int pad_w);

/* col2im: inverse of im2col, used in backward pass */
ax_tensor_t *ax_col2im(ax_tensor_t *cols, int64_t channels,
                        int64_t height, int64_t width,
                        int kh, int kw,
                        int stride_h, int stride_w,
                        int pad_h, int pad_w);

#endif /* AX_CONV_H */
