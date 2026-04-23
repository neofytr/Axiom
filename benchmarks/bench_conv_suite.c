/* bench_conv_suite.c — conv2d + pooling + fused ConvBNReLU. */

#include "axiom/axiom.h"
#include "bench_common.h"
#include <stdio.h>
#include <stdlib.h>

static double conv_flops(int N, int C_in, int C_out, int H_out, int W_out, int K) {
    return 2.0 * N * C_out * H_out * W_out * C_in * K * K;
}

static void bench_conv(int N, int C_in, int H, int W, int C_out, int K, int stride, int pad) {
    ax_layer_t *conv = ax_conv2d_create_ex(C_in, C_out, K, K, stride, stride, pad, pad, true);
    int64_t in_sh[] = {N, C_in, H, W};
    ax_tensor_t *x = ax_tensor_rand(in_sh, 4, -0.5f, 0.5f);
    ax_no_grad();
    ax_layer_eval(conv);

    /* warm up */
    for (int w = 0; w < 2; w++) { ax_tensor_t *y = ax_layer_forward(conv, x); ax_tensor_destroy(y); }

    ax_tensor_t *probe = ax_layer_forward(conv, x);
    int H_out = (int)probe->shape[2];
    int W_out = (int)probe->shape[3];
    ax_tensor_destroy(probe);

    int iters = (int)(2e10 / conv_flops(N, C_in, C_out, H_out, W_out, K));
    if (iters < 2) iters = 2;
    if (iters > 50) iters = 50;

    double t0 = bench_now_ms();
    for (int i = 0; i < iters; i++) {
        ax_tensor_t *y = ax_layer_forward(conv, x);
        ax_tensor_destroy(y);
    }
    double lat = (bench_now_ms() - t0) / iters;
    ax_enable_grad();

    char cs[96];
    snprintf(cs, sizeof(cs), "conv_%dx%dx%dx%d_%dx%d_k%d_s%d", N, C_in, H, W, C_out, K, K, stride);
    BENCH_EMIT_FLOPS("conv", cs, lat, conv_flops(N, C_in, C_out, H_out, W_out, K));

    ax_tensor_destroy(x); ax_layer_destroy(conv);
}

static void bench_conv_bn_relu(int N, int C_in, int H, int W, int C_out, int K, int stride, int pad) {
    /* build the fused layer from its pieces — conv + bn paired into one.
       matches how model code typically constructs CBR. */
    ax_layer_t *conv = ax_conv2d_create_ex(C_in, C_out, K, K, stride, stride, pad, pad, false);
    ax_layer_t *bn = ax_batchnorm_create(C_out, 1e-5f, 0.1f);
    ax_layer_t *cbr = ax_conv_bn_relu_create_from(conv, bn);
    int64_t in_sh[] = {N, C_in, H, W};
    ax_tensor_t *x = ax_tensor_rand(in_sh, 4, -0.5f, 0.5f);
    ax_no_grad();
    ax_layer_eval(cbr);

    for (int w = 0; w < 2; w++) { ax_tensor_t *y = ax_layer_forward(cbr, x); ax_tensor_destroy(y); }

    ax_tensor_t *probe = ax_layer_forward(cbr, x);
    int H_out = (int)probe->shape[2];
    int W_out = (int)probe->shape[3];
    ax_tensor_destroy(probe);

    int iters = (int)(2e10 / conv_flops(N, C_in, C_out, H_out, W_out, K));
    if (iters < 2) iters = 2;
    if (iters > 50) iters = 50;

    double t0 = bench_now_ms();
    for (int i = 0; i < iters; i++) {
        ax_tensor_t *y = ax_layer_forward(cbr, x);
        ax_tensor_destroy(y);
    }
    double lat = (bench_now_ms() - t0) / iters;
    ax_enable_grad();

    char cs[96];
    snprintf(cs, sizeof(cs), "cbr_%dx%dx%dx%d_%dx%d_k%d_s%d", N, C_in, H, W, C_out, K, K, stride);
    BENCH_EMIT_FLOPS("conv", cs, lat, conv_flops(N, C_in, C_out, H_out, W_out, K));

    ax_tensor_destroy(x); ax_layer_destroy(cbr);
}

static void bench_maxpool(int N, int C, int H, int W, int K, int stride) {
    ax_layer_t *p = ax_maxpool2d_create(K, stride, 0);
    int64_t in_sh[] = {N, C, H, W};
    ax_tensor_t *x = ax_tensor_rand(in_sh, 4, -0.5f, 0.5f);
    ax_no_grad();
    for (int w = 0; w < 3; w++) { ax_tensor_t *y = ax_layer_forward(p, x); ax_tensor_destroy(y); }
    int iters = (int)(1e9 / (N * C * H * W));
    if (iters < 5) iters = 5;
    if (iters > 500) iters = 500;
    double t0 = bench_now_ms();
    for (int i = 0; i < iters; i++) { ax_tensor_t *y = ax_layer_forward(p, x); ax_tensor_destroy(y); }
    double lat = (bench_now_ms() - t0) / iters;
    ax_enable_grad();
    char cs[96]; snprintf(cs, sizeof(cs), "maxpool_%dx%dx%dx%d_k%d", N, C, H, W, K);
    BENCH_EMIT_BW("conv", cs, lat, 4.0 * (double)(N * C * H * W));
    ax_tensor_destroy(x); ax_layer_destroy(p);
}

/* NHWC variant: convert input to NHWC once, then time only the pool. */
static void bench_maxpool_nhwc(int N, int C, int H, int W, int K, int stride) {
    ax_layer_t *p = ax_maxpool2d_create(K, stride, 0);
    int64_t in_sh[] = {N, C, H, W};
    ax_tensor_t *x_nchw = ax_tensor_rand(in_sh, 4, -0.5f, 0.5f);
    ax_tensor_t *x = ax_tensor_to_nhwc(x_nchw);
    ax_no_grad();
    for (int w = 0; w < 3; w++) { ax_tensor_t *y = ax_layer_forward(p, x); ax_tensor_destroy(y); }
    int iters = (int)(1e9 / (N * C * H * W));
    if (iters < 5) iters = 5;
    if (iters > 500) iters = 500;
    double t0 = bench_now_ms();
    for (int i = 0; i < iters; i++) { ax_tensor_t *y = ax_layer_forward(p, x); ax_tensor_destroy(y); }
    double lat = (bench_now_ms() - t0) / iters;
    ax_enable_grad();
    char cs[96]; snprintf(cs, sizeof(cs), "maxpool_%dx%dx%dx%d_k%d_nhwc", N, C, H, W, K);
    BENCH_EMIT_BW("conv", cs, lat, 4.0 * (double)(N * C * H * W));
    ax_tensor_destroy(x); ax_tensor_destroy(x_nchw); ax_layer_destroy(p);
}

/* NHWC conv variant: convert input to NHWC once, time only the conv. */
static void bench_conv_nhwc(int N, int C_in, int H, int W, int C_out, int K, int stride, int pad) {
    ax_layer_t *conv = ax_conv2d_create_ex(C_in, C_out, K, K, stride, stride, pad, pad, true);
    int64_t in_sh[] = {N, C_in, H, W};
    ax_tensor_t *x_nchw = ax_tensor_rand(in_sh, 4, -0.5f, 0.5f);
    ax_tensor_t *x = ax_tensor_to_nhwc(x_nchw);
    ax_no_grad();
    ax_layer_eval(conv);
    for (int w = 0; w < 2; w++) { ax_tensor_t *y = ax_layer_forward(conv, x); ax_tensor_destroy(y); }
    ax_tensor_t *probe = ax_layer_forward(conv, x);
    int H_out = (int)probe->shape[1];   /* NHWC: dim 1 is H */
    int W_out = (int)probe->shape[2];
    ax_tensor_destroy(probe);
    int iters = (int)(2e10 / conv_flops(N, C_in, C_out, H_out, W_out, K));
    if (iters < 2) iters = 2;
    if (iters > 50) iters = 50;
    double t0 = bench_now_ms();
    for (int i = 0; i < iters; i++) { ax_tensor_t *y = ax_layer_forward(conv, x); ax_tensor_destroy(y); }
    double lat = (bench_now_ms() - t0) / iters;
    ax_enable_grad();
    char cs[96]; snprintf(cs, sizeof(cs), "conv_%dx%dx%dx%d_%dx%d_k%d_s%d_nhwc", N, C_in, H, W, C_out, K, K, stride);
    BENCH_EMIT_FLOPS("conv", cs, lat, conv_flops(N, C_in, C_out, H_out, W_out, K));
    ax_tensor_destroy(x); ax_tensor_destroy(x_nchw); ax_layer_destroy(conv);
}

int main(void) {
    ax_init();

    /* VGG-ish shapes — representative CNN workload */
    struct { int N, Cin, H, W, Cout, K, stride, pad; } cases[] = {
        /* early stages: few channels, large spatial */
        { 32,   3, 224, 224,  64, 3, 1, 1},
        { 32,  64, 112, 112, 128, 3, 1, 1},
        /* mid stages */
        { 32, 128,  56,  56, 256, 3, 1, 1},
        { 32, 256,  28,  28, 512, 3, 1, 1},
        /* late: heavy channels, small spatial */
        { 32, 512,  14,  14, 512, 3, 1, 1},
        /* strided (downsample) */
        { 32,  64, 112, 112, 128, 3, 2, 1},
        /* 1x1 (pointwise) */
        { 32, 512,  14,  14, 512, 1, 1, 0},
    };
    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); i++) {
        bench_conv(cases[i].N, cases[i].Cin, cases[i].H, cases[i].W,
                   cases[i].Cout, cases[i].K, cases[i].stride, cases[i].pad);
    }

    /* fused ConvBNReLU at two representative shapes */
    bench_conv_bn_relu(32,  64, 112, 112, 128, 3, 1, 1);
    bench_conv_bn_relu(32, 256,  28,  28, 512, 3, 1, 1);

    /* pooling */
    bench_maxpool(32, 128, 112, 112, 2, 2);
    bench_maxpool(32, 256,  56,  56, 2, 2);

    /* NHWC variants — measure layout-aware kernel speed on the bench cases
       where the NCHW path was previously TF-bound. */
    bench_maxpool_nhwc(32, 128, 112, 112, 2, 2);
    bench_maxpool_nhwc(32, 256,  56,  56, 2, 2);
    bench_conv_nhwc(32,  64, 112, 112, 128, 3, 2, 1);  /* the stride-2 regression */
    bench_conv_nhwc(32, 128,  56,  56, 256, 3, 1, 1);
    bench_conv_nhwc(32, 256,  28,  28, 512, 3, 1, 1);
    bench_conv_nhwc(32, 512,  14,  14, 512, 3, 1, 1);

    ax_shutdown();
    return 0;
}
