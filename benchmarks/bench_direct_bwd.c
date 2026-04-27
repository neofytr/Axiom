/* bench_direct_bwd.c — targeted conv backward benchmark for small spatial dims.
   measures shapes where M = out_h * out_w < 128 (direct backward territory). */

#include "axiom/axiom.h"
#include "axiom/autograd.h"
#include "bench_common.h"
#include <stdio.h>
#include <stdlib.h>

static void bench_conv_bwd(const char *tag, int N, int C_in, int H, int W,
                            int C_out, int K, int stride, int pad)
{
    ax_layer_t *conv = ax_conv2d_create_ex(C_in, C_out, K, K, stride, stride,
                                            pad, pad, true);
    int64_t in_sh[] = {N, C_in, H, W};
    ax_tensor_t *x = ax_tensor_rand(in_sh, 4, -0.5f, 0.5f);
    x->requires_grad = true;

    ax_enable_grad();
    ax_layer_train(conv);

    ax_tensor_t *probe = ax_layer_forward(conv, x);
    int out_h = (int)probe->shape[2], out_w = (int)probe->shape[3];
    int M = out_h * out_w;
    ax_tensor_t *loss = ax_sum(probe, -1);
    ax_backward(loss);
    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss);
    ax_tensor_destroy(probe);

    int iters = 30;

    for (int w = 0; w < 5; w++) {
        ax_tensor_t *y = ax_layer_forward(conv, x);
        ax_tensor_t *l = ax_sum(y, -1);
        ax_backward(l);
        ax_graph_cleanup(l);
        ax_tensor_destroy(l);
        ax_tensor_destroy(y);
    }

    double t0 = bench_now_ms();
    for (int i = 0; i < iters; i++) {
        ax_tensor_t *y = ax_layer_forward(conv, x);
        ax_tensor_t *l = ax_sum(y, -1);
        ax_backward(l);
        ax_graph_cleanup(l);
        ax_tensor_destroy(l);
        ax_tensor_destroy(y);
    }
    double lat = (bench_now_ms() - t0) / iters;

    printf("%-24s  N=%2d Cin=%3d H=%2d W=%2d Cout=%3d K=%d s=%d  M=%4d  %.1f us\n",
           tag, N, C_in, H, W, C_out, K, stride, M, lat * 1000.0);

    ax_tensor_destroy(x);
    ax_layer_destroy(conv);
}

int main(void)
{
    ax_init();
    printf("=== conv backward: direct bwd shapes (M < 128) ===\n\n");

    /* resnet stage4 4x4 spatial: M=16 */
    bench_conv_bwd("res_s4_256_512",   1, 256, 4, 4, 512, 3, 1, 1);
    bench_conv_bwd("res_s4_512_512",   1, 512, 4, 4, 512, 3, 1, 1);

    /* tiny 16x32 spatial 4x4: M=16 */
    bench_conv_bwd("tiny_16x32",       1, 16,  4, 4, 32,  3, 1, 1);

    /* resnet stage3 8x8: M=64 */
    bench_conv_bwd("res_s3_128_256",   1, 128, 8, 8, 256, 3, 1, 1);
    bench_conv_bwd("res_s3_256_256",   1, 256, 8, 8, 256, 3, 1, 1);

    /* M=49 (7x7 output) */
    bench_conv_bwd("res_7x7_256_512",  1, 256, 7, 7, 512, 3, 1, 1);

    /* M=100 (10x10 output, near boundary) */
    bench_conv_bwd("m100_64_128",      1, 64, 10, 10, 128, 3, 1, 1);

    /* batch > 1 shapes */
    bench_conv_bwd("res_s4_b4",        4, 256, 4, 4, 512, 3, 1, 1);
    bench_conv_bwd("res_s4_b8",        8, 256, 4, 4, 512, 3, 1, 1);
    bench_conv_bwd("res_s4_b16",      16, 256, 4, 4, 512, 3, 1, 1);

    printf("\n=== conv backward: non-direct shapes (M >= 128, control) ===\n\n");

    /* M=256 (16x16) — should NOT use direct, control group */
    bench_conv_bwd("res_s2_64_128",    1, 64,  16, 16, 128, 3, 1, 1);
    bench_conv_bwd("res_s2_128_128",   1, 128, 16, 16, 128, 3, 1, 1);

    /* M=1024 (32x32) — control */
    bench_conv_bwd("res_s1_64_64",     1, 64,  32, 32, 64,  3, 1, 1);

    return 0;
}
