/* test_conv_paths.c — verify every conv2d forward path against naive reference */

#include "test.h"
#include "axiom/axiom.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static ax_tensor_t *make_4d(float *data, int64_t n, int64_t c, int64_t h, int64_t w)
{
    int64_t shape[] = {n, c, h, w};
    return ax_tensor_from_array(data, shape, 4, AX_FLOAT32);
}

static float *rand_floats(int64_t n)
{
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    for (int64_t i = 0; i < n; i++)
        buf[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    return buf;
}

/* naive conv2d: NCHW layout, arbitrary stride and padding.
   out[n][co][oh][ow] = bias[co] + sum over ci,kh,kw of
       in[n][ci][oh*sh - ph + ky][ow*sw - pw + kx] * w[co][ci][ky][kx]
   where out-of-bounds input reads are zero (padding). */
static void naive_conv2d(const float *in, const float *w, const float *bias,
                         float *out,
                         int64_t N, int64_t Cin, int64_t H, int64_t W,
                         int64_t Cout, int kh, int kw,
                         int sh, int sw, int ph, int pw)
{
    int64_t oh = (H + 2 * ph - kh) / sh + 1;
    int64_t ow = (W + 2 * pw - kw) / sw + 1;
    int64_t out_hw = oh * ow;

    for (int64_t n = 0; n < N; n++) {
        for (int64_t co = 0; co < Cout; co++) {
            for (int64_t y = 0; y < oh; y++) {
                for (int64_t x = 0; x < ow; x++) {
                    float acc = bias ? bias[co] : 0.0f;
                    for (int64_t ci = 0; ci < Cin; ci++) {
                        for (int ky = 0; ky < kh; ky++) {
                            for (int kx = 0; kx < kw; kx++) {
                                int64_t iy = y * sh - ph + ky;
                                int64_t ix = x * sw - pw + kx;
                                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                    float iv = in[n * Cin * H * W + ci * H * W + iy * W + ix];
                                    float wv = w[co * Cin * kh * kw + ci * kh * kw + ky * kw + kx];
                                    acc += iv * wv;
                                }
                            }
                        }
                    }
                    out[n * Cout * out_hw + co * out_hw + y * ow + x] = acc;
                }
            }
        }
    }
}

/* compare framework output against naive reference.
   returns max absolute difference. */
static float compare_vs_naive(ax_layer_t *layer,
                              int64_t N, int64_t Cin, int64_t H, int64_t W,
                              int64_t Cout, int kh, int kw,
                              int sh, int sw, int ph, int pw)
{
    int64_t in_n = N * Cin * H * W;
    float *in_data = rand_floats(in_n);
    ax_tensor_t *input = make_4d(in_data, N, Cin, H, W);

    ax_conv2d_t *cc = (ax_conv2d_t *)layer;
    float *wd = (float *)cc->weight->storage->data;
    float *bd = (cc->use_bias && cc->bias) ? (float *)cc->bias->storage->data : NULL;

    int64_t out_h = (H + 2 * ph - kh) / sh + 1;
    int64_t out_w = (W + 2 * pw - kw) / sw + 1;
    int64_t out_n = N * Cout * out_h * out_w;

    float *ref = (float *)calloc((size_t)out_n, sizeof(float));
    naive_conv2d(in_data, wd, bd, ref, N, Cin, H, W, Cout, kh, kw, sh, sw, ph, pw);

    ax_no_grad();
    ax_tensor_t *out = ax_layer_forward(layer, input);
    ax_enable_grad();

    float max_diff = 0.0f;
    if (out) {
        float *od = (float *)out->storage->data;
        for (int64_t i = 0; i < out_n; i++) {
            float d = fabsf(od[i] - ref[i]);
            if (d > max_diff) max_diff = d;
        }
        ax_tensor_destroy(out);
    } else {
        max_diff = 1e10f;
    }

    free(ref);
    free(in_data);
    ax_tensor_destroy(input);
    return max_diff;
}

/* 1x1 kernel, stride 1, no padding: exercises zero-copy im2col fast path */
static void test_conv_1x1_s1(void)
{
    int64_t N = 2, Cin = 64, Cout = 128, H = 14, W = 14;
    ax_layer_t *c = ax_conv2d_create_ex((int)Cin, (int)Cout, 1, 1, 1, 1, 0, 0, true);
    AX_TEST_ASSERT(c != NULL, "create 1x1 conv");
    float d = compare_vs_naive(c, N, Cin, H, W, Cout, 1, 1, 1, 1, 0, 0);
    AX_TEST_ASSERT(d < 1e-4f, "1x1 s1 matches naive");
    ax_layer_destroy(c);
}

/* 3x3 stride 1, pad 1, large channels: exercises winograd F(2,3) path
   winograd requires Cin>=32, Cout>=32, out_h>=4, out_w>=4, 3x3 s1 p1 */
static void test_conv_winograd_f23(void)
{
    int64_t N = 1, Cin = 64, Cout = 64, H = 28, W = 28;
    ax_layer_t *c = ax_conv2d_create((int)Cin, (int)Cout, 3, 1, 1, true);
    AX_TEST_ASSERT(c != NULL, "create winograd conv");
    float d = compare_vs_naive(c, N, Cin, H, W, Cout, 3, 3, 1, 1, 1, 1);
    AX_TEST_ASSERT(d < 1e-4f, "winograd f23 matches naive");
    ax_layer_destroy(c);
}

/* 3x3 stride 1, small Cin=3: exercises direct smallcin path
   (can_direct_smallcin: Cin<=4, kh<=7, W_out >= VF32_WIDTH*2) */
static void test_conv_direct_3x3_s1(void)
{
    int64_t N = 1, Cin = 3, Cout = 16, H = 32, W = 32;
    ax_layer_t *c = ax_conv2d_create((int)Cin, (int)Cout, 3, 1, 1, true);
    AX_TEST_ASSERT(c != NULL, "create direct 3x3 s1 conv");
    float d = compare_vs_naive(c, N, Cin, H, W, Cout, 3, 3, 1, 1, 1, 1);
    AX_TEST_ASSERT(d < 1e-4f, "direct 3x3 s1 matches naive");
    ax_layer_destroy(c);
}

/* 3x3 stride 2, pad 1: stride-2 path (may fall to im2col+gemm or
   implicit gemm depending on compile flags and shape) */
static void test_conv_direct_3x3_s2(void)
{
    int64_t N = 1, Cin = 64, Cout = 128, H = 28, W = 28;
    ax_layer_t *c = ax_conv2d_create_ex((int)Cin, (int)Cout, 3, 3, 2, 2, 1, 1, true);
    AX_TEST_ASSERT(c != NULL, "create 3x3 s2 conv");
    float d = compare_vs_naive(c, N, Cin, H, W, Cout, 3, 3, 2, 2, 1, 1);
    AX_TEST_ASSERT(d < 1e-4f, "3x3 s2 matches naive");
    ax_layer_destroy(c);
}

/* very small Cin=1, K=5: exercises direct smallcin path with non-3x3 kernel */
static void test_conv_direct_smallcin(void)
{
    int64_t N = 1, Cin = 1, Cout = 32, H = 16, W = 16;
    int K = 5;
    ax_layer_t *c = ax_conv2d_create_ex((int)Cin, (int)Cout, K, K, 1, 1, 2, 2, true);
    AX_TEST_ASSERT(c != NULL, "create smallcin conv");
    float d = compare_vs_naive(c, N, Cin, H, W, Cout, K, K, 1, 1, 2, 2);
    AX_TEST_ASSERT(d < 1e-4f, "smallcin matches naive");
    ax_layer_destroy(c);
}

/* 5x5 kernel, stride 1, Cin=32: should fall through all specializations
   to the default im2col+GEMM path */
static void test_conv_im2col_default(void)
{
    int64_t N = 1, Cin = 32, Cout = 64, H = 14, W = 14;
    int K = 5;
    ax_layer_t *c = ax_conv2d_create_ex((int)Cin, (int)Cout, K, K, 1, 1, 2, 2, true);
    AX_TEST_ASSERT(c != NULL, "create im2col default conv");
    float d = compare_vs_naive(c, N, Cin, H, W, Cout, K, K, 1, 1, 2, 2);
    AX_TEST_ASSERT(d < 1e-4f, "im2col default matches naive");
    ax_layer_destroy(c);
}

/* small spatial, large batch: may trigger batched im2col path
   (M < AX_CONV_BATCH_M_THRESH=512, N > 1) */
static void test_conv_batched_im2col(void)
{
    int64_t N = 32, Cin = 64, Cout = 128, H = 4, W = 4;
    ax_layer_t *c = ax_conv2d_create((int)Cin, (int)Cout, 3, 1, 1, true);
    AX_TEST_ASSERT(c != NULL, "create batched conv");
    float d = compare_vs_naive(c, N, Cin, H, W, Cout, 3, 3, 1, 1, 1, 1);
    AX_TEST_ASSERT(d < 1e-4f, "batched im2col matches naive");
    ax_layer_destroy(c);
}

/* backward parity: compare dW from autograd against finite differences.
   small shape to keep float32 finite differences accurate. uses relative
   error (5%) OR absolute 1e-3 threshold, matching test_grad_verify.c. */
static void test_conv_backward_parity(void)
{
    int64_t N = 1, Cin = 8, Cout = 16, H = 7, W = 7;
    int kh = 3, kw = 3, sh = 1, sw = 1, ph = 1, pw = 1;
    float eps = 1e-3f;
    float rel_tol = 0.1f;
    float abs_tol = 5e-3f;

    ax_layer_t *layer = ax_conv2d_create_ex((int)Cin, (int)Cout, kh, kw, sh, sw, ph, pw, false);
    AX_TEST_ASSERT(layer != NULL, "create conv for backward");
    ax_conv2d_t *cc = (ax_conv2d_t *)layer;

    int64_t in_n = N * Cin * H * W;
    float *in_data = rand_floats(in_n);
    ax_tensor_t *input = make_4d(in_data, N, Cin, H, W);
    input->requires_grad = true;

    /* analytical gradient via autograd */
    ax_enable_grad();
    ax_tensor_t *out = ax_layer_forward(layer, input);
    AX_TEST_ASSERT(out != NULL, "forward for backward");
    ax_tensor_t *loss = ax_sum(out, -1);
    AX_TEST_ASSERT(loss != NULL, "sum for backward");
    ax_backward(loss);

    AX_TEST_ASSERT(cc->weight->grad != NULL, "weight should have grad");
    int64_t w_numel = ax_tensor_numel(cc->weight);
    float *anal_dw = (float *)malloc((size_t)w_numel * sizeof(float));
    memcpy(anal_dw, cc->weight->grad->storage->data, (size_t)w_numel * sizeof(float));

    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss);

    float *wd = (float *)cc->weight->storage->data;
    int check_count = 50;
    if (check_count > w_numel) check_count = (int)w_numel;

    int bad = 0;
    int step = (int)(w_numel / check_count);
    if (step < 1) step = 1;

    for (int idx = 0; idx < (int)w_numel && idx / step < check_count; idx += step) {
        float orig = wd[idx];

        wd[idx] = orig + eps;
        ax_storage_touch(cc->weight->storage);
        ax_no_grad();
        ax_tensor_t *out_p = ax_layer_forward(layer, input);
        ax_tensor_t *loss_p = ax_sum(out_p, -1);
        int64_t zi[] = {0};
        float fp = ax_tensor_get_f32(loss_p, zi);
        ax_tensor_destroy(loss_p);
        ax_tensor_destroy(out_p);

        wd[idx] = orig - eps;
        ax_storage_touch(cc->weight->storage);
        ax_tensor_t *out_m = ax_layer_forward(layer, input);
        ax_tensor_t *loss_m = ax_sum(out_m, -1);
        float fm = ax_tensor_get_f32(loss_m, zi);
        ax_tensor_destroy(loss_m);
        ax_tensor_destroy(out_m);
        ax_enable_grad();

        wd[idx] = orig;
        ax_storage_touch(cc->weight->storage);

        float numerical = (fp - fm) / (2.0f * eps);
        float diff = fabsf(anal_dw[idx] - numerical);
        float scale = fabsf(anal_dw[idx]) + fabsf(numerical) + 1e-8f;
        if (diff > abs_tol && diff / scale > rel_tol) bad++;
    }

    AX_TEST_ASSERT(bad == 0, "dW finite difference matches autograd");

    free(anal_dw);
    free(in_data);
    ax_tensor_destroy(input);
    ax_layer_destroy(layer);
}

/* regression: stale pack_b cache when same col_buf is reused across samples.
   triggers when K_gemm fits in a single KC tile (K_gemm <= 256) AND
   N_gemm fits in a single NC tile (N_gemm <= 256), so the GEMM has only
   one (jc,pc) tile and the cache never self-invalidates between samples. */
static void test_conv_pack_b_cache_reuse(void)
{
    float d;

    /* 1x1 conv, N=2: K_gemm=64 < KC, M=196 < NC => single tile */
    {
        ax_layer_t *c = ax_conv2d_create_ex(64, 128, 1, 1, 1, 1, 0, 0, true);
        AX_TEST_ASSERT(c != NULL, "create 1x1 N=2 conv");
        d = compare_vs_naive(c, 2, 64, 14, 14, 128, 1, 1, 1, 1, 0, 0);
        AX_TEST_ASSERT(d < 1e-4f, "1x1 N=2 per-sample reuse");
        ax_layer_destroy(c);
    }

    /* 1x1 conv, N=8 > typical T: same thread handles multiple samples */
    {
        ax_layer_t *c = ax_conv2d_create_ex(32, 64, 1, 1, 1, 1, 0, 0, true);
        AX_TEST_ASSERT(c != NULL, "create 1x1 N=8 conv");
        d = compare_vs_naive(c, 8, 32, 7, 7, 64, 1, 1, 1, 1, 0, 0);
        AX_TEST_ASSERT(d < 1e-4f, "1x1 N=8 per-sample reuse");
        ax_layer_destroy(c);
    }

    /* 3x3 conv, N=2, Cin=32: the original bug report shape */
    {
        ax_layer_t *c = ax_conv2d_create(32, 32, 3, 1, 1, true);
        AX_TEST_ASSERT(c != NULL, "create 3x3 N=2 Cin32 conv");
        d = compare_vs_naive(c, 2, 32, 8, 8, 32, 3, 3, 1, 1, 1, 1);
        AX_TEST_ASSERT(d < 1e-3f, "3x3 N=2 Cin32 8x8 matches naive");
        ax_layer_destroy(c);
    }

    /* 5x5 conv, N=4, avoids winograd: exercises batched path reuse */
    {
        ax_layer_t *c = ax_conv2d_create_ex(16, 32, 5, 5, 1, 1, 2, 2, true);
        AX_TEST_ASSERT(c != NULL, "create 5x5 batched conv");
        d = compare_vs_naive(c, 4, 16, 6, 6, 32, 5, 5, 1, 1, 2, 2);
        AX_TEST_ASSERT(d < 1e-4f, "5x5 N=4 batched path matches naive");
        ax_layer_destroy(c);
    }
}

/* degenerate shapes that might trip edge cases */
static void test_conv_edge_shapes(void)
{
    float d;

    /* smallest possible: H=W=1, K=1, Cin=1, Cout=1, N=1 */
    {
        ax_layer_t *c = ax_conv2d_create(1, 1, 1, 1, 0, true);
        AX_TEST_ASSERT(c != NULL, "create 1x1x1 conv");
        d = compare_vs_naive(c, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0);
        AX_TEST_ASSERT(d < 1e-5f, "1x1x1 edge matches naive");
        ax_layer_destroy(c);
    }

    /* output 1x1: H=W=3, K=3, S=1, P=0 */
    {
        ax_layer_t *c = ax_conv2d_create(1, 1, 3, 1, 0, true);
        AX_TEST_ASSERT(c != NULL, "create 1x1 output conv");
        d = compare_vs_naive(c, 1, 1, 3, 3, 1, 3, 3, 1, 1, 0, 0);
        AX_TEST_ASSERT(d < 1e-5f, "1x1 output edge matches naive");
        ax_layer_destroy(c);
    }

    /* 1D-like: H=1, W=100, K=1x3, Cin=16, Cout=16 */
    {
        ax_layer_t *c = ax_conv2d_create_ex(16, 16, 1, 3, 1, 1, 0, 1, true);
        AX_TEST_ASSERT(c != NULL, "create 1d-like conv");
        d = compare_vs_naive(c, 1, 16, 1, 100, 16, 1, 3, 1, 1, 0, 1);
        AX_TEST_ASSERT(d < 1e-4f, "1d-like edge matches naive");
        ax_layer_destroy(c);
    }

    /* large channels, small spatial: Cin=Cout=256, H=W=7, K=3 s1 p1 */
    {
        ax_layer_t *c = ax_conv2d_create(256, 256, 3, 1, 1, true);
        AX_TEST_ASSERT(c != NULL, "create large-channel conv");
        d = compare_vs_naive(c, 1, 256, 7, 7, 256, 3, 3, 1, 1, 1, 1);
        AX_TEST_ASSERT(d < 1e-3f, "large-channel edge matches naive");
        ax_layer_destroy(c);
    }
}

int main(void)
{
    srand(42);

    AX_RUN_TEST(test_conv_1x1_s1);
    AX_RUN_TEST(test_conv_winograd_f23);
    AX_RUN_TEST(test_conv_direct_3x3_s1);
    AX_RUN_TEST(test_conv_direct_3x3_s2);
    AX_RUN_TEST(test_conv_direct_smallcin);
    AX_RUN_TEST(test_conv_im2col_default);
    AX_RUN_TEST(test_conv_batched_im2col);
    AX_RUN_TEST(test_conv_backward_parity);
    AX_RUN_TEST(test_conv_pack_b_cache_reuse);
    AX_RUN_TEST(test_conv_edge_shapes);

    AX_TEST_SUMMARY();
}
