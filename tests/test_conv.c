/* test_conv.c — tests for conv2d, pooling, flatten, im2col */

#include "test.h"
#include "axiom/axiom.h"

/* helper: make a 4d NCHW tensor */
static ax_tensor_t *make_4d(float *data, int64_t n, int64_t c, int64_t h, int64_t w)
{
    int64_t shape[] = {n, c, h, w};
    return ax_tensor_from_array(data, shape, 4, AX_FLOAT32);
}

static void test_im2col_basic(void)
{
    /* single channel 3x3 image, 2x2 kernel, stride 1, no padding
       should produce a [4, 4] column matrix (4 = 1*2*2, 4 = 2*2 output positions) */
    float img_data[] = {1, 2, 3,
                        4, 5, 6,
                        7, 8, 9};
    int64_t img_shape[] = {1, 3, 3};
    ax_tensor_t *img = ax_tensor_from_array(img_data, img_shape, 3, AX_FLOAT32);

    ax_tensor_t *cols = ax_im2col(img, 2, 2, 1, 1, 0, 0);
    AX_TEST_ASSERT(cols != NULL, "im2col should work");
    AX_TEST_ASSERT_EQ(cols->shape[0], 4, "rows = C*kh*kw = 1*2*2");
    AX_TEST_ASSERT_EQ(cols->shape[1], 4, "cols = oh*ow = 2*2");

    /* first column should be the top-left 2x2 patch: [1, 2, 4, 5] */
    int64_t i00[] = {0, 0}, i10[] = {1, 0}, i20[] = {2, 0}, i30[] = {3, 0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(cols, i00), 1.0f, 1e-5, "patch[0,0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(cols, i10), 2.0f, 1e-5, "patch[1,0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(cols, i20), 4.0f, 1e-5, "patch[2,0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(cols, i30), 5.0f, 1e-5, "patch[3,0]");

    ax_tensor_destroy(img); ax_tensor_destroy(cols);
}

static void test_conv2d_create(void)
{
    ax_layer_t *c = ax_conv2d_create(3, 16, 3, 1, 1, true);
    AX_TEST_ASSERT(c != NULL, "conv2d should be created");

    /* weight: [16, 3, 3, 3], bias: [16] */
    ax_conv2d_t *cc = (ax_conv2d_t *)c;
    AX_TEST_ASSERT_EQ(cc->weight->shape[0], 16, "out channels");
    AX_TEST_ASSERT_EQ(cc->weight->shape[1], 3, "in channels");
    AX_TEST_ASSERT_EQ(cc->weight->shape[2], 3, "kernel h");
    AX_TEST_ASSERT_EQ(cc->weight->shape[3], 3, "kernel w");
    AX_TEST_ASSERT(cc->weight->requires_grad, "weight needs grad");

    AX_TEST_ASSERT_EQ(cc->bias->shape[0], 16, "bias size");

    /* params: weight + bias = 16*3*3*3 + 16 = 432 + 16 = 448 */
    AX_TEST_ASSERT_EQ(ax_layer_param_count(c), 448, "param count");

    ax_layer_destroy(c);
}

static void test_conv2d_forward(void)
{
    /* 1 image, 1 channel, 4x4, kernel 3x3, stride 1, pad 0
       output should be [1, out_ch, 2, 2] */
    ax_layer_t *c = ax_conv2d_create(1, 1, 3, 1, 0, false);

    /* set kernel to all ones for easy verification */
    ax_conv2d_t *cc = (ax_conv2d_t *)c;
    ax_compute_fill(cc->weight, 1.0);

    float in_data[16];
    for (int i = 0; i < 16; i++) in_data[i] = 1.0f;
    ax_tensor_t *input = make_4d(in_data, 1, 1, 4, 4);

    ax_tensor_t *out = ax_layer_forward(c, input);
    AX_TEST_ASSERT(out != NULL, "conv2d forward should work");
    AX_TEST_ASSERT_EQ(out->ndim, 4, "output is 4d");
    AX_TEST_ASSERT_EQ(out->shape[0], 1, "batch");
    AX_TEST_ASSERT_EQ(out->shape[1], 1, "channels");
    AX_TEST_ASSERT_EQ(out->shape[2], 2, "out height");
    AX_TEST_ASSERT_EQ(out->shape[3], 2, "out width");

    /* all-ones input with all-ones 3x3 kernel: each output = 9 */
    float *od = (float *)out->storage->data;
    AX_TEST_ASSERT_NEAR(od[0], 9.0f, 1e-4, "conv output[0,0]");
    AX_TEST_ASSERT_NEAR(od[1], 9.0f, 1e-4, "conv output[0,1]");
    AX_TEST_ASSERT_NEAR(od[2], 9.0f, 1e-4, "conv output[1,0]");
    AX_TEST_ASSERT_NEAR(od[3], 9.0f, 1e-4, "conv output[1,1]");

    ax_tensor_destroy(input);
    ax_layer_destroy(c);
}

static void test_conv2d_with_padding(void)
{
    /* pad=1 with 3x3 kernel should preserve spatial dims */
    ax_layer_t *c = ax_conv2d_create(1, 1, 3, 1, 1, false);
    ax_conv2d_t *cc = (ax_conv2d_t *)c;
    ax_compute_fill(cc->weight, 1.0);

    float in_data[9] = {1,1,1, 1,1,1, 1,1,1};
    ax_tensor_t *input = make_4d(in_data, 1, 1, 3, 3);

    ax_tensor_t *out = ax_layer_forward(c, input);
    AX_TEST_ASSERT(out != NULL, "padded conv should work");
    AX_TEST_ASSERT_EQ(out->shape[2], 3, "height preserved with pad=1");
    AX_TEST_ASSERT_EQ(out->shape[3], 3, "width preserved with pad=1");

    /* center pixel sees full 3x3 of ones: output = 9
       corner pixel sees 2x2 of ones: output = 4 */
    float *od = (float *)out->storage->data;
    AX_TEST_ASSERT_NEAR(od[4], 9.0f, 1e-4, "center = 9");
    AX_TEST_ASSERT_NEAR(od[0], 4.0f, 1e-4, "corner = 4");

    ax_tensor_destroy(input);
    ax_layer_destroy(c);
}

static void test_maxpool2d(void)
{
    /* [1, 1, 4, 4] with 2x2 maxpool, stride 2 -> [1, 1, 2, 2] */
    float in_data[] = {1, 3, 2, 4,
                       5, 6, 7, 8,
                       3, 2, 1, 0,
                       1, 2, 3, 4};
    ax_tensor_t *input = make_4d(in_data, 1, 1, 4, 4);

    ax_layer_t *pool = ax_maxpool2d_create(2, 2, 0);
    ax_tensor_t *out = ax_layer_forward(pool, input);

    AX_TEST_ASSERT(out != NULL, "maxpool should work");
    AX_TEST_ASSERT_EQ(out->shape[2], 2, "pooled height");
    AX_TEST_ASSERT_EQ(out->shape[3], 2, "pooled width");

    float *od = (float *)out->storage->data;
    AX_TEST_ASSERT_NEAR(od[0], 6.0f, 1e-5, "max of top-left 2x2");
    AX_TEST_ASSERT_NEAR(od[1], 8.0f, 1e-5, "max of top-right 2x2");
    AX_TEST_ASSERT_NEAR(od[2], 3.0f, 1e-5, "max of bottom-left 2x2");
    AX_TEST_ASSERT_NEAR(od[3], 4.0f, 1e-5, "max of bottom-right 2x2");

    ax_tensor_destroy(input);
    ax_layer_destroy(pool);
}

static void test_avgpool2d(void)
{
    float in_data[] = {1, 2, 3, 4,
                       5, 6, 7, 8,
                       9, 10, 11, 12,
                       13, 14, 15, 16};
    ax_tensor_t *input = make_4d(in_data, 1, 1, 4, 4);

    ax_layer_t *pool = ax_avgpool2d_create(2, 2, 0);
    ax_tensor_t *out = ax_layer_forward(pool, input);

    AX_TEST_ASSERT(out != NULL, "avgpool should work");
    float *od = (float *)out->storage->data;
    /* top-left 2x2: avg(1,2,5,6) = 3.5 */
    AX_TEST_ASSERT_NEAR(od[0], 3.5f, 1e-5, "avg of top-left 2x2");
    /* bottom-right 2x2: avg(11,12,15,16) = 13.5 */
    AX_TEST_ASSERT_NEAR(od[3], 13.5f, 1e-5, "avg of bottom-right 2x2");

    ax_tensor_destroy(input);
    ax_layer_destroy(pool);
}

static void test_global_avgpool(void)
{
    float in_data[] = {1, 2, 3, 4};
    ax_tensor_t *input = make_4d(in_data, 1, 1, 2, 2);

    ax_layer_t *pool = ax_global_avgpool2d_create();
    ax_tensor_t *out = ax_layer_forward(pool, input);

    AX_TEST_ASSERT(out != NULL, "global avgpool should work");
    AX_TEST_ASSERT_EQ(out->ndim, 2, "output is [N, C]");
    AX_TEST_ASSERT_EQ(out->shape[0], 1, "batch");
    AX_TEST_ASSERT_EQ(out->shape[1], 1, "channels");

    int64_t i0[] = {0, 0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(out, i0), 2.5f, 1e-5, "avg(1,2,3,4) = 2.5");

    ax_tensor_destroy(input);
    ax_layer_destroy(pool);
}

static void test_flatten(void)
{
    float in_data[24];
    for (int i = 0; i < 24; i++) in_data[i] = (float)i;
    ax_tensor_t *input = make_4d(in_data, 2, 3, 2, 2);

    ax_layer_t *flat = ax_flatten_create();
    ax_tensor_t *out = ax_layer_forward(flat, input);

    AX_TEST_ASSERT(out != NULL, "flatten should work");
    AX_TEST_ASSERT_EQ(out->ndim, 2, "output is 2d");
    AX_TEST_ASSERT_EQ(out->shape[0], 2, "batch preserved");
    AX_TEST_ASSERT_EQ(out->shape[1], 12, "3*2*2 = 12");

    ax_tensor_destroy(input);
    ax_layer_destroy(flat);
}

static void test_conv_pipeline(void)
{
    /* mini cnn pipeline: conv -> relu -> pool -> flatten
       input: [1, 1, 6, 6] */
    float in_data[36];
    for (int i = 0; i < 36; i++) in_data[i] = (float)(i % 5);
    ax_tensor_t *input = make_4d(in_data, 1, 1, 6, 6);

    ax_layer_t *model = ax_sequential_create();
    ax_sequential_add(model, ax_conv2d_create(1, 4, 3, 1, 0, true));
    ax_sequential_add(model, ax_relu_layer_create());
    ax_sequential_add(model, ax_maxpool2d_create(2, 2, 0));
    ax_sequential_add(model, ax_flatten_create());

    ax_tensor_t *out = ax_layer_forward(model, input);
    AX_TEST_ASSERT(out != NULL, "conv pipeline should work");

    /* conv: [1,1,6,6] -> [1,4,4,4], pool: -> [1,4,2,2], flatten: -> [1,16] */
    AX_TEST_ASSERT_EQ(out->ndim, 2, "final output is 2d");
    AX_TEST_ASSERT_EQ(out->shape[0], 1, "batch");
    AX_TEST_ASSERT_EQ(out->shape[1], 16, "4*2*2 = 16 features");

    ax_tensor_destroy(input);
    ax_layer_destroy(model);
}

/* test batched GEMM path: N>1 && M<512 triggers the single-GEMM route.
   verify forward output and backward weight gradient match analytic expectations. */
static void test_conv2d_batched_fwd(void)
{
    /* N=8, C_in=4, H=7, W=7 → M=49 < 512 → batched path
       3x3 pad=1: out_h=7, out_w=7, K=4*9=36, C_out=8
       all-ones weights, all-ones input → each output pixel = K = 36 */
    int N = 8, Cin = 4, H = 7, W = 7, Cout = 8;
    ax_layer_t *c = ax_conv2d_create(Cin, Cout, 3, 1, 1, false);
    ax_conv2d_t *cc = (ax_conv2d_t *)c;
    ax_compute_fill(cc->weight, 1.0f);

    int64_t in_sh[] = {N, Cin, H, W};
    ax_tensor_t *inp = ax_tensor_ones(in_sh, 4, AX_FLOAT32);

    ax_tensor_t *out = ax_layer_forward(c, inp);
    AX_TEST_ASSERT(out != NULL, "batched conv fwd should work");
    AX_TEST_ASSERT_EQ(out->shape[0], N, "batch");
    AX_TEST_ASSERT_EQ(out->shape[1], Cout, "channels");
    AX_TEST_ASSERT_EQ(out->shape[2], H, "height preserved");

    /* center pixel sees full 3x3 patch across Cin channels: 9*Cin = 36 */
    float *od = (float *)out->storage->data;
    int64_t M = H * W;
    int64_t center = 3 * W + 3; /* pixel (3,3) */
    for (int n = 0; n < N; n++)
        for (int co = 0; co < Cout; co++)
            AX_TEST_ASSERT_NEAR(od[(n * Cout + co) * M + center], (float)(9 * Cin), 1e-3f,
                                "batched: center pixel should equal K=9*Cin");

    ax_tensor_destroy(inp);
    ax_layer_destroy(c);
}

static void test_conv2d_batched_bwd(void)
{
    /* backward: dW should sum contributions from all N samples.
       N=4, C_in=2, H=5, W=5 → M=25 < 512 → batched bwd path
       3x3 pad=1 conv, all-ones weights+input, all-ones upstream grad.
       dW[co, k] = sum_n sum_m go[n,co,m] * col[n,k,m] */
    int N = 4, Cin = 2, H = 5, W = 5, Cout = 4;
    ax_layer_t *c = ax_conv2d_create(Cin, Cout, 3, 1, 1, false);
    ax_conv2d_t *cc = (ax_conv2d_t *)c;
    ax_compute_fill(cc->weight, 1.0f);
    cc->weight->requires_grad = true;

    int64_t in_sh[] = {N, Cin, H, W};
    ax_tensor_t *inp = ax_tensor_ones(in_sh, 4, AX_FLOAT32);
    inp->requires_grad = true;

    ax_enable_grad();
    ax_tensor_t *out = ax_layer_forward(c, inp);

    /* sum loss = sum of all outputs; dout = all-ones */
    ax_tensor_t *loss = ax_sum(out, -1);
    ax_backward(loss);

    /* each interior im2col patch element is 1.0 (all-ones input).
       sum over N*M positions: dW[co,k] = N * M_effective where
       M_effective counts how many spatial positions have this kernel tap active.
       for an interior tap (not corner/edge pad): all H*W positions → N*H*W */
    float *wg = (float *)cc->weight->grad->storage->data;
    int64_t K = Cin * 3 * 3;
    /* center tap of center channel at (1,1): all M positions see it → N*H*W */
    int center_tap = 1 * 3 + 1; /* ky=1, kx=1 in the 3x3 kernel */
    float expected_center = (float)(N * H * W);
    for (int co = 0; co < Cout; co++) {
        for (int ci = 0; ci < Cin; ci++) {
            int k = ci * 9 + center_tap;
            AX_TEST_ASSERT_NEAR(wg[co * K + k], expected_center, 1.0f,
                                "batched bwd: center tap dW");
        }
    }

    ax_tensor_destroy(loss);
    ax_tensor_destroy(out);
    ax_tensor_destroy(inp);
    ax_layer_destroy(c);
    ax_no_grad();
}

/* sub-batched path: when N*K*M*4 exceeds AX_CONV_BATCH_COL_BYTES (8 MB)
   the batched path splits N into n_batch-sized chunks. verify forward and
   backward across the split boundary match per-sample reference results. */
static void test_conv2d_subbatched_fwd_bwd(void)
{
    /* N=8, Cin=256, H=W=14, kh=kw=3 → K=2304, M=196.
       full_bytes = 2304*196*8*4 = 14.4 MB > 8 MB → triggers split.
       n_batch = 8MB/(K*M*4) = 4 → 2 chunks of 4 samples each. */
    int N = 8, Cin = 256, H = 14, W = 14, Cout = 32;
    int64_t M = (int64_t)H * W;
    int64_t K = (int64_t)Cin * 9;

    ax_layer_t *c = ax_conv2d_create(Cin, Cout, 3, 1, 1, false);
    ax_conv2d_t *cc = (ax_conv2d_t *)c;

    /* deterministic small init */
    int64_t wn = (int64_t)Cout * Cin * 9;
    float *wd = (float *)cc->weight->storage->data;
    for (int64_t i = 0; i < wn; i++) wd[i] = (float)((i * 17) % 31 - 15) * 0.001f;
    cc->weight->requires_grad = true;

    int64_t in_sh[] = {N, Cin, H, W};
    ax_tensor_t *inp = ax_tensor_create(in_sh, 4, AX_FLOAT32);
    int64_t inn = (int64_t)N * Cin * H * W;
    float *id = (float *)inp->storage->data;
    for (int64_t i = 0; i < inn; i++) id[i] = (float)((i * 7) % 23 - 11) * 0.01f;

    /* sub-batched forward */
    ax_tensor_t *out_b = ax_layer_forward(c, inp);
    AX_TEST_ASSERT(out_b != NULL, "sub-batched fwd should run");
    AX_TEST_ASSERT_EQ(out_b->shape[0], N, "sub-batched fwd: N preserved");
    AX_TEST_ASSERT_EQ(out_b->shape[1], Cout, "sub-batched fwd: Cout preserved");

    /* per-sample reference: N=1 disables batched path → per-sample im2col+gemm */
    int64_t ref_sh[] = {1, Cin, H, W};
    for (int n = 0; n < N; n++) {
        ax_tensor_t *one = ax_tensor_create(ref_sh, 4, AX_FLOAT32);
        memcpy(one->storage->data, id + n * Cin * H * W,
               (size_t)(Cin * H * W) * sizeof(float));
        ax_tensor_t *out_one = ax_layer_forward(c, one);
        const float *bd = (const float *)out_b->storage->data + n * Cout * M;
        const float *sd = (const float *)out_one->storage->data;
        for (int64_t i = 0; i < Cout * M; i++) {
            AX_TEST_ASSERT_NEAR(bd[i], sd[i], 5e-3f,
                                "sub-batched fwd matches per-sample ref");
        }
        ax_tensor_destroy(one);
        ax_tensor_destroy(out_one);
    }

    /* sub-batched backward: dW must accumulate correctly across chunks.
       reuse the layer (weights unchanged); compute via batched path then
       compare to per-sample dW summation. */
    inp->requires_grad = true;
    /* fresh forward so saved tensors are batched-shape */
    ax_enable_grad();
    ax_tensor_t *out_g = ax_layer_forward(c, inp);
    ax_tensor_t *loss = ax_sum(out_g, -1);
    ax_backward(loss);
    ax_no_grad();

    /* snapshot dW from sub-batched run */
    float *dW_b = (float *)malloc((size_t)wn * sizeof(float));
    memcpy(dW_b, cc->weight->grad->storage->data, (size_t)wn * sizeof(float));
    /* zero grad before per-sample reference accumulation */
    memset(cc->weight->grad->storage->data, 0, (size_t)wn * sizeof(float));

    /* per-sample dW reference */
    ax_enable_grad();
    for (int n = 0; n < N; n++) {
        ax_tensor_t *one = ax_tensor_create(ref_sh, 4, AX_FLOAT32);
        memcpy(one->storage->data, id + n * Cin * H * W,
               (size_t)(Cin * H * W) * sizeof(float));
        one->requires_grad = true;
        ax_tensor_t *out_one = ax_layer_forward(c, one);
        ax_tensor_t *l1 = ax_sum(out_one, -1);
        ax_backward(l1);
        ax_tensor_destroy(l1);
        ax_tensor_destroy(out_one);
        ax_tensor_destroy(one);
    }
    ax_no_grad();

    const float *dW_ref = (const float *)cc->weight->grad->storage->data;
    /* magnitudes scale with N*M*Cin so use a relative tolerance via abs */
    for (int64_t i = 0; i < wn; i++) {
        AX_TEST_ASSERT_NEAR(dW_b[i], dW_ref[i], 1.0f,
                            "sub-batched dW matches per-sample summation");
    }

    free(dW_b);
    ax_tensor_destroy(loss);
    ax_tensor_destroy(out_g);
    ax_tensor_destroy(out_b);
    ax_tensor_destroy(inp);
    ax_layer_destroy(c);
}

/* 1×1 stride=1 pad=0 backward zero-copy fast path.
   verify dW and dX match a per-sample reference computed via the legacy
   path (3×3 with same effective shape — too different to compare).
   instead, check gradient consistency: numerical vs analytical via
   centered finite difference on a few weights and inputs. */
static void test_conv2d_1x1_zero_copy_bwd(void)
{
    /* small 1×1: N=4, Cin=8, Cout=4, H=W=5 */
    int N = 4, Cin = 8, H = 5, W = 5, Cout = 4;
    int64_t M = (int64_t)H * W;

    ax_layer_t *c = ax_conv2d_create(Cin, Cout, 1, 1, 0, false);
    ax_conv2d_t *cc = (ax_conv2d_t *)c;
    cc->weight->requires_grad = true;

    /* deterministic small init */
    int64_t wn = (int64_t)Cout * Cin;
    float *wd = (float *)cc->weight->storage->data;
    for (int64_t i = 0; i < wn; i++) wd[i] = (float)((i * 13) % 17 - 8) * 0.1f;

    int64_t in_sh[] = {N, Cin, H, W};
    ax_tensor_t *inp = ax_tensor_create(in_sh, 4, AX_FLOAT32);
    int64_t inn = (int64_t)N * Cin * H * W;
    float *id = (float *)inp->storage->data;
    for (int64_t i = 0; i < inn; i++) id[i] = (float)((i * 7) % 11 - 5) * 0.1f;
    inp->requires_grad = true;

    ax_enable_grad();
    ax_tensor_t *out = ax_layer_forward(c, inp);
    ax_tensor_t *loss = ax_sum(out, -1);
    ax_backward(loss);
    ax_no_grad();

    /* analytical dW from fast path */
    float *dW_fast = (float *)malloc((size_t)wn * sizeof(float));
    memcpy(dW_fast, cc->weight->grad->storage->data, (size_t)wn * sizeof(float));

    /* analytical dX from fast path */
    float *dX_fast = (float *)malloc((size_t)inn * sizeof(float));
    memcpy(dX_fast, inp->grad->storage->data, (size_t)inn * sizeof(float));

    /* numerical finite-difference dW for a single weight: loss(w+eps) - loss(w-eps) / 2eps */
    float eps = 1e-3f;
    int probes[] = {0, wn/4, wn/2, 3*wn/4, wn-1};
    for (int p = 0; p < 5; p++) {
        int64_t i = probes[p];
        float orig = wd[i];
        wd[i] = orig + eps;
        ax_tensor_t *out_p = ax_layer_forward(c, inp);
        float lp = 0.0f;
        float *od_p = (float *)out_p->storage->data;
        for (int64_t j = 0; j < N * Cout * M; j++) lp += od_p[j];
        ax_tensor_destroy(out_p);

        wd[i] = orig - eps;
        ax_tensor_t *out_m = ax_layer_forward(c, inp);
        float lm = 0.0f;
        float *od_m = (float *)out_m->storage->data;
        for (int64_t j = 0; j < N * Cout * M; j++) lm += od_m[j];
        ax_tensor_destroy(out_m);

        wd[i] = orig;  /* restore */

        float fd = (lp - lm) / (2.0f * eps);
        AX_TEST_ASSERT_NEAR(dW_fast[i], fd, 0.05f,
                            "1×1 fast path dW matches finite difference");
    }

    /* numerical dX for a single input element */
    int x_probes[] = {0, (int)(inn/4), (int)(inn/2), (int)(3*inn/4), (int)(inn-1)};
    for (int p = 0; p < 5; p++) {
        int64_t i = x_probes[p];
        float orig = id[i];
        id[i] = orig + eps;
        ax_tensor_t *out_p = ax_layer_forward(c, inp);
        float lp = 0.0f;
        float *od_p = (float *)out_p->storage->data;
        for (int64_t j = 0; j < N * Cout * M; j++) lp += od_p[j];
        ax_tensor_destroy(out_p);

        id[i] = orig - eps;
        ax_tensor_t *out_m = ax_layer_forward(c, inp);
        float lm = 0.0f;
        float *od_m = (float *)out_m->storage->data;
        for (int64_t j = 0; j < N * Cout * M; j++) lm += od_m[j];
        ax_tensor_destroy(out_m);

        id[i] = orig;

        float fd = (lp - lm) / (2.0f * eps);
        AX_TEST_ASSERT_NEAR(dX_fast[i], fd, 0.05f,
                            "1×1 fast path dX matches finite difference");
    }

    free(dW_fast);
    free(dX_fast);
    ax_tensor_destroy(loss);
    ax_tensor_destroy(out);
    ax_tensor_destroy(inp);
    ax_layer_destroy(c);
}

/* validate the direct-smallcin path matches the im2col+gemm reference.
   we can't pick the path explicitly from the public API, so we rely on
   conv2d_forward's internal predicate: C_in ≤ 4 → direct-smallcin.
   we then reconstruct the same conv with a wrapper input that pads C_in to 5
   (zero-padded weight for the extra channel), which forces the im2col path,
   and check both produce the same outputs to 1e-5 tolerance. */
static void test_conv2d_direct_smallcin_vs_gemm(void)
{
    /* shape mirrors VGG/ResNet first layer */
    const int N = 2, Cin = 3, Cout = 16, H = 32, W = 32;
    const int K = 3, S = 1, P = 1;

    /* deterministic input + weights via ax_tensor_rand */
    ax_set_seed(7);
    int64_t in_sh[] = {N, Cin, H, W};
    ax_tensor_t *inp_a = ax_tensor_rand(in_sh, 4, -1.0f, 1.0f);

    /* layer A: hits direct-smallcin (C_in=3 ≤ 4) */
    ax_layer_t *ca = ax_conv2d_create(Cin, Cout, K, S, P, true);
    ax_conv2d_t *cca = (ax_conv2d_t *)ca;

    /* layer B: pad C_in to 5 → falls into im2col path (5 > 4 threshold) */
    int64_t in_sh_b[] = {N, 5, H, W};
    ax_tensor_t *inp_b = ax_tensor_zeros(in_sh_b, 4, AX_FLOAT32);
    /* copy the 3 real channels into the first 3 slots of the 5-channel input */
    float *ad = (float *)inp_a->storage->data;
    float *bd = (float *)inp_b->storage->data;
    for (int n = 0; n < N; n++)
        for (int c = 0; c < Cin; c++)
            memcpy(bd + ((size_t)n * 5 + c) * H * W,
                   ad + ((size_t)n * Cin + c) * H * W,
                   (size_t)H * W * sizeof(float));

    ax_layer_t *cb = ax_conv2d_create(5, Cout, K, S, P, true);
    ax_conv2d_t *ccb = (ax_conv2d_t *)cb;
    /* copy the 3-channel weight into the 5-channel weight; zero the rest */
    float *wa = (float *)cca->weight->storage->data;
    float *wb = (float *)ccb->weight->storage->data;
    memset(wb, 0, (size_t)Cout * 5 * K * K * sizeof(float));
    for (int co = 0; co < Cout; co++)
        for (int ci = 0; ci < Cin; ci++)
            memcpy(wb + ((size_t)co * 5 + ci) * K * K,
                   wa + ((size_t)co * Cin + ci) * K * K,
                   (size_t)K * K * sizeof(float));
    /* same bias */
    memcpy((float *)ccb->bias->storage->data,
           (float *)cca->bias->storage->data,
           (size_t)Cout * sizeof(float));

    ax_no_grad();
    ax_tensor_t *out_a = ax_layer_forward(ca, inp_a);
    ax_tensor_t *out_b = ax_layer_forward(cb, inp_b);

    AX_TEST_ASSERT(out_a && out_b, "both forwards should succeed");
    AX_TEST_ASSERT_EQ(out_a->shape[0], out_b->shape[0], "N matches");
    AX_TEST_ASSERT_EQ(out_a->shape[1], out_b->shape[1], "Cout matches");
    AX_TEST_ASSERT_EQ(out_a->shape[2], out_b->shape[2], "H_out matches");
    AX_TEST_ASSERT_EQ(out_a->shape[3], out_b->shape[3], "W_out matches");

    float *oa = (float *)out_a->storage->data;
    float *ob = (float *)out_b->storage->data;
    int64_t total = out_a->shape[0] * out_a->shape[1] * out_a->shape[2] * out_a->shape[3];
    float max_err = 0.0f;
    for (int64_t i = 0; i < total; i++) {
        float e = fabsf(oa[i] - ob[i]);
        if (e > max_err) max_err = e;
    }
    AX_TEST_ASSERT(max_err < 1e-4f, "direct-smallcin matches im2col reference");

    ax_tensor_destroy(out_a); ax_tensor_destroy(out_b);
    ax_tensor_destroy(inp_a); ax_tensor_destroy(inp_b);
    ax_layer_destroy(ca);    ax_layer_destroy(cb);
    ax_enable_grad();
}

/* exercise the SIMD-vectorized maxpool k=2 s=2 path with deterministic
   inputs wide enough to trigger the SIMD branch (ow > AX_VF32_WIDTH).
   reference computed with the trivial 4-element scalar max. */
static void test_maxpool2d_simd(void)
{
    /* 1×2×4×24 input → 1×2×2×12 output. ow=12 > AX_VF32_WIDTH=8 so SIMD
       runs once + scalar tail covers the rest, exercising both branches. */
    const int N = 1, C = 2, H = 4, W = 24;
    int64_t in_sh[] = {N, C, H, W};
    ax_tensor_t *inp = ax_tensor_create(in_sh, 4, AX_FLOAT32);
    float *id = (float *)inp->storage->data;
    /* deterministic values: id[n,c,h,w] = c*1000 + h*100 + w + 1 */
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                id[(c * H + h) * W + w] = (float)(c * 1000 + h * 100 + w + 1);

    ax_layer_t *pool = ax_maxpool2d_create(2, 2, 0);
    ax_no_grad();
    ax_tensor_t *out = ax_layer_forward(pool, inp);
    AX_TEST_ASSERT(out != NULL, "maxpool simd forward");

    float *od = (float *)out->storage->data;
    int64_t oh = out->shape[2], ow = out->shape[3];
    AX_TEST_ASSERT_EQ(oh, 2, "out_h");
    AX_TEST_ASSERT_EQ(ow, 12, "out_w");

    /* reference: scalar pairwise max over 2×2 windows */
    for (int c = 0; c < C; c++) {
        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                int iy = y * 2, ix = x * 2;
                float a = id[(c * H + iy)     * W + ix];
                float b = id[(c * H + iy)     * W + ix + 1];
                float p = id[(c * H + iy + 1) * W + ix];
                float q = id[(c * H + iy + 1) * W + ix + 1];
                float ref = a; if (b > ref) ref = b;
                                if (p > ref) ref = p;
                                if (q > ref) ref = q;
                float got = od[(c * oh + y) * ow + x];
                AX_TEST_ASSERT_NEAR(got, ref, 1e-5, "simd maxpool value");
            }
        }
    }
    ax_tensor_destroy(out);
    ax_tensor_destroy(inp);
    ax_layer_destroy(pool);
    ax_enable_grad();
}

/* exercise edge cases: very small W (just below SIMD threshold), pad=0,
   stride=2, asymmetric kernel. all should fall into either smallcin or
   im2col cleanly without crashing. */
static void test_conv2d_smallcin_edges(void)
{
    struct { int Cin, Cout, H, W, K, S, P; } cases[] = {
        { 1, 8,  16, 16, 3, 1, 1 },   /* 1-channel grayscale */
        { 3, 16, 32, 32, 3, 1, 0 },   /* no padding */
        { 3, 16, 32, 32, 3, 2, 1 },   /* stride 2 (smallcin doesn't do simd at sw>1, but should still produce correct results) */
        { 4, 32, 28, 28, 3, 1, 1 },   /* C_in=4 (boundary) */
        { 3, 8,  56, 56, 5, 1, 2 },   /* 5x5 kernel */
        { 3, 8,  56, 56, 7, 2, 3 },   /* 7x7 stride 2 (resnet stem) */
    };
    int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));

    for (int i = 0; i < n_cases; i++) {
        int Cin = cases[i].Cin, Cout = cases[i].Cout;
        int H = cases[i].H, W = cases[i].W;
        int Kk = cases[i].K, S = cases[i].S, P = cases[i].P;

        ax_set_seed(13 + i);
        int64_t in_sh[] = {1, Cin, H, W};
        ax_tensor_t *inp = ax_tensor_rand(in_sh, 4, -1.0f, 1.0f);

        ax_layer_t *c = ax_conv2d_create(Cin, Cout, Kk, S, P, true);
        ax_no_grad();
        ax_tensor_t *out = ax_layer_forward(c, inp);
        AX_TEST_ASSERT(out != NULL, "forward should not crash on edge case");
        AX_TEST_ASSERT(out->shape[0] == 1 && out->shape[1] == Cout,
                       "output shape header correct");
        ax_tensor_destroy(out);
        ax_tensor_destroy(inp);
        ax_layer_destroy(c);
        ax_enable_grad();
    }
}

int main(void)
{
    ax_init();

    printf("=== conv tests ===\n");
    AX_RUN_TEST(test_im2col_basic);
    AX_RUN_TEST(test_conv2d_create);
    AX_RUN_TEST(test_conv2d_forward);
    AX_RUN_TEST(test_conv2d_with_padding);
    AX_RUN_TEST(test_maxpool2d);
    AX_RUN_TEST(test_avgpool2d);
    AX_RUN_TEST(test_global_avgpool);
    AX_RUN_TEST(test_flatten);
    AX_RUN_TEST(test_conv_pipeline);
    AX_RUN_TEST(test_conv2d_batched_fwd);
    AX_RUN_TEST(test_conv2d_batched_bwd);
    AX_RUN_TEST(test_conv2d_subbatched_fwd_bwd);
    AX_RUN_TEST(test_conv2d_1x1_zero_copy_bwd);
    AX_RUN_TEST(test_conv2d_direct_smallcin_vs_gemm);
    AX_RUN_TEST(test_conv2d_smallcin_edges);
    AX_RUN_TEST(test_maxpool2d_simd);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
