/* test_cuda.c — CUDA backend tests.
   Without AX_HAVE_CUDA: verifies CPU path is unaffected and stub APIs fail gracefully.
   With AX_HAVE_CUDA + a GPU: exercises the full GPU path end-to-end. */

#include "test.h"
#include "axiom/axiom.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float cpu_f32(ax_tensor_t *t, int64_t i) {
    return ((float *)t->storage->data)[t->offset + i];
}

/* ── tests that run regardless of CUDA availability ──────────────────────── */

static void test_default_device_is_cpu(void) {
    AX_TEST_ASSERT(ax_get_default_device() == AX_DEVICE_CPU,
                   "default device should be CPU");
}

static void test_cpu_tensors_unaffected(void) {
    int64_t shape[] = {4, 4};
    ax_tensor_t *a = ax_tensor_zeros(shape, 2, AX_FLOAT32);
    AX_TEST_ASSERT(a != NULL, "cpu zeros should succeed");
    AX_TEST_ASSERT(a->storage->device == AX_DEVICE_CPU, "storage should be CPU");
    AX_TEST_ASSERT(cpu_f32(a, 0) == 0.0f, "zeros should be zero");
    ax_tensor_destroy(a);
}

static void test_to_cpu_on_cpu_tensor(void) {
    int64_t shape[] = {3};
    float data[] = {1.0f, 2.0f, 3.0f};
    ax_tensor_t *t = ax_tensor_from_array(data, shape, 1, AX_FLOAT32);
    ax_tensor_t *c = ax_tensor_to_cpu(t);
    AX_TEST_ASSERT(c != NULL, "ax_tensor_to_cpu on cpu tensor should succeed");
    AX_TEST_ASSERT(c->storage == t->storage, "should share storage (view)");
    AX_TEST_ASSERT_NEAR(cpu_f32(c, 0), 1.0f, 1e-6f, "element 0");
    AX_TEST_ASSERT_NEAR(cpu_f32(c, 2), 3.0f, 1e-6f, "element 2");
    ax_tensor_destroy(c);
    ax_tensor_destroy(t);
}

static void test_cpu_gemm_still_works(void) {
    float a_data[] = {1.f, 2.f, 3.f, 4.f};
    float b_data[] = {5.f, 6.f, 7.f, 8.f};
    int64_t s22[] = {2, 2};
    ax_tensor_t *a = ax_tensor_from_array(a_data, s22, 2, AX_FLOAT32);
    ax_tensor_t *b = ax_tensor_from_array(b_data, s22, 2, AX_FLOAT32);
    ax_tensor_t *c = ax_tensor_zeros(s22, 2, AX_FLOAT32);
    ax_status_t st = ax_compute_gemm(a, b, c);
    AX_TEST_ASSERT(st == AX_OK, "cpu gemm should succeed");
    AX_TEST_ASSERT_NEAR(cpu_f32(c, 0), 19.f, 1e-4f, "c[0,0]=19");
    AX_TEST_ASSERT_NEAR(cpu_f32(c, 1), 22.f, 1e-4f, "c[0,1]=22");
    AX_TEST_ASSERT_NEAR(cpu_f32(c, 2), 43.f, 1e-4f, "c[1,0]=43");
    AX_TEST_ASSERT_NEAR(cpu_f32(c, 3), 50.f, 1e-4f, "c[1,1]=50");
    ax_tensor_destroy(a); ax_tensor_destroy(b); ax_tensor_destroy(c);
}

/* ── GPU tests (only when compiled with AX_HAVE_CUDA and a device is found) ── */

#ifdef AX_HAVE_CUDA
#include <cuda_runtime.h>

static float gpu_read(ax_tensor_t *t, int64_t i) {
    float v;
    cudaMemcpy(&v, (float *)t->storage->data + t->offset + i,
               sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

static ax_tensor_t *gpu_from_array(const float *data, const int64_t *shape, int ndim) {
    ax_tensor_t *cpu = ax_tensor_from_array(data, shape, ndim, AX_FLOAT32);
    ax_tensor_t *gpu = ax_tensor_to_cuda(cpu);
    ax_tensor_destroy(cpu);
    return gpu;
}

static void test_gpu_device_count(void) {
    int n = ax_cuda_device_count();
    AX_TEST_ASSERT(n > 0, "should have at least one CUDA device");
    printf("  (cuda device count: %d)\n", n);
}

static void test_gpu_tensor_zeros(void) {
    ax_set_default_device(AX_DEVICE_CUDA);
    int64_t shape[] = {8};
    ax_tensor_t *t = ax_tensor_zeros(shape, 1, AX_FLOAT32);
    AX_TEST_ASSERT(t != NULL, "gpu zeros should succeed");
    AX_TEST_ASSERT(t->storage->device == AX_DEVICE_CUDA, "storage on GPU");
    for (int i = 0; i < 8; i++)
        AX_TEST_ASSERT_NEAR(gpu_read(t, i), 0.0f, 1e-6f, "element should be 0");
    ax_tensor_destroy(t);
    ax_set_default_device(AX_DEVICE_CPU);
}

static void test_gpu_roundtrip(void) {
    float data[] = {1.f, 2.f, 3.f, 4.f, 5.f};
    int64_t shape[] = {5};
    ax_tensor_t *cpu_in  = ax_tensor_from_array(data, shape, 1, AX_FLOAT32);
    ax_tensor_t *gpu     = ax_tensor_to_cuda(cpu_in);
    ax_tensor_t *cpu_out = ax_tensor_to_cpu(gpu);
    AX_TEST_ASSERT(gpu     != NULL, "to_cuda should succeed");
    AX_TEST_ASSERT(cpu_out != NULL, "to_cpu should succeed");
    AX_TEST_ASSERT(gpu->storage->device     == AX_DEVICE_CUDA, "gpu on CUDA");
    AX_TEST_ASSERT(cpu_out->storage->device == AX_DEVICE_CPU,  "cpu_out on CPU");
    for (int i = 0; i < 5; i++)
        AX_TEST_ASSERT_NEAR(cpu_f32(cpu_out, i), data[i], 1e-6f, "roundtrip value");
    ax_tensor_destroy(cpu_in); ax_tensor_destroy(gpu); ax_tensor_destroy(cpu_out);
}

static void test_gpu_fill_and_scalar(void) {
    ax_compute_set_backend(AX_BACKEND_CUDA);
    ax_set_default_device(AX_DEVICE_CUDA);
    int64_t shape[] = {16};

    ax_tensor_t *t   = ax_tensor_zeros(shape, 1, AX_FLOAT32);
    ax_tensor_t *out = ax_tensor_zeros(shape, 1, AX_FLOAT32);
    ax_compute_fill(t, 3.0);
    for (int i = 0; i < 16; i++)
        AX_TEST_ASSERT_NEAR(gpu_read(t, i), 3.0f, 1e-5f, "fill value");

    ax_compute_add_scalar(t, 2.0, out);
    ax_cuda_synchronize();
    for (int i = 0; i < 16; i++)
        AX_TEST_ASSERT_NEAR(gpu_read(out, i), 5.0f, 1e-5f, "add_scalar result");

    ax_tensor_destroy(t); ax_tensor_destroy(out);
    ax_set_default_device(AX_DEVICE_CPU);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
}

static void test_gpu_elementwise(void) {
    ax_compute_set_backend(AX_BACKEND_CUDA);
    ax_set_default_device(AX_DEVICE_CUDA);
    int64_t shape[] = {4};

    float a_data[] = {1.f, 2.f, 3.f, 4.f};
    float b_data[] = {4.f, 3.f, 2.f, 1.f};
    ax_tensor_t *a   = gpu_from_array(a_data, shape, 1);
    ax_tensor_t *b   = gpu_from_array(b_data, shape, 1);
    ax_tensor_t *out = ax_tensor_zeros(shape, 1, AX_FLOAT32);

    ax_compute_add(a, b, out);
    ax_cuda_synchronize();
    for (int i = 0; i < 4; i++)
        AX_TEST_ASSERT_NEAR(gpu_read(out, i), 5.f, 1e-5f, "add: all should be 5");

    ax_compute_mul(a, b, out);
    ax_cuda_synchronize();
    float expected_mul[] = {4.f, 6.f, 6.f, 4.f};
    for (int i = 0; i < 4; i++)
        AX_TEST_ASSERT_NEAR(gpu_read(out, i), expected_mul[i], 1e-5f, "mul result");

    float neg_data[] = {-2.f, -1.f, 0.f, 1.f};
    ax_tensor_t *neg = gpu_from_array(neg_data, shape, 1);
    ax_compute_relu(neg, out);
    ax_cuda_synchronize();
    float expected_relu[] = {0.f, 0.f, 0.f, 1.f};
    for (int i = 0; i < 4; i++)
        AX_TEST_ASSERT_NEAR(gpu_read(out, i), expected_relu[i], 1e-5f, "relu result");

    ax_tensor_destroy(a); ax_tensor_destroy(b);
    ax_tensor_destroy(neg); ax_tensor_destroy(out);
    ax_set_default_device(AX_DEVICE_CPU);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
}

static void test_gpu_gemm(void) {
    ax_compute_set_backend(AX_BACKEND_CUDA);
    ax_set_default_device(AX_DEVICE_CUDA);

    float a_data[] = {1.f, 2.f, 3.f, 4.f};
    float b_data[] = {5.f, 6.f, 7.f, 8.f};
    int64_t s22[] = {2, 2};
    ax_tensor_t *a = gpu_from_array(a_data, s22, 2);
    ax_tensor_t *b = gpu_from_array(b_data, s22, 2);
    ax_tensor_t *c = ax_tensor_zeros(s22, 2, AX_FLOAT32);

    ax_status_t st = ax_compute_gemm(a, b, c);
    ax_cuda_synchronize();
    AX_TEST_ASSERT(st == AX_OK, "gpu gemm should succeed");

    float expected[] = {19.f, 22.f, 43.f, 50.f};
    for (int i = 0; i < 4; i++)
        AX_TEST_ASSERT_NEAR(gpu_read(c, i), expected[i], 1e-3f, "gemm element");

    ax_tensor_destroy(a); ax_tensor_destroy(b); ax_tensor_destroy(c);
    ax_set_default_device(AX_DEVICE_CPU);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
}

static void test_gpu_sum_reduction(void) {
    ax_compute_set_backend(AX_BACKEND_CUDA);
    ax_set_default_device(AX_DEVICE_CUDA);

    float data[64];
    for (int i = 0; i < 64; i++) data[i] = 1.0f;
    int64_t shape[] = {64};
    int64_t scalar_shape[] = {1};
    ax_tensor_t *in  = gpu_from_array(data, shape, 1);
    ax_tensor_t *out = ax_tensor_zeros(scalar_shape, 1, AX_FLOAT32);

    ax_compute_sum(in, -1, out);
    ax_cuda_synchronize();
    AX_TEST_ASSERT_NEAR(gpu_read(out, 0), 64.0f, 1e-3f, "sum of 64 ones = 64");

    ax_tensor_destroy(in); ax_tensor_destroy(out);
    ax_set_default_device(AX_DEVICE_CPU);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
}

static void test_gpu_larger_gemm(void) {
    int M = 64, K = 128, N = 32;
    int64_t sa[] = {M, K}, sb[] = {K, N}, sc[] = {M, N};

    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
    ax_set_default_device(AX_DEVICE_CPU);
    ax_tensor_t *cpu_a = ax_tensor_rand(sa, 2, -0.1f, 0.1f);
    ax_tensor_t *cpu_b = ax_tensor_rand(sb, 2, -0.1f, 0.1f);
    ax_tensor_t *cpu_c = ax_tensor_zeros(sc, 2, AX_FLOAT32);
    ax_compute_gemm(cpu_a, cpu_b, cpu_c);

    ax_compute_set_backend(AX_BACKEND_CUDA);
    ax_tensor_t *gpu_a = ax_tensor_to_cuda(cpu_a);
    ax_tensor_t *gpu_b = ax_tensor_to_cuda(cpu_b);
    ax_set_default_device(AX_DEVICE_CUDA);
    ax_tensor_t *gpu_c = ax_tensor_zeros(sc, 2, AX_FLOAT32);
    ax_compute_gemm(gpu_a, gpu_b, gpu_c);
    ax_cuda_synchronize();

    ax_tensor_t *check = ax_tensor_to_cpu(gpu_c);
    int mismatches = 0;
    for (int i = 0; i < M * N; i++)
        if (fabsf(cpu_f32(check, i) - cpu_f32(cpu_c, i)) > 1e-3f) mismatches++;
    AX_TEST_ASSERT(mismatches == 0, "large gemm: gpu vs cpu mismatch");
    printf("  (64x128 @ 128x32: %d mismatches vs CPU)\n", mismatches);

    ax_tensor_destroy(cpu_a); ax_tensor_destroy(cpu_b); ax_tensor_destroy(cpu_c);
    ax_tensor_destroy(gpu_a); ax_tensor_destroy(gpu_b); ax_tensor_destroy(gpu_c);
    ax_tensor_destroy(check);
    ax_set_default_device(AX_DEVICE_CPU);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
}

/* conv_gemm parity: run cuda_conv_gemm on a small sample and compare
   against the cpu reference computed via the current active cpu backend's
   im2col + gemm. */
static void test_gpu_conv_gemm(void) {
    /* tiny 3-channel 4x4 input, 2 output channels, 3x3 kernel, stride 1,
       pad 1 -> out 4x4. sizes chosen to keep the reference fast. */
    const int C_in = 3, C_out = 2, H = 4, W = 4;
    const int kh = 3, kw = 3, sh = 1, sw = 1, ph = 1, pw = 1;
    const int out_h = (H + 2*ph - kh) / sh + 1;
    const int out_w = (W + 2*pw - kw) / sw + 1;
    const int K = C_in * kh * kw, M = out_h * out_w;

    /* cpu reference: build input and weight on cpu, run cpu gemm via the
       explicit im2col path using the existing ax_compute_conv_gemm. the
       cpu backend implements conv_gemm too, so we can reuse the same api. */
    int64_t in_shape[]  = {C_in, H, W};
    int64_t w_shape[]   = {C_out, K};
    int64_t out_shape[] = {C_out, M};

    float *in_data = (float *)malloc((size_t)C_in * H * W * sizeof(float));
    float *w_data  = (float *)malloc((size_t)C_out * K * sizeof(float));
    for (int i = 0; i < C_in * H * W; i++) in_data[i] = 0.01f * (float)(i + 1);
    for (int i = 0; i < C_out * K;    i++) w_data[i]  = 0.02f * (float)(i + 1) - 0.1f;

    /* cpu leg */
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
    ax_set_default_device(AX_DEVICE_CPU);
    ax_tensor_t *cpu_in  = ax_tensor_from_array(in_data, in_shape, 3, AX_FLOAT32);
    ax_tensor_t *cpu_w   = ax_tensor_from_array(w_data,  w_shape,  2, AX_FLOAT32);
    ax_tensor_t *cpu_out = ax_tensor_zeros(out_shape, 2, AX_FLOAT32);

    AX_TEST_ASSERT(ax_compute_has_conv_gemm() != 0,
                   "cpu backend should implement conv_gemm for reference");
    ax_conv_params_t cp_cpu = {
        .input = (const float *)cpu_in->storage->data,
        .C_in = C_in, .H = H, .W = W,
        .kh = kh, .kw = kw,
        .sh = sh, .sw = sw,
        .ph = ph, .pw = pw,
        .out_h = out_h, .out_w = out_w,
    };
    AX_TEST_ASSERT(ax_compute_conv_gemm(cpu_w, &cp_cpu, cpu_out) == AX_OK,
                   "cpu conv_gemm should succeed");

    /* gpu leg */
    ax_compute_set_backend(AX_BACKEND_CUDA);
    ax_tensor_t *gpu_in  = ax_tensor_to_cuda(cpu_in);
    ax_tensor_t *gpu_w   = ax_tensor_to_cuda(cpu_w);
    ax_set_default_device(AX_DEVICE_CUDA);
    ax_tensor_t *gpu_out = ax_tensor_zeros(out_shape, 2, AX_FLOAT32);

    AX_TEST_ASSERT(ax_compute_has_conv_gemm() != 0,
                   "cuda backend should implement conv_gemm");
    ax_conv_params_t cp_gpu = {
        .input = (const float *)gpu_in->storage->data,
        .C_in = C_in, .H = H, .W = W,
        .kh = kh, .kw = kw,
        .sh = sh, .sw = sw,
        .ph = ph, .pw = pw,
        .out_h = out_h, .out_w = out_w,
    };
    AX_TEST_ASSERT(ax_compute_conv_gemm(gpu_w, &cp_gpu, gpu_out) == AX_OK,
                   "gpu conv_gemm should succeed");
    ax_cuda_synchronize();

    /* compare via bulk d2h */
    ax_tensor_t *check = ax_tensor_to_cpu(gpu_out);
    int mismatches = 0;
    for (int i = 0; i < C_out * M; i++) {
        float a = cpu_f32(cpu_out, i);
        float b = cpu_f32(check,   i);
        if (fabsf(a - b) > 1e-3f) mismatches++;
    }
    AX_TEST_ASSERT(mismatches == 0, "gpu vs cpu conv_gemm mismatch");
    printf("  (%dx%d @ %d,%d kernel: %d mismatches vs cpu)\n",
           C_in, H*W, kh, kw, mismatches);

    ax_tensor_destroy(cpu_in); ax_tensor_destroy(cpu_w); ax_tensor_destroy(cpu_out);
    ax_tensor_destroy(gpu_in); ax_tensor_destroy(gpu_w); ax_tensor_destroy(gpu_out);
    ax_tensor_destroy(check);
    free(in_data); free(w_data);
    ax_set_default_device(AX_DEVICE_CPU);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
}

/* I.2: Winograd F(2,3) parity vs the existing im2col + cuBLAS path.
   builds N=2 samples, C_in=C_out=32, 8x8 input, 3x3 kernel pad=1
   (a Winograd-friendly shape: even dims, C ≥ heuristic threshold).
   exercises:
     1. weight transform U = G * w * G^T
     2. input transform V = B^T * d * B over multiple tiles + samples
     3. cuBLAS batched gemm
     4. output transform Y = A^T * M * A back to NCHW
   compares against cuda_conv_gemm_batched element-wise; mismatches
   indicate either a Winograd-matrix bug or an indexing bug. */
extern ax_status_t cuda_conv_gemm_batched(const ax_tensor_t *,
                                            const float *, int64_t,
                                            const ax_conv_params_t *,
                                            ax_tensor_t *);
extern ax_status_t cuda_conv_winograd_f23(const ax_tensor_t *,
                                            const float *, int64_t,
                                            const ax_conv_params_t *,
                                            ax_tensor_t *);
static void test_gpu_winograd_f23_parity(void) {
    const int N = 2, C_in = 32, C_out = 32, H = 8, W = 8;
    const int kh = 3, kw = 3, sh = 1, sw = 1, ph = 1, pw = 1;
    const int out_h = (H + 2*ph - kh) / sh + 1;
    const int out_w = (W + 2*pw - kw) / sw + 1;
    const int M = out_h * out_w;
    const int K = C_in * kh * kw;

    int64_t in_shape[]  = {N, C_in, H, W};
    int64_t w_shape[]   = {C_out, K};
    int64_t out_shape[] = {N, C_out, M};

    /* deterministic init */
    int64_t in_n  = (int64_t)N * C_in * H * W;
    int64_t w_n   = (int64_t)C_out * K;
    float *h_in  = (float *)malloc((size_t)in_n * sizeof(float));
    float *h_w   = (float *)malloc((size_t)w_n  * sizeof(float));
    for (int64_t i = 0; i < in_n; i++) h_in[i] = 0.005f * (float)((i * 7 + 3) % 199 - 99);
    for (int64_t i = 0; i < w_n;  i++) h_w[i]  = 0.01f  * (float)((i * 5 + 1) % 79  - 39);

    /* push to gpu */
    ax_compute_set_backend(AX_BACKEND_CUDA);
    ax_set_default_device(AX_DEVICE_CUDA);
    ax_tensor_t *cpu_in  = ax_tensor_from_array(h_in, in_shape, 4, AX_FLOAT32);
    ax_tensor_t *cpu_w   = ax_tensor_from_array(h_w,  w_shape,  2, AX_FLOAT32);
    ax_tensor_t *gpu_in  = ax_tensor_to_cuda(cpu_in);
    ax_tensor_t *gpu_w   = ax_tensor_to_cuda(cpu_w);
    ax_tensor_t *out_ref = ax_tensor_zeros(out_shape, 3, AX_FLOAT32);
    ax_tensor_t *out_wno = ax_tensor_zeros(out_shape, 3, AX_FLOAT32);

    ax_conv_params_t cp = {
        .input = NULL,  /* unused — input passed separately via input_batched */
        .C_in = C_in, .H = H, .W = W,
        .kh = kh, .kw = kw,
        .sh = sh, .sw = sw,
        .ph = ph, .pw = pw,
        .out_h = out_h, .out_w = out_w,
    };
    const float *in_dev = (const float *)gpu_in->storage->data;

    /* reference path */
    AX_TEST_ASSERT(cuda_conv_gemm_batched(gpu_w, in_dev, N, &cp, out_ref) == AX_OK,
                   "im2col reference conv should succeed");
    /* winograd path */
    AX_TEST_ASSERT(cuda_conv_winograd_f23(gpu_w, in_dev, N, &cp, out_wno) == AX_OK,
                   "winograd_f23 conv should succeed");
    ax_cuda_synchronize();

    ax_tensor_t *check_ref = ax_tensor_to_cpu(out_ref);
    ax_tensor_t *check_wno = ax_tensor_to_cpu(out_wno);
    int mismatches = 0;
    float max_err = 0.0f;
    int64_t total = (int64_t)N * C_out * M;
    for (int64_t i = 0; i < total; i++) {
        float a = cpu_f32(check_ref, i);
        float b = cpu_f32(check_wno, i);
        float e = fabsf(a - b);
        if (e > max_err) max_err = e;
        /* Winograd has slightly worse fp32 numerical conditioning than
           direct due to the larger sum of products in the transforms;
           tol of 5e-3 is comfortable for this shape. */
        if (e > 5e-3f) mismatches++;
    }
    AX_TEST_ASSERT(mismatches == 0, "winograd_f23 vs im2col mismatch");
    printf("  (N=%d C=%d %dx%d: %d mismatches, max_err=%g)\n",
           N, C_in, H, W, mismatches, max_err);

    ax_tensor_destroy(cpu_in); ax_tensor_destroy(cpu_w);
    ax_tensor_destroy(gpu_in); ax_tensor_destroy(gpu_w);
    ax_tensor_destroy(out_ref); ax_tensor_destroy(out_wno);
    ax_tensor_destroy(check_ref); ax_tensor_destroy(check_wno);
    free(h_in); free(h_w);
    ax_set_default_device(AX_DEVICE_CPU);
    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
}

/* fused-op parity: add_relu, axpy, softmax_rowwise vs cpu reference. */
static void test_gpu_fused_ops(void) {
    /* add_relu */
    {
        float a_data[] = {-1.f,  2.f, -3.f, 4.f,  5.f, -6.f};
        float b_data[] = { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
        int64_t shape[] = {2, 3};
        ax_compute_set_backend(AX_BACKEND_CUDA);
        ax_tensor_t *ga = gpu_from_array(a_data, shape, 2);
        ax_tensor_t *gb = gpu_from_array(b_data, shape, 2);
        ax_set_default_device(AX_DEVICE_CUDA);
        ax_tensor_t *gout = ax_tensor_zeros(shape, 2, AX_FLOAT32);

        AX_TEST_ASSERT(ax_compute_has_add_relu(), "cuda backend should have add_relu");
        AX_TEST_ASSERT(ax_compute_add_relu(ga, gb, gout) == AX_OK, "add_relu should succeed");
        ax_cuda_synchronize();

        /* expected: relu(a + b) */
        float expected[] = {0.f, 2.5f, 0.f, 4.5f, 5.5f, 0.f};
        for (int i = 0; i < 6; i++)
            AX_TEST_ASSERT_NEAR(gpu_read(gout, i), expected[i], 1e-5f, "add_relu element");

        ax_tensor_destroy(ga); ax_tensor_destroy(gb); ax_tensor_destroy(gout);
        ax_set_default_device(AX_DEVICE_CPU);
    }

    /* axpy: y += 0.5 * x */
    {
        float x_data[] = {1.f, 2.f, 3.f, 4.f};
        float y_data[] = {10.f, 20.f, 30.f, 40.f};
        int64_t shape[] = {4};
        ax_compute_set_backend(AX_BACKEND_CUDA);
        ax_tensor_t *gx = gpu_from_array(x_data, shape, 1);
        ax_tensor_t *gy = gpu_from_array(y_data, shape, 1);

        AX_TEST_ASSERT(ax_compute_has_axpy(), "cuda backend should have axpy");
        AX_TEST_ASSERT(ax_compute_axpy(gx, 0.5f, gy) == AX_OK, "axpy should succeed");
        ax_cuda_synchronize();

        float expected[] = {10.5f, 21.f, 31.5f, 42.f};
        for (int i = 0; i < 4; i++)
            AX_TEST_ASSERT_NEAR(gpu_read(gy, i), expected[i], 1e-5f, "axpy result");

        ax_tensor_destroy(gx); ax_tensor_destroy(gy);
    }

    /* softmax_rowwise: stable row-wise softmax */
    {
        float in_data[] = {1.f, 2.f, 3.f,
                           -1.f, -2.f, -3.f};
        int64_t shape[] = {2, 3};
        ax_compute_set_backend(AX_BACKEND_CUDA);
        ax_tensor_t *gin = gpu_from_array(in_data, shape, 2);
        ax_set_default_device(AX_DEVICE_CUDA);
        ax_tensor_t *gout = ax_tensor_zeros(shape, 2, AX_FLOAT32);

        AX_TEST_ASSERT(ax_compute_has_softmax_rowwise(), "cuda backend should have softmax_rowwise");
        AX_TEST_ASSERT(ax_compute_softmax_rowwise(gin, gout) == AX_OK, "softmax_rowwise should succeed");
        ax_cuda_synchronize();

        /* expected: softmax([1,2,3]) = [0.0900, 0.2447, 0.6652] */
        AX_TEST_ASSERT_NEAR(gpu_read(gout, 0), 0.09003f, 1e-4f, "softmax row0 [0]");
        AX_TEST_ASSERT_NEAR(gpu_read(gout, 1), 0.24473f, 1e-4f, "softmax row0 [1]");
        AX_TEST_ASSERT_NEAR(gpu_read(gout, 2), 0.66524f, 1e-4f, "softmax row0 [2]");
        /* row 1 is the same but reversed because of the negative sign */
        AX_TEST_ASSERT_NEAR(gpu_read(gout, 3), 0.66524f, 1e-4f, "softmax row1 [0]");
        AX_TEST_ASSERT_NEAR(gpu_read(gout, 4), 0.24473f, 1e-4f, "softmax row1 [1]");
        AX_TEST_ASSERT_NEAR(gpu_read(gout, 5), 0.09003f, 1e-4f, "softmax row1 [2]");

        /* verify each row sums to 1 */
        float r0 = gpu_read(gout, 0) + gpu_read(gout, 1) + gpu_read(gout, 2);
        float r1 = gpu_read(gout, 3) + gpu_read(gout, 4) + gpu_read(gout, 5);
        AX_TEST_ASSERT_NEAR(r0, 1.0f, 1e-4f, "softmax row0 sums to 1");
        AX_TEST_ASSERT_NEAR(r1, 1.0f, 1e-4f, "softmax row1 sums to 1");

        ax_tensor_destroy(gin); ax_tensor_destroy(gout);
        ax_set_default_device(AX_DEVICE_CPU);
    }

    ax_compute_set_backend(AX_BACKEND_CPU_SIMD);
}
#endif /* AX_HAVE_CUDA */

/* ── main ─────────────────────────────────────────────────────────────────── */

int main(void) {
    ax_init();
    printf("=== test_cuda (compiled %s CUDA) ===\n",
#ifdef AX_HAVE_CUDA
           "WITH"
#else
           "WITHOUT"
#endif
    );

    AX_RUN_TEST(test_default_device_is_cpu);
    AX_RUN_TEST(test_cpu_tensors_unaffected);
    AX_RUN_TEST(test_to_cpu_on_cpu_tensor);
    AX_RUN_TEST(test_cpu_gemm_still_works);

#ifdef AX_HAVE_CUDA
    if (ax_cuda_device_count() > 0) {
        printf("--- GPU present, running GPU tests ---\n");
        AX_RUN_TEST(test_gpu_device_count);
        AX_RUN_TEST(test_gpu_tensor_zeros);
        AX_RUN_TEST(test_gpu_roundtrip);
        AX_RUN_TEST(test_gpu_fill_and_scalar);
        AX_RUN_TEST(test_gpu_elementwise);
        AX_RUN_TEST(test_gpu_gemm);
        AX_RUN_TEST(test_gpu_sum_reduction);
        AX_RUN_TEST(test_gpu_larger_gemm);
        AX_RUN_TEST(test_gpu_conv_gemm);
        AX_RUN_TEST(test_gpu_winograd_f23_parity);
        AX_RUN_TEST(test_gpu_fused_ops);
    } else {
        printf("--- no GPU found, skipping GPU tests ---\n");
    }
#endif

    ax_shutdown();
    AX_TEST_SUMMARY();
}
