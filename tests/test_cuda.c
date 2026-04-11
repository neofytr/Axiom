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
    } else {
        printf("--- no GPU found, skipping GPU tests ---\n");
    }
#endif

    ax_shutdown();
    AX_TEST_SUMMARY();
}
