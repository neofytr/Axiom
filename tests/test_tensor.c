/* test_tensor.c — tests for tensor creation, storage, and shape ops */

#include "test.h"
#include "axiom/axiom.h"

/* test tensor creation and basic properties */
static void test_tensor_create(void) {
    int64_t shape[] = {3, 4};
    ax_tensor_t *t = ax_tensor_create(shape, 2, AX_FLOAT32);

    AX_TEST_ASSERT(t != NULL, "tensor should be created");
    AX_TEST_ASSERT_EQ(t->ndim, 2, "ndim should be 2");
    AX_TEST_ASSERT_EQ(t->shape[0], 3, "shape[0] should be 3");
    AX_TEST_ASSERT_EQ(t->shape[1], 4, "shape[1] should be 4");
    AX_TEST_ASSERT_EQ(t->dtype, AX_FLOAT32, "dtype should be float32");
    AX_TEST_ASSERT_EQ(ax_tensor_numel(t), 12, "numel should be 12");
    AX_TEST_ASSERT(ax_tensor_is_contiguous(t), "new tensor should be contiguous");

    /* verify strides are c-contiguous */
    AX_TEST_ASSERT_EQ(t->strides[0], 4, "stride[0] should be 4");
    AX_TEST_ASSERT_EQ(t->strides[1], 1, "stride[1] should be 1");

    ax_tensor_destroy(t);
}

/* test zeros, ones, full creation */
static void test_tensor_fill_constructors(void) {
    int64_t shape[] = {2, 3};

    ax_tensor_t *z = ax_tensor_zeros(shape, 2, AX_FLOAT32);
    AX_TEST_ASSERT(z != NULL, "zeros should succeed");
    int64_t idx[] = {0, 0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(z, idx), 0.0f, 1e-7, "zeros should be 0");
    int64_t idx2[] = {1, 2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(z, idx2), 0.0f, 1e-7, "zeros should be 0 everywhere");
    ax_tensor_destroy(z);

    ax_tensor_t *o = ax_tensor_ones(shape, 2, AX_FLOAT32);
    AX_TEST_ASSERT(o != NULL, "ones should succeed");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(o, idx), 1.0f, 1e-7, "ones should be 1");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(o, idx2), 1.0f, 1e-7, "ones should be 1 everywhere");
    ax_tensor_destroy(o);

    ax_tensor_t *f = ax_tensor_full(shape, 2, AX_FLOAT32, 3.14);
    AX_TEST_ASSERT(f != NULL, "full should succeed");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(f, idx), 3.14f, 1e-5, "full should be 3.14");
    ax_tensor_destroy(f);
}

/* test from_array */
static void test_tensor_from_array(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int64_t shape[] = {2, 3};

    ax_tensor_t *t = ax_tensor_from_array(data, shape, 2, AX_FLOAT32);
    AX_TEST_ASSERT(t != NULL, "from_array should succeed");

    int64_t idx00[] = {0, 0};
    int64_t idx01[] = {0, 1};
    int64_t idx12[] = {1, 2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(t, idx00), 1.0f, 1e-7, "[0,0] should be 1");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(t, idx01), 2.0f, 1e-7, "[0,1] should be 2");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(t, idx12), 6.0f, 1e-7, "[1,2] should be 6");

    ax_tensor_destroy(t);
}

/* test arange */
static void test_tensor_arange(void) {
    ax_tensor_t *t = ax_tensor_arange(0, 5, AX_FLOAT32);
    AX_TEST_ASSERT(t != NULL, "arange should succeed");
    AX_TEST_ASSERT_EQ(ax_tensor_numel(t), 5, "should have 5 elements");

    int64_t i0[] = {0}, i1[] = {1}, i4[] = {4};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(t, i0), 0.0f, 1e-7, "[0] should be 0");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(t, i1), 1.0f, 1e-7, "[1] should be 1");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(t, i4), 4.0f, 1e-7, "[4] should be 4");

    ax_tensor_destroy(t);
}

/* test rand */
static void test_tensor_rand(void) {
    int64_t shape[] = {100};
    ax_tensor_t *t = ax_tensor_rand(shape, 1, -1.0f, 1.0f);
    AX_TEST_ASSERT(t != NULL, "rand should succeed");

    /* check all values are in range */
    int in_range = 1;
    for (int64_t i = 0; i < 100; i++) {
        int64_t idx[] = {i};
        float v = ax_tensor_get_f32(t, idx);
        if (v < -1.0f || v >= 1.0f) { in_range = 0; break; }
    }
    AX_TEST_ASSERT(in_range, "all random values should be in [-1, 1)");

    ax_tensor_destroy(t);
}

/* test scalar tensor */
static void test_tensor_scalar(void) {
    ax_tensor_t *t = ax_tensor_scalar(42.0f);
    AX_TEST_ASSERT(t != NULL, "scalar should succeed");
    AX_TEST_ASSERT_EQ(ax_tensor_numel(t), 1, "scalar should have 1 element");

    int64_t idx[] = {0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(t, idx), 42.0f, 1e-7, "value should be 42");
    ax_tensor_destroy(t);
}

/* test storage reference counting */
static void test_storage_refcount(void) {
    int64_t shape[] = {4};
    ax_tensor_t *t = ax_tensor_from_array((float[]){1, 2, 3, 4}, shape, 1, AX_FLOAT32);

    /* create a view — shares storage */
    ax_tensor_t *v = ax_tensor_view(t);
    AX_TEST_ASSERT_EQ(t->storage->refcount, 2, "refcount should be 2 after view");

    /* both should see the same data */
    int64_t i0[] = {0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(v, i0), 1.0f, 1e-7, "view should see same data");

    /* modify through view, original should see it */
    ax_tensor_set_f32(v, i0, 99.0f);
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(t, i0), 99.0f, 1e-7, "original should see view's write");

    ax_tensor_destroy(v);
    AX_TEST_ASSERT_EQ(t->storage->refcount, 1, "refcount should drop to 1");
    ax_tensor_destroy(t);
}

/* test reshape */
static void test_tensor_reshape(void) {
    float data[] = {1, 2, 3, 4, 5, 6};
    int64_t shape[] = {2, 3};
    ax_tensor_t *t = ax_tensor_from_array(data, shape, 2, AX_FLOAT32);

    int64_t new_shape[] = {3, 2};
    ax_tensor_t *r = ax_tensor_reshape(t, new_shape, 2);
    AX_TEST_ASSERT(r != NULL, "reshape should succeed");
    AX_TEST_ASSERT_EQ(r->shape[0], 3, "reshaped shape[0] should be 3");
    AX_TEST_ASSERT_EQ(r->shape[1], 2, "reshaped shape[1] should be 2");
    AX_TEST_ASSERT_EQ(ax_tensor_numel(r), 6, "numel should be preserved");

    /* data should be shared (zero-copy) */
    AX_TEST_ASSERT(r->storage == t->storage, "reshape should share storage");

    /* verify data layout: row-major means [1,2,3,4,5,6] -> [[1,2],[3,4],[5,6]] */
    int64_t idx[] = {0, 0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(r, idx), 1.0f, 1e-7, "[0,0] should be 1");
    int64_t idx2[] = {2, 1};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(r, idx2), 6.0f, 1e-7, "[2,1] should be 6");

    ax_tensor_destroy(r);
    ax_tensor_destroy(t);
}

/* test transpose */
static void test_tensor_transpose(void) {
    float data[] = {1, 2, 3, 4, 5, 6};
    int64_t shape[] = {2, 3};
    ax_tensor_t *t = ax_tensor_from_array(data, shape, 2, AX_FLOAT32);

    ax_tensor_t *tr = ax_tensor_transpose(t, 0, 1);
    AX_TEST_ASSERT(tr != NULL, "transpose should succeed");
    AX_TEST_ASSERT_EQ(tr->shape[0], 3, "transposed shape[0] should be 3");
    AX_TEST_ASSERT_EQ(tr->shape[1], 2, "transposed shape[1] should be 2");

    /* verify zero-copy */
    AX_TEST_ASSERT(tr->storage == t->storage, "transpose should share storage");

    /* original: [[1,2,3],[4,5,6]] -> transposed: [[1,4],[2,5],[3,6]] */
    int64_t idx00[] = {0, 0};
    int64_t idx01[] = {0, 1};
    int64_t idx10[] = {1, 0};
    int64_t idx21[] = {2, 1};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(tr, idx00), 1.0f, 1e-7, "T[0,0] should be 1");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(tr, idx01), 4.0f, 1e-7, "T[0,1] should be 4");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(tr, idx10), 2.0f, 1e-7, "T[1,0] should be 2");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(tr, idx21), 6.0f, 1e-7, "T[2,1] should be 6");

    /* transposed tensor should NOT be contiguous */
    AX_TEST_ASSERT(!ax_tensor_is_contiguous(tr), "transposed tensor should not be contiguous");

    ax_tensor_destroy(tr);
    ax_tensor_destroy(t);
}

/* test squeeze and unsqueeze */
static void test_tensor_squeeze_unsqueeze(void) {
    int64_t shape[] = {1, 3, 1, 4};
    ax_tensor_t *t = ax_tensor_ones(shape, 4, AX_FLOAT32);

    /* squeeze all size-1 dims */
    ax_tensor_t *sq = ax_tensor_squeeze(t, -1);
    AX_TEST_ASSERT(sq != NULL, "squeeze should succeed");
    AX_TEST_ASSERT_EQ(sq->ndim, 2, "squeezed ndim should be 2");
    AX_TEST_ASSERT_EQ(sq->shape[0], 3, "squeezed shape[0] should be 3");
    AX_TEST_ASSERT_EQ(sq->shape[1], 4, "squeezed shape[1] should be 4");

    /* unsqueeze at dim 0 */
    ax_tensor_t *us = ax_tensor_unsqueeze(sq, 0);
    AX_TEST_ASSERT(us != NULL, "unsqueeze should succeed");
    AX_TEST_ASSERT_EQ(us->ndim, 3, "unsqueezed ndim should be 3");
    AX_TEST_ASSERT_EQ(us->shape[0], 1, "unsqueezed shape[0] should be 1");
    AX_TEST_ASSERT_EQ(us->shape[1], 3, "unsqueezed shape[1] should be 3");
    AX_TEST_ASSERT_EQ(us->shape[2], 4, "unsqueezed shape[2] should be 4");

    ax_tensor_destroy(us);
    ax_tensor_destroy(sq);
    ax_tensor_destroy(t);
}

/* test contiguous copy of non-contiguous tensor */
static void test_tensor_contiguous(void) {
    float data[] = {1, 2, 3, 4, 5, 6};
    int64_t shape[] = {2, 3};
    ax_tensor_t *t = ax_tensor_from_array(data, shape, 2, AX_FLOAT32);

    /* transpose makes it non-contiguous */
    ax_tensor_t *tr = ax_tensor_transpose(t, 0, 1);
    AX_TEST_ASSERT(!ax_tensor_is_contiguous(tr), "should be non-contiguous");

    /* make contiguous copy */
    ax_tensor_t *c = ax_tensor_contiguous(tr);
    AX_TEST_ASSERT(c != NULL, "contiguous should succeed");
    AX_TEST_ASSERT(ax_tensor_is_contiguous(c), "result should be contiguous");
    AX_TEST_ASSERT(c->storage != tr->storage, "should have new storage");

    /* verify data is correct: transposed [[1,4],[2,5],[3,6]] */
    int64_t idx00[] = {0, 0};
    int64_t idx01[] = {0, 1};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(c, idx00), 1.0f, 1e-7, "c[0,0] should be 1");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(c, idx01), 4.0f, 1e-7, "c[0,1] should be 4");

    ax_tensor_destroy(c);
    ax_tensor_destroy(tr);
    ax_tensor_destroy(t);
}

/* test dtype utilities */
static void test_dtype_utils(void) {
    AX_TEST_ASSERT_EQ(ax_dtype_size(AX_FLOAT32), 4, "float32 should be 4 bytes");
    AX_TEST_ASSERT_EQ(ax_dtype_size(AX_FLOAT64), 8, "float64 should be 8 bytes");
    AX_TEST_ASSERT_EQ(ax_dtype_size(AX_INT32), 4, "int32 should be 4 bytes");
    AX_TEST_ASSERT_EQ(ax_dtype_size(AX_UINT8), 1, "uint8 should be 1 byte");

    AX_TEST_ASSERT(strcmp(ax_dtype_name(AX_FLOAT32), "float32") == 0, "float32 name");
    AX_TEST_ASSERT(strcmp(ax_dtype_name(AX_BOOL), "bool") == 0, "bool name");
}

/* layout: NCHW <-> NHWC round-trip preserves data, default layout is NCHW. */
static void test_tensor_layout_roundtrip(void) {
    int64_t sh[] = {2, 4, 3, 5};  /* N=2, C=4, H=3, W=5 */
    ax_tensor_t *nchw = ax_tensor_create(sh, 4, AX_FLOAT32);
    AX_TEST_ASSERT(nchw != NULL, "create NCHW tensor");
    AX_TEST_ASSERT(ax_tensor_get_layout(nchw) == AX_LAYOUT_NCHW, "default layout is NCHW");

    /* fill with i for nchw[n, c, h, w] = encode(n, c, h, w) */
    float *d = (float *)nchw->storage->data;
    int64_t N = sh[0], C = sh[1], H = sh[2], W = sh[3];
    for (int64_t n = 0; n < N; n++)
        for (int64_t c = 0; c < C; c++)
            for (int64_t h = 0; h < H; h++)
                for (int64_t w = 0; w < W; w++)
                    d[((n * C + c) * H + h) * W + w] = (float)(((n * C + c) * H + h) * W + w);

    /* convert to NHWC */
    ax_tensor_t *nhwc = ax_tensor_to_nhwc(nchw);
    AX_TEST_ASSERT(nhwc != NULL, "convert to NHWC");
    AX_TEST_ASSERT(ax_tensor_get_layout(nhwc) == AX_LAYOUT_NHWC, "layout flag is NHWC");
    AX_TEST_ASSERT_EQ(nhwc->shape[0], N, "N preserved");
    AX_TEST_ASSERT_EQ(nhwc->shape[1], H, "H is dim 1 in NHWC");
    AX_TEST_ASSERT_EQ(nhwc->shape[2], W, "W is dim 2 in NHWC");
    AX_TEST_ASSERT_EQ(nhwc->shape[3], C, "C is dim 3 in NHWC");

    /* verify data placement: nhwc[n, h, w, c] should equal nchw[n, c, h, w] */
    float *nd = (float *)nhwc->storage->data;
    for (int64_t n = 0; n < N; n++)
        for (int64_t h = 0; h < H; h++)
            for (int64_t w = 0; w < W; w++)
                for (int64_t c = 0; c < C; c++) {
                    float expected = (float)(((n * C + c) * H + h) * W + w);
                    float actual = nd[((n * H + h) * W + w) * C + c];
                    AX_TEST_ASSERT_NEAR(actual, expected, 0.0f, "NHWC data matches NCHW source");
                }

    /* round-trip back */
    ax_tensor_t *nchw2 = ax_tensor_to_nchw(nhwc);
    AX_TEST_ASSERT(nchw2 != NULL, "convert back to NCHW");
    AX_TEST_ASSERT(ax_tensor_get_layout(nchw2) == AX_LAYOUT_NCHW, "layout flag is NCHW");
    AX_TEST_ASSERT_EQ(nchw2->shape[1], C, "C is dim 1 again");

    float *d2 = (float *)nchw2->storage->data;
    for (int64_t i = 0; i < N * C * H * W; i++)
        AX_TEST_ASSERT_NEAR(d2[i], d[i], 0.0f, "round-trip exact");

    /* idempotent: NHWC -> NHWC is no-op */
    ax_tensor_t *nhwc2 = ax_tensor_to_nhwc(nhwc);
    AX_TEST_ASSERT(nhwc2 != NULL, "NHWC -> NHWC");
    AX_TEST_ASSERT(ax_tensor_get_layout(nhwc2) == AX_LAYOUT_NHWC, "layout still NHWC");

    ax_tensor_destroy(nhwc2);
    ax_tensor_destroy(nchw2);
    ax_tensor_destroy(nhwc);
    ax_tensor_destroy(nchw);
}

int main(void) {
    ax_init();

    printf("=== tensor tests ===\n");
    AX_RUN_TEST(test_tensor_create);
    AX_RUN_TEST(test_tensor_fill_constructors);
    AX_RUN_TEST(test_tensor_from_array);
    AX_RUN_TEST(test_tensor_arange);
    AX_RUN_TEST(test_tensor_rand);
    AX_RUN_TEST(test_tensor_scalar);
    AX_RUN_TEST(test_storage_refcount);
    AX_RUN_TEST(test_tensor_reshape);
    AX_RUN_TEST(test_tensor_transpose);
    AX_RUN_TEST(test_tensor_squeeze_unsqueeze);
    AX_RUN_TEST(test_tensor_contiguous);
    AX_RUN_TEST(test_tensor_layout_roundtrip);
    AX_RUN_TEST(test_dtype_utils);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
