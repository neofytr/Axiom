/* ad-hoc verification of inplace activations */
#include "axiom/axiom.h"
#include "axiom/ops.h"
#include "axiom/autograd.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdatomic.h>

static int failures = 0;

#define EXPECT(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); failures++; } \
} while (0)

static bool close_enough(float a, float b) {
    return fabsf(a - b) < 1e-5f;
}

static void test_relu_forward(void)
{
    printf("test: relu inplace forward equality\n");
    int64_t shape[] = {4, 8};
    float src[32];
    for (int i = 0; i < 32; i++) src[i] = (float)(i - 16) * 0.3f;

    ax_tensor_t *a = ax_tensor_from_array(src, shape, 2, AX_FLOAT32);
    ax_tensor_t *b = ax_tensor_from_array(src, shape, 2, AX_FLOAT32);

    ax_tensor_t *ref = ax_relu(a);
    void *b_data_before = b->storage->data;
    ax_tensor_t *got = ax_relu_inplace(b);

    EXPECT(got == b, "relu_inplace returns same tensor pointer");
    EXPECT(got->storage->data == b_data_before, "relu_inplace keeps same storage");

    float *rd = (float *)ref->storage->data;
    float *gd = (float *)got->storage->data;
    bool match = true;
    for (int i = 0; i < 32; i++)
        if (!close_enough(rd[i], gd[i])) { match = false; break; }
    EXPECT(match, "relu_inplace matches ax_relu element-by-element");

    ax_tensor_destroy(a);
    ax_tensor_destroy(ref);
    ax_tensor_destroy(got);
}

static void test_sigmoid_tanh_forward(void)
{
    printf("test: sigmoid/tanh inplace forward equality\n");
    int64_t shape[] = {16};
    float src[16];
    for (int i = 0; i < 16; i++) src[i] = (float)(i - 8) * 0.4f;

    ax_tensor_t *a = ax_tensor_from_array(src, shape, 1, AX_FLOAT32);
    ax_tensor_t *b = ax_tensor_from_array(src, shape, 1, AX_FLOAT32);
    ax_tensor_t *ref = ax_sigmoid(a);
    ax_tensor_t *got = ax_sigmoid_inplace(b);
    EXPECT(got == b, "sigmoid_inplace returns same pointer");
    float *rd = (float *)ref->storage->data;
    float *gd = (float *)got->storage->data;
    bool match = true;
    for (int i = 0; i < 16; i++) if (!close_enough(rd[i], gd[i])) match = false;
    EXPECT(match, "sigmoid_inplace matches ax_sigmoid");
    ax_tensor_destroy(a); ax_tensor_destroy(ref); ax_tensor_destroy(got);

    ax_tensor_t *a2 = ax_tensor_from_array(src, shape, 1, AX_FLOAT32);
    ax_tensor_t *b2 = ax_tensor_from_array(src, shape, 1, AX_FLOAT32);
    ax_tensor_t *ref2 = ax_tanh_op(a2);
    ax_tensor_t *got2 = ax_tanh_inplace(b2);
    EXPECT(got2 == b2, "tanh_inplace returns same pointer");
    float *rd2 = (float *)ref2->storage->data;
    float *gd2 = (float *)got2->storage->data;
    match = true;
    for (int i = 0; i < 16; i++) if (!close_enough(rd2[i], gd2[i])) match = false;
    EXPECT(match, "tanh_inplace matches ax_tanh_op");
    ax_tensor_destroy(a2); ax_tensor_destroy(ref2); ax_tensor_destroy(got2);
}

static void test_relu_backward(void)
{
    printf("test: relu inplace backward equals non-inplace\n");
    int64_t shape[] = {12};
    float src[12] = {-3,-2,-1,0,0.5f,1,-0.1f,2,3,-4,0.7f,-0.3f};

    /* non-inplace */
    ax_tensor_t *a = ax_tensor_from_array(src, shape, 1, AX_FLOAT32);
    a->requires_grad = true;
    ax_tensor_t *y = ax_relu(a);
    ax_tensor_t *loss = ax_sum(y, -1);
    ax_backward(loss);
    float a_grad_ref[12];
    memcpy(a_grad_ref, a->grad->storage->data, sizeof(float)*12);

    /* inplace */
    ax_tensor_t *b = ax_tensor_from_array(src, shape, 1, AX_FLOAT32);
    b->requires_grad = true;
    ax_tensor_t *y2 = ax_relu_inplace(b);
    EXPECT(y2 == b, "backward: relu_inplace returns same pointer");
    ax_tensor_t *loss2 = ax_sum(y2, -1);
    ax_backward(loss2);
    float *gi = (float *)b->grad->storage->data;
    bool match = true;
    for (int i = 0; i < 12; i++) if (!close_enough(a_grad_ref[i], gi[i])) match = false;
    EXPECT(match, "relu inplace backward == non-inplace backward");

    ax_tensor_destroy(a); ax_tensor_destroy(b);
}

static void test_sigmoid_backward(void)
{
    printf("test: sigmoid inplace backward equals non-inplace\n");
    int64_t shape[] = {8};
    float src[8] = {-2,-1,0,1,2,-0.5f,0.5f,3};

    ax_tensor_t *a = ax_tensor_from_array(src, shape, 1, AX_FLOAT32);
    a->requires_grad = true;
    ax_tensor_t *y = ax_sigmoid(a);
    ax_tensor_t *loss = ax_sum(y, -1);
    ax_backward(loss);
    float ref[8];
    memcpy(ref, a->grad->storage->data, sizeof(float)*8);

    ax_tensor_t *b = ax_tensor_from_array(src, shape, 1, AX_FLOAT32);
    b->requires_grad = true;
    ax_tensor_t *y2 = ax_sigmoid_inplace(b);
    ax_tensor_t *loss2 = ax_sum(y2, -1);
    ax_backward(loss2);
    float *gi = (float *)b->grad->storage->data;
    bool match = true;
    for (int i = 0; i < 8; i++) if (!close_enough(ref[i], gi[i])) match = false;
    EXPECT(match, "sigmoid inplace backward == non-inplace");

    ax_tensor_destroy(a); ax_tensor_destroy(b);
}

static void test_tanh_backward(void)
{
    printf("test: tanh inplace backward equals non-inplace\n");
    int64_t shape[] = {8};
    float src[8] = {-2,-1,0,1,2,-0.5f,0.5f,3};

    ax_tensor_t *a = ax_tensor_from_array(src, shape, 1, AX_FLOAT32);
    a->requires_grad = true;
    ax_tensor_t *y = ax_tanh_op(a);
    ax_tensor_t *loss = ax_sum(y, -1);
    ax_backward(loss);
    float ref[8];
    memcpy(ref, a->grad->storage->data, sizeof(float)*8);

    ax_tensor_t *b = ax_tensor_from_array(src, shape, 1, AX_FLOAT32);
    b->requires_grad = true;
    ax_tensor_t *y2 = ax_tanh_inplace(b);
    ax_tensor_t *loss2 = ax_sum(y2, -1);
    ax_backward(loss2);
    float *gi = (float *)b->grad->storage->data;
    bool match = true;
    for (int i = 0; i < 8; i++) if (!close_enough(ref[i], gi[i])) match = false;
    EXPECT(match, "tanh inplace backward == non-inplace");

    ax_tensor_destroy(a); ax_tensor_destroy(b);
}

static void test_refcount_fallback(void)
{
    printf("test: inplace falls back when refcount > 1\n");
    int64_t shape[] = {4};
    float src[4] = {-1, 0.5f, -2, 3};

    ax_tensor_t *a = ax_tensor_from_array(src, shape, 1, AX_FLOAT32);
    ax_storage_retain(a->storage);  /* bump refcount to 2 */

    void *orig_data = a->storage->data;
    float orig[4];
    memcpy(orig, a->storage->data, sizeof(float)*4);

    ax_tensor_t *out = ax_relu_inplace(a);

    EXPECT(out != a, "fallback: refcount>1 must allocate new tensor");
    /* original a must be UNTOUCHED */
    float *ad = (float *)a->storage->data;
    bool untouched = true;
    for (int i = 0; i < 4; i++) if (ad[i] != orig[i]) untouched = false;
    EXPECT(untouched, "fallback: input storage unchanged when shared");
    /* and new out should hold relu values */
    float *od = (float *)out->storage->data;
    EXPECT(od[0] == 0.0f && od[1] == 0.5f && od[2] == 0.0f && od[3] == 3.0f,
           "fallback: result correct");

    (void)orig_data;
    ax_storage_release(a->storage);  /* release extra ref */
    ax_tensor_destroy(a);
    ax_tensor_destroy(out);
}

int main(void)
{
    test_relu_forward();
    test_sigmoid_tanh_forward();
    test_relu_backward();
    test_sigmoid_backward();
    test_tanh_backward();
    test_refcount_fallback();

    if (failures == 0) printf("\nALL INPLACE TESTS PASSED\n");
    else printf("\n%d FAILURES\n", failures);
    return failures ? 1 : 0;
}
