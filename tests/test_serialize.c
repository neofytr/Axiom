/* test_serialize.c — test save/load roundtrip for tensors and models */

#include "test.h"
#include "axiom/axiom.h"
#include <unistd.h>

static void test_tensor_save_load(void)
{
    float data[] = {1, 2, 3, 4, 5, 6};
    int64_t shape[] = {2, 3};
    ax_tensor_t *t = ax_tensor_from_array(data, shape, 2, AX_FLOAT32);

    ax_status_t s = ax_tensor_save(t, "/tmp/ax_test_tensor.axt");
    AX_TEST_ASSERT_EQ(s, AX_OK, "tensor save should succeed");

    ax_tensor_t *loaded = ax_tensor_load("/tmp/ax_test_tensor.axt");
    AX_TEST_ASSERT(loaded != NULL, "tensor load should succeed");
    AX_TEST_ASSERT_EQ(loaded->ndim, 2, "ndim preserved");
    AX_TEST_ASSERT_EQ(loaded->shape[0], 2, "shape[0] preserved");
    AX_TEST_ASSERT_EQ(loaded->shape[1], 3, "shape[1] preserved");

    /* check values */
    int64_t i00[] = {0, 0}, i12[] = {1, 2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(loaded, i00), 1.0f, 1e-7, "value [0,0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(loaded, i12), 6.0f, 1e-7, "value [1,2]");

    ax_tensor_destroy(t);
    ax_tensor_destroy(loaded);
    unlink("/tmp/ax_test_tensor.axt");
}

static void test_model_save_load(void)
{
    /* build a small model */
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(2, 4, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(4, 1, true));

    ax_model_t *model = ax_model_create(net);

    /* get a prediction before saving */
    float in_data[] = {1, 2};
    int64_t in_shape[] = {1, 2};
    ax_tensor_t *input = ax_tensor_from_array(in_data, in_shape, 2, AX_FLOAT32);

    ax_tensor_t *pred_before = ax_model_predict(model, input);
    int64_t i0[] = {0, 0};
    float val_before = ax_tensor_get_f32(pred_before, i0);

    /* save */
    ax_status_t s = ax_model_save(model, "/tmp/ax_test_model.axm");
    AX_TEST_ASSERT_EQ(s, AX_OK, "model save should succeed");

    /* load into a new model */
    ax_model_t *loaded = ax_model_load("/tmp/ax_test_model.axm");
    AX_TEST_ASSERT(loaded != NULL, "model load should succeed");

    /* predict with loaded model — should give same result */
    ax_tensor_t *pred_after = ax_model_predict(loaded, input);
    float val_after = ax_tensor_get_f32(pred_after, i0);

    AX_TEST_ASSERT_NEAR(val_before, val_after, 1e-5, "loaded model should predict same as original");

    /* check param count matches */
    AX_TEST_ASSERT_EQ(ax_layer_param_count(model->net),
                       ax_layer_param_count(loaded->net),
                       "param count should match");

    ax_tensor_destroy(input);
    ax_model_destroy(model);
    ax_model_destroy(loaded);
    unlink("/tmp/ax_test_model.axm");
}

static void test_load_nonexistent(void)
{
    ax_tensor_t *t = ax_tensor_load("/tmp/nonexistent_file_12345.axt");
    AX_TEST_ASSERT(t == NULL, "loading nonexistent file should return null");

    ax_model_t *m = ax_model_load("/tmp/nonexistent_file_12345.axm");
    AX_TEST_ASSERT(m == NULL, "loading nonexistent model should return null");
}

int main(void)
{
    ax_init();

    printf("=== serialize tests ===\n");
    AX_RUN_TEST(test_tensor_save_load);
    AX_RUN_TEST(test_model_save_load);
    AX_RUN_TEST(test_load_nonexistent);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
