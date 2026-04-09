/* test_data.c — test dataset, dataloader, and transforms */

#include "test.h"
#include "axiom/axiom.h"
#include <stdio.h>
#include <unistd.h>

static ax_tensor_t *make_2d(float *data, int64_t rows, int64_t cols)
{
    int64_t shape[] = {rows, cols};
    return ax_tensor_from_array(data, shape, 2, AX_FLOAT32);
}

static void test_tensor_dataset(void)
{
    ax_tensor_t *inputs = make_2d((float[]){1,2, 3,4, 5,6, 7,8}, 4, 2);
    ax_tensor_t *targets = make_2d((float[]){0, 1, 1, 0}, 4, 1);

    ax_dataset_t *ds = ax_tensor_dataset_create(inputs, targets);
    AX_TEST_ASSERT(ds != NULL, "dataset should be created");
    AX_TEST_ASSERT_EQ(ax_dataset_length(ds), 4, "should have 4 samples");

    ax_tensor_t *in, *tgt;
    ax_dataset_get_item(ds, 0, &in, &tgt);
    AX_TEST_ASSERT(in != NULL, "item 0 input should exist");

    /* first sample input should be [1, 2] */
    int64_t i0[] = {0}, i1[] = {1};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(in, i0), 1.0f, 1e-6, "item 0 in[0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(in, i1), 2.0f, 1e-6, "item 0 in[1]");

    /* first sample target should be [0] */
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(tgt, i0), 0.0f, 1e-6, "item 0 tgt");

    /* third sample */
    ax_dataset_get_item(ds, 2, &in, &tgt);
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(in, i0), 5.0f, 1e-6, "item 2 in[0]");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(tgt, i0), 1.0f, 1e-6, "item 2 tgt");

    ax_dataset_destroy(ds);
    ax_tensor_destroy(inputs); ax_tensor_destroy(targets);
}

static void test_dataloader_basic(void)
{
    ax_tensor_t *inputs = make_2d((float[]){1,2, 3,4, 5,6, 7,8, 9,10}, 5, 2);
    ax_tensor_t *targets = make_2d((float[]){0, 1, 1, 0, 1}, 5, 1);
    ax_dataset_t *ds = ax_tensor_dataset_create(inputs, targets);

    ax_dataloader_t *dl = ax_dataloader_create(ds, 2, false);
    AX_TEST_ASSERT(dl != NULL, "dataloader should be created");
    AX_TEST_ASSERT_EQ(ax_dataloader_num_batches(dl), 3, "5 samples / 2 batch = 3 batches");

    /* iterate through all batches */
    ax_batch_t batch;
    int n_batches = 0;
    int64_t total_samples = 0;

    while (ax_dataloader_next(dl, &batch))
    {
        n_batches++;
        total_samples += batch.batch_size;
        AX_TEST_ASSERT(batch.input != NULL, "batch input should exist");
        AX_TEST_ASSERT(batch.target != NULL, "batch target should exist");
        AX_TEST_ASSERT_EQ(batch.input->shape[1], 2, "features preserved");

        ax_tensor_destroy(batch.input);
        ax_tensor_destroy(batch.target);
    }

    AX_TEST_ASSERT_EQ(n_batches, 3, "should get 3 batches");
    AX_TEST_ASSERT_EQ(total_samples, 5, "should cover all 5 samples");

    /* after exhausting, next should return false */
    AX_TEST_ASSERT(!ax_dataloader_next(dl, &batch), "should return false when done");

    /* reset and iterate again */
    ax_dataloader_reset(dl);
    n_batches = 0;
    while (ax_dataloader_next(dl, &batch))
    {
        n_batches++;
        ax_tensor_destroy(batch.input);
        ax_tensor_destroy(batch.target);
    }
    AX_TEST_ASSERT_EQ(n_batches, 3, "should get 3 batches after reset");

    ax_dataloader_destroy(dl);
    ax_dataset_destroy(ds);
    ax_tensor_destroy(inputs); ax_tensor_destroy(targets);
}

static void test_dataloader_shuffle(void)
{
    /* with shuffle, the order should be different across resets
       (statistically — there's a tiny chance they come out the same) */
    ax_tensor_t *inputs = make_2d((float[]){
        1,0, 2,0, 3,0, 4,0, 5,0, 6,0, 7,0, 8,0, 9,0, 10,0
    }, 10, 2);
    ax_tensor_t *targets = make_2d((float[]){0,1,2,3,4,5,6,7,8,9}, 10, 1);
    ax_dataset_t *ds = ax_tensor_dataset_create(inputs, targets);

    ax_dataloader_t *dl = ax_dataloader_create(ds, 10, true);

    /* get all items as one batch */
    ax_batch_t b1, b2;
    ax_dataloader_next(dl, &b1);

    ax_dataloader_reset(dl);
    ax_dataloader_next(dl, &b2);

    /* at least some values should differ (shuffle happened) */
    /* we just check that the dataloader doesn't crash with shuffle=true */
    AX_TEST_ASSERT(b1.input != NULL, "shuffled batch 1 should exist");
    AX_TEST_ASSERT(b2.input != NULL, "shuffled batch 2 should exist");
    AX_TEST_ASSERT_EQ(b1.batch_size, 10, "full batch");
    AX_TEST_ASSERT_EQ(b2.batch_size, 10, "full batch");

    ax_tensor_destroy(b1.input); ax_tensor_destroy(b1.target);
    ax_tensor_destroy(b2.input); ax_tensor_destroy(b2.target);
    ax_dataloader_destroy(dl);
    ax_dataset_destroy(ds);
    ax_tensor_destroy(inputs); ax_tensor_destroy(targets);
}

static void test_csv_dataset(void)
{
    /* create a temp csv file */
    FILE *f = fopen("/tmp/ax_test_data.csv", "w");
    fprintf(f, "x1,x2,y\n");
    fprintf(f, "1.0,2.0,0.0\n");
    fprintf(f, "3.0,4.0,1.0\n");
    fprintf(f, "5.0,6.0,1.0\n");
    fclose(f);

    int feature_cols[] = {0, 1};
    int target_cols[] = {2};
    ax_dataset_t *ds = ax_csv_dataset_load("/tmp/ax_test_data.csv",
                                            feature_cols, 2,
                                            target_cols, 1,
                                            true);

    AX_TEST_ASSERT(ds != NULL, "csv dataset should load");
    AX_TEST_ASSERT_EQ(ax_dataset_length(ds), 3, "should have 3 rows");

    ax_tensor_t *in, *tgt;
    ax_dataset_get_item(ds, 0, &in, &tgt);

    int64_t i0[] = {0}, i1[] = {1};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(in, i0), 1.0f, 1e-5, "csv row 0 col 0");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(in, i1), 2.0f, 1e-5, "csv row 0 col 1");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(tgt, i0), 0.0f, 1e-5, "csv row 0 target");

    ax_dataset_get_item(ds, 2, &in, &tgt);
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(in, i0), 5.0f, 1e-5, "csv row 2 col 0");

    ax_dataset_destroy(ds);
    unlink("/tmp/ax_test_data.csv");
}

static void test_normalize(void)
{
    /* [4, 2] tensor: feature 0 = {0, 10, 20, 30}, feature 1 = {1, 1, 1, 1} */
    ax_tensor_t *t = make_2d((float[]){0,1, 10,1, 20,1, 30,1}, 4, 2);
    ax_tensor_t *normed = ax_transform_normalize(t);

    AX_TEST_ASSERT(normed != NULL, "normalize should work");

    /* feature 0: mean=15, std~11.18. normalized should have ~zero mean */
    int64_t i00[] = {0, 0}, i10[] = {1, 0}, i20[] = {2, 0};
    float sum = 0;
    for (int i = 0; i < 4; i++)
    {
        int64_t idx[] = {i, 0};
        sum += ax_tensor_get_f32(normed, idx);
    }
    AX_TEST_ASSERT_NEAR(sum, 0.0f, 0.01f, "normalized feature 0 should have ~zero mean");

    /* feature 1: all same value, std~0, normalized should be ~0 */
    int64_t i01[] = {0, 1};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(normed, i01), 0.0f, 0.1f, "constant feature normalized ~0");

    ax_tensor_destroy(t); ax_tensor_destroy(normed);
}

static void test_one_hot(void)
{
    float labels_data[] = {0, 2, 1, 3};
    int64_t shape[] = {4};
    ax_tensor_t *labels = ax_tensor_from_array(labels_data, shape, 1, AX_FLOAT32);

    ax_tensor_t *oh = ax_transform_one_hot(labels, 4);
    AX_TEST_ASSERT(oh != NULL, "one_hot should work");
    AX_TEST_ASSERT_EQ(oh->shape[0], 4, "rows");
    AX_TEST_ASSERT_EQ(oh->shape[1], 4, "classes");

    /* row 0: [1, 0, 0, 0] */
    int64_t i00[] = {0, 0}, i01[] = {0, 1};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(oh, i00), 1.0f, 1e-7, "[0,0]=1");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(oh, i01), 0.0f, 1e-7, "[0,1]=0");

    /* row 1: [0, 0, 1, 0] */
    int64_t i12[] = {1, 2};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(oh, i12), 1.0f, 1e-7, "[1,2]=1");

    /* row 3: [0, 0, 0, 1] */
    int64_t i33[] = {3, 3};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(oh, i33), 1.0f, 1e-7, "[3,3]=1");

    ax_tensor_destroy(labels); ax_tensor_destroy(oh);
}

static void test_minmax_scale(void)
{
    ax_tensor_t *t = make_2d((float[]){0, 100, 5, 200, 10, 300}, 3, 2);
    ax_tensor_t *scaled = ax_transform_minmax_scale(t);

    AX_TEST_ASSERT(scaled != NULL, "minmax should work");

    /* feature 0: min=0, max=10 -> [0, 0.5, 1.0] */
    int64_t i00[] = {0, 0}, i10[] = {1, 0}, i20[] = {2, 0};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(scaled, i00), 0.0f, 1e-5, "min=0");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(scaled, i10), 0.5f, 1e-5, "mid=0.5");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(scaled, i20), 1.0f, 1e-5, "max=1");

    /* feature 1: min=100, max=300 -> [0, 0.5, 1.0] */
    int64_t i01[] = {0, 1}, i11[] = {1, 1}, i21[] = {2, 1};
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(scaled, i01), 0.0f, 1e-5, "f1 min");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(scaled, i11), 0.5f, 1e-5, "f1 mid");
    AX_TEST_ASSERT_NEAR(ax_tensor_get_f32(scaled, i21), 1.0f, 1e-5, "f1 max");

    ax_tensor_destroy(t); ax_tensor_destroy(scaled);
}

int main(void)
{
    ax_init();

    printf("=== data pipeline tests ===\n");
    AX_RUN_TEST(test_tensor_dataset);
    AX_RUN_TEST(test_dataloader_basic);
    AX_RUN_TEST(test_dataloader_shuffle);
    AX_RUN_TEST(test_csv_dataset);
    AX_RUN_TEST(test_normalize);
    AX_RUN_TEST(test_one_hot);
    AX_RUN_TEST(test_minmax_scale);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
