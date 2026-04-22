/* axiom/data.h — data loading and preprocessing pipeline.

   provides:
   - dataset interface (get_item, length)
   - tensor dataset (wraps existing tensors)
   - csv loader (parses csv files into tensors)
   - dataloader (batching, shuffling, iteration)
   - basic transforms (normalize, one-hot encode)

   ownership: ax_*_create returns a heap dataset/dataloader; release
   with ax_dataset_destroy / ax_dataloader_destroy. tensor dataset
   wrappers borrow the underlying tensor (caller keeps ownership);
   csv loader allocates its own tensor that the dataset owns.
   batches returned by the dataloader are owned by the loader and
   recycled per iteration — copy them out if you need to keep them.

   error handling: ax_*_create returns NULL on alloc failure or invalid
   shapes. csv loader returns NULL on file-open / parse error;
   ax_err_last_message describes the cause.

   thread-safety: dataloader iteration is not concurrent-safe. for
   parallel loaders, instantiate one dataloader per thread. */

#ifndef AX_DATA_H
#define AX_DATA_H

#include "tensor.h"

/* dataset interface: anything that has items you can index into.
   concrete datasets embed this as first field (same pattern as layers). */

typedef struct ax_dataset ax_dataset_t;

typedef struct
{
    /* get input and target tensors for the given index.
       the returned tensors are owned by the dataset (don't destroy them). */
    void (*get_item)(ax_dataset_t *self, int64_t index,
                     ax_tensor_t **input, ax_tensor_t **target);
    int64_t (*length)(ax_dataset_t *self);
    void (*destroy)(ax_dataset_t *self);
} ax_dataset_ops_t;

struct ax_dataset
{
    ax_dataset_ops_t ops;
};

/* convenience wrappers */
static inline void ax_dataset_get_item(ax_dataset_t *ds, int64_t idx,
                                        ax_tensor_t **input, ax_tensor_t **target)
{
    ds->ops.get_item(ds, idx, input, target);
}

static inline int64_t ax_dataset_length(ax_dataset_t *ds)
{
    return ds->ops.length(ds);
}

static inline void ax_dataset_destroy(ax_dataset_t *ds)
{
    if (ds && ds->ops.destroy) ds->ops.destroy(ds);
}


/* tensor dataset: wraps a pair of tensors (inputs, targets).
   the first dimension is the sample dimension.
   get_item returns a view (slice) of each tensor at the given index. */

ax_dataset_t *ax_tensor_dataset_create(ax_tensor_t *inputs, ax_tensor_t *targets);


/* csv dataset: loads a csv file where each row is a sample.
   you specify which columns are features and which are targets.
   all values are parsed as float32.

   example:
     ax_dataset_t *ds = ax_csv_dataset_load("data.csv",
         (int[]){0, 1, 2}, 3,     // feature columns
         (int[]){3}, 1,            // target columns
         true);                    // has header row */

ax_dataset_t *ax_csv_dataset_load(const char *path,
                                   const int *feature_cols, int n_features,
                                   const int *target_cols, int n_targets,
                                   bool has_header);


/* dataloader: iterates over a dataset in batches.
   supports shuffling (fisher-yates on indices). */

typedef struct
{
    ax_dataset_t *dataset;
    int64_t batch_size;
    bool shuffle;

    /* internal state */
    int64_t *indices;       /* shuffled index array */
    int64_t n_samples;
    int64_t current_pos;    /* where we are in the epoch */
} ax_dataloader_t;

/* a batch: pair of tensors [batch_size, ...] */
typedef struct
{
    ax_tensor_t *input;
    ax_tensor_t *target;
    int64_t batch_size;     /* actual size (last batch may be smaller) */
} ax_batch_t;

/* create a dataloader for the given dataset */
ax_dataloader_t *ax_dataloader_create(ax_dataset_t *dataset,
                                       int64_t batch_size,
                                       bool shuffle);

/* reset to the start of the dataset (and reshuffle if shuffle=true) */
void ax_dataloader_reset(ax_dataloader_t *dl);

/* get next batch. returns false when epoch is done.
   the batch tensors are freshly allocated; caller must destroy them. */
bool ax_dataloader_next(ax_dataloader_t *dl, ax_batch_t *batch);

/* how many batches in one epoch */
int64_t ax_dataloader_num_batches(ax_dataloader_t *dl);

/* free the dataloader (does NOT free the dataset) */
void ax_dataloader_destroy(ax_dataloader_t *dl);


/* transforms */

/* normalize tensor to zero mean, unit variance along axis 0 (per-feature).
   returns new tensor. useful for preprocessing inputs. */
ax_tensor_t *ax_transform_normalize(ax_tensor_t *t);

/* normalize with known mean and std (for applying training stats to test data) */
ax_tensor_t *ax_transform_normalize_with(ax_tensor_t *t, ax_tensor_t *mean, ax_tensor_t *std);

/* one-hot encode an integer tensor into a float tensor.
   input: [n] of int values in [0, num_classes)
   output: [n, num_classes] of float 0/1 */
ax_tensor_t *ax_transform_one_hot(ax_tensor_t *labels, int num_classes);

/* min-max scaling to [0, 1] range along axis 0 */
ax_tensor_t *ax_transform_minmax_scale(ax_tensor_t *t);

#endif /* AX_DATA_H */
