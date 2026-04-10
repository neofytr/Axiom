/* posix source for strtok_r */
#define _POSIX_C_SOURCE 200809L

/* data.c — dataset, dataloader, and transform implementations.

   the general pattern: dataset provides random access to samples,
   dataloader iterates over them in shuffled batches. transforms
   preprocess the data before feeding it to the model.

   on embedded, you'd typically skip the dataloader and feed tensors
   directly. but for training on desktop this is essential. */

#include "axiom/data.h"
#include "axiom/compute.h"
#include "axiom/error.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <inttypes.h>


/* tensor dataset: wraps two tensors where dim 0 is the sample axis */

typedef struct
{
    ax_dataset_t base;
    ax_tensor_t *inputs;   /* [n_samples, ...] */
    ax_tensor_t *targets;  /* [n_samples, ...] */
    int64_t n_samples;

    /* per-sample views (allocated lazily, cached) */
    ax_tensor_t **input_views;
    ax_tensor_t **target_views;
} ax_tensor_dataset_t;

/* create a view of a single sample (row) from a 2d tensor */
static ax_tensor_t *make_row_view(ax_tensor_t *t, int64_t row)
{
    if (t->ndim == 1)
    {
        /* scalar target per sample: return a 1-element view */
        ax_tensor_t *v = ax_tensor_create(&(int64_t){1}, 1, t->dtype);
        float *src = (float *)t->storage->data;
        float *dst = (float *)v->storage->data;
        dst[0] = src[t->offset + row * t->strides[0]];
        return v;
    }

    /* 2d case: return a view of row */
    int new_ndim = t->ndim - 1;
    int64_t new_shape[AX_MAX_DIMS];
    int64_t new_strides[AX_MAX_DIMS];

    for (int i = 0; i < new_ndim; i++)
    {
        new_shape[i] = t->shape[i + 1];
        new_strides[i] = t->strides[i + 1];
    }

    ax_tensor_t *v = calloc(1, sizeof(ax_tensor_t));
    if (!v) return NULL;

    v->storage = t->storage;
    ax_storage_retain(v->storage);
    v->ndim = new_ndim;
    v->dtype = t->dtype;
    v->offset = t->offset + row * t->strides[0];
    memcpy(v->shape, new_shape, sizeof(int64_t) * new_ndim);
    memcpy(v->strides, new_strides, sizeof(int64_t) * new_ndim);
    return v;
}

static void tensor_ds_get_item(ax_dataset_t *self, int64_t idx,
                                ax_tensor_t **input, ax_tensor_t **target)
{
    ax_tensor_dataset_t *ds = (ax_tensor_dataset_t *)self;

    /* lazy init views */
    if (!ds->input_views)
    {
        ds->input_views = calloc((size_t)ds->n_samples, sizeof(ax_tensor_t *));
        ds->target_views = calloc((size_t)ds->n_samples, sizeof(ax_tensor_t *));
        if (!ds->input_views || !ds->target_views)
        {
            free(ds->input_views); ds->input_views = NULL;
            free(ds->target_views); ds->target_views = NULL;
            *input = NULL;
            *target = NULL;
            return;
        }
    }

    /* bounds check on idx */
    if (idx < 0 || idx >= ds->n_samples)
    {
        ax_err_set(AX_ERR_OUT_OF_BOUNDS,
                   "dataset index %" PRId64 " out of range [0, %" PRId64 ")", idx, ds->n_samples);
        *input = NULL;
        *target = NULL;
        return;
    }

    if (!ds->input_views[idx])
        ds->input_views[idx] = make_row_view(ds->inputs, idx);
    if (!ds->target_views[idx])
        ds->target_views[idx] = make_row_view(ds->targets, idx);

    *input = ds->input_views[idx];
    *target = ds->target_views[idx];
}

static int64_t tensor_ds_length(ax_dataset_t *self)
{
    return ((ax_tensor_dataset_t *)self)->n_samples;
}

static void tensor_ds_destroy(ax_dataset_t *self)
{
    ax_tensor_dataset_t *ds = (ax_tensor_dataset_t *)self;
    if (ds->input_views)
    {
        for (int64_t i = 0; i < ds->n_samples; i++)
        {
            if (ds->input_views[i]) ax_tensor_destroy(ds->input_views[i]);
            if (ds->target_views[i]) ax_tensor_destroy(ds->target_views[i]);
        }
        free(ds->input_views);
        free(ds->target_views);
    }
    /* don't destroy inputs/targets — we don't own them */
    free(ds);
}

ax_dataset_t *ax_tensor_dataset_create(ax_tensor_t *inputs, ax_tensor_t *targets)
{
    if (!inputs || !targets) return NULL;

    ax_tensor_dataset_t *ds = calloc(1, sizeof(ax_tensor_dataset_t));
    if (!ds) return NULL;

    ds->base.ops.get_item = tensor_ds_get_item;
    ds->base.ops.length = tensor_ds_length;
    ds->base.ops.destroy = tensor_ds_destroy;
    ds->inputs = inputs;
    ds->targets = targets;
    ds->n_samples = inputs->shape[0];
    return (ax_dataset_t *)ds;
}


/* csv dataset: loads from a csv file */

typedef struct
{
    ax_dataset_t base;
    ax_tensor_t *all_inputs;   /* [n_rows, n_features] */
    ax_tensor_t *all_targets;  /* [n_rows, n_targets] */
    int64_t n_rows;
    ax_tensor_t **input_views;
    ax_tensor_t **target_views;
} ax_csv_dataset_t;

/* count lines in a file (skipping header if present) */
static int64_t count_csv_lines(const char *path, bool has_header)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    int64_t count = 0;
    char buf[4096];
    while (fgets(buf, sizeof(buf), f)) count++;
    fclose(f);

    return has_header ? count - 1 : count;
}

/* parse a single float from a csv field.
   uses strtof instead of atof for error detection. */
static float parse_float(const char *s)
{
    if (!s || *s == '\0') return 0.0f;
    char *end;
    float val = strtof(s, &end);
    if (end == s) return 0.0f; /* no conversion performed */
    return val;
}

static void csv_ds_get_item(ax_dataset_t *self, int64_t idx,
                             ax_tensor_t **input, ax_tensor_t **target)
{
    ax_csv_dataset_t *ds = (ax_csv_dataset_t *)self;

    if (!ds->input_views)
    {
        ds->input_views = calloc((size_t)ds->n_rows, sizeof(ax_tensor_t *));
        ds->target_views = calloc((size_t)ds->n_rows, sizeof(ax_tensor_t *));
        if (!ds->input_views || !ds->target_views)
        {
            free(ds->input_views); ds->input_views = NULL;
            free(ds->target_views); ds->target_views = NULL;
            *input = NULL;
            *target = NULL;
            return;
        }
    }

    /* bounds check on idx */
    if (idx < 0 || idx >= ds->n_rows)
    {
        ax_err_set(AX_ERR_OUT_OF_BOUNDS,
                   "csv dataset index %" PRId64 " out of range [0, %" PRId64 ")", idx, ds->n_rows);
        *input = NULL;
        *target = NULL;
        return;
    }

    if (!ds->input_views[idx])
        ds->input_views[idx] = make_row_view(ds->all_inputs, idx);
    if (!ds->target_views[idx])
        ds->target_views[idx] = make_row_view(ds->all_targets, idx);

    *input = ds->input_views[idx];
    *target = ds->target_views[idx];
}

static int64_t csv_ds_length(ax_dataset_t *self)
{
    return ((ax_csv_dataset_t *)self)->n_rows;
}

static void csv_ds_destroy(ax_dataset_t *self)
{
    ax_csv_dataset_t *ds = (ax_csv_dataset_t *)self;
    if (ds->input_views)
    {
        for (int64_t i = 0; i < ds->n_rows; i++)
        {
            if (ds->input_views[i]) ax_tensor_destroy(ds->input_views[i]);
            if (ds->target_views[i]) ax_tensor_destroy(ds->target_views[i]);
        }
        free(ds->input_views);
        free(ds->target_views);
    }
    if (ds->all_inputs) ax_tensor_destroy(ds->all_inputs);
    if (ds->all_targets) ax_tensor_destroy(ds->all_targets);
    free(ds);
}

ax_dataset_t *ax_csv_dataset_load(const char *path,
                                   const int *feature_cols, int n_features,
                                   const int *target_cols, int n_targets,
                                   bool has_header)
{
    int64_t n_rows = count_csv_lines(path, has_header);
    if (n_rows <= 0)
    {
        ax_err_set(AX_ERR_INTERNAL, "csv file empty or not found: %s", path);
        return NULL;
    }

    /* allocate storage */
    int64_t in_shape[] = {n_rows, n_features};
    int64_t tgt_shape[] = {n_rows, n_targets};

    ax_tensor_t *inputs = ax_tensor_zeros(in_shape, 2, AX_FLOAT32);
    ax_tensor_t *targets = ax_tensor_zeros(tgt_shape, 2, AX_FLOAT32);
    if (!inputs || !targets)
    {
        if (inputs) ax_tensor_destroy(inputs);
        if (targets) ax_tensor_destroy(targets);
        return NULL;
    }

    float *in_data = (float *)inputs->storage->data;
    float *tgt_data = (float *)targets->storage->data;

    /* parse file */
    FILE *f = fopen(path, "r");
    if (!f) { ax_tensor_destroy(inputs); ax_tensor_destroy(targets); return NULL; }

    #define CSV_LINE_MAX 8192
    #define CSV_FIELDS_MAX 256
    char line[CSV_LINE_MAX];
    if (has_header) fgets(line, sizeof(line), f); /* skip header */

    int64_t row = 0;
    while (fgets(line, sizeof(line), f) && row < n_rows)
    {
        /* check for line truncation: if line fills buffer and has no newline,
           the line was too long and data may be corrupt */
        size_t line_len = strlen(line);
        if (line_len == CSV_LINE_MAX - 1 && line[line_len - 1] != '\n')
        {
            ax_err_set(AX_ERR_INTERNAL,
                       "csv line %" PRId64 " exceeds maximum length (%d bytes)",
                       row + (has_header ? 2 : 1), CSV_LINE_MAX);
            /* skip rest of truncated line */
            int ch;
            while ((ch = fgetc(f)) != '\n' && ch != EOF) {}
            continue;
        }

        /* tokenize by comma (strtok_r for thread safety) */
        char *fields[CSV_FIELDS_MAX];
        int n_fields = 0;
        char *saveptr = NULL;
        char *tok = strtok_r(line, ",\n\r", &saveptr);
        while (tok && n_fields < CSV_FIELDS_MAX)
        {
            fields[n_fields++] = tok;
            tok = strtok_r(NULL, ",\n\r", &saveptr);
        }

        /* validate column indices don't exceed parsed fields */
        /* fill feature columns */
        for (int c = 0; c < n_features; c++)
        {
            if (feature_cols[c] >= 0 && feature_cols[c] < n_fields)
                in_data[row * n_features + c] = parse_float(fields[feature_cols[c]]);
        }

        /* fill target columns */
        for (int c = 0; c < n_targets; c++)
        {
            if (target_cols[c] >= 0 && target_cols[c] < n_fields)
                tgt_data[row * n_targets + c] = parse_float(fields[target_cols[c]]);
        }

        row++;
    }
    #undef CSV_LINE_MAX
    #undef CSV_FIELDS_MAX
    fclose(f);

    /* build dataset */
    ax_csv_dataset_t *ds = calloc(1, sizeof(ax_csv_dataset_t));
    if (!ds) { ax_tensor_destroy(inputs); ax_tensor_destroy(targets); return NULL; }

    ds->base.ops.get_item = csv_ds_get_item;
    ds->base.ops.length = csv_ds_length;
    ds->base.ops.destroy = csv_ds_destroy;
    ds->all_inputs = inputs;
    ds->all_targets = targets;
    ds->n_rows = row;
    return (ax_dataset_t *)ds;
}


/* dataloader */

static void shuffle_indices(int64_t *arr, int64_t n)
{
    /* fisher-yates shuffle */
    for (int64_t i = n - 1; i > 0; i--)
    {
        int64_t j = rand() % (i + 1);
        int64_t tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

ax_dataloader_t *ax_dataloader_create(ax_dataset_t *dataset,
                                       int64_t batch_size,
                                       bool shuffle)
{
    if (!dataset) return NULL;

    ax_dataloader_t *dl = calloc(1, sizeof(ax_dataloader_t));
    if (!dl) return NULL;

    dl->dataset = dataset;
    dl->batch_size = batch_size;
    dl->shuffle = shuffle;
    dl->n_samples = ax_dataset_length(dataset);
    dl->current_pos = 0;

    /* allocate and fill index array — check for overflow */
    if (dl->n_samples <= 0 || (size_t)dl->n_samples > SIZE_MAX / sizeof(int64_t))
    {
        free(dl);
        return NULL;
    }
    dl->indices = malloc((size_t)dl->n_samples * sizeof(int64_t));
    if (!dl->indices) { free(dl); return NULL; }

    for (int64_t i = 0; i < dl->n_samples; i++)
        dl->indices[i] = i;

    if (shuffle)
    {
        srand((unsigned)time(NULL));
        shuffle_indices(dl->indices, dl->n_samples);
    }

    return dl;
}

void ax_dataloader_reset(ax_dataloader_t *dl)
{
    if (!dl) return;
    dl->current_pos = 0;
    if (dl->shuffle)
        shuffle_indices(dl->indices, dl->n_samples);
}

bool ax_dataloader_next(ax_dataloader_t *dl, ax_batch_t *batch)
{
    if (!dl || !batch) return false;
    if (dl->current_pos >= dl->n_samples) return false;

    int64_t remaining = dl->n_samples - dl->current_pos;
    int64_t bs = remaining < dl->batch_size ? remaining : dl->batch_size;

    /* get first item to know shapes */
    ax_tensor_t *first_in, *first_tgt;
    ax_dataset_get_item(dl->dataset, dl->indices[dl->current_pos],
                        &first_in, &first_tgt);
    if (!first_in || !first_tgt) return false;

    /* allocate batch tensors: [bs, ...item_shape] */
    if (first_in->ndim + 1 > AX_MAX_DIMS || first_tgt->ndim + 1 > AX_MAX_DIMS)
        return false;

    int64_t in_shape[AX_MAX_DIMS], tgt_shape[AX_MAX_DIMS];
    in_shape[0] = bs;
    tgt_shape[0] = bs;

    for (int d = 0; d < first_in->ndim; d++)
        in_shape[d + 1] = first_in->shape[d];
    for (int d = 0; d < first_tgt->ndim; d++)
        tgt_shape[d + 1] = first_tgt->shape[d];

    batch->input = ax_tensor_zeros(in_shape, first_in->ndim + 1, AX_FLOAT32);
    batch->target = ax_tensor_zeros(tgt_shape, first_tgt->ndim + 1, AX_FLOAT32);
    batch->batch_size = bs;

    if (!batch->input || !batch->target) return false;

    /* fill batch by copying each sample */
    float *in_data = (float *)batch->input->storage->data;
    float *tgt_data = (float *)batch->target->storage->data;

    int64_t in_sample_size = ax_tensor_numel(first_in);
    int64_t tgt_sample_size = ax_tensor_numel(first_tgt);

    for (int64_t i = 0; i < bs; i++)
    {
        int64_t idx = dl->indices[dl->current_pos + i];
        ax_tensor_t *item_in, *item_tgt;
        ax_dataset_get_item(dl->dataset, idx, &item_in, &item_tgt);

        /* copy sample data into batch position */
        float *src_in = (float *)item_in->storage->data;
        float *src_tgt = (float *)item_tgt->storage->data;

        memcpy(in_data + i * in_sample_size,
               src_in + item_in->offset,
               in_sample_size * sizeof(float));

        memcpy(tgt_data + i * tgt_sample_size,
               src_tgt + item_tgt->offset,
               tgt_sample_size * sizeof(float));
    }

    dl->current_pos += bs;
    return true;
}

int64_t ax_dataloader_num_batches(ax_dataloader_t *dl)
{
    if (!dl || dl->batch_size == 0) return 0;
    return (dl->n_samples + dl->batch_size - 1) / dl->batch_size;
}

void ax_dataloader_destroy(ax_dataloader_t *dl)
{
    if (!dl) return;
    free(dl->indices);
    free(dl);
}


/* transforms */

ax_tensor_t *ax_transform_normalize(ax_tensor_t *t)
{
    if (!t || t->ndim < 1) return NULL;

    /* compute mean and std along axis 0 */
    int64_t n = t->shape[0];
    int64_t features = (t->ndim == 1) ? 1 : t->shape[1];
    int64_t n_features = features;

    ax_tensor_t *out = ax_tensor_zeros(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    float *td = (float *)t->storage->data;
    float *od = (float *)out->storage->data;

    for (int64_t f = 0; f < n_features; f++)
    {
        /* mean */
        float sum = 0;
        for (int64_t i = 0; i < n; i++)
        {
            int64_t idx = (t->ndim == 1) ? i : i * t->strides[0] + f * t->strides[1];
            sum += td[t->offset + idx];
        }
        float mean = sum / (float)n;

        /* std */
        float var_sum = 0;
        for (int64_t i = 0; i < n; i++)
        {
            int64_t idx = (t->ndim == 1) ? i : i * t->strides[0] + f * t->strides[1];
            float d = td[t->offset + idx] - mean;
            var_sum += d * d;
        }
        float std = sqrtf(var_sum / (float)n + 1e-7f);

        /* normalize */
        for (int64_t i = 0; i < n; i++)
        {
            int64_t in_idx = (t->ndim == 1) ? i : i * t->strides[0] + f * t->strides[1];
            int64_t out_idx = (out->ndim == 1) ? i : i * out->strides[0] + f * out->strides[1];
            od[out->offset + out_idx] = (td[t->offset + in_idx] - mean) / std;
        }
    }
    return out;
}

ax_tensor_t *ax_transform_normalize_with(ax_tensor_t *t, ax_tensor_t *mean, ax_tensor_t *std)
{
    if (!t || !mean || !std) return NULL;

    ax_tensor_t *out = ax_tensor_zeros(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    float *td = (float *)t->storage->data;
    float *od = (float *)out->storage->data;
    float *md = (float *)mean->storage->data;
    float *sd = (float *)std->storage->data;

    int64_t n = t->shape[0];
    int64_t features = (t->ndim == 1) ? 1 : t->shape[1];

    for (int64_t i = 0; i < n; i++)
    {
        for (int64_t f = 0; f < features; f++)
        {
            int64_t idx = (t->ndim == 1) ? i : i * t->strides[0] + f * t->strides[1];
            int64_t oidx = (out->ndim == 1) ? i : i * out->strides[0] + f * out->strides[1];
            od[out->offset + oidx] = (td[t->offset + idx] - md[f]) / (sd[f] + 1e-7f);
        }
    }
    return out;
}

ax_tensor_t *ax_transform_one_hot(ax_tensor_t *labels, int num_classes)
{
    if (!labels || labels->ndim != 1) return NULL;

    int64_t n = labels->shape[0];
    int64_t shape[] = {n, num_classes};
    ax_tensor_t *out = ax_tensor_zeros(shape, 2, AX_FLOAT32);
    if (!out) return NULL;

    float *od = (float *)out->storage->data;

    /* labels can be float32 (we'll cast to int) or int32 */
    for (int64_t i = 0; i < n; i++)
    {
        int cls;
        if (labels->dtype == AX_FLOAT32)
        {
            float *ld = (float *)labels->storage->data;
            cls = (int)ld[labels->offset + i];
        }
        else if (labels->dtype == AX_INT32)
        {
            int32_t *ld = (int32_t *)labels->storage->data;
            cls = (int)ld[labels->offset + i];
        }
        else
        {
            ax_tensor_destroy(out);
            return NULL;
        }

        if (cls >= 0 && cls < num_classes)
            od[i * num_classes + cls] = 1.0f;
    }
    return out;
}

ax_tensor_t *ax_transform_minmax_scale(ax_tensor_t *t)
{
    if (!t || t->ndim < 1) return NULL;

    int64_t n = t->shape[0];
    int64_t features = (t->ndim == 1) ? 1 : t->shape[1];

    ax_tensor_t *out = ax_tensor_zeros(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    float *td = (float *)t->storage->data;
    float *od = (float *)out->storage->data;

    for (int64_t f = 0; f < features; f++)
    {
        float mn = FLT_MAX, mx = -FLT_MAX;
        for (int64_t i = 0; i < n; i++)
        {
            int64_t idx = (t->ndim == 1) ? i : i * t->strides[0] + f * t->strides[1];
            float v = td[t->offset + idx];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }

        float range = mx - mn;
        if (range < 1e-7f) range = 1.0f; /* avoid div by zero for constant features */

        for (int64_t i = 0; i < n; i++)
        {
            int64_t in_idx = (t->ndim == 1) ? i : i * t->strides[0] + f * t->strides[1];
            int64_t out_idx = (out->ndim == 1) ? i : i * out->strides[0] + f * out->strides[1];
            od[out->offset + out_idx] = (td[t->offset + in_idx] - mn) / range;
        }
    }
    return out;
}
