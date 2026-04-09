/* serialize.c — model save/load implementation.

   file format (all little-endian):

   tensor file (.axt):
     u32  magic (0x41585430 = "AXT0")
     u32  dtype
     u32  ndim
     i64  shape[ndim]
     ...  raw data (numel * dtype_size bytes)

   model file (.axm):
     u32  magic (0x41584F4E = "AXON")
     u32  version
     u32  n_layers
     for each layer:
       u32  layer_type
       u32  n_params_in_layer
       i64  input_features
       i64  output_features
       f32  extra (alpha for leaky relu, axis for softmax, etc.)
       u8   flags (bit 0 = use_bias)
     for each param tensor (flattened across all layers):
       u32  dtype
       u32  ndim
       i64  shape[ndim]
       ...  raw data bytes

   the format is deliberately simple. no compression, no alignment padding,
   no variable-length strings. a microcontroller with fread() can load this. */

#include "axiom/serialize.h"
#include "axiom/error.h"
#include "axiom/activations.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define TENSOR_MAGIC 0x41585430  /* "AXT0" */

/* write helpers (little-endian on most platforms we care about;
   for big-endian targets you'd add byte swapping here) */

static bool write_u32(FILE *f, uint32_t v) { return fwrite(&v, 4, 1, f) == 1; }
static bool write_i64(FILE *f, int64_t v)  { return fwrite(&v, 8, 1, f) == 1; }
static bool write_f32(FILE *f, float v)    { return fwrite(&v, 4, 1, f) == 1; }
static bool write_u8(FILE *f, uint8_t v)   { return fwrite(&v, 1, 1, f) == 1; }

static bool read_u32(FILE *f, uint32_t *v) { return fread(v, 4, 1, f) == 1; }
static bool read_i64(FILE *f, int64_t *v)  { return fread(v, 8, 1, f) == 1; }
static bool read_f32(FILE *f, float *v)    { return fread(v, 4, 1, f) == 1; }
static bool read_u8(FILE *f, uint8_t *v)   { return fread(v, 1, 1, f) == 1; }

/* write a tensor's metadata + data to file */
static bool write_tensor(FILE *f, ax_tensor_t *t)
{
    if (!write_u32(f, (uint32_t)t->dtype)) return false;
    if (!write_u32(f, (uint32_t)t->ndim)) return false;
    for (int i = 0; i < t->ndim; i++)
    {
        if (!write_i64(f, t->shape[i])) return false;
    }

    /* make contiguous before writing (in case of transposed/strided tensor) */
    int64_t n = ax_tensor_numel(t);
    if (n < 0) return false; /* overflow in numel */
    size_t bytes = (size_t)n * ax_dtype_size(t->dtype);

    if (ax_tensor_is_contiguous(t) && t->offset == 0)
    {
        if (fwrite(t->storage->data, 1, bytes, f) != bytes) return false;
    }
    else
    {
        /* slow path: copy element by element for non-contiguous tensors */
        ax_tensor_t *c = ax_tensor_contiguous(t);
        if (!c) return false;
        bool ok = fwrite(c->storage->data, 1, bytes, f) == bytes;
        ax_tensor_destroy(c);
        if (!ok) return false;
    }
    return true;
}

/* safe multiply for serialization validation */
static bool ser_safe_mul(int64_t a, int64_t b, int64_t *result) {
    if (a == 0 || b == 0) { *result = 0; return true; }
    if (a > 0 && b > 0 && a > INT64_MAX / b) return false;
    *result = a * b;
    return true;
}

/* read a tensor from file, with full validation of untrusted data */
static ax_tensor_t *read_tensor(FILE *f)
{
    uint32_t dtype, ndim;
    if (!read_u32(f, &dtype)) return NULL;
    if (!read_u32(f, &ndim)) return NULL;

    /* validate dtype */
    if (dtype >= AX_DTYPE_COUNT) {
        ax_err_set(AX_ERR_INVALID_DTYPE,
                   "invalid dtype %u in file (max %d)", dtype, AX_DTYPE_COUNT - 1);
        return NULL;
    }

    /* validate ndim */
    if (ndim > AX_MAX_DIMS) {
        ax_err_set(AX_ERR_INVALID_SHAPE,
                   "ndim %u exceeds AX_MAX_DIMS (%d)", ndim, AX_MAX_DIMS);
        return NULL;
    }

    int64_t shape[AX_MAX_DIMS];
    int64_t numel = 1;
    for (uint32_t i = 0; i < ndim; i++)
    {
        if (!read_i64(f, &shape[i])) return NULL;
        /* validate each shape dimension is positive */
        if (shape[i] <= 0) {
            ax_err_set(AX_ERR_INVALID_SHAPE,
                       "shape[%u] = %ld is non-positive in file", i, shape[i]);
            return NULL;
        }
        /* check for overflow in numel computation */
        if (!ser_safe_mul(numel, shape[i], &numel)) {
            ax_err_set(AX_ERR_INVALID_SHAPE,
                       "integer overflow computing numel at dim %u", i);
            return NULL;
        }
    }

    /* check that numel * dtype_size doesn't overflow size_t */
    size_t elem_size = ax_dtype_size((ax_dtype_t)dtype);
    if (elem_size > 0 && (size_t)numel > SIZE_MAX / elem_size) {
        ax_err_set(AX_ERR_INVALID_SHAPE,
                   "allocation size overflow: %ld elements * %zu bytes", numel, elem_size);
        return NULL;
    }

    ax_tensor_t *t = ax_tensor_create(shape, (int)ndim, (ax_dtype_t)dtype);
    if (!t) return NULL;

    int64_t n = ax_tensor_numel(t);
    if (n < 0) { ax_tensor_destroy(t); return NULL; }
    size_t bytes = (size_t)n * ax_dtype_size(t->dtype);
    if (fread(t->storage->data, 1, bytes, f) != bytes)
    {
        ax_tensor_destroy(t);
        return NULL;
    }
    return t;
}


/* tensor save/load */

ax_status_t ax_tensor_save(ax_tensor_t *t, const char *path)
{
    if (!t || !path)
    {
        ax_err_set(AX_ERR_NULL_ARG, "null tensor or path");
        return AX_ERR_NULL_ARG;
    }

    FILE *f = fopen(path, "wb");
    if (!f)
    {
        ax_err_set(AX_ERR_INTERNAL, "cannot open %s for writing", path);
        return AX_ERR_INTERNAL;
    }

    bool ok = write_u32(f, TENSOR_MAGIC) && write_tensor(f, t);
    fclose(f);

    if (!ok)
    {
        ax_err_set(AX_ERR_INTERNAL, "write failed");
        return AX_ERR_INTERNAL;
    }
    return AX_OK;
}

ax_tensor_t *ax_tensor_load(const char *path)
{
    if (!path) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f)
    {
        ax_err_set(AX_ERR_INTERNAL, "cannot open %s for reading", path);
        return NULL;
    }

    uint32_t magic;
    if (!read_u32(f, &magic) || magic != TENSOR_MAGIC)
    {
        ax_err_set(AX_ERR_INTERNAL, "not a valid axiom tensor file");
        fclose(f);
        return NULL;
    }

    ax_tensor_t *t = read_tensor(f);
    fclose(f);
    return t;
}


/* model save */

ax_status_t ax_model_save(ax_model_t *model, const char *path)
{
    if (!model || !model->net || !path)
    {
        ax_err_set(AX_ERR_NULL_ARG, "null model or path");
        return AX_ERR_NULL_ARG;
    }

    if (model->net->type != AX_LAYER_SEQUENTIAL)
    {
        ax_err_set(AX_ERR_NOT_IMPLEMENTED, "only sequential models can be saved");
        return AX_ERR_NOT_IMPLEMENTED;
    }

    ax_sequential_t *seq = (ax_sequential_t *)model->net;
    FILE *f = fopen(path, "wb");
    if (!f)
    {
        ax_err_set(AX_ERR_INTERNAL, "cannot open %s for writing", path);
        return AX_ERR_INTERNAL;
    }

    /* header */
    write_u32(f, AX_MAGIC);
    write_u32(f, AX_FORMAT_VERSION);
    write_u32(f, (uint32_t)seq->n_layers);

    /* layer descriptors */
    for (int i = 0; i < seq->n_layers; i++)
    {
        ax_layer_t *l = seq->layers[i];
        write_u32(f, (uint32_t)l->type);
        write_u32(f, (uint32_t)l->n_params);
        write_i64(f, l->input_features);
        write_i64(f, l->output_features);

        /* extra context for parameterized activations */
        float extra = 0.0f;
        uint8_t flags = 0;

        if (l->type == AX_LAYER_DENSE)
        {
            ax_dense_t *d = (ax_dense_t *)l;
            flags = d->use_bias ? 1 : 0;
        }

        write_f32(f, extra);
        write_u8(f, flags);
    }

    /* parameter data */
    for (int i = 0; i < seq->n_layers; i++)
    {
        ax_layer_t *l = seq->layers[i];
        for (int p = 0; p < l->n_params; p++)
        {
            if (!write_tensor(f, l->params[p]))
            {
                fclose(f);
                ax_err_set(AX_ERR_INTERNAL, "failed writing params for layer %d", i);
                return AX_ERR_INTERNAL;
            }
        }
    }

    fclose(f);
    return AX_OK;
}


/* model load */

ax_model_t *ax_model_load(const char *path)
{
    if (!path) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f)
    {
        ax_err_set(AX_ERR_INTERNAL, "cannot open %s", path);
        return NULL;
    }

    /* header */
    uint32_t magic, version, n_layers;
    if (!read_u32(f, &magic) || !read_u32(f, &version) || !read_u32(f, &n_layers))
    {
        ax_err_set(AX_ERR_INTERNAL, "truncated model file header");
        fclose(f);
        return NULL;
    }

    if (magic != AX_MAGIC)
    {
        ax_err_set(AX_ERR_INTERNAL, "not a valid axiom model file");
        fclose(f);
        return NULL;
    }

    if (version > AX_FORMAT_VERSION)
    {
        ax_err_set(AX_ERR_INTERNAL, "model version %u newer than supported %u",
                   version, AX_FORMAT_VERSION);
        fclose(f);
        return NULL;
    }

    /* validate n_layers against maximum */
    if (n_layers == 0 || n_layers > AX_SEQ_MAX_LAYERS)
    {
        ax_err_set(AX_ERR_INVALID_SHAPE,
                   "n_layers %u invalid (max %d)", n_layers, AX_SEQ_MAX_LAYERS);
        fclose(f);
        return NULL;
    }

    /* read layer descriptors */
    typedef struct { uint32_t type, n_params; int64_t in_f, out_f; float extra; uint8_t flags; } layer_desc_t;
    layer_desc_t *descs = calloc(n_layers, sizeof(layer_desc_t));
    if (!descs)
    {
        ax_err_set(AX_ERR_ALLOC, "failed to allocate layer descriptors");
        fclose(f);
        return NULL;
    }

    for (uint32_t i = 0; i < n_layers; i++)
    {
        if (!read_u32(f, &descs[i].type) ||
            !read_u32(f, &descs[i].n_params) ||
            !read_i64(f, &descs[i].in_f) ||
            !read_i64(f, &descs[i].out_f) ||
            !read_f32(f, &descs[i].extra) ||
            !read_u8(f, &descs[i].flags))
        {
            ax_err_set(AX_ERR_INTERNAL,
                       "truncated layer descriptor at layer %u", i);
            free(descs);
            fclose(f);
            return NULL;
        }

        /* validate n_params per layer */
        if (descs[i].n_params > AX_LAYER_MAX_PARAMS) {
            ax_err_set(AX_ERR_INVALID_SHAPE,
                       "layer %u has %u params (max %d)",
                       i, descs[i].n_params, AX_LAYER_MAX_PARAMS);
            free(descs);
            fclose(f);
            return NULL;
        }

        /* validate feature dimensions are positive for parameterized layers */
        if (descs[i].type == AX_LAYER_DENSE && (descs[i].in_f <= 0 || descs[i].out_f <= 0)) {
            ax_err_set(AX_ERR_INVALID_SHAPE,
                       "layer %u has invalid features: in=%ld out=%ld",
                       i, descs[i].in_f, descs[i].out_f);
            free(descs);
            fclose(f);
            return NULL;
        }
    }

    /* reconstruct layers */
    ax_layer_t *seq = ax_sequential_create();

    for (uint32_t i = 0; i < n_layers; i++)
    {
        ax_layer_t *layer = NULL;

        switch (descs[i].type)
        {
            case AX_LAYER_DENSE:
                layer = ax_dense_create(descs[i].in_f, descs[i].out_f,
                                        descs[i].flags & 1);
                break;
            case AX_LAYER_RELU:       layer = ax_relu_layer_create(); break;
            case AX_LAYER_SIGMOID:    layer = ax_sigmoid_layer_create(); break;
            case AX_LAYER_TANH:       layer = ax_tanh_layer_create(); break;
            case AX_LAYER_LEAKY_RELU: layer = ax_leaky_relu_layer_create(descs[i].extra); break;
            case AX_LAYER_ELU:        layer = ax_elu_layer_create(descs[i].extra); break;
            case AX_LAYER_GELU:       layer = ax_gelu_layer_create(); break;
            case AX_LAYER_SWISH:      layer = ax_swish_layer_create(); break;
            case AX_LAYER_SOFTMAX:    layer = ax_softmax_layer_create((int)descs[i].extra); break;
            default:
                ax_err_set(AX_ERR_INTERNAL, "unknown layer type %u", descs[i].type);
                free(descs);
                ax_layer_destroy(seq);
                fclose(f);
                return NULL;
        }

        if (!layer)
        {
            free(descs);
            ax_layer_destroy(seq);
            fclose(f);
            return NULL;
        }

        ax_sequential_add(seq, layer);
    }

    /* load parameter data */
    for (uint32_t i = 0; i < n_layers; i++)
    {
        ax_sequential_t *s = (ax_sequential_t *)seq;
        ax_layer_t *layer = s->layers[i];

        for (int p = 0; p < layer->n_params; p++)
        {
            ax_tensor_t *loaded = read_tensor(f);
            if (!loaded)
            {
                ax_err_set(AX_ERR_INTERNAL, "failed reading params for layer %u param %d", i, p);
                free(descs);
                ax_layer_destroy(seq);
                fclose(f);
                return NULL;
            }

            /* validate loaded tensor matches existing parameter shape */
            ax_tensor_t *existing = layer->params[p];
            if (existing && existing->storage)
            {
                int64_t n_existing = ax_tensor_numel(existing);
                int64_t n_loaded = ax_tensor_numel(loaded);
                if (n_existing < 0 || n_loaded < 0 || n_existing != n_loaded)
                {
                    ax_err_set(AX_ERR_SHAPE_MISMATCH,
                               "layer %u param %d: expected %ld elements, file has %ld",
                               i, p, n_existing, n_loaded);
                    ax_tensor_destroy(loaded);
                    free(descs);
                    ax_layer_destroy(seq);
                    fclose(f);
                    return NULL;
                }
                size_t bytes = (size_t)n_existing * ax_dtype_size(existing->dtype);
                memcpy(existing->storage->data, loaded->storage->data, bytes);
            }
            ax_tensor_destroy(loaded);
        }
    }

    free(descs);
    fclose(f);

    ax_model_t *model = ax_model_create(seq);
    return model;
}
