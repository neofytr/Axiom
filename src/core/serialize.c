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
    size_t bytes = n * ax_dtype_size(t->dtype);

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

/* read a tensor from file */
static ax_tensor_t *read_tensor(FILE *f)
{
    uint32_t dtype, ndim;
    if (!read_u32(f, &dtype)) return NULL;
    if (!read_u32(f, &ndim)) return NULL;

    int64_t shape[AX_MAX_DIMS];
    for (uint32_t i = 0; i < ndim; i++)
    {
        if (!read_i64(f, &shape[i])) return NULL;
    }

    ax_tensor_t *t = ax_tensor_create(shape, (int)ndim, (ax_dtype_t)dtype);
    if (!t) return NULL;

    int64_t n = ax_tensor_numel(t);
    size_t bytes = n * ax_dtype_size(t->dtype);
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
    read_u32(f, &magic);
    read_u32(f, &version);
    read_u32(f, &n_layers);

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

    /* read layer descriptors */
    typedef struct { uint32_t type, n_params; int64_t in_f, out_f; float extra; uint8_t flags; } layer_desc_t;
    layer_desc_t *descs = calloc(n_layers, sizeof(layer_desc_t));

    for (uint32_t i = 0; i < n_layers; i++)
    {
        read_u32(f, &descs[i].type);
        read_u32(f, &descs[i].n_params);
        read_i64(f, &descs[i].in_f);
        read_i64(f, &descs[i].out_f);
        read_f32(f, &descs[i].extra);
        read_u8(f, &descs[i].flags);
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

            /* overwrite the randomly-initialized weights with loaded data */
            ax_tensor_t *existing = layer->params[p];
            if (existing && existing->storage)
            {
                int64_t n = ax_tensor_numel(existing);
                size_t bytes = n * ax_dtype_size(existing->dtype);
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
