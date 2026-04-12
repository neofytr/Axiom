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
#include "axiom/conv.h"
#include "axiom/norm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <inttypes.h>

#define TENSOR_MAGIC 0x41585430  /* "AXT0" */

/* max bytes we'll allocate for a single tensor from a file.
   prevents a crafted file from causing unbounded malloc (DoS).
   2 GB should cover any reasonable model. */
#define AX_MAX_TENSOR_BYTES ((size_t)2u * 1024u * 1024u * 1024u)

/* crc32 (iso 3309 polynomial, same as zlib/png).
   computed incrementally so we don't need to buffer the whole file. */
static uint32_t crc32_table[256];
static volatile int crc32_table_built = 0;

static void crc32_build_table(void)
{
    /* benign race: worst case two threads build the same table
       simultaneously. the result is identical either way because
       the polynomial is fixed. volatile flag ensures visibility. */
    if (crc32_table_built) return;
    for (uint32_t i = 0; i < 256; i++)
    {
        uint32_t c = i;
        for (int j = 0; j < 8; j++)
            c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        crc32_table[i] = c;
    }
    crc32_table_built = 1;
}

static uint32_t crc32_update(uint32_t crc, const void *data, size_t len)
{
    if (!crc32_table_built) crc32_build_table();
    const uint8_t *p = (const uint8_t *)data;
    crc = ~crc;
    for (size_t i = 0; i < len; i++)
        crc = crc32_table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

/* compute crc32 of a file region [start_pos, end_pos).
   if end_pos < 0, reads to EOF. */
static uint32_t crc32_of_file(FILE *f, long start_pos, long end_pos)
{
    fseek(f, start_pos, SEEK_SET);
    uint32_t crc = 0;
    uint8_t buf[4096];
    long remaining = (end_pos >= 0) ? (end_pos - start_pos) : -1;
    for (;;)
    {
        size_t to_read = sizeof(buf);
        if (remaining >= 0 && (long)to_read > remaining)
            to_read = (size_t)remaining;
        if (to_read == 0) break;
        size_t n = fread(buf, 1, to_read, f);
        if (n == 0) break;
        crc = crc32_update(crc, buf, n);
        if (remaining >= 0) remaining -= (long)n;
    }
    return crc;
}

#ifndef AX_INFERENCE_ONLY
/* write helpers with incremental crc tracking.
   pass crc=NULL to skip crc accumulation. compiled out in inference-only
   builds — no model save path exists there, only load. */

static bool write_bytes(FILE *f, const void *data, size_t len, uint32_t *crc)
{
    if (fwrite(data, 1, len, f) != len) return false;
    if (crc) *crc = crc32_update(*crc, data, len);
    return true;
}

static bool write_u32(FILE *f, uint32_t v)  { return write_bytes(f, &v, 4, NULL); }

/* crc-tracked variants */
static bool write_u32_c(FILE *f, uint32_t v, uint32_t *c)  { return write_bytes(f, &v, 4, c); }
static bool write_i64_c(FILE *f, int64_t v, uint32_t *c)   { return write_bytes(f, &v, 8, c); }
static bool write_f32_c(FILE *f, float v, uint32_t *c)     { return write_bytes(f, &v, 4, c); }
static bool write_u8_c(FILE *f, uint8_t v, uint32_t *c)    { return write_bytes(f, &v, 1, c); }
#endif /* !AX_INFERENCE_ONLY */

static bool read_u32(FILE *f, uint32_t *v) { return fread(v, 4, 1, f) == 1; }
static bool read_i64(FILE *f, int64_t *v)  { return fread(v, 8, 1, f) == 1; }
static bool read_f32(FILE *f, float *v)    { return fread(v, 4, 1, f) == 1; }
static bool read_u8(FILE *f, uint8_t *v)   { return fread(v, 1, 1, f) == 1; }

#ifndef AX_INFERENCE_ONLY
/* write a tensor's metadata + data to file, optionally tracking crc */
static bool write_tensor_crc(FILE *f, ax_tensor_t *t, uint32_t *crc)
{
    if (!write_u32_c(f, (uint32_t)t->dtype, crc)) return false;
    if (!write_u32_c(f, (uint32_t)t->ndim, crc)) return false;
    for (int i = 0; i < t->ndim; i++)
    {
        if (!write_i64_c(f, t->shape[i], crc)) return false;
    }

    /* make contiguous before writing (in case of transposed/strided tensor) */
    int64_t n = ax_tensor_numel(t);
    if (n < 0) return false; /* overflow in numel */
    size_t bytes = (size_t)n * ax_dtype_size(t->dtype);

    if (ax_tensor_is_contiguous(t) && t->offset == 0)
    {
        if (!write_bytes(f, t->storage->data, bytes, crc)) return false;
    }
    else
    {
        ax_tensor_t *c = ax_tensor_contiguous(t);
        if (!c) return false;
        bool ok = write_bytes(f, c->storage->data, bytes, crc);
        ax_tensor_destroy(c);
        if (!ok) return false;
    }
    return true;
}

static bool write_tensor(FILE *f, ax_tensor_t *t)
{
    return write_tensor_crc(f, t, NULL);
}
#endif /* !AX_INFERENCE_ONLY */

/* safe multiply for serialization validation — all four sign combinations */
static bool ser_safe_mul(int64_t a, int64_t b, int64_t *result) {
    if (a == 0 || b == 0) { *result = 0; return true; }
    if (a > 0 && b > 0 && a > INT64_MAX / b) return false;
    if (a > 0 && b < 0 && b < INT64_MIN / a) return false;
    if (a < 0 && b > 0 && a < INT64_MIN / b) return false;
    if (a < 0 && b < 0 && a < INT64_MAX / b) return false;
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
                       "shape[%u] = %" PRId64 " is non-positive in file", i, shape[i]);
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
                   "allocation size overflow: %" PRId64 " elements * %zu bytes", numel, elem_size);
        return NULL;
    }

    /* cap allocation size to prevent DoS from crafted files */
    size_t total_bytes = (size_t)numel * elem_size;
    if (total_bytes > AX_MAX_TENSOR_BYTES) {
        ax_err_set(AX_ERR_INVALID_SHAPE,
                   "tensor too large for deserialization: %zu bytes (max %zu)",
                   total_bytes, AX_MAX_TENSOR_BYTES);
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

#ifndef AX_INFERENCE_ONLY
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
#endif /* !AX_INFERENCE_ONLY */

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

#ifndef AX_INFERENCE_ONLY
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

    /* all writes go through crc-tracked variants */
    uint32_t crc = 0;

    /* header */
    write_u32_c(f, AX_MAGIC, &crc);
    write_u32_c(f, AX_FORMAT_VERSION, &crc);
    write_u32_c(f, (uint32_t)seq->n_layers, &crc);

    /* layer descriptors */
    for (int i = 0; i < seq->n_layers; i++)
    {
        ax_layer_t *l = seq->layers[i];
        write_u32_c(f, (uint32_t)l->type, &crc);
        write_u32_c(f, (uint32_t)l->n_params, &crc);
        write_u32_c(f, (uint32_t)l->n_buffers, &crc);

        switch (l->type)
        {
            case AX_LAYER_DENSE: {
                ax_dense_t *d = (ax_dense_t *)l;
                write_i64_c(f, l->input_features, &crc);
                write_i64_c(f, l->output_features, &crc);
                write_u8_c(f, d->use_bias ? 1 : 0, &crc);
                break;
            }
            case AX_LAYER_CONV2D: {
                ax_conv2d_t *c = (ax_conv2d_t *)l;
                write_u32_c(f, (uint32_t)c->in_channels, &crc);
                write_u32_c(f, (uint32_t)c->out_channels, &crc);
                write_u32_c(f, (uint32_t)c->kernel_h, &crc);
                write_u32_c(f, (uint32_t)c->kernel_w, &crc);
                write_u32_c(f, (uint32_t)c->stride_h, &crc);
                write_u32_c(f, (uint32_t)c->stride_w, &crc);
                write_u32_c(f, (uint32_t)c->pad_h, &crc);
                write_u32_c(f, (uint32_t)c->pad_w, &crc);
                write_u8_c(f, c->use_bias ? 1 : 0, &crc);
                break;
            }
            case AX_LAYER_MAXPOOL2D: {
                ax_maxpool2d_t *p = (ax_maxpool2d_t *)l;
                write_u32_c(f, (uint32_t)p->kernel_size, &crc);
                write_u32_c(f, (uint32_t)p->stride, &crc);
                write_u32_c(f, (uint32_t)p->padding, &crc);
                break;
            }
            case AX_LAYER_AVGPOOL2D: {
                ax_avgpool2d_t *p = (ax_avgpool2d_t *)l;
                write_u32_c(f, (uint32_t)p->kernel_size, &crc);
                write_u32_c(f, (uint32_t)p->stride, &crc);
                write_u32_c(f, (uint32_t)p->padding, &crc);
                break;
            }
            case AX_LAYER_BATCHNORM: {
                ax_batchnorm_t *bn = (ax_batchnorm_t *)l;
                write_i64_c(f, bn->num_features, &crc);
                write_f32_c(f, bn->eps, &crc);
                write_f32_c(f, bn->momentum, &crc);
                break;
            }
            case AX_LAYER_LAYERNORM: {
                ax_layernorm_t *ln = (ax_layernorm_t *)l;
                write_i64_c(f, ln->num_features, &crc);
                write_f32_c(f, ln->eps, &crc);
                break;
            }
            case AX_LAYER_DROPOUT: {
                ax_dropout_t *dp = (ax_dropout_t *)l;
                write_f32_c(f, dp->p, &crc);
                break;
            }
            case AX_LAYER_LEAKY_RELU:
            case AX_LAYER_ELU: {
                ax_activation_layer_t *al = (ax_activation_layer_t *)l;
                write_f32_c(f, al->alpha, &crc);
                break;
            }
            case AX_LAYER_SOFTMAX: {
                write_u32_c(f, 1, &crc);
                break;
            }
            default:
                break;
        }
    }

    /* parameter and buffer data */
    for (int i = 0; i < seq->n_layers; i++)
    {
        ax_layer_t *l = seq->layers[i];
        for (int p = 0; p < l->n_params; p++)
        {
            if (!write_tensor_crc(f, l->params[p], &crc))
            {
                fclose(f);
                ax_err_set(AX_ERR_INTERNAL, "failed writing params for layer %d", i);
                return AX_ERR_INTERNAL;
            }
        }
        for (int p = 0; p < l->n_buffers; p++)
        {
            if (!write_tensor_crc(f, l->buffers[p], &crc))
            {
                fclose(f);
                ax_err_set(AX_ERR_INTERNAL, "failed writing buffers for layer %d", i);
                return AX_ERR_INTERNAL;
            }
        }
    }

    /* append crc32 (not included in the checksum itself) */
    write_u32(f, crc);

    fclose(f);
    return AX_OK;
}
#endif /* !AX_INFERENCE_ONLY */


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

    /* reconstruct layers from type-specific descriptors */
    uint32_t layer_n_buffers[AX_SEQ_MAX_LAYERS];
    memset(layer_n_buffers, 0, sizeof(layer_n_buffers));
    ax_layer_t *seq = ax_sequential_create();
    if (!seq) { fclose(f); return NULL; }

    for (uint32_t i = 0; i < n_layers; i++)
    {
        uint32_t type, n_params, n_buffers_file = 0;
        if (!read_u32(f, &type) || !read_u32(f, &n_params))
        {
            ax_err_set(AX_ERR_INTERNAL, "truncated layer header at layer %u", i);
            ax_layer_destroy(seq);
            fclose(f);
            return NULL;
        }

        if (version >= 3) {
            if (!read_u32(f, &n_buffers_file))
            {
                ax_err_set(AX_ERR_INTERNAL, "truncated layer header (n_buffers) at layer %u", i);
                ax_layer_destroy(seq);
                fclose(f);
                return NULL;
            }
        }

        if (n_params > AX_LAYER_MAX_PARAMS)
        {
            ax_err_set(AX_ERR_INVALID_SHAPE, "layer %u: %u params exceeds max", i, n_params);
            ax_layer_destroy(seq); fclose(f); return NULL;
        }

        layer_n_buffers[i] = n_buffers_file;
        ax_layer_t *layer = NULL;

        switch (type)
        {
            case AX_LAYER_DENSE: {
                int64_t in_f, out_f; uint8_t bias;
                if (!read_i64(f, &in_f) || !read_i64(f, &out_f) || !read_u8(f, &bias))
                { ax_layer_destroy(seq); fclose(f); return NULL; }
                if (in_f <= 0 || out_f <= 0)
                { ax_layer_destroy(seq); fclose(f); return NULL; }
                layer = ax_dense_create(in_f, out_f, bias & 1);
                break;
            }
            case AX_LAYER_CONV2D: {
                uint32_t in_ch, out_ch, kh, kw, sh, sw, ph, pw; uint8_t bias;
                if (!read_u32(f, &in_ch) || !read_u32(f, &out_ch) ||
                    !read_u32(f, &kh) || !read_u32(f, &kw) ||
                    !read_u32(f, &sh) || !read_u32(f, &sw) ||
                    !read_u32(f, &ph) || !read_u32(f, &pw) ||
                    !read_u8(f, &bias))
                { ax_layer_destroy(seq); fclose(f); return NULL; }
                layer = ax_conv2d_create_ex((int)in_ch, (int)out_ch, (int)kh, (int)kw, (int)sh, (int)sw, (int)ph, (int)pw, bias & 1);
                break;
            }
            case AX_LAYER_MAXPOOL2D: {
                uint32_t ks, st, pd;
                if (!read_u32(f, &ks) || !read_u32(f, &st) || !read_u32(f, &pd))
                { ax_layer_destroy(seq); fclose(f); return NULL; }
                layer = ax_maxpool2d_create((int)ks, (int)st, (int)pd);
                break;
            }
            case AX_LAYER_AVGPOOL2D: {
                uint32_t ks, st, pd;
                if (!read_u32(f, &ks) || !read_u32(f, &st) || !read_u32(f, &pd))
                { ax_layer_destroy(seq); fclose(f); return NULL; }
                layer = ax_avgpool2d_create((int)ks, (int)st, (int)pd);
                break;
            }
            case AX_LAYER_BATCHNORM: {
                int64_t nf; float eps, mom;
                if (!read_i64(f, &nf) || !read_f32(f, &eps) || !read_f32(f, &mom))
                { ax_layer_destroy(seq); fclose(f); return NULL; }
                if (nf <= 0) { ax_layer_destroy(seq); fclose(f); return NULL; }
                layer = ax_batchnorm_create(nf, eps, mom);
                break;
            }
            case AX_LAYER_LAYERNORM: {
                int64_t nf; float eps;
                if (!read_i64(f, &nf) || !read_f32(f, &eps))
                { ax_layer_destroy(seq); fclose(f); return NULL; }
                if (nf <= 0) { ax_layer_destroy(seq); fclose(f); return NULL; }
                layer = ax_layernorm_create(nf, eps);
                break;
            }
            case AX_LAYER_DROPOUT: {
                float p;
                if (!read_f32(f, &p))
                { ax_layer_destroy(seq); fclose(f); return NULL; }
                layer = ax_dropout_create(p);
                break;
            }
            case AX_LAYER_LEAKY_RELU:
            case AX_LAYER_ELU: {
                float alpha;
                if (!read_f32(f, &alpha))
                { ax_layer_destroy(seq); fclose(f); return NULL; }
                layer = (type == AX_LAYER_LEAKY_RELU)
                    ? ax_leaky_relu_layer_create(alpha)
                    : ax_elu_layer_create(alpha);
                break;
            }
            case AX_LAYER_SOFTMAX: {
                uint32_t axis;
                if (!read_u32(f, &axis))
                { ax_layer_destroy(seq); fclose(f); return NULL; }
                layer = ax_softmax_layer_create((int)axis);
                break;
            }
            case AX_LAYER_RELU:           layer = ax_relu_layer_create(); break;
            case AX_LAYER_SIGMOID:        layer = ax_sigmoid_layer_create(); break;
            case AX_LAYER_TANH:           layer = ax_tanh_layer_create(); break;
            case AX_LAYER_GELU:           layer = ax_gelu_layer_create(); break;
            case AX_LAYER_SWISH:          layer = ax_swish_layer_create(); break;
            case AX_LAYER_FLATTEN:        layer = ax_flatten_create(); break;
            case AX_LAYER_GLOBAL_AVGPOOL2D: layer = ax_global_avgpool2d_create(); break;
            default:
                ax_err_set(AX_ERR_INTERNAL, "unknown layer type %u at layer %u", type, i);
                ax_layer_destroy(seq); fclose(f); return NULL;
        }

        if (!layer) { ax_layer_destroy(seq); fclose(f); return NULL; }
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
                               "layer %u param %d: expected %" PRId64 " elements, file has %" PRId64,
                               i, p, n_existing, n_loaded);
                    ax_tensor_destroy(loaded);
                    ax_layer_destroy(seq);
                    fclose(f);
                    return NULL;
                }
                size_t bytes = (size_t)n_existing * ax_dtype_size(existing->dtype);
                memcpy(existing->storage->data, loaded->storage->data, bytes);
            }
            ax_tensor_destroy(loaded);
        }

        /* load buffer data (version >= 3) */
        uint32_t nb = layer_n_buffers[i];
        for (uint32_t b = 0; b < nb; b++)
        {
            ax_tensor_t *loaded = read_tensor(f);
            if (!loaded)
            {
                ax_err_set(AX_ERR_INTERNAL, "failed reading buffer for layer %u buffer %u", i, b);
                ax_layer_destroy(seq);
                fclose(f);
                return NULL;
            }

            if (b < (uint32_t)layer->n_buffers)
            {
                ax_tensor_t *existing = layer->buffers[b];
                if (existing && existing->storage)
                {
                    int64_t n_existing = ax_tensor_numel(existing);
                    int64_t n_loaded = ax_tensor_numel(loaded);
                    if (n_existing == n_loaded)
                    {
                        size_t bytes = (size_t)n_existing * ax_dtype_size(existing->dtype);
                        memcpy(existing->storage->data, loaded->storage->data, bytes);
                    }
                }
            }
            ax_tensor_destroy(loaded);
        }
    }

    /* verify crc32 (version >= 4) */
    if (version >= 4)
    {
        long data_end = ftell(f);
        uint32_t stored_crc;
        if (!read_u32(f, &stored_crc))
        {
            ax_err_set(AX_ERR_INTERNAL, "missing checksum in model file");
            ax_layer_destroy(seq);
            fclose(f);
            return NULL;
        }
        uint32_t computed_crc = crc32_of_file(f, 0, data_end);
        if (computed_crc != stored_crc)
        {
            ax_err_set(AX_ERR_INTERNAL,
                       "model file corrupted: checksum mismatch (stored 0x%08x, computed 0x%08x)",
                       stored_crc, computed_crc);
            ax_layer_destroy(seq);
            fclose(f);
            return NULL;
        }
    }

    fclose(f);

    ax_model_t *model = ax_model_create(seq);
    return model;
}
