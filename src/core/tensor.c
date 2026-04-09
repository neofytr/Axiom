/* tensor.c — tensor creation, storage management, shape ops */

#include "axiom/tensor.h"
#include "axiom/memory.h"
#include "axiom/compute.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* track whether rng has been seeded */
static bool rng_seeded = false;

/* storage */

ax_storage_t *ax_storage_create(size_t size_bytes, ax_device_t device) {
    ax_storage_t *s = (ax_storage_t *)malloc(sizeof(ax_storage_t));
    if (!s) return NULL;

    s->data = ax_aligned_alloc(size_bytes, AX_DEFAULT_ALIGNMENT);
    if (!s->data) {
        free(s);
        return NULL;
    }

    s->size_bytes = size_bytes;
    s->refcount = 1;
    s->device = device;
    return s;
}

void ax_storage_retain(ax_storage_t *s) {
    if (s) s->refcount++;
}

void ax_storage_release(ax_storage_t *s) {
    if (!s) return;
    s->refcount--;
    if (s->refcount <= 0) {
        ax_aligned_free(s->data);
        free(s);
    }
}

/* internal helpers */

/* compute total element count from shape */
static int64_t compute_numel(const int64_t *shape, int ndim) {
    int64_t n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    return n;
}

/* compute default row-major (c-contiguous) strides from shape */
static void compute_strides(const int64_t *shape, int ndim, int64_t *strides) {
    if (ndim == 0) return;
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

/* allocate tensor struct and fill metadata (no storage allocation) */
static ax_tensor_t *tensor_alloc_meta(const int64_t *shape, int ndim, ax_dtype_t dtype) {
    if (ndim < 0 || ndim > AX_MAX_DIMS) return NULL;

    ax_tensor_t *t = (ax_tensor_t *)calloc(1, sizeof(ax_tensor_t));
    if (!t) return NULL;

    t->ndim = ndim;
    t->dtype = dtype;
    t->offset = 0;
    t->requires_grad = false;
    t->grad = NULL;
    t->grad_fn = NULL;

    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
    }
    compute_strides(shape, ndim, t->strides);

    return t;
}

/* tensor creation */

ax_tensor_t *ax_tensor_create(const int64_t *shape, int ndim, ax_dtype_t dtype) {
    ax_tensor_t *t = tensor_alloc_meta(shape, ndim, dtype);
    if (!t) return NULL;

    int64_t n = compute_numel(shape, ndim);
    size_t bytes = (size_t)n * ax_dtype_size(dtype);
    if (bytes == 0) bytes = ax_dtype_size(dtype); /* scalar: at least one element */

    t->storage = ax_storage_create(bytes, AX_DEVICE_CPU);
    if (!t->storage) {
        free(t);
        return NULL;
    }
    return t;
}

ax_tensor_t *ax_tensor_zeros(const int64_t *shape, int ndim, ax_dtype_t dtype) {
    ax_tensor_t *t = ax_tensor_create(shape, ndim, dtype);
    if (!t) return NULL;
    memset(t->storage->data, 0, t->storage->size_bytes);
    return t;
}

ax_tensor_t *ax_tensor_ones(const int64_t *shape, int ndim, ax_dtype_t dtype) {
    return ax_tensor_full(shape, ndim, dtype, 1.0);
}

ax_tensor_t *ax_tensor_full(const int64_t *shape, int ndim, ax_dtype_t dtype, double value) {
    ax_tensor_t *t = ax_tensor_create(shape, ndim, dtype);
    if (!t) return NULL;
    ax_compute_fill(t, value);
    return t;
}

ax_tensor_t *ax_tensor_from_array(const void *data, const int64_t *shape, int ndim, ax_dtype_t dtype) {
    ax_tensor_t *t = ax_tensor_create(shape, ndim, dtype);
    if (!t) return NULL;

    int64_t n = compute_numel(shape, ndim);
    size_t bytes = (size_t)n * ax_dtype_size(dtype);
    memcpy(t->storage->data, data, bytes);
    return t;
}

ax_tensor_t *ax_tensor_arange(int64_t start, int64_t end, ax_dtype_t dtype) {
    int64_t len = end - start;
    if (len <= 0) return NULL;

    ax_tensor_t *t = ax_tensor_create(&len, 1, dtype);
    if (!t) return NULL;

    /* fill with sequential values */
    if (dtype == AX_FLOAT32) {
        float *d = (float *)t->storage->data;
        for (int64_t i = 0; i < len; i++) d[i] = (float)(start + i);
    } else if (dtype == AX_INT32) {
        int32_t *d = (int32_t *)t->storage->data;
        for (int64_t i = 0; i < len; i++) d[i] = (int32_t)(start + i);
    } else if (dtype == AX_FLOAT64) {
        double *d = (double *)t->storage->data;
        for (int64_t i = 0; i < len; i++) d[i] = (double)(start + i);
    } else if (dtype == AX_INT64) {
        int64_t *d = (int64_t *)t->storage->data;
        for (int64_t i = 0; i < len; i++) d[i] = start + i;
    }
    return t;
}

ax_tensor_t *ax_tensor_rand(const int64_t *shape, int ndim, float low, float high) {
    if (!rng_seeded) {
        srand((unsigned)time(NULL));
        rng_seeded = true;
    }

    ax_tensor_t *t = ax_tensor_create(shape, ndim, AX_FLOAT32);
    if (!t) return NULL;

    float range = high - low;
    int64_t n = compute_numel(shape, ndim);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++) {
        d[i] = low + ((float)rand() / (float)RAND_MAX) * range;
    }
    return t;
}

ax_tensor_t *ax_tensor_scalar(float value) {
    int64_t shape[] = {1};
    ax_tensor_t *t = ax_tensor_create(shape, 1, AX_FLOAT32);
    if (!t) return NULL;
    ((float *)t->storage->data)[0] = value;
    return t;
}

/* destruction */

void ax_tensor_destroy(ax_tensor_t *t) {
    if (!t) return;
    if (t->grad) ax_tensor_destroy(t->grad);
    ax_storage_release(t->storage);
    free(t);
}

/* shape queries */

int64_t ax_tensor_numel(const ax_tensor_t *t) {
    if (!t) return 0;
    return compute_numel(t->shape, t->ndim);
}

bool ax_tensor_is_contiguous(const ax_tensor_t *t) {
    if (!t || t->ndim == 0) return true;

    /* check strides match c-contiguous layout */
    int64_t expected = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        if (t->strides[i] != expected) return false;
        expected *= t->shape[i];
    }
    return true;
}

/* shape manipulation */

ax_tensor_t *ax_tensor_reshape(ax_tensor_t *t, const int64_t *new_shape, int new_ndim) {
    if (!t) return NULL;

    /* verify element count matches */
    int64_t old_n = compute_numel(t->shape, t->ndim);
    int64_t new_n = compute_numel(new_shape, new_ndim);
    if (old_n != new_n) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "reshape: %ld elements vs %ld elements", old_n, new_n);
        return NULL;
    }

    /* if contiguous, share storage (zero-copy) */
    if (ax_tensor_is_contiguous(t)) {
        ax_tensor_t *r = tensor_alloc_meta(new_shape, new_ndim, t->dtype);
        if (!r) return NULL;

        r->storage = t->storage;
        ax_storage_retain(r->storage);
        r->offset = t->offset;
        return r;
    }

    /* not contiguous: must copy first */
    ax_tensor_t *contig = ax_tensor_contiguous(t);
    if (!contig) return NULL;

    ax_tensor_t *r = tensor_alloc_meta(new_shape, new_ndim, t->dtype);
    if (!r) {
        ax_tensor_destroy(contig);
        return NULL;
    }

    r->storage = contig->storage;
    ax_storage_retain(r->storage);
    r->offset = 0;
    ax_tensor_destroy(contig);
    return r;
}

ax_tensor_t *ax_tensor_transpose(ax_tensor_t *t, int dim0, int dim1) {
    if (!t) return NULL;
    if (dim0 < 0 || dim0 >= t->ndim || dim1 < 0 || dim1 >= t->ndim) {
        ax_err_set(AX_ERR_INVALID_AXIS, "transpose dims out of range");
        return NULL;
    }

    /* create a view that shares storage, swap shape and strides */
    ax_tensor_t *r = tensor_alloc_meta(t->shape, t->ndim, t->dtype);
    if (!r) return NULL;

    r->storage = t->storage;
    ax_storage_retain(r->storage);
    r->offset = t->offset;

    /* copy strides from original, then swap */
    memcpy(r->strides, t->strides, sizeof(int64_t) * t->ndim);

    int64_t tmp_shape = r->shape[dim0];
    r->shape[dim0] = r->shape[dim1];
    r->shape[dim1] = tmp_shape;

    int64_t tmp_stride = r->strides[dim0];
    r->strides[dim0] = r->strides[dim1];
    r->strides[dim1] = tmp_stride;

    return r;
}

ax_tensor_t *ax_tensor_squeeze(ax_tensor_t *t, int dim) {
    if (!t) return NULL;
    if (dim < -1 || dim >= t->ndim) {
        ax_err_set(AX_ERR_INVALID_AXIS, "squeeze dim out of range");
        return NULL;
    }

    /* count resulting dimensions */
    int new_ndim = 0;
    int64_t new_shape[AX_MAX_DIMS];
    int64_t new_strides[AX_MAX_DIMS];

    for (int i = 0; i < t->ndim; i++) {
        /* dim == -1 means squeeze all size-1 dims */
        bool should_squeeze = (dim == -1) ? (t->shape[i] == 1) : (i == dim && t->shape[i] == 1);
        if (!should_squeeze) {
            new_shape[new_ndim] = t->shape[i];
            new_strides[new_ndim] = t->strides[i];
            new_ndim++;
        }
    }

    ax_tensor_t *r = tensor_alloc_meta(new_shape, new_ndim, t->dtype);
    if (!r) return NULL;

    r->storage = t->storage;
    ax_storage_retain(r->storage);
    r->offset = t->offset;
    memcpy(r->strides, new_strides, sizeof(int64_t) * new_ndim);
    return r;
}

ax_tensor_t *ax_tensor_unsqueeze(ax_tensor_t *t, int dim) {
    if (!t) return NULL;
    if (dim < 0 || dim > t->ndim) {
        ax_err_set(AX_ERR_INVALID_AXIS, "unsqueeze dim out of range");
        return NULL;
    }
    if (t->ndim >= AX_MAX_DIMS) {
        ax_err_set(AX_ERR_INVALID_SHAPE, "cannot unsqueeze: already at max dims");
        return NULL;
    }

    int64_t new_shape[AX_MAX_DIMS];
    int64_t new_strides[AX_MAX_DIMS];
    int new_ndim = t->ndim + 1;

    for (int i = 0, j = 0; i < new_ndim; i++) {
        if (i == dim) {
            new_shape[i] = 1;
            /* stride for size-1 dim doesn't matter, but set it sensibly */
            new_strides[i] = (dim < t->ndim) ? t->strides[j] * t->shape[j] : 1;
            if (i == new_ndim - 1 && j > 0) {
                new_strides[i] = 1;
            }
        } else {
            new_shape[i] = t->shape[j];
            new_strides[i] = t->strides[j];
            j++;
        }
    }

    ax_tensor_t *r = tensor_alloc_meta(new_shape, new_ndim, t->dtype);
    if (!r) return NULL;

    r->storage = t->storage;
    ax_storage_retain(r->storage);
    r->offset = t->offset;
    memcpy(r->strides, new_strides, sizeof(int64_t) * new_ndim);
    return r;
}

/* element access */

float ax_tensor_get_f32(const ax_tensor_t *t, const int64_t *indices) {
    size_t offset = t->offset;
    for (int i = 0; i < t->ndim; i++) {
        offset += indices[i] * t->strides[i];
    }
    return ((float *)t->storage->data)[offset];
}

void ax_tensor_set_f32(ax_tensor_t *t, const int64_t *indices, float value) {
    size_t offset = t->offset;
    for (int i = 0; i < t->ndim; i++) {
        offset += indices[i] * t->strides[i];
    }
    ((float *)t->storage->data)[offset] = value;
}

/* view / copy */

ax_tensor_t *ax_tensor_view(ax_tensor_t *t) {
    if (!t) return NULL;

    ax_tensor_t *v = tensor_alloc_meta(t->shape, t->ndim, t->dtype);
    if (!v) return NULL;

    v->storage = t->storage;
    ax_storage_retain(v->storage);
    v->offset = t->offset;
    memcpy(v->strides, t->strides, sizeof(int64_t) * t->ndim);
    return v;
}

ax_tensor_t *ax_tensor_contiguous(ax_tensor_t *t) {
    if (!t) return NULL;

    /* if already contiguous, just return a view */
    if (ax_tensor_is_contiguous(t) && t->offset == 0) {
        return ax_tensor_view(t);
    }

    /* allocate new contiguous storage and copy */
    ax_tensor_t *c = ax_tensor_create(t->shape, t->ndim, t->dtype);
    if (!c) return NULL;

    ax_compute_copy(t, c);
    return c;
}

/* printing */

void ax_tensor_print_shape(const ax_tensor_t *t) {
    if (!t) { printf("(null)\n"); return; }
    printf("tensor(shape=[");
    for (int i = 0; i < t->ndim; i++) {
        printf("%ld%s", t->shape[i], i < t->ndim - 1 ? ", " : "");
    }
    printf("], dtype=%s)\n", ax_dtype_name(t->dtype));
}

void ax_tensor_print(const ax_tensor_t *t) {
    if (!t) { printf("(null)\n"); return; }

    ax_tensor_print_shape(t);

    if (t->dtype != AX_FLOAT32) {
        printf("  (print only supports float32)\n");
        return;
    }

    int64_t n = ax_tensor_numel(t);
    int64_t limit = n > 20 ? 20 : n; /* print at most 20 elements */

    printf("  [");
    for (int64_t i = 0; i < limit; i++) {
        /* compute nd indices from flat index */
        int64_t remaining = i;
        int64_t indices[AX_MAX_DIMS];
        for (int d = t->ndim - 1; d >= 0; d--) {
            indices[d] = remaining % t->shape[d];
            remaining /= t->shape[d];
        }
        float v = ax_tensor_get_f32(t, indices);
        printf("%.4f%s", v, i < limit - 1 ? ", " : "");
    }
    if (n > limit) printf(", ... (%ld more)", n - limit);
    printf("]\n");
}
