/* tensor.c — tensor creation, storage management, shape ops */

#include "axiom/tensor.h"
#include "axiom/autograd.h"
#include "axiom/memory.h"
#include "axiom/compute.h"
#include "axiom/device.h"
#include "axiom/rng.h"
#include <stdalign.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <inttypes.h>

/* default device for all new tensor allocations (thread-local) */
static _Thread_local ax_device_t tl_default_device = AX_DEVICE_CPU;

void ax_set_default_device(ax_device_t dev) { tl_default_device = dev; }
ax_device_t ax_get_default_device(void)     { return tl_default_device; }

void ax_set_seed(unsigned int seed)
{
    ax_rng_seed((uint64_t)seed);
}

/* storage */

/* thread-local slab allocator for ax_tensor_t and ax_storage_t structs.
   these fixed-size metadata structs are allocated/freed hundreds of times per
   training step — a free-list eliminates all per-struct malloc traffic. */

static _Thread_local ax_tensor_t *tensor_freelist = NULL;
static _Thread_local ax_storage_t *storage_freelist = NULL;

static ax_tensor_t *slab_tensor_alloc(void) {
    ax_tensor_t *t = tensor_freelist;
    if (t) {
        tensor_freelist = *(ax_tensor_t **)t; /* next pointer lives in first slot */
        memset(t, 0, sizeof(ax_tensor_t));
        return t;
    }
    return (ax_tensor_t *)calloc(1, sizeof(ax_tensor_t));
}

static void slab_tensor_free(ax_tensor_t *t) {
    if (!t) return;
    *(ax_tensor_t **)t = tensor_freelist;
    tensor_freelist = t;
}

static ax_storage_t *slab_storage_alloc(void) {
    ax_storage_t *s = storage_freelist;
    if (s) {
        storage_freelist = *(ax_storage_t **)s;
        return s;
    }
    return (ax_storage_t *)malloc(sizeof(ax_storage_t));
}

static void slab_storage_free(ax_storage_t *s) {
    if (!s) return;
    *(ax_storage_t **)s = storage_freelist;
    storage_freelist = s;
}


/* thread-local storage DATA pool — eliminates malloc/free for buffer bytes.
   buckets are indexed by ceil(log2(size_bytes)), from 6 (64 B) to 28 (256 MB).
   each bucket is a singly-linked free list capped at AX_POOL_BUCKET_CAP entries.
   storages keep their original capacity in size_bytes; pool reuses if capacity
   meets the request. _Thread_local means zero locking on the fast path. */

#ifndef AX_NO_STORAGE_POOL
#define AX_POOL_MIN_BITS    6        /* 64 B  */
#define AX_POOL_MAX_BITS    28       /* 256 MB */
#define AX_POOL_NUM_BUCKETS (AX_POOL_MAX_BITS - AX_POOL_MIN_BITS + 1)
#define AX_POOL_BUCKET_CAP  64       /* raised from 16 per agent report point 9 */

typedef struct pool_node {
    ax_storage_t *storage;
    struct pool_node *next;
} pool_node_t;

/* per-thread bucket state */
static _Thread_local pool_node_t *pool_buckets[AX_POOL_NUM_BUCKETS] = {0};
static _Thread_local int pool_bucket_count[AX_POOL_NUM_BUCKETS] = {0};
/* free-list of empty pool_node_t structs to avoid mallocing those too */
static _Thread_local pool_node_t *node_freelist = NULL;

/* ceil(log2(x)) for x >= 1; returns AX_POOL_MIN_BITS for tiny x */
static inline int pool_bucket_for(size_t bytes) {
    if (bytes <= (1u << AX_POOL_MIN_BITS)) return 0;
    int b = 0;
    size_t v = bytes - 1;
    while (v) { v >>= 1; b++; }
    /* b is now ceil(log2(bytes)); subtract MIN to get bucket index */
    if (b < AX_POOL_MIN_BITS) b = AX_POOL_MIN_BITS;
    int idx = b - AX_POOL_MIN_BITS;
    if (idx >= AX_POOL_NUM_BUCKETS) return -1; /* too large for pool */
    return idx;
}

/* try to pop a storage from the pool that has at least `bytes` capacity.
   returns NULL on miss. */
static ax_storage_t *pool_get(size_t bytes) {
    int idx = pool_bucket_for(bytes);
    if (idx < 0) return NULL;
    pool_node_t *n = pool_buckets[idx];
    if (!n) return NULL;
    /* the bucket holds storages whose capacity is >= 2^(MIN+idx-1)+1 and <= 2^(MIN+idx).
       so head storage's capacity is at least 2^(MIN+idx-1)+1 which may be < bytes.
       guard: only return if capacity >= bytes. */
    if (n->storage->size_bytes < bytes) return NULL;
    pool_buckets[idx] = n->next;
    pool_bucket_count[idx]--;
    ax_storage_t *s = n->storage;
    /* recycle the node */
    n->next = node_freelist;
    node_freelist = n;
    return s;
}

/* return a storage to the pool. returns true if pooled, false if caller should free. */
static bool pool_put(ax_storage_t *s) {
    /* CUDA unified memory must NOT mix with CPU pool — keep them separate */
    if (s->device != AX_DEVICE_CPU) return false;
    int idx = pool_bucket_for(s->size_bytes);
    if (idx < 0) return false;
    if (pool_bucket_count[idx] >= AX_POOL_BUCKET_CAP) return false;

    pool_node_t *n = node_freelist;
    if (n) {
        node_freelist = n->next;
    } else {
        n = (pool_node_t *)malloc(sizeof(pool_node_t));
        if (!n) return false;
    }
    n->storage = s;
    n->next = pool_buckets[idx];
    pool_buckets[idx] = n;
    pool_bucket_count[idx]++;
    return true;
}
#endif /* !AX_NO_STORAGE_POOL */

ax_storage_t *ax_storage_create(size_t size_bytes, ax_device_t device) {
    /* non-cpu devices are owned by a backend module; route allocation
       through its vtable so core has zero device-specific code. */
    if (device != AX_DEVICE_CPU) {
        const ax_backend_ops_t *ops = ax_backend_for_device(device);
        if (!ops || !ops->storage_alloc) {
            ax_err_set(AX_ERR_BACKEND,
                       "no backend registered for device %d", (int)device);
            return NULL;
        }
        ax_storage_t *s = slab_storage_alloc();
        if (!s) return NULL;
        s->data = ops->storage_alloc(size_bytes);
        if (!s->data) {
            slab_storage_free(s);
            return NULL;
        }
        s->size_bytes = size_bytes;
        atomic_init(&s->refcount, 1);
        s->device = device;
        s->is_arena_temp = false;
        s->generation = 1;
        return s;
    }

#ifndef AX_NO_STORAGE_POOL
    /* try the pool first — returns a storage with pre-allocated data buffer */
    ax_storage_t *pooled = pool_get(size_bytes);
    if (pooled) {
        atomic_init(&pooled->refcount, 1);
        pooled->device = device;
        pooled->is_arena_temp = false;
        /* pooled buffers are recycled — bump generation so any stale cache
           entry keyed on this storage pointer from a previous owner is
           invalidated on first use. */
        pooled->generation++;
        return pooled;
    }
#endif
    /* slab-allocate the struct; only the data buffer hits aligned_alloc */
    ax_storage_t *s = slab_storage_alloc();
    if (!s) return NULL;

    s->data = ax_aligned_alloc(size_bytes, AX_DEFAULT_ALIGNMENT);
    if (!s->data) {
        slab_storage_free(s);
        return NULL;
    }

    s->size_bytes = size_bytes;
    atomic_init(&s->refcount, 1);
    s->device = device;
    s->is_arena_temp = false;
    s->generation = 1;
    return s;
}

void ax_storage_retain(ax_storage_t *s) {
    if (s) atomic_fetch_add(&s->refcount, 1);
}

void ax_storage_release(ax_storage_t *s) {
    if (!s) return;
    /* arena-allocated storages are owned by the arena — never free or pool them. */
    if (s->is_arena_temp) return;
    int prev = atomic_fetch_sub(&s->refcount, 1);
    if (prev <= 1) {
        /* non-cpu storage: ask the owning backend to free it. no pooling
           for foreign devices — their allocators are typically hotter
           than our pool anyway, and mixing device pointers across
           backends would be a footgun. */
        if (s->device != AX_DEVICE_CPU) {
            const ax_backend_ops_t *ops = ax_backend_for_device(s->device);
            if (ops && ops->storage_free) ops->storage_free(s->data);
            slab_storage_free(s);
            return;
        }
#ifndef AX_NO_STORAGE_POOL
        /* cpu path: try to return to pool; on success, struct AND data stay alive */
        if (pool_put(s)) return;
#endif
        /* cpu pool miss: free the data buffer, return the struct to slab */
        ax_aligned_free(s->data);
        slab_storage_free(s);
    }
}

/* internal helpers */

/* safe multiply that checks for overflow. returns false on overflow. */
static bool safe_mul_i64(int64_t a, int64_t b, int64_t *result) {
    if (a == 0 || b == 0) { *result = 0; return true; }
    if (a > 0 && b > 0 && a > INT64_MAX / b) return false;
    if (a > 0 && b < 0 && b < INT64_MIN / a) return false;
    if (a < 0 && b > 0 && a < INT64_MIN / b) return false;
    if (a < 0 && b < 0 && a < INT64_MAX / b) return false;
    *result = a * b;
    return true;
}

/* compute total element count from shape. returns -1 on overflow. */
static int64_t compute_numel(const int64_t *shape, int ndim) {
    int64_t n = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            ax_err_set(AX_ERR_INVALID_SHAPE,
                       "shape dimension %d is non-positive: %" PRId64, i, shape[i]);
            return -1;
        }
        if (!safe_mul_i64(n, shape[i], &n)) {
            ax_err_set(AX_ERR_INVALID_SHAPE,
                       "integer overflow computing numel at dim %d", i);
            return -1;
        }
    }
    return n;
}

/* compute default row-major (c-contiguous) strides from shape.
   returns false on overflow. */
static bool compute_strides(const int64_t *shape, int ndim, int64_t *strides) {
    if (ndim == 0) return true;
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        if (!safe_mul_i64(strides[i + 1], shape[i + 1], &strides[i])) {
            ax_err_set(AX_ERR_INVALID_SHAPE,
                       "integer overflow computing strides at dim %d", i);
            return false;
        }
    }
    return true;
}

/* allocate tensor struct and fill metadata (no storage allocation) */
static ax_tensor_t *tensor_alloc_meta(const int64_t *shape, int ndim, ax_dtype_t dtype) {
    if (ndim < 0 || ndim > AX_MAX_DIMS) return NULL;
    if ((int)dtype < 0 || dtype >= AX_DTYPE_COUNT) {
        ax_err_set(AX_ERR_INVALID_DTYPE, "invalid dtype value: %d", (int)dtype);
        return NULL;
    }

    /* validate all shape dimensions are positive */
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            ax_err_set(AX_ERR_INVALID_SHAPE,
                       "shape dimension %d must be positive, got %" PRId64, i, shape[i]);
            return NULL;
        }
    }

    /* slab-allocated; already zeroed by slab_tensor_alloc */
    ax_tensor_t *t = slab_tensor_alloc();
    if (!t) return NULL;

    t->ndim = ndim;
    t->dtype = dtype;

    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
    }
    if (!compute_strides(shape, ndim, t->strides)) {
        slab_tensor_free(t);
        return NULL;
    }

    return t;
}

/* tensor creation */

ax_tensor_t *ax_tensor_create(const int64_t *shape, int ndim, ax_dtype_t dtype) {
    ax_tensor_t *t = tensor_alloc_meta(shape, ndim, dtype);
    if (!t) return NULL;

    int64_t n = compute_numel(shape, ndim);
    if (n < 0) { slab_tensor_free(t); return NULL; } /* overflow in numel */

    size_t elem_size = ax_dtype_size(dtype);
    if (elem_size > 0 && (size_t)n > SIZE_MAX / elem_size) {
        ax_err_set(AX_ERR_INVALID_SHAPE,
                   "allocation size overflow: %" PRId64 " elements * %zu bytes", n, elem_size);
        slab_tensor_free(t);
        return NULL;
    }
    size_t bytes = (size_t)n * elem_size;
    if (bytes == 0) bytes = elem_size;

    t->storage = ax_storage_create(bytes, tl_default_device);
    if (!t->storage) {
        slab_tensor_free(t);
        return NULL;
    }
    return t;
}

/* device-aware host-to-tensor helpers.

   core code historically wrote to tensor->storage->data directly with
   memcpy/memset, assuming the buffer was host-addressable. that was
   true as long as the cuda backend used cudaMallocManaged, because the
   driver transparently page-faulted host reads/writes into gpu pages.
   now that we're moving toward explicit device memory, every host-side
   write has to check the tensor's device and either memcpy (cpu) or
   go through the owning backend's h2d / fill hooks (non-cpu).

   these helpers encapsulate that. they return AX_OK on success and set
   ax_err_set on failure. callers that don't check the return value get
   silent failures, which matches the existing init.c void-return
   contract. */

/* fill entire tensor storage with a single scalar value. */
static ax_status_t tensor_fill_value(ax_tensor_t *t, double value) {
    if (t->storage->device == AX_DEVICE_CPU) {
        /* fast path: in-process memset for zero, kernel loop otherwise */
        if (value == 0.0) {
            memset(t->storage->data, 0, t->storage->size_bytes);
            ax_storage_touch(t->storage);
            return AX_OK;
        }
        /* ax_compute_fill bumps generation via dispatch_touch_on_ok */
        return ax_compute_fill(t, value);
    }
    const ax_backend_ops_t *ops = ax_backend_for_device(t->storage->device);
    if (!ops || !ops->fill) {
        ax_err_set(AX_ERR_BACKEND,
                   "device %d has no fill hook", (int)t->storage->device);
        return AX_ERR_BACKEND;
    }
    ax_status_t s = ops->fill(t, value);
    if (s == AX_OK) ax_storage_touch(t->storage);
    return s;
}

/* copy `bytes` from host buffer `data` into tensor storage at t->offset.
   `bytes` must equal numel * dtype_size; caller's job to enforce. */
static ax_status_t tensor_write_from_host(ax_tensor_t *t, const void *data, size_t bytes) {
    void *dst = (char *)t->storage->data + t->offset * ax_dtype_size(t->dtype);
    if (t->storage->device == AX_DEVICE_CPU) {
        memcpy(dst, data, bytes);
        ax_storage_touch(t->storage);
        return AX_OK;
    }
    const ax_backend_ops_t *ops = ax_backend_for_device(t->storage->device);
    if (!ops || !ops->memcpy_h2d) {
        ax_err_set(AX_ERR_BACKEND,
                   "device %d has no h2d hook", (int)t->storage->device);
        return AX_ERR_BACKEND;
    }
    ax_status_t s = ops->memcpy_h2d(dst, data, bytes);
    if (s == AX_OK) ax_storage_touch(t->storage);
    return s;
}

ax_tensor_t *ax_tensor_zeros(const int64_t *shape, int ndim, ax_dtype_t dtype) {
    ax_tensor_t *t = ax_tensor_create(shape, ndim, dtype);
    if (!t) return NULL;
    if (tensor_fill_value(t, 0.0) != AX_OK) { ax_tensor_destroy(t); return NULL; }
    return t;
}

ax_tensor_t *ax_tensor_ones(const int64_t *shape, int ndim, ax_dtype_t dtype) {
    return ax_tensor_full(shape, ndim, dtype, 1.0);
}

ax_tensor_t *ax_tensor_full(const int64_t *shape, int ndim, ax_dtype_t dtype, double value) {
    ax_tensor_t *t = ax_tensor_create(shape, ndim, dtype);
    if (!t) return NULL;
    if (tensor_fill_value(t, value) != AX_OK) { ax_tensor_destroy(t); return NULL; }
    return t;
}

ax_tensor_t *ax_tensor_from_array(const void *data, const int64_t *shape, int ndim, ax_dtype_t dtype) {
    if (!data) return NULL;
    ax_tensor_t *t = ax_tensor_create(shape, ndim, dtype);
    if (!t) return NULL;

    int64_t n = compute_numel(shape, ndim);
    if (n < 0) { ax_tensor_destroy(t); return NULL; }
    size_t bytes = (size_t)n * ax_dtype_size(dtype);
    if (tensor_write_from_host(t, data, bytes) != AX_OK) {
        ax_tensor_destroy(t);
        return NULL;
    }
    return t;
}

ax_tensor_t *ax_tensor_arange(int64_t start, int64_t end, ax_dtype_t dtype) {
    int64_t len = end - start;
    if (len <= 0) return NULL;

    ax_tensor_t *t = ax_tensor_create(&len, 1, dtype);
    if (!t) return NULL;

    /* build sequential values into a host scratch buffer and commit
       via tensor_write_from_host so this works for any device. */
    size_t elem = ax_dtype_size(dtype);
    void *scratch = malloc((size_t)len * elem);
    if (!scratch) { ax_tensor_destroy(t); return NULL; }

    if (dtype == AX_FLOAT32) {
        float *d = (float *)scratch;
        for (int64_t i = 0; i < len; i++) d[i] = (float)(start + i);
    } else if (dtype == AX_INT32) {
        int32_t *d = (int32_t *)scratch;
        for (int64_t i = 0; i < len; i++) d[i] = (int32_t)(start + i);
    } else if (dtype == AX_FLOAT64) {
        double *d = (double *)scratch;
        for (int64_t i = 0; i < len; i++) d[i] = (double)(start + i);
    } else if (dtype == AX_INT64) {
        int64_t *d = (int64_t *)scratch;
        for (int64_t i = 0; i < len; i++) d[i] = start + i;
    }

    ax_status_t st = tensor_write_from_host(t, scratch, (size_t)len * elem);
    free(scratch);
    if (st != AX_OK) { ax_tensor_destroy(t); return NULL; }
    return t;
}

ax_tensor_t *ax_tensor_rand(const int64_t *shape, int ndim, float low, float high) {
    ax_tensor_t *t = ax_tensor_create(shape, ndim, AX_FLOAT32);
    if (!t) return NULL;

    int64_t n = compute_numel(shape, ndim);
    if (n < 0) { ax_tensor_destroy(t); return NULL; }

    if (t->storage->device == AX_DEVICE_CPU) {
        /* fast path: rng straight into device-visible buffer */
        float *d = (float *)t->storage->data;
        for (int64_t i = 0; i < n; i++) d[i] = ax_rng_uniform(low, high);
        return t;
    }

    /* non-cpu: rng into a host scratch buffer, then h2d transfer */
    float *scratch = (float *)malloc((size_t)n * sizeof(float));
    if (!scratch) { ax_tensor_destroy(t); return NULL; }
    for (int64_t i = 0; i < n; i++) scratch[i] = ax_rng_uniform(low, high);
    ax_status_t st = tensor_write_from_host(t, scratch, (size_t)n * sizeof(float));
    free(scratch);
    if (st != AX_OK) { ax_tensor_destroy(t); return NULL; }
    return t;
}

ax_tensor_t *ax_tensor_scalar(float value) {
    int64_t shape[] = {1};
    ax_tensor_t *t = ax_tensor_create(shape, 1, AX_FLOAT32);
    if (!t) return NULL;
    if (tensor_write_from_host(t, &value, sizeof(float)) != AX_OK) {
        ax_tensor_destroy(t);
        return NULL;
    }
    return t;
}

/* shared helper for arena-allocated tensors. callers choose whether
   the data buffer is zero-initialised (zeros=true) or left uninitialised
   (zeros=false). uninitialised is safe when the caller will immediately
   overwrite the entire buffer via a compute op (gemm, copy, etc.). */
static ax_tensor_t *tensor_arena_alloc_impl(struct ax_arena *arena,
        const int64_t *shape, int ndim, ax_dtype_t dtype, bool zeros)
{
    if (!arena) return zeros ? ax_tensor_zeros(shape, ndim, dtype)
                             : ax_tensor_create(shape, ndim, dtype);
    if (ndim < 0 || ndim > AX_MAX_DIMS) return NULL;
    if ((int)dtype < 0 || dtype >= AX_DTYPE_COUNT) return NULL;

    int64_t n = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) return NULL;
        if (!safe_mul_i64(n, shape[i], &n)) return NULL;
    }
    size_t elem_size = ax_dtype_size(dtype);
    if (elem_size > 0 && (size_t)n > SIZE_MAX / elem_size) return NULL;
    size_t bytes = (size_t)n * elem_size;
    if (bytes == 0) bytes = elem_size;

    ax_tensor_t *t = (ax_tensor_t *)ax_arena_alloc(arena, sizeof(ax_tensor_t), alignof(ax_tensor_t));
    ax_storage_t *s = t ? (ax_storage_t *)ax_arena_alloc(arena, sizeof(ax_storage_t), alignof(ax_storage_t)) : NULL;
    void *data = s ? ax_arena_alloc(arena, bytes, AX_DEFAULT_ALIGNMENT) : NULL;
    if (!t || !s || !data) {
        return zeros ? ax_tensor_zeros(shape, ndim, dtype)
                     : ax_tensor_create(shape, ndim, dtype);
    }

    memset(t, 0, sizeof(*t));
    t->ndim = ndim;
    t->dtype = dtype;
    for (int i = 0; i < ndim; i++) t->shape[i] = shape[i];
    compute_strides(shape, ndim, t->strides);

    s->data = data;
    s->size_bytes = bytes;
    atomic_init(&s->refcount, 1);
    s->device = AX_DEVICE_CPU;
    s->is_arena_temp = true;
    s->generation = 1;
    if (zeros) memset(data, 0, bytes);

    t->storage = s;
    return t;
}

/* arena-allocated zero tensor — see tensor_arena_alloc_impl. */
ax_tensor_t *ax_tensor_arena_zeros(struct ax_arena *arena, const int64_t *shape, int ndim, ax_dtype_t dtype) {
    return tensor_arena_alloc_impl(arena, shape, ndim, dtype, true);
}

/* arena-allocated uninitialised tensor — use when the caller will
   immediately overwrite the entire data buffer (gemm output, copy
   target, etc.). skips the memset, saving one full memory pass per
   allocation. for a 256×1024 matmul grad that's 1 MB of memset
   avoided per backward call. */
ax_tensor_t *ax_tensor_arena_create(struct ax_arena *arena, const int64_t *shape, int ndim, ax_dtype_t dtype) {
    return tensor_arena_alloc_impl(arena, shape, ndim, dtype, false);
}

/* destruction */

void ax_tensor_destroy(ax_tensor_t *t) {
    if (!t) return;
    /* arena-temp tensors live in a bump arena — caller resets the arena to free
       them in bulk. skip everything (including slab return, which would corrupt
       arena memory). */
    if (t->storage && t->storage->is_arena_temp) return;
    if (t->grad) {
        ax_tensor_destroy(t->grad);
        t->grad = NULL;
    }
    if (t->grad_fn) {
        ax_grad_fn_t *gf = (ax_grad_fn_t *)t->grad_fn;
        /* free owned saved tensors (helpers created during forward
           that aren't part of the graph and nobody else references) */
        for (int i = 0; i < gf->n_saved; i++)
        {
            if (gf->saved_owned[i] && gf->saved[i])
            {
                ax_tensor_destroy(gf->saved[i]);
                gf->saved[i] = NULL;
            }
        }
        /* free ctx if backward didn't already clean it up.
           only call ctx_cleanup if one was set — if none was set,
           ctx might point to a layer struct or other non-owned memory.
           backward functions that malloc their own ctx MUST set ctx_cleanup
           or null out self->ctx after freeing. */
        if (gf->ctx && gf->ctx_cleanup)
        {
            gf->ctx_cleanup(gf->ctx);
            gf->ctx = NULL;
        }
        ax_grad_fn_destroy(gf);
        t->grad_fn = NULL;
    }
    ax_storage_release(t->storage);
    slab_tensor_free(t);
}

/* shape queries */

int64_t ax_tensor_numel(const ax_tensor_t *t) {
    if (!t) return 0;
    int64_t n = compute_numel(t->shape, t->ndim);
    return n < 0 ? 0 : n;
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
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "reshape: %" PRId64 " elements vs %" PRId64 " elements", old_n, new_n);
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
    memcpy(r->strides, t->strides, sizeof(int64_t) * (size_t)t->ndim);

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
    memcpy(r->strides, new_strides, sizeof(int64_t) * (size_t)new_ndim);
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
    memcpy(r->strides, new_strides, sizeof(int64_t) * (size_t)new_ndim);
    return r;
}

/* element access */

float ax_tensor_get_f32(const ax_tensor_t *t, const int64_t *indices) {
    for (int i = 0; i < t->ndim; i++) {
        if (indices[i] < 0 || indices[i] >= t->shape[i]) {
            ax_err_set(AX_ERR_OUT_OF_BOUNDS,
                       "get_f32: index %" PRId64 " out of bounds for dim %d (size %" PRId64 ")",
                       indices[i], i, t->shape[i]);
            return 0.0f;
        }
    }
    size_t offset = t->offset;
    for (int i = 0; i < t->ndim; i++) {
        offset += indices[i] * t->strides[i];
    }
    /* non-cpu: one-element d2h round trip. expensive for bulk access;
       callers that care about throughput should ax_tensor_to_cpu() the
       whole thing first. for diagnostics this is fine. */
    if (t->storage->device != AX_DEVICE_CPU) {
        const ax_backend_ops_t *ops = ax_backend_for_device(t->storage->device);
        if (!ops || !ops->memcpy_d2h) return 0.0f;
        float v = 0.0f;
        const void *src = (const char *)t->storage->data + offset * sizeof(float);
        ops->memcpy_d2h(&v, src, sizeof(float));
        return v;
    }
    return ((float *)t->storage->data)[offset];
}

void ax_tensor_set_f32(ax_tensor_t *t, const int64_t *indices, float value) {
    for (int i = 0; i < t->ndim; i++) {
        if (indices[i] < 0 || indices[i] >= t->shape[i]) {
            ax_err_set(AX_ERR_OUT_OF_BOUNDS,
                       "set_f32: index %" PRId64 " out of bounds for dim %d (size %" PRId64 ")",
                       indices[i], i, t->shape[i]);
            return;
        }
    }
    size_t offset = t->offset;
    for (int i = 0; i < t->ndim; i++) {
        offset += indices[i] * t->strides[i];
    }
    /* non-cpu: one-element h2d round trip. same cost caveat as get. */
    if (t->storage->device != AX_DEVICE_CPU) {
        const ax_backend_ops_t *ops = ax_backend_for_device(t->storage->device);
        if (!ops || !ops->memcpy_h2d) return;
        void *dst = (char *)t->storage->data + offset * sizeof(float);
        ops->memcpy_h2d(dst, &value, sizeof(float));
        return;
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
    memcpy(v->strides, t->strides, sizeof(int64_t) * (size_t)t->ndim);
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

/* device transfers.
   both functions route through the owning backend's vtable. core has no
   idea how the device memory is actually allocated or copied. */

ax_tensor_t *ax_tensor_to_cuda(ax_tensor_t *t) {
    if (!t) return NULL;
    if (t->storage->device == AX_DEVICE_CUDA)
        return ax_tensor_view(t);

    const ax_backend_ops_t *ops = ax_backend_for_device(AX_DEVICE_CUDA);
    if (!ops || !ops->storage_alloc || !ops->memcpy_h2d) {
        ax_err_set(AX_ERR_BACKEND, "cuda backend not available");
        return NULL;
    }

    ax_tensor_t *src = ax_ensure_contiguous(t);
    if (!src) return NULL;

    int64_t n = ax_tensor_numel(src);
    size_t bytes = (size_t)n * ax_dtype_size(src->dtype);

    ax_tensor_t *gpu = tensor_alloc_meta(src->shape, src->ndim, src->dtype);
    if (!gpu) { if (src != t) ax_tensor_destroy(src); return NULL; }

    gpu->storage = ax_storage_create(bytes, AX_DEVICE_CUDA);
    if (!gpu->storage) {
        slab_tensor_free(gpu);
        if (src != t) ax_tensor_destroy(src);
        return NULL;
    }

    /* source is cpu-contiguous here, so a single h2d memcpy suffices */
    const char *src_bytes = (const char *)src->storage->data
                          + src->offset * ax_dtype_size(src->dtype);
    if (ops->memcpy_h2d(gpu->storage->data, src_bytes, bytes) != AX_OK) {
        ax_tensor_destroy(gpu);
        if (src != t) ax_tensor_destroy(src);
        return NULL;
    }

    if (src != t) ax_tensor_destroy(src);
    return gpu;
}

ax_tensor_t *ax_tensor_to_cpu(ax_tensor_t *t) {
    if (!t) return NULL;
    if (t->storage->device == AX_DEVICE_CPU)
        return ax_tensor_view(t);

    const ax_backend_ops_t *ops = ax_backend_for_device(t->storage->device);
    if (!ops || !ops->memcpy_d2h) {
        ax_err_set(AX_ERR_BACKEND,
                   "no backend registered for source device %d",
                   (int)t->storage->device);
        return NULL;
    }

    int64_t n = ax_tensor_numel(t);
    size_t bytes = (size_t)n * ax_dtype_size(t->dtype);

    ax_tensor_t *cpu = tensor_alloc_meta(t->shape, t->ndim, t->dtype);
    if (!cpu) return NULL;

    cpu->storage = ax_storage_create(bytes, AX_DEVICE_CPU);
    if (!cpu->storage) { slab_tensor_free(cpu); return NULL; }

    const char *src_bytes = (const char *)t->storage->data
                          + t->offset * ax_dtype_size(t->dtype);
    if (ops->memcpy_d2h(cpu->storage->data, src_bytes, bytes) != AX_OK) {
        ax_tensor_destroy(cpu);
        return NULL;
    }
    return cpu;
}

/* printing */

void ax_tensor_print_shape(const ax_tensor_t *t) {
    if (!t) { printf("(null)\n"); return; }
    printf("tensor(shape=[");
    for (int i = 0; i < t->ndim; i++) {
        printf("%" PRId64 "%s", t->shape[i], i < t->ndim - 1 ? ", " : "");
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

    /* for non-cpu tensors, mirror the whole thing to cpu first so the
       per-element reads below stay free. one bulk d2h beats 20 tiny
       ones and prints are cold-path anyway. */
    ax_tensor_t *view = NULL;
    const ax_tensor_t *readable = t;
    if (t->storage->device != AX_DEVICE_CPU) {
        view = ax_tensor_to_cpu((ax_tensor_t *)t);
        if (!view) { printf("  (failed to copy to cpu)\n"); return; }
        readable = view;
    }

    int64_t n = ax_tensor_numel(readable);
    int64_t limit = n > 20 ? 20 : n; /* print at most 20 elements */

    printf("  [");
    for (int64_t i = 0; i < limit; i++) {
        /* compute nd indices from flat index */
        int64_t remaining = i;
        int64_t indices[AX_MAX_DIMS];
        for (int d = readable->ndim - 1; d >= 0; d--) {
            indices[d] = remaining % readable->shape[d];
            remaining /= readable->shape[d];
        }
        float v = ax_tensor_get_f32(readable, indices);
        printf("%.4f%s", v, i < limit - 1 ? ", " : "");
    }
    if (n > limit) printf(", ... (%" PRId64 " more)", n - limit);
    printf("]\n");

    if (view) ax_tensor_destroy(view);
}
