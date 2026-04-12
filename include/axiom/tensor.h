/* axiom/tensor.h — n-dimensional tensor with refcounted storage */

#ifndef AX_TENSOR_H
#define AX_TENSOR_H

#include "types.h"
#include "error.h"
#ifndef __cplusplus
#include <stdatomic.h>
#endif

/* ref-counted data buffer shared across tensor views */
typedef struct {
    void *data;            /* raw data pointer (aligned) */
    size_t size_bytes;     /* total allocated bytes */
    atomic_int refcount;   /* atomic reference count; freed when it hits 0 */
    ax_device_t device;    /* where this memory lives */
    bool is_arena_temp;    /* true if allocated from a bump arena (release is no-op) */
    /* generation counter — bumped on every write to the buffer contents.
       read-only caches (pack_b in cpu_opt.c, ...) that key on a raw
       pointer use this to detect when the buffer has mutated in place
       (e.g. optimizer step rewrote weights without changing the pointer)
       and invalidate stale entries. starts at 1. mutate via
       ax_storage_touch(); direct manipulation is allowed but discouraged. */
    uint64_t generation;
} ax_storage_t;

/* mark storage as modified — bump its generation counter so any stale
   cache keyed on the previous value invalidates on next lookup. cheap
   (single increment); call at every in-place write site. safe on NULL. */
static inline void ax_storage_touch(ax_storage_t *s) {
    if (s) s->generation++;
}

/* the tensor: metadata wrapper around a storage buffer */
typedef struct ax_tensor {
    ax_storage_t *storage;             /* shared data buffer */
    int64_t shape[AX_MAX_DIMS];        /* size of each dimension */
    int64_t strides[AX_MAX_DIMS];      /* element stride per dimension */
    int ndim;                          /* number of dimensions */
    ax_dtype_t dtype;                  /* element type */
    size_t offset;                     /* element offset into storage */

    /* autograd fields — used in phase 2, reserved now */
    bool requires_grad;
    struct ax_tensor *grad;
    void *grad_fn;                     /* opaque pointer to grad function */
} ax_tensor_t;

#ifdef __cplusplus
extern "C" {
#endif

/* storage management */

/* create a new storage with the given byte count */
ax_storage_t *ax_storage_create(size_t size_bytes, ax_device_t device);

/* increment reference count */
void ax_storage_retain(ax_storage_t *s);

/* decrement reference count; frees if it hits 0 */
void ax_storage_release(ax_storage_t *s);

/* tensor creation */

/* create a tensor with the given shape and dtype; memory is uninitialized */
ax_tensor_t *ax_tensor_create(const int64_t *shape, int ndim, ax_dtype_t dtype);

/* create and fill with zeros */
ax_tensor_t *ax_tensor_zeros(const int64_t *shape, int ndim, ax_dtype_t dtype);

/* arena-allocated zero tensor: metadata, storage struct, AND data buffer all bumped
   from the supplied arena. extremely fast (no malloc, no slab, no atomics).
   the returned tensor is invalidated when the arena is reset. ax_tensor_destroy
   on it is a safe no-op. intended for short-lived backward-pass scratch tensors.
   never store the returned pointer in grad_fn->saved[] — it becomes stale after
   ax_arena_reset. on arena exhaustion, falls back to ax_tensor_zeros. */
struct ax_arena;
ax_tensor_t *ax_tensor_arena_zeros(struct ax_arena *arena, const int64_t *shape, int ndim, ax_dtype_t dtype);

/* create and fill with ones */
ax_tensor_t *ax_tensor_ones(const int64_t *shape, int ndim, ax_dtype_t dtype);

/* create and fill with a constant value */
ax_tensor_t *ax_tensor_full(const int64_t *shape, int ndim, ax_dtype_t dtype, double value);

/* create a tensor from existing data (copies the data) */
ax_tensor_t *ax_tensor_from_array(const void *data, const int64_t *shape, int ndim, ax_dtype_t dtype);

/* create a 1d tensor with values from start to end (exclusive), step 1 */
ax_tensor_t *ax_tensor_arange(int64_t start, int64_t end, ax_dtype_t dtype);

/* create a tensor filled with uniform random values in [low, high) */
ax_tensor_t *ax_tensor_rand(const int64_t *shape, int ndim, float low, float high);

/* seed the global RNG used by ax_tensor_rand and weight initializers.
   wraps ax_rng_seed() from rng.h for backward compatibility. */
void ax_set_seed(unsigned int seed);

/* create a scalar tensor (shape [1], 1-dim).
   not truly 0-dim since many ops assume ndim >= 1. */
ax_tensor_t *ax_tensor_scalar(float value);

/* tensor destruction */

/* release the tensor and its storage (if refcount drops to 0) */
void ax_tensor_destroy(ax_tensor_t *t);

/* shape queries */

/* total number of elements */
int64_t ax_tensor_numel(const ax_tensor_t *t);

/* check if tensor data is contiguous in memory */
bool ax_tensor_is_contiguous(const ax_tensor_t *t);

/* shape manipulation (zero-copy where possible) */

/* reshape to new shape; returns new tensor sharing storage if contiguous */
ax_tensor_t *ax_tensor_reshape(ax_tensor_t *t, const int64_t *new_shape, int new_ndim);

/* transpose two dimensions; zero-copy (swaps strides) */
ax_tensor_t *ax_tensor_transpose(ax_tensor_t *t, int dim0, int dim1);

/* remove dimensions of size 1 */
ax_tensor_t *ax_tensor_squeeze(ax_tensor_t *t, int dim);

/* insert a dimension of size 1 at the given position */
ax_tensor_t *ax_tensor_unsqueeze(ax_tensor_t *t, int dim);

/* element access */

/* get/set a single f32 element using ndim indices */
float ax_tensor_get_f32(const ax_tensor_t *t, const int64_t *indices);
void ax_tensor_set_f32(ax_tensor_t *t, const int64_t *indices, float value);

/* view creation */

/* create a view (shares storage, different shape/strides/offset) */
ax_tensor_t *ax_tensor_view(ax_tensor_t *t);

/* make a contiguous copy of the tensor */
ax_tensor_t *ax_tensor_contiguous(ax_tensor_t *t);

/* return the tensor if contiguous, or a contiguous copy. caller must
   check if the return value differs from input to know whether to free it. */
static inline ax_tensor_t *ax_ensure_contiguous(ax_tensor_t *t) {
    if (!t) return NULL;
    if (ax_tensor_is_contiguous(t) && t->offset == 0) return t;
    return ax_tensor_contiguous(t);
}

/* device management */

/* set/get the default device for all new tensor allocations.
   ax_set_default_device(AX_DEVICE_CUDA) combined with
   ax_compute_set_backend(AX_BACKEND_CUDA) routes all tensor creation
   and compute to the GPU with no changes to training code. */
void ax_set_default_device(ax_device_t dev);
ax_device_t ax_get_default_device(void);

/* move a tensor to the cuda device (nop + retain if already on gpu).
   returns NULL and sets AX_ERR_BACKEND if no backend is registered as
   the owner of AX_DEVICE_CUDA (e.g. cuda support not compiled in). */
ax_tensor_t *ax_tensor_to_cuda(ax_tensor_t *t);

/* move a tensor to CPU (nop + retain if already on CPU). */
ax_tensor_t *ax_tensor_to_cpu(ax_tensor_t *t);

/* printing */

/* print tensor shape and a few elements for debugging */
void ax_tensor_print(const ax_tensor_t *t);

/* print just the shape */
void ax_tensor_print_shape(const ax_tensor_t *t);

#ifdef __cplusplus
}
#endif

#endif /* AX_TENSOR_H */
