/* axiom/tensor.h — n-dimensional tensor with refcounted storage.

   ownership pattern: every ax_tensor_* function that returns ax_tensor_t*
   produces a heap tensor owned by the caller; release with
   ax_tensor_destroy(). zero-copy view constructors (reshape/transpose/
   squeeze/unsqueeze/view) return a NEW heap tensor that shares storage
   with the parent — destroying the view does not free the underlying
   buffer (ref-counted via ax_storage_t).

   thread safety: ax_storage_t refcount is atomic; ax_storage_retain/release
   are concurrent-safe. all other tensor mutations (set, reshape, etc.) on
   the SAME tensor object are not concurrent-safe. independent tensors
   may be operated on concurrently as long as they share no storage.

   failure mode: constructors return NULL on allocation failure;
   ax_err_last_message() describes the cause. */


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
    int64_t offset;                    /* element offset into storage */

    /* autograd fields — used in phase 2, reserved now */
    bool requires_grad;
    struct ax_tensor *grad;
    void *grad_fn;                     /* opaque pointer to grad function */

    /* layout hint for 4D image tensors. defaults to AX_LAYOUT_NCHW (=0) for
       backward compat — calloc/memset(0) leaves it correct. layers that care
       about layout (Conv2d, Pool2d, BatchNorm2d) inspect this to dispatch
       to the right kernel. ignored for ndim != 4. */
    ax_layout_t layout;
} ax_tensor_t;

#ifdef __cplusplus
extern "C" {
#endif

/* storage management. storage is ref-counted with atomic operations and
   shared between views. callers usually let the tensor lifecycle drive
   storage release; direct retain/release is only needed for advanced
   ownership patterns (e.g. handing storage to a non-axiom subsystem). */

/* allocate a fresh storage of `size_bytes` bytes on `device`. refcount = 1.
   ownership: caller; release with ax_storage_release.
   returns NULL on allocation failure or unknown device. */
ax_storage_t *ax_storage_create(size_t size_bytes, ax_device_t device);

/* atomically increment refcount. NULL is a safe no-op. */
void ax_storage_retain(ax_storage_t *s);

/* atomically decrement refcount; frees the buffer when it reaches zero.
   NULL is a safe no-op. */
void ax_storage_release(ax_storage_t *s);

/* tensor creation. all constructors below allocate a heap tensor owned
   by the caller; release with ax_tensor_destroy. they return NULL on
   allocation failure with ax_err_last_message() set. */

/* create with given shape and dtype; storage is uninitialised. */
ax_tensor_t *ax_tensor_create(const int64_t *shape, int ndim, ax_dtype_t dtype);

/* create and fill with zeros. */
ax_tensor_t *ax_tensor_zeros(const int64_t *shape, int ndim, ax_dtype_t dtype);

/* arena-allocated zero tensor: metadata, storage struct, AND data buffer all bumped
   from the supplied arena. extremely fast (no malloc, no slab, no atomics).
   the returned tensor is invalidated when the arena is reset. ax_tensor_destroy
   on it is a safe no-op. intended for short-lived backward-pass scratch tensors.
   never store the returned pointer in grad_fn->saved[] — it becomes stale after
   ax_arena_reset. on arena exhaustion, falls back to ax_tensor_zeros. */
struct ax_arena;
ax_tensor_t *ax_tensor_arena_zeros(struct ax_arena *arena, const int64_t *shape, int ndim, ax_dtype_t dtype);

/* arena-allocated uninitialised tensor — same lifecycle as arena_zeros but
   skips the memset. use when the caller will immediately overwrite the
   entire buffer via a compute op (gemm, copy). saves one full memory pass. */
ax_tensor_t *ax_tensor_arena_create(struct ax_arena *arena, const int64_t *shape, int ndim, ax_dtype_t dtype);

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

/* release the tensor metadata; storage is released too if its refcount
   hits zero (otherwise other views keep it alive). NULL is a safe no-op.
   thread-safety: concurrent-safe across distinct tensor pointers; one
   caller per tensor. */
void ax_tensor_destroy(ax_tensor_t *t);

/* shape queries — read-only and concurrent-safe on the same tensor. */

/* total number of elements (product of shape). returns 0 for NULL t. */
int64_t ax_tensor_numel(const ax_tensor_t *t);

/* true if the data is contiguous in memory (row-major, no holes).
   false for NULL t or non-contiguous strides. */
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

/* element access. slow path — favour vectorised compute ops in tight loops.
   bounds checking is debug-only (NDEBUG removes it). */

/* read a single f32 element. returns 0.0f for NULL t or out-of-bounds
   indices in debug builds (debug build aborts via AX_BOUNDS_CHECK).
   thread-safety: concurrent-safe on the same tensor as long as no
   writer is active. */
float ax_tensor_get_f32(const ax_tensor_t *t, const int64_t *indices);

/* write a single f32 element. NULL t is a no-op. mutates storage in
   place; bumps storage->generation for cache invalidation.
   thread-safety: not concurrent-safe with other writers or readers
   on the same tensor. */
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

/* layout management (only meaningful for 4D image tensors) */

/* get the layout hint for a 4D image tensor. returns AX_LAYOUT_NCHW for
   non-4D tensors (default / no meaningful layout). */
static inline ax_layout_t ax_tensor_get_layout(const ax_tensor_t *t) {
    return t ? t->layout : AX_LAYOUT_NCHW;
}

/* tag a tensor with a layout hint. does NOT transpose data — only updates
   the metadata. caller must ensure the underlying memory order matches the
   declared layout (i.e. for AX_LAYOUT_NHWC: shape must be [N, H, W, C] with
   contiguous channels-last memory). use ax_tensor_to_nhwc / to_nchw for
   the data-moving conversion. */
static inline void ax_tensor_set_layout(ax_tensor_t *t, ax_layout_t layout) {
    if (t) t->layout = layout;
}

/* convert tensor to NHWC layout (transpose if currently NCHW; no-op + retain
   if already NHWC). returns a NEW tensor; caller frees both. always
   contiguous in the new layout. ndim must be 4 — returns NULL otherwise. */
ax_tensor_t *ax_tensor_to_nhwc(ax_tensor_t *t);

/* convert tensor to NCHW layout (transpose if NHWC; no-op + retain if
   already NCHW). returns a NEW tensor; caller frees both. ndim must be 4. */
ax_tensor_t *ax_tensor_to_nchw(ax_tensor_t *t);

/* device management. process-wide setting; takes effect for any tensor
   constructor called after the change.
   thread-safety: not concurrent-safe with parallel tensor construction;
   set during single-threaded initialisation, then leave alone. */

/* set the default device used by every subsequent tensor constructor.
   pair with ax_compute_set_backend() to route both allocation and
   compute to the same device. */
void ax_set_default_device(ax_device_t dev);

/* return the currently selected default device. */
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
