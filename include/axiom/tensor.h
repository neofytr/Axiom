/* axiom/tensor.h — n-dimensional tensor with refcounted storage */

#ifndef AX_TENSOR_H
#define AX_TENSOR_H

#include "types.h"
#include "error.h"

/* ref-counted data buffer shared across tensor views */
typedef struct {
    void *data;            /* raw data pointer (aligned) */
    size_t size_bytes;     /* total allocated bytes */
    int refcount;          /* reference count; freed when it hits 0 */
    ax_device_t device;    /* where this memory lives */
} ax_storage_t;

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

/* create a scalar tensor (0-dim) */
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

/* printing */

/* print tensor shape and a few elements for debugging */
void ax_tensor_print(const ax_tensor_t *t);

/* print just the shape */
void ax_tensor_print_shape(const ax_tensor_t *t);

#endif /* AX_TENSOR_H */
