# Unit 1: Vectors, Matrices, and Tensors

## Why This Matters

Every piece of data in a neural network — inputs, weights, outputs, gradients — is
stored in a tensor. Understanding tensors is understanding the language neural networks
speak. This unit covers the math and then shows how Axiom implements it in C.


## 1.1 Scalars, Vectors, Matrices

A **scalar** is a single number: `3.14`, `-7`, `0.001`.

A **vector** is an ordered list of scalars. A vector with n elements lives in R^n:

    v = [v_1, v_2, ..., v_n]

A **matrix** is a 2D grid of scalars with m rows and n columns:

    M = | m_11  m_12  ...  m_1n |
        | m_21  m_22  ...  m_2n |
        |  ...                   |
        | m_m1  m_m2  ...  m_mn |


## 1.2 Tensors: The Generalization

A **tensor** is the generalization to any number of dimensions:

- 0D tensor: scalar (shape `[]` or `[1]`)
- 1D tensor: vector (shape `[n]`)
- 2D tensor: matrix (shape `[m, n]`)
- 3D tensor: a "cube" of numbers (shape `[d1, d2, d3]`)
- 4D tensor: common for image batches (shape `[N, C, H, W]`)

The number of dimensions is called the **rank** or **ndim**. The size along each
dimension is the **shape**. The total number of elements is the **numel** (number of
elements), equal to the product of all shape dimensions.


## 1.3 Memory Layout

A tensor's data lives in a flat, contiguous block of memory. To find element
`[i, j, k]` in a 3D tensor, you need **strides**: the number of elements you skip
when you increment an index by 1 along a given dimension.

For a tensor with shape `[d0, d1, d2]` in row-major (C-contiguous) layout:

    stride[2] = 1
    stride[1] = d2
    stride[0] = d1 * d2

The physical offset of element `[i, j, k]` is:

    offset = i * stride[0] + j * stride[1] + k * stride[2]

This is how Axiom computes element access in `ax_tensor_get_f32`:

```c
size_t offset = t->offset;
for (int i = 0; i < t->ndim; i++) {
    offset += indices[i] * t->strides[i];
}
return ((float *)t->storage->data)[offset];
```


## 1.4 Axiom's Tensor Structure

In Axiom (see `include/axiom/tensor.h`), a tensor is:

```c
typedef struct ax_tensor {
    ax_storage_t *storage;             // shared data buffer
    int64_t shape[AX_MAX_DIMS];        // size of each dimension
    int64_t strides[AX_MAX_DIMS];      // element stride per dimension
    int ndim;                          // number of dimensions
    ax_dtype_t dtype;                  // element type (float32, int32, etc.)
    size_t offset;                     // element offset into storage

    bool requires_grad;                // autograd tracking
    struct ax_tensor *grad;            // gradient tensor
    void *grad_fn;                     // backward function
} ax_tensor_t;
```

Key design decisions:

- **Fixed-size arrays** (`AX_MAX_DIMS = 8`): no heap allocation for shape/strides.
  This is critical for embedded targets where malloc is expensive or unavailable.
- **Refcounted storage**: multiple tensors can share the same data buffer. Views,
  reshapes, and transposes are zero-copy when possible.
- **Offset field**: enables slicing without copying data. A view into row 5 of a
  matrix just sets `offset = 5 * stride[0]`.


## 1.5 Storage and Reference Counting

The actual data buffer is wrapped in `ax_storage_t`:

```c
typedef struct {
    void *data;            // raw data pointer (64-byte aligned)
    size_t size_bytes;     // total allocated bytes
    int refcount;          // freed when it hits 0
    ax_device_t device;    // CPU or CUDA
} ax_storage_t;
```

When you create a view or reshape, the new tensor increments the storage refcount.
When a tensor is destroyed, it decrements the refcount. The data is freed only when
the last reference is released. This is manual reference counting — simple and
deterministic, no garbage collector needed.


## 1.6 Supported Data Types

Axiom supports six dtypes (see `types.h`):

| Enum         | C Type    | Bytes |
|-------------|-----------|-------|
| AX_FLOAT32  | float     | 4     |
| AX_FLOAT64  | double    | 8     |
| AX_INT32    | int32_t   | 4     |
| AX_INT64    | int64_t   | 8     |
| AX_UINT8    | uint8_t   | 1     |
| AX_BOOL     | uint8_t   | 1     |

Most neural network computation uses AX_FLOAT32. The others exist for data loading
(INT32 labels), indexing (INT64), and quantized inference (UINT8, future).


## 1.7 Key Operations

**Creating tensors:**

```c
ax_tensor_t *a = ax_tensor_zeros(shape, ndim, AX_FLOAT32);   // all zeros
ax_tensor_t *b = ax_tensor_ones(shape, ndim, AX_FLOAT32);    // all ones
ax_tensor_t *c = ax_tensor_rand(shape, ndim, 0.0f, 1.0f);   // uniform random
ax_tensor_t *d = ax_tensor_from_array(data, shape, ndim, AX_FLOAT32);  // from C array
```

**Shape manipulation (zero-copy when contiguous):**

```c
ax_tensor_t *r = ax_tensor_reshape(t, new_shape, new_ndim);  // share storage
ax_tensor_t *tr = ax_tensor_transpose(t, 0, 1);              // swap dims
ax_tensor_t *sq = ax_tensor_squeeze(t, dim);                  // remove size-1 dim
ax_tensor_t *us = ax_tensor_unsqueeze(t, dim);                // insert size-1 dim
```

**Contiguity check:**

A tensor is contiguous if its strides match the expected row-major layout. Transpose
breaks contiguity (it swaps strides but not data). `ax_tensor_contiguous()` makes a
contiguous copy when needed.


## 1.8 Matrix Multiplication — The Core of Neural Networks

Almost everything in a neural network reduces to matrix multiplication:

    C = A @ B

Where A is `[m, k]`, B is `[k, n]`, and C is `[m, n]`. Each element:

    C[i, j] = sum over p of A[i, p] * B[p, j]

This is O(m * n * k) — cubic complexity. Making this fast (BLAS, SIMD, tiling)
is the single most important optimization in deep learning.

In Axiom, `ax_matmul()` handles this and hooks into the autograd system so gradients
flow back through it automatically.


## 1.9 Integer Overflow Safety

Computing `numel` (product of all shape dimensions) can overflow. Axiom uses
checked multiplication everywhere:

```c
static bool safe_mul_i64(int64_t a, int64_t b, int64_t *result) {
    if (a > 0 && b > 0 && a > INT64_MAX / b) return false;
    *result = a * b;
    return true;
}
```

`compute_numel()` returns -1 on overflow, and `ax_tensor_create()` checks that
`numel * elem_size` doesn't overflow `size_t` before allocating. Without this,
a malicious shape like `[INT64_MAX, 2]` would silently wrap around and allocate
a tiny buffer, causing a heap overflow on writes.


## Key Takeaways

1. A tensor is a multidimensional array with shape, strides, and a shared data buffer.
2. Strides enable zero-copy views, reshapes, and transposes.
3. Reference-counted storage prevents data leaks and unnecessary copies.
4. All shape arithmetic uses overflow-checked multiplication.
5. Everything in deep learning is ultimately tensor operations — especially matmul.
