import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/docs/api/tensor')({
  component: TensorApiPage,
})

function TensorApiPage() {
  return (
    <>
      <h1>Tensor API</h1>
      <p>
        Tensors are the fundamental data type. All operations accept and return tensors.
        Header: <code>axiom/tensor.h</code>
      </p>

      <h2>Creation</h2>

      <h3>ax_tensor_create</h3>
      <pre><code className="language-c">{`ax_tensor_t *ax_tensor_create(const int64_t *shape, int ndim, ax_dtype_t dtype);`}</code></pre>
      <p>
        Creates a tensor with the given shape and dtype. Memory is allocated but <strong>not
        initialized</strong>. Use this when you're about to immediately overwrite the contents
        (e.g., before a compute op fills it).
      </p>

      <h3>ax_tensor_zeros</h3>
      <pre><code className="language-c">{`ax_tensor_t *ax_tensor_zeros(const int64_t *shape, int ndim, ax_dtype_t dtype);`}</code></pre>
      <p>Creates a tensor filled with zeros.</p>

      <h3>ax_tensor_ones</h3>
      <pre><code className="language-c">{`ax_tensor_t *ax_tensor_ones(const int64_t *shape, int ndim, ax_dtype_t dtype);`}</code></pre>
      <p>Creates a tensor filled with ones.</p>

      <h3>ax_tensor_full</h3>
      <pre><code className="language-c">{`ax_tensor_t *ax_tensor_full(const int64_t *shape, int ndim, ax_dtype_t dtype, double value);`}</code></pre>
      <p>Creates a tensor filled with a constant value.</p>

      <h3>ax_tensor_from_array</h3>
      <pre><code className="language-c">{`ax_tensor_t *ax_tensor_from_array(const void *data, const int64_t *shape,
                                  int ndim, ax_dtype_t dtype);`}</code></pre>
      <p>
        Creates a tensor from existing data. <strong>Copies</strong> the data into a new storage
        buffer. The source array must contain at least <code>product(shape) * dtype_size</code> bytes.
      </p>

      <h3>ax_tensor_rand</h3>
      <pre><code className="language-c">{`ax_tensor_t *ax_tensor_rand(const int64_t *shape, int ndim, float low, float high);`}</code></pre>
      <p>Creates a tensor filled with uniform random values in [low, high).</p>

      <h3>ax_tensor_arange</h3>
      <pre><code className="language-c">{`ax_tensor_t *ax_tensor_arange(int64_t start, int64_t end, ax_dtype_t dtype);`}</code></pre>
      <p>Creates a 1D tensor with values from start to end (exclusive), step 1.</p>

      <h3>ax_tensor_scalar</h3>
      <pre><code className="language-c">{`ax_tensor_t *ax_tensor_scalar(float value);`}</code></pre>
      <p>Creates a scalar tensor (shape [1], 1-dim). Not truly 0-dim since many ops assume ndim &gt;= 1.</p>

      <h3>ax_set_seed</h3>
      <pre><code className="language-c">{`void ax_set_seed(unsigned int seed);`}</code></pre>
      <p>Seeds the global RNG used by <code>ax_tensor_rand</code> and weight initializers.</p>

      <h2>Arena allocation</h2>
      <p>
        These variants allocate from a bump arena instead of the heap. Extremely fast (pointer
        bump, no malloc). The returned tensor is invalidated when the arena is reset.{' '}
        <code>ax_tensor_destroy</code> on an arena tensor is a safe no-op.
      </p>

      <h3>ax_tensor_arena_zeros</h3>
      <pre><code className="language-c">{`ax_tensor_t *ax_tensor_arena_zeros(ax_arena_t *arena, const int64_t *shape,
                                    int ndim, ax_dtype_t dtype);`}</code></pre>

      <h3>ax_tensor_arena_create</h3>
      <pre><code className="language-c">{`ax_tensor_t *ax_tensor_arena_create(ax_arena_t *arena, const int64_t *shape,
                                     int ndim, ax_dtype_t dtype);`}</code></pre>
      <p>Same as arena_zeros but skips the memset. Use when you'll immediately overwrite the buffer.</p>

      <h2>Destruction</h2>

      <h3>ax_tensor_destroy</h3>
      <pre><code className="language-c">{`void ax_tensor_destroy(ax_tensor_t *t);`}</code></pre>
      <p>
        Releases the storage (decrements refcount; frees if it hits 0) and frees the tensor struct.
        Safe on NULL. Safe on arena-allocated tensors (no-op for the storage release).
      </p>

      <h2>Shape queries</h2>

      <h3>ax_tensor_numel</h3>
      <pre><code className="language-c">{`int64_t ax_tensor_numel(const ax_tensor_t *t);`}</code></pre>
      <p>Total number of elements (product of all dimensions).</p>

      <h3>ax_tensor_is_contiguous</h3>
      <pre><code className="language-c">{`bool ax_tensor_is_contiguous(const ax_tensor_t *t);`}</code></pre>
      <p>
        Returns true if the tensor's strides match the row-major (C-contiguous) pattern.
        The optimized backend's SIMD paths require contiguous tensors.
      </p>

      <h2>Shape manipulation</h2>
      <p>These are zero-copy where possible (they share storage with the input).</p>

      <h3>ax_tensor_reshape</h3>
      <pre><code className="language-c">{`ax_tensor_t *ax_tensor_reshape(ax_tensor_t *t, const int64_t *new_shape, int new_ndim);`}</code></pre>
      <p>
        Reshapes to the new shape. If the tensor is contiguous, returns a view sharing the same
        storage. If not contiguous, copies the data first. The total number of elements must match.
      </p>

      <h3>ax_tensor_transpose</h3>
      <pre><code className="language-c">{`ax_tensor_t *ax_tensor_transpose(ax_tensor_t *t, int dim0, int dim1);`}</code></pre>
      <p>
        Swaps two dimensions by swapping their strides and shape entries. Zero-copy. The result
        is generally not contiguous.
      </p>

      <h3>ax_tensor_squeeze</h3>
      <pre><code className="language-c">{`ax_tensor_t *ax_tensor_squeeze(ax_tensor_t *t, int dim);`}</code></pre>
      <p>Removes a dimension of size 1 at the given position. Zero-copy.</p>

      <h3>ax_tensor_unsqueeze</h3>
      <pre><code className="language-c">{`ax_tensor_t *ax_tensor_unsqueeze(ax_tensor_t *t, int dim);`}</code></pre>
      <p>Inserts a dimension of size 1 at the given position. Zero-copy.</p>

      <h2>Element access</h2>

      <h3>ax_tensor_get_f32 / ax_tensor_set_f32</h3>
      <pre><code className="language-c">{`float ax_tensor_get_f32(const ax_tensor_t *t, const int64_t *indices);
void ax_tensor_set_f32(ax_tensor_t *t, const int64_t *indices, float value);`}</code></pre>
      <p>
        Get/set a single float32 element using ndim indices. These respect strides and offset,
        so they work correctly on views and transposed tensors.
      </p>

      <h2>Views and copies</h2>

      <h3>ax_tensor_view</h3>
      <pre><code className="language-c">{`ax_tensor_t *ax_tensor_view(ax_tensor_t *t);`}</code></pre>
      <p>Creates a view that shares storage. The storage refcount is incremented.</p>

      <h3>ax_tensor_contiguous</h3>
      <pre><code className="language-c">{`ax_tensor_t *ax_tensor_contiguous(ax_tensor_t *t);`}</code></pre>
      <p>Makes a contiguous copy of the tensor (new storage, data is copied).</p>

      <h3>ax_ensure_contiguous</h3>
      <pre><code className="language-c">{`static inline ax_tensor_t *ax_ensure_contiguous(ax_tensor_t *t);`}</code></pre>
      <p>
        Returns the tensor unchanged if already contiguous (with offset 0), or makes a contiguous
        copy. Check if the return value differs from the input to know whether to free it.
      </p>

      <h2>Device management</h2>

      <pre><code className="language-c">{`void ax_set_default_device(ax_device_t dev);
ax_device_t ax_get_default_device(void);

ax_tensor_t *ax_tensor_to_cuda(ax_tensor_t *t);
ax_tensor_t *ax_tensor_to_cpu(ax_tensor_t *t);`}</code></pre>
      <p>
        <code>ax_tensor_to_cuda</code> moves a tensor to the GPU (no-op + retain if already there).
        Returns NULL if no CUDA backend is registered. <code>ax_tensor_to_cpu</code> does the reverse.
        Setting the default device causes all new tensor allocations to go to that device.
      </p>

      <h2>Printing</h2>

      <pre><code className="language-c">{`void ax_tensor_print(const ax_tensor_t *t);
void ax_tensor_print_shape(const ax_tensor_t *t);`}</code></pre>
      <p>Print tensor contents or just the shape for debugging.</p>

      <h2>Storage utilities</h2>

      <pre><code className="language-c">{`ax_storage_t *ax_storage_create(size_t size_bytes, ax_device_t device);
void ax_storage_retain(ax_storage_t *s);
void ax_storage_release(ax_storage_t *s);
static inline void ax_storage_touch(ax_storage_t *s);`}</code></pre>
      <p>
        You rarely need these directly. <code>ax_storage_touch</code> bumps the generation counter;
        call it after any in-place mutation to invalidate caches.
      </p>
    </>
  )
}
