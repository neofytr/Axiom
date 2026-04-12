import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/docs/architecture/tensors')({
  component: TensorsPage,
})

function TensorsPage() {
  return (
    <>
      <h1>Tensors and Storage</h1>
      <p>
        The tensor is the fundamental data structure. Everything in Axiom operates on tensors:
        inputs, weights, gradients, loss values. Understanding how they work internally is
        important for writing efficient code and avoiding leaks.
      </p>

      <h2>The two-level design</h2>
      <p>
        A tensor is split into two parts: the <strong>metadata</strong> (shape, strides, dtype)
        and the <strong>storage</strong> (the actual data buffer). This separation enables
        zero-copy views, reshapes, and transposes.
      </p>

      <h3>ax_storage_t</h3>
      <pre><code className="language-c">{`typedef struct {
    void *data;            // raw data pointer (aligned)
    size_t size_bytes;     // total allocated bytes
    atomic_int refcount;   // freed when it hits 0
    ax_device_t device;    // AX_DEVICE_CPU or AX_DEVICE_CUDA
    bool is_arena_temp;    // true if from a bump arena (release is no-op)
    uint64_t generation;   // bumped on every in-place write
} ax_storage_t;`}</code></pre>
      <p>
        Storage is reference-counted with C11 atomics. Multiple tensors can share the same
        storage (views, reshapes). When the last reference is released, the buffer is freed.
        The <code>generation</code> counter is bumped on every in-place mutation so caches
        (like the GEMM pack_b cache) can detect stale data.
      </p>

      <h3>ax_tensor_t</h3>
      <pre><code className="language-c">{`typedef struct ax_tensor {
    ax_storage_t *storage;          // shared data buffer
    int64_t shape[AX_MAX_DIMS];     // size of each dimension
    int64_t strides[AX_MAX_DIMS];   // element stride per dimension
    int ndim;                       // number of dimensions
    ax_dtype_t dtype;               // element type (AX_FLOAT32, etc)
    int64_t offset;                 // element offset into storage

    // autograd fields
    bool requires_grad;
    struct ax_tensor *grad;         // accumulated gradient
    void *grad_fn;                  // pointer to grad function node
} ax_tensor_t;`}</code></pre>

      <h2>Strides and contiguity</h2>
      <p>
        Strides define how many elements to skip to advance one step along each dimension.
        A freshly created tensor has row-major (C-contiguous) strides: the last dimension
        has stride 1, and each dimension before it has stride equal to the product of all
        following dimensions.
      </p>
      <p>
        For a tensor with shape [2, 3, 4], the strides are [12, 4, 1]. Element (i, j, k)
        lives at offset <code>i*12 + j*4 + k*1</code> in the storage buffer.
      </p>
      <p>
        <code>ax_tensor_is_contiguous()</code> checks whether the strides match the row-major
        pattern. The optimized backend requires contiguous tensors for SIMD ops; non-contiguous
        inputs fall back to the naive backend.
      </p>

      <h2>Views and zero-copy operations</h2>
      <p>
        Several operations return a new tensor that <strong>shares storage</strong> with the input:
      </p>
      <ul>
        <li><code>ax_tensor_view(t)</code> — creates a view with the same shape/strides. The storage refcount is incremented.</li>
        <li><code>ax_tensor_reshape(t, new_shape, new_ndim)</code> — if the tensor is contiguous, returns a view with new shape and recomputed strides. If not contiguous, copies the data first.</li>
        <li><code>ax_tensor_transpose(t, dim0, dim1)</code> — swaps two stride entries. Zero-copy, but the result is generally not contiguous.</li>
        <li><code>ax_tensor_squeeze(t, dim)</code> / <code>ax_tensor_unsqueeze(t, dim)</code> — add or remove dimensions of size 1. Zero-copy.</li>
      </ul>
      <p>
        When you need a guaranteed contiguous copy, call <code>ax_tensor_contiguous(t)</code>.
        The helper <code>ax_ensure_contiguous(t)</code> returns the input unchanged if it's already
        contiguous, or makes a copy if not. Check whether the return value differs from the input
        to know if you need to free it.
      </p>

      <h2>The offset field</h2>
      <p>
        The <code>offset</code> field lets a tensor point into the middle of a storage buffer.
        This is used for slicing: a slice of a batch tensor can share the same allocation
        with an offset pointing to the start of the relevant sample. The optimized backend
        checks <code>offset == 0</code> as part of the contiguity fast path.
      </p>

      <h2>Reference counting rules</h2>
      <ul>
        <li><code>ax_storage_create()</code> — refcount starts at 1</li>
        <li><code>ax_storage_retain(s)</code> — increments refcount (atomic)</li>
        <li><code>ax_storage_release(s)</code> — decrements; frees on zero</li>
        <li><code>ax_tensor_destroy(t)</code> — releases the storage, frees the tensor struct</li>
      </ul>
      <p>
        Arena-allocated tensors (<code>is_arena_temp = true</code>) skip the release entirely.
        They're invalidated when the arena is reset. This is used for backward-pass scratch
        tensors where bulk deallocation is far cheaper than individual frees.
      </p>

      <h2>Generation counter</h2>
      <p>
        The <code>generation</code> field on storage is a monotonically increasing counter bumped
        by <code>ax_storage_touch(s)</code> on every in-place write. It starts at 1.
      </p>
      <p>
        The primary consumer is the GEMM pack_b cache in <code>cpu_opt.c</code>. The cache keys on
        the raw B pointer plus the generation. If an optimizer step rewrites weight values in-place
        (same pointer, new contents), the generation mismatch invalidates the cached packed panel
        and forces a re-pack. Without this, the backward pass could silently compute on stale weights.
      </p>

      <h2>Device management</h2>
      <p>
        Each storage knows which device it lives on. To move tensors between CPU and GPU:
      </p>
      <pre><code className="language-c">{`// move to GPU (no-op if already there)
ax_tensor_t *gpu_t = ax_tensor_to_cuda(t);

// move back to CPU
ax_tensor_t *cpu_t = ax_tensor_to_cpu(gpu_t);

// set default device for all new allocations
ax_set_default_device(AX_DEVICE_CUDA);`}</code></pre>
      <p>
        When you set the default device to CUDA and the backend to CUDA, all new tensor
        allocations and compute operations route to the GPU with no changes to training code.
      </p>
    </>
  )
}
