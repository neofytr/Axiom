import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/docs/architecture/cuda')({
  component: CudaPage,
})

function CudaPage() {
  return (
    <>
      <h1>CUDA Backend</h1>
      <p>
        The CUDA backend provides GPU acceleration by implementing the same{' '}
        <code>ax_backend_ops_t</code> vtable as the CPU backends. It uses cuBLAS for matrix
        operations and custom CUDA kernels for element-wise ops. Training code doesn't need
        to change; you just set the backend and default device.
      </p>

      <h2>Enabling CUDA</h2>
      <pre><code className="language-bash">{`cmake -DAX_CUDA=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)`}</code></pre>
      <p>Requires the CUDA toolkit (nvcc, cuBLAS) to be installed.</p>

      <h2>Using the GPU</h2>
      <pre><code className="language-c">{`ax_init();

// route compute to GPU
ax_compute_set_backend(AX_BACKEND_CUDA);
ax_set_default_device(AX_DEVICE_CUDA);

// from here, all tensor creation and compute uses the GPU
ax_layer_t *net = ax_sequential_create();
ax_sequential_add(net, ax_dense_create(784, 128, true));
// ... rest of training code is identical`}</code></pre>

      <h2>How it plugs in</h2>
      <p>
        The CUDA backend registers itself in <code>ax_compute_init()</code> by filling the
        backend table at <code>AX_BACKEND_CUDA</code> with the <code>ax_cuda_ops</code> vtable.
        It also registers as the device owner for <code>AX_DEVICE_CUDA</code>, which means:
      </p>
      <ul>
        <li>
          <code>ax_storage_create()</code> with <code>device = AX_DEVICE_CUDA</code> calls{' '}
          <code>ax_cuda_ops.storage_alloc()</code> (which wraps <code>cudaMalloc</code>)
        </li>
        <li>
          <code>ax_storage_release()</code> on GPU storage calls{' '}
          <code>ax_cuda_ops.storage_free()</code> (which wraps <code>cudaFree</code>)
        </li>
        <li>
          <code>ax_tensor_to_cuda(t)</code> allocates GPU storage, copies data via{' '}
          <code>memcpy_h2d</code>, and returns a new tensor on the GPU
        </li>
      </ul>

      <h2>GEMM via cuBLAS</h2>
      <p>
        The <code>gemm</code> slot calls <code>cublasSgemm</code> with the appropriate transpose
        flags and leading dimensions. cuBLAS expects column-major layout, so the backend swaps
        A and B and transposes the result: <code>C = A @ B</code> in row-major is equivalent to{' '}
        <code>C^T = B^T @ A^T</code> in column-major.
      </p>
      <p>
        The <code>gemm_nt</code> and <code>gemm_tn</code> variants map directly to cuBLAS
        transpose flags, avoiding any CPU-side transpose.
      </p>

      <h2>Element-wise kernels</h2>
      <p>
        Element-wise operations (add, mul, relu, sigmoid, etc.) use custom CUDA kernels that
        process one element per thread with grid-stride loops. These are straightforward parallel
        maps. The block size is typically 256 threads, with the grid sized to cover the total
        element count.
      </p>

      <h2>Fused optimizer kernels</h2>
      <p>
        The CUDA backend implements <code>adam_update</code> and <code>sgd_update</code> as
        fused kernels. A single kernel reads weight, gradient, and moment tensors, computes
        the update rule, and writes back the updated weight and moments. This avoids multiple
        kernel launches and multiple memory passes that a decomposed implementation would
        require.
      </p>

      <h2>Memory transfers</h2>
      <p>
        Explicit host-to-device and device-to-host transfers are handled by the vtable's
        memcpy hooks:
      </p>
      <pre><code className="language-c">{`// move tensor to GPU
ax_tensor_t *gpu_t = ax_tensor_to_cuda(cpu_t);

// move back to CPU (e.g., for printing or saving)
ax_tensor_t *cpu_t = ax_tensor_to_cpu(gpu_t);`}</code></pre>
      <p>
        When the default device is CUDA, new tensors are allocated on the GPU directly.
        There's no automatic data movement; you explicitly move tensors when needed.
      </p>

      <h2>Arena behavior on GPU</h2>
      <p>
        The CPU bump arenas (backward arena, forward arena) are disabled when the default device
        is not CPU. <code>ax_backward_arena()</code> and <code>ax_forward_arena()</code> return
        NULL, and arena-aware allocation functions fall back to the normal heap path, which
        routes through the CUDA backend's <code>storage_alloc</code>. The arena only holds host
        memory and can't be used for device tensors.
      </p>

      <h2>Synchronization</h2>
      <p>
        The backend provides a <code>synchronize</code> hook that wraps{' '}
        <code>cudaDeviceSynchronize()</code>. This blocks until all pending GPU work completes.
        It's called automatically where needed (e.g., before timing measurements or data
        transfers).
      </p>
    </>
  )
}
