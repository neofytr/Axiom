import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/docs/architecture/overview')({
  component: OverviewPage,
})

function OverviewPage() {
  return (
    <>
      <h1>Architecture Overview</h1>
      <p>
        Axiom is organized as a layered pipeline. Each layer has a clear responsibility
        and a clean boundary. From bottom to top:
      </p>

      <h2>The four layers</h2>

      <h3>1. Backend (compute kernels)</h3>
      <p>
        The lowest layer. Each backend is a vtable (<code>ax_backend_ops_t</code>) full of function
        pointers: one for each math operation (add, mul, gemm, relu, etc). Axiom ships with
        three backends:
      </p>
      <ul>
        <li><strong>cpu_naive</strong> — pure C loops. Always available, always correct. Used as the reference for testing.</li>
        <li><strong>cpu_opt</strong> — SIMD-vectorized ops with BLIS-style tiled GEMM. This is what makes Axiom fast. Under ISA dispatch, this compiles into three variants (AVX-512, AVX2, scalar) and the right one is selected at startup.</li>
        <li><strong>cuda</strong> — GPU backend using cuBLAS for GEMM and custom CUDA kernels for element-wise ops. Registers device memory hooks so tensors can live on the GPU.</li>
      </ul>

      <h3>2. Compute dispatch</h3>
      <p>
        A thin routing layer (<code>dispatch.c</code>) that forwards every compute call to the active
        backend's function pointer. User code never touches backends directly. The dispatch layer
        handles:
      </p>
      <ul>
        <li>Backend registration and selection at startup</li>
        <li>ISA probing (<code>__builtin_cpu_supports</code>) to pick the best CPU variant</li>
        <li>Device ownership table: maps <code>AX_DEVICE_CUDA</code> to the CUDA backend so memory allocation routes correctly</li>
        <li>Optional op fallback: if a backend doesn't implement an optional op (like <code>gemm_relu</code>), the dispatch returns <code>AX_ERR_NOT_IMPLEMENTED</code> and the caller falls back</li>
      </ul>

      <h3>3. Ops (tensor operations)</h3>
      <p>
        The <code>ops.c</code> layer sits between user code and compute. Each op function
        (<code>ax_add</code>, <code>ax_matmul</code>, <code>ax_relu</code>, etc) does three things:
      </p>
      <ol>
        <li><strong>Allocate the output tensor</strong> with the correct shape (handling broadcasting rules)</li>
        <li><strong>Call the compute dispatch</strong> to do the actual math</li>
        <li><strong>Record the autograd node</strong> if gradient tracking is enabled. This attaches a <code>grad_fn</code> to the output tensor with a backward function and saved tensors.</li>
      </ol>

      <h3>4. Layers and model</h3>
      <p>
        Layers (<code>ax_layer_t</code>) are higher-level building blocks. Each layer has a forward
        function, a list of parameters, and a train/eval mode toggle. The model container
        (<code>ax_model_t</code>) bundles a layer tree with an optimizer and loss function
        for a simple train/predict interface.
      </p>

      <h2>Data flow: forward pass</h2>
      <pre><code className="language-text">{`user code
  ax_model_train_step(model, input, target)
    ax_layer_forward(net, input)          // layer
      ax_matmul_bias(input, weight, bias) // op — allocates output, records grad_fn
        ax_compute_gemm(a, b, out)        // dispatch — routes to backend
          opt_gemm(a, b, out)             // backend — BLIS tiled GEMM with SIMD
    ax_cross_entropy_loss(pred, target)   // loss op — returns scalar tensor
    ax_backward(loss)                     // autograd — topo-sort, walk backward
    ax_optimizer_step(opt)                // update weights using gradients`}</code></pre>

      <h2>Data flow: backward pass</h2>
      <p>
        <code>ax_backward(loss)</code> starts from the scalar loss tensor and walks the computation
        graph in reverse topological order. At each node, it calls the <code>grad_fn&gt;backward</code>
        function, which computes the gradient with respect to its inputs and accumulates it into
        each input's <code>.grad</code> tensor. The topo-sort is done via a DFS with a visited set.
      </p>
      <p>
        After backward completes, every parameter tensor's <code>.grad</code> field contains the
        gradient of the loss with respect to that parameter. The optimizer reads these gradients
        and updates the weights.
      </p>

      <h2>Memory model</h2>
      <p>
        Tensors are metadata wrappers around a shared <code>ax_storage_t</code> buffer. Storage is
        reference-counted with atomics, so views and reshapes share memory without copying.
        When the refcount hits zero, the buffer is freed (or returned to the pool on embedded).
      </p>
      <p>
        The autograd engine uses two arenas for scratch memory:
      </p>
      <ul>
        <li><strong>Forward arena</strong> — for tensors saved during forward that backward needs (batchnorm's x_hat, etc). Reset by <code>ax_graph_cleanup()</code>.</li>
        <li><strong>Backward arena</strong> — for temporary buffers during the backward pass. Reset after <code>ax_backward()</code> completes.</li>
      </ul>
      <p>
        Grad nodes themselves come from a thread-local slab allocator (free-list of pre-sized blocks),
        avoiding malloc/free per operation during training.
      </p>

      <h2>Threading model</h2>
      <p>
        Parallelism uses OpenMP pragmas at specific points:
      </p>
      <ul>
        <li>GEMM: the JC (column panel) tile loop runs in parallel with per-thread pack buffers</li>
        <li>BatchNorm/LayerNorm: parallel over channels</li>
        <li>Optimizers: parallel over parameters with dynamic scheduling</li>
        <li>Element-wise ops: parallel for when n &gt; 8192 * num_threads</li>
      </ul>
      <p>
        An <code>omp_in_parallel()</code> guard prevents nesting (e.g., conv2d's batch loop calling
        a GEMM that tries to fork again). The GEMM thread count auto-shrinks when there aren't
        enough JC tiles to fill all threads.
      </p>

      <h2>Project layout</h2>
      <pre><code className="language-text">{`include/axiom/     24 public headers
src/core/          tensor, autograd, ops, layers, optimizers, losses
src/compute/       dispatch + backends (cpu_naive, cpu_opt, cuda/)
tests/             22 test binaries
examples/          xor, mnist, mnist_cnn, deep_mlp
benchmarks/        perf comparisons, TF baselines`}</code></pre>
      <p>About 19K lines of C for the whole framework.</p>
    </>
  )
}
