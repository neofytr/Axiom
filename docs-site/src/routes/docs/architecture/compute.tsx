import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/docs/architecture/compute')({
  component: ComputePage,
})

function ComputePage() {
  return (
    <>
      <h1>Compute Dispatch</h1>
      <p>
        The compute layer is the routing table between tensor operations and backend kernels.
        It's a thin indirection that lets the same training code run on different hardware
        without any code changes.
      </p>

      <h2>Backend vtable</h2>
      <p>
        Every backend is defined by a single struct of function pointers:
      </p>
      <pre><code className="language-c">{`typedef struct {
    const char *name;    // "cpu_naive", "cpu_opt_avx2", "cuda", etc

    // element-wise binary
    ax_status_t (*add)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
    ax_status_t (*sub)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
    ax_status_t (*mul)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
    ax_status_t (*div_op)(...);

    // unary
    ax_status_t (*neg)(...);
    ax_status_t (*exp_op)(...);
    // ... (relu, sigmoid, tanh, etc)

    // matrix multiply
    ax_status_t (*gemm)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);

    // optional ops (NULL when not implemented)
    ax_status_t (*gemm_nt)(...);     // out = a @ b^T
    ax_status_t (*gemm_tn)(...);     // out = a^T @ b
    ax_status_t (*gemm_relu)(...);   // out = relu(a @ b + bias)
    ax_status_t (*gemm_ex)(...);     // out = alpha*(a @ b) + beta*out
    ax_status_t (*axpy)(...);        // y += alpha * x
    ax_status_t (*conv_gemm)(...);   // implicit im2col convolution
    ax_status_t (*adam_update)(...);  // fused optimizer kernel
    ax_status_t (*sgd_update)(...);

    // device memory hooks (non-CPU backends only)
    ax_device_t device;
    void *(*storage_alloc)(size_t);
    void  (*storage_free)(void *);
    ax_status_t (*memcpy_h2d)(...);
    ax_status_t (*memcpy_d2h)(...);
} ax_backend_ops_t;`}</code></pre>
      <p>
        Required ops (add, mul, gemm, relu, etc) must be filled in by every backend. Optional
        ops (gemm_nt, gemm_relu, conv_gemm, etc) can be NULL. The dispatch layer checks for
        NULL and returns <code>AX_ERR_NOT_IMPLEMENTED</code>, letting the caller decide on a fallback.
      </p>

      <h2>Backend registration</h2>
      <p>
        <code>ax_compute_init()</code> runs once at startup. It:
      </p>
      <ol>
        <li>Registers <code>cpu_naive</code> as the always-available fallback</li>
        <li>
          Under <code>AX_CPU_ISA_DISPATCH</code>, probes the CPU:
          <ul>
            <li>If AVX-512 is available: <code>ax_cpu_opt_ops_avx512</code></li>
            <li>Else if AVX2+FMA: <code>ax_cpu_opt_ops_avx2</code></li>
            <li>Else: <code>ax_cpu_opt_ops_scalar</code></li>
          </ul>
          Without ISA dispatch, the single <code>ax_cpu_opt_ops</code> is used.
        </li>
        <li>If CUDA is compiled in, registers <code>ax_cuda_ops</code></li>
        <li>Selects the best available backend as the default</li>
        <li>Runs tile auto-tuning for the selected CPU backend</li>
      </ol>

      <h2>ISA dispatch</h2>
      <p>
        The <code>AX_CPU_ISA_DISPATCH</code> build flag compiles <code>cpu_opt.c</code> three times
        with different compiler flags:
      </p>
      <ul>
        <li>Once with <code>-mavx512f -mavx512vl</code> — produces <code>ax_cpu_opt_ops_avx512</code></li>
        <li>Once with <code>-mavx2 -mfma</code> — produces <code>ax_cpu_opt_ops_avx2</code></li>
        <li>Once with no SIMD flags — produces <code>ax_cpu_opt_ops_scalar</code></li>
      </ul>
      <p>
        The C preprocessor macro <code>AX_CPU_OPT_SUFFIX</code> is set differently for each
        compilation, and a paste macro (<code>AX_SYM</code>) appends the suffix to all externally
        visible symbols. This means one source file produces three distinct object files with
        no symbol collisions. At runtime, <code>__builtin_cpu_supports("avx512f")</code> picks
        the right one.
      </p>

      <h2>Device ownership</h2>
      <p>
        Backends that manage device memory (CUDA) declare which device they own via{' '}
        <code>ops&gt;device</code>. During init, this is registered in a device-to-backend table.
        Core code uses <code>ax_backend_for_device(AX_DEVICE_CUDA)</code> to look up the
        memory hooks (alloc, free, memcpy) without knowing anything about CUDA specifically.
      </p>
      <p>
        CPU backends set <code>device = AX_DEVICE_COUNT</code> (sentinel value) since CPU memory
        is handled inline by <code>tensor.c</code>.
      </p>

      <h2>Dispatch functions</h2>
      <p>
        Each <code>ax_compute_*</code> function is a one-liner that calls through the active
        backend's function pointer:
      </p>
      <pre><code className="language-c">{`ax_status_t ax_compute_add(const ax_tensor_t *a,
                           const ax_tensor_t *b,
                           ax_tensor_t *out) {
    ensure_compute_init();
    return active_ops&gt;add(a, b, out);
}`}</code></pre>
      <p>
        The <code>ensure_compute_init()</code> call is a fast check (single branch on a static
        flag) that lazily initializes the compute system if it hasn't been initialized yet.
      </p>

      <h2>Optional op pattern</h2>
      <p>
        For optional operations, the pattern is:
      </p>
      <pre><code className="language-c">{`// check if available
int ax_compute_has_gemm_relu(void) {
    ensure_compute_init();
    return active_ops&gt;gemm_relu != NULL;
}

// dispatch with fallback
ax_status_t ax_compute_gemm_relu(...) {
    ensure_compute_init();
    if (!active_ops&gt;gemm_relu)
        return AX_ERR_NOT_IMPLEMENTED;
    return active_ops&gt;gemm_relu(a, b, bias, out);
}`}</code></pre>
      <p>
        Callers check <code>ax_compute_has_gemm_relu()</code> once at the start of a forward pass
        and branch to either the fused path or the separate gemm+relu path. This avoids paying
        the dispatch overhead on every sample in a batch.
      </p>

      <h2>Thread control</h2>
      <pre><code className="language-c">{`// set thread count (wraps omp_set_num_threads)
ax_set_num_threads(4);

// auto-tune for hybrid CPUs (P-core/E-core detection)
int fast_cores = ax_autotune_threads();`}</code></pre>
      <p>
        <code>ax_autotune_threads()</code> pins a micro-benchmark to each logical core via{' '}
        <code>sched_setaffinity</code>, measures throughput, clusters cores within 25% of the
        fastest as "fast", and sets the thread count accordingly. Total calibration time
        is bounded under 200ms. Skipped if <code>OMP_NUM_THREADS</code> or{' '}
        <code>AX_NO_AUTOTUNE=1</code> is set.
      </p>
    </>
  )
}
