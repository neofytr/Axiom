import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/docs/guides/performance')({
  component: PerformanceGuidePage,
})

function PerformanceGuidePage() {
  return (
    <>
      <h1>Performance Tuning</h1>
      <p>
        Axiom is fast by default, but there are knobs you can turn to get more out of your
        hardware. This guide covers build-time and runtime optimizations.
      </p>

      <h2>Build with Release mode</h2>
      <pre><code className="language-bash">{`cmake -DCMAKE_BUILD_TYPE=Release ..`}</code></pre>
      <p>
        This enables <code>-O3</code> and link-time optimization (LTO). The difference between
        Debug and Release can be 5-10x for compute-bound workloads. Always benchmark with Release.
      </p>

      <h2>Enable ISA dispatch</h2>
      <pre><code className="language-bash">{`cmake -DAX_CPU_ISA_DISPATCH=ON -DCMAKE_BUILD_TYPE=Release ..`}</code></pre>
      <p>
        Builds three variants of the optimized backend (AVX-512, AVX2, scalar) and picks the
        best one at runtime. Without this flag, the compiler generates code for whatever
        <code>-march</code> you specify (or the default, which is usually SSE2 on x86).
      </p>
      <p>
        If you're building for a known target, you can also just pass the right architecture:
      </p>
      <pre><code className="language-bash">{`cmake -DCMAKE_C_FLAGS="-march=native" -DCMAKE_BUILD_TYPE=Release ..`}</code></pre>

      <h2>Thread count</h2>
      <pre><code className="language-c">{`ax_set_num_threads(4);`}</code></pre>
      <p>
        Or set the <code>OMP_NUM_THREADS</code> environment variable. More threads help for
        large GEMMs and parallel element-wise ops, but there's diminishing returns. For
        GEMM-heavy workloads, matching the number of physical cores (not hyperthreads)
        usually works best.
      </p>

      <h3>Hybrid CPU auto-tuning</h3>
      <pre><code className="language-c">{`ax_autotune_threads();`}</code></pre>
      <p>
        On hybrid CPUs (Intel 12th gen+ P-core/E-core, ARM big.LITTLE), this benchmarks each
        core, identifies the fast ones, and sets the thread count to the fast-core count.
        This avoids the common problem where OpenMP barrier synchronization stalls the fast
        cores waiting for slow efficiency cores.
      </p>
      <p>
        Calibration takes under 200ms. Skipped if <code>OMP_NUM_THREADS</code> is already set
        or <code>AX_NO_AUTOTUNE=1</code>.
      </p>

      <h2>GEMM tile tuning</h2>
      <p>
        The tiled GEMM auto-tunes MC and NC panel sizes to fit your CPU's L1 and L2 cache
        at startup. The auto-tuner reads cache sizes from <code>/sys/devices/system/cpu/</code>
        on Linux and adjusts:
      </p>
      <ul>
        <li><strong>MC</strong> — height of the packed A panel. Sized so MC * KC * 4 bytes fits in 3/4 of L1d.</li>
        <li><strong>NC</strong> — width of the packed B panel. Sized so (MC*KC + NC*KC) * 4 bytes fits in L2.</li>
        <li><strong>KC</strong> — depth of each panel block. Default varies by ISA.</li>
      </ul>
      <p>
        You can override with environment variables:
      </p>
      <pre><code className="language-bash">{`AX_GEMM_MC=168 AX_GEMM_NC=2048 AX_GEMM_KC=256 ./train`}</code></pre>
      <p>
        MC must be a multiple of MR, NC must be a multiple of NR. The auto-tuner prints
        its choices to stderr on startup, so you can see what it picked.
      </p>

      <h2>Op fusion</h2>
      <p>
        Several operations are automatically fused when possible:
      </p>
      <ul>
        <li>
          <strong>Dense + ReLU</strong>: when a ReLU layer follows a Dense layer in a sequential
          model, the forward pass uses <code>ax_matmul_bias_relu</code>, which dispatches to the
          backend's <code>gemm_relu</code>. The activation is applied during the GEMM writeback
          while the output is still in registers, saving a full read-write pass over the output.
        </li>
        <li>
          <strong>Cross-entropy backward</strong>: computes <code>softmax(pred) - target</code>
          in a single SIMD pass instead of materializing the full softmax Jacobian.
        </li>
        <li>
          <strong>In-place activations</strong>: <code>ax_relu_inplace</code>,{' '}
          <code>ax_sigmoid_inplace</code>, <code>ax_tanh_inplace</code> mutate the input buffer
          directly, avoiding one allocation and one memory pass.
        </li>
        <li>
          <strong>Fused bias add</strong>: <code>bias_add</code> adds a 1D bias to each row of
          a 2D output in a single pass, instead of broadcasting + add.
        </li>
        <li>
          <strong>AXPY</strong>: <code>y += alpha * x</code> in-place, used for gradient
          accumulation and optimizer updates.
        </li>
      </ul>

      <h2>Transposed GEMM</h2>
      <p>
        The backward pass needs <code>dX = dY @ W^T</code> and <code>dW = X^T @ dY</code>.
        Instead of materializing physical transposes, the optimized backend provides{' '}
        <code>gemm_nt</code> and <code>gemm_tn</code> which walk the matrix with transposed
        indexing during the pack phase. This saves one full copy of the weight or input matrix.
      </p>

      <h2>pack_b caching</h2>
      <p>
        The GEMM pack_b buffer is cached between calls. In backward passes, the same weight
        matrix often appears in two consecutive GEMMs. The second call skips the pack phase
        entirely and reuses the packed panel from the first call. The cache validates on the
        B pointer address, tile indices, and storage generation counter.
      </p>

      <h2>Memory tips</h2>
      <ul>
        <li>
          <strong>Contiguous tensors are fast.</strong> The SIMD fast paths in the optimized
          backend only activate for contiguous tensors with offset 0. Non-contiguous tensors
          fall back to the naive backend. If you transpose or slice, call{' '}
          <code>ax_tensor_contiguous()</code> before heavy compute.
        </li>
        <li>
          <strong>Destroy temporaries promptly.</strong> The training loop creates intermediate
          tensors during forward and backward passes. <code>ax_graph_cleanup(loss)</code> frees
          them all in one shot. Don't skip this call or memory usage will grow linearly with
          training steps.
        </li>
        <li>
          <strong>Arena allocations are free.</strong> The forward and backward arenas use bump
          allocation: a pointer increment instead of malloc. Hundreds of temporary tensors are
          allocated and freed in bulk. This matters at 12.6M parameter scale where backward
          creates hundreds of intermediate tensors.
        </li>
      </ul>

      <h2>Profiling</h2>
      <p>
        Use standard profiling tools:
      </p>
      <pre><code className="language-bash">{`# perf stat for high-level counters
perf stat ./build/benchmark_mnist

# perf record + report for hotspots
perf record -g ./build/benchmark_mnist
perf report

# valgrind for memory (slow but thorough)
valgrind --tool=callgrind ./build/benchmark_mnist
kcachegrind callgrind.out.*`}</code></pre>
      <p>
        The GEMM micro-kernel and pack functions should dominate the profile for training
        workloads. If element-wise ops or memory allocation show up prominently, something
        is wrong.
      </p>

      <h2>Benchmark numbers</h2>
      <p>
        For reference, on GitHub Actions CI (same machine, same dataset, same thread count):
      </p>
      <table>
        <thead>
          <tr>
            <th>Workload</th>
            <th>Axiom</th>
            <th>TensorFlow</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Training 12.6M params (EPYC, 4T)</td>
            <td><strong>69s</strong></td>
            <td>87s</td>
          </tr>
          <tr>
            <td>Training 4.2M params (Graviton3, 4T)</td>
            <td><strong>26s</strong></td>
            <td>30s</td>
          </tr>
          <tr>
            <td>Inference 1.46M (throughput)</td>
            <td><strong>51,226 img/s</strong></td>
            <td>22,185 img/s</td>
          </tr>
        </tbody>
      </table>
      <p>
        The inference gap (2.3x) comes from eliminating Python dispatch overhead and graph
        executor overhead that TensorFlow pays per-op.
      </p>
    </>
  )
}
