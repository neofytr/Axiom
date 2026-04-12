import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/docs/building')({
  component: BuildingPage,
})

function BuildingPage() {
  return (
    <>
      <h1>Building Axiom</h1>
      <p>
        Axiom builds with CMake and requires only a C compiler. No external libraries,
        no package manager, no downloads.
      </p>

      <h2>Basic build</h2>
      <pre><code className="language-bash">{`mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
ctest --output-on-failure`}</code></pre>
      <p>
        This builds the static library <code>libaxiom.a</code>, all test binaries, and the examples.
        The default build uses the optimized CPU backend with whatever SIMD the compiler targets.
      </p>

      <h2>CMake flags</h2>
      <table>
        <thead>
          <tr>
            <th>Flag</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>-DCMAKE_BUILD_TYPE=Release</code></td>
            <td>Enables -O3 and LTO. Always use this for benchmarks or deployment.</td>
          </tr>
          <tr>
            <td><code>-DAX_CPU_ISA_DISPATCH=ON</code></td>
            <td>
              Compiles the optimized backend three times: AVX-512, AVX2, and scalar.
              At startup, <code>ax_compute_init()</code> probes the CPU with{' '}
              <code>__builtin_cpu_supports</code> and picks the best variant. The resulting
              binary runs on any x86 or ARM machine.
            </td>
          </tr>
          <tr>
            <td><code>-DAX_CUDA=ON</code></td>
            <td>
              Enables the CUDA backend. Requires the CUDA toolkit (nvcc). Builds{' '}
              <code>cuda_backend.cu</code> which registers a vtable with cuBLAS GEMM,
              fused element-wise kernels, and device memory hooks.
            </td>
          </tr>
          <tr>
            <td><code>-DAX_INFERENCE_ONLY=ON</code></td>
            <td>
              Strips all training code: autograd engine, optimizers, losses, backward functions.
              The resulting binary is under 100KB on ARM. Forward-only functions still work.
              Any code that calls <code>ax_backward()</code> or optimizer functions gets a link error,
              which is the clearest signal that this build doesn't support training.
            </td>
          </tr>
          <tr>
            <td><code>-DAX_PROFILE=embedded-linux</code></td>
            <td>
              Smaller buffer defaults, trimmed for embedded Linux targets (Raspberry Pi, Jetson Nano, etc).
              Still uses stdio and pthreads.
            </td>
          </tr>
          <tr>
            <td><code>-DAX_PROFILE=embedded-baremetal</code></td>
            <td>
              No stdio, no heap allocation, no threads. Designed for Cortex-M and similar bare-metal
              targets. Combine with <code>AX_INFERENCE_ONLY=ON</code> for the smallest possible binary.
            </td>
          </tr>
        </tbody>
      </table>

      <h2>OpenMP</h2>
      <p>
        OpenMP is auto-detected by CMake. When available, it parallelizes the GEMM JC tile loop,
        batchnorm/layernorm channel loops, optimizer updates, and element-wise ops above a size
        threshold (~8K elements per thread). If you want to control thread count:
      </p>
      <pre><code className="language-c">{`ax_set_num_threads(4);  // or set OMP_NUM_THREADS=4`}</code></pre>
      <p>
        On hybrid CPUs (Intel 12th gen+, ARM big.LITTLE), call{' '}
        <code>ax_autotune_threads()</code> at startup. It benchmarks each core, clusters them by
        speed, and sets the thread count to the number of fast cores. This avoids barrier stalls
        where slow efficiency cores block fast performance cores.
      </p>

      <h2>GEMM tile tuning</h2>
      <p>
        The tiled GEMM auto-tunes MC and NC (panel sizes) to fit L1 and L2 cache at startup.
        You can override with environment variables:
      </p>
      <pre><code className="language-bash">{`AX_GEMM_MC=168 AX_GEMM_NC=2048 AX_GEMM_KC=256 ./my_program`}</code></pre>
      <p>
        MC must be a multiple of MR (14 for AVX-512, 6 for AVX2, 8 for NEON, 4 for scalar).
        NC must be a multiple of NR (32, 16, 12, 4 respectively).
      </p>

      <h2>Linking</h2>
      <pre><code className="language-bash">{`# static link
gcc -o my_app my_app.c -Iinclude -Lbuild -laxiom -lm -lpthread

# with CUDA
gcc -o my_app my_app.c -Iinclude -Lbuild -laxiom -lm -lpthread -lcudart -lcublas`}</code></pre>

      <h2>Running tests</h2>
      <pre><code className="language-bash">{`# all tests
ctest --output-on-failure

# specific test
./build/test_tensor
./build/test_autograd

# with address sanitizer (built-in cmake support)
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS="-fsanitize=address" ..
make -j$(nproc) && ctest`}</code></pre>
      <p>
        The CI runs 22 test binaries on x86 (AVX-512 Xeon), ARM (Graviton3 NEON), plus ASan
        and inference-only builds.
      </p>
    </>
  )
}
