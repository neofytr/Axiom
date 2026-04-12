import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/docs/architecture/gemm')({
  component: GemmPage,
})

function GemmPage() {
  return (
    <>
      <h1>GEMM Kernels</h1>
      <p>
        Matrix multiplication is the computational bottleneck of neural network training.
        A dense layer forward pass is a GEMM. The backward pass is two more GEMMs.
        Convolutions are also reduced to GEMM via im2col. Getting GEMM fast is the single
        most important optimization in the entire framework.
      </p>
      <p>
        Axiom uses a BLIS-style tiled GEMM with micro-kernels sized to the CPU's register file.
        This section explains how it works in detail.
      </p>

      <h2>The problem with naive matmul</h2>
      <p>
        A naive three-loop matrix multiply (i, j, k) has terrible cache behavior. For C = A * B
        where A is MxK and B is KxN, the inner loop over k accesses A row-wise (sequential)
        but B column-wise (strided). For large N, every B access is a cache miss. The naive
        approach achieves maybe 2-5% of peak FLOPS on a modern CPU.
      </p>

      <h2>BLIS tiling strategy</h2>
      <p>
        BLIS (BLAS-like Library Instantiation Software) introduced a five-loop tiling strategy
        that maps matrix panels to cache levels. Axiom implements the key ideas:
      </p>
      <ol>
        <li>
          <strong>JC loop</strong> — partitions B into column panels of width NC. Each panel
          fits in L3 cache (or L2 on systems without L3). This is the outer parallelism loop.
        </li>
        <li>
          <strong>PC loop</strong> — partitions the K dimension into blocks of KC. This controls
          how much of A and B we process at a time, sized so the packed panels fit in L2/L1.
        </li>
        <li>
          <strong>IC loop</strong> — partitions A into row panels of height MC. The packed A
          panel (MC x KC) fits in L1.
        </li>
        <li>
          <strong>JR loop</strong> — within a B panel, iterates in micro-tiles of width NR.
        </li>
        <li>
          <strong>IR loop</strong> — within an A panel, iterates in micro-tiles of height MR.
          Each (MR x NR) micro-tile is computed by the micro-kernel.
        </li>
      </ol>

      <h2>Micro-kernel register allocation</h2>
      <p>
        The micro-kernel is where almost all FLOPs happen. It computes a small MR x NR tile of
        the output using FMA (fused multiply-add) instructions. The register allocation is
        designed to use every available register:
      </p>
      <table>
        <thead>
          <tr>
            <th>ISA</th>
            <th>MR x NR</th>
            <th>Registers used</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>AVX-512</td>
            <td>14 x 32</td>
            <td>28 ZMM accumulators + 2 B loads + 1 A broadcast = 31 of 32</td>
            <td>Two 16-wide vectors per NR row</td>
          </tr>
          <tr>
            <td>AVX2</td>
            <td>6 x 16</td>
            <td>12 YMM accumulators + 2 B loads + 1 A broadcast = 15 of 16</td>
            <td>Two 8-wide vectors per NR row</td>
          </tr>
          <tr>
            <td>NEON</td>
            <td>8 x 12</td>
            <td>24 accumulators + 3 B loads + 1 A = 28 of 32</td>
            <td>Three 4-wide vectors per NR row</td>
          </tr>
          <tr>
            <td>Scalar</td>
            <td>4 x 4</td>
            <td>16 accumulators</td>
            <td>Fallback for any architecture</td>
          </tr>
        </tbody>
      </table>
      <p>
        The key insight: we want MR * NR accumulator registers to hold the output tile,
        plus a handful for loading A and B values. If we spill to memory, we lose.
        The micro-kernel sizes above are chosen to fill the register file completely.
      </p>

      <h2>Panel packing</h2>
      <p>
        Before the micro-kernel runs, the A and B panels are "packed" into contiguous buffers
        with a layout that the micro-kernel can stream through sequentially:
      </p>
      <ul>
        <li>
          <strong>pack_a</strong>: packs an MC x KC panel of A into MR-wide strips.
          Elements within each strip are contiguous, so the micro-kernel's A broadcast loads
          hit sequential cache lines.
        </li>
        <li>
          <strong>pack_b</strong>: packs a KC x NC panel of B into NR-wide strips.
          Same idea: the micro-kernel's B vector loads are sequential.
        </li>
      </ul>
      <p>
        Pack buffers are <strong>per-thread and persistent</strong>. They're allocated once (on
        first use) and reused across every GEMM call, eliminating ~16 malloc/free pairs per
        GEMM invocation.
      </p>

      <h3>pack_b cache</h3>
      <p>
        The packed B buffer is cached and reused when the same B tile is requested back-to-back.
        This is common in backward passes where the same weight matrix appears in two GEMMs
        (dY @ W for computing dX, then X^T @ dY for computing dW). The cache key is the tuple
        (B pointer, ldb, jc, pc, kc, nc, storage generation). The generation check prevents
        stale hits after in-place weight updates.
      </p>

      <h2>Adaptive tile sizes</h2>
      <p>
        The default tile sizes (MC, NC, KC) are compile-time constants, but at startup the
        auto-tuner adapts them to the actual cache sizes:
      </p>
      <ul>
        <li>
          <strong>MC adaptation</strong>: if the packed A panel (MC * KC * 4 bytes) exceeds
          3/4 of L1d, MC is shrunk to fit. It's rounded down to a multiple of MR and floored
          at 8*MR to keep enough tiles for IC parallelism.
        </li>
        <li>
          <strong>NC adaptation</strong>: the packed B panel (NC * KC * 4 bytes) plus packed A
          should fit in L2. If they don't, NC is shrunk. Rounded down to a multiple of NR.
        </li>
      </ul>
      <p>
        Environment variables <code>AX_GEMM_MC</code>, <code>AX_GEMM_NC</code>,{' '}
        <code>AX_GEMM_KC</code> override the auto-tuned values.
      </p>

      <h2>Parallelism</h2>
      <p>
        The JC loop (outer column panel loop) is parallelized with OpenMP. Each thread works on
        a different NC-wide column panel of B, with its own pack buffers. There's no shared
        mutable state between threads during the GEMM.
      </p>
      <p>
        When there aren't enough JC tiles to fill all threads (small N), the thread count
        auto-shrinks to avoid idle workers sitting at the barrier. An{' '}
        <code>omp_in_parallel()</code> guard prevents nested parallelism: if a conv2d batch
        loop is already parallel, the inner GEMM runs single-threaded.
      </p>

      <h2>Transposed variants</h2>
      <p>
        The optimized backend also provides <code>gemm_nt</code> (A @ B^T) and{' '}
        <code>gemm_tn</code> (A^T @ B) which walk the B or A matrix with transposed indexing
        during packing, avoiding the cost of materializing a physical transpose. These are
        used in the backward pass:
      </p>
      <ul>
        <li><code>dX = dY @ W^T</code> — uses gemm_nt with W stored in its original layout</li>
        <li><code>dW = X^T @ dY</code> — uses gemm_tn with X stored in its original layout</li>
      </ul>

      <h2>Fused variants</h2>
      <ul>
        <li>
          <strong>gemm_relu</strong>: applies <code>max(0, x)</code> during the micro-kernel's
          writeback step. The output tile is still in registers, so the relu is essentially free
          (no extra memory pass). Used for Dense+ReLU fusion.
        </li>
        <li>
          <strong>gemm_ex</strong>: <code>out = alpha * (A @ B) + beta * out</code>. The classic
          BLAS sgemm interface. Used for gradient accumulation (beta=1) and scaling.
        </li>
      </ul>

      <h2>Implicit im2col convolution</h2>
      <p>
        The backend also provides <code>conv_gemm</code>, which performs convolution without
        materializing the full im2col matrix. During the GEMM's pack_b phase, input patches
        are gathered directly from the image buffer using the convolution's kernel/stride/padding
        parameters. This halves the memory footprint for large convolutions.
      </p>
    </>
  )
}
