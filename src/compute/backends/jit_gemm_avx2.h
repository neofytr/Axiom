/* jit_gemm_avx2.h — JIT-emitted AVX2 6x16 GEMM micro-kernel.

   the JITed kernel computes a single (mr=GEMM_MR_AVX2=6, nr=GEMM_NR_AVX2=16)
   tile of C += pack_a * pack_b for an arbitrary kc passed at call time.
   the kernel itself is shape-independent (kc is a runtime arg), but the
   instructions are emitted as a single contiguous block so the cpu can
   prefetch and the compiler-induced overhead of the C version is gone.

   later phases can specialize per-kc (full unroll) — phase 36 just does
   the runtime-kc K-loop for now. */

#ifndef AX_JIT_GEMM_AVX2_H
#define AX_JIT_GEMM_AVX2_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* JITed kernel signature. ldc_bytes is C row stride in BYTES (caller
   multiplies elements by sizeof(float) before invoking). */
typedef void (*ax_jit_gemm_kernel_fn)(int64_t kc, const float *ap,
                                       const float *bp, float *c,
                                       int64_t ldc_bytes);

/* lazily emit the 6x16 AVX2 micro-kernel and return a pointer to it.
   the emitted code is owned by an internal global buffer; subsequent
   calls return the same pointer. returns NULL on JIT failure (e.g. no
   mmap). thread-safe via a one-time init guarded by a mutex. */
ax_jit_gemm_kernel_fn ax_jit_gemm_avx2_get_6x16(void);

/* check if JIT is available and the kernel was successfully emitted. */
bool ax_jit_gemm_avx2_available(void);

/* phase i.1.a — strided-A 6x16 micro-kernel.

   same shape as the packed-A 6x16 kernel above but reads A directly from
   row-major memory with caller-supplied row stride `lda_bytes`. used by
   sdpa_bwd to skip the pack_a_t step that would otherwise pre-transpose
   P_tile / dS_tile into MR-strip-packed form.

   semantics: C += sum_{k=0..kc-1} A[k*lda] * B[k*NR],
              A[k*lda] is a vector of MR=6 floats at byte offset
                k*lda_bytes from `ap`.
              B[k*NR] is the standard NR=16-strip-packed b panel
                (so b[k*NR] is k*64 bytes from `bp`).
              C is row-major, leading dim ldc_bytes.

   per-kc fully-unrolled emitter mirrors the packed variant:
     - kc range 1..256
     - cached per kc; one mmap'd page per kc emitted on demand
     - thread-safe (each kc lives in its own page; aligned ptr publish) */
typedef void (*ax_jit_gemm_stridedA_kernel_fn)(int64_t kc,
                                                const float *ap, int64_t lda_bytes,
                                                const float *bp,
                                                float *c, int64_t ldc_bytes);

ax_jit_gemm_stridedA_kernel_fn
ax_jit_gemm_avx2_get_6x16_stridedA_kc(int64_t kc);

/* batch C: per-kc fully-unrolled kernel. for small kc, emits a kernel
   with the K-loop fully unrolled — no dec/branch overhead, register-
   allocator can schedule FMAs ideally. results cached by kc.

   thread-safety: each cached kc lives in its OWN mmap'd page, so
   emission for kc=A never touches a page that's executing kc=B.
   the cache slot itself is published with a release store so concurrent
   readers either see NULL (and fall back) or a fully-written page.
   safe to call from any thread, anytime, including under omp parallel. */
ax_jit_gemm_kernel_fn ax_jit_gemm_avx2_get_6x16_kc(int64_t kc);

#ifdef __cplusplus
}
#endif

#endif
