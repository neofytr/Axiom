/* jit_gemm_neon.h — JIT-emitted NEON 8x12 GEMM micro-kernel.

   matches the C intrinsics layout:
     - 24 V accumulators (8 rows × 3 vectors of 4 floats)
     - per K iter: 3 b loads (b0,b1,b2) + 2 a loads (a_lo, a_hi as Q regs)
       + 24 FMLA via lane-broadcast (vfmaq_laneq_f32 = single FMLA-by-lane).

   ldc_bytes is C row stride in BYTES (caller multiplies by sizeof(float)). */

#ifndef AX_JIT_GEMM_NEON_H
#define AX_JIT_GEMM_NEON_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ax_jit_gemm_neon_kernel_fn)(int64_t kc, const float *ap,
                                            const float *bp, float *c,
                                            int64_t ldc_bytes);

ax_jit_gemm_neon_kernel_fn ax_jit_gemm_neon_get_8x12(void);
bool ax_jit_gemm_neon_available(void);

#ifdef __cplusplus
}
#endif

#endif
