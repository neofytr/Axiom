/* jit_gemm_avx512.h — JIT-emitted AVX-512 14x32 GEMM micro-kernel.

   matches the C intrinsics kernel layout: 28 ZMM accumulators (14 rows ×
   2 cols of 16-wide vectors), 1 zmm for `a` broadcast, 2 for `b0, b1`.
   ldc_bytes is C row stride in BYTES (caller multiplies). */

#ifndef AX_JIT_GEMM_AVX512_H
#define AX_JIT_GEMM_AVX512_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ax_jit_gemm_zmm_kernel_fn)(int64_t kc, const float *ap,
                                           const float *bp, float *c,
                                           int64_t ldc_bytes);

ax_jit_gemm_zmm_kernel_fn ax_jit_gemm_avx512_get_14x32(void);
bool ax_jit_gemm_avx512_available(void);

#ifdef __cplusplus
}
#endif

#endif
