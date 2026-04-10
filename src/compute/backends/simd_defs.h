/* simd_defs.h — platform abstraction for SIMD intrinsics.
   provides a uniform interface over AVX2, NEON, and scalar fallback.
   phase 0: scalar-only. phase 2+ fills in real SIMD. */

#ifndef AX_SIMD_DEFS_H
#define AX_SIMD_DEFS_H

#include <stdint.h>
#include <math.h>

/* max scratch bytes any single kernel may allocate.
   prevents crafted shapes from causing unbounded malloc. */
#define AX_MAX_SCRATCH_BYTES ((size_t)64u * 1024u * 1024u)

/* platform detection */
#if defined(__AVX2__) && defined(__FMA__)
    #define AX_SIMD_AVX2 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define AX_SIMD_NEON 1
#endif

#if defined(AX_SIMD_AVX2) || defined(AX_SIMD_NEON)
    #define AX_HAS_SIMD 1
#endif

/* for now: scalar-only definitions. SIMD intrinsics added in phase 2. */

#define AX_VF32_WIDTH 1

typedef float ax_vf32;

static inline ax_vf32 ax_vf32_load(const float *p) { return *p; }
static inline void    ax_vf32_store(float *p, ax_vf32 v) { *p = v; }
static inline ax_vf32 ax_vf32_set1(float v) { return v; }
static inline ax_vf32 ax_vf32_zero(void) { return 0.0f; }
static inline ax_vf32 ax_vf32_add(ax_vf32 a, ax_vf32 b) { return a + b; }
static inline ax_vf32 ax_vf32_sub(ax_vf32 a, ax_vf32 b) { return a - b; }
static inline ax_vf32 ax_vf32_mul(ax_vf32 a, ax_vf32 b) { return a * b; }
static inline ax_vf32 ax_vf32_div(ax_vf32 a, ax_vf32 b) { return a / b; }
static inline ax_vf32 ax_vf32_neg(ax_vf32 a) { return -a; }
static inline ax_vf32 ax_vf32_abs(ax_vf32 a) { return fabsf(a); }
static inline ax_vf32 ax_vf32_max(ax_vf32 a, ax_vf32 b) { return a > b ? a : b; }
static inline ax_vf32 ax_vf32_min(ax_vf32 a, ax_vf32 b) { return a < b ? a : b; }
static inline ax_vf32 ax_vf32_fmadd(ax_vf32 a, ax_vf32 b, ax_vf32 c) { return a * b + c; }
static inline ax_vf32 ax_vf32_sqrt(ax_vf32 a) { return sqrtf(a); }
static inline ax_vf32 ax_vf32_exp(ax_vf32 a) { return expf(a); }
static inline ax_vf32 ax_vf32_log(ax_vf32 a) { return logf(a); }
static inline ax_vf32 ax_vf32_tanh(ax_vf32 a) { return tanhf(a); }

/* relu: max(0, x) — branchless in SIMD, simple compare in scalar */
static inline ax_vf32 ax_vf32_relu(ax_vf32 a) { return a > 0.0f ? a : 0.0f; }

/* sigmoid: 1 / (1 + exp(-x)) */
static inline ax_vf32 ax_vf32_sigmoid(ax_vf32 a) { return 1.0f / (1.0f + expf(-a)); }

/* comparison returning float mask (1.0 or 0.0) */
static inline ax_vf32 ax_vf32_cmpgt(ax_vf32 a, ax_vf32 b) { return a > b ? 1.0f : 0.0f; }
static inline ax_vf32 ax_vf32_cmpeq(ax_vf32 a, ax_vf32 b) { return a == b ? 1.0f : 0.0f; }

#endif /* AX_SIMD_DEFS_H */
