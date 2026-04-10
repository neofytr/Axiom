/* simd_defs.h — platform abstraction for SIMD intrinsics.
   provides a uniform interface over AVX2, NEON, and scalar fallback. */

#ifndef AX_SIMD_DEFS_H
#define AX_SIMD_DEFS_H

#include <stdint.h>
#include <math.h>

/* max scratch bytes any single kernel may allocate */
#define AX_MAX_SCRATCH_BYTES ((size_t)64u * 1024u * 1024u)

/* platform detection */
#if defined(__AVX2__) && defined(__FMA__)
    #define AX_SIMD_AVX2 1
    #define AX_HAS_SIMD 1
    #include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define AX_SIMD_NEON 1
    #define AX_HAS_SIMD 1
    #include <arm_neon.h>
#endif


#if defined(AX_SIMD_AVX2)

/* avx2: 8-wide float vectors */
#define AX_VF32_WIDTH 8
typedef __m256 ax_vf32;

static inline ax_vf32 ax_vf32_load(const float *p)    { return _mm256_load_ps(p); }
static inline ax_vf32 ax_vf32_loadu(const float *p)   { return _mm256_loadu_ps(p); }
static inline void    ax_vf32_store(float *p, ax_vf32 v)  { _mm256_store_ps(p, v); }
static inline void    ax_vf32_storeu(float *p, ax_vf32 v) { _mm256_storeu_ps(p, v); }
static inline ax_vf32 ax_vf32_set1(float v)            { return _mm256_set1_ps(v); }
static inline ax_vf32 ax_vf32_zero(void)               { return _mm256_setzero_ps(); }
static inline ax_vf32 ax_vf32_add(ax_vf32 a, ax_vf32 b) { return _mm256_add_ps(a, b); }
static inline ax_vf32 ax_vf32_sub(ax_vf32 a, ax_vf32 b) { return _mm256_sub_ps(a, b); }
static inline ax_vf32 ax_vf32_mul(ax_vf32 a, ax_vf32 b) { return _mm256_mul_ps(a, b); }
static inline ax_vf32 ax_vf32_div(ax_vf32 a, ax_vf32 b) { return _mm256_div_ps(a, b); }
static inline ax_vf32 ax_vf32_neg(ax_vf32 a)           { return _mm256_sub_ps(_mm256_setzero_ps(), a); }
static inline ax_vf32 ax_vf32_max(ax_vf32 a, ax_vf32 b) { return _mm256_max_ps(a, b); }
static inline ax_vf32 ax_vf32_min(ax_vf32 a, ax_vf32 b) { return _mm256_min_ps(a, b); }
static inline ax_vf32 ax_vf32_fmadd(ax_vf32 a, ax_vf32 b, ax_vf32 c) { return _mm256_fmadd_ps(a, b, c); }
static inline ax_vf32 ax_vf32_sqrt(ax_vf32 a)          { return _mm256_sqrt_ps(a); }

/* branchless abs via clearing sign bit */
static inline ax_vf32 ax_vf32_abs(ax_vf32 a) {
    __m256i mask = _mm256_set1_epi32(0x7FFFFFFF);
    return _mm256_and_ps(a, _mm256_castsi256_ps(mask));
}

/* branchless relu: max(0, x) */
static inline ax_vf32 ax_vf32_relu(ax_vf32 a) {
    return _mm256_max_ps(a, _mm256_setzero_ps());
}

/* fast exp approximation (schraudolph-style with refinement).
   accurate to ~1e-4 relative error. clamped to [-88, 88] to avoid inf/nan. */
static inline ax_vf32 ax_vf32_exp(ax_vf32 x) {
    /* clamp input to prevent overflow/underflow */
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));

    /* exp(x) = 2^(x * log2(e)) = 2^(n + f) where n=floor, f=fraction */
    __m256 ln2_inv = _mm256_set1_ps(1.4426950408889634f);  /* 1/ln(2) */
    __m256 t = _mm256_mul_ps(x, ln2_inv);

    /* round to nearest integer */
    __m256 tn = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 f = _mm256_sub_ps(t, tn);

    /* polynomial approximation of 2^f for f in [-0.5, 0.5] */
    __m256 c0 = _mm256_set1_ps(1.0f);
    __m256 c1 = _mm256_set1_ps(0.6931472f);
    __m256 c2 = _mm256_set1_ps(0.2402265f);
    __m256 c3 = _mm256_set1_ps(0.0554953f);
    __m256 c4 = _mm256_set1_ps(0.0096813f);
    __m256 c5 = _mm256_set1_ps(0.0013376f);

    __m256 p = _mm256_fmadd_ps(c5, f, c4);
    p = _mm256_fmadd_ps(p, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, c1);
    p = _mm256_fmadd_ps(p, f, c0);

    /* multiply by 2^n using integer exponent manipulation */
    __m256i ni = _mm256_cvtps_epi32(tn);
    ni = _mm256_add_epi32(ni, _mm256_set1_epi32(127));
    ni = _mm256_slli_epi32(ni, 23);
    __m256 pow2n = _mm256_castsi256_ps(ni);

    return _mm256_mul_ps(p, pow2n);
}

/* log via standard scalar fallback (vectorized log is complex and error-prone) */
static inline ax_vf32 ax_vf32_log(ax_vf32 a) {
    float tmp[8] __attribute__((aligned(32)));
    _mm256_store_ps(tmp, a);
    for (int i = 0; i < 8; i++) tmp[i] = logf(tmp[i]);
    return _mm256_load_ps(tmp);
}

/* tanh via scalar fallback */
static inline ax_vf32 ax_vf32_tanh(ax_vf32 a) {
    float tmp[8] __attribute__((aligned(32)));
    _mm256_store_ps(tmp, a);
    for (int i = 0; i < 8; i++) tmp[i] = tanhf(tmp[i]);
    return _mm256_load_ps(tmp);
}

/* sigmoid: 1 / (1 + exp(-x)) — uses fast exp */
static inline ax_vf32 ax_vf32_sigmoid(ax_vf32 x) {
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 exp_neg = ax_vf32_exp(neg_x);
    __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
}

/* comparison returning float mask */
static inline ax_vf32 ax_vf32_cmpgt(ax_vf32 a, ax_vf32 b) {
    return _mm256_and_ps(_mm256_cmp_ps(a, b, _CMP_GT_OQ), _mm256_set1_ps(1.0f));
}
static inline ax_vf32 ax_vf32_cmpeq(ax_vf32 a, ax_vf32 b) {
    return _mm256_and_ps(_mm256_cmp_ps(a, b, _CMP_EQ_OQ), _mm256_set1_ps(1.0f));
}

/* horizontal sum of 8 floats */
static inline float ax_vf32_hsum(ax_vf32 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}


#elif defined(AX_SIMD_NEON)

/* neon: 4-wide float vectors */
#define AX_VF32_WIDTH 4
typedef float32x4_t ax_vf32;

static inline ax_vf32 ax_vf32_load(const float *p)    { return vld1q_f32(p); }
static inline ax_vf32 ax_vf32_loadu(const float *p)   { return vld1q_f32(p); }
static inline void    ax_vf32_store(float *p, ax_vf32 v)  { vst1q_f32(p, v); }
static inline void    ax_vf32_storeu(float *p, ax_vf32 v) { vst1q_f32(p, v); }
static inline ax_vf32 ax_vf32_set1(float v)            { return vdupq_n_f32(v); }
static inline ax_vf32 ax_vf32_zero(void)               { return vdupq_n_f32(0.0f); }
static inline ax_vf32 ax_vf32_add(ax_vf32 a, ax_vf32 b) { return vaddq_f32(a, b); }
static inline ax_vf32 ax_vf32_sub(ax_vf32 a, ax_vf32 b) { return vsubq_f32(a, b); }
static inline ax_vf32 ax_vf32_mul(ax_vf32 a, ax_vf32 b) { return vmulq_f32(a, b); }
static inline ax_vf32 ax_vf32_div(ax_vf32 a, ax_vf32 b) { return vdivq_f32(a, b); }
static inline ax_vf32 ax_vf32_neg(ax_vf32 a)           { return vnegq_f32(a); }
static inline ax_vf32 ax_vf32_abs(ax_vf32 a)           { return vabsq_f32(a); }
static inline ax_vf32 ax_vf32_max(ax_vf32 a, ax_vf32 b) { return vmaxq_f32(a, b); }
static inline ax_vf32 ax_vf32_min(ax_vf32 a, ax_vf32 b) { return vminq_f32(a, b); }
static inline ax_vf32 ax_vf32_fmadd(ax_vf32 a, ax_vf32 b, ax_vf32 c) { return vfmaq_f32(c, a, b); }
static inline ax_vf32 ax_vf32_sqrt(ax_vf32 a)          { return vsqrtq_f32(a); }
static inline ax_vf32 ax_vf32_relu(ax_vf32 a)          { return vmaxq_f32(a, vdupq_n_f32(0.0f)); }

/* scalar fallbacks for transcendentals on NEON */
static inline ax_vf32 ax_vf32_exp(ax_vf32 a) {
    float tmp[4]; vst1q_f32(tmp, a);
    for (int i = 0; i < 4; i++) {
        if (tmp[i] > 88.0f) tmp[i] = 88.0f;
        if (tmp[i] < -88.0f) tmp[i] = -88.0f;
        tmp[i] = expf(tmp[i]);
    }
    return vld1q_f32(tmp);
}
static inline ax_vf32 ax_vf32_log(ax_vf32 a) {
    float tmp[4]; vst1q_f32(tmp, a);
    for (int i = 0; i < 4; i++) tmp[i] = logf(tmp[i]);
    return vld1q_f32(tmp);
}
static inline ax_vf32 ax_vf32_tanh(ax_vf32 a) {
    float tmp[4]; vst1q_f32(tmp, a);
    for (int i = 0; i < 4; i++) tmp[i] = tanhf(tmp[i]);
    return vld1q_f32(tmp);
}
static inline ax_vf32 ax_vf32_sigmoid(ax_vf32 x) {
    ax_vf32 neg_x = vnegq_f32(x);
    ax_vf32 exp_neg = ax_vf32_exp(neg_x);
    ax_vf32 one = vdupq_n_f32(1.0f);
    return vdivq_f32(one, vaddq_f32(one, exp_neg));
}
static inline ax_vf32 ax_vf32_cmpgt(ax_vf32 a, ax_vf32 b) {
    uint32x4_t mask = vcgtq_f32(a, b);
    return vreinterpretq_f32_u32(vandq_u32(mask, vreinterpretq_u32_f32(vdupq_n_f32(1.0f))));
}
static inline ax_vf32 ax_vf32_cmpeq(ax_vf32 a, ax_vf32 b) {
    uint32x4_t mask = vceqq_f32(a, b);
    return vreinterpretq_f32_u32(vandq_u32(mask, vreinterpretq_u32_f32(vdupq_n_f32(1.0f))));
}
static inline float ax_vf32_hsum(ax_vf32 v) {
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    s = vpadd_f32(s, s);
    return vget_lane_f32(s, 0);
}


#else

/* scalar fallback: width 1 */
#define AX_VF32_WIDTH 1
typedef float ax_vf32;

static inline ax_vf32 ax_vf32_load(const float *p)    { return *p; }
static inline ax_vf32 ax_vf32_loadu(const float *p)   { return *p; }
static inline void    ax_vf32_store(float *p, ax_vf32 v)  { *p = v; }
static inline void    ax_vf32_storeu(float *p, ax_vf32 v) { *p = v; }
static inline ax_vf32 ax_vf32_set1(float v)            { return v; }
static inline ax_vf32 ax_vf32_zero(void)               { return 0.0f; }
static inline ax_vf32 ax_vf32_add(ax_vf32 a, ax_vf32 b) { return a + b; }
static inline ax_vf32 ax_vf32_sub(ax_vf32 a, ax_vf32 b) { return a - b; }
static inline ax_vf32 ax_vf32_mul(ax_vf32 a, ax_vf32 b) { return a * b; }
static inline ax_vf32 ax_vf32_div(ax_vf32 a, ax_vf32 b) { return a / b; }
static inline ax_vf32 ax_vf32_neg(ax_vf32 a)           { return -a; }
static inline ax_vf32 ax_vf32_abs(ax_vf32 a)           { return fabsf(a); }
static inline ax_vf32 ax_vf32_max(ax_vf32 a, ax_vf32 b) { return a > b ? a : b; }
static inline ax_vf32 ax_vf32_min(ax_vf32 a, ax_vf32 b) { return a < b ? a : b; }
static inline ax_vf32 ax_vf32_fmadd(ax_vf32 a, ax_vf32 b, ax_vf32 c) { return a * b + c; }
static inline ax_vf32 ax_vf32_sqrt(ax_vf32 a)          { return sqrtf(a); }
static inline ax_vf32 ax_vf32_exp(ax_vf32 a) {
    if (a > 88.0f) a = 88.0f;
    if (a < -88.0f) a = -88.0f;
    return expf(a);
}
static inline ax_vf32 ax_vf32_log(ax_vf32 a)           { return logf(a); }
static inline ax_vf32 ax_vf32_tanh(ax_vf32 a)          { return tanhf(a); }
static inline ax_vf32 ax_vf32_relu(ax_vf32 a)          { return a > 0.0f ? a : 0.0f; }
static inline ax_vf32 ax_vf32_sigmoid(ax_vf32 a)       { return 1.0f / (1.0f + expf(-a)); }
static inline ax_vf32 ax_vf32_cmpgt(ax_vf32 a, ax_vf32 b) { return a > b ? 1.0f : 0.0f; }
static inline ax_vf32 ax_vf32_cmpeq(ax_vf32 a, ax_vf32 b) { return a == b ? 1.0f : 0.0f; }
static inline float   ax_vf32_hsum(ax_vf32 v)          { return v; }

#endif

#endif /* AX_SIMD_DEFS_H */
