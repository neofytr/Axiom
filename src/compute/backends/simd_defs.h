/* simd_defs.h — platform abstraction for SIMD intrinsics.
   provides a uniform interface over AVX2, NEON, and scalar fallback. */

#ifndef AX_SIMD_DEFS_H
#define AX_SIMD_DEFS_H

#include <stdint.h>
#include <math.h>

/* max scratch bytes any single kernel may allocate */
#define AX_MAX_SCRATCH_BYTES ((size_t)64u * 1024u * 1024u)

/* platform detection — ordered from widest to narrowest. avx-512 must
   be checked BEFORE avx2 because avx-512 cpus also define __AVX2__. */
#if defined(__AVX512F__) && defined(__FMA__)
    #define AX_SIMD_AVX512 1
    #define AX_HAS_SIMD 1
    #include <immintrin.h>
#elif defined(__AVX2__) && defined(__FMA__)
    #define AX_SIMD_AVX2 1
    #define AX_HAS_SIMD 1
    #include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define AX_SIMD_NEON 1
    #define AX_HAS_SIMD 1
    #include <arm_neon.h>
#endif


/* ================================================================
   AVX-512 tier: 16-wide float vectors using ZMM registers.
   32 ZMM registers enable a 14×32 GEMM micro-kernel (28 accumulators
   + 2 B loads + 1 A broadcast + 1 spare = 32 registers, zero spills).
   ================================================================ */
#if defined(AX_SIMD_AVX512)

#define AX_VF32_WIDTH 16
typedef __m512 ax_vf32;

static inline ax_vf32 ax_vf32_load(const float *p)    { return _mm512_load_ps(p); }
static inline ax_vf32 ax_vf32_loadu(const float *p)   { return _mm512_loadu_ps(p); }
static inline void    ax_vf32_store(float *p, ax_vf32 v)  { _mm512_store_ps(p, v); }
static inline void    ax_vf32_storeu(float *p, ax_vf32 v) { _mm512_storeu_ps(p, v); }
static inline void    ax_vf32_stream(float *p, ax_vf32 v) { _mm512_stream_ps(p, v); }
static inline ax_vf32 ax_vf32_set1(float v)            { return _mm512_set1_ps(v); }
static inline ax_vf32 ax_vf32_zero(void)               { return _mm512_setzero_ps(); }
static inline ax_vf32 ax_vf32_add(ax_vf32 a, ax_vf32 b) { return _mm512_add_ps(a, b); }
static inline ax_vf32 ax_vf32_sub(ax_vf32 a, ax_vf32 b) { return _mm512_sub_ps(a, b); }
static inline ax_vf32 ax_vf32_mul(ax_vf32 a, ax_vf32 b) { return _mm512_mul_ps(a, b); }
static inline ax_vf32 ax_vf32_div(ax_vf32 a, ax_vf32 b) { return _mm512_div_ps(a, b); }
static inline ax_vf32 ax_vf32_neg(ax_vf32 a)           { return _mm512_sub_ps(_mm512_setzero_ps(), a); }
static inline ax_vf32 ax_vf32_max(ax_vf32 a, ax_vf32 b) { return _mm512_max_ps(a, b); }
static inline ax_vf32 ax_vf32_min(ax_vf32 a, ax_vf32 b) { return _mm512_min_ps(a, b); }
static inline ax_vf32 ax_vf32_fmadd(ax_vf32 a, ax_vf32 b, ax_vf32 c) { return _mm512_fmadd_ps(a, b, c); }
static inline ax_vf32 ax_vf32_sqrt(ax_vf32 a)          { return _mm512_sqrt_ps(a); }

static inline ax_vf32 ax_vf32_abs(ax_vf32 a) {
    return _mm512_abs_ps(a);
}
static inline ax_vf32 ax_vf32_relu(ax_vf32 a) {
    return _mm512_max_ps(a, _mm512_setzero_ps());
}

/* avx-512 compare: returns mask. for cmpgt producing a float vector
   of 1.0/0.0 (like the avx2 path), use mask blend. */
static inline ax_vf32 ax_vf32_cmpgt(ax_vf32 a, ax_vf32 b) {
    __mmask16 m = _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);
    return _mm512_mask_blend_ps(m, _mm512_setzero_ps(), _mm512_set1_ps(1.0f));
}

/* horizontal sum: reduce 16 floats to 1 */
static inline float ax_vf32_hsum(ax_vf32 v) {
    return _mm512_reduce_add_ps(v);
}

/* horizontal max: reduce 16 floats to max */
static inline float ax_vf32_hmax(ax_vf32 v) {
    return _mm512_reduce_max_ps(v);
}

/* fast exp: same polynomial as avx2 but 16-wide */
static inline ax_vf32 ax_vf32_exp(ax_vf32 x) {
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.0f));
    x = _mm512_min_ps(x, _mm512_set1_ps(88.0f));
    __m512 ln2_inv = _mm512_set1_ps(1.4426950408889634f);
    __m512 t = _mm512_mul_ps(x, ln2_inv);
    /* round to nearest integer. _MM_FROUND_TO_NEAREST_INT = 0x00:
       bits[3:0]=0 (M=0, integer precision), bits[5:4]=0 (nearest-even). */
    __m512 tn = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT);
    __m512 f = _mm512_sub_ps(t, tn);
    __m512i ni = _mm512_cvtps_epi32(tn);
    __m512i exp_bits = _mm512_slli_epi32(_mm512_add_epi32(ni, _mm512_set1_epi32(127)), 23);
    __m512 pow2n = _mm512_castsi512_ps(exp_bits);
    __m512 c0 = _mm512_set1_ps(1.0f);
    __m512 c1 = _mm512_set1_ps(0.6931472f);
    __m512 c2 = _mm512_set1_ps(0.2402265f);
    __m512 c3 = _mm512_set1_ps(0.0554953f);
    __m512 c4 = _mm512_set1_ps(0.0096813f);
    __m512 c5 = _mm512_set1_ps(0.0013376f);
    __m512 p = _mm512_fmadd_ps(c5, f, c4);
    p = _mm512_fmadd_ps(p, f, c3);
    p = _mm512_fmadd_ps(p, f, c2);
    p = _mm512_fmadd_ps(p, f, c1);
    p = _mm512_fmadd_ps(p, f, c0);
    return _mm512_mul_ps(p, pow2n);
}

/* vector log: same Cephes-style polynomial as avx2 but 16-wide */
static inline ax_vf32 ax_vf32_log(ax_vf32 x) {
    __m512i xi = _mm512_castps_si512(x);
    __m512i exp = _mm512_sub_epi32(_mm512_srli_epi32(xi, 23), _mm512_set1_epi32(127));
    __m512i mant = _mm512_or_si512(_mm512_and_si512(xi, _mm512_set1_epi32(0x007FFFFF)),
                                    _mm512_set1_epi32(0x3F800000));
    __m512 m = _mm512_castsi512_ps(mant);
    __m512 e = _mm512_cvtepi32_ps(exp);
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 f = _mm512_sub_ps(m, one);
    __m512 s = _mm512_div_ps(f, _mm512_add_ps(m, one));
    __m512 s2 = _mm512_mul_ps(s, s);
    __m512 r = _mm512_fmadd_ps(_mm512_set1_ps(0.2392088f), s2, _mm512_set1_ps(0.2850606f));
    r = _mm512_fmadd_ps(r, s2, _mm512_set1_ps(0.4000006f));
    r = _mm512_fmadd_ps(r, s2, _mm512_set1_ps(0.6666667f));
    r = _mm512_fmadd_ps(r, s2, _mm512_set1_ps(2.0f));
    r = _mm512_mul_ps(r, s);
    return _mm512_fmadd_ps(e, _mm512_set1_ps(0.6931472f), r);
}

/* sigmoid: 1 / (1 + exp(-x)) */
static inline ax_vf32 ax_vf32_sigmoid(ax_vf32 x) {
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
    return _mm512_div_ps(one, _mm512_add_ps(one, ax_vf32_exp(neg_x)));
}

/* vector tanh via 2*sigmoid(2x) - 1 identity */
static inline ax_vf32 ax_vf32_tanh(ax_vf32 x) {
    __m512 two = _mm512_set1_ps(2.0f);
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 exp_neg = ax_vf32_exp(_mm512_sub_ps(_mm512_setzero_ps(), _mm512_mul_ps(two, x)));
    __m512 sig2 = _mm512_div_ps(two, _mm512_add_ps(one, exp_neg));
    return _mm512_sub_ps(sig2, one);
}

/* horizontal min */
static inline float ax_vf32_hmin(ax_vf32 v) {
    return _mm512_reduce_min_ps(v);
}

/* compare equal: returns 1.0f where a==b, 0.0f elsewhere */
static inline ax_vf32 ax_vf32_cmpeq(ax_vf32 a, ax_vf32 b) {
    __mmask16 m = _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ);
    return _mm512_mask_blend_ps(m, _mm512_setzero_ps(), _mm512_set1_ps(1.0f));
}

/* paired horizontal max/sum: takes 2 vectors of WIDTH floats and returns
   1 vector of WIDTH floats where output[i] = max/sum(input[2i], input[2i+1])
   treating (a, b) concatenated as a single 2*WIDTH input. used by maxpool
   and avgpool k=2 fast paths. */
static inline ax_vf32 ax_vf32_pmax_pack(ax_vf32 a, ax_vf32 b) {
    __m512i idx_evens = _mm512_setr_epi32(0,2,4,6,8,10,12,14, 16,18,20,22,24,26,28,30);
    __m512i idx_odds  = _mm512_setr_epi32(1,3,5,7,9,11,13,15, 17,19,21,23,25,27,29,31);
    __m512 evens = _mm512_permutex2var_ps(a, idx_evens, b);
    __m512 odds  = _mm512_permutex2var_ps(a, idx_odds, b);
    return _mm512_max_ps(evens, odds);
}
static inline ax_vf32 ax_vf32_padd_pack(ax_vf32 a, ax_vf32 b) {
    __m512i idx_evens = _mm512_setr_epi32(0,2,4,6,8,10,12,14, 16,18,20,22,24,26,28,30);
    __m512i idx_odds  = _mm512_setr_epi32(1,3,5,7,9,11,13,15, 17,19,21,23,25,27,29,31);
    __m512 evens = _mm512_permutex2var_ps(a, idx_evens, b);
    __m512 odds  = _mm512_permutex2var_ps(a, idx_odds, b);
    return _mm512_add_ps(evens, odds);
}


/* ================================================================
   AVX2 tier: 8-wide float vectors using YMM registers.
   ================================================================ */
#elif defined(AX_SIMD_AVX2)

/* avx2: 8-wide float vectors */
#define AX_VF32_WIDTH 8
typedef __m256 ax_vf32;

static inline ax_vf32 ax_vf32_load(const float *p)    { return _mm256_load_ps(p); }
static inline ax_vf32 ax_vf32_loadu(const float *p)   { return _mm256_loadu_ps(p); }
static inline void    ax_vf32_store(float *p, ax_vf32 v)  { _mm256_store_ps(p, v); }
static inline void    ax_vf32_storeu(float *p, ax_vf32 v) { _mm256_storeu_ps(p, v); }
static inline void    ax_vf32_stream(float *p, ax_vf32 v) { _mm256_stream_ps(p, v); }
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

/* vectorized log via frexp-style decomposition plus degree-8 cephes poly.
   extracts mantissa into [0.5, 1), optionally doubles to center around 1,
   then evaluates log(1+z) = z - 0.5*z^2 + z^3 * P(z). ~ulp accurate.
   coefficients from cephes / julien pommier's public domain avx math lib. */
static inline ax_vf32 ax_vf32_log(ax_vf32 x) {
    /* clamp negatives/zero to smallest normal so result is finite */
    __m256 min_norm = _mm256_set1_ps(1.17549435e-38f);
    x = _mm256_max_ps(x, min_norm);

    __m256i xi = _mm256_castps_si256(x);

    /* extract biased exponent, unbias by 127, will add 1 back after shift */
    __m256i ei = _mm256_srli_epi32(xi, 23);
    ei = _mm256_sub_epi32(ei, _mm256_set1_epi32(0x7f));

    /* mantissa m in [0.5, 1): clear exponent bits, or in 0.5 pattern (0x3F000000) */
    __m256i mi = _mm256_and_si256(xi, _mm256_set1_epi32(0x007FFFFF));
    mi = _mm256_or_si256(mi, _mm256_set1_epi32(0x3F000000));
    __m256 m = _mm256_castsi256_ps(mi);

    __m256 e = _mm256_cvtepi32_ps(ei);
    e = _mm256_add_ps(e, _mm256_set1_ps(1.0f));

    /* if m < sqrt(0.5), set m = 2*m - 1 and e -= 1. else set m = m - 1.
       this centers the poly around 0 with |z| < sqrt(2) - 1 ~ 0.414. */
    __m256 sqrt_half = _mm256_set1_ps(0.707106781f);
    __m256 mask = _mm256_cmp_ps(m, sqrt_half, _CMP_LT_OQ);
    __m256 m_masked = _mm256_and_ps(mask, m);
    m = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));
    e = _mm256_sub_ps(e, _mm256_and_ps(mask, _mm256_set1_ps(1.0f)));
    m = _mm256_add_ps(m, m_masked);

    __m256 z = m;
    __m256 z2 = _mm256_mul_ps(z, z);

    /* degree-8 horner poly, cephes coefficients */
    __m256 p = _mm256_set1_ps(7.0376836292e-2f);
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(-1.1514610310e-1f));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps( 1.1676998740e-1f));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(-1.2420140846e-1f));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps( 1.4249322787e-1f));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(-1.6668057665e-1f));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps( 2.0000714765e-1f));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(-2.4999993993e-1f));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps( 3.3333331174e-1f));
    p = _mm256_mul_ps(p, z);
    p = _mm256_mul_ps(p, z2);

    /* log(1+z) = z - 0.5*z^2 + z^3 * P(z) */
    __m256 half_z2 = _mm256_mul_ps(z2, _mm256_set1_ps(-0.5f));
    __m256 log1pz = _mm256_add_ps(_mm256_add_ps(z, half_z2), p);

    /* combine: log(x) = e * ln2 + log(1+z) */
    __m256 ln2 = _mm256_set1_ps(0.6931471805f);
    return _mm256_fmadd_ps(e, ln2, log1pz);
}

/* sigmoid: 1 / (1 + exp(-x)) — uses fast exp */
static inline ax_vf32 ax_vf32_sigmoid(ax_vf32 x) {
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 exp_neg = ax_vf32_exp(neg_x);
    __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
}

/* tanh via identity tanh(x) = 2*sigmoid(2x) - 1. reuses vectorized sigmoid. */
static inline ax_vf32 ax_vf32_tanh(ax_vf32 x) {
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 s = ax_vf32_sigmoid(_mm256_mul_ps(two, x));
    return _mm256_sub_ps(_mm256_mul_ps(two, s), one);
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

/* horizontal max of 8 floats */
static inline float ax_vf32_hmax(ax_vf32 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 m = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    return _mm_cvtss_f32(m);
}

/* horizontal min of 8 floats */
static inline float ax_vf32_hmin(ax_vf32 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 m = _mm_min_ps(lo, hi);
    m = _mm_min_ps(m, _mm_movehl_ps(m, m));
    m = _mm_min_ss(m, _mm_shuffle_ps(m, m, 1));
    return _mm_cvtss_f32(m);
}

/* paired horizontal max/sum: out[i] = max/sum(in[2i], in[2i+1])
   over the 16-element concatenation of (a, b). avx2 shuffle works
   per-128-bit-lane so we need a final cross-lane permutevar to fix
   the resulting layout into [a01,a23,a45,a67,b01,b23,b45,b67]. */
static inline ax_vf32 ax_vf32_pmax_pack(ax_vf32 a, ax_vf32 b) {
    __m256 evens = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
    __m256 odds  = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1));
    __m256 m = _mm256_max_ps(evens, odds);
    return _mm256_permutevar8x32_ps(m, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
}
static inline ax_vf32 ax_vf32_padd_pack(ax_vf32 a, ax_vf32 b) {
    __m256 evens = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
    __m256 odds  = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1));
    __m256 s = _mm256_add_ps(evens, odds);
    return _mm256_permutevar8x32_ps(s, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
}


#elif defined(AX_SIMD_NEON)

/* neon: 4-wide float vectors */
#define AX_VF32_WIDTH 4
typedef float32x4_t ax_vf32;

static inline ax_vf32 ax_vf32_load(const float *p)    { return vld1q_f32(p); }
static inline ax_vf32 ax_vf32_loadu(const float *p)   { return vld1q_f32(p); }
static inline void    ax_vf32_store(float *p, ax_vf32 v)  { vst1q_f32(p, v); }
static inline void    ax_vf32_storeu(float *p, ax_vf32 v) { vst1q_f32(p, v); }
static inline void    ax_vf32_stream(float *p, ax_vf32 v) { vst1q_f32(p, v); } /* no NT on neon */
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

/* vectorized exp: same polynomial as avx2 ported to neon intrinsics.
   exp(x) = 2^n * poly(f) where n = round(x/ln2), f = x/ln2 - n */
static inline ax_vf32 ax_vf32_exp(ax_vf32 x) {
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));
    x = vminq_f32(x, vdupq_n_f32(88.0f));

    float32x4_t ln2_inv = vdupq_n_f32(1.4426950408889634f);
    float32x4_t t = vmulq_f32(x, ln2_inv);
    float32x4_t tn = vrndnq_f32(t);
    float32x4_t f = vsubq_f32(t, tn);

    /* 2^n via integer exponent bit shift */
    int32x4_t ni = vcvtq_s32_f32(tn);
    int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23);
    float32x4_t pow2n = vreinterpretq_f32_s32(exp_bits);

    /* degree-5 poly for 2^f, f in [-0.5, 0.5] */
    float32x4_t p = vdupq_n_f32(0.0013376f);
    p = vfmaq_f32(vdupq_n_f32(0.0096813f), p, f);
    p = vfmaq_f32(vdupq_n_f32(0.0554953f), p, f);
    p = vfmaq_f32(vdupq_n_f32(0.2402265f), p, f);
    p = vfmaq_f32(vdupq_n_f32(0.6931472f), p, f);
    p = vfmaq_f32(vdupq_n_f32(1.0f),       p, f);

    return vmulq_f32(p, pow2n);
}

/* vectorized log: cephes-style frexp decomposition with degree-8 poly.
   same algorithm as avx2 log ported to neon. */
static inline ax_vf32 ax_vf32_log(ax_vf32 x) {
    float32x4_t min_norm = vdupq_n_f32(1.17549435e-38f);
    x = vmaxq_f32(x, min_norm);

    int32x4_t xi = vreinterpretq_s32_f32(x);

    /* extract exponent, unbias */
    int32x4_t ei = vsubq_s32(vshrq_n_s32(xi, 23), vdupq_n_s32(0x7f));

    /* mantissa in [0.5, 1) */
    int32x4_t mi = vorrq_s32(vandq_s32(xi, vdupq_n_s32(0x007FFFFF)),
                              vdupq_n_s32(0x3F000000));
    float32x4_t m = vreinterpretq_f32_s32(mi);
    float32x4_t e = vcvtq_f32_s32(ei);
    e = vaddq_f32(e, vdupq_n_f32(1.0f));

    /* if m < sqrt(0.5), double m and decrement e */
    float32x4_t sqrt_half = vdupq_n_f32(0.707106781f);
    uint32x4_t lt_mask = vcltq_f32(m, sqrt_half);
    float32x4_t m_masked = vreinterpretq_f32_u32(
        vandq_u32(lt_mask, vreinterpretq_u32_f32(m)));
    m = vsubq_f32(m, vdupq_n_f32(1.0f));
    e = vsubq_f32(e, vreinterpretq_f32_u32(
        vandq_u32(lt_mask, vreinterpretq_u32_f32(vdupq_n_f32(1.0f)))));
    m = vaddq_f32(m, m_masked);

    float32x4_t z = m;
    float32x4_t z2 = vmulq_f32(z, z);

    /* degree-8 horner poly, cephes coefficients */
    float32x4_t p = vdupq_n_f32(7.0376836292e-2f);
    p = vfmaq_f32(vdupq_n_f32(-1.1514610310e-1f), p, z);
    p = vfmaq_f32(vdupq_n_f32( 1.1676998740e-1f), p, z);
    p = vfmaq_f32(vdupq_n_f32(-1.2420140846e-1f), p, z);
    p = vfmaq_f32(vdupq_n_f32( 1.4249322787e-1f), p, z);
    p = vfmaq_f32(vdupq_n_f32(-1.6668057665e-1f), p, z);
    p = vfmaq_f32(vdupq_n_f32( 2.0000714765e-1f), p, z);
    p = vfmaq_f32(vdupq_n_f32(-2.4999993993e-1f), p, z);
    p = vfmaq_f32(vdupq_n_f32( 3.3333331174e-1f), p, z);
    p = vmulq_f32(p, z);
    p = vmulq_f32(p, z2);

    /* log(1+z) = z - 0.5*z^2 + z^3 * P(z) */
    float32x4_t half_z2 = vmulq_f32(z2, vdupq_n_f32(-0.5f));
    float32x4_t log1pz = vaddq_f32(vaddq_f32(z, half_z2), p);

    /* combine: log(x) = e * ln2 + log(1+z) */
    return vfmaq_f32(log1pz, e, vdupq_n_f32(0.6931471805f));
}
static inline ax_vf32 ax_vf32_sigmoid(ax_vf32 x) {
    ax_vf32 neg_x = vnegq_f32(x);
    ax_vf32 exp_neg = ax_vf32_exp(neg_x);
    ax_vf32 one = vdupq_n_f32(1.0f);
    return vdivq_f32(one, vaddq_f32(one, exp_neg));
}
/* tanh via identity tanh(x) = 2*sigmoid(2x) - 1. reuses vectorized sigmoid. */
static inline ax_vf32 ax_vf32_tanh(ax_vf32 x) {
    ax_vf32 two = vdupq_n_f32(2.0f);
    ax_vf32 one = vdupq_n_f32(1.0f);
    ax_vf32 s = ax_vf32_sigmoid(vmulq_f32(two, x));
    return vsubq_f32(vmulq_f32(two, s), one);
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
static inline float ax_vf32_hmax(ax_vf32 v) {
    float32x2_t s = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
    s = vpmax_f32(s, s);
    return vget_lane_f32(s, 0);
}
static inline float ax_vf32_hmin(ax_vf32 v) {
    float32x2_t s = vpmin_f32(vget_low_f32(v), vget_high_f32(v));
    s = vpmin_f32(s, s);
    return vget_lane_f32(s, 0);
}

/* paired horizontal max/sum: native ARMv8 intrinsics. */
static inline ax_vf32 ax_vf32_pmax_pack(ax_vf32 a, ax_vf32 b) {
    return vpmaxq_f32(a, b);
}
static inline ax_vf32 ax_vf32_padd_pack(ax_vf32 a, ax_vf32 b) {
    return vpaddq_f32(a, b);
}


#else

/* scalar fallback: width 1 */
#define AX_VF32_WIDTH 1
typedef float ax_vf32;

static inline ax_vf32 ax_vf32_load(const float *p)    { return *p; }
static inline ax_vf32 ax_vf32_loadu(const float *p)   { return *p; }
static inline void    ax_vf32_store(float *p, ax_vf32 v)  { *p = v; }
static inline void    ax_vf32_storeu(float *p, ax_vf32 v) { *p = v; }
static inline void    ax_vf32_stream(float *p, ax_vf32 v) { *p = v; }
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
static inline float   ax_vf32_hmax(ax_vf32 v)          { return v; }
static inline float   ax_vf32_hmin(ax_vf32 v)          { return v; }
/* paired pack: with WIDTH=1 the pair is just (a, b) → max/sum. */
static inline ax_vf32 ax_vf32_pmax_pack(ax_vf32 a, ax_vf32 b) { return a > b ? a : b; }
static inline ax_vf32 ax_vf32_padd_pack(ax_vf32 a, ax_vf32 b) { return a + b; }

#endif

#endif /* AX_SIMD_DEFS_H */
