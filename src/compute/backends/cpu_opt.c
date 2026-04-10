/* cpu_opt.c — optimized cpu backend with contiguous fast paths.
   falls back to cpu_naive for strided/broadcast cases.
   every op validates storage bounds before pointer access. */

#include "axiom/backend_ops.h"
#include "axiom/tensor.h"
#include "axiom/error.h"
#include "axiom/memory.h"
#include "simd_defs.h"
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdint.h>

/* reference backend for fallback */
extern const ax_backend_ops_t ax_cpu_naive_ops;

/* validation helpers */

static inline int64_t fast_numel(const ax_tensor_t *t) {
    int64_t n = 1;
    for (int d = 0; d < t->ndim; d++) n *= t->shape[d];
    return n;
}

/* check that a tensor is contiguous float32 with valid storage.
   returns numel on success, -1 on failure. */
static inline int64_t validate_contig_f32(const ax_tensor_t *t) {
    if (!t || !t->storage || !t->storage->data) return -1;
    if (t->dtype != AX_FLOAT32) return -1;
    if (t->offset != 0) return -1;
    if (!ax_tensor_is_contiguous(t)) return -1;
    int64_t n = fast_numel(t);
    if (n <= 0) return -1;
    /* bounds check: numel * sizeof(float) must fit in storage */
    if ((size_t)n > t->storage->size_bytes / sizeof(float)) return -1;
    return n;
}

/* get raw float pointer for a validated contiguous tensor */
static inline float *raw_f32(const ax_tensor_t *t) {
    return (float *)t->storage->data;
}

/* check two tensors are both contiguous f32 with matching numel */
static inline int64_t validate_pair(const ax_tensor_t *a, const ax_tensor_t *b) {
    int64_t na = validate_contig_f32(a);
    int64_t nb = validate_contig_f32(b);
    if (na < 0 || nb < 0) return -1;
    /* for binary ops, shapes may differ (broadcast) — caller handles that */
    return na;
}

/* check all three tensors are contiguous f32 with same numel (no broadcast) */
static inline int64_t validate_triple_same(const ax_tensor_t *a, const ax_tensor_t *b, const ax_tensor_t *out) {
    int64_t na = validate_contig_f32(a);
    int64_t nb = validate_contig_f32(b);
    int64_t no = validate_contig_f32(out);
    if (na < 0 || nb < 0 || no < 0) return -1;
    if (na != nb || na != no) return -1;
    return na;
}


/* element-wise binary ops (contiguous, no broadcast).
   separate SIMD and scalar expressions to avoid type conflicts. */

#define DEFINE_OPT_BINOP(name, simd_expr, scalar_expr, naive_fn) \
static ax_status_t opt_##name(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) { \
    int64_t n = validate_triple_same(a, b, out); \
    if (n < 0) return ax_cpu_naive_ops.naive_fn(a, b, out); \
    const float *ad = raw_f32(a); \
    const float *bd = raw_f32(b); \
    float *od = raw_f32(out); \
    int64_t i = 0; \
    int64_t vec_end = n - (n % AX_VF32_WIDTH); \
    for (; i < vec_end; i += AX_VF32_WIDTH) { \
        ax_vf32 va = ax_vf32_load(ad + i); \
        ax_vf32 vb = ax_vf32_load(bd + i); \
        ax_vf32_store(od + i, simd_expr); \
    } \
    for (; i < n; i++) { \
        float sa = ad[i], sb = bd[i]; \
        od[i] = scalar_expr; \
    } \
    return AX_OK; \
}

DEFINE_OPT_BINOP(add, ax_vf32_add(va, vb), sa + sb, add)
DEFINE_OPT_BINOP(sub, ax_vf32_sub(va, vb), sa - sb, sub)
DEFINE_OPT_BINOP(mul, ax_vf32_mul(va, vb), sa * sb, mul)
DEFINE_OPT_BINOP(div_op, ax_vf32_div(va, vb), sa / sb, div_op)


/* element-wise unary ops (contiguous) */

#define DEFINE_OPT_UNOP(name, simd_expr, scalar_expr, naive_fn) \
static ax_status_t opt_##name(const ax_tensor_t *in, ax_tensor_t *out) { \
    int64_t ni = validate_contig_f32(in); \
    int64_t no = validate_contig_f32(out); \
    if (ni < 0 || no < 0 || ni != no) return ax_cpu_naive_ops.naive_fn(in, out); \
    const float *id = raw_f32(in); \
    float *od = raw_f32(out); \
    int64_t n = ni; \
    int64_t i = 0; \
    int64_t vec_end = n - (n % AX_VF32_WIDTH); \
    for (; i < vec_end; i += AX_VF32_WIDTH) { \
        ax_vf32 v = ax_vf32_load(id + i); \
        ax_vf32_store(od + i, simd_expr); \
    } \
    for (; i < n; i++) { \
        float sv = id[i]; \
        od[i] = scalar_expr; \
    } \
    return AX_OK; \
}

DEFINE_OPT_UNOP(neg, ax_vf32_neg(v), -sv, neg)
DEFINE_OPT_UNOP(abs_op, ax_vf32_abs(v), fabsf(sv), abs_op)
DEFINE_OPT_UNOP(exp_op, ax_vf32_exp(v), expf(sv > 88.0f ? 88.0f : (sv < -88.0f ? -88.0f : sv)), exp_op)
DEFINE_OPT_UNOP(log_op, ax_vf32_log(v), (sv > 0.0f ? logf(sv) : -FLT_MAX), log_op)
DEFINE_OPT_UNOP(sqrt_op, ax_vf32_sqrt(v), (sv >= 0.0f ? sqrtf(sv) : 0.0f), sqrt_op)
DEFINE_OPT_UNOP(square, ax_vf32_mul(v, v), sv * sv, square)

/* activations */
DEFINE_OPT_UNOP(relu, ax_vf32_relu(v), (sv > 0.0f ? sv : 0.0f), relu)
DEFINE_OPT_UNOP(sigmoid, ax_vf32_sigmoid(v), (1.0f / (1.0f + expf(-sv))), sigmoid)
DEFINE_OPT_UNOP(tanh_op, ax_vf32_tanh(v), tanhf(sv), tanh_op)


/* scalar ops */

static ax_status_t opt_add_scalar(const ax_tensor_t *in, double scalar, ax_tensor_t *out) {
    int64_t ni = validate_contig_f32(in);
    int64_t no = validate_contig_f32(out);
    if (ni < 0 || no < 0 || ni != no) return ax_cpu_naive_ops.add_scalar(in, scalar, out);
    const float *id = raw_f32(in);
    float *od = raw_f32(out);
    float s = (float)scalar;
    for (int64_t i = 0; i < ni; i++)
        od[i] = id[i] + s;
    return AX_OK;
}

static ax_status_t opt_mul_scalar(const ax_tensor_t *in, double scalar, ax_tensor_t *out) {
    int64_t ni = validate_contig_f32(in);
    int64_t no = validate_contig_f32(out);
    if (ni < 0 || no < 0 || ni != no) return ax_cpu_naive_ops.mul_scalar(in, scalar, out);
    const float *id = raw_f32(in);
    float *od = raw_f32(out);
    float s = (float)scalar;
    for (int64_t i = 0; i < ni; i++)
        od[i] = id[i] * s;
    return AX_OK;
}


/* tiled gemm: cache-blocked with panel packing (BLIS-style).
   6x16 micro-kernel on AVX2: 12 YMM accumulators + 2 B loads + 1 A broadcast
   = 15 of 16 registers. no spills, near-peak FMA throughput. */

#define GEMM_MC 72     /* rows of A panel (multiple of MR=6) */
#define GEMM_NC 256    /* cols of B panel (multiple of NR=16) */
#define GEMM_KC 256    /* depth of both panels */

#if defined(AX_SIMD_AVX2)
    #define GEMM_MR 6
    #define GEMM_NR 16     /* 2 x 8-wide AVX2 vectors per row */
#elif defined(AX_SIMD_NEON)
    #define GEMM_MR 4
    #define GEMM_NR 8      /* 2 x 4-wide NEON vectors per row */
#else
    #define GEMM_MR 4
    #define GEMM_NR 4
#endif

/* pack a MC x KC panel of A (row-major) into contiguous MR-row strips */
static void pack_a(const float *a, int64_t lda, int64_t mc, int64_t kc,
                    int64_t m_remain, float *packed)
{
    for (int64_t i = 0; i < mc; i += GEMM_MR) {
        int64_t mr = (i + GEMM_MR <= m_remain) ? GEMM_MR : (m_remain > i ? m_remain - i : 0);
        for (int64_t p = 0; p < kc; p++) {
            for (int64_t ii = 0; ii < GEMM_MR; ii++) {
                if (ii < mr)
                    packed[ii] = a[(i + ii) * lda + p];
                else
                    packed[ii] = 0.0f;
            }
            packed += GEMM_MR;
        }
    }
}

/* pack a KC x NC panel of B (row-major) into contiguous NR-col strips */
static void pack_b(const float *b, int64_t ldb, int64_t kc, int64_t nc,
                    int64_t n_remain, float *packed)
{
    for (int64_t j = 0; j < nc; j += GEMM_NR) {
        int64_t nr = (j + GEMM_NR <= n_remain) ? GEMM_NR : (n_remain > j ? n_remain - j : 0);
        for (int64_t p = 0; p < kc; p++) {
            for (int64_t jj = 0; jj < GEMM_NR; jj++) {
                if (jj < nr)
                    packed[jj] = b[p * ldb + (j + jj)];
                else
                    packed[jj] = 0.0f;
            }
            packed += GEMM_NR;
        }
    }
}


#if defined(AX_SIMD_AVX2)

/* 6x16 AVX2+FMA micro-kernel.
   12 YMM accumulators (6 rows x 2 vectors), fully pinned in registers.
   A is broadcast per row, B is loaded as 2 contiguous vectors.
   no register spills — verified by inspecting generated assembly. */
static void micro_kernel(int64_t kc, const float * restrict ap, const float * restrict bp,
                          float * restrict c, int64_t ldc, int64_t mr, int64_t nr)
{
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    for (int64_t p = 0; p < kc; p++) {
        __m256 b0 = _mm256_load_ps(bp);
        __m256 b1 = _mm256_load_ps(bp + 8);

        __m256 a0 = _mm256_broadcast_ss(ap + 0);
        c00 = _mm256_fmadd_ps(a0, b0, c00); c01 = _mm256_fmadd_ps(a0, b1, c01);
        __m256 a1 = _mm256_broadcast_ss(ap + 1);
        c10 = _mm256_fmadd_ps(a1, b0, c10); c11 = _mm256_fmadd_ps(a1, b1, c11);
        __m256 a2 = _mm256_broadcast_ss(ap + 2);
        c20 = _mm256_fmadd_ps(a2, b0, c20); c21 = _mm256_fmadd_ps(a2, b1, c21);
        __m256 a3 = _mm256_broadcast_ss(ap + 3);
        c30 = _mm256_fmadd_ps(a3, b0, c30); c31 = _mm256_fmadd_ps(a3, b1, c31);
        __m256 a4 = _mm256_broadcast_ss(ap + 4);
        c40 = _mm256_fmadd_ps(a4, b0, c40); c41 = _mm256_fmadd_ps(a4, b1, c41);
        __m256 a5 = _mm256_broadcast_ss(ap + 5);
        c50 = _mm256_fmadd_ps(a5, b0, c50); c51 = _mm256_fmadd_ps(a5, b1, c51);

        ap += GEMM_MR;
        bp += GEMM_NR;
    }

    /* writeback: full tile uses aligned store, edge uses scalar */
    if (mr == GEMM_MR && nr == GEMM_NR) {
        #define STORE_ROW(row, lo, hi) \
            _mm256_store_ps(c + (row)*ldc,     _mm256_add_ps(lo, _mm256_load_ps(c + (row)*ldc))); \
            _mm256_store_ps(c + (row)*ldc + 8, _mm256_add_ps(hi, _mm256_load_ps(c + (row)*ldc + 8)));
        STORE_ROW(0, c00, c01); STORE_ROW(1, c10, c11);
        STORE_ROW(2, c20, c21); STORE_ROW(3, c30, c31);
        STORE_ROW(4, c40, c41); STORE_ROW(5, c50, c51);
        #undef STORE_ROW
    } else {
        /* edge tile: extract to aligned stack buffer, scalar write */
        float buf[GEMM_MR * GEMM_NR] __attribute__((aligned(64)));
        #define EXTRACT_ROW(row, lo, hi) \
            _mm256_store_ps(buf + (row)*GEMM_NR,     lo); \
            _mm256_store_ps(buf + (row)*GEMM_NR + 8, hi);
        EXTRACT_ROW(0, c00, c01); EXTRACT_ROW(1, c10, c11);
        EXTRACT_ROW(2, c20, c21); EXTRACT_ROW(3, c30, c31);
        EXTRACT_ROW(4, c40, c41); EXTRACT_ROW(5, c50, c51);
        #undef EXTRACT_ROW
        for (int64_t ii = 0; ii < mr; ii++)
            for (int64_t jj = 0; jj < nr; jj++)
                c[ii * ldc + jj] += buf[ii * GEMM_NR + jj];
    }
}

#else

/* generic micro-kernel using the SIMD abstraction layer.
   on NEON: 4x8 (8 accumulators). on scalar: 4x4. */
static void micro_kernel(int64_t kc, const float *ap, const float *bp,
                          float *c, int64_t ldc, int64_t mr, int64_t nr)
{
    /* NR/AX_VF32_WIDTH vectors per row */
    #define NVEC (GEMM_NR / AX_VF32_WIDTH)
    ax_vf32 acc[GEMM_MR][NVEC];
    for (int ii = 0; ii < GEMM_MR; ii++)
        for (int v = 0; v < NVEC; v++)
            acc[ii][v] = ax_vf32_zero();

    for (int64_t p = 0; p < kc; p++) {
        for (int v = 0; v < NVEC; v++) {
            ax_vf32 bv = ax_vf32_load(bp + v * AX_VF32_WIDTH);
            for (int ii = 0; ii < GEMM_MR; ii++) {
                ax_vf32 av = ax_vf32_set1(ap[ii]);
                acc[ii][v] = ax_vf32_fmadd(av, bv, acc[ii][v]);
            }
        }
        ap += GEMM_MR;
        bp += GEMM_NR;
    }

    /* writeback */
    if (mr == GEMM_MR && nr == GEMM_NR) {
        for (int ii = 0; ii < GEMM_MR; ii++)
            for (int v = 0; v < NVEC; v++) {
                float *cp = c + ii * ldc + v * AX_VF32_WIDTH;
                ax_vf32_store(cp, ax_vf32_add(ax_vf32_load(cp), acc[ii][v]));
            }
    } else {
        float buf[GEMM_MR * GEMM_NR] __attribute__((aligned(64)));
        for (int ii = 0; ii < GEMM_MR; ii++)
            for (int v = 0; v < NVEC; v++)
                ax_vf32_store(buf + ii * GEMM_NR + v * AX_VF32_WIDTH, acc[ii][v]);
        for (int64_t ii = 0; ii < mr; ii++)
            for (int64_t jj = 0; jj < nr; jj++)
                c[ii * ldc + jj] += buf[ii * GEMM_NR + jj];
    }
    #undef NVEC
}

#endif

static ax_status_t opt_gemm(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    if (a->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "gemm only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    if (a->ndim != 2 || b->ndim != 2 || out->ndim != 2) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "gemm requires 2d tensors");
        return AX_ERR_SHAPE_MISMATCH;
    }

    int64_t m = a->shape[0], k = a->shape[1], n = b->shape[1];
    if (b->shape[0] != k || out->shape[0] != m || out->shape[1] != n) {
        return ax_cpu_naive_ops.gemm(a, b, out);
    }

    int64_t na = validate_contig_f32(a);
    int64_t nb = validate_contig_f32(b);
    int64_t no_v = validate_contig_f32(out);
    if (na < 0 || nb < 0 || no_v < 0) {
        return ax_cpu_naive_ops.gemm(a, b, out);
    }

    /* for very small matrices, use simple loop (tiling overhead not worth it) */
    if (m * n * k < 4096) {
        const float *ad = raw_f32(a);
        const float *bd = raw_f32(b);
        float *od = raw_f32(out);
        memset(od, 0, (size_t)(m * n) * sizeof(float));
        for (int64_t i = 0; i < m; i++)
            for (int64_t p = 0; p < k; p++) {
                float a_ip = ad[i * k + p];
                for (int64_t j = 0; j < n; j++)
                    od[i * n + j] += a_ip * bd[p * n + j];
            }
        return AX_OK;
    }

    /* round up tile counts for packing */
    int64_t mc_padded = ((m + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;
    int64_t nc_padded = ((n + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;

    /* checked scratch allocation — bounded by AX_MAX_SCRATCH_BYTES */
    size_t pack_a_size = (size_t)GEMM_MC * (size_t)GEMM_KC * sizeof(float);
    size_t pack_b_size = (size_t)GEMM_NC * (size_t)GEMM_KC * sizeof(float);
    if (pack_a_size + pack_b_size > AX_MAX_SCRATCH_BYTES) {
        return ax_cpu_naive_ops.gemm(a, b, out);
    }

    float *pack_a_buf = (float *)ax_aligned_alloc(pack_a_size, 64);
    float *pack_b_buf = (float *)ax_aligned_alloc(pack_b_size, 64);
    if (!pack_a_buf || !pack_b_buf) {
        ax_aligned_free(pack_a_buf);
        ax_aligned_free(pack_b_buf);
        return ax_cpu_naive_ops.gemm(a, b, out);
    }

    const float *ad = raw_f32(a);
    const float *bd = raw_f32(b);
    float *od = raw_f32(out);
    memset(od, 0, (size_t)(m * n) * sizeof(float));

    /* main tiling loop: iterate over KC tiles, then MC, then NC */
    for (int64_t pc = 0; pc < k; pc += GEMM_KC) {
        int64_t kc = (pc + GEMM_KC <= k) ? GEMM_KC : (k - pc);

        for (int64_t ic = 0; ic < m; ic += GEMM_MC) {
            int64_t mc = (ic + GEMM_MC <= m) ? GEMM_MC : (m - ic);

            /* pack A panel [ic:ic+mc, pc:pc+kc] */
            int64_t mc_pack = ((mc + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;
            pack_a(ad + ic * k + pc, k, mc_pack, kc, mc, pack_a_buf);

            for (int64_t jc = 0; jc < n; jc += GEMM_NC) {
                int64_t nc = (jc + GEMM_NC <= n) ? GEMM_NC : (n - jc);

                /* pack B panel [pc:pc+kc, jc:jc+nc] */
                int64_t nc_pack = ((nc + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
                pack_b(bd + pc * n + jc, n, kc, nc_pack, nc, pack_b_buf);

                /* micro-kernel over MR x NR blocks */
                int64_t mc_round = ((mc + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;
                int64_t nc_round = ((nc + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;

                for (int64_t ir = 0; ir < mc_round; ir += GEMM_MR) {
                    int64_t mr = (ir + GEMM_MR <= mc) ? GEMM_MR : (mc - ir);
                    for (int64_t jr = 0; jr < nc_round; jr += GEMM_NR) {
                        int64_t nr = (jr + GEMM_NR <= nc) ? GEMM_NR : (nc - jr);

                        const float *ap = pack_a_buf + ir * kc;
                        const float *bp = pack_b_buf + jr * kc;
                        float *cp = od + (ic + ir) * n + (jc + jr);

                        micro_kernel(kc, ap, bp, cp, n, mr, nr);
                    }
                }
            }
        }
    }

    ax_aligned_free(pack_a_buf);
    ax_aligned_free(pack_b_buf);
    return AX_OK;
}


/* reductions */

static ax_status_t opt_sum(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    /* fast path: full reduction on contiguous tensor */
    if (axis == -1) {
        int64_t n = validate_contig_f32(in);
        int64_t no = validate_contig_f32(out);
        if (n < 0 || no < 0 || no != 1) return ax_cpu_naive_ops.sum(in, axis, out);
        const float *d = raw_f32(in);
        double total = 0.0; /* double accumulator for precision */
        for (int64_t i = 0; i < n; i++) total += (double)d[i];
        raw_f32(out)[0] = (float)total;
        return AX_OK;
    }
    return ax_cpu_naive_ops.sum(in, axis, out);
}

static ax_status_t opt_mean(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    if (axis == -1) {
        int64_t n = validate_contig_f32(in);
        int64_t no = validate_contig_f32(out);
        if (n < 0 || no < 0 || no != 1) return ax_cpu_naive_ops.mean(in, axis, out);
        const float *d = raw_f32(in);
        double total = 0.0;
        for (int64_t i = 0; i < n; i++) total += (double)d[i];
        raw_f32(out)[0] = (float)(total / (double)n);
        return AX_OK;
    }
    return ax_cpu_naive_ops.mean(in, axis, out);
}

static ax_status_t opt_max(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    if (axis == -1) {
        int64_t n = validate_contig_f32(in);
        int64_t no = validate_contig_f32(out);
        if (n < 0 || no < 0 || no != 1) return ax_cpu_naive_ops.max_op(in, axis, out);
        const float *d = raw_f32(in);
        float mx = -FLT_MAX;
        for (int64_t i = 0; i < n; i++) if (d[i] > mx) mx = d[i];
        raw_f32(out)[0] = mx;
        return AX_OK;
    }
    return ax_cpu_naive_ops.max_op(in, axis, out);
}

static ax_status_t opt_min(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    if (axis == -1) {
        int64_t n = validate_contig_f32(in);
        int64_t no = validate_contig_f32(out);
        if (n < 0 || no < 0 || no != 1) return ax_cpu_naive_ops.min_op(in, axis, out);
        const float *d = raw_f32(in);
        float mn = FLT_MAX;
        for (int64_t i = 0; i < n; i++) if (d[i] < mn) mn = d[i];
        raw_f32(out)[0] = mn;
        return AX_OK;
    }
    return ax_cpu_naive_ops.min_op(in, axis, out);
}


/* comparisons */

static ax_status_t opt_equal(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    int64_t n = validate_triple_same(a, b, out);
    if (n < 0) return ax_cpu_naive_ops.equal(a, b, out);
    const float *ad = raw_f32(a);
    const float *bd = raw_f32(b);
    float *od = raw_f32(out);
    for (int64_t i = 0; i < n; i++)
        od[i] = (ad[i] == bd[i]) ? 1.0f : 0.0f;
    return AX_OK;
}

static ax_status_t opt_greater(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    int64_t n = validate_triple_same(a, b, out);
    if (n < 0) return ax_cpu_naive_ops.greater(a, b, out);
    const float *ad = raw_f32(a);
    const float *bd = raw_f32(b);
    float *od = raw_f32(out);
    for (int64_t i = 0; i < n; i++)
        od[i] = (ad[i] > bd[i]) ? 1.0f : 0.0f;
    return AX_OK;
}


/* data movement */

static ax_status_t opt_fill(ax_tensor_t *t, double value) {
    int64_t n = validate_contig_f32(t);
    if (n < 0) return ax_cpu_naive_ops.fill(t, value);
    float v = (float)value;
    float *d = raw_f32(t);
    /* special case: fill with 0 uses memset (fast) */
    if (v == 0.0f) {
        memset(d, 0, (size_t)n * sizeof(float));
    } else {
        for (int64_t i = 0; i < n; i++) d[i] = v;
    }
    return AX_OK;
}

static ax_status_t opt_copy(const ax_tensor_t *src, ax_tensor_t *dst) {
    int64_t ns = validate_contig_f32(src);
    int64_t nd = validate_contig_f32(dst);
    if (ns < 0 || nd < 0 || ns != nd) return ax_cpu_naive_ops.copy(src, dst);
    memcpy(raw_f32(dst), raw_f32(src), (size_t)ns * sizeof(float));
    return AX_OK;
}


/* vtable registration */

const ax_backend_ops_t ax_cpu_opt_ops = {
    .name       = "cpu_opt",
    .add        = opt_add,
    .sub        = opt_sub,
    .mul        = opt_mul,
    .div_op     = opt_div_op,
    .neg        = opt_neg,
    .abs_op     = opt_abs_op,
    .exp_op     = opt_exp_op,
    .log_op     = opt_log_op,
    .sqrt_op    = opt_sqrt_op,
    .square     = opt_square,
    .add_scalar = opt_add_scalar,
    .mul_scalar = opt_mul_scalar,
    .gemm       = opt_gemm,
    .sum        = opt_sum,
    .mean       = opt_mean,
    .max_op     = opt_max,
    .min_op     = opt_min,
    .equal      = opt_equal,
    .greater    = opt_greater,
    .fill       = opt_fill,
    .copy       = opt_copy,
    .relu       = opt_relu,
    .sigmoid    = opt_sigmoid,
    .tanh_op    = opt_tanh_op,
};
