/* cpu_opt.c — optimized cpu backend with contiguous fast paths.
   falls back to cpu_naive for strided/broadcast cases.
   every op validates storage bounds before pointer access.

   runtime isa dispatch: this file can be compiled twice under the
   AX_CPU_ISA_DISPATCH build flag, once with -mavx2 -mfma and once
   without. the externally-visible symbols (the vtable and the tune
   init) get suffixed by AX_CPU_OPT_SUFFIX so the two object files
   don't collide when linked together. dispatch.c then picks between
   ax_cpu_opt_ops_avx2 and ax_cpu_opt_ops_scalar at ax_compute_init
   time via __builtin_cpu_supports. single-build users (the default)
   see the unchanged symbol names. */

#ifdef AX_CPU_OPT_SUFFIX
#define AX_PASTE2(a, b) a ## b
#define AX_PASTE(a, b)  AX_PASTE2(a, b)
#define AX_SYM(name)    AX_PASTE(name, AX_CPU_OPT_SUFFIX)
#else
#define AX_SYM(name)    name
#endif

#include "axiom/backend_ops.h"
#include "axiom/tensor.h"
#include "axiom/error.h"
#include "axiom/memory.h"
#include "simd_defs.h"
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdint.h>
#include <stdlib.h>
#ifndef AX_NO_STDIO
#include <stdio.h>
#endif
#ifdef __linux__
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* thread-local storage qualifier. on hosted builds, _Thread_local backs
   the gemm pack buffers and pack_b cache keys per worker thread. on
   baremetal / rtos targets (AX_SINGLE_THREADED) we drop it and rely on
   the single-threaded invariant — still correct because AX_SINGLE_THREADED
   implies AX_OPENMP=OFF so nothing ever races. */
#ifdef AX_SINGLE_THREADED
#define AX_TLS static
#else
#define AX_TLS static _Thread_local
#endif

/* log sink: fprintf(stderr, ...) normally, nothing on baremetal. keeps
   the autotuner / gemm-tile diagnostics from pulling in 20+ kb of stdio
   machinery on flash-constrained targets. */
#ifdef AX_NO_STDIO
#define AX_LOG(...) ((void)0)
#else
#define AX_LOG(...) fprintf(stderr, __VA_ARGS__)
#endif

/* threshold for parallelizing element-wise ops.
   below this, openmp fork-join overhead exceeds compute time. scales
   with thread count so more threads only kick in for proportionally
   more work. the 8k-per-thread constant matches measured fork-join
   overhead of ~5us on modern x86. */
#define AX_PAR_THRESHOLD_PER_THREAD 8192

#ifdef _OPENMP
static inline int64_t ax_par_threshold(void) {
    int nt = omp_in_parallel() ? 1 : omp_get_max_threads();
    if (nt < 1) nt = 1;
    return (int64_t)AX_PAR_THRESHOLD_PER_THREAD * nt;
}
#else
static inline int64_t ax_par_threshold(void) { return (int64_t)1 << 62; }
#endif

/* per-thread persistent pack buffers for GEMM — allocated once per thread,
   reused on every call. eliminates ~16 malloc/free per GEMM invocation.
   uses AX_TLS so single-threaded baremetal builds drop the _Thread_local. */
AX_TLS float *tl_pack_a_buf = NULL;  /* GEMM_MC * GEMM_KC floats */
AX_TLS float *tl_pack_b_buf = NULL;  /* GEMM_NC * GEMM_KC floats */

/* pack_b cache: skip re-packing B when the same tile is requested back-to-
   back. hit path is typical in backward passes where the same weight
   matrix is used twice (e.g. dY @ W for dX, then X^T @ dY for dW). the
   key is the exact set of inputs pack_b reads — bptr, ldb,
   (jc, pc, kc, nc_pack, nc) — plus the storage generation at pack time.

   the generation check is the correctness fix for stale cache hits. the
   cache only stored a raw pointer before, so if the buffer contents got
   mutated in place (optimizer step on a weight, for instance) between
   two gemm calls sharing the same B pointer, the cache would return the
   previous pack and the backward pass would silently compute on stale
   weights. now every in-place write site bumps storage->generation
   (dispatch.c wrappers, tensor.c host helpers, optim.c, ops.c inplace
   activations) and this cache rejects the hit when the generation
   doesn't match.

   a NULL bptr means the cache is empty / invalidated. */
AX_TLS const float *tl_pack_b_cache_bptr = NULL;
AX_TLS uint64_t tl_pack_b_cache_gen = 0;
AX_TLS int64_t tl_pack_b_cache_ldb = 0;
AX_TLS int64_t tl_pack_b_cache_jc = 0;
AX_TLS int64_t tl_pack_b_cache_pc = 0;
AX_TLS int64_t tl_pack_b_cache_kc = 0;
AX_TLS int64_t tl_pack_b_cache_nc = 0;
AX_TLS int64_t tl_pack_b_cache_ncp = 0;

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

#ifdef _OPENMP
#define AX_OMP_PAR_FOR_IF(n) _Pragma("omp parallel for schedule(static) if((n) > ax_par_threshold())")
#else
/* non-omp build: the alias variable used as the macro arg stays live
   (casts to void) so callers don't trip -Wunused-variable. */
#define AX_OMP_PAR_FOR_IF(n) (void)(n);
#endif

#define DEFINE_OPT_BINOP(name, simd_expr, scalar_expr, naive_fn) \
static ax_status_t opt_##name(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) { \
    int64_t n = validate_triple_same(a, b, out); \
    if (n < 0) return ax_cpu_naive_ops.naive_fn(a, b, out); \
    const float *ad = raw_f32(a); \
    const float *bd = raw_f32(b); \
    float *od = raw_f32(out); \
    int64_t vec_end = n - (n % AX_VF32_WIDTH); \
    AX_OMP_PAR_FOR_IF(n) \
    for (int64_t i = 0; i < vec_end; i += AX_VF32_WIDTH) { \
        ax_vf32 va = ax_vf32_load(ad + i); \
        ax_vf32 vb = ax_vf32_load(bd + i); \
        ax_vf32_store(od + i, simd_expr); \
    } \
    for (int64_t i = vec_end; i < n; i++) { \
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
    int64_t vec_end = n - (n % AX_VF32_WIDTH); \
    AX_OMP_PAR_FOR_IF(n) \
    for (int64_t i = 0; i < vec_end; i += AX_VF32_WIDTH) { \
        ax_vf32 v = ax_vf32_load(id + i); \
        ax_vf32_store(od + i, simd_expr); \
    } \
    for (int64_t i = vec_end; i < n; i++) { \
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


/* scalar ops — vectorized */

static ax_status_t opt_add_scalar(const ax_tensor_t *in, double scalar, ax_tensor_t *out) {
    int64_t ni = validate_contig_f32(in);
    int64_t no = validate_contig_f32(out);
    if (ni < 0 || no < 0 || ni != no) return ax_cpu_naive_ops.add_scalar(in, scalar, out);
    const float *id = raw_f32(in);
    float *od = raw_f32(out);
    float s = (float)scalar;
    ax_vf32 vs = ax_vf32_set1(s);
    int64_t n = ni; /* alias so AX_OMP_PAR_FOR_IF(n) resolves correctly */
    int64_t vec_end = n - (n % AX_VF32_WIDTH);
    AX_OMP_PAR_FOR_IF(n)
    for (int64_t i = 0; i < vec_end; i += AX_VF32_WIDTH)
        ax_vf32_store(od + i, ax_vf32_add(ax_vf32_load(id + i), vs));
    for (int64_t i = vec_end; i < n; i++)
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
    ax_vf32 vs = ax_vf32_set1(s);
    int64_t n = ni; /* alias so AX_OMP_PAR_FOR_IF(n) resolves correctly */
    int64_t vec_end = n - (n % AX_VF32_WIDTH);
    AX_OMP_PAR_FOR_IF(n)
    for (int64_t i = 0; i < vec_end; i += AX_VF32_WIDTH)
        ax_vf32_store(od + i, ax_vf32_mul(ax_vf32_load(id + i), vs));
    for (int64_t i = vec_end; i < n; i++)
        od[i] = id[i] * s;
    return AX_OK;
}


/* tiled gemm: cache-blocked with panel packing (BLIS-style).
   6x16 micro-kernel on AVX2: 12 YMM accumulators + 2 B loads + 1 A broadcast
   = 15 of 16 registers. no spills, near-peak FMA throughput. */

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

/* macro-kernel tile sizes. compile-time defaults come from the build
   profile via the AX_GEMM_DEFAULT_MC / NC / KC preprocessor defines
   set in cmake profiles: 72/256/256 for desktop, 48/128/128 for
   embedded-linux, 24/32/64 for baremetal.
   at runtime AX_GEMM_MC/NC/KC env vars override these (hosted targets
   only). the init hook also logs a warning if the pack_b panel exceeds
   detected l2 so users know to tune. */
#ifndef AX_GEMM_DEFAULT_MC
#define AX_GEMM_DEFAULT_MC 72
#endif
#ifndef AX_GEMM_DEFAULT_NC
#define AX_GEMM_DEFAULT_NC 256
#endif
#ifndef AX_GEMM_DEFAULT_KC
#define AX_GEMM_DEFAULT_KC 256
#endif

static int64_t GEMM_MC = AX_GEMM_DEFAULT_MC;
static int64_t GEMM_NC = AX_GEMM_DEFAULT_NC;
static int64_t GEMM_KC = AX_GEMM_DEFAULT_KC;

static void ax_gemm_read_env(const char *name, int64_t *slot, int64_t multiple_of) {
    const char *s = getenv(name);
    if (!s || !*s) return;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s || v <= 0) return;
    if (multiple_of > 1) v = (v / multiple_of) * multiple_of;
    if (v <= 0) return;
    *slot = (int64_t)v;
}

/* one-shot init: read env overrides + emit l2 size warning.
   called both from an attribute((constructor)) (hosted builds, runs
   before main) AND explicitly from ax_compute_init as a fallback for
   baremetal crt0 that doesn't walk .init_array. guarded by a flag so
   a second invocation is a no-op. */
static bool ax_cpu_opt_init_done = false;

static void ax_cpu_opt_init_impl(void) {
    if (ax_cpu_opt_init_done) return;
    ax_cpu_opt_init_done = true;

    ax_gemm_read_env("AX_GEMM_MC", &GEMM_MC, GEMM_MR);
    ax_gemm_read_env("AX_GEMM_NC", &GEMM_NC, GEMM_NR);
    ax_gemm_read_env("AX_GEMM_KC", &GEMM_KC, 1);

#if defined(__linux__) && !defined(AX_NO_STDIO)
    long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (l2 > 0) {
        size_t pack_b_bytes = (size_t)GEMM_NC * (size_t)GEMM_KC * sizeof(float);
        if ((long)pack_b_bytes > l2) {
            AX_LOG("axiom: gemm pack_b buffer %zu kB exceeds detected l2 %ld kB; "
                   "consider setting AX_GEMM_KC / AX_GEMM_NC smaller on this target\n",
                   pack_b_bytes / 1024, l2 / 1024);
        }
    }
#endif
}

/* public entry point — called from ax_compute_init() in dispatch.c so
   baremetal targets without .init_array support still apply the env
   overrides. idempotent: second call is a no-op. suffixed variant
   under AX_CPU_ISA_DISPATCH so both isa variants coexist. */
void AX_SYM(ax_cpu_opt_tune_init)(void) {
    ax_cpu_opt_init_impl();
}

/* constructor: only emit under the single-build path. under
   AX_CPU_ISA_DISPATCH we drive init explicitly from dispatch.c
   (because only one of the two variants ends up being the active
   backend — registering both constructors is redundant and the
   scalar variant would overwrite the autotuned tile sizes the
   avx2 variant picked). */
#if !defined(AX_NO_CONSTRUCTORS) && !defined(AX_CPU_OPT_SUFFIX) \
    && (defined(__GNUC__) || defined(__clang__))
static __attribute__((constructor)) void ax_cpu_opt_ctor(void) {
    ax_cpu_opt_init_impl();
}
#endif

/* lazily allocate this thread's pack_a and pack_b buffers (once per thread) */
static bool ensure_tl_pack_bufs(void) {
    if (!tl_pack_a_buf)
        tl_pack_a_buf = (float *)ax_aligned_alloc((size_t)GEMM_MC * (size_t)GEMM_KC * sizeof(float), 64);
    if (!tl_pack_b_buf)
        tl_pack_b_buf = (float *)ax_aligned_alloc((size_t)GEMM_NC * (size_t)GEMM_KC * sizeof(float), 64);
    return tl_pack_a_buf && tl_pack_b_buf;
}

/* forward decl — pack_b is defined further down */
static void pack_b(const float *b, int64_t ldb, int64_t kc, int64_t nc,
                    int64_t n_remain, float *packed);

/* invalidate pack_b cache — call when any pack_b buffer contents might differ
   from what the cache key currently describes. */
static inline void pack_b_cache_invalidate(void) {
    tl_pack_b_cache_bptr = NULL;
}

/* pack_b with cache: if the last pack_b into tl_pack_b_buf used the exact
   same (bptr, ldb, jc, pc, kc, nc, nc_pack) AND the storage generation
   hasn't changed since then, the buffer is still valid and we can skip
   the copy. otherwise re-pack and update the cache key. 'tile_bptr' must
   be (bd + pc*ldb + jc) — the actual address pack_b reads from.
   'b_gen' is the storage generation counter of the B tensor at call
   time; the caller passes b->storage->generation. */
static inline void pack_b_cached(const float *tile_bptr, uint64_t b_gen,
                                  int64_t ldb,
                                  int64_t kc, int64_t nc_pack, int64_t nc,
                                  int64_t jc, int64_t pc)
{
    if (tl_pack_b_cache_bptr == tile_bptr
        && tl_pack_b_cache_gen == b_gen
        && tl_pack_b_cache_ldb == ldb
        && tl_pack_b_cache_jc  == jc
        && tl_pack_b_cache_pc  == pc
        && tl_pack_b_cache_kc  == kc
        && tl_pack_b_cache_nc  == nc
        && tl_pack_b_cache_ncp == nc_pack) {
        return;  /* hit — buffer already contains exactly this tile */
    }
    pack_b(tile_bptr, ldb, kc, nc_pack, nc, tl_pack_b_buf);
    tl_pack_b_cache_bptr = tile_bptr;
    tl_pack_b_cache_gen  = b_gen;
    tl_pack_b_cache_ldb  = ldb;
    tl_pack_b_cache_jc   = jc;
    tl_pack_b_cache_pc   = pc;
    tl_pack_b_cache_kc   = kc;
    tl_pack_b_cache_nc   = nc;
    tl_pack_b_cache_ncp  = nc_pack;
}

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

/* pack an MC x KC panel of A^T, where the physical source a_src is
   stored [K, M] row-major with lda == M. A^T[i,p] = a_src[p*lda + i].
   produces the same layout as pack_a so the shared micro_kernel works. */
static void pack_a_t(const float *a_src, int64_t lda_src, int64_t mc, int64_t kc,
                      int64_t m_remain, float *packed)
{
    for (int64_t i = 0; i < mc; i += GEMM_MR) {
        int64_t mr = (i + GEMM_MR <= m_remain) ? GEMM_MR : (m_remain > i ? m_remain - i : 0);
        for (int64_t p = 0; p < kc; p++) {
            for (int64_t ii = 0; ii < GEMM_MR; ii++) {
                if (ii < mr)
                    packed[ii] = a_src[p * lda_src + (i + ii)];
                else
                    packed[ii] = 0.0f;
            }
            packed += GEMM_MR;
        }
    }
}

/* pack a KC x NC panel of B^T, where the physical source b_src is
   stored [N, K] row-major with ldb == K. B^T[p,j] = b_src[j*ldb + p].
   produces the same layout as pack_b. */
static void pack_b_t(const float *b_src, int64_t ldb_src, int64_t kc, int64_t nc,
                      int64_t n_remain, float *packed)
{
    /* b_src is [N, K] (row-major). pack as if transposed:
       packed[p * NR + jj] = b_src[(j + jj) * ldb + p].

       the naive approach reads column-by-column with stride=ldb between
       consecutive elements — one L1 miss per element for large ldb.

       avx2 optimisation: process 8×8 sub-blocks where we load 8
       consecutive floats from 8 different rows (8 sequential reads),
       transpose in-register using vpunpcklps/vunpckhps/vperm2f128, and
       write 8 transposed rows sequentially. this converts the NR strided
       reads into NR/8 groups of 8 sequential reads, dramatically
       improving cache utilisation. */

#if defined(AX_SIMD_AVX2)
    for (int64_t j = 0; j < nc; j += GEMM_NR) {
        int64_t nr = (j + GEMM_NR <= n_remain) ? GEMM_NR : (n_remain > j ? n_remain - j : 0);

        /* fast path: full NR-wide strip with K >= 8, use 8×8 block transpose.
           NR=16 means two groups of 8 rows, each processed in 8-column chunks. */
        if (nr == GEMM_NR) {
            int64_t p = 0;
            for (; p + 8 <= kc; p += 8) {
                /* process rows j..j+7 and j+8..j+15 */
                for (int g = 0; g < 2; g++) {
                    int64_t base_row = j + g * 8;
                    /* load 8 floats from each of 8 rows (sequential within each row) */
                    __m256 r0 = _mm256_loadu_ps(b_src + (base_row + 0) * ldb_src + p);
                    __m256 r1 = _mm256_loadu_ps(b_src + (base_row + 1) * ldb_src + p);
                    __m256 r2 = _mm256_loadu_ps(b_src + (base_row + 2) * ldb_src + p);
                    __m256 r3 = _mm256_loadu_ps(b_src + (base_row + 3) * ldb_src + p);
                    __m256 r4 = _mm256_loadu_ps(b_src + (base_row + 4) * ldb_src + p);
                    __m256 r5 = _mm256_loadu_ps(b_src + (base_row + 5) * ldb_src + p);
                    __m256 r6 = _mm256_loadu_ps(b_src + (base_row + 6) * ldb_src + p);
                    __m256 r7 = _mm256_loadu_ps(b_src + (base_row + 7) * ldb_src + p);

                    /* 8×8 in-register transpose.
                       step 1: interleave pairs of 32-bit floats */
                    __m256 t0 = _mm256_unpacklo_ps(r0, r1);
                    __m256 t1 = _mm256_unpackhi_ps(r0, r1);
                    __m256 t2 = _mm256_unpacklo_ps(r2, r3);
                    __m256 t3 = _mm256_unpackhi_ps(r2, r3);
                    __m256 t4 = _mm256_unpacklo_ps(r4, r5);
                    __m256 t5 = _mm256_unpackhi_ps(r4, r5);
                    __m256 t6 = _mm256_unpacklo_ps(r6, r7);
                    __m256 t7 = _mm256_unpackhi_ps(r6, r7);

                    /* step 2: interleave pairs of 64-bit groups */
                    r0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(t0), _mm256_castps_pd(t2)));
                    r1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(t0), _mm256_castps_pd(t2)));
                    r2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(t1), _mm256_castps_pd(t3)));
                    r3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(t1), _mm256_castps_pd(t3)));
                    r4 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(t4), _mm256_castps_pd(t6)));
                    r5 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(t4), _mm256_castps_pd(t6)));
                    r6 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(t5), _mm256_castps_pd(t7)));
                    r7 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(t5), _mm256_castps_pd(t7)));

                    /* step 3: swap 128-bit lanes */
                    t0 = _mm256_permute2f128_ps(r0, r4, 0x20);
                    t1 = _mm256_permute2f128_ps(r1, r5, 0x20);
                    t2 = _mm256_permute2f128_ps(r2, r6, 0x20);
                    t3 = _mm256_permute2f128_ps(r3, r7, 0x20);
                    t4 = _mm256_permute2f128_ps(r0, r4, 0x31);
                    t5 = _mm256_permute2f128_ps(r1, r5, 0x31);
                    t6 = _mm256_permute2f128_ps(r2, r6, 0x31);
                    t7 = _mm256_permute2f128_ps(r3, r7, 0x31);

                    /* store: each t[i] holds 8 values from column (p+i) of the
                       original B, across 8 rows. write them into the packed layout
                       where the NR=16 slot for this group starts at packed[(p+i)*NR + g*8]. */
                    _mm256_storeu_ps(packed + (p + 0) * GEMM_NR + g * 8, t0);
                    _mm256_storeu_ps(packed + (p + 1) * GEMM_NR + g * 8, t1);
                    _mm256_storeu_ps(packed + (p + 2) * GEMM_NR + g * 8, t2);
                    _mm256_storeu_ps(packed + (p + 3) * GEMM_NR + g * 8, t3);
                    _mm256_storeu_ps(packed + (p + 4) * GEMM_NR + g * 8, t4);
                    _mm256_storeu_ps(packed + (p + 5) * GEMM_NR + g * 8, t5);
                    _mm256_storeu_ps(packed + (p + 6) * GEMM_NR + g * 8, t6);
                    _mm256_storeu_ps(packed + (p + 7) * GEMM_NR + g * 8, t7);
                }
            }
            /* scalar tail for remaining K columns */
            for (; p < kc; p++) {
                for (int64_t jj = 0; jj < GEMM_NR; jj++)
                    packed[p * GEMM_NR + jj] = b_src[(j + jj) * ldb_src + p];
            }
            packed += kc * GEMM_NR;
            continue;
        }

        /* edge strip: NR not full — scalar fallback with zero padding */
        for (int64_t p = 0; p < kc; p++) {
            for (int64_t jj = 0; jj < GEMM_NR; jj++) {
                if (jj < nr)
                    packed[jj] = b_src[(j + jj) * ldb_src + p];
                else
                    packed[jj] = 0.0f;
            }
            packed += GEMM_NR;
        }
    }
#else
    /* generic scalar fallback */
    for (int64_t j = 0; j < nc; j += GEMM_NR) {
        int64_t nr = (j + GEMM_NR <= n_remain) ? GEMM_NR : (n_remain > j ? n_remain - j : 0);
        for (int64_t p = 0; p < kc; p++) {
            for (int64_t jj = 0; jj < GEMM_NR; jj++) {
                if (jj < nr)
                    packed[jj] = b_src[(j + jj) * ldb_src + p];
                else
                    packed[jj] = 0.0f;
            }
            packed += GEMM_NR;
        }
    }
#endif
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

    /* prefetch output C into L1 before the compute loop so the writeback
       step doesn't stall on L2 latency. use read hint (0) not write
       hint (1) because prefetchw can segfault on some os/memory configs. */
    for (int64_t row = 0; row < mr; row++) {
        __builtin_prefetch(c + row * ldc, 0, 3);
        if (nr > 8) __builtin_prefetch(c + row * ldc + 8, 0, 3);
    }

    /* prefetch distance 4 iterations ahead (~96 B for A, ~256 B for B).
       tighter than 8 to avoid evicting hot data from L1 on small kc. */
    for (int64_t p = 0; p < kc; p++) {
        __builtin_prefetch(ap + 4 * GEMM_MR, 0, 3);
        __builtin_prefetch(bp + 4 * GEMM_NR, 0, 3);

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

    /* writeback: C matrix may not be aligned at tile boundaries, use unaligned ops */
    if (mr == GEMM_MR && nr == GEMM_NR) {
        #define STORE_ROW(row, lo, hi) \
            _mm256_storeu_ps(c + (row)*ldc,     _mm256_add_ps(lo, _mm256_loadu_ps(c + (row)*ldc))); \
            _mm256_storeu_ps(c + (row)*ldc + 8, _mm256_add_ps(hi, _mm256_loadu_ps(c + (row)*ldc + 8)));
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
   on NEON: 4x8 (8 accumulators). on scalar: 4x4.
   uses loadu/storeu for safe unaligned writeback at tile edges. */
static void micro_kernel(int64_t kc, const float *ap, const float *bp,
                          float *c, int64_t ldc, int64_t mr, int64_t nr)
{
    /* NR/AX_VF32_WIDTH vectors per row */
    #define NVEC (GEMM_NR / AX_VF32_WIDTH)
    ax_vf32 acc[GEMM_MR][NVEC];
    for (int ii = 0; ii < GEMM_MR; ii++)
        for (int v = 0; v < NVEC; v++)
            acc[ii][v] = ax_vf32_zero();

    /* prefetch 8 iterations ahead keeps both packed panels warm in l1
       without evicting current lines. portable via __builtin_prefetch. */
    for (int64_t p = 0; p < kc; p++) {
        __builtin_prefetch(ap + 8 * GEMM_MR, 0, 3);
        __builtin_prefetch(bp + 8 * GEMM_NR, 0, 3);

        for (int v = 0; v < NVEC; v++) {
            ax_vf32 bv = ax_vf32_loadu(bp + v * AX_VF32_WIDTH);
            for (int ii = 0; ii < GEMM_MR; ii++) {
                ax_vf32 av = ax_vf32_set1(ap[ii]);
                acc[ii][v] = ax_vf32_fmadd(av, bv, acc[ii][v]);
            }
        }
        ap += GEMM_MR;
        bp += GEMM_NR;
    }

    /* writeback: C may not be aligned at tile boundaries */
    if (mr == GEMM_MR && nr == GEMM_NR) {
        for (int ii = 0; ii < GEMM_MR; ii++)
            for (int v = 0; v < NVEC; v++) {
                float *cp = c + ii * ldc + v * AX_VF32_WIDTH;
                ax_vf32_storeu(cp, ax_vf32_add(ax_vf32_loadu(cp), acc[ii][v]));
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

/* compute an effective NC that ensures at least max_threads JC tiles
   when the default GEMM_NC would leave threads idle. rounds to GEMM_NR. */
static inline int64_t ax_adaptive_nc(int64_t n, int max_threads) {
    int64_t nc = GEMM_NC;
    int64_t tiles = (n + nc - 1) / nc;
    if (tiles < (int64_t)max_threads && n >= (int64_t)max_threads * GEMM_NR) {
        nc = ((n / max_threads + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
        if (nc < GEMM_NR) nc = GEMM_NR;
        if (nc > GEMM_NC) nc = GEMM_NC;
    }
    return nc;
}

/* adaptive KC: when K is small enough to fit in a single tile, use K
   directly as KC to avoid the overhead of multiple K-tile iterations
   (each tile re-packs A and B). for large K, cap at the L2 budget.
   the L2 budget constraint: pack_a (MC×KC) + pack_b (NC×KC) must fit.
   for MC=72, NC=256: (72+256)×KC×4 ≤ ~768 KB → KC ≤ ~600.
   use 512 as a safe maximum that fits comfortably in any L2 ≥ 1 MB. */
static inline int64_t ax_adaptive_kc(int64_t k) {
    if (k <= 512) return k;  /* single tile: no re-packing */
    return GEMM_KC;          /* default tiled path */
}

static ax_status_t opt_gemm(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    if (!a || !b || !out) {
        ax_err_set(AX_ERR_NULL_ARG, "gemm: NULL tensor");
        return AX_ERR_NULL_ARG;
    }
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

    /* pack_b cache invalidation: if A's data pointer equals the cached B pointer
       range, a prior call's pack_b data no longer describes the current B and must
       be re-packed. we only hold a base tile pointer, so also invalidate when the
       output buffer equals it (aliased write). the usual backward pattern
       (same B, different A and out) is still a clean cache hit. */
    const float *a_raw = raw_f32(a);
    const float *b_raw = raw_f32(b);
    float *o_raw = raw_f32(out);
    if (tl_pack_b_cache_bptr != NULL) {
        /* bptr points into b_raw region; if a or out aliases b's storage or the
           role of a and b swapped, drop the cache. cheap conservative check: any
           pointer overlap of a or out with the cached tile invalidates it. */
        if ((const float *)a_raw == tl_pack_b_cache_bptr
            || (const float *)o_raw == tl_pack_b_cache_bptr) {
            pack_b_cache_invalidate();
        }
    }

    /* SIMD small/medium path. tiled BLIS overhead dominates below ~100k FLOPs;
       a straight vectorized AXPY (C[i,j] += a_ip * B[p,j]) with SIMD inner on j
       runs faster for small-to-medium shapes. */
    if (m * n * k < 100000) {
        const float *ad = a_raw;
        const float *bd = b_raw;
        float *od = o_raw;
        memset(od, 0, (size_t)(m * n) * sizeof(float));
        int64_t vec_end = n - (n % AX_VF32_WIDTH);
        for (int64_t i = 0; i < m; i++) {
            float *oi = od + i * n;
            const float *ai = ad + i * k;
            for (int64_t p = 0; p < k; p++) {
                float a_ip = ai[p];
                const float *bp = bd + p * n;
                ax_vf32 va = ax_vf32_set1(a_ip);
                int64_t j = 0;
                for (; j < vec_end; j += AX_VF32_WIDTH) {
                    ax_vf32 vo = ax_vf32_loadu(oi + j);
                    ax_vf32 vb = ax_vf32_loadu(bp + j);
                    ax_vf32_storeu(oi + j, ax_vf32_fmadd(va, vb, vo));
                }
                for (; j < n; j++)
                    oi[j] += a_ip * bp[j];
            }
        }
        /* this path writes nothing to tl_pack_b_buf, so the cache remains valid. */
        return AX_OK;
    }

    /* per-thread pack buffers — guarded against nested parallel regions
       (conv2d batch loop already runs in a parallel region). */
    int max_threads = 1;
    #ifdef _OPENMP
    if (!omp_in_parallel()) max_threads = omp_get_max_threads();
    #endif

    int64_t nc_eff = ax_adaptive_nc(n, max_threads);
    int64_t n_jc_tiles = (n + nc_eff - 1) / nc_eff;
    int64_t n_ic_tiles = (m + GEMM_MC - 1) / GEMM_MC;

    /* parallelism strategy:
       - JC (column strips): each thread takes a JC tile, needs its own pack_b.
         best when n is large (n_jc_tiles >= 2).
       - IC (row strips): pack_b is done once, threads split IC tiles, each needs
         its own pack_a. best when n is small (e.g. MNIST: n=128 → jc_tiles=1).
       - Fine: when neither M nor N is big enough for tile-level splitting, pack
         A and B once, then parallelize the inner (ir, jr) micro-kernel grid. This
         covers the typical "narrow forward dense GEMM" case (e.g., m=batch=64,
         n=hidden=256, k=3136 — m<MC=72 and n=NC=256 so both tile counts are 1,
         but m has ~11 MR-rows of independent work). */
    int64_t n_mr_tiles = (m + GEMM_MR - 1) / GEMM_MR;
    int64_t n_nr_tiles_first_jc = ((n < GEMM_NC ? n : GEMM_NC) + GEMM_NR - 1) / GEMM_NR;
    int64_t fine_units = n_mr_tiles * n_nr_tiles_first_jc;

    bool use_jc_par = (max_threads > 1) && (n_jc_tiles >= 2);
    bool use_ic_par = !use_jc_par && (max_threads > 1) && (n_ic_tiles >= 2);
    /* fine-grained parallel is only worth it when there's enough total work to
       amortize OMP fork-join overhead AND enough work units per thread. ~1M
       FLOPs is a conservative threshold (< 50us serial). */
    int64_t total_flops = m * n * k;
    bool use_fine_par = !use_jc_par && !use_ic_par && (max_threads > 1)
                        && (m <= GEMM_MC) && (fine_units >= 4)
                        && (total_flops > 1000000);

    int gemm_threads = 1;
    if (use_jc_par) {
        gemm_threads = (int)(n_jc_tiles < (int64_t)max_threads ? n_jc_tiles : (int64_t)max_threads);
    } else if (use_ic_par) {
        gemm_threads = (int)(n_ic_tiles < (int64_t)max_threads ? n_ic_tiles : (int64_t)max_threads);
    } else if (use_fine_par) {
        gemm_threads = (int)(fine_units < (int64_t)max_threads ? fine_units : (int64_t)max_threads);
    }
    (void)gemm_threads; /* only read by num_threads() in the omp pragmas below */

    /* ensure the calling thread has its pack buffers (serial init before parallel region) */
    if (!ensure_tl_pack_bufs())
        return ax_cpu_naive_ops.gemm(a, b, out);

    const float *ad = a_raw;
    const float *bd = b_raw;
    float *od = o_raw;
    memset(od, 0, (size_t)(m * n) * sizeof(float));

    if (use_jc_par) {
        /* JC parallel: each thread owns a column strip of C and its own pack buffers.
           Thread-local buffers — lazily allocated once per thread, reused every call.
           Writes to disjoint columns → no synchronization needed. */
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(gemm_threads) schedule(static)
        #endif
        for (int64_t jct = 0; jct < n_jc_tiles; jct++) {
            /* each thread lazily inits its own TLS pack buffers on first use */
            ensure_tl_pack_bufs();
            float *pack_a_buf = tl_pack_a_buf;
            float *pack_b_buf = tl_pack_b_buf;
            if (!pack_a_buf || !pack_b_buf) continue;

            int64_t jc = jct * nc_eff;
            int64_t nc = (jc + nc_eff <= n) ? nc_eff : (n - jc);
            int64_t nc_pack = ((nc + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;

            for (int64_t pc = 0; pc < k; pc += GEMM_KC) {
                int64_t kc = (pc + GEMM_KC <= k) ? GEMM_KC : (k - pc);
                pack_b_cached(bd + pc * n + jc, b->storage->generation, n, kc, nc_pack, nc, jc, pc);

                for (int64_t ic = 0; ic < m; ic += GEMM_MC) {
                    int64_t mc = (ic + GEMM_MC <= m) ? GEMM_MC : (m - ic);
                    int64_t mc_pack = ((mc + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;
                    pack_a(ad + ic * k + pc, k, mc_pack, kc, mc, pack_a_buf);

                    for (int64_t ir = 0; ir < mc_pack; ir += GEMM_MR) {
                        int64_t mr = (ir + GEMM_MR <= mc) ? GEMM_MR : (mc - ir);
                        for (int64_t jr = 0; jr < nc_pack; jr += GEMM_NR) {
                            int64_t nr = (jr + GEMM_NR <= nc) ? GEMM_NR : (nc - jr);
                            micro_kernel(kc, pack_a_buf + ir * kc, pack_b_buf + jr * kc,
                                         od + (ic + ir) * n + (jc + jr), n, mr, nr);
                        }
                    }
                }
            }
        }
    } else if (use_fine_par) {
        /* Fine-grained parallel: pack A and B serially (small data), then
           parallelize over the (ir, jr) micro-kernel grid using collapse(2).
           Each (ir, jr) writes a disjoint MR x NR block of C. Used when neither
           m nor n is big enough for MC/NC-level tile splitting. */
        float *main_pack_b = tl_pack_b_buf;
        float *main_pack_a = tl_pack_a_buf;

        for (int64_t jc = 0; jc < n; jc += nc_eff) {
            int64_t nc = (jc + nc_eff <= n) ? nc_eff : (n - jc);
            int64_t nc_pack = ((nc + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;

            for (int64_t pc = 0; pc < k; pc += GEMM_KC) {
                int64_t kc = (pc + GEMM_KC <= k) ? GEMM_KC : (k - pc);
                int64_t mc = m;  /* whole m fits since m <= GEMM_MC */
                int64_t mc_pack = ((mc + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;

                /* fine-parallel: pack_b target is main_pack_b == tl_pack_b_buf,
                   same target as pack_b_cached. */
                pack_b_cached(bd + pc * n + jc, b->storage->generation, n, kc, nc_pack, nc, jc, pc);
                pack_a(ad + pc, k, mc_pack, kc, mc, main_pack_a);

                int64_t ir_tiles = mc_pack / GEMM_MR;
                int64_t jr_tiles = nc_pack / GEMM_NR;

                #ifdef _OPENMP
                #pragma omp parallel for num_threads(gemm_threads) schedule(static) collapse(2)
                #endif
                for (int64_t irt = 0; irt < ir_tiles; irt++) {
                    for (int64_t jrt = 0; jrt < jr_tiles; jrt++) {
                        int64_t ir = irt * GEMM_MR;
                        int64_t jr = jrt * GEMM_NR;
                        int64_t mr = (ir + GEMM_MR <= mc) ? GEMM_MR : (mc - ir);
                        int64_t nr = (jr + GEMM_NR <= nc) ? GEMM_NR : (nc - jr);
                        micro_kernel(kc, main_pack_a + ir * kc, main_pack_b + jr * kc,
                                     od + ir * n + (jc + jr), n, mr, nr);
                    }
                }
            }
        }
    } else {
        /* IC parallel (or serial): pack_b once per (jc, pc) into the calling thread's
           TLS pack_b buffer, pass as a raw pointer into the parallel region (read-only
           from worker threads — safe, TLS buffer is ordinary memory).
           Serial path: gemm_threads==1, pragma is a no-op. */
        float *main_pack_b = tl_pack_b_buf;  /* packed by serial outer loop */

        for (int64_t jc = 0; jc < n; jc += nc_eff) {
            int64_t nc = (jc + nc_eff <= n) ? nc_eff : (n - jc);
            int64_t nc_pack = ((nc + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;

            for (int64_t pc = 0; pc < k; pc += GEMM_KC) {
                int64_t kc = (pc + GEMM_KC <= k) ? GEMM_KC : (k - pc);
                /* serial/IC path: pack_b target is tl_pack_b_buf of the calling thread. */
                pack_b_cached(bd + pc * n + jc, b->storage->generation, n, kc, nc_pack, nc, jc, pc);
                const float *pack_b_buf = main_pack_b;  /* shared read-only in parallel region */

                #ifdef _OPENMP
                #pragma omp parallel for num_threads(gemm_threads) schedule(static) if(use_ic_par)
                #endif
                for (int64_t ict = 0; ict < n_ic_tiles; ict++) {
                    ensure_tl_pack_bufs();
                    float *pack_a_buf = tl_pack_a_buf;
                    if (!pack_a_buf) continue;

                    int64_t ic = ict * GEMM_MC;
                    int64_t mc = (ic + GEMM_MC <= m) ? GEMM_MC : (m - ic);
                    int64_t mc_pack = ((mc + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;
                    pack_a(ad + ic * k + pc, k, mc_pack, kc, mc, pack_a_buf);

                    for (int64_t ir = 0; ir < mc_pack; ir += GEMM_MR) {
                        int64_t mr = (ir + GEMM_MR <= mc) ? GEMM_MR : (mc - ir);
                        for (int64_t jr = 0; jr < nc_pack; jr += GEMM_NR) {
                            int64_t nr = (jr + GEMM_NR <= nc) ? GEMM_NR : (nc - jr);
                            micro_kernel(kc, pack_a_buf + ir * kc, pack_b_buf + jr * kc,
                                         od + (ic + ir) * n + (jc + jr), n, mr, nr);
                        }
                    }
                }
            }
        }
    }

    return AX_OK;
}


/* implicit im2col conv gemm.

   computes out[C_out, M] = weight[C_out, K] @ im2col(input)[K, M]
   for a single sample [C_in, H, W], without ever materializing the im2col
   matrix. reuses pack_a and the micro_kernel; only pack_b is replaced by
   pack_b_im2col, which gathers directly from the input image.

   index mapping for a packed b element at (row r in [0..kc), col jj in [0..NR)):
     global k = pc + r, global m = jc + jr + jj
     ci = k / (kh*kw); rem = k % (kh*kw); ky = rem / kw; kx = rem % kw
     oh = m / out_w; ow = m % out_w
     ih = oh*sh - ph + ky; iw = ow*sw - pw + kx
     value = input[ci, ih, iw] if in-bounds else 0 */

/* pack a KC x NC panel of im2col(input) into NR-col strips. identical layout
   to pack_b() so the micro_kernel is untouched.

   fast path: when a strip lies entirely on one output row AND stride_w==1,
   the iw positions within the strip are contiguous. that reduces the gather
   to a memset(leading pad) + memcpy(valid middle) + memset(trailing pad) —
   glibc memset/memcpy are simd-tuned, so the common case (stride-1 convs)
   stops being bottlenecked by per-element branches.

   slow path: general scalar gather with per-element bounds checks.
   used when the strip spans a row boundary or stride_w != 1. */
static void pack_b_im2col(const ax_conv_params_t *p,
                           int64_t kc, int64_t nc_pack, int64_t nc,
                           int64_t jc, int64_t pc,
                           float *packed)
{
    const float *input = p->input;
    const int64_t H = p->H, W = p->W;
    const int kh = p->kh, kw = p->kw;
    const int sh = p->sh, sw = p->sw;
    const int ph = p->ph, pw = p->pw;
    const int64_t out_w = p->out_w;
    const int64_t khkw = (int64_t)kh * (int64_t)kw;
    const int64_t HW = H * W;

    for (int64_t j = 0; j < nc_pack; j += GEMM_NR) {
        int64_t nr = (j + GEMM_NR <= nc) ? GEMM_NR : (nc > j ? nc - j : 0);

        /* strip geometry, independent of r */
        int64_t gm_first = jc + j;
        int64_t oh_first = (nr > 0) ? (gm_first / out_w) : 0;
        int64_t ow_first = gm_first - oh_first * out_w;
        int64_t gm_last = (nr > 0) ? (gm_first + nr - 1) : gm_first;
        int64_t oh_last = (nr > 0) ? (gm_last / out_w) : oh_first;
        bool fast_ok = (nr > 0) && (sw == 1) && (oh_first == oh_last);

        for (int64_t r = 0; r < kc; r++) {
            int64_t gk = pc + r;
            int64_t ci = gk / khkw;
            int64_t rem = gk - ci * khkw;
            int ky = (int)(rem / kw);
            int kx = (int)(rem - (int64_t)ky * kw);

            if (fast_ok) {
                int64_t ih = oh_first * sh - ph + ky;
                if (ih < 0 || ih >= H) {
                    memset(packed, 0, (size_t)GEMM_NR * sizeof(float));
                } else {
                    int64_t iw_start = ow_first - pw + kx;
                    int64_t lo = 0, hi = nr;
                    if (iw_start < 0) lo = -iw_start;
                    int64_t iw_last = iw_start + nr - 1;
                    if (iw_last >= W) hi = W - iw_start;
                    if (hi < 0) hi = 0;
                    if (lo > nr) lo = nr;
                    if (hi > nr) hi = nr;
                    if (lo > hi) lo = hi;

                    if (lo > 0)
                        memset(packed, 0, (size_t)lo * sizeof(float));
                    if (hi > lo)
                        memcpy(packed + lo,
                               input + ci * HW + ih * W + (iw_start + lo),
                               (size_t)(hi - lo) * sizeof(float));
                    if (nr > hi)
                        memset(packed + hi, 0, (size_t)(nr - hi) * sizeof(float));
                    if (GEMM_NR > nr)
                        memset(packed + nr, 0, (size_t)(GEMM_NR - nr) * sizeof(float));
                }
            } else {
                for (int64_t jj = 0; jj < GEMM_NR; jj++) {
                    float val = 0.0f;
                    if (jj < nr) {
                        int64_t gm = jc + j + jj;
                        int64_t oh = gm / out_w;
                        int64_t ow = gm - oh * out_w;
                        int64_t ih = oh * sh - ph + ky;
                        int64_t iw = ow * sw - pw + kx;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            val = input[ci * HW + ih * W + iw];
                        }
                    }
                    packed[jj] = val;
                }
            }
            packed += GEMM_NR;
        }
    }
}

static ax_status_t opt_conv_gemm(const ax_tensor_t *weight,
                                  const ax_conv_params_t *params,
                                  ax_tensor_t *out)
{
    if (!weight || !params || !out) {
        ax_err_set(AX_ERR_NULL_ARG, "conv_gemm: NULL arg");
        return AX_ERR_NULL_ARG;
    }
    if (weight->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32) {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "conv_gemm only supports float32");
        return AX_ERR_DTYPE_MISMATCH;
    }
    if (weight->ndim != 2 || out->ndim != 2) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "conv_gemm expects 2d weight and out");
        return AX_ERR_SHAPE_MISMATCH;
    }
    int64_t C_out = weight->shape[0];
    int64_t K = weight->shape[1];
    int64_t M = params->out_h * params->out_w;
    int64_t K_expected = params->C_in * params->kh * params->kw;
    if (K != K_expected || out->shape[0] != C_out || out->shape[1] != M) {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "conv_gemm shape mismatch");
        return AX_ERR_SHAPE_MISMATCH;
    }
    if (validate_contig_f32(weight) < 0 || validate_contig_f32(out) < 0 || !params->input) {
        ax_err_set(AX_ERR_BACKEND, "conv_gemm: non-contiguous or null input");
        return AX_ERR_BACKEND;
    }

    const float *wd = raw_f32(weight);
    float *od = raw_f32(out);
    memset(od, 0, (size_t)(C_out * M) * sizeof(float));

    int64_t m = C_out, n = M, k = K;

    /* small-problem fallback: straight scalar-simd loop with on-the-fly gather.
       tiled BLIS overhead dominates below ~100k FLOPs just like opt_gemm. */
    if (m * n * k < 100000) {
        const int64_t H = params->H, W = params->W;
        const int kh = params->kh, kw = params->kw;
        const int sh = params->sh, sw = params->sw;
        const int ph = params->ph, pw = params->pw;
        const int64_t out_w = params->out_w;
        const int64_t khkw = (int64_t)kh * (int64_t)kw;
        const int64_t HW = H * W;
        const float *input = params->input;

        for (int64_t i = 0; i < m; i++) {
            float *oi = od + i * n;
            const float *wi = wd + i * k;
            for (int64_t p = 0; p < k; p++) {
                float w_ip = wi[p];
                int64_t ci = p / khkw;
                int64_t rem = p - ci * khkw;
                int ky = (int)(rem / kw);
                int kx = (int)(rem - (int64_t)ky * kw);
                for (int64_t j = 0; j < n; j++) {
                    int64_t oh = j / out_w;
                    int64_t ow = j - oh * out_w;
                    int64_t ih = oh * sh - ph + ky;
                    int64_t iw = ow * sw - pw + kx;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        oi[j] += w_ip * input[ci * HW + ih * W + iw];
                }
            }
        }
        return AX_OK;
    }

    /* ensure this thread has pack buffers. conv_gemm is typically called from
       inside conv2d_forward's parallel batch loop — each thread already has its
       own tl_pack_a/b_buf. */
    if (!ensure_tl_pack_bufs()) {
        ax_err_set(AX_ERR_ALLOC, "conv_gemm: pack buffer alloc failed");
        return AX_ERR_ALLOC;
    }

    /* our pack_b writes custom data into tl_pack_b_buf; invalidate the cache so
       a subsequent plain opt_gemm call re-packs. */
    pack_b_cache_invalidate();

    float *pack_a_buf = tl_pack_a_buf;
    float *pack_b_buf = tl_pack_b_buf;

    for (int64_t jc = 0; jc < n; jc += GEMM_NC) {
        int64_t nc = (jc + GEMM_NC <= n) ? GEMM_NC : (n - jc);
        int64_t nc_pack = ((nc + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;

        for (int64_t pc = 0; pc < k; pc += GEMM_KC) {
            int64_t kc = (pc + GEMM_KC <= k) ? GEMM_KC : (k - pc);

            /* gather B directly from input image */
            pack_b_im2col(params, kc, nc_pack, nc, jc, pc, pack_b_buf);

            for (int64_t ic = 0; ic < m; ic += GEMM_MC) {
                int64_t mc = (ic + GEMM_MC <= m) ? GEMM_MC : (m - ic);
                int64_t mc_pack = ((mc + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;
                pack_a(wd + ic * k + pc, k, mc_pack, kc, mc, pack_a_buf);

                for (int64_t ir = 0; ir < mc_pack; ir += GEMM_MR) {
                    int64_t mr = (ir + GEMM_MR <= mc) ? GEMM_MR : (mc - ir);
                    for (int64_t jr = 0; jr < nc_pack; jr += GEMM_NR) {
                        int64_t nr = (jr + GEMM_NR <= nc) ? GEMM_NR : (nc - jr);
                        micro_kernel(kc, pack_a_buf + ir * kc, pack_b_buf + jr * kc,
                                     od + (ic + ir) * n + (jc + jr), n, mr, nr);
                    }
                }
            }
        }
    }

    /* leave pack_b cache invalidated (we clobbered it with custom data). */
    pack_b_cache_invalidate();
    return AX_OK;
}


/* helper: SIMD row sum for a contiguous float array of length n.
   uses 4 independent accumulators to hide FP add latency. */
static inline float simd_row_sum(const float *d, int64_t n)
{
    ax_vf32 acc0 = ax_vf32_zero(), acc1 = ax_vf32_zero();
    ax_vf32 acc2 = ax_vf32_zero(), acc3 = ax_vf32_zero();
    int64_t unroll4 = n - (n % (AX_VF32_WIDTH * 4));
    int64_t i = 0;
    for (; i < unroll4; i += AX_VF32_WIDTH * 4) {
        acc0 = ax_vf32_add(acc0, ax_vf32_loadu(d + i));
        acc1 = ax_vf32_add(acc1, ax_vf32_loadu(d + i + AX_VF32_WIDTH));
        acc2 = ax_vf32_add(acc2, ax_vf32_loadu(d + i + AX_VF32_WIDTH * 2));
        acc3 = ax_vf32_add(acc3, ax_vf32_loadu(d + i + AX_VF32_WIDTH * 3));
    }
    acc0 = ax_vf32_add(ax_vf32_add(acc0, acc1), ax_vf32_add(acc2, acc3));
    int64_t vec_end = n - (n % AX_VF32_WIDTH);
    for (; i < vec_end; i += AX_VF32_WIDTH)
        acc0 = ax_vf32_add(acc0, ax_vf32_loadu(d + i));
    double total = (double)ax_vf32_hsum(acc0);
    for (; i < n; i++) total += (double)d[i];
    return (float)total;
}

/* helper: SIMD row max/min for a contiguous float array */
static inline float simd_row_max(const float *d, int64_t n)
{
    ax_vf32 v0 = ax_vf32_set1(-FLT_MAX), v1 = ax_vf32_set1(-FLT_MAX);
    ax_vf32 v2 = ax_vf32_set1(-FLT_MAX), v3 = ax_vf32_set1(-FLT_MAX);
    int64_t unroll4 = n - (n % (AX_VF32_WIDTH * 4));
    int64_t i = 0;
    for (; i < unroll4; i += AX_VF32_WIDTH * 4) {
        v0 = ax_vf32_max(v0, ax_vf32_loadu(d + i));
        v1 = ax_vf32_max(v1, ax_vf32_loadu(d + i + AX_VF32_WIDTH));
        v2 = ax_vf32_max(v2, ax_vf32_loadu(d + i + AX_VF32_WIDTH * 2));
        v3 = ax_vf32_max(v3, ax_vf32_loadu(d + i + AX_VF32_WIDTH * 3));
    }
    v0 = ax_vf32_max(ax_vf32_max(v0, v1), ax_vf32_max(v2, v3));
    int64_t vec_end = n - (n % AX_VF32_WIDTH);
    for (; i < vec_end; i += AX_VF32_WIDTH)
        v0 = ax_vf32_max(v0, ax_vf32_loadu(d + i));
    float mx = ax_vf32_hmax(v0);
    for (; i < n; i++) if (d[i] > mx) mx = d[i];
    return mx;
}

static inline float simd_row_min(const float *d, int64_t n)
{
    ax_vf32 v0 = ax_vf32_set1(FLT_MAX), v1 = ax_vf32_set1(FLT_MAX);
    ax_vf32 v2 = ax_vf32_set1(FLT_MAX), v3 = ax_vf32_set1(FLT_MAX);
    int64_t unroll4 = n - (n % (AX_VF32_WIDTH * 4));
    int64_t i = 0;
    for (; i < unroll4; i += AX_VF32_WIDTH * 4) {
        v0 = ax_vf32_min(v0, ax_vf32_loadu(d + i));
        v1 = ax_vf32_min(v1, ax_vf32_loadu(d + i + AX_VF32_WIDTH));
        v2 = ax_vf32_min(v2, ax_vf32_loadu(d + i + AX_VF32_WIDTH * 2));
        v3 = ax_vf32_min(v3, ax_vf32_loadu(d + i + AX_VF32_WIDTH * 3));
    }
    v0 = ax_vf32_min(ax_vf32_min(v0, v1), ax_vf32_min(v2, v3));
    int64_t vec_end = n - (n % AX_VF32_WIDTH);
    for (; i < vec_end; i += AX_VF32_WIDTH)
        v0 = ax_vf32_min(v0, ax_vf32_loadu(d + i));
    float mn = ax_vf32_hmin(v0);
    for (; i < n; i++) if (d[i] < mn) mn = d[i];
    return mn;
}


/* axis-0 reductions for contig tensors: treat shape[0] as rows and the
   product of the remaining dims as cols. output is a contig [cols]-shape
   buffer. vectorize the col axis (simd-wide chunks) so each output lane
   has its own vector accumulator that traverses rows linearly — unit-stride
   loads, no horizontal reduction. parallel over col chunks (disjoint
   output writes). used by dense bias gradients (sum axis=0) and
   classification paths (argmax axis=0). */

static void simd_axis0_sum(const float *d, float *od, int64_t rows, int64_t cols) {
    int64_t vec_end = cols - (cols % AX_VF32_WIDTH);
    int64_t work = rows * cols;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(work > ax_par_threshold())
#else
    (void)work;
#endif
    for (int64_t j = 0; j < vec_end; j += AX_VF32_WIDTH) {
        ax_vf32 acc = ax_vf32_zero();
        for (int64_t i = 0; i < rows; i++)
            acc = ax_vf32_add(acc, ax_vf32_loadu(d + i * cols + j));
        ax_vf32_storeu(od + j, acc);
    }
    for (int64_t j = vec_end; j < cols; j++) {
        float acc = 0.0f;
        for (int64_t i = 0; i < rows; i++) acc += d[i * cols + j];
        od[j] = acc;
    }
}

static void simd_axis0_max(const float *d, float *od, int64_t rows, int64_t cols) {
    int64_t vec_end = cols - (cols % AX_VF32_WIDTH);
    int64_t work = rows * cols;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(work > ax_par_threshold())
#else
    (void)work;
#endif
    for (int64_t j = 0; j < vec_end; j += AX_VF32_WIDTH) {
        ax_vf32 acc = ax_vf32_set1(-FLT_MAX);
        for (int64_t i = 0; i < rows; i++)
            acc = ax_vf32_max(acc, ax_vf32_loadu(d + i * cols + j));
        ax_vf32_storeu(od + j, acc);
    }
    for (int64_t j = vec_end; j < cols; j++) {
        float acc = -FLT_MAX;
        for (int64_t i = 0; i < rows; i++) { float v = d[i * cols + j]; if (v > acc) acc = v; }
        od[j] = acc;
    }
}

static void simd_axis0_min(const float *d, float *od, int64_t rows, int64_t cols) {
    int64_t vec_end = cols - (cols % AX_VF32_WIDTH);
    int64_t work = rows * cols;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(work > ax_par_threshold())
#else
    (void)work;
#endif
    for (int64_t j = 0; j < vec_end; j += AX_VF32_WIDTH) {
        ax_vf32 acc = ax_vf32_set1(FLT_MAX);
        for (int64_t i = 0; i < rows; i++)
            acc = ax_vf32_min(acc, ax_vf32_loadu(d + i * cols + j));
        ax_vf32_storeu(od + j, acc);
    }
    for (int64_t j = vec_end; j < cols; j++) {
        float acc = FLT_MAX;
        for (int64_t i = 0; i < rows; i++) { float v = d[i * cols + j]; if (v < acc) acc = v; }
        od[j] = acc;
    }
}

/* validate an axis-0 reduction: input must be contig, output must be
   contig with numel == product(in->shape[1..]) and shape match.
   returns rows, cols via out params; returns -1 on failure. */
static int axis0_shape_ok(const ax_tensor_t *in, const ax_tensor_t *out,
                           int64_t *rows, int64_t *cols) {
    if (in->ndim < 2) return -1;
    if (out->ndim != in->ndim - 1) return -1;
    for (int d = 0; d < out->ndim; d++)
        if (out->shape[d] != in->shape[d + 1]) return -1;
    int64_t r = in->shape[0];
    int64_t c = 1;
    for (int d = 1; d < in->ndim; d++) c *= in->shape[d];
    if (r <= 0 || c <= 0) return -1;
    *rows = r;
    *cols = c;
    return 0;
}


/* reductions */

static ax_status_t opt_sum(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    /* fast path: full reduction */
    if (axis == -1) {
        int64_t n = validate_contig_f32(in);
        int64_t no = validate_contig_f32(out);
        if (n < 0 || no < 0 || no != 1) return ax_cpu_naive_ops.sum(in, axis, out);
        raw_f32(out)[0] = simd_row_sum(raw_f32(in), n);
        return AX_OK;
    }

    /* fast path: axis-1 sum on 2D contiguous tensor → row-wise reduction */
    if (axis == 1 && in->ndim == 2) {
        int64_t ni = validate_contig_f32(in);
        int64_t no = validate_contig_f32(out);
        if (ni < 0 || no < 0) return ax_cpu_naive_ops.sum(in, axis, out);
        int64_t rows = in->shape[0], cols = in->shape[1];
        if (out->ndim != 1 || out->shape[0] != rows)
            return ax_cpu_naive_ops.sum(in, axis, out);
        const float *d = raw_f32(in);
        float *od = raw_f32(out);
        int64_t n = rows; /* alias so AX_OMP_PAR_FOR_IF(n) resolves correctly */
        AX_OMP_PAR_FOR_IF(n)
        for (int64_t i = 0; i < rows; i++)
            od[i] = simd_row_sum(d + i * cols, cols);
        return AX_OK;
    }

    /* fast path: axis-0 sum on contig nd tensor. column-parallel
       vector accumulators; hot for dense bias gradients. */
    if (axis == 0) {
        int64_t ni = validate_contig_f32(in);
        int64_t no = validate_contig_f32(out);
        if (ni < 0 || no < 0) return ax_cpu_naive_ops.sum(in, axis, out);
        int64_t rows, cols;
        if (axis0_shape_ok(in, out, &rows, &cols) < 0)
            return ax_cpu_naive_ops.sum(in, axis, out);
        simd_axis0_sum(raw_f32(in), raw_f32(out), rows, cols);
        return AX_OK;
    }

    return ax_cpu_naive_ops.sum(in, axis, out);
}

static ax_status_t opt_mean(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    /* fast path: full reduction */
    if (axis == -1) {
        int64_t n = validate_contig_f32(in);
        int64_t no = validate_contig_f32(out);
        if (n < 0 || no < 0 || no != 1) return ax_cpu_naive_ops.mean(in, axis, out);
        raw_f32(out)[0] = simd_row_sum(raw_f32(in), n) / (float)n;
        return AX_OK;
    }

    /* fast path: axis-1 mean on 2D contiguous tensor → row-wise mean */
    if (axis == 1 && in->ndim == 2) {
        int64_t ni = validate_contig_f32(in);
        int64_t no = validate_contig_f32(out);
        if (ni < 0 || no < 0) return ax_cpu_naive_ops.mean(in, axis, out);
        int64_t rows = in->shape[0], cols = in->shape[1];
        if (out->ndim != 1 || out->shape[0] != rows)
            return ax_cpu_naive_ops.mean(in, axis, out);
        const float *d = raw_f32(in);
        float *od = raw_f32(out);
        float inv_cols = 1.0f / (float)cols;
        int64_t n = rows; /* alias so AX_OMP_PAR_FOR_IF(n) resolves correctly */
        AX_OMP_PAR_FOR_IF(n)
        for (int64_t i = 0; i < rows; i++)
            od[i] = simd_row_sum(d + i * cols, cols) * inv_cols;
        return AX_OK;
    }

    /* fast path: axis-0 mean = axis-0 sum / rows. */
    if (axis == 0) {
        int64_t ni = validate_contig_f32(in);
        int64_t no = validate_contig_f32(out);
        if (ni < 0 || no < 0) return ax_cpu_naive_ops.mean(in, axis, out);
        int64_t rows, cols;
        if (axis0_shape_ok(in, out, &rows, &cols) < 0)
            return ax_cpu_naive_ops.mean(in, axis, out);
        float *od = raw_f32(out);
        simd_axis0_sum(raw_f32(in), od, rows, cols);
        float inv_rows = 1.0f / (float)rows;
        ax_vf32 vinv = ax_vf32_set1(inv_rows);
        int64_t vec_end = cols - (cols % AX_VF32_WIDTH);
        for (int64_t j = 0; j < vec_end; j += AX_VF32_WIDTH)
            ax_vf32_storeu(od + j, ax_vf32_mul(ax_vf32_loadu(od + j), vinv));
        for (int64_t j = vec_end; j < cols; j++) od[j] *= inv_rows;
        return AX_OK;
    }

    return ax_cpu_naive_ops.mean(in, axis, out);
}

static ax_status_t opt_max(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    if (axis == -1) {
        int64_t n = validate_contig_f32(in);
        int64_t no = validate_contig_f32(out);
        if (n < 0 || no < 0 || no != 1) return ax_cpu_naive_ops.max_op(in, axis, out);
        raw_f32(out)[0] = simd_row_max(raw_f32(in), n);
        return AX_OK;
    }
    if (axis == 0) {
        int64_t ni = validate_contig_f32(in);
        int64_t no = validate_contig_f32(out);
        if (ni < 0 || no < 0) return ax_cpu_naive_ops.max_op(in, axis, out);
        int64_t rows, cols;
        if (axis0_shape_ok(in, out, &rows, &cols) < 0)
            return ax_cpu_naive_ops.max_op(in, axis, out);
        simd_axis0_max(raw_f32(in), raw_f32(out), rows, cols);
        return AX_OK;
    }
    return ax_cpu_naive_ops.max_op(in, axis, out);
}

static ax_status_t opt_min(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    if (axis == -1) {
        int64_t n = validate_contig_f32(in);
        int64_t no = validate_contig_f32(out);
        if (n < 0 || no < 0 || no != 1) return ax_cpu_naive_ops.min_op(in, axis, out);
        raw_f32(out)[0] = simd_row_min(raw_f32(in), n);
        return AX_OK;
    }
    if (axis == 0) {
        int64_t ni = validate_contig_f32(in);
        int64_t no = validate_contig_f32(out);
        if (ni < 0 || no < 0) return ax_cpu_naive_ops.min_op(in, axis, out);
        int64_t rows, cols;
        if (axis0_shape_ok(in, out, &rows, &cols) < 0)
            return ax_cpu_naive_ops.min_op(in, axis, out);
        simd_axis0_min(raw_f32(in), raw_f32(out), rows, cols);
        return AX_OK;
    }
    return ax_cpu_naive_ops.min_op(in, axis, out);
}


/* comparisons — vectorized via simd_defs cmpeq/cmpgt */

static ax_status_t opt_equal(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    int64_t n = validate_triple_same(a, b, out);
    if (n < 0) return ax_cpu_naive_ops.equal(a, b, out);
    const float *ad = raw_f32(a);
    const float *bd = raw_f32(b);
    float *od = raw_f32(out);
    int64_t i = 0;
    int64_t vec_end = n - (n % AX_VF32_WIDTH);
    for (; i < vec_end; i += AX_VF32_WIDTH)
        ax_vf32_store(od + i, ax_vf32_cmpeq(ax_vf32_load(ad + i), ax_vf32_load(bd + i)));
    for (; i < n; i++)
        od[i] = (ad[i] == bd[i]) ? 1.0f : 0.0f;
    return AX_OK;
}

static ax_status_t opt_greater(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    int64_t n = validate_triple_same(a, b, out);
    if (n < 0) return ax_cpu_naive_ops.greater(a, b, out);
    const float *ad = raw_f32(a);
    const float *bd = raw_f32(b);
    float *od = raw_f32(out);
    int64_t i = 0;
    int64_t vec_end = n - (n % AX_VF32_WIDTH);
    for (; i < vec_end; i += AX_VF32_WIDTH)
        ax_vf32_store(od + i, ax_vf32_cmpgt(ax_vf32_load(ad + i), ax_vf32_load(bd + i)));
    for (; i < n; i++)
        od[i] = (ad[i] > bd[i]) ? 1.0f : 0.0f;
    return AX_OK;
}


/* data movement */

static ax_status_t opt_fill(ax_tensor_t *t, double value) {
    int64_t n = validate_contig_f32(t);
    if (n < 0) return ax_cpu_naive_ops.fill(t, value);
    float v = (float)value;
    float *d = raw_f32(t);
    if (v == 0.0f) {
        memset(d, 0, (size_t)n * sizeof(float));
    } else {
        ax_vf32 vv = ax_vf32_set1(v);
        int64_t vec_end = n - (n % AX_VF32_WIDTH);
        AX_OMP_PAR_FOR_IF(n)
        for (int64_t i = 0; i < vec_end; i += AX_VF32_WIDTH)
            ax_vf32_store(d + i, vv);
        for (int64_t i = vec_end; i < n; i++)
            d[i] = v;
    }
    return AX_OK;
}

static ax_status_t opt_copy(const ax_tensor_t *src, ax_tensor_t *dst) {
    int64_t ns = validate_contig_f32(src);
    int64_t nd = validate_contig_f32(dst);
    if (ns < 0 || nd < 0 || ns != nd) return ax_cpu_naive_ops.copy(src, dst);
    const float *sd = raw_f32(src);
    float *dd = raw_f32(dst);
    /* small copies: a single glibc memcpy beats fork-join. large copies
       chunk the buffer so each worker calls memcpy on its own slice —
       lets the simd-tuned libc routine run per thread without the
       element-at-a-time overhead of a parallel scalar loop. */
    if (ns <= ax_par_threshold()) {
        memcpy(dd, sd, (size_t)ns * sizeof(float));
        return AX_OK;
    }
#ifdef _OPENMP
    int nt = omp_in_parallel() ? 1 : omp_get_max_threads();
    if (nt < 1) nt = 1;
    int64_t chunk = (ns + nt - 1) / nt;
    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int t = 0; t < nt; t++) {
        int64_t start = (int64_t)t * chunk;
        int64_t end = start + chunk;
        if (end > ns) end = ns;
        if (end > start) memcpy(dd + start, sd + start, (size_t)(end - start) * sizeof(float));
    }
#else
    memcpy(dd, sd, (size_t)ns * sizeof(float));
#endif
    return AX_OK;
}


/* transposed-b gemm: out = a @ b^T.
   shape contract: a is [m, k], b is [n, k], out is [m, n].
   reuses the packed BLIS tile loop with pack_b_t in place of pack_b.
   for simplicity this path skips the small-matrix fast-path and the
   pack_b cache — backward-pass gemms are called once per layer per
   step with unique shapes, so the cache would miss anyway. */
static ax_status_t opt_gemm_nt(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    if (!a || !b || !out) { ax_err_set(AX_ERR_NULL_ARG, "gemm_nt: NULL"); return AX_ERR_NULL_ARG; }
    if (a->dtype != AX_FLOAT32 || b->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32)
        return AX_ERR_DTYPE_MISMATCH;
    if (a->ndim != 2 || b->ndim != 2 || out->ndim != 2) return AX_ERR_SHAPE_MISMATCH;

    int64_t m = a->shape[0], k = a->shape[1];
    int64_t n = b->shape[0];
    if (b->shape[1] != k || out->shape[0] != m || out->shape[1] != n) return AX_ERR_SHAPE_MISMATCH;
    if (validate_contig_f32(a) < 0 || validate_contig_f32(b) < 0 || validate_contig_f32(out) < 0)
        return AX_ERR_BACKEND;

    const float *ad = raw_f32(a);
    const float *bd = raw_f32(b);  /* [n, k] — walked as if transposed */
    float *od = raw_f32(out);

    /* plain b cache would give stale/incorrect hits here — different packer */
    pack_b_cache_invalidate();

    /* small-path: straight vectorized inner loop. b[j, p] is accessed
       column-wise by j for each p — strided. scalar fallback is simplest. */
    if (m * n * k < 100000) {
        memset(od, 0, (size_t)(m * n) * sizeof(float));
        for (int64_t i = 0; i < m; i++) {
            const float *ai = ad + i * k;
            float *oi = od + i * n;
            for (int64_t j = 0; j < n; j++) {
                const float *bj = bd + j * k;
                ax_vf32 acc = ax_vf32_zero();
                int64_t p = 0;
                int64_t vec_end = k - (k % AX_VF32_WIDTH);
                for (; p < vec_end; p += AX_VF32_WIDTH)
                    acc = ax_vf32_fmadd(ax_vf32_loadu(ai + p), ax_vf32_loadu(bj + p), acc);
                float s = ax_vf32_hsum(acc);
                for (; p < k; p++) s += ai[p] * bj[p];
                oi[j] = s;
            }
        }
        return AX_OK;
    }

    if (!ensure_tl_pack_bufs()) return AX_ERR_ALLOC;
    memset(od, 0, (size_t)(m * n) * sizeof(float));

    /* JC parallelism when there's enough compute to justify OMP overhead
       (~1M FLOPs threshold matches the standard gemm heuristic). */
    int max_threads = 1;
    int64_t total_flops = m * n * k;
    #ifdef _OPENMP
    if (!omp_in_parallel() && total_flops > 1000000) max_threads = omp_get_max_threads();
    #endif
    int64_t nc_eff_nt = ax_adaptive_nc(n, max_threads);
    int64_t n_jc_tiles = (n + nc_eff_nt - 1) / nc_eff_nt;
    int gemm_threads = (int)(n_jc_tiles < (int64_t)max_threads ? n_jc_tiles : (int64_t)max_threads);
    (void)gemm_threads;

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(gemm_threads) schedule(static)
    #endif
    for (int64_t jct = 0; jct < n_jc_tiles; jct++) {
        ensure_tl_pack_bufs();
        float *pack_a_buf = tl_pack_a_buf;
        float *pack_b_buf = tl_pack_b_buf;
        if (!pack_a_buf || !pack_b_buf) continue;

        int64_t jc = jct * nc_eff_nt;
        int64_t nc = (jc + nc_eff_nt <= n) ? nc_eff_nt : (n - jc);
        int64_t nc_pack = ((nc + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
        for (int64_t pc = 0; pc < k; pc += GEMM_KC) {
            int64_t kc = (pc + GEMM_KC <= k) ? GEMM_KC : (k - pc);
            pack_b_t(bd + jc * k + pc, k, kc, nc_pack, nc, pack_b_buf);
            for (int64_t ic = 0; ic < m; ic += GEMM_MC) {
                int64_t mc = (ic + GEMM_MC <= m) ? GEMM_MC : (m - ic);
                int64_t mc_pack = ((mc + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;
                pack_a(ad + ic * k + pc, k, mc_pack, kc, mc, pack_a_buf);
                for (int64_t ir = 0; ir < mc_pack; ir += GEMM_MR) {
                    int64_t mr = (ir + GEMM_MR <= mc) ? GEMM_MR : (mc - ir);
                    for (int64_t jr = 0; jr < nc_pack; jr += GEMM_NR) {
                        int64_t nr = (jr + GEMM_NR <= nc) ? GEMM_NR : (nc - jr);
                        micro_kernel(kc, pack_a_buf + ir * kc, pack_b_buf + jr * kc,
                                     od + (ic + ir) * n + (jc + jr), n, mr, nr);
                    }
                }
            }
        }
    }
    pack_b_cache_invalidate();
    return AX_OK;
}

/* transposed-a gemm: out = a^T @ b.
   shape contract: a is [k, m], b is [k, n], out is [m, n].
   reuses the tile loop with pack_a_t in place of pack_a. */
static ax_status_t opt_gemm_tn(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) {
    if (!a || !b || !out) { ax_err_set(AX_ERR_NULL_ARG, "gemm_tn: NULL"); return AX_ERR_NULL_ARG; }
    if (a->dtype != AX_FLOAT32 || b->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32)
        return AX_ERR_DTYPE_MISMATCH;
    if (a->ndim != 2 || b->ndim != 2 || out->ndim != 2) return AX_ERR_SHAPE_MISMATCH;

    int64_t k = a->shape[0], m = a->shape[1];
    int64_t n = b->shape[1];
    if (b->shape[0] != k || out->shape[0] != m || out->shape[1] != n) return AX_ERR_SHAPE_MISMATCH;
    if (validate_contig_f32(a) < 0 || validate_contig_f32(b) < 0 || validate_contig_f32(out) < 0)
        return AX_ERR_BACKEND;

    const float *ad = raw_f32(a);  /* [k, m] — walked as if transposed */
    const float *bd = raw_f32(b);
    float *od = raw_f32(out);

    pack_b_cache_invalidate();

    if (m * n * k < 100000) {
        memset(od, 0, (size_t)(m * n) * sizeof(float));
        /* scalar-simd inner: out[i,j] = sum_p a[p,i] * b[p,j].
           iterate p outer, (i,j) inner to keep b[p,:] contiguous. */
        for (int64_t p = 0; p < k; p++) {
            const float *bp = bd + p * n;
            const float *ap = ad + p * m;
            for (int64_t i = 0; i < m; i++) {
                float ai = ap[i];
                float *oi = od + i * n;
                ax_vf32 va = ax_vf32_set1(ai);
                int64_t j = 0;
                int64_t vec_end = n - (n % AX_VF32_WIDTH);
                if (p == 0) {
                    /* first pass: initialize oi to ai * bp */
                    for (; j < vec_end; j += AX_VF32_WIDTH)
                        ax_vf32_storeu(oi + j, ax_vf32_mul(va, ax_vf32_loadu(bp + j)));
                    for (; j < n; j++) oi[j] = ai * bp[j];
                } else {
                    for (; j < vec_end; j += AX_VF32_WIDTH)
                        ax_vf32_storeu(oi + j, ax_vf32_fmadd(va, ax_vf32_loadu(bp + j), ax_vf32_loadu(oi + j)));
                    for (; j < n; j++) oi[j] += ai * bp[j];
                }
            }
        }
        return AX_OK;
    }

    if (!ensure_tl_pack_bufs()) return AX_ERR_ALLOC;
    memset(od, 0, (size_t)(m * n) * sizeof(float));

    int max_threads = 1;
    int64_t total_flops = m * n * k;
    #ifdef _OPENMP
    if (!omp_in_parallel() && total_flops > 1000000) max_threads = omp_get_max_threads();
    #endif
    int64_t nc_eff_tn = ax_adaptive_nc(n, max_threads);
    int64_t n_jc_tiles = (n + nc_eff_tn - 1) / nc_eff_tn;
    int gemm_threads = (int)(n_jc_tiles < (int64_t)max_threads ? n_jc_tiles : (int64_t)max_threads);
    (void)gemm_threads;

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(gemm_threads) schedule(static)
    #endif
    for (int64_t jct = 0; jct < n_jc_tiles; jct++) {
        ensure_tl_pack_bufs();
        float *pack_a_buf = tl_pack_a_buf;
        float *pack_b_buf = tl_pack_b_buf;
        if (!pack_a_buf || !pack_b_buf) continue;

        int64_t jc = jct * nc_eff_tn;
        int64_t nc = (jc + nc_eff_tn <= n) ? nc_eff_tn : (n - jc);
        int64_t nc_pack = ((nc + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
        for (int64_t pc = 0; pc < k; pc += GEMM_KC) {
            int64_t kc = (pc + GEMM_KC <= k) ? GEMM_KC : (k - pc);
            pack_b(bd + pc * n + jc, n, kc, nc_pack, nc, pack_b_buf);
            for (int64_t ic = 0; ic < m; ic += GEMM_MC) {
                int64_t mc = (ic + GEMM_MC <= m) ? GEMM_MC : (m - ic);
                int64_t mc_pack = ((mc + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;
                pack_a_t(ad + pc * m + ic, m, mc_pack, kc, mc, pack_a_buf);
                for (int64_t ir = 0; ir < mc_pack; ir += GEMM_MR) {
                    int64_t mr = (ir + GEMM_MR <= mc) ? GEMM_MR : (mc - ir);
                    for (int64_t jr = 0; jr < nc_pack; jr += GEMM_NR) {
                        int64_t nr = (jr + GEMM_NR <= nc) ? GEMM_NR : (nc - jr);
                        micro_kernel(kc, pack_a_buf + ir * kc, pack_b_buf + jr * kc,
                                     od + (ic + ir) * n + (jc + jr), n, mr, nr);
                    }
                }
            }
        }
    }
    pack_b_cache_invalidate();
    return AX_OK;
}

/* fused bias add: out[..., axis, ...] = in[..., axis, ...] + bias.
   bias is rank-1, numel == in->shape[axis]. single-pass fused write
   to out. broadcast is along `axis` — the stride of the bias lookup
   is 1 per AX_VF32_WIDTH elements ONLY when axis == ndim-1 (innermost).
   for axis != innermost, use a per-group scalar fill + simd add. */
static ax_status_t opt_bias_add(const ax_tensor_t *in, const ax_tensor_t *bias,
                                 int axis, ax_tensor_t *out) {
    if (!in || !bias || !out) return AX_ERR_NULL_ARG;
    if (validate_contig_f32(in) < 0 || validate_contig_f32(bias) < 0 || validate_contig_f32(out) < 0)
        return ax_cpu_naive_ops.add ? ax_cpu_naive_ops.add(in, bias, out) : AX_ERR_BACKEND;
    if (bias->ndim != 1) return AX_ERR_SHAPE_MISMATCH;
    if (axis < 0 || axis >= in->ndim) return AX_ERR_INVALID_AXIS;
    if (bias->shape[0] != in->shape[axis]) return AX_ERR_SHAPE_MISMATCH;
    if (in->ndim != out->ndim) return AX_ERR_SHAPE_MISMATCH;
    for (int d = 0; d < in->ndim; d++)
        if (in->shape[d] != out->shape[d]) return AX_ERR_SHAPE_MISMATCH;

    const float *id = raw_f32(in);
    const float *bd = raw_f32(bias);
    float *od = raw_f32(out);
    int64_t n = fast_numel(in);

    /* compute outer, axis_len, inner:
       outer = product of shape[0..axis)
       axis_len = shape[axis]
       inner = product of shape[axis+1..ndim) */
    int64_t outer = 1, inner = 1;
    for (int d = 0; d < axis; d++) outer *= in->shape[d];
    int64_t axis_len = in->shape[axis];
    for (int d = axis + 1; d < in->ndim; d++) inner *= in->shape[d];

    /* for every (outer, axis, inner) group, broadcast bias[axis] over inner elements */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2) if(n > ax_par_threshold())
#else
    (void)n;
#endif
    for (int64_t o = 0; o < outer; o++) {
        for (int64_t a = 0; a < axis_len; a++) {
            const float *ip = id + (o * axis_len + a) * inner;
            float *op = od + (o * axis_len + a) * inner;
            float bv = bd[a];
            ax_vf32 vb = ax_vf32_set1(bv);
            int64_t i = 0;
            int64_t vec_end = inner - (inner % AX_VF32_WIDTH);
            for (; i < vec_end; i += AX_VF32_WIDTH)
                ax_vf32_storeu(op + i, ax_vf32_add(ax_vf32_loadu(ip + i), vb));
            for (; i < inner; i++) op[i] = ip[i] + bv;
        }
    }
    return AX_OK;
}

/* argmax along an axis. output is contig int64 with rank in->ndim-1
   (reduced dim removed). axis=-1 reduces everything to one scalar.
   fast path: full reduction via one linear scan over the flat buffer.
   general axis=k: outer x axis_len x inner layout with scalar scan along
   axis — simd is not straightforward because we'd need to track indices
   alongside values. scalar is fine for an accuracy-counting hot path. */
static ax_status_t opt_argmax(const ax_tensor_t *in, int axis, ax_tensor_t *out) {
    if (!in || !out) return AX_ERR_NULL_ARG;
    if (in->dtype != AX_FLOAT32) return AX_ERR_DTYPE_MISMATCH;
    if (out->dtype != AX_INT64) return AX_ERR_DTYPE_MISMATCH;
    if (validate_contig_f32(in) < 0) return AX_ERR_BACKEND;
    if (!out->storage || !out->storage->data) return AX_ERR_BACKEND;

    const float *id = raw_f32(in);
    int64_t *od = (int64_t *)out->storage->data;
    int64_t n = fast_numel(in);

    if (axis == -1) {
        if (fast_numel(out) != 1) return AX_ERR_SHAPE_MISMATCH;
        int64_t best = 0;
        float bestv = id[0];
        for (int64_t i = 1; i < n; i++) {
            if (id[i] > bestv) { bestv = id[i]; best = i; }
        }
        od[0] = best;
        return AX_OK;
    }

    if (axis < 0 || axis >= in->ndim) return AX_ERR_INVALID_AXIS;

    int64_t outer = 1, inner = 1;
    for (int d = 0; d < axis; d++) outer *= in->shape[d];
    int64_t axis_len = in->shape[axis];
    for (int d = axis + 1; d < in->ndim; d++) inner *= in->shape[d];

    if (fast_numel(out) != outer * inner) return AX_ERR_SHAPE_MISMATCH;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(outer * inner > ax_par_threshold() / 4)
#endif
    for (int64_t oi = 0; oi < outer * inner; oi++) {
        int64_t o = oi / inner;
        int64_t i = oi - o * inner;
        const float *base = id + o * axis_len * inner + i;
        int64_t best = 0;
        float bestv = base[0];
        for (int64_t a = 1; a < axis_len; a++) {
            float v = base[a * inner];
            if (v > bestv) { bestv = v; best = a; }
        }
        od[oi] = best;
    }
    return AX_OK;
}


/* ── fused primitives ──────────────────────────────────────────────
   add_relu / axpy / softmax_rowwise — single-pass fused ops whose
   win over the equivalent dispatch chain is memory traffic. all
   three require contiguous, same-shape tensors; anything else
   falls back to cpu_naive. */

static ax_status_t opt_add_relu(const ax_tensor_t *a, const ax_tensor_t *b,
                                 ax_tensor_t *out)
{
    int64_t na = validate_contig_f32(a);
    int64_t nb = validate_contig_f32(b);
    int64_t no_v = validate_contig_f32(out);
    if (na < 0 || nb < 0 || no_v < 0 || na != nb || na != no_v) {
        return ax_cpu_naive_ops.add_relu(a, b, out);
    }
    const float *ad = raw_f32(a);
    const float *bd = raw_f32(b);
    float *od = raw_f32(out);

    int64_t i = 0, ve = na - (na % AX_VF32_WIDTH);
    for (; i < ve; i += AX_VF32_WIDTH) {
        ax_vf32 va = ax_vf32_load(ad + i);
        ax_vf32 vb = ax_vf32_load(bd + i);
        ax_vf32_store(od + i, ax_vf32_relu(ax_vf32_add(va, vb)));
    }
    for (; i < na; i++) {
        float v = ad[i] + bd[i];
        od[i] = v > 0.0f ? v : 0.0f;
    }
    return AX_OK;
}

static ax_status_t opt_axpy(const ax_tensor_t *x, float alpha, ax_tensor_t *y)
{
    int64_t nx = validate_contig_f32(x);
    int64_t ny = validate_contig_f32(y);
    if (nx < 0 || ny < 0 || nx != ny) {
        return ax_cpu_naive_ops.axpy(x, alpha, y);
    }
    const float *xd = raw_f32(x);
    float *yd = raw_f32(y);
    ax_vf32 v_alpha = ax_vf32_set1(alpha);
    int64_t i = 0, ve = nx - (nx % AX_VF32_WIDTH);
    for (; i < ve; i += AX_VF32_WIDTH) {
        ax_vf32 vy = ax_vf32_load(yd + i);
        ax_vf32 vx = ax_vf32_load(xd + i);
        ax_vf32_store(yd + i, ax_vf32_fmadd(v_alpha, vx, vy));
    }
    for (; i < nx; i++) yd[i] += alpha * xd[i];
    return AX_OK;
}

static ax_status_t opt_softmax_rowwise(const ax_tensor_t *in, ax_tensor_t *out)
{
    if (in->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32)
        return AX_ERR_DTYPE_MISMATCH;
    if (in->ndim != 2 || out->ndim != 2)
        return ax_cpu_naive_ops.softmax_rowwise(in, out);
    /* stricter: require contiguous, offset 0 */
    if (validate_contig_f32(in) < 0 || validate_contig_f32(out) < 0)
        return ax_cpu_naive_ops.softmax_rowwise(in, out);
    if (in->shape[0] != out->shape[0] || in->shape[1] != out->shape[1])
        return AX_ERR_SHAPE_MISMATCH;

    int64_t rows = in->shape[0];
    int64_t cols = in->shape[1];
    const float *id = raw_f32(in);
    float *od = raw_f32(out);

    /* parallel over rows — each row is an independent softmax. threshold
       avoids omp fork-join overhead for tiny batches. */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(rows * cols > ax_par_threshold() / 2)
    #endif
    for (int64_t r = 0; r < rows; r++) {
        const float *irow = id + r * cols;
        float *orow = od + r * cols;

        /* pass 1: row-max with simd. use loadu/storeu because rows are
           not guaranteed to be aligned (cols may not be a multiple of
           AX_VF32_WIDTH, making row offsets misaligned). */
        int64_t c = 0, ve = cols - (cols % AX_VF32_WIDTH);
        float mx;
        if (ve > 0) {
            ax_vf32 vmx = ax_vf32_loadu(irow);
            for (c = AX_VF32_WIDTH; c < ve; c += AX_VF32_WIDTH)
                vmx = ax_vf32_max(vmx, ax_vf32_loadu(irow + c));
            mx = ax_vf32_hmax(vmx);
        } else {
            mx = irow[0];
            c = 1;
        }
        for (; c < cols; c++) if (irow[c] > mx) mx = irow[c];

        /* pass 2: exp(x - max) + row sum. */
        ax_vf32 vmx_b = ax_vf32_set1(mx);
        ax_vf32 vsum = ax_vf32_zero();
        c = 0;
        for (; c < ve; c += AX_VF32_WIDTH) {
            ax_vf32 ve_vals = ax_vf32_exp(ax_vf32_sub(ax_vf32_loadu(irow + c), vmx_b));
            ax_vf32_storeu(orow + c, ve_vals);
            vsum = ax_vf32_add(vsum, ve_vals);
        }
        float sum = ax_vf32_hsum(vsum);
        for (; c < cols; c++) {
            float e = expf(irow[c] - mx);
            orow[c] = e;
            sum += e;
        }

        /* pass 3: divide by row sum. */
        float inv = 1.0f / sum;
        ax_vf32 vinv = ax_vf32_set1(inv);
        c = 0;
        for (; c < ve; c += AX_VF32_WIDTH)
            ax_vf32_storeu(orow + c, ax_vf32_mul(ax_vf32_loadu(orow + c), vinv));
        for (; c < cols; c++) orow[c] *= inv;
    }
    return AX_OK;
}


/* fused-scaling gemm: out = alpha * (a @ b) + beta * out.

   dispatch strategy:
   - alpha==1, beta==0: identical to plain gemm. reuse opt_gemm.
   - alpha==1, beta==1: accumulate into out. run opt_gemm into a
     temporary, then a simd add pass. (or, alternatively, skip the
     memset(0) inside opt_gemm — but that needs a parameterised
     variant; the temp+add version reuses the existing optimized
     gemm unchanged.)
   - general case: fall through to cpu_naive's reference impl.

   the primary callers are backward passes and bias-accumulation
   patterns where alpha=1 is the norm and beta is either 0 or 1.
   other alphas are rare in practice. */

static ax_status_t opt_gemm_ex(const ax_tensor_t *a, const ax_tensor_t *b,
                                float alpha, float beta, ax_tensor_t *out)
{
    if (!a || !b || !out) {
        ax_err_set(AX_ERR_NULL_ARG, "gemm_ex: NULL tensor");
        return AX_ERR_NULL_ARG;
    }
    if (a->dtype != AX_FLOAT32 || b->dtype != AX_FLOAT32 || out->dtype != AX_FLOAT32)
        return AX_ERR_DTYPE_MISMATCH;
    if (a->ndim != 2 || b->ndim != 2 || out->ndim != 2)
        return AX_ERR_SHAPE_MISMATCH;

    /* fast path 1: alpha==1, beta==0 → plain gemm semantics. */
    if (alpha == 1.0f && beta == 0.0f) {
        return opt_gemm(a, b, out);
    }

    /* fast path 2: alpha==1, beta==1 → accumulate a@b into out.
       do the gemm into a thread-local scratch, then simd-add the
       scratch into out. cheaper than scaling up two operands. */
    if (alpha == 1.0f && beta == 1.0f) {
        int64_t m = a->shape[0], n = b->shape[1];
        if (validate_contig_f32(out) < 0)
            return ax_cpu_naive_ops.gemm_ex(a, b, alpha, beta, out);
        int64_t numel = m * n;
        float *scratch = (float *)ax_aligned_alloc((size_t)numel * sizeof(float), 64);
        if (!scratch) return ax_cpu_naive_ops.gemm_ex(a, b, alpha, beta, out);
        /* wrap the scratch in a throwaway tensor descriptor so opt_gemm
           can write into it. share metadata with out (same shape). */
        ax_storage_t fake_storage = {
            .data = scratch, .size_bytes = (size_t)numel * sizeof(float),
            .device = AX_DEVICE_CPU, .is_arena_temp = false, .generation = 1,
        };
        atomic_init(&fake_storage.refcount, 1);
        ax_tensor_t fake = *out;
        fake.storage = &fake_storage;
        fake.offset = 0;
        ax_status_t s = opt_gemm(a, b, &fake);
        if (s != AX_OK) { ax_aligned_free(scratch); return s; }
        /* simd accumulate: out += scratch */
        float *od = raw_f32(out);
        int64_t i = 0, ve = numel - (numel % AX_VF32_WIDTH);
        for (; i < ve; i += AX_VF32_WIDTH) {
            ax_vf32 o = ax_vf32_load(od + i);
            ax_vf32 r = ax_vf32_load(scratch + i);
            ax_vf32_store(od + i, ax_vf32_add(o, r));
        }
        for (; i < numel; i++) od[i] += scratch[i];
        ax_aligned_free(scratch);
        return AX_OK;
    }

    /* general (alpha, beta) — reference implementation is fine;
       this path is rare in training workloads. */
    return ax_cpu_naive_ops.gemm_ex(a, b, alpha, beta, out);
}


/* vtable registration */

const ax_backend_ops_t AX_SYM(ax_cpu_opt_ops) = {
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
    .gemm_ex    = opt_gemm_ex,
    .gemm_nt    = opt_gemm_nt,
    .gemm_tn    = opt_gemm_tn,
    .add_relu   = opt_add_relu,
    .axpy       = opt_axpy,
    .softmax_rowwise = opt_softmax_rowwise,
    .bias_add   = opt_bias_add,
    .conv_gemm  = opt_conv_gemm,
    .sum        = opt_sum,
    .mean       = opt_mean,
    .max_op     = opt_max,
    .min_op     = opt_min,
    .argmax     = opt_argmax,
    .equal      = opt_equal,
    .greater    = opt_greater,
    .fill       = opt_fill,
    .copy       = opt_copy,
    .relu       = opt_relu,
    .sigmoid    = opt_sigmoid,
    .tanh_op    = opt_tanh_op,
};
