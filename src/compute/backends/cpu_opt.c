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
#include <stdatomic.h>
#include <time.h>
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

/* hybrid cpu thread counts from dispatch.c. large GEMMs use all_threads
   (P+E cores), small ops use fast_threads (P-cores only). both are 0
   until ax_autotune_threads runs, after which they're set. */
extern int ax_gemm_fast_threads;
extern int ax_gemm_all_threads;

/* forward declaration — definition after GEMM_MC/NC are declared */
static int ax_gemm_threads_for_shape(int64_t m, int64_t n, int64_t k);

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

/* per-thread scratch for SDPA softmax (row_max, row_sum) and attn packing
   (Kt, V). reused across calls so sdpa forward doesn't malloc 128+ times
   per invocation on a BH=64 workload. sized lazily to fit the largest
   (S, dk) seen so far. */
AX_TLS float *tl_sdpa_row_max = NULL; AX_TLS int64_t tl_sdpa_row_max_S = 0;
AX_TLS float *tl_sdpa_row_sum = NULL; AX_TLS int64_t tl_sdpa_row_sum_S = 0;
AX_TLS float *tl_sdpa_kt_packed = NULL; AX_TLS int64_t tl_sdpa_kt_bytes = 0;
AX_TLS float *tl_sdpa_v_packed  = NULL; AX_TLS int64_t tl_sdpa_v_bytes = 0;

/* sdpa backward scratch. attn_bwd_head has 8 aligned_alloc per head
   call; at BH=64 a single sdpa_bwd hit malloc ~128 times. move all
   eight to tls with grow-on-demand. */
AX_TLS float *tl_bwd_kt_packed  = NULL; AX_TLS int64_t tl_bwd_kt_bytes  = 0;
AX_TLS float *tl_bwd_vt_packed  = NULL; AX_TLS int64_t tl_bwd_vt_bytes  = 0;
AX_TLS float *tl_bwd_k_packed   = NULL; AX_TLS int64_t tl_bwd_k_bytes   = 0;
AX_TLS float *tl_bwd_p_tile     = NULL; AX_TLS int64_t tl_bwd_p_bytes   = 0;
AX_TLS float *tl_bwd_dp_tile    = NULL; AX_TLS int64_t tl_bwd_dp_bytes  = 0;
AX_TLS float *tl_bwd_ds_tile    = NULL; AX_TLS int64_t tl_bwd_ds_bytes  = 0;
AX_TLS float *tl_bwd_pa         = NULL; AX_TLS int64_t tl_bwd_pa_bytes  = 0;
AX_TLS float *tl_bwd_pb         = NULL; AX_TLS int64_t tl_bwd_pb_bytes  = 0;
AX_TLS float *tl_bwd_di         = NULL; AX_TLS int64_t tl_bwd_di_bytes  = 0;

static inline float *ax_tls_grow(float **p, int64_t *cap_bytes, int64_t want_bytes) {
    if (*p && *cap_bytes >= want_bytes) return *p;
    if (*p) { ax_aligned_free(*p); *p = NULL; *cap_bytes = 0; }
    *p = (float *)ax_aligned_alloc((size_t)want_bytes, 64);
    if (*p) *cap_bytes = want_bytes;
    return *p;
}

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
    for (int d = 0; d < t->ndim; d++) {
        if (t->shape[d] <= 0) return -1;
        /* overflow check: if n * shape[d] would exceed INT64_MAX, bail */
        if (n > INT64_MAX / t->shape[d]) return -1;
        n *= t->shape[d];
    }
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


/* non-temporal store threshold: arrays larger than half L3 benefit from
   NT stores because they won't fit in L3 anyway (no cache pollution).
   defaults to 2M floats (8MB) if L3 not detected. updated at init. */
static int64_t AX_NT_ELEMS = 2 * 1024 * 1024;

/* issue memory-ordering fence after a block of NT stores */
static inline void ax_nt_fence(void) {
#if defined(AX_SIMD_AVX2) || defined(AX_SIMD_AVX512)
    _mm_sfence();
#endif
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
    int use_nt = (n >= AX_NT_ELEMS); \
    AX_OMP_PAR_FOR_IF(n) \
    for (int64_t i = 0; i < vec_end; i += AX_VF32_WIDTH) { \
        ax_vf32 va = ax_vf32_load(ad + i); \
        ax_vf32 vb = ax_vf32_load(bd + i); \
        if (use_nt) ax_vf32_stream(od + i, simd_expr); \
        else        ax_vf32_store(od + i, simd_expr); \
    } \
    if (use_nt) ax_nt_fence(); \
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
    int use_nt = (n >= AX_NT_ELEMS); \
    AX_OMP_PAR_FOR_IF(n) \
    for (int64_t i = 0; i < vec_end; i += AX_VF32_WIDTH) { \
        ax_vf32 v = ax_vf32_load(id + i); \
        if (use_nt) ax_vf32_stream(od + i, simd_expr); \
        else        ax_vf32_store(od + i, simd_expr); \
    } \
    if (use_nt) ax_nt_fence(); \
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
    int64_t n = ni;
    int64_t vec_end = n - (n % AX_VF32_WIDTH);
    int use_nt = (n >= AX_NT_ELEMS);
    AX_OMP_PAR_FOR_IF(n)
    for (int64_t i = 0; i < vec_end; i += AX_VF32_WIDTH) {
        if (use_nt) ax_vf32_stream(od + i, ax_vf32_add(ax_vf32_load(id + i), vs));
        else        ax_vf32_store(od + i, ax_vf32_add(ax_vf32_load(id + i), vs));
    }
    if (use_nt) ax_nt_fence();
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
    int64_t n = ni;
    int64_t vec_end = n - (n % AX_VF32_WIDTH);
    int use_nt = (n >= AX_NT_ELEMS);
    AX_OMP_PAR_FOR_IF(n)
    for (int64_t i = 0; i < vec_end; i += AX_VF32_WIDTH) {
        if (use_nt) ax_vf32_stream(od + i, ax_vf32_mul(ax_vf32_load(id + i), vs));
        else        ax_vf32_store(od + i, ax_vf32_mul(ax_vf32_load(id + i), vs));
    }
    if (use_nt) ax_nt_fence();
    for (int64_t i = vec_end; i < n; i++)
        od[i] = id[i] * s;
    return AX_OK;
}


/* tiled gemm: cache-blocked with panel packing (BLIS-style).
   6x16 micro-kernel on AVX2: 12 YMM accumulators + 2 B loads + 1 A broadcast
   = 15 of 16 registers. no spills, near-peak FMA throughput. */

#if defined(AX_SIMD_AVX512)
    #define GEMM_MR 14
    #define GEMM_NR 32     /* 2 x 16-wide AVX-512 vectors per row.
                              28 accumulators + 2 B + 1 A + 1 spare = 32 ZMM.
                              matches BLIS reference for Skylake-X. */
#elif defined(AX_SIMD_AVX2)
    #define GEMM_MR 6
    #define GEMM_NR 16     /* 2 x 8-wide AVX2 vectors per row */
#elif defined(AX_SIMD_NEON)
    #define GEMM_MR 8
    #define GEMM_NR 12     /* 3 x 4-wide NEON vectors per row.
                              24 accumulators + 3 B + 2 A = 29 Q regs.
                              uses vfmaq_laneq_f32 (lane-broadcast FMA). */
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
/* isa-adaptive tile defaults. the cmake profile sets one set of defaults
   for the target class (desktop / embedded-linux / baremetal). when the
   isa is known at compile time, override with values that fill the typical
   cache hierarchy for that ISA. env vars still win over everything. */
#if defined(AX_SIMD_AVX512) && !defined(AX_GEMM_DEFAULT_MC)
    /* xeon: 1 MB L2. MC=168=12×14, pack_a=168×256×4=168KB.
       NC=512, pack_b=512×256×4=512KB. total=680KB < 1MB. */
    #define AX_GEMM_DEFAULT_MC 168
    #define AX_GEMM_DEFAULT_NC 512
    #define AX_GEMM_DEFAULT_KC 256
#endif
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

/* pick thread count for a GEMM. strategy is 3-tier:
   - tiny GEMMs (< 1M FLOPs): serial, fork-join cost exceeds compute
   - small-medium (< ~100M FLOPs): fast cores only, E-cores add latency
   - large (>= ~100M FLOPs): all cores, parallelism dominates latency
   the thresholds scale with number of fast cores (more cores = higher
   absolute throughput = larger crossover point). */
static int ax_gemm_threads_for_shape(int64_t m, int64_t n, int64_t k) {
#ifdef _OPENMP
    if (omp_in_parallel()) return 1;

    int fast = ax_gemm_fast_threads > 0 ? ax_gemm_fast_threads : omp_get_max_threads();
    int all  = ax_gemm_all_threads  > 0 ? ax_gemm_all_threads  : fast;

    int64_t total_flops = 2 * m * n * k;

    /* tiny: fork-join amortization threshold */
    if (total_flops < 1000000) return 1;

    /* if non-hybrid, just return all (fast == all) */
    if (all == fast) return all;

    /* crossover from fast-only to all-cores. scales with fast thread
       count: more P-cores means their throughput is higher, pushing the
       crossover higher before E-cores help. ~25M FLOPs per fast thread. */
    int64_t crossover = (int64_t)fast * 25000000LL;
    return (total_flops >= crossover) ? all : fast;
#else
    (void)m; (void)n; (void)k;
    return 1;
#endif
}

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

    /* runtime cache-size auto-tuning. detect L1d and L2, then adjust
       MC and NC so pack_a fits in L1d and pack_a+pack_b fits in L2.
       general: adapts to any cpu's cache hierarchy automatically.
       env vars always override. */
#if defined(__linux__) && !defined(AX_NO_STDIO)
    /* L1d auto-tune: pack_a (MC×KC) should fit in ~80% of L1d.
       when it doesn't, shrink MC (rounded to MR). this matters on
       neoverse-n2 (64KB L1d) where the desktop default MC=72 gives
       pack_a=72KB > 64KB, and on zen3/4 (32KB L1d).

       fallback to reading /sys when sysconf returns -1 (some ARM
       containers don't expose cache info via sysconf but do via sysfs). */
    long l1d = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    if (l1d <= 0) {
        /* try /sys/devices/system/cpu/cpu0/cache/index0/size (usually L1d) */
        for (int idx = 0; idx < 4 && l1d <= 0; idx++) {
            char path[128];
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu0/cache/index%d/level", idx);
            FILE *fl = fopen(path, "r");
            if (!fl) continue;
            int lvl = 0; fscanf(fl, "%d", &lvl); fclose(fl);
            if (lvl != 1) continue;
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu0/cache/index%d/type", idx);
            FILE *ft = fopen(path, "r");
            if (!ft) continue;
            char type[32] = {0};
            fscanf(ft, "%31s", type); fclose(ft);
            if (strcmp(type, "Data") != 0 && strcmp(type, "Unified") != 0) continue;
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu0/cache/index%d/size", idx);
            FILE *fs = fopen(path, "r");
            if (!fs) continue;
            long sz = 0; char unit = 0;
            if (fscanf(fs, "%ld%c", &sz, &unit) >= 1) {
                if (unit == 'K' || unit == 'k') sz *= 1024;
                else if (unit == 'M' || unit == 'm') sz *= 1024 * 1024;
                l1d = sz;
            }
            fclose(fs);
        }
    }
    if (l1d > 0) {
        long l1_budget = (long)((double)l1d * 0.80);
        size_t pack_a_bytes = (size_t)GEMM_MC * (size_t)GEMM_KC * sizeof(float);
        if ((long)pack_a_bytes > l1_budget) {
            int64_t new_mc = l1_budget / ((int64_t)GEMM_KC * (int64_t)sizeof(float));
            new_mc = (new_mc / GEMM_MR) * GEMM_MR;
            /* floor: keep at least 8 MR-tiles so IC parallelism stays useful.
               on 32KB L1d (zen3/4) this means MC=max(24,48)=48 — pack_a
               spills ~25% past L1d but trades that for better IC occupancy. */
            int64_t mc_floor = GEMM_MR * 8;
            if (new_mc < mc_floor) new_mc = mc_floor;
            if (new_mc < GEMM_MC) {
                AX_LOG("axiom: auto-tuned MC from %ld to %ld to fit L1d (%ld kB)\n",
                       (long)GEMM_MC, (long)new_mc, l1d / 1024);
                GEMM_MC = new_mc;
            }
        }
    }

    /* L2 auto-tune: pack_a (MC×KC) + pack_b (NC×KC) ≤ 80% of L2 */
    long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (l2 > 0) {
        long budget = (long)((double)l2 * 0.80);
        size_t pack_a_bytes = (size_t)GEMM_MC * (size_t)GEMM_KC * sizeof(float);
        size_t pack_b_bytes = (size_t)GEMM_NC * (size_t)GEMM_KC * sizeof(float);

        if ((long)(pack_a_bytes + pack_b_bytes) > budget) {
            long avail_for_b = budget - (long)pack_a_bytes;
            if (avail_for_b > 0) {
                int64_t new_nc = avail_for_b / ((int64_t)GEMM_KC * (int64_t)sizeof(float));
                new_nc = (new_nc / GEMM_NR) * GEMM_NR;
                if (new_nc >= GEMM_NR && new_nc < GEMM_NC) {
                    AX_LOG("axiom: auto-tuned NC from %ld to %ld to fit L2 (%ld kB)\n",
                           (long)GEMM_NC, (long)new_nc, l2 / 1024);
                    GEMM_NC = new_nc;
                }
            }
        }
    }

    /* L3 detection + NC bound for multi-threaded JC parallel.
       each JC thread owns a pack_b panel (NC×KC×4 bytes). when all
       threads' panels exceed 80% of shared L3, shrink NC. */
    long l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
    if (l3 > 0 && !getenv("AX_GEMM_NC")) {
        int nt = 1;
#ifdef _OPENMP
        nt = omp_get_max_threads();
#endif
        if (nt > 1) {
            long l3_budget = (long)((double)l3 * 0.80);
            long total_pb = (long)nt * (long)GEMM_NC * (long)GEMM_KC * (long)sizeof(float);
            if (total_pb > l3_budget) {
                int64_t new_nc = l3_budget / ((int64_t)nt * (int64_t)GEMM_KC * (int64_t)sizeof(float));
                new_nc = (new_nc / GEMM_NR) * GEMM_NR;
                if (new_nc >= GEMM_NR && new_nc < GEMM_NC) {
                    AX_LOG("axiom: auto-tuned NC from %ld to %ld for L3 (%ld kB, %d threads)\n",
                           (long)GEMM_NC, (long)new_nc, l3 / 1024, nt);
                    GEMM_NC = new_nc;
                }
            }
        }
    }

    /* KC auto-tune: if combined panels (MC+NC)×KC×4 exceed L2, shrink KC.
       this triggers on avx-512 configs (MC=168, NC=512) with smaller L2. */
    if (l2 > 0 && !getenv("AX_GEMM_KC")) {
        long budget = (long)((double)l2 * 0.80);
        long panels = (long)(GEMM_MC + GEMM_NC) * (long)GEMM_KC * (long)sizeof(float);
        if (panels > budget) {
            int64_t new_kc = budget / ((int64_t)(GEMM_MC + GEMM_NC) * (int64_t)sizeof(float));
            new_kc = (new_kc / 8) * 8;
            if (new_kc >= 64 && new_kc < GEMM_KC) {
                AX_LOG("axiom: auto-tuned KC from %ld to %ld to fit L2\n",
                       (long)GEMM_KC, (long)new_kc);
                GEMM_KC = new_kc;
            }
        }
    }

    AX_LOG("axiom: tiles MC=%ld NC=%ld KC=%ld (L1d=%ld kB, L2=%ld kB, L3=%ld kB)\n",
           (long)GEMM_MC, (long)GEMM_NC, (long)GEMM_KC,
           l1d > 0 ? l1d/1024 : -1, l2 > 0 ? l2/1024 : -1, l3 > 0 ? l3/1024 : -1);
#endif

    /* NT store threshold: half L3. arrays larger than this won't fit in L3
       so NT stores avoid cache pollution without extra cost. */
    if (l3 > 0) {
        AX_NT_ELEMS = (int64_t)((l3 / 2) / (long)sizeof(float));
        if (AX_NT_ELEMS < 256 * 1024) AX_NT_ELEMS = 256 * 1024; /* floor at 1MB */
    }
}

/* public entry point — called from ax_compute_init() in dispatch.c so
   baremetal targets without .init_array support still apply the env
   overrides. idempotent: second call is a no-op. suffixed variant
   under AX_CPU_ISA_DISPATCH so both isa variants coexist. */
void AX_SYM(ax_cpu_opt_tune_init)(void) {
    ax_cpu_opt_init_impl();
}

/* always-on: pre-allocate per-thread pack buffers on every omp worker.
   lazy allocation per-thread inside ensure_tl_pack_bufs() otherwise
   happens on the first gemm call that crosses the tile threshold,
   which adds tens of ms of jitter to whichever benchmark catches it
   first. paying the cost up-front makes first-call timings realistic.
   cheap: ~2-10ms depending on thread count. definition deferred until
   after ensure_tl_pack_bufs is defined; declaration only here. */
void AX_SYM(ax_cpu_opt_prewarm)(void);

/* measured tile-size calibration: opt-in via AX_GEMM_CALIBRATE=1.
   runs a small grid of (mc, nc, kc) candidates around the heuristic
   default on a 1024^3 gemm and keeps the fastest. budget ≈ 500ms. */
static double ax_tile_cal_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* forward decl — defined later in this file as a vtable entry */
static ax_status_t opt_gemm(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
static bool ensure_tl_pack_bufs(void);
static inline void pack_b_cache_invalidate(void);

void AX_SYM(ax_cpu_opt_calibrate_tiles)(void) {
#if defined(AX_NO_AUTOTUNE) || !defined(_OPENMP)
    return;
#else
    const char *calibrate = getenv("AX_GEMM_CALIBRATE");
    if (!calibrate || calibrate[0] != '1') return;
    if (getenv("AX_GEMM_MC") || getenv("AX_GEMM_NC") || getenv("AX_GEMM_KC")) return;

    int64_t mc_orig = GEMM_MC, nc_orig = GEMM_NC, kc_orig = GEMM_KC;
    struct { int64_t mc, nc, kc; } configs[] = {
        { mc_orig,             nc_orig,           kc_orig      },
        { mc_orig,             nc_orig,           kc_orig / 2  },
        { mc_orig,             nc_orig / 2,       kc_orig      },
        { (int64_t)GEMM_MR * 8,  nc_orig,         kc_orig      },
        { (int64_t)GEMM_MR * 16, nc_orig,         kc_orig      },
        { mc_orig,             (int64_t)GEMM_NR * 32, kc_orig  },
    };
    int nc_configs = (int)(sizeof(configs) / sizeof(configs[0]));

    /* score each config on TWO shapes: a square 1024^3 and a skinny
       256x1152x3136. the square exercises pack_b reuse and the JC parallel
       loop; the skinny mirrors typical conv per-sample GEMM (small M, large
       N). picking on the square alone biased toward configs that overfit
       BLAS-style benchmarks but underperformed on conv layers. score is
       the geometric mean of per-shape times so neither shape dominates. */
    struct { int64_t M, N, K; float *A, *B, *C; ax_storage_t sa, sb, sc; ax_tensor_t ta, tb, tc; }
        sh[2] = { { 1024, 1024, 1024, NULL, NULL, NULL, {0}, {0}, {0}, {0}, {0}, {0} },
                  { 256,  3136, 1152, NULL, NULL, NULL, {0}, {0}, {0}, {0}, {0}, {0} } };
    uint32_t s = 2463534242u;
    for (int sh_i = 0; sh_i < 2; sh_i++) {
        int64_t M = sh[sh_i].M, N = sh[sh_i].N, K = sh[sh_i].K;
        sh[sh_i].A = (float *)ax_aligned_alloc((size_t)M * (size_t)K * sizeof(float), 64);
        sh[sh_i].B = (float *)ax_aligned_alloc((size_t)K * (size_t)N * sizeof(float), 64);
        sh[sh_i].C = (float *)ax_aligned_alloc((size_t)M * (size_t)N * sizeof(float), 64);
        if (!sh[sh_i].A || !sh[sh_i].B || !sh[sh_i].C) {
            for (int j = 0; j <= sh_i; j++) {
                ax_aligned_free(sh[j].A); ax_aligned_free(sh[j].B); ax_aligned_free(sh[j].C);
            }
            return;
        }
        for (int64_t i = 0; i < M * K; i++) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            sh[sh_i].A[i] = (float)(s & 0xFFFF) * (1.0f / 65536.0f) - 0.5f;
        }
        for (int64_t i = 0; i < K * N; i++) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            sh[sh_i].B[i] = (float)(s & 0xFFFF) * (1.0f / 65536.0f) - 0.5f;
        }
        atomic_init(&sh[sh_i].sa.refcount, 0); sh[sh_i].sa.data = sh[sh_i].A; sh[sh_i].sa.size_bytes = (size_t)M*K*sizeof(float); sh[sh_i].sa.device = AX_DEVICE_CPU; sh[sh_i].sa.is_arena_temp = true; sh[sh_i].sa.generation = 1;
        atomic_init(&sh[sh_i].sb.refcount, 0); sh[sh_i].sb.data = sh[sh_i].B; sh[sh_i].sb.size_bytes = (size_t)K*N*sizeof(float); sh[sh_i].sb.device = AX_DEVICE_CPU; sh[sh_i].sb.is_arena_temp = true; sh[sh_i].sb.generation = 1;
        atomic_init(&sh[sh_i].sc.refcount, 0); sh[sh_i].sc.data = sh[sh_i].C; sh[sh_i].sc.size_bytes = (size_t)M*N*sizeof(float); sh[sh_i].sc.device = AX_DEVICE_CPU; sh[sh_i].sc.is_arena_temp = true; sh[sh_i].sc.generation = 1;
        sh[sh_i].ta.storage = &sh[sh_i].sa; sh[sh_i].ta.ndim = 2; sh[sh_i].ta.dtype = AX_FLOAT32; sh[sh_i].ta.shape[0] = M; sh[sh_i].ta.shape[1] = K; sh[sh_i].ta.strides[0] = K; sh[sh_i].ta.strides[1] = 1;
        sh[sh_i].tb.storage = &sh[sh_i].sb; sh[sh_i].tb.ndim = 2; sh[sh_i].tb.dtype = AX_FLOAT32; sh[sh_i].tb.shape[0] = K; sh[sh_i].tb.shape[1] = N; sh[sh_i].tb.strides[0] = N; sh[sh_i].tb.strides[1] = 1;
        sh[sh_i].tc.storage = &sh[sh_i].sc; sh[sh_i].tc.ndim = 2; sh[sh_i].tc.dtype = AX_FLOAT32; sh[sh_i].tc.shape[0] = M; sh[sh_i].tc.shape[1] = N; sh[sh_i].tc.strides[0] = N; sh[sh_i].tc.strides[1] = 1;
    }

    /* pre-size the per-thread pack buffers for the MAX of all candidate
       configs before the sweep. if we let each iteration's first gemm
       allocate them lazily, a later iteration with larger MC/KC would
       overflow the earlier-sized buffer. forcing max allocation up-front
       keeps every iteration safe. the prewarm step at init may have
       allocated tl buffers at the default MC/NC/KC; since our candidates
       can be larger, free them in every thread first so they get
       re-allocated at the max candidate size. */
    int64_t max_mc = mc_orig, max_nc = nc_orig, max_kc = kc_orig;
    for (int i = 0; i < nc_configs; i++) {
        if (configs[i].mc > max_mc) max_mc = configs[i].mc;
        if (configs[i].nc > max_nc) max_nc = configs[i].nc;
        if (configs[i].kc > max_kc) max_kc = configs[i].kc;
    }
    #pragma omp parallel
    {
        if (tl_pack_a_buf) { ax_aligned_free(tl_pack_a_buf); tl_pack_a_buf = NULL; }
        if (tl_pack_b_buf) { ax_aligned_free(tl_pack_b_buf); tl_pack_b_buf = NULL; }
        pack_b_cache_invalidate();
    }
    GEMM_MC = max_mc; GEMM_NC = max_nc; GEMM_KC = max_kc;
    #pragma omp parallel
    { ensure_tl_pack_bufs(); }

    const int warm_iters = 2;
    const int timed_iters = 5;
    double best_score = 1e30;
    double best_sq_ms = 1e30, best_sk_ms = 1e30;
    int best_i = 0;
    double t_start = ax_tile_cal_now_ms();

    for (int i = 0; i < nc_configs; i++) {
        GEMM_MC = configs[i].mc;
        GEMM_NC = configs[i].nc;
        GEMM_KC = configs[i].kc;
        /* pack_b cache keys on bptr+kc+jc+pc/ldb; any of those changing
           between iterations invalidates automatically inside pack_b_cached.
           no manual invalidation needed here. */
        double per_shape[2];
        for (int sh_i = 0; sh_i < 2; sh_i++) {
            for (int w = 0; w < warm_iters; w++)
                opt_gemm(&sh[sh_i].ta, &sh[sh_i].tb, &sh[sh_i].tc);
            double t0 = ax_tile_cal_now_ms();
            for (int r = 0; r < timed_iters; r++)
                opt_gemm(&sh[sh_i].ta, &sh[sh_i].tb, &sh[sh_i].tc);
            per_shape[sh_i] = (ax_tile_cal_now_ms() - t0) / (double)timed_iters;
        }
        /* geometric mean: equal weight to each shape's relative slowdown */
        double score = sqrt(per_shape[0] * per_shape[1]);
        if (score < best_score) {
            best_score = score; best_i = i;
            best_sq_ms = per_shape[0]; best_sk_ms = per_shape[1];
        }
    }
    GEMM_MC = configs[best_i].mc;
    GEMM_NC = configs[best_i].nc;
    GEMM_KC = configs[best_i].kc;
    double total_ms = ax_tile_cal_now_ms() - t_start;

#ifndef AX_NO_STDIO
    double sq_gflops = (2.0 * (double)sh[0].M * (double)sh[0].N * (double)sh[0].K) / (best_sq_ms * 1e-3) / 1e9;
    double sk_gflops = (2.0 * (double)sh[1].M * (double)sh[1].N * (double)sh[1].K) / (best_sk_ms * 1e-3) / 1e9;
    fprintf(stderr,
        "axiom: gemm calibrate MC=%ld NC=%ld KC=%ld (square %.1f gflops, skinny %.1f gflops, sweep %.0fms)\n",
        (long)GEMM_MC, (long)GEMM_NC, (long)GEMM_KC, sq_gflops, sk_gflops, total_ms);
#endif

    for (int sh_i = 0; sh_i < 2; sh_i++) {
        ax_aligned_free(sh[sh_i].A); ax_aligned_free(sh[sh_i].B); ax_aligned_free(sh[sh_i].C);
    }
#endif
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
    if (!tl_pack_a_buf) {
        size_t pa = (size_t)GEMM_MC * (size_t)GEMM_KC;
        if (pa / (size_t)GEMM_MC != (size_t)GEMM_KC) return false; /* overflow */
        if (pa > SIZE_MAX / sizeof(float)) return false;
        tl_pack_a_buf = (float *)ax_aligned_alloc(pa * sizeof(float), 64);
    }
    if (!tl_pack_b_buf) {
        /* round NC up to a multiple of NR for the pack buffer. when NR
           doesn't evenly divide NC (e.g. NR=12, NC=128 → nc_pack=132),
           pack_b writes ceil(NC/NR)*NR floats per KC row. without the
           round-up the buffer overflows. */
        size_t nc_rounded = (size_t)((GEMM_NC + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
        size_t pb = nc_rounded * (size_t)GEMM_KC;
        if (pb / nc_rounded != (size_t)GEMM_KC) return false;
        if (pb > SIZE_MAX / sizeof(float)) return false;
        tl_pack_b_buf = (float *)ax_aligned_alloc(pb * sizeof(float), 64);
    }
    return tl_pack_a_buf && tl_pack_b_buf;
}

void AX_SYM(ax_cpu_opt_prewarm)(void) {
#ifdef _OPENMP
    #pragma omp parallel
    { ensure_tl_pack_bufs(); }
#else
    ensure_tl_pack_bufs();
#endif
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


#if defined(AX_SIMD_AVX512)

/* 14×32 AVX-512 micro-kernel.
   28 ZMM accumulators (14 rows × 2 vectors), fully pinned in registers.
   per K iteration: 2 B loads + 14 A broadcasts + 28 FMA.
   FMA throughput: 28/2 = 14 cycles (2 FMA ports).
   broadcast throughput: 14/1 = 14 cycles (port 5).
   both co-bottleneck at 14 cycles → 28×16×2/14 = 64 FLOPs/cycle
   = theoretical peak for 2-FMA-port AVX-512. */

static void micro_kernel(int64_t kc, const float *restrict ap, const float *restrict bp,
                          float *restrict c, int64_t ldc, int64_t mr, int64_t nr)
{
    __m512 c00=_mm512_setzero_ps(), c01=_mm512_setzero_ps();
    __m512 c10=_mm512_setzero_ps(), c11=_mm512_setzero_ps();
    __m512 c20=_mm512_setzero_ps(), c21=_mm512_setzero_ps();
    __m512 c30=_mm512_setzero_ps(), c31=_mm512_setzero_ps();
    __m512 c40=_mm512_setzero_ps(), c41=_mm512_setzero_ps();
    __m512 c50=_mm512_setzero_ps(), c51=_mm512_setzero_ps();
    __m512 c60=_mm512_setzero_ps(), c61=_mm512_setzero_ps();
    __m512 c70=_mm512_setzero_ps(), c71=_mm512_setzero_ps();
    __m512 c80=_mm512_setzero_ps(), c81=_mm512_setzero_ps();
    __m512 c90=_mm512_setzero_ps(), c91=_mm512_setzero_ps();
    __m512 cA0=_mm512_setzero_ps(), cA1=_mm512_setzero_ps();
    __m512 cB0=_mm512_setzero_ps(), cB1=_mm512_setzero_ps();
    __m512 cC0=_mm512_setzero_ps(), cC1=_mm512_setzero_ps();
    __m512 cD0=_mm512_setzero_ps(), cD1=_mm512_setzero_ps();

    /* prefetch output C */
    for (int64_t row = 0; row < mr; row++) {
        __builtin_prefetch(c + row * ldc, 0, 3);
        if (nr > 16) __builtin_prefetch(c + row * ldc + 16, 0, 3);
    }

    /* 2× unrolled K loop with prefetch */
    #define AVX512_BODY(a_ptr, b_ptr) { \
        const float *_ap = (const float *)(a_ptr); \
        const float *_bp = (const float *)(b_ptr); \
        __m512 b0 = _mm512_load_ps(_bp); \
        __m512 b1 = _mm512_load_ps(_bp + 16); \
        __m512 a; \
        a=_mm512_set1_ps(_ap[ 0]); c00=_mm512_fmadd_ps(a,b0,c00); c01=_mm512_fmadd_ps(a,b1,c01); \
        a=_mm512_set1_ps(_ap[ 1]); c10=_mm512_fmadd_ps(a,b0,c10); c11=_mm512_fmadd_ps(a,b1,c11); \
        a=_mm512_set1_ps(_ap[ 2]); c20=_mm512_fmadd_ps(a,b0,c20); c21=_mm512_fmadd_ps(a,b1,c21); \
        a=_mm512_set1_ps(_ap[ 3]); c30=_mm512_fmadd_ps(a,b0,c30); c31=_mm512_fmadd_ps(a,b1,c31); \
        a=_mm512_set1_ps(_ap[ 4]); c40=_mm512_fmadd_ps(a,b0,c40); c41=_mm512_fmadd_ps(a,b1,c41); \
        a=_mm512_set1_ps(_ap[ 5]); c50=_mm512_fmadd_ps(a,b0,c50); c51=_mm512_fmadd_ps(a,b1,c51); \
        a=_mm512_set1_ps(_ap[ 6]); c60=_mm512_fmadd_ps(a,b0,c60); c61=_mm512_fmadd_ps(a,b1,c61); \
        a=_mm512_set1_ps(_ap[ 7]); c70=_mm512_fmadd_ps(a,b0,c70); c71=_mm512_fmadd_ps(a,b1,c71); \
        a=_mm512_set1_ps(_ap[ 8]); c80=_mm512_fmadd_ps(a,b0,c80); c81=_mm512_fmadd_ps(a,b1,c81); \
        a=_mm512_set1_ps(_ap[ 9]); c90=_mm512_fmadd_ps(a,b0,c90); c91=_mm512_fmadd_ps(a,b1,c91); \
        a=_mm512_set1_ps(_ap[10]); cA0=_mm512_fmadd_ps(a,b0,cA0); cA1=_mm512_fmadd_ps(a,b1,cA1); \
        a=_mm512_set1_ps(_ap[11]); cB0=_mm512_fmadd_ps(a,b0,cB0); cB1=_mm512_fmadd_ps(a,b1,cB1); \
        a=_mm512_set1_ps(_ap[12]); cC0=_mm512_fmadd_ps(a,b0,cC0); cC1=_mm512_fmadd_ps(a,b1,cC1); \
        a=_mm512_set1_ps(_ap[13]); cD0=_mm512_fmadd_ps(a,b0,cD0); cD1=_mm512_fmadd_ps(a,b1,cD1); \
    }

    int64_t p = 0;
    int64_t kc2 = kc - (kc & 1);
    for (; p < kc2; p += 2) {
        __builtin_prefetch(ap + 8 * GEMM_MR, 0, 3);
        __builtin_prefetch(bp + 8 * GEMM_NR, 0, 3);
        AVX512_BODY(ap, bp);
        AVX512_BODY(ap + GEMM_MR, bp + GEMM_NR);
        ap += 2 * GEMM_MR;
        bp += 2 * GEMM_NR;
    }
    if (p < kc) {
        AVX512_BODY(ap, bp);
        ap += GEMM_MR;
        bp += GEMM_NR;
    }
    #undef AVX512_BODY

    /* writeback */
    if (mr == GEMM_MR && nr == GEMM_NR) {
        #define STORE_ROW(row, lo, hi) \
            _mm512_storeu_ps(c + (row)*ldc,      _mm512_add_ps(lo, _mm512_loadu_ps(c + (row)*ldc))); \
            _mm512_storeu_ps(c + (row)*ldc + 16, _mm512_add_ps(hi, _mm512_loadu_ps(c + (row)*ldc + 16)));
        STORE_ROW( 0,c00,c01); STORE_ROW( 1,c10,c11); STORE_ROW( 2,c20,c21); STORE_ROW( 3,c30,c31);
        STORE_ROW( 4,c40,c41); STORE_ROW( 5,c50,c51); STORE_ROW( 6,c60,c61); STORE_ROW( 7,c70,c71);
        STORE_ROW( 8,c80,c81); STORE_ROW( 9,c90,c91); STORE_ROW(10,cA0,cA1); STORE_ROW(11,cB0,cB1);
        STORE_ROW(12,cC0,cC1); STORE_ROW(13,cD0,cD1);
        #undef STORE_ROW
    } else {
        float buf[GEMM_MR * GEMM_NR] __attribute__((aligned(64)));
        #define EXT_ROW(row, lo, hi) \
            _mm512_store_ps(buf + (row)*GEMM_NR,      lo); \
            _mm512_store_ps(buf + (row)*GEMM_NR + 16, hi);
        EXT_ROW( 0,c00,c01); EXT_ROW( 1,c10,c11); EXT_ROW( 2,c20,c21); EXT_ROW( 3,c30,c31);
        EXT_ROW( 4,c40,c41); EXT_ROW( 5,c50,c51); EXT_ROW( 6,c60,c61); EXT_ROW( 7,c70,c71);
        EXT_ROW( 8,c80,c81); EXT_ROW( 9,c90,c91); EXT_ROW(10,cA0,cA1); EXT_ROW(11,cB0,cB1);
        EXT_ROW(12,cC0,cC1); EXT_ROW(13,cD0,cD1);
        #undef EXT_ROW
        for (int64_t ii = 0; ii < mr; ii++)
            for (int64_t jj = 0; jj < nr; jj++)
                c[ii * ldc + jj] += buf[ii * GEMM_NR + jj];
    }
}

#elif defined(AX_SIMD_AVX2)

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

    /* 2× unrolled K loop. processes two K iterations per loop body:
       - amortizes the loop branch overhead (cmp+jne every 2 iters instead of 1)
       - the ooo engine can overlap iteration N+1's loads with N's FMA
       - prefetch at 4-iteration distance into L1 */

    #define KERNEL_BODY(a_ptr, b_ptr) \
    { \
        __m256 b0 = _mm256_load_ps(b_ptr); \
        __m256 b1 = _mm256_load_ps(b_ptr + 8); \
        __m256 a0 = _mm256_broadcast_ss(a_ptr + 0); \
        c00 = _mm256_fmadd_ps(a0, b0, c00); c01 = _mm256_fmadd_ps(a0, b1, c01); \
        __m256 a1 = _mm256_broadcast_ss(a_ptr + 1); \
        c10 = _mm256_fmadd_ps(a1, b0, c10); c11 = _mm256_fmadd_ps(a1, b1, c11); \
        __m256 a2 = _mm256_broadcast_ss(a_ptr + 2); \
        c20 = _mm256_fmadd_ps(a2, b0, c20); c21 = _mm256_fmadd_ps(a2, b1, c21); \
        __m256 a3 = _mm256_broadcast_ss(a_ptr + 3); \
        c30 = _mm256_fmadd_ps(a3, b0, c30); c31 = _mm256_fmadd_ps(a3, b1, c31); \
        __m256 a4 = _mm256_broadcast_ss(a_ptr + 4); \
        c40 = _mm256_fmadd_ps(a4, b0, c40); c41 = _mm256_fmadd_ps(a4, b1, c41); \
        __m256 a5 = _mm256_broadcast_ss(a_ptr + 5); \
        c50 = _mm256_fmadd_ps(a5, b0, c50); c51 = _mm256_fmadd_ps(a5, b1, c51); \
    }

    /* prefetch distance tuned for Haswell/Broadwell/Zen: 16 K-steps
       ahead for B (the larger stream, one cache line per K), 8 for A.
       this lets two ooo iterations overlap their B-line fetch with the
       current FMA cascade without evicting the working set. measured
       +2-3% on huge gemm 4096/8192 vs the previous 8-step distance. */
    int64_t p = 0;
    int64_t kc2 = kc - (kc & 1);
    for (; p < kc2; p += 2) {
        __builtin_prefetch(ap + 8 * GEMM_MR, 0, 3);
        __builtin_prefetch(bp + 16 * GEMM_NR, 0, 3);
        __builtin_prefetch(bp + 16 * GEMM_NR + 8, 0, 3);
        KERNEL_BODY(ap, bp);
        KERNEL_BODY(ap + GEMM_MR, bp + GEMM_NR);
        ap += 2 * GEMM_MR;
        bp += 2 * GEMM_NR;
    }
    /* odd tail */
    if (p < kc) {
        KERNEL_BODY(ap, bp);
        ap += GEMM_MR;
        bp += GEMM_NR;
    }
    #undef KERNEL_BODY

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

#elif defined(AX_SIMD_NEON)

/* 8×12 NEON micro-kernel using vfmaq_laneq_f32.
   24 Q accumulators (8 rows × 3 vectors of 4 floats).
   per K iteration: 3 B loads (b0..b2) + 2 A loads (a_lo, a_hi as Q regs)
   + 24 FMLA via lane-broadcast (vfmaq_laneq_f32 is a single instruction
   on A64: FMLA Vd.4S, Vn.4S, Vm.S[lane]).
   this avoids the scalar broadcast + separate FMA of the generic path,
   giving ~2× the throughput on Cortex-A76 and Neoverse N1. */

static void micro_kernel(int64_t kc, const float *restrict ap, const float *restrict bp,
                          float *restrict c, int64_t ldc, int64_t mr, int64_t nr)
{
    float32x4_t c00=vdupq_n_f32(0), c01=vdupq_n_f32(0), c02=vdupq_n_f32(0);
    float32x4_t c10=vdupq_n_f32(0), c11=vdupq_n_f32(0), c12=vdupq_n_f32(0);
    float32x4_t c20=vdupq_n_f32(0), c21=vdupq_n_f32(0), c22=vdupq_n_f32(0);
    float32x4_t c30=vdupq_n_f32(0), c31=vdupq_n_f32(0), c32=vdupq_n_f32(0);
    float32x4_t c40=vdupq_n_f32(0), c41=vdupq_n_f32(0), c42=vdupq_n_f32(0);
    float32x4_t c50=vdupq_n_f32(0), c51=vdupq_n_f32(0), c52=vdupq_n_f32(0);
    float32x4_t c60=vdupq_n_f32(0), c61=vdupq_n_f32(0), c62=vdupq_n_f32(0);
    float32x4_t c70=vdupq_n_f32(0), c71=vdupq_n_f32(0), c72=vdupq_n_f32(0);

    /* prefetch output C into L1 */
    for (int64_t row = 0; row < mr; row++) {
        __builtin_prefetch(c + row * ldc, 0, 3);
        if (nr > 4) __builtin_prefetch(c + row * ldc + 4, 0, 3);
        if (nr > 8) __builtin_prefetch(c + row * ldc + 8, 0, 3);
    }

    /* 2× unrolled K loop with prefetch */
    #define NEON_BODY(a_ptr, b_ptr) { \
        const float *_ap = (const float *)(a_ptr); \
        const float *_bp = (const float *)(b_ptr); \
        float32x4_t b0 = vld1q_f32(_bp); \
        float32x4_t b1 = vld1q_f32(_bp + 4); \
        float32x4_t b2 = vld1q_f32(_bp + 8); \
        float32x4_t a_lo = vld1q_f32(_ap); \
        float32x4_t a_hi = vld1q_f32(_ap + 4); \
        c00=vfmaq_laneq_f32(c00,b0,a_lo,0); c01=vfmaq_laneq_f32(c01,b1,a_lo,0); c02=vfmaq_laneq_f32(c02,b2,a_lo,0); \
        c10=vfmaq_laneq_f32(c10,b0,a_lo,1); c11=vfmaq_laneq_f32(c11,b1,a_lo,1); c12=vfmaq_laneq_f32(c12,b2,a_lo,1); \
        c20=vfmaq_laneq_f32(c20,b0,a_lo,2); c21=vfmaq_laneq_f32(c21,b1,a_lo,2); c22=vfmaq_laneq_f32(c22,b2,a_lo,2); \
        c30=vfmaq_laneq_f32(c30,b0,a_lo,3); c31=vfmaq_laneq_f32(c31,b1,a_lo,3); c32=vfmaq_laneq_f32(c32,b2,a_lo,3); \
        c40=vfmaq_laneq_f32(c40,b0,a_hi,0); c41=vfmaq_laneq_f32(c41,b1,a_hi,0); c42=vfmaq_laneq_f32(c42,b2,a_hi,0); \
        c50=vfmaq_laneq_f32(c50,b0,a_hi,1); c51=vfmaq_laneq_f32(c51,b1,a_hi,1); c52=vfmaq_laneq_f32(c52,b2,a_hi,1); \
        c60=vfmaq_laneq_f32(c60,b0,a_hi,2); c61=vfmaq_laneq_f32(c61,b1,a_hi,2); c62=vfmaq_laneq_f32(c62,b2,a_hi,2); \
        c70=vfmaq_laneq_f32(c70,b0,a_hi,3); c71=vfmaq_laneq_f32(c71,b1,a_hi,3); c72=vfmaq_laneq_f32(c72,b2,a_hi,3); \
    }

    int64_t p = 0;
    int64_t kc2 = kc - (kc & 1);
    for (; p < kc2; p += 2) {
        __builtin_prefetch(ap + 8 * GEMM_MR, 0, 3);
        __builtin_prefetch(bp + 8 * GEMM_NR, 0, 3);
        NEON_BODY(ap, bp);
        NEON_BODY(ap + GEMM_MR, bp + GEMM_NR);
        ap += 2 * GEMM_MR;
        bp += 2 * GEMM_NR;
    }
    if (p < kc) {
        NEON_BODY(ap, bp);
        ap += GEMM_MR;
        bp += GEMM_NR;
    }
    #undef NEON_BODY

    /* writeback */
    if (mr == GEMM_MR && nr == GEMM_NR) {
        #define NEON_STORE_ROW(row, v0, v1, v2) \
            vst1q_f32(c + (row)*ldc,     vaddq_f32(v0, vld1q_f32(c + (row)*ldc))); \
            vst1q_f32(c + (row)*ldc + 4, vaddq_f32(v1, vld1q_f32(c + (row)*ldc + 4))); \
            vst1q_f32(c + (row)*ldc + 8, vaddq_f32(v2, vld1q_f32(c + (row)*ldc + 8)));
        NEON_STORE_ROW(0, c00, c01, c02); NEON_STORE_ROW(1, c10, c11, c12);
        NEON_STORE_ROW(2, c20, c21, c22); NEON_STORE_ROW(3, c30, c31, c32);
        NEON_STORE_ROW(4, c40, c41, c42); NEON_STORE_ROW(5, c50, c51, c52);
        NEON_STORE_ROW(6, c60, c61, c62); NEON_STORE_ROW(7, c70, c71, c72);
        #undef NEON_STORE_ROW
    } else {
        float buf[GEMM_MR * GEMM_NR] __attribute__((aligned(64)));
        #define NEON_EXT(row, v0, v1, v2) \
            vst1q_f32(buf + (row)*GEMM_NR,     v0); \
            vst1q_f32(buf + (row)*GEMM_NR + 4, v1); \
            vst1q_f32(buf + (row)*GEMM_NR + 8, v2);
        NEON_EXT(0, c00, c01, c02); NEON_EXT(1, c10, c11, c12);
        NEON_EXT(2, c20, c21, c22); NEON_EXT(3, c30, c31, c32);
        NEON_EXT(4, c40, c41, c42); NEON_EXT(5, c50, c51, c52);
        NEON_EXT(6, c60, c61, c62); NEON_EXT(7, c70, c71, c72);
        #undef NEON_EXT
        for (int64_t ii = 0; ii < mr; ii++)
            for (int64_t jj = 0; jj < nr; jj++)
                c[ii * ldc + jj] += buf[ii * GEMM_NR + jj];
    }
}

#else

/* generic scalar micro-kernel. slowest but correct on any platform. */
static void micro_kernel(int64_t kc, const float *ap, const float *bp,
                          float *c, int64_t ldc, int64_t mr, int64_t nr)
{
    #define NVEC (GEMM_NR / AX_VF32_WIDTH)
    ax_vf32 acc[GEMM_MR][NVEC];
    for (int ii = 0; ii < GEMM_MR; ii++)
        for (int v = 0; v < NVEC; v++)
            acc[ii][v] = ax_vf32_zero();

    for (int64_t p = 0; p < kc; p++) {
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

    /* per-thread pack buffers — guarded against nested parallel regions.
       on hybrid cpus, large GEMMs use all cores (P+E), small ones use
       only P-cores for lower latency. */
    int64_t total_flops_est = 2 * m * n * k;
    int max_threads = ax_gemm_threads_for_shape(m, n, k);

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
    /* fine-grained parallel: ~1M FLOPs threshold for fork-join amortization */
    bool use_fine_par = !use_jc_par && !use_ic_par && (max_threads > 1)
                        && (m <= GEMM_MC) && (fine_units >= 4)
                        && (total_flops_est > 1000000);

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

/* SIMD + OMP sum across a contiguous float array. the crossover point
   between serial and parallel depends on measured omp fork/join cost,
   set at init time via ax_par_threshold_elems. below that threshold
   the serial simd_row_sum wins. */
extern int64_t ax_par_threshold_elems;
static inline float simd_row_sum_par(const float *d, int64_t n)
{
#ifdef _OPENMP
    if (n >= ax_par_threshold_elems) {
        int T = omp_get_max_threads();
        if (T > 16) T = 16;
        if (T > 1) {
            double partials[16];
            int64_t chunk = (n + T - 1) / T;
            #pragma omp parallel for num_threads(T) schedule(static)
            for (int t = 0; t < T; t++) {
                int64_t lo = (int64_t)t * chunk;
                int64_t hi = lo + chunk; if (hi > n) hi = n;
                partials[t] = (lo < hi) ? (double)simd_row_sum(d + lo, hi - lo) : 0.0;
            }
            double total = 0.0;
            for (int t = 0; t < T; t++) total += partials[t];
            return (float)total;
        }
    }
#endif
    return simd_row_sum(d, n);
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
        raw_f32(out)[0] = simd_row_sum_par(raw_f32(in), n);
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
        raw_f32(out)[0] = simd_row_sum_par(raw_f32(in), n) / (float)n;
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

    int64_t total_flops = 2 * m * n * k;
    int max_threads = ax_gemm_threads_for_shape(m, n, k);
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

    int64_t total_flops = 2 * m * n * k;
    int max_threads = ax_gemm_threads_for_shape(m, n, k);
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
    int use_nt = (na >= AX_NT_ELEMS);
    for (; i < ve; i += AX_VF32_WIDTH) {
        ax_vf32 va = ax_vf32_load(ad + i);
        ax_vf32 vb = ax_vf32_load(bd + i);
        if (use_nt) ax_vf32_stream(od + i, ax_vf32_relu(ax_vf32_add(va, vb)));
        else        ax_vf32_store(od + i, ax_vf32_relu(ax_vf32_add(va, vb)));
    }
    if (use_nt) ax_nt_fence();
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
    /* axpy reads and writes yd, so nt stores aren't correct here
       (we'd stream past cached yd values that other ops still read).
       skip NT for axpy. */
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

/* fused matmul + bias + relu: out = relu(a @ b + bias).
   does the gemm via opt_gemm, then a single simd pass that adds bias
   and applies relu in-place. the output is cache-hot from the gemm
   write, so the fused pass hits L2/L3 instead of streaming from DRAM.
   saves one full read+write pass vs separate gemm → bias_add → relu. */
static ax_status_t opt_gemm_relu(const ax_tensor_t *a, const ax_tensor_t *b,
                                  const ax_tensor_t *bias, ax_tensor_t *out)
{
    ax_status_t s = opt_gemm(a, b, out);
    if (s != AX_OK) return s;

    int64_t m = out->shape[0], n = out->shape[1];
    float *od = raw_f32(out);
    const float *bd = bias ? (const float *)bias->storage->data + bias->offset : NULL;

    /* fused bias + relu in a single pass while output is cache-hot */
    if (bd) {
        ax_vf32 vzero = ax_vf32_zero();
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(m * n > ax_par_threshold())
        #endif
        for (int64_t i = 0; i < m; i++) {
            float *row = od + i * n;
            int64_t j = 0, ve = n - (n % AX_VF32_WIDTH);
            for (; j < ve; j += AX_VF32_WIDTH) {
                ax_vf32 v = ax_vf32_add(ax_vf32_loadu(row + j), ax_vf32_loadu(bd + j));
                ax_vf32_storeu(row + j, ax_vf32_max(v, vzero));
            }
            for (; j < n; j++) {
                float v = row[j] + bd[j];
                row[j] = v > 0.0f ? v : 0.0f;
            }
        }
    } else {
        /* no bias — just relu */
        int64_t total = m * n;
        ax_vf32 vzero = ax_vf32_zero();
        int64_t i = 0, ve = total - (total % AX_VF32_WIDTH);
        for (; i < ve; i += AX_VF32_WIDTH)
            ax_vf32_storeu(od + i, ax_vf32_max(ax_vf32_loadu(od + i), vzero));
        for (; i < total; i++)
            od[i] = od[i] > 0.0f ? od[i] : 0.0f;
    }
    return AX_OK;
}

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


/* free per-thread pack buffers so sanitizer runs come up clean.
   called from ax_compute_shutdown via the vtable .shutdown hook. */
static void opt_shutdown(void) {
#ifdef _OPENMP
    #pragma omp parallel
    {
        if (tl_pack_a_buf) { ax_aligned_free(tl_pack_a_buf); tl_pack_a_buf = NULL; }
        if (tl_pack_b_buf) { ax_aligned_free(tl_pack_b_buf); tl_pack_b_buf = NULL; }
    }
#else
    if (tl_pack_a_buf) { ax_aligned_free(tl_pack_a_buf); tl_pack_a_buf = NULL; }
    if (tl_pack_b_buf) { ax_aligned_free(tl_pack_b_buf); tl_pack_b_buf = NULL; }
#endif
}

/* vtable registration */

const ax_backend_ops_t AX_SYM(ax_cpu_opt_ops) = {
    .name       = "cpu_opt",
    .shutdown   = opt_shutdown,
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
    .gemm_relu  = opt_gemm_relu,
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

/* ================================================================
   SCALED DOT-PRODUCT ATTENTION — fused flashattention-style kernel

   design principles:
   1. score tiles never leave L1. process Q in MR-row strips (6 rows for
      AVX2, 14 for AVX-512, 8 for NEON). score_strip is MR × BK ≤ 3-8 KB,
      fits easily in L1d. GEMM → softmax → V-multiply all operate on
      this L1-resident strip, eliminating the 64 KB L2 round-trip that a
      naive (separate score tile, separate softmax pass) has.

   2. scale is baked into packed K^T, not applied as a separate pass.
      scale_packed() multiplies the NR-col-strip layout of Kt by 1/√dk
      once per kj block. all subsequent score computations using this Kt
      produce pre-scaled values.

   3. K^T and V are packed ONCE per kj block and reused across every qi
      block. this is the key locality win — a 128-row K panel is 64 KB
      but gets accessed S/BQ = 8 times for S=1024.

   4. online softmax with output correction fused into the strip loop.
      as we accumulate contributions from each kj block, the running
      row_max may increase. when it does, previous output contributions
      get multiplied by exp(old_max - new_max) to correct them. this is
      done row-by-row inside the strip loop while out[qi+ir] is L1-hot.

   5. causal masking uses three-tier logic:
      - entire strip is future (qi+ir+MR-1 < kj): skip the strip
      - entire strip is past (qi+ir >= kj+BK-1): normal, no mask
      - partial overlap: apply causal mask within the softmax row loop
      this avoids per-element masking when it's not needed.

   6. backward uses recompute, not P-materialization. on CPUs with
      modest L3 (4-18 MB typical), materializing per-head S×S scores
      thrashes L3. recomputing scores from Q/K inside the backward is
      actually cheaper when per-head S*S*4 exceeds ~half of shared L3.
      the score recompute uses the SAME MR-strip fused pipeline as
      the forward, so it costs ~30% of a full forward, not 100%.

   ================================================================ */

#include <time.h>

#define ATTN_BQ_MAX 128
#define ATTN_BK_MAX 128
#define ATTN_MAX_DK 256

/* attention tile sizes — runtime, defaults picked so score_strip stays in L1d. */
static int64_t ATTN_BQ = 128;
static int64_t ATTN_BK = 128;

/* in-place multiply a packed panel by a scalar (SIMD-vectorized).
   used to bake 1/√dk into Kt_packed so the score GEMM output is
   pre-scaled without a separate pass over the score tile. */
static inline void attn_scale_packed(float *buf, int64_t n, float s) {
    int64_t c = 0;
#if defined(AX_HAS_SIMD)
    ax_vf32 vs = ax_vf32_set1(s);
    int64_t ve = n - (n % AX_VF32_WIDTH);
    for (; c < ve; c += AX_VF32_WIDTH)
        ax_vf32_storeu(buf + c, ax_vf32_mul(ax_vf32_loadu(buf + c), vs));
#endif
    for (; c < n; c++) buf[c] *= s;
}

/* online softmax + output correction for a single row of the score strip.
   sr is the scaled score row (length Bk), to be overwritten with exp(sr - new_max).
   row_max_io and row_sum_io track running state across kj blocks.
   out_row is the accumulated output row, to be corrected in-place.
   returns exp(old_max - new_max), the correction factor. */
static inline void attn_fwd_softmax_row(float *sr, float *out_row,
                                         float *row_max_io, float *row_sum_io,
                                         int64_t Bk, int64_t dk)
{
    float tmx = -FLT_MAX;
    int64_t j = 0;
#if defined(AX_HAS_SIMD)
    ax_vf32 vmx = ax_vf32_set1(-FLT_MAX);
    int64_t bve = Bk - (Bk % AX_VF32_WIDTH);
    for (; j < bve; j += AX_VF32_WIDTH)
        vmx = ax_vf32_max(vmx, ax_vf32_loadu(sr + j));
    tmx = ax_vf32_hmax(vmx);
#endif
    for (; j < Bk; j++) if (sr[j] > tmx) tmx = sr[j];

    float om = *row_max_io;
    float nm = (om > tmx) ? om : tmx;
    float corr = (om == -FLT_MAX) ? 0.0f : expf(om - nm);

    float ts = 0;
    j = 0;
#if defined(AX_HAS_SIMD)
    ax_vf32 vnm = ax_vf32_set1(nm), vts = ax_vf32_zero();
    for (; j < bve; j += AX_VF32_WIDTH) {
        ax_vf32 v = ax_vf32_exp(ax_vf32_sub(ax_vf32_loadu(sr + j), vnm));
        ax_vf32_storeu(sr + j, v);
        vts = ax_vf32_add(vts, v);
    }
    ts = ax_vf32_hsum(vts);
#endif
    for (; j < Bk; j++) { sr[j] = expf(sr[j] - nm); ts += sr[j]; }

    *row_sum_io = *row_sum_io * corr + ts;
    *row_max_io = nm;

    /* apply correction to previously-accumulated output row (unless this
       is the first kj block, signaled by old_max == -FLT_MAX). */
    if (om != -FLT_MAX) {
        int64_t d = 0;
#if defined(AX_HAS_SIMD)
        ax_vf32 vc = ax_vf32_set1(corr);
        int64_t dve = dk - (dk % AX_VF32_WIDTH);
        for (; d < dve; d += AX_VF32_WIDTH)
            ax_vf32_storeu(out_row + d, ax_vf32_mul(ax_vf32_loadu(out_row + d), vc));
#endif
        for (; d < dk; d++) out_row[d] *= corr;
    }
}

/* apply causal mask to a score strip: set sr[j] = -inf for j > qi_abs.
   qi_abs is the absolute query position (qi + ir + r).
   kj is the column offset of this kj block.
   Bk is the width of this block. */
static inline void attn_apply_causal_mask(float *sr, int64_t qi_abs,
                                            int64_t kj, int64_t Bk)
{
    /* sr corresponds to score for positions kj, kj+1, ..., kj+Bk-1.
       mask all positions strictly greater than qi_abs. */
    if (qi_abs >= kj + Bk - 1) return; /* nothing to mask in this block */
    int64_t first_mask = qi_abs - kj + 1;
    if (first_mask < 0) first_mask = 0;
    for (int64_t j = first_mask; j < Bk; j++) sr[j] = -FLT_MAX;
}

/* apply padding mask to a score strip: set sr[j] = -inf wherever pad_mask[kj+j] == 0. */
static inline void attn_apply_pad_mask(float *sr, int64_t kj, int64_t Bk,
                                         const int8_t *pad_mask)
{
    for (int64_t j = 0; j < Bk; j++)
        if (!pad_mask[kj + j]) sr[j] = -FLT_MAX;
}

/* process one (qi, kj) forward tile in MR-row strips.
   Kt_packed must be pre-scaled by 1/√dk.
   Bq, Bk are the actual tile dimensions (may be less than ATTN_BQ/BK on edges).
   qi_base, kj_base are absolute positions within the sequence (for masking). */
static inline void attn_fwd_tile_mr(const float *Q_qi, int64_t dk,
                                      const float *Kt_packed, const float *V_packed,
                                      float *out_qi, float *row_max_qi, float *row_sum_qi,
                                      int64_t Bq, int64_t Bk, int64_t dk_np,
                                      int64_t qi_base, int64_t kj_base,
                                      bool causal, const int8_t *pad_mask)
{
    int64_t Bk_np = ((Bk + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
    float score_strip[GEMM_MR * ATTN_BK_MAX] __attribute__((aligned(64)));
    float a_q[GEMM_MR * ATTN_MAX_DK] __attribute__((aligned(64)));
    float a_s[GEMM_MR * ATTN_BK_MAX] __attribute__((aligned(64)));

    int64_t dk_p = ((dk + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;

    /* causal three-tier: skip entire strip if fully in the future. */
    for (int64_t ir = 0; ir < Bq; ir += GEMM_MR) {
        int64_t mr = (ir + GEMM_MR <= Bq) ? GEMM_MR : (Bq - ir);
        int64_t qi_abs_start = qi_base + ir;
        int64_t qi_abs_end = qi_abs_start + mr - 1;

        if (causal && qi_abs_end < kj_base) continue; /* entire strip is future */

        int64_t mr_p = ((mr + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;
        /* need masking iff ANY row in the strip has a future key in this
           block. the earliest row qi_abs_start gates the earliest future
           column; if qi_abs_start < kj+Bk-1 then row 0 still sees future
           keys. using qi_abs_end would miss strips that straddle the
           causal boundary (later rows fully masked, earlier rows partial). */
        bool need_causal_mask = causal && (qi_abs_start < kj_base + Bk - 1);

        /* 1. pack Q strip and compute score = Q @ Kt (scale baked in) */
        pack_a(Q_qi + ir * dk, dk, mr_p, dk, mr, a_q);
        memset(score_strip, 0, (size_t)(mr * Bk) * sizeof(float));
        for (int64_t jr = 0; jr < Bk_np; jr += GEMM_NR) {
            int64_t nr = (jr + GEMM_NR <= Bk) ? GEMM_NR : (Bk > jr ? Bk - jr : 0);
            if (nr <= 0) break;
            micro_kernel(dk, a_q, Kt_packed + jr * dk,
                         score_strip + jr, Bk, mr, nr);
        }

        /* 2. apply masks if needed */
        if (pad_mask) {
            for (int64_t r = 0; r < mr; r++)
                attn_apply_pad_mask(score_strip + r * Bk, kj_base, Bk, pad_mask);
        }
        if (need_causal_mask) {
            for (int64_t r = 0; r < mr; r++)
                attn_apply_causal_mask(score_strip + r * Bk,
                                        qi_base + ir + r, kj_base, Bk);
        }

        /* 3. online softmax + output correction (fused per row) */
        for (int64_t r = 0; r < mr; r++) {
            attn_fwd_softmax_row(score_strip + r * Bk,
                                  out_qi + (ir + r) * dk,
                                  row_max_qi + ir + r,
                                  row_sum_qi + ir + r,
                                  Bk, dk);
        }

        /* 4. V multiply: out[MR, dk] += score_strip[MR, Bk] @ V_packed[Bk, dk] */
        pack_a(score_strip, Bk, mr_p, Bk, mr, a_s);
        (void)dk_p;
        for (int64_t jr = 0; jr < dk_np; jr += GEMM_NR) {
            int64_t nr = (jr + GEMM_NR <= dk) ? GEMM_NR : (dk > jr ? dk - jr : 0);
            if (nr <= 0) break;
            micro_kernel(Bk, a_s, V_packed + jr * Bk,
                         out_qi + ir * dk + jr, dk, mr, nr);
        }
    }
}

/* per-head forward pass. Q/K/V/out are [S, dk] contiguous for this head.
   row_max, row_sum are [S] working buffers (caller allocates + initializes to
   -FLT_MAX / 0). L may be NULL to skip logsumexp writeout (inference). */
static void attn_fwd_head(const float *Q, const float *K, const float *V,
                           float *out, float *L, float *row_max, float *row_sum,
                           int64_t S, int64_t dk, float scale,
                           bool causal, const int8_t *pad_mask)
{
    int64_t dk_np = ((dk + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
    int64_t Bk_np_max = ((ATTN_BK + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;

    /* per-kj-block packed panels. Kt gets pre-scaled. TLS-backed so the
       per-head loop doesn't allocate each iteration — on BH=64 each
       thread would otherwise do 32 aligned_alloc + free round trips. */
    int64_t kt_want = (int64_t)(dk * Bk_np_max) * (int64_t)sizeof(float);
    int64_t v_want  = (int64_t)(ATTN_BK * dk_np) * (int64_t)sizeof(float);
    float *Kt_packed = ax_tls_grow(&tl_sdpa_kt_packed, &tl_sdpa_kt_bytes, kt_want);
    float *V_packed  = ax_tls_grow(&tl_sdpa_v_packed,  &tl_sdpa_v_bytes,  v_want);
    if (!Kt_packed || !V_packed) return;

    for (int64_t i = 0; i < S; i++) { row_max[i] = -FLT_MAX; row_sum[i] = 0.0f; }
    memset(out, 0, (size_t)(S * dk) * sizeof(float));

    for (int64_t kj = 0; kj < S; kj += ATTN_BK) {
        int64_t Bk = (kj + ATTN_BK <= S) ? ATTN_BK : (S - kj);
        int64_t Bk_np = ((Bk + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;

        /* pack K^T and V for this kj block */
        pack_b_t(K + kj * dk, dk, dk, Bk_np, Bk, Kt_packed);
        attn_scale_packed(Kt_packed, dk * Bk_np, scale);  /* bake 1/√dk into Kt */
        pack_b(V + kj * dk, dk, Bk, dk_np, dk, V_packed);

        for (int64_t qi = 0; qi < S; qi += ATTN_BQ) {
            int64_t Bq = (qi + ATTN_BQ <= S) ? ATTN_BQ : (S - qi);

            /* causal: skip this (qi, kj) if entire block is in the future */
            if (causal && qi + Bq - 1 < kj) continue;

            attn_fwd_tile_mr(Q + qi * dk, dk, Kt_packed, V_packed,
                              out + qi * dk, row_max + qi, row_sum + qi,
                              Bq, Bk, dk_np, qi, kj, causal, pad_mask);
        }
    }

    /* final normalization: out[i] /= row_sum[i], L[i] = row_max[i] + log(row_sum[i]) */
    for (int64_t i = 0; i < S; i++) {
        float rs = row_sum[i];
        if (rs > 0.0f) {
            float inv = 1.0f / rs;
            float *o = out + i * dk;
            int64_t d = 0;
#if defined(AX_HAS_SIMD)
            ax_vf32 vi = ax_vf32_set1(inv);
            int64_t ve = dk - (dk % AX_VF32_WIDTH);
            for (; d < ve; d += AX_VF32_WIDTH)
                ax_vf32_storeu(o + d, ax_vf32_mul(ax_vf32_loadu(o + d), vi));
#endif
            for (; d < dk; d++) o[d] *= inv;
            if (L) L[i] = row_max[i] + logf(rs);
        } else if (L) {
            L[i] = -FLT_MAX;
        }
    }
    /* Kt_packed and V_packed are TLS, do not free */
}

/* batched SDPA forward. processes BH heads in parallel. per-thread
   scratch for row_max/row_sum is TLS so we don't malloc 2 * nthreads
   times per call — sdpa_fwd ran on a hot training loop allocates
   thousands of times per epoch otherwise. */
void AX_SYM(ax_cpu_sdpa_fwd)(const float *Q, const float *K, const float *V,
                              float *out, float *L,
                              int64_t BH, int64_t S, int64_t dk, float scale,
                              bool causal, const int8_t *pad_mask)
{
    int64_t head_sz = S * dk;

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        /* grow per-thread row_max/row_sum lazily to fit this call's S. */
        int64_t want = (int64_t)S * (int64_t)sizeof(float);
        float *row_max = ax_tls_grow(&tl_sdpa_row_max, &tl_sdpa_row_max_S, want);
        float *row_sum = ax_tls_grow(&tl_sdpa_row_sum, &tl_sdpa_row_sum_S, want);
        if (!row_max || !row_sum) goto done;

#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int64_t h = 0; h < BH; h++) {
            attn_fwd_head(Q + h * head_sz, K + h * head_sz, V + h * head_sz,
                           out + h * head_sz, L ? L + h * S : NULL,
                           row_max, row_sum, S, dk, scale, causal, pad_mask);
        }
    done:;
    }
}

/* ================================================================
   backward pass

   per-head backward does 5 GEMMs per (qi, kj) tile:
     1. recompute score = Q @ Kt (scale baked into Kt)
     2. softmax: P = exp(score - L) — baked scale means no extra multiply
     3. dV += P^T @ dO
     4. dP = dO @ V^T
     5. dS = P * (dP - Di[:]) * scale_raw  (scale_raw = original 1/√dk,
        needed because chain rule through score scaling)
     6. dQ += dS @ K
     7. dK += dS^T @ Q
   ================================================================ */

static inline void attn_bwd_softmax_row(const float *sr_pre_mask,
                                         float Li, float *sr_out,
                                         int64_t Bk)
{
    /* P = exp(score - L). score already has scale baked in from Kt.
       exp(x - L) where L = row_max + log(row_sum) gives correctly-
       normalized softmax without dividing. */
    int64_t j = 0;
#if defined(AX_HAS_SIMD)
    ax_vf32 vLi = ax_vf32_set1(Li);
    int64_t bve = Bk - (Bk % AX_VF32_WIDTH);
    for (; j < bve; j += AX_VF32_WIDTH)
        ax_vf32_storeu(sr_out + j,
                       ax_vf32_exp(ax_vf32_sub(ax_vf32_loadu(sr_pre_mask + j), vLi)));
#endif
    for (; j < Bk; j++) sr_out[j] = expf(sr_pre_mask[j] - Li);
}

static void attn_bwd_head(const float *Q, const float *K, const float *V,
                           const float *dO, const float *L, const float *Di,
                           float *dQ, float *dK, float *dV,
                           int64_t S, int64_t dk, float scale, bool causal)
{
    int64_t dk_np = ((dk + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
    int64_t Bk_np_max = ((ATTN_BK + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
    int64_t Bq_p_max = ((ATTN_BQ + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;
    int64_t Bk_p_max = ((ATTN_BK + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;

    /* packed panels + per-tile scratch — all TLS so we don't malloc 8
       times per head invocation (BH=64 → 128 mallocs per sdpa_bwd call
       without this). each grow-on-demand based on this call's shape. */
    int64_t kt_want  = (int64_t)(dk * Bk_np_max) * (int64_t)sizeof(float);
    int64_t vt_want  = (int64_t)(dk * Bk_np_max) * (int64_t)sizeof(float);
    int64_t k_want   = (int64_t)(ATTN_BK * dk_np) * (int64_t)sizeof(float);
    int64_t tile_scores_sz = ATTN_BQ * ATTN_BK;
    int64_t p_want   = (int64_t)tile_scores_sz * (int64_t)sizeof(float);
    int64_t pa_sz = Bq_p_max * ATTN_MAX_DK;
    if (Bk_p_max * ATTN_BQ > pa_sz) pa_sz = Bk_p_max * ATTN_BQ;
    if (Bq_p_max * ATTN_BK > pa_sz) pa_sz = Bq_p_max * ATTN_BK;
    int64_t pa_want  = pa_sz * (int64_t)sizeof(float);
    int64_t pb_want  = (int64_t)(ATTN_BQ * dk_np) * (int64_t)sizeof(float);

    float *Kt_packed = ax_tls_grow(&tl_bwd_kt_packed, &tl_bwd_kt_bytes, kt_want);
    float *Vt_packed = ax_tls_grow(&tl_bwd_vt_packed, &tl_bwd_vt_bytes, vt_want);
    float *K_packed  = ax_tls_grow(&tl_bwd_k_packed,  &tl_bwd_k_bytes,  k_want);
    float *P_tile    = ax_tls_grow(&tl_bwd_p_tile,    &tl_bwd_p_bytes,  p_want);
    float *dP_tile   = ax_tls_grow(&tl_bwd_dp_tile,   &tl_bwd_dp_bytes, p_want);
    float *dS_tile   = ax_tls_grow(&tl_bwd_ds_tile,   &tl_bwd_ds_bytes, p_want);
    float *pa        = ax_tls_grow(&tl_bwd_pa,        &tl_bwd_pa_bytes, pa_want);
    float *pb        = ax_tls_grow(&tl_bwd_pb,        &tl_bwd_pb_bytes, pb_want);

    /* MR-strip score scratch for the forward-recompute pass */
    float score_strip[GEMM_MR * ATTN_BK_MAX] __attribute__((aligned(64)));
    float a_q[GEMM_MR * ATTN_MAX_DK] __attribute__((aligned(64)));

    if (!Kt_packed || !Vt_packed || !K_packed || !P_tile || !dP_tile ||
        !dS_tile || !pa || !pb) { return; }

    for (int64_t kj = 0; kj < S; kj += ATTN_BK) {
        int64_t Bk = (kj + ATTN_BK <= S) ? ATTN_BK : (S - kj);
        int64_t Bk_np = ((Bk + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
        int64_t Bk_p = ((Bk + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;

        pack_b_t(K + kj * dk, dk, dk, Bk_np, Bk, Kt_packed);
        attn_scale_packed(Kt_packed, dk * Bk_np, scale);
        pack_b_t(V + kj * dk, dk, dk, Bk_np, Bk, Vt_packed);
        pack_b(K + kj * dk, dk, Bk, dk_np, dk, K_packed);

        for (int64_t qi = 0; qi < S; qi += ATTN_BQ) {
            int64_t Bq = (qi + ATTN_BQ <= S) ? ATTN_BQ : (S - qi);
            int64_t Bq_p = ((Bq + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;

            /* causal: skip this (qi, kj) if entire block is in future */
            if (causal && qi + Bq - 1 < kj) continue;

            /* recompute P[Bq, Bk] via MR-strip fused score+softmax.
               the score uses Kt_packed (scaled), so P = exp(score - L). */
            for (int64_t ir = 0; ir < Bq; ir += GEMM_MR) {
                int64_t mr = (ir + GEMM_MR <= Bq) ? GEMM_MR : (Bq - ir);
                int64_t mr_p = ((mr + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;

                if (causal && qi + ir + mr - 1 < kj) {
                    /* entire strip in future — write zeros to P */
                    for (int64_t r = 0; r < mr; r++)
                        memset(P_tile + (ir + r) * Bk, 0, (size_t)Bk * sizeof(float));
                    continue;
                }
                /* need masking iff ANY row in the strip has a future key —
                   gated by the EARLIEST row's absolute position (qi+ir),
                   not the latest. see fwd path for the same invariant. */
                bool need_mask = causal && (qi + ir < kj + Bk - 1);

                pack_a(Q + (qi + ir) * dk, dk, mr_p, dk, mr, a_q);
                memset(score_strip, 0, (size_t)(mr * Bk) * sizeof(float));
                for (int64_t jr = 0; jr < Bk_np; jr += GEMM_NR) {
                    int64_t nr = (jr + GEMM_NR <= Bk) ? GEMM_NR : (Bk > jr ? Bk - jr : 0);
                    if (nr <= 0) break;
                    micro_kernel(dk, a_q, Kt_packed + jr * dk,
                                 score_strip + jr, Bk, mr, nr);
                }
                if (need_mask) {
                    for (int64_t r = 0; r < mr; r++)
                        attn_apply_causal_mask(score_strip + r * Bk,
                                                qi + ir + r, kj, Bk);
                }
                /* P = exp(score - L) into P_tile */
                for (int64_t r = 0; r < mr; r++) {
                    attn_bwd_softmax_row(score_strip + r * Bk, L[qi + ir + r],
                                          P_tile + (ir + r) * Bk, Bk);
                }
            }

            /* dV += P^T @ dO (full tile GEMM) */
            pack_a_t(P_tile, Bk, Bk_p, Bq, Bk, pa);
            pack_b(dO + qi * dk, dk, Bq, dk_np, dk, pb);
            for (int64_t ir = 0; ir < Bk_p; ir += GEMM_MR) {
                int64_t mr = (ir + GEMM_MR <= Bk) ? GEMM_MR : (Bk > ir ? Bk - ir : 0);
                if (mr <= 0) break;
                for (int64_t jr = 0; jr < dk_np; jr += GEMM_NR) {
                    int64_t nr = (jr + GEMM_NR <= dk) ? GEMM_NR : (dk > jr ? dk - jr : 0);
                    if (nr <= 0) break;
                    micro_kernel(Bq, pa + ir * Bq, pb + jr * Bq,
                                 dV + (kj + ir) * dk + jr, dk, mr, nr);
                }
            }

            /* dP = dO @ V^T (full tile GEMM) */
            pack_a(dO + qi * dk, dk, Bq_p, dk, Bq, pa);
            memset(dP_tile, 0, (size_t)(Bq * Bk) * sizeof(float));
            for (int64_t ir = 0; ir < Bq_p; ir += GEMM_MR) {
                int64_t mr = (ir + GEMM_MR <= Bq) ? GEMM_MR : (Bq > ir ? Bq - ir : 0);
                if (mr <= 0) break;
                for (int64_t jr = 0; jr < Bk_np; jr += GEMM_NR) {
                    int64_t nr = (jr + GEMM_NR <= Bk) ? GEMM_NR : (Bk > jr ? Bk - jr : 0);
                    if (nr <= 0) break;
                    micro_kernel(dk, pa + ir * dk, Vt_packed + jr * dk,
                                 dP_tile + ir * Bk + jr, Bk, mr, nr);
                }
            }

            /* dS = P * (dP - Di) * scale (element-wise) */
            for (int64_t r = 0; r < Bq; r++) {
                float di = Di[qi + r];
                float *pr = P_tile + r * Bk;
                float *dpr = dP_tile + r * Bk;
                float *dsr = dS_tile + r * Bk;
                int64_t j = 0;
#if defined(AX_HAS_SIMD)
                ax_vf32 vdi = ax_vf32_set1(di), vsc = ax_vf32_set1(scale);
                int64_t bve = Bk - (Bk % AX_VF32_WIDTH);
                for (; j < bve; j += AX_VF32_WIDTH) {
                    ax_vf32 p = ax_vf32_loadu(pr + j), dp = ax_vf32_loadu(dpr + j);
                    ax_vf32_storeu(dsr + j, ax_vf32_mul(ax_vf32_mul(p, ax_vf32_sub(dp, vdi)), vsc));
                }
#endif
                for (; j < Bk; j++) dsr[j] = pr[j] * (dpr[j] - di) * scale;
            }

            /* dQ += dS @ K */
            pack_a(dS_tile, Bk, Bq_p, Bk, Bq, pa);
            for (int64_t ir = 0; ir < Bq_p; ir += GEMM_MR) {
                int64_t mr = (ir + GEMM_MR <= Bq) ? GEMM_MR : (Bq > ir ? Bq - ir : 0);
                if (mr <= 0) break;
                for (int64_t jr = 0; jr < dk_np; jr += GEMM_NR) {
                    int64_t nr = (jr + GEMM_NR <= dk) ? GEMM_NR : (dk > jr ? dk - jr : 0);
                    if (nr <= 0) break;
                    micro_kernel(Bk, pa + ir * Bk, K_packed + jr * Bk,
                                 dQ + (qi + ir) * dk + jr, dk, mr, nr);
                }
            }

            /* dK += dS^T @ Q */
            pack_a_t(dS_tile, Bk, Bk_p, Bq, Bk, pa);
            pack_b(Q + qi * dk, dk, Bq, dk_np, dk, pb);
            for (int64_t ir = 0; ir < Bk_p; ir += GEMM_MR) {
                int64_t mr = (ir + GEMM_MR <= Bk) ? GEMM_MR : (Bk > ir ? Bk - ir : 0);
                if (mr <= 0) break;
                for (int64_t jr = 0; jr < dk_np; jr += GEMM_NR) {
                    int64_t nr = (jr + GEMM_NR <= dk) ? GEMM_NR : (dk > jr ? dk - jr : 0);
                    if (nr <= 0) break;
                    micro_kernel(Bq, pa + ir * Bq, pb + jr * Bq,
                                 dK + (kj + ir) * dk + jr, dk, mr, nr);
                }
            }
        }
    }

    /* all packs are TLS; nothing to free */
}

void AX_SYM(ax_cpu_sdpa_bwd)(const float *Q, const float *K, const float *V,
                              const float *O, const float *dO, const float *L,
                              float *dQ, float *dK, float *dV,
                              int64_t BH, int64_t S, int64_t dk, float scale,
                              bool causal)
{
    int64_t head_sz = S * dk;

    /* zero output grads (caller may pass accumulating buffers;
       in the MHA layer path we zero for a fresh backward) */
    memset(dQ, 0, (size_t)(BH * head_sz) * sizeof(float));
    memset(dK, 0, (size_t)(BH * head_sz) * sizeof(float));
    memset(dV, 0, (size_t)(BH * head_sz) * sizeof(float));

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        /* Di is TLS + grow-on-demand; lives for thread lifetime. */
        float *Di = ax_tls_grow(&tl_bwd_di, &tl_bwd_di_bytes,
                                (int64_t)S * (int64_t)sizeof(float));
        if (!Di) goto done;

#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int64_t h = 0; h < BH; h++) {
            const float *dOh = dO + h * head_sz;
            const float *Oh = O + h * head_sz;
            /* Di[i] = dot(dO[i], O[i]) */
            for (int64_t i = 0; i < S; i++) {
                const float *do_r = dOh + i * dk;
                const float *o_r  = Oh  + i * dk;
                float dot = 0;
                int64_t d = 0;
#if defined(AX_HAS_SIMD)
                ax_vf32 vd = ax_vf32_zero();
                int64_t ve = dk - (dk % AX_VF32_WIDTH);
                for (; d < ve; d += AX_VF32_WIDTH)
                    vd = ax_vf32_fmadd(ax_vf32_loadu(do_r + d),
                                       ax_vf32_loadu(o_r + d), vd);
                dot = ax_vf32_hsum(vd);
#endif
                for (; d < dk; d++) dot += do_r[d] * o_r[d];
                Di[i] = dot;
            }
            attn_bwd_head(Q + h * head_sz, K + h * head_sz, V + h * head_sz,
                           dOh, L + h * S, Di,
                           dQ + h * head_sz, dK + h * head_sz, dV + h * head_sz,
                           S, dk, scale, causal);
        }
    done:;
    }
}

/* ================================================================
   Rotary Position Embeddings (RoPE)

   applies rotation by angle θ_{i,k} to dimension pair (2k, 2k+1) at
   token position i, where θ_{i,k} = i / theta_base^(2k/dk).
   θ_{i,k} is the inverse frequency — long-wavelength for early dim
   pairs, short-wavelength for later ones.

   for each token position i and each dim pair (2k, 2k+1):
     x'[2k]   = x[2k] * cos(θ) - x[2k+1] * sin(θ)
     x'[2k+1] = x[2k] * sin(θ) + x[2k+1] * cos(θ)

   applied in-place to Q and K (but NOT V — V is position-independent). */
void AX_SYM(ax_cpu_rope_apply)(float *Q, float *K, const int64_t *positions,
                                int64_t BH, int64_t S, int64_t dk, float theta_base)
{
    if (dk % 2 != 0) return; /* RoPE requires even dk */
    int64_t half = dk / 2;

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int64_t h = 0; h < BH; h++) {
        for (int64_t i = 0; i < S; i++) {
            int64_t pos = positions ? positions[i] : i;
            float *q_row = Q + h * S * dk + i * dk;
            float *k_row = K + h * S * dk + i * dk;
            for (int64_t d = 0; d < half; d++) {
                float inv_freq = 1.0f / powf(theta_base, (float)(2 * d) / (float)dk);
                float theta = (float)pos * inv_freq;
                float c = cosf(theta), s = sinf(theta);
                float q0 = q_row[d],       q1 = q_row[d + half];
                float k0 = k_row[d],       k1 = k_row[d + half];
                q_row[d]        = q0 * c - q1 * s;
                q_row[d + half] = q0 * s + q1 * c;
                k_row[d]        = k0 * c - k1 * s;
                k_row[d + half] = k0 * s + k1 * c;
            }
        }
    }
}

/* ================================================================
   KV cache attention: rank-1 attention for single-token inference.
   Q_new is [BH, dk], attending against cached K/V [BH, cur_len, dk].
   out is [BH, dk]. this is a GEMV, not a GEMM — much faster than
   going through the full SDPA path for a single new query.
   ================================================================ */
void AX_SYM(ax_cpu_kv_cache_attend)(const float *K_cache, const float *V_cache,
                                      const float *Q_new, float *out,
                                      int64_t BH, int64_t cur_len, int64_t max_seq,
                                      int64_t dk, float scale)
{
    (void)max_seq;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t h = 0; h < BH; h++) {
        const float *K_h = K_cache + h * max_seq * dk;
        const float *V_h = V_cache + h * max_seq * dk;
        const float *q = Q_new + h * dk;
        float *o = out + h * dk;

        /* allocate scores on stack (bounded by cur_len which is reasonable) */
        float scores_buf[8192];
        float *scores = scores_buf;
        float *heap_scores = NULL;
        if (cur_len > 8192) {
            heap_scores = (float *)malloc((size_t)cur_len * sizeof(float));
            if (!heap_scores) continue;
            scores = heap_scores;
        }

        /* 1. compute scores[j] = dot(Q, K[j]) * scale  — this is a [cur_len, dk] @ [dk] GEMV */
        float mx = -FLT_MAX;
        for (int64_t j = 0; j < cur_len; j++) {
            const float *kj = K_h + j * dk;
            float dot = 0;
            int64_t d = 0;
#if defined(AX_HAS_SIMD)
            ax_vf32 vd = ax_vf32_zero();
            int64_t ve = dk - (dk % AX_VF32_WIDTH);
            for (; d < ve; d += AX_VF32_WIDTH)
                vd = ax_vf32_fmadd(ax_vf32_loadu(q + d), ax_vf32_loadu(kj + d), vd);
            dot = ax_vf32_hsum(vd);
#endif
            for (; d < dk; d++) dot += q[d] * kj[d];
            scores[j] = dot * scale;
            if (scores[j] > mx) mx = scores[j];
        }

        /* 2. softmax */
        float sum = 0;
        for (int64_t j = 0; j < cur_len; j++) {
            scores[j] = expf(scores[j] - mx);
            sum += scores[j];
        }
        float inv = 1.0f / sum;

        /* 3. out = sum_j scores[j] * V[j]  — GEMV */
        memset(o, 0, (size_t)dk * sizeof(float));
        for (int64_t j = 0; j < cur_len; j++) {
            float w = scores[j] * inv;
            const float *vj = V_h + j * dk;
            int64_t d = 0;
#if defined(AX_HAS_SIMD)
            ax_vf32 vw = ax_vf32_set1(w);
            int64_t ve = dk - (dk % AX_VF32_WIDTH);
            for (; d < ve; d += AX_VF32_WIDTH)
                ax_vf32_storeu(o + d, ax_vf32_fmadd(vw, ax_vf32_loadu(vj + d), ax_vf32_loadu(o + d)));
#endif
            for (; d < dk; d++) o[d] += w * vj[d];
        }

        free(heap_scores);
    }
}
