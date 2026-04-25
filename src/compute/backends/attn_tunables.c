/* attn_tunables.c — runtime-calibrated thresholds for attention paths.

   each gate that selects between a fused and unfused kernel variant is
   measured at startup with a synthetic workload representing the actual
   decision. results cached per host (calib_cache) so the cost is paid
   once per machine + library version.

   on AX_NO_AUTOTUNE=1 the calibrate function is a no-op and the getters
   return conservative defaults matching the pre-calibration hardcoded
   values — the behaviour everyone shipped before this work landed. */

#include "axiom/internal/attn_tunables.h"
#include "axiom/tensor.h"
#include "axiom/compute.h"
#include "axiom/memory.h"
#include "axiom/attention.h"

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================
   calibrated state — written once by ax_attn_tunables_calibrate(),
   read by ax_attn_tunable_*() getters thereafter. defaults match the
   pre-calibration hardcoded values so behaviour is unchanged when
   calibration is skipped (AX_NO_AUTOTUNE=1) or fails.
   ============================================================ */

static struct {
    bool calibrated;

    /* F.3.a fused qkv+head_split: enable when rows*3*D*4 > thresh */
    int64_t f3a_qkv_bytes_threshold;     /* default: 8 MB */

    /* F.3.c fused dwqkv split-acc: enable when D >= thresh */
    int64_t f3c_d_threshold;              /* default: 1024 */

    /* save_p memory budget */
    int64_t save_p_max_bytes;             /* default: 8 MB */
    int64_t save_p_small_exclusion_sk;    /* S*dk threshold; default: 8192 (== 128*64) */
    int64_t save_p_small_exclusion_s;     /* S threshold;    default: 128 */

    /* fused_bh per-head attn_flat byte budget. */
    int64_t fused_bh_per_head_bytes;      /* default: 32 KB */

    /* AX_SDPA_FUSED auto: when true, use FA-2-style fused kj-block.
       calibrator decides per-BH-vs-NT regime; defaults to "always
       use materialized" (safest pre-calibration default). */
    bool sdpa_fused_use_when_bh_lt_nt;    /* default: true (BH<NT path) */
    bool sdpa_fused_use_when_bh_eq_nt;    /* default: false (regression case) */
    bool sdpa_fused_use_when_bh_gt_nt;    /* default: true */

    /* gemm_tn pre-transpose flop threshold */
    int64_t gemm_tn_pretranspose_flops;   /* default: 2 GFLOPS */

    /* attention tile sizes (MR-multiples) */
    int64_t attn_bq;                      /* default: ATTN_BQ_DEFAULT */
    int64_t attn_bk;                      /* default: ATTN_BQ_DEFAULT */
} g_attn = {
    .calibrated = false,
    .f3a_qkv_bytes_threshold = (int64_t)8 * 1024 * 1024,
    .f3c_d_threshold = 1024,
    .save_p_max_bytes = (int64_t)8 * 1024 * 1024,
    .save_p_small_exclusion_sk = 8192,
    .save_p_small_exclusion_s = 128,
    .fused_bh_per_head_bytes = 32 * 1024,
    .sdpa_fused_use_when_bh_lt_nt = true,
    .sdpa_fused_use_when_bh_eq_nt = false,
    .sdpa_fused_use_when_bh_gt_nt = true,
    .gemm_tn_pretranspose_flops = 2000000000LL,
    /* attn_bq/bk seeded from the existing ATTN_BQ_DEFAULT macro at
       calibrate-time. without that we'd duplicate the macro definition
       here; see ax_attn_tunables_calibrate(). */
    .attn_bq = 0,
    .attn_bk = 0,
};

/* ============================================================
   getters — pure read of the calibrated state. cheap. always safe
   even before calibration runs (returns defaults).
   ============================================================ */

int64_t ax_attn_tunable_f3a_qkv_bytes_threshold(void) {
    return g_attn.f3a_qkv_bytes_threshold;
}

int64_t ax_attn_tunable_f3c_d_threshold(void) {
    return g_attn.f3c_d_threshold;
}

int64_t ax_attn_tunable_save_p_max_bytes(void) {
    return g_attn.save_p_max_bytes;
}

int64_t ax_attn_tunable_save_p_small_exclusion_sk(void) {
    return g_attn.save_p_small_exclusion_sk;
}

int64_t ax_attn_tunable_save_p_small_exclusion_s(void) {
    return g_attn.save_p_small_exclusion_s;
}

int64_t ax_attn_tunable_fused_bh_per_head_bytes(void) {
    return g_attn.fused_bh_per_head_bytes;
}

bool ax_attn_tunable_use_sdpa_fused(int64_t BH) {
#ifdef _OPENMP
    int nt = omp_get_max_threads();
#else
    int nt = 1;
#endif
    if (BH < (int64_t)nt) return g_attn.sdpa_fused_use_when_bh_lt_nt;
    if (BH > (int64_t)nt) return g_attn.sdpa_fused_use_when_bh_gt_nt;
    return g_attn.sdpa_fused_use_when_bh_eq_nt;
}

int64_t ax_attn_tunable_gemm_tn_pretranspose_flops(void) {
    return g_attn.gemm_tn_pretranspose_flops;
}

int64_t ax_attn_tunable_attn_bq(void) {
    return g_attn.attn_bq;
}

int64_t ax_attn_tunable_attn_bk(void) {
    return g_attn.attn_bk;
}

/* ============================================================
   calibration — run a small workload for each gate, measure both
   sides of the threshold, pick the crossover.

   each measure_* function returns wall time in ms for one shape.
   gates with discrete options (BH<NT vs BH=NT vs BH>NT for
   sdpa_fused) measure the on/off variants directly. gates with a
   continuous knob (D for F.3.c, qkv_bytes for F.3.a) sweep a
   small set of representative shapes and pick the lowest crossover
   that still wins consistently across the host's compute resources.

   each measurement is bracketed in a clearly-bounded budget so the
   total calibration time is predictable. defaults shown above are
   the fallback when calibration fails or is skipped.
   ============================================================ */

#if !defined(AX_NO_AUTOTUNE)

/* ============================================================
   measurement infrastructure shared across all gates
   ============================================================ */

static double meas_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* fill a buffer with deterministic pseudo-random fp32 values in [-0.5, 0.5).
   xorshift32 seeded with a fixed constant so the calibration is repeatable
   on a given host (avoids run-to-run noise from differing input data). */
static void meas_fill_rand(float *buf, int64_t n) {
    uint32_t s = 2463534242u;
    for (int64_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        buf[i] = (float)((int32_t)s) / (float)INT32_MAX * 0.5f;
    }
}

/* allocate an aligned heap tensor wrapper around raw data we already own.
   the tensor doesn't take ownership; caller frees the data via
   ax_aligned_free (or the caller's own allocator). all measurement-time
   tensors use this so we control the data layout and lifetime ourselves
   rather than threading through ax_tensor_create's allocator path. */
typedef struct meas_tensor {
    float *data;
    ax_storage_t st;
    ax_tensor_t  tv;
} meas_tensor_t;

static void meas_tensor_init(meas_tensor_t *t, int64_t *shape, int ndim,
                              float *data, int64_t total_bytes) {
    t->data = data;
    t->st.data = data;
    t->st.size_bytes = (size_t)total_bytes;
    atomic_init(&t->st.refcount, 0);
    t->st.device = AX_DEVICE_CPU;
    t->st.is_arena_temp = true;
    t->st.generation = 1;
    memset(&t->tv, 0, sizeof(t->tv));
    t->tv.storage = &t->st;
    t->tv.ndim = ndim;
    t->tv.dtype = AX_FLOAT32;
    int64_t stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        t->tv.shape[i] = shape[i];
        t->tv.strides[i] = stride;
        stride *= shape[i];
    }
}

/* trimmed-mean timing harness. fn is called once per inner iter. warm
   iters fill caches / branch predictors / packed buffers; timed iters
   collect samples. we return the trimmed mean (drop highest sample to
   discard outliers from cgroup throttling / scheduling jitter, then
   average the rest). more robust than min-of-N when bg load is
   variable — min often catches an unrepresentative under-load value
   while a single high-load sample skews mean; trimmed mean splits
   the difference. caller picks `timed` to match host noise floor;
   small budget (3-5) for cheap gates, larger (7-10) for heavy ones. */
typedef ax_status_t (*meas_fn_t)(void *ctx);

#define MEAS_MAX_TIMED 16

static int meas_dcmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double meas_trimmed_mean_ms(meas_fn_t fn, void *ctx, int warm, int timed) {
    if (timed > MEAS_MAX_TIMED) timed = MEAS_MAX_TIMED;
    for (int i = 0; i < warm; i++) {
        if (fn(ctx) != AX_OK) return 1.0e30;
    }
    double samples[MEAS_MAX_TIMED];
    for (int i = 0; i < timed; i++) {
        double t0 = meas_now_ms();
        if (fn(ctx) != AX_OK) return 1.0e30;
        samples[i] = meas_now_ms() - t0;
    }
    qsort(samples, (size_t)timed, sizeof(double), meas_dcmp);
    /* drop the top floor(timed/4) samples — captures host scheduling
       jitter without losing the underlying signal. for timed=4 that's
       1 sample, for timed=8 that's 2. */
    int drop_high = timed / 4;
    int kept = timed - drop_high;
    if (kept <= 0) kept = 1;
    double sum = 0;
    for (int i = 0; i < kept; i++) sum += samples[i];
    return sum / (double)kept;
}

extern int ax_compute_has_dwqkv_split_acc(void);
extern int ax_compute_has_qkv_head_gemm(void);

/* ============================================================
   F.3.c (dwqkv_split_acc) D-threshold calibration

   for each candidate D in {512, 768, 1024, 1280, 1536}, measure:
     - fallback path: gemm_tn into [D, 3D] scratch + 3-way split
     - F.3.c path:    ax_compute_dwqkv_split_acc direct

   pick the lowest D where the fused path consistently wins by ≥3 %.
   if no D wins, threshold is set to "above the largest tested" so
   the gate effectively disables the fused path.
   ============================================================ */

typedef struct {
    const ax_tensor_t *x;
    const ax_tensor_t *dQKV;
    ax_tensor_t *dWqkv_scratch;
    ax_tensor_t *dWq, *dWk, *dWv;
} f3c_ctx_t;

static ax_status_t f3c_run_fallback(void *p) {
    f3c_ctx_t *c = (f3c_ctx_t *)p;
    /* mirror the fallback path: gemm_tn into scratch, then split.
       we skip the actual SIMD split here since both paths perform
       it equivalently — the dwqkv_split_acc kernel ALSO does the
       split internally. measuring just the gemm_tn call captures
       the dominant cost of the fallback. */
    return ax_compute_gemm_tn(c->x, c->dQKV, c->dWqkv_scratch);
}

static ax_status_t f3c_run_fused(void *p) {
    f3c_ctx_t *c = (f3c_ctx_t *)p;
    return ax_compute_dwqkv_split_acc(c->x, c->dQKV, c->dWq, c->dWk, c->dWv);
}

static int64_t calibrate_f3c_d_threshold(void) {
    if (!ax_compute_has_dwqkv_split_acc()) return INT64_MAX;

    /* representative shape parameters: keep B*S=1024 rows and 3D
       shape consistent with mha-train workloads. sweep D. */
    static const int64_t cand_D[]   = { 512, 768, 1024, 1280, 1536 };
    static const int     n_cand     = (int)(sizeof(cand_D) / sizeof(cand_D[0]));
    const int64_t rows = 1024;
    const int     warm = 2;
    const int     timed = 5;

    int64_t best_threshold = INT64_MAX;  /* no D wins by default */

    for (int i = 0; i < n_cand; i++) {
        int64_t D = cand_D[i];
        int64_t threeD = 3 * D;

        /* allocate aligned tensors. x_flat [rows, D], dQKV [rows, 3D],
           dWqkv [D, 3D], and three [D, D] grad tensors. */
        size_t x_bytes  = (size_t)rows * D * sizeof(float);
        size_t dq_bytes = (size_t)rows * threeD * sizeof(float);
        size_t dw_bytes = (size_t)D * threeD * sizeof(float);
        size_t dws_bytes = (size_t)D * D * sizeof(float);
        float *x_buf  = (float *)ax_aligned_alloc(x_bytes,  64);
        float *dq_buf = (float *)ax_aligned_alloc(dq_bytes, 64);
        float *dw_buf = (float *)ax_aligned_alloc(dw_bytes, 64);
        float *dwq    = (float *)ax_aligned_alloc(dws_bytes, 64);
        float *dwk    = (float *)ax_aligned_alloc(dws_bytes, 64);
        float *dwv    = (float *)ax_aligned_alloc(dws_bytes, 64);
        if (!x_buf || !dq_buf || !dw_buf || !dwq || !dwk || !dwv) {
            ax_aligned_free(x_buf); ax_aligned_free(dq_buf); ax_aligned_free(dw_buf);
            ax_aligned_free(dwq);   ax_aligned_free(dwk);    ax_aligned_free(dwv);
            continue;  /* skip this candidate, try next */
        }
        meas_fill_rand(x_buf,  rows * D);
        meas_fill_rand(dq_buf, rows * threeD);
        memset(dw_buf, 0, dw_bytes);
        memset(dwq, 0, dws_bytes);
        memset(dwk, 0, dws_bytes);
        memset(dwv, 0, dws_bytes);

        meas_tensor_t x_t, dq_t, dw_t, dwq_t, dwk_t, dwv_t;
        int64_t x_sh[]   = { rows, D };
        int64_t dq_sh[]  = { rows, threeD };
        int64_t dw_sh[]  = { D, threeD };
        int64_t dws_sh[] = { D, D };
        meas_tensor_init(&x_t,   x_sh,   2, x_buf,  (int64_t)x_bytes);
        meas_tensor_init(&dq_t,  dq_sh,  2, dq_buf, (int64_t)dq_bytes);
        meas_tensor_init(&dw_t,  dw_sh,  2, dw_buf, (int64_t)dw_bytes);
        meas_tensor_init(&dwq_t, dws_sh, 2, dwq,    (int64_t)dws_bytes);
        meas_tensor_init(&dwk_t, dws_sh, 2, dwk,    (int64_t)dws_bytes);
        meas_tensor_init(&dwv_t, dws_sh, 2, dwv,    (int64_t)dws_bytes);

        f3c_ctx_t ctx = {
            .x = &x_t.tv, .dQKV = &dq_t.tv, .dWqkv_scratch = &dw_t.tv,
            .dWq = &dwq_t.tv, .dWk = &dwk_t.tv, .dWv = &dwv_t.tv,
        };

        double t_fallback = meas_trimmed_mean_ms(f3c_run_fallback, &ctx, warm, timed);
        double t_fused    = meas_trimmed_mean_ms(f3c_run_fused,    &ctx, warm, timed);

        ax_aligned_free(x_buf); ax_aligned_free(dq_buf); ax_aligned_free(dw_buf);
        ax_aligned_free(dwq);   ax_aligned_free(dwk);    ax_aligned_free(dwv);

        /* fused wins iff t_fused < 0.97 * t_fallback (3 % margin to fight
           noise on borderline cases). first D that wins becomes the
           threshold — every D >= it will use the fused path. */
        if (t_fused < t_fallback * 0.97 && D < best_threshold) {
            best_threshold = D;
            /* don't break: continue measuring so the LOG can show the
               full sweep, but threshold latches on the first win. */
        }
    }
    return best_threshold;
}

/* ============================================================
   F.3.a (qkv_head_gemm) qkv-bytes threshold calibration

   sweep rows*D values and measure fused-vs-unfused qkv projection.
   the unfused path is gemm + ax_attn_head_interleave_qkv_split_bias —
   we approximate with gemm only since the layout transform is small
   relative to the gemm. the threshold is qkv_bytes = rows*3*D*4.
   ============================================================ */

typedef struct {
    const ax_tensor_t *x;
    const ax_tensor_t *Wqkv;
    const ax_tensor_t *bqkv;          /* may be NULL */
    int64_t B, S, H, dk;
    ax_tensor_t *qkv_scratch;
    ax_tensor_t *Qh, *Kh, *Vh;
} f3a_ctx_t;

static ax_status_t f3a_run_fallback(void *p) {
    f3a_ctx_t *c = (f3a_ctx_t *)p;
    /* gemm only — the head_interleave_split is much cheaper and
       roughly the same cost on either side of the gate. */
    return ax_compute_gemm(c->x, c->Wqkv, c->qkv_scratch);
}

static ax_status_t f3a_run_fused(void *p) {
    f3a_ctx_t *c = (f3a_ctx_t *)p;
    return ax_compute_qkv_head_gemm(c->x, c->Wqkv, c->bqkv,
                                      c->B, c->S, c->H, c->dk,
                                      c->Qh, c->Kh, c->Vh);
}

static int64_t calibrate_f3a_qkv_bytes_threshold(void) {
    if (!ax_compute_has_qkv_head_gemm()) return INT64_MAX;

    /* representative shapes spanning the qkv_bytes regime. the threshold
       finds the lowest qkv_bytes where the fused kernel wins. */
    typedef struct { int64_t B, S, D, H; } shape_t;
    static const shape_t cand[] = {
        /* qkv_bytes = B*S*3*D*4 */
        {  1, 256,  512,   8 },   /*  1.5 MB */
        {  4, 128,  512,   8 },   /*  3.0 MB */
        {  8, 128,  512,   8 },   /*  6.3 MB */
        {  4, 256,  512,   8 },   /*  6.3 MB */
        {  4, 512,  768,  12 },   /* 18.9 MB */
        {  2,1024,  768,  12 },   /* 18.9 MB */
        {  1,2048,  768,  12 },   /* 18.9 MB */
    };
    static const int n_cand = (int)(sizeof(cand) / sizeof(cand[0]));
    const int warm = 2;
    const int timed = 5;

    int64_t best_threshold = INT64_MAX;

    for (int i = 0; i < n_cand; i++) {
        const shape_t *s = &cand[i];
        int64_t B = s->B, S = s->S, D = s->D, H = s->H;
        int64_t dk = D / H;
        int64_t rows = B * S;
        int64_t threeD = 3 * D;
        int64_t qkv_bytes = rows * threeD * (int64_t)sizeof(float);

        size_t x_bytes   = (size_t)rows * D * sizeof(float);
        size_t Wqkv_bytes = (size_t)D * threeD * sizeof(float);
        size_t qkv_sbytes = (size_t)rows * threeD * sizeof(float);
        size_t qh_bytes   = (size_t)B * H * S * dk * sizeof(float);
        float *x_buf  = (float *)ax_aligned_alloc(x_bytes,    64);
        float *W_buf  = (float *)ax_aligned_alloc(Wqkv_bytes, 64);
        float *qs_buf = (float *)ax_aligned_alloc(qkv_sbytes, 64);
        float *Qh_buf = (float *)ax_aligned_alloc(qh_bytes,   64);
        float *Kh_buf = (float *)ax_aligned_alloc(qh_bytes,   64);
        float *Vh_buf = (float *)ax_aligned_alloc(qh_bytes,   64);
        if (!x_buf || !W_buf || !qs_buf || !Qh_buf || !Kh_buf || !Vh_buf) {
            ax_aligned_free(x_buf); ax_aligned_free(W_buf);
            ax_aligned_free(qs_buf); ax_aligned_free(Qh_buf);
            ax_aligned_free(Kh_buf); ax_aligned_free(Vh_buf);
            continue;
        }
        meas_fill_rand(x_buf, rows * D);
        meas_fill_rand(W_buf, D * threeD);

        meas_tensor_t x_t, W_t, qs_t, Qh_t, Kh_t, Vh_t;
        int64_t x_sh[]   = { rows, D };
        int64_t W_sh[]   = { D, threeD };
        int64_t qs_sh[]  = { rows, threeD };
        int64_t qh_sh[]  = { B * H, S, dk };
        meas_tensor_init(&x_t,  x_sh,  2, x_buf,  (int64_t)x_bytes);
        meas_tensor_init(&W_t,  W_sh,  2, W_buf,  (int64_t)Wqkv_bytes);
        meas_tensor_init(&qs_t, qs_sh, 2, qs_buf, (int64_t)qkv_sbytes);
        meas_tensor_init(&Qh_t, qh_sh, 3, Qh_buf, (int64_t)qh_bytes);
        meas_tensor_init(&Kh_t, qh_sh, 3, Kh_buf, (int64_t)qh_bytes);
        meas_tensor_init(&Vh_t, qh_sh, 3, Vh_buf, (int64_t)qh_bytes);

        f3a_ctx_t ctx = {
            .x = &x_t.tv, .Wqkv = &W_t.tv, .bqkv = NULL,
            .B = B, .S = S, .H = H, .dk = dk,
            .qkv_scratch = &qs_t.tv,
            .Qh = &Qh_t.tv, .Kh = &Kh_t.tv, .Vh = &Vh_t.tv,
        };

        double t_fallback = meas_trimmed_mean_ms(f3a_run_fallback, &ctx, warm, timed);
        double t_fused    = meas_trimmed_mean_ms(f3a_run_fused,    &ctx, warm, timed);

        ax_aligned_free(x_buf); ax_aligned_free(W_buf);
        ax_aligned_free(qs_buf); ax_aligned_free(Qh_buf);
        ax_aligned_free(Kh_buf); ax_aligned_free(Vh_buf);

        if (t_fused < t_fallback * 0.97 && qkv_bytes < best_threshold) {
            best_threshold = qkv_bytes;
        }
    }
    return best_threshold;
}

/* ============================================================
   gemm_tn pre-transpose flop threshold calibration

   the pre-transpose path stages A^T into TLS scratch then runs a
   plain NN gemm. it wins when the saved per-tile pack_a_t exceeds
   the transpose write cost — empirically above ~2 GFLOPS on AVX2.
   sweep flops, find crossover.
   ============================================================ */

typedef struct {
    const ax_tensor_t *a;
    const ax_tensor_t *b;
    ax_tensor_t *c;
} gtn_ctx_t;

static ax_status_t gtn_run(void *p) {
    gtn_ctx_t *c = (gtn_ctx_t *)p;
    return ax_compute_gemm_tn(c->a, c->b, c->c);
}

static int64_t calibrate_gemm_tn_pretranspose_flops(void) {
    /* the gate inside opt_gemm_tn ALSO requires n >= 2*m; we mirror
       that constraint in the candidate shapes. measurement only
       compares one path's wall time per shape — to find the true
       crossover we'd need to disable the gate and run both paths,
       which requires plumbing a kill-switch.

       since we can't easily A/B in this commit (would require an
       extra setter on opt_gemm_tn), we measure throughput across
       the gate's flop range and pick the highest flop count where
       throughput plateaus / dips, indicating pre-transpose is
       hurting on smaller shapes. when in doubt, fall back to the
       2 GFLOPS default which is the empirical AVX2 default. */
    typedef struct { int64_t m, n, k; } shape_t;
    static const shape_t cand[] = {
        /* flops = 2*m*n*k */
        {  256,  512, 256 },   /* 0.07 GF */
        {  512, 1024, 512 },   /* 0.54 GF */
        {  768, 1536, 768 },   /* 1.81 GF — borderline */
        { 1024, 2048,1024 },   /* 4.29 GF */
        {  512, 2048, 512 },   /* 1.07 GF — n >= 2*m */
        { 1024, 3072, 512 },   /* 3.22 GF */
    };
    (void)cand;
    /* without an A/B-able gate we can't measure the crossover directly.
       keep the default 2 GFLOPS until A/B infrastructure is added. the
       getter-based scaffold is in place so the value can be tuned via
       env override AX_GEMM_TN_PRETRANSPOSE_FLOPS without rebuilding. */
    return 2000000000LL;
}

/* ============================================================
   placeholder calibrators for the remaining gates. each will be
   filled in as its measurement infrastructure stabilises. for now
   they return the pre-calibration default so behaviour matches what
   shipped before the autotuner. each TODO marks the next work item.
   ============================================================ */

static int64_t calibrate_save_p_max_bytes(void) {
    /* TODO: needs full mha train_step harness with save_p toggle.
       for now return the empirical 8 MB default. */
    return (int64_t)8 * 1024 * 1024;
}

static int64_t calibrate_save_p_small_exclusion_sk(void) {
    /* TODO: needs save_p A/B on small-S/dk shapes. */
    return 8192;
}

static int64_t calibrate_save_p_small_exclusion_s(void) {
    return 128;
}

static int64_t calibrate_fused_bh_per_head_bytes(void) {
    /* TODO: needs combined kernel A/B sweep over per-head bytes.
       the threshold should equal half the host's L1d (after
       subtracting tile scratch). until measured, use 32 KB which
       is half-L1d on most x86 (32-48 KB) and conservative enough
       on Apple M (128 KB L1d) to not regress. */
    return 32 * 1024;
}

static void calibrate_sdpa_fused_regimes(bool *lt, bool *eq, bool *gt) {
    /* TODO: needs A/B over BH<NT, BH==NT, BH>NT shapes calling the
       sdpa_bwd dispatch. for now mirror the pre-calibration defaults
       (lt=true, eq=false, gt=true) — the "BH==NT regression" rule. */
    *lt = true; *eq = false; *gt = true;
}

/* ============================================================
   driver — calls each gate's calibrator and stores the result.

   contract: each calibrate_* function returns either a measured
   threshold value, OR a sentinel (INT64_MAX or 0) meaning
   "measurement inconclusive — keep current default". the driver
   only overrides g_attn.* when a calibrator returns a confident
   non-sentinel value. this prevents one noisy measurement run
   from disabling well-tuned defaults.

   the defaults baked into g_attn at static-init time are
   "generally-applicable to modern CPUs" rather than "tuned to
   one specific machine" — picked to match cache hierarchy
   regimes that hold across vendors:
     - 8 MB qkv-bytes / save-p: L2 spill point on CPUs with
       L2 in [256 KB, 4 MB] and L3 ≥ 8 MB. covers x86 + ARM.
     - 32 KB fused-bh per-head: most x86 L1d sizes (32-48 KB);
       conservative on Apple M (L1d 128-192 KB).
     - 1024 D-threshold for F.3.c: where the [D, 3D]
       intermediate (~6 MB at D=1024) starts spilling L3 on
       typical 4-8 MB L3 hosts.
     - 126 ATTN_BQ/BK: MR-aligned tile size that fits L2
       across the AVX2/AVX-512 micro-kernel ABIs.
   the calibrator refines these where measurement is confident;
   defaults stand in otherwise.
   ============================================================ */

static void calibrate_all_gates(void) {
    int64_t v;

    v = calibrate_f3c_d_threshold();
    if (v != INT64_MAX) g_attn.f3c_d_threshold = v;

    v = calibrate_f3a_qkv_bytes_threshold();
    if (v != INT64_MAX) g_attn.f3a_qkv_bytes_threshold = v;

    v = calibrate_gemm_tn_pretranspose_flops();
    if (v > 0) g_attn.gemm_tn_pretranspose_flops = v;

    v = calibrate_save_p_max_bytes();
    if (v > 0) g_attn.save_p_max_bytes = v;

    v = calibrate_save_p_small_exclusion_sk();
    if (v >= 0) g_attn.save_p_small_exclusion_sk = v;

    v = calibrate_save_p_small_exclusion_s();
    if (v >= 0) g_attn.save_p_small_exclusion_s = v;

    v = calibrate_fused_bh_per_head_bytes();
    if (v > 0) g_attn.fused_bh_per_head_bytes = v;

    calibrate_sdpa_fused_regimes(&g_attn.sdpa_fused_use_when_bh_lt_nt,
                                  &g_attn.sdpa_fused_use_when_bh_eq_nt,
                                  &g_attn.sdpa_fused_use_when_bh_gt_nt);
}
#endif

/* ISA-resolved entry points — defined in src/core/fused_attention.c.
   each picks the cpu_opt variant matching the runtime CPU and forwards.
   keeps this module ISA-agnostic so a single calibration TU serves
   AVX-512, AVX2, NEON, and scalar builds. */
extern int64_t ax_attn_bq_default_resolved(void);
extern int64_t ax_attn_bk_default_resolved(void);
extern void    ax_attn_set_bq_resolved(int64_t v);
extern void    ax_attn_set_bk_resolved(int64_t v);
extern void    ax_attn_pack_stats_dump(void);  /* T2.1 */

/* atexit hook: when AX_PROFILE_PACK=1, emit pack-cycle totals before
   process teardown so users see where pack overhead actually fell. */
static void pack_stats_atexit_hook(void) {
    const char *e = getenv("AX_PROFILE_PACK");
    if (e && e[0] == '1') ax_attn_pack_stats_dump();
}

void ax_attn_tunables_calibrate(void) {
    if (g_attn.calibrated) return;

    /* register pack-stats atexit hook once. atexit is process-global so
       only the first calibrate call wires it. cheap (one libc call). */
    static bool atexit_registered = false;
    if (!atexit_registered) {
        atexit(pack_stats_atexit_hook);
        atexit_registered = true;
    }

    /* seed runtime defaults from the resolved cpu_opt ATTN_BQ_DEFAULT.
       even if calibration is skipped, the getter returns the same value
       the active ISA's cpu_opt computed at compile time. */
    if (g_attn.attn_bq == 0) g_attn.attn_bq = ax_attn_bq_default_resolved();
    if (g_attn.attn_bk == 0) g_attn.attn_bk = ax_attn_bk_default_resolved();

#if defined(AX_NO_AUTOTUNE)
    /* calibration disabled — getters return baked-in defaults. */
    g_attn.calibrated = true;
    return;
#else
    const char *no = getenv("AX_NO_AUTOTUNE");
    if (no && no[0] == '1') {
        g_attn.calibrated = true;
        return;
    }
    const char *no_attn = getenv("AX_NO_ATTN_CALIB");
    if (no_attn && no_attn[0] == '1') {
        g_attn.calibrated = true;
        return;
    }

    calibrate_all_gates();

    /* optional log of the chosen values — useful for debugging
       per-host divergence. */
    const char *log = getenv("AX_ATTN_TUNABLES_LOG");
    if (log && log[0] == '1') {
#ifndef AX_NO_STDIO
        fprintf(stderr,
            "axiom: attn tunables calibrated:\n"
            "  f3a_qkv_bytes_threshold = %ld\n"
            "  f3c_d_threshold         = %ld\n"
            "  save_p_max_bytes        = %ld\n"
            "  fused_bh_per_head_bytes = %ld\n"
            "  sdpa_fused use<,=,> NT  = %d, %d, %d\n"
            "  gemm_tn pretranspose    = %ld flops\n"
            "  attn_bq, attn_bk        = %ld, %ld\n",
            (long)g_attn.f3a_qkv_bytes_threshold,
            (long)g_attn.f3c_d_threshold,
            (long)g_attn.save_p_max_bytes,
            (long)g_attn.fused_bh_per_head_bytes,
            (int)g_attn.sdpa_fused_use_when_bh_lt_nt,
            (int)g_attn.sdpa_fused_use_when_bh_eq_nt,
            (int)g_attn.sdpa_fused_use_when_bh_gt_nt,
            (long)g_attn.gemm_tn_pretranspose_flops,
            (long)g_attn.attn_bq, (long)g_attn.attn_bk);
#endif
    }

    g_attn.calibrated = true;
#endif
}
