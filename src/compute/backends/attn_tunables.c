/* attn_tunables.c — runtime-calibrated thresholds for attention paths.

   each gate that selects between a fused and unfused kernel variant is
   measured at startup with a synthetic workload representing the actual
   decision. results cached per host (calib_cache) so the cost is paid
   once per machine + library version.

   on AX_NO_AUTOTUNE=1 the calibrate function is a no-op and the getters
   return conservative defaults matching the pre-calibration hardcoded
   values — the behaviour everyone shipped before this work landed. */

#include "axiom/internal/attn_tunables.h"
#include "axiom/internal/calib_cache.h"
#include "axiom/tensor.h"
#include "axiom/compute.h"
#include "axiom/memory.h"
#include "axiom/attention.h"
#include "axiom/layer.h"

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
       calibrator decides per-BH-vs-NT regime; conservative pre-cal
       default is "always materialized" (false everywhere) so that
       AX_NO_AUTOTUNE=1 + unset env var matches the pre-tunable
       "force off when unset" behaviour. the calibrator flips
       individual regimes to true only when measurement is confident. */
    bool sdpa_fused_use_when_bh_lt_nt;    /* default: false */
    bool sdpa_fused_use_when_bh_eq_nt;    /* default: false */
    bool sdpa_fused_use_when_bh_gt_nt;    /* default: false */

    /* gemm_tn pre-transpose flop threshold */
    int64_t gemm_tn_pretranspose_flops;   /* default: 2 GFLOPS */

    /* attention tile sizes (MR-multiples) */
    int64_t attn_bq;                      /* default: ATTN_BQ_DEFAULT */
    int64_t attn_bk;                      /* default: ATTN_BQ_DEFAULT */

    /* GEMM tile calibration switch margin. cand wins iff
       cand_score < base_score * gemm_tile_switch_margin. values
       in [0.5, 1.0); 0.94 = "must beat baseline by ≥6 %". */
    double gemm_tile_switch_margin;       /* default: 0.94 */

    /* (a) unpacked-A micro-kernel max K. when kc <= this threshold AND
       the matrix is contiguous, opt_gemm uses micro_kernel_unpacked_a
       to skip pack_a entirely (saves the write+read traffic, ~14-21 %
       of total bench time on small-shape gemms per pack-profile data).
       above this kc, packed remains the default — strided A reads
       across MR rows blow L1d when each row is too long. default 512
       fits MR=14 rows × 512 floats = 28 KB ≤ typical 32 KB L1d on
       AVX-512, comfortably for AVX2 MR=6 (12 KB). calibrator sweeps
       per-host. set to 0 to disable the unpacked path entirely. */
    int64_t gemm_unpacked_a_max_k;        /* default: 512 */
} g_attn = {
    .calibrated = false,
    .f3a_qkv_bytes_threshold = (int64_t)8 * 1024 * 1024,
    .f3c_d_threshold = 1024,
    .save_p_max_bytes = (int64_t)8 * 1024 * 1024,
    .save_p_small_exclusion_sk = 8192,
    .save_p_small_exclusion_s = 128,
    .fused_bh_per_head_bytes = 32 * 1024,
    .sdpa_fused_use_when_bh_lt_nt = false,
    .sdpa_fused_use_when_bh_eq_nt = false,
    .sdpa_fused_use_when_bh_gt_nt = false,
    .gemm_tn_pretranspose_flops = 2000000000LL,
    /* attn_bq/bk seeded from the existing ATTN_BQ_DEFAULT macro at
       calibrate-time. without that we'd duplicate the macro definition
       here; see ax_attn_tunables_calibrate(). */
    .attn_bq = 0,
    .attn_bk = 0,
    .gemm_tile_switch_margin = 0.94,
    .gemm_unpacked_a_max_k = 512,
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

double ax_attn_tunable_gemm_tile_switch_margin(void) {
    return g_attn.gemm_tile_switch_margin;
}

int64_t ax_attn_tunable_gemm_unpacked_a_max_k(void) {
    return g_attn.gemm_unpacked_a_max_k;
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
   mha-train micro-harness — owns one MHA layer + Q/K/V buffers
   for the duration of a calibration sweep, runs ax_mha_train_step
   (autograd path) or ax_mha_train_step_v4 (fused-bh path) per call.

   the harness is the shared substrate for save_p / fused_bh /
   sdpa_fused calibration: each calibrator flips one of g_attn's
   gate fields between a pair of values, runs the harness through
   the trimmed-mean timer, and picks the winner.

   we use the autograd train_step (not train_step_fused) for the
   save_p calibration so the gate inside train_step.c is the one
   exercised. fused_bh calibration uses train_step_v4 because
   that's the path with the per_head_bytes gate. sdpa_fused
   calibration uses train_step too — its bwd routes through the
   sdpa_bwd dispatch where AX_SDPA_FUSED resolves.
   ============================================================ */

typedef struct meas_mha {
    ax_layer_t  *mha;
    ax_tensor_t *x;
    ax_tensor_t *dout;
    ax_tensor_t *y_out;
    /* shape kept for sanity/printing */
    int64_t B, S, D;
    int     H;
} meas_mha_t;

static ax_status_t meas_mha_create(meas_mha_t *m, int64_t B, int64_t S,
                                    int64_t D, int H, bool causal) {
    memset(m, 0, sizeof(*m));
    m->B = B; m->S = S; m->D = D; m->H = H;
    m->mha = ax_mha_create(D, H, /*use_bias*/false, causal);
    if (!m->mha) return AX_ERR_ALLOC;
    ax_layer_train(m->mha);
    int64_t shape[3] = { B, S, D };
    m->x     = ax_tensor_rand(shape, 3, -0.5f, 0.5f);
    m->dout  = ax_tensor_rand(shape, 3, -0.5f, 0.5f);
    m->y_out = ax_tensor_create(shape, 3, AX_FLOAT32);
    if (!m->x || !m->dout || !m->y_out) {
        if (m->x)     ax_tensor_destroy(m->x);
        if (m->dout)  ax_tensor_destroy(m->dout);
        if (m->y_out) ax_tensor_destroy(m->y_out);
        ax_layer_destroy(m->mha);
        memset(m, 0, sizeof(*m));
        return AX_ERR_ALLOC;
    }
    return AX_OK;
}

static void meas_mha_destroy(meas_mha_t *m) {
    if (m->y_out) ax_tensor_destroy(m->y_out);
    if (m->dout)  ax_tensor_destroy(m->dout);
    if (m->x)     ax_tensor_destroy(m->x);
    if (m->mha)   ax_layer_destroy(m->mha);
    memset(m, 0, sizeof(*m));
}

static ax_status_t meas_mha_run_step(void *p) {
    meas_mha_t *c = (meas_mha_t *)p;
    return ax_mha_train_step(c->mha, c->x, c->dout, c->y_out);
}

static ax_status_t meas_mha_run_step_v4(void *p) {
    meas_mha_t *c = (meas_mha_t *)p;
    return ax_mha_train_step_v4(c->mha, c->x, c->dout, c->y_out);
}

/* ============================================================
   save_p calibration

   the gate sits in train_step.c at the SDPA fwd point:
     save_p = (p_save_bytes <= ax_attn_tunable_save_p_max_bytes())

   for each candidate (B, S, H, dk), we time a full mha_train_step
   twice — once with save_p forced ON (bytes-cap = INT64_MAX,
   small-S/dk exclusions = 0/0), once forced OFF (bytes-cap = 0).
   the threshold is the largest p_save_bytes where ON wins; above
   that, OFF (recompute) is faster.

   small-S/dk exclusion thresholds are calibrated separately
   (calibrate_save_p_small_exclusion_*) by sweeping S in {64, 96,
   128, 192, 256} at fixed dk=64 and finding the largest S where
   recompute beats save even though save's memory budget is below
   the bytes cap. that gives the s_threshold; the sk_threshold is
   the corresponding S*dk product.
   ============================================================ */

typedef struct {
    /* shape spans cache regimes:
        (4, 128, 8, 64) → p = 4*8*128*128*4 = 2 MB    (small)
        (8, 128, 8, 64) → p = 4 MB
        (4, 256, 8, 64) → p = 4 MB
        (8, 256, 8, 64) → p = 16 MB                    (above default 8 MB)
        (4, 512, 8, 64) → p = 16 MB                    (above default)
        (2,1024,12,64) → p = 96 MB                    (well above) */
    int64_t B, S;
    int64_t D;
    int     H;
} sp_shape_t;

static int64_t calibrate_save_p_max_bytes(void) {
    /* representative shapes covering the regime where save_p is
       contested. the gate matters most when p_save_bytes is in
       [1 MB, 100 MB] — below that the small-S exclusion handles
       it; above, recompute always wins. */
    static const sp_shape_t cand[] = {
        { 4, 128, 512,  8 },   /*  p=2 MB, s_p<=8 MB → save ON */
        { 8, 128, 512,  8 },   /*  p=4 MB */
        { 4, 256, 512,  8 },   /*  p=4 MB */
        { 8, 256, 512,  8 },   /*  p=16 MB */
        { 4, 512, 512,  8 },   /*  p=16 MB */
        { 2,1024, 768, 12 },   /*  p=96 MB */
    };
    static const int n_cand = (int)(sizeof(cand) / sizeof(cand[0]));
    /* heavier sample budget than per-gate calibrators that hit small
       gemms: each iter is a full mha_train_step (5-50 ms), so noise
       at 3 % margin is borderline. warm=2 lets the layer's TLS scratch
       and arena settle; timed=8 → trim 2 high samples → 6 averaged. */
    const int warm = 2, timed = 8;
    const double margin = 0.95;  /* 5 % win margin for the wider noise band */

    /* save existing g_attn fields we'll mutate so we can restore
       them between/after measurements. */
    int64_t saved_max  = g_attn.save_p_max_bytes;
    int64_t saved_excl_s  = g_attn.save_p_small_exclusion_s;
    int64_t saved_excl_sk = g_attn.save_p_small_exclusion_sk;

    /* the running answer: largest p_save_bytes where save_p ON wins.
       we initialise to the pre-calibration default (8 MB) so if no
       shape provides a confident measurement, we keep that default. */
    int64_t best = saved_max;
    bool    any_measured = false;

    for (int i = 0; i < n_cand; i++) {
        const sp_shape_t *s = &cand[i];
        int64_t bh = s->B * s->H;
        int64_t dk = s->D / s->H;
        int64_t p_bytes = bh * s->S * s->S * (int64_t)sizeof(float);

        meas_mha_t mh;
        if (meas_mha_create(&mh, s->B, s->S, s->D, s->H, /*causal*/false) != AX_OK)
            continue;

        /* force save_p ON: huge cap, no small-S/dk exclusion. */
        g_attn.save_p_max_bytes        = INT64_MAX;
        g_attn.save_p_small_exclusion_s  = 0;
        g_attn.save_p_small_exclusion_sk = 0;
        double t_on = meas_trimmed_mean_ms(meas_mha_run_step, &mh, warm, timed);

        /* force save_p OFF: cap = 0 → save_p = (p_bytes <= 0) = false. */
        g_attn.save_p_max_bytes = 0;
        double t_off = meas_trimmed_mean_ms(meas_mha_run_step, &mh, warm, timed);

        meas_mha_destroy(&mh);

        if (t_on >= 1.0e29 || t_off >= 1.0e29) continue;
        any_measured = true;

        /* save wins: t_on < margin * t_off. if save wins on this shape,
           we know p_bytes is below the crossover, so the threshold
           should be at least p_bytes. pick the largest such p_bytes.
           if save loses (t_on >= t_off), p_bytes is at/above the
           crossover; we don't lower `best` from a single noisy
           sample — a smaller shape may have already delivered a
           higher confirmed bound. */
        if (t_on < t_off * margin && p_bytes > best) {
            best = p_bytes;
        }
        (void)dk; (void)bh;
    }

    g_attn.save_p_max_bytes        = saved_max;
    g_attn.save_p_small_exclusion_s  = saved_excl_s;
    g_attn.save_p_small_exclusion_sk = saved_excl_sk;

    if (!any_measured) return 0;  /* sentinel: keep default */
    return best;
}

/* small-S exclusion: at small S, save_p ON saves a small bytes count
   but the per-head materialisation overhead (zero pollution + extra
   memory pass) often outweighs the bwd recompute saving. measure at
   a small dk fixed, sweep S, find largest S where save loses. */
static int64_t calibrate_save_p_small_exclusion_s(void) {
    /* fixed B*H*dk so total compute is comparable across S. */
    static const int64_t cand_S[] = { 64, 96, 128, 192 };
    static const int     n_cand = (int)(sizeof(cand_S) / sizeof(cand_S[0]));
    const int warm = 2, timed = 8;
    const double margin = 0.95;  /* 5 % margin */
    const int64_t B = 8;
    const int64_t D = 512;
    const int     H = 8;

    int64_t saved_max  = g_attn.save_p_max_bytes;
    int64_t saved_excl_s  = g_attn.save_p_small_exclusion_s;
    int64_t saved_excl_sk = g_attn.save_p_small_exclusion_sk;

    int64_t threshold_s = 0;     /* largest S where recompute beats save */
    bool    any_measured = false;

    for (int i = 0; i < n_cand; i++) {
        int64_t S = cand_S[i];
        meas_mha_t mh;
        if (meas_mha_create(&mh, B, S, D, H, /*causal*/false) != AX_OK)
            continue;

        g_attn.save_p_max_bytes        = INT64_MAX;
        g_attn.save_p_small_exclusion_s  = 0;
        g_attn.save_p_small_exclusion_sk = 0;
        double t_on = meas_trimmed_mean_ms(meas_mha_run_step, &mh, warm, timed);

        g_attn.save_p_max_bytes = 0;
        double t_off = meas_trimmed_mean_ms(meas_mha_run_step, &mh, warm, timed);

        meas_mha_destroy(&mh);

        if (t_on >= 1.0e29 || t_off >= 1.0e29) continue;
        any_measured = true;

        /* recompute wins (save loses) at this S → exclude S from save. */
        if (t_off < t_on * margin && S > threshold_s) {
            threshold_s = S;
        }
    }

    g_attn.save_p_max_bytes        = saved_max;
    g_attn.save_p_small_exclusion_s  = saved_excl_s;
    g_attn.save_p_small_exclusion_sk = saved_excl_sk;

    if (!any_measured) return -1;  /* sentinel: keep default */
    return threshold_s;
}

static int64_t calibrate_save_p_small_exclusion_sk(void) {
    /* the original gate uses S*dk as a separate criterion (S<=128 AND
       S*dk<=8192, both must hold for exclusion to fire). dk is fixed
       at 64 in the calibrate_save_p_small_exclusion_s sweep, so the
       sk threshold should track 128*64 = 8192 as the canonical
       crossover. measuring sk independently would require a separate
       sweep over dk while holding S small; that's marginal value
       since dk is typically 64 in the workloads we care about.
       return the canonical 8192. */
    return 8192;
}

/* ============================================================
   fused_bh calibration

   the gate sits in train_step_v4.c (auto_enable_fused_bh):
     fits_l1 = per_head_bytes <= ax_attn_tunable_fused_bh_per_head_bytes()
     enabled = fits_l1 && BH >= NT

   for each candidate per_head_bytes, we time train_step_v4 twice:
     ON:  per_head_bytes_threshold = INT64_MAX → fits_l1 = true
     OFF: per_head_bytes_threshold = 0          → fits_l1 = false
   (BH >= NT must hold for the gate to fire; we pick shapes that
    satisfy it.)

   per_head_bytes = S * dk * 4. shapes vary S + dk to span the L1
   regime (~16 KB) up to L2 spill (~256 KB).
   ============================================================ */

typedef struct {
    int64_t B, S;
    int64_t D;
    int     H;
} fbh_shape_t;

static int64_t calibrate_fused_bh_per_head_bytes(void) {
    /* shapes ordered by per_head_bytes ascending. BH ≥ NT for all. */
    static const fbh_shape_t cand[] = {
        /* per_head_bytes = S * dk * 4, dk = D/H */
        { 4,   64, 512,  8 },   /*  S=64, dk=64 → 16 KB */
        { 4,  128, 512,  8 },   /*  32 KB */
        { 4,  192, 512,  8 },   /*  48 KB */
        { 4,  256, 512,  8 },   /*  64 KB */
        { 2,  512, 768, 12 },   /*  S=512, dk=64 → 128 KB */
    };
    static const int n_cand = (int)(sizeof(cand) / sizeof(cand[0]));
    const int warm = 2, timed = 8;
    const double margin = 0.95;

#ifdef _OPENMP
    int nt = omp_get_max_threads();
#else
    int nt = 1;
#endif

    int64_t saved = g_attn.fused_bh_per_head_bytes;
    int64_t best  = saved;
    bool    any_measured = false;

    for (int i = 0; i < n_cand; i++) {
        const fbh_shape_t *s = &cand[i];
        int64_t bh = s->B * s->H;
        if (bh < (int64_t)nt) continue;  /* gate's enough_heads requires it */
        int64_t dk = s->D / s->H;
        int64_t per_head_bytes = s->S * dk * (int64_t)sizeof(float);

        meas_mha_t mh;
        if (meas_mha_create(&mh, s->B, s->S, s->D, s->H, /*causal*/false) != AX_OK)
            continue;

        /* force fused_bh ON */
        g_attn.fused_bh_per_head_bytes = INT64_MAX;
        double t_on = meas_trimmed_mean_ms(meas_mha_run_step_v4, &mh, warm, timed);

        /* force fused_bh OFF */
        g_attn.fused_bh_per_head_bytes = 0;
        double t_off = meas_trimmed_mean_ms(meas_mha_run_step_v4, &mh, warm, timed);

        meas_mha_destroy(&mh);

        if (t_on >= 1.0e29 || t_off >= 1.0e29) continue;
        any_measured = true;

        if (t_on < t_off * margin && per_head_bytes > best) {
            best = per_head_bytes;
        }
    }

    g_attn.fused_bh_per_head_bytes = saved;
    if (!any_measured) return 0;
    return best;
}

/* ============================================================
   sdpa_fused regimes calibration

   the dispatch (ax_attn_bwd_get_impl in cpu_opt.c) consults
   ax_attn_tunable_use_sdpa_fused(BH) when the env var is unset
   or =auto. that getter returns one of three booleans depending
   on BH vs NT.

   for each regime (BH<NT, BH=NT, BH>NT), we pick a representative
   shape and A/B by flipping the corresponding boolean in g_attn
   while running mha train_step. since the dispatch caches env_mode
   on first call, we ensure no AX_SDPA_FUSED env override is set
   when calibration runs (the default env_mode now resolves to
   "consult tunable" — the dispatch then reads g_attn directly).
   ============================================================ */

static double meas_sdpa_regime(int64_t B, int64_t S, int64_t D, int H,
                                bool *target_flag, bool force_value,
                                int warm, int timed) {
    meas_mha_t mh;
    if (meas_mha_create(&mh, B, S, D, H, /*causal*/false) != AX_OK)
        return 1.0e30;
    bool saved = *target_flag;
    *target_flag = force_value;
    double t = meas_trimmed_mean_ms(meas_mha_run_step, &mh, warm, timed);
    *target_flag = saved;
    meas_mha_destroy(&mh);
    return t;
}

static void calibrate_sdpa_fused_regimes(bool *lt, bool *eq, bool *gt) {
#ifdef _OPENMP
    int nt = omp_get_max_threads();
#else
    int nt = 1;
#endif

    /* skip calibration if AX_SDPA_FUSED is set — the env override
       wins over our tunable settings, so any A/B we attempt won't
       be reflected in the dispatch. retain conservative defaults. */
    const char *env = getenv("AX_SDPA_FUSED");
    if (env && env[0] != '\0' && env[0] != 'a' && env[0] != 'A') {
        *lt = g_attn.sdpa_fused_use_when_bh_lt_nt;
        *eq = g_attn.sdpa_fused_use_when_bh_eq_nt;
        *gt = g_attn.sdpa_fused_use_when_bh_gt_nt;
        return;
    }

    /* shape parameters tuned so each regime hits with a workload that
       still fits a calibration time budget (< ~500 ms / measurement).
       D=512, H=8 gives BH = 8*B; we pick B to land on/above/below NT. */
    const int64_t D = 512;
    const int     H = 8;
    const int64_t S = 256;
    const int     warm = 2, timed = 8;
    const double  margin = 0.95;  /* 5 % win margin */

    /* BH < NT: pick B such that BH = max(1, nt/2). */
    int64_t B_lt = (nt >= 16) ? 1 : 1;  /* BH = 8 < NT=16; BH=8 < NT for NT>=10 */
    /* BH = NT: pick B = nt/H (rounded). if NT/H is not integer, skip. */
    int64_t B_eq = (nt % H == 0) ? (nt / H) : 0;
    /* BH > NT: pick B such that BH = 2*NT (rounded up). */
    int64_t B_gt = ((int64_t)2 * nt + H - 1) / H;
    if (B_gt < 1) B_gt = 1;

    /* lt regime */
    if (B_lt * H < nt) {
        double t_on  = meas_sdpa_regime(B_lt, S, D, H,
                                         &g_attn.sdpa_fused_use_when_bh_lt_nt, true,
                                         warm, timed);
        double t_off = meas_sdpa_regime(B_lt, S, D, H,
                                         &g_attn.sdpa_fused_use_when_bh_lt_nt, false,
                                         warm, timed);
        if (t_on < 1.0e29 && t_off < 1.0e29) {
            *lt = (t_on < t_off * margin);  /* fused wins by margin */
        } else {
            *lt = g_attn.sdpa_fused_use_when_bh_lt_nt;
        }
    } else {
        *lt = g_attn.sdpa_fused_use_when_bh_lt_nt;
    }

    /* eq regime — only measurable when BH==NT is reachable */
    if (B_eq > 0 && B_eq * H == nt) {
        double t_on  = meas_sdpa_regime(B_eq, S, D, H,
                                         &g_attn.sdpa_fused_use_when_bh_eq_nt, true,
                                         warm, timed);
        double t_off = meas_sdpa_regime(B_eq, S, D, H,
                                         &g_attn.sdpa_fused_use_when_bh_eq_nt, false,
                                         warm, timed);
        if (t_on < 1.0e29 && t_off < 1.0e29) {
            *eq = (t_on < t_off * margin);
        } else {
            *eq = g_attn.sdpa_fused_use_when_bh_eq_nt;
        }
    } else {
        *eq = g_attn.sdpa_fused_use_when_bh_eq_nt;
    }

    /* gt regime */
    {
        double t_on  = meas_sdpa_regime(B_gt, S, D, H,
                                         &g_attn.sdpa_fused_use_when_bh_gt_nt, true,
                                         warm, timed);
        double t_off = meas_sdpa_regime(B_gt, S, D, H,
                                         &g_attn.sdpa_fused_use_when_bh_gt_nt, false,
                                         warm, timed);
        if (t_on < 1.0e29 && t_off < 1.0e29) {
            *gt = (t_on < t_off * margin);
        } else {
            *gt = g_attn.sdpa_fused_use_when_bh_gt_nt;
        }
    }
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
extern void    ax_gemm_tiles_get_resolved(int64_t *, int64_t *, int64_t *);  /* A4 */
extern void    ax_gemm_tiles_set_resolved(int64_t, int64_t, int64_t);          /* A4 */
extern void    ax_gemm_tiles_get_regime_resolved(int, int64_t *, int64_t *, int64_t *);  /* A3 */
extern void    ax_gemm_tiles_set_regime_resolved(int, int64_t, int64_t, int64_t);          /* A3 */

/* atexit hook: when AX_PROFILE_PACK=1, emit pack-cycle totals before
   process teardown so users see where pack overhead actually fell. */
static void pack_stats_atexit_hook(void) {
    const char *e = getenv("AX_PROFILE_PACK");
    if (e && e[0] == '1') ax_attn_pack_stats_dump();
}

/* early init: resolves env-var overrides for tunables that downstream
   calibrators (notably ax_calibrate_gemm_tiles) consult. registers
   atexit hooks. NO measurements — that's ax_attn_tunables_calibrate.
   safe to call before the active backend is initialised; idempotent. */
void ax_attn_tunables_init_early(void) {
    static bool inited = false;
    if (inited) return;
    inited = true;

    /* register pack-stats atexit hook once. cheap (one libc call). */
    atexit(pack_stats_atexit_hook);

    /* AX_GEMM_TILE_SWITCH_MARGIN: numeric override, e.g. 0.92 ("must
       beat baseline by 8 %") or 0.96 ("only switch on a >= 4 % win").
       reject out-of-range values (sticking with default 0.94 if user
       passes garbage). exposed as env-var so users on noisier hosts
       can loosen the margin without rebuilding. */
    const char *env = getenv("AX_GEMM_TILE_SWITCH_MARGIN");
    if (env && env[0] != '\0') {
        char *endp = NULL;
        double v = strtod(env, &endp);
        if (endp != env && v >= 0.5 && v < 1.0) {
            g_attn.gemm_tile_switch_margin = v;
        }
    }

    /* A2: SDPA tile size overrides — AX_ATTN_BQ / AX_ATTN_BK. resolved
       early so the cpu_opt static ATTN_BQ/ATTN_BK are set before the
       first SDPA call. cpu_opt's setters validate alignment; we just
       parse and forward. proper measurement-based calibration of these
       remains future work (would need SDPA-specific A/B harness; the
       defaults are MR/NR-aligned values that fit L2 across vendors). */
    const char *bq_env = getenv("AX_ATTN_BQ");
    if (bq_env && bq_env[0] != '\0') {
        char *endp = NULL;
        long v = strtol(bq_env, &endp, 10);
        if (endp != bq_env && v > 0 && v <= 4096) {
            g_attn.attn_bq = v;
        }
    }
    const char *bk_env = getenv("AX_ATTN_BK");
    if (bk_env && bk_env[0] != '\0') {
        char *endp = NULL;
        long v = strtol(bk_env, &endp, 10);
        if (endp != bk_env && v > 0 && v <= 4096) {
            g_attn.attn_bk = v;
        }
    }

    /* (a) AX_GEMM_UNPACKED_A_MAX_K: override the kc threshold above
       which the unpacked-A micro-kernel is skipped. set to 0 to disable
       the unpacked path entirely (forces packed for all shapes). */
    const char *uak_env = getenv("AX_GEMM_UNPACKED_A_MAX_K");
    if (uak_env && uak_env[0] != '\0') {
        char *endp = NULL;
        long v = strtol(uak_env, &endp, 10);
        if (endp != uak_env && v >= 0 && v <= 8192) {
            g_attn.gemm_unpacked_a_max_k = v;
        }
    }
}

void ax_attn_tunables_calibrate(void) {
    if (g_attn.calibrated) return;

    /* run the early-init phase if it hasn't been run yet (some entry
       points may call calibrate without going through the normal
       compute-init path). cheap and idempotent. */
    ax_attn_tunables_init_early();

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

/* ============================================================
   A4: calibration cache integration
   ============================================================ */

/* indicates that ax_calib_cache_try_apply() loaded a payload — we
   use this to skip both the gemm-tile sweep and the attn calibrator,
   and to bypass the post-calibration cache save. */
static bool g_calib_loaded_from_cache = false;

bool ax_calib_cache_try_apply(void) {
    if (g_attn.calibrated) return false;  /* already done some other way */

    /* run early init first so env-var overrides (notably the
       gemm_tile_switch_margin) hit before we even consider a cache
       hit — env beats cache, by design. */
    ax_attn_tunables_init_early();

    ax_calib_payload_t pl;
    if (!ax_calib_cache_load(&pl)) return false;

    /* apply cached values to the live tunable state. */
    g_attn.f3a_qkv_bytes_threshold       = pl.f3a_qkv_bytes_threshold;
    g_attn.f3c_d_threshold               = pl.f3c_d_threshold;
    g_attn.save_p_max_bytes              = pl.save_p_max_bytes;
    g_attn.save_p_small_exclusion_sk     = pl.save_p_small_exclusion_sk;
    g_attn.save_p_small_exclusion_s      = pl.save_p_small_exclusion_s;
    g_attn.fused_bh_per_head_bytes       = pl.fused_bh_per_head_bytes;
    g_attn.gemm_tn_pretranspose_flops    = pl.gemm_tn_pretranspose_flops;
    g_attn.attn_bq                       = pl.attn_bq;
    g_attn.attn_bk                       = pl.attn_bk;
    g_attn.sdpa_fused_use_when_bh_lt_nt  = (bool)pl.sdpa_fused_use_when_bh_lt_nt;
    g_attn.sdpa_fused_use_when_bh_eq_nt  = (bool)pl.sdpa_fused_use_when_bh_eq_nt;
    g_attn.sdpa_fused_use_when_bh_gt_nt  = (bool)pl.sdpa_fused_use_when_bh_gt_nt;
    /* preserve whatever init_early set the margin to (env-var override
       wins over cached value). only fall through to cache if init_early
       didn't change it. */
    if (g_attn.gemm_tile_switch_margin == 0.94)  /* default unchanged */
        g_attn.gemm_tile_switch_margin = pl.gemm_tile_switch_margin;

    /* push attn tile sizes into cpu_opt's static so attn fwd/bwd take
       effect immediately. gemm tiles ditto via the resolved setter,
       both the legacy "global" tile (for callers not using per-regime
       dispatch) and the three per-regime sets used by apply_tiles_for_
       shape() in cpu_opt.c. */
    ax_attn_set_bq_resolved(g_attn.attn_bq);
    ax_attn_set_bk_resolved(g_attn.attn_bk);
    ax_gemm_tiles_set_resolved(pl.gemm_mc, pl.gemm_nc, pl.gemm_kc);
    ax_gemm_tiles_set_regime_resolved(0, pl.gemm_mc_small, pl.gemm_nc_small, pl.gemm_kc_small);
    ax_gemm_tiles_set_regime_resolved(1, pl.gemm_mc_med,   pl.gemm_nc_med,   pl.gemm_kc_med);
    ax_gemm_tiles_set_regime_resolved(2, pl.gemm_mc_large, pl.gemm_nc_large, pl.gemm_kc_large);

    g_attn.calibrated = true;
    g_calib_loaded_from_cache = true;
    return true;
}

bool ax_calib_loaded_from_cache(void) {
    return g_calib_loaded_from_cache;
}

void ax_calib_cache_save_current(void) {
    if (!g_attn.calibrated) return;
    if (g_calib_loaded_from_cache) return;  /* nothing new to persist */

    ax_calib_payload_t pl;
    pl.f3a_qkv_bytes_threshold       = g_attn.f3a_qkv_bytes_threshold;
    pl.f3c_d_threshold               = g_attn.f3c_d_threshold;
    pl.save_p_max_bytes              = g_attn.save_p_max_bytes;
    pl.save_p_small_exclusion_sk     = g_attn.save_p_small_exclusion_sk;
    pl.save_p_small_exclusion_s      = g_attn.save_p_small_exclusion_s;
    pl.fused_bh_per_head_bytes       = g_attn.fused_bh_per_head_bytes;
    pl.gemm_tn_pretranspose_flops    = g_attn.gemm_tn_pretranspose_flops;
    pl.attn_bq                       = g_attn.attn_bq;
    pl.attn_bk                       = g_attn.attn_bk;
    pl.sdpa_fused_use_when_bh_lt_nt  = (int8_t)g_attn.sdpa_fused_use_when_bh_lt_nt;
    pl.sdpa_fused_use_when_bh_eq_nt  = (int8_t)g_attn.sdpa_fused_use_when_bh_eq_nt;
    pl.sdpa_fused_use_when_bh_gt_nt  = (int8_t)g_attn.sdpa_fused_use_when_bh_gt_nt;
    pl._pad0                         = 0;
    pl.gemm_tile_switch_margin       = g_attn.gemm_tile_switch_margin;

    int64_t mc = 0, nc = 0, kc = 0;
    ax_gemm_tiles_get_resolved(&mc, &nc, &kc);
    pl.gemm_mc = mc; pl.gemm_nc = nc; pl.gemm_kc = kc;

    /* A3: per-regime tile sets. these are populated by the gemm-tile
       calibrator (cpu_opt.c) which now picks per-regime winners. */
    ax_gemm_tiles_get_regime_resolved(0, &mc, &nc, &kc);
    pl.gemm_mc_small = mc; pl.gemm_nc_small = nc; pl.gemm_kc_small = kc;
    ax_gemm_tiles_get_regime_resolved(1, &mc, &nc, &kc);
    pl.gemm_mc_med   = mc; pl.gemm_nc_med   = nc; pl.gemm_kc_med   = kc;
    ax_gemm_tiles_get_regime_resolved(2, &mc, &nc, &kc);
    pl.gemm_mc_large = mc; pl.gemm_nc_large = nc; pl.gemm_kc_large = kc;

    (void)ax_calib_cache_save(&pl);
}
