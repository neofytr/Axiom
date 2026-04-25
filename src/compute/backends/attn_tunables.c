/* attn_tunables.c — runtime-calibrated thresholds for attention paths.

   each gate that selects between a fused and unfused kernel variant is
   measured at startup with a synthetic workload representing the actual
   decision. results cached per host (calib_cache) so the cost is paid
   once per machine + library version.

   on AX_NO_AUTOTUNE=1 the calibrate function is a no-op and the getters
   return conservative defaults matching the pre-calibration hardcoded
   values — the behaviour everyone shipped before this work landed. */

#include "axiom/internal/attn_tunables.h"

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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
/* extern declarations for the kernels we calibrate against. these are
   private to the cpu_opt build and aren't in any public header — declared
   inline here. */
extern int ax_compute_has_dwqkv_split_acc(void);
extern int ax_compute_has_qkv_head_gemm(void);

/* placeholder — real measurement TUs land in subsequent commits.
   for the current commit, the calibration is a no-op that just marks
   the state as calibrated and records the env-driven debug log if
   AX_ATTN_TUNABLES_LOG is set. each gate's measurement function will
   be added incrementally and call into this state struct. */
static void calibrate_all_gates(void) {
    /* TODO: per-gate measurements land here in follow-up commits. */
    (void)0;
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

void ax_attn_tunables_calibrate(void) {
    if (g_attn.calibrated) return;

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
