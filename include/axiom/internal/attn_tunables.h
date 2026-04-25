/* axiom/internal/attn_tunables.h — calibrated thresholds for attention.

   the attention forward and backward paths previously hardcoded several
   shape / cache-byte gates that select between fused and unfused kernel
   variants. those values were tuned on i5-12500H AVX2 and won't be
   optimal on AMD Zen, Apple M-series, or AVX-512 hosts.

   this header exposes the runtime-calibrated equivalents. each gate is
   measured at process startup against a small synthetic workload that
   represents the actual decision the gate makes (fused vs unfused
   kernel for the relevant primitive). measurements run with the AVX2 /
   AVX-512 selected ISA, the calibrated GEMM tile config, and the host's
   thread count, so they reflect the real per-call cost of each path.

   getters (ax_attn_tunable_*) are the only public API. callers don't
   see the calibration mechanism — they just call the getter and branch.

   on AX_NO_AUTOTUNE the getters return conservative defaults that
   roughly match the pre-calibration hardcoded values so behaviour is
   unchanged for users who explicitly opt out.

   persistence: see axiom/internal/calib_cache.h. calibration results are
   cached to disk keyed on (cpuid_str, isa_label, n_threads) so the cost
   is paid at most once per machine + version. */

#ifndef AX_INTERNAL_ATTN_TUNABLES_H
#define AX_INTERNAL_ATTN_TUNABLES_H

#include "axiom/types.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* early init: resolves env-var overrides for tunables that downstream
   calibrators consult (notably AX_GEMM_TILE_SWITCH_MARGIN, read by
   ax_calibrate_gemm_tiles). called from ax_compute_init BEFORE the
   gemm-tile sweep so the override has effect. registers atexit hooks
   used by the pack-profile and similar instrumentation. NO heavy
   measurements — that's ax_attn_tunables_calibrate. idempotent;
   safe to call before the active backend is initialised. */
void ax_attn_tunables_init_early(void);

/* run all attention-tunable measurements. called once from ax_compute_init
   after gemm tile calibration. budget ~2-10 s on a quiet machine; cached
   by calib_cache so subsequent runs are instantaneous.

   no-op under AX_NO_AUTOTUNE=1 (getters return conservative defaults that
   match the pre-calibration hardcoded values). idempotent. */
void ax_attn_tunables_calibrate(void);

/* A4: try to load a cached payload matching this host. on hit, applies
   the values to live state (g_attn + cpu_opt's static GEMM tile config)
   and marks calibration as done so subsequent calibrators short-circuit.
   returns true on hit, false on miss / env-disabled / error. */
bool ax_calib_cache_try_apply(void);

/* A4: returns true if the most recent calibrate cycle came from disk.
   used by the post-calibration save hook to avoid re-writing what we
   just read. */
bool ax_calib_loaded_from_cache(void);

/* A4: persist the current live tunable + tile state to the calibration
   cache. fire after both gemm-tile and attn-tunable calibrators have run.
   no-op when the values were loaded from cache (nothing new to save). */
void ax_calib_cache_save_current(void);

/* getters. all return values appropriate for the current host. callers
   should branch on these instead of hardcoded constants. */

/* F.3.a (qkv_head_gemm) gate: enable the fused kernel when
   rows * 3 * D * sizeof(float) > this threshold. corresponds to the
   L2-spill point where the fused kernel saves real DRAM traffic. */
int64_t ax_attn_tunable_f3a_qkv_bytes_threshold(void);

/* F.3.c (dwqkv_split_acc) gate: enable when D >= this value AND all
   three Wq/Wk/Wv require grad. lower D doesn't beat the [D, 3D]
   intermediate's L3 fit on most hosts. */
int64_t ax_attn_tunable_f3c_d_threshold(void);

/* save_p memory budget: enable forward P-save when
   bh * S * S * sizeof(float) <= this. the saved scores let bwd skip
   QK^T recompute. tuned to half-L3 typical. */
int64_t ax_attn_tunable_save_p_max_bytes(void);

/* save_p small-shape exclusion: skip save_p when (S * dk) <= this AND
   S <= some limit. on small workloads the recompute is cheap enough
   that the save-write cost dominates. set to 0 to never exclude. */
int64_t ax_attn_tunable_save_p_small_exclusion_sk(void);
int64_t ax_attn_tunable_save_p_small_exclusion_s(void);

/* fused_bh (Phase E combined fwd+bwd) auto-enable: per-head attn_flat
   slice in bytes (S * dk * sizeof(float)) must be <= this for the
   combined kernel to win. corresponds to L1d minus working scratch. */
int64_t ax_attn_tunable_fused_bh_per_head_bytes(void);

/* AX_SDPA_FUSED auto: returns true iff the FA-2-style fused kj-block
   variant should be used for this BH given the host's NT. captures the
   "BH != NT regression" rule but lets the calibrator override. */
bool ax_attn_tunable_use_sdpa_fused(int64_t BH);

/* gemm_tn pre-transpose flop threshold. the optimization is worth it
   only above this flop count (below, the transpose write traffic
   exceeds the saved pack_a_t). units: flops. */
int64_t ax_attn_tunable_gemm_tn_pretranspose_flops(void);

/* attention tile sizes. ATTN_BQ and ATTN_BK can be calibrated to fit
   the host's L1d / L2 budget rather than the compile-time MR-aligned
   default. returned values are MR-multiples. */
int64_t ax_attn_tunable_attn_bq(void);
int64_t ax_attn_tunable_attn_bk(void);

/* GEMM tile calibration: candidate-vs-baseline switch margin. the tile
   sweep prefers a non-baseline (mc, nc, kc) combo only when its score
   is < base_score * margin (default 0.94 → "must beat baseline by 6 %").
   tuned to sit between 5 % (too loose) and 8 % (too tight) on x86 +
   ARM hosts. exposed as a tunable so users on noisier hosts can
   loosen / tighten as needed without recompiling.

   getter returns a fraction in [0.5, 1.0); 1.0 - margin is the
   minimum-required improvement. callers should multiply baseline by
   the returned value to get the cutoff. NOT consumed by the attention
   path — lives here because attn_tunables.c is the consolidated
   runtime-tunables home. */
double  ax_attn_tunable_gemm_tile_switch_margin(void);

/* override / debug: AX_ATTN_TUNABLES_LOG=1 prints chosen values to
   stderr at calibration time. AX_NO_ATTN_CALIB=1 forces fallback to
   conservative defaults (== old hardcoded values). */

#ifdef __cplusplus
}
#endif

#endif /* AX_INTERNAL_ATTN_TUNABLES_H */
