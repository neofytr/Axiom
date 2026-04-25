/* axiom/internal/calib_cache.h — disk persistence for autotuner output.

   the gemm-tile sweep + attn-tunables sweep collectively cost ~10-15 s
   on a typical x86 host. running them every process startup is a poor
   developer experience and a real cost in CI / serverless environments.

   this module persists the calibrated values to a small binary file
   (~150 bytes) keyed on the host fingerprint (cpu brand + ISA + thread
   count + library build identifier). on startup, we read the file
   first; if the key matches the current host, the cached values are
   loaded and the sweep is skipped entirely.

   key components:
     - cpu brand: distinguishes machines with similar topologies but
       different microarch. obtained from cpuid leaf 0x80000002-4 or
       /proc/cpuinfo.
     - ISA label: "avx512" / "avx2" / "scalar" — which kernel family is
       active. a different binary built for a different ISA gets a
       different cache entry.
     - thread count: the calibrator pre-tests at the active
       max_threads. running with a different OMP_NUM_THREADS would
       invalidate the gemm-tile / fused-bh decisions.
     - schema version: bumped whenever the payload struct changes so
       old caches aren't read into a new struct layout.

   default cache path: ${XDG_CACHE_HOME}/axiom/calib.bin (falling back
   to ${HOME}/.cache/axiom/calib.bin). overrideable with AX_CALIB_CACHE_PATH
   env var. set AX_NO_CALIB_CACHE=1 to disable both load and save. */

#ifndef AX_INTERNAL_CALIB_CACHE_H
#define AX_INTERNAL_CALIB_CACHE_H

#include "axiom/types.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* every persisted tunable goes here. when adding fields, bump the
   schema version in calib_cache.c so old caches are rejected. */
typedef struct ax_calib_payload {
    /* attn tunables */
    int64_t f3a_qkv_bytes_threshold;
    int64_t f3c_d_threshold;
    int64_t save_p_max_bytes;
    int64_t save_p_small_exclusion_sk;
    int64_t save_p_small_exclusion_s;
    int64_t fused_bh_per_head_bytes;
    int64_t gemm_tn_pretranspose_flops;
    int64_t attn_bq;
    int64_t attn_bk;
    /* sdpa_fused regime booleans expanded to int8 for stable ABI */
    int8_t  sdpa_fused_use_when_bh_lt_nt;
    int8_t  sdpa_fused_use_when_bh_eq_nt;
    int8_t  sdpa_fused_use_when_bh_gt_nt;
    int8_t  _pad0;
    double  gemm_tile_switch_margin;
    /* gemm tile config */
    int64_t gemm_mc, gemm_nc, gemm_kc;
} ax_calib_payload_t;

/* attempt to load a cached payload matching the current host. returns
   true on hit (out filled), false on miss / disabled / IO error.
   the function never modifies global state — caller decides whether
   to apply the cached values. */
bool ax_calib_cache_load(ax_calib_payload_t *out);

/* persist the supplied payload to disk under the current host's key.
   returns true on success, false on failure (creating the cache dir
   failed, write failed, etc.). errors are silent — calibration always
   continues even if persistence fails. */
bool ax_calib_cache_save(const ax_calib_payload_t *in);

/* compute the host fingerprint as a hex string suitable for logging.
   format: "<schema>-<cpu_hash>-<isa>-<nt>". buf must be ≥ 64 bytes. */
void ax_calib_cache_fingerprint(char *buf, size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif /* AX_INTERNAL_CALIB_CACHE_H */
