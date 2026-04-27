/* axiom/internal/compute_internal.h — implementation-only compute APIs.

   not part of the public abi. include this from src/ only. these
   declarations expose the backend vtable (ax_backend_ops_t) and the
   parallelism thresholds that the core kernels read directly. moving
   them out of the public compute.h keeps the published surface to the
   high-level dispatch functions and prevents library users from coding
   against internal-only contracts that may change without notice. */

#ifndef AX_INTERNAL_COMPUTE_INTERNAL_H
#define AX_INTERNAL_COMPUTE_INTERNAL_H

#include "axiom/types.h"
#include "axiom/device.h"
#include "axiom/compute.h"
#include "axiom/internal/backend_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

/* get the ops vtable for the active backend (used internally by tensor ops). */
const ax_backend_ops_t *ax_compute_get_ops(void);

/* register a backend module's vtable. backends self-register at init,
   or core registers the static-build backends at startup. */
ax_status_t ax_compute_register_backend(ax_backend_id_t id, const ax_backend_ops_t *ops);

/* look up the backend that owns memory/lifecycle for a given device.
   returns NULL for AX_DEVICE_CPU or for devices whose backend module is
   not compiled in. core uses this to route storage allocation and
   host<->device transfers without knowing specific device types. */
const ax_backend_ops_t *ax_backend_for_device(ax_device_t device);

/* calibrated parallelism thresholds. set by ax_calibrate_thresholds()
   based on measured omp fork/join overhead at init time. kernels with
   per-iteration work below the threshold skip omp and run serial
   (fork-join barrier would dominate). */
extern int64_t ax_par_threshold_elems;   /* reduction / elementwise */
extern int64_t ax_par_threshold_batch;   /* per-sample loops (norm, loss) */
extern int64_t ax_par_threshold_flops;   /* generic flop-count threshold */
extern double  ax_omp_overhead_ms;       /* measured fork/join cost */

/* per-regime variants of ax_par_threshold_elems for kernels that know
   their compute density. light kernels (low flop/byte) defer to a higher
   threshold; heavy kernels (high flop/byte) parallelise sooner. */
extern int64_t ax_par_threshold_elems_light;
extern int64_t ax_par_threshold_elems_heavy;

/* bandwidth-bound parallelism threshold. total tensor bytes below which
   omp fork/join overhead exceeds the gain for memory-bound kernels
   (batchnorm, layernorm). calibrated via L2-resident memcpy probe. */
extern int64_t ax_par_threshold_bw_bytes;

/* returns 1 if the CPU has heterogeneous cores (P+E, big.LITTLE).
   GEMM uses this to select dynamic scheduling for load balance. */
int ax_compute_cores_heterogeneous(void);

/* measure omp fork/join overhead and derive the thresholds above. cheap
   (<50ms). called lazily from ax_compute_init(). safe to call multiple
   times (idempotent). no-op under AX_NO_AUTOTUNE=1 or without openmp. */
void ax_calibrate_thresholds(void);

/* sweep gemm tile (mc/nc/kc) shapes on representative workloads and
   pick the best. opt-in via AX_GEMM_CALIBRATE=1 (adds ~500 ms startup). */
void ax_calibrate_gemm_tiles(void);

/* hybrid CPU fast-vs-all crossover calibration. measures gemm throughput
   on a representative shape and derives the FLOPs threshold above which
   spreading work to all cores beats fast-only. cheap (~30ms). no-op on
   non-hybrid CPUs and under AX_NO_AUTOTUNE=1. */
void ax_calibrate_hybrid_crossover(void);

/* phase 34: measure each omp thread's relative throughput so gemm work
   distribution can be weighted proportionally. cheap (~5ms). no-op under
   AX_NO_AUTOTUNE=1 or without openmp. */
void ax_measure_thread_speeds(void);

#ifdef __cplusplus
}
#endif

#endif /* AX_INTERNAL_COMPUTE_INTERNAL_H */
