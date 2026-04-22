/* axiom/axiom.h — master include for the axiom framework.

   ============================================================
   CONVENTIONS
   ============================================================

   ERROR HANDLING
     - functions that report success/failure return ax_status_t (AX_OK on
       success, AX_ERR_* on failure). some constructors return a pointer
       and signal failure with NULL.
     - on AX_ERR_x or NULL, the failing thread's ax_err_last_status() and
       ax_err_last_message() describe the cause. consult them once after
       a call returns an error.
     - axiom never aborts in release builds. asserts and AX_BOUNDS_CHECK
       are debug-only (compile with NDEBUG to remove).

   OWNERSHIP
     There are four object lifetimes in axiom:
       - heap:  caller owns; release with the matching ax_*_destroy call.
       - arena: lives in the caller-supplied arena; freed when the arena
                is reset/destroyed. never free individually.
       - pool:  lives in the per-thread tensor storage pool; recycled by
                ax_graph_cleanup or by overwriting the storage slot.
       - view:  borrows the parent storage; destroying the view does NOT
                free the underlying buffer. parent must outlive view.
     Function comments call out which model applies when not obvious from
     the name (create -> heap, alloc_in -> arena, view -> view, etc.).
     For pointers returned via output parameters, the comment names the
     owner explicitly.

   THREAD SAFETY
     Default contract for read-only queries (ax_*_get_*, ax_compute_get_*,
     ax_err_last_*) is process- or thread-safe as appropriate. Mutating
     operations on a single object are typically NOT concurrent-safe.
     The kernels themselves use OpenMP internally; user code should not
     parallelise calls into axiom across threads on the same tensor.
     Per-function comments call out deviations (e.g. error TLS is
     per-thread; backend registration is process-wide).

   NAMING
     Public symbols use the ax_ prefix. Enum values use AX_. Header
     guards use AX_<NAME>_H. Verbs follow the create / destroy / get /
     set / forward / backward pattern with these accepted shorthands:
       - _fwd / _bwd: short form of forward / backward used by the SDPA
         primitive layer (ax_fused_attention_*); chosen for parity with
         pytorch / jax / tf and to keep the function names compact.
       - clip_* / transform_*: domain verbs (gradient clipping; data
         pipeline transforms) where create/get/set don't fit.
     Internal-only declarations live under axiom/internal/ and are not
     part of the published abi.

   VERSIONING / ABI STABILITY
     axiom follows SemVer (MAJOR.MINOR.PATCH). version constants and
     the AX_DEPRECATED / AX_ABI_STABLE_SINCE macros live in
     axiom/types.h. every symbol declared in a public header (anything
     under axiom/ but not axiom/internal/) is implicitly abi-stable
     since the current minor release. removals or signature changes
     require:
       1) one minor release with the old symbol marked AX_DEPRECATED;
       2) removal at the next major bump.
     internal headers carry no abi promise and may break between minor
     releases. version is exposed as AX_VERSION_{MAJOR,MINOR,PATCH},
     AX_VERSION (packed integer), and AX_VERSION_STRING.

   ============================================================ */

#ifndef AX_AXIOM_H
#define AX_AXIOM_H

#include "types.h"
#include "error.h"
#include "memory.h"
#include "tensor.h"
#include "compute.h"
#include "broadcast.h"
#include "ops.h"
#include "autograd.h"
#include "activations.h"
#include "rng.h"
#include "init.h"
#include "layer.h"
#include "model.h"
#include "serialize.h"
#include "conv.h"
#include "norm.h"
#include "attention.h"
#include "cuda.h"

/* training-only subset: excluded when building inference-only.
   the library also excludes their .c files under AX_INFERENCE_ONLY
   so user code trying to call these symbols gets a clean compile
   error at the include site rather than a buried linker error. */
#ifndef AX_INFERENCE_ONLY
#include "losses.h"
#include "optim.h"
#include "data.h"
#include "lr_scheduler.h"
#endif

/* initialize axiom — must be called before any other axiom functions.
   sets up compute backends, memory systems, etc. */
static inline ax_status_t ax_init(void) {
    return ax_compute_init();
}

/* shut down axiom — frees global resources */
static inline void ax_shutdown(void) {
    ax_compute_shutdown();
}

#endif /* AX_AXIOM_H */
