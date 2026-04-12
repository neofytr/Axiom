/* axiom/axiom.h — master include for the axiom framework */

#ifndef AXIOM_H
#define AXIOM_H

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

#endif /* AXIOM_H */
