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
