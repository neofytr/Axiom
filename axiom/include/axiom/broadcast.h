/* axiom/broadcast.h — shape broadcasting logic (numpy-style rules) */

#ifndef AX_BROADCAST_H
#define AX_BROADCAST_H

#include "types.h"

/* compute the broadcast output shape for two input shapes.
   follows numpy rules: dimensions are compared right-to-left,
   each pair must be equal or one of them must be 1.
   returns AX_OK on success, fills out_shape and out_ndim. */
ax_status_t ax_broadcast_shape(
    const int64_t *a_shape, int a_ndim,
    const int64_t *b_shape, int b_ndim,
    int64_t *out_shape, int *out_ndim);

/* check if shape a can broadcast to shape b (i.e., a is
   broadcastable into b without needing b to change) */
bool ax_broadcast_compatible(
    const int64_t *a_shape, int a_ndim,
    const int64_t *b_shape, int b_ndim);

#endif /* AX_BROADCAST_H */
