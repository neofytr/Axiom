/* axiom/init.h — weight initialization schemes.

   proper initialization is surprisingly important. bad init can make
   deep networks untrainable.

   ownership: each init function fills the supplied tensor in-place. no
   allocation. NULL t is a safe no-op.

   error handling: return type is void; on dtype/dim errors the function
   logs via ax_err_set and leaves the tensor partially or fully written.
   inspect ax_err_last_status() to detect.

   thread-safety: writes to the supplied tensor — single-thread per
   tensor. the underlying RNG is process-global; concurrent init calls
   to disjoint tensors race on the rng state. seed before init for a
   deterministic sequence (single-threaded). */

#ifndef AX_INIT_H
#define AX_INIT_H

#include "tensor.h"

/* xavier/glorot uniform: U[-limit, limit] where limit = sqrt(6 / (fan_in + fan_out))
   good for sigmoid and tanh activations. keeps variance stable across layers. */
void ax_init_xavier_uniform(ax_tensor_t *t, int64_t fan_in, int64_t fan_out);

/* xavier/glorot normal: N(0, std) where std = sqrt(2 / (fan_in + fan_out)) */
void ax_init_xavier_normal(ax_tensor_t *t, int64_t fan_in, int64_t fan_out);

/* he/kaiming uniform: U[-limit, limit] where limit = sqrt(6 / fan_in)
   designed for relu activations. accounts for relu killing half the values. */
void ax_init_kaiming_uniform(ax_tensor_t *t, int64_t fan_in);

/* he/kaiming normal: N(0, std) where std = sqrt(2 / fan_in) */
void ax_init_kaiming_normal(ax_tensor_t *t, int64_t fan_in);

/* lecun normal: N(0, std) where std = sqrt(1 / fan_in)
   for selu activations (self-normalizing networks) */
void ax_init_lecun_normal(ax_tensor_t *t, int64_t fan_in);

/* uniform random in [low, high) */
void ax_init_uniform(ax_tensor_t *t, float low, float high);

/* normal/gaussian with given mean and std */
void ax_init_normal(ax_tensor_t *t, float mean, float std);

/* fill with zeros */
void ax_init_zeros(ax_tensor_t *t);

/* fill with ones */
void ax_init_ones(ax_tensor_t *t);

/* fill with a constant */
void ax_init_constant(ax_tensor_t *t, float value);

#endif /* AX_INIT_H */
