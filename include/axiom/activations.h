/* axiom/activations.h — activation functions with autograd support.
   relu and sigmoid already live in ops.h; these are the rest. */

#ifndef AX_ACTIVATIONS_H
#define AX_ACTIVATIONS_H

#include "tensor.h"

/* leaky relu: max(alpha * x, x) where alpha is typically 0.01 */
ax_tensor_t *ax_leaky_relu(ax_tensor_t *a, float alpha);

/* elu: x if x > 0, alpha * (exp(x) - 1) otherwise */
ax_tensor_t *ax_elu(ax_tensor_t *a, float alpha);

/* selu: scale * elu(x, alpha) with specific constants for self-normalizing */
ax_tensor_t *ax_selu(ax_tensor_t *a);

/* gelu: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
   this is the approximate version used in most implementations */
ax_tensor_t *ax_gelu(ax_tensor_t *a);

/* swish (silu): x * sigmoid(x) */
ax_tensor_t *ax_swish(ax_tensor_t *a);

/* softplus: log(1 + exp(x)) */
ax_tensor_t *ax_softplus(ax_tensor_t *a);

/* mish: x * tanh(softplus(x)) */
ax_tensor_t *ax_mish(ax_tensor_t *a);

/* softmax along a given axis. numerically stable (subtracts max first).
   axis = -1 means last dimension. */
ax_tensor_t *ax_softmax(ax_tensor_t *a, int axis);

#endif /* AX_ACTIVATIONS_H */
